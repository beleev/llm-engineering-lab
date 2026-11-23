"""
M10 — 训练稳定性: LR 调度 (cosine / WSD) · 全局梯度裁剪 · NaN guard · z-loss

是什么: 让长跑任务不炸的四件朴素工具。
解决的瓶颈: 不是显存也不是通信, 是 "loss spike 之后要不要回滚 3 天" 的工程风险。
关键公式: cosine  lr = ½·base·(1 + cos(π·progress))            需要预先知道总步数
          WSD     warmup → 恒定 (stable) → 最后 ~10% 快速衰减     稳定段与总步数无关, 可随时续训/分叉
          clip    g ← g · min(1, c / ‖g‖₂),  ‖·‖ 是 **所有参数拼在一起** 的范数 (方向不变)
          z-loss  L += c_z · mean(log² Z),  Z = Σ exp(logit)      交叉熵对 logit 整体平移不敏感,
                  这个方向没人管, log Z 就会一路往上漂; 加了这一项它被拉回 0 附近
读代码盯住: `wsd_lr` 里 stable 段不依赖 total_steps; `guarded_step` 里坏 step 对参数零影响;
            `z_loss` 的梯度 2·c_z·log Z·softmax —— 它只压整体高度, 不压 logit 之间的差。
"""
from __future__ import annotations

import math
import numpy as np

from llm_train.core import banner, clip_by_global_norm, global_norm, has_overflow, kv, make_rng, softmax


def warmup_cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.0) -> float:
    """step 从 0 计 (llm_basic/optim.py 的 cosine_lr 从 1 计)。

    前 warmup_steps 步线性升到 base_lr, 之后半个余弦降向 min_ratio·base_lr。
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps         # +1: 第 0 步的 lr 不为 0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)   # 0 → 1
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def wsd_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int,
           decay_frac: float = 0.1, min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay (MiniCPM / DeepSeek-V3 / Kimi 同类调度)。"""
    decay_start = total_steps - int(total_steps * decay_frac)   # 最后 decay_frac 比例的步数用来衰减
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return base_lr                                     # 稳定段: 与 total_steps 无关
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return base_lr * (1 - (1 - min_ratio) * progress)      # 线性退火 (也常用 1-sqrt / 指数)


def guarded_step(params, grads, lr: float, max_norm: float) -> bool:
    """NaN guard + clip + SGD。坏 step 返回 False 且不碰参数。"""
    if has_overflow(grads):
        return False
    clipped, _, _ = clip_by_global_norm(grads, max_norm)
    for k in params:
        params[k] -= lr * clipped[k]
    return True


# ---- z-loss: 把 log Z 拉回 0 附近 ------------------------------------- #

def log_partition(logits: np.ndarray) -> np.ndarray:
    """逐行的 log Z = log Σ exp(logit)。减最大值只为防 exp 溢出, 结果不变。"""
    m = logits.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))[..., 0]


def z_loss(logits: np.ndarray, coef: float) -> tuple[float, np.ndarray]:
    """PaLM 的 z-loss: coef · mean(log² Z)。返回 (惩罚值, 对 logits 的梯度)。

    ∂ log Z / ∂ logit_j = softmax_j, 所以 ∂ log²Z / ∂ logit_j = 2 · log Z · softmax_j。
    ST-MoE 的 router z-loss 是同一个式子, 只是作用在路由器的 logits 上, 系数更大。
    """
    lz = log_partition(logits)                                   # [B]
    grad = coef * 2.0 * lz[:, None] / logits.shape[0] * softmax(logits)
    return coef * float(np.mean(lz ** 2)), grad


def train_softmax(coef: float, lr: float, steps: int, n_class: int, d_in: int = 8,
                  batch: int = 64, seed: int = 0) -> dict:
    """一层 softmax 分类器 (logits = x @ W + b) 上的 z-loss 对照。

    数据可分, 所以交叉熵会一直往下走, 把胜出的 logit 一路顶高 —— log Z 跟着漂。
    n_class 当成输出头的词表大小, 或者 MoE 路由器的专家数, 两种用法只差一个系数。
    """
    rs = make_rng(seed)
    teacher = rs.randn(d_in, n_class).astype(np.float32)         # 生成标签的线性 teacher
    x = rs.randn(batch, d_in).astype(np.float32)                 # [batch, d_in]
    y = np.argmax(x @ teacher, axis=1)                           # [batch]
    onehot = np.eye(n_class, dtype=np.float32)[y]
    W = (rs.randn(d_in, n_class) * 0.02).astype(np.float32)
    b = np.zeros(n_class, dtype=np.float32)
    for _ in range(steps):
        logits = x @ W + b                                       # [batch, n_class]
        probs = softmax(logits)
        d_logits = (probs - onehot) / batch                      # 交叉熵对 logits 的梯度
        d_logits = d_logits + z_loss(logits, coef)[1]            # ★ 唯一的改动: 加上 z-loss 的梯度
        W -= lr * (x.T @ d_logits)
        b -= lr * d_logits.sum(axis=0)
    logits = x @ W + b
    probs = softmax(logits)
    ce = float(-np.mean(np.log(probs[np.arange(batch), y] + 1e-30)))
    return {
        "ce": ce,                                                # 不含 z-loss 的那部分
        "log_z": float(np.mean(log_partition(logits))),
        "max_logit": float(logits.max()),
        "top1": float(np.mean(np.argmax(logits, axis=1) == y)),  # 排序有没有被 z-loss 改坏
    }


def main() -> None:
    banner("M10 - Training Stability")

    # ---- 1) 两种调度 ----
    total, warm, base = 100, 10, 1e-3
    cos = [warmup_cosine_lr(s, total, base, warm) for s in range(total)]
    wsd = [wsd_lr(s, total, base, warm) for s in range(total)]
    marks = [0, 9, 30, 60, 89, 95, 99]
    print(f"\n[1] LR 调度 (total={total}, warmup={warm}, base={base:g})")
    kv("步", marks)
    kv("cosine ×1e-3", [round(cos[s] * 1e3, 3) for s in marks])
    kv("WSD    ×1e-3", [round(wsd[s] * 1e3, 3) for s in marks])

    # "想多训一倍" 时: WSD 已经走过的稳定段 lr 完全不变, cosine 整条曲线都变了
    total2, keep = 2 * total, 90                           # 总步数翻倍; 比较前 keep 步 (都还在 WSD 稳定段里)
    cos200 = [warmup_cosine_lr(s, total2, base, warm) for s in range(keep)]
    wsd200 = [wsd_lr(s, total2, base, warm) for s in range(keep)]
    kv(f"总步数 {total}→{total2}, 前 {keep} 步 lr 最大变化", f"cosine {max(abs(a - b) for a, b in zip(cos, cos200)):.2e},  "
                                                 f"WSD {max(abs(a - b) for a, b in zip(wsd, wsd200)):.2e}")
    assert wsd[:keep] == wsd200, "WSD: 稳定段 checkpoint 可以直接续训或分叉出一个 decay 分支"
    assert max(abs(a - b) for a, b in zip(cos, cos200)) > 0.3 * base, \
        "cosine: 总步数 100 → 200 后, 前 90 步的 lr 最大要变 0.3·base 以上"
    assert all(lr == base for lr in wsd[warm:keep]), "WSD 稳定段 (step 10~89) 的 lr 必须恒等于 base"
    assert wsd[-1] < 0.11 * base, "WSD 最后一步应衰减到 base 的 11% 以下"
    assert abs(cos[warm] - base) < 1e-12, "cosine: warmup 刚结束 (step=10) 时 lr 应等于 base"
    assert cos[-1] < 0.01 * base, "cosine 最后一步应降到 base 的 1% 以下"
    assert all(a < b for a, b in zip(cos[:warm - 1], cos[1:warm])), "warmup 单调上升"

    # ---- 2) 全局范数裁剪 ----
    print("\n[2] 全局范数裁剪 (global-norm clipping)")
    grads = {
        "W": np.array([[1.0, 2.0], [100.0, -50.0]], dtype=np.float32),      # 一个 spike
        "b": np.array([0.5, -0.25], dtype=np.float32),
    }
    clipped, old_norm, scale = clip_by_global_norm(grads, max_norm=5.0)
    cosine_sim = float(np.sum(grads["W"] * clipped["W"]) / (np.linalg.norm(grads["W"]) * np.linalg.norm(clipped["W"])))
    kv("‖g‖ 裁剪前 → 后", f"{old_norm:.2f} → {global_norm(clipped):.2f}  (scale={scale:.4f})")
    kv("方向 cos 相似度", f"{cosine_sim:.6f}")
    assert abs(global_norm(clipped) - 5.0) < 1e-5, "裁剪后的全局范数必须等于 max_norm = 5"
    assert abs(cosine_sim - 1) < 1e-6, "裁剪只缩长度: 前后的 cos 相似度必须是 1"
    small = {"W": np.array([0.1], dtype=np.float32)}
    assert clip_by_global_norm(small, 5.0)[2] == 1.0, "范数没超阈值就原样放行"

    # ---- 3) NaN guard ----
    print("\n[3] NaN/Inf guard")
    params = {"W": np.ones((2, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    before = {k: v.copy() for k, v in params.items()}
    bad = {"W": np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    ok_bad = guarded_step(params, bad, lr=0.1, max_norm=5.0)
    untouched = all(np.array_equal(before[k], params[k]) for k in params)
    ok_good = guarded_step(params, grads, lr=0.1, max_norm=5.0)
    kv("坏 step: 执行? / 参数被改?", f"{ok_bad} / {not untouched}")
    kv("好 step: 执行?", ok_good)
    assert not ok_bad, "含 NaN 的 step 必须被拒绝"
    assert untouched, "坏 step 不能改动任何参数"
    assert ok_good, "正常的 step 必须照常执行"
    assert np.isfinite(params["W"]).all(), "好 step 之后参数里不能有 NaN / Inf"
    # clip 救不了 NaN: 范数是 NaN, 任何缩放系数乘上去还是 NaN (torch 里系数本身也变 NaN, 污染全部参数)
    assert has_overflow(clip_by_global_norm(bad, 5.0)[0]), "所以 guard 必须排在 clip 之前"

    print("\n  OK: WSD 稳定段与总步数解耦; clip 只缩长度不改方向; NaN guard 必须排在 clip 之前。")

    # ---- 4) z-loss: 输出头的 log Z 漂移 ----
    print("\n[4] z-loss (PaLM 的 1e-4·log²Z): 输出头 16 类, 3000 步, 可分数据")
    for lr_big, tag in [(0.5, "lr=0.5"), (8.0, "lr=8.0 (偏大)")]:
        row = {c: train_softmax(c, lr_big, 3000, n_class=16) for c in (0.0, 1e-4, 1e-3)}
        kv(f"{tag} mean log Z (c_z = 0 / 1e-4 / 1e-3)",
           " / ".join(f"{r['log_z']:.2f}" for r in row.values()))
        kv(f"{tag} max logit", " / ".join(f"{r['max_logit']:.2f}" for r in row.values()))
        kv(f"{tag} 交叉熵 (不含 z-loss)", " / ".join(f"{r['ce']:.4f}" for r in row.values()))
        kv(f"{tag} top-1 正确率", " / ".join(f"{r['top1']:.3f}" for r in row.values()))
    big = {c: train_softmax(c, 8.0, 3000, n_class=16) for c in (0.0, 1e-4, 1e-3)}
    assert big[0.0]["log_z"] > 20, f"lr 偏大时不加 z-loss 的 log Z 应漂到 20 以上, 实得 {big[0.0]['log_z']:.2f}"
    assert big[1e-4]["log_z"] < 0.6 * big[0.0]["log_z"], "c_z = 1e-4 应把 log Z 压到不加时的 60% 以下"
    assert big[1e-3]["log_z"] < big[1e-4]["log_z"], "系数越大, log Z 被拉得越靠近 0"
    assert big[1e-3]["ce"] > big[0.0]["ce"], "z-loss 是额外约束, 交叉熵本身会变差"
    assert all(r["top1"] == 1.0 for r in big.values()), "三档的 top-1 排序都不应被改坏"
    # 为什么在意 log Z: softmax 实现里先减最大值, 但 logit 本身要先能被存下来
    for fmt, mx in [("fp16", 65504.0), ("bf16", 3.3895e38)]:
        kv(f"{fmt} 下 exp(logit) 溢出的 logit 阈值", f"{math.log(mx):.1f}")
    # 三个种子: 结论的方向稳定, 具体数值有波动
    seeds = [(train_softmax(0.0, 8.0, 3000, 16, seed=s)["log_z"],
              train_softmax(1e-4, 8.0, 3000, 16, seed=s)["log_z"]) for s in (0, 1, 2)]
    kv("3 个种子的 log Z (无 / 1e-4)", "; ".join(f"{a:.1f}→{b:.1f}" for a, b in seeds))
    assert all(b < a for a, b in seeds), "三个种子上 z-loss 都应把 log Z 压低"

    # ---- 5) router z-loss: 同一个式子, 作用在路由器上 ----
    print("\n[5] router z-loss (ST-MoE 的 1e-3): 路由器 8 个专家, 3000 步, lr=8.0")
    routers = {c: train_softmax(c, 8.0, 3000, n_class=8) for c in (0.0, 1e-3)}
    kv("router mean log Z (c_z = 0 / 1e-3)", " / ".join(f"{r['log_z']:.2f}" for r in routers.values()))
    kv("router max logit", " / ".join(f"{r['max_logit']:.2f}" for r in routers.values()))
    kv("router top-1 专家选对的比例", " / ".join(f"{r['top1']:.3f}" for r in routers.values()))
    assert routers[1e-3]["log_z"] < routers[0.0]["log_z"], "router z-loss 也应把 log Z 压低"
    assert routers[1e-3]["top1"] == routers[0.0]["top1"] == 1.0, "压 log Z 不改变选哪个专家"

    print("\n  OK: z-loss 只压 log Z 的整体高度, 不改 logit 之间的差, 所以 top-1 不变;")
    print("      代价是交叉熵本身变差, 系数要在这两者之间挑。")


if __name__ == "__main__":
    main()
