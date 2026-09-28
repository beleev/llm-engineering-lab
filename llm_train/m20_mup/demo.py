"""
M20 — μP (Maximal Update Parametrization): 在小模型上调 lr, 直接搬到大模型

是什么: 标准参数化 (SP) 下, 网络变宽时最优学习率会漂移 (Adam 下大约 ∝ 1/width), 小模型上调好的 lr 到大模型就不对了。
        μP 按宽度改写 "初始化方差 + 每层学习率 (+ 输出乘子)", 让每层更新对输出的影响与宽度无关 →
        最优 lr 不随宽度移动, 可以在窄的 proxy 模型上扫超参再 zero-shot 迁移 (μTransfer, GPT-4 / Cerebras-GPT / MiniCPM 在用)。
为什么 SP 会漂: Adam 的更新每个元素大小 ≈ lr, 与梯度大小无关。隐藏层 / 输出层有 fan_in = width 个输入,
        width 个 ≈lr 的同向小更新叠加 → 下一层激活的变化 ∝ lr · width。宽 4 倍, lr 就得小 4 倍。
μP (Adam, Tensor Programs V 表 3; m = width / base_width):
        输入层      init Var 1/fan_in     lr η          (fan_in 是固定的 d_in)
        隐藏层      init Var 1/fan_in     lr η / m
        输出层      init Var 1/(fan_in·m) lr η / m      (初始输出也要随宽度变小, 否则一开始就被随机头主导)
        m = 1 时与 SP 完全相同 → base 宽度上调出的 η 两种参数化通用。
读代码盯住: `init_and_lrs` 里 SP 与 μP 只差两个 `/ m`, 和 main 里每个宽度的 argmin。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import adam_update, banner, kv, make_rng

D_IN, BASE = 16, 32
WIDTHS = [32, 128, 512]
LOG2_LRS = np.arange(-13, -3)                                            # lr = 2^-13 … 2^-4
STEPS, BATCH = 150, 64
NAMES = {"sp": "SP", "mup": "μP"}

_rs = make_rng(20)
T1, T2 = _rs.randn(D_IN, 64) / 4, _rs.randn(64) / 8                    # 固定的 teacher: 单隐层 ReLU 网络


def init_and_lrs(width, lr, param, rs):
    """两层隐藏层 MLP  x → W1 → relu → W2 → relu → W3 → y 的初始权重和逐层 lr。"""
    m = width / BASE
    W = [rs.randn(D_IN, width) / np.sqrt(D_IN),
         rs.randn(width, width) / np.sqrt(width),
         rs.randn(width, 1) / np.sqrt(width) / (m if param == "mup" else 1)]
    lrs = [lr, lr / m, lr / m] if param == "mup" else [lr, lr, lr]
    return [w.astype(np.float32) for w in W], lrs


def train(width, lr, param):
    """Adam + 线性衰减到 0, 每步取新数据; 返回最后 10 步的平均 loss (发散返回 inf)。"""
    W, lrs = init_and_lrs(width, lr, param, make_rng(0))
    M = [np.zeros_like(w) for w in W]
    S = [np.zeros_like(w) for w in W]
    data_rs, losses = make_rng(99), []
    for t in range(1, STEPS + 1):
        x = data_rs.randn(BATCH, D_IN).astype(np.float32)
        y = (np.maximum(x @ T1, 0) @ T2)[:, None].astype(np.float32)
        h1 = np.maximum(x @ W[0], 0)
        h2 = np.maximum(h1 @ W[1], 0)
        out = h2 @ W[2]
        losses.append(float(np.mean((out - y) ** 2)))
        if not np.isfinite(losses[-1]):
            return np.inf
        g = 2 * (out - y) / BATCH
        g2 = (g @ W[2].T) * (h2 > 0)
        g1 = (g2 @ W[1].T) * (h1 > 0)
        for w, grad, m_, s_, lr_i in zip(W, (x.T @ g1, h1.T @ g2, h2.T @ g), M, S, lrs):
            adam_update(w, grad, m_, s_, t, lr_i * (1 - t / (STEPS + 1)))
    return float(np.mean(losses[-10:]))


def main() -> None:
    banner("M20 - muP Hyperparameter Transfer")
    print(f"\n  MLP 宽度 {WIDTHS}, base = {BASE}; 每个 (参数化, 宽度) 扫 lr = 2^{LOG2_LRS[0]} … 2^{LOG2_LRS[-1]}, {STEPS} 步 Adam")
    best, curves = {}, {}
    for param in ("sp", "mup"):
        print(f"\n  [{NAMES[param]}]  {'width':>6}  {'best log2(lr)':>13}   loss @ log2(lr) = " + " ".join(f"{v:>6}" for v in LOG2_LRS))
        for w in WIDTHS:
            curves[param, w] = np.array([train(w, 2.0**k, param) for k in LOG2_LRS])
            best[param, w] = int(LOG2_LRS[np.argmin(curves[param, w])])
            print(f"        {w:>6}  {best[param, w]:>13}   {' ' * 17}" + " ".join(f"{v:6.4f}" for v in curves[param, w]))

    i_base = list(LOG2_LRS).index(best["sp", BASE])                      # 在 base 宽度上调出的 lr (两种参数化此处相同)
    big = WIDTHS[-1]
    print()
    kv("SP  最优 log2(lr) 随宽度", [best["sp", w] for w in WIDTHS])
    kv("μP  最优 log2(lr) 随宽度", [best["mup", w] for w in WIDTHS])
    kv(f"把 base 的 lr 直接用到 width={big}", f"SP loss {curves['sp', big][i_base]:.4f} (该宽度最优 {curves['sp', big].min():.4f})"
       f" / μP loss {curves['mup', big][i_base]:.4f} (最优 {curves['mup', big].min():.4f})")
    kv("μP 同一 lr 下 loss 随宽度", [round(float(curves["mup", w][i_base]), 4) for w in WIDTHS])

    assert np.array_equal(curves["sp", BASE], curves["mup", BASE]), "base 宽度上 μP 就是 SP"
    assert all(abs(best["mup", w] - best["mup", BASE]) <= 1 for w in WIDTHS), "μP: 最优 lr 随宽度基本不动 (±1 个格点)"
    assert best["sp", big] <= best["sp", BASE] - 3, "SP: 宽 16 倍, 最优 lr 左移 ≥ 3 个 2 倍格点"
    assert curves["sp", big][i_base] > 3 * curves["sp", big].min(), "SP 下直接搬 lr 明显变差"
    assert curves["mup", big][i_base] < 1.1 * curves["mup", big].min(), "μP 下直接搬 lr 几乎就是最优"
    assert np.all(np.diff([curves["mup", w][i_base] for w in WIDTHS]) < 0), "μP: 同一 lr 下越宽越好"

    print("\n  OK: SP 的最优 lr 随宽度左移; μP 下 base 宽度调出的 lr 直接迁移到 16 倍宽, 且越宽越好。")


if __name__ == "__main__":
    main()
