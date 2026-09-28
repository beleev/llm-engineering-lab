"""
M14 — Muon 优化器 (Kimi K2 / Moonlight 等在用)

是什么: 对 **2-D 权重矩阵**, 把动量矩阵 M 正交化后再更新:  W ← W − lr · NS(M),  NS(M) ≈ U·Vᵀ  (M = UΣVᵀ)。
        embedding / 输出头 / bias / norm 等非矩阵参数仍用 AdamW。
解决的瓶颈: 样本效率 (同样 loss 约省 ~1/2 FLOPs) 和优化器显存 (只存 1 份动量, Adam 要 m+v 两份)。
关键公式: Newton–Schulz 五次迭代  X ← aX + (bA + cA²)X,  A = XXᵀ,  (a,b,c) = (3.4445, −4.7750, 2.0315)
          只用矩阵乘 → GPU 友好; 5 步把奇异值推到 ~1 附近 (不求精确, 落在 [0.6, 1.25] 即可)。
          太小的方向来不及到 1: 本 demo 里 σ=1e-4 的方向 5 步后是 0.041, 但也放大了两个数量级以上。
直觉: 梯度的奇异值谱极不均匀, SGD/Adam 的更新被少数大奇异方向主导; 正交化让每个方向步长相同。
读代码盯住: `newton_schulz` 前后的奇异值, 和 `0.2·√max(n,m)` 这个让 Muon 复用 AdamW 学习率的缩放。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import adam_update, banner, kv, make_rng, softmax


def newton_schulz(G: np.ndarray, steps: int = 5) -> np.ndarray:
    """G [n, m] → 近似 U·Vᵀ [n, m] (G = UΣVᵀ)。只用矩阵乘, 不做 SVD。

    结果的奇异值落在 1 附近, 不求精确。
    """
    a, b, c = 3.4445, -4.7750, 2.0315                # 文件头公式里的三个系数
    # +1e-7: G 全 0 时范数为 0, 防除零
    X = G / (np.linalg.norm(G) + 1e-7)               # Frobenius 归一 → 奇异值全部 ≤ 1, 迭代才收敛
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                                      # 让 A = XXᵀ 是较小的那个方阵
    for _ in range(steps):
        A = X @ X.T                                  # [min(n,m), min(n,m)]
        X = a * X + (b * A + c * A @ A) @ X          # 对每个奇异值 σ 作用多项式 aσ + bσ³ + cσ⁵
    return X.T if tall else X


class Muon:
    """只管一个 2-D 权重矩阵的 Muon。状态只有一份动量 buf, 与权重同形。"""

    def __init__(self, shape, lr, momentum=0.95, wd=0.0):
        self.lr, self.mu, self.wd = lr, momentum, wd # mu: 动量系数; wd: 权重衰减系数
        self.buf = np.zeros(shape)                   # 唯一的优化器状态
        self.rms_match = 0.2 * np.sqrt(max(shape))   # Moonlight: 让更新的 RMS ≈ AdamW 的 ~0.2, lr 可直接复用

    def step(self, W, G):
        """原地更新 W。G 是 W 的梯度, 两者同形。"""
        self.buf = self.mu * self.buf + G
        O = newton_schulz(G + self.mu * self.buf)    # Nesterov 式动量, 再正交化
        W *= 1 - self.lr * self.wd                   # 解耦权重衰减 (同 AdamW): 直接缩 W, 不进动量
        W -= self.lr * self.rms_match * O


def qk_clip(Wq, Wk, x, tau: float):
    """MuonClip 的 QK-clip: 若最大 attention logit 超过 τ, 把 Wq/Wk 各缩 √(τ/S_max)。

    Muon 的更新是满秩的, 大规模训练时 QK logit 容易爆; Kimi K2 每步后做这个裁剪。
    """
    # x [N, d], Wq / Wk [d, d_k]。logit = (x·Wq)(x·Wk)ᵀ / √d_k, 形状 [N, N]; s_max 是其中绝对值最大的
    s_max = np.abs((x @ Wq) @ (x @ Wk).T).max() / np.sqrt(Wq.shape[1])
    eta = min(1.0, tau / s_max)                      # 没超过 τ 时 eta = 1, 不动
    # logit 里 Wq 和 Wk 各出现一次: 两边各乘 √eta, logit 正好乘 eta
    return Wq * np.sqrt(eta), Wk * np.sqrt(eta), s_max


def run(opt_name: str, lr: float, rotate: bool = True, steps: int = 150):
    """病态的矩阵回归: 输入协方差条件数 1e4 且 **不与坐标轴对齐** (乘了随机旋转 Q)。

    Adam 的逐元素缩放只能修 "对角" 的病态; 旋转之后它无能为力, 而 NS 正交化与基无关。
    W 走 Muon 或 Adam, 1-D 的 b 两边都用 Adam。返回每一步的 loss 列表。

    两边的权重衰减都是 0 (adam_update 和 Muon 的 wd 都用默认值), 所以对照臂是 Adam 不是 AdamW。
    这里比的是 "逐元素缩放 vs 正交化" 这一处差别; 无噪声回归上加衰减只会把 W 往 0 拉, 让两臂多一个不相干的变量。
    """
    rs = make_rng(14)
    d_in, d_out, B = 32, 32, 64
    feat_scale = np.logspace(0, -2, d_in)            # 主轴标准差 1 → 0.01
    Q = np.linalg.qr(rs.randn(d_in, d_in))[0] if rotate else np.eye(d_in)   # 随机旋转: 病态方向不再是单个坐标
    W_true, b_true = rs.randn(d_in, d_out), rs.randn(d_out)   # 要学的目标: [d_in, d_out], [d_out]
    W, b = np.zeros((d_in, d_out)), np.zeros(d_out)
    muon = Muon(W.shape, lr)
    # Adam 的一阶 / 二阶矩, W 和 b 各一套
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    losses = []
    for t in range(1, steps + 1):
        x = (rs.randn(B, d_in) * feat_scale) @ Q     # [B, d_in]
        diff = x @ W + b - (x @ W_true + b_true)     # [B, d_out]
        losses.append(float(np.mean(diff**2)))
        # MSE 的梯度: gW [d_in, B] @ [B, d_out] → [d_in, d_out]; gb 沿 batch 求和 → [d_out]
        gW, gb = x.T @ diff * (2 / diff.size), diff.sum(0) * (2 / diff.size)
        lr_t = lr * (1 - t / steps)                  # 线性降到 0
        if opt_name == "muon":
            muon.lr = lr_t
            muon.step(W, gW)
        else:
            adam_update(W, gW, mW, vW, t, lr_t)
        # b 的 lr 固定从 0.05 起, 不跟着被扫描的 lr 变: 两臂的差别只在 W 怎么更新
        adam_update(b, gb, mb, vb, t, 0.05 * (1 - t / steps))     # 1-D 参数: 两边都用 Adam
    return losses


def main() -> None:
    banner("M14 - Muon Optimizer")
    rs = make_rng(0)

    # ---- 1) Newton–Schulz 到底做了什么 ----
    U, _ = np.linalg.qr(rs.randn(16, 16))
    V, _ = np.linalg.qr(rs.randn(32, 32))
    sigma = np.logspace(0, -4, 16)                   # 16 个奇异值, 从 1 到 1e-4
    G = U @ np.diag(sigma) @ V[:16]                  # [16, 16] @ [16, 16] @ [16, 32] → [16, 32]
    print(f"\n[1] Newton–Schulz 正交化 (条件数 {sigma.max() / sigma.min():.0e} 的 {G.shape[0]}×{G.shape[1]} 矩阵)")
    sv = {k: np.linalg.svd(newton_schulz(G, steps=k), compute_uv=False) for k in (1, 3, 5)}
    kv("输入奇异值 max / min", f"{sigma.max():.0e} / {sigma.min():.0e}")
    for k, s in sv.items():
        kv(f"NS {k} 步后奇异值 max / min", f"{s.max():.3f} / {s.min():.3f}")
    O = newton_schulz(G)
    exact = U @ V[:16]                               # 精确的 U·Vᵀ (需要 SVD, GPU 上很慢)
    big = sigma > 1e-2                               # 只在 "来得及被推到 1" 的方向上比较
    kv("与精确 UVᵀ 的方向余弦", f"{np.sum(O * exact) / np.linalg.norm(O) / np.linalg.norm(exact):.3f}")
    assert sv[5][: big.sum()].min() > 0.6, "σ ≥ 1e-2 的方向 5 步后奇异值应高于 0.6"
    assert sv[5].max() < 1.25, "5 步后最大奇异值应低于 1.25"
    assert sv[5].min() > 100 * sigma.min(), "哪怕 σ=1e-4 的方向也被放大了两个数量级以上"

    # ---- 2) Muon vs Adam ----
    steps = 150
    print(f"\n[2] 病态矩阵回归 {steps} 步, 各自扫 lr 取最好")
    grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    best = {}                                        # (优化器, 是否旋转) → 扫 lr 后最低的末步 loss
    for rotate in (True, False):
        for name in ("adam", "muon"):
            finals = {lr: run(name, lr, rotate, steps)[-1] for lr in grid}
            best[name, rotate] = min(finals.values())
            if rotate:
                kv(f"{name.capitalize()}: 各 lr 的末步 loss", {lr: float(f"{v:.2e}") for lr, v in finals.items()})
    a, m_ = best["adam", True], best["muon", True]
    kv("旋转过的病态 (非轴对齐)", f"Adam {a:.2e} / Muon {m_:.2e}  → Muon 好 {a / m_:.1f}x")
    a0, m0 = best["adam", False], best["muon", False]
    kv("轴对齐的病态 (Adam 的主场)", f"Adam {a0:.2e} / Muon {m0:.2e}  → Adam 好 {m0 / a0:.1f}x")
    assert m_ < 0.5 * a, "病态方向不与坐标轴对齐时, 各自最优 lr 下 Muon 明显更低"
    assert a / m_ > a0 / m0, "轴对齐时 Adam 的逐元素缩放本来就够用, Muon 的优势缩小"
    kv("优化器状态 (每个矩阵参数)", "Adam 2 份 (m, v) / Muon 1 份 (动量)")

    # ---- 3) QK-clip ----
    tau = 100.0
    print(f"\n[3] MuonClip 的 QK-clip (τ = {tau:g})")
    x = rs.randn(16, 32)                             # [N=16, d=32]
    Wq, Wk = rs.randn(32, 8) * 3, rs.randn(32, 8) * 3    # [d, d_k=8]; 乘 3 故意让 logit 爆掉
    Wq2, Wk2, s_before = qk_clip(Wq, Wk, x, tau=tau)
    _, _, s_after = qk_clip(Wq2, Wk2, x, tau=tau)
    kv("max attention logit 裁剪前 → 后", f"{s_before:.1f} → {s_after:.1f}")
    p = softmax((x @ Wq) @ (x @ Wk).T / np.sqrt(Wq.shape[1]))
    kv("裁剪前 softmax 最大概率均值", f"{p.max(1).mean():.3f}  (≈1 即 one-hot, 梯度消失)")
    assert s_before > tau, f"裁剪前最大 logit 应超过 τ = {tau:g}, 否则演示不到裁剪"
    assert abs(s_after - tau) < 1e-6, "裁剪后最大 logit 必须正好等于 τ"

    print("\n  OK: NS 把梯度谱拉平; 非轴对齐的病态问题上 Muon 胜过调好 lr 的 Adam (轴对齐时反而 Adam 赢); 大规模时配 QK-clip。")


if __name__ == "__main__":
    main()
