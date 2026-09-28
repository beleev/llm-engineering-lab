"""
M19 — Scaling Laws: Chinchilla 算力最优

是什么: 训练 loss 同时受模型大小 N (参数量) 和数据量 D (训练 token 数) 限制, 经验上
        L(N, D) = E + A / N^α + B / D^β
        E 是不可约损失 (数据本身的噪声), 后两项分别是 "模型太小" 和 "数据太少" 的代价。
解决的问题: 给定算力 C ≈ 6·N·D (前向 2ND + 反向 4ND FLOPs), N 和 D 各给多少?
        Kaplan (2020) 说优先堆 N; Chinchilla (2022) 重新拟合得 N_opt ∝ C^0.5, D_opt ∝ C^0.5, 约 20 token / 参数。
关键公式: 把 D = C/(6N) 代入 L 对 N 求导 = 0:
        N_opt = G·(C/6)^a,  D_opt = (C/6)^b / G,  a = β/(α+β),  b = α/(α+β),  G = (αA / βB)^(1/(α+β))
做法: 6 个宽度 × 6 个数据量, 每格真训一遍 (单遍数据, cosine 退火到 0), 用变量投影拟合 (α, β 网格 + E, A, B 线性最小二乘)。
读代码盯住: `train` 里 "每个样本只看一次" (D 就是见过的样本数), `fit_scaling_law`, 和 `compute_optimal` 的闭式解。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import adam_update, banner, kv, make_rng

D_IN, K, NOISE = 8, 512, 0.1
WIDTHS = [2, 4, 8, 16, 32, 64]
DATA = [2048, 4096, 8192, 16384, 32768, 65536]
LR, BATCH = 3e-2, 32                                                     # 所有规模同一个 lr (在 1e-2 / 3e-2 / 1e-1 里粗扫过)

_rs = make_rng(19)
U_T = _rs.randn(K, D_IN) / np.sqrt(D_IN) * 2                             # teacher: 512 个 tanh 单元
C_T = _rs.randn(K) * 0.5
A_T = _rs.choice([-1, 1], K) * np.arange(1, K + 1) ** -1.0               # 输出权重按 1/k 衰减 → 窄网络只学得到头部, 误差随宽度幂律下降


def sample(rs, n):
    x = rs.randn(n, D_IN)
    return x, np.tanh(x @ U_T.T + C_T) @ A_T + NOISE * rs.randn(n)      # 标签噪声方差 σ² = 0.01 就是真实的 E


def n_params(width):
    return width * (D_IN + 2) + 1                                        # W1 + b1 + w2 + b2


def train(width, n_data, seed=0):
    """学生: 单隐层 tanh MLP。每步取新样本, 一共见 n_data 个 → 单遍数据, 与 LLM 预训练一致。返回测试 MSE。"""
    rs = make_rng(seed)
    P = {"W1": rs.randn(D_IN, width) / np.sqrt(D_IN), "b1": np.zeros(width),
         "w2": rs.randn(width) / np.sqrt(width), "b2": np.zeros(1)}
    M = {k: np.zeros_like(v) for k, v in P.items()}
    S = {k: np.zeros_like(v) for k, v in P.items()}
    data_rs, steps = make_rng(1000 + seed), n_data // BATCH
    for t in range(1, steps + 1):
        x, y = sample(data_rs, BATCH)
        h = np.tanh(x @ P["W1"] + P["b1"])
        g = 2 * (h @ P["w2"] + P["b2"][0] - y) / BATCH                  # dL/dpred
        gh = np.outer(g, P["w2"]) * (1 - h**2)
        grads = {"W1": x.T @ gh, "b1": gh.sum(0), "w2": h.T @ g, "b2": np.array([g.sum()])}
        lr_t = LR * 0.5 * (1 + np.cos(np.pi * t / steps))               # cosine 退火长度 = 本次的数据量 (Chinchilla 的关键修正)
        for k in P:
            adam_update(P[k], grads[k], M[k], S[k], t, lr_t)
    x, y = sample(make_rng(7), 4096)
    return float(np.mean((np.tanh(x @ P["W1"] + P["b1"]) @ P["w2"] + P["b2"][0] - y) ** 2))


def fit_scaling_law(N, D, L):
    """变量投影: 对每组 (α, β) 解线性最小二乘得 E, A, B ≥ 0; 残差按相对误差计。返回 (α, β, E, A, B, 相对 RMS)。"""
    best = None
    grid = np.linspace(0.05, 2.0, 79)
    for al in grid:
        for be in grid:
            X = np.stack([np.ones_like(N), N**-al, D**-be], axis=1) / L[:, None]
            coef = np.linalg.lstsq(X, np.ones_like(L), rcond=None)[0]
            if (coef < 0).any():
                continue
            r = float(np.sum((X @ coef - 1) ** 2))
            if best is None or r < best[0]:
                best = (r, al, be, *coef)
    r, al, be, E, A, B = best
    return al, be, E, A, B, np.sqrt(r / len(L))


def compute_optimal(C, al, be, A, B):
    a, b = be / (al + be), al / (al + be)
    G = (al * A / (be * B)) ** (1 / (al + be))
    return G * (C / 6) ** a, (C / 6) ** b / G


def main() -> None:
    banner("M19 - Scaling Laws (Chinchilla)")

    # ---- 1) 网格上真训 ----
    print(f"\n[1] 真训 {len(WIDTHS)}×{len(DATA)} 个 (N, D) 组合, 每格测试 MSE (不可约噪声 σ² = {NOISE**2:g})")
    print("  " + "N \\ D".rjust(8) + "".join(f"{d:>9}" for d in DATA))
    loss = {}
    for w in WIDTHS:
        row = [loss.setdefault((w, d), train(w, d)) for d in DATA]
        print("  " + f"{n_params(w):>8}" + "".join(f"{v:>9.4f}" for v in row))
    N = np.array([n_params(w) for w, _ in loss], dtype=float)
    D = np.array([d for _, d in loss], dtype=float)
    L = np.array(list(loss.values()))
    assert loss[64, 65536] < loss[64, 2048] and loss[64, 65536] < loss[2, 65536], "更多数据、更大模型都降 loss"
    assert abs(loss[2, 65536] - loss[2, 16384]) < 0.005, "N=21 的小模型: 数据再多也降不下去 (被 A/N^α 卡住)"

    # ---- 2) 拟合 ----
    al, be, E, A, B, rel = fit_scaling_law(N, D, L)
    print("\n[2] 拟合 L(N, D) = E + A/N^α + B/D^β")
    kv("E / A / α / B / β", f"{E:.4f} / {A:.3f} / {al:.3f} / {B:.1f} / {be:.3f}")
    kv("相对残差 RMS", f"{rel:.3f}  (单个种子, 训练本身有噪声)")
    kv("Chinchilla 论文 (Approach 3)", "E=1.69, α=0.34, β=0.28  → 玩具任务的指数完全不同, 正常")
    assert rel < 0.15, "36 个点被 5 个参数的幂律描述到 ~10% 以内"
    assert 0.5 * NOISE**2 < E < 1.5 * NOISE**2, "拟合出的 E 接近真实的标签噪声方差 —— E 确实是 '不可约' 的那部分"

    # ---- 3) 固定算力下的最优配比 ----
    a, b = be / (al + be), al / (al + be)
    print("\n[3] 固定 C = 6ND, 最优 N / D")
    kv("N_opt ∝ C^a, D_opt ∝ C^b", f"a = {a:.3f}, b = {b:.3f}   (Chinchilla: a ≈ b ≈ 0.5; Kaplan: a ≈ 0.73)")
    for C in (1e7, 1e8, 1e9):
        n_opt, d_opt = compute_optimal(C, al, be, A, B)
        ns = np.logspace(0, 6, 20001)                                    # 数值检查闭式解
        n_num = ns[np.argmin(A / ns**al + B / (C / 6 / ns) ** be)]
        assert abs(n_num / n_opt - 1) < 0.01
        kv(f"C = {C:.0e}", f"N_opt = {n_opt:7.0f}, D_opt = {d_opt:9.0f}, D/N = {d_opt / n_opt:5.0f}")

    # IsoFLOP: 网格的反对角线 (宽度 ×2, 数据 ÷2) 上 6ND 近似相等, 拿真训的点对照预测
    diag = [(w, DATA[-1 - i]) for i, w in enumerate(WIDTHS)]
    C_iso = np.mean([6 * n_params(w) * d for w, d in diag])
    n_pred = compute_optimal(C_iso, al, be, A, B)[0]
    w_best = min(diag, key=lambda wd: loss[wd])[0]
    print(f"\n  IsoFLOP (C ≈ {C_iso:.2e}, 实测) : " + ", ".join(f"N={n_params(w)}:{loss[w, d]:.4f}" for w, d in diag))
    kv("实测最优 N / 拟合预测 N_opt", f"{n_params(w_best)} / {n_pred:.0f}")
    assert 0.5 < n_pred / n_params(w_best) < 2, "预测的最优规模落在实测最优格点的相邻范围内"

    print("\n  OK: 玩具网格上 L(N,D) 拟合良好, E ≈ 噪声底; 固定算力下存在最优 N, 预测与 IsoFLOP 实测一致。指数不照搬论文。")


if __name__ == "__main__":
    main()
