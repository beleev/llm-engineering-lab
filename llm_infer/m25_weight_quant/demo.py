"""
m25 demo — 另外三条量化路线: GPTQ (补偿误差) / SmoothQuant W8A8 (压平激活) / FP8 (换格点)

[A] GPTQ vs RTN (INT4 group=32): 同一组权重、同一批校准激活; 误差在 held-out 激活上报告。
    A1 激活通道相关 → GPTQ 明显赢;  A2 激活 iid → 无相关可用, 校准集小时反而过拟合;
    A3 相关 + 离群通道 → AWQ 与 GPTQ 各治一半, 叠加最好 (与 m08 的分工)。
[B] SmoothQuant W8A8: 激活离群通道让 per-token INT8 失真; 把缩放迁到权重后下降; 扫 α。
[C] FP8 E4M3 / E5M2: 舍入正确性; per-tensor vs per-channel scale (与 INT8 对照); W8A8 激活 per-tensor。
数据全部人工合成 (相关结构、离群通道位置与倍数都打印出来)。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m08_quantization.int4_awq import quant_dequant, output_err, awq_quantize
from llm_infer.m25_weight_quant.gptq import gptq_quantize
from llm_infer.m25_weight_quant.smoothquant import fake_quant_sym, w8a8_matmul, smooth_scales
from llm_infer.m25_weight_quant.fp8 import FORMATS, fp8_round, fake_quant_fp8

D_IN, D_OUT, G = 256, 256, 32
OUTLIERS = [7, 50, 131, 200]


def rel(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def correlated_acts(rs, n, rank=32, noise=0.3):
    """X = 低秩公共成分 + 独立噪声: 输入通道之间强相关 (真实 LLM 激活的典型形态, H = XᵀX 远非对角)。"""
    return (rs.randn(n, rank) @ rs.randn(rank, D_IN) / np.sqrt(rank) + noise * rs.randn(n, D_IN)).astype(np.float32)


def gptq_demo(rs, W):
    N = 512
    print(f"\n[A] GPTQ vs RTN: INT4 group={G} 非对称, 存储格式相同; 误差 = ‖XW−XŴ‖/‖XW‖")
    print("  A1 激活通道相关 (rank-32 公共成分 + 0.3 噪声)")
    X = correlated_acts(rs, 2 * N)
    Xc, Xt = X[:N], X[N:]
    W_rtn, W_gptq = quant_dequant(W, 4, G), gptq_quantize(W, Xc, 4, G)
    print(f"  {'方案':<12} {'calib 误差':>10} {'held-out 误差':>13}")
    for name, W_hat in (("RTN", W_rtn), ("GPTQ", W_gptq)):
        print(f"  {name:<12} {output_err(Xc, W, W_hat):>10.4f} {output_err(Xt, W, W_hat):>13.4f}")
    e_rtn, e_gptq = output_err(Xt, W, W_rtn), output_err(Xt, W, W_gptq)
    assert e_gptq < 0.6 * e_rtn, (e_gptq, e_rtn)                          # 至少降 40%

    print("  A2 激活 iid (H ≈ 对角, 没有相关性可补偿): 扫校准样本数 N")
    X_test = rs.randn(2048, D_IN).astype(np.float32)
    e_rtn_iid = output_err(X_test, W, W_rtn)
    gaps = {}
    for n in (256, 512, 2048, 8192):
        e = output_err(X_test, W, gptq_quantize(W, rs.randn(n, D_IN).astype(np.float32), 4, G))
        gaps[n] = e / e_rtn_iid
        print(f"  N={n:<5} GPTQ {e:.4f}  vs RTN {e_rtn_iid:.4f}  ({gaps[n]:.2f}x)")
    assert gaps[256] > 1.2                                # 校准集小: 把噪声当相关性去"补偿", 泛化更差
    assert gaps[8192] < 1.05 and gaps[8192] < gaps[2048] < gaps[512] < gaps[256]   # N 大了才打平, 不会更好

    print("  A3 相关激活 + 离群输入通道 × 30 (AWQ 的主场)")
    X = correlated_acts(rs, 2 * N)
    X[:, OUTLIERS] *= 30
    Xc, Xt = X[:N], X[N:]
    W_awq, alpha, s, _ = awq_quantize(W, Xc, 4, G)
    W_both = gptq_quantize(W * s[:, None], Xc / s, 4, G) / s[:, None]    # 先 AWQ 缩放, 再对缩放后的问题做 GPTQ
    res = {name: output_err(Xt, W, W_hat) for name, W_hat in
           (("RTN", quant_dequant(W, 4, G)), (f"AWQ (α={alpha:g})", W_awq), ("GPTQ", gptq_quantize(W, Xc, 4, G)),
            ("AWQ + GPTQ", W_both))}
    for name, e in res.items():
        print(f"  {name:<14} {e:.4f}")
    e_r, e_a, e_g, e_b = res.values()
    assert max(e_a, e_g) < 0.5 * e_r                      # 两者各自都远好于 RTN
    assert e_b < 0.6 * min(e_a, e_g)                      # 治的是不同的病, 叠加还能再降
    return e_rtn, e_gptq


def smoothquant_demo(rs, W):
    N, mult = 256, 50.0
    print(f"\n[B] SmoothQuant W8A8: INT8 激活 per-token + INT8 权重 per-channel, 离群输入通道 {OUTLIERS} × {mult:g}")
    X = rs.randn(2 * N, D_IN).astype(np.float32)
    X[:, OUTLIERS] *= mult
    Xc, Xt = X[:N], X[N:]                                 # s 只用 calib 的 max|X_j|
    Y = Xt @ W

    def errs(X_, W_):                                     # (只量化激活, 只量化权重, 都量化)
        return tuple(rel(Y, w8a8_matmul(X_, W_, qx, qw)) for qx, qw in ((True, False), (False, True), (True, True)))

    print(f"  {'':<10} {'只量化 X':>9} {'只量化 W':>9} {'W8A8':>8}")
    base = errs(Xt, W)
    print(f"  {'不平滑':<10} {base[0]:>9.4f} {base[1]:>9.4f} {base[2]:>8.4f}")
    sweep = {}
    for alpha in np.round(np.arange(0, 1.01, 0.1), 1):
        s = smooth_scales(Xc, W, alpha)
        sweep[float(alpha)] = r = errs(Xt / s, W * s[:, None])            # 数学上 (X/s)(s⊙W) = XW
        print(f"  α={alpha:<8g} {r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>8.4f}")
    best = min(sweep, key=lambda a: sweep[a][2])
    kv("最优 α / W8A8 误差", f"{best:g} / {sweep[best][2]:.4f}  (不平滑 {base[2]:.4f}, 降 {base[2] / sweep[best][2]:.1f}x)")
    assert base[0] > 3 * base[1]                          # 不平滑时误差几乎全来自激活
    assert 0 < best < 1 and sweep[best][2] < 0.4 * base[2]
    assert sweep[1.0][1] > 3 * sweep[best][1]             # α=1: 离群值全压到权重, 权重误差反过来爆
    return base[2], best, sweep[best][2]


def fp8_demo(rs, W):
    print("\n[C] FP8: E4M3 (3 bit 尾数) / E5M2 (2 bit 尾数)")
    for fmt, (e_bits, m_bits, fmax) in FORMATS.items():
        v = np.abs(rs.randn(100_000) * 100).astype(np.float32)
        v = v[(v >= 2.0 ** (2 - 2 ** (e_bits - 1))) & (v <= fmax)]     # 正规区
        worst = float(np.max(np.abs(fp8_round(v, fmt) - v) / v))
        print(f"  {fmt}: max={fmax:g}, 正规区最大相对误差 {worst:.4f} (上界 2^-{m_bits + 1} = {2.0 ** -(m_bits + 1):.4f}), "
              f"fp8({fmax * 2:g}) = {fp8_round(np.float32(fmax * 2), fmt):g} (饱和)")
        assert worst <= 2.0 ** -(m_bits + 1) and fp8_round(np.float32(fmax * 2), fmt) == fmax

    print("  C1 不加 scale 直接转: 权重幅度一小, E4M3 就掉进非正规区 (E5M2 范围大, 不受影响)")
    for std in (0.02, 0.002):
        Ws = (rs.randn(D_IN, D_OUT) * std).astype(np.float32)
        r = {f: (rel(Ws, fp8_round(Ws, f)), rel(Ws, fake_quant_fp8(Ws, f))) for f in FORMATS}
        print(f"  std={std:<6g} e4m3 直接转 {r['e4m3'][0]:.4f} / 加 scale {r['e4m3'][1]:.4f}    "
              f"e5m2 直接转 {r['e5m2'][0]:.4f} / 加 scale {r['e5m2'][1]:.4f}")
    assert r["e4m3"][0] > 5 * r["e4m3"][1] and r["e5m2"][0] < 1.1 * r["e5m2"][1]

    print("  C2 输出通道幅度相差悬殊 (列缩放 exp(1.5·N(0,1))): per-tensor vs per-channel scale, 权重相对误差")
    Wc = W * np.exp(rs.randn(D_OUT) * 1.5).astype(np.float32)
    rows = {"FP8 e4m3": [rel(Wc, fake_quant_fp8(Wc, "e4m3", ax)) for ax in (None, 0)],
            "FP8 e5m2": [rel(Wc, fake_quant_fp8(Wc, "e5m2", ax)) for ax in (None, 0)],
            "INT8": [rel(Wc, fake_quant_sym(Wc, 8, ax)) for ax in (None, 0)]}
    print(f"  {'格式':<10} {'per-tensor':>10} {'per-channel':>11} {'改善':>6}")
    for name, (t, c) in rows.items():
        print(f"  {name:<10} {t:>10.4f} {c:>11.4f} {t / c:>5.2f}x")
    (t4, c4), (t5, c5), (ti, ci) = rows.values()
    assert ti / ci > 5 and t4 / c4 < 1.2                  # INT8 靠 per-channel 救命; FP8 的格点自带伸缩, 几乎不在乎
    assert c4 < c5 and ci < c4                            # 同粒度: INT8 per-channel < E4M3 < E5M2 (分布无离群时均匀格点更准)

    print(f"  C3 W8A8, 激活 per-tensor 动态 scale (vLLM FP8 默认), 离群输入通道 × 50")
    X = rs.randn(256, D_IN).astype(np.float32)
    X[:, OUTLIERS] *= 50
    Y = X @ W
    r = {"FP8 e4m3": rel(Y, fake_quant_fp8(X, "e4m3") @ fake_quant_fp8(W, "e4m3", 0)),
         "FP8 e5m2": rel(Y, fake_quant_fp8(X, "e5m2") @ fake_quant_fp8(W, "e5m2", 0)),
         "INT8": rel(Y, fake_quant_sym(X, 8, None) @ fake_quant_sym(W, 8, 0))}
    print("  " + "   ".join(f"{k} {v:.4f}" for k, v in r.items()))
    assert r["FP8 e4m3"] < 0.7 * r["INT8"]               # 激活有离群值时, 浮点格点的大动态范围占优
    return rows, r


def main():
    banner("M25 - Weight quant: GPTQ / SmoothQuant W8A8 / FP8")
    rs = np.random.RandomState(0)
    W = (rs.randn(D_IN, D_OUT) * 0.05).astype(np.float32)
    kv("W", f"({D_IN},{D_OUT}), std 0.05")
    gptq_demo(rs, W)
    smoothquant_demo(rs, W)
    rows, _ = fp8_demo(rs, W)
    gain = {k: t / c for k, (t, c) in rows.items()}
    print("\n  ✓ GPTQ: 激活相关时输出误差远低于 RTN; iid 激活下打平 (校准集小还会更差)")
    print("  ✓ AWQ 保护重要通道、GPTQ 补偿误差, 两者叠加最好; SmoothQuant 让激活也能 INT8")
    print(f"  ✓ FP8: per-channel 只改善 {gain['FP8 e4m3']:.2f}x (INT8 是 {gain['INT8']:.1f}x); 激活有离群值时 FP8 per-tensor 优于 INT8")


if __name__ == "__main__":
    main()
