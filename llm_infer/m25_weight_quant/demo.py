"""
m25 demo — 另外三条量化路线: GPTQ (补偿误差) / SmoothQuant W8A8 (压平激活) / FP8 (换格点)

[A] GPTQ vs RTN (INT4 group=32): 同一组权重、同一批校准激活; 误差在 held-out 激活上报告。
    A1 激活通道相关 → GPTQ 明显赢;  A2 激活 iid → 无相关可用, 校准集小时反而过拟合;
    A3 相关 + 离群通道 → AWQ 与 GPTQ 各治一半, 叠加最好 (与 m08 的分工)。
[B] SmoothQuant W8A8: 激活离群通道让 per-token INT8 失真; 把缩放迁到权重后下降; 扫 α。
[C] FP8 E4M3 / E5M2: 舍入正确性; per-tensor vs per-channel scale (与 INT8 对照); W8A8 激活 per-tensor。
数据全部人工合成 (相关结构、离群通道位置与倍数都打印出来)。

运行: python -m llm_infer.m25_weight_quant.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m08_quantization.int4_awq import quant_dequant, output_err, awq_quantize
from llm_infer.m25_weight_quant.gptq import gptq_quantize
from llm_infer.m25_weight_quant.smoothquant import fake_quant_sym, w8a8_matmul, smooth_scales
from llm_infer.m25_weight_quant.fp8 import FORMATS, fp8_round, fake_quant_fp8

D_IN, D_OUT, G = 256, 256, 32         # 权重形状 (D_IN, D_OUT); G = INT4 的分组大小
OUTLIERS = [7, 50, 131, 200]          # 人工放大的离群输入通道
W_STD = 0.05                          # 权重的初始化标准差
ACT_RANK, ACT_NOISE = 32, 0.3         # 相关激活: 公共成分的秩, 独立噪声的幅度


def rel(A, B):
    """相对误差 ‖A − B‖ / ‖A‖, A 是基准。"""
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def correlated_acts(rs, n, rank=ACT_RANK, noise=ACT_NOISE):
    """X = 低秩公共成分 + 独立噪声: 输入通道之间强相关 (真实 LLM 激活的典型形态, H = XᵀX 远非对角)。"""
    return (rs.randn(n, rank) @ rs.randn(rank, D_IN) / np.sqrt(rank) + noise * rs.randn(n, D_IN)).astype(np.float32)


def gptq_demo(rs, W):
    """[A] 三种激活分布下比较 RTN / GPTQ / AWQ。返回 A1 的 (RTN 误差, GPTQ 误差)。"""
    N = 512                                               # 校准样本数, 另有同样多的 held-out 样本
    print(f"\n[A] GPTQ vs RTN: INT4 group={G} 非对称, 存储格式相同; 误差 = ‖XW−XŴ‖/‖XW‖")
    print(f"  A1 激活通道相关 (rank-{ACT_RANK} 公共成分 + {ACT_NOISE:g} 噪声)")
    X = correlated_acts(rs, 2 * N)
    Xc, Xt = X[:N], X[N:]
    W_rtn, W_gptq = quant_dequant(W, 4, G), gptq_quantize(W, Xc, 4, G)
    print(f"  {'方案':<12} {'calib 误差':>10} {'held-out 误差':>13}")
    for name, W_hat in (("RTN", W_rtn), ("GPTQ", W_gptq)):
        print(f"  {name:<12} {output_err(Xc, W, W_hat):>10.4f} {output_err(Xt, W, W_hat):>13.4f}")
    e_rtn, e_gptq = output_err(Xt, W, W_rtn), output_err(Xt, W, W_gptq)
    assert e_gptq < 0.6 * e_rtn, f"A1 激活相关: GPTQ 的输出误差应比 RTN 至少降 40%: {e_gptq:.4f} vs {e_rtn:.4f}"

    print("  A2 激活 iid (H ≈ 对角, 没有相关性可补偿): 扫校准样本数 N")
    X_test = rs.randn(2048, D_IN).astype(np.float32)
    e_rtn_iid = output_err(X_test, W, W_rtn)
    gaps = {}
    for n in (256, 512, 2048, 8192):
        e = output_err(X_test, W, gptq_quantize(W, rs.randn(n, D_IN).astype(np.float32), 4, G))
        gaps[n] = e / e_rtn_iid
        print(f"  N={n:<5} GPTQ {e:.4f}  vs RTN {e_rtn_iid:.4f}  ({gaps[n]:.2f}x)")
    # gaps[n] = GPTQ 误差 / RTN 误差; > 1 表示 GPTQ 更差
    assert gaps[256] > 1.2, "A2 校准集小: GPTQ 把噪声当相关性去补偿, held-out 误差应比 RTN 差 20% 以上"
    assert gaps[8192] < 1.05, "A2 校准集够大: GPTQ 应与 RTN 基本打平"
    assert gaps[8192] < gaps[2048] < gaps[512] < gaps[256], "A2: 校准样本越多, GPTQ 相对 RTN 的劣势应越小"

    mult = 30                                             # 离群通道的放大倍数
    print(f"  A3 相关激活 + 离群输入通道 × {mult} (AWQ 的主场)")
    X = correlated_acts(rs, 2 * N)
    X[:, OUTLIERS] *= mult
    Xc, Xt = X[:N], X[N:]
    W_awq, alpha, s, _ = awq_quantize(W, Xc, 4, G)
    W_both = gptq_quantize(W * s[:, None], Xc / s, 4, G) / s[:, None]    # 先 AWQ 缩放, 再对缩放后的问题做 GPTQ
    res = {name: output_err(Xt, W, W_hat) for name, W_hat in
           (("RTN", quant_dequant(W, 4, G)), (f"AWQ (α={alpha:g})", W_awq), ("GPTQ", gptq_quantize(W, Xc, 4, G)),
            ("AWQ + GPTQ", W_both))}
    for name, e in res.items():
        print(f"  {name:<14} {e:.4f}")
    e_r, e_a, e_g, e_b = res.values()
    assert max(e_a, e_g) < 0.5 * e_r, "A3: AWQ 和 GPTQ 各自的误差都应不到 RTN 的一半"
    assert e_b < 0.6 * min(e_a, e_g), "A3: AWQ + GPTQ 叠加后, 误差应比两者中较好的再降 40% 以上"
    return e_rtn, e_gptq


def smoothquant_demo(rs, W):
    """[B] 扫 α, 把 W8A8 的误差拆成激活贡献和权重贡献。返回 (不平滑的误差, 最优 α, 最优误差)。"""
    N, mult = 256, 50.0                                   # 校准 / 测试样本数, 离群通道放大倍数
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
    # errs 的三个分量: [0] 只量化激活, [1] 只量化权重, [2] 都量化
    assert base[0] > 3 * base[1], "不平滑时误差应几乎全来自激活: 只量化 X 的误差超过只量化 W 的 3 倍"
    assert 0 < best < 1, f"最优 α 应在 0 和 1 之间, 实际 {best}"
    assert sweep[best][2] < 0.4 * base[2], "最优 α 下 W8A8 误差应降到不平滑时的 40% 以下"
    assert sweep[1.0][1] > 3 * sweep[best][1], "α=1 把离群值全压到权重上, 权重的量化误差应反过来变大"
    return base[2], best, sweep[best][2]


def fp8_demo(rs, W):
    """[C] FP8 的舍入正确性, 以及 scale 粒度对 FP8 和 INT8 的不同影响。"""
    print("\n[C] FP8: E4M3 (3 bit 尾数) / E5M2 (2 bit 尾数)")
    for fmt, (e_bits, m_bits, fmax) in FORMATS.items():
        v = np.abs(rs.randn(100_000) * 100).astype(np.float32)
        v = v[(v >= 2.0 ** (2 - 2 ** (e_bits - 1))) & (v <= fmax)]     # 正规区
        worst = float(np.max(np.abs(fp8_round(v, fmt) - v) / v))
        print(f"  {fmt}: max={fmax:g}, 正规区最大相对误差 {worst:.4f} (上界 2^-{m_bits + 1} = {2.0 ** -(m_bits + 1):.4f}), "
              f"fp8({fmax * 2:g}) = {fp8_round(np.float32(fmax * 2), fmt):g} (饱和)")
        assert worst <= 2.0 ** -(m_bits + 1), \
            f"{fmt}: 正规区的相对舍入误差应不超过 2^-{m_bits + 1}, 实际 {worst:.4f}"
        assert fp8_round(np.float32(fmax * 2), fmt) == fmax, f"{fmt}: 超出范围的值应饱和到 {fmax:g}, 不产生 inf"

    print("  C1 不加 scale 直接转: 权重幅度一小, E4M3 就掉进非正规区 (E5M2 范围大, 不受影响)")
    for std in (0.02, 0.002):
        Ws = (rs.randn(D_IN, D_OUT) * std).astype(np.float32)
        r = {f: (rel(Ws, fp8_round(Ws, f)), rel(Ws, fake_quant_fp8(Ws, f))) for f in FORMATS}
        print(f"  std={std:<6g} e4m3 直接转 {r['e4m3'][0]:.4f} / 加 scale {r['e4m3'][1]:.4f}    "
              f"e5m2 直接转 {r['e5m2'][0]:.4f} / 加 scale {r['e5m2'][1]:.4f}")
    # r 是循环最后一轮 (std=0.002) 的结果: (直接转的误差, 加 scale 的误差)
    assert r["e4m3"][0] > 5 * r["e4m3"][1], "C1 std=0.002: E4M3 不加 scale 会掉进非正规区, 误差应超过加 scale 的 5 倍"
    assert r["e5m2"][0] < 1.1 * r["e5m2"][1], "C1 std=0.002: E5M2 范围大, 加不加 scale 误差应差不多"

    print("  C2 输出通道幅度相差悬殊 (列缩放 exp(1.5·N(0,1))): per-tensor vs per-channel scale, 权重相对误差")
    Wc = W * np.exp(rs.randn(D_OUT) * 1.5).astype(np.float32)
    rows = {"FP8 e4m3": [rel(Wc, fake_quant_fp8(Wc, "e4m3", ax)) for ax in (None, 0)],
            "FP8 e5m2": [rel(Wc, fake_quant_fp8(Wc, "e5m2", ax)) for ax in (None, 0)],
            "INT8": [rel(Wc, fake_quant_sym(Wc, 8, ax)) for ax in (None, 0)]}
    print(f"  {'格式':<10} {'per-tensor':>10} {'per-channel':>11} {'改善':>6}")
    for name, (t, c) in rows.items():
        print(f"  {name:<10} {t:>10.4f} {c:>11.4f} {t / c:>5.2f}x")
    (t4, c4), (t5, c5), (ti, ci) = rows.values()
    assert ti / ci > 5, "C2: INT8 从 per-tensor 换成 per-channel, 误差应降到 1/5 以下"
    assert t4 / c4 < 1.2, "C2: FP8 的格点随数值大小伸缩, per-channel 带来的改善应不到 20%"
    assert c4 < c5, "C2 per-channel: E4M3 多 1 bit 尾数, 误差应小于 E5M2"
    assert ci < c4, "C2 per-channel: 分布无离群时均匀格点更准, INT8 误差应小于 E4M3"

    mult = 50                                             # 离群通道的放大倍数
    print(f"  C3 W8A8, 激活 per-tensor 动态 scale (vLLM FP8 默认), 离群输入通道 × {mult}")
    X = rs.randn(256, D_IN).astype(np.float32)
    X[:, OUTLIERS] *= mult
    Y = X @ W
    r = {"FP8 e4m3": rel(Y, fake_quant_fp8(X, "e4m3") @ fake_quant_fp8(W, "e4m3", 0)),
         "FP8 e5m2": rel(Y, fake_quant_fp8(X, "e5m2") @ fake_quant_fp8(W, "e5m2", 0)),
         "INT8": rel(Y, fake_quant_sym(X, 8, None) @ fake_quant_sym(W, 8, 0))}
    print("  " + "   ".join(f"{k} {v:.4f}" for k, v in r.items()))
    assert r["FP8 e4m3"] < 0.7 * r["INT8"], \
        "C3: 激活有离群值且用 per-tensor scale 时, FP8 E4M3 的误差应不到 INT8 的 70%"
    return rows, r


def main():
    banner("M25 - Weight quant: GPTQ / SmoothQuant W8A8 / FP8")
    rs = np.random.RandomState(0)
    W = (rs.randn(D_IN, D_OUT) * W_STD).astype(np.float32)
    kv("W", f"({D_IN},{D_OUT}), std {W_STD:g}")
    gptq_demo(rs, W)
    smoothquant_demo(rs, W)
    rows, _ = fp8_demo(rs, W)
    gain = {k: t / c for k, (t, c) in rows.items()}
    print("\n  ✓ GPTQ: 激活相关时输出误差远低于 RTN; iid 激活下打平 (校准集小还会更差)")
    print("  ✓ AWQ 保护重要通道、GPTQ 补偿误差, 两者叠加最好; SmoothQuant 让激活也能 INT8")
    print(f"  ✓ FP8: per-channel 只改善 {gain['FP8 e4m3']:.2f}x (INT8 是 {gain['INT8']:.1f}x); 激活有离群值时 FP8 per-tensor 优于 INT8")


if __name__ == "__main__":
    main()
