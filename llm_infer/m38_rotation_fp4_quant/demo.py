"""
m38 demo — 旋转消除离群值 + 推理侧 FP4 格式

    [A] 计算不变性: Hadamard 正交, (XQ)(QᵀW) 与 XW 一致; 旋转把每个 token 的 max/RMS 压下来
    [B] W4A4: 带离群通道的激活, 旋转前后的 per-token INT4 误差; 与 SmoothQuant (m25) 对照;
        B2: 离群值出现在哪个通道随 token 变化时, 静态的 SmoothQuant 缩放失效, 旋转不受影响
    [C] KV 4 bit: 每个头的 q、k 乘同一个 Hadamard, q·k 不变, K 的 per-token INT4 误差下降
    [D] INT4 分组 / MXFP4 / NVFP4: 权重和激活上的误差与每元素位数
    [E] QuIP# 的非相干处理: 权重两侧各乘一个随机 Hadamard, 尖峰被摊平, weight-only 低比特误差下降
    [F] 学习旋转 (SpinQuant 的单层简化): 随机 Hadamard 之间的差异, 以及 Cayley 更新能再降多少
数据全部人工合成: 离群通道位置与倍数、尖峰比例都打印出来。误差都是 fake quant 的相对误差, 不说明速度。

运行: python -m llm_infer.m38_rotation_fp4_quant.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m25_weight_quant.smoothquant import smooth_scales
from llm_infer.m25_weight_quant.gptq import gptq_quantize
from llm_infer.m38_rotation_fp4_quant.rotation_fp4 import (
    BITS, FORMATS, hadamard, random_hadamard, int4_sym, rel, w4a4_err, incoherence, learn_rotation,
)

N_DIM = 256                       # 隐藏维 (输入通道数), 2 的幂
OUTLIERS = [7, 50, 131, 200]      # 人工放大的离群输入通道
MULT = 20                         # 离群通道的放大倍数
N_TOK = 512                       # 校准 / 测试各多少个 token
W_STD = 0.05                      # 权重的初始化标准差
NORMAL = np.setdiff1d(np.arange(N_DIM), OUTLIERS)


def acts(rs, n=N_TOK, mult=MULT, spike_frac=0.0, spike=0.0):
    """激活 (n, N_DIM): 标准正态, 固定离群通道 × mult; spike_frac 比例的 token 在一个随机通道上额外出现 ±spike。"""
    X = rs.randn(n, N_DIM).astype(np.float32)
    X[:, OUTLIERS] *= mult
    hit = np.where(rs.rand(n) < spike_frac)[0]
    X[hit, rs.randint(0, N_DIM, len(hit))] = spike * np.sign(rs.randn(len(hit)))
    return X


def max_rms(X):
    """每个 token 的 max|x| / RMS(x) 的平均。per-token 量化的格距正比于它: 越大, 普通通道分到的格点越少。"""
    return float(np.mean(np.abs(X).max(1) / np.sqrt((X ** 2).mean(1))))


def invariance_demo(rs, X, W):
    """[A] 正交性与计算不变性。"""
    print("\n[A] 计算不变性: XW = (XQ)(QᵀW), Q 正交")
    H, Hr = hadamard(N_DIM), random_hadamard(N_DIM, rs)
    I = np.eye(N_DIM)
    e_orth = max(np.abs(H @ H.T - I).max(), np.abs(Hr @ Hr.T - I).max())
    e_inv = rel(X @ W, (X @ H) @ (H.T @ W))
    kv("max|QQᵀ − I| (H 与随机符号 H)", f"{e_orth:.1e}")
    kv("‖XW − (XH)(HᵀW)‖ / ‖XW‖", f"{e_inv:.1e}")
    before, after = max_rms(X), max_rms(X @ H)
    kv("每 token max/RMS: 旋转前 → 后", f"{before:.2f} → {after:.2f}  (纯高斯约 3)")
    assert e_orth < 1e-5, f"Hadamard 矩阵应正交, 实际 max|QQᵀ−I| = {e_orth:.1e}"
    assert e_inv < 1e-5, f"旋转前后矩阵乘结果应一致 (只差浮点舍入), 实际相对误差 {e_inv:.1e}"
    assert before > 3 * after, f"旋转应把每个 token 的 max/RMS 压到 1/3 以下: {before:.2f} → {after:.2f}"
    return H


def w4a4_demo(rs, H, X, Xt, W):
    """[B] per-token INT4 激活 + per-channel INT4 权重。"""
    print(f"\n[B] W4A4: 激活 per-token INT4, 权重 per-output-channel INT4 (都是对称 RTN); 离群通道 {OUTLIERS} × {MULT}")
    qX, qXH = int4_sym(Xt, axis=1), int4_sym(Xt @ H, axis=1) @ H.T          # 后者在原坐标系里比较
    e_n, e_r = rel(Xt[:, NORMAL], qX[:, NORMAL]), rel(Xt[:, NORMAL], qXH[:, NORMAL])
    kv("普通通道的激活误差: 不处理 / 旋转", f"{e_n:.3f} / {e_r:.3f}  (1.0 = 全被舍成 0)")
    s = smooth_scales(X, W, 0.5)                                             # m25 的 SmoothQuant 缩放
    Y = Xt @ W
    res = {
        "不处理": w4a4_err(Xt, W),
        "SmoothQuant α=0.5": rel(Y, int4_sym(Xt / s, axis=1) @ int4_sym(W * s[:, None], axis=0)),
        "Hadamard 旋转": w4a4_err(Xt, W, H),
        "旋转 + GPTQ 权重": rel(Y, int4_sym(Xt @ H, axis=1) @ gptq_quantize(H.T @ W, X @ H, 4, N_DIM)),
    }
    for name, e in res.items():
        print(f"  {name:<18} W4A4 输出误差 {e:.4f}")
    e0, e_sq, e_h, e_hg = res.values()
    assert e_r < 0.4 * e_n, f"旋转后普通通道的激活误差应降到 40% 以下: {e_n:.3f} → {e_r:.3f}"
    assert e_h < 0.6 * e0, f"旋转后 W4A4 输出误差应降到不处理的 60% 以下: {e0:.4f} → {e_h:.4f}"
    assert e_h < e_sq, f"固定离群通道下, 旋转的 W4A4 误差应低于 SmoothQuant: {e_h:.4f} vs {e_sq:.4f}"
    assert e_hg < 0.8 * e_h, f"权重换成 GPTQ (QuaRot 的默认) 应再降 20% 以上: {e_h:.4f} → {e_hg:.4f}"

    frac, spike = 0.1, 60.0
    print(f"  B2 没有固定离群通道, 但 {frac:.0%} 的 token 在一个随机通道上出现 ±{spike:g}")
    Xs, Xst = acts(rs, mult=1, spike_frac=frac, spike=spike), acts(rs, mult=1, spike_frac=frac, spike=spike)
    s2 = smooth_scales(Xs, W, 0.5)
    r2 = {"不处理": w4a4_err(Xst, W),
          "SmoothQuant α=0.5": rel(Xst @ W, int4_sym(Xst / s2, axis=1) @ int4_sym(W * s2[:, None], axis=0)),
          "Hadamard 旋转": w4a4_err(Xst, W, H)}
    print("  " + "   ".join(f"{k} {v:.4f}" for k, v in r2.items()))
    b0, b_sq, b_h = r2.values()
    assert b_sq > b0, f"B2: 校准集里的随机尖峰撑大了很多通道的 max, 静态 SmoothQuant 应比不处理更差: {b_sq:.4f} vs {b0:.4f}"
    assert b_h < 0.8 * b0, f"B2: 旋转不依赖校准统计, 误差应比不处理低 20% 以上: {b0:.4f} → {b_h:.4f}"
    return res


def kv_demo(rs):
    """[C] K 的 per-token INT4, 每个头的 q、k 先乘同一个 Hadamard (QuaRot 在 RoPE 之后做)。"""
    d, T = 64, 256                                                           # 一个头的维度, 上下文长度
    print(f"\n[C] KV 4 bit: 单头 d={d}, T={T}, K 的通道 3 和 40 × 15; 打分 = q·k")
    K = rs.randn(T, d).astype(np.float32)
    K[:, [3, 40]] *= 15
    q = rs.randn(8, d).astype(np.float32)
    Hd = hadamard(d)
    S = q @ K.T
    e_inv = rel(S, (q @ Hd) @ (K @ Hd).T)
    e_raw, e_rot = rel(S, q @ int4_sym(K, axis=1).T), rel(S, (q @ Hd) @ int4_sym(K @ Hd, axis=1).T)
    kv("q·k 旋转前后的相对差", f"{e_inv:.1e}")
    kv("K per-token INT4 后打分误差: 不旋转 / 旋转", f"{e_raw:.4f} / {e_rot:.4f}")
    assert e_inv < 1e-5, f"q、k 乘同一个正交矩阵, 打分应不变, 实际差 {e_inv:.1e}"
    assert e_rot < 0.5 * e_raw, f"旋转后 K 的 4 bit 打分误差应降到一半以下: {e_raw:.4f} → {e_rot:.4f}"


def format_demo(H, Xt, W):
    """[D] 四种 4 bit 格式, 都沿最后一维分组; 激活看普通通道在原坐标系里的误差。"""
    print("\n[D] 4 bit 格式: 每元素位数含 scale; 激活误差只算普通通道 (离群通道本身在哪种格式下都量得准)")
    print(f"  {'格式':<10} {'bit/元素':>8} {'权重':>8} {'激活 不旋转':>11} {'激活 旋转':>9} {'旋转改善':>8}")
    XH, rows = Xt @ H, {}
    for name, f in FORMATS.items():
        ew = rel(W, f(W.T).T)                                                # 权重沿输入维分组: 转置后最后一维是输入维
        ea = rel(Xt[:, NORMAL], f(Xt)[:, NORMAL])
        ear = rel(Xt[:, NORMAL], (f(XH) @ H.T)[:, NORMAL])
        rows[name] = (ew, ea, ear)
        print(f"  {name:<10} {BITS[name]:>9.3f} {ew:>9.4f} {ea:>12.4f} {ear:>10.4f} {ea / ear:>9.2f}x")
    assert BITS["MXFP4"] == 4.25 and BITS["NVFP4"] == 4.5, "MXFP4 = 4 + 8/32, NVFP4 = 4 + 8/16"
    assert rows["NVFP4"][0] < rows["MXFP4"][0], "权重: NVFP4 的 E4M3 scale 能对准 amax, 误差应低于 2 的幂 scale 的 MXFP4"
    g128 = rows["INT4 g128"][1] / rows["INT4 g128"][2]
    nv = rows["NVFP4"][1] / rows["NVFP4"][2]
    assert g128 > 2, f"激活: 128 个一组时离群值毁掉整组, 旋转应改善 2 倍以上, 实际 {g128:.2f}x"
    assert nv < 1.2, f"激活: NVFP4 每 16 个一组, 离群值本来就被关在小组里, 旋转改善应不到 1.2 倍, 实际 {nv:.2f}x"
    return rows


def incoherence_demo(rs, Xt):
    """[E] 权重里有少量尖峰: 两侧随机 Hadamard (QuIP#) 后再做 weight-only per-channel 量化。"""
    W = (rs.randn(N_DIM, N_DIM) * W_STD).astype(np.float32)
    spikes = rs.rand(N_DIM, N_DIM) < 0.002
    W[spikes] *= 25
    print(f"\n[E] 非相干处理: 权重里 {spikes.mean():.1%} 的元素 × 25; W' = UᵀWV, U、V 是随机 Hadamard")
    U, V = random_hadamard(N_DIM, rs), random_hadamard(N_DIM, rs)
    Wi = U.T @ W @ V
    mu0, mu1 = incoherence(W), incoherence(Wi)
    kv("非相干度 μ = max|W|·√(mn)/‖W‖_F", f"{mu0:.1f} → {mu1:.1f}")
    Y = Xt @ W
    out = {}
    for b in (4, 3):
        def q(M):                                                            # 对称 b bit, per-output-channel
            qmax = 2 ** (b - 1) - 1
            s = np.maximum(np.abs(M).max(0, keepdims=True), 1e-12) / qmax
            return np.clip(np.round(M / s), -qmax, qmax) * s
        out[b] = (rel(Y, Xt @ q(W)), rel(Y, Xt @ (U @ q(Wi) @ V.T)))       # 推理时 Xt U · q(W') · Vᵀ
        kv(f"INT{b} weight-only 输出误差: 直接 / 非相干处理后", f"{out[b][0]:.4f} / {out[b][1]:.4f}")
    assert mu0 > 5 * mu1, f"两侧旋转应把非相干度降到 1/5 以下: {mu0:.1f} → {mu1:.1f}"
    assert out[4][1] < 0.5 * out[4][0], f"INT4: 非相干处理后误差应降到一半以下: {out[4]}"


def learned_demo(X, Xt, W):
    """[F] 20 个随机 Hadamard (只换符号) 与随机正交矩阵的 W4A4 误差, 再从其中一个出发学习旋转。"""
    print("\n[F] 选哪个旋转: 随机 Hadamard (20 组符号) vs 随机正交矩阵 vs 学习的旋转 (W4A4 输出误差)")
    hs = [random_hadamard(N_DIM, np.random.RandomState(100 + i)) for i in range(20)]
    e_h = np.array([w4a4_err(Xt, W, Q) for Q in hs])
    G = np.linalg.qr(np.random.RandomState(99).randn(N_DIM, N_DIM))[0].astype(np.float32)
    e_g = w4a4_err(Xt, W, G)
    kv("随机 Hadamard: 最好 / 平均 / 最差", f"{e_h.min():.4f} / {e_h.mean():.4f} / {e_h.max():.4f}")
    kv("随机正交矩阵 (QR)", f"{e_g:.4f}")
    Q0 = hs[int(np.argmax([w4a4_err(X, W, Q) for Q in hs]))]                # 从校准集上最差的那个出发
    Ql = learn_rotation(X, W, Q0)
    c0, c1, t0, t1 = w4a4_err(X, W, Q0), w4a4_err(X, W, Ql), w4a4_err(Xt, W, Q0), w4a4_err(Xt, W, Ql)
    e_orth = float(np.abs(Ql @ Ql.T - np.eye(N_DIM)).max())
    kv("学习旋转 (60 步 Cayley): 校准集", f"{c0:.4f} → {c1:.4f}")
    kv("学习旋转: held-out", f"{t0:.4f} → {t1:.4f}  (降 {1 - t1 / t0:.1%}); max|QQᵀ−I| = {e_orth:.1e}")
    assert e_orth < 1e-4, f"Cayley 更新应保持正交, 实际 max|QQᵀ−I| = {e_orth:.1e}"
    assert e_g > e_h.mean(), f"随机正交矩阵的误差应高于随机 Hadamard 的平均: {e_g:.4f} vs {e_h.mean():.4f}"
    assert t1 < t0, f"学习后的旋转在 held-out 上应比起点更好: {t0:.4f} → {t1:.4f}"
    return e_h, t0, t1


def main():
    banner("M38 - 旋转消除离群值 + FP4 推理格式")
    rs = np.random.RandomState(0)
    W = (rs.randn(N_DIM, N_DIM) * W_STD).astype(np.float32)
    X, Xt = acts(rs), acts(rs)                                              # 校准 / 测试激活
    kv("W / 激活", f"({N_DIM},{N_DIM}) std {W_STD:g} / {N_TOK} token × {N_DIM}, 离群通道 {OUTLIERS} × {MULT}")
    H = invariance_demo(rs, X, W)
    w4a4_demo(rs, H, X, Xt, W)
    kv_demo(rs)
    format_demo(H, Xt, W)
    incoherence_demo(rs, Xt)
    e_h, _, t1 = learned_demo(X, Xt, W)
    print("\n  ✓ 旋转不改变矩阵乘结果, 只改变数值分布: 离群值被摊平, per-token INT4 和粗分组的误差大幅下降")
    print("  ✓ 小 block 的 FP4 (NVFP4) 本身就能关住离群值, 旋转收益小; MXFP4 的 2 的幂 scale 比 NVFP4 粗")
    print(f"  ✓ 这一层上学习旋转比随机 Hadamard 的平均好 {1 - t1 / e_h.mean():.0%}; 整个网络上差多少要另外量")


if __name__ == "__main__":
    main()
