"""
m26 demo — Flash-Decoding: split-K 的结果 == 普通 attention; 小 batch 长上下文时把闲着的 SM 用起来

    [1] 数值对拍: 各种 (B,H,T,S) 下 O 与 core.dense_attention、lse 与朴素 logsumexp 一致 (< 1e-5)
    [2] 延迟估算 (代价模型, 不是实测): B=1, H=32, T=32k 时扫 S, 看 SM 利用率与延迟
    [3] 延迟估算: 不同 (B, T) 下最优 S 相对 S=1 的加速比 —— 小 batch + 长上下文收益最大, 大 batch 无收益

运行: python -m llm_infer.m26_flash_decoding.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention
from llm_infer.core.utils import banner, kv
from llm_infer.m26_flash_decoding.flash_decoding import (
    flash_decoding, decode_latency_us, N_SM, HBM_GBPS, KV_BYTES_PER_TOKEN, LAUNCH_US,
)

TOL = 1e-5                                 # fp32 舍入误差的量级
SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)     # [2][3] 里要扫的 split 数 S


def best_split(B, H, T):
    """代价模型下延迟最低的 S。"""
    return min(SPLITS, key=lambda s: decode_latency_us(B, H, T, s)[0])


def main():
    banner("M26 - Flash-Decoding (split-K + LSE reduce)")
    rs = np.random.RandomState(0)

    print("\n[1] 数值对拍 vs core.dense_attention (decode: 每条序列 1 个 query, 看全部 T 个 key)")
    print(f"  {'B':>3} {'H':>3} {'T':>6} {'S':>5}  {'max|ΔO|':>10}  {'max|Δlse|':>10}")
    worst = 0.0
    for B, H, T in [(1, 1, 1), (2, 4, 100), (1, 8, 4096), (4, 2, 777)]:
        q = rs.randn(B, H, 1, 64).astype(np.float32)
        K, V = rs.randn(B, H, T, 64).astype(np.float32), rs.randn(B, H, T, 64).astype(np.float32)
        ref = dense_attention(q, K, V)                                   # (B,H,1,64); Tq=1 的 causal 就是全可见
        # 朴素 lse: 先落地整行分数。8 = √d, d=64
        S_full = (q @ np.swapaxes(K, -1, -2) / 8.0).astype(np.float64)  # (B,H,1,T)
        ref_lse = np.log(np.exp(S_full - S_full.max(-1, keepdims=True)).sum(-1, keepdims=True)) + S_full.max(-1, keepdims=True)
        for S in sorted({1, 3, 16, T} & set(range(1, T + 1))):          # S 不整除 T、S = T (每段 1 个 key) 都要对
            O, lse = flash_decoding(q, K, V, S)
            dO, dl = float(np.max(np.abs(O - ref))), float(np.max(np.abs(lse - ref_lse)))
            worst = max(worst, dO, dl)
            print(f"  {B:>3} {H:>3} {T:>6} {S:>5}  {dO:>10.2e}  {dl:>10.2e}")
    kv("最坏误差", f"{worst:.2e}")
    assert worst < TOL, f"split-K 再按 lse 合并的结果 (O 和 lse) 必须等于普通 attention, 最坏误差 {worst:.2e}"

    print("\n  以下 [2][3] 全部来自代价模型 (cost model), 不是实测。模型参数:")
    kv("SM 数 / HBM 带宽", f"{N_SM} / {HBM_GBPS:g} GB/s (A100); 每个 SM 最多 1/{N_SM} 的带宽")
    kv("每 token 每 head 的 KV 字节", f"{KV_BYTES_PER_TOKEN} (d=128, fp16 K+V)")
    kv("kernel 固定开销", f"{LAUNCH_US} μs × (1 个主 kernel + S>1 时 1 个 reduce kernel)")

    B, H, T = 1, 32, 32768
    print(f"\n[2] B={B}, H={H}, T={T:,}: 并行单位 = B×H×S")
    print(f"  {'S':>4} {'units':>6} {'waves':>6} {'SM 利用率':>9} {'延迟(估算)':>11}")
    lat = {}
    for S in SPLITS:
        lat[S], util = decode_latency_us(B, H, T, S)
        units = B * H * S
        print(f"  {S:>4} {units:>6} {-(-units // N_SM):>6} {util:>9.0%} {lat[S]:>8.1f} μs")
    s_best = min(lat, key=lat.get)
    kv("最优 S / 加速比", f"{s_best} / {lat[1] / lat[s_best]:.2f}x  (上限 N_SM/(B·H) = {N_SM / (B * H):.2f}x: 最多把闲着的 SM 全用上)")
    assert lat[1] / lat[s_best] <= N_SM / (B * H), "加速比不应超过 N_SM/(B·H): 最多把闲着的 SM 全用上"
    assert decode_latency_us(B, H, T, 1)[1] < 0.3, "S=1 时只有 32 个 block, SM 利用率应低于 30%"
    assert lat[1] / lat[s_best] > 2.5, f"B=1, T=32k 下最优 S 的加速比应超过 2.5×, 实际 {lat[1] / lat[s_best]:.2f}×"

    print(f"\n[3] 最优 S 相对 S=1 的加速比 (H={H})")
    Ts = (512, 4096, 32768, 131072)
    print(f"  {'B':>4} " + " ".join(f"{'T=' + format(t, ','):>14}" for t in Ts))
    speed = {}
    Bs = (1, 4, 16, 64)
    for B in Bs:
        cells = []
        for T in Ts:
            s = best_split(B, H, T)
            speed[B, T] = decode_latency_us(B, H, T, 1)[0] / decode_latency_us(B, H, T, s)[0]
            cells.append(f"{speed[B, T]:.2f}x (S={s})")
        print(f"  {B:>4} " + " ".join(f"{c:>14}" for c in cells))
    assert speed[1, 131072] > 2.5, "B=1, T=128k: 加速比应超过 2.5×"
    assert all(speed[1, a] <= speed[1, b] for a, b in zip(Ts, Ts[1:])), \
        "B=1: 上下文越长加速比应越高 (两次 kernel 的固定开销被摊薄)"
    assert speed[1, 512] < 0.6 * speed[1, 131072], "短上下文 (T=512): 固定开销应吃掉一大块收益"
    assert all(speed[4, T] < speed[1, T] for T in Ts), "batch 越大, 本来闲着的 SM 越少, 加速比应越低"
    assert all(speed[64, T] < 1.1 for T in Ts), "B=64 时 B×H=2048 早就填满 SM, split 应基本无收益"

    print(f"\n  ✓ split-K + lse reduce 与普通 attention 数值一致 (< {TOL:g}), 任意 S")
    print(f"  ✓ (估算) B={Bs[0]}, T={Ts[-1] // 1024}k: {speed[Bs[0], Ts[-1]]:.2f}x; "
          f"B={Bs[-1]}: ≤ {max(speed[Bs[-1], T] for T in Ts):.2f}x —— 收益只在 SM 填不满时")


if __name__ == "__main__":
    main()
