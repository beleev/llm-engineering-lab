"""
m11 demo — FlashAttention: 分块结果 == 朴素 attention, 工作集与 T 无关

运行: python -m llm_infer.m11_flash_attention.demo

四件事, 全部带 assert:
    1) O 与 core.dense_attention 对拍, lse 与朴素 logsumexp 对拍 (< 1e-5)
    2) causal 整块跳过: 统计 full / partial / skipped 块数
    3) 峰值工作集 b_q×b_k vs T² (由 stats 实测, 不是公式估算)
    4) 用 lse 合并两段部分 attention == 全量 attention (ring attention / chunked prefill 原语)
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention, causal_mask
from llm_infer.core.utils import banner
from llm_infer.m11_flash_attention.flash_attention import flash_attention, merge_attention

TOL = 1e-5            # fp32 舍入误差的量级
SHAPES = [(16, 16), (64, 64), (256, 256), (8, 64), (1, 100)]   # [1] 对拍的 (Tq, Tk)
BLOCKS = [(4, 4), (16, 8), (64, 32)]                            # [1] 对拍的 (b_q, b_k)
SHOW = (16, 8)        # [1] 只打印这一组块大小, 其余只计入最坏误差
B = 64                # [2][3] 的块大小 b_q = b_k


def naive_lse(Q, K, mask):
    """朴素 logsumexp: 先落地整张 (Tq,Tk) 分数矩阵。"""
    S = (Q @ K.T / np.sqrt(Q.shape[-1]) + mask).astype(np.float64)   # (Tq, Tk)
    m = S.max(-1)                                                    # (Tq,) 减最大值再 exp, 防溢出
    return m + np.log(np.exp(S - m[:, None]).sum(-1))


def main():
    banner("M11 - FlashAttention (tiling + online softmax + LSE)")
    rs = np.random.RandomState(0)
    D = 32
    rand = lambda *s: rs.randn(*s).astype(np.float32)

    print("\n[1] 数值对拍 vs core.dense_attention (Tq<Tk 即 chunked prefill / decode 形态)")
    print(f"  {'Tq':>5} {'Tk':>5} {'b_q':>4} {'b_k':>4} {'causal':>6}  {'max|ΔO|':>10}  {'max|Δlse|':>10}")
    worst = 0.0
    for Tq, Tk in SHAPES:
        Q, K, V = rand(Tq, D), rand(Tk, D), rand(Tk, D)
        for bq, bk in BLOCKS:
            for causal in (True, False):
                mask = causal_mask(Tq, Tk) if causal else np.zeros((Tq, Tk), np.float32)
                O, lse, _ = flash_attention(Q, K, V, bq, bk, causal)
                dO = np.max(np.abs(O - dense_attention(Q, K, V, mask)))
                dl = np.max(np.abs(lse - naive_lse(Q, K, mask)))
                worst = max(worst, dO, dl)
                if (bq, bk) == SHOW:
                    print(f"  {Tq:>5} {Tk:>5} {bq:>4} {bk:>4} {str(causal):>6}  {dO:>10.2e}  {dl:>10.2e}")
    print(f"  {len(SHAPES) * len(BLOCKS) * 2} 组配置的最坏误差 = {worst:.2e}")        # ×2: causal 开 / 关
    assert worst < TOL, f"分块结果 (O 和 lse) 必须等于朴素 attention, 最坏误差 {worst:.2e}"

    print(f"\n[2] causal 整块跳过 + [3] 峰值工作集 (b_q=b_k={B}, stats 实测)")
    print(f"  {'T':>5} {'full':>6} {'partial':>8} {'skipped':>8} {'跳过比例':>8}  {'peak b_q×b_k':>12} {'T²':>12} {'省':>7}")
    for T in [256, 1024, 4096]:
        Q, K, V = rand(T, D), rand(T, D), rand(T, D)
        O, _, st = flash_attention(Q, K, V, B, B, causal=True)
        n = T // B                                        # 每个方向的块数; 全图 n×n 块
        total = st["full"] + st["partial"] + st["skipped"]
        print(f"  {T:>5} {st['full']:>6} {st['partial']:>8} {st['skipped']:>8} {st['skipped'] / total:>8.1%}"
              f"  {st['peak_elems']:>12,} {T * T:>12,} {T * T // st['peak_elems']:>6}x")
        assert total == n * n, f"T={T}: 三类块数之和应为 {n}×{n}, 实际 {total}"
        assert st["skipped"] == n * (n - 1) // 2, f"T={T}: 对角线上方的 n(n-1)/2 块应整块跳过"
        assert st["partial"] == n, f"T={T}: 只有对角线上的 {n} 块需要逐元素 mask"
        assert st["peak_elems"] == B * B, f"T={T}: 峰值工作集应恒为 b_q×b_k = {B * B}, 与 T 无关"
        if T <= 1024:                                     # T=4096 的朴素基线要 16M 元素, 不跑
            assert np.max(np.abs(O - dense_attention(Q, K, V))) < TOL, f"T={T}: 分块结果必须等于朴素 attention"

    print("\n[4] 用 lse 合并两段部分 attention (ring attention / chunked prefill 原语)")
    Tk, Tq = 128, 32                                      # 最后 32 个 query, K/V 前后两半各 64 (想象在两张卡上)
    Q, K, V = rand(Tq, D), rand(Tk, D), rand(Tk, D)
    h = Tk // 2
    O1, l1, _ = flash_attention(Q, K[:h], V[:h], causal=False)   # 前半段全在过去 → 全可见
    O2, l2, _ = flash_attention(Q, K[h:], V[h:], causal=True)    # 后半段含 query 自己 → causal
    O_m, l_m = merge_attention(O1, l1, O2, l2)
    dO = np.max(np.abs(O_m - dense_attention(Q, K, V)))
    dl = np.max(np.abs(l_m - naive_lse(Q, K, causal_mask(Tq, Tk))))
    d_wrong = np.max(np.abs((O1 + O2) / 2 - dense_attention(Q, K, V)))
    print(f"  merge(O1,lse1,O2,lse2) vs 全量: max|ΔO| = {dO:.2e}, max|Δlse| = {dl:.2e}")
    print(f"  反例: 不用 lse, 直接 (O1+O2)/2  : max|ΔO| = {d_wrong:.2e}  ← 两段的 softmax 质量不相等")
    assert dO < TOL, f"用 lse 合并两段后的 O 应等于全量 attention, 实际差 {dO:.2e}"
    assert dl < TOL, f"合并后的 lse 应等于全量的 logsumexp, 实际差 {dl:.2e}"
    assert d_wrong > 100 * TOL, "反例: 不用 lse 直接平均两段输出, 结果应明显不同"

    print(f"\n  ✓ 分块输出与 lse 均与朴素实现一致 (< {TOL:g})")
    print(f"  ✓ causal 下 (n-1)/2n ≈ 一半的块整块跳过; 工作集恒为 b_q×b_k = {B * B:,} 个元素")
    print("  ✓ 只凭 (O, lse) 就能跨设备 / 跨 chunk 精确合并 attention")


if __name__ == "__main__":
    main()
