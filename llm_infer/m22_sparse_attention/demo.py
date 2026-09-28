"""
m22 demo — 稀疏 attention decode: block 打分选 top-k vs 随机选块 vs 全量

    [1] Quest 上界的正确性: 每个 block 的 bound ≥ 该 block 内真实 max q·k
    [2] "针" 负载 (attention 真的集中): 误差 / attention 质量召回 随 "读取的 KV 比例" 的变化
    [3] 真实 TinyLM 的 KV (随机权重 → attention 弥散): 同样的表, 打分相对随机的优势变小

运行: python -m llm_infer.m22_sparse_attention.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import TinyLM, ModelConfig, rms_norm
from llm_infer.core.tiny_model import block_forward, apply_rope
from llm_infer.core.utils import banner, kv
from llm_infer.m22_sparse_attention.sparse_attention import (
    quest_upper_bound, mean_score, select_blocks, random_blocks, evaluate, needle_context, block_view,
)

BS = 16           # block 大小 (每 block 几个 token)
N_RANDOM = 5      # 随机对照取几个种子的平均
TOP = 8           # [2] 检查 quest 选的前 TOP 个 block 是否覆盖全部 needle


def sweep(cases, ks):
    """cases: [(q, K, V)]; 对每个 k 返回 {method: (mean err, mean recall)}, 在 cases 上取平均。"""
    rs = np.random.RandomState(0)
    table = {}
    for k in ks:
        acc = {"quest": [], "mean": [], "random": []}
        for q, K, V in cases:
            nb = K.shape[0] // BS
            acc["quest"].append(evaluate(q, K, V, select_blocks(quest_upper_bound(q, K, BS), k), BS))
            acc["mean"].append(evaluate(q, K, V, select_blocks(mean_score(q, K, BS), k), BS))
            acc["random"] += [evaluate(q, K, V, random_blocks(nb, k, rs), BS) for _ in range(N_RANDOM)]
        table[k] = {m: np.mean(v, axis=0) for m, v in acc.items()}
    return table


def show(table, nb):
    """把 sweep 的结果打印成表。KV 读取 = k / nb, 即读了多大比例的 KV; 召回 = 选中 block 覆盖的 attention 质量。"""
    # 中文字符占两格, 表头的格式宽度按显示宽度少补几格, 和下面的数字列对齐
    print(f"  {'k':>4}{'KV 读取':>7} | {'quest 误差':>8}{'召回':>6} | {'mean 误差':>8}{'召回':>6} | "
          f"{'random 误差':>9}{'召回':>6}")
    for k, row in table.items():
        print(f"  {k:>4}{k / nb:>9.1%} | {row['quest'][0]:>10.4f}{row['quest'][1]:>8.1%} | "
              f"{row['mean'][0]:>10.4f}{row['mean'][1]:>8.1%} | {row['random'][0]:>11.4f}{row['random'][1]:>8.1%}")


def tinylm_qkv(T: int):
    """真实模型的 (q, K, V): 长随机序列 prefill, 取每层最后一个位置的 query 和整层 post-RoPE KV。"""
    lm = TinyLM(ModelConfig(max_seq_len=T))
    x = lm.w.tok_emb[np.random.RandomState(0).randint(0, lm.cfg.vocab_size, T)]      # (T, D)
    cases = []
    for layer in lm.w.layers:
        q = apply_rope(rms_norm(x, layer.norm1_g) @ layer.wq, lm.cos, lm.sin)[-1]   # (D,) 与 attn_forward 内部同一个 q
        x, (K, V) = block_forward(x, layer, lm.cos, lm.sin)
        cases.append((q, K, V))
    return cases


def main():
    banner("M22 - 稀疏 attention decode (Quest / NSA / DSA 式)")
    q, K, V, needles = needle_context(T=4096, d=64, bs=BS)
    nb = K.shape[0] // BS
    kv("needle 上下文", f"T={K.shape[0]}, d={K.shape[1]}, block={BS} → {nb} 个 block, "
                      f"needle 所在 block={needles.tolist()}")

    print("\n[1] Quest 上界 ≥ block 内真实 max q·k")
    ub = quest_upper_bound(q, K, BS)
    true_max = (block_view(K, BS) @ q).max(1)                     # (nb,)
    kv("各 block 的 min(bound − 真实 max)", f"{(ub - true_max).min():.3f}  (≥ 0)")
    kv("bound 的松紧: mean(bound / 真实 max)", f"{np.mean(ub / true_max):.2f}x")
    # -1e-9 / -1e-4: 浮点舍入余量 (合成数据是 fp64, TinyLM 的 KV 是 fp32)
    assert (ub >= true_max - 1e-9).all(), "Quest 上界必须 ≥ block 内真实的 max q·k"
    for qq, KK, _ in tinylm_qkv(1024):                            # 真实 KV 上同样成立
        assert (quest_upper_bound(qq, KK, BS) >= (block_view(KK, BS) @ qq).max(1) - 1e-4).all(), \
            "真实 TinyLM 的 KV 上, Quest 上界同样必须 ≥ block 内真实的 max q·k"

    ks = [4, 8, 16, 32, 64, 128, nb]
    print(f"\n[2] needle 负载 (random = {N_RANDOM} 个种子平均; 所有方法都强制保留首块+末块)")
    t = sweep([(q, K, V)], ks)
    show(t, nb)
    top = select_blocks(ub, TOP)
    kv(f"quest top-{TOP} 是否包含全部 needle block", set(needles) <= set(top))
    assert set(needles) <= set(top), \
        f"quest 的 top-{TOP} 应包含全部 needle block {needles.tolist()}, 实际选了 {top.tolist()}"
    for m in ("quest", "mean", "random"):
        assert t[nb][m][0] < 1e-6, f"{m}: k = 全部 block 时应与全量 attention 一致"
    for k in ks[:-1]:
        assert t[k]["quest"][0] < t[k]["random"][0], f"k={k}: 同样读 k 个 block, 打分选块的误差应小于随机选块"
        assert t[k]["quest"][1] > t[k]["random"][1], f"k={k}: 打分选块覆盖的 attention 质量应多于随机选块"
    errs = [t[k]["quest"][0] for k in ks]
    # +0.02: 允许小幅回升, 只要求大体单调
    assert all(b <= a + 0.02 for a, b in zip(errs, errs[1:])), f"误差应随 k 增大而大体下降, 实际 {errs}"

    T2 = 1024
    cases = tinylm_qkv(T2)
    nb2 = T2 // BS
    print(f"\n[3] 真实 TinyLM KV (T={T2} 随机 token, {len(cases)} 层取平均, d={cases[0][1].shape[1]}, {nb2} blocks)")
    print("  预期: 随机权重模型的 attention 近乎均匀弥散, 没有 '少数重要 token', 稀疏化的前提不成立 →")
    print("  打分相对随机的优势会小很多; 真实训练过的 LLM 注意力高度集中, 才有 Quest/NSA/DSA 的收益。")
    ks2 = [4, 8, 16, 32, nb2]
    t2 = sweep(cases, ks2)
    show(t2, nb2)
    for m in ("quest", "mean", "random"):
        assert t2[nb2][m][0] < 1e-6, f"TinyLM KV, {m}: k = 全部 block 时应与全量 attention 一致"
    for k in ks2[:-1]:
        # 只断言召回; attention 弥散时误差谁大谁小没有保证
        assert t2[k]["quest"][1] >= t2[k]["random"][1], f"TinyLM KV, k={k}: 打分选块的召回至少不应输给随机选块"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
