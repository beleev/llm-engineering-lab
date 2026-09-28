"""
M18 — Sequence Packing: 把变长文档拼进定长序列

是什么: 预训练 / SFT 的文档长短不一。每篇单独占一行再补 padding, 大部分算力花在 pad 上;
        packing 把多篇文档首尾相接塞进同一行 [L], 几乎不留空位。
代价: 同一行里的文档会互相 "看见"。要和逐篇单独 forward 完全等价, 需要三件事:
    1. 文档级 block-diagonal 因果 mask: token 只能看同一篇里在它之前的 token;
    2. 位置 id 在每篇文档开头重置为 0;
    3. 每篇最后一个 token 不去预测下一篇的第一个 token (label 置为忽略)。
解决的瓶颈: 有效 token 比例 (MFU 的分子)。真实系统里 1 由 FlashAttention varlen (cu_seqlens) 实现, 不物化 [L, L] mask。
关键公式: 有效比例 = Σ len_i / (行数 · L);   注意力算量 Σ len_i² / L² (varlen 只算对角块)
读代码盯住: `doc_mask` 的一行 (因果 ∧ 同文档), `pack_ffd`, 和表格里 RoPE 那一列 "不重置位置也没事" 的原因。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv, make_rng, max_abs_diff, softmax

V, D, H = 50, 32, 4                                                      # 词表, 隐维, 头数
L = 128                                                                  # 定长序列长度


def rope(x, pos):
    """x: [T, H, dh], pos: [T] → [T, H, dh]。按位置旋转每对维度。

    q·k 只依赖位置差 → 天然是相对位置。
    """
    half = x.shape[-1] // 2
    ang = pos[:, None] / 10000 ** (np.arange(half) / half)              # [T, 1] / [half] → [T, half] 每对维度的转角
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]         # [T, 1, half], 中间的 1 在头维广播
    x1, x2 = x[..., :half], x[..., half:]                               # 各 [T, H, half]: 第 i 维和第 i+half 维配成一对
    # 每对 (x1, x2) 当成平面上的一个点, 转 ang 角; 拼回 [T, H, dh]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def abs_pos_emb(pos):
    """原版 Transformer 的正弦绝对位置编码, 直接加在 embedding 上。pos [T] → [T, D]。

    GPT-2 也是绝对位置, 但用可学习的表。
    """
    ang = pos[:, None] / 10000 ** (np.arange(0, D, 2) / D)              # [T, 1] / [D/2] → [T, D/2]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)          # [T, D/2] 两份 → [T, D]


def init_params(rs):
    """E [V, D] embedding 表; Wq / Wk / Wv / Wo [D, D] 注意力的四个矩阵; Wout [D, V] 输出头。"""
    s = 1 / np.sqrt(D)                                                   # 输入每维方差为 1 时, x @ W 每维方差也约为 1
    return {"E": rs.randn(V, D), **{k: rs.randn(D, D) * s for k in ("Wq", "Wk", "Wv", "Wo")},
            "Wout": rs.randn(D, V) * s}


def forward(P, tokens, pos, mask, pos_kind):
    """单层注意力 + 残差 + 输出头。mask[t, s] = True 表示 t 可以看 s。返回 logits [T, V]。

    tokens [T] 整数, pos [T] 每个 token 的位置 id, mask [T, T] bool。
    pos_kind: "abs" 把位置加在 embedding 上; "rope" 把位置转进 q 和 k。
    """
    T, dh = len(tokens), D // H                                          # dh: 每个头的维度
    x = P["E"][tokens] + (abs_pos_emb(pos) if pos_kind == "abs" else 0)  # [T, D]
    q, k, v = ((x @ P[w]).reshape(T, H, dh) for w in ("Wq", "Wk", "Wv"))     # [T, D] → 各 [T, H, dh]
    if pos_kind == "rope":
        q, k = rope(q, pos), rope(k, pos)                                # 只转 q 和 k, v 不带位置; 形状不变
    # t 是 query 的位置, s 是 key 的位置, 每个头各算各的
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(dh)               # [T,H,dh]·[T,H,dh] → [H, T, T]
    scores[:, ~mask] = -np.inf                                           # mask [T, T], 对每个头广播; 不许看的 → -inf
    out = np.einsum("hts,shd->thd", softmax(scores), v).reshape(T, D)    # [H,T,T]·[T,H,dh] → [T, H, dh] → [T, D]
    return (x + out @ P["Wo"]) @ P["Wout"]                               # 残差后过输出头: [T, D] @ [D, V] → [T, V]


def doc_mask(doc_ids):
    """block-diagonal 因果 mask: 因果 ∧ 同一篇文档。doc_ids [T] → [T, T] bool。"""
    t = np.arange(len(doc_ids))
    # [T, 1] 和 [1, T] 广播成 [T, T]: 行是 query, 列是 key
    return (t[:, None] >= t[None, :]) & (doc_ids[:, None] == doc_ids[None, :])


def reset_positions(doc_ids):
    """每篇文档开头位置归零: [0,1,2, 0,1, 0,1,2,3, ...]。"""
    # 每篇第一个 token 的下标 [n_docs]。diff ≠ 0 的位置是一篇的最后一个 token, +1 才是下一篇的开头
    starts = np.r_[0, np.flatnonzero(np.diff(doc_ids)) + 1]
    # 每个 token 的下标减去它那一篇的起点。repeat 把起点按每篇长度铺开成 [T]
    return np.arange(len(doc_ids)) - np.repeat(starts, np.diff(np.r_[starts, len(doc_ids)]))


def pack_ffd(lengths):
    """First-Fit-Decreasing: 从长到短, 放进第一个还装得下的行。不切断文档。返回每行的文档下标列表。"""
    rows, free = [], []                                                  # 每行的文档下标; 每行还剩几个空位
    for i in np.argsort(lengths)[::-1]:                                  # 从最长的文档开始
        for r, f in enumerate(free):
            if f >= lengths[i]:
                rows[r].append(i); free[r] -= lengths[i]
                break
        else:                                                            # 没有 break = 哪一行都放不下 → 新开一行
            rows.append([i]); free.append(L - lengths[i])
    return rows


def cross_entropy_sum(logits, targets, valid):
    """logits [T, V], targets [T], valid [T] bool → valid 位置上交叉熵的总和 (标量)。"""
    logp = logits - logits.max(-1, keepdims=True)                        # 先减每行最大值, exp 不溢出
    logp -= np.log(np.exp(logp).sum(-1, keepdims=True))                  # [T, V] log-softmax
    return -logp[np.arange(len(targets)), targets][valid].sum()          # 取正确答案那一列 [T] → 只留 valid 的


def main() -> None:
    banner("M18 - Sequence Packing")
    rs = make_rng(18)

    # ---- 1) padding vs packing 的有效 token 比例 ----
    # [n_docs] 对数正态: 短文档多, 长文档少; 长度夹在 [4, L]
    n_docs = 400
    lengths = np.clip(rs.lognormal(3.3, 0.8, size=n_docs), 4, L).astype(int)
    print(f"\n[1] {n_docs} 篇文档, 长度中位数 {int(np.median(lengths))}, 最长 {lengths.max()}, 序列长 L = {L}")
    total = lengths.sum()
    batch = 8
    order = rs.permutation(len(lengths))
    # 动态 padding 占的格子数: 每个 batch 补齐到本 batch 最长的那篇 → 最长 × 篇数, 再对所有 batch 求和
    dyn_slots = sum(lengths[order[i:i + batch]].max() * len(order[i:i + batch]) for i in range(0, len(order), batch))
    rows = pack_ffd(lengths)
    ratios = {
        "每篇一行, 补齐到 L": total / (len(lengths) * L),
        f"动态 padding (batch={batch} 内补齐到最长)": total / dyn_slots,
        "packing (FFD, 不切文档)": total / (len(rows) * L),
    }
    for name, r in ratios.items():
        kv(name, f"有效 token {r:.3f}")
    kv("packing 行数 / 文档数", f"{len(rows)} / {len(lengths)}")
    varlen = sum(lengths[i] ** 2 for row in rows for i in row) / (len(rows) * L**2)
    kv("注意力算量 varlen / 整行因果", f"{varlen:.3f}  (只算对角块)")
    assert ratios["packing (FFD, 不切文档)"] > 0.95 > 0.5 > ratios["每篇一行, 补齐到 L"], \
        "有效 token 比例: packing 应高于 0.95, 每篇一行补齐到 L 应低于 0.5"
    assert ratios["每篇一行, 补齐到 L"] < ratios[f"动态 padding (batch={batch} 内补齐到最长)"] < 0.95, \
        "动态 padding 夹在中间: 高于每篇一行, 低于 0.95"

    # ---- 2) 带文档 mask 的 packing == 逐篇单独 forward ----
    row = next(r for r in rows if len(r) >= 4)                           # 第一个装了 ≥4 篇的行: 长短混合
    docs = [rs.randint(0, V, size=lengths[i]) for i in row]
    tokens = np.concatenate(docs)                                        # [T] 几篇文档首尾相接
    doc_ids = np.repeat(np.arange(len(docs)), [len(d) for d in docs])    # [T] 每个 token 属于第几篇
    bounds = np.r_[0, np.cumsum([len(d) for d in docs])]                 # [n_docs+1] 每篇的起止下标
    print(f"\n[2] 一行里塞了 {len(docs)} 篇文档, 长度 {[len(d) for d in docs]}, 共 {len(tokens)} token")
    P = init_params(rs)
    causal = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))    # [T, T] 普通因果 mask, 不分文档
    configs = {
        "文档 mask + 位置重置": (doc_mask(doc_ids), reset_positions(doc_ids)),
        "文档 mask, 位置连续": (doc_mask(doc_ids), np.arange(len(tokens))),
        "只有因果 mask + 位置重置": (causal, reset_positions(doc_ids)),
    }
    err = {}
    print(f"  {'配置':<22}{'RoPE max|Δ|':>14}{'绝对位置 max|Δ|':>18}")
    for name, (mask, pos) in configs.items():
        for pos_kind in ("rope", "abs"):
            packed = forward(P, tokens, pos, mask, pos_kind)             # [T, V] 整行一起算
            # 参照: 每篇单独 forward, 位置从 0 起, 普通因果 mask
            ref = [forward(P, d, np.arange(len(d)), np.tril(np.ones((len(d), len(d)), dtype=bool)), pos_kind)
                   for d in docs]
            err[name, pos_kind] = max(max_abs_diff(packed[bounds[j]:bounds[j + 1]], ref[j]) for j in range(len(docs)))
        print(f"  {name:<22}{err[name, 'rope']:>14.1e}{err[name, 'abs']:>18.1e}")
    assert err["文档 mask + 位置重置", "rope"] < 1e-12 and err["文档 mask + 位置重置", "abs"] < 1e-12, "逐元素一致"
    assert err["只有因果 mask + 位置重置", "rope"] > 0.1 and err["只有因果 mask + 位置重置", "abs"] > 0.1, "不带 mask 就串文档"
    assert err["文档 mask, 位置连续", "abs"] > 0.1, "绝对位置编码: 不重置位置, 后面的文档就不等价"
    assert err["文档 mask, 位置连续", "rope"] < 1e-9, "RoPE 只看位置差, 块内相对位置不变 → 仅有浮点误差"

    # ---- 3) label: 不跨文档预测 ----
    mask, pos = configs["文档 mask + 位置重置"]
    logits = forward(P, tokens, pos, mask, "rope")
    targets = np.r_[tokens[1:], 0]                                       # [T] 右移一位; 末尾补的 0 只占位, 不参与 loss
    valid = np.ones(len(tokens), dtype=bool)
    valid[bounds[1:] - 1] = False                                        # 每篇最后一个 token 没有 "下一个"
    packed_ce = cross_entropy_sum(logits, targets, valid)
    ref_ce = sum(cross_entropy_sum(forward(P, d, np.arange(len(d)), np.tril(np.ones((len(d),) * 2, dtype=bool)), "rope"),
                                   np.r_[d[1:], 0], np.arange(len(d)) < len(d) - 1) for d in docs)
    naive_ce = cross_entropy_sum(logits, targets, np.arange(len(tokens)) < len(tokens) - 1)
    print()
    kv("有效预测数: 屏蔽边界 / 不屏蔽", f"{valid.sum()} / {len(tokens) - 1}")
    kv("CE 总和: packing / 逐篇 / 不屏蔽边界", f"{packed_ce:.6f} / {ref_ce:.6f} / {naive_ce:.6f}")
    assert abs(packed_ce - ref_ce) < 1e-9, "屏蔽边界后, packing 的 CE 总和必须等于逐篇 forward 之和"
    assert abs(naive_ce - ref_ce) > 1, "不屏蔽边界就多算了跨文档的预测, CE 总和与逐篇之和应差 1 以上"

    print(f"\n  OK: packing 有效 token 从 {ratios['每篇一行, 补齐到 L']:.3f} 提到 {ratios['packing (FFD, 不切文档)']:.3f};"
          " 文档 mask + 位置重置 + 边界 label 屏蔽 == 逐篇 forward。")


if __name__ == "__main__":
    main()
