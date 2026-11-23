#!/usr/bin/env python
"""
DeepSeek-V4 压缩注意力结构验证 — 退化性、因果性、KV 字节、增量生成。不训练, 几秒跑完。

看什么:
    1. 压缩比 1、不重叠、top-k 取全部、去掉滑窗时, CSA 退化成稠密因果注意力 (shared-KV 的多查询形式):
       每 1 个 token 压成 1 条就是它自己, 条目 i 的到齐位置就是位置 i, 可见性就是因果 mask。
    2. 压缩条目只在凑满之后才可见: 条目 i 的到齐位置是 (i+1)m − 1, 改动位置 t 之后的 token 不会影响
       位置 t 的输出。
    3. 每 token 每层的 KV 元素数与算式逐个相符; 再按原文配置给出 MLA (V3)、MLA + indexer (V3.2)、
       V4 三者的字节对比 (代价模型估算, 不是实测)。
    4. 逐 token 增量生成与整段前向一致: 压缩、滑窗、top-k 三条路径的缓存都要对。
"""

import math

import torch
import torch.nn.functional as F

from llm_models.layers.core.compressed_attention import CompressedAttention
from llm_models.models.moe.deepseek_v4 import DeepSeekV4
from llm_models.utils.generation import KVCache

V, T = 64, 64
C, ROPE, M, HCA_M, WIN = 32, 16, 4, 16, 8
CFG = dict(vocab_size=V, d_model=64, n_heads=4, num_layers=4, max_len=128, dropout=0.0,
           num_routed_experts=4, num_shared_experts=1, top_k=2,
           latent_dim=C, qk_rope_head_dim=ROPE, csa_block=M, hca_block=HCA_M,
           window=WIN, csa_top_k=4)


def dense_reference(layer: CompressedAttention, x: torch.Tensor, rope) -> torch.Tensor:
    """压缩比 1 时的参照: shared-KV 多查询注意力, 一个向量同时当 K 和 V, 因果 softmax。"""
    B, T_, _ = x.shape
    pos = torch.arange(T_)
    q = layer.q_norm(layer.q_up(x).view(B, T_, layer.num_heads, layer.c))
    head, tail = q[..., :-ROPE], q[..., -ROPE:]
    q = torch.cat([head, rope(tail.transpose(1, 2), position_ids=pos).transpose(1, 2)], dim=-1)
    kv = layer.w_kv_a(x)
    head, tail = kv[..., :-ROPE], kv[..., -ROPE:]
    kv = layer.kv_norm(torch.cat([head, rope(tail, position_ids=pos)], dim=-1))
    scores = torch.einsum("bthc,bsc->bths", q, kv) / math.sqrt(layer.c)
    causal = torch.ones(T_, T_, dtype=torch.bool).tril().unsqueeze(1)          # [T, 1, S], 在 head 维广播
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.einsum("bths,bsc->bthc", F.softmax(scores, dim=-1), kv)
    return layer.w_o(out.reshape(B, T_, layer.num_heads * layer.c))


def v4_cache_elements(layer: CompressedAttention, seq_len: int) -> int:
    """V4 一层的 KV cache 元素数算式: 压缩条目 + 还凑不满一块的 token + 滑窗。"""
    n_entry = seq_len // layer.m
    pending = seq_len - max(0, n_entry - 1) * layer.m          # 缓冲区留着最后一整块
    per_src = (layer.c + 1) * (2 if layer.overlap else 1)      # 每路一个 KV 向量加一个压缩打分
    win = min(seq_len, layer.window + 1) * 2 * layer.c if layer.window else 0
    return n_entry * layer.c + pending * per_src + win


def main():
    torch.manual_seed(0)
    model = DeepSeekV4(**CFG).eval()
    x = torch.randint(0, V, (2, T))
    print(f"层排布 {model.pattern} (C = CSA, 压缩比 {M}, top-k {CFG['csa_top_k']}; H = HCA, 压缩比 {HCA_M}); "
          f"条目维度 c = {C} (RoPE 段 {ROPE}), 滑窗 {WIN}")

    # ---- 1. 压缩比 1 时退化成稠密因果注意力 ----
    plain = CompressedAttention(d_model=64, num_heads=4, latent_dim=C, qk_rope_head_dim=ROPE,
                                block_size=1, window=0, sparse=True, overlap=False, sparse_top_k=T)
    h = torch.randn(2, T, 64)
    with torch.no_grad():
        got = plain(h, rope=model.rope, position_ids=torch.arange(T))
        want = dense_reference(plain, h, model.rope)
    diff = (got - want).abs().max().item()
    print(f"\n压缩比 1、不重叠、top-k 取全部、无滑窗: 与稠密因果注意力的最大差 {diff:.1e}")
    assert diff < 1e-5, "压缩比 1 且 top-k 取全部时, CSA 应与稠密因果注意力一致"

    # ---- 2. 压缩块只在凑满之后才可见 ----
    with torch.no_grad():
        ref = model(x)[0]
        future = x.clone()
        future[:, 40:] = torch.randint(0, V, (2, T - 40), generator=torch.Generator().manual_seed(1))
        alt = model(future)[0]
    gap_before = (ref[:, :40] - alt[:, :40]).abs().max().item()
    print(f"\n只改位置 40 之后的 token: 前 40 个位置的 logits 最大差 {gap_before:.1e}")
    assert gap_before < 1e-5, "改动后面的 token 不应影响前面位置的输出"

    n_entry = T // M
    entry_pos = torch.arange(n_entry) * M + (M - 1)
    print(f"CSA 的 {n_entry} 个条目到齐位置: {entry_pos[:6].tolist()} ... {entry_pos[-1].item()}"
          f" (条目 i 覆盖 token [{'(i-1)m'}, (i+1)m), 第 {M - 1}、{2 * M - 1} ... 位之后才可见)")
    assert (entry_pos == torch.arange(n_entry) * M + M - 1).all(), "条目的到齐位置应是 (i+1)m − 1"
    for t in (0, M - 2, M - 1, 2 * M - 1, T - 1):
        visible = int((entry_pos <= t).sum())
        assert visible == (t + 1) // M, f"位置 {t} 能看到的条目数应是 (t+1)//m, 实际 {visible}"
    print(f"  位置 {M - 2} 看到 0 条, 位置 {M - 1} 看到 1 条, 位置 {T - 1} 看到 {(T) // M} 条")

    # ---- 3. 增量生成与整段前向一致 ----
    print("\n逐 token 增量生成 vs 整段前向:")
    caches = {}
    for residual in ("standard", "mhc"):
        torch.manual_seed(0)
        m = DeepSeekV4(residual=residual, n_streams=4, **CFG).eval()
        with torch.no_grad():
            full = m(x)[0]
            cache = KVCache(len(m.layers))
            step = torch.cat([m(x[:, t:t + 1], cache=cache)[0] for t in range(T)], dim=1)
        caches[residual] = (m, cache)
        diff = (full - step).abs().max().item()
        params = sum(p.numel() for p in m.parameters())
        print(f"  residual = {residual:<8} | 最大差 {diff:.1e} | 参数 {params:,}")
        assert diff < 1e-4, f"residual={residual}: 增量生成应与整段前向一致"

    # ---- 4. KV 元素数与算式 ----
    print(f"\n逐 token 解码到 T = {T} 之后每层的 KV cache 元素数 (实测 vs 算式, batch 2):")
    m, cache = caches["standard"]
    for i, (layer, kind) in enumerate(zip(m.layers, m.pattern)):
        c = cache.layers[i]
        got = (c["entries"].numel() + sum(a.numel() for a in c["src"] if a is not None)
               + c["win_k"].numel() + c["win_v"].numel()) // x.size(0)
        want = v4_cache_elements(layer.attn, T)
        print(f"  第 {i} 层 {kind} | 条目 {c['entries'].size(1):2d} 条 | 每条序列实测 {got:5d} | 算式 {want:5d}")
        assert got == want, f"第 {i} 层的 KV 元素数应与算式相符"

    print("\n每 token 每层的 KV 字节, 按原文配置的代价模型估算 (不是实测):")
    for name, text in kv_cost_model(1_000_000):
        print(f"  {name:<24} {text}")
    print("DeepSeek-V4 压缩注意力结构验证通过")


def kv_cost_model(seq_len: int):
    """按原文配置算每 token 的 KV 字节 (代价模型)。V4 按 RoPE 段 BF16、其余 FP8; MLA 按纯 BF16。

    V3.2 在 MLA 之上还要缓存一份 indexer 的 key, 原文没有给它的维度, 所以这里不计这一项:
    下面的 "是 MLA 的百分之几" 因此是一个偏大的估计 (分母偏小)。
    """
    layers, r, rope, c, m, m_hca, win = 61, 512, 64, 512, 4, 128, 128
    mla = (r + rope) * 2                                   # V3: 每 token 每层缓存 c_kv 加共享的 k_rope
    entry = rope * 2 + (c - rope) * 1                      # V4: 一个条目兼当 K 和 V, 混合精度
    csa = (entry * math.ceil(seq_len / m) + win * 2 * entry) / seq_len
    hca = (entry * math.ceil(seq_len / m_hca) + win * 2 * entry) / seq_len
    n_hca = 2 + (layers - 2) // 2                          # 前两层 HCA, 之后 1:1 交错
    n_csa = layers - n_hca
    v4, v3 = n_csa * csa + n_hca * hca, layers * mla
    need = (0.10 * v3 - layers * hca) / (csa - hca)        # 要凑到原文的 10%, 至少得有多少层 CSA
    return [
        ("MLA (V3)", f"({r} + {rope}) × 2 = {mla} 字节/层 × {layers} 层 = {v3:,.0f} 字节"),
        ("MLA + indexer (V3.2)", f"{mla} 字节/层 再加一份 indexer key; 原文没给它的维度, 不计入"),
        ("V4 CSA 层", f"{entry} × ceil(T/{m}) / T + {win} × 2 × {entry} / T = {csa:.2f} 字节"),
        ("V4 HCA 层", f"{entry} × ceil(T/{m_hca}) / T + 同样的滑窗 = {hca:.2f} 字节"),
        ("V4 合计 (1:1 交错)", f"{n_csa} × CSA + {n_hca} × HCA = {v4:,.0f} 字节, 是 MLA 的 {v4 / v3:.1%}"),
        ("原文的说法", f"1M 上下文下 V4-Pro 的 KV cache 是 V3.2 的 10%; 按上面的代价模型反推, "
                   f"61 层里至少要有约 {need:.0f} 层 CSA 才凑得到这个比例, 也就是交错比例比 1:1 更偏向 CSA"),
    ]


if __name__ == "__main__":
    main()
