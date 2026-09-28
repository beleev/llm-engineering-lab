#!/usr/bin/env python
"""
DeepSeek-V3.2 推理示例: DSA = MLA + Lightning Indexer

验证四件事 (全是 assert):
  1. 构造不浪费: 每层的 MLA 只建一次; 参数量 = 同配置 V3 + indexer 参数
  2. 稀疏 mask: 每个 query 最多看 sparse_top_k 个 key, 且绝不看未来
  3. 稀疏确实生效: 序列长于 top-k 时, V3.2 的输出与 "同权重走稠密注意力" 不同
  4. decode 时 indexer 也走 cache (多缓存一份 indexer key), 贪心输出与无 cache 完全相同;
     左 pad 不改变真实位置的 logits (不打印)
"""

import torch

from llm_models.layers.core.attention import MultiHeadLatentAttention
from llm_models.models.moe.deepseekV3 import DeepSeekV3, DeepSeekV3_2
from llm_models.utils.generation import KVCache

# 每个 query 最多看几个 key, indexer 的头数和每头维数, MLA 的 latent 维数 r, 带 RoPE 的那段 key 的维数
SPARSE_TOP_K, IDX_HEADS, IDX_DIM, R, ROPE = 8, 2, 32, 64, 32


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def main():
    torch.manual_seed(42)
    base = dict(
        vocab_size=1000, d_model=256, n_heads=4, num_layers=2, max_len=128,
        num_routed_experts=4, num_shared_experts=1, top_k=2, latent_dim=R, qk_rope_head_dim=ROPE,
    )
    dsa = dict(sparse_top_k=SPARSE_TOP_K, indexer_heads=IDX_HEADS, indexer_head_dim=IDX_DIM)

    # ---- 1) 每层只建一次 MLA (先建 V3 的层再把注意力换掉的写法, 每层的 MLA 会多建一遍再丢掉) ----
    # 计数办法: 临时把 MLA.__init__ 换成 "先记一笔再调原函数" 的版本, 建完模型在 finally 里还原
    built, orig_init = [], MultiHeadLatentAttention.__init__
    MultiHeadLatentAttention.__init__ = lambda self, *a, **k: (built.append(1), orig_init(self, *a, **k))[1]
    try:
        model = DeepSeekV3_2(**base, **dsa).eval()
    finally:
        MultiHeadLatentAttention.__init__ = orig_init
    assert len(built) == base["num_layers"], f"MLA 被构造了 {len(built)} 次"

    count = lambda m: sum(p.numel() for p in m.parameters())
    indexer_params = sum(count(layer.attn.indexer) for layer in model.layers)
    assert indexer_params == base["num_layers"] * 2 * base["d_model"] * IDX_HEADS * IDX_DIM, \
        "每层 indexer 应只有 w_q 和 w_k 两个投影, 各 d_model × (头数 × 每头维数) 个参数"
    assert count(model) == count(DeepSeekV3(**base)) + indexer_params, "V3.2 的参数量应等于同配置 V3 加上 indexer"
    print(f"MLA 构造次数 {len(built)} (= 层数) | 总参数 {count(model):,} = V3 + indexer {indexer_params:,}")

    # ---- 2) 稀疏 mask 的两条不变量 ----
    T = 24
    attn = model.layers[0].attn
    causal = model._causal_mask(T)  # [1, T, T]
    with torch.inference_mode():
        scores = attn.indexer(torch.randn(2, T, base["d_model"]), mask=causal)  # [2, T, T]
        sparse = attn._sparse_mask_from_topk(scores, causal)
    assert not (sparse & ~causal).any(), "选中了未来 token"
    per_row = sparse.sum(-1)  # [2, T] 每个 query 实际能看的 key 数
    assert per_row.max() == SPARSE_TOP_K, "每个 query 最多只能看 sparse_top_k 个 key"
    # 第 t 行因果可见 t+1 个 key; 不足 k 个时没什么可筛的, 应全部保留
    assert (per_row[:, :SPARSE_TOP_K] == torch.arange(1, SPARSE_TOP_K + 1)).all(), \
        "前 k 行可见的 key 不足 k 个, 应全部保留 (第 t 行看 t+1 个)"
    print(f"T={T}: 每个 query 看到的 key 数 = {per_row[0].tolist()}  (上限 {SPARSE_TOP_K}, 稠密时是 1..{T})")

    # ---- 3) 稀疏真的改变了计算 ----
    idx = torch.randint(0, base["vocab_size"], (1, T))
    with torch.inference_mode():
        sparse_logits = model(idx)[0]
        model.set_dense_warmup(True)  # 同一套权重, 临时切到稠密注意力当对照
        dense_logits = model(idx)[0]
        model.set_dense_warmup(False)
    # 前 k 个位置本来就全看
    assert torch.allclose(sparse_logits[:, :SPARSE_TOP_K], dense_logits[:, :SPARSE_TOP_K], atol=1e-5), \
        "前 k 个位置可见的 key 不超过 k 个, 稀疏与稠密的输出应相同"
    assert not torch.allclose(sparse_logits[:, -1], dense_logits[:, -1], atol=1e-5), \
        "序列长于 top-k 时稀疏应改变结果: 最后位置的 logits 不应与稠密相同"
    print(f"稀疏 vs 稠密 最后位置 logits 最大差 {(sparse_logits[:, -1] - dense_logits[:, -1]).abs().max():.4f} (> 0)")

    # ---- 4) 带 cache 的 decode: indexer key 也缓存 ----
    full = torch.randint(0, base["vocab_size"], (2, 30))
    prefill = 10
    with torch.inference_mode():
        diff = (model(full)[0] - cached_logits(model, full, prefill=prefill)).abs().max().item()
    print(f"prefill {prefill} + 逐 token decode {full.size(1) - prefill} 步 vs 一次性 forward: "
          f"logits 最大差 {diff:.2e}")
    assert diff < 1e-4, "cache 路径与无 cache 路径的 logits 不一致"

    # 左 pad (不打印): indexer 先 mask 再选 top-k, pad 不会被选中, 真实位置的 logits 不变
    P = 4
    padded = torch.cat([torch.zeros(2, P, dtype=torch.long), full], dim=1)   # [B, P+T]
    mask = torch.cat([torch.zeros(2, P), torch.ones_like(full)], dim=1)       # [B, P+T], 0 = pad
    with torch.inference_mode():
        d_pad = (model(padded, attention_mask=mask)[0][:, P:] - model(full)[0]).abs().max().item()
    assert d_pad < 1e-4, f"左 pad 改变了真实位置的 logits: {d_pad}"

    prompt = idx[:, :12]
    n_gen = 20
    with_cache = model.generate(prompt, max_new_tokens=n_gen, temperature=0, use_cache=True)
    no_cache = model.generate(prompt, max_new_tokens=n_gen, temperature=0, use_cache=False)
    assert torch.equal(with_cache, no_cache), "cache 与无 cache 的贪心输出必须完全相同"

    cache = KVCache(len(model.layers))
    with torch.inference_mode():
        model(prompt, cache=cache)
    per_token = {k: v.numel() // prompt.size(1) for k, v in cache.layers[0].items()}  # 每项每个 token 占几个数
    assert per_token == {"idx_k": IDX_HEADS * IDX_DIM, "c_kv": R, "k_rope": ROPE}, \
        "每 token 每层应缓存三项: indexer key (头数 × 每头维数)、latent c_kv (r)、k_rope (rope)"
    print(f"贪心生成 {n_gen} token, cache 与无 cache 一致 | 每 token 每层 cache: {per_token} "
          f"(idx_k 是 DSA 多付的代价)")


if __name__ == "__main__":
    main()
