#!/usr/bin/env python
"""
DeepSeek-V3 推理示例: MoE 路由 + MLA 的 latent KV cache

三件事, 每件都有 assert:
  1. 路由: 每 token 选 K 个不同专家, 权重和为 1; 激活参数 ≪ 总参数
  2. 带 cache 的 prefill+decode 与一次性 forward 的 logits 一致; 贪心 generate 输出逐 token 相同;
     左 pad 不改变真实位置的 logits; 超过 max_len 抛 ValueError
  3. 每 token 每层的 cache 浮点数 (从真实 cache 张量里数出来, 再与公式对照):
        MHA 2·H·Dh   vs   GQA 2·Hkv·Dh   vs   MLA r + rope
模型是随机初始化的, 生成内容无意义; 这里验证的是机制。
"""

import torch

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.models.moe.deepseekV3 import DeepSeekV3
from llm_models.utils.generation import KVCache

# d_model, 头数 H, GQA 的 KV 头数 Hkv, MLA 的 latent 维数 r, 单独带 RoPE 的那段 key 的维数
D_MODEL, N_HEADS, N_KV_HEADS, R, ROPE = 512, 8, 2, 64, 32


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def floats_per_token(cache: dict, seq_len: int) -> int:
    """一层 cache 里每个 token 占多少个浮点数: 所有张量的元素数之和 ÷ 序列长。"""
    return sum(t.numel() for t in cache.values()) // seq_len  # batch = 1


def main():
    torch.manual_seed(42)
    vocab_size, top_k, num_experts = 1000, 2, 8
    model = DeepSeekV3(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, num_layers=2, max_len=128,
        num_routed_experts=num_experts, num_shared_experts=1, top_k=top_k,
        latent_dim=R, qk_rope_head_dim=ROPE,
    ).eval()
    idx = torch.randint(0, vocab_size, (1, 10))

    # ---- 1) 路由 ----
    with torch.inference_mode():
        logits, all_routing_info = model(idx)
    assert logits.shape == (1, 10, vocab_size), "logits 形状应为 [B, T, V]"
    for i, info in enumerate(all_routing_info):  # 每个 MoE 层一份路由记录
        sel, w = info["selected_experts"], info["routing_weights"]  # [N, K], N = B·T 个 token
        assert sel.shape == (10, top_k), f"Layer {i}: 每个 token 应选出 K 个专家, 形状应为 [N, K]"
        assert (sel[:, 0] != sel[:, 1]).all(), f"Layer {i}: 同一个 token 选中的两个专家应不同"
        assert sel.max() < num_experts, f"Layer {i}: 专家下标应小于专家数 E"
        assert torch.allclose(w.sum(-1), torch.ones(10), atol=1e-5), f"Layer {i}: 每个 token 的路由权重之和应为 1"
        n = 3                                                # 只打印前 n 个 token
        print(f"第 {i} 层: 前 {n} 个 token 选中专家 {sel[:n].tolist()}, "
              f"权重 {[[round(v, 3) for v in r] for r in w[:n].tolist()]}")

    p = model.get_num_active_params()
    assert p["total_params"] == sum(t.numel() for t in model.parameters()), \
        "get_num_active_params 报的总参数应与逐个数出来的一致"
    assert p["active_params"] < p["total_params"], "每个 token 只过 K 个路由专家, 激活参数应少于总参数"
    print(f"总参数 {p['total_params']:,} | 每 token 激活 {p['active_params']:,} "
          f"({p['active_params'] / p['total_params']:.1%})")

    # ---- 2) KV cache 不改变结果 ----
    full = torch.randint(0, vocab_size, (2, 30))  # [B=2, T=30], 前 prefill 个一次喂入, 其余逐个 decode
    prefill = 10
    with torch.inference_mode():
        diff = (model(full)[0] - cached_logits(model, full, prefill=prefill)).abs().max().item()
    print(f"prefill {prefill} + 逐 token decode {full.size(1) - prefill} 步 vs 一次性 forward: "
          f"logits 最大差 {diff:.2e}")
    assert diff < 1e-4, "cache 路径与无 cache 路径的 logits 不一致"
    n_gen = 20
    with_cache = model.generate(idx, max_new_tokens=n_gen, temperature=0, use_cache=True)
    no_cache = model.generate(idx, max_new_tokens=n_gen, temperature=0, use_cache=False)
    assert torch.equal(with_cache, no_cache), "cache 与无 cache 的贪心输出必须完全相同"
    print(f"贪心生成 {n_gen} token, cache 与无 cache 完全一致: {with_cache[0, idx.size(1):].tolist()}")

    # 左 pad (不打印): MLA 的位置只在 RoPE 段, pad 被 mask 挡掉后真实位置的 logits 不变
    P = 4
    padded = torch.cat([torch.zeros(2, P, dtype=torch.long), full], dim=1)   # [B, P+T]
    mask = torch.cat([torch.zeros(2, P), torch.ones_like(full)], dim=1)       # [B, P+T], 0 = pad
    with torch.inference_mode():
        d_pad = (model(padded, attention_mask=mask)[0][:, P:] - model(full)[0]).abs().max().item()
    assert d_pad < 1e-4, f"左 pad 改变了真实位置的 logits: {d_pad}"
    # 超过 max_len 要给出说明上限的错误, 而不是 RoPE 查表越界的 IndexError
    try:
        model(torch.zeros(1, model.max_len + 1, dtype=torch.long))
        too_long_ok = True
    except ValueError:
        too_long_ok = False
    assert not too_long_ok, "序列超过 max_len 时 forward 应抛 ValueError"

    # ---- 3) cache 体积: 直接数真实 cache 里的浮点数 ----
    S = idx.size(1)
    cache = KVCache(len(model.layers))
    with torch.inference_mode():
        model(idx, cache=cache)
        assert set(cache.layers[0]) == {"c_kv", "k_rope"}, "MLA 只该缓存 latent 和共享 k_rope"
        assert cache.layers[0]["c_kv"].shape == (1, S, R), "c_kv 应是 [B, S, r] 的 latent, 不分头"
        assert cache.layers[0]["k_rope"].shape == (1, 1, S, ROPE), "k_rope 应是 [B, 1, S, rope], 所有头共用一份"
        mla = floats_per_token(cache.layers[0], S)

        # 对照组: 同样的 d_model 和头数, 单独建一层 GQA 跑一遍, 数它的 cache。Hkv = H 时 GQA 就是 MHA
        x, measured = torch.randn(1, S, D_MODEL), {}
        for name, hkv in [("MHA", N_HEADS), ("GQA", N_KV_HEADS)]:
            c = {}
            GroupedQueryAttention(D_MODEL, N_HEADS, num_kv_heads=hkv)(x, cache=c)
            measured[name] = floats_per_token(c, S)

    d_head = D_MODEL // N_HEADS
    assert mla == R + ROPE, "MLA 每 token 每层应只缓存 r + rope 个数"
    assert measured["MHA"] == 2 * N_HEADS * d_head, "MHA 每 token 每层应缓存 2·H·Dh 个数 (K 和 V 各 H 个头)"
    assert measured["GQA"] == 2 * N_KV_HEADS * d_head, "GQA 每 token 每层应缓存 2·Hkv·Dh 个数"
    print(f"\n每 token 每层 cache 浮点数 (d_model={D_MODEL}, H={N_HEADS}, Dh={d_head}):")
    print(f"  MHA  2·H·Dh   = {measured['MHA']}")
    print(f"  GQA  2·Hkv·Dh = {measured['GQA']}  (Hkv={N_KV_HEADS}, MHA 的 {measured['GQA'] / measured['MHA']:.1%})")
    print(f"  MLA  r + rope = {mla}  (r={R}, rope={ROPE}, MHA 的 {mla / measured['MHA']:.1%}, GQA 的 {mla / measured['GQA']:.1%})")
    assert mla < measured["GQA"] < measured["MHA"], "cache 体积应满足 MLA < GQA < MHA"


if __name__ == "__main__":
    main()
