#!/usr/bin/env python
"""
Mixtral 推理示例: softmax top-2 路由 + GQA KV cache

assert 验证: 每 token 选 2 个不同专家且权重和为 1; 激活参数 < 总参数;
generate 带 cache 与不带 cache 的贪心输出逐 token 相同。模型随机初始化, 生成内容无意义。
"""

import torch

from llm_models.models.moe.mixtral import Mixtral
from llm_models.utils.generation import KVCache


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def main():
    torch.manual_seed(42)

    vocab_size, num_experts, top_k = 1000, 4, 2
    model = Mixtral(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, num_experts=num_experts, top_k=top_k, max_len=128,
    ).eval()

    count = lambda m: sum(p.numel() for p in m.parameters())
    total = count(model)
    experts = sum(count(layer.moe.experts) for layer in model.layers)
    active = total - experts + experts * top_k // num_experts  # 专家部分每个 token 只用到 K/E
    print(f"Mixtral Mini | 总参数 {total:,} | 每 token 激活 {active:,} ({active / total:.1%})")
    assert active < total, "每个 token 只过 K 个专家, 激活参数应少于总参数"

    idx = torch.randint(0, vocab_size, (1, 10))
    with torch.inference_mode():
        logits, all_routing = model(idx)
    assert logits.shape == (1, 10, vocab_size), "logits 形状应为 [B, T, V]"

    for i, info in enumerate(all_routing):  # 每个 MoE 层一份路由记录
        # sel [N, K]: 选中的专家下标; w [N, K]: 选中后重归一的权重; probs [N, E]: 对全部专家的 softmax
        sel, w, probs = info["selected_experts"], info["routing_weights"], info["routing_probs"]
        assert (sel[:, 0] != sel[:, 1]).all(), f"Layer {i}: 同一个 token 选中的两个专家应不同"
        assert sel.max() < num_experts, f"Layer {i}: 专家下标应小于专家数 E"
        assert torch.allclose(w.sum(-1), torch.ones(10), atol=1e-5), f"Layer {i}: 选中专家的权重重归一后之和应为 1"
        assert torch.allclose(probs.sum(-1), torch.ones(10), atol=1e-5), f"Layer {i}: softmax 路由概率的行和应为 1"
        n = 3                                                # 只打印前 n 个 token
        print(f"  第 {i} 层: 前 {n} 个 token 的专家 {sel[:n].tolist()}  "
              f"权重 {[[round(v, 3) for v in r] for r in w[:n].tolist()]}")

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


if __name__ == "__main__":
    main()
