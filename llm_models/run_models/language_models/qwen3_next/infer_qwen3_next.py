#!/usr/bin/env python
"""
Qwen3-Next 推理示例 — 混合线性注意力 (Gated DeltaNet 3 : 1 全注意力)

    1. 层排布 [Δ, Δ, Δ, A, ...]
    2. 因果性: DeltaNet 没有 mask, 但递推天然因果 (前缀 logits 不受未来 token 影响)
    3. 缓存: attn 层的 k/v 随 T 增长, delta 层的 state 大小恒定 —— 直接量 cache 里的张量
    4. 有/无 cache 输出逐 token 相同 (无 cache 时 DeltaNet 每步要从头递推, 所以加速比比纯注意力模型大)
    5. 左 padding: delta 层在 pad 位置不写状态, 真实位置的 logits 与不 pad 一致, 左 pad 批量生成 == 逐条生成
"""

import torch

from llm_models.models.language_models.qwen3_next import Qwen3Next
from llm_models.utils.generation import KVCache, benchmark_kv_cache


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size, T = 1000, 32
    model = Qwen3Next(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=8, max_len=128, linear_ratio=3, dropout=0.0,
    ).eval()
    pattern = " ".join("Δ" if t == "delta" else "A" for t in model.layer_types)
    print(f"Qwen3-Next Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[1] 层排布: [{pattern}]")
    assert model.layer_types.count("delta") == 6, "8 层、linear_ratio=3 时应有 6 层 DeltaNet"
    assert model.layer_types.count("attn") == 2, "8 层、linear_ratio=3 时应有 2 层全注意力"

    idx = torch.randint(1, vocab_size, (2, T))
    logits = model(idx)
    assert logits.shape == (2, T, vocab_size), "logits 形状应为 [B, T, V]"

    # 整段前向取前 short 位, 与只喂前 short 个 token 的前向比: 相等说明后面的 token 没影响前面
    short = 8
    diff = float((logits[:, :short] - model(idx[:, :short])).abs().max())
    assert diff < 1e-4, "DeltaNet 递推必须天然因果"
    print(f"[2] 因果性: |logits(全长)[:{short}] - logits(截断到 {short})| = {diff:.1e}")

    # 分别 prefill 8 个和 32 个 token, 按层的类型把 cache 里所有张量的元素数加起来
    sizes = {}
    long = T
    for n in (short, long):
        cache = KVCache(len(model.layers))
        model(idx[:, :n], cache=cache)
        sizes[n] = {
            kind: sum(t.numel() for c, k in zip(cache.layers, model.layer_types) if k == kind
                      for t in c.values())
            for kind in ("attn", "delta")
        }
    ratio = long // short
    assert sizes[long]["attn"] == ratio * sizes[short]["attn"], "attn 层的 k/v 缓存应随 T 线性增长"   # O(T)
    assert sizes[long]["delta"] == sizes[short]["delta"], "delta 层的状态大小应与 T 无关"             # O(1)
    print(f"[3] 缓存元素数 T={short} → T={long}:  attn 层 {sizes[short]['attn']} → {sizes[long]['attn']} "
          f"(×{ratio}),  delta 层 {sizes[short]['delta']} → {sizes[long]['delta']} (不变)")

    n_gen = 60
    speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=n_gen)
    print(f"[4] 生成 {n_gen} token: 有/无 cache 输出一致, 加速 {speedup:.1f}x")

    # ---- 5) 左 padding ----
    # 第 0 条前面补 P 个 pad。delta 层在 pad 位置 β=0、α=1, 状态原样传过去;
    # 不这样做的话 pad 会写进状态, 真实位置的 logits 就和不 pad 时不同
    P = 5
    real = idx[:1]                                                       # [1, T]
    padded = torch.cat([torch.zeros(1, P, dtype=torch.long), real], dim=1)       # [1, P+T], pad id = 0
    mask = torch.cat([torch.zeros(1, P), torch.ones(1, T)], dim=1)               # [1, P+T], 0 = pad
    pad_diff = float((model(padded, attention_mask=mask)[:, P:] - model(real)).abs().max())
    assert pad_diff < 1e-4, f"左 pad 改变了真实位置的 logits: {pad_diff}"
    # 长短两条 prompt 左 pad 成一批贪心生成, 应与各自单独生成逐 token 相同
    batch = torch.cat([padded[:, : P + short], idx[1:, : P + short]])   # [2, P+short]: 第 0 条短, 第 1 条满长
    bmask = torch.cat([mask[:, : P + short], torch.ones(1, P + short)])
    out = model.generate(batch, 10, temperature=0, attention_mask=bmask)
    solo = model.generate(real[:, :short], 10, temperature=0)
    assert torch.equal(out[:1, P:], solo), "左 pad 批量生成应与单独生成逐 token 相同"
    print(f"[5] 左 padding {P} 位: 真实位置 logits 最大偏差 {pad_diff:.1e}; 批量生成 == 单独生成")


if __name__ == "__main__":
    main()
