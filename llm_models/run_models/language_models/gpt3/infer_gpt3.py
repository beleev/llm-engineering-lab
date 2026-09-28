#!/usr/bin/env python
"""
GPT-3 推理示例 — 因果性 + KV cache

验证 (全部是 assert):
    1. 因果 mask: 改动未来的 token, 过去位置的 logits 一位都不变
    2. KV cache: 有/无 cache 的贪心生成逐 token 相同, 打印加速比
       (Sin-PE 与 RoPE 两种位置编码都测: 前者要平移 offset, 后者要平移 position_ids)
    3. 左 padding (不打印): RoPE 下真实位置 logits 不变; Sin-PE 下含 pad 的 mask 抛 NotImplementedError
"""

import torch

from llm_models.models.language_models.gpt3 import GPT3
from llm_models.utils.generation import benchmark_kv_cache


def main():
    torch.manual_seed(42)
    vocab_size, T = 1000, 16

    for use_rope in (False, True):
        # eval() 关掉 dropout=0.1: 两次前向才能逐位相等
        model = GPT3(
            vocab_size=vocab_size, d_model=256, n_heads=8, num_layers=4,
            max_len=256, dropout=0.1, use_rope=use_rope,
        ).eval()
        name = "RoPE" if use_rope else "Sin-PE"
        print(f"GPT-3 Mini ({name}) | 参数量: {sum(p.numel() for p in model.parameters()):,}")

        idx = torch.randint(1, vocab_size, (2, T))
        with torch.inference_mode():
            logits = model(idx)                                   # [B, T, V]
            future_changed = idx.clone()
            # 前半段原样保留, 后半段 (位置 ≥ T/2) 整段换成新的随机 token
            future_changed[:, T // 2:] = torch.randint(1, vocab_size, (2, T - T // 2))
            logits2 = model(future_changed)
        assert logits.shape == (2, T, vocab_size), "logits 形状应为 [B, T, V]"
        # 位置 < T/2 看不到被改动的后半段 → logits 必须完全不变
        assert torch.equal(logits[:, : T // 2], logits2[:, : T // 2]), "因果 mask 失效: 过去看到了未来"
        assert not torch.allclose(logits[:, T // 2:], logits2[:, T // 2:]), \
            "后半段输入换了, 后半段的 logits 却没变: 模型没在读输入"

        # 左 pad P 位: RoPE 只看相对位置, 真 token 的输出不变;
        # Sin-PE 是绝对位置, pad 会挪动真 token 的位置, 所以收到有 0 的 mask 直接报错
        P = 3
        padded = torch.cat([torch.zeros(2, P, dtype=torch.long), idx], dim=1)   # [B, P+T], pad id = 0
        mask = torch.cat([torch.zeros(2, P), torch.ones(2, T)], dim=1)          # [B, P+T], 0 = pad
        with torch.inference_mode():
            if use_rope:
                d_pad = (model(padded, attention_mask=mask)[:, P:] - logits).abs().max().item()
                assert d_pad < 1e-4, f"RoPE 下左 pad 不应改变真实位置的 logits: {d_pad}"
            else:
                try:
                    model(padded, attention_mask=mask)
                    rejected = False
                except NotImplementedError:
                    rejected = True
                assert rejected, "Sin-PE 下收到含 pad 的 mask 应抛 NotImplementedError, 不能静默算错"

        # 8 个 token 的 prompt 贪心生成 n_gen 个; 函数内部断言有/无 cache 逐 token 相同
        n_gen = 200
        speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=n_gen)
        print(f"  因果性通过 | 生成 {n_gen} token: 有/无 cache 输出完全一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()
