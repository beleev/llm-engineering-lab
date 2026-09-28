#!/usr/bin/env python
"""
Transformer (Encoder-Decoder) 推理: 用断言验证三种 mask 与 cross-attention, 再做一次贪心解码

盯住: encode() 只跑一次得到 memory; decode() 每步重跑 (教学版无 KV cache)。
"""

import torch

from llm_models.models.foundation.transformer import Transformer
from llm_models.utils.masks import get_pad_mask, get_subsequent_mask


def main():
    torch.manual_seed(42)
    V_src, V_tgt, B, S, T = 100, 120, 2, 7, 6                          # 源/目标词表, batch, 源长, 目标长
    model = Transformer(V_src, V_tgt, d_model=64, n_heads=4, num_layers=2, d_ff=128, use_rope=True).eval()
    print(f"Transformer Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    src = torch.randint(1, V_src, (B, S)); src[:, -2:] = 0             # 末尾 2 个 pad (pad_idx=0)
    tgt = torch.randint(1, V_tgt, (B, T))
    src_mask = get_pad_mask(src, pad_idx=0)                            # [B, 1, S]
    tgt_mask = get_pad_mask(tgt, pad_idx=0) & get_subsequent_mask(tgt)  # [B,1,T] & [1,T,T] -> [B,T,T]
    assert src_mask.shape == (B, 1, S), "src_mask 形状应为 [B, 1, S] (对所有 query 广播)"
    assert tgt_mask.shape == (B, T, T), "tgt_mask 形状应为 [B, T, T] (padding ∩ 因果)"

    with torch.inference_mode():
        out = model(src, tgt, src_mask, tgt_mask)                      # [B, T, V_tgt]
        assert out.shape == (B, T, V_tgt), "logits 形状应为 [B, T, V_tgt]"

        # 下面三个实验都是 "改一处输入, 看输出哪里变"; mask 始终用原来的
        # 1) 因果: 改 tgt 最后一个 token, 前面位置的 logits 不变
        #    x % (V-1) + 1 把 [1, V-1] 映射回 [1, V-1] 且一定不等于原值
        tgt2 = tgt.clone(); tgt2[:, -1] = tgt2[:, -1] % (V_tgt - 1) + 1
        d_causal = (model(src, tgt2, src_mask, tgt_mask)[:, :-1] - out[:, :-1]).abs().max().item()
        # 2) cross-attention: 改 src 的有效 token, 所有 tgt 位置的 logits 都变
        src2 = src.clone(); src2[:, 0] = src2[:, 0] % (V_src - 1) + 1
        #    amax(dim=(0, 2)) 得到每个 tgt 位置的最大变化 [T], 再取 min: 变化最小的位置也要变
        d_cross = (model(src2, tgt, src_mask, tgt_mask) - out).abs().amax(dim=(0, 2)).min().item()
        # 3) 源 padding: 改 pad 位置上的 token (mask 不变), 输出不变
        src3 = src.clone(); src3[:, -2:] = 5
        d_pad = (model(src3, tgt, src_mask, tgt_mask) - out).abs().max().item()
    print(f"改未来 tgt → 过去 logits 变化 {d_causal:.2e} | 改 src → 每个 tgt 位置至少变化 {d_cross:.4f} | 改 src pad → {d_pad:.2e}")
    assert d_causal < 1e-5, "因果 mask 失效: 改未来 token 影响了过去"
    assert d_cross > 1e-4, "cross-attn 没读到 src: 改了 src 的有效 token, 有的 tgt 位置输出没变"
    assert d_pad < 1e-5, "src padding 没被屏蔽: 改 pad 处的 token 影响了输出"

    # 4) 贪心解码: encoder 只跑一次; BOS=1
    with torch.inference_mode():
        memory = model.encode(src, src_mask)                           # [B, S, D]
        ys = torch.ones(B, 1, dtype=torch.long)                        # [B, 1] 起始只有 BOS
        for _ in range(5):
            # 每步把已生成的 ys 整段重跑 decoder, 只取最后一个位置的 logits 选下一个 token
            logits = model.decode(ys, memory, src_mask, get_subsequent_mask(ys))
            ys = torch.cat([ys, logits[:, -1].argmax(-1, keepdim=True)], dim=1)   # [B, t] → [B, t+1]
    print(f"贪心解码输出: {ys.tolist()}")
    assert ys.shape == (B, 6), "BOS 加 5 步贪心解码, 输出长度应为 6"


if __name__ == "__main__":
    main()
