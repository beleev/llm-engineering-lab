#!/usr/bin/env python
"""
Attention 的四个性质, 每个都用断言验证

1) 为什么除以 √d_k   2) 权重每行和为 1, 被 mask 的列为 0   3) 因果 mask 下过去不受未来影响
4) 没有位置编码的 self-attention 是置换等变的 (打乱输入 = 打乱输出) —— 所以 Transformer 必须另加位置信息
另有一条不打印的断言: rope 模块内部抛的 TypeError 会传出来, 不会被 _call_rope 吞掉。
"""

import math

import torch

from llm_models.layers.core.attention import MultiHeadAttention, ScaledDotProductAttention, _call_rope
from llm_models.utils.masks import build_causal_mask


def main():
    torch.manual_seed(42)
    B, T, D, H = 2, 16, 128, 4                                         # batch, 序列长, d_model, 头数

    # 1) q·k 的方差 ≈ d_k ⇒ 不缩放时 softmax 饱和
    #    q, k 各维独立 N(0,1): 每维乘积的方差是 1, d_k 维相加后方差就是 d_k
    d_k = 64
    Q, K = torch.randn(1000, d_k), torch.randn(1000, d_k)              # 1000 对随机 (q, k)
    raw = (Q * K).sum(-1)                                              # [1000] 每对的点积 q·k
    # p_raw / p_scaled: T 个 query 对 T 个 key 做 softmax, 取每行最大权重再平均。越接近 1 越饱和
    p_raw = torch.softmax(torch.randn(T, d_k) @ torch.randn(T, d_k).T, dim=-1).max(-1).values.mean().item()
    p_scaled = torch.softmax(torch.randn(T, d_k) @ torch.randn(T, d_k).T / math.sqrt(d_k), dim=-1).max(-1).values.mean().item()
    print(f"1) var(q·k) = {raw.var().item():.1f} (d_k={d_k}), 缩放后 {(raw / math.sqrt(d_k)).var().item():.2f}; "
          f"softmax 最大权重: 不缩放 {p_raw:.2f} vs 缩放 {p_scaled:.2f}")
    assert abs(raw.var().item() / d_k - 1) < 0.2, "q·k 的方差应 ≈ d_k (相对误差 < 20%)"
    assert p_raw > 2 * p_scaled, "不除以 √d_k 时 softmax 应更尖: 最大权重要超过缩放后的 2 倍"

    # 2) 权重是概率分布; 被 mask 的列权重为 0
    x = torch.randn(B, T, D)
    causal = build_causal_mask(T, x.device)                            # [1, T, T]
    _, w = ScaledDotProductAttention()(x, x, x, mask=causal)           # w: [B, T, T]
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5), "注意力权重每行之和应为 1"
    # ~causal 是被挡住的格子 (未来位置); 这些格子上的权重必须恰好是 0, 不能只是很小
    assert (w.masked_select(~causal.bool().expand_as(w)) == 0).all(), "被 mask 挡住的位置权重应恰为 0"
    print(f"2) 权重行和 = 1; 因果 mask 下第 0 行只看自己: w[0,0,:3] = {[round(v, 2) for v in w[0, 0, :3].tolist()]}")

    # 3)、4) 换成带 W_q / W_k / W_v / W_o 投影的多头注意力, 测的是完整的一层
    mha = MultiHeadAttention(D, H).eval()
    with torch.inference_mode():
        y = mha(x, mask=causal)                                        # [B, T, D]
        assert y.shape == x.shape, "MHA 输出形状应与输入相同 (形状不变才能接残差)"

        # 3) 因果: 改最后一个 token, 前 T-1 个输出不变
        x2 = x.clone(); x2[:, -1] += 1.0
        # [:, :-1] 去掉被改的最后一位, 只比前 T-1 个位置
        d_past = (mha(x2, mask=causal)[:, :-1] - y[:, :-1]).abs().max().item()
        assert d_past < 1e-6, "因果 mask 失效: 改最后一个 token 影响了前面位置的输出"

        # 4) 置换等变 (无 mask、无位置编码)
        #    先打乱输入再过 attention, 与先过 attention 再按同样顺序打乱输出, 两者应相等
        perm = torch.randperm(T)
        d_perm = (mha(x[:, perm]) - mha(x)[:, perm]).abs().max().item()
        assert d_perm < 1e-5, "无位置编码的 self-attention 应置换等变: attn(打乱 x) 应等于 打乱 attn(x)"
    print(f"3) 改未来 → 过去输出变化 {d_past:.1e}   4) attn(打乱 x) 与 打乱 attn(x) 的差 {d_perm:.1e}")

    # 注意力层靠 _call_rope 给 Q/K 加位置。rope 内部出错时, 错误必须冒出来,
    # 不能被当成 "这个 rope 不收 position_ids" 悄悄改成按 0..T-1 旋转
    class BrokenRope(torch.nn.Module):
        def forward(self, x, position_ids=None):
            if position_ids is not None:
                raise TypeError("rope 内部出错")
            return x

    try:
        _call_rope(BrokenRope(), x, torch.arange(T))
        swallowed = True
    except TypeError:
        swallowed = False
    assert not swallowed, "rope 内部的 TypeError 被 _call_rope 吞掉了"


if __name__ == "__main__":
    main()
