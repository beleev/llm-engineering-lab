"""
门控注意力 — Gated Attention (Qiu et al., Qwen, arXiv 2505.06708, NeurIPS 2025)

是什么: 在缩放点积注意力的输出上, 按 token 现算一个 sigmoid 门再逐元素乘, 然后才过输出投影

    y = W_o [ σ(W_g x) ⊙ SDPA(q, k, v) ]

    门的形状是 [H·Dh], 每个 head 的每个通道各一个, 由当前位置的隐状态决定。
解决什么: softmax 的行和恒为 1, 每个 query 必须把这 1 份概率质量分完。没有合适的 key 时, 模型
    只能把质量倒给一个 "无害" 的位置, 最常见的就是第 0 个 token, 这就是 attention sink;
    与它伴随的是少数几个维度上的大激活 (massive activations), BF16 下容易放大数值误差。
    输出门让注意力可以 "读了也不用": 质量照样分完, 但乘上一个接近 0 的门之后不进残差。
    论文实测基线平均有 46.7% 的注意力分数落在第 0 个 token 上, 加门之后降到 4.8%,
    大激活也随之减小; 论文还指出大激活不是 attention sink 的前提条件。
谁在用: Qwen3-Next 的全注意力层用的就是这一种 (论文里最有效的 SDPA 输出门);
    Qwen3.5-397B-A17B 的 60 层按 15 组 "3 层 Gated DeltaNet + 1 层 Gated Attention" 排布。
    Kimi K3 给 MLA 加的也是同一类逐通道输出门 (那边叫 Gated MLA)。
对照: GPT-OSS 的 attention sink 是另一条路 —— 给每个 head 一个可学的额外 logit 参与 softmax
    再丢掉, 让行和可以小于 1; 门控注意力不动 softmax, 在输出侧解决同一个问题。
本库的做法: 继承 GroupedQueryAttention, 只加一个门。use_gate=False 时与它逐位相同,
    所以 "有门 / 无门" 的对照是同一条代码路径, 只差那一次乘法。
简化: 只实现论文里的 SDPA 输出门, 没有实现它比较过的 query / key / value 侧门控与逐头标量门。
读代码时盯住: last_sink_mass —— 每次前向顺手记下第 0 列分走的平均注意力质量, 两种模式可直接比。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention, _call_rope, _normalize_attn_mask


class GatedAttention(GroupedQueryAttention):
    """
    GQA 加一个 SDPA 输出门。接口与 GroupedQueryAttention 完全相同。

    Args:
        use_gate: False 时不乘门, 与 GroupedQueryAttention 逐位相同 (做对照用)
        其余参数与 GroupedQueryAttention 相同 (d_model / num_heads / num_kv_heads / bias /
        qk_norm / use_sink)
    """

    def __init__(self, d_model: int, num_heads: int, use_gate: bool = True, **kwargs):
        super().__init__(d_model, num_heads, **kwargs)
        self.use_gate = use_gate
        # 逐 head 逐通道的门: [D] → [H·Dh]。初始权重由 utils/init.py 给 N(0, 0.02²),
        # 所以起步时 logit ≈ 0、门 ≈ 0.5, 相当于把输出整体缩小一半再交给 w_o
        self.w_g = nn.Linear(d_model, num_heads * self.head_dim, bias=False) if use_gate else None
        self.last_sink_mass: Optional[float] = None    # 第 0 列分走的平均注意力质量

    def forward(
        self, q, k=None, v=None, mask=None, rope=None, position_ids=None,
        cache: Optional[dict] = None,
    ):
        """q [B, T, D], k / v [B, S, D] (None = 自注意力) → [B, T, D]。"""
        x = q
        k = q if k is None else k
        v = q if v is None else v
        B, T, _ = q.shape
        S = k.size(1)

        Q = self.w_q(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T, Dh]
        K = self.w_k(k).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, Hkv, S, Dh]
        V = self.w_v(v).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            Q, K = self.q_norm(Q), self.k_norm(K)
        if rope is not None:
            Q = _call_rope(rope, Q, position_ids)
            K = _call_rope(rope, K, position_ids)
        if cache is not None:                      # 先旋转再进 cache, 与 GQA 一致
            if "k" in cache:
                K = torch.cat([cache["k"], K], dim=2)
                V = torch.cat([cache["v"], V], dim=2)
            cache["k"], cache["v"] = K, V
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale                     # [B, H, T, S]
        norm_mask = _normalize_attn_mask(mask)
        if norm_mask is not None:
            scores = scores.masked_fill(norm_mask == 0, float("-inf"))
        if self.sink is not None:
            sink = self.sink.view(1, -1, 1, 1).expand(B, -1, T, 1)
            attn = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        else:
            attn = F.softmax(scores, dim=-1)                                           # [B, H, T, S]
        out = torch.matmul(attn, V)                                                    # [B, H, T, Dh]

        # 统计第 0 列分走多少: 跳过第 0 行 (它只有自己可看, 必然是 1), 剩下的行取平均
        if T > 1:
            self.last_sink_mass = attn[:, :, 1:, 0].mean().item()

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        if self.w_g is not None:
            out = out * torch.sigmoid(self.w_g(x))     # ★ 输出门: 读了也可以不用
        return self.w_o(out)
