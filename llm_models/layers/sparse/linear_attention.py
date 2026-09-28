"""
线性注意力 — Gated DeltaNet (Yang et al., 2024)

论文出处:
    "Gated Delta Networks: Improving Mamba2 with Delta Rule" (NeurIPS 2024)
    Qwen3-Next (2025) 用它替换了 75% 的注意力层

动机:
    标准注意力的 KV cache 随上下文线性增长 (O(T)), 计算 O(T^2)。
    线性注意力把"显式缓存所有历史 K/V"换成"一个固定大小的状态矩阵":

        S_t ∈ R^{d_k × d_v}    —— 不管上下文多长, 状态恒定大小 (O(1) "cache")
        o_t = S_t^T q_t        —— 读取: 用 query 检索状态

    问题是怎么"写"这个状态。三代写法的演进:

        线性注意力 (2020):  S_t = S_{t-1} + k_t v_t^T          只加不减, 状态会"挤爆"
        DeltaNet   (2021):  S_t = (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
                             先擦掉 k_t 方向的旧值, 再写新值 —— delta rule,
                             等价于对 ||S^T k - v||^2 做一步在线梯度下降
        Gated DeltaNet:     S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
                             再加一个整体衰减门 α_t (Mamba2 的遗忘门),
                             "该忘的整体淡出, 该改的精准覆写"

混合架构 (Qwen3-Next / MiniMax 的共同选择):
    线性注意力召回精度有限 (固定大小状态是有损压缩), 纯线性模型长程检索弱。
    解法: 大部分层用线性注意力 (省), 周期性插入少量全注意力层 (准)。
    Qwen3-Next 取 3:1 —— 75% 层 O(1) 状态, 25% 层保留完整 KV。

实现说明:
    教学版用显式时间步循环 (与 layers/sparse/ssm.py 的 scan 一致, CPU 可读),
    真实实现用 chunk-wise 并行扫描在 GPU 上达到训练并行度。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet 线性注意力层 (多头, 教学版)。

    接口与库内注意力层对齐 (可直接装进 PreLNBlock):
        forward(q, k=None, v=None, mask=None, rope=None, position_ids=None)
    其中 rope 被接受但忽略: 位置信息由衰减门 α 隐式编码 (越旧的信息衰减越多), 不需要 RoPE
    (Qwen3-Next 只在全注意力层上用部分 RoPE)。
    mask 的含义和注意力层不同: 因果性由递推天然保证 (状态只能从过去流向未来), 不需要因果 mask;
    这里的 mask 是 [B, T] 的 "哪些是真 token", 用来让 pad 不写进状态 (见 forward)。

    Args:
        d_model:   隐藏维度
        num_heads: 头数 (每头独立维护一个 d_h × d_h 状态矩阵)
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 两个门都是"每头一个标量/步": α 整体遗忘, β 写入强度
        # α 初始化偏向"记住" (bias=+2 → sigmoid≈0.88), 训练初期不至于瞬间遗忘
        self.gate_alpha = nn.Linear(d_model, num_heads, bias=True)
        nn.init.constant_(self.gate_alpha.bias, 2.0)
        self.gate_beta = nn.Linear(d_model, num_heads, bias=True)

        # 输出门 (Mamba2 / Qwen3-Next 惯例): o ⊙ silu(W_g x)
        self.w_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,        # [B, T] 1=真 token 0=pad; None = 全是真 token
        rope: Optional[nn.Module] = None,            # 接受但忽略: 衰减门隐式编码位置
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,                # 解码用: cache["state"] 就是全部 "KV cache"
    ) -> torch.Tensor:
        """q [B, T, D] → [B, T, D]。只做自注意力: k / v 不读, 只用 q。

        mask: [B, T], 只管本次的 T 个 token。pad 位置取 β=0 (不写)、α=1 (不衰减),
            状态原样传过去, 所以左 pad 不改真 token 的输出。pad 位置自己的输出没有意义。
        cache: 给了就从 cache["state"] [B, H, Dh, Dh] 接着递推, 结束时写回。
        """
        x = q                                        # PreLNBlock 传 q=k=v=h
        B, T, D = x.shape
        H, Dh = self.num_heads, self.d_head

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, Dh).transpose(1, 2)          # [B, T, D] → [B, T, H, Dh] → [B, H, T, Dh]

        # q/k 按 DeltaNet 惯例做 L2 归一化: 保证 (I - β k k^T) 的谱半径 <= 1,
        # 递推不会数值爆炸 (k k^T 是到 k 方向的投影, β∈(0,1) 时是收缩映射)
        qh = F.normalize(split_heads(self.w_q(x)), dim=-1)      # [B, H, T, Dh], 每个向量模长 1
        kh = F.normalize(split_heads(self.w_k(x)), dim=-1)      # [B, H, T, Dh]
        vh = split_heads(self.w_v(x))                           # [B, H, T, Dh]

        # 每步每头一个标量门: [B, T, D] → [B, T, H] → [B, H, T], sigmoid 压到 (0, 1)
        alpha = torch.sigmoid(self.gate_alpha(x)).transpose(1, 2)   # [B, H, T]
        beta = torch.sigmoid(self.gate_beta(x)).transpose(1, 2)     # [B, H, T]
        if mask is not None:
            # pad 位置: β=0 写入项为 0, α=1 旧状态不衰减 → S 原样传给下一步
            real = mask.to(x.dtype).unsqueeze(1)                    # [B, T] → [B, 1, T], 在 head 维广播
            beta = beta * real
            alpha = alpha * real + (1 - real)

        # 状态矩阵: 每个 (batch, head) 一个 Dh×Dh —— 这就是全部"KV cache"
        # 有 cache 时从上一步的状态接着递推 (大小与已读过多少 token 无关 → O(1))
        state = cache["state"] if cache and "state" in cache else x.new_zeros(B, H, Dh, Dh)
        outs = []
        for t in range(T):
            k_t = kh[:, :, t]                        # [B, H, Dh]
            v_t = vh[:, :, t]                        # [B, H, Dh]
            a_t = alpha[:, :, t, None, None]         # [B, H] → [B, H, 1, 1], 好和状态矩阵广播
            b_t = beta[:, :, t, None, None]          # [B, H, 1, 1]

            # delta rule: 先读出 k_t 方向当前存的值, 擦掉, 再写入新值
            #   S ← α (S - β k (k^T S)) + β k v^T
            # einsum 下标: d = key 维, e = value 维。"bhd,bhe->bhde" 是外积 k·vᵀ
            k_read = torch.einsum("bhd,bhde->bhe", k_t, state)       # k^T S  [B,H,Dh]
            state = a_t * (state - b_t * torch.einsum("bhd,bhe->bhde", k_t, k_read)) \
                + b_t * torch.einsum("bhd,bhe->bhde", k_t, v_t)      # [B, H, Dh, Dh]

            # 读取: o_t = S^T q_t。先写后读, 所以位置 t 能读到自己刚写进去的内容
            outs.append(torch.einsum("bhd,bhde->bhe", qh[:, :, t], state))   # [B, H, Dh]

        if cache is not None:
            cache["state"] = state

        o = torch.stack(outs, dim=2)                 # T 个 [B, H, Dh] → [B, H, T, Dh]
        o = o.transpose(1, 2).reshape(B, T, D)       # [B, H, T, Dh] → [B, T, H, Dh] → [B, T, D]

        # 输出门 + 输出投影
        o = o * F.silu(self.w_gate(x))               # [B, T, D] 逐元素门控
        return self.w_o(o)

    def state_size_per_token(self) -> int:
        """与序列长度无关的状态大小 (元素数) —— 对比 GQA 的 O(T) KV cache。"""
        return self.num_heads * self.d_head * self.d_head
