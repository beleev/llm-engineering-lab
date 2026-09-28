"""
注意力模块 — 一条主线: 每一代都在压 KV cache / 计算量, 同时尽量不掉效果

    softmax(QKᵀ/√d)·V                       ScaledDotProductAttention (2017, 数学核心)
    → MHA   每 head 一份 K/V, cache 最大      MultiHeadAttention (教学版逐头循环, 便于可视化)
    → GQA   H 个 Q head 共享 Hkv 对 K/V       GroupedQueryAttention (cache ÷ H/Hkv; Hkv=1 即 MQA)
            + 可选 QK-Norm / attention sink / KV cache
    → MLA   K/V 压成低秩 latent c_kv          MultiHeadLatentAttention (DeepSeek-V2/V3, 解耦 RoPE)
            cache 每 token 从 2·H·Dh 个数降到 r + rope 个 (V3 配置下 ↓ ~98%, 算式见类 docstring)
    → DSA   MLA + Lightning Indexer 选 top-k  MultiHeadLatentSparseAttention (V3.2, O(L²) → O(L·k))

统一接口 (SingleHead / MHA / GQA / MLA / DSA):
    forward(q, k, v, mask, rope, position_ids[, cache]) → Tensor [B, T, D_out]; mask 为 bool, True = 可见。
例外:
    - ScaledDotProductAttention 返回 (out, weights)
    - LightningIndexer 返回打分 [B, T, S]
    - MLA 的 return_attn=True 返回 (out, attn)
KV cache 协议见 llm_models/utils/generation.py。
读代码时盯住: scores 的形状 [B, H, T, S] —— T 是 query 数 (解码时 = 1), S 是 key 数 (= 已缓存 + 本步)。
"""

import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.normalization import RMSNorm


def _normalize_attn_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """把任意 2D/3D/4D 掩码扩到 [B, 1, T, S] 或 [1, 1, T, S] 以便广播。"""
    if mask is None:
        return None
    if mask.dim() == 2:  # [T, S] -> [1, 1, T, S]
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:  # [B, T, S] -> [B, 1, T, S]
        return mask.unsqueeze(1)
    return mask  # [B, 1, T, S] or [B, H, T, S]


def _call_rope(
    rope: nn.Module,
    x: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """给 x 加旋转位置。position_ids 为 None 时按 0..T-1 旋转, 否则原样传给 rope。

    rope 模块的 forward 没有 position_ids 形参时退回 rope(x), 按 0..T-1 旋转。
    看签名而不用 try/except TypeError: 后者会把 rope 内部真正的 TypeError 吞掉,
    悄悄改成按 0..T-1 旋转, 算错了也不报错。
    """
    if position_ids is None or "position_ids" not in inspect.signature(rope.forward).parameters:
        return rope(x)
    return rope(x, position_ids=position_ids)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 — Transformer 的最小数学核心

    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    为什么除以 sqrt(d_k):
        Q·K 内积的方差随 d_k 线性增长。d_k 大时 softmax 被推到饱和区
        (概率几乎全压在一个 token 上), 反传梯度趋近于 0。
        除以 sqrt(d_k) 把内积方差拉回 O(1)。
    """

    def forward(self, Q, K, V, mask=None):
        """Q [..., T, d_k], K [..., S, d_k], V [..., S, d_v], mask 可广播到 [..., T, S]。

        返回 (output [..., T, d_v], attn_weights [..., T, S])。T = query 数, S = key 数。
        """
        d_k = Q.size(-1)

        # K 转置后 [..., d_k, S]; Q·Kᵀ 得到每个 query 对每个 key 的分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # [..., T, S]

        if mask is not None:
            # mask 为 0/False 表示屏蔽: 填 -inf 后 softmax 给出的概率恰好是 0
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [..., T, S] 沿 key 维归一化, 每行和为 1
        output = torch.matmul(attn_weights, V)    # [..., T, d_v] 按权重对 V 加权求和
        return output, attn_weights


class SingleHeadSelfAttention(nn.Module):
    """
    单头自注意力 (教学用)

    生产代码通常直接写多头; 单头版本保留用于教学:
        - 便于把注意力权重可视化成单张热力图
        - 便于断点观察 Q/K/V 的形状与数值范围
    支持可选的 RoPE 注入; 真实多头见下方 MultiHeadAttention / GQA。
    """

    def __init__(self, d_input: int, d_out: int):
        super().__init__()
        self.w_q = nn.Linear(d_input, d_out)
        self.w_k = nn.Linear(d_input, d_out)
        self.w_v = nn.Linear(d_input, d_out)
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        """q [B, T, d_input], k / v [B, S, d_input] → [B, T, d_out]。

        k / v 为 None 时是自注意力 (Q=K=V 同源), 否则是交叉注意力。
        mask 可广播到 [B, T, S], True = 可见。注意力权重被丢弃, 只返回输出。
        """
        k = k if k is not None else q
        v = v if v is not None else q

        q_proj = self.w_q(q)   # [B, T, d_out]
        k_proj = self.w_k(k)   # [B, S, d_out]
        v_proj = self.w_v(v)   # [B, S, d_out]

        # RoPE 只转 Q/K (位置参与匹配), V 不旋转 (V 携带的是内容值)
        if rope is not None:
            q_proj = _call_rope(rope, q_proj, position_ids)
            k_proj = _call_rope(rope, k_proj, position_ids)

        output, _ = self.attention(q_proj, k_proj, v_proj, mask)   # [B, T, d_out]
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力 (教学版) — Vaswani et al. 2017 原始 MHA

    多头的动机:
        单头 attention 只能学到一种 (Q,K) 相关性; 多头让不同 head 在不同子空间
        关注不同模式 (语法 / 语义 / 位置邻近 …), 再拼接融合。

    本实现用 nn.ModuleList 逐头循环, 运算等价于标准 MHA 但更慢,
    好处是可以逐头单独观察、调试。生产代码请用 GroupedQueryAttention
    (num_kv_heads = num_heads 即等价于 MHA)。
    不支持 KV cache: forward 没有 cache 参数。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(d_input=d_model, d_out=self.d_head) for _ in range(num_heads)]
        )
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        """q [B, T, D], k / v [B, S, D] (None = 自注意力) → [B, T, D]。

        mask 可广播到 [B, T, S], True = 可见; 所有 head 共用同一张 mask。
        """
        # 每个 head 独立投影并算 attention, 各输出 [B, T, d_head]
        head_outputs = [
            head(q, k, v, mask, rope=rope, position_ids=position_ids) for head in self.heads
        ]
        # 沿特征维拼接: H 个 [B, T, d_head] → [B, T, H·d_head] = [B, T, D]
        concat_output = torch.cat(head_outputs, dim=-1)
        return self.w_o(concat_output)   # 输出投影 W_O 把各头的结果混合, [B, T, D]


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) — Ainslie et al., 2023

    KV cache 随 KV head 数线性增长。GQA 让 num_groups = H / Hkv 个 Q head 共享一对 K/V head,
    cache ÷ num_groups, 质量几乎不掉。Hkv == H → MHA;  Hkv == 1 → MQA。

    可选零件 (默认全关, 关掉时与经典 GQA 逐位相同):
        qk_norm:  Q/K 在 RoPE 之前各过一个 head_dim 上的 RMSNorm (Qwen3 / OLMo-2)。
                  logit = |q||k|cosθ·scale, scale = 1/√Dh。
                  归一化后 |q| = |k| = √Dh (γ=1 时), 所以 |logit| ≤ √Dh。
                  γ 可学, 上界实际是 √Dh·max|γ_q|·max|γ_k|:
                  由两组小参数控制, 与 W_q / W_k 的大小无关。
        use_sink: 每个 head 一个可学 logit, 作为额外一列参与 softmax 后丢弃 (GPT-OSS)。
                  让 head 可以 "谁都不看" (概率质量倒进 sink), 而不是被迫把 1 分完。
        cache:    见 llm_models/utils/generation.py。存的是 RoPE 之后、复制分组之前的 K/V。

    Args:
        d_model / num_heads: 模型维度 / Q head 数
        num_kv_heads: K/V head 数, 默认等于 num_heads (MHA)
        bias: 现代 LLM 普遍 False; GPT-3 风格传 True
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = False,
        qk_norm: bool = False,
        use_sink: bool = False,
    ):
        super().__init__()

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_q = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        # sink logit 初始 0: 相当于多一个 "分数为 0 的空 key"
        self.sink = nn.Parameter(torch.zeros(num_heads)) if use_sink else None

    def forward(
        self, q, k=None, v=None, mask=None, rope=None, position_ids=None,
        cache: Optional[dict] = None,
    ):
        """q [B, T, D], k / v [B, S, D] (None = 自注意力) → [B, T, D]。

        mask: 2D / 3D / 4D 都行, 会扩到 [B, 1, T, S]; True = 可见。
        cache: 本层的 dict。给了就把本步 K/V 追加进去, 此时 S = 已缓存 + 本步。
        """
        k = q if k is None else k
        v = q if v is None else v

        B, T, _ = q.shape
        S = k.size(1)

        # 切头: [B, T, D] → [B, T, H, Dh] → [B, H, T, Dh]
        # head 维挪到前面后, matmul 对每个 head 独立做矩阵乘
        Q = self.w_q(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T, Dh]
        K = self.w_k(k).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, Hkv, S, Dh]
        V = self.w_v(v).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, Hkv, S, Dh]

        if self.q_norm is not None:            # QK-Norm 在 RoPE 之前: 旋转不改范数
            Q, K = self.q_norm(Q), self.k_norm(K)

        if rope is not None:                   # 只转 Q/K; V 携带内容, 不带位置
            Q = _call_rope(rope, Q, position_ids)
            K = _call_rope(rope, K, position_ids)

        # 顺序不能换: 先旋转再进 cache。cache 里的 K 已带各自的绝对位置, 以后不用再转
        if cache is not None:                  # 追加本步 K/V (已旋转), 然后对全部历史做注意力
            if "k" in cache:
                K = torch.cat([cache["k"], K], dim=2)     # [B, Hkv, S_past+S, Dh]
                V = torch.cat([cache["v"], V], dim=2)     # [B, Hkv, S_past+S, Dh]
            cache["k"], cache["v"] = K, V

        # KV head 复制 num_groups 份对齐 Q head (FlashAttention 等 kernel 会跳过这步拷贝)
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)   # [B, Hkv, S, Dh] → [B, H, S, Dh]
            V = V.repeat_interleave(self.num_groups, dim=1)   # [B, Hkv, S, Dh] → [B, H, S, Dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # [B, H, T, S]

        norm_mask = _normalize_attn_mask(mask)                        # [B|1, 1, T, S], 在 head 维广播
        if norm_mask is not None:
            scores = scores.masked_fill(norm_mask == 0, float("-inf"))

        if self.sink is not None:
            sink = self.sink.view(1, -1, 1, 1).expand(B, -1, T, 1)   # [H] → [1, H, 1, 1] → [B, H, T, 1]
            # sink 当作第 S+1 个 key 一起 softmax, 再用 [..., :-1] 丢掉它那一列:
            # 它分走的概率不乘任何 V, 所以剩下 S 列的行和 < 1
            attn = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]  # [B, H, T, S]
        else:
            attn = F.softmax(scores, dim=-1)                          # [B, H, T, S]
        out = torch.matmul(attn, V)                                   # [B, H, T, Dh]

        # 合头: [B, H, T, Dh] → [B, T, H, Dh] → [B, T, D]
        # transpose 后内存不连续, view 前必须 contiguous
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.w_o(out)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) — DeepSeek-V2/V3 论文核心

    GQA 靠减少 KV head 省 cache, 但每 token 仍要存 2·Hkv·Dh 个数。
    MLA 换个思路: 把 K/V 压进一个低秩 latent, 每 token 只存 (r + rope) 个数:

        c_kv   = W_DKV x                 [B, T, r]      ← 缓存它
        k_rope = RoPE(W_KR x)            [B, 1, T, rope] ← 和它 (所有 head 共享)
        K = [W_UK c_kv | k_rope],  V = W_UV c_kv        ← 每步现场升维, 不缓存

    为什么要 "解耦 RoPE": RoPE 是随位置变化的旋转, 直接转 c_kv 升维出的 K 会让 W_UK
    无法被吸收进 Q 侧; 所以每头维度拆成 nope 段 (走 latent, 不旋转) + rope 段 (小投影, 旋转)。

    关键数字: DeepSeek-V3 r=512, rope=64 → 576 floats/token/层; 同规模 MHA (128 头×128 维)
    要 32768 → 省 ~98%。读代码时盯住 forward 里的 c_kv / k_rope 两个变量。

    Args:
        kv_lora_rank:     latent 维度 r
        qk_nope_head_dim: Q/K 每头不旋转段
        qk_rope_head_dim: Q/K 每头旋转段 (传入的 rope 模块 head_dim 必须等于它)
        v_head_dim:       V 每头维度, 默认 = qk_nope_head_dim
        q_lora_rank:      Q 的低秩维度 (可选, V3 用它省训练激活显存)
        latent_dim:       kv_lora_rank 的别名 (DeepSeekV3 的构造参数用这个名字);
                          两个都给时以 latent_dim 为准
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        bias: bool = False,
        latent_dim: Optional[int] = None,   # kv_lora_rank 的别名, 见类 docstring
    ):
        super().__init__()

        if latent_dim is not None:
            kv_lora_rank = latent_dim

        v_head_dim = v_head_dim if v_head_dim is not None else qk_nope_head_dim

        self.d_model = d_model
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # --- Q 投影 (可选低秩) ---
        if q_lora_rank is not None:
            self.q_down = nn.Linear(d_model, q_lora_rank, bias=bias)
            self.q_up = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.q_down = None
            self.q_up = nn.Linear(d_model, num_heads * self.qk_head_dim, bias=bias)

        # --- KV 低秩 + 解耦 K rope (一次投影到 [kv_lora + rope_dim]) ---
        self.kv_down = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.k_up = nn.Linear(kv_lora_rank, num_heads * qk_nope_head_dim, bias=bias)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=bias)

        # --- 输出投影 ---
        self.w_o = nn.Linear(num_heads * v_head_dim, d_model, bias=bias)

    def forward(
        self, q, k=None, v=None, mask=None, rope=None, position_ids=None,
        cache: Optional[dict] = None, return_attn: bool = False,
    ):
        """
        q [B, T, D] → [B, T, D]; return_attn=True 时返回 (out, attn)。

        cache:       每层一个可变 dict。只存 c_kv [B, S, r] 与 post-RoPE 的共享 k_rope
                     [B, 1, S, rope]; K/V 每步从 latent 现场升维 —— MLA 省 cache 的全部秘密。
        return_attn: True 时额外返回 detach 的注意力概率 [B, H, T, S] (DSA indexer 对齐 loss 的目标)。
        """
        # MLA 仅用于 self-attention; k/v 参数只为统一签名
        x = q
        B, T, _ = x.shape
        H = self.num_heads

        # --- 1) Q: (可选低秩) 投影 → 切头 → 拆 nope/rope 两段 ---
        q_proj = self.q_up(self.q_down(x)) if self.q_down is not None else self.q_up(x)  # [B, T, H·(nope+rope)]
        q_proj = q_proj.view(B, T, H, self.qk_head_dim).transpose(1, 2)  # [B, H, T, nope+rope]
        q_nope, q_rope = torch.split(                                    # [B, H, T, nope], [B, H, T, rope]
            q_proj, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # --- 2) KV 压缩: 一次投到 [c_kv | k_rope] ---
        kv_mix = self.kv_down(x)  # [B, T, r + rope]
        c_kv, k_rope = torch.split(                                      # c_kv [B, T, r], k_rope [B, T, rope]
            kv_mix, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_rope = k_rope.unsqueeze(1)  # [B, T, rope] → [B, 1, T, rope], 所有 head 共享

        # --- 3) 只旋转 rope 段 (c_kv 不能旋转, 否则 W_UK 无法被吸收) ---
        if rope is not None:
            q_rope = _call_rope(rope, q_rope, position_ids)
            k_rope = _call_rope(rope, k_rope, position_ids)

        # --- KV cache: 追加本步的 latent, 之后对全部 S 个位置做注意力 ---
        if cache is not None:
            if "c_kv" in cache:
                c_kv = torch.cat([cache["c_kv"], c_kv], dim=1)        # [B, S, r]
                k_rope = torch.cat([cache["k_rope"], k_rope], dim=2)  # [B, 1, S, rope]
            cache["c_kv"], cache["k_rope"] = c_kv, k_rope
        S = c_kv.size(1)

        # 从 (缓存的) latent 现场升维出 K-nope 和 V
        k_nope = self.k_up(c_kv).view(B, S, H, self.qk_nope_head_dim).transpose(1, 2)  # [B, H, S, nope]
        v_heads = self.v_up(c_kv).view(B, S, H, self.v_head_dim).transpose(1, 2)       # [B, H, S, v]

        # --- 4) 拼 [nope | rope] 后做标准缩放点积注意力 ---
        q_combined = torch.cat([q_nope, q_rope], dim=-1)                        # [B, H, T, nope+rope]
        k_combined = torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], dim=-1)  # [B, H, S, nope+rope]

        scores = torch.matmul(q_combined, k_combined.transpose(-2, -1)) * self.scale  # [B, H, T, S]

        norm_mask = _normalize_attn_mask(mask)
        if norm_mask is not None:
            scores = scores.masked_fill(norm_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)   # [B, H, T, S]
        out = torch.matmul(attn, v_heads)  # [B, H, T, v]

        # 合头: [B, H, T, v] → [B, T, H, v] → [B, T, H·v] → w_o → [B, T, D]
        out = self.w_o(out.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim))
        # attn 要 detach: 它只当 indexer 的学习目标, index_loss 的梯度不该流回主注意力
        return (out, attn.detach()) if return_attn else out


class LightningIndexer(nn.Module):
    """
    Lightning Indexer — DSA (DeepSeek V3.2) 的廉价选择器

    "先粗选后细算": 用几个又小又便宜的 head 给每对 (t, s) 打分, 只留每行 top-k,
    昂贵的 MLA 只在这 k 个位置上算。

        I[t, s] = Σ_h ReLU( q_h[t] · k_h[s] / sqrt(d) )      # 只需要排序, 不需要归一化

    top-k 不可导 → indexer 从 LM loss 拿不到任何梯度。它靠单独的对齐 loss 训练
    (见 MultiHeadLatentSparseAttention): 让 softmax(I) 去拟合主注意力的分布。
    教学简化: 官方还有按 query 生成的 head 权重 w[t,h] 和 indexer 自己的 RoPE, 此处省略。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除，"
                    "或显式指定 head_dim"
                )
            head_dim = d_model // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.w_q = nn.Linear(d_model, num_heads * head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, num_heads * head_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Hi = indexer 的头数, Di = indexer 每头维度 (都比主注意力小)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, T, Hi·Di] → [B, Hi, T, Di]

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """返回 index_scores [B, T, S]; 不可见位置为 -inf。cache 里多存一份 indexer 的 key。"""
        k = k if k is not None else q
        q_heads = self._split_heads(self.w_q(q))  # [B, Hi, T, Di]
        k_heads = self._split_heads(self.w_k(k))  # [B, Hi, S, Di]

        if cache is not None:  # decode: 历史 token 的 indexer key 也要缓存, 否则没法给旧位置打分
            if "idx_k" in cache:
                k_heads = torch.cat([cache["idx_k"], k_heads], dim=2)   # [B, Hi, S_past+S, Di]
            cache["idx_k"] = k_heads

        # 每头的 q·k 内积: 对最后一维 d 求和
        scores = torch.einsum("bhtd,bhsd->bhts", q_heads, k_heads) * self.scale   # [B, Hi, T, S]
        scores = F.relu(scores).sum(dim=1)  # 各头 ReLU 后相加, [B, Hi, T, S] → [B, T, S]

        norm_mask = _normalize_attn_mask(mask)                                    # [B|1, 1, T, S]
        if norm_mask is not None:
            # squeeze(1) 去掉 head 维, 和 [B, T, S] 的 scores 对齐
            scores = scores.masked_fill(~norm_mask.squeeze(1).bool(), float("-inf"))

        return scores


class MultiHeadLatentSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) + MLA — DeepSeek V3.2 (2025)

    MLA 省了 cache, 但注意力算力仍是 O(L²)。DSA: indexer 打分 → (先 mask 再) top-k →
    MLA 只在选中的 k 个 key 上算, 复杂度 O(L·k)。

    indexer 怎么训练 (top-k 不可导, LM loss 给不了梯度):
        p[t,:] = mean_h 主注意力概率 (detach)             # 老师
        q[t,:] = softmax(I[t,:])  (同一可见集合上)         # 学生
        index_loss = KL(p ‖ q)
      - indexer 的输入也 detach: index_loss 只更新 indexer, LM loss 只更新主模型, 互不干扰。
      - dense_warmup=True  (论文第 1 阶段): MLA 看全部因果可见位置, indexer 拟合稠密注意力;
        dense_warmup=False (第 2 阶段):     MLA 只看 top-k, KL 也只在选中集合上算。

    每次有梯度的 forward 后可读: last_index_loss (标量), last_index_scores / last_attn_target
    ([B, T, S], detach, 供监控 top-k 召回率)。
    教学实现仍构造稠密 [B, T, S] mask; 生产版靠自定义 kernel 真正跳过未选中的 KV。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        indexer_heads: int = 4,
        indexer_head_dim: Optional[int] = None,
        sparse_top_k: int = 128,
        bias: bool = False,
        latent_dim: Optional[int] = None,   # kv_lora_rank 的别名, 原样传给 MLA
    ):
        super().__init__()

        if sparse_top_k <= 0:
            raise ValueError(f"sparse_top_k 必须为正数，当前 {sparse_top_k}")

        self.sparse_top_k = sparse_top_k
        self.dense_warmup = False    # True: MLA 走稠密, indexer 只旁听学习 (论文 warm-up 阶段)
        self.last_index_loss = None  # 每次带梯度的 forward 后刷新

        self.mla =MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            bias=bias,
            latent_dim=latent_dim,
        )

        self.indexer = LightningIndexer(
            d_model=d_model,
            num_heads=indexer_heads,
            head_dim=indexer_head_dim,
            bias=bias,
        )

    def _sparse_mask_from_topk(
        self,
        index_scores: torch.Tensor,
        base_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        在 base_mask 限定的可见区域内, 每行取分数最高的 k 个 key。

        index_scores [B, T, S], base_mask [B|1, T, S] (True = 可见) → sparse_mask [B, T, S] bool。
        key 总数 S ≤ k 时不做稀疏, 直接返回 base_mask (没有就全可见)。
        """
        B, T, S = index_scores.shape
        k = min(self.sparse_top_k, S)

        if k >= S:  # 稀疏退化为稠密
            if base_mask is None:
                return torch.ones_like(index_scores, dtype=torch.bool)
            return base_mask.bool().expand(B, T, S)   # [B|1, T, S] → [B, T, S]

        # 必须先 mask 再 top-k, 否则会选中未来 token (数据泄漏)
        scores = index_scores
        if base_mask is not None:
            scores = scores.masked_fill(~base_mask.bool(), float("-inf"))

        # 不可导: 梯度到此为止。用 stable sort 而非 torch.topk: ReLU 后大量分数恰为 0,
        # topk 对并列的取舍随行长而变 → 带 cache 的 decode 会和整段 forward 选出不同的 key。
        # stable sort 下并列者固定取靠前的位置, 两条路径一致。
        topk_indices = scores.sort(dim=-1, descending=True, stable=True).indices[..., :k]   # [B, T, k]
        sparse = torch.zeros(B, T, S, dtype=torch.bool, device=index_scores.device)
        sparse.scatter_(-1, topk_indices, True)   # 每行把选中的 k 个下标置 True, [B, T, S]

        if base_mask is not None:
            sparse = sparse & base_mask.bool()  # 可见位置不足 k 个的行会选到 -inf, 这里剔掉
        return sparse

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None, cache=None):
        """q [B, T, D] → [B, T, D]。只做自注意力, k / v 参数只为统一签名。

        cache 同时装 MLA 的 c_kv / k_rope 和 indexer 的 idx_k。
        副作用: 带梯度时刷新 self.last_index_loss (标量), 由外层模型取走加进总 loss。
        """
        # 1) indexer 打分。输入 detach: index_loss 不应改动主干的表示
        index_scores = self.indexer(q.detach(), mask=mask, cache=cache)  # [B, T, S]

        base_mask = None if mask is None else _normalize_attn_mask(mask).squeeze(1)  # [B|1, T, S]
        sparse_mask = self._sparse_mask_from_topk(index_scores, base_mask)           # [B, T, S]
        used_mask = sparse_mask
        if self.dense_warmup and base_mask is not None:
            used_mask = base_mask.bool().expand_as(sparse_mask)                      # [B, T, S]

        # 2) MLA 只在 used_mask 允许的位置上做注意力
        out, attn = self.mla(
            q, mask=used_mask, rope=rope, position_ids=position_ids,
            cache=cache, return_attn=True,
        )  # attn: [B, H, T, S], 已 detach

        # 3) indexer 对齐 loss: KL( 主注意力 ‖ softmax(indexer 分数) ), 都限定在 used_mask 上
        self.last_index_loss = None
        if torch.is_grad_enabled():   # 推理 (no_grad) 时不算 loss, 省一次 log_softmax
            p = attn.mean(dim=1)  # [B, H, T, S] → [B, T, S]; 各头都已归一, 平均后每行仍和为 1
            # 学生分布只在 used_mask 内归一化, 和老师 p 的支撑集一致
            log_q = F.log_softmax(index_scores.masked_fill(~used_mask, float("-inf")), dim=-1)
            log_q = log_q.masked_fill(~used_mask, 0.0)  # 这些位置 p=0, 置 0 避免 0·(-inf)=nan
            # clamp_min(1e-9): p=0 时 log(0) = -inf, 0·(-inf) 是 nan; 垫到 1e-9 后这一项是 0·log(1e-9) = 0
            kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)  # [B, T]
            self.last_index_loss = kl.mean()
            self.last_index_scores, self.last_attn_target = index_scores.detach(), p
        return out
