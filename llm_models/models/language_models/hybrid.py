"""
HybridLM — 2025-2026 主流混合架构的零件对照台

是什么: 一个 decoder-only LM, 每层用哪种序列混合器由一个排布字符串决定。七种混合器:
    gdn        Gated DeltaNet, 定长状态, 无位置编码            (Qwen3-Next)
    kda        Kimi Delta Attention, 逐通道遗忘门, 无位置编码   (Kimi Linear / Kimi K3)
    gqa        GQA + RoPE, 全注意力
    gated_gqa  GQA + RoPE + SDPA 输出门                        (Qwen3-Next / Qwen3.5 的全注意力层)
    mla_nope   MLA, 不加位置编码                                (Kimi Linear / Kimi K3 的全注意力层)
    gqa_nope   GQA, 不加位置编码, 看全长                        (Llama 4 iRoPE 里的 NoPE 层)
    gqa_chunk  GQA + RoPE, 注意力限制在自己那一块之内            (Llama 4 iRoPE 里的 RoPE 层)
解决什么: 把几家 2025-2026 的排布放进同一个类里按同一套口径比缓存和外推:
    Kimi Linear / Kimi K3: 3 层 KDA 配 1 层 MLA (3:1), MLA 层一律 NoPE; K3 的 MLA 还加了逐通道输出门。
    Qwen3-Next / Qwen3.5: 3 层 Gated DeltaNet 配 1 层 Gated Attention (3:1); Qwen3.5 是 60 层、
        397B 总参数 / 17B 激活的稀疏 MoE, 原生多模态。
    Llama 4 iRoPE: NoPE 层与 RoPE 层交错, RoPE 层用分块局部注意力, 推理时对注意力做温度缩放。
    Nemotron 3.5 / Jamba: 把线性那一半换成 Mamba, 本库的 Mamba 在 mamba.py 里单列一章, 这里不收。
关键数字: 缓存元素数 = Σ_{全注意力层} T · (每 token 的 K/V) + Σ_{线性层} H·Dh² (后一项与 T 无关)。
    gqa / gated_gqa / gqa_nope / gqa_chunk 每 token 是 2·Hkv·Dh, mla_nope 是 r + rope, gdn / kda 是 0。
本库的做法 (教学约定, 不代表原模型): Pre-LN、weight tying、embedding 乘 √D、N(0, 0.02²) 初始化照旧。
简化: NoPE 的 MLA 层仍保留 rope 段的投影和缓存 (本库的 MLA 把两段写在一个投影里);
      真实实现会把这一段去掉, 每 token 再少 rope 个数。
简化: 没有实现 Llama 4 推理时的注意力温度缩放 (Meta 只说 "inference time temperature scaling",
      具体公式来自实现而不是官方描述)。分块大小、NoPE 层的间隔由构造参数给, 不写死成 Llama 4 的值。
简化: 没有 MoE。Qwen3.5 与 Kimi K3 的稀疏 MoE 见 moe/ 下的几个模型。
读代码时盯住: _MIXERS 里每种混合器的三元组 (构造函数, 要不要 RoPE, 掩码类型) ——
    掩码类型决定这一层拿到的是 [B, T, S] 的注意力掩码、分块掩码, 还是 [B, T] 的 "哪些是真 token"。
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadLatentAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.gated_attention import GatedAttention
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.layers.sparse.linear_attention import GatedDeltaNet, KimiDeltaAttention
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask

# 混合器名 → (要不要 RoPE, 掩码类型): "full" 全长因果, "chunk" 分块局部, "token" 只要 [B, T] 的真 token 标记
_MIXERS = {
    "gdn": (False, "token"),
    "kda": (False, "token"),
    "gqa": (True, "full"),
    "gated_gqa": (True, "full"),
    "mla_nope": (False, "full"),
    "gqa_nope": (False, "full"),
    "gqa_chunk": (True, "chunk"),
}


class HybridLM(GenerationMixin, nn.Module):
    """
    idx → Embed·sqrt(D) → N × PreLNBlock(按排布选混合器 + SwiGLU) → RMSNorm → lm_head (tied)

    forward 返回 Tensor; 接受 attention_mask; 支持 KV cache (generate() 来自 GenerationMixin)。

    Args:
        layout:       空格或逗号分隔的混合器名, 一个名字一层, 例如 "kda kda kda mla_nope"
        chunk_size:   gqa_chunk 层的分块大小
        num_kv_heads: GQA 系列的 K/V head 数, None = n_heads
        latent_dim / qk_rope_head_dim: mla_nope 的 latent 维度 r 与 rope 段宽度
        其余参数与 LLaMA 相同
    """

    def __init__(
        self,
        layout: str = "kda kda kda mla_nope",
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        max_len: int = 512,
        chunk_size: int = 16,
        d_ff: Optional[int] = None,
        latent_dim: int = 32,
        qk_rope_head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        kinds = layout.replace(",", " ").split()
        unknown = set(kinds) - set(_MIXERS)
        if unknown:
            raise ValueError(f"未知的混合器 {sorted(unknown)}, 可选: {sorted(_MIXERS)}")

        self.layout, self.kinds = layout, kinds
        self.d_model, self.max_len, self.chunk_size = d_model, max_len, chunk_size
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.latent_dim, self.qk_rope_head_dim = latent_dim, qk_rope_head_dim

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # RoPE 只给 GQA 系列用, head_dim 按它们算; mla_nope 和线性层都不用位置编码
        self.rope = RotaryPositionalEncoding(self.head_dim, max_len)
        if d_ff is None:
            d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64

        self.layers = nn.ModuleList([
            PreLNBlock(
                d_model=d_model, attn=self._make_mixer(kind), ffn=SwiGLUFeedForward(d_model, d_ff),
                norm_cls=RMSNorm, dropout=dropout,
            )
            for kind in kinds
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight          # weight tying
        self.register_buffer("causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False)
        init_weights(self)
        # GatedDeltaNet / KimiDeltaAttention 的 α 门 bias 要在 init_weights 之后补回
        for layer, kind in zip(self.layers, kinds):
            if kind in ("gdn", "kda"):
                nn.init.constant_(layer.attn.gate_alpha.bias, 2.0)

    def _make_mixer(self, kind: str) -> nn.Module:
        if kind == "gdn":
            return GatedDeltaNet(self.d_model, self.n_heads)
        if kind == "kda":
            return KimiDeltaAttention(self.d_model, self.n_heads)
        if kind == "mla_nope":
            return MultiHeadLatentAttention(
                d_model=self.d_model, num_heads=self.n_heads, kv_lora_rank=self.latent_dim,
                qk_nope_head_dim=self.head_dim, qk_rope_head_dim=self.qk_rope_head_dim,
            )
        gate = kind == "gated_gqa"
        return GatedAttention(self.d_model, self.n_heads, use_gate=gate, num_kv_heads=self.num_kv_heads)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def cache_elements(self, seq_len: int) -> Dict[str, int]:
        """长度 seq_len 时一条序列的缓存元素数算式 (与实测张量大小逐个相符, 见 infer 脚本)。"""
        per_token = {
            "gqa": 2 * self.num_kv_heads * self.head_dim,
            "gated_gqa": 2 * self.num_kv_heads * self.head_dim,
            "gqa_nope": 2 * self.num_kv_heads * self.head_dim,
            "gqa_chunk": 2 * self.num_kv_heads * self.head_dim,
            "mla_nope": self.latent_dim + self.qk_rope_head_dim,
            "gdn": 0,
            "kda": 0,
        }
        state = {"gdn": self.n_heads * self.head_dim ** 2, "kda": self.n_heads * self.head_dim ** 2}
        grows = sum(per_token[k] for k in self.kinds) * seq_len
        fixed = sum(state.get(k, 0) for k in self.kinds)
        return {"grows_with_t": grows, "fixed": fixed, "total": grows + fixed}

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                                  # [B, T, V]
        """idx [B, T] → logits [B, T, V]。

        三类掩码: 全注意力层拿 [B, T, past+T] 的因果加 padding 掩码; 分块层在它之上再要求
        query 和 key 落在同一块里; 线性层只要 [B, T] 的 "本次哪些 token 是真的"。
        """
        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        position_ids = torch.arange(past, past + T, device=idx.device)
        causal = self._causal_mask(past + T)[:, past:]
        full = combine_causal_and_padding_mask(causal, attention_mask)          # [B 或 1, T, past+T]
        token_mask = None if attention_mask is None else attention_mask[:, past:]
        chunk = None
        if "gqa_chunk" in self.kinds:
            k_pos = torch.arange(past + T, device=idx.device)
            same = (position_ids[:, None] // self.chunk_size) == (k_pos[None, :] // self.chunk_size)
            chunk = full & same.unsqueeze(0)                                    # [B 或 1, T, past+T]

        masks = {"full": full, "chunk": chunk, "token": token_mask}
        for i, (layer, kind) in enumerate(zip(self.layers, self.kinds)):
            use_rope, mask_kind = _MIXERS[kind]
            x = layer(
                x, mask=masks[mask_kind], rope=self.rope if use_rope else None,
                position_ids=position_ids, cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T
        return self.lm_head(self.ln_f(x))

    def sink_mass(self) -> List[float]:
        """上一次前向里, 各 GQA 系列层落在第 0 个 token 上的平均注意力质量。"""
        return [l.attn.last_sink_mass for l in self.layers if isinstance(l.attn, GatedAttention)]
