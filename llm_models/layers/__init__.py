"""
基础层 (layers) 模块 — Transformer 全栈零件库

按功能分类成四个子包:

  core/        通用 Transformer 基础组件 (任何 Transformer 都需要)
                - attention:         SDPA → MHA → GQA → MLA → DSA (含 LightningIndexer)
                - position_encoding: Sinusoidal / RoPE (可选 NTK / YaRN 缩放) / M-RoPE
                - feedforward:       ReLU / GELU / SwiGLU
                - normalization:     RMSNorm
                - blocks:            PreLN Transformer Block 组装器

  sparse/      稀疏 / 非注意力序列分支
                - moe:              MixtralMoE (经典稀疏 MoE)
                - ssm:              SelectiveSSM (Mamba)
                - linear_attention: GatedDeltaNet (Qwen3-Next)

  diffusion/   扩散 / 生成模型专用
                - adaln: adaLN-Zero (DiT 条件注入): AdaLNZeroBlock / FinalLayer / TimestepEmbedding / modulate
                - vq:    VectorQuantizer (VQ-VAE) / MultiScaleVQ (VAR)

  multimodal/  跨模态构建块
                - PatchEmbed2D / 3D, PatchTransformerEncoder,
                  PerceiverResampler (+ PerceiverResamplerBlock), ModalityProjector

本包不转发、要按完整路径 import 的 3 个名字 (顶层 llm_models 也导出了它们):
    core.attention.LightningIndexer, core.position_encoding.scaled_inv_freq, diffusion.vq.MultiScaleVQ
    (GatedDeltaNet 本包有转发)。

通过组合这些零件可拼出 BERT / GPT-3 / LLaMA / Mistral / Qwen3-Next / LLaDA / Mixtral / GPT-OSS /
Qwen2-VL / DeepSeek-V3 / Mamba / CLIP / Whisper / DiT / MM-DiT / Video DiT / VAR 等模型。
"""

from llm_models.layers.core import (
    ScaledDotProductAttention,
    SingleHeadSelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
    SinPositionalEncoding,
    RotaryPositionalEncoding,
    MultimodalRotaryEmbedding,
    apply_rotary_pos_emb,
    sinusoidal_embedding,
    FeedForward,
    GeLUFeedForward,
    SwiGLUFeedForward,
    RMSNorm,
    PreLNBlock,
    PreLNCrossBlock,
)

from llm_models.layers.sparse import MixtralMoE, SelectiveSSM, GatedDeltaNet

from llm_models.layers.diffusion import (
    AdaLNZeroBlock,
    FinalLayer,
    TimestepEmbedding,
    modulate,
    VectorQuantizer,
)

from llm_models.layers.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)

__all__ = [
    # core/attention
    "ScaledDotProductAttention",
    "SingleHeadSelfAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "MultiHeadLatentSparseAttention",
    # core/position
    "SinPositionalEncoding",
    "RotaryPositionalEncoding",
    "MultimodalRotaryEmbedding",
    "apply_rotary_pos_emb",
    "sinusoidal_embedding",
    # core/feedforward
    "FeedForward",
    "GeLUFeedForward",
    "SwiGLUFeedForward",
    # core/normalization
    "RMSNorm",
    # core/blocks
    "PreLNBlock",
    "PreLNCrossBlock",
    # sparse
    "MixtralMoE",
    "SelectiveSSM",
    "GatedDeltaNet",
    # diffusion
    "AdaLNZeroBlock",
    "FinalLayer",
    "TimestepEmbedding",
    "modulate",
    "VectorQuantizer",
    # multimodal
    "PatchEmbed2D",
    "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock",
    "PerceiverResampler",
    "ModalityProjector",
]
