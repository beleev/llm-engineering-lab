"""
core — 通用 Transformer 基础组件

任何 Transformer 系模型都会复用的零件:
  attention:         SDPA → MHA → GQA → MLA → DSA 的完整演进链
  position_encoding: Sinusoidal / RoPE / M-RoPE 三代位置编码, 外加通用的 sinusoidal_embedding
  feedforward:       ReLU / GELU / SwiGLU 三代 FFN
  normalization:     RMSNorm (LLaMA 系标准)
  blocks:            PreLN 通用 Transformer Block 组装器

本包没有转发的两个名字, 按完整路径 import:
  attention.LightningIndexer (DSA 的打分器), position_encoding.scaled_inv_freq (NTK / YaRN 频率缩放)
"""

from llm_models.layers.core.attention import (
    ScaledDotProductAttention,
    SingleHeadSelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
)
from llm_models.layers.core.position_encoding import (
    SinPositionalEncoding,
    RotaryPositionalEncoding,
    MultimodalRotaryEmbedding,
    apply_rotary_pos_emb,
    sinusoidal_embedding,
)
from llm_models.layers.core.feedforward import (
    FeedForward,
    GeLUFeedForward,
    SwiGLUFeedForward,
)
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.blocks import PreLNBlock, PreLNCrossBlock

__all__ = [
    "ScaledDotProductAttention",
    "SingleHeadSelfAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "MultiHeadLatentSparseAttention",
    "SinPositionalEncoding",
    "RotaryPositionalEncoding",
    "MultimodalRotaryEmbedding",
    "apply_rotary_pos_emb",
    "sinusoidal_embedding",
    "FeedForward",
    "GeLUFeedForward",
    "SwiGLUFeedForward",
    "RMSNorm",
    "PreLNBlock",
    "PreLNCrossBlock",
]
