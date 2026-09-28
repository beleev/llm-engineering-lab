"""
multimodal — 跨模态构建块

  PatchEmbed2D / PatchEmbed3D: 图像 (含 mel 声谱图) / 视频 patch 化
  PatchTransformerEncoder:    标准 ViT 风格编码器
  PerceiverResampler:         把任意多个 token 重采样成固定的 K 个 (Flamingo 系)
  PerceiverResamplerBlock:    Resampler 的一层 (cross-attn + FFN)
  ModalityProjector:          把编码器维度投影到 LLM 维度
"""

from llm_models.layers.multimodal.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)

__all__ = [
    "PatchEmbed2D",
    "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock",
    "PerceiverResampler",
    "ModalityProjector",
]
