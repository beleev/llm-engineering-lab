"""
diffusion — 扩散 / 生成模型专用零件

  adaln: adaLN-Zero 模块 (DiT / 扩散 Transformer 条件注入)
  vq:    VectorQuantizer (VQ-VAE / VAR / LlamaGen 的离散化骨架)
         MultiScaleVQ (VAR 的多尺度残差量化)。本包没有转发它,
         用 `from llm_models.layers.diffusion.vq import MultiScaleVQ`
"""

from llm_models.layers.diffusion.adaln import (
    AdaLNZeroBlock,
    FinalLayer,
    TimestepEmbedding,
    modulate,
)
from llm_models.layers.diffusion.vq import VectorQuantizer

__all__ = [
    "AdaLNZeroBlock",
    "FinalLayer",
    "TimestepEmbedding",
    "modulate",
    "VectorQuantizer",
]
