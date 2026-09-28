"""
utils — 掩码、初始化、生成三组通用工具

本包直接导出的只有掩码 (masks.py):
- get_pad_mask: padding 掩码, 屏蔽 pad token
- get_subsequent_mask / build_causal_mask: 因果掩码, 自回归模型不看未来
- build_sliding_window_mask: 带状因果掩码 (SWA, Mistral / Gemma / GPT-OSS)
- combine_causal_and_padding_mask / combine_masks: 多种掩码的组合

另外两个文件按完整路径 import (顶层 llm_models 也导出了这些名字):
- init.py:       init_weights, 把 Linear / Embedding 初始化成 N(0, 0.02²)
- generation.py: KVCache / GenerationMixin / benchmark_kv_cache, 自回归生成 + KV cache
"""

from llm_models.utils.masks import (
    get_pad_mask,
    get_subsequent_mask,
    build_causal_mask,
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
    combine_masks,
)

# __all__ 显式声明 from llm_models.utils import * 时导出的符号
__all__ = [
    "get_pad_mask",
    "get_subsequent_mask",
    "build_causal_mask",
    "build_sliding_window_mask",
    "combine_causal_and_padding_mask",
    "combine_masks",
]
