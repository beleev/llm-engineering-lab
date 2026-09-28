"""
GPT-3 (Brown et al., 2020) — decoder-only Transformer, "一切任务 = 文本续写"

是什么: N × PreLNBlock(MHA + GELU-FFN + LayerNorm), 目标只有一个: 预测下一个 token。
解决什么: 相比原始 Transformer / GPT-1 的 Post-LN, Pre-LN `x + f(LN(x))` 让深层训练稳定、不再强依赖 warmup;
          175B 参数首次系统验证 Scaling Law 与 in-context learning。
关键数字: d_ff = 4·d_model;  初始 CE ≈ ln V (N(0,0.02²) 初始化)。
          换成 PyTorch 默认的 N(0,1) embedding + weight tying + ·√D, 初始 CE ≈ D
          (run_models 的 mini 配置 D=256, V=1000 下 ~250)。成因见 models/__init__.py 的「本库约定」。
三个实现细节: causal mask 用 register_buffer 缓存;  lm_head 与 embedding 共享权重 (Press & Wolf 2017);
             generate() 来自 GenerationMixin, 带 KV cache (每步只算 1 个新 token)。
与论文的差异:
    - 原版是可学习的绝对位置 embedding。这里用 Sinusoidal: 零参数, 便于和 RoPE 对照。
    - embedding 乘 √D 是本库约定, 原版不乘。
    - Pre-LN 自 GPT-2 起就有, 不是 GPT-3 引入的。
    - use_rope=True 是本库加的开关。
读代码时盯住: forward 里的 `past` —— 它平移 Sin-PE 的 offset (或 RoPE 的 position_ids) 和 mask 的行。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.core.position_encoding import (
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


def _make_gpt_block(d_model: int, n_heads: int, d_ff: int, dropout: float) -> PreLNBlock:
    """GPT-3 Block = PreLNBlock(MHA + GELU-FFN + LayerNorm)。

    MHA 用 GQA(num_kv_heads = num_heads, bias=True) 实现 —— 数学上就是 MHA, 且自带 KV cache;
    逐头循环的教学版见 layers/core/attention.py::MultiHeadAttention。
    """
    return PreLNBlock(
        d_model=d_model,
        attn=GroupedQueryAttention(d_model, n_heads, bias=True),
        ffn=GeLUFeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


GPTBlock = PreLNBlock   # 别名: GPTBlock 就是 PreLNBlock


class GPT3(GenerationMixin, nn.Module):
    """
    idx -> Embed·sqrt(D) -> (Sin-PE 加到 x | RoPE 注入每层 Q/K)
        -> N × PreLNBlock(MHA + GELU-FFN + LN) -> LayerNorm -> lm_head (tied)

    末尾 ln_f 是 Pre-LN 必需的 "出口规范化": 残差主路从不过 norm, 范数随层数累积。
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 12,
        max_len: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_rope = use_rope
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_rope:
            self.pos_encoder = RotaryPositionalEncoding(d_model // n_heads, max_len)  # 作用于 head 内 Q/K
        else:
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)                # 加在 embedding 上

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [_make_gpt_block(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying: 省一个 V×D 矩阵, 输入/输出 embedding 共用一套梯度
        self.lm_head.weight = self.token_embedding.weight

        # 下三角 mask 只建一次, [1, max_len, max_len]。persistent=False: 不进 state_dict
        self.register_buffer(
            "causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False
        )

        init_weights(self)   # weight tying 之后; 初始 CE ≈ ln V

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """取左上角 [1, seq_len, seq_len] 的下三角 mask; 超过缓存大小就现建一张。"""
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                                  # [B, T, V]
        """
        idx [B, T] -> logits [B, T, V], 返回 Tensor。
        attention_mask: use_rope=True 时照常屏蔽 pad, 左 pad 不改真 token 的输出。
            默认的 Sin-PE 是绝对位置: 左 pad 会把真 token 挤到别的位置, 输出就变了。
            所以 Sin-PE 下只收全 1 的 mask, 有 0 就抛 NotImplementedError。批量生成请用等长 prompt。
        cache: 给了就走 KV cache, idx 只含新 token; 每层的 K/V 存在 cache.layers[i]。
        """
        _, T = idx.size()
        past = cache.pos if cache is not None else 0                     # 已缓存的 token 数
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过最大上下文窗口 {self.max_len}")
        if not self.use_rope and attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError(
                "GPT3 默认用 Sin-PE (绝对位置), pad 会挪动真 token 的位置, 算出的 logits 是错的。"
                "请用等长 prompt, 或构造时传 use_rope=True"
            )

        # ·√D 是本库约定: embedding 每维 std 0.02 → 0.02·√D, 与幅度 ~1 的 Sin-PE 同量级
        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]

        # 两种位置编码二选一: RoPE 不动主干, 交给每层去旋转 Q/K; Sin-PE 直接加到 x 上
        rope = self.pos_encoder if self.use_rope else None
        if not self.use_rope:
            x = self.pos_encoder(x, offset=past)                         # 新 token 的位置从 past 起算

        position_ids = torch.arange(past, past + T, device=idx.device)   # [T] 新 token 的绝对位置
        # 行 past: 是新 token (query), 列 :past+T 是全部历史 (key)
        causal = self._causal_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)   # [B 或 1, T, past+T]
        for i, layer in enumerate(self.layers):
            x = layer(
                x, mask=mask, rope=rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T                                               # 下一次调用从这里接着数

        return self.lm_head(self.ln_f(x))                                # [B, T, V]
