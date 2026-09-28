"""
Transformer — 原始 Encoder-Decoder 架构 ("Attention Is All You Need", Vaswani et al., 2017)

是什么: Encoder 双向读完源句; Decoder 因果地生成目标句, 每层通过 cross-attention 回看整个源句。
解决了什么: RNN seq2seq 必须逐 token 串行, 且长距离依赖要穿过很多步; attention 让任意两个位置一步直连, 训练可并行。
关键公式: Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V;  cross-attn 中 Q 来自 decoder, K/V 来自 encoder 输出。
         x = emb·√d_model + PE  (embedding 用 N(0, 0.02²) 初始化, 乘 √D 后与幅度 ~1 的 sin PE 同量级)
后来分化成两支: 只留 Encoder → BERT; 只留 Decoder → GPT。与论文的差异: 这里用 Pre-LN (更稳, 不依赖 warmup)。
与论文的其它差异:
    - 论文让两个 embedding 和输出层共用一个矩阵; 这里三者各自独立 (源/目标词表可以不同)。
    - use_rope=True 是本库加的开关, 论文只有 Sin-PE。
    - ·√D 与论文一致。
本实现没做的: KV cache 和 generate()。推理时 encode 跑一次, decode 每步整段重算。
读代码时盯住: 三种 mask —— src_mask (源 padding) / tgt_mask (因果 ∧ 目标 padding) / cross-attn 复用 src_mask。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock, PreLNCrossBlock
from llm_models.layers.core.feedforward import FeedForward
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding, SinPositionalEncoding
from llm_models.utils.init import init_weights


def _make_encoder_layer(d_model: int, n_heads: int, d_ff: int, dropout: float) -> PreLNBlock:
    """self-attn + ReLU-FFN (与论文一致的 ReLU / LayerNorm)。"""
    return PreLNBlock(
        d_model=d_model,
        attn=MultiHeadAttention(d_model, n_heads),
        ffn=FeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


def _make_decoder_layer(d_model: int, n_heads: int, d_ff: int, dropout: float) -> PreLNCrossBlock:
    """因果 self-attn + cross-attn (Q=decoder, K/V=encoder 输出) + FFN。"""
    return PreLNCrossBlock(
        d_model=d_model,
        self_attn=MultiHeadAttention(d_model, n_heads),
        cross_attn=MultiHeadAttention(d_model, n_heads),
        ffn=FeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


# 别名: EncoderLayer / DecoderLayer 就是 PreLNBlock / PreLNCrossBlock
EncoderLayer = PreLNBlock
DecoderLayer = PreLNCrossBlock


class Transformer(nn.Module):
    """
    src -> src_emb·√D (+PE) -> N × EncoderLayer -> LN ─────────────┐ memory [B, S, D]
    tgt -> tgt_emb·√D (+PE) -> N × DecoderLayer(cross-attn memory) -> LN -> fc_out -> [B, T, V_tgt]

    use_rope=True 时不加 Sin-PE, 改为在 self-attn 内旋转 Q/K。

    三个入口:
        forward(src, tgt, ...)  训练用, 一次走完 encode + decode。
        encode(src)             推理时只跑一次, 得到 memory。
        decode(tgt, memory)     推理时每生成一个 token 调一次。
    返回值都是 Tensor。padding 不走 attention_mask, 由调用方建好 src_mask / tgt_mask 传进来
    (见 utils/masks.py)。不支持 KV cache。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.d_model = d_model

        if use_rope:
            self.pos_encoder = RotaryPositionalEncoding(d_model // n_heads, max_len)   # 作用在每个 head 上
        else:
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)                 # 直接加到 embedding 上

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)     # 源/目标词表不同, 各用各的
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList(
            [_make_encoder_layer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [_make_decoder_layer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.enc_final_norm = nn.LayerNorm(d_model)
        self.dec_final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # 换成 PyTorch 默认的 N(0,1) embedding 会怎样: 乘 √D 后每维 std ≈ √D, 把幅度 ~1 的 Sin-PE 淹没
        init_weights(self)

    def _embed(self, ids: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        """ids [B, T] -> [B, T, D]。查表, 乘 √D, 再加 Sin-PE (use_rope=True 时不加)。"""
        # ·√D: embedding 每维 std 0.02, 乘完 ≈ 0.02·√D, 与幅度 ~1 的 Sin-PE 同量级
        emb = embedding(ids) * math.sqrt(self.d_model)                 # [B, T, D]
        return emb if self.use_rope else self.pos_encoder(emb)

    @property
    def _rope(self) -> Optional[nn.Module]:
        """传给每层 self-attn 的 RoPE 模块; 用 Sin-PE 时是 None。"""
        return self.pos_encoder if self.use_rope else None

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """src [B, S], src_mask [B, 1, S] (True=有效) -> memory [B, S, D]。推理时只需跑一次。"""
        x = self._embed(src, self.src_embedding)
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask, rope=self._rope)               # 双向 self-attn
        return self.enc_final_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt [B, T], memory [B, S, D] -> logits [B, T, V_tgt]
        tgt_mask [B, T, T] 管 self-attn (因果 ∧ 目标 padding); src_mask [B, 1, S] 管 cross-attn。
        """
        x = self._embed(tgt, self.tgt_embedding)
        for layer in self.decoder_layers:
            # cross-attn 不用 RoPE: Q 与 K 来自两条不同序列, 相对位置没有意义。
            # PreLNCrossBlock 内部只对 self-attn 传 rope
            x = layer(x, context=memory, self_mask=tgt_mask, context_mask=src_mask, rope=self._rope)
        return self.fc_out(self.dec_final_norm(x))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        src [B, S], tgt [B, T] (teacher forcing 输入);
        src_mask [B, 1, S] 源 padding; tgt_mask [B, T, T] 因果 ∧ 目标 padding  -> logits [B, T, V_tgt]

        返回 Tensor。mask 都是 bool, True = 可见; 传 None 表示不屏蔽 (decoder 也就不因果了)。
        不接受 attention_mask, 不支持 KV cache。
        """
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)
