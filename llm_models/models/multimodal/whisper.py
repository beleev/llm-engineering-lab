"""
Whisper — 语音识别版的 Encoder-Decoder Transformer (Radford et al., OpenAI, 2022)

是什么: 和 2017 原版 Transformer 同构, 只把 Encoder 的输入从 token 换成 log-mel 声谱图。
解决了什么: 传统 ASR 是声学模型 + 语言模型 + 对齐的多级流水线; Whisper 用一个 seq2seq 模型端到端完成,
           并用 decoder 开头的 task token (语言 / 转写 / 翻译 / 时间戳) 切换任务。
关键数字: mel [B, 80, 3000] (30 s, 每 10 ms 一帧) → 2 层 Conv1d (第二层 stride=2) → 1500 帧 (50 Hz) → Transformer。
         Encoder 用固定 sin 位置编码, Decoder 用可学习位置 embedding, lm_head 与 token embedding 共享权重。
与官方实现的差异: 结构一致, 默认参数即 base (6 + 6 层, d=512, 8 头)。
本实现没做的:
    - KV cache 和 generate(): 解码时每步整段重算。
    - padding mask: decoder 没有 attention_mask, 变长的 batch 要自己保证等长。
读代码时盯住: encoder_hidden [B, T_mel/2, D] —— 音频进入 decoder 的唯一通道 (cross-attn 的 K/V)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock, PreLNCrossBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.core.position_encoding import SinPositionalEncoding
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask


class WhisperAudioEncoder(nn.Module):
    """mel [B, n_mels, T_mel] -> Conv stem (时间维 ÷2) -> + sin PE -> N × PreLNBlock (双向) -> LN -> [B, T_mel/2, D]"""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        max_source_len: int = 1500,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Conv stem 就是语音的 "tokenizer": 把频率维当通道, 沿时间卷积
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = SinPositionalEncoding(d_model, max_len=max_source_len)
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, 4 * d_model),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, n_mels, T_mel] -> encoder_hidden [B, T_enc, D], T_enc = T_mel/2 (奇数时向上取整)。"""
        x = F.gelu(self.conv1(mel))                                    # [B, D, T_mel]
        x = F.gelu(self.conv2(x))                                      # [B, D, T_mel/2], stride=2 把帧数减半
        # Conv 输出通道在前, Transformer 要时间在前: [B, D, T_enc] → [B, T_enc, D], 再加 sin PE
        x = self.pos_encoding(x.transpose(1, 2))                       # [B, T_enc, D]
        for layer in self.layers:
            x = layer(x)                                               # 无 mask: 整段音频互相可见
        return self.ln_f(x)                                            # [B, T_enc, D]


class WhisperTextDecoder(nn.Module):
    """
    标准 Transformer decoder: 因果 self-attn + cross-attn(encoder_hidden) + GELU-FFN。

    forward 返回 Tensor。不接受 attention_mask, 不支持 KV cache。
    embedding 不乘 √D, 位置是可学习 embedding (与官方一致)。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_target_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_target_len = max_target_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_target_len, d_model)
        self.layers = nn.ModuleList(
            [
                PreLNCrossBlock(
                    d_model=d_model,
                    self_attn=MultiHeadAttention(d_model, n_heads),
                    cross_attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, 4 * d_model),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight              # weight tying

        # 下三角 mask 只建一次, [1, max_target_len, max_target_len]
        self.register_buffer("causal_mask", build_causal_mask(max_target_len, torch.device("cpu")), persistent=False)

    def forward(self, input_ids: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] (含 task prompt), encoder_hidden [B, T_enc, D] -> logits [B, T, V]"""
        T = input_ids.size(1)
        if T > self.max_target_len:
            raise ValueError(f"target 长度 {T} 超过 max_target_len={self.max_target_len}")

        pos = torch.arange(T, device=input_ids.device)                 # [T], 广播到 batch
        x = self.token_embedding(input_ids) + self.position_embedding(pos)   # [B, T, D]
        self_mask = self.causal_mask[:, :T, :T]                        # [1, T, T]

        for layer in self.layers:
            # cross-attn 不设 mask: 每个文本位置都能看整段音频
            x = layer(x, context=encoder_hidden, self_mask=self_mask, context_mask=None)
        return self.lm_head(self.ln_f(x))                              # [B, T, V]


class Whisper(nn.Module):
    """
    mel -> WhisperAudioEncoder -> encoder_hidden;  tokens -> WhisperTextDecoder(cross-attn) -> logits

    forward 返回 Tensor [B, T, V]。不接受 attention_mask, 不支持 KV cache。
    推理时可以分开调: self.encoder(mel) 只跑一次, self.decoder(ids, encoder_hidden) 每步调一次。
    """

    def __init__(
        self,
        vocab_size: int = 51865,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        max_source_len: int = 1500,
        max_target_len: int = 448,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = WhisperAudioEncoder(
            n_mels=n_mels, d_model=d_model, n_heads=n_heads,
            num_layers=encoder_layers, max_source_len=max_source_len, dropout=dropout,
        )
        self.decoder = WhisperTextDecoder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            num_layers=decoder_layers, max_target_len=max_target_len, dropout=dropout,
        )
        # 换成 PyTorch 默认的 N(0,1) embedding + weight tying 会怎样: 初始 CE ≈ 80
        # (run_models 的 mini 配置 D=128, V=500 下); N(0, 0.02²) 后 ≈ ln V。
        # Conv stem 保持 PyTorch 默认初始化: init_weights 只动 Linear / Embedding
        init_weights(self)

    def forward(self, mel: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """
        mel [B, n_mels, T_mel], decoder_input_ids [B, T] -> logits [B, T, V], 返回 Tensor。
        T_mel/2 不能超过 max_source_len (sin PE 表的长度), T 不能超过 max_target_len。
        """
        return self.decoder(decoder_input_ids, encoder_hidden=self.encoder(mel))
