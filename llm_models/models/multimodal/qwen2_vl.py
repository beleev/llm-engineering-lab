"""
Qwen2-VL — 视觉 token 当作前缀拼进 LLM 的视觉语言模型 (Wang et al., 阿里, 2024)

是什么: 图像 → ViT → (Resampler 压缩) → Projector → 得到若干 "视觉 token", 拼在文本 token 前面, 送进同一个 decoder-only LLM。
解决了什么: CLIP 只能打分不能生成; Flamingo 式 cross-attention 要改 LLM 结构。prefix 路线不动 LLM, 天然支持多图交错和 KV cache。
关键公式: 序列 = [v_1..v_N ; t_1..t_T], 因果 mask ⇒ 文本看得到全部视觉 token, 反之不行; loss 只算文本位置 (视觉位置 label = -100)。
M-RoPE: 把 head 维切成 (时间, 行, 列) 三段各自旋转。视觉 patch 的位置是 (0, 行, 列); 文本三轴取同一个值 ⇒ 对文本仍是 1-D 相对位置编码。
本实现的简化: 固定分辨率 (原版是动态分辨率); 启用 Resampler 后视觉 token 不再对应网格, 此时 M-RoPE 无意义 (演示 M-RoPE 请设 vision_num_latents=0)。
    - 视觉压缩用 Perceiver Resampler (Flamingo 式); 原版是把相邻 2×2 个 patch 合并后过 MLP。
    - decoder 没接 KV cache 和 generate(): prefix 路线本身支持 KV cache, 本实现没做。
    - use_mrope 默认 False, 默认配置下位置编码是普通的 1-D RoPE。
    - lm_head 与 embedding 共享权重、embedding 乘 √D 是本库约定 (见 models/__init__.py)。
读代码时盯住: combined_embeds 的拼接顺序, 以及 labels / attention_mask / position_ids 如何与它逐位对齐。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import (
    MultimodalRotaryEmbedding,
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.layers.multimodal import (
    ModalityProjector,
    PatchEmbed2D,
    PatchTransformerEncoder,
    PerceiverResampler,
)
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


def build_mrope_position_ids(batch_size: int, grid_hw: Tuple[int, int], text_len: int) -> torch.Tensor:
    """
    [一张图的 H×W 个 patch ; text_len 个文本 token] 的 M-RoPE 位置, 返回 [3, B, H·W + text_len] (时间, 行, 列)。

    视觉 patch: (0, 行, 列)。文本: 三轴相同, 从 max(H, W) 起步 (Qwen2-VL: 接在视觉段最大位置之后)。
    返回的张量在 CPU 上, 用之前自己 .to(device)。
    """
    H, W = grid_hw
    rows = torch.arange(H).repeat_interleave(W)                        # [H·W]  0,0,..,1,1,..
    cols = torch.arange(W).repeat(H)                                   # [H·W]  0,1,..,0,1,..
    vision = torch.stack([torch.zeros_like(rows), rows, cols])         # [3, H·W]
    text = (max(H, W) + torch.arange(text_len)).expand(3, -1)          # [T] → [3, T], 三轴同值
    # [3, H·W + T] → 插入 batch 维 [3, 1, H·W + T] → 复制到 [3, B, H·W + T] (每个样本位置相同)
    return torch.cat([vision, text], dim=1).unsqueeze(1).expand(-1, batch_size, -1)


class Qwen2VLDecoder(nn.Module):
    """
    LLaMA 式 decoder-only LLM (GQA + SwiGLU + RMSNorm + RoPE), 既接受 input_ids 也接受外部拼好的 inputs_embeds。

    position_ids: None / [T] / [B, T] → 1-D RoPE;  [3, B, T] → M-RoPE (需 use_mrope=True)。
    use_rope=False 时退回 Sinusoidal, 仅供对比。

    forward 默认返回 Tensor, return_hidden=True 时返回 tuple (logits, hidden); 接受 attention_mask。
    没有 cache 参数, 也没接 GenerationMixin: 不支持 KV cache 和 generate()。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_len: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_mrope: bool = False,
        mrope_sections: Optional[tuple] = None,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        if use_mrope and not use_rope:
            raise ValueError("use_mrope=True 需要 use_rope=True")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_rope = use_rope
        self.use_mrope = use_mrope

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        d_head = d_model // n_heads
        if use_mrope:
            self.pos_encoder = MultimodalRotaryEmbedding(d_head=d_head, section_dims=mrope_sections, max_len=max_len)
        elif use_rope:
            self.pos_encoder = RotaryPositionalEncoding(d_head, max_len)
        else:
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)

        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)      # SwiGLU 有 3 个矩阵, 乘 2/3 使参数量与 4·D 的两矩阵 FFN 持平

        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=GroupedQueryAttention(d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight              # weight tying

        self.register_buffer("causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False)

        # 换成 PyTorch 默认的 N(0,1) embedding + tying + ·√D 会怎样: 初始 CE ≈ D
        # (run_models 的 mini 配置 D=128, V=500 下 ≈ 130); N(0, 0.02²) 后 ≈ ln V
        init_weights(self)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] -> [B, T, D]。外层模型用它拿文本 embedding, 再和视觉 token 拼接。"""
        # ·√D 是本库约定。视觉 token 不乘: 它们的尺度由 ModalityProjector 末尾的 LayerNorm 定
        return self.token_embedding(input_ids) * math.sqrt(self.d_model)   # [B, T, D]

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        return self.causal_mask[:, :seq_len, :seq_len]                 # [1, T, T]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        input_ids [B, T] 或 inputs_embeds [B, T, D] (二选一) -> logits [B, T, V] (可选再返回 hidden [B, T, D])

        attention_mask: [B, T], 1=有效 0=pad, 长度要和送进来的序列一致 (含视觉前缀)。
        返回类型: 默认 Tensor; return_hidden=True 时是 tuple (logits, hidden)。
            LLaMA 的同名参数是 "用 hidden 代替 logits", 语义不同。
        传 inputs_embeds 时不再乘 √D, 调用方要自己先用 embed_tokens 处理文本部分。
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 与 inputs_embeds 必须且只能传入一个")

        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")

        rope = self.pos_encoder if self.use_rope else None             # RoPE 在每层 attention 内旋转 Q/K
        if rope is None:
            x = self.pos_encoder(x)                                    # Sinusoidal 一次性加到输入上

        if self.use_mrope and (position_ids is None or position_ids.dim() != 3 or position_ids.size(0) != 3):
            raise ValueError("use_mrope=True 时必须传入 [3, B, T] 的 position_ids")

        # 因果 ∧ padding。视觉前缀同样走因果 mask: 文本能看到全部视觉 token。结果 [B 或 1, T, T]
        attn_mask = combine_causal_and_padding_mask(self._causal_mask(seq_len), attention_mask)

        for layer in self.layers:
            x = layer(x, mask=attn_mask, rope=rope, position_ids=position_ids)

        hidden = self.ln_f(x)                                          # [B, T, D]
        logits = self.lm_head(hidden)                                  # [B, T, V]
        return (logits, hidden) if return_hidden else logits


class Qwen2VLModel(nn.Module):
    """
    images -> ViT -> (PerceiverResampler: N 个 patch → num_latents 个 token) -> Projector ─┐
    input_ids -> token embedding ──────────────────────────────────────────────────────────┴ concat -> Qwen2VLDecoder

    vision_num_latents <= 0: 关闭 Resampler, 视觉 token 数 = patch 数。
    use_modality_embedding:  给视觉段 / 文本段各加一个可学习向量 (类似 BERT 的 segment embedding)。

    forward 返回 Tensor。padding mask 的参数名是 text_attention_mask (只管文本段)。
    不支持 KV cache 和 generate()。
    """

    MODALITY_VISION = 0
    MODALITY_TEXT = 1

    def __init__(
        self,
        vocab_size: int,
        text_d_model: int = 1024,
        text_n_heads: int = 16,
        text_num_kv_heads: Optional[int] = None,
        text_num_layers: int = 24,
        max_len: int = 2048,
        vision_image_size: int = 224,
        vision_patch_size: int = 14,
        vision_d_model: int = 1024,
        vision_n_heads: int = 16,
        vision_num_layers: int = 24,
        vision_num_latents: int = 64,
        vision_num_latent_layers: int = 2,
        projector_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_mrope: bool = False,
        use_modality_embedding: bool = True,
    ):
        super().__init__()

        self.vision_encoder = PatchTransformerEncoder(
            patch_embed=PatchEmbed2D(
                input_size=vision_image_size, patch_size=vision_patch_size,
                in_channels=3, embed_dim=vision_d_model,
            ),
            d_model=vision_d_model, n_heads=vision_n_heads, num_layers=vision_num_layers, dropout=dropout,
        )

        self.vision_resampler: Optional[PerceiverResampler] = None
        if vision_num_latents and vision_num_latents > 0:
            self.vision_resampler = PerceiverResampler(
                num_latents=vision_num_latents, d_model=vision_d_model, n_heads=vision_n_heads,
                num_layers=vision_num_latent_layers, dropout=dropout,
            )

        self.vision_projector = ModalityProjector(
            in_dim=vision_d_model, out_dim=text_d_model, hidden_dim=projector_hidden_dim,
        )

        self.text_decoder = Qwen2VLDecoder(
            vocab_size=vocab_size, d_model=text_d_model, n_heads=text_n_heads,
            num_kv_heads=text_num_kv_heads, num_layers=text_num_layers, max_len=max_len,
            dropout=dropout, use_rope=use_rope, use_mrope=use_mrope,
        )

        self.use_modality_embedding = use_modality_embedding
        self.modality_embedding = nn.Embedding(2, text_d_model) if use_modality_embedding else None

        init_weights(self)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, 3, H, W] -> 视觉 token [B, N_v, D_text]"""
        tokens = self.vision_encoder(images)                           # [B, N_patch, D_vis]
        if self.vision_resampler is not None:
            tokens = self.vision_resampler(tokens)                     # [B, num_latents, D_vis]
        return self.vision_projector(tokens)                           # [B, N_v, D_text]

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids [B, T]; images [B, 3, H, W] 或 None (退化为纯文本 LLM); text_attention_mask [B, T];
        position_ids: 1-D RoPE 用 [T_total] / [B, T_total], M-RoPE 用 [3, B, T_total]
        -> logits [B, N_v + T, V]  (前 N_v 个位置对应视觉 token, 训练时 label = -100)

        返回 Tensor。N_v = vision_num_latents; 关掉 Resampler 时 N_v = patch 数。
        labels 和 position_ids 的长度都要按 N_v + T 准备, 否则和 logits 对不上。
        """
        B, T = input_ids.shape
        embeds = self.text_decoder.embed_tokens(input_ids)             # [B, T, D]
        attention_mask = text_attention_mask
        n_vis = 0                                                      # 没有图像时视觉 token 数为 0

        if images is not None:
            vision_embeds = self.encode_images(images)                 # [B, N_v, D]
            n_vis = vision_embeds.size(1)
            embeds = torch.cat([vision_embeds, embeds], dim=1)         # [B, N_v + T, D] 视觉在前
            if text_attention_mask is not None:                        # 视觉 token 没有 padding
                # 前面补 N_v 个 1: [B, T] → [B, N_v + T], 顺序与 embeds 的拼接一致
                attention_mask = torch.cat([text_attention_mask.new_ones(B, n_vis), text_attention_mask], dim=1)

        if self.modality_embedding is not None:
            modality_ids = torch.cat([
                torch.full((n_vis,), self.MODALITY_VISION, device=embeds.device),
                torch.full((T,), self.MODALITY_TEXT, device=embeds.device),
            ])                                                         # [N_v + T]
            embeds = embeds + self.modality_embedding(modality_ids)    # [N_v + T, D] 广播到 batch

        return self.text_decoder(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids)
