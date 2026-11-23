"""
Transfusion — 一个 transformer 同时做文本自回归和图像扩散 (Zhou et al., Meta, 2024, arXiv 2408.11039)

是什么: 文本是离散 token, 用 next-token CE 训练; 图像是连续 patch 向量, 用扩散损失训练。
        两种位置排在同一条序列里, 共用同一套 Transformer 层, 只有输入层和输出头按位置类型分开。
解决什么: Qwen2-VL 这类前缀路线能看图但只输出文本, 生成图像要另接一个扩散模型;
          Chameleon 把图像量化成离散 token, 统一成 next-token 预测, 但量化会丢细节, 一张图还要逐 token 生成上千步。
          Transfusion 让图像保持连续向量, 训练时算扩散损失, 生成时在同一个模型里去噪。
关键公式:
    序列: [BOS, 文本..., BOI, p_1 .. p_N, EOI, 文本...]       p_i 是加噪后的图像 patch
    掩码: M[i, j] = (j ≤ i) OR (i 和 j 属于同一张图)           整条序列因果, 同一张图内部双向
    损失: L = L_LM + λ·L_img                                   论文 λ = 5
          L_LM  = 文本位置的 next-token CE
          L_img = 图像位置的 MSE(v_θ(x_t, t), ε − x_0)
    加噪: x_t = (1 − t)·x_0 + t·ε, t 的嵌入加到这张图的每个 patch 输入上
与论文的差异:
    简化: 图像损失用 flow matching 回归速度 v = ε − x_0; 论文是 DDPM 的 ε 预测 + cosine 日程。
    简化: 直接对像素切 patch, 线性层进出; 论文先用 VAE 把图压成 latent, 再按 2×2 latent patch 经线性层或 U-Net 下/上采样块进出。
    简化: 一条序列里的图共用一个 t, 位置编码是整条序列的 1D RoPE (论文用 Llama 式块, 含 RoPE)。
    简化: 生成时由调用方放好 BOI, 不用 CFG; 论文是模型采到 BOI 后切到扩散模式, 去噪完再接 EOI 回到文本模式。
    本库约定: 文本 embedding 乘 √D、lm_head 与 embedding 共享权重、N(0, 0.02²) 初始化, 见 models/__init__.py。
读代码时盯住: transfusion_mask 里 "同一张图" 那一项, 以及 forward 里文本位置和图像位置怎样各取各的输入、各出各的头。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.layers.diffusion.adaln import TimestepEmbedding
from llm_models.models.language_models.llama import _make_llama_block
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask


def transfusion_mask(image_id: torch.Tensor) -> torch.Tensor:
    """
    image_id [B, S]: 文本位置为 -1, 图像位置为这张图的编号 (0, 1, ...)。
    返回 [B, S, S] bool, True = 可见: 因果 OR 同属一张图。
    """
    S = image_id.size(1)
    causal = build_causal_mask(S, image_id.device)                       # [1, S, S]
    same_image = (image_id[:, :, None] == image_id[:, None, :]) & (image_id[:, :, None] >= 0)  # [B, S, S]
    return causal | same_image                                           # ★ 图像段内的 "未来" patch 也可见


class TransfusionModel(nn.Module):
    """
    文本 token 与图像 patch 共用 N 个 LLaMA 块 (GQA + SwiGLU + RMSNorm + RoPE)。

    输入按位置二选一:
        文本位置: token_embedding(tokens) · √D
        图像位置: patch_in(patches) + time_embed(t)
    输出也按位置二选一: 文本位置看 logits (lm_head, 与 embedding 共享), 图像位置看 velocity (patch_out)。
    forward 返回 dict {logits [B, S, V], velocity [B, S, P]}, 两个头都对所有位置算, 由 loss 按位置挑。
    """

    def __init__(
        self,
        vocab_size: int,
        patch_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 3,
        max_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_in = nn.Linear(patch_dim, d_model)
        self.time_embed = TimestepEmbedding(d_model)                     # t 用 [0, 1000) 量纲, 见 training/diffusion.py
        self.rope = RotaryPositionalEncoding(d_model // n_heads, max_len)
        d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64
        self.layers = nn.ModuleList(
            [_make_llama_block(d_model, n_heads, num_kv_heads, d_ff, 0.0) for _ in range(num_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight                # weight tying
        self.patch_out = nn.Linear(d_model, patch_dim)
        init_weights(self)                                               # 初始 logits ≈ 0 → 文本 CE ≈ ln V

    def forward(
        self,
        tokens: torch.Tensor,          # [B, S] 图像位置的值不用
        patches: torch.Tensor,         # [B, S, P] 文本位置的值不用
        image_id: torch.Tensor,        # [B, S] -1 = 文本
        t: torch.Tensor,               # [B] 这条序列里图像的噪声时间, [0, 1000) 量纲
        mask: Optional[torch.Tensor] = None,  # 默认 transfusion_mask(image_id); 传因果 mask 可做对照
    ) -> Dict[str, torch.Tensor]:
        S = tokens.size(1)
        if S > self.max_len:
            raise ValueError(f"序列长度 {S} 超过 max_len={self.max_len}")
        is_image = (image_id >= 0).unsqueeze(-1)                         # [B, S, 1]
        text_x = self.token_embedding(tokens) * math.sqrt(self.d_model)  # ·√D 是本库约定
        image_x = self.patch_in(patches) + self.time_embed(t).unsqueeze(1)  # 同一张图的每个 patch 加同一份 t 嵌入
        x = torch.where(is_image, image_x, text_x)                       # [B, S, D]
        mask = transfusion_mask(image_id) if mask is None else mask
        for layer in self.layers:
            x = layer(x, mask=mask, rope=self.rope)
        h = self.ln_f(x)
        return {"logits": self.lm_head(h), "velocity": self.patch_out(h)}


def transfusion_loss(
    out: Dict[str, torch.Tensor],
    tokens: torch.Tensor,          # [B, S]
    target_v: torch.Tensor,        # [B, S, P] 图像位置是 ε − x_0, 文本位置的值不用
    image_id: torch.Tensor,        # [B, S]
    lm_mask: torch.Tensor,         # [B, S-1] bool: 位置 i 是否算 next-token loss (目标是 tokens[:, i+1])
    lam: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """L = L_LM + λ·L_img, 各自只在自己的位置上取平均。"""
    # cross_entropy 要求类别维在 dim=1: [B, S-1, V] → [B, V, S-1]
    ce = F.cross_entropy(out["logits"][:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none")  # [B, S-1]
    lm = (ce * lm_mask).sum() / lm_mask.sum().clamp(min=1)
    img = (image_id >= 0).float()                                        # [B, S]
    mse = ((out["velocity"] - target_v) ** 2).mean(-1)                   # [B, S] 每个 patch 的 MSE
    im = (mse * img).sum() / img.sum().clamp(min=1)
    return {"total_loss": lm + lam * im, "lm_loss": lm, "image_loss": im}
