"""
VAR — Visual AutoRegressive, 用 "next-scale prediction" 做图像自回归 (Tian et al., 2024)

前一代 (VQGAN / LlamaGen) 把 token 网格按光栅序拉平做 next-token: 二维结构被硬塞成一维,
且生成 N 个 token 要 N 步。VAR 把 "一步" 改成 "一整张更细的 token map":

    p(r_1, ..., r_K) = Π_k p(r_k | r_1, ..., r_{k-1})      r_k: 第 k 级 s_k×s_k 的 token map

  - tokenizer: 多尺度残差 VQ (layers/diffusion/vq.py::MultiScaleVQ), 粗尺度管低频, 细尺度补残差
  - transformer: 第 k 级的输入 = 前 k-1 级累计重建下采样到 s_k (第 1 级用可学习的 start token),
                 同一级内 s_k² 个 token **并行** 预测; 注意力是 block-causal (只看同级与更粗级)
  - 采样: K 步 (本例 3 步出 1+4+16=21 个 token), 词表里没有 BOS, 采到的一定是合法码字

简化:
    - 原版是类别条件生成 (类别 embedding 当 start token); 这里无条件, start 是一个可学习向量。
    - 采样不带 KV cache: 每一级都把全部前缀重算一遍。
    - tokenizer 的简化见 layers/diffusion/vq.py::MultiScaleVQ。
读代码时盯住: VARModel.block_causal_mask, 以及 forward 里 "输入来自 <k 级, 标签是第 k 级"。
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.diffusion.vq import MultiScaleVQ
from llm_models.models.generative.vae import ImageVAEDecoder, ImageVAEEncoder
from llm_models.utils.init import init_weights


class ImageTokenizer(nn.Module):
    """
    多尺度 VQ-VAE: 图像 ↔ token 金字塔。conv 主干复用 ImageVAE 的 encoder/decoder,
    瓶颈换成 MultiScaleVQ。必须先训练它 (重建 + vq_loss), 码本才有意义。

    Args:
        image_size:    输入边长; latent 边长 = image_size / 2^levels, 必须等于 scales[-1]
        codebook_size: 码本大小 K (也是 VAR Transformer 的词表大小)
        latent_dim:    码字维度 D
        scales:        token 金字塔各级边长, 由粗到细

    forward 返回 dict (四个键见 forward)。
    """

    def __init__(
        self,
        image_size: int = 16,
        image_channels: int = 3,
        codebook_size: int = 64,
        latent_dim: int = 16,
        base_channels: int = 16,
        levels: int = 2,
        scales: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        if image_size // (2 ** levels) != scales[-1] or image_size % (2 ** levels) != 0:
            raise ValueError(f"image_size/2^levels 必须等于最细尺度 {scales[-1]}")

        self.scales = tuple(scales)
        self.num_tokens = sum(s * s for s in scales)          # L = Σ s²
        self.encoder = ImageVAEEncoder(
            in_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, downsample_levels=levels,
        )
        self.decoder = ImageVAEDecoder(
            out_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, upsample_levels=levels,
        )
        self.quantizer = MultiScaleVQ(codebook_size, latent_dim, scales)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokenizer 训练前向: loss = MSE(recon, x) + vq_loss。

        x [B, 3, H, W] -> 返回 dict, 四个键:
            "recon":      [B, 3, H, W]   量化后再解码的重建
            "indices":    list, 第 k 项 [B, s_k, s_k], 各级 token
            "vq_loss":    标量, 码本 loss + commitment loss
            "perplexity": 标量, 码本使用度 (越接近码本大小越均匀)
        """
        f, _ = self.encoder(x)                                # 只用 mean 头; [B, D, h, h]
        f_hat, info = self.quantizer(f)                       # f_hat [B, D, h, h], 梯度直通回 encoder
        return {"recon": self.decoder(f_hat), **info}

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> List[torch.Tensor]:
        """图像 [B, 3, H, W] → K 个 token map [B, s_k, s_k]。"""
        f, _ = self.encoder(x)                                # [B, D, h, h]
        return self.quantizer(f)[1]["indices"]                # quantizer 返回 (f_hat, info), 取 info

    def decode_from_indices(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """K 个 token map [B, s_k, s_k] → 图像 [B, 3, H, W]。"""
        return self.decoder(self.quantizer.decode(indices))


class VARModel(nn.Module):
    """
    冻结的 ImageTokenizer + block-causal Transformer。

    序列布局 (scales=(1,2,4), L=21):  [ 级1: 1 个 | 级2: 4 个 | 级3: 16 个 ]
    位置 i 的输入来自 "更粗级的累计重建", 输出 logits 预测 "本级该位置的 token"。
    没有 shift-by-one: 错位发生在尺度之间, 不在 token 之间。

    forward(images) 返回 dict {"logits", "labels"}。
    没有 attention_mask (每张图 token 数相同, 没有 padding), 没有 KV cache。
    """

    def __init__(
        self,
        tokenizer: ImageTokenizer,
        d_model: int = 96,
        n_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        for p in tokenizer.parameters():                      # tokenizer 已训练好, 这里只训 Transformer
            p.requires_grad_(False)

        self.scales = tokenizer.scales
        self.codebook_size = tokenizer.quantizer.vq.num_embeddings
        self.num_tokens = tokenizer.num_tokens                # L = Σ s²
        D = tokenizer.quantizer.vq.embedding_dim              # 码字维度, 不是 d_model

        tf = nn.Module()                                      # 可训练部分单独成组, 便于只对它 init
        tf.start = nn.Parameter(torch.zeros(1, 1, d_model))   # 第 1 级的输入 (无条件生成的 "起点")
        tf.in_proj = nn.Linear(D, d_model)                    # 码字空间 → d_model
        tf.pos_embed = nn.Embedding(self.num_tokens, d_model)
        tf.scale_embed = nn.Embedding(len(self.scales), d_model)
        tf.blocks = nn.ModuleList([
            PreLNBlock(d_model, MultiHeadAttention(d_model, n_heads),
                       GeLUFeedForward(d_model, 4 * d_model), dropout=0.0)
            for _ in range(num_layers)
        ])
        tf.ln_f = nn.LayerNorm(d_model)
        tf.head = nn.Linear(d_model, self.codebook_size)      # 词表 = 码本, 没有 BOS
        self.tf = init_weights(tf)                            # 初始 CE ≈ ln K
        nn.init.normal_(tf.start, std=0.02)                   # start 是裸 Parameter, init_weights 不碰它

        # 每个位置属于第几级: [0, 1,1,1,1, 2×16]
        scale_ids = torch.cat([torch.full((s * s,), k) for k, s in enumerate(self.scales)])
        self.register_buffer("scale_ids", scale_ids, persistent=False)
        self.register_buffer("mask", self.block_causal_mask(self.scales), persistent=False)

    @staticmethod
    def block_causal_mask(scales: Sequence[int]) -> torch.Tensor:
        """[L, L] bool, mask[i, j]=True 表示 i 能看 j: 当且仅当 j 的级别 ≤ i 的级别 (同级互相可见)。"""
        ids = torch.cat([torch.full((s * s,), k) for k, s in enumerate(scales)])   # [L] 每个位置的级别
        return ids[:, None] >= ids[None, :]                   # [L, 1] 对 [1, L] 广播比较 → [L, L]

    def _run(self, feats: List[torch.Tensor], B: int) -> torch.Tensor:
        """feats: 第 2..n 级的输入特征 [B, D, s_k, s_k] (n ≤ K) → 前 n 级的 logits [B, L_n, K]。"""
        # 1) 拼输入序列: 第 1 级是 start, 其余各级是特征图摊平后投到 d_model
        x = [self.tf.start.expand(B, -1, -1)]                                  # [1, 1, d] → [B, 1, d]
        # 每级 [B, D, s, s] → 摊平 [B, D, s²] → [B, s², D] → in_proj → [B, s², d]
        x += [self.tf.in_proj(f.flatten(2).transpose(1, 2)) for f in feats]    # [B, s², d]
        x = torch.cat(x, dim=1)                                                # [B, L_n, d]
        # 2) 加两种位置: 序列内第几个 + 属于第几级
        L = x.size(1)
        pos = torch.arange(L, device=x.device)                                 # [L_n]
        x = x + self.tf.pos_embed(pos) + self.tf.scale_embed(self.scale_ids[:L])
        # 3) block-causal 注意力。mask [L_n, L_n] 广播到 batch
        for block in self.tf.blocks:
            x = block(x, mask=self.mask[:L, :L])
        return self.tf.head(self.tf.ln_f(x))                                   # [B, L_n, K]

    def forward_tokens(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """teacher forcing: 真实 token 金字塔 → 全部 L 个位置的 logits [B, L, K]。"""
        feats = self.tokenizer.quantizer.next_scale_inputs(indices)            # K-1 个 [B, D, s_k, s_k]
        return self._run(feats, indices[0].size(0))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images [B, 3, H, W] -> 返回 dict, 两个键:
            "logits": [B, L, K]  每个位置在码本上的打分, K = codebook_size
            "labels": [B, L]     tokenizer 给出的真实 token, 按 "级 1 | 级 2 | ..." 摊平
        两者逐位对齐, 直接算交叉熵 (training/loss.py::VARLoss)。
        tokenizer 在 no_grad 下跑, 梯度只流向 Transformer。
        """
        indices = self.tokenizer.encode_to_indices(images)                     # K 个 [B, s_k, s_k]
        # 每级 [B, s, s] → [B, s²], 再按级拼接
        labels = torch.cat([i.flatten(1) for i in indices], dim=1)             # [B, L]
        return {"logits": self.forward_tokens(indices), "labels": labels}

    @torch.inference_mode()
    def sample_tokens(
        self, batch_size: int, temperature: float = 1.0, top_k: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        逐级采样, K 次前向。返回 K 个 token map, 第 k 项 [B, s_k, s_k]。
        简化: 每级重算全部前缀 (无 KV cache), L=21 时无所谓。
        """
        q = self.tokenizer.quantizer
        H = self.scales[-1]                                                    # 最细一级的边长 = latent 边长
        indices: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []                                         # 喂给 _run 的各级输入, 逐级变长
        f_hat = 0.0                                                            # 到目前为止的累计重建
        for k, s in enumerate(self.scales):
            if k > 0:
                f_hat = f_hat + q.lookup(indices[-1], H)                       # 累计重建 [B, D, H, H]
                feats.append(q._down(f_hat, s))
            logits = self._run(feats, batch_size)[:, -s * s:]                  # 只要最新一级 [B, s², K]
            logits = logits / max(temperature, 1e-5)                           # 1e-5 防除零 (temperature=0)
            if top_k is not None:
                kth = logits.topk(top_k, dim=-1).values[..., -1:]              # [B, s², 1] 第 k 大的分数
                logits = logits.masked_fill(logits < kth, float("-inf"))       # 比它小的都不采
            probs = F.softmax(logits, dim=-1)
            # multinomial 只收 2 维: [B, s², K] → [B·s², K] → 采样 [B·s², 1] → [B, s, s]
            idx = torch.multinomial(probs.flatten(0, 1), 1).view(batch_size, s, s)
            indices.append(idx)
        return indices

    @torch.inference_mode()
    def sample(self, batch_size: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """采样 token 金字塔, 再用 tokenizer 的 decoder 解码 → 图像 [B, 3, H, W]。"""
        return self.tokenizer.decode_from_indices(self.sample_tokens(batch_size, temperature, top_k))
