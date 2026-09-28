"""
VAE — Latent Diffusion 的前置压缩器 (Kingma & Welling 2013; Rombach et al. 2022)

解决的问题: 在 512×512×3 像素上做扩散太贵; 先压到 64×64×4 的 latent (元素数 ÷48) 再扩散。
为什么不是普通 AE: AE 的 latent 分布任意, 扩散模型无从假设; VAE 用 KL 把它拉向 N(0, I)。

    μ, logσ² = Encoder(x);   z = μ + σ·ε, ε~N(0,I)      # 重参数化: 随机性挪到 ε, 梯度能穿过采样
    loss = MSE(Decoder(z), x) + kl_weight · KL,   KL = -0.5·Σ(1 + logσ² - μ² - σ²)

关键数字: SD 的 kl_weight ~1e-6 —— 几乎就是 AE, 只要 latent 别离 N(0,I) 太远。
简化 (相对 SD 的 VAE):
    - 主干只有 Conv → GroupNorm → SiLU 的直筒堆叠; SD 用 ResNet block, 瓶颈处还有注意力。
    - 重建 loss 只用 MSE; SD 还加了感知 loss 和对抗 loss。
    - 默认下采样 2 次 (空间 ÷4); SD 是 3 次 (÷8)。
读代码时盯住: ImageVAE.reparameterize, 以及 encoder 的两个 1×1 头 (mean_head / logvar_head)。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """一个"Conv → GroupNorm → SiLU"块, GN 在小 batch 上比 BN 稳定。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
        nn.SiLU(),
    )


class ImageVAEEncoder(nn.Module):
    """
    VAE 图像 encoder

    forward: x [B, 3, H, W] -> tuple (mean, logvar), 各 [B, latent_dim, H/2^L, W/2^L],
    L = downsample_levels。logvar 是 log σ², 这里还没 clamp。

    Args:
        in_channels:   输入图像通道 (3 for RGB)
        base_channels: 首层通道数, 每次下采样翻倍 (默认 64 → 128 → 256)
        latent_dim:    潜空间通道数 (SD 1.5 用 4; 教学默认 4)
        downsample_levels: 下采样次数 (3 → 空间 ÷8; 教学默认 2 → ÷4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        downsample_levels: int = 2,
    ):
        super().__init__()

        # 首层卷积: 进入 base_channels
        layers = [_conv_block(in_channels, base_channels)]

        ch = base_channels
        for _ in range(downsample_levels):
            # 每级: 一次 stride-2 下采样 + 一次普通 conv
            layers.append(_conv_block(ch, ch * 2, stride=2))
            layers.append(_conv_block(ch * 2, ch * 2))
            ch *= 2

        self.trunk = nn.Sequential(*layers)
        # 瓶颈用 1×1 卷积产出 μ 和 log σ² 两组张量
        self.mean_head = nn.Conv2d(ch, latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv2d(ch, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)                                   # [B, ch, H/2^L, W/2^L]
        return self.mean_head(h), self.logvar_head(h)       # 2 × [B, latent_dim, H/2^L, W/2^L]


class ImageVAEDecoder(nn.Module):
    """
    VAE 图像 decoder (与 encoder 镜像)

    forward: z [B, latent_dim, h, w] -> [B, 3, h·2^L, w·2^L], 值域 [-1, 1] (末尾 tanh)。
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        upsample_levels: int = 2,
    ):
        super().__init__()

        # 最深通道 = base_channels * 2^upsample_levels
        ch = base_channels * (2 ** upsample_levels)
        layers = [_conv_block(latent_dim, ch)]

        for _ in range(upsample_levels):
            # 每级: upsample (bilinear) + conv, 比 ConvTranspose 不易出 checkerboard
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            layers.append(_conv_block(ch, ch // 2))
            layers.append(_conv_block(ch // 2, ch // 2))
            ch //= 2

        # 输出层: 回到 pixel 通道 (forward 里再过 tanh → [-1, 1])
        layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))

        self.trunk = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(z))


class ImageVAE(nn.Module):
    """
    完整 VAE (encoder + decoder + reparameterization)

    forward 流程:
        μ, logσ² = encoder(x)
        ε ~ N(0, I)
        z = μ + σ · ε                   # reparameterization trick (让梯度可回传)
        x̂ = decoder(z)

    返回重建 + 分布参数, 供外部计算 loss:
        recon_loss = ||x - x̂||²
        kl_loss    = -0.5 · Σ (1 + logσ² - μ² - σ²)

    forward 返回 dict, 四个键 (见 forward)。没有 attention_mask, 没有 KV cache。

    Args:
        image_channels / base_channels / latent_dim / levels 见 encoder/decoder
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        levels: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = ImageVAEEncoder(
            in_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, downsample_levels=levels,
        )
        self.decoder = ImageVAEDecoder(
            out_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, upsample_levels=levels,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x [B, 3, H, W] -> {"mean", "logvar"}, 各 [B, latent_dim, h, w]。"""
        mean, logvar = self.encoder(x)
        # 数值安全: clamp logvar 避免 exp 爆炸 (SD 常见做法)。
        # σ = exp(logvar / 2) 被限制在 [exp(-15), exp(10)]: 不会溢出成 inf, 也不会下溢成 0
        logvar = logvar.clamp(-30.0, 20.0)
        return {"mean": mean, "logvar": logvar}

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + σ·ε。随机性全在 ε 里, 梯度能穿过 μ 和 σ 回到 encoder。"""
        std = torch.exp(0.5 * logvar)                       # logσ² → σ
        eps = torch.randn_like(std)                         # ε ~ N(0, I), 不带梯度
        return mean + std * eps                             # [B, latent_dim, h, w]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, latent_dim, h, w] -> 图像 [B, 3, H, W], 值域 [-1, 1]。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x [B, 3, H, W] (值域 [-1, 1]) -> 返回 dict, 四个键:
            "recon":  [B, 3, H, W]            重建图像
            "z":      [B, latent_dim, h, w]   采样出的 latent, h = H / 2^levels
            "mean":   [B, latent_dim, h, w]   μ
            "logvar": [B, latent_dim, h, w]   log σ², 已 clamp 到 [-30, 20]
        loss 不在这里算, 见 training/loss.py::VAELoss。
        """
        enc = self.encode(x)
        z = self.reparameterize(enc["mean"], enc["logvar"])
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mean": enc["mean"], "logvar": enc["logvar"]}
