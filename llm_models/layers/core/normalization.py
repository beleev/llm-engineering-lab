"""
RMSNorm (Zhang & Sennrich, 2019) — LLaMA / Qwen / DeepSeek / Gemma 的标配归一化

是什么: RMSNorm(x) = x / sqrt(mean(x²) + eps) · γ
解决什么: LayerNorm = (x − mean)/std · γ + β。实验表明起作用的主要是 "缩放" 而不是 "去均值";
          去掉 mean 和 bias 后少一次统计、一次减法、一组参数, 大模型上效果几乎不变。
也用于 QK-Norm: 在 head_dim 上归一化 Q/K, 归一化后 |q| = |k| = sqrt(Dh) (γ=1 时), attention logit ≤ sqrt(Dh)。
          γ 可学, 所以上界实际是 sqrt(Dh)·max|γ_q|·max|γ_k|: 由两组小参数控制, 与 W_q / W_k 的大小无关
          (见 attention.py::GroupedQueryAttention)。
读代码时盯住: `dim=-1` —— 只在最后一维 (特征维) 上统计, 每个 token 独立归一化。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Args:
        d_model: 最后一维的大小
        eps: 加在 sqrt 内部防除零 (LLaMA 风格, 1e-6 ~ 1e-5)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))     # γ, 初始 1 = 纯归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # [..., d_model] -> 同形状
        # rms = 这个 token 各维的均方根。keepdim 留住最后一维, 才能和 x 广播相除
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # [..., 1]
        return (x / rms) * self.weight                      # 除完每个 token 的 RMS = 1, 再逐维乘 γ [d_model]
