"""
QAT — 量化感知训练: 前向走假量化 (Jacob et al. 2018), 反向用直通估计器 STE (Bengio et al. 2013 起的名字)

是什么: 训练时每次前向都把权重 "量化再反量化" (假量化), 让 loss 看到的就是部署时的 INT4 权重;
        反向时把取整当成恒等函数, 梯度原样流回高精度的影子权重 (latent weight)。
解决什么: 训练后量化 (RTN / GPTQ / AWQ) 是训完再压, 模型从来没在量化误差下被训练过, 低位宽时掉点;
          QAT 让模型在训练里就学会 "落在量化格点上也能用" 的权重。QLoRA 的量化基座是冻结的, 不在此列。
核心公式 (分组对称 INT4, 每 group_size 个权重共用一个 scale):
    s = max|w_group| / q_max,  q_max = 2^{b−1} − 1 (INT4 为 7)
    q = clamp(round(w / s), −q_max, q_max),   ŵ = q · s                     (假量化: 量化再反量化)
    STE: ∂L/∂w := ∂L/∂ŵ                                                    (round 的导数几乎处处为 0, 直接用它梯度就断了)
    代码里写成 w + (ŵ − w).detach(): 前向值是 ŵ, 反向对 w 的导数是 1。
与论文 / 真实系统的差异:
    简化: 只量化权重 (weight-only), 激活和 KV cache 保持 fp32。LLM-QAT (Liu et al. 2023) 还量化激活与 KV cache。
    简化: scale 每步按当前权重的 absmax 现算, 不学习 (LSQ 一类方法把 scale 也当参数训)。
    简化: STE 是纯恒等, 不在截断区间外置零梯度。absmax 定 scale 时没有权重落在截断区间外, 两种写法等价。
    简化: 对称量化取 [−7, 7] 共 15 个格点, 不用 −8 (真实 INT4 kernel 常用非对称或 [−8, 7])。
    LLM-QAT 用模型自己生成的数据做蒸馏; 本库直接用任务数据和交叉熵做 QAT。
量化谁: 每个 block 的 7 个线性层 (与 qlora.py 一致)。embedding / lm_head 保持高精度, 理由见 qlora.py 文件头。
读代码时盯住: `fake_quant` 末行的 detach 写法, 以及 `FakeQuantLinear.forward` 里训练和部署走的是同一个 ŵ。
依赖 llm_models: 靠属性名找层, 复用 lora.replace_linears。
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.methods.lora import ALL_LINEARS, replace_linears


def quantize_dequantize(w: torch.Tensor, bits: int = 4, group_size: int = 32) -> torch.Tensor:
    """
    w [d_out, d_in] → ŵ 同形状: 沿输入维每 group_size 个一组, 对称取整到 [−q_max, q_max] 再乘回 scale。
    不带梯度路径 (round 的导数是 0); 要梯度就用 fake_quant。group_size 必须整除 d_in。
    """
    d_out, d_in = w.shape
    if d_in % group_size:
        raise ValueError(f"group_size={group_size} 不整除 d_in={d_in}")
    q_max = 2 ** (bits - 1) - 1                                       # INT4: 7
    g = w.reshape(d_out, d_in // group_size, group_size)              # [d_out, n_groups, gs]
    # clamp_min: 整组全 0 时 absmax = 0, 下面要除以它
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / q_max   # [d_out, n_groups, 1]
    q = (g / scale).round().clamp(-q_max, q_max)                      # 整数格点
    return (q * scale).reshape(d_out, d_in)


def fake_quant(w: torch.Tensor, bits: int = 4, group_size: int = 32) -> torch.Tensor:
    """前向 = ŵ, 反向对 w 的梯度 = 1 (STE)。"""
    # ★ (ŵ − w).detach() 在前向里把 w 换成 ŵ, 反向时这一项没有梯度, 只剩 w 的恒等路径
    return w + (quantize_dequantize(w, bits, group_size) - w).detach()


class FakeQuantLinear(nn.Module):
    """
    包住一个 nn.Linear: 参数还是原来的高精度 weight (影子权重, 可训), 前向时用它的假量化版本。
    训练和部署共用 quantize_dequantize, 所以 QAT 训完直接取整, 部署时的输出与训练最后一步的前向一致。
    """

    def __init__(self, base: nn.Linear, bits: int = 4, group_size: int = 32) -> None:
        super().__init__()
        self.weight, self.bias = base.weight, base.bias               # 直接接管原参数, 不复制
        self.in_features, self.out_features = base.in_features, base.out_features
        self.bits, self.group_size = bits, group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, fake_quant(self.weight, self.bits, self.group_size), self.bias)


def apply_qat(model: nn.Module, bits: int = 4, group_size: int = 32,
              target_modules: Optional[Sequence[str]] = None) -> nn.Module:
    """原地把命中的 nn.Linear 换成 FakeQuantLinear。默认命中全部 7 种线性层。"""
    replace_linears(model, target_modules or ALL_LINEARS, lambda m: FakeQuantLinear(m, bits, group_size))
    return model


@torch.no_grad()
def rtn_quantize_(model: nn.Module, bits: int = 4, group_size: int = 32,
                  target_modules: Optional[Sequence[str]] = None) -> nn.Module:
    """
    训练后量化 (RTN, round-to-nearest): 把命中的线性层权重原地换成 ŵ, 返回普通 nn.Linear 的模型。
    FakeQuantLinear 会被换回 nn.Linear(ŵ): QAT 训完的模型也走这一步部署。
    """
    def to_linear(layer: nn.Module) -> nn.Linear:
        lin = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        lin.weight.copy_(quantize_dequantize(layer.weight, bits, group_size))
        if layer.bias is not None:
            lin.bias.copy_(layer.bias)
        return lin

    replace_linears(model, target_modules or ALL_LINEARS, to_linear, types=(nn.Linear, FakeQuantLinear))
    return model
