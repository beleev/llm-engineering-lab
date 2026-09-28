"""
DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)

是什么: 把权重拆成 "每行的长度 m" × "方向 V/‖V‖", LoRA 只改方向, 长度单独用一个向量学。
解决什么: 论文观察到全参微调常常 "方向变很多、长度几乎不变" (或反过来), 而 LoRA 的 ΔW = BA 把两者绑在一起按比例变;
          拆开后低秩支路只需表达方向变化, 同样的 r 更接近全参微调。
核心公式:  W' = m ⊙ (W + (α/r)BA) / ‖W + (α/r)BA‖_row        m 初始化为 ‖W‖_row ⇒ 第 0 步 W' = W
           比 LoRA 每层只多 d_out 个参数。
读代码时盯住: `norm.detach()` —— 论文 §4.3: 反传时把分母当常数, 省掉一整份 [d_out, d_in] 的梯度显存, 精度几乎不变。
与论文的差异 (范数沿哪个方向取): 本库的做法是, weight 形状 [d_out, d_in], 对每个输出神经元的那一行权重取范数 (dim=1),
              m 有 d_out 个分量。论文把这个范数记作逐列的 ‖·‖_c, 对着论文公式读代码时留意这一点。
              PEFT 库的 DoRA 同样对 dim=1 取范数。
与 LoRA 的另一处不同: dropout > 0 时, 这里丢的是整层的输入 (基座通路也受影响); LoRALinear 只丢低秩支路的输入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.methods.lora import LoRALinear


class DoRALinear(LoRALinear):
    """
    LoRALinear + 一个可训的长度向量 lora_magnitude [d_out]。A、B、scaling、delta() 全部继承。
    只重写两处: merged_weight() 多了 "归一化再乘 m", forward() 每步用它现算整个 W'。
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0) -> None:
        super().__init__(base, r=r, alpha=alpha, dropout=dropout)
        # 名字带 "lora_": 复用 mark_only_lora_as_trainable / get_lora_state_dict
        # m 的初值 = 基座每一行的范数。base 是 NF4Linear 时, .weight 给出的是反量化后的权重
        self.lora_magnitude = nn.Parameter(base.weight.detach().norm(dim=1))       # [d_out, d_in] → [d_out]

    def merged_weight(self) -> torch.Tensor:
        direction = self.base.weight + self.delta()                  # V  [d_out, d_in] (未归一化)
        norm = direction.norm(dim=1, keepdim=True).detach()          # [d_out, 1] 当常数: 梯度只经分子和 m
        return self.lora_magnitude.unsqueeze(1) * direction / norm   # [d_out, 1] · [d_out, d_in] / [d_out, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化是对**整行权重**做的, 没法像 LoRA 那样拆成两次小 matmul: 每步都要显式构造 W'
        # dropout > 0 时这里丢的是整层输入 (含基座通路); LoRALinear 只丢低秩支路
        return F.linear(self.lora_dropout(x), self.merged_weight(), self.base.bias)
