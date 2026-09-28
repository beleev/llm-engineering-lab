"""
LoRA — Low-Rank Adaptation (Hu et al., 2021)

是什么: 冻结 W, 在旁边加一条低秩支路, 只训 A、B。
解决什么: 全参 SFT 要为每个参数存梯度 + 两份 Adam 状态 (≈ 权重的 3~4 倍显存), 每个下游任务还要存一整份模型。
核心公式:  y = W x + (α/r) · B A x        A ∈ R^{r×d_in} (Kaiming uniform),  B ∈ R^{d_out×r} (全零)
           B = 0 ⇒ 训练起点严格等于原模型; 训完 W' = W + (α/r)BA 可合并, 推理零开销。
           参数量 d_out·d_in → r·(d_in + d_out);  α/r 让改 r 时不必重调 lr。
读代码时盯住: `delta()` —— 训练 / 合并 / DoRA / QLoRA 全都围着这一个 ΔW 转。
省的是显存和存储, 不是步数: 同一任务上 LoRA 通常比全参收敛**慢** (实测见 run_finetune/lora/readme.md)。
与论文的差异: 论文正文写 A 用随机高斯初始化, 本库用 Kaiming uniform (官方 loralib 的代码也是这样写的)。
              论文的实验多数只注入 W_q / W_v, 本库默认注入注意力的四个投影。
依赖 llm_models: 靠**属性名**找层。w_q / w_k / w_v / w_o 是注意力的四个投影, w_gate / w_up / w_down 是 SwiGLU 的三个矩阵。
                 上游改了这些名字, 这里就一层也命中不了 (replace_linears 会抛 ValueError)。
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.utils.param_utils import freeze_module

ATTENTION_LINEARS: List[str] = ["w_q", "w_k", "w_v", "w_o"]                    # GQA 的四个投影 (默认)
ALL_LINEARS: List[str] = ATTENTION_LINEARS + ["w_gate", "w_up", "w_down"]      # + SwiGLU 三个矩阵


class LoRALinear(nn.Module):
    """
    包住一个线性层 `base` (nn.Linear, 或 qlora.NF4Linear —— 只要有 .weight / .bias / in,out_features)。
    可训参数名都以 "lora_" 开头, mark_only_lora_as_trainable / get_lora_state_dict 靠这个前缀识别。
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r 必须为正, 当前 r={r}")
        self.base = base
        freeze_module(base)                                          # 基座只读: 梯度只流进 lora_A / lora_B
        self.scaling = alpha / r                                     # α/r: ΔW 前面的缩放系数
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # A: Kaiming uniform (nn.Linear 的默认初始化), 让 A x 的方差与输入同量级
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))             # [r, d_in]
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)            # a=√5: nn.Linear 默认初始化用的同一个值
        # B: 全零 ⇒ BA = 0 ⇒ 第 0 步输出与原模型逐位相同 ("无害启动")
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))            # [d_out, r]

    def delta(self) -> torch.Tensor:
        """ΔW = (α/r)·B A   [d_out, d_in]"""
        return self.scaling * (self.lora_B @ self.lora_A)            # [d_out, r] @ [r, d_in] → [d_out, d_in]

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # [..., d_in] → [..., d_out]
        # 两次小 matmul (d_in→r→d_out), 不显式构造 d_out×d_in 的 ΔW
        # dropout 只丢低秩支路的输入; 基座那一路 self.base(x) 吃的是原始 x
        # 形状: [..., d_in] → [..., r] → [..., d_out]
        low_rank = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * low_rank

    def merged_weight(self) -> torch.Tensor:
        """W' = W + ΔW; base 是 NF4 时 .weight 先反量化 (dequant → merge)。"""
        return self.base.weight + self.delta()


def replace_linears(
    model: nn.Module,
    targets: Sequence[str],
    build: Callable[[nn.Module], nn.Module],
    types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """把属性名 (路径最后一段, 如 "w_q") 命中 targets 的子层换成 build(子层), 返回替换个数。"""
    # 先收集再替换: 不能一边遍历 named_modules 一边改它
    hits = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if name in targets and isinstance(child, types)
    ]
    if not hits:
        raise ValueError(f"模型里没有名字在 {list(targets)} 中的 {types} 层; 检查 target_modules")
    for parent, name, child in hits:
        setattr(parent, name, build(child))
    return len(hits)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    layer_cls: type = LoRALinear,
) -> nn.Module:
    """原地注入适配器。layer_cls=DoRALinear 即得 DoRA。只注入, 不冻结其余参数 (见下一个函数)。"""
    replace_linears(model, target_modules or ATTENTION_LINEARS,
                    lambda m: layer_cls(m, r=r, alpha=alpha, dropout=dropout))
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """embedding / norm / lm_head 也一并冻结: 只有名字含 "lora_" 的参数可训。"""
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    把每个适配器层换回一个普通 nn.Linear(W') —— 之后模型里不再有任何 LoRA 痕迹, 推理零开销。
    LoRA / DoRA / QLoRA 通用: 只要求该层实现 merged_weight()。QLoRA 走 "反量化 → 合并", 结果是高精度权重
    (重新量化会把刚学到的 ΔW 再抹掉一部分, 所以业界合并后一般保持 16-bit)。
    """
    def to_linear(layer: nn.Module) -> nn.Linear:
        # 新建一个同形状的 nn.Linear, 把合并后的 W' 和原 bias 抄进去
        bias = layer.base.bias
        lin = nn.Linear(layer.base.in_features, layer.base.out_features, bias=bias is not None)
        lin.weight.copy_(layer.merged_weight())
        if bias is not None:
            lin.bias.copy_(bias)
        return lin

    # 收集所有适配器层的属性名 (DoRALinear 是 LoRALinear 的子类, 一并命中)
    names = {n for m in model.modules() for n, c in m.named_children() if isinstance(c, LoRALinear)}
    if not names:
        raise ValueError("merge_lora_weights: 模型里没有 LoRA 层 (已经合并过, 或从未 apply_lora)")
    replace_linears(model, names, to_linear, types=(LoRALinear,))
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """只含适配器的 state_dict: 部署时 基座 + apply_lora + load_state_dict(adapter, strict=False)。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
