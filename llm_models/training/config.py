"""
训练超参 — 一个 frozen dataclass + "linear warmup → cosine decay" 调度

为什么 frozen: 配置构造后不可改, 训练中途被意外篡改是最难复现的一类 bug。
调度: step < warmup 时 lr 系数 = step / warmup, 从 **0** 线性升到 1。
        为什么要 warmup: Adam 前几步的动量估计噪声大, 直接用大 lr 容易发散。
      之后 0.5·(1 + cos(π·progress)), progress 从 0 到 1, 系数从 1 平滑降到 0。
        比线性衰减在末期降得更慢, 也不需要 step decay 那样的额外超参。
读代码时盯住: `lr_lambda(0) == 0`。
      第 1 步的 loss 在任何更新之前算出, 就是未训练模型的 loss。
      lr_lambda(0) = 0 让第 1 步的 optimizer.step 不改参数, 真正的更新从第 2 步开始。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """
    训练配置 (不可变)。

    Trainer 读的 6 个字段:
        learning_rate: 3e-4 是 Karpathy 戏称的 "Adam 默认最佳学习率",
                       中小规模 Transformer 通常不用再调。
        weight_decay:  AdamW 的解耦权重衰减系数。
                       简化: 严格做法只衰减矩阵权重, 排除 LayerNorm / bias 这类小参数;
                       本库对所有参数统一衰减。
        max_grad_norm: 梯度裁剪阈值, 常见经验值 1.0。
                       Transformer 训练初期容易出现梯度尖峰 (注意力 softmax 饱和时尤其),
                       不裁剪的话一步大更新就能毁掉整个模型。
        num_steps:     总训练步数 (本库只训几十步做演示)。
        warmup_steps:  lr 从 0 线性升到 learning_rate 要走的步数。
        log_interval:  每多少步打印一次训练指标。

    Trainer 不读的 5 个字段。它们是给脚本用的记事本, 只改 config 不会生效:
        batch_size / seq_len: 要由脚本传给 DataGenerator。
                       显存不够时可以用梯度累积模拟大 batch; 注意力计算量是 O(seq_len²)。
        aux_loss_weight: MoE 负载均衡 loss 的权重, 要传给 MoELMLoss。
                       太大损伤主任务, 太小路由坍塌到少数专家;
                       DeepSeek / Switch Transformer 的经验值约 0.01。
        audio_loss_weight: Qwen2.5-Omni 的 Talker (音频) loss 相对 Thinker (文本) loss 的权重,
                       要传给 OmniLoss。
        seed:          随机种子, 要由脚本自己调 torch.manual_seed。
    """
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_steps: int = 50
    warmup_steps: int = 10
    batch_size: int = 2
    seq_len: int = 32
    aux_loss_weight: float = 0.01
    audio_loss_weight: float = 0.5
    log_interval: int = 10
    seed: int = 42

    def get_lr_lambda(self, total_steps: int):
        """返回 lr_lambda(step) -> float, 直接传给 LambdaLR。公式和理由见文件头。

        total_steps: 总训练步数 (含 warmup)。
        返回值是乘在 learning_rate 上的系数, 不是 lr 本身。
        step 是 scheduler 内部的计数: 从 0 开始, 每调一次 scheduler.step() 加 1。
        """
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)                 # max(1, ·): warmup_steps = 0 时不除零
            # progress: 退火阶段走了多少, 0 → 1
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda
