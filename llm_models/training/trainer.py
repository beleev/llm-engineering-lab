"""
通用训练器 — 一个训练循环跑全库所有模型

是什么: Trainer 只认两个策略接口 —— SyntheticDataGenerator (batch 怎么来) 与 LossComputer (loss 怎么算);
        新模型只需配一对 generator / loss, 循环本身不改。
单步: generate_batch → pop labels → model(**batch) → loss → backward → clip_grad_norm → AdamW.step → scheduler.step
关键数字: 第 1 步的 loss 在任何更新之前算出, 就是 "未训练模型" 的 loss,
          train 脚本据此断言 `|loss₁ − ln V| < 0.5`。
          另外第 1 步的 lr = 0 (线性 warmup 从 0 起): 这一步的 optimizer.step 不改参数,
          真正的更新从第 2 步开始。
注意: 默认数据是固定的一个 batch (见 data.py), 所以 "loss 下降" = 能背下这个 batch。
读代码时盯住: `batch.pop("labels", None)` —— dict 里剩下的 key 必须正好是 model.forward 的形参名。
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn

from llm_models.training.config import TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer

# Metric: 训练步内部的指标值。tensor 用于 GPU 上累积, float 用于纯标量 (如 lr)。
Metric = Union[torch.Tensor, float]


class Trainer:
    """
    AdamW + (linear warmup → cosine) + 梯度裁剪。

    - AdamW (LLM 训练的事实标准): weight decay 与梯度解耦, 不被 Adam 的自适应缩放扭曲。
      简化: 严格做法应排除 norm / bias, 这里对所有参数统一衰减。
    - clip_grad_norm_: Transformer 训练初期常有梯度尖峰 (softmax 饱和), 一次大更新就能毁掉模型。
    - metrics 以 tensor 返回, 只在打日志时 .item(): 每步 .item() 会强制 GPU→CPU 同步,
      大模型 + 大 batch 下会卡住 GPU 流水线。
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_generator: SyntheticDataGenerator,
        loss_computer: LossComputer,
    ):
        self.model = model
        self.config = config
        self.data_generator = data_generator
        self.loss_computer = loss_computer

        # 只放可训参数: LoRA 冻结的基座 / 蒸馏的 teacher / DPO 的 ref 不进 optimizer。
        # 所以冻结 (requires_grad=False) 必须在构造 Trainer 之前做
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 先线性 warmup 到峰值, 再余弦退火到 0 (公式见 config.py)。
        # LambdaLR 每步把 base_lr 乘上 lr_lambda(step) 当作实际 lr。
        # 调度的总长就是 config.num_steps: 走过它之后余弦会掉头, lr 又升回峰值,
        # 所以 train_step 不许超过这个步数 (要训更久就改 config.num_steps)
        lr_lambda = config.get_lr_lambda(config.num_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self._global_step = 0

    def train_step(self) -> Dict[str, Metric]:
        """跑一步训练。返回 dict: 各项 loss (tensor)、grad_norm (tensor)、lr (float)。

        loss 保持 tensor 不转 float, 原因见类 docstring。
        已经走满 config.num_steps 步时抛 ValueError: 再走 lr 会回升 (见 __init__)。
        """
        if self._global_step >= self.config.num_steps:
            raise ValueError(
                f"已训练 {self._global_step} 步, 达到 config.num_steps={self.config.num_steps}; "
                "学习率调度按这个步数建, 再训 lr 会回升。要训更久请调大 num_steps"
            )
        self.model.train()

        # ---- 1) 取数据 ----
        batch = self.data_generator.generate_batch()

        # ---- 2) 拆出 labels / extra_labels, 让 batch 中只剩 model.forward 入参 ----
        # 用 pop 而非 索引: 原地从 dict 移除, 保证后面 **batch 不会把 labels 误传给 forward
        # 没有 labels 键 (对比 / 偏好 / VAE 这类不需要标签的 loss) 时传 None, 由 LossComputer 自己决定用不用
        labels = batch.pop("labels", None)
        extra_labels = {}
        if "audio_labels" in batch:
            extra_labels["audio_labels"] = batch.pop("audio_labels")

        # ---- 3) 模型前向 ----
        # **batch 解包: 各模型的 forward 签名不同, 由各自的 data_generator 保证字段对齐
        output = self.model(**batch)

        # ---- 4) 计算损失 ----
        loss_dict = self.loss_computer.compute(output, labels, **extra_labels)

        # ---- 5) 反向传播 ----
        loss_dict["total_loss"].backward()

        # ---- 6) 梯度裁剪: 防梯度爆炸, 尤其是 Transformer 训练前几百步 ----
        # 返回值是裁剪 **之前** 的总梯度范数, 日志里看到的 grad_norm 可以大于 max_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # ---- 7) 参数更新 → 学习率调度 → 清零梯度 ----
        # 顺序约定: optimizer.step 必须先于 scheduler.step (PyTorch >= 1.1)。
        # 反过来的话第 1 步就会跳过 lr_lambda(0), 调度整体错一位
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()      # 不清零的话下一步的梯度会叠加在这一步上

        self._global_step += 1

        metrics: Dict[str, Metric] = dict(loss_dict)
        metrics["grad_norm"] = grad_norm
        # scheduler 已经 step 过: 这里记的是 **下一步** 要用的 lr, 所以第 1 条日志的 lr 不是 0
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        return metrics

    def train(self, num_steps: int = 0) -> List[Dict[str, float]]:
        """跑完整个训练循环。num_steps 传 0 就用 config.num_steps。

        num_steps 可以少于 config.num_steps (lr 没退火到 0 就停), 不能多:
        加上已训的步数超过 config.num_steps 时, 开训前就抛 ValueError (原因见 __init__)。
        返回打过日志的那几步的 metrics 列表 (第 1 步 / 每 log_interval 步 / 最后一步),
        值已转成 float, 便于序列化和绘图。
        """
        steps = num_steps if num_steps > 0 else self.config.num_steps
        if self._global_step + steps > self.config.num_steps:
            raise ValueError(
                f"要训 {steps} 步, 已训 {self._global_step} 步, 合计超过 config.num_steps="
                f"{self.config.num_steps}; 学习率调度按它建, 超出部分 lr 会回升。请调大 num_steps"
            )
        all_metrics: List[Dict[str, float]] = []

        # 训练横幅
        print(f"\n{'=' * 60}")
        print(f"开始训练 | 总步数: {steps} | LR: {self.config.learning_rate}")
        print(f"{'=' * 60}")

        for step in range(1, steps + 1):
            metrics = self.train_step()

            # 只在 第一步 / 间隔点 / 最后一步 打印
            if step == 1 or step % self.config.log_interval == 0 or step == steps:
                scalar_metrics = self._to_scalar(metrics)
                self._log_metrics(step, steps, scalar_metrics)
                all_metrics.append(scalar_metrics)

        # 训练结束横幅, 顺带打印 loss 收敛情况, 方便快速判断训练是否生效
        print(f"{'=' * 60}")
        print("训练完成!")
        if len(all_metrics) >= 2:
            first_loss = all_metrics[0]["total_loss"]
            final_loss = all_metrics[-1]["total_loss"]
            print(f"Loss: {first_loss:.4f} -> {final_loss:.4f}")
        print(f"{'=' * 60}\n")

        return all_metrics

    @staticmethod
    def _to_scalar(metrics: Dict[str, Metric]) -> Dict[str, float]:
        """把 tensor 形式的 metrics 转成 float。Trainer 里唯一调 .item() 的地方, 只在打日志时进来。"""
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            out[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return out

    def _log_metrics(self, step: int, total_steps: int, metrics: Dict[str, float]):
        """打印一行训练指标。lr 用科学计数法 (它跨多个量级), 其余保留 4 位小数。"""
        parts = [f"Step [{step:>4d}/{total_steps}]"]
        for k, v in metrics.items():
            if k == "lr":
                parts.append(f"{k}: {v:.2e}")
            else:
                parts.append(f"{k}: {v:.4f}")
        print(" | ".join(parts))
