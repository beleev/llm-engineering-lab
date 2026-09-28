#!/usr/bin/env python
"""
SFT: 在 "反转 prompt" 任务上全参微调, 用**留出集** exact-match 验收; 顺带复现 prompt-mask 差一位的后果。

    python -m llm_finetune.run_finetune.sft.train_sft

实验设计:
    实验组: 随机初始化的小 LLaMA, 用正确的 labels 训 600 步 (prompt mask 只盖前 P−1 列)。
    对照组: 同一个种子、同样的步数和 lr, 只把 mask 多盖一位 (labels[:, :P] 全是 -100)。
            多盖的第 P−1 列正是 "SEP 这个位置要预测第一个回复 token y_1"。
    两组唯一的差别就是 labels 的这一列。
断言 (每条验证一个结论):
    1. 第 1 步 loss ≈ ln V。Trainer 第 1 步 lr = 0, 这是未训练模型的 loss, 相当于在 V 个 token 里均匀瞎猜。
    2. 实验组的留出集 exact-match > 0.9。留出集的 prompt 训练时没出现过, 模型学到的是规则。
    3. 对照组的 exact-match < 0.5。第一个回复 token 没人教, 它一错整条就判错。
依赖 llm_models: Trainer 第 1 步 lr = 0 (见 common.py 文件头); exact_match 靠 LLaMA.generate。
"""

import math

import torch

from llm_finetune.data import InstructionDataGenerator, SeqTask
from llm_finetune.methods.sft import SFTLoss
from llm_finetune.run_finetune.common import fit, make_model
from llm_finetune.utils import print_trainable_parameters

STEPS, LR = 600, 3e-3


class OffByOneMask(InstructionDataGenerator):
    """反面教材: 把 labels[:, :P] (而不是 [:, :P-1]) 置 -100 —— 第一个回复 token 从此没有监督。"""

    def _sample(self):
        batch = super()._sample()
        batch["labels"][:, self.task.prompt_len - 1] = -100      # 第 P−1 列原本是 y_1 (见 make_labels 的对齐图)
        return batch


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("reverse")

    model = make_model(task)
    print_trainable_parameters(model, "SFT (全参)")
    em_before = task.exact_match(model)
    metrics = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR)   # metrics[0] = 第 1 步的指标
    em = task.exact_match(model)

    # 对照组: 重置成同一个种子, 初始化与实验组相同, 只换数据生成器
    torch.manual_seed(0)
    buggy = make_model(task)
    fit(buggy, OffByOneMask(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
    em_buggy = task.exact_match(buggy)

    first = metrics[0]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(task.vocab_size):.3f})")
    print(f"留出集 exact-match: 训练前 {em_before:.3f} → 训练后 {em:.3f}")
    print(f"prompt mask 多盖一位 (labels[:, :P]) 的同配置模型: exact-match {em_buggy:.3f}")

    assert abs(first - math.log(task.vocab_size)) < 0.5, (
        f"第 1 步 loss 应 ≈ ln V = {math.log(task.vocab_size):.3f} (未训练模型在均匀瞎猜), 实际 {first:.3f}")
    assert em > 0.9, f"SFT 没学会任务: 留出集 exact-match {em:.3f}, 应 > 0.9"
    assert em_buggy < 0.5 < em, (
        f"mask 多盖一位后第一个回复 token 没有监督, exact-match 应 < 0.5; "
        f"实际 {em_buggy:.3f} (正确 mask 的模型 {em:.3f})")


if __name__ == "__main__":
    main()
