#!/usr/bin/env python
"""
On-policy 蒸馏 (reverse KL) vs off-policy 蒸馏 (forward KL): 同一个 teacher、同一个热身后的 student、同样的步数。

    python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill

实验设计:
    teacher、student 的结构、数据、evaluate 都从 train_distill.py 直接 import。
    teacher 学的是 "70% sort / 30% copy" 两种答案的混合; student 容量不够, 学不全。
    热身: student 先用 forward KL 训 200 步。on-policy 要求 student 已经能说出点东西 (真实流程里它先经过 SFT 或 off-policy 蒸馏)。
    两臂都从热身后的 student 的 deepcopy 出发, 各 300 步, lr=1e-3:
        + off-policy forward KL    数据来自混合数据集, 走通用 Trainer       ← 对照组
        + on-policy reverse KL     数据是 student 自己采的回复, 手写循环
    forward KL 这一臂用 DistillLoss(T=1, α=0): 纯 KL(teacher‖student), 不掺硬标签, 才能和 reverse KL 对称地比。
断言 (每条验证一个结论):
    1. 各自在自己优化的指标上赢: on-policy 的 reverse KL 更低, off-policy 的 forward KL 更低。
    2. on-policy 的样本合格率比 off-policy 高 0.1 以上。reverse KL 是 mode-seeking: 容量不够时宁可只做好一种答案。
    3. on-policy 给多数派答案 (sort) 的概率更高, 给少数派答案 (copy, 30%) 的概率更低。
       这是第 2 条的代价: 少数派被放弃得更彻底。forward KL 两头都留着概率。
依赖 llm_models: off-policy 一臂走 Trainer; on-policy 一臂靠 `LLaMA.generate` 采样。
"""

import copy

import torch

from llm_finetune import DistillLoss, PromptDataGenerator, TeacherStudent, on_policy_distill_step
from llm_finetune.run_finetune.common import fit
from llm_finetune.run_finetune.distill.train_distill import (
    MAJOR, MixtureData, evaluate, make_student, report, train_teacher,
)

WARMUP_STEPS, STEPS, LR = 200, 300, 1e-3


def main() -> None:
    torch.manual_seed(0)
    teacher = train_teacher()
    forward_kl = DistillLoss(temperature=1.0, alpha=0.0)       # 纯 KL(teacher‖student), 不掺硬标签, 与 reverse KL 对称比较

    # on-policy 需要一个 "已经能说出点东西" 的 student (真实流程里它先经过 SFT / off-policy 蒸馏)
    torch.manual_seed(1)
    warm = make_student()
    fit(TeacherStudent(warm, teacher), MixtureData(), forward_kl, WARMUP_STEPS, 3e-3, log_interval=WARMUP_STEPS)
    rows = {"teacher": evaluate(teacher, teacher), "热身后的 student": evaluate(warm, teacher)}

    off = copy.deepcopy(warm)
    fit(TeacherStudent(off, teacher), MixtureData(), forward_kl, STEPS, LR, log_interval=STEPS)
    rows["+ off-policy forward KL"] = evaluate(off, teacher)

    on = copy.deepcopy(warm)
    optimizer = torch.optim.AdamW(on.parameters(), lr=LR)
    prompts = PromptDataGenerator(MAJOR, 64)                   # 只出 prompt; 回复由 student 现场采
    for step in range(1, STEPS + 1):
        m = on_policy_distill_step(on, teacher, prompts.generate_batch()["prompts"], MAJOR.response_len, optimizer)
        if step % 100 == 0:
            print(f"on-policy 第 {step} 步: 自己样本上的 reverse KL {m['reverse_kl']:.3f}")
    rows["+ on-policy reverse KL"] = evaluate(on, teacher)
    report(rows)

    off_m, on_m = rows["+ off-policy forward KL"], rows["+ on-policy reverse KL"]
    assert on_m["reverse_kl"] < off_m["reverse_kl"], (
        f"各自在自己优化的那个指标上赢: on-policy 的 reverse KL 应更低, "
        f"实际 on {on_m['reverse_kl']:.3f}, off {off_m['reverse_kl']:.3f}")
    assert off_m["forward_kl"] < on_m["forward_kl"], (
        f"各自在自己优化的那个指标上赢: off-policy 的 forward KL 应更低, "
        f"实际 off {off_m['forward_kl']:.3f}, on {on_m['forward_kl']:.3f}")
    assert on_m["valid"] > off_m["valid"] + 0.1, (
        f"mode-seeking: on-policy 自己采样出来的回复合格率应更高 (容量不够时宁可只做好一种答案), "
        f"高 0.1 以上: on {on_m['valid']:.3f}, off {off_m['valid']:.3f}")
    assert on_m["logp_major"] > off_m["logp_major"], (
        f"on-policy 应把更多概率给多数派答案 (sort): "
        f"log π on {on_m['logp_major']:.2f}, off {off_m['logp_major']:.2f}")
    assert on_m["logp_minor"] < off_m["logp_minor"], (
        f"代价: teacher 的少数派答案 (copy, 30%) 应被 on-policy 放弃得更彻底, forward KL 则两头都留着概率: "
        f"log π on {on_m['logp_minor']:.2f}, off {off_m['logp_minor']:.2f}")


if __name__ == "__main__":
    main()
