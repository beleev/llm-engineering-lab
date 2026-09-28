#!/usr/bin/env python
"""
Reward Model: SFT 主干 + 标量头, Bradley-Terry loss, 通用 Trainer 训练; 在**留出**偏好对上验收,
并单测 "右 pad 时分数取自最后一个真 token"。

    python -m llm_finetune.run_finetune.rm.train_rm

实验设计:
    主干: sort 任务上 SFT 100 步的小 LLaMA, 后面接一个标量头。
    训练: 每步新采一批偏好对 (正确回复 vs 损坏回复), Bradley-Terry loss, 300 步。
    验收: 512 对留出偏好对, 固定不变。训练前后各评一次, 训练前那次是对照。
    右 pad 单测: 取 8 条 chosen 序列, 在后面多垫 5 个 PAD, 比三种读法的分数:
        clean    不垫, 带 mask                            ← 基准
        padded   垫 5 个 PAD, 带 mask (pad 处为 0)        应与 clean 相同
        naive    垫 5 个 PAD, 不给 mask                   读到的是最后一个 pad 位置的分
断言 (每条验证一个结论):
    1. 留出集的 rejected 里确实有带右 pad 的序列。没有的话, "取最后一个真 token" 这条分支就没被测到。
    2. 第 1 步 loss ≈ ln 2。标量头小初始化, 两边分数几乎相等; Trainer 第 1 步 lr = 0。
    3. 留出集偏好准确率: 训练前 < 0.9 < 训练后。
    4. padded 与 clean 相同 (误差 < 1e-4): 垫 pad 不改变分数。
    5. naive 与 clean 相差 > 1e-2: 不给 mask 真的会读错位置。这条保证第 4 条单测有区分力。
依赖 llm_models: RewardModel 靠 `LLaMA.forward(..., return_hidden=True)` 取隐状态; Trainer 第 1 步 lr = 0。
"""

import torch
import torch.nn.functional as F

from llm_finetune import BradleyTerryLoss, PairwiseForward, PreferenceDataGenerator, RewardModel, SeqTask
from llm_finetune.data import PAD
from llm_finetune.run_finetune.common import fit, make_model, sft_warmup

SFT_STEPS, STEPS, LR = 100, 300, 1e-3
N_PAD = 5                                                 # 右 pad 单测在序列后面多垫几个 PAD


@torch.no_grad()
def heldout_accuracy(rm: RewardModel, batch) -> float:
    """batch 是一批偏好对 → RM 给 chosen 的分高于 rejected 的比例。"""
    rm.eval()
    r_w = rm(batch["chosen_input_ids"], batch["chosen_attention_mask"])
    r_l = rm(batch["rejected_input_ids"], batch["rejected_attention_mask"])
    return float((r_w > r_l).float().mean())


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    backbone = make_model(task)
    sft_warmup(backbone, task, SFT_STEPS)                     # 业界惯例: RM 从 SFT checkpoint 起步
    rm = RewardModel(backbone)                                # 取 backbone 的 return_hidden=True 隐状态接标量头

    heldout = PreferenceDataGenerator(task, 512, split="test", fixed=True).generate_batch()
    acc_before = heldout_accuracy(rm, heldout)
    hist = fit(PairwiseForward(rm), PreferenceDataGenerator(task), BradleyTerryLoss(), STEPS, LR, log_interval=100)
    acc = heldout_accuracy(rm, heldout)
    print(f"第 1 步 loss {hist[0]['total_loss']:.4f} (≈ ln 2); 留出集偏好准确率 {acc_before:.3f} → {acc:.3f}")

    # ---- 右 pad 单测: 同一条序列, 后面多垫 N_PAD 个 PAD, 分数必须不变 ----
    ids, mask = heldout["chosen_input_ids"][:8], heldout["chosen_attention_mask"][:8]
    with torch.no_grad():
        clean = rm(ids, mask)
        padded = rm(F.pad(ids, (0, N_PAD), value=PAD), F.pad(mask, (0, N_PAD), value=0))
        naive = rm(F.pad(ids, (0, N_PAD), value=PAD))           # 不给 mask 就退化成 scores[:, -1]: 读到的是 pad 位置
    print(f"右 pad {N_PAD} 位后的分数偏移: 取最后一个真 token {float((padded - clean).abs().max()):.1e} | "
          f"取 [:, -1] {float((naive - clean).abs().max()):.2f}")

    assert (heldout["rejected_attention_mask"].sum(1) < heldout["rejected_attention_mask"].size(1)).any(), (
        "留出集的 rejected 里应有带右 pad 的序列 ('漏一个 token' 的那一半); "
        "有了它, 训练 / 评估才全程在走 '取最后一个真 token' 的 gather 分支")
    assert abs(hist[0]["total_loss"] - 0.6931) < 0.05, (
        f"第 1 步 loss 应 ≈ ln 2 = 0.6931 (标量头小初始化, r_w − r_l ≈ 0), 实际 {hist[0]['total_loss']:.4f}")
    assert acc > 0.9 > acc_before, (
        f"RM 没学会在留出集上排序: 偏好准确率应从 < 0.9 升到 > 0.9, 实际 {acc_before:.3f} → {acc:.3f}")
    assert torch.allclose(padded, clean, atol=1e-4), (
        f"pad 不应改变分数: 垫 {N_PAD} 个 PAD 后分数最大偏移 {float((padded - clean).abs().max()):.1e}, 应 < 1e-4")
    assert (naive - clean).abs().max() > 1e-2, (
        f"不给 mask 时读到的是 pad 位置的分, 应与真 token 处的分相差 > 1e-2; "
        f"实际只差 {float((naive - clean).abs().max()):.1e}, 这样右 pad 单测就测不出读错位置的 bug")


if __name__ == "__main__":
    main()
