#!/usr/bin/env python
"""
DPO: SFT 过的 policy + 偏好对 (正确回复 vs 损坏回复)。用通用 Trainer 训练, 在**留出**偏好对上验收;
同时展示 DPO 的著名副作用: 差值变大了, 但 chosen 自己的概率也掉了。

    python -m llm_finetune.run_finetune.dpo.train_dpo

实验设计:
    起点: sort 任务上 SFT 100 步的 policy。故意只热身到 "半会", 给偏好训练留出提升空间。
    训练: ref = 起点的冻结副本; 偏好对每步新采; β=0.5, lr=3e-4, 200 步。
    对照 = 训练前的同一个 policy (表里 "SFT 后" 那一行)。前后两次在同一批留出偏好对上评。
    四个指标: 偏好准确率、log π(chosen)、log π(rejected)、贪心 exact-match。
断言 (每条验证一个结论):
    1. 第 1 步 loss = ln 2 (误差 < 1e-3)。此时 policy 与 ref 权重相同, 两个 log-ratio 都是 0; Trainer 第 1 步 lr = 0。
    2. 留出集偏好准确率不下降, 且 > 0.97。
    3. log π(chosen) − log π(rejected) 比训练前多拉开 2 nat 以上。
    4. log π(chosen) 本身下降。DPO 的 loss 只看差值: 两边一起降、rejected 降得更多, 也算 "优化成功"
       (likelihood displacement)。这一条是本配置下的观察。
依赖 llm_models: Trainer 第 1 步 lr = 0, 第 1 条断言靠它。
"""

import torch

from llm_finetune import DPOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, DPO_STEPS, LR, BETA = 100, 200, 3e-4, 0.5


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    policy = make_model(task)
    em_sft = sft_warmup(policy, task, SFT_STEPS)              # 故意只热身到 "半会": 留出提升空间
    before = preference_accuracy(policy, task)

    # ref = 此刻 policy 的冻结副本; PairwiseForward 把 policy、ref 各 1 次前向 (每次 2B 条序列) 包成一个 Module 交给通用 Trainer
    hist = fit(PairwiseForward.with_frozen_copy(policy), PreferenceDataGenerator(task), DPOLoss(BETA),
               DPO_STEPS, LR, log_interval=50)
    after, em_dpo = preference_accuracy(policy, task), task.exact_match(policy)

    print(f"第 1 步 loss {hist[0]['total_loss']:.4f} (policy = ref ⇒ 恰为 ln 2 = 0.6931)")
    print(f"{'留出集':<10}{'偏好准确率':>10}{'log π(chosen)':>16}{'log π(rejected)':>18}{'exact-match':>14}")
    print(f"{'SFT 后':<10}{before['accuracy']:>14.3f}{before['logp_chosen']:>16.2f}{before['logp_rejected']:>18.2f}{em_sft:>14.3f}")
    print(f"{'DPO 后':<10}{after['accuracy']:>14.3f}{after['logp_chosen']:>16.2f}{after['logp_rejected']:>18.2f}{em_dpo:>14.3f}")

    gap = lambda m: m["logp_chosen"] - m["logp_rejected"]     # 两种回复的平均 log-prob 差, 单位 nat
    assert abs(hist[0]["total_loss"] - 0.6931) < 1e-3, (
        f"第 1 步 loss 应恰为 ln 2 = 0.6931 (policy = ref, 两个 log-ratio 都是 0), 实际 {hist[0]['total_loss']:.4f}")
    assert after["accuracy"] >= before["accuracy"], (
        f"留出集偏好准确率不应下降: {before['accuracy']:.3f} → {after['accuracy']:.3f}")
    assert after["accuracy"] > 0.97, f"留出集偏好准确率应提升到 > 0.97, 实际 {after['accuracy']:.3f}"
    assert gap(after) > gap(before) + 2, (
        f"chosen 与 rejected 的 log-prob 差应被拉开 2 nat 以上: {gap(before):.2f} → {gap(after):.2f}")
    assert after["logp_chosen"] < before["logp_chosen"], (
        f"本配置下应能观察到 chosen 的 log-prob 下降 (DPO 的 loss 只看差值: 两边一起降、rejected 降得更多, "
        f"也算 '优化成功', 即 likelihood displacement): {before['logp_chosen']:.2f} → {after['logp_chosen']:.2f}")


if __name__ == "__main__":
    main()
