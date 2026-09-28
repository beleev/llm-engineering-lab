#!/usr/bin/env python
"""
KTO vs DPO: 同一个 SFT 起点、同样的 β / lr / 步数、同样的每步前向量 (policy + ref 各 128 条序列)。
DPO 吃偏好对; KTO 吃把偏好对**拆开**后的单条 "好 / 坏" 样本 (每个 prompt 只出现一条, DPO 一对也凑不出)。
再把好坏比例压到 1:9, 看 KTO 在 DPO 根本用不了的数据上还能不能学到偏好方向。

    python -m llm_finetune.run_finetune.kto.train_kto

实验设计 (同一个 SFT 起点的 4 个副本, 每臂开始前重置种子, 都训 200 步):
    DPO 成对                 64 对 = 128 条序列                         ← 对照组
    KTO 拆开 好:坏=1:1       64 条样本 + 64 条错配样本 = 128 条
    KTO 好:坏=1:9 λ_U=1/9    坏样本多到 9 倍, 坏样本的权重降到 1/9
    KTO 好:坏=1:9 不调 λ     坏样本多到 9 倍, 权重不动                   ← 反面对照
    第三臂里 λ_D·n_D / (λ_U·n_U) = 1, 落在论文建议的 [1, 4/3] 里。
    验收全在留出偏好对上 (common.preference_accuracy), 另看贪心 EM。
断言 (每条验证一个结论):
    1. KTO 1:1 和调过 λ 的 KTO 1:9: 偏好准确率比 SFT 起点高 0.02 以上, 两种回复的 log-prob 差多拉开 2 nat 以上。
       没有成对数据, 也学得到偏好方向。
    2. KTO 1:1 对比 DPO: log π(chosen) 更高, 贪心 EM 高出 0.1 以上。
       KTO 的好样本有自己的一项 (把 r 往 z0 之上推), DPO 只看差值。
    3. KTO 1:9 不调 λ: log π(chosen) 比起点低 10 nat 以上, 偏好准确率低于起点, 贪心 EM < 0.05。
       坏样本的梯度压倒了好样本, 模型把所有回复一起往下压。
依赖 llm_models: fit 返回的 hist 里带着 KTOLoss 输出的 "z0" / "kl_estimate" (Trainer 把 loss 返回的每个键都记下来)。
"""

import copy

import torch

from llm_finetune import (
    DPOLoss, KTOForward, KTOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask, UnpairedDataGenerator,
)
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, STEPS, LR, BETA = 100, 200, 3e-4, 0.5             # 与 train_dpo 相同


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    rows = {"SFT 起点": (preference_accuracy(sft, task), em_sft, None)}   # 名字 → (留出集偏好指标, 贪心 EM, 训练日志)

    runs = [
        # 名字, 包装, 数据, loss。DPO: 64 对 = 128 条; KTO: 64 条样本 + 64 条错配 KL 样本 = 128 条
        ("DPO 成对", PairwiseForward, lambda: PreferenceDataGenerator(task, 64), DPOLoss(BETA)),
        ("KTO 拆开 好:坏=1:1", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.5), KTOLoss(BETA)),
        ("KTO 好:坏=1:9 λ_U=1/9", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.1), KTOLoss(BETA, lambda_u=1 / 9)),
        ("KTO 好:坏=1:9 不调 λ", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.1), KTOLoss(BETA)),
    ]
    for name, wrapper, data, loss in runs:
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)                           # 每臂都从同一个 SFT 起点出发, 互不影响
        # with_frozen_copy: ref = 此刻 policy 的冻结副本
        hist = fit(wrapper.with_frozen_copy(policy), data(), loss, STEPS, LR, log_interval=STEPS)
        rows[name] = (preference_accuracy(policy, task), task.exact_match(policy), hist)

    print(f"\n{'留出集':<22}{'偏好准确率':>8}{'log π(chosen)':>15}{'log π(rejected)':>17}{'差值':>7}{'贪心EM':>9}")
    gap = lambda m: m["logp_chosen"] - m["logp_rejected"]
    for name, (m, em, _) in rows.items():
        print(f"{name:<24}{m['accuracy']:>9.3f}{m['logp_chosen']:>15.2f}{m['logp_rejected']:>17.2f}{gap(m):>8.2f}{em:>9.3f}")
    hist = rows["KTO 拆开 好:坏=1:1"][2]
    print(f"KTO 错配样本上的平均 log π/π_ref: 第 1 步 {hist[0]['kl_estimate']:.3f} → 最后一步 {hist[-1]['kl_estimate']:.3f}; "
          f"clamp 后的 z0 最大 {max(h['z0'] for h in hist):.3f}")

    before, (dpo, em_dpo, _) = rows["SFT 起点"][0], rows["DPO 成对"]
    for name in ("KTO 拆开 好:坏=1:1", "KTO 好:坏=1:9 λ_U=1/9"):
        m = rows[name][0]
        assert m["accuracy"] > before["accuracy"] + 0.02, (
            f"{name}: 留出集偏好准确率应比 SFT 起点高 0.02 以上: {before['accuracy']:.3f} → {m['accuracy']:.3f}")
        assert gap(m) > gap(before) + 2, (
            f"{name}: 没有成对数据也应把 chosen / rejected 拉开 2 nat 以上: {gap(before):.2f} → {gap(m):.2f}")
    kto, em_kto, _ = rows["KTO 拆开 好:坏=1:1"]
    assert kto["logp_chosen"] > dpo["logp_chosen"], (
        f"KTO 的好样本有自己的项 (把 r 往 z0 之上推), 不像 DPO 只看差值, chosen 的 log-prob 应更高: "
        f"KTO {kto['logp_chosen']:.2f}, DPO {dpo['logp_chosen']:.2f}")
    assert em_kto > em_dpo + 0.1, (
        f"chosen 的 log-prob 不掉, 生成质量也不该掉: KTO 的贪心 EM 应比 DPO 高 0.1 以上, "
        f"实际 KTO {em_kto:.3f}, DPO {em_dpo:.3f}")
    # 以下三条看反面对照: 1:9 且不调 λ, 坏样本的梯度压倒一切, 模型把所有回复一起往下压
    bad, em_bad, _ = rows["KTO 好:坏=1:9 不调 λ"]
    assert bad["logp_chosen"] < before["logp_chosen"] - 10, (
        f"1:9 且不调 λ: chosen 的 log-prob 应被一起压低 10 nat 以上: "
        f"{before['logp_chosen']:.2f} → {bad['logp_chosen']:.2f}")
    assert bad["accuracy"] < before["accuracy"], (
        f"1:9 且不调 λ: 偏好方向应丢失, 准确率低于 SFT 起点: {before['accuracy']:.3f} → {bad['accuracy']:.3f}")
    assert em_bad < 0.05, f"1:9 且不调 λ: 生成应崩掉, 贪心 EM 应 < 0.05, 实际 {em_bad:.3f}"


if __name__ == "__main__":
    main()
