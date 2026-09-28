#!/usr/bin/env python
"""
KTO vs DPO: 同一个 SFT 起点、同样的 β / lr / 步数、同样的每步前向量 (policy + ref 各 128 条序列)。
DPO 吃偏好对; KTO 吃把偏好对**拆开**后的单条 "好 / 坏" 样本 (每个 prompt 只出现一条, DPO 一对也凑不出)。
再把好坏比例压到 1:9, 看 KTO 在 DPO 根本用不了的数据上还能不能学到偏好方向。

    python -m llm_finetune.run_finetune.kto.train_kto
"""

import copy

import torch

from llm_finetune import DPOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask
from llm_finetune.methods.kto import KTOForward, KTOLoss, UnpairedDataGenerator
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, STEPS, LR, BETA = 100, 200, 3e-4, 0.5             # 与 train_dpo 相同


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    rows = {"SFT 起点": (preference_accuracy(sft, task), em_sft, None)}

    runs = [
        # 名字, 包装, 数据, loss。DPO: 64 对 = 128 条; KTO: 64 条样本 + 64 条错配 KL 样本 = 128 条
        ("DPO 成对", PairwiseForward, lambda: PreferenceDataGenerator(task, 64), DPOLoss(BETA)),
        ("KTO 拆开 好:坏=1:1", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.5), KTOLoss(BETA)),
        ("KTO 好:坏=1:9 λ_U=1/9", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.1), KTOLoss(BETA, lambda_u=1 / 9)),
        ("KTO 好:坏=1:9 不调 λ", KTOForward, lambda: UnpairedDataGenerator(task, 64, 0.1), KTOLoss(BETA)),
    ]
    for name, wrapper, data, loss in runs:
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
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
        assert m["accuracy"] > before["accuracy"] + 0.02, f"{name}: 留出集偏好准确率应提升"
        assert gap(m) > gap(before) + 2, f"{name}: 没有成对数据也应把 chosen / rejected 拉开"
    kto, em_kto, _ = rows["KTO 拆开 好:坏=1:1"]
    # 好样本有自己的项 (把 r 往 z0 之上推), 不像 DPO 只看差值 —— chosen 的 log-prob 不掉, 生成质量也不掉
    assert kto["logp_chosen"] > dpo["logp_chosen"] and em_kto > em_dpo + 0.1
    # 1:9 且不调 λ: 坏样本的梯度压倒一切, 模型把所有回复一起往下压 → 偏好方向丢了, 生成崩了
    bad, em_bad, _ = rows["KTO 好:坏=1:9 不调 λ"]
    assert bad["logp_chosen"] < before["logp_chosen"] - 10 and bad["accuracy"] < before["accuracy"] and em_bad < 0.05


if __name__ == "__main__":
    main()
