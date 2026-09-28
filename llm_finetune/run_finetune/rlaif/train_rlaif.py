#!/usr/bin/env python
"""
RLAIF / Constitutional AI: "只求有用" 的 policy (照抄一切, 包括脏话) → 规则 judge 按宪法批评、改写 → 自动造偏好对 →
现成的 DPO 训一轮。在留出 prompt 上验收: judge 查得到的违规下降; judge 查不到的 (宪法写了、规则没写) 基本不动。

    python -m llm_finetune.run_finetune.rlaif.train_rlaif
"""

import torch

from llm_finetune import DPOLoss, PairwiseForward, SeqTask
from llm_finetune.data.tasks import PAD, completion_mask
from llm_finetune.methods.rlaif import (
    PHONE, SWEAR, SWEAR_VARIANT, AIPreferenceData, build_preference_pairs, content, critique, revise, violation_rate,
)
from llm_finetune.run_finetune.common import fit, pretrained_base

N_PROMPTS, DPO_STEPS, LR, BETA = 4096, 150, 1e-4, 0.5    # lr 比 dpo 脚本小 3×: 3e-4 时有用性被 DPO 的副作用吃掉 (见 readme)


def respond(policy, task: SeqTask, prompts: torch.Tensor, temperature: float) -> torch.Tensor:
    """[n, P] → [n, R] 回复; EOS 之后清成 PAD。"""
    out = policy.generate(prompts, task.response_len, temperature=temperature)[:, task.prompt_len:]
    return out.masked_fill(~completion_mask(out), PAD)


def evaluate(policy, task: SeqTask) -> dict:
    """留出 prompt 上贪心回复: judge 的违规率 + judge 看不见的变体率 + 有用性 (非敏感位置照抄率、整条 = 理想回复)。"""
    state = torch.get_rng_state()
    torch.manual_seed(1234)
    prompts = task.sample_prompts(256, "test")
    torch.set_rng_state(state)
    out = respond(policy, task, prompts, temperature=0)
    ideal = torch.stack([revise(y) for y in task.target(prompts)])       # 有用 (照抄) + 守规矩
    stats = violation_rate(out)
    stats[f"变体 {SWEAR_VARIANT} (judge 漏检)"] = sum(SWEAR_VARIANT in content(y) for y in out) / len(out)
    src, L = prompts[:, : task.length], task.length
    harmless = ~torch.isin(src, torch.tensor([SWEAR, PHONE]))            # 不该打码的位置: 该照抄
    stats["照抄率 (非敏感位置)"] = float((out[:, :L] == src)[harmless].float().mean())
    stats["= 理想回复"] = float((out == ideal).all(dim=1).float().mean())
    return stats


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("copy")
    policy = pretrained_base()                      # 只求有用的 SFT 模型: 什么都照抄, 包括脏话和电话号码
    before = evaluate(policy, task)

    # ---- 一轮 AI 反馈: 采样 → 批评 → 改写 → 偏好对 ----
    prompts = task.sample_prompts(N_PROMPTS, "train")
    responses = respond(policy, task, prompts, temperature=1.0)
    pairs = build_preference_pairs(prompts, responses)
    n_pairs = len(pairs["chosen_input_ids"])
    P = task.prompt_len
    bad, good = pairs["rejected_input_ids"][0], pairs["chosen_input_ids"][0]
    print(f"{N_PROMPTS} 条采样里 judge 批评了 {n_pairs} 条 → {n_pairs} 个偏好对 (没问题的回复不成对)。示例:")
    print(f"  prompt   {bad[:P - 1].tolist()}")
    print(f"  回复     {content(bad[P:])}  ← {critique(bad[P:])}")
    print(f"  改写后   {content(good[P:])}")

    fit(PairwiseForward.with_frozen_copy(policy), AIPreferenceData(pairs), DPOLoss(BETA), DPO_STEPS, LR,
        log_interval=100)
    after = evaluate(policy, task)

    print(f"\n{'留出集 (贪心)':<22}{'DPO 前':>8}{'DPO 后':>8}")
    for k in before:
        print(f"{k:<22}{before[k]:>8.3f}{after[k]:>8.3f}")

    variant = f"变体 {SWEAR_VARIANT} (judge 漏检)"
    assert before["任一"] > 0.4, "只求有用的 policy 本应经常违规 (照抄脏话 / 电话号码)"
    assert after["任一"] < 0.2 * before["任一"], f"训练后违反原则的比例应大幅下降: {before['任一']:.3f} → {after['任一']:.3f}"
    assert after["= 理想回复"] > before["= 理想回复"] + 0.05, "整体应更多地给出 '照抄但守规矩' 的回复"
    assert after["照抄率 (非敏感位置)"] > 0.8, "不应靠 '什么都不说' 来守规矩"
    # AI 反馈的上限 = judge 的上限: 规则没查的违规在 chosen / rejected 里同时出现, 训练对它没有直接压力。
    # 它也会降一点 —— 来自 DPO 的副作用 (首 token 被带偏; 偶尔把 14 也打了码), 不是 judge 教的
    assert after[variant] > 0.6 * before[variant], "judge 漏检的违规不应被训练消除"


if __name__ == "__main__":
    main()
