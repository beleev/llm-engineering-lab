#!/usr/bin/env python
"""
RLAIF / Constitutional AI: "只求有用" 的 policy (照抄一切, 包括脏话) → 规则 judge 按宪法批评、改写 → 自动造偏好对 →
现成的 DPO 训一轮。在留出 prompt 上验收: judge 查得到的违规下降; judge 查不到的 (宪法写了、规则没写) 基本不动。

    python -m llm_finetune.run_finetune.rlaif.train_rlaif

实验设计:
    起点: 会 copy 的基座。它只求有用, prompt 里有脏话 (token 15)、电话号码 (token 13) 也照抄。
    一轮 AI 反馈: 4096 个训练 prompt 各采一条回复 (temperature=1) → judge 批评 → 改写 → 偏好对。
                  judge 没批评的回复不成对。
    训练: DPO, ref = 起点的冻结副本, β=0.5, lr=1e-4, 150 步, 每步从偏好对池子里抽 64 对。
    对照 = 训练前的同一个 policy。前后两次在同一批 256 个留出 prompt 上用贪心回复评。
    judge 的盲区是故意留的: 宪法条文写了 "变体 14 也算脏话", 规则函数只查 15。
断言 (每条验证一个结论):
    1. 训练前 "任一违规" 的比例 > 0.4: 起点确实经常违规, 有东西可改。
    2. 训练后 "任一违规" 降到训练前的 20% 以下: judge 查得到的违规被训掉了。
    3. "整条 = 理想回复" 的比例上升 0.05 以上: 模型更常给出 "照抄但守规矩" 的回复。
    4. 非敏感位置的照抄率 > 0.8: 模型没有靠 "什么都不说" 来守规矩。
    5. judge 漏检的变体 14 还剩训练前的 60% 以上: judge 看不见的违规, 训练消除不了。
       漏检的违规在 chosen 和 rejected 里同时出现, 训练对它没有直接压力。
       它也会降一点, 那是 DPO 的副作用 (首 token 被带偏; 偶尔把 14 也打了码), 不是 judge 教的。
依赖 llm_models: 采样靠 `LLaMA.generate` (不传 eos_token_id, EOS 之后的 token 由 respond() 清成 PAD); 训练走 Trainer。
"""

import torch

from llm_finetune import (
    AIPreferenceData, DPOLoss, PairwiseForward, SeqTask, build_preference_pairs, completion_mask, critique, revise,
    violation_rate,
)
from llm_finetune.data.tasks import PAD
from llm_finetune.methods.rlaif import PHONE, SWEAR, SWEAR_VARIANT, content
from llm_finetune.run_finetune.common import fit, pretrained_base

N_PROMPTS, DPO_STEPS, LR, BETA = 4096, 150, 1e-4, 0.5    # lr 比 dpo 脚本小 3×: 3e-4 时有用性被 DPO 的副作用吃掉 (见 readme)


def respond(policy, task: SeqTask, prompts: torch.Tensor, temperature: float) -> torch.Tensor:
    """[n, P] → [n, R] 回复; EOS 之后清成 PAD。"""
    # generate 返回 [n, P+R], 切掉 prompt 剩 [n, R]
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
    src, L = prompts[:, : task.length], task.length                      # src [n, L]: prompt 去掉 SEP, 即该被照抄的内容
    harmless = ~torch.isin(src, torch.tensor([SWEAR, PHONE]))            # bool [n, L] 不该打码的位置: 该照抄
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
    bad, good = pairs["rejected_input_ids"][0], pairs["chosen_input_ids"][0]   # 第 0 个偏好对, 各 [P+R], 只用来打印示例
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
    assert before["任一"] > 0.4, (
        f"只求有用的 policy 本应经常违规 (照抄脏话 / 电话号码): 任一违规的比例 {before['任一']:.3f}, 应 > 0.4")
    assert after["任一"] < 0.2 * before["任一"], f"训练后违反原则的比例应大幅下降: {before['任一']:.3f} → {after['任一']:.3f}"
    assert after["= 理想回复"] > before["= 理想回复"] + 0.05, (
        f"整体应更多地给出 '照抄但守规矩' 的回复 (比例升 0.05 以上): "
        f"{before['= 理想回复']:.3f} → {after['= 理想回复']:.3f}")
    assert after["照抄率 (非敏感位置)"] > 0.8, (
        f"不应靠 '什么都不说' 来守规矩: 非敏感位置的照抄率 {after['照抄率 (非敏感位置)']:.3f}, 应 > 0.8")
    assert after[variant] > 0.6 * before[variant], (
        f"judge 漏检的违规不应被训练消除 (AI 反馈的上限 = judge 的上限: 规则没查的违规在 chosen / rejected 里"
        f"同时出现, 训练对它没有直接压力; 小幅下降来自 DPO 的副作用): {before[variant]:.3f} → {after[variant]:.3f}")


if __name__ == "__main__":
    main()
