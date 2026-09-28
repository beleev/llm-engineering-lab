#!/usr/bin/env python
"""
GRPO / DAPO / Dr.GRPO / GSPO: 同一个 SFT 起点、同样的 prompt 流和步数, 奖励 = 任务 verifier (排序全对才得 1 分)。
先用几个确定性的小实验把每个变体改动的那一项单独量出来, 再看训练结果。

    python -m llm_finetune.run_finetune.grpo.train_grpo

实验设计:
    第一部分 diagnostics(): 三个不训练的小实验, 每个只量一个变体改动的那一项。
        (a) ÷std       人造两道题的奖励 (8 条里对 1 条 / 对 4 条), 比 ÷std 前后两题权重之比。
        (b) 聚合方式   一条 3-token 回复 + 一条 9-token 回复, 对 per-token loss 求导, 梯度就是每个 token 的权重。
        (c) 温度       用 T=0.5 采 4000 个首 token, 看经验分布贴近 softmax(z/T) 还是 softmax(z)。
    第二部分 训练: 起点是 sort 任务上 SFT 150 步的 policy (R1 式流水线: 先 SFT 到 "半会", 再 RL)。
        四臂: grpo (对照组) / dapo / dr_grpo / gspo。都从起点的 deepcopy 出发, 每臂开始前重置种子。
        每步 32 个 prompt × G=8 条回复, 共 60 步, lr=3e-4。各臂之间只差 VARIANTS 里那几个开关。
        max_new = R+2: 比正确回复多采 2 个 token。EOS 之后的部分由 completion_mask 裁掉, 不进 loss。
    第三部分 β > 0: 上面四臂都是 β=0 (不建 ref)。再从同一起点跑两臂 grpo, 各 20 步, 只差 β = 0 / 0.5,
        在留出 prompt 的样本上量 KL(policy‖SFT 起点), 看 KL 惩罚有没有把 policy 拴住。
    两个验收指标, 都在留出集上:
        贪心 EM   temperature=0, 模型最有把握的答案对不对。
        pass@1    temperature=1, 按策略采样一次对不对。RL 直接优化的是它。
断言 (每条验证一个结论):
    1. (a) ÷std 让 "难题 / 中等题" 的权重比变大: 标准化把极端题目重新放大。
    2. (b) seq_mean 下短回复的 token 权重是长回复的 3 倍 (9/3); token_mean 和 fixed_len 下是 1 倍。
    3. (c) 经验分布离 softmax(z/T) 更近; T=0.5 和 T=1 算出的 log-prob 不相同; generate() 之后模型仍在 train 状态。
    4. 每个变体: 第 1 个 epoch 的 |ρ−1| 恒为 0, 最后一个 epoch 的 > 0。
    5. 每个变体: 留出集 pass@1 比 SFT 起点高 0.04 以上; 贪心 EM 最多比起点低 0.15。
    6. dapo 进 loss 的样本里没有 A = 0 的; grpo 有。动态采样把 A≡0 的组剔掉了。
    7. gspo 的 log ρ 标准差不到 grpo 的一半: 序列级比率的波动小。
    8. β > 0 时建了 ref, 每个有效步都报了 KL 项且大于 0; 留出集上离 SFT 起点的 KL 不到 β=0 时的一半。
依赖 llm_models: `LLaMA.generate` 采样时用 softmax(z/T), 结束后恢复 train / eval 状态 (第 3 条断言直接测这两点)。
"""

import copy

import torch
import torch.nn.functional as F

from llm_finetune import (
    GRPOTrainer, PromptDataGenerator, SeqTask, aggregate, completion_kl, completion_logprobs, group_advantages,
    make_config,
)
from llm_finetune.run_finetune.common import make_model, sft_warmup

SFT_STEPS, RL_STEPS, PROMPTS_PER_STEP, G, LR = 150, 60, 32, 8, 3e-4
BETA, BETA_STEPS = 0.5, 20                                   # 第三部分: KL 惩罚的系数和短跑步数


def diagnostics(policy, task) -> None:
    print("---- 变体各改了什么 (确定性小实验) ----")
    # (a) ÷std: 8 条里对 1 条 (难题) vs 对 4 条 (中等题), 看每道题拿到的总权重 mean|A|
    hard, medium = torch.tensor([1.] + [0.] * 7), torch.tensor([1.] * 4 + [0.] * 4)   # [8] ×2, 一道题 8 条回复的奖励
    # weight[是否 ÷std] = [难题的 mean|A|, 中等题的 mean|A|]
    weight = {norm: [float(group_advantages(r, 8, norm).abs().mean()) for r in (hard, medium)] for norm in (True, False)}
    print(f"(a) 题目权重 难题/中等题: GRPO (÷std) {weight[True][0] / weight[True][1]:.2f} | "
          f"Dr.GRPO (不÷std) {weight[False][0] / weight[False][1]:.2f}   ← ÷std 把没什么可学的极端题目重新放大")
    assert weight[True][0] / weight[True][1] > weight[False][0] / weight[False][1], (
        f"÷std 应把极端题目 (8 条里只对 1 条) 相对中等题的权重放大: "
        f"÷std 时权重比 {weight[True][0] / weight[True][1]:.2f}, 不 ÷std 时 {weight[False][0] / weight[False][1]:.2f}")

    # (b) 聚合方式: 一条 3-token 回复和一条 9-token 回复, 对 per-token loss 求导 = 每个 token 的权重
    mask = torch.zeros(2, 9)                                  # [2, 9] 两条回复, 最长 9 个 token
    mask[0, :3], mask[1, :] = 1, 1                            # 第 0 条只有 3 个 token, 第 1 条 9 个
    ratio = {}
    for how in ("seq_mean", "token_mean", "fixed_len"):
        # loss 对 per_token 是线性的, 所以 ∂loss/∂per_token[i, t] 就是这个 token 在 loss 里的权重
        per_token = torch.zeros(2, 9, requires_grad=True)
        aggregate(per_token, mask, how).backward()
        ratio[how] = float(per_token.grad[0, 0] / per_token.grad[1, 0])   # 短回复首 token 的权重 / 长回复首 token 的权重
    print(f"(b) 短回复 token 权重 / 长回复 token 权重: GRPO seq_mean {ratio['seq_mean']:.1f} | DAPO token_mean "
          f"{ratio['token_mean']:.1f} | Dr.GRPO fixed_len {ratio['fixed_len']:.1f}   "
          f"← seq_mean 下答错的长回复每 token 受罚只有 1/{ratio['seq_mean']:.0f}")
    assert ratio["seq_mean"] == 3.0, (
        f"seq_mean 下 3-token 回复里每个 token 的权重应是 9-token 回复的 3 倍 (9/3), 实际 {ratio['seq_mean']}")
    assert ratio["token_mean"] == ratio["fixed_len"] == 1.0, (
        f"token_mean / fixed_len 下每个 token 的权重应与回复长度无关 (比值 1): "
        f"token_mean {ratio['token_mean']}, fixed_len {ratio['fixed_len']}")

    # (c) 温度一致性: T=0.5 采样 4000 个首 token, 经验分布应贴合 softmax(z/T) 而不是 softmax(z)
    T, prompt = 0.5, task.sample_prompts(1)                   # prompt [1, P]
    first = policy.generate(prompt.repeat(4000, 1), 1, temperature=T)[:, -1]   # [4000] 同一个 prompt 各采 1 个 token
    freq = torch.bincount(first, minlength=task.vocab_size) / 4000             # [V] 经验分布
    with torch.no_grad():
        z = policy(prompt)[0, -1]                             # [V] prompt 末位置的 logits = 首个回复 token 的分布
    # TV 距离 = ½·Σ|p − q|, 0 表示两个分布相同
    tv = {name: float((freq - F.softmax(z / t, -1)).abs().sum() / 2) for name, t in (("z/T", T), ("z", 1.0))}
    print(f"(c) T={T} 采样的经验分布与 softmax(z/T) 的 TV 距离 {tv['z/T']:.3f}, 与 softmax(z) 的 {tv['z']:.3f}   "
          f"← log-prob 必须带同一个 T, 否则 ρ 的分母不是行为策略")
    assert tv["z/T"] < tv["z"], (
        f"T={T} 采样的经验分布应更贴近 softmax(z/T): "
        f"与 softmax(z/T) 的 TV 距离 {tv['z/T']:.3f}, 与 softmax(z) 的 {tv['z']:.3f}")
    policy.train()
    seqs = policy.generate(prompt.repeat(4, 1), 3, temperature=T)               # [4, P+3]
    assert policy.training, "generate() 结束后应恢复调用前的 train 状态 (否则采样一次就把模型永久留在 eval)"
    with torch.no_grad():
        assert not torch.allclose(completion_logprobs(policy, seqs, prompt.size(1), T),
                                  completion_logprobs(policy, seqs, prompt.size(1), 1.0)), (
            "completion_logprobs 没有用上 temperature: T=0.5 与 T=1 算出的 log-prob 不应相同")


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)                 # R1 式流水线: 先 SFT 到 "半会", 再 RL
    pass1_sft = task.exact_match(sft, n=512, temperature=1.0)  # 按策略采样的留出集正确率: RL 直接优化的就是它
    diagnostics(sft, task)

    print(f"\n---- 训练: 每步 {PROMPTS_PER_STEP} 个 prompt × G={G}, 共 {RL_STEPS} 步; SFT 起点留出集: 贪心 EM {em_sft:.3f}, 采样 pass@1 {pass1_sft:.3f} ----")
    results = {}
    for variant in ("grpo", "dapo", "dr_grpo", "gspo"):
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        trainer = GRPOTrainer(policy, task.verify, make_config(variant, lr=LR, max_new=task.response_len + 2, group_size=G))
        prompts = PromptDataGenerator(task, PROMPTS_PER_STEP)
        hist = [trainer.step(prompts.generate_batch()["prompts"]) for _ in range(RL_STEPS)]
        hist = [h for h in hist if "skipped" not in h]        # 整批都没有可学的组时, step() 不更新并标 "skipped"
        avg = lambda k: sum(h[k] for h in hist) / len(hist)   # 某个指标在全部有效步上的平均
        results[variant] = dict(em=task.exact_match(policy), pass1=task.exact_match(policy, n=512, temperature=1.0), reward_first=hist[0]["reward"], reward_last=hist[-1]["reward"],
                                **{k: avg(k) for k in ("zero_adv_frac", "used_zero_adv_frac", "clip_frac",
                                                       "ratio_dev_epoch1", "ratio_dev_last", "log_ratio_std", "length")})

    print(f"{'':<9}{'贪心EM':>8}{'pass@1':>8}{'训练奖励 首→末':>16}{'A≡0 的 prompt':>14}{'进 loss 的 A=0':>14}"
          f"{'|ρ−1| ep1':>11}{'|ρ−1| ep2':>11}{'std log ρ':>11}{'裁剪比例':>9}")
    for v, r in results.items():
        print(f"{v:<9}{r['em']:>9.3f}{r['pass1']:>8.3f}{r['reward_first']:>11.3f}→{r['reward_last']:.3f}{r['zero_adv_frac']:>16.1%}"
              f"{r['used_zero_adv_frac']:>16.1%}{r['ratio_dev_epoch1']:>11.3f}{r['ratio_dev_last']:>11.3f}"
              f"{r['log_ratio_std']:>11.4f}{r['clip_frac']:>10.2%}")

    g, dapo, gspo = results["grpo"], results["dapo"], results["gspo"]
    for v, r in results.items():
        assert r["ratio_dev_epoch1"] == 0 and r["ratio_dev_last"] > 0, f"{v}: 第 1 个 epoch ρ≡1, 第 2 个 epoch 起 ρ≠1"
        assert r["pass1"] > pass1_sft + 0.04, f"{v}: 留出集 pass@1 应高于 SFT 起点 ({r['pass1']:.3f} vs {pass1_sft:.3f})"
        assert r["em"] > em_sft - 0.15, f"{v}: 贪心 EM 不应崩 ({r['em']:.3f} vs {em_sft:.3f})"
    assert g["used_zero_adv_frac"] > 0 == dapo["used_zero_adv_frac"], (
        f"动态采样应把 A≡0 的组全部剔除: 进 loss 的样本里 A=0 的比例, "
        f"grpo {g['used_zero_adv_frac']:.1%} (应 > 0), dapo {dapo['used_zero_adv_frac']:.1%} (应 = 0)")
    assert gspo["log_ratio_std"] < 0.5 * g["log_ratio_std"], (
        f"序列级 ρ 的波动应远小于 token 级 (不到一半): "
        f"log ρ 的标准差 gspo {gspo['log_ratio_std']:.4f}, grpo {g['log_ratio_std']:.4f}")

    # ---- 第三部分 β > 0: KL 惩罚把 policy 拴在 SFT 起点附近。两臂只差 β, 放在最后, 不扰动上面各臂的随机流 ----
    drift, kl_term = {}, {}
    for beta in (0.0, BETA):
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        trainer = GRPOTrainer(policy, task.verify, make_config("grpo", lr=LR, max_new=task.response_len + 2,
                                                               group_size=G, beta=beta))
        assert (trainer.ref is not None) == (beta > 0), f"β={beta}: 只有 β > 0 时才该建 ref"
        prompts = PromptDataGenerator(task, PROMPTS_PER_STEP)
        hist = [h for h in (trainer.step(prompts.generate_batch()["prompts"]) for _ in range(BETA_STEPS))
                if "skipped" not in h]
        kl_term[beta] = [h.get("kl") for h in hist]                     # β=0 时 step() 不报 "kl", 这里全是 None
        torch.manual_seed(99)
        seqs = policy.generate(task.sample_prompts(256, "test"), task.response_len + 2, temperature=1.0)   # [256, P+C]
        drift[beta] = completion_kl(policy, sft, seqs, task.prompt_len)   # 留出 prompt 上 KL(policy‖SFT 起点)
    avg_kl = sum(kl_term[BETA]) / len(kl_term[BETA])
    print(f"\n---- β > 0: grpo 再跑 {BETA_STEPS} 步, 只差 β ----")
    print(f"留出集上 KL(policy‖SFT 起点): β=0 {drift[0.0]:.4f} | β={BETA} {drift[BETA]:.4f}; "
          f"β={BETA} 训练中 KL 项的平均 {avg_kl:.4f}")
    assert all(k is None for k in kl_term[0.0]), "β=0 时不应计算 KL 项"
    assert all(k is not None and k > 0 for k in kl_term[BETA]), (
        f"β={BETA} 时每个有效步都应报 KL 项且 > 0 (第 2 个 epoch 起 policy 已离开 ref): {kl_term[BETA]}")
    assert drift[BETA] < 0.5 * drift[0.0], (
        f"KL 惩罚应把 policy 拴在 SFT 起点附近: 留出集 KL(policy‖SFT 起点) "
        f"β={BETA} {drift[BETA]:.4f}, 应不到 β=0 时 {drift[0.0]:.4f} 的一半")


if __name__ == "__main__":
    main()
