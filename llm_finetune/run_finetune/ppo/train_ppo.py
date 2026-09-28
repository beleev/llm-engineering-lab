#!/usr/bin/env python
"""
PPO (带 critic) vs GRPO: 同一个 SFT 起点、同样的 verifier 奖励、同样的**采样预算** (每步 256 条回复) 和步数。
GRPO = 32 题 × 8 条; PPO = 256 题 × 1 条。再加一个对照 "PPO, critic 冻结" (V ≈ 0, 不学), 把 critic 自己的贡献单独量出来。

    python -m llm_finetune.run_finetune.ppo.train_ppo

实验设计:
    先跑 check_gae(): 在随机数上验 GAE 的两个端点 (λ=0 和 λ=1), 不涉及模型。
    起点: sort 任务上 SFT 150 步的 policy。三臂都从它的 deepcopy 出发, 每臂开始前重置种子, 60 步, lr=3e-4:
        grpo                 32 题 × 8 条, baseline = 同题 8 条的均值       ← 对照组
        ppo                  256 题 × 1 条, baseline = 学出来的 critic
        ppo_frozen_critic    同 ppo, 但 critic 的 lr = 0                    ← 第二个对照
    critic 冻结时停在初始状态 (value head 小初始化, V ≈ 0), 给不出按前缀区分的 baseline。
    这时剩下的只有批内白化减掉的那个常数。
    ppo 对 grpo 的差距里, 多少来自 critic、多少来自 "一题一采", 靠第三臂分开。
    常驻参数 = 训练时要同时放在内存里的模型: grpo 只有 policy; ppo 有 policy + ref + critic。
断言 (每条验证一个结论):
    1. GAE: λ=0 时 A 等于 TD(0) 误差; λ=1 时回报目标等于 Monte-Carlo 回报, A = 回报 − V。
    2. ppo 的训练奖励: 后 10 步的平均比前 10 步高 0.1 以上。
    3. ppo 的留出集 pass@1 比 SFT 起点高 0.1 以上。
    4. ppo 要训的参数 > grpo 的 1.9 倍: critic 与 policy 同尺寸。
    5. ppo 的 pass@1 比 grpo 高 0.03 以上。
    6. ppo 与 ppo_frozen_critic 的 pass@1 相差 < 0.05: 本规模下 critic 学没学, 结果在噪声内。
       结合第 5 条: ppo 对 grpo 的优势来自 "一题一采" 的采样方式 (256 道不同的题), 不来自 critic。
    7. baseline 解释的方差: ppo 的 V(s_0) 低于 grpo 的组均值。本规模下学出来的 critic 不如免费的组均值。
    第 5–7 条是本规模下的观察, 换模型尺寸或步数可能不成立。
依赖 llm_models: critic 是 RewardModel(LLaMA), 靠 `return_hidden=True` 取隐状态; 采样靠 `LLaMA.generate`。
"""

import copy
import time

import torch

from llm_finetune import (
    GRPOTrainer, PPOConfig, PPOTrainer, PromptDataGenerator, RewardModel, SeqTask, gae, group_advantages, make_config,
    token_values,
)
from llm_finetune.run_finetune.common import make_model, sft_warmup
from llm_finetune.utils.param_utils import count_parameters

SFT_STEPS, RL_STEPS, SAMPLES_PER_STEP, G, LR = 150, 60, 256, 8, 3e-4
WINDOW = 10                                               # 训练奖励取前 / 后多少步的平均


def check_gae() -> None:
    """λ=0 → TD(0): r_t + γV(s_{t+1}) − V(s_t);  λ=1 → 回报目标 = Monte-Carlo 回报 Σ γ^k r_{t+k}。长度不等, 带 mask。"""
    torch.manual_seed(0)
    r, v, gamma = torch.randn(4, 7), torch.randn(4, 7), 0.9               # 随机的奖励 / 估值 [4, 7]
    # [7] 与 [4, 1] 广播成 [4, 7]: 四条回复的长度分别是 7 / 5 / 3 / 1
    mask = (torch.arange(7) < torch.tensor([[7], [5], [3], [1]])).float()
    r, v = r * mask, v * mask                                             # mask 外 r = V = 0 (终止)
    next_v = torch.cat([v[:, 1:], torch.zeros(4, 1)], dim=1)              # [4, 7] V 左移一位, 末列补 0 (终止态)
    td0 = (r + gamma * next_v - v) * mask                                 # [4, 7] 手算的 TD(0) 误差
    # [4, 7] 手算的 Monte-Carlo 回报: 第 t 列 = Σ_k γ^k · r_{t+k}
    mc = torch.stack([sum(gamma ** k * r[:, t + k] for k in range(7 - t)) for t in range(7)], dim=1)
    a0, _ = gae(r, v, mask, gamma, 0.0)
    a1, g1 = gae(r, v, mask, gamma, 1.0)
    print(f"GAE 两端: λ=0 的 A 与 TD(0) 误差 {float((a0 - td0).abs().max()):.1e} | "
          f"λ=1 的回报目标与 Monte-Carlo 回报 误差 {float((g1 - mc).abs().max()):.1e}")
    assert torch.allclose(a0, td0, atol=1e-6), (
        f"λ=0 应退化成 TD(0): A 与 r_t + γV(s_{{t+1}}) − V(s_t) 的最大差 {float((a0 - td0).abs().max()):.1e}")
    assert torch.allclose(g1, mc, atol=1e-5), (
        f"λ=1 应退化成 MC 回报: 回报目标与 Σ γ^k r 的最大差 {float((g1 - mc).abs().max()):.1e}")
    assert torch.allclose(a1, (mc - v) * mask, atol=1e-5), "λ=1 时 A 应等于 MC 回报 − V(s_t)"


@torch.no_grad()
def baseline_explained_var(policy, task, critic=None) -> float:
    """留出集整条回复的 0/1 奖励 R, baseline 解释掉的方差 1 − Var(R − b)/Var(R)。
    GRPO: b = 同题 8 条的均值;  PPO: b = V(s_0) (critic 只看 prompt 时的估值)。"""
    torch.manual_seed(7)
    prompts = task.sample_prompts(64, "test").repeat_interleave(G, dim=0)          # [512, P]
    seqs = policy.generate(prompts, task.response_len)                             # [512, P+R], 默认 temperature=1
    R = task.verify(prompts, seqs[:, task.prompt_len:])                            # [512] 每条回复的 0/1 奖励
    if critic is None:
        resid = group_advantages(R, G, std_norm=False)                             # R − 组均值
    else:
        # token_values → [512, R], 第 0 列是 V(s_0): 只看完 prompt、还没生成任何 token 时的估值
        resid = R - token_values(critic, seqs, task.prompt_len)[:, 0]
    return float(1 - resid.var() / R.var())


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    pass1_sft = task.exact_match(sft, n=512, temperature=1.0)
    check_gae()

    print(f"\n---- 训练: 每步 {SAMPLES_PER_STEP} 条回复, {RL_STEPS} 步; SFT 起点留出集 贪心 EM {em_sft:.3f}, 采样 pass@1 {pass1_sft:.3f} ----")
    results = {}
    for name in ("grpo", "ppo", "ppo_frozen_critic"):
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        critic = None
        if name == "grpo":
            trainer = GRPOTrainer(policy, task.verify, make_config("grpo", lr=LR, max_new=task.response_len + 2, group_size=G))
            prompts = PromptDataGenerator(task, SAMPLES_PER_STEP // G)
            resident = [policy]
        else:
            critic = RewardModel(copy.deepcopy(sft))                      # 与 policy 同尺寸, 从 SFT 权重起步
            # lr = 0: 优化器照常 step, 但更新量是 0, critic 停在初始状态
            critic_lr = 0.0 if name == "ppo_frozen_critic" else PPOConfig.critic_lr
            trainer = PPOTrainer(policy, critic, task.verify,
                                 PPOConfig(lr=LR, max_new=task.response_len + 2, critic_lr=critic_lr))
            prompts = PromptDataGenerator(task, SAMPLES_PER_STEP)
            resident = [policy, trainer.ref, critic]
        t0 = time.perf_counter()
        hist = [trainer.step(prompts.generate_batch()["prompts"]) for _ in range(RL_STEPS)]
        dt = time.perf_counter() - t0
        # 前 WINDOW 步 / 后 WINDOW 步的平均训练奖励: 单步的奖励抖动大, 取平均再比
        first, last = (sum(h["reward"] for h in hist[s]) / WINDOW for s in (slice(0, WINDOW), slice(-WINDOW, None)))
        results[name] = dict(em=task.exact_match(policy), pass1=task.exact_match(policy, n=512, temperature=1.0),
                             first=first, last=last, sec=dt, ev=baseline_explained_var(policy, task, critic),
                             params=sum(count_parameters(m)["total"] for m in resident),
                             trainable=sum(count_parameters(m)["trainable"] for m in resident))

    print(f"{'':<18}{'贪心EM':>7}{'pass@1':>8}{f'训练奖励 前{WINDOW}步→后{WINDOW}步':>20}{'baseline 解释的方差':>16}"
          f"{'常驻参数':>9}{'要训的参数':>9}{'耗时':>7}")
    for k, r in results.items():
        print(f"{k:<18}{r['em']:>9.3f}{r['pass1']:>8.3f}{r['first']:>14.3f} → {r['last']:.3f}{r['ev']:>18.3f}"
              f"{r['params']:>15,}{r['trainable']:>12,}{r['sec']:>7.1f}s")

    grpo, ppo, frozen = results["grpo"], results["ppo"], results["ppo_frozen_critic"]
    assert ppo["last"] > ppo["first"] + 0.1, (
        f"PPO 训练奖励应上升 0.1 以上: 前 {WINDOW} 步平均 {ppo['first']:.3f} → 后 {WINDOW} 步平均 {ppo['last']:.3f}")
    assert ppo["pass1"] > pass1_sft + 0.1, f"PPO 留出集 pass@1 应高于 SFT 起点 ({ppo['pass1']:.3f} vs {pass1_sft:.3f})"
    assert ppo["trainable"] > 1.9 * grpo["trainable"], (
        f"critic 与 policy 同尺寸: 要训的参数翻倍, 实际 ppo {ppo['trainable']:,} vs grpo {grpo['trainable']:,}")
    assert ppo["pass1"] > grpo["pass1"] + 0.03, (
        f"同样 {SAMPLES_PER_STEP} 条/步, 本配置下 PPO 的 pass@1 应比 GRPO 高 0.03 以上 (下一条断言说明: critic 冻结的对照一样高, "
        f"差别来自 '一题一采' 的采样方式): ppo {ppo['pass1']:.3f}, grpo {grpo['pass1']:.3f}")
    assert abs(ppo["pass1"] - frozen["pass1"]) < 0.05, (
        f"本规模下 critic 学没学, pass@1 应在噪声内 (差 < 0.05): "
        f"ppo {ppo['pass1']:.3f}, critic 冻结 {frozen['pass1']:.3f}")
    assert ppo["ev"] < grpo["ev"], (
        f"本规模下学出来的 V(s_0) 不如免费的组均值: baseline 解释的方差 ppo {ppo['ev']:.3f}, grpo {grpo['ev']:.3f}")


if __name__ == "__main__":
    main()
