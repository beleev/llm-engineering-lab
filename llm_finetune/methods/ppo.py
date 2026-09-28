"""
PPO (带 critic) — InstructGPT 式 RLHF 的 RL 那一步 (Schulman 2017 → Ouyang 2022)

是什么: policy 采样 → 判分 (verifier 分落在末 token, 每个 token 再扣一份对 ref 的 KL) → critic 估每个前缀的 V(s_t)
        → GAE 算**逐 token** 优势 → clipped surrogate 更新 policy, MSE 更新 critic。
解决什么: REINFORCE 的回报方差大, 要一个 baseline; PPO 的 baseline 是一个学出来的 V(s_t), 而且每个 token 各有一个 A_t
          (critic 看得到 "从哪个 token 起答案已经没救了")。代价: 多养一个和 policy 同尺寸的 critic (权重 + 梯度 + Adam)。
核心公式:
    r_t = −β·(log π_old(o_t) − log π_ref(o_t)) + [t = 末 token]·R(x, o)
    δ_t = r_t + γ·V(s_{t+1}) − V(s_t)        A_t = Σ_k (γλ)^k δ_{t+k}        回报目标 G_t = A_t + V(s_t)
        λ = 0 → A_t = δ_t                        TD(0): 只信下一步的 V, 方差小, V 不准时有偏
        λ = 1 → A_t = Σ_k γ^k r_{t+k} − V(s_t)    Monte-Carlo: 只用真实回报, 无偏, 方差大
    L_π = −E_t[ min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t) ]        L_V = ½·E_t[ (V_θ(s_t) − G_t)² ]
与 GRPO 的差别: GRPO 的 baseline 是 "同一题 G 条回复的均值", 整条回复共用一个 A, 必须一题多采;
               PPO 一题采 1 条就够, A 逐 token 不同, 但多一个 critic 要训。
读代码时盯住: `token_values` 的下标 —— V(s_t) 是 "读完 prompt 与 o_<t、还没生成 o_t" 时的估值, 取自位置 P−1+t,
              与 completion_logprobs 同一个错位; 末 token 之后的 V 记 0 (终止态)。
"""

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from llm_finetune.data.tasks import completion_mask
from llm_finetune.methods.grpo import completion_logprobs
from llm_finetune.methods.reward_model import RewardModel
from llm_finetune.utils.param_utils import freeze_module


@dataclass(frozen=True)
class PPOConfig:
    max_new: int = 9                 # C
    lr: float = 3e-4
    critic_lr: float = 1e-3          # critic 从 0 学起 (value head 小初始化), 给大一点
    inner_epochs: int = 2            # 同 GRPO 的 μ
    clip: float = 0.2                # ε
    beta: float = 0.02               # 每 token 的 KL 惩罚系数, 写进奖励 (InstructGPT 的做法), 不是写进 loss
    gamma: float = 1.0               # LLM 场景惯例: 不折扣
    lam: float = 0.95                # GAE 的 λ


def token_values(critic: RewardModel, seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """seqs [N, P+C] → V [N, C]: 复用 RewardModel 的主干 + 标量头, 但取**每个**位置的分, 不只是最后一个。"""
    h = critic.backbone(seqs[:, :-1])                                 # [N, P+C-1, D]
    return critic.value_head(h).squeeze(-1)[:, prompt_len - 1:]       # 位置 P−1+t 看到的是 prompt + o_<t


def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
        gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """rewards, values, mask [N, C] (mask 是前缀全 1) → (A, G) [N, C]。mask 外视为终止: r = V = 0。"""
    rewards, values = rewards * mask, values * mask
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])                         # A_{t+1}
    next_v = torch.zeros_like(rewards[:, 0])                          # V(s_{t+1}); 最后一个位置之后是终止态
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = delta + gamma * lam * running
        adv[:, t] = running
        next_v = values[:, t]
    return adv * mask, (adv + values) * mask


class PPOTrainer:
    """policy: LM; critic: RewardModel (通常由 SFT 权重初始化); reward_fn(prompts, completions) → [N]。"""

    def __init__(self, policy: nn.Module, critic: RewardModel,
                 reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 config: PPOConfig = PPOConfig()) -> None:
        self.policy, self.critic, self.reward_fn, self.cfg = policy, critic, reward_fn, config
        self.ref = copy.deepcopy(policy)                              # KL 惩罚要一份冻结的 SFT 起点
        freeze_module(self.ref)
        self.ref.eval()
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)

    def step(self, prompts: torch.Tensor) -> Dict[str, float]:
        cfg, P = self.cfg, prompts.size(1)

        # ---- 1) rollout: 每个 prompt 采 1 条 (不需要一题多采) ----
        seqs = self.policy.generate(prompts, cfg.max_new, temperature=1.0).clone()   # [N, P+C]
        completions = seqs[:, P:]
        mask = completion_mask(completions).float()                                  # [N, C]
        score = self.reward_fn(prompts, completions).float()                         # [N]

        # ---- 2) 冻结行为策略 / ref / critic 的读数, 拼逐 token 奖励, 算 GAE ----
        with torch.no_grad():
            old_logp = completion_logprobs(self.policy, seqs, P)                     # [N, C]
            ref_logp = completion_logprobs(self.ref, seqs, P)
            old_values = token_values(self.critic, seqs, P)                          # [N, C]
        kl = (old_logp - ref_logp) * mask
        rewards = -cfg.beta * kl
        last = mask.sum(1).long() - 1                                                # [N] 末 token 下标
        rewards[torch.arange(len(seqs)), last] += score                              # verifier 分只落在末 token
        adv, returns = gae(rewards, old_values, mask, cfg.gamma, cfg.lam)
        valid = mask > 0
        A = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-8)                    # 批内白化: 只改尺度, 不改方向

        metrics = {
            "reward": float(score.mean()),
            "kl": float(kl.sum(1).mean()),
            # critic 解释了多少回报方差 (1 = 完美 baseline, 0 = 和常数一样, <0 = 帮倒忙)
            "explained_var": float(1 - (returns - old_values)[valid].var() / returns[valid].var()),
        }

        # ---- 3) 同一批 rollout 上更新 μ 次: policy 与 critic 各有自己的 loss 和优化器 ----
        self.policy.train()
        self.critic.train()
        n_tok = mask.sum()
        for _ in range(cfg.inner_epochs):
            logp = completion_logprobs(self.policy, seqs, P)
            ratio = (logp - old_logp).exp()
            surrogate = torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
            policy_loss = -(surrogate * mask).sum() / n_tok
            value_loss = 0.5 * (((token_values(self.critic, seqs, P) - returns) ** 2) * mask).sum() / n_tok

            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (policy_loss + value_loss).backward()                                    # 两个网络不共享参数, 梯度互不干扰
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer.step()
            self.critic_optimizer.step()
        metrics["value_loss"] = float(value_loss.detach())
        return metrics
