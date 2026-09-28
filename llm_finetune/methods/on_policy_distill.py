"""
On-policy 蒸馏 (reverse KL) — GKD (Agarwal et al., 2024) / MiniLLM (Gu et al., 2024); Qwen3、Thinking Machines 2025 的配方

是什么: student **自己采样**回复, teacher 在这些回复的每个 token 位置上给出分布, student 逐 token 最小化 KL(student ‖ teacher)。
解决什么: off-policy 蒸馏 (distill.py) 只在 teacher / 数据集的前缀上训练; student 推理时一旦走偏, 就进入从没被训练过的前缀
          (exposure bias)。on-policy 直接在 student 会走到的状态上纠正它 —— 像 RL 一样 on-policy, 又像蒸馏一样每个 token 都有稠密信号。
核心公式:  y_i ~ π_s(·|x_i) (no_grad 采样)
           L = Σ_{i,t} KL( π_s(·|x_i, y_{i,<t}) ‖ π_teacher(·|x_i, y_{i,<t}) ) / Σ_i |y_i|
           分母是全批的 token 总数: 所有回复的 token 放在一起平均, 不是先在每条回复内平均。
           reverse KL = mode-seeking: student 在 teacher 认为不可能的地方放质量会被重罚, 但**漏掉** teacher 的某个模式不受罚。
           forward KL (distill.py) 正相反 = mode-covering。
读代码时盯住: 采样不回传梯度 (不是 REINFORCE), 梯度只来自对**整个词表**解析求和的逐位置 KL —— 所以方差远小于 RL。
为什么不接通用 Trainer: 与 GRPO 同理, 训练数据由当前 student 现场生成。
与论文的差异 (GKD 和 MiniLLM 处理采样的方式不同, 本库取前一种):
    本库的做法是把采样出的前缀当常数。"采到哪些前缀也随 θ 变" 这一项梯度被丢掉, 所以梯度有偏, 方差小。
    GKD 同样不对采样过程求导。MiniLLM 保留这一项, 用策略梯度估计它。
    本库只用 student 自己的样本, 没有混入固定数据集的样本。
依赖 llm_models: `student.generate(prompts, max_new_tokens, temperature)` 在 no_grad 下采样并返回普通张量。
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.data.tasks import completion_mask


def token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """逐位置 KL(p‖q): [..., V] ×2 → [...], 对词表解析求和。"""
    log_p, log_q = F.log_softmax(logits_p, dim=-1), F.log_softmax(logits_q, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


@torch.no_grad()
def completion_kl(model_p: nn.Module, model_q: nn.Module, seqs: torch.Tensor, prompt_len: int) -> float:
    """
    seqs [N, P+C] → 回复段上逐 token KL(p‖q) 的平均, 一个 float。评估用, 不回传梯度。
    只算第一个 EOS (含) 之前的位置, 分母是全批的有效 token 数, 与 on_policy_distill_step 的 loss 同一个口径。
    """
    x = seqs[:, :-1]                                                              # [N, P+C-1] 位置 t 的输出预测 token t+1
    kl = token_kl(model_p(x)[:, prompt_len - 1:], model_q(x)[:, prompt_len - 1:])  # [N, C] 回复的 C 个位置
    mask = completion_mask(seqs[:, prompt_len:]).float()                          # [N, C] EOS 之后不算
    return float((kl * mask).sum() / mask.sum())


def on_policy_distill_step(student: nn.Module, teacher: nn.Module, prompts: torch.Tensor,
                           max_new: int, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    prompts [B, P] → 采样、算 reverse KL、更新 student 一步, 返回 {"reverse_kl": 本步 loss}。
    max_new 是每条回复采多少个 token (记作 C)。teacher 由调用方冻结并置 eval, 这里只保证不给它算梯度。
    """
    P = prompts.size(1)
    # 1) student 自己采样; generate 在 no_grad 下运行, 返回普通张量, 可直接做带梯度的前向
    seqs = student.generate(prompts, max_new, temperature=1.0)                    # [B, P+C]
    mask = completion_mask(seqs[:, P:]).float()                                   # [B, C] EOS 之后不算

    # 2) 两个模型在**同一条 student 轨迹**上前向; 位置 P−1 … P+C−2 的输出预测回复的 C 个 token
    student.train()
    s_logits = student(seqs[:, :-1])[:, P - 1:]                                   # [B, C, V] 带梯度
    with torch.no_grad():                                                         # teacher 只当评分者
        t_logits = teacher(seqs[:, :-1])[:, P - 1:]                               # [B, C, V]

    # 3) reverse KL: student 在前 (期望在 student 自己的分布下取)
    # token_kl → [B, C]; 乘 mask 去掉 EOS 之后的位置, 再除以全批的有效 token 数
    loss = (token_kl(s_logits, t_logits) * mask).sum() / mask.sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)                     # 梯度范数上限 1.0, 防单步走太远
    optimizer.step()
    return {"reverse_kl": float(loss.detach())}
