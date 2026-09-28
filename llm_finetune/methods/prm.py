"""
PRM vs ORM — 过程奖励模型与结果奖励模型 (Lightman et al. 2023 "Let's Verify Step by Step")

是什么: 同一个 "主干 + 标量头" 的打分器, 监督信号不同:
        ORM 只在最后一个 token 上学 "最终答案对不对" (1 个标签 / 条);
        PRM 在**每一步**的末 token 上学 "这一步对不对" (K 个标签 / 条)。
解决什么: 多步推理里, ORM 要自己从一个 0/1 里猜出是哪一步坏的 (稀疏信号, 信用分配难), 还会给 "过程错、答案蒙对" 的解打高分;
          PRM 的信号是稠密的, 而且能指出**第一个错步** —— 可以拿来做 best-of-N 重排, 也可以当逐步的 RL 奖励。
核心公式:  s_t = σ(value_head(h_t))            h_t = 主干在第 t 个位置的隐状态 (因果: 只看到 t 及之前)
           L_ORM = BCE(s_末, [答案对])      L_PRM = mean_i BCE(s_{step_i}, [第 i 步对])
           解的 PRM 分 = min_i s_{step_i}    (一步错就整条错: 取最短板)
与论文的差异: 本库的做法是取 min。Lightman et al. 的论文把整条解的分定为各步概率的连乘 Π_i s_i,
              min 是它比较过的另一种聚合方式。取 min 时, 分数不会因为步数 K 变多而变小。
玩具数据 (ArithChain): a0 op1 k1 … opK kK SEP | op1 k1 v1 … opK kK vK EOS, 模 10 的 + / − / ×; 每步写成一行 "op k 结果"。
           第 i 步 "对" = v_i == op_i(v_{i−1}, k_i), v_{i−1} 是**解里写的**上一步 (错了之后照着错的值往下算, 像真人一样)。
           候选解来自一个带噪的程序求解器 (每步以 p_err 写错), 不是 LM —— 这里只研究 "打分器", 不研究 "生成器"。
读代码时盯住: `step_positions` —— 第 i 步的分取自 v_i 所在位置 (读完 v_i 之后), 不是 v_i 的前一个位置。
依赖 llm_models: `rm.backbone(seqs, return_hidden=True)` 返回隐状态 [N, T, D] (LLaMA.forward 的公开参数)。
"""

from typing import Dict

import torch
import torch.nn.functional as F

from llm_finetune.methods.reward_model import RewardModel

PAD, SEP, EOS = 0, 1, 2                             # 与 data/tasks.py 同值
MOD = 10
DIGIT0, PLUS, MINUS, TIMES = 3, 13, 14, 15          # 数字 0–9 → token 3–12


def apply_op(op: torch.Tensor, a: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """op ∈ {0:+, 1:−, 2:×}, a / k / 返回值都是 0–9 的数字 [N]。"""
    return torch.stack([a + k, a - k, a * k], dim=1).gather(1, op.unsqueeze(1)).squeeze(1) % MOD


class ArithChain:
    """K 步模 10 算术链。vocab_size 与 SeqTask 同为 16, 可以直接交给 common.make_model。"""

    vocab_size = 16

    def __init__(self, steps: int = 4) -> None:
        self.K = steps
        self.prompt_len = 2 * steps + 2                                   # a0, (op, k)×K, SEP
        # 解里每步占 3 个 token (op k v), v 是第 3 个 → 第 i 步 (从 0 数) 的 v 在 prompt_len + 3i + 2
        self.step_positions = self.prompt_len + 3 * torch.arange(steps) + 2   # [K] v_1 … v_K 的位置
        self.eos_position = self.prompt_len + 3 * steps                   # 3K 个解 token 之后就是 EOS

    def sample(self, n: int, p_err: float, split: str = "train", n_cand: int = 1) -> Dict[str, torch.Tensor]:
        """n 道题, 每题 n_cand 条带噪解 → seqs [n·n_cand, T], step_ok [n·n_cand, K], outcome_ok [n·n_cand]。
        留出集 = (a0 + Σk) ≡ 0 (mod 5) 的题, 训练集 = 其余。"""
        # ---- 1) 采题目: 多采再按 mod 5 筛 (同 SeqTask.sample_prompts) ----
        keep = []
        while sum(len(k) for k in keep) < n:
            a0 = torch.randint(0, MOD, (4 * n, 1))                        # [4n, 1] 起始数字 0–9
            ks = torch.randint(1, MOD, (4 * n, self.K))                   # [4n, K] 每步的操作数 1–9
            is_test = (a0.squeeze(1) + ks.sum(1)) % 5 == 0
            keep.append(torch.cat([a0, ks], 1)[is_test if split == "test" else ~is_test])
        prob = torch.cat(keep)[:n].repeat_interleave(n_cand, dim=0)       # [N, 1+K]
        a0, ks = prob[:, 0], prob[:, 1:]
        ops = torch.randint(0, 3, ks.shape)                               # [N, K] 0:+ 1:− 2:×

        # ---- 2) 带噪求解器逐步写解 ----
        N = len(prob)
        # truth: 每一步都算对时的值; written: 解里实际写下的值 (可能已经带错)。各 [N]
        truth, written = a0.clone(), a0.clone()
        vals, step_ok = torch.zeros(N, self.K, dtype=torch.long), torch.zeros(N, self.K)   # [N, K] ×2
        for i in range(self.K):
            # right: 从**写下的**上一步出发, 这一步算对应该得到的值
            truth, right = apply_op(ops[:, i], truth, ks[:, i]), apply_op(ops[:, i], written, ks[:, i])
            wrong = (right + torch.randint(1, MOD, (N,))) % MOD                # 一定 ≠ right
            err = torch.rand(N) < p_err
            written = torch.where(err, wrong, right)
            vals[:, i], step_ok[:, i] = written, (~err).float()
        # ---- 3) 拼成 token 序列: 数字 +DIGIT0, 运算符 +PLUS ----
        prompt = torch.stack([a0 + DIGIT0] + [t for i in range(self.K) for t in (ops[:, i] + PLUS, ks[:, i] + DIGIT0)]
                             + [torch.full((N,), SEP)], dim=1)
        solution = torch.stack([t for i in range(self.K) for t in (ops[:, i] + PLUS, ks[:, i] + DIGIT0, vals[:, i] + DIGIT0)],
                               dim=1)                                     # 每步写成 "op k v_i", 像草稿纸上的一行算式
        seqs = torch.cat([prompt, solution, torch.full((N, 1), EOS)], dim=1)   # [N, T], T = (2K+2) + 3K + 1
        # outcome_ok: 最后写下的值等于真值。中间错过、最后碰巧相等也算对 (过程错、答案蒙对)
        return {"seqs": seqs, "step_ok": step_ok, "outcome_ok": (written == truth).float()}


def token_scores(rm: RewardModel, seqs: torch.Tensor) -> torch.Tensor:
    """[N, T] → [N, T] 每个位置一个 logit (RewardModel 只取末位置; 这里要逐位置)。"""
    return rm.value_head(rm.backbone(seqs, return_hidden=True)).squeeze(-1)


def prm_loss(rm: RewardModel, task: ArithChain, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """PRM 的 loss: 在 K 个步末位置上各做一次二分类 (这一步对不对), 对 N·K 个标签取平均 BCE。batch 来自 task.sample。"""
    logits = token_scores(rm, batch["seqs"])[:, task.step_positions]                  # [N, K]
    return F.binary_cross_entropy_with_logits(logits, batch["step_ok"])


def orm_loss(rm: RewardModel, task: ArithChain, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """ORM 的 loss: 只在 EOS 位置上做一次二分类 (最终答案对不对), 对 N 个标签取平均 BCE。"""
    logits = token_scores(rm, batch["seqs"])[:, task.eos_position]                   # [N]
    return F.binary_cross_entropy_with_logits(logits, batch["outcome_ok"])


@torch.no_grad()
def solution_scores(rm: RewardModel, task: ArithChain, seqs: torch.Tensor, kind: str) -> torch.Tensor:
    """[N, T] → [N] 整条解的分。PRM: 各步概率的最小值; ORM: 末位置概率。"""
    p = torch.sigmoid(token_scores(rm, seqs))                         # [N, T]; 只有受过监督的位置 (步末 / EOS) 的值有意义
    return p[:, task.step_positions].min(1).values if kind == "prm" else p[:, task.eos_position]
