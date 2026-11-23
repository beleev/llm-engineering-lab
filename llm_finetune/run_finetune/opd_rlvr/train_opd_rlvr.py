#!/usr/bin/env python
"""
在线蒸馏 (OPD) 与 RLVR 怎么组合: 只 RLVR、只 OPD、先 OPD 再 RLVR、两种损失联合; 再加一个按领域路由的多教师 OPD。

    python -m llm_finetune.run_finetune.opd_rlvr.train_opd_rlvr

两种信号的差别:
    RLVR (GRPO + task.verify)   一条回复一个 0/1 分, 稀疏; 不需要老师, 学生采不到对的回复时这道题没有梯度。
    OPD (逐 token reverse KL)   学生自己采样, 老师在每个位置给一个 V 维分布, 稠密; 但学生学的是老师的分布, 老师的错也一起学。
实验设计 (任务 sort; 学生是 common.make_model 经 SFT 150 步的 "半会" 起点, 与 grpo 脚本相同):
    第一部分 弱老师 + 五臂:
        老师: 同结构, SFT 250 步。它的采样 pass@1 约 0.55, 只比学生好一截, 是 "学生能追上的老师"。
        各臂都从学生起点的 deepcopy 出发, 各 160 步, 每步 256 条学生 rollout, 开始前重置种子:
            只 RLVR          GRPOTrainer, 每步 32 个 prompt × G=8, lr=3e-4          ← 对照组
            只 OPD           on_policy_distill_step, 每步 256 个 prompt × 1 条, lr=1e-3
            先 OPD 再 RLVR   前 80 步 OPD, 后 80 步 RLVR, 两段各用自己的 lr 和一个新的 AdamW
            联合             同一批 GRPO rollout 上 loss = GRPO surrogate + λ·逐 token reverse KL, λ=1。
                             lr 各跑一次: 3e-4 (RLVR 的 lr) 和 1e-3 (OPD 的 lr), 免得输赢只是 lr 选得不好
        验收 (留出集): 采样 pass@1 (n=512, T=1)、贪心 EM、学生样本上的 KL(学生‖老师)。
    第二部分 强老师 + 两臂: 老师 SFT 800 步 (pass@1 接近 1), 只比 "只 OPD" 和 "先 OPD 再 RLVR", 同样 160 步、80 步切换。
        用来看切换时机: 老师远强于学生时, 80 步的 OPD 还没学到头。
    第三部分 多教师 OPD (MixTask: prompt 末尾的分隔符是 SEP 就做 sort, 是 CPY 就做 copy):
        两位领域老师各只训自己的子任务 (各 800 步); 学生在两个子任务的混合数据上 SFT 150 步。
        两臂各 OPD 100 步, 每步 256 个 prompt (两个子任务各约一半), lr=1e-3:
            单老师       所有样本都交给 sort 老师打分                             ← 对照组
            多教师路由   按 prompt 的分隔符把样本交给对应领域的老师 (MiMo-V2-Flash 的 MOPD 就是按领域选老师)
        验收: 两个子任务各自的留出集采样 pass@1。
断言 (每条验证一个结论; 只断言实测成立的, 单种子):
    1. 弱老师: 先 OPD 再 RLVR 的 pass@1 高于只 OPD、只 RLVR 和两个 lr 的联合。
    2. 弱老师: 先 OPD 再 RLVR 的 pass@1 高于老师自己。RLVR 的奖励来自 verifier, 不受老师水平限制。
    3. 弱老师: 只 OPD 离老师最近 (KL 最小), 而且 pass@1 不超过老师。OPD 的上限就是老师。
    4. 强老师: 只 OPD 的 pass@1 高于先 OPD 再 RLVR。老师还远没被学完时就切到 RLVR, 是把稠密信号换成了稀疏信号。
    5. 多教师: 单老师把学生的 copy 能力压到起点以下, 多教师路由的 copy pass@1 比单老师高 0.5 以上, sort 也更高。
与论文 / 真实系统的差异:
    简化: 老师与学生同结构 (2 层 d=64), 老师只是多训了几步; 真实 OPD 的老师比学生大一个量级。
    简化: 切换点固定在一半 (第 80 步)。Sequential Beats Joint (arXiv 2609.04108) 建议按 OPD 的验证分数决定何时切到 RL。
    简化: 多教师只路由逐 token 的老师分布, 没有像 MiMo-V2-Flash 那样再叠加可验证的结果奖励。
依赖 llm_models: 学生和老师都是 LLaMA; OPD 与 GRPO 都靠 `LLaMA.generate` 在 no_grad 下采样。
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune import (
    GRPOTrainer, PromptDataGenerator, SeqTask, aggregate, completion_kl, completion_logprobs, completion_mask,
    freeze_module, group_advantages, make_config, on_policy_distill_step, token_kl,
)
from llm_finetune.data.tasks import EOS
from llm_finetune.run_finetune.common import make_model, sft_warmup

STUDENT_SFT, WEAK_SFT, STRONG_SFT = 150, 250, 800
STEPS, PROMPTS, G = 160, 32, 8                    # 每步 32 × 8 = 256 条 rollout; OPD 每步 256 个 prompt × 1 条
SWITCH = STEPS // 2                               # 先 OPD 再 RLVR: 第 80 步切换
LR_RL, LR_OPD, LAMBDA = 3e-4, 1e-3, 1.0
MT_STEPS = 100                                    # 第三部分每臂的 OPD 步数
CPY = 16                                          # MixTask 里 "做 copy" 的分隔符; 内容 token 仍是 3–15


def evaluate(student: nn.Module, teacher: nn.Module, task: SeqTask) -> Dict[str, float]:
    """留出集: 采样 pass@1, 贪心 EM, 学生自己样本上的 KL(学生‖老师)。固定种子, 不扰动训练的随机流。"""
    state = torch.get_rng_state()
    torch.manual_seed(99)
    seqs = student.generate(task.sample_prompts(256, "test"), task.response_len, temperature=1.0)   # [256, P+R]
    kl = completion_kl(student, teacher, seqs, task.prompt_len)
    torch.set_rng_state(state)
    return {"pass1": task.exact_match(student, n=512, temperature=1.0), "em": task.exact_match(student), "kl": kl}


def make_teacher(task: SeqTask, steps: int) -> nn.Module:
    teacher = make_model(task)
    sft_warmup(teacher, task, steps)
    teacher.eval()
    freeze_module(teacher)                                           # 老师只打分, 不进 optimizer
    return teacher


def rlvr_steps(policy: nn.Module, task: SeqTask, steps: int) -> None:
    trainer = GRPOTrainer(policy, task.verify, make_config("grpo", lr=LR_RL, max_new=task.response_len + 2, group_size=G))
    prompts = PromptDataGenerator(task, PROMPTS)
    for _ in range(steps):
        trainer.step(prompts.generate_batch()["prompts"])


def opd_steps(student: nn.Module, teacher: nn.Module, task: SeqTask, steps: int) -> None:
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR_OPD)
    prompts = PromptDataGenerator(task, PROMPTS * G)                 # 与 RLVR 同样每步 256 条 rollout
    for _ in range(steps):
        on_policy_distill_step(student, teacher, prompts.generate_batch()["prompts"], task.response_len, optimizer)


def joint_steps(policy: nn.Module, teacher: nn.Module, task: SeqTask, steps: int, lr: float) -> None:
    """同一批 GRPO rollout 上同时算两种 loss: 每个 token 的 loss = −surrogate + λ·KL(学生‖老师), 聚合方式同 GRPO。"""
    cfg = make_config("grpo", lr=lr, max_new=task.response_len + 2, group_size=G)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    prompts, P = PromptDataGenerator(task, PROMPTS), task.prompt_len
    for _ in range(steps):
        expanded = prompts.generate_batch()["prompts"].repeat_interleave(G, dim=0)   # [N=B·G, P]
        seqs = policy.generate(expanded, cfg.max_new, temperature=1.0)               # [N, P+C]
        mask = completion_mask(seqs[:, P:]).float()                                  # [N, C]
        A = group_advantages(task.verify(expanded, seqs[:, P:]), G, cfg.std_norm).unsqueeze(1)   # [N, 1]
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seqs, P)                          # [N, C] 行为策略
            t_logits = teacher(seqs[:, :-1])[:, P - 1:]                              # [N, C, V] 老师只打分
        policy.train()
        for _ in range(cfg.inner_epochs):
            logits = policy(seqs[:, :-1])[:, P - 1:]                                 # [N, C, V] 一次前向, 两项共用
            logp = F.log_softmax(logits, -1).gather(-1, seqs[:, P:].unsqueeze(-1)).squeeze(-1)   # [N, C]
            ratio = (logp - old_logp).exp()
            surrogate = torch.min(ratio * A, ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high) * A)
            # ★ 稀疏项和稠密项加在同一个 loss 里, 共用一次反传和一个 lr
            loss = aggregate(-surrogate + LAMBDA * token_kl(logits, t_logits), mask, cfg.loss_agg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


def print_rows(title: str, rows: Dict[str, Dict[str, float]]) -> None:
    print(title)
    print(f"{'':<16}{'采样 pass@1':>12}{'贪心 EM':>10}{'KL(学生‖老师)':>16}")
    for name, r in rows.items():
        print(f"{name:<16}{r['pass1']:>12.3f}{r['em']:>10.3f}{r['kl']:>16.4f}")


def combine(task: SeqTask) -> Dict[str, Dict[str, Dict[str, float]]]:
    """第一、二部分。返回 {"weak": 弱老师五臂, "strong": 强老师两臂}, 每臂是 evaluate 的结果。"""
    torch.manual_seed(0)
    student = make_model(task)
    sft_warmup(student, task, STUDENT_SFT)                           # 与 grpo 脚本同一个 "半会" 起点
    out = {}
    for regime, teacher_steps in (("weak", WEAK_SFT), ("strong", STRONG_SFT)):
        torch.manual_seed(7)
        teacher = make_teacher(task, teacher_steps)
        arms = {
            "只 OPD": lambda m: opd_steps(m, teacher, task, STEPS),
            "先 OPD 再 RLVR": lambda m: (opd_steps(m, teacher, task, SWITCH), rlvr_steps(m, task, STEPS - SWITCH)),
        }
        if regime == "weak":                                         # 强老师只跑两臂, 看切换时机
            arms = {"只 RLVR": lambda m: rlvr_steps(m, task, STEPS), **arms,
                    "联合 lr=3e-4": lambda m: joint_steps(m, teacher, task, STEPS, LR_RL),
                    "联合 lr=1e-3": lambda m: joint_steps(m, teacher, task, STEPS, LR_OPD)}
        rows = {"老师": evaluate(teacher, teacher, task), "学生起点": evaluate(student, teacher, task)}
        for name, run in arms.items():
            torch.manual_seed(1)
            model = copy.deepcopy(student)
            run(model)
            rows[name] = evaluate(model, teacher, task)
        print_rows(f"\n---- {'弱' if regime == 'weak' else '强'}老师 (SFT {teacher_steps} 步), 各 {STEPS} 步, "
                   f"每步 {PROMPTS * G} 条学生 rollout; 留出集 ----", rows)
        out[regime] = rows
    return out


# ---------------------------------------------------------------- 第三部分: 多教师 OPD

class MixTask(SeqTask):
    """
    两个子任务共用一个模型: prompt 末尾的分隔符是 SEP 就做 sort, 是 CPY 就做 copy。
    domain=None 时两种各约一半; 指定 domain 时只出这一种 (用来训领域老师、分领域验收)。
    verify / exact_match 沿用 SeqTask, 它们只调 sample_prompts 和 target。
    """

    def __init__(self, domain: Optional[str] = None) -> None:
        super().__init__("sort", vocab_size=CPY + 1)                 # 词表多一个 CPY
        self.domain = domain

    def sample_prompts(self, n: int, split: str = "train") -> torch.Tensor:
        x = SeqTask("sort").sample_prompts(n, split)                 # [n, P] 内容 token 取 3–15, 不会撞上 CPY
        is_copy = torch.rand(n) < 0.5 if self.domain is None else torch.full((n,), self.domain == "copy")
        x[is_copy, -1] = CPY
        return x

    def target(self, prompts: torch.Tensor) -> torch.Tensor:
        x = prompts[:, : self.length]
        y = torch.where(prompts[:, -1:] == CPY, x, x.sort(dim=1).values)               # [n, L] 按分隔符选规则
        return torch.cat([y, torch.full((len(x), 1), EOS)], dim=1)


class RoutedTeacher(nn.Module):
    """按每条序列 prompt 末尾的分隔符, 把它交给对应领域的老师打分。on_policy_distill_step 不用改。"""

    def __init__(self, sort_teacher: nn.Module, copy_teacher: nn.Module, prompt_len: int) -> None:
        super().__init__()
        self.sort_teacher, self.copy_teacher, self.sep_pos = sort_teacher, copy_teacher, prompt_len - 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # ponytail: 两位老师都对整批前向一遍再按行挑, 算力翻倍; 真实系统按领域分批, 每条只调一位老师
        is_copy = (idx[:, self.sep_pos] == CPY).view(-1, 1, 1)
        return torch.where(is_copy, self.copy_teacher(idx), self.sort_teacher(idx))


def domain_pass1(model: nn.Module) -> Dict[str, float]:
    """两个子任务各自的留出集采样 pass@1。"""
    return {d: MixTask(d).exact_match(model, n=512, temperature=1.0) for d in ("sort", "copy")}


def multi_teacher() -> Dict[str, Dict[str, float]]:
    torch.manual_seed(2)
    mix = MixTask()
    teachers = {d: make_teacher(MixTask(d), STRONG_SFT) for d in ("sort", "copy")}   # 领域老师只见过自己的子任务
    torch.manual_seed(5)
    student = make_model(mix)
    sft_warmup(student, mix, STUDENT_SFT)

    rows = {f"{d} 老师": domain_pass1(t) for d, t in teachers.items()}
    rows["学生起点"] = domain_pass1(student)
    for name, teacher in (("单老师 (sort)", teachers["sort"]),
                          ("多教师路由", RoutedTeacher(teachers["sort"], teachers["copy"], mix.prompt_len))):
        torch.manual_seed(3)
        model = copy.deepcopy(student)
        opd_steps(model, teacher, mix, MT_STEPS)
        rows[name] = domain_pass1(model)

    print(f"\n---- 多教师 OPD: 各 {MT_STEPS} 步; 留出集采样 pass@1 ----")
    print(f"{'':<16}{'sort':>8}{'copy':>8}")
    for name, r in rows.items():
        print(f"{name:<16}{r['sort']:>8.3f}{r['copy']:>8.3f}")
    return rows


def main() -> None:
    res = combine(SeqTask("sort"))
    weak, strong = res["weak"], res["strong"]
    seq = weak["先 OPD 再 RLVR"]["pass1"]
    for arm in ("只 OPD", "只 RLVR", "联合 lr=3e-4", "联合 lr=1e-3"):
        assert seq > weak[arm]["pass1"], (
            f"弱老师: 先 OPD 再 RLVR 的 pass@1 应高于{arm}: {seq:.3f} vs {weak[arm]['pass1']:.3f}")
    assert seq > weak["老师"]["pass1"], (
        f"弱老师: 先 OPD 再 RLVR 应超过老师自己 (RLVR 的奖励来自 verifier, 不以老师为上限): "
        f"pass@1 {seq:.3f} vs 老师 {weak['老师']['pass1']:.3f}")
    assert weak["只 OPD"]["pass1"] <= weak["老师"]["pass1"], (
        f"弱老师: 只 OPD 的 pass@1 不应超过老师 (OPD 的目标就是贴近老师的分布): "
        f"{weak['只 OPD']['pass1']:.3f} vs 老师 {weak['老师']['pass1']:.3f}")
    nearest = min(r["kl"] for name, r in weak.items() if name not in ("老师", "学生起点", "只 OPD"))
    assert weak["只 OPD"]["kl"] < nearest, (
        f"弱老师: 只 OPD 应离老师最近: KL(学生‖老师) {weak['只 OPD']['kl']:.4f}, 其余各臂最小 {nearest:.4f}")
    assert strong["只 OPD"]["pass1"] > strong["先 OPD 再 RLVR"]["pass1"], (
        f"强老师: OPD 还没学到头时切到 RLVR 应该吃亏: pass@1 只 OPD {strong['只 OPD']['pass1']:.3f}, "
        f"先 OPD 再 RLVR {strong['先 OPD 再 RLVR']['pass1']:.3f}")

    mt = multi_teacher()
    start, single, routed = mt["学生起点"], mt["单老师 (sort)"], mt["多教师路由"]
    assert single["copy"] < start["copy"], (
        f"单老师: sort 老师在 copy 样本上按 sort 打分, 应把学生的 copy 能力压到起点以下: "
        f"{single['copy']:.3f} vs 起点 {start['copy']:.3f}")
    assert routed["copy"] > single["copy"] + 0.5, (
        f"多教师路由: copy 样本交给 copy 老师, pass@1 应比单老师高 0.5 以上: {routed['copy']:.3f} vs {single['copy']:.3f}")
    assert routed["sort"] > single["sort"], (
        f"多教师路由: 不再被 copy 样本上的错误分布拖累, sort 的 pass@1 也应更高: "
        f"{routed['sort']:.3f} vs {single['sort']:.3f}")


if __name__ == "__main__":
    main()
