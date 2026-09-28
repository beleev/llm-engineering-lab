#!/usr/bin/env python
"""
PRM vs ORM: 同一个主干 + 标量头、同一份带噪解 (4 步模 10 算术链)、同样步数; 唯一差别是监督信号 ——
ORM 每条解 1 个 "最终答案对不对", PRM 每条解 4 个 "这一步对不对"。
在**留出**题上每题给 8 个候选, 比 best-of-8 的正确率, 再看 PRM 能不能指出第一个错步。

    python -m llm_finetune.run_finetune.prm.train_prm
"""

import math

import torch

from llm_finetune.methods.prm import ArithChain, orm_loss, prm_loss, solution_scores, token_scores
from llm_finetune.methods.reward_model import RewardModel
from llm_finetune.run_finetune.common import make_model

STEPS, LR, BATCH, P_ERR, N_TEST, N_CAND = 600, 5e-4, 96, 0.2, 512, 8


def train(task: ArithChain, loss_fn) -> tuple:
    torch.manual_seed(0)                                     # 两个打分器同一个初始化、同一串训练数据
    rm = RewardModel(make_model(task))
    opt = torch.optim.AdamW(rm.parameters(), lr=LR)
    for _ in range(STEPS):
        loss = loss_fn(rm, task, task.sample(BATCH, P_ERR))
        opt.zero_grad()
        loss.backward()
        opt.step()
    rm.eval()
    return rm, float(loss.detach())


def main() -> None:
    task = ArithChain(steps=4)
    torch.manual_seed(1)
    test = task.sample(N_TEST, P_ERR, split="test", n_cand=N_CAND)
    outcome = test["outcome_ok"].view(N_TEST, N_CAND)                         # 最终答案对
    process = test["step_ok"].min(1).values.view(N_TEST, N_CAND)             # 每一步都对
    rows = torch.arange(N_TEST)
    lucky = float(((test["outcome_ok"] == 1) & (test["step_ok"].min(1).values == 0)).float().mean())
    print(f"候选解: 答案对 {float(outcome.mean()):.3f} | 每步都对 {float(process.mean()):.3f} | "
          f"过程错但答案蒙对 {lucky:.3f} | 8 个里至少一个答案对 {float(outcome.max(1).values.mean()):.3f}")

    pick = {"随机挑 1 个": torch.zeros(N_TEST, dtype=torch.long)}
    models, final_loss = {}, {}
    for kind, loss_fn in (("orm", orm_loss), ("prm", prm_loss)):
        models[kind], final_loss[kind] = train(task, loss_fn)
        pick[kind.upper()] = solution_scores(models[kind], task, test["seqs"], kind).view(N_TEST, N_CAND).argmax(1)
    print(f"训练 {STEPS} 步后的 BCE: ORM {final_loss['orm']:.3f} (ln 2 = {math.log(2):.3f}) | "
          f"PRM {final_loss['prm']:.3f} (只猜 '都对' = {-(0.8 * math.log(0.8) + 0.2 * math.log(0.2)):.3f})")

    print(f"\n{'best-of-' + str(N_CAND):<12}{'答案对':>8}{'每步都对':>10}")
    acc = {}
    for name, idx in pick.items():
        acc[name] = float(outcome[rows, idx].mean())
        print(f"{name:<14}{acc[name]:>8.3f}{float(process[rows, idx].mean()):>10.3f}")

    # ---- 定位第一个错步: PRM 打分最低的那一步 vs 真正第一个错步 (只看至少错了一步的候选) ----
    with torch.no_grad():
        p_step = torch.sigmoid(token_scores(models["prm"], test["seqs"]))[:, task.step_positions]   # [N, K]
    bad = test["step_ok"].min(1).values == 0
    first_err = (test["step_ok"] == 0).float().argmax(1)
    loc = float((p_step.argmin(1) == first_err)[bad].float().mean())
    always_first = float((first_err == 0)[bad].float().mean())              # 最好的常数猜法: 总说 "第 1 步错"
    print(f"\n定位第一个错步 (至少错一步的候选): PRM {loc:.3f} | 总猜第 1 步 {always_first:.3f} | ORM 只有一个分, 无从定位")

    assert acc["PRM"] > acc["ORM"] + 0.1, "同样的数据和步数, PRM 挑出的解应明显更常答对"
    assert acc["PRM"] > acc["随机挑 1 个"] + 0.15
    assert loc > always_first + 0.1, "PRM 应能定位第一个错步"


if __name__ == "__main__":
    main()
