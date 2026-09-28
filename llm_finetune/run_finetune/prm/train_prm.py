#!/usr/bin/env python
"""
PRM vs ORM: 同一个主干 + 标量头、同一份带噪解 (4 步模 10 算术链)、同样步数; 唯一差别是监督信号 ——
ORM 每条解 1 个 "最终答案对不对", PRM 每条解 4 个 "这一步对不对"。
在**留出**题上每题给 8 个候选, 比 best-of-8 的正确率, 再看 PRM 能不能指出第一个错步。

    python -m llm_finetune.run_finetune.prm.train_prm

实验设计:
    两臂: ORM (对照组) / PRM。同一个种子 → 同一个初始化、同一串训练数据; 600 步, 每步 96 条带噪解。
          两臂唯一的差别是 loss 函数, 也就是监督信号落在哪些位置。
    带噪解: 程序求解器每一步以 p_err = 0.2 的概率写错。写错之后照着错的值往下算。
    验收: 512 道留出题, 每题 8 条候选解。打分器给 8 条各打一个分, 挑最高的那条, 看它答案对不对。
    两个参照:
        随机挑 1 个    永远取第 0 条候选。8 条候选独立同分布, 取第 0 条等于随机挑。
        总猜第 1 步    定位错步时最好的常数猜法。
断言 (每条验证一个结论):
    1. PRM 挑出的解, 答案正确率比 ORM 高 0.1 以上。同样的数据和步数, 稠密的监督信号学得更快。
    2. PRM 挑出的解, 答案正确率比随机挑高 0.15 以上。
    3. PRM 打分最低的那一步 = 真正的第一个错步, 命中率比 "总猜第 1 步" 高 0.1 以上。
依赖 llm_models: 打分器是 RewardModel(LLaMA), 靠 `return_hidden=True` 取每个位置的隐状态。
                 这个脚本不用 Trainer, 手写的循环里没有 warmup 和梯度裁剪。
"""

import math

import torch

from llm_finetune import ArithChain, RewardModel, orm_loss, prm_loss, solution_scores, token_scores
from llm_finetune.run_finetune.common import make_model

STEPS, LR, BATCH, P_ERR, N_TEST, N_CAND = 600, 5e-4, 96, 0.2, 512, 8


def train(task: ArithChain, loss_fn) -> tuple:
    """训一个打分器, 返回 (eval 状态的 RewardModel, 最后一步的 loss)。loss_fn 是 prm_loss 或 orm_loss。"""
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
    # 采样时同一题的 8 条候选是连着放的, 所以 [N_TEST·N_CAND] 可以直接 view 成 [N_TEST, N_CAND]
    outcome = test["outcome_ok"].view(N_TEST, N_CAND)                         # 最终答案对
    process = test["step_ok"].min(1).values.view(N_TEST, N_CAND)             # 每一步都对 ([N, K] 按步取 min → [N])
    rows = torch.arange(N_TEST)
    lucky = float(((test["outcome_ok"] == 1) & (test["step_ok"].min(1).values == 0)).float().mean())
    print(f"候选解: 答案对 {float(outcome.mean()):.3f} | 每步都对 {float(process.mean()):.3f} | "
          f"过程错但答案蒙对 {lucky:.3f} | {N_CAND} 个里至少一个答案对 {float(outcome.max(1).values.mean()):.3f}")

    # pick[名字] = [N_TEST] 每道题挑中的候选下标。候选独立同分布, 永远取第 0 条就等于随机挑
    pick = {"随机挑 1 个": torch.zeros(N_TEST, dtype=torch.long)}
    models, final_loss = {}, {}
    for kind, loss_fn in (("orm", orm_loss), ("prm", prm_loss)):
        models[kind], final_loss[kind] = train(task, loss_fn)
        pick[kind.upper()] = solution_scores(models[kind], task, test["seqs"], kind).view(N_TEST, N_CAND).argmax(1)
    print(f"训练 {STEPS} 步后的 BCE: ORM {final_loss['orm']:.3f} (ln 2 = {math.log(2):.3f}) | "
          f"PRM {final_loss['prm']:.3f} (只猜 '都对' = {-((1 - P_ERR) * math.log(1 - P_ERR) + P_ERR * math.log(P_ERR)):.3f})")

    print(f"\n{'best-of-' + str(N_CAND):<12}{'答案对':>8}{'每步都对':>10}")
    acc = {}
    for name, idx in pick.items():
        acc[name] = float(outcome[rows, idx].mean())
        print(f"{name:<14}{acc[name]:>8.3f}{float(process[rows, idx].mean()):>10.3f}")

    # ---- 定位第一个错步: PRM 打分最低的那一步 vs 真正第一个错步 (只看至少错了一步的候选) ----
    with torch.no_grad():
        p_step = torch.sigmoid(token_scores(models["prm"], test["seqs"]))[:, task.step_positions]   # [N, K]
    bad = test["step_ok"].min(1).values == 0                                # [N] 至少错了一步的候选
    # argmax 在并列时返回第一个最大值的下标 → 第一个 "错" 的步。全对的行会得到 0, 由上面的 bad 过滤掉
    first_err = (test["step_ok"] == 0).float().argmax(1)                    # [N]
    loc = float((p_step.argmin(1) == first_err)[bad].float().mean())
    always_first = float((first_err == 0)[bad].float().mean())              # 最好的常数猜法: 总说 "第 1 步错"
    print(f"\n定位第一个错步 (至少错一步的候选): PRM {loc:.3f} | 总猜第 1 步 {always_first:.3f} | ORM 只有一个分, 无从定位")

    assert acc["PRM"] > acc["ORM"] + 0.1, (
        f"同样的数据和步数, PRM 挑出的解应明显更常答对 (高 0.1 以上): PRM {acc['PRM']:.3f}, ORM {acc['ORM']:.3f}")
    assert acc["PRM"] > acc["随机挑 1 个"] + 0.15, (
        f"PRM 做 best-of-{N_CAND} 重排应比随机挑 1 个高 0.15 以上: "
        f"PRM {acc['PRM']:.3f}, 随机 {acc['随机挑 1 个']:.3f}")
    assert loc > always_first + 0.1, (
        f"PRM 应能定位第一个错步: 命中率 {loc:.3f}, 应比 '总猜第 1 步' 的 {always_first:.3f} 高 0.1 以上")


if __name__ == "__main__":
    main()
