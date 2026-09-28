#!/usr/bin/env python
"""
模型合并: 同一个基座 (会 copy) 分别全参微调出两个 "局部改写" 任务, 再不训练、只在权重空间里把两份微调合成一个模型。
比较 Task Arithmetic / TIES / DARE / SLERP 在 "两个任务互相干扰" 上的差别, 全部看留出集。

    python -m llm_finetune.run_finetune.merge.train_merge

实验设计:
    基座: 会 copy 的模型。两个任务都是 "copy, 但改写一段 token":
        任务 A   低段 token (3–8) 各 +1, 其余照抄
        任务 B   高段 token (10–15) 各 −1, 其余照抄
    两段不相交, token 9 两边都不改。两个任务各自全参微调 300 步, 得到 θ_A、θ_B。
    评测任务 = A 和 B 同时做 (低段 +1 且高段 −1)。单个微调模型做不到, 合并后的模型才可能做到。
    对照行: 基座 / 只微调 A / 只微调 B / 简单平均 (λ=0.5)。
    实验行: Task Arithmetic / TIES d=0.5 / DARE p=0.5 / DARE p=0.9 / SLERP t=0.5。
    λ 怎么定: 前四种方法各自在 LAMBDAS 里挑。挑的时候用训练分布的 prompt (split="train", seed=99),
              报告的数字用留出集。SLERP 没有 λ。
    三个指标: A 段准确率、B 段准确率、两段全对 EM。
    干扰 = 相对 "只微调该任务" 的掉点, A / B 两段取平均。
断言 (每条验证一个结论):
    1. TIES 的三步在一个能手算的 3 维例子上得到 [3, −1.5, 0]。
    2. 只微调 A 的模型 A 段准确率 > 0.95, 只微调 B 的 B 段 > 0.95: 单任务微调训好了。
    3. 两个单任务模型的两段全对 EM 都 < 0.1: 单个模型不能同时做两段改写。
    4. Task Arithmetic 的 A 段、B 段准确率都比基座高 0.7 以上, 两段全对 EM > 0.5。
    5. 简单平均的 EM 比 Task Arithmetic 低 0.3 以上。λ=0.5 时每个任务向量只加了一半。
    6. DARE p=0.9 的 EM 比 Task Arithmetic 低 0.2 以上。
       DARE 论文里 "丢 90% 几乎无损" 依赖大模型任务向量的冗余。这里模型太小, 没有这份冗余。
依赖 llm_models: 合并只读写 state_dict, 用 `load_state_dict` 装回一个 LLaMA 再评测。
"""

import copy
from typing import Dict, List, Sequence

import torch

from llm_finetune import (
    InstructionDataGenerator, SeqTask, SFTLoss, dare, sign_conflict, slerp, task_arithmetic, task_vectors, ties_merge,
)
from llm_finetune.data.tasks import EOS
from llm_finetune.run_finetune.common import fit, pretrained_base

STEPS, LR = 300, 1e-3
LOW, HIGH = list(range(3, 9)), list(range(10, 16))   # 两个任务各管一段 token, 不相交 (token 9 两边都不改)
LAMBDAS = [0.5, 0.7, 1.0, 1.5]                       # λ 在验证集 (训练分布的 prompt) 上选, 在留出集上报告


class EditTask(SeqTask):
    """
    copy, 但把指定的 token 平移: edits = [(token 列表, 平移量), ...]。
    A = 低段 +1, B = 高段 −1。关键: A 的训练数据里高段 token 照抄 (反之亦然) —— 每个任务向量只被要求改 "自己那段",
    两个任务在函数上几乎不重叠, 这正是任务向量能相加的前提。
    """

    def __init__(self, edits: Sequence) -> None:
        super().__init__("copy")
        self.table = torch.arange(self.vocab_size)            # [V] 查找表: 初始每个 token 映射到自己
        for tokens, shift in edits:
            self.table[tokens] += shift

    def target(self, prompts: torch.Tensor) -> torch.Tensor:
        return self.table[super().target(prompts)]            # 先 copy 出 [n, R], 再逐 token 查表改写; EOS 映射到自己


@torch.no_grad()
def scores(model, split: str = "test", seed: int = 1234) -> List[float]:
    """[A 段准确率, B 段准确率, 两段全对 EM]: 低段 token 所在位置对了几成 / 高段 / 整条回复全对。"""
    both = EditTask([(LOW, 1), (HIGH, -1)])
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    prompts = both.sample_prompts(256, split)
    torch.set_rng_state(state)
    out = model.generate(prompts, both.response_len, temperature=0)[:, both.prompt_len:]
    ok = out == both.target(prompts)                                              # [n, R] 每个位置对不对
    # src [n, R]: 回复的第 t 个位置对应 prompt 的第 t 个 token (copy 任务), 末位是 EOS。用它判断这个位置属于哪一段
    src = torch.cat([prompts[:, : both.length], torch.full((len(prompts), 1), EOS)], dim=1)   # 每个回复位置抄的是谁
    return [float(ok[torch.isin(src, torch.tensor(seg))].float().mean()) for seg in (LOW, HIGH)] + \
           [float(ok.all(dim=1).float().mean())]


def main() -> None:
    # TIES 三步在一个能手算的例子上: 各留 |·| 最大的 2 个 → 符号 sign(3−2, −1−2) = (+, −) → 只对同号的取平均
    toy = ties_merge({"w": torch.zeros(3)}, [{"w": torch.tensor([3., -1., .1])}, {"w": torch.tensor([-2., -2., .1])}],
                     density=2 / 3)
    assert torch.allclose(toy["w"], torch.tensor([3., -1.5, 0.])), (
        f"TIES 手算例子应得 [3, -1.5, 0] (修剪 → 选符号 → 只对同号的取平均), 实际 {toy['w'].tolist()}")

    torch.manual_seed(0)
    base = pretrained_base()
    probe = copy.deepcopy(base)                               # 一个空壳: 每次把合并出的权重装进去再评测

    def evaluate(sd: Dict[str, torch.Tensor], **kw) -> List[float]:
        """state_dict → [A 段准确率, B 段准确率, 两段全对 EM]。"""
        probe.load_state_dict(sd)
        return scores(probe, **kw)

    finetuned = []
    for edits in ([(LOW, 1)], [(HIGH, -1)]):
        torch.manual_seed(1)
        model = copy.deepcopy(base)
        fit(model, InstructionDataGenerator(EditTask(edits)), SFTLoss(), STEPS, LR, log_interval=STEPS)
        finetuned.append(model.state_dict())
    sd0, (sd_a, sd_b) = base.state_dict(), finetuned
    taus = task_vectors(sd0, finetuned)                       # [τ_A, τ_B]
    flat = [torch.cat([v.flatten() for v in t.values()]) for t in taus]   # 每个任务向量摊平成一个长向量
    cos = float(flat[0] @ flat[1] / (flat[0].norm() * flat[1].norm()))
    print(f"‖τ_A‖ {flat[0].norm():.2f}  ‖τ_B‖ {flat[1].norm():.2f}  cos(τ_A, τ_B) {cos:+.3f}  "
          f"符号冲突 {sign_conflict(taus):.1%} (两者都非零的坐标里)")

    methods = {
        "Task Arithmetic": lambda lam: task_arithmetic(sd0, taus, lam),
        "TIES d=0.5": lambda lam: ties_merge(sd0, taus, density=0.5, lam=lam),
        "DARE p=0.5": lambda lam: task_arithmetic(sd0, dare(taus, p=0.5), lam),
        "DARE p=0.9": lambda lam: task_arithmetic(sd0, dare(taus, p=0.9), lam),
    }
    # 名字 → (λ 或 None, [A 段准确率, B 段准确率, 两段全对 EM])
    rows = {"基座": (None, evaluate(sd0)), "只微调 A": (None, evaluate(sd_a)), "只微调 B": (None, evaluate(sd_b)),
            "简单平均": (0.5, evaluate(task_arithmetic(sd0, taus, 0.5)))}
    for name, merge in methods.items():
        # 每种方法各自在验证集上挑 λ (按两段全对 EM), 公平比较; 论文里也都是这么调的
        lam = max(LAMBDAS, key=lambda l: evaluate(merge(l), split="train", seed=99)[2])
        rows[name] = (lam, evaluate(merge(lam)))
    rows["SLERP t=0.5"] = (None, evaluate(slerp(sd_a, sd_b, 0.5)))

    print(f"\n{'':<16}{'λ':>5}{'A 段准确率':>11}{'B 段准确率':>11}{'两段全对 EM':>12}{'干扰(A/B 平均掉点)':>18}")
    for name, (lam, (acc_a, acc_b, em)) in rows.items():
        # 干扰 = 相对 "只微调该任务" 的掉点; 只对合并出来的模型有意义
        merged = lam is not None or name.startswith("SLERP")
        drop = f"{((rows['只微调 A'][1][0] - acc_a) + (rows['只微调 B'][1][1] - acc_b)) / 2:.3f}" if merged else "-"
        print(f"{name:<16}{'-' if lam is None else f'{lam:g}':>5}{acc_a:>11.3f}{acc_b:>11.3f}{em:>12.3f}{drop:>18}")

    ta, avg, dare9 = rows["Task Arithmetic"][1], rows["简单平均"][1], rows["DARE p=0.9"][1]
    best_other = max(rows, key=lambda n: rows[n][1][2] if n not in ("Task Arithmetic", "基座") else -1)
    verdict = "没有赢过" if rows[best_other][1][2] <= ta[2] else "赢过了"
    print(f"\n除 Task Arithmetic 外两段全对 EM 最高的是 {best_other} ({rows[best_other][1][2]:.3f}), "
          f"{verdict}调好 λ 的 Task Arithmetic ({ta[2]:.3f})")
    # SLERP 插的是完整权重 θ_A、θ_B: 两者共享 θ₀, 夹角不大, t=0.5 时 ≈ 简单平均再把范数补回一点
    wa, wb = (torch.cat([v.flatten() for v in sd.values() if v.is_floating_point()]) for sd in (sd_a, sd_b))
    omega = float(torch.arccos(wa @ wb / (wa.norm() * wb.norm())))
    print(f"θ_A 与 θ_B 夹角 Ω = {omega:.3f} rad; t=0.5 时 SLERP 相对线性平均的放大倍数 1/cos(Ω/2) = {1 / torch.cos(torch.tensor(omega / 2)):.3f}")

    assert rows["只微调 A"][1][0] > 0.95, f"单任务微调没训好: 只微调 A 的 A 段准确率 {rows['只微调 A'][1][0]:.3f}, 应 > 0.95"
    assert rows["只微调 B"][1][1] > 0.95, f"单任务微调没训好: 只微调 B 的 B 段准确率 {rows['只微调 B'][1][1]:.3f}, 应 > 0.95"
    assert max(rows["只微调 A"][1][2], rows["只微调 B"][1][2]) < 0.1, (
        f"单个微调模型不应能同时做两段改写 (两段全对 EM 应 < 0.1): "
        f"只微调 A {rows['只微调 A'][1][2]:.3f}, 只微调 B {rows['只微调 B'][1][2]:.3f}")
    assert min(ta[:2]) > max(rows["基座"][1][:2]) + 0.7, f"合并后应在两个任务上都明显好于基座: {ta}"
    assert ta[2] > 0.5, f"合并后应能同时完成两段改写: EM {ta[2]:.3f}"
    assert avg[2] < ta[2] - 0.3, (
        f"简单平均应明显弱于调过 λ 的 Task Arithmetic (简单平均 = λ=0.5: 每个任务向量只加了一半, "
        f"两段都只改到一半): EM {avg[2]:.3f} vs {ta[2]:.3f}")
    assert dare9[2] < ta[2] - 0.2, (
        f"玩具模型上 DARE p=0.9 本应明显掉点 (论文里 '丢 90% 几乎无损' 依赖大模型任务向量的冗余, "
        f"这里模型太小): EM {dare9[2]:.3f} vs Task Arithmetic {ta[2]:.3f}")


if __name__ == "__main__":
    main()
