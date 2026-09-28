#!/usr/bin/env python
"""
SimPO / ORPO vs DPO: 同一个 SFT 起点、同样的偏好数据和步数。比两件事 ——
(1) 不要 reference 省了什么 (前向次数 / 常驻权重 / 实测耗时);  (2) 留出集上各自把模型带去了哪里。

    python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo

实验设计:
    第一组, 从同一个 SFT 起点 (100 步) 出发, 每臂 200 步, lr=3e-4, 开始前重置种子:
        DPO β=0.5          带 ref              ← 对照组
        SimPO β=2 γ=1      不带 ref
        ORPO λ=0.5         不带 ref
    第二组, 从**随机初始化**出发, 每臂 300 步, lr=3e-3 (ORPO 的卖点是不需要先 SFT):
        纯 SFT (从零)      只有 NLL            ← 对照组
        ORPO (从零)        NLL + odds-ratio 项
    怎么数前向: 给 policy 和 ref 各挂一个 forward hook, LLaMA.forward 每被调一次记 1。
                PairwiseForward 把 chosen / rejected 拼成一个 [2B, T] 的 batch, 所以 policy 每步只算 1 次。
                "纯 SFT (从零)" 这一臂没有 wrapper, hook 直接挂在 policy 上, 每步 1 次。
                "SFT 起点" 那一行不是训练臂, 前向次数和耗时都记 0。
    常驻权重 = policy 的字节数; 带 ref 时再乘 2。
断言 (每条验证一个结论):
    1. 每步前向次数: DPO 2 次, SimPO / ORPO / 纯 SFT 1 次。DPO 多出来的是 ref 那一次。
    2. DPO 的常驻权重是 SimPO 的 2 倍。
    3. 三种方法的留出集偏好准确率都不低于 SFT 起点。
    4. log π(chosen): ORPO 高于起点, DPO 和 SimPO 低于起点。
       ORPO 的 NLL 项直接抬高 chosen 的概率; DPO / SimPO 的 loss 只看差值, 没有这一项。
       这是本配置 (λ=0.5) 下的实测。λ 大时 odds-ratio 项仍可能把 chosen 拉低。
    5. ORPO 的贪心 EM 高于 SFT 起点。
    6. ORPO (从零) 的贪心 EM > 0.5: 不经 SFT 阶段也学会了任务。
    7. ORPO (从零) 的 log π(rejected) 低于纯 SFT (从零): odds-ratio 项把 rejected 多压了一截。
依赖 llm_models: 靠 nn.Module 的 forward hook 数 LLaMA.forward 的调用次数; 还读了 PairwiseForward 的内部属性 `_ref`。
"""

import copy
import time

import torch

from llm_finetune import (
    DPOLoss, InstructionDataGenerator, ORPOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask,
    SFTLoss, SimPOLoss, weight_bytes,
)
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, STEPS, LR, SCRATCH_LR = 100, 200, 3e-4, 3e-3   # 从零训练用 SFT 的 lr, 对齐阶段用小 lr


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    # 名字 → (留出集偏好指标, 贪心 EM, 每步前向次数, 耗时 (秒), 常驻权重 (字节))
    rows = {"SFT 起点": (preference_accuracy(sft, task), em_sft, 0, 0.0, weight_bytes(sft))}

    # 用只有一个元素的 list 当计数器: lambda 里不能给外层变量赋值, 但可以改 list 的内容
    forwards = [0]                                            # 数 LLaMA.forward 被调了几次 (policy 与 ref 都算)
    hook = lambda *_: forwards.__setitem__(0, forwards[0] + 1)

    def run(name, policy, wrapper, loss, steps=STEPS, data=None, lr=LR):
        """训一臂并把结果记进 rows。wrapper=None 表示直接训裸 policy (纯 SFT)。"""
        # 要挂 hook 的模型 = {policy, ref}; 没有 wrapper 或没有 ref 时 ref 记 None, 从集合里减掉
        ref = wrapper._ref[0] if wrapper else None
        handles = [m.register_forward_hook(hook) for m in {policy, ref} - {None}]
        forwards[0], t0 = 0, time.perf_counter()
        fit(wrapper or policy, data or PreferenceDataGenerator(task), loss, steps, lr, log_interval=steps)
        dt = time.perf_counter() - t0
        n_fwd = forwards[0] / steps
        for h in handles:
            h.remove()
        # ref 是 policy 的 deepcopy, 字节数相同: 带 ref 就是两份
        resident = weight_bytes(policy) * (2 if ref is not None else 1)
        rows[name] = (preference_accuracy(policy, task), task.exact_match(policy), n_fwd, dt, resident)

    for name, loss, with_ref in [("DPO β=0.5", DPOLoss(0.5), True), ("SimPO β=2 γ=1", SimPOLoss(2.0, 1.0), False),
                                 ("ORPO λ=0.5", ORPOLoss(0.5), False)]:
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        run(name, policy, PairwiseForward.with_frozen_copy(policy) if with_ref else PairwiseForward(policy), loss)

    # ORPO 的卖点是 "不需要先 SFT": 从**随机初始化**出发, 与同样步数的纯 SFT 对照
    for name, loss in [("纯 SFT (从零)", None), ("ORPO (从零)", ORPOLoss(0.5))]:
        torch.manual_seed(1)
        policy = make_model(task)
        if loss is None:
            run(name, policy, None, SFTLoss(), SFT_STEPS + STEPS, InstructionDataGenerator(task), lr=SCRATCH_LR)
        else:
            run(name, policy, PairwiseForward(policy), loss, SFT_STEPS + STEPS, lr=SCRATCH_LR)

    print(f"\n{'':<16}{'偏好准确率':>8}{'logπ(chosen)':>14}{'logπ(rejected)':>16}{'EM':>8}{'前向/步':>9}{'耗时':>8}{'常驻权重':>10}")
    for name, (p, em, n_fwd, dt, mem) in rows.items():
        print(f"{name:<16}{p['accuracy']:>12.3f}{p['logp_chosen']:>14.2f}{p['logp_rejected']:>16.2f}"
              f"{em:>8.3f}{n_fwd:>10.0f}{dt:>8.1f}s{mem / 1024:>7.0f}KiB")

    dpo, simpo, orpo = rows["DPO β=0.5"], rows["SimPO β=2 γ=1"], rows["ORPO λ=0.5"]
    # 元组下标: [0] = 偏好指标 dict, [1] = 贪心 EM, [2] = 每步前向次数, [4] = 常驻权重
    assert (dpo[2], simpo[2], orpo[2]) == (2, 1, 1), (
        f"DPO 每步多一次 ref 前向: 每步前向次数应为 DPO 2 / SimPO 1 / ORPO 1, 实际 {dpo[2]:g} / {simpo[2]:g} / {orpo[2]:g}")
    assert rows["纯 SFT (从零)"][2] == 1, f"纯 SFT 每步 1 次前向 (只有 policy), 实际 {rows['纯 SFT (从零)'][2]:g}"
    assert dpo[4] == 2 * simpo[4], f"DPO 常驻两份权重 (policy + ref): DPO {dpo[4]} 字节, SimPO {simpo[4]} 字节"
    for name in ("DPO β=0.5", "SimPO β=2 γ=1", "ORPO λ=0.5"):
        assert rows[name][0]["accuracy"] >= rows["SFT 起点"][0]["accuracy"], (
            f"{name}: 留出集偏好准确率不应下降: "
            f"{rows['SFT 起点'][0]['accuracy']:.3f} → {rows[name][0]['accuracy']:.3f}")
    assert orpo[0]["logp_chosen"] > rows["SFT 起点"][0]["logp_chosen"] > max(dpo[0]["logp_chosen"], simpo[0]["logp_chosen"]), (
        f"本配置下 log π(chosen) 应是 ORPO > SFT 起点 > DPO 和 SimPO (ORPO 的 NLL 项直接抬高 chosen; "
        f"DPO / SimPO 的 loss 只看差值): ORPO {orpo[0]['logp_chosen']:.2f}, 起点 {rows['SFT 起点'][0]['logp_chosen']:.2f}, "
        f"DPO {dpo[0]['logp_chosen']:.2f}, SimPO {simpo[0]['logp_chosen']:.2f}")
    assert orpo[1] > em_sft, f"ORPO 应同时提升 exact-match: {em_sft:.3f} → {orpo[1]:.3f}"
    scratch_sft, scratch_orpo = rows["纯 SFT (从零)"], rows["ORPO (从零)"]
    assert scratch_orpo[1] > 0.5, f"ORPO 应能不经 SFT 阶段直接学会任务: 贪心 EM {scratch_orpo[1]:.3f}, 应 > 0.5"
    assert scratch_orpo[0]["logp_rejected"] < scratch_sft[0]["logp_rejected"], (
        f"odds-ratio 项应把 rejected 压得比纯 SFT 更低: "
        f"ORPO {scratch_orpo[0]['logp_rejected']:.2f}, 纯 SFT {scratch_sft[0]['logp_rejected']:.2f}")


if __name__ == "__main__":
    main()
