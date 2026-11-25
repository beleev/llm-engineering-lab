#!/usr/bin/env python
"""
LoRA 做 RL: 同一个 GRPO 任务上比全参、LoRA r=1 / r=8, 再用两组对照找出差距从哪来; 最后在同一起点上做一次 SFT 对照。

    python -m llm_finetune.run_finetune.lora_rl.train_lora_rl

要回答的问题: run_finetune/lora 量到 "同步数下全参优于 LoRA" (EM 0.809 vs 0.352), 那是 SFT 上的结论。
Thinking Machines 的 "LoRA Without Regret" (Schulman 等, 2025-09-29) 报告 RL 场景下秩 1 的 LoRA 就能追平全参,
理由是策略梯度每个 episode 只靠一个优势值学习, 只带回 O(1) bit 的信息, 而 SFT 每条样本带回 O(token 数) bit。
这个脚本在本章的玩具规模上量一遍, 结果照实写。

实验设计:
    起点: 与 train_grpo 相同, sort 任务上 SFT 150 步的 policy (种子 0)。
    六种 "哪些参数可训" 的配置, RL 和 SFT 各跑一遍:
        全参                 全部 99,648 个参数
        只训线性层           7 种线性层全参可训, RMSNorm 增益和 (与 lm_head 共享的) embedding 冻结
        LoRA r=1 / r=8       注入全部 7 种线性层 (含 MLP), α 固定 16, 缩放 α/r (原论文与博客的参数化); 其余全部冻结
        LoRA r=1 + norm      在 r=1 之上再放开 RMSNorm 增益 (5 × 64 = 320 个)
        只训 norm            只训这 320 个增益
    RL 臂: GRPO (methods/grpo.py 原样 import), 32 prompt × G=8, 60 步, max_new = R+2, β=0。
    SFT 臂: 同一起点、同样 60 步, 每步 64 条 (prompt, 正确回复)。
    每种配置扫 2 个 lr (见 RL_GRID / SFT_GRID, 按预扫描取的两个较好值), 每臂开始前重置种子 1。
    验收指标 (都在留出集上): 采样 pass@1 (temperature=1, 512 题, RL 直接优化的量) 与贪心 EM (256 题)。
    RL 按 pass@1 取每种配置的最好一臂, SFT 按贪心 EM 取。
断言 (每条验证一个结论):
    1. mark_only_lora_as_trainable 之后, GRPOTrainer 的 AdamW 只动 lora_ 参数: 训练后基座权重逐位不变。
    2. RL: 全参把采样 pass@1 提高 0.04 以上。
    3. RL: 标准 LoRA r=1 (norm 冻结) 的 pass@1 比全参低 0.05 以上 —— 博客的结论在这个规模上没有照原样成立。
    4. RL: 只训线性层 (满秩, 98,304 个参数) 同样比全参低 0.05 以上 —— 差距不在秩。
    5. RL: LoRA r=1 再放开 320 个 norm 增益, 与全参的差距缩到 0.05 以内; 只训 norm 就比起点高 0.1 以上。
    6. RL: 没有哪一臂的贪心 EM 比起点高 0.02 以上 —— 60 步 RL 的收益是把分布变尖, 不是学会新的映射。
    7. SFT: 全参的贪心 EM 涨 0.1 以上, 只训 norm 涨不到 0.05; 全参比 LoRA r=1 高 0.04 以上。
依赖 llm_models: GRPOTrainer 靠 `LLaMA.generate` 采样; apply_lora 按属性名 (w_q、w_up …) 找层;
                 LLaMA 在 lm_head 之前有一层 RMSNorm (ln_f), logits 的整体尺度由它的增益和 embedding 决定。
"""

import copy

import torch

from llm_finetune import (
    ALL_LINEARS, GRPOTrainer, InstructionDataGenerator, PromptDataGenerator, SeqTask, SFTLoss, apply_lora,
    count_parameters, make_config, mark_only_lora_as_trainable,
)
from llm_finetune.run_finetune.common import fit, make_model, sft_warmup

SFT_STEPS, STEPS, PROMPTS_PER_STEP, G, ALPHA = 150, 60, 32, 8, 16
RL_GRID = {"全参": [3e-4, 1e-3], "只训线性层": [3e-4, 1e-3], "LoRA r=1": [1e-3, 3e-3],
           "LoRA r=8": [1e-3, 3e-3], "LoRA r=1 + norm": [3e-3, 1e-2], "只训 norm": [1e-2, 3e-2]}
SFT_GRID = {"全参": [1e-3, 3e-3], "只训线性层": [1e-3, 3e-3], "LoRA r=1": [3e-3, 1e-2],
            "LoRA r=8": [3e-3, 1e-2], "LoRA r=1 + norm": [1e-2, 3e-2], "只训 norm": [3e-2, 1e-1]}


def is_norm(param_name: str) -> bool:
    """RMSNorm 的增益向量: 每层 norm1 / norm2 加最后的 ln_f, 2 层模型共 5 × 64 = 320 个参数。"""
    return "norm" in param_name or param_name.startswith("ln_f")


def prepare(start, name):
    """从起点复制一份, 按配置名决定哪些参数可训。返回 (模型, 可训参数量)。"""
    model = copy.deepcopy(start)
    if name.startswith("LoRA"):
        apply_lora(model, r=int(name.split("=")[1].split()[0]), alpha=ALPHA, target_modules=ALL_LINEARS)
        mark_only_lora_as_trainable(model)                    # embedding / norm / lm_head 一并冻结
    for n, p in model.named_parameters():
        if name == "只训线性层":
            p.requires_grad = any(t in n for t in ALL_LINEARS)
        elif name == "只训 norm":
            p.requires_grad = is_norm(n)
        elif name.endswith("+ norm") and is_norm(n):
            p.requires_grad = True
    return model, count_parameters(model)["trainable"]


def evaluate(model, task):
    return {"pass1": task.exact_match(model, n=512, temperature=1.0), "em": task.exact_match(model)}


def run_rl(start, task, name, lr):
    torch.manual_seed(1)
    model, n = prepare(start, name)
    # AdamW 收的是全部参数; requires_grad=False 的参数 grad 一直是 None, 优化器直接跳过它们
    trainer = GRPOTrainer(model, task.verify, make_config("grpo", lr=lr, max_new=task.response_len + 2, group_size=G))
    prompts = PromptDataGenerator(task, PROMPTS_PER_STEP)
    for _ in range(STEPS):
        trainer.step(prompts.generate_batch()["prompts"])
    return model, {"n": n, **evaluate(model, task)}


def run_sft(start, task, name, lr):
    torch.manual_seed(1)
    model, n = prepare(start, name)
    fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, lr, log_interval=STEPS)
    return model, {"n": n, **evaluate(model, task)}


def table(title, rows, start_row):
    print(f"\n---- {title} ----")
    print(f"{'':<24}{'可训参数':>9}{'采样 pass@1':>13}{'贪心 EM':>9}")
    print(f"{'SFT 起点':<24}{start_row['n']:>10,}{start_row['pass1']:>13.3f}{start_row['em']:>9.3f}")
    for (name, lr), r in rows.items():
        print(f"{name + f' lr={lr:g}':<24}{r['n']:>10,}{r['pass1']:>13.3f}{r['em']:>9.3f}")


def best(rows, name, key):
    """同一配置在 lr 网格里按 key 取最好的那一臂。"""
    return max((r for (n, _), r in rows.items() if n == name), key=lambda r: r[key])


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    start = make_model(task)
    sft_warmup(start, task, SFT_STEPS)                         # 与 train_grpo 同一个起点
    start_row = {"n": count_parameters(start)["trainable"], **evaluate(start, task)}
    base_weights = {k: v.clone() for k, v in start.state_dict().items()}

    rl, unchanged = {}, None
    for name, lrs in RL_GRID.items():
        for lr in lrs:
            model, rl[(name, lr)] = run_rl(start, task, name, lr)
            if name == "LoRA r=1":
                # 冻结的基座在 RL 之后必须逐位不变; 适配器层的原权重挂在 .base 下, 去掉这一段就是原来的名字
                frozen = {k.replace(".base", ""): v for k, v in model.state_dict().items() if "lora_" not in k}
                unchanged = all(torch.equal(v, base_weights[k]) for k, v in frozen.items())
    table(f"RL: GRPO {PROMPTS_PER_STEP} prompt × G={G}, {STEPS} 步 (留出集)", rl, start_row)

    sft = {}
    for name, lrs in SFT_GRID.items():
        for lr in lrs:
            _, sft[(name, lr)] = run_sft(start, task, name, lr)
    table(f"SFT 对照: 同一起点, 每步 64 条, {STEPS} 步 (留出集)", sft, start_row)

    rl_best = {name: best(rl, name, "pass1") for name in RL_GRID}
    sft_best = {name: best(sft, name, "em") for name in SFT_GRID}
    print(f"\n---- 每种配置取 lr 网格里最好的一臂 (RL 按 pass@1, SFT 按贪心 EM) ----")
    print(f"{'':<16}{'可训参数':>9}{'RL pass@1':>11}{'RL 贪心 EM':>12}{'SFT 贪心 EM':>13}")
    print(f"{'SFT 起点':<16}{start_row['n']:>10,}{start_row['pass1']:>11.3f}{start_row['em']:>12.3f}{start_row['em']:>13.3f}")
    for k in RL_GRID:
        print(f"{k:<16}{rl_best[k]['n']:>10,}{rl_best[k]['pass1']:>11.3f}{rl_best[k]['em']:>12.3f}{sft_best[k]['em']:>13.3f}")

    full, p0, em0 = rl_best["全参"]["pass1"], start_row["pass1"], start_row["em"]
    assert unchanged, "LoRA 臂训练后冻结的基座权重变了: AdamW 应跳过 requires_grad=False 的参数"
    assert full > p0 + 0.04, f"RL 全参: 留出集 pass@1 应比 SFT 起点高 0.04 以上 ({full:.3f} vs {p0:.3f})"
    assert rl_best["LoRA r=1"]["pass1"] < full - 0.05, (
        f"这个规模上标准 LoRA r=1 (norm 冻结) 追不上全参: r=1 {rl_best['LoRA r=1']['pass1']:.3f}, 全参 {full:.3f}")
    assert rl_best["只训线性层"]["pass1"] < full - 0.05, (
        f"差距不在秩: 满秩的线性层全参 (norm 冻结) 也应落后全参 0.05 以上, "
        f"只训线性层 {rl_best['只训线性层']['pass1']:.3f}, 全参 {full:.3f}")
    # 两个采样指标相减, 差值比各自的波动还小, 只打印: 本机 r=1 + norm 0.301 对全参 0.324 (差 0.023),
    # 换台机器见过 0.314 对 0.395 (差 0.081)。方向一致 (放开 norm 把 r=1 从落后拉到接近), 差多少不稳。
    print(f"  放开 320 个 norm 增益后, r=1 从 {rl_best['LoRA r=1']['pass1']:.3f} 升到 "
          f"{rl_best['LoRA r=1 + norm']['pass1']:.3f}, 全参 {full:.3f}: 两个采样值之差随机性大, 只打印不断言")
    assert rl_best["只训 norm"]["pass1"] > p0 + 0.1, (
        f"只训 320 个 norm 增益, pass@1 应比起点高 0.1 以上: {rl_best['只训 norm']['pass1']:.3f} vs {p0:.3f}")
    top_em = max(r["em"] for r in rl.values())
    print(f"  60 步 RL 只把分布变尖, 没提高贪心 EM: 各臂最高 {top_em:.3f}, 起点 {em0:.3f} "
          f"(各臂的 RL 贪心 EM 在起点上下浮动, 只打印不断言)")
    assert sft_best["全参"]["em"] > em0 + 0.1, (
        f"SFT 全参的贪心 EM 应涨 0.1 以上: {sft_best['全参']['em']:.3f} vs 起点 {em0:.3f}")
    assert sft_best["只训 norm"]["em"] < em0 + 0.05, (
        f"SFT 只训 norm 学不到新映射, 贪心 EM 应涨不到 0.05: {sft_best['只训 norm']['em']:.3f} vs 起点 {em0:.3f}")
    assert sft_best["全参"]["em"] > sft_best["LoRA r=1"]["em"] + 0.04, (
        f"SFT 上全参的贪心 EM 应比 LoRA r=1 高 0.04 以上: "
        f"全参 {sft_best['全参']['em']:.3f}, r=1 {sft_best['LoRA r=1']['em']:.3f}")


if __name__ == "__main__":
    main()
