#!/usr/bin/env python
"""
DoRA vs LoRA: 同一个基座、同样的 r / lr / 步数 / 随机种子, 只换适配器类型。

    python -m llm_finetune.run_finetune.dora.train_dora

实验设计:
    基座和新任务同 train_lora: 会 copy 的基座, 适配到 sort。
    两臂: LoRALinear (对照组) / DoRALinear。r=8, α=2r, lr=1e-2, 300 步, 都注入全部 7 种线性层。
    每臂跑 3 个种子 (1, 2, 3), 比的是 3 次的平均: 单个种子上两者的差距可能落在噪声里。
    附带两项观察: 训练后长度向量 m 相对初值变了多少; 合并前后输出变没变。
断言 (每条验证一个结论):
    1. 注入后、训练前, 输出与基座相同 (误差 < 1e-5)。m 初始化为 ‖W‖_row 且 B = 0 ⇒ W' = W。
    2. DoRA 比 LoRA 多出的可训参数 = 被注入的各层 d_out 之和。每层只多一个长度向量 m。
    3. 合并前后 logits 最大差 < 1e-4。
    4. 3 个种子平均的末步 loss: DoRA 低于 LoRA。
依赖 llm_models: 用到属性路径 `model.layers[0].attn.w_q` 和 `base.lm_head`。
"""

import copy

import torch

from llm_finetune import (
    ALL_LINEARS, DoRALinear, InstructionDataGenerator, LoRALinear, SeqTask, SFTLoss, apply_lora,
    count_parameters, mark_only_lora_as_trainable, merge_lora_weights,
)
from llm_finetune.run_finetune.common import fit, pretrained_base

STEPS, RANK, LR, SEEDS = 300, 8, 1e-2, (1, 2, 3)


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    base = pretrained_base()
    probe = task.sample_prompts(8, "test")                    # [8, P] 固定输入, 用来比 "改动前后输出变没变"

    results = {}                                              # 类名 → (可训参数量, [(末步 loss, 留出集 EM) × 3 个种子])
    for cls in (LoRALinear, DoRALinear):
        runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = copy.deepcopy(base)
            apply_lora(model, r=RANK, alpha=2 * RANK, target_modules=ALL_LINEARS, layer_cls=cls)
            mark_only_lora_as_trainable(model)
            assert torch.allclose(model(probe), base(probe), atol=1e-5), (
                f"{cls.__name__} 注入改变了初始输出: m 初始化为 ‖W‖_row 且 B = 0 ⇒ W' = W, "
                "与 LoRA 一样是无害启动, 只允许浮点舍入级别的差 (1e-5)")
            hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
            runs.append((hist[-1]["total_loss"], task.exact_match(model)))
        results[cls.__name__] = (count_parameters(model)["trainable"], runs)

    print(f"\n{'':<12}{'可训参数':>10}{f'末步 loss ({len(SEEDS)} 个种子)':>26}{f'留出集 EM ({len(SEEDS)} 个种子)':>26}")
    mean = {}
    for name, (n, runs) in results.items():
        mean[name] = [sum(x) / len(x) for x in zip(*runs)]    # [平均末步 loss, 平均留出集 EM]
        print(f"{name:<12}{n:>10,}{'  '.join(f'{l:.3f}' for l, _ in runs):>26}{'  '.join(f'{e:.3f}' for _, e in runs):>26}")
    print(f"平均: LoRA loss {mean['LoRALinear'][0]:.3f} EM {mean['LoRALinear'][1]:.3f} | "
          f"DoRA loss {mean['DoRALinear'][0]:.3f} EM {mean['DoRALinear'][1]:.3f}")

    # 训练后 m 相对初值 ‖W₀‖ 的变化: DoRA 多出来的那个自由度到底动了多少
    # 这里的 model 是循环留下的最后一个: DoRA, 种子 3
    layer = model.layers[0].attn.w_q
    drift = (layer.lora_magnitude.detach() / layer.base.weight.norm(dim=1) - 1).abs().mean()
    print(f"layers.0.attn.w_q: 长度向量 m 平均变化 {float(drift):.1%}")

    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        gap = float((before - model(probe)).abs().max())

    extra = results["DoRALinear"][0] - results["LoRALinear"][0]
    # 基座里除 lm_head 之外的 nn.Linear, 正好就是 ALL_LINEARS 命中的那些层
    assert extra == sum(m.out_features for m in base.modules() if isinstance(m, torch.nn.Linear)
                        and m is not base.lm_head), f"DoRA 只比 LoRA 每层多 d_out 个参数, 实际多了 {extra}"
    assert gap < 1e-4, f"DoRA 合并前后输出应数值相等: logits 最大差 {gap:.1e}, 应 < 1e-4"
    assert mean["DoRALinear"][0] < mean["LoRALinear"][0], (
        f"同 r 下 DoRA 的平均 loss 应低于 LoRA: DoRA {mean['DoRALinear'][0]:.3f}, LoRA {mean['LoRALinear'][0]:.3f}")


if __name__ == "__main__":
    main()
