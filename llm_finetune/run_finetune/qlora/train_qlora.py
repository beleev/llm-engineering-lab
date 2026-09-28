#!/usr/bin/env python
"""
QLoRA: 把基座的全部线性层压成 NF4, 看 (1) 整个模型省了多少 (2) 量化伤了多少能力 (3) 还能不能照常适配新任务。

    python -m llm_finetune.run_finetune.qlora.train_qlora

实验设计 (只有一个模型, 分四段, 每段拿 "这一段之前的自己" 当对照):
    0) 打包单测: 3×5 的随机矩阵, block_size=5 → 15 个 4-bit 索引。个数是奇数, 要补一个才凑满 8 个字节。
    1) 量化: 会 copy 的基座, 每个 block 的 7 个线性层全换成 NF4。对照 = 量化前的同一个模型。
       此刻 B = 0, LoRA 支路输出是 0, 测到的就是纯量化的影响。
    2) 适配: 在量化基座上训 LoRA 300 步, 新任务 sort。
    3) 合并: 反量化 → 加 ΔW → 普通 nn.Linear。对照 = 合并前的输出。
断言 (每条验证一个结论):
    1. 15 个索引打包成 8 字节, 反量化后形状不变, 相对误差 < 0.2。
    2. 可训参数全是 lora_ 开头的: 基座是 buffer, 拿不到梯度。
    3. 被量化的层数 = 7 × block 数 (4 个注意力投影 + 3 个 SwiGLU 矩阵)。
    4. 整模型压缩比 > 5×。只算被量化的层, 理论上限是 4 / 0.5625 = 7.1×; embedding 没量化, 所以到不了。
    5. NF4 基座在原任务 copy 上的留出集 EM > 0.9: 量化没有毁掉已有的能力。
    6. QLoRA 在 sort 上的留出集 EM > 0.2: 量化基座照样能适配新任务。
    7. 合并前后 logits 最大差 < 1e-4, 合并后模型里没有 NF4Linear。
依赖 llm_models: 用到属性路径 `model.layers[0].ffn.w_up`; apply_qlora 按属性名找层。
"""

import torch

from llm_finetune import (
    InstructionDataGenerator, NF4Linear, SeqTask, SFTLoss, apply_qlora, count_parameters,
    mark_only_lora_as_trainable, merge_lora_weights, nf4_dequantize, nf4_quantize, weight_bytes,
)
from llm_finetune.run_finetune.common import fit, pretrained_base

STEPS, RANK, LR = 300, 8, 1e-2


def main() -> None:
    torch.manual_seed(0)
    old_task, task = SeqTask("copy"), SeqTask("sort")
    model = pretrained_base()
    probe = task.sample_prompts(8, "test")

    # ---- 0) 打包的边界情况: block_size 为奇数 → 索引个数为奇数 ----
    w = torch.randn(3, 5)
    packed, scales = nf4_quantize(w, block_size=5)                    # 15 个索引 → 8 字节
    w_hat = nf4_dequantize(packed, scales, w.shape, block_size=5)
    assert packed.numel() == 8, f"15 个 4-bit 索引应补 1 个后打包成 8 字节, 实际 {packed.numel()} 字节"
    assert w_hat.shape == w.shape, f"反量化应还原成原形状 {tuple(w.shape)}, 实际 {tuple(w_hat.shape)}"
    assert (w_hat - w).norm() / w.norm() < 0.2, (
        f"奇数长度打包后反量化结果不对: 相对误差 {float((w_hat - w).norm() / w.norm()):.3f}, 应 < 0.2")

    # ---- 1) 量化全部线性层, 整模型对账 ----
    fp32_bytes, em_fp32 = weight_bytes(model), old_task.exact_match(model)   # 量化前的对照读数
    w_orig = model.layers[0].ffn.w_up.weight.detach().clone()                # 留一层的原始权重, 用来算量化误差
    apply_qlora(model, r=RANK, alpha=2 * RANK, block_size=64)
    mark_only_lora_as_trainable(model)

    n_q = sum(isinstance(m, NF4Linear) for m in model.modules())
    adapter_bytes = count_parameters(model)["trainable"] * 4          # 可训的只有 LoRA, fp32 每个参数 4 字节
    base_bytes = weight_bytes(model) - adapter_bytes                  # 剩下的 = NF4 基座 + 没量化的 embedding / norm
    rel_err = float((model.layers[0].ffn.w_up.base.weight - w_orig).norm() / w_orig.norm())
    em_nf4 = old_task.exact_match(model)                              # B = 0: 此刻测到的就是 "纯量化" 的影响
    print(f"量化了 {n_q} 个线性层; embedding / lm_head (共享) 与 RMSNorm 保持 fp32")
    print(f"整模型权重: fp32 {fp32_bytes / 1024:.0f} KiB → NF4 基座 {base_bytes / 1024:.0f} KiB "
          f"({fp32_bytes / base_bytes:.2f}×) + LoRA {adapter_bytes / 1024:.0f} KiB")
    print(f"单层相对量化误差 {rel_err:.1%}; 基座原任务 (copy) 留出集 EM: fp32 {em_fp32:.3f} → NF4 {em_nf4:.3f}")

    # ---- 2) 在量化基座上训练 LoRA ----
    hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
    em = task.exact_match(model)
    assert all(not p.requires_grad or "lora_" in n for n, p in model.named_parameters()), (
        "可训参数里混进了非 LoRA 参数: QLoRA 只该训 lora_A / lora_B")
    print(f"QLoRA 适配到 sort: loss {hist[0]['total_loss']:.3f} → {hist[-1]['total_loss']:.3f}, 留出集 EM {em:.3f}")

    # ---- 3) 合并: 反量化 → 加 ΔW → 普通 nn.Linear ----
    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        gap = float((before - model(probe)).abs().max())
    print(f"合并 (dequant → merge) 前后 logits 最大差 {gap:.1e}; 合并后模型回到 {weight_bytes(model) / 1024:.0f} KiB 的高精度权重")

    assert n_q == 7 * len(model.layers), (
        f"每个 block 应量化 4 个注意力投影 + 3 个 SwiGLU 矩阵, 共 {7 * len(model.layers)} 层, 实际 {n_q}")
    assert fp32_bytes / base_bytes > 5, (
        f"整模型压缩比应 > 5× (理论上限 4 / 0.5625 = 7.1×), 实际 {fp32_bytes / base_bytes:.2f}×")
    assert em_nf4 > 0.9, f"NF4 不应毁掉基座已有的能力: copy 任务留出集 EM {em_nf4:.3f}, 应 > 0.9"
    assert em > 0.2, f"QLoRA 没学到新任务: 留出集 EM {em:.3f}, 应 > 0.2"
    assert gap < 1e-4, f"合并 (dequant → merge) 前后输出应数值相等: logits 最大差 {gap:.1e}, 应 < 1e-4"
    assert not any(isinstance(m, NF4Linear) for m in model.modules()), "合并后模型里不应再有 NF4Linear"


if __name__ == "__main__":
    main()
