#!/usr/bin/env python
"""
LoRA vs 全参微调: 同一个基座 (会 copy) 适配到新任务 (sort), 同样的步数, 留出集 exact-match 说话。

    python -m llm_finetune.run_finetune.lora.train_lora

实验设计:
    基座: common.pretrained_base(), 已经会 copy。新任务: sort。基座在 sort 上的留出集 EM 先打印出来当起点。
    三臂都从基座的 deepcopy 出发, 每臂开始前重置种子, 步数都是 300:
        全参 lr=3e-3    所有参数都训              ← 对照组
        LoRA lr=3e-3   与全参同一个 lr, 只训 A、B
        LoRA lr=1e-2   lr 调大到约 3 倍
    LoRA 注入全部 7 种线性层 (ALL_LINEARS), r=8, α=2r。
    最后对第三臂做合并: 适配器层换回普通 nn.Linear, 比合并前后的输出。
断言 (每条验证一个结论):
    1. 注入后、训练前, 输出与基座逐位相同。B = 0 ⇒ ΔW = 0。
    2. 合并前后 logits 最大差 < 1e-4, 合并后模型里没有 lora_ 参数。
    3. LoRA lr=1e-2 的留出集 EM > 0.2: 学到了新任务。
    4. LoRA lr=1e-2 的末步 loss 低于 LoRA lr=3e-3: LoRA 要用比全参更大的 lr。
    5. 全参的末步 loss 低于 LoRA: 同样步数下全参收敛更快。LoRA 省的是显存和存储。
依赖 llm_models: apply_lora 按属性名 (w_q、w_up …) 找层, 见 methods/lora.py 文件头。
"""

import copy

import torch

from llm_finetune import (
    ALL_LINEARS, InstructionDataGenerator, SeqTask, SFTLoss, apply_lora, count_parameters,
    get_lora_state_dict, mark_only_lora_as_trainable, merge_lora_weights,
)
from llm_finetune.run_finetune.common import fit, make_model, pretrained_base

STEPS, RANK = 300, 8
REAL_D = 4096                                                 # 7B 级模型的 d_model, 只用来对比可训比例


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    base = pretrained_base()
    probe = task.sample_prompts(8, "test")                    # [8, P] 固定的一小批输入, 用来比 "改动前后输出变没变"
    print(f"基座在新任务 sort 上的留出集 EM: {task.exact_match(base):.3f}")

    rows = {}                                                 # 名字 → (可训参数量, 末步 loss, 留出集 EM)
    for name, lr in [("全参 lr=3e-3", 3e-3), ("LoRA lr=3e-3", 3e-3), ("LoRA lr=1e-2", 1e-2)]:
        torch.manual_seed(1)
        model = copy.deepcopy(base)
        if name.startswith("LoRA"):
            apply_lora(model, r=RANK, alpha=2 * RANK, target_modules=ALL_LINEARS)
            mark_only_lora_as_trainable(model)
            assert torch.equal(model(probe), base(probe)), (
                "LoRA 注入改变了初始输出: B = 0 ⇒ 注入后、训练前的输出必须与基座逐位相同")
        hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, lr, log_interval=STEPS)
        rows[name] = (count_parameters(model)["trainable"], hist[-1]["total_loss"], task.exact_match(model))

    # Adam 为每个可训参数存 m、v 两个 fp32 = 8 字节, 所以下面是 n * 8 / 1024 KiB
    total = count_parameters(base)["total"]
    print(f"\n{'':<14}{'可训参数':>10}{'Adam 状态':>12}{'末步 loss':>11}{'留出集 EM':>11}")
    for name, (n, loss, em) in rows.items():
        print(f"{name:<14}{n:>10,}{n * 8 / 1024:>9.0f}KiB{loss:>11.3f}{em:>11.3f}")
    print(f"(基座共 {total:,} 参数; 2r/d = {2 * RANK}/{base.d_model} = {2 * RANK / base.d_model:.0%}: d_model 太小, 比例不好看。"
          f"d={REAL_D} 时同样的 r 只占 {2 * RANK / REAL_D:.1%})")

    # ---- 合并: 换回普通 nn.Linear, 输出必须数值相等 ----
    # 这里的 model 是循环留下的最后一臂 (LoRA lr=1e-2)
    adapter = get_lora_state_dict(model)
    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        after = model(probe)
    gap = float((before - after).abs().max())
    print(f"adapter {sum(v.numel() for v in adapter.values()) * 4 / 1024:.0f} KiB; 合并前后 logits 最大差 {gap:.1e}")

    full, lora_same_lr, lora = rows["全参 lr=3e-3"], rows["LoRA lr=3e-3"], rows["LoRA lr=1e-2"]
    # 元组下标: [1] = 末步 loss, [2] = 留出集 EM
    assert gap < 1e-4, f"合并后输出必须与合并前数值相等: logits 最大差 {gap:.1e}, 应 < 1e-4"
    assert not any("lora_" in n for n, _ in model.named_parameters()), "合并后不应再有 LoRA 参数"
    assert lora[2] > 0.2, f"LoRA 没学到新任务: 留出集 EM {lora[2]:.3f}, 应 > 0.2"
    assert lora[1] < lora_same_lr[1], (
        f"LoRA 需要比全参更大的 lr (B 从 0 起步, ΔW 的有效步长小): "
        f"lr=1e-2 的末步 loss {lora[1]:.3f} 应低于 lr=3e-3 的 {lora_same_lr[1]:.3f}")
    assert full[1] < lora[1], (
        f"同样步数下全参应收敛得更快 —— LoRA 的卖点是显存 / 存储, 不是速度: "
        f"全参末步 loss {full[1]:.3f}, LoRA {lora[1]:.3f}")


if __name__ == "__main__":
    main()
