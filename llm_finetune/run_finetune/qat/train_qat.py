#!/usr/bin/env python
"""
QAT vs 训练后 RTN: 同一个训好的模型, 直接取整到低位宽会掉多少; 带着假量化再训一小段, 能拿回多少。

    python -m llm_finetune.run_finetune.qat.train_qat

实验设计:
    第一部分 STE 单测: 一个 [4, 8] 的权重, group_size=4。上游梯度 c 随机, loss = Σ ŵ·c。
        fake_quant 的梯度应该原样等于 c (直通); 直接对 round 求导, 只有每组 absmax 那个元素经 scale 拿到梯度, 其余全是 0。
    第二部分 训练后量化 (RTN): reverse 任务 SFT 600 步 (留出集 EM 1.000) 当 fp32 基线。
        对 7 种线性层做分组对称 RTN, 扫 bits ∈ {4, 3, 2} × group_size ∈ {8, 32, 64}, 报留出集贪心 EM。
        每个权重的存储 = bits + 16/group_size 位 (每组一个 fp16 scale): 组越小, scale 的开销越大。
    第三部分 QAT: 从 fp32 基线出发, 换成 FakeQuantLinear 再训 100 步 (lr=1e-3), 训完 RTN 部署, 报 EM。
        对照组: 同样 100 步、同样 lr, 但不加假量化, 训完再 RTN。两臂只差 "训练时前向有没有看到量化误差"。
        INT2 扫三个 group_size, INT4 只跑 g=32。
断言 (每条验证一个结论):
    1. STE: 影子权重的梯度形状与权重相同, 数值等于上游梯度; 直接对 round 求导, 非零梯度的个数恰好等于组数。
    2. INT4 RTN 在这个规模下几乎不掉点 (EM 降幅 ≤ 0.02): d_model=64 的小模型权重分布窄, 16 个格点够用。
    3. INT2 RTN (g=32) 掉点 > 0.5。
    4. INT2 下组越小 RTN 越准: g=8 的 EM > g=32 > g=64。
    5. QAT (INT2, g=32) 部署后的 EM 比直接 RTN 高 0.5 以上, 也比 "多训 100 步再 RTN" 的对照高 0.5 以上。
    6. QAT 训练最后的前向与 RTN 部署后的前向逐位一致 (logits 最大差 < 1e-5): 训练看到的就是部署的权重。
依赖 llm_models: 靠属性名找 7 种线性层 (lora.replace_linears)。
"""

import copy

import torch

from llm_finetune import InstructionDataGenerator, SeqTask, SFTLoss
from llm_finetune.methods.qat import apply_qat, fake_quant, quantize_dequantize, rtn_quantize_
from llm_finetune.run_finetune.common import fit, make_model, sft_warmup

FP_STEPS, QAT_STEPS, QAT_LR = 600, 100, 1e-3
BITS, GROUPS = (4, 3, 2), (8, 32, 64)


def ste_check() -> None:
    print("---- STE: 梯度怎么穿过取整 ----")
    torch.manual_seed(0)
    w = torch.randn(4, 8, requires_grad=True)
    c = torch.randn(4, 8)                                     # 上游梯度 ∂L/∂ŵ
    (fake_quant(w, 4, 4) * c).sum().backward()
    print(f"fake_quant: 前向值 = ŵ ({torch.equal(fake_quant(w, 4, 4), quantize_dequantize(w, 4, 4))}), "
          f"∂L/∂w 形状 {tuple(w.grad.shape)}, 与上游梯度最大差 {float((w.grad - c).abs().max()):.1e}")
    assert w.grad.shape == w.shape, f"STE 的梯度形状应与权重相同: {tuple(w.grad.shape)} vs {tuple(w.shape)}"
    assert torch.equal(w.grad, c), "STE 应把上游梯度原样传给影子权重 (round 当恒等函数)"

    naive = torch.randn(4, 8, requires_grad=True)
    (quantize_dequantize(naive, 4, 4) * c).sum().backward()   # 不用 STE, 直接对 round 求导
    n_live, n_groups = int((naive.grad != 0).sum()), naive.numel() // 4
    print(f"直接对 round 求导: ∂L/∂w 里只有 {n_live}/{naive.numel()} 个非零   ← 每组 absmax 那一个经 scale 拿到梯度, "
          f"其余的梯度在取整处断掉")
    # 不用 STE 时, ŵ = round(w/s)·s 只通过 s = max|w|/7 依赖 w: 每组只有 absmax 那一个元素有梯度
    assert n_live == n_groups, f"不用 STE 时只有每组 absmax 的元素有梯度: 非零 {n_live} 个, 组数 {n_groups}"


def main() -> None:
    ste_check()

    torch.manual_seed(0)
    task = SeqTask("reverse")
    fp = make_model(task)
    em_fp = sft_warmup(fp, task, FP_STEPS)
    print(f"\n---- 训练后量化 RTN: fp32 基线 (reverse, SFT {FP_STEPS} 步) 留出集 EM {em_fp:.3f} ----")
    rtn = {(b, g): task.exact_match(rtn_quantize_(copy.deepcopy(fp), b, g)) for b in BITS for g in GROUPS}
    print(f"{'':<8}" + "".join(f"{f'g={g}':>16}" for g in GROUPS))
    for b in BITS:                                            # 括号里是每个权重的平均位数: bits + 16/g
        print(f"INT{b:<5}" + "".join(f"{rtn[b, g]:>9.3f} ({b + 16 / g:.1f}b)" for g in GROUPS))

    print(f"\n---- QAT: 从 fp32 基线再训 {QAT_STEPS} 步 (lr={QAT_LR}), 训完 RTN 部署 ----")
    print(f"{'':<12}{'直接 RTN':>10}{'多训再 RTN':>12}{'QAT 再 RTN':>12}")
    qat, more = {}, {}
    for b, g in [(4, 32), (2, 8), (2, 32), (2, 64)]:
        torch.manual_seed(1)
        plain = copy.deepcopy(fp)                             # 对照: 同样的步数和 lr, 前向不加假量化
        fit(plain, InstructionDataGenerator(task), SFTLoss(), QAT_STEPS, QAT_LR, log_interval=QAT_STEPS)
        more[b, g] = task.exact_match(rtn_quantize_(plain, b, g))
        torch.manual_seed(1)
        model = apply_qat(copy.deepcopy(fp), b, g)
        fit(model, InstructionDataGenerator(task), SFTLoss(), QAT_STEPS, QAT_LR, log_interval=QAT_STEPS)
        probe = task.sample_prompts(8, "test")
        with torch.no_grad():
            before = model(probe)                             # 训练形态: FakeQuantLinear
            rtn_quantize_(model, b, g)                        # 部署形态: 普通 nn.Linear(ŵ)
            gap = float((model(probe) - before).abs().max())
        qat[b, g] = task.exact_match(model)
        print(f"INT{b} g={g:<5}{rtn[b, g]:>10.3f}{more[b, g]:>12.3f}{qat[b, g]:>12.3f}   (训练与部署 logits 最大差 {gap:.1e})")
        assert gap < 1e-5, f"QAT 训练时的前向应与 RTN 部署后一致: INT{b} g={g} logits 最大差 {gap:.1e}"

    assert em_fp - rtn[4, 32] <= 0.02, (
        f"INT4 RTN 在 d_model=64 的小模型上应几乎不掉点: fp32 {em_fp:.3f} → INT4 g=32 {rtn[4, 32]:.3f}")
    assert em_fp - rtn[2, 32] > 0.5, f"INT2 RTN (g=32) 应明显掉点: fp32 {em_fp:.3f} → {rtn[2, 32]:.3f}"
    assert rtn[2, 8] > rtn[2, 32], f"INT2 下组越小 RTN 越准: g=8 {rtn[2, 8]:.3f} 应 > g=32 {rtn[2, 32]:.3f}"
    assert rtn[2, 32] > rtn[2, 64], f"INT2 下组越小 RTN 越准: g=32 {rtn[2, 32]:.3f} 应 > g=64 {rtn[2, 64]:.3f}"
    assert qat[2, 32] > rtn[2, 32] + 0.5, f"QAT 应拿回 INT2 掉的点: QAT {qat[2, 32]:.3f} vs 直接 RTN {rtn[2, 32]:.3f}"
    assert qat[2, 32] > more[2, 32] + 0.5, (
        f"恢复来自假量化, 不是多训了 100 步: QAT {qat[2, 32]:.3f} vs 多训再 RTN {more[2, 32]:.3f}")


if __name__ == "__main__":
    main()
