"""
权重初始化 — 让 "初始 loss ≈ ln V" 成立的那一行

问题: nn.Embedding 默认 N(0,1)。一旦 lm_head 与 embedding 共享权重 (weight tying),
      最后一层 norm 的输出 h 几乎指向输入 token 自己的 embedding, 范数 sqrt(D);
      它与 lm_head 里同一行 (范数也 ≈ sqrt(D)) 的内积 ≈ D。
      模型确信 "下一个 token 还是它自己", 初始 CE ≈ D, 而不是均匀猜测的 ln V
      (run_models 的 mini 配置实测: D=256 时 ≈ 250, D=128 时 ≈ 127, 见 train_gpt3.py)。
解法: GPT-2 起的惯例 —— 所有 Linear / Embedding 用 N(0, 0.02²), bias 置 0。
      此时初始 logits std ≈ 0.02·sqrt(D) ≪ 1, softmax 近似均匀, CE ≈ ln V。
·sqrt(D): embedding 每维 std 0.02, 乘 sqrt(D) 后每维 std ≈ 0.02·sqrt(D)。
      - 用 Sin-PE 的模型: 这让 embedding 与幅度 ~1 的位置编码同量级。
      - 用 RoPE 的模型: 没有加性位置编码要对齐, 效果是 embedding 相对各层残差分支的输出放大 sqrt(D) 倍。
      Mamba 不乘, 沿用官方实现。各模型的约定汇总在 models/__init__.py。

读代码时盯住: std —— Linear / Embedding 共用的一个初始化超参。
会覆盖专用初始化: 凡是用 Linear / Embedding 承载的专用初始值, 调用本函数后都被改写。
      - SSM 的 dt_proj: Mamba 调完本函数再调 reset_dt 补回
      - GatedDeltaNet 的 gate_alpha.bias = 2: Qwen3Next 调完本函数再补回
      - adaLN 的零初始化: DiT / MMDiT / VideoDiT 不调用本函数, 否则 adaLN-Zero 的恒等起点就没了
      - VQ 码本的均匀初始化: VAR 只对 Transformer 部分调用, 不碰 tokenizer
不归它管的初始化: ViT 的 pos_embed (trunc_normal 0.02)、Resampler 的 latents (N(0,1))
      都是裸 Parameter, 各自在定义处初始化。
"""

import torch.nn as nn


def init_weights(model: nn.Module, std: float = 0.02) -> nn.Module:
    """对 model 内所有 Linear / Embedding 做 N(0, std²) 初始化, 返回 model 便于链式调用。

    只碰 Linear / Embedding: RMSNorm 的 γ、Mamba 的 A_log、MoE 的 routing_bias 等
    不属于这两类模块的参数和 buffer 保持原样。tied 权重是同一个 Parameter, 重复初始化无害。
    用 Linear / Embedding 承载的专用初始化会被覆盖, 清单见文件头。
    """
    for m in model.modules():   # modules() 递归遍历所有子模块, 包括 model 自己
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
    return model
