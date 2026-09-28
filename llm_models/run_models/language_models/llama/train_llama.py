#!/usr/bin/env python
"""
LLaMA 训练示例 — 验证两件事, 都用 assert 写死:
    1. 初始 loss ≈ ln V: 刚初始化的模型应当 "均匀瞎猜"。偏离很多说明初始化有问题
       (用默认 N(0,1) embedding + weight tying 会得到 ~250)。
    2. loss 明显下降: 数据是 **固定的一个随机 batch** (SyntheticDataGenerator.fixed=True),
       随机 token 没有规律可学, 所以这里的 "下降" = 模型背下了这个 batch —— 只证明
       forward / backward / 优化器通路正确, 不代表学到了语言。
"""

import math

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import (
    Trainer, TrainingConfig, StandardLMLoss, DecoderOnlyDataGenerator,
)


def main():
    # 共 60 步, 前 5 步线性 warmup; 第 1 步、每 10 步、最后一步各记一条 metrics
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=32,
        num_steps=60, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = LLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_layers=2, max_len=128, dropout=0.0, num_kv_heads=2,
    )
    print(f"LLaMA Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # idx 是 [B, seq_len] 的随机 token, labels 是同一条序列右移一位 (位置 t 的目标 = 第 t+1 个 token)
    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    metrics = Trainer(model, cfg, data_gen, StandardLMLoss()).train()

    # 第 1 步的 loss 在任何更新之前算出, 就是未训练模型的 loss。
    # 另外 lr_lambda(0) = 0: 第 1 步的 optimizer.step 不改参数, 真正的更新从第 2 步开始
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    ln_v = math.log(vocab_size)                                        # 在 V 个 token 里均匀瞎猜的交叉熵
    print(f"初始 loss {first:.3f}  vs  ln V = {ln_v:.3f}   |   最终 loss {last:.3f}")
    # 0.5 是给随机初始化留的余量; 0.5 × first 要求 loss 至少降一半
    assert abs(first - ln_v) < 0.5, f"初始 loss {first:.2f} 应 ≈ ln V = {ln_v:.2f}"
    assert last < 0.5 * first, f"loss 未明显下降: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    main()
