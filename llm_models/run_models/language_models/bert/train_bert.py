#!/usr/bin/env python
"""
BERT MLM 训练: 初始 loss ≈ ln V; 只有被 mask 的位置贡献 loss; 能背下固定 batch

注意: 数据是固定的随机 token (无规律), "loss 下降" = 记忆, 只验证梯度链路。
"""

import math

import torch

from llm_models.models.language_models.bert import BERT
from llm_models.training import Trainer, TrainingConfig, MaskedLMLoss, MaskedLMDataGenerator


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=4, seq_len=32,
        num_steps=60, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    V = 500
    model = BERT(vocab_size=V, d_model=128, n_heads=4, num_layers=2, max_len=64, dropout=0.0)
    print(f"BERT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = MaskedLMDataGenerator(vocab_size=V, batch_size=cfg.batch_size, seq_len=cfg.seq_len)
    loss_fn = MaskedLMLoss()

    # 只有 label != -100 的位置有梯度: 对 logits 求导, 未选中位置的梯度必须恰为 0
    batch = data_gen.generate_batch()
    labels = batch.pop("labels")                                       # [B, T]; 没被选中的位置是 -100
    logits = model(**batch)                                            # [B, T, V]
    logits.retain_grad()                                               # logits 是中间结果, 不声明就不保留 .grad
    loss_fn.compute(logits, labels)["total_loss"].backward()
    g = logits.grad.abs().sum(-1)                                      # [B, T] 每个位置的梯度大小
    picked = labels != -100                                            # [B, T] 被选中做 MLM 的位置
    print(f"被选中做 MLM 的位置: {picked.sum().item()}/{picked.numel()} ({picked.float().mean():.1%})")
    assert (g[~picked] == 0).all(), "没被选中的位置 (label = -100) 对 logits 的梯度应恰为 0"
    assert (g[picked] > 0).all(), "被选中做 MLM 的位置应收到非零梯度"
    model.zero_grad()                                                  # 清掉这次探针反传留下的梯度, 不带进下面的训练

    # metrics[0] 是第 1 步的 loss: 它在任何参数更新之前算出, 就是未训练模型的 loss
    metrics = Trainer(model, cfg, data_gen, loss_fn).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(V):.3f}) -> 终态 {last:.3f}")
    assert abs(first - math.log(V)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last < 0.5 * first, "固定 batch 上 loss 应明显下降"


if __name__ == "__main__":
    main()
