"""
M01 — 梯度累积 (Gradient Accumulation)

是什么: 大 batch 拆成 K 个 micro-batch 依次前向/反向, 梯度累加, 累够再 step 一次。
解决的瓶颈: 显存。激活峰值只按 micro-batch 计, 代价是 K 倍的串行时间, 不省通信也不省算力。
关键公式: loss 取 mean 时  g_full = Σ_k (n_k / N) · g_k   —— 权重是样本数占比, 不是 1。
读代码盯住: `accumulate` 里的 `scale=n / len(x)`; 漏掉它梯度放大 K 倍, 对 SGD 等于把学习率放大 K 倍。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, banner, kv, max_abs_diff
from llm_train.core.utils import add_inplace, zeros_like


def accumulate(model: LinearModel, x, y, sizes, weighted: bool = True):
    """按 sizes 切 micro-batch 并累积梯度。weighted=False 演示 "忘了缩放" 的错误写法。"""
    accum = zeros_like(model.params())               # 累积梯度, 与参数同结构, 全 0 起步
    start = 0                                        # 当前 micro-batch 在 x 里的起点
    for n in sizes:
        xb, yb = x[start : start + n], y[start : start + n]          # [n, d_in], [n, d_out]
        _, grads = model.loss_and_grads(xb, yb)                       # 每个 micro 的 loss 是自己 n 个样本的 mean
        # 权重 n/N: 每个 micro 的梯度是自己 n 个样本的平均, 乘回样本数占比才等于 N 个样本的平均
        add_inplace(accum, grads, scale=n / len(x) if weighted else 1.0)
        start += n
    return accum


def main() -> None:
    banner("M01 - Gradient Accumulation")

    rs = np.random.RandomState(0)
    model = LinearModel.init(d_in=5, d_out=3, seed=1)
    x = rs.randn(8, 5).astype(np.float32)                             # [N=8, d_in=5]
    y = rs.randn(8, 3).astype(np.float32)                             # [N=8, d_out=3]

    _, full = model.loss_and_grads(x, y)                              # 基线: 真的一次喂 8 个样本

    even_sizes, ragged_sizes = [2, 2, 2, 2], [3, 3, 2]                # 每个 micro-batch 的样本数, 和都是 8
    K = len(even_sizes)                                               # 累积步数
    even = accumulate(model, x, y, sizes=even_sizes)
    ragged = accumulate(model, x, y, sizes=ragged_sizes)              # 最后一个 micro 不满: 权重必须按样本数
    naive = accumulate(model, x, y, sizes=even_sizes, weighted=False)

    kv(f"max |full - accum({','.join(map(str, even_sizes))})|", f"{max_abs_diff(full, even):.2e}")
    kv(f"max |full - accum({','.join(map(str, ragged_sizes))})|", f"{max_abs_diff(full, ragged):.2e}")
    kv("忘记缩放: |naive| / |full|", f"{np.linalg.norm(naive['W']) / np.linalg.norm(full['W']):.2f}x")
    kv("激活峰值 (样本数)", f"{len(x)} -> {max(even_sizes)}")
    kv("optimizer.step 频率", f"每 {K} 个 micro-batch 一次")

    assert max_abs_diff(full, even) < 1e-6, "等长 micro-batch 的累积梯度必须等于大 batch 梯度"
    assert max_abs_diff(full, ragged) < 1e-6, "micro-batch 不等长时, 按样本数占比加权后也必须相等"
    assert np.allclose(naive["W"], K * full["W"], atol=1e-5), "不缩放 = 梯度放大 K 倍"
    print(f"\n  OK: 累积梯度 == 大 batch 梯度 (含不等长 micro-batch); 不缩放则恰好放大 K={K} 倍。")


if __name__ == "__main__":
    main()
