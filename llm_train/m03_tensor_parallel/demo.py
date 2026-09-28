"""
M03 — 张量并行 (Megatron-style TP)

是什么: 把一层的权重矩阵切给 N 个 rank。MLP 的标准切法: W1 按列切, W2 按行切。
解决的瓶颈: 单层参数/激活放不下一张卡。代价是 **每层** 前向 1 次 + 反向 1 次 all-reduce,
           阻塞在关键路径上, 所以 TP 只在 NVLink 机内用 (通常 ≤ 8)。
关键公式: 前向  Y = Σ_r relu(X·W1_r)·W2_r          ← g 算子: 前向 all-reduce, 反向恒等
          反向  dX = Σ_r dZ_r · W1_rᵀ               ← f 算子: 前向恒等, 反向 all-reduce
读代码盯住: 两次 `all_reduce_sum` 的位置; relu 夹在列切和行切之间不需要通信 (逐元素)。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_reduce_sum, banner, comm, kv, max_abs_diff, relu


def dense_mlp_grads(x, target, W1, b1, W2, b2):
    """对照基线: 不切分的 MLP。x [B, D], target [B, O] → (loss, 各参数梯度, out [B, O])。

    W1 [D, H], b1 [H], W2 [H, O], b2 [O]。grads["x"] [B, D] 是传给上一层的梯度。
    """
    # ---- 前向 ----
    z = x @ W1 + b1                           # [B, D] @ [D, H] → [B, H]
    h = relu(z)                               # [B, H]
    out = h @ W2 + b2                         # [B, H] @ [H, O] → [B, O]
    diff = out - target                       # [B, O]
    loss = float(np.mean(diff * diff))
    # ---- 反向 ----
    d_out = (2.0 / diff.size) * diff          # [B, O], mean(diff²) 对 out 求导
    d_z = (d_out @ W2.T) * (z > 0)            # [B, O] @ [O, H] → [B, H]; relu: z ≤ 0 的位置梯度为 0
    grads = {
        "W2": h.T @ d_out, "b2": d_out.sum(0),    # [H, O], [O]
        "W1": x.T @ d_z, "b1": d_z.sum(0),        # [D, H], [H]
        "x": d_z @ W1.T,                      # [B, D] 传给上一层的梯度 —— TP 的反向通信就为它
    }
    return loss, grads, out


def main() -> None:
    banner("M03 - Tensor Parallel MLP")

    rs = np.random.RandomState(4)
    B, D, H, O, world = 3, 4, 8, 2, 2
    # B batch, D 输入维, H 中间维, O 输出维, world 卡数
    x = rs.randn(B, D).astype(np.float32)
    target = rs.randn(B, O).astype(np.float32)
    W1 = (rs.randn(D, H) * 0.1).astype(np.float32)
    b1 = (rs.randn(H) * 0.1).astype(np.float32)
    W2 = (rs.randn(H, O) * 0.1).astype(np.float32)
    b2 = np.zeros(O, dtype=np.float32)

    dense_loss, dense, dense_out = dense_mlp_grads(x, target, W1, b1, W2, b2)

    W1_s = np.split(W1, world, axis=1)        # 列切: [D, H] -> N × [D, H/N]
    b1_s = np.split(b1, world)                #        [H]    -> N × [H/N]
    W2_s = np.split(W2, world, axis=0)        # 行切: [H, O] -> N × [H/N, O]

    comm.reset()
    # ---- 前向: x 在每个 rank 上是完整副本 (f 算子前向 = 恒等) ----
    z_s = [x @ w + b for w, b in zip(W1_s, b1_s)]            # N × [B, H/N]
    h_s = [relu(z) for z in z_s]                             # 逐元素, 无需通信
    partial = [h @ w for h, w in zip(h_s, W2_s)]             # N × [B, O], 每份只是部分和
    tp_out = all_reduce_sum(partial)[0] + b2                 # g 算子; b2 只加一次, 不能每 rank 都加

    # ---- 反向: d_out 在每个 rank 上相同 (g 算子反向 = 恒等) ----
    diff = tp_out - target
    tp_loss = float(np.mean(diff * diff))
    d_out = (2.0 / diff.size) * diff                         # [B, O]
    gW2_s = [h.T @ d_out for h in h_s]                       # N × [H/N, O]
    d_z_s = [(d_out @ w.T) * (z > 0) for w, z in zip(W2_s, z_s)]   # N × [B, H/N]
    gW1_s = [x.T @ dz for dz in d_z_s]                       # N × [D, H/N]
    gb1_s = [dz.sum(0) for dz in d_z_s]                      # N × [H/N]
    dx_partial = [dz @ w.T for dz, w in zip(d_z_s, W1_s)]    # N × [B, D], 每份只含本 rank 那 H/N 列的贡献
    tp_dx = all_reduce_sum(dx_partial)[0]                    # f 算子反向: 不做这一步, 上一层拿到的梯度就是错的

    # 把各 rank 的梯度分片按切分方向拼回整块, 只为和 dense 对拍; 真实 TP 里各 rank 只更新自己那片
    tp = {
        "W1": np.concatenate(gW1_s, axis=1), "b1": np.concatenate(gb1_s),    # [D, H], [H]
        "W2": np.concatenate(gW2_s, axis=0), "b2": d_out.sum(0), "x": tp_dx,  # [H, O], [O], [B, D]
    }

    kv("loss: dense / TP", f"{dense_loss:.6f} / {tp_loss:.6f}")
    kv("输出 max|Δ|", f"{max_abs_diff(dense_out, tp_out):.2e}")
    for name in ("W1", "b1", "W2", "b2", "x"):
        kv(f"梯度 {name} max|Δ|", f"{max_abs_diff(dense[name], tp[name]):.2e}")
    kv("单 rank 的 dX 与真值差", f"{max_abs_diff(dense['x'], dx_partial[0]):.2e}  (未 all-reduce → 错)")
    kv("每 rank 权重", f"{W1_s[0].size + W2_s[0].size} / {W1.size + W2.size} 个参数")
    kv("通信量 (1 个 MLP 块, 前向+反向)", comm.summary())

    assert max_abs_diff(dense_out, tp_out) < 1e-6, "TP 输出必须等于不切分的 MLP"
    assert all(max_abs_diff(dense[k], tp[k]) < 1e-6 for k in dense), \
        "TP 的每个参数梯度和 dX 都必须等于不切分的 MLP"
    assert max_abs_diff(dense["x"], dx_partial[0]) > 1e-4, "局部 dX 不等于真 dX, 反向 all-reduce 不可省"
    assert comm.calls["all_reduce"] == 2, "一个 MLP 块恰好 2 次 all-reduce: 前向 1 次, 反向 1 次"
    print("\n  OK: 输出、全部参数梯度、dX 都与 dense 一致; 每个 MLP 块 = 前向 1 次 + 反向 1 次 all-reduce。")


if __name__ == "__main__":
    main()
