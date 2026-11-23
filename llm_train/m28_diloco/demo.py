"""
M28 — DiLoCo: 本地多步 + 外层动量, 把同步次数降到 1/H

是什么: worker 各自连着跑 H 步再同步一次参数, 外层把 "旧参数 - 新平均参数" 当梯度走一步动量 SGD。
解决的瓶颈: 跨数据中心训练时, DDP 每步一次 all-reduce 的带宽和延迟撑不住。
关键数字: 同步次数 = 总步数 / H; 通信量同比例下降。
读代码盯住: H 很小时外层动量把每一步放大 lr_out/(1-mu) ≈ 7 倍, 反而走坏 —— 论文里 H 取几百。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, average_grad_trees, banner, bytes_of, comm, kv, max_abs_diff
from llm_train.m28_diloco.diloco import (
    D_IN, D_OUT, evaluate, holdout, run_ddp, run_diloco, run_streaming_diloco, worker_streams,
)

STEPS, LR_IN, WORLD = 960, 0.02, 4


def main() -> None:
    banner("M28 - DiLoCo: local steps + outer momentum")

    x_eval, y_eval = holdout()
    kv("留出集 MSE 的坏到底参照 (全 0 预测)", f"{float(np.mean(y_eval ** 2)):.3f}")

    # ---- 1) 退化检查: H=1 + 内层朴素 SGD + 外层 lr 1 无动量 == DDP 的一步 ----
    print("\n[1] 退化检查: H=1, 内层 SGD, 外层 lr=1 且无动量")
    sgd_steps, sgd_lr = 40, 0.05
    ddp_sgd = run_diloco(sgd_steps, 1, sgd_lr, lr_out=1.0, momentum=0.0, nesterov=False,
                         world=WORLD, inner="sgd")
    ref = run_diloco(sgd_steps, sgd_steps, sgd_lr, lr_out=1.0, momentum=0.0, nesterov=False,
                     world=WORLD, inner="sgd")
    plain = LinearModel.init(D_IN, D_OUT, seed=5)
    streams = worker_streams(WORLD)
    for _ in range(sgd_steps):                                 # 手写一遍 DDP: 平均梯度, 一步 SGD
        local = []
        for stream in streams:
            xb, yb = stream.next_batch()
            local.append(plain.loss_and_grads(xb, yb)[1])
        plain.apply_grads(average_grad_trees(local), sgd_lr)
    gap = max_abs_diff(plain.params(), ddp_sgd)
    kv("max |DDP - DiLoCo(H=1, 外层退化)|", f"{gap:.2e}")
    kv("max |DDP - DiLoCo(H=40, 同样的外层)|", f"{max_abs_diff(plain.params(), ref):.2e}")
    # theta - 1.0 * (theta - avg) 在 fp32 下不是逐位的 avg, 40 步攒下来是 1e-7 量级
    assert gap < 1e-6, f"H=1 且外层退化成恒等映射时每步都等于 DDP, 实得 {gap:.2e}"
    assert max_abs_diff(plain.params(), ref) > 1e-3, "H=40 时就不该再等于 DDP 了"

    # ---- 2) 同样的总步数, 扫 H ----
    print(f"\n[2] 同样 {STEPS} 步 (world={WORLD}, 内层 AdamW lr={LR_IN}, 外层 Nesterov lr=0.7 mu=0.9)")
    comm.reset()
    ddp_params = run_ddp(STEPS, LR_IN, world=WORLD)
    ddp_loss, ddp_bytes = evaluate(ddp_params, x_eval, y_eval), comm.total
    param_bytes = bytes_of(ddp_params)
    kv("模型大小 (W + b)", f"{param_bytes} B")
    kv("DDP: 留出集 MSE / 同步次数 / 通信", f"{ddp_loss:.2e} / {STEPS} / {int(ddp_bytes)} B/rank")

    results = {}
    for inner_steps in [1, 4, 16, 48, 120, 240]:
        comm.reset()
        params = run_diloco(STEPS, inner_steps, LR_IN, world=WORLD)
        results[inner_steps] = (evaluate(params, x_eval, y_eval), STEPS // inner_steps, comm.total)
        loss, syncs, nbytes = results[inner_steps]
        kv(f"DiLoCo H={inner_steps}: MSE / 同步次数 / 通信",
           f"{loss:.2e} / {syncs} / {int(nbytes)} B/rank ({ddp_bytes / nbytes:.0f}× 少)")

    best = min(results, key=lambda h: results[h][0])
    kv("最好的 H", f"{best} (MSE {results[best][0]:.2e}, 通信比 DDP 少 {ddp_bytes / results[best][2]:.0f}×)")
    assert results[16][2] * 16 == results[1][2], "通信量与 H 成反比"
    assert results[16][0] < 1e-6 and results[48][0] < 1e-6, \
        f"H=16 / 48 应该训得动 (实得 {results[16][0]:.2e} / {results[48][0]:.2e})"
    assert results[1][0] > results[16][0] * 1e6, \
        f"H=1 配论文的外层超参会走坏: 外层有效步长被动量放大, 实得 MSE {results[1][0]:.2e}"
    assert results[240][0] > results[48][0], "H 太大 (只同步 4 次) 时 MSE 回升"
    assert ddp_loss < results[best][0], "DDP 每步同步, 在这个玩具上仍然最准"

    # ---- 3) Streaming DiLoCo: 分片错峰同步 ----
    print("\n[3] Streaming DiLoCo: 参数按输出维分 2 / 4 份错峰同步 (H=48)")
    comm.reset()
    _ = run_diloco(STEPS, 48, LR_IN, world=WORLD)
    full_bytes, full_syncs = comm.total, STEPS // 48
    kv("整块同步: 同步次数 / 单次字节 / 总通信",
       f"{full_syncs} / {int(full_bytes / full_syncs)} / {int(full_bytes)} B/rank")
    for groups in [2, 4]:
        comm.reset()
        params = run_streaming_diloco(STEPS, 48, LR_IN, groups=groups, world=WORLD)
        syncs = STEPS // (48 // groups)
        kv(f"分 {groups} 份: MSE / 同步次数 / 单次字节 / 总通信",
           f"{evaluate(params, x_eval, y_eval):.2e} / {syncs} / {int(comm.total / syncs)} / {int(comm.total)} B/rank")
        assert abs(comm.total - full_bytes) < 1e-6, "错峰只改每次发多少, 总通信量不变"
        assert comm.total / syncs < full_bytes / full_syncs, "单次同步的字节数应降到 1/G"

    print("\n  OK: 通信次数与 H 成反比, 但 H 太小会被外层动量放大、太大会跟不上;")
    print("      Streaming 的分片错峰只降单次字节数 (峰值带宽), 总通信量和整块同步相同。")


if __name__ == "__main__":
    main()
