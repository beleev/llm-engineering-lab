"""
M27 — 容错训练: checkpoint 间隔、goodput、慢节点、SDC、弹性缩容

是什么: 把 "挂掉" 当常态来排训练计划: 多久存一次盘、慢节点吃掉多少时间、损坏怎么发现、少几张卡怎么接着跑。
解决的瓶颈: goodput (有效训练时间 / 墙钟时间)。卡越多整体 MTBF 越短, 不算一下就会在落盘和重跑之间两头亏。
关键公式: M_sys = M_gpu / N; Young tau = sqrt(2·delta·M); Daly tau = sqrt(2·delta·(M+R)) - delta
读代码盯住: 扫间隔那一段 —— goodput 曲线在最优点附近很平, 所以 Young 和 Daly 的差别常常无所谓。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, banner, kv, max_abs_diff
from llm_train.m27_fault_tolerance.fault_tolerance import (
    accumulate_grads, cluster_mtbf, daly_interval, flip_bit, goodput_model, grad_checksum,
    sdc_report, simulate_goodput, straggler_step_time, sweep_interval, young_interval,
)

HOUR = 3600.0


def main() -> None:
    banner("M27 - Fault Tolerance: checkpoint / goodput / straggler / SDC")

    # ---- 1) 卡越多, 整体 MTBF 越短 ----
    print("\n[1] 整体 MTBF = 单卡 MTBF / N  (单卡取 10 万小时)")
    gpu_mtbf = 100_000.0                                   # 小时
    for n in [8, 1024, 16384]:
        m = cluster_mtbf(gpu_mtbf, n)
        kv(f"{n:>5} 张卡的整体 MTBF", f"{m:.2f} 小时 = {m * 60:.0f} 分钟")
    assert cluster_mtbf(gpu_mtbf, 16384) * 16384 == gpu_mtbf, "整体 MTBF × N 必须还是单卡 MTBF"
    assert cluster_mtbf(gpu_mtbf, 1024) > cluster_mtbf(gpu_mtbf, 16384), "卡越多整体 MTBF 越短"

    # ---- 2) Young / Daly 的最优间隔, 与蒙特卡洛扫出来的最优点对照 ----
    print("\n[2] 最优 checkpoint 间隔 (整体 MTBF 6 小时, 重启恢复 10 分钟)")
    mtbf, restart, horizon = 6 * HOUR, 10 * 60.0, 200 * HOUR
    for dump_min, tag in [(10.0, "同步落盘 10 分钟"), (0.5, "异步快照 30 秒")]:
        dump = dump_min * 60.0
        y, d = young_interval(dump, mtbf), daly_interval(dump, mtbf, restart)
        grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]           # 小时
        sweep = sweep_interval([g * HOUR for g in grid], dump, mtbf, restart, horizon)
        best, best_gp = max(sweep, key=lambda kvp: kvp[1])
        kv(f"{tag}: Young / Daly 间隔", f"{y / 60:.0f} 分钟 / {d / 60:.0f} 分钟")
        kv(f"{tag}: 扫出来的最优间隔 / goodput", f"{best / 60:.0f} 分钟 / {best_gp:.1%}")
        kv(f"{tag}: goodput @ Daly 间隔",
           f"{simulate_at(d, dump, mtbf, restart, horizon):.1%}")
        for g in [0.25, 1.0, 6.0]:
            gp = dict((round(i / HOUR, 2), v) for i, v in sweep)[g]
            kv(f"{tag}: goodput @ {g * 60:.0f} 分钟", f"{gp:.1%}")

    # 异步快照把阻塞时间缩短, 最优间隔跟着变密, goodput 变高
    dump_sync, dump_async = 10 * 60.0, 30.0
    gp_sync = simulate_at(daly_interval(dump_sync, mtbf, restart), dump_sync, mtbf, restart, horizon)
    gp_async = simulate_at(daly_interval(dump_async, mtbf, restart), dump_async, mtbf, restart, horizon)
    kv("最优 goodput 同步 → 异步", f"{gp_sync:.1%} → {gp_async:.1%}")
    # 最优点的来历: 落盘占比 delta/tau 与重跑占比 tau/(2M) 在最优间隔上相等
    tau = young_interval(dump_sync, mtbf)
    kv("Young 间隔处 落盘占比 / 重跑占比", f"{dump_sync / tau:.3f} / {tau / (2 * mtbf):.3f}")
    assert abs(dump_sync / tau - tau / (2 * mtbf)) < 1e-9, \
        "Young 的最优间隔就是让这两笔开销相等的那个点"
    # 解析模型与蒙特卡洛对表 (网页实验台用的是解析式, 这里核对两者是同一件事)
    gaps = []
    for d in (dump_sync, dump_async):
        for tau_min in (15, 60, 180, 360):
            mc = simulate_goodput(tau_min * 60.0, d, mtbf, restart, horizon, np.random.RandomState(0))
            gaps.append(abs(mc - goodput_model(tau_min * 60.0, d, mtbf, restart)))
    kv("解析模型 vs 蒙特卡洛 最大偏差", f"{max(gaps):.4f}")
    assert max(gaps) < 0.01, f"两种算法应在 1 个百分点内一致, 实得 {max(gaps):.4f}"
    assert young_interval(dump_sync, mtbf) > young_interval(dump_async, mtbf), \
        "落盘越快, 最优间隔越短 (存得更勤更划算)"
    assert daly_interval(dump_sync, mtbf, restart) < young_interval(dump_sync, mtbf), \
        "Daly 的式子比 Young 多减了一个 delta, 在这组参数下给出更短的间隔"
    assert gp_async > gp_sync, "异步快照应提高最优 goodput"
    assert daly_interval(20 * HOUR, mtbf, restart) == float("inf"), \
        "落盘比 2(M+R) 还长时, Daly 的式子不再有意义"

    # ---- 3) 慢节点: 同步一步要等 N 个里最慢的那个 ----
    print("\n[3] 慢节点 (每个 rank 单步 = 1.0 + Exp(0.1), 4000 次抽样)")
    rng = np.random.RandomState(27)
    straggler = {n: straggler_step_time(n, rng) for n in [1, 8, 64, 1024]}
    kv("每步时间 (N = 1 / 8 / 64 / 1024)", " / ".join(f"{r['step']:.3f}" for r in straggler.values()))
    kv("被最慢的那个拖掉的比例", " / ".join(f"{r['loss']:.1%}" for r in straggler.values()))
    assert straggler[1024]["step"] > straggler[8]["step"] > straggler[1]["step"], \
        "N 越大, 最大值的期望越大"
    assert straggler[1]["loss"] < 0.01, "只有一个 rank 时没有 '等最慢的' 这回事"
    assert straggler[1024]["loss"] > 0.3, "1024 个 rank 时被拖掉的比例应超过 30%"

    # ---- 4) SDC: 副本间对梯度校验和 ----
    print("\n[4] 静默数据损坏 (SDC): 两个副本各算一遍, 对校验和")
    model = LinearModel.init(d_in=8, d_out=4, seed=27)
    rs = np.random.RandomState(270)
    x = rs.randn(32, 8).astype(np.float32)
    y = rs.randn(32, 4).astype(np.float32)
    _, clean = model.loss_and_grads(x, y)
    kv("同一份输入重算两遍的校验和差", f"{abs(grad_checksum(clean) - grad_checksum(model.loss_and_grads(x, y)[1])):.1e}")
    for bit, tag in [(30, "高位 (指数)"), (22, "中位 (尾数高位)"), (0, "最低位 (尾数末位)")]:
        dirty = dict(clean, W=flip_bit(clean["W"], 0, bit))
        r = sdc_report(clean, dirty, tol=1e-6)
        kv(f"翻 W[0,0] 的第 {bit} 位 {tag}", f"逐位相同 {r['bitwise']}, 最大差 {r['max_gap']:.3e}, 阈值 1e-6 下发现 {r['caught']}")
    high = sdc_report(clean, dict(clean, W=flip_bit(clean["W"], 0, 30)), tol=1e-6)
    low = sdc_report(clean, dict(clean, W=flip_bit(clean["W"], 0, 0)), tol=1e-6)
    assert not high["bitwise"] and not low["bitwise"], "任何一位翻掉, 逐位比较都发现得了"
    assert high["caught"], "指数位翻掉会让数值差很远, 阈值 1e-6 一定抓得到"
    assert not low["caught"], "尾数末位翻掉的差小于 1e-6, 阈值法抓不到"
    assert grad_checksum(clean) == grad_checksum(model.loss_and_grads(x, y)[1]), \
        "确定性重算: 同一个 rank 同一份输入, 校验和逐位相同"

    # ---- 5) 弹性缩容: 8 张卡掉到 6 张, 用梯度累积保持全局 batch ----
    print("\n[5] 弹性缩容: 全局 batch 48 不变, 8 张卡 → 6 张卡")
    xb = rs.randn(48, 8).astype(np.float32)
    yb = rs.randn(48, 4).astype(np.float32)
    g8 = accumulate_grads(model, xb, yb, n_rank=8, micro=6)        # 8 × 1 × 6
    g6 = accumulate_grads(model, xb, yb, n_rank=6, micro=4)        # 6 × 2 × 4
    g1 = accumulate_grads(model, xb, yb, n_rank=1, micro=48)       # 单卡整批
    kv("8 卡 vs 单卡整批 max|Δg|", f"{max_abs_diff(g8, g1):.2e}")
    kv("6 卡 (每卡累积 2 次) vs 8 卡 max|Δg|", f"{max_abs_diff(g8, g6):.2e}")
    kv("6 卡 vs 8 卡 逐位相同", np.array_equal(g8["W"], g6["W"]))
    assert max_abs_diff(g8, g6) < 1e-7, "重切之后的全局平均梯度在 fp32 噪声内与原来相同"
    assert max_abs_diff(g8, g1) < 1e-7, "两种切法都等于单卡整批"
    assert not np.array_equal(g8["W"], g6["W"]), \
        "但不是逐位相同: 求和顺序变了, 浮点加法不满足结合律"

    print("\n  OK: 最优间隔由落盘耗时和整体 MTBF 决定, 异步快照把两头的亏都压小;")
    print("      慢节点和 SDC 是另两笔账; 弹性缩容能接着训, 但接不回逐位一致 (求和顺序变了)。")


def simulate_at(interval: float, dump: float, mtbf: float, restart: float, horizon: float) -> float:
    """在给定间隔上重开一个随机源跑一次 goodput 模拟 (固定种子, 可复现)。"""
    from llm_train.m27_fault_tolerance.fault_tolerance import simulate_goodput
    return simulate_goodput(interval, dump, mtbf, restart, horizon, np.random.RandomState(0))


if __name__ == "__main__":
    main()
