"""
m40 demo — Attention–FFN 分离部署 (AFD) 的代价模型

    [1] 一个配置的拆账: 注意力卡、FFN 卡、网络各花多少时间 (每层、每个 micro-batch)
    [2] A:F 实例比: 固定 16 张 FFN 卡, 扫注意力卡数; 每卡吞吐的最优点落在 t_a ≈ t_f; 上下文翻倍, 最优比例也翻倍;
        网络慢时 FFN 卡收发先到顶, 最优比例被压低
    [3] micro-batch 数: 1 个没有重叠, 2 个有气泡, 3 个把通信藏住; 再多, 每份太小, 权重被反复读, 吞吐下降
    [4] 什么时候不划算: 对照小规模同卡部署和大 EP 同卡部署, 扫 模型 × 上下文 × 网络 8 个场景
全部是代价模型的估算 (硬件和模型参数见 afd.py, 都是假设), 没有任何实测。

运行: python -m llm_infer.m40_af_disaggregation.demo
"""
from __future__ import annotations
from dataclasses import replace

from llm_infer.core.utils import banner, kv
from llm_infer.m40_af_disaggregation.afd import COARSE, FINE, HW, afd, colocated, with_sla

FAST, SLOW = 450e9, 50e9          # 每卡网络带宽 B/s: rack 级 NVLink 量级 / 400 Gb/s RDMA
SLA = 0.05                        # TPOT 上限 50 ms ([4] 用)
US = 1e6                          # 秒 → 微秒


def breakdown(hw):
    """[1] 粗粒度专家, 4K 上下文, 184 张注意力卡 + 16 张 FFN 卡, 3 个 micro-batch。"""
    m, n_a, n_f, n_mb = COARSE, 184, 16, 3
    r = afd(m, hw, n_a, n_f, 4096, n_mb)
    tokens = n_a * r["b"] / n_mb
    read = m.experts / n_f * m.expert_params / hw.bw
    print(f"\n[1] 拆账 (估算): {m.name}, 上下文 4096, {n_a} 张注意力卡 + {n_f} 张 FFN 卡, {n_mb} 个 micro-batch, 网络 {hw.net / 1e9:g} GB/s")
    kv("每张注意力卡的序列数 (显存塞满)", f"{r['b']:.0f}  → 每个 micro-batch 共 {tokens:.0f} 个 token")
    kv("t_a 注意力 (读权重 + 读 KV)", f"{r['t_a'] * US:.0f} µs")
    kv("t_f 专家 FFN (读权重 / 算, 取慢的)", f"{r['t_f'] * US:.0f} µs  (只读权重要 {read * US:.0f} µs)")
    kv("t_c 一个方向的 all-to-all", f"{r['t_c'] * US:.0f} µs")
    kv("TPOT / 每卡吞吐", f"{r['tpot'] * 1e3:.1f} ms / {r['tgs']:.0f} token/s")
    assert r["hidden"], "这个配置下通信应被 ping-pong 完全藏住"
    assert r["t_f"] > read, "184 张注意力卡汇聚的 token 应足以让 FFN 卡从读权重受限变成算力受限"


def ratio_sweep(hw):
    """[2] 固定 16 张 FFN 卡和 3 个 micro-batch, 扫注意力卡数。"""
    print(f"\n[2] A:F 实例比 (估算): {COARSE.name}, 16 张 FFN 卡, 3 个 micro-batch")
    best = {}
    for ctx, net in ((4096, FAST), (8192, FAST), (4096, SLOW)):
        h = replace(hw, net=net)
        rows = {n_a: afd(COARSE, h, n_a, 16, ctx, 3) for n_a in range(8, 513, 8)}
        n_opt = max(rows, key=lambda n: rows[n]["tgs"])
        best[ctx, net] = (n_opt, rows[n_opt])
        print(f"  上下文 {ctx}, 网络 {net / 1e9:g} GB/s:")
        for n_a in sorted({32, 64, 128, 256, 384, n_opt}):
            r = rows[n_a]
            mark = "  ← 最优" if n_a == n_opt else ""
            print(f"    {n_a:>3}A:16F  t_a {r['t_a'] * US:>4.0f} µs  t_f {r['t_f'] * US:>4.0f} µs  t_c {r['t_c'] * US:>5.0f} µs"
                  f"  每卡 {r['tgs']:>5.0f} token/s{mark}")
    (n4, r4), (n8, _), (ns, rs) = best[4096, FAST], best[8192, FAST], best[4096, SLOW]
    assert 0.8 < r4["t_f"] / r4["t_a"] < 1.25, f"最优点应落在 t_a ≈ t_f, 实际 t_f/t_a = {r4['t_f'] / r4['t_a']:.2f}"
    assert 1.7 < n8 / n4 < 2.3, f"上下文翻倍, 每张注意力卡的序列数减半, 最优注意力卡数应约翻倍: {n4} → {n8}"
    assert ns < n4 / 3, f"网络慢时 FFN 卡的收发先到顶, 最优注意力卡数应压到 1/3 以下: {n4} → {ns}"
    assert rs["tgs"] < 0.9 * r4["tgs"], "网络慢时最优点的每卡吞吐应明显更低"
    return n4


def microbatch_sweep(hw, n_a):
    """[3] 固定 A:F 比, 扫 micro-batch 数。"""
    print(f"\n[3] micro-batch 数 (估算): {COARSE.name}, 4K, {n_a}A:16F; MegaScale-Infer 的条件 m ≥ 2·(1 + t_c/t_f)")
    first_hidden = {}
    for net in (FAST, 100e9):
        h = replace(hw, net=net)
        rows = {m: afd(COARSE, h, n_a, 16, 4096, m) for m in range(1, 7)}
        print(f"  网络 {net / 1e9:g} GB/s:")
        for m, r in rows.items():
            need = 2 * (1 + r["t_c"] / max(r["t_a"], r["t_f"]))
            print(f"    m={m}  t_a {r['t_a'] * US:>4.0f}  t_f {r['t_f'] * US:>4.0f}  t_c {r['t_c'] * US:>4.0f} µs  "
                  f"需要 m ≥ {need:.1f}  {'藏住' if r['hidden'] else '不重叠' if m == 1 else '有气泡'}  每卡 {r['tgs']:>5.0f} token/s")
        first_hidden[net] = min((m for m, r in rows.items() if r["hidden"]), default=99)
        if net == FAST:
            best = max(rows, key=lambda m: rows[m]["tgs"])
            assert not rows[1]["hidden"] and not rows[2]["hidden"], "1 个 micro-batch 没有重叠, 2 个时通信仍有气泡"
            assert first_hidden[net] == 3 and best == 3, f"快网络下 3 个 micro-batch 应正好藏住通信且吞吐最高, 实际最优 m={best}"
            assert rows[6]["tgs"] < 0.85 * rows[3]["tgs"], "micro-batch 切太碎, 每份都要重读权重, 吞吐应明显下降"
    assert first_hidden[100e9] > first_hidden[FAST], "网络越慢, 藏住通信需要的 micro-batch 越多"


def scenarios(hw):
    """[4] 模型 × 上下文 × 网络: AFD 最优配置 vs 16 卡同卡部署 vs 大 EP 同卡部署 (32~256 卡里最好的)。"""
    print(f"\n[4] 什么时候不划算 (估算, TPOT ≤ {SLA * 1e3:.0f} ms): 每卡吞吐 token/s")
    print(f"  {'模型':<16} {'上下文':>6} {'网络':>6} {'同卡 16':>7} {'大 EP 同卡':>9} {'AFD 最优':>8} {'配置':>12} {'AFD/大EP':>8}")
    out = {}
    for m in (FINE, COARSE):
        for ctx in (4096, 32768):
            for net in (SLOW, FAST):
                h = replace(hw, net=net)
                small = with_sla(colocated, SLA, m, h, 16, ctx)["tgs"]
                big = max(with_sla(colocated, SLA, m, h, n, ctx)["tgs"] for n in (32, 64, 128, 256))
                cands = [(with_sla(afd, SLA, m, h, n_a, n_f, ctx, k), n_a, n_f, k)
                         for n_f in (16, 32) if m.layers * m.experts * m.expert_params / n_f <= h.mem
                         for n_a in range(8, 257, 8) for k in (2, 3, 4)]
                r, n_a, n_f, k = max(cands, key=lambda c: c[0]["tgs"])
                out[m.name, ctx, net] = (small, big, r["tgs"])
                print(f"  {m.name:<14} {ctx:>7} {net / 1e9:>5.0f}G {small:>8.0f} {big:>9.0f} {r['tgs']:>9.0f}"
                      f"   {n_a:>3}A{n_f}F m={k} {r['tgs'] / big:>7.2f}x")
    for key, (small, big, a) in out.items():
        assert a > 1.2 * small, f"{key}: 对照只有 16 卡的同卡部署时, AFD 应至少快 20%"
        assert a < big, f"{key}: 这个代价模型里, 同构硬件上 AFD 不应超过调好的大 EP 同卡部署"
    for m in (FINE, COARSE):
        small, big, a = out[m.name, 4096, SLOW]
        assert a < 0.8 * big, f"{m.name}: 短上下文 + 慢网络时 AFD 应比大 EP 差 20% 以上"
    return out


def main():
    banner("M40 - Attention–FFN 分离部署 (代价模型, 全部是估算)")
    hw = replace(HW(), net=FAST)
    kv("硬件假设 (每卡)", f"显存带宽 {hw.bw / 1e12:g} TB/s, FP8 {hw.flops / 1e15:g} PFLOPS, 可用显存 {hw.mem / 1e9:g} GB")
    for m in (FINE, COARSE):
        kv(f"模型假设: {m.name}", f"{m.layers} 层, 每专家 {m.expert_params / 1e6:.0f}M 参数, 全部专家 "
                                 f"{m.layers * m.experts * m.expert_params / 1e9:.0f} GB (FP8)")
    breakdown(hw)
    n4 = ratio_sweep(hw)
    microbatch_sweep(hw, n4)
    scenarios(hw)
    print("\n  ✓ A:F 比的最优点在两边耗时相等处; 上下文变长只需加注意力卡; 网络慢时比例被压低")
    print("  ✓ 3 个 micro-batch 藏住通信, 切得更碎反而变慢")
    print("  ✓ AFD 比小规模同卡部署快得多, 但在这个代价模型里赢不了调好的大 EP 同卡部署")


if __name__ == "__main__":
    main()
