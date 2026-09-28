"""
m27 demo — 多副本路由: 轮询 / 最少负载 / 前缀感知, 以及热点前缀下的负载阈值兜底

    [1] 三种路由在"多轮对话 + 共享 system prompt"请求流上的前缀命中率、负载均衡度 (max/mean)、估算 TTFT
    [2] 纯前缀感知在热点前缀下的倾斜: 各副本分到的活
    [3] 扫负载阈值: 命中率 ↔ 均衡度 ↔ TTFT 的折中
TTFT 与负载全部来自代价模型 (FIFO 工作队列 + 固定 prefill/decode 速度), 不是实测; 命中率是真实跑 RadixCache 得到的。

运行: python -m llm_infer.m27_multi_replica_routing.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m27_multi_replica_routing.router import (
    simulate, chat_stream, PREFILL_TOK_S, DECODE_TOK_S, CACHE_TOKENS,
)

N_REPLICAS = 8
N_CONV, N_SYS, SYS_LEN, SPAN_S = 200, 4, 512, 400.0   # 对话数, system prompt 个数和长度, 到达时间跨度 (秒)
TURNS = (3, 8)                                   # 每段对话的轮数范围 (含两端)
THRESHOLDS = (4.0, 2.0, 1.0, 0.5, 0.2, 0.05)     # [3] 里要扫的负载阈值 (秒)
DEFAULT_THR = 1.0                                # 拿来和 least_load 对比的那个阈值


def row(name, r):
    """打印一种路由策略的一行结果: 命中率, miss, max/mean, TTFT 均值, TTFT p90。"""
    print(f"  {name:<22} {r['hit']:>7.1%} {1 - r['hit']:>7.1%} {r['balance']:>8.2f} "
          f"{r['ttft_mean']:>10.3f} {r['ttft_p90']:>10.3f}")


def main():
    banner("M27 - 多副本路由 (前缀感知 + 负载阈值)")
    reqs = chat_stream(N_CONV, N_SYS, SYS_LEN, SPAN_S, turns=TURNS)
    n_tok = sum(len(p) for _, p, _ in reqs)
    pop = 1.0 / np.arange(1, N_SYS + 1)                  # 与 chat_stream 里的流行度同一个公式
    kv("请求流", f"{N_CONV} 段对话 × {TURNS[0]}~{TURNS[1]} 轮 = {len(reqs)} 个请求, "
                f"{SPAN_S:g} s 内到达, prompt 共 {n_tok:,} tokens")
    kv("system prompt", f"{N_SYS} 个, 各 {SYS_LEN} tokens, 流行度 ∝ 1/rank "
                        f"({' / '.join(f'{x:.0%}' for x in pop / pop.sum())})")
    kv("副本", f"{N_REPLICAS} 个, 每个 RadixCache 容量 {CACHE_TOKENS:,} tokens (LRU 驱逐叶子)")
    kv("代价模型 (估算)", f"prefill {PREFILL_TOK_S:g} tok/s, decode {DECODE_TOK_S:g} tok/s, 每副本 FIFO 工作队列")

    print("\n[1] 三种路由")
    print(f"  {'策略':<20} {'命中率':>7} {'miss':>7} {'max/mean':>8} {'TTFT 均值':>10} {'TTFT p90':>10}  (秒, 估算)")
    rr = simulate(reqs, "round_robin", N_REPLICAS)
    ll = simulate(reqs, "least_load", N_REPLICAS)
    px = simulate(reqs, "prefix", N_REPLICAS)
    for name, r in (("round_robin", rr), ("least_load", ll), ("prefix (最长前缀匹配)", px)):
        row(name, r)
    assert px["hit"] > max(rr["hit"], ll["hit"]) + 0.25, \
        "前缀感知的命中率应比轮询和最少负载高 25 个百分点以上 (多轮历史只在处理过它的副本上是热的)"
    assert 1 - px["hit"] < 0.25 * (1 - ll["hit"]), "前缀感知下需要重算的 token 应不到最少负载的 1/4"
    assert ll["balance"] < 1.1, f"最少负载的 max/mean 应低于 1.1, 实际 {ll['balance']:.2f}"
    assert rr["balance"] < 1.1, f"轮询的 max/mean 应低于 1.1, 实际 {rr['balance']:.2f}"

    print("\n[2] 纯前缀感知的代价: 新对话都被已缓存 system prompt 的副本吸走")
    kv("各副本分到的活 (占比)", " ".join(f"{s:.0%}" for s in px["share"]))
    busy = int(np.sum(px["share"] > 0.01))
    kv("真正在干活的副本", f"{busy} / {N_REPLICAS}")
    assert px["balance"] > 3, f"纯前缀感知在热点前缀下应严重倾斜, max/mean 实际 {px['balance']:.2f}"
    assert busy <= N_SYS, f"只有 {N_SYS} 个 system prompt, 在干活的副本应不超过 {N_SYS} 个, 实际 {busy}"
    assert px["ttft_mean"] > 50 * ll["ttft_mean"], "热点副本过载后队列越排越长, TTFT 应比最少负载高 50 倍以上"

    print(f"\n[3] 负载阈值兜底: 最佳副本积压比最闲副本多 > threshold 秒, 就改走最少负载")
    print(f"  {'策略':<20} {'命中率':>7} {'miss':>7} {'max/mean':>8} {'TTFT 均值':>10} {'TTFT p90':>10}")
    res = {}
    for thr in THRESHOLDS:
        res[thr] = simulate(reqs, "prefix", N_REPLICAS, threshold_s=thr)
        row(f"prefix, thr={thr:g}s", res[thr])
    d = res[DEFAULT_THR]
    kv(f"thr={DEFAULT_THR:g}s vs least_load", f"命中率 {d['hit']:.1%} vs {ll['hit']:.1%}, TTFT {d['ttft_mean']:.3f} vs {ll['ttft_mean']:.3f} s "
                                            f"({1 - d['ttft_mean'] / ll['ttft_mean']:.0%} ↓)")
    assert d["balance"] < 1.5, f"thr={DEFAULT_THR:g}s: 加阈值后 max/mean 应低于 1.5, 实际 {d['balance']:.2f}"
    assert d["hit"] > ll["hit"] + 0.15, f"thr={DEFAULT_THR:g}s: 命中率应比最少负载高 15 个百分点以上"
    assert d["ttft_mean"] < 0.8 * ll["ttft_mean"], f"thr={DEFAULT_THR:g}s: TTFT 均值应比最少负载低 20% 以上"
    assert res[0.05]["hit"] < res[1.0]["hit"] < res[4.0]["hit"], "阈值越紧, 越常放弃缓存, 命中率应越低"
    assert res[0.05]["balance"] < res[4.0]["balance"] < px["balance"], "阈值越紧, 负载应越均衡"
    best = min(res, key=lambda t: res[t]["ttft_mean"])
    assert 0.05 < best < 4.0, f"TTFT 最优的阈值应在中间 (两头分别输给重算和排队), 实际 {best:g}"
    kv("TTFT 最优阈值", f"{best:g} s")

    print("\n  ✓ 前缀感知把 miss 从 {:.0%} 降到 {:.0%}; 但纯前缀感知在热点下只用 {} 个副本, TTFT 爆炸".format(
        1 - ll["hit"], 1 - px["hit"], busy))
    print(f"  ✓ 加负载阈值后 max/mean {d['balance']:.2f}, 命中率 {d['hit']:.1%}, TTFT 比最少负载低 {1 - d['ttft_mean'] / ll['ttft_mean']:.0%} (估算)")


if __name__ == "__main__":
    main()
