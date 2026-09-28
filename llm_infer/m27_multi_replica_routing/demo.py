"""
m27 demo — 多副本路由: 轮询 / 最少负载 / 前缀感知, 以及热点前缀下的负载阈值兜底

    [1] 三种路由在"多轮对话 + 共享 system prompt"请求流上的前缀命中率、负载均衡度 (max/mean)、估算 TTFT
    [2] 纯前缀感知在热点前缀下的倾斜: 各副本分到的活
    [3] 扫负载阈值: 命中率 ↔ 均衡度 ↔ TTFT 的折中
TTFT 与负载全部来自代价模型 (FIFO 工作队列 + 固定 prefill/decode 速度), 不是实测; 命中率是真实跑 RadixCache 得到的。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m27_multi_replica_routing.router import (
    simulate, chat_stream, PREFILL_TOK_S, DECODE_TOK_S, CACHE_TOKENS,
)

N_REPLICAS = 8
THRESHOLDS = (4.0, 2.0, 1.0, 0.5, 0.2, 0.05)
DEFAULT_THR = 1.0


def row(name, r):
    print(f"  {name:<22} {r['hit']:>7.1%} {1 - r['hit']:>7.1%} {r['balance']:>8.2f} "
          f"{r['ttft_mean']:>10.3f} {r['ttft_p90']:>10.3f}")


def main():
    banner("M27 - Multi-replica routing (prefix-aware + load threshold)")
    reqs = chat_stream()
    n_tok = sum(len(p) for _, p, _ in reqs)
    kv("请求流", f"200 段对话 × 3~8 轮 = {len(reqs)} 个请求, 400 s 内到达, prompt 共 {n_tok:,} tokens")
    kv("system prompt", "4 个, 各 512 tokens, 流行度 ∝ 1/rank (48% / 24% / 16% / 12%)")
    kv("副本", f"{N_REPLICAS} 个, 每个 RadixCache 容量 {CACHE_TOKENS:,} tokens (LRU 驱逐叶子)")
    kv("代价模型 (估算)", f"prefill {PREFILL_TOK_S:g} tok/s, decode {DECODE_TOK_S:g} tok/s, 每副本 FIFO 工作队列")

    print("\n[1] 三种路由")
    print(f"  {'policy':<22} {'命中率':>7} {'miss':>7} {'max/mean':>8} {'TTFT 均值':>10} {'TTFT p90':>10}  (秒, 估算)")
    rr = simulate(reqs, "round_robin", N_REPLICAS)
    ll = simulate(reqs, "least_load", N_REPLICAS)
    px = simulate(reqs, "prefix", N_REPLICAS)
    for name, r in (("round_robin", rr), ("least_load", ll), ("prefix (最长前缀匹配)", px)):
        row(name, r)
    assert px["hit"] > max(rr["hit"], ll["hit"]) + 0.25                # 多轮历史只在处理过它的副本上是热的
    assert 1 - px["hit"] < 0.25 * (1 - ll["hit"])                      # 需要重算的 token 少 4 倍以上
    assert ll["balance"] < 1.1 and rr["balance"] < 1.1

    print("\n[2] 纯前缀感知的代价: 新对话都被已缓存 system prompt 的副本吸走")
    kv("各副本分到的活 (占比)", " ".join(f"{s:.0%}" for s in px["share"]))
    busy = int(np.sum(px["share"] > 0.01))
    kv("真正在干活的副本", f"{busy} / {N_REPLICAS}")
    assert px["balance"] > 3 and busy <= 4                             # 至多 4 个 system prompt → 至多 4 个副本
    assert px["ttft_mean"] > 50 * ll["ttft_mean"]                      # 热点副本过载, 队列越排越长

    print(f"\n[3] 负载阈值兜底: 最佳副本积压比最闲副本多 > threshold 秒, 就改走最少负载")
    print(f"  {'policy':<22} {'命中率':>7} {'miss':>7} {'max/mean':>8} {'TTFT 均值':>10} {'TTFT p90':>10}")
    res = {}
    for thr in THRESHOLDS:
        res[thr] = simulate(reqs, "prefix", N_REPLICAS, threshold_s=thr)
        row(f"prefix, thr={thr:g}s", res[thr])
    d = res[DEFAULT_THR]
    kv(f"thr={DEFAULT_THR:g}s vs least_load", f"命中率 {d['hit']:.1%} vs {ll['hit']:.1%}, TTFT {d['ttft_mean']:.3f} vs {ll['ttft_mean']:.3f} s "
                                            f"({1 - d['ttft_mean'] / ll['ttft_mean']:.0%} ↓)")
    assert d["balance"] < 1.5 and d["hit"] > ll["hit"] + 0.15 and d["ttft_mean"] < 0.8 * ll["ttft_mean"]
    assert res[0.05]["hit"] < res[1.0]["hit"] < res[4.0]["hit"]         # 阈值越紧, 越常放弃缓存
    assert res[0.05]["balance"] < res[4.0]["balance"] < px["balance"]   # ... 换来更均衡
    best = min(res, key=lambda t: res[t]["ttft_mean"])
    assert 0.05 < best < 4.0                                             # TTFT 最优在中间: 两头分别输给排队和重算
    kv("TTFT 最优阈值", f"{best:g} s")

    print("\n  ✓ 前缀感知把 miss 从 {:.0%} 降到 {:.0%}; 但纯前缀感知在热点下只用 {} 个副本, TTFT 爆炸".format(
        1 - ll["hit"], 1 - px["hit"], busy))
    print(f"  ✓ 加负载阈值后 max/mean {d['balance']:.2f}, 命中率 {d['hit']:.1%}, TTFT 比最少负载低 {1 - d['ttft_mean'] / ll['ttft_mean']:.0%} (估算)")


if __name__ == "__main__":
    main()
