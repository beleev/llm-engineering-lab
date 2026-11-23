"""
m41 demo — 推理模型与 Agent 负载下的服务端 (全部数字来自代价模型和离散事件模拟, 是估算, 不是实测)

    [1] 思考 token 占输出的大头; 三种历史拼法下, 多轮工具循环逐次调用的前缀命中
    [2] effort (思考预算): 单请求延迟随输出线性涨, 显存约束下集群吞吐掉得比线性更快
    [3] 工具停顿期间的 KV: LRU 淘汰 / 钉住 / 固定 TTL / 按应用提示设 TTL, 3 个随机种子

运行: python -m llm_infer.m41_agentic_serving.demo
"""
from __future__ import annotations

import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m41_agentic_serving.agentic_sim import (
    POLICIES, AgentShape, DecodeCost, SimConfig, effort_row, history_hits, simulate)


def rj(s, w):
    """按显示宽度右对齐 (中文占两格)。"""
    return " " * max(0, w - len(s) - sum(ord(c) > 0x2E80 for c in s)) + s


def lj(s, w):
    """按显示宽度左对齐。"""
    return s + " " * max(0, w - len(s) - sum(ord(c) > 0x2E80 for c in s))


def main():
    banner("M41 - 推理模型与 Agent 负载的服务端 (估算)")

    # ---------------- [1] ---------------- #
    shape = AgentShape()
    print(f"\n[1] 一个 Agent 会话: {shape.turns} 轮用户发言, 每轮 {shape.steps} 次工具调用 + 1 次回答 "
          f"(思考 {shape.think}, 工具调用 {shape.call}, 工具结果 {shape.result}, 回答 {shape.answer} token)")
    res = {}
    print("  " + lj("历史里的思考块", 34) + rj("prompt 合计", 12) + rj("要算的 (miss)", 15) + rj("命中率", 9) + rj("最终上下文", 12))
    for p, label in POLICIES.items():
        rows, out_tok, think_tok, ctx = history_hits(shape, p)
        P, H = sum(r[0] for r in rows), sum(r[1] for r in rows)
        res[p] = dict(rows=rows, P=P, miss=P - H, ctx=ctx)
        print(f"  {lj(label, 34)}{P:>12,}{P - H:>15,}{H / P:>9.1%}{ctx:>12,}")
    kv("每次调用的输出里思考占比", f"{think_tok / out_tok:.1%}  ({think_tok:,} / {out_tok:,} token)")
    assert think_tok / out_tok > 0.8, "推理模型的输出应以思考 token 为主"
    n1 = shape.steps + 1                                             # 第 2 轮第一次调用的下标
    for p in POLICIES:
        print(f"  逐次命中率 ({POLICIES[p]}): " + " ".join(f"{h / P:.0%}" for P, h in res[p]["rows"]))
    boundary = {p: res[p]["rows"][n1][1] / res[p]["rows"][n1][0] for p in POLICIES}
    kv("第 2 轮第一次调用的命中率", " / ".join(f"{boundary[p]:.0%}" for p in POLICIES) + "  (全部保留 / 新用户消息时删 / 每次都删)")
    assert boundary["strip_on_user"] < 0.5, "新用户消息到来时删掉上一轮的思考, 上一轮的工具结果全部要重算"
    assert boundary["keep_all"] > 0.9, "全部保留时, 新一轮只需要算新的用户消息"
    assert res["strip_on_user"]["miss"] > res["keep_all"]["miss"], "删思考块改了历史, 要算的 token 更多"
    assert res["keep_all"]["ctx"] > res["strip_all"]["ctx"], "全部保留的代价: 上下文 (KV) 更长"
    # 在一个工具循环内部, 三种拼法的差别: 每次都删的那种, 上一次的工具调用 token 也要重算
    assert res["strip_all"]["miss"] > res["keep_all"]["miss"], "每次都删思考, 上一次的可见输出也要重算"

    # ---------------- [2] ---------------- #
    cost = DecodeCost()
    print(f"\n[2] effort: prompt 4000 + 可见输出 300, 思考预算 4 档 (代价模型: 每步读权重 {cost.weight_ms} ms + 读 KV, "
          f"KV 池 {cost.kv_pool_tokens:,} token)")
    print("  " + rj("思考", 7) + rj("输出", 8) + rj("单请求延迟", 12) + rj("最大并发", 10) + rj("集群 tok/s", 12) + rj("集群 请求/s", 13))
    rows = [effort_row(cost, 4000, 300, t) for t in (512, 2048, 8192, 32768)]
    for r in rows:
        print(f"  {r['think']:>7,}{r['out']:>8,}{r['solo_s']:>10.1f} s{r['batch']:>10}{r['tok_s']:>12.0f}{r['req_s']:>13.3f}")
    for a, b in zip(rows, rows[1:]):
        out_ratio, lat_ratio, thr_ratio = b["out"] / a["out"], b["solo_s"] / a["solo_s"], a["req_s"] / b["req_s"]
        assert lat_ratio < out_ratio * 1.1, "单请求延迟大致随输出长度线性增长"
        assert thr_ratio > out_ratio, "显存约束下并发变少, 集群请求吞吐下降得比输出变长更快"
    kv(f"思考 512 → 32768 (输出 ×{rows[-1]['out'] / rows[0]['out']:.1f})", f"单请求延迟 ×{rows[-1]['solo_s'] / rows[0]['solo_s']:.1f}, 集群请求吞吐 ÷{rows[0]['req_s'] / rows[-1]['req_s']:.0f}")

    # ---------------- [3] ---------------- #
    cfg = SimConfig()
    print(f"\n[3] 工具停顿期间的 KV: {cfg.n_sessions} 个会话 (每个 {cfg.steps} 次调用), KV 池 {cfg.kv_pool:,} token, "
          f"{cfg.slots} 个并发槽; 工具 {1 - cfg.p_slow:.0%} 快 (中位 {cfg.fast_s:g} s) / {cfg.p_slow:.0%} 慢 (中位 {cfg.slow_s:g} s)")
    names = {"evict": "LRU 淘汰", "pin": "钉住到回来", "ttl": f"固定 TTL {cfg.ttl_s:g} s", "hint": "按提示设 TTL"}
    print("  " + lj("策略", 16) + rj("重算 token", 12) + rj("平均 TTFT", 11) + rj("p90 TTFT", 10) + rj("平均完成时间", 14) + rj("被迫淘汰", 10))
    agg, runs = {}, {}
    for pol in names:
        runs[pol] = [simulate(SimConfig(seed=s), pol) for s in (0, 1, 2)]
        rr = runs[pol]
        agg[pol] = dict(re=np.mean([r.recompute_tokens for r in rr]), ttft=np.mean([np.mean(r.ttft_s) for r in rr]),
                        p90=np.mean([np.percentile(r.ttft_s, 90) for r in rr]), jct=np.mean([np.mean(r.jct_s) for r in rr]),
                        forced=np.mean([r.forced for r in rr]))
        a = agg[pol]
        print(f"  {lj(names[pol], 16)}{a['re']:>12,.0f}{a['ttft']:>9.1f} s{a['p90']:>8.1f} s{a['jct']:>12.0f} s{a['forced']:>10.0f}")
    for h, e in zip(runs["hint"], runs["evict"]):
        assert h.recompute_tokens < e.recompute_tokens, "逐种子: 按提示设 TTL 的重算应少于 LRU"
        assert np.mean(h.ttft_s) < np.mean(e.ttft_s), "逐种子: 少重算也让 prefill 队列变短, TTFT 应低于 LRU"
    kv("按提示设 TTL vs LRU", f"重算 ÷{agg['evict']['re'] / agg['hint']['re']:.2f}, 平均完成时间 {agg['hint']['jct']:.0f} vs {agg['evict']['jct']:.0f} s")
    assert agg["pin"]["ttft"] > 10 * agg["evict"]["ttft"], "全部钉住会占满 KV 池, 新调用排长队"
    assert agg["pin"]["forced"] > 0, "钉满之后只能被迫淘汰, 钉住并不能保证不重算"
    assert agg["ttl"]["re"] < agg["evict"]["re"], "固定 TTL 能少重算"
    assert agg["ttl"]["ttft"] > agg["evict"]["ttft"], "固定 TTL 把慢工具会话也钉了 8 s, TTFT 反而变差"
    assert agg["hint"]["ttft"] < agg["ttl"]["ttft"], "知道哪些是慢工具, 就不必白钉它们"
    assert agg["hint"]["jct"] < agg["evict"]["jct"], "按提示设 TTL 的平均完成时间应最短"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
