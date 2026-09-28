"""
router.py — 多副本 serving 的请求路由: 轮询 / 最少负载 / 前缀感知 (最长前缀匹配 + 负载阈值兜底)

是什么: 同一个模型起 N 个引擎副本 (data parallel), 每个副本有自己的前缀缓存 (这里直接用 m05 的 RadixCache)。
    路由器决定每个请求去哪个副本。副本之间**不共享** KV: 同一段对话历史只在处理过它的副本上是热的。
解决的瓶颈: 多轮对话第 k 轮的 prompt = system prompt + 前 k−1 轮全部历史。轮询 / 最少负载把下一轮随手扔到
    别的副本, 那里没有这段历史 → 整段重新 prefill。前缀感知路由把请求送到"缓存里已有最长前缀"的副本。
代价: 热点前缀 (大家都用的 system prompt) 会把所有新对话吸到同一个副本 → 负载倾斜、排队。
    兜底: 最佳副本比最闲副本多积压超过 `threshold_s` 秒的活, 就放弃缓存, 改走最少负载 (SGLang router 的
    balance threshold 同一思路); 被分流过去的副本随后也缓存了热点前缀, 热点自然复制开。
读代码盯住: `route` 的三个分支, 以及 `prefix_len` 为什么是只读的 (路由探测不能刷新 LRU、不能 split 树)。
对应真实系统: SGLang Model Gateway (sgl-router) 的 cache_aware 策略 (每个 worker 一棵近似 radix tree + balance 阈值);
    vLLM production-stack / llm-d 的 prefix-aware routing; NVIDIA Dynamo 的 KV-aware router。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np

from llm_infer.m05_radix_cache.radix_tree import RadixCache

# 代价模型参数 (估算用, 不是实测): 取得偏慢, 排队效应才看得见。
# chat_stream 默认参数 + 8 个副本时, 利用率在 30~40%。
PREFILL_TOK_S = 1000.0        # 每个副本 prefill 速度 (tokens/s)
DECODE_TOK_S = 100.0          # 每个副本 decode 吞吐 (tokens/s)
CACHE_TOKENS = 24_000         # 每个副本前缀缓存容量 (tokens)


@dataclass
class Replica:
    """一个引擎副本: 自己的前缀缓存 + 一条工作队列的积压。"""
    cache: RadixCache = field(default_factory=RadixCache)
    backlog_s: float = 0.0    # 还没做完的活 (秒); 路由器眼里的"负载"
    last_t: float = 0.0       # 上次更新积压的时刻 (秒)
    work_s: float = 0.0       # 累计分到的活, 用来算负载均衡度


def prefix_len(cache: RadixCache, tokens: List[int]) -> int:
    """只读的最长前缀匹配: 与 RadixCache.match 相同的走法, 但不 split、不刷新 last_used。"""
    node, i = cache.root, 0
    while i < len(tokens):
        child = node.children.get(tokens[i])
        if child is None:
            return i
        n = 0                                                            # 这条边上匹配了几个 token
        for a, b in zip(child.edge_tokens, tokens[i:]):
            if a != b:
                return i + n
            n += 1
        if n < len(child.edge_tokens):                                   # tokens 在边中间用完了
            return i + n
        node, i = child, i + n
    return i


def route(policy: str, replicas: List[Replica], tokens: List[int], rr: int, threshold_s: float = float("inf")) -> int:
    """→ 这个请求该去的副本下标。rr: 请求序号, 轮询和平手时的轮转都靠它。

    threshold_s 只对 "prefix" 起作用; 默认 inf = 永不放弃缓存 (纯前缀感知)。
    """
    loads = [r.backlog_s for r in replicas]
    if policy == "round_robin":
        return rr % len(replicas)
    turn = [(i - rr) % len(replicas) for i in range(len(replicas))]     # 平手时轮转, 空闲期不全压给 0 号
    least = min(range(len(replicas)), key=lambda i: (loads[i], turn[i]))
    if policy == "least_load":
        return least
    assert policy == "prefix", f"未知路由策略 {policy!r}, 可选 round_robin / least_load / prefix"
    # 命中最长者优先, 平手 (比如都只命中 system prompt) 选负载低的
    best = max(range(len(replicas)), key=lambda i: (prefix_len(replicas[i].cache, tokens), -loads[i], -turn[i]))
    return least if loads[best] - loads[least] > threshold_s else best


def simulate(reqs, policy: str, n_replicas: int = 8, threshold_s: float = float("inf")) -> dict:
    """reqs: [(到达时刻 s, prompt tokens, 回复 tokens)] 按时间排序。

    每个副本是一条 FIFO 工作队列: 活 = miss 部分的 prefill + 回复的 decode, 按固定速度做完。
    TTFT (估算) = 到达时的积压 + 本请求 miss 部分的 prefill。
    """
    reps = [Replica() for _ in range(n_replicas)]
    hit = total = 0
    ttft = []
    for k, (t, prompt, reply) in enumerate(reqs):
        for r in reps:                                                   # 时间推进: 积压按 1 秒/秒 消化
            r.backlog_s, r.last_t = max(0.0, r.backlog_s - (t - r.last_t)), t
        rep = reps[route(policy, reps, prompt, k, threshold_s)]
        _, slots = rep.cache.match(prompt)                               # 真正使用: 刷新 LRU
        h = min(len(slots), len(prompt) - 1)                             # 命中数; 至少算最后 1 个 token 才有 logits
        prefill_s = (len(prompt) - h) / PREFILL_TOK_S
        ttft.append(rep.backlog_s + prefill_s)
        work = prefill_s + len(reply) / DECODE_TOK_S
        rep.backlog_s += work
        rep.work_s += work
        full = prompt + reply                                            # 回复的 KV 在 decode 时生成, 也进缓存
        rep.cache.insert(full, [0] * len(full))                          # 只关心命中长度, slot 号用占位
        over = rep.cache.total_tokens() - CACHE_TOKENS
        if over > 0:
            rep.cache.evict(over)                                        # LRU 驱逐叶子
        hit, total = hit + h, total + len(prompt)
    work = np.array([r.work_s for r in reps])                            # (n_replicas,)
    ttft = np.array(ttft)                                                # (请求数,)
    # balance = 最忙副本 / 平均; util = 总工作量 / (副本数 × 请求流的时间跨度)
    return dict(hit=hit / total, balance=float(work.max() / work.mean()), share=work / work.sum(),
                ttft_mean=float(ttft.mean()), ttft_p90=float(np.percentile(ttft, 90)),
                util=float(work.sum() / (n_replicas * reqs[-1][0])))


def chat_stream(n_conv=200, n_sys=4, sys_len=512, span_s=400.0, seed=0, turns=(3, 8)):
    """多轮对话 + 共享 system prompt: n_sys 个 system prompt, 流行度 ∝ 1/rank (头部是热点);
    每段对话 turns[0]~turns[1] 轮 (含两端), 每轮 prompt = system + 全部历史 + 新消息, 轮间思考时间 ~ Exp(15 s)。"""
    rs = np.random.RandomState(seed)
    systems = [list(rs.randint(0, 1 << 30, sys_len)) for _ in range(n_sys)]   # 1<<30: token 取值范围够大, 不会偶然撞前缀
    pop = 1.0 / np.arange(1, n_sys + 1)                                   # (n_sys,) 没归一化的流行度
    reqs = []
    for _ in range(n_conv):
        hist = list(systems[rs.choice(n_sys, p=pop / pop.sum())])
        t = rs.uniform(0, span_s)                                         # 这段对话第一轮的到达时刻
        for _ in range(rs.randint(turns[0], turns[1] + 1)):              # randint 不含上界, 所以 +1
            hist += list(rs.randint(0, 1 << 30, rs.randint(30, 80)))      # 用户消息
            reply = list(rs.randint(0, 1 << 30, rs.randint(80, 200)))
            reqs.append((t, list(hist), reply))
            hist += reply
            t += rs.exponential(15.0)
    return sorted(reqs, key=lambda r: r[0])
