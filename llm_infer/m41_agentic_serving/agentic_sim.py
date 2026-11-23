"""
agentic_sim.py — 推理模型与 Agent 负载下的服务端: 思考 token、工具停顿、前缀命中

是什么: 推理模型每次调用先写一段思考 (thinking) 再给工具调用或回答; Agent 一个任务要来回调用很多次工具,
      每次调用的 prompt 都是「整段历史 + 新的工具结果」, 两次调用之间会话停在工具上等结果。
瓶颈:   1. 历史怎么拼决定前缀缓存能不能命中: 删掉之前的思考块等于改了历史, 被删位置之后的 token 全部要重算。
      2. 工具调用期间这条会话的 KV 留不留: 留着占显存, 别的请求进不来; 丢掉, 回来时整段重算。
      3. 思考预算 (effort) 让每个请求的输出变长: 单请求延迟按输出长度线性涨, 集群吞吐涨得更快地往下掉,
         因为 KV 占满显存后能同时跑的请求数也跟着变少。
做法:   - history_hits: 三种历史拼法下逐次调用算最长公共前缀, 命中多少是数出来的, 不是公式套的。
      - effort_table: 解码代价模型 (读权重 + 读 KV), 算单请求延迟和显存约束下的集群吞吐。
      - simulate: 离散事件模拟, 一个 KV 池 + 若干并发槽, 比较工具停顿期间 KV 的四种处理: 直接可淘汰 (LRU)、
        钉住到会话回来、按 TTL 钉住、按应用给的工具类型提示决定钉不钉。
简化:   所有时间和吞吐来自代价模型, 不是实测; 解码每 token 的时间不随批大小变 (simulate 里), 也不建模 prefill 和
      decode 抢同一块 GPU; 会话之间不共享前缀; 删思考块对模型答题质量的影响不在模拟范围内。
真实系统: Qwen3 / DeepSeek-V3.2 的对话模板在新用户消息到来时删掉之前的思考; Anthropic 较新的模型默认保留;
      Continuum (arXiv 2511.02230) 给工具调用期间的 KV 设 TTL; Leyline (arXiv 2606.01065) 让应用直接编辑缓存里的一段。
盯住:   render() 里思考块什么时候被跳过; simulate() 里 _free() 先淘汰谁、pinned 什么时候解除。
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# 1. 历史怎么拼: 三种处理思考块的方式, 逐次调用数前缀命中                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgentShape:
    """一个 Agent 会话的形状 (token 数)。"""
    system: int = 2000          # system prompt + 工具定义, 假设已被别的会话缓存
    user: int = 100             # 每条用户消息
    think: int = 1200           # 每次调用的思考
    call: int = 60              # 工具调用 (可见输出)
    result: int = 800           # 工具结果
    answer: int = 300           # 每轮结束时的最终回答
    turns: int = 2              # 用户发言几轮
    steps: int = 5              # 每轮里调用几次工具


POLICIES = {
    "keep_all": "全部保留",
    "strip_on_user": "新用户消息到来时删掉之前的思考",
    "strip_all": "每次都删掉之前的思考",
}


def build_session(shape: AgentShape):
    """生成会话的块序列。每块是 (类型, 所在用户轮, token id 列表); token id 全局唯一, 公共前缀只能来自同一段内容。"""
    ids = itertools.count()
    blk = lambda kind, turn, n: (kind, turn, [next(ids) for _ in range(n)])
    out = [blk("system", -1, shape.system)]
    calls = []                                       # 每次模型调用之前的块数, 和它产出的块
    for t in range(shape.turns):
        out.append(blk("user", t, shape.user))
        for s in range(shape.steps + 1):
            last = s == shape.steps
            produced = [blk("think", t, shape.think), blk("answer" if last else "call", t, shape.answer if last else shape.call)]
            calls.append((len(out), produced))
            out += produced
            if not last:
                out.append(blk("result", t, shape.result))
    return out, calls


def render(blocks, policy: str, cur_turn: int) -> List[int]:
    """把块序列拼成 prompt。cur_turn = 当前调用所在的用户轮。"""
    toks = []
    for kind, turn, t in blocks:
        if kind == "think" and policy == "strip_all":
            continue                                 # 一律不带之前的思考
        if kind == "think" and policy == "strip_on_user" and turn < cur_turn:
            continue                                 # ★ 只删更早用户轮的思考: 当前工具循环里的照留
        toks += t
    return toks


def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    diff = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(diff[0]) if len(diff) else n


def history_hits(shape: AgentShape, policy: str):
    """逐次调用: prompt 长度、命中 (与之前任何一次完整序列的最长公共前缀)、miss。缓存不淘汰。
    → (每次调用 [(prompt, hit)], 输出 token 数, 其中思考 token 数, 最后一次调用的上下文长度)。"""
    blocks, calls = build_session(shape)
    cached = [blocks[0][2]]                          # system prompt 已缓存
    rows, out_tok, think_tok = [], 0, 0
    for start, produced in calls:
        turn = blocks[start - 1][1]
        prompt = render(blocks[:start], policy, turn)
        hit = max(_lcp(prompt, c) for c in cached)
        rows.append((len(prompt), hit))
        gen = [x for _, _, t in produced for x in t]
        cached.append(prompt + gen)                  # 解码出来的 token 也进缓存 (vLLM / SGLang 都这么做)
        out_tok += len(gen)
        think_tok += len(produced[0][2])
    return rows, out_tok, think_tok, len(prompt) + len(gen)


# --------------------------------------------------------------------- #
# 2. effort: 思考预算对单请求延迟和集群吞吐的影响 (代价模型)                #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class DecodeCost:
    """解码一步 = 读一遍权重 + 读批里所有请求的 KV。参数取 8B 级模型、单卡、fp16 的量级, 不是实测。"""
    weight_ms: float = 8.0               # 16 GB 权重 / 2 TB/s
    kv_bytes_per_token: int = 131072     # LLaMA-3-8B fp16, 与 m20 相同
    hbm_GBps: float = 2000.0
    kv_pool_tokens: int = 300_000        # 约 40 GB 给 KV
    prefill_tok_per_s: float = 8000.0
    max_batch: int = 256

    def step_ms(self, batch: int, ctx_sum: float) -> float:
        return self.weight_ms + ctx_sum * self.kv_bytes_per_token / (self.hbm_GBps * 1e9) * 1e3


def effort_row(cost: DecodeCost, prompt: int, visible: int, think: int) -> Dict[str, float]:
    """一档思考预算 → 单请求延迟 (独占一张卡) 和集群稳态吞吐 (显存允许的最大并发)。"""
    out = think + visible
    avg_ctx = prompt + out / 2                       # 解码过程中的平均上下文长度
    solo_s = prompt / cost.prefill_tok_per_s + out * cost.step_ms(1, avg_ctx) / 1e3
    batch = max(1, min(cost.max_batch, cost.kv_pool_tokens // (prompt + out)))   # ★ 按最长上下文预留 KV
    tok_s = batch / (cost.step_ms(batch, batch * avg_ctx) / 1e3)
    return dict(think=think, out=out, solo_s=solo_s, batch=batch, tok_s=tok_s, req_s=tok_s / out)


# --------------------------------------------------------------------- #
# 3. 工具停顿期间的 KV: 离散事件模拟                                      #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimConfig:
    n_sessions: int = 120
    arrival_per_s: float = 0.15
    steps: int = 8                       # 每个会话调用几次模型 (之间是工具)
    first_prompt: int = 3000
    new_tokens: int = 600                # 每次带回来的工具结果
    out_tokens: int = 500                # 每次输出 (思考 + 工具调用)
    p_slow: float = 0.25                 # 慢工具 (跑测试、等人确认) 的比例
    fast_s: float = 2.0                  # 快工具耗时中位数
    slow_s: float = 40.0                 # 慢工具耗时中位数
    kv_pool: int = 120_000               # KV 池容量 (token)
    slots: int = 16                      # 同时在跑的调用数上限
    prefill_tok_per_s: float = 8000.0
    tpot_ms: float = 20.0                # 每个输出 token 的时间 (不随批大小变, 简化)
    ttl_s: float = 8.0                   # TTL 策略: 停顿后钉住多久
    seed: int = 0


@dataclass
class SimResult:
    recompute_tokens: int = 0            # 被淘汰后重新 prefill 的历史 token
    ttft_s: List[float] = field(default_factory=list)      # 每次调用从工具返回 (或会话到达) 到首 token: 排队 + prefill
    jct_s: List[float] = field(default_factory=list)       # 每个会话从到达到结束
    evictions: int = 0
    forced: int = 0                      # 被迫淘汰的钉住会话数 (pin 的兜底)
    peak_pinned: int = 0                 # 停在工具上、被钉住的 KV 峰值 (token)


def simulate(cfg: SimConfig, policy: str) -> SimResult:
    """policy ∈ {"evict", "pin", "ttl", "hint"}。

    evict: 停顿时 KV 留在池里但随时可淘汰 (LRU), 等于引擎的默认前缀缓存。
    pin  : 停顿期间钉住, 会话回来或结束才放。
    ttl  : 停顿后钉住 ttl_s 秒, 过期变成可淘汰。
    hint : 应用告诉引擎这次是快工具还是慢工具; 快的按它的典型耗时设 TTL, 慢的立刻可淘汰且优先淘汰。
    """
    rng = np.random.RandomState(cfg.seed)
    res = SimResult()
    ev, seq = [], itertools.count()                  # 事件堆: (时间, 序号, 类型, 会话)
    t_arr = np.cumsum(rng.exponential(1 / cfg.arrival_per_s, cfg.n_sessions))
    # 每个会话每次工具调用的快慢和耗时预先抽好, 四种策略看到的是同一份负载
    slow = rng.rand(cfg.n_sessions, cfg.steps) < cfg.p_slow
    dur = np.where(slow, cfg.slow_s, cfg.fast_s) * rng.lognormal(0, 0.5, (cfg.n_sessions, cfg.steps))
    sess = [dict(arrive=t_arr[i], step=0, ctx=0, resident=0, pinned_until=-1.0, pause_at=0.0, slow=False)
            for i in range(cfg.n_sessions)]
    for i in range(cfg.n_sessions):
        heapq.heappush(ev, (t_arr[i], next(seq), "ready", i))
    queue: List[Tuple[float, int]] = []              # (进队时间, 会话)
    running, used = 0, 0                             # used = 池里已占的 token (常驻的会话 KV + 运行中的预留)
    prefill_free_at = 0.0                            # prefill 是算力瓶颈, 所有调用排同一条 prefill 队 (FIFO)

    def evictable(s, now):
        return s["resident"] > 0 and s["pause_at"] >= 0 and now >= s["pinned_until"]

    def free(need, now, force=False):
        """淘汰停在工具上的会话, 直到腾出 need 个 token。hint 策略下先淘汰慢工具的会话, 其余按 LRU。
        force=True: 池子被钉住的会话占满、又没有调用在跑时, 连钉住的也淘汰, 否则会死锁。"""
        nonlocal used
        cands = [i for i, s in enumerate(sess) if evictable(s, now) or (force and s["resident"] > 0 and s["pause_at"] >= 0)]
        cands.sort(key=lambda i: (not sess[i]["slow"] if policy == "hint" else 0, sess[i]["pause_at"]))
        for i in cands:
            if cfg.kv_pool - used >= need:
                break
            res.forced += not evictable(sess[i], now)
            used -= sess[i]["resident"]
            sess[i]["resident"] = 0
            res.evictions += 1
        return cfg.kv_pool - used >= need

    def try_start(now):
        nonlocal running, used, prefill_free_at
        for qi, (t_q, i) in list(enumerate(queue)):  # 按顺序扫, 放得下的先跑 (避免钉住的会话把队首堵死)
            if running >= cfg.slots:
                break
            s = sess[i]
            new = cfg.first_prompt if s["step"] == 0 else cfg.new_tokens
            miss_hist = s["ctx"] if s["resident"] == 0 else 0     # KV 被淘汰过: 整段历史重算
            need = miss_hist + new + cfg.out_tokens  # 已常驻的部分早就算在 used 里
            s_pause = s["pause_at"]
            s["pause_at"] = -1.0                     # 不再是停顿状态, 不会被自己的 free 淘汰
            if not free(need, now, force=running == 0):
                s["pause_at"] = s_pause
                continue
            queue.remove((t_q, i))
            used += need
            s["resident"] = s["ctx"] + new + cfg.out_tokens
            s["pinned_until"] = -1.0
            res.recompute_tokens += miss_hist
            running += 1
            start = max(now, prefill_free_at)        # ★ 重算的历史也要占 prefill 算力, 拖慢别人的首 token
            prefill_free_at = start + (miss_hist + new) / cfg.prefill_tok_per_s
            res.ttft_s.append(prefill_free_at - t_q)
            s["ctx"] += new + cfg.out_tokens
            heapq.heappush(ev, (prefill_free_at + cfg.out_tokens * cfg.tpot_ms / 1e3, next(seq), "done", i))

    while ev:
        now, _, kind, i = heapq.heappop(ev)
        s = sess[i]
        if kind == "ready":
            queue.append((now, i))
        elif kind == "done":
            running -= 1
            s["step"] += 1
            if s["step"] == cfg.steps:               # 会话结束, KV 全部释放
                used -= s["resident"]
                s["resident"] = 0
                res.jct_s.append(now - s["arrive"])
            else:
                d = dur[i, s["step"] - 1]
                s["slow"] = bool(slow[i, s["step"] - 1])
                s["pause_at"] = now
                if policy == "pin":
                    s["pinned_until"] = float("inf")
                elif policy == "ttl":
                    s["pinned_until"] = now + cfg.ttl_s
                    heapq.heappush(ev, (now + cfg.ttl_s, next(seq), "tick", i))   # 到期时让调度器再看一眼
                elif policy == "hint":                   # 应用说这是快工具: TTL 取它典型耗时的 2 倍; 慢工具不钉
                    s["pinned_until"] = -1.0 if s["slow"] else now + 2 * cfg.fast_s
                    heapq.heappush(ev, (now + 2 * cfg.fast_s, next(seq), "tick", i))
                else:
                    s["pinned_until"] = -1.0
                heapq.heappush(ev, (now + d, next(seq), "ready", i))
        pinned = sum(x["resident"] for x in sess if x["pause_at"] >= 0 and now < x["pinned_until"])
        res.peak_pinned = max(res.peak_pinned, pinned)
        try_start(now)
    assert len(res.jct_s) == cfg.n_sessions, "每个会话都应跑完"
    return res
