"""M19 — Prompt caching: 前缀逐字节相同才命中; 写入比普通输入贵, 读取便宜得多; 有 TTL。

没有它: agent loop 每一轮都重发整个上下文 (m04 / m14), 同一段几千 token 的 system prompt 和工具定义被反复全价计费、反复 prefill。
关键设计 (模拟 Messages API 的 cache_control 断点):
  - 请求按 tools → system → messages 的顺序拼成 block 序列; 缓存键是"到某个断点为止的整个前缀"的哈希。
  - 断点放两处: system 末尾 (静态部分) 和最后一个 block (对话增量)。查找时从断点往回看最多 20 个 block。
  - 计费三档: 普通输入 ×1, 缓存写入 ×1.25, 缓存读取 ×0.1 (相对基础输入单价; 示意值, 以官方定价为准)。
  - TTL: 条目 5 分钟没被读就过期; 每次命中刷新。
  - 反例: 把"当前时间"放在 system 开头 —— 前缀每次都变, 一次也命中不了, 还每次付写入溢价, 比不开缓存更贵。
    正确做法: 稳定内容在前, 易变内容在后 (时间戳跟着那一轮的用户消息走, 写进历史后就不再变)。
复用: 请求体用 m15 的 to_api_messages 转换; 工具顺序由 ToolRegistry.schemas() 排序保证稳定。
对应: Anthropic prompt caching (cache_control: {"type": "ephemeral"}); 各家 API 的 cached input 计价。
差异:
  - 真实 API 一个请求最多 4 个断点; 前缀短于最小可缓存长度时不缓存也不报错 (长度随模型不同)。这里没有这两条限制
  - 真实 API 另有 1 小时 TTL, 写入倍率是 2。这里只模拟 5 分钟这一档
  - 这里回看时每个 block 算一格; token 数是 estimate_tokens 的粗估, 耗时是按公式算的, 没有真的计时
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from llm_agent.core import Agent, CalculatorTool, Message, ModelAction, PermissionGate, RuleBasedLLM, SearchDocsTool, ToolRegistry
from llm_agent.core.claude_llm import to_api_messages
from llm_agent.core.utils import banner, estimate_tokens, kv

PRICE = {"input": 1.0, "cache_write": 1.25, "cache_read": 0.10}  # 相对基础输入单价的倍率 (示意值, 以官方定价为准)
LATENCY_MS = {"base": 300, "per_uncached_token": 0.20, "per_cached_token": 0.02}  # prefill 耗时模型 (示意值)
TTL_S, LOOKBACK = 300, 20  # 缓存条目存活 300 秒 (5 分钟); 从断点往回最多看 20 个 block

TOPICS = ["kv cache", "lora", "dpo", "sampling", "rope", "moe"]
DOCS = {t.replace(" ", "_"): f"{t}: key fact FACT-{i}." for i, t in enumerate(TOPICS, 1)}
# 120 条规则拼成一段很长的 system prompt: 这就是每次请求都要重发、最值得缓存的静态前缀
POLICY = "你是公司内部的知识库助手。\n" + "\n".join(
    f"规则 {i}: 回答第 {i} 类问题时, 先检索内部文档, 引用文档标题, 不确定就说不知道, 不要编造数字或链接。" for i in range(1, 121)
)
T0 = datetime(2026, 9, 27, 9, 0, 0)  # 模拟时钟的起点。固定值, 输出才可复现


def request_blocks(tools: List[Dict[str, Any]], system: str, messages: List[Message]) -> List[str]:
    """按缓存前缀顺序 tools → system → messages 序列化。sort_keys: 同一内容永远是同一串字节。"""
    _, api_messages = to_api_messages(messages)
    blocks = [json.dumps(t, sort_keys=True, ensure_ascii=False) for t in tools]
    blocks.append(json.dumps({"system": system}, ensure_ascii=False))
    blocks += [json.dumps({"role": m["role"], **b}, sort_keys=True, ensure_ascii=False) for m in api_messages for b in m["content"]]
    return blocks


class PromptCache:
    """模拟服务端的缓存。只记"哪些前缀被缓存过、何时过期", 不存内容。"""

    def __init__(self, ttl: float = TTL_S) -> None:
        self.ttl = ttl
        self.entries: Dict[str, float] = {}  # 前缀哈希 -> 过期时刻 (秒)

    def request(self, blocks: List[str], breakpoints: List[int], now: float) -> Dict[str, int]:
        """返回本次请求的 token 账单 {read, write, input}。breakpoints = 带 cache_control 的 block 下标。"""
        hashes, h = [], hashlib.sha256()
        for block in blocks:  # 前缀哈希链: 第 i 个哈希覆盖 blocks[0..i], 中间任何一个字节变了, 之后全变
            h.update(block.encode())
            hashes.append(h.copy().hexdigest())
        tokens = [estimate_tokens(b) for b in blocks]  # 每个 block 的 token 数

        # 第 1 步 读: 每个断点各自往回找, 取命中的最长前缀
        hit = 0  # 命中的前缀长度 (block 数)
        for bp in breakpoints:
            for i in range(bp, max(-1, bp - LOOKBACK), -1):  # 从断点往回找最长的已缓存前缀
                if self.entries.get(hashes[i], -1) > now:  # 有这个条目, 且还没过期
                    hit = max(hit, i + 1)
                    break
        if hit:
            self.entries[hashes[hit - 1]] = now + self.ttl  # 命中刷新 TTL
        # 第 2 步 写: 断点落在命中前缀之外的, 把"到该断点为止的前缀"写进缓存
        last = max(breakpoints) + 1  # 最后一个断点之后的 block 不参与缓存
        for bp in breakpoints:
            if bp + 1 > hit:
                self.entries[hashes[bp]] = now + self.ttl
        # 三段账单: [0, hit) 读缓存; [hit, last) 写缓存; [last, 末尾) 按普通输入计费
        return {"read": sum(tokens[:hit]), "write": sum(tokens[hit:last]), "input": sum(tokens[last:])}


def bill(usage: Dict[str, int]) -> Tuple[float, float]:
    """token 账单 → (费用, 首 token 延迟 ms)。费用的单位是"1 个普通输入 token 的价钱"。"""
    cost = usage["input"] * PRICE["input"] + usage["write"] * PRICE["cache_write"] + usage["read"] * PRICE["cache_read"]
    ms = LATENCY_MS["base"] + (usage["input"] + usage["write"]) * LATENCY_MS["per_uncached_token"] + usage["read"] * LATENCY_MS["per_cached_token"]
    return cost, ms


class RecordingLLM:
    """记录每次调用时模型看到的上下文和"墙上时间"; 每次调用耗时 15s 模拟时钟。"""

    def __init__(self) -> None:
        # calls: 每次调用的 (上下文, 工具定义, 当时的时刻)。now: 模拟时钟, 单位秒
        self.inner, self.calls, self.now = RuleBasedLLM(), [], 0.0

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        self.calls.append((list(messages), tools, self.now))
        self.now += 15
        return self.inner.next(messages, tools)


PAUSE_S, IDLE_S = 30, 360  # 两轮之间停多久 (秒): 正常节奏 30 秒; [3] 故意停 6 分钟, 超过 TTL


def session(timestamp_in: str, pause_s: float = PAUSE_S) -> RecordingLLM:
    """6 轮检索对话。timestamp_in="user": 时间戳跟在用户消息末尾; "system": 之后在 system 开头现拼。"""
    llm = RecordingLLM()
    agent = Agent(llm, ToolRegistry([SearchDocsTool(DOCS), CalculatorTool()]), PermissionGate("auto"),
                  system_prompt=POLICY, context_budget_chars=10**6, name=timestamp_in)  # 预算够大: 不让清理/压缩改写历史
    for topic in TOPICS:
        stamp = f" (当前时间 {(T0 + timedelta(seconds=llm.now)).isoformat()})" if timestamp_in == "user" else ""
        agent.run(f"检索 {topic}{stamp}", verbose=False)
        llm.now += pause_s
    return llm


def replay(llm: RecordingLLM, cached: bool, timestamp_in_system: bool = False) -> Dict[str, float]:
    """把录下来的每次调用重放一遍, 按给定的缓存策略算总账。

    同一段录音可以用不同策略重放, 所以几种策略比的是完全相同的请求。
    返回的 base 是这批请求不开缓存的价钱, hits 是读到过缓存的调用次数, ms 是平均延迟。
    """
    cache, total = PromptCache(), {"read": 0, "write": 0, "input": 0, "cost": 0.0, "ms": 0.0, "hits": 0, "base": 0.0}
    for messages, tools, now in llm.calls:
        system = messages[0].text  # agent 拼上下文时, 第一条永远是 system prompt
        if timestamp_in_system:
            system = f"当前时间 {(T0 + timedelta(seconds=now)).isoformat()}\n{system}"  # 反例: 易变内容放在最前面
        blocks = request_blocks(tools, system, messages[1:])
        if cached:
            usage = cache.request(blocks, [len(tools), len(blocks) - 1], now)  # 断点: system 末尾 + 最后一个 block
        else:
            usage = {"read": 0, "write": 0, "input": sum(estimate_tokens(b) for b in blocks)}
        cost, ms = bill(usage)
        total["base"] += sum(estimate_tokens(b) for b in blocks) * PRICE["input"]  # 同一批请求不开缓存的价钱
        for key in ("read", "write", "input"):
            total[key] += usage[key]
        total["cost"] += cost
        total["ms"] += ms / len(llm.calls)
        total["hits"] += usage["read"] > 0
    return total


def main() -> None:
    banner("M19 - Prompt caching")

    print("\n[1] 机制: 同一前缀第二次读缓存; 工具顺序一换 (字节变了) 就全部重写")
    tools = ToolRegistry([SearchDocsTool(DOCS), CalculatorTool()]).schemas()
    first = [Message("user", "检索 lora")]
    blocks = request_blocks(tools, POLICY, first)
    cache = PromptCache()
    a = cache.request(blocks, [len(tools), len(blocks) - 1], now=0)
    b = cache.request(blocks, [len(tools), len(blocks) - 1], now=10)
    flipped = request_blocks(tools[::-1], POLICY, first)
    c = cache.request(flipped, [len(tools), len(flipped) - 1], now=20)
    d = cache.request(blocks, [len(tools), len(blocks) - 1], now=20 + TTL_S + 1)
    for name, usage in (("第 1 次 (冷)", a), ("第 2 次 (同前缀)", b), ("工具顺序反转", c), (f"静默 {TTL_S // 60} 分钟后", d)):
        cost, ms = bill(usage)
        kv(name, f"{usage}  cost={cost:.0f}  ttft≈{ms:.0f}ms")
    assert a["read"] == 0, "第一次请求时缓存是空的, 不应读到任何东西"
    assert b["write"] == 0, "第二次请求的前缀完全相同, 不应再写缓存"
    assert b["read"] == a["write"], "第二次读到的 token 数应等于第一次写入的"
    # 一个字节不同 = 全新前缀; 过了 TTL = 从没缓存过
    assert c["read"] == 0, "工具顺序一换, 前缀的字节变了, 不应命中"
    assert d["read"] == 0, "静默超过 TTL 后条目已过期, 不应命中"
    # 读 ≈ 1/10 价; 冷写比不缓存还贵 25%
    assert bill(b)[0] < 0.1 * bill(a)[0], f"全读缓存的费用应不到冷写的 1/10, 实际: {bill(b)[0]:.0f} vs {bill(a)[0]:.0f}"
    assert bill(a)[0] > sum(a.values()), "冷写要付写入溢价, 费用应高于不开缓存"

    good, stamped = session("user"), session("system")
    print(f"\n[2] {len(TOPICS)} 轮检索会话 ({len(good.calls)} 次模型调用): 不缓存 vs 缓存, 以及时间戳放错位置")
    rows = {
        "不缓存": replay(good, cached=False),
        "缓存 (时间戳在 system 开头)": replay(stamped, cached=True, timestamp_in_system=True),
        "缓存 (时间戳在用户消息末尾)": replay(good, cached=True),
    }
    for name, r in rows.items():
        print(f"    {name:<22} read={r['read']:>6} write={r['write']:>6} input={r['input']:>6}  "
              f"cost={r['cost']:>6.0f} (不缓存的 {r['cost'] / r['base']:.2f}x)  平均ttft≈{r['ms']:>5.0f}ms  命中 {r['hits']}/{len(good.calls)}")
    none, bad, ok = rows.values()  # 依次是: 不缓存 / 时间戳在 system 开头 / 时间戳在用户消息末尾
    assert len(good.calls) == 12, f"6 轮, 每轮问模型 2 次, 应有 12 次调用, 实际: {len(good.calls)}"
    assert bad["hits"] == 0, f"时间戳在 system 开头时前缀每次都变, 应一次都命中不了, 实际: {bad['hits']}"
    # 只有第一次调用是冷的
    assert ok["hits"] == len(good.calls) - 1, f"时间戳放在后面时, 除第一次外应全部命中, 实际: {ok['hits']}"
    # 每次都付写入溢价, 一次都读不到: 比不开缓存还贵
    assert bad["cost"] > 1.2 * bad["base"], (
        f"前缀放错时的费用应超过不开缓存的 1.2 倍, 实际: {bad['cost'] / bad['base']:.2f} 倍"
    )
    assert ok["cost"] < 0.25 * ok["base"], f"前缀放对时的费用应不到不开缓存的 1/4, 实际: {ok['cost'] / ok['base']:.2f} 倍"
    assert ok["ms"] < 0.4 * none["ms"], f"平均延迟应不到不开缓存的 40%, 实际: {ok['ms']:.0f}ms vs {none['ms']:.0f}ms"

    print(f"\n[3] TTL: 同样的会话, 每轮之间停 {IDLE_S // 60} 分钟")
    idle = replay(session("user", pause_s=IDLE_S), cached=True)
    detail = f"cost={idle['cost']:.0f}  命中 {idle['hits']}/{len(good.calls)}  (间隔 {PAUSE_S} 秒时 cost={ok['cost']:.0f})"
    kv(f"每轮间隔 {IDLE_S // 60} 分钟", detail)
    # 只剩每轮内第 2 次调用能命中
    assert idle["hits"] == len(TOPICS), (
        f"轮间隔超过 TTL 时, 每轮应只有第 2 次调用命中, 共 {len(TOPICS)} 次, 实际: {idle['hits']}"
    )
    assert ok["cost"] < idle["cost"] < none["cost"], (
        f"费用应介于缓存常热和不开缓存之间, 实际: {ok['cost']:.0f} / {idle['cost']:.0f} / {none['cost']:.0f}"
    )

    print("\n  OK: 缓存按前缀字节匹配 —— 稳定的放前面, 易变的放后面, 别让 TTL 在两轮之间过期。")


if __name__ == "__main__":
    main()
