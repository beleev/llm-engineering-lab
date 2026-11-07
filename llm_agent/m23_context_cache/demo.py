"""M23 — 压缩、动态状态、前缀缓存一起算账。

复用 m19 的 PromptCache/bill。固定长度 ASCII 块代表工具日志, 数字全是模型值,
不是供应商账单。摘要由确定性转换器代替 LLM, 只能验证保留规则和成本机制。
"""
from __future__ import annotations

from llm_agent.core.utils import banner, estimate_tokens, kv
from llm_agent.m19_prompt_caching.demo import PromptCache, bill


def block(label: str, tokens: int) -> str:
    if len(label) > tokens * 4:
        raise ValueError("标签超过块长度")
    return label + "." * (tokens * 4 - len(label))


def summarize(facts: dict[str, str], keep_scope: bool = True) -> dict[str, str]:
    # scope 与否定必须显式保留; 这不是自然语言摘要质量评测。
    return dict(facts) if keep_scope else {"language": facts["language"]}


def simulate(turns: int = 6, placement: str = "tail", threshold: int = 0,
             keep_scope: bool = True) -> dict:
    if turns < 1 or threshold < 0 or placement not in {"front", "tail"}:
        raise ValueError("无效轮数、阈值或放置位置")
    cache = PromptCache()
    policy = block("system: fixed rules", 200)
    facts = {"scope": "project-A", "language": "TypeScript", "prohibition": "do-not-delete"}
    active = dict(facts)
    history, audit, rows = [], [], []
    totals = {"read": 0, "write": 0, "input": 0}
    cost, summaries, summary_cost, peak = 0.0, 0, 0.0, 0

    def charge(blocks: list[str], now: float) -> float:
        usage = cache.request(blocks, [0, len(blocks) - 1], now)
        for key in totals:
            totals[key] += usage[key]
        return bill(usage)[0]

    for turn in range(turns):
        if threshold and len(history) >= threshold:
            # 摘要输入也查缓存, 输出按 1 倍示意单价计入; 不把摘要当免费。
            expense = charge([policy, *history, block("summarize", 10)], turn * 10) + 50
            cost += expense
            summary_cost += expense
            summaries += 1
            active = summarize(active, keep_scope)
            history = [block("summary:" + str(sorted(active.items())), 50)]
        state = block(f"memory:{sorted(active.items())};done:{turn}", 40)
        query = block(f"user: task-{turn}", 20)
        blocks = [state, policy, *history, query] if placement == "front" else [policy, *history, state, query]
        amount = charge(blocks, turn * 10 + 1)
        cost += amount
        peak = max(peak, sum(estimate_tokens(b) for b in blocks))
        rows.append({"turn": turn + 1, "input": sum(estimate_tokens(b) for b in blocks), "cost": round(amount, 2)})
        pair = [query, block(f"tool_result:{turn}: immutable evidence", 200)]
        history.extend(pair)
        audit.extend(pair)
    return {"cost": round(cost, 2), "summary_cost": round(summary_cost, 2), "summaries": summaries,
            "peak": peak, "retained": sum(active.get(k) == v for k, v in facts.items()),
            "audit_blocks": len(audit), "usage": totals, "rows": rows}


def main() -> None:
    banner("M23 - Context / cache tradeoff (modeled)")
    front, tail = simulate(placement="front"), simulate()
    compact = simulate(threshold=4)
    lossy = simulate(threshold=4, keep_scope=False)
    for name, row in [("状态放前部", front), ("状态追加尾部", tail), ("保限定摘要", compact), ("丢限定摘要", lossy)]:
        kv(name, {k: row[k] for k in ("cost", "summary_cost", "peak", "retained", "audit_blocks")})
    assert tail["cost"] < front["cost"], "动态前缀破坏历史复用"
    assert compact["peak"] < tail["peak"] and compact["summary_cost"] > 0
    assert compact["retained"] == 3 and lossy["retained"] == 1
    assert all(row["audit_blocks"] == 12 for row in (front, tail, compact, lossy))
    short, compressed = simulate(turns=2), simulate(turns=2, threshold=2)
    assert compressed["cost"] > short["cost"], "短会话的摘要开销可能大于节省"
    kv("两轮: 不摘要 / 摘要", f"{short['cost']} / {compressed['cost']} 模拟费用单位")
    print("\n  OK: 上下文更短不保证总价更低; 证据与限定保留要单独检查。")


if __name__ == "__main__":
    main()
