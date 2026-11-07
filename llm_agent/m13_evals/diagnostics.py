"""首错诊断与前缀回归: 消费真实 Agent 轨迹, 用已知故障的离线环境校验归因。

首错 = 第一次违反本任务验收条件的位置, 不是通用因果推断。并行工具不在本实验内。
前缀回归使用动作集合; 文本相同不代表环境相同, 文本不同也不一定动作错误。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import random

from llm_agent.core import Agent, Message, ModelAction, PermissionGate, ToolCall, ToolRegistry
from llm_agent.core.utils import kv
from llm_agent.m21_completion_verification.demo import PutNote


class ScriptedLLM:
    def __init__(self, titles: tuple[str, ...], notes: dict, outage: bool = False) -> None:
        self.titles, self.notes, self.outage = titles, notes, outage
        self.prefixes: list[dict] = []

    def next(self, messages, tools) -> ModelAction:
        # 冻结的是动作发生前的观察, 后续写入不应改掉这里的证据。
        self.prefixes.append({"messages": deepcopy([m.to_dict() for m in messages]), "state": dict(self.notes)})
        if self.outage:
            raise TimeoutError("model unavailable")
        step = len(self.prefixes) - 1
        return (ModelAction.tool(ToolCall("put_note", {"title": self.titles[step]}))
                if step < len(self.titles) else ModelAction.final("已完成"))


def diagnose(messages: list[Message], notes: dict, required: tuple[str, ...]) -> dict | None:
    """按 assistant 动作顺序读轨迹; 工具失败按配对结果归类。"""
    step = 0
    for i, message in enumerate(messages):
        if message.role != "assistant":
            continue
        step += 1
        uses = message.tool_uses()
        for call in uses:
            if call["name"] != "put_note" or call["input"].get("title") not in required:
                return {"step": step, "category": "scope", "evidence": call}
            results = messages[i + 1].tool_results() if i + 1 < len(messages) else []
            result = next((r for r in results if r["tool_use_id"] == call["id"]), None)
            if result is None:
                return {"step": step, "category": "infrastructure", "evidence": "missing tool result"}
            if result["is_error"]:
                return {"step": step, "category": "tool", "evidence": result}
        if not uses and any(notes.get(k) != "saved" for k in required):
            return {"step": step, "category": "false_completion", "evidence": dict(notes)}
    return None


def trace_case(titles: tuple[str, ...], broken: bool = False, outage: bool = False) -> dict:
    notes: dict[str, str] = {}
    llm = ScriptedLLM(titles, notes, outage)
    agent = Agent(llm, ToolRegistry([PutNote(notes, broken)]), PermissionGate("auto"), max_turns=5,
                  context_budget_chars=100000, compaction="none")
    try:
        final = agent.run("只写 alpha、beta 两条笔记", verbose=False)
        failure = diagnose(agent.messages, notes, ("alpha", "beta"))
    except TimeoutError as exc:
        final, failure = "", {"step": len(llm.prefixes), "category": "infrastructure", "evidence": str(exc)}
    calls = sum(len(m.tool_uses()) for m in agent.messages)
    result = {"passed": notes == {"alpha": "saved", "beta": "saved"} and failure is None,
              "final": final, "failure": failure, "state": notes, "tool_calls": calls,
              "usage_estimate": dict(agent.usage), "model_attempts": len(llm.prefixes)}
    if failure:
        result["prefix"] = llm.prefixes[failure["step"] - 1]
    return result


@dataclass(frozen=True)
class PrefixCase:
    name: str
    context: str
    acceptable: frozenset[str]


PREFIX_CASES = (
    PrefixCase("project-exception", '{"global":"Python","project":"TypeScript"}', frozenset({"use:TypeScript"})),
    PrefixCase("current-request", '{"global":"Python","project":"TypeScript","current":"Rust"}', frozenset({"use:Rust"})),
    PrefixCase("ambiguous-owner", '{"global":"Python","ambiguous":true}', frozenset({"clarify", "inspect"})),
)


def next_action(context: str, scoped: bool) -> str:
    data = json.loads(context)
    if not scoped:
        return "use:" + data["global"]
    if data.get("ambiguous"):
        return "clarify"
    return "use:" + data.get("current", data.get("project", data["global"]))


def paired_interval(before: list[bool], after: list[bool], seed: int = 13, samples: int = 1000) -> tuple[float, float]:
    """按题配对 bootstrap, 不把同题 A/B 当两组独立样本。小套件不支持泛化结论。"""
    if not before or len(before) != len(after) or samples < 2:
        raise ValueError("需要非空、等长的配对结果和至少两次重采样")
    delta = [int(b) - int(a) for a, b in zip(before, after)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(samples))
    return means[int((samples - 1) * .025)], means[int((samples - 1) * .975)]


def main() -> None:
    print("\n[3] 相同回答, 不同环境: 定位第一个错误动作")
    cases = {
        "correct": trace_case(("alpha", "beta")),
        "premature": trace_case(("alpha",)),
        "extra-write": trace_case(("alpha", "extra", "beta")),
        "tool-failure": trace_case(("alpha", "beta"), broken=True),
        "outage": trace_case((), outage=True),
    }
    expected = {"correct": None, "premature": (2, "false_completion"), "extra-write": (2, "scope"),
                "tool-failure": (1, "tool"), "outage": (1, "infrastructure")}
    for name, row in cases.items():
        failure = row["failure"]
        got = (failure["step"], failure["category"]) if failure else None
        kv(name, f"passed={row['passed']}, 首错={got}, 工具调用={row['tool_calls']}")
        assert got == expected[name]
        assert row["passed"] == (name == "correct")
    assert len({cases[n]["final"] for n in ("correct", "premature", "extra-write", "tool-failure")}) == 1
    assert "extra" not in cases["extra-write"]["prefix"]["state"], "首错前快照不能被后续写入污染"

    print("\n[4] 同一个冻结前缀, 只换下一步策略 (规则替身, 不是 LLM 实测)")
    before, after = [], []
    for case in PREFIX_CASES:
        a, b = next_action(case.context, False), next_action(case.context, True)
        before.append(a in case.acceptable)
        after.append(b in case.acceptable)
        kv(case.name, f"{a} → {b}; 可接受 {sorted(case.acceptable)}")
    assert before == [False] * 3 and after == [True] * 3
    assert "inspect" in PREFIX_CASES[2].acceptable, "可接受动作不只一条参考答案"
    assert paired_interval(before, before) == (0, 0)
    kv("配对 bootstrap 区间", f"{paired_interval(before, after)}; 仅 3 个手写反例, 不外推模型能力")
