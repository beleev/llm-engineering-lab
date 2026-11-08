"""M21 — 完成声明只是候选; 只读验收通过后才算结束。

复用 Agent 的工具/权限/轨迹。验收放在外部驱动器, 不改变 Stop Hook 的通知语义。
规则模型故意少写一条, 再根据验收反馈修复; 不声称真实模型会同样服从。
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from llm_agent.core import Agent, ModelAction, PermissionGate, ToolCall, ToolRegistry
from llm_agent.core.schema import ToolResult, validate_transcript
from llm_agent.core.tools import Tool, _obj
from llm_agent.core.utils import banner, kv


class PutNote(Tool):
    name, description = "put_note", "按标题写一条笔记"
    read_only, risk = False, "medium"
    parameters = _obj(["title"], title={"type": "string"})

    def __init__(self, notes: dict[str, str], broken: bool = False) -> None:
        self.notes, self.broken = notes, broken

    def execute(self, args: dict) -> ToolResult:
        if self.broken:
            return ToolResult(self.name, "temporary write failure", ok=False)
        self.notes[args["title"]] = "saved"
        return ToolResult(self.name, "saved")


class EarlyFinalLLM:
    """初次少写 gamma, 看到修复提示才补。stubborn 模式永远只说完成。"""

    def __init__(self, stubborn: bool = False) -> None:
        self.stubborn = stubborn

    def next(self, messages, tools) -> ModelAction:
        last = messages[-1]
        if last.tool_results() or self.stubborn:
            return ModelAction.final("已完成")
        titles = json.loads(last.text.removeprefix("repair:")) if last.text.startswith("repair:") else ["alpha", "beta"]
        return ModelAction.tool(*(ToolCall("put_note", {"title": title}) for title in titles))


@dataclass(frozen=True)
class Verification:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing or self.unexpected or self.changed)


def verify_notes(notes: dict[str, str], required: tuple[str, ...], before: dict[str, str]) -> Verification:
    """纯读取: 判结果和作用域, 不替模型补任务。"""
    return Verification(tuple(k for k in required if notes.get(k) != "saved"),
                        tuple(sorted(set(notes) - set(before) - set(required))),
                        tuple(k for k, v in before.items() if notes.get(k) != v))


def run_verified(agent: Agent, notes: dict[str, str], required: tuple[str, ...],
                 max_repairs: int = 1, max_calls: int = 4) -> dict:
    if max_repairs < 0 or max_calls < 1:
        raise ValueError("修复次数非负, 模型预算至少为 1")
    before, checks, start = dict(notes), [], agent.usage["llm_calls"]
    prompt = "写入 alpha、beta、gamma 三条笔记"
    old_limit = agent.max_turns
    final = "not started"
    try:
        for attempt in range(max_repairs + 1):
            remaining = max_calls - (agent.usage["llm_calls"] - start)
            if remaining <= 0:
                break
            agent.max_turns = min(old_limit, remaining)
            final = agent.run(prompt, verbose=False)
            check = verify_notes(notes, required, before)
            checks.append(check)
            if check.passed:
                return {"passed": True, "final": final, "checks": checks, "repairs": attempt}
            # 不允许模型靠补写掩盖越界修改; 这种情况直接失败。
            if check.unexpected or check.changed:
                break
            prompt = "repair:" + json.dumps(check.missing)
        return {"passed": False, "final": final, "checks": checks, "repairs": max(0, len(checks) - 1)}
    finally:
        agent.max_turns = old_limit


def experiment(mode: str = "verified", max_repairs: int = 1, stubborn: bool = False,
               broken: bool = False, max_calls: int = 4) -> dict:
    notes = {"existing": "keep"}
    agent = Agent(EarlyFinalLLM(stubborn), ToolRegistry([PutNote(notes, broken)]), PermissionGate("auto"),
                  max_turns=2, context_budget_chars=100000, compaction="none")
    required = ("alpha", "beta", "gamma")
    if mode == "verified":
        result = run_verified(agent, notes, required, max_repairs, max_calls)
    elif mode in {"direct", "text"}:
        final = agent.run("写三条笔记", verbose=False)
        result = {"passed": mode == "direct" or "完成" in final, "final": final, "repairs": 0}
    else:
        raise ValueError(mode)
    result.update(notes=notes, calls=agent.usage["llm_calls"], actual=verify_notes(notes, required, {"existing": "keep"}).passed)
    assert not validate_transcript(agent.messages), "修复不能破坏工具配对"
    return result


def main() -> None:
    banner("M21 - Completion verification")
    for mode in ("direct", "text", "verified"):
        row = experiment(mode)
        kv(mode, f"宣告通过={row['passed']}, 环境正确={row['actual']}, 模型调用={row['calls']}")
        assert row["actual"] == (mode == "verified")
        assert row["passed"]
    for kwargs in ({"max_repairs": 0}, {"stubborn": True}, {"broken": True}, {"max_calls": 1}):
        row = experiment(**kwargs)
        assert not row["passed"] and row["calls"] <= kwargs.get("max_calls", 4)
    notes = {"existing": "changed", "alpha": "saved", "beta": "saved", "gamma": "saved", "extra": "saved"}
    before = dict(notes)
    verdict = verify_notes(notes, ("alpha", "beta", "gamma"), {"existing": "keep"})
    assert not verdict.passed and verdict.changed == ("existing",) and verdict.unexpected == ("extra",)
    assert notes == before, "验收必须只读"
    print("\n  OK: 环境证据挡住假完成; 修复有预算; 验收不替模型写结果。")


if __name__ == "__main__":
    main()
