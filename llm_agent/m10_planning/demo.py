"""M10 — Planning: todo 工具、plan 模式、并行工具调用。

没有它: 多步任务里模型边想边做 —— 做到一半忘了还剩什么; 用户也没机会在它动手之前说"不"。
关键设计:
  - todo_write: 计划是模型自己维护的显式状态 (每次整表覆写), 留在上下文里, 也能展示给用户。
  - plan 模式: 权限门只放行只读工具; 模型用 exit_plan_mode 提交计划, 人批准后模式才切走。
    "只读"由 harness 强制, 不靠模型自觉。
  - 互不依赖的调用放进同一个 assistant turn, harness 用线程池并行执行: 耗时 ≈ 最慢的一个。
对应: Claude Code 的 TodoWrite、plan mode / ExitPlanMode; Claude API 的 parallel tool use。
简化: 一个工具算不算只读, 看的是它自己声明的 read_only 属性, 权限门不核对。
  "人"是一个返回 True / False 的函数。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from llm_agent.core import (
    Agent,
    DelegateTool,
    ExitPlanModeTool,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    TodoWriteTool,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    WriteNoteTool,
)
from llm_agent.core.utils import banner, kv

DOCS = {"kv_cache": "The kv cache stores past keys and values so decoding is incremental."}
PROMPT = "搜索 kv cache，并写入笔记"
DELAY_S = 0.2  # 每次假天气查询睡多久, 模拟网络延迟
CITIES = ["北京", "上海", "深圳"]  # [4] 一次问几个城市, 就是同一 turn 里并行几个调用


class SlowWeatherTool(Tool):
    """每次调用睡 DELAY_S 秒的假天气工具, 用来量并行和串行的耗时差。"""

    name = "weather"
    description = f"Weather lookup that takes {DELAY_S}s (simulates network latency)."
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        time.sleep(DELAY_S)
        return ToolResult(self.name, f"{args['city']}: sunny")


def run_plan_mode(approve: bool):
    """在 plan 模式下跑一遍 PROMPT。approve 决定"人"批不批准计划。

    返回 (agent, 权限门, 笔记本, todo 工具, 提交过的计划, 最终回答), 供 main 断言。
    """
    notes: list = []
    todos = TodoWriteTool()
    gate = PermissionGate("plan")
    plans = []
    # list.append 返回 None, 所以 `append(plan) or approve` 先记下计划, 再返回 approve
    exit_tool = ExitPlanModeTool(gate, approve=lambda plan: plans.append(plan) or approve)
    tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes), todos, exit_tool])
    agent = Agent(RuleBasedLLM(), tools, gate, max_turns=8, name="approved" if approve else "rejected")
    final = agent.run(PROMPT)
    return agent, gate, notes, todos, plans, final


def main() -> None:
    banner("M10 - Planning: todos, plan mode, parallel calls")

    print("\n[1] plan 模式 + 用户拒绝计划: 一个字都没写")
    agent, gate, notes, todos, plans, final = run_plan_mode(approve=False)
    print("  提交的计划:\n    " + plans[0].replace("\n", "\n    "))
    assert notes == [], f"计划被拒, 笔记本应是空的, 实际: {notes}"
    assert gate.mode == "plan", f"计划被拒, 权限模式应停在 plan, 实际: {gate.mode}"
    assert "未获批准" in final, f"模型应告诉用户计划没获批, 实际: {final}"
    assert "write_note" not in [b["name"] for m in agent.messages for b in m.tool_uses()], (
        "计划被拒后, 模型不应再尝试写操作"
    )

    print("\n[2] 用户批准: 模式切到 accept_edits, 按计划执行, todo 从 pending 走到 completed")
    agent, gate, notes, todos, plans, final = run_plan_mode(approve=True)
    order = [b["name"] for m in agent.messages for b in m.tool_uses()]
    kv("工具调用顺序", order)
    assert order == ["todo_write", "exit_plan_mode", "search_docs", "write_note", "todo_write"], (
        f"应先列 todo、交计划, 获批后才检索和写入, 最后更新 todo, 实际: {order}"
    )
    assert gate.mode == "accept_edits", f"计划获批后模式应切到 accept_edits, 实际: {gate.mode}"
    assert len(notes) == 1, f"获批后应写入 1 条笔记, 实际: {notes}"
    assert [t["status"] for t in todos.history[0]] == ["pending", "pending"], "第一次列 todo 时两项都应是 pending"
    assert [t["status"] for t in todos.history[-1]] == ["completed", "completed"], "收尾时两项都应是 completed"

    print("\n[3] plan 模式是 harness 强制的: 就算模型不交计划直接写, 门也不开")
    gate = PermissionGate("plan")
    rogue = Agent(RuleBasedLLM(), ToolRegistry([WriteNoteTool(notes_b := [])]), gate, name="rogue")
    # 这个 agent 没有 exit_plan_mode 工具, 模型会直接去写
    assert "DENIED: plan mode is read-only" in rogue.run("写入笔记: 偷偷写"), "plan 模式下的写操作应被权限门拒绝"
    assert notes_b == [], f"被拒的写操作不应留下笔记, 实际: {notes_b}"
    sneaky = DelegateTool({"writer": lambda: ToolRegistry([WriteNoteTool([])])})  # 想靠子 agent 代写? 委托不是只读操作
    assert not gate.evaluate(ToolCall("delegate", {"task": "写入笔记"}), sneaky).allowed, (
        "delegate 标的是非只读, plan 模式下不应放行"
    )

    print(f"\n[4] 并行工具调用: {len(CITIES)} 个 {DELAY_S}s 的调用")
    timings = {}
    for label, workers in (("串行", 1), ("并行", 4)):
        agent = Agent(RuleBasedLLM(), ToolRegistry([SlowWeatherTool()]), PermissionGate("auto"), max_parallel=workers, name=label)
        start = time.perf_counter()
        agent.run(f"查询{'、'.join(CITIES)}天气", verbose=False)
        timings[label] = time.perf_counter() - start
        n = len(CITIES)
        assert len(agent.messages[1].tool_uses()) == n, f"{n} 个城市应在同一个 turn 里发出 {n} 个 tool_use"
        assert len(agent.messages[2].tool_results()) == n, f"{n} 个结果应放在同一条 user 消息里"
        kv(f"{label} (max_parallel={workers})", f"{timings[label]:.2f}s")
    # 3 个调用各睡 0.2s: 串行至少 0.6s。并行理想值约 0.2s, 阈值放宽到串行的 70%, 给线程调度留余量
    floor = len(CITIES) * DELAY_S
    assert timings["串行"] >= floor, f"串行执行应不少于 {floor:.1f}s, 实际: {timings['串行']:.2f}s"
    assert timings["并行"] < timings["串行"] * 0.7, (
        f"并行应明显快于串行 (不到 70%), 实际: {timings['并行']:.2f}s vs {timings['串行']:.2f}s"
    )

    print("\n  OK: 先计划后动手; 只读靠门强制; 独立调用并行发。")


if __name__ == "__main__":
    main()
