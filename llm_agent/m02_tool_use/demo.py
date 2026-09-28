"""M02 — Tool Use: schema 给模型看, execute 由确定性代码跑, 结果回填上下文。

没有 JSON Schema: 模型只知道工具名, 参数靠猜; harness 也没法在执行前拦下畸形参数。
关键设计:
  - Tool.schema() 就是 Claude API 的 tool 定义 {name, description, input_schema}。
  - 模型输出是不可信输入: 先按 schema 校验, 不合格直接回一个 is_error 的 tool_result,
    让模型下一轮自己改 —— 而不是让工具带着坏参数崩溃。
  - 一个 assistant turn 可带多个 tool_use (并行工具调用), 结果必须合在同一条 user 消息里。
对应: Anthropic tool use (`tools=[...]`, `strict: true`, parallel tool use)。
差异: strict: true 由 API 保证模型给出的参数符合 schema。这里是拿到模型输出之后在本地校验, 只校验一层。
"""

from __future__ import annotations

import json

from llm_agent.core import (
    Agent,
    CalculatorTool,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    WriteNoteTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

DOCS = {
    "agent_loop": "Agent loop = assemble context, call model, run tools, repeat.",
    "permissions": "Deny-first gates keep unknown or risky actions under human control.",
    "context": "Context windows are scarce, so agents compact history and retrieve memory.",
}


def main() -> None:
    banner("M02 - Tool Use")

    notes: list = []
    tools = ToolRegistry([CalculatorTool(), SearchDocsTool(DOCS), WriteNoteTool(notes)])

    print("\n[1] 发给模型的工具定义 (Claude API 格式)")
    for schema in tools.schemas():
        print("  " + json.dumps(schema, ensure_ascii=False))
        assert set(schema) == {"name", "description", "input_schema"}, (
            f"工具定义应只有这三项, risk 等元数据不应发给模型, 实际: {sorted(schema)}"
        )
        assert schema["input_schema"]["type"] == "object", "input_schema 的顶层应是 object"

    print("\n[2] 执行前校验: 模型给错参数时, 得到的是可自我纠正的错误, 不是崩溃")
    for bad in (
        ToolCall("calculator", {"expression": "1+1"}),  # 参数名猜错
        ToolCall("calculator", {"expr": 42}),  # 类型不对
        ToolCall("calculator", {"expr": "__import__('os').system('id')"}),  # 合法字符串, 但不是算式
    ):
        result = tools.execute(bad)
        print(f"  {bad.args!s:<52} -> {result.output}")
        assert not result.ok, f"坏参数应得到失败结果, 不应执行成功: {bad.args}"
    # 对照: 参数正确时照常执行
    assert tools.execute(ToolCall("calculator", {"expr": "1+1"})).output == "1+1 = 2", "合法参数应正常算出结果"

    print("\n[3] 并行 gather → 串行 act")
    agent = Agent(RuleBasedLLM(), tools, PermissionGate(mode="auto"), max_turns=5, name="m02")
    final = agent.run("计算 6 * 7，同时搜索 agent loop，并写入笔记")

    first = agent.messages[1]  # messages[0] 是用户 prompt, [1] 是模型的第一次回复
    # 一个 turn, 两个 tool_use
    assert [b["name"] for b in first.tool_uses()] == ["calculator", "search_docs"], (
        "两个互不依赖的只读调用应在同一个 turn 里一起发出"
    )
    assert len(agent.messages[2].tool_results()) == 2, "两个结果应放在同一条 user 消息里"
    assert validate_transcript(agent.messages) == [], "并行调用的 tool_use / tool_result 应全部配对"
    # 笔记 = 纯工具数据, 没有混进别的文字
    assert notes == ["agent_loop: " + DOCS["agent_loop"]], f"笔记应只有检索到的那一条原文, 实际: {notes}"
    assert "6 * 7 = 42" in final, f"最终回答应带上计算结果, 实际: {final}"

    kv("笔记", notes)
    print("\n  OK: schema 约束输入, 校验守住边界, 并行调用省掉来回轮次。")


if __name__ == "__main__":
    main()
