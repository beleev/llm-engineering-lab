"""M01 — Agent Loop: 让模型"行动-观察-再行动"的那个 while 循环。

没有它: 模型只能一次性回答, 算不了它心算不准的式子, 也看不到任何外部信息。
关键设计: loop 本身很薄 —— 问模型 → 有 tool_use 就执行并回填 tool_result → 没有就结束。
助手"决定调用什么"和工具"返回了什么"都作为 content block 进 transcript, 两者靠 id 配对。
这里的模型是 RuleBasedLLM (关键词规则), 为的是让 harness 的每一步都确定、可断言。
对应: Claude Code / 任何 ReAct 式 agent 的主循环 (core/agent.py)。
"""

from __future__ import annotations

from llm_agent.core import Agent, CalculatorTool, PermissionGate, RuleBasedLLM, ToolRegistry, validate_transcript
from llm_agent.core.utils import banner, kv


def main() -> None:
    banner("M01 - Minimal Agent Loop")

    agent = Agent(
        llm=RuleBasedLLM(),
        tools=ToolRegistry([CalculatorTool()]),
        # bypass_permissions: 没命中规则的一律放行。这里没有规则, 等于全部放行, 先把权限放到一边只看 loop
        permissions=PermissionGate(mode="bypass_permissions"),
        max_turns=4,
        name="m01",
    )

    print("\n[1] 一次完整闭环")
    final = agent.run("请计算 2 + 3 * 4")
    assert "2 + 3 * 4 = 14" in final, f"最终回答应带上计算器的结果, 实际: {final}"

    print("\n[transcript]")
    for msg in agent.messages:
        print(f"  {msg.role:<9} {msg.content}")
    # 每条消息只留 (角色, 各 block 的类型), 用来核对 transcript 的骨架
    shape = [(m.role, [b["type"] for b in m.blocks]) for m in agent.messages]
    assert shape == [
        ("user", ["text"]),
        ("assistant", ["tool_use"]),  # 模型的决定本身也是 transcript 的一部分
        ("user", ["tool_result"]),  # 工具结果以 user 角色回填, 用 tool_use_id 指回上一条
        ("assistant", ["text"]),
    ], f"transcript 应是 user → tool_use → tool_result → 最终回答 这四条, 实际: {shape}"
    assert validate_transcript(agent.messages) == [], "每个 tool_use 都应配上同 id 的 tool_result"

    print("\n[2] 同一会话的第二个问题: 必须重新调用工具, 不能拿上一轮结果充数")
    final2 = agent.run("再算 10 / 4")
    assert "10 / 4 = 2.5" in final2, f"第二个问题应重新调计算器算出 2.5, 实际: {final2}"
    assert "14" not in final2, f"第二个回答不应混进上一轮的结果 14, 实际: {final2}"
    assert sum(len(m.tool_uses()) for m in agent.messages) == 2, "两个问题应各调一次工具, 共 2 个 tool_use"

    kv("模型调用次数", agent.usage["llm_calls"])
    # 每个问题两次: 一次决定调工具, 一次读结果作答
    assert agent.usage["llm_calls"] == 4, f"2 个问题应问模型 4 次, 实际: {agent.usage['llm_calls']}"
    print("\n  OK: loop 很薄 —— 复杂度都在它周围的 harness (工具/权限/上下文/持久化) 里。")


if __name__ == "__main__":
    main()
