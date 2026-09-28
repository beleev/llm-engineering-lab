"""M07 — Subagents: 把一件脏活关进隔离的上下文, 只拿回摘要。

没有它: 调研过程中读到的每一篇文档都堆进主上下文, 主 agent 很快被细节淹没, 每轮还要为它们重复付费。
关键设计:
  - delegate 是一个普通工具; 子 agent 有自己的 messages、工具集、权限门。
  - 父级只收到一行摘要; 子级完整 transcript 落盘备查, 不回流。
  - 隔离是双向的: 子级也拿不到父级的工具 (这里父级有 shell, 子级没有)。
  - delegate 是高风险工具: 父级要有 allow 规则才能委托。子级跑满 max_turns 时结果是 is_error。
多个子级并行扇出 + token 记账见 m11。
对应: Claude Code 的 Task (Agent) 工具与 SubagentStop hook。
不防: 子级的权限门没有规则, 也没有 hooks。父级的 deny 规则和 pre_tool_use 拦截都管不到子级,
  子级能做什么只取决于给它的工具集 (见 core/subagents.py)。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    Decision,
    DelegateTool,
    HookManager,
    JsonlSessionStore,
    ModelAction,
    PermissionGate,
    PermissionRule,
    RuleBasedLLM,
    SearchDocsTool,
    ShellTool,
    ToolCall,
    ToolRegistry,
)
from llm_agent.core.utils import banner, kv

DOCS = {
    # 后面拼 120 个 "detail ": 故意造一篇长文档, 看它撑大的是谁的上下文
    "agent_loop": "The loop is small; harness systems around it carry most complexity. " + "detail " * 120,
    "context": "Subagent side transcripts should not flood the parent context.",
}


def main() -> None:
    banner("M07 - Isolated Subagents")

    stops = []
    hooks = HookManager()
    hooks.register("subagent_stop", lambda agent_type, summary: stops.append(agent_type))

    with tempfile.TemporaryDirectory(prefix="llm_agent_sub_") as tmp:
        delegate = DelegateTool(
            {"researcher": lambda: ToolRegistry([SearchDocsTool(DOCS)])}, hooks=hooks, transcript_dir=Path(tmp)
        )
        # delegate 是 high 风险, auto 模式下没有 allow 规则就会被拒 (没人可问)
        gate = PermissionGate("auto", [PermissionRule("delegate", "", Decision.ALLOW, "delegation is reviewed")])
        parent = Agent(RuleBasedLLM(), ToolRegistry([delegate, ShellTool()]), gate, hooks=hooks, name="parent")
        parent.run("请委托子智能体调研 agent loop")

        child = delegate.children[0]
        print("\n[父级 transcript]")
        for msg in parent.messages:
            print(f"  {msg.role:<9} {msg.text[:100]}")
        print("\n[子级 transcript — 只在磁盘上, 不在父上下文里]")
        for msg in child["messages"]:
            print(f"  {msg.role:<9} {msg.text[:100]}")

        parent_tools = [b["name"] for m in parent.messages for b in m.tool_uses()]
        child_tools = [b["name"] for m in child["messages"] for b in m.tool_uses()]
        assert parent_tools == ["delegate"], f"父级应只调 delegate, 自己不动手, 实际: {parent_tools}"
        assert child_tools == ["search_docs"], f"检索应发生在子级, 实际: {child_tools}"
        # user / tool_use / tool_result(摘要) / final —— 子级的 4 条消息一条都没进来
        assert len(parent.messages) == 4, f"父 transcript 应只有 4 条, 实际: {len(parent.messages)}"
        assert stops == ["researcher"], f"子级结束时应触发一次 subagent_stop, 实际: {stops}"
        assert len(JsonlSessionStore(child["path"]).load()) == len(child["messages"]), (
            "子 transcript 应完整落盘, 条数与内存里的一致"
        )

        kv("父级峰值上下文 tokens", parent.usage["peak_context_tokens"])
        kv("子级峰值上下文 tokens", child["usage"]["peak_context_tokens"])
        # 长文档只撑大了子级的上下文
        assert parent.usage["peak_context_tokens"] < child["usage"]["peak_context_tokens"], (
            "父级只收到摘要, 它的上下文峰值应小于读了长文档的子级"
        )

        class Endless:
            """写死的子级模型: 永远还要再搜一次, 从不给最终回答。"""

            def next(self, messages, tools):
                return ModelAction.tool(ToolCall("search_docs", {"query": "agent loop"}))

        stuck = DelegateTool({"researcher": lambda: ToolRegistry([SearchDocsTool(DOCS)])}, llm_factory=Endless)
        result = stuck.execute({"task": "调研 agent loop"})
        kv("子级没做完时的结果", f"ok={result.ok} {result.output}")
        assert not result.ok, f"子级跑满 max_turns 没给出最终回答, 委托结果应是 is_error, 实际: {result}"

    print("\n  OK: 父级拿结论, 子级留细节; 隔离的是上下文, 也是工具权限。")


if __name__ == "__main__":
    main()
