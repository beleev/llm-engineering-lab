"""M06 — Persistence & Resume: append-only JSONL 会话日志。

没有它: 进程一退会话全丢; 出事后也没法复盘"模型当时到底要求了什么、harness 放行了什么"。
关键设计:
  - 每条消息 (含助手的 tool_use 和回填的 tool_result) 立刻追加一行 —— 先记录"模型要什么", 再去执行。
  - resume = 读回 transcript。恢复的只有上下文: 权限要由新会话重新建立, SessionStart 注入不重复。
  - 落盘的 block 格式与 id 配对和 Messages API 一致; harness 注入的 system 消息发送前由 claude_llm.to_api_messages 转换。
对应: Claude Code 的 ~/.claude/projects/*.jsonl 与 `claude --resume`。
  - 崩在 tool_use 之后: resume 给悬空的调用补一条 is_error 占位结果, 并写回 JSONL, 再 resume 也合法 ([3])。
上限: 没有 fsync 和文件锁。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    HookManager,
    JsonlSessionStore,
    Message,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    WriteNoteTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

DOCS = {"agent_loop": "Agent loop uses messages, tools, permissions and persistence."}


def main() -> None:
    banner("M06 - Append-only Session Persistence")

    sources = []
    hooks = HookManager()
    # list.append 返回 None, 所以 `append(...) or "..."` 先记下来源, 再把后面的字符串作为返回值
    hooks.register("session_start", lambda source: sources.append(source) or "Policy: answer in Chinese.")

    with tempfile.TemporaryDirectory(prefix="llm_agent_session_") as tmp:
        path = Path(tmp) / "session.jsonl"
        notes: list = []
        tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes)])

        print("\n[1] session A: 搜索, 然后进程'退出'")
        Agent(RuleBasedLLM(), tools, PermissionGate("auto"), hooks=hooks, store=JsonlSessionStore(path), name="session-A").run(
            "搜索 agent loop"
        )

        print("\n[JSONL 原文]")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in lines:
            print("  " + json.dumps(record, ensure_ascii=False)[:150])
        kinds = [b["type"] for r in lines if isinstance(r["content"], list) for b in r["content"]]
        # 只记工具结果的话, 看不到模型发起过什么调用, 这种 transcript 对真实 API 也不合法
        assert kinds == ["tool_use", "tool_result"], f"模型发起的调用和工具结果都应落盘, 实际: {kinds}"

        print("\n[2] session B: 新进程读回 transcript 继续; 引用'刚才的结果'")
        agent_b = Agent(
            RuleBasedLLM(),
            tools,
            # 会话 A 是 auto 模式; B 退回 default, 写操作要重新问人 —— 恢复上下文 ≠ 恢复授权
            PermissionGate("default", ask_policy=lambda call: call.name == "write_note"),
            hooks=hooks,
            store=JsonlSessionStore(path),
            load_history=True,
            name="session-B",
        )
        assert len(agent_b.messages) == len(lines), "读回的消息数应等于文件行数: 一行不多, 一行不少"
        agent_b.run("把刚才结果写入笔记")

        # 写进笔记的是上一个会话的工具数据, 原样无杂质
        assert notes == ["agent_loop: " + DOCS["agent_loop"]], f"笔记应是会话 A 检索到的原文, 实际: {notes}"
        assert validate_transcript(agent_b.messages) == [], "跨会话接起来的 transcript 配对应仍然合法"
        assert sources == ["startup", "resume"], f"A 应是 startup, B 应是 resume, 实际: {sources}"
        assert len([m for m in agent_b.messages if m.name == "session_start"]) == 1, (
            "resume 时历史里已有 session_start 消息, 不应重复注入"
        )
        ids = [b["id"] for m in agent_b.messages for b in m.tool_uses()]
        assert ids == ["toolu_0001", "toolu_0002"], f"id 应跨会话续号, 不与会话 A 的撞号, 实际: {ids}"

        kv("JSONL 行数", len(JsonlSessionStore(path).load_all()))
        kv("笔记", notes)

        print("\n[3] 崩在 tool_use 之后: resume 补上占位结果并写回 JSONL, 连续 resume 两次")
        crashed = JsonlSessionStore(Path(tmp) / "crashed.jsonl")
        crashed.append(Message("user", "搜索 agent loop"))
        # 模型要了工具, 进程在工具返回之前退出: 日志最后一行是悬空的 tool_use
        crashed.append(Message("assistant", [ToolCall("search_docs", {"query": "agent loop"}, id="toolu_0001").to_block()]))
        for attempt in (1, 2):
            store = JsonlSessionStore(crashed.path)
            resumed = Agent(RuleBasedLLM(), tools, PermissionGate("auto"), store=store, load_history=True, name=f"resume-{attempt}")
            problems = validate_transcript(resumed.messages)
            assert problems == [], f"第 {attempt} 次 resume 读回的 transcript 应合法, 实际: {problems}"
            resumed.run("搜索 agent loop", verbose=False)
        audit = crashed.load_all()
        placeholders = [b for m in audit for b in m.tool_results() if b["is_error"]]
        kv("占位结果", placeholders[0]["content"])
        assert len(placeholders) == 1, f"占位结果应只补一次 (写进了 JSONL, 第二次 resume 不再悬空), 实际: {placeholders}"
        assert validate_transcript(audit) == [], "JSONL 全量记录的配对也应合法"

    print("\n  OK: transcript 可完整复盘、可恢复; 权限不随 transcript 恢复。")


if __name__ == "__main__":
    main()
