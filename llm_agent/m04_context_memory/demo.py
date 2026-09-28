"""M04 — Context & Memory: 上下文窗口是稀缺资源, 要主动管。

没有它: 长会话撑爆窗口; "爆了再截断"会把最早说的目标一起截掉, agent 从此答非所问。
关键设计:
  - 文件记忆: markdown 文件, 每轮按 prompt 检索相关片段拼进上下文 (中文靠 bigram 分词)。
  - 瘦身三档, 由便宜到贵: 清旧工具结果 → 模型写摘要 → (反例) 硬截断。
    本模块在函数层面对比三者; 它们如何接进 agent loop 并真正缩小持久化的会话, 见 m14。
对应: CLAUDE.md 记忆文件; Claude API context editing 与 compaction; Claude Code /compact。
差异: CLAUDE.md 是整份载入上下文。这里的 FileMemory 按当前 prompt 做关键词检索, 每次最多取 3 个文件。
  这里的长度和预算都按字符算, 不按 token。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    FileMemory,
    Message,
    RuleBasedLLM,
    clear_tool_results,
    summarize_with_llm,
    total_chars,
    truncate_messages,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv


def fake_session(turns: int = 10) -> list:
    """手工拼一段"多轮 + 工具调用"的 transcript, 工具结果故意很长。"""
    messages = [Message("system", "You are an agent.")]
    for i in range(1, turns + 1):
        messages += [
            Message("user", f"目标{i}: 检索主题 {i}"),
            Message("assistant", [{"type": "tool_use", "id": f"t{i}", "name": "search_docs", "input": {"query": f"topic {i}"}}]),
            Message("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "padding " * 20 + f"fact-{i} " + "padding " * 20, "is_error": False}]),
            Message("assistant", f"主题 {i} 的结论是 fact-{i}"),
        ]
    return messages


def main() -> None:
    banner("M04 - Context, Memory, Compaction")

    print("\n[1] 文件记忆检索 (含中文查询)")
    with tempfile.TemporaryDirectory(prefix="llm_agent_memory_") as tmp:
        memory = FileMemory(Path(tmp))
        memory.add("permissions", "Deny-first permission rules protect shell and file writes.")
        memory.add("subagents", "Subagents should keep separate context and return summaries.")
        memory.add("回答风格", "回答用中文; 先给结论, 再给关键原因。")
        for query, expect in (("permission shell", "permissions.md"), ("应该用什么风格回答", "回答风格.md")):
            hits = [name for name, _ in memory.search(query)]
            kv(query, hits)
            assert hits[0] == expect, f"查询 {query!r} 的第一条应是 {expect}, 实际: {hits}"

    messages = fake_session()
    before = total_chars(messages)

    print("\n[2] 第 1 档: 清旧工具结果 —— 零模型开销, 对话结构原样保留")
    cleared = clear_tool_results(messages, keep_last=1)
    kv("字符数", f"{before} -> {total_chars(cleared)}")
    assert total_chars(cleared) < before * 0.4, (
        f"清掉 9 个旧工具结果后应不到原来的 40%, 实际: {before} -> {total_chars(cleared)}"
    )
    assert validate_transcript(cleared) == [], "清理只换正文, tool_use / tool_result 的配对应原样保留"
    # 只有最近一个结果留着正文。cleared[-2] 是第 10 轮的 tool_result, cleared[3] 是第 1 轮的
    assert "fact-10" in cleared[-2].text, "最近一个工具结果的正文应保留"
    assert "fact-1 " not in cleared[3].text, "第 1 轮的工具结果应已换成占位符"
    assert "fact-1 " in messages[3].text, "原 transcript 不应被改动: 清理只作用于发给模型的视图"

    print("\n[3] 第 2 档: 模型写摘要, 替换掉旧轮次 (保留当前轮)")
    # messages[0] 是 system; 每轮 4 条, 所以最后 4 条是当前轮 (第 10 轮), 中间的是要压缩的前 9 轮
    old, current_turn = messages[1:-4], messages[-4:]
    summary = summarize_with_llm(RuleBasedLLM(), old, keep="用户偏好中文")
    compacted = [messages[0], summary] + current_turn
    print("  " + summary.text.replace("\n", "\n  "))
    kv("字符数", f"{before} -> {total_chars(compacted)}")
    assert total_chars(compacted) < before * 0.5, (
        f"前 9 轮换成摘要后应不到原来的一半, 实际: {before} -> {total_chars(compacted)}"
    )
    assert validate_transcript(compacted) == [], "当前轮原样保留, 配对不应被切断"
    # 目标和结论都还在
    assert all(f"目标{i}" in summary.text and f"fact-{i}" in summary.text for i in range(1, 10)), (
        "前 9 轮的目标和结论都应留在摘要里"
    )
    # keep 参数指定的必留信息。接进 agent 后, 这个参数的值来自 pre_compact hook
    assert "用户偏好中文" in summary.text, "keep 指定的内容应出现在摘要里"

    print("\n[4] 反例: 同等预算下硬截断 —— 每条消息留个开头, 预算用完就一刀切")
    truncated = truncate_messages(messages, max_chars=total_chars(compacted))
    kv("字符数", f"{before} -> {total_chars(truncated)}")
    lost = [i for i in range(1, 10) if f"fact-{i} " not in " ".join(m.text for m in truncated)]
    kv("丢失的结论", lost)
    assert lost, "截断应当丢信息"
    # 结构信息全没了, 只剩文本碎片
    assert truncated[-1].role == "assistant", "截断保留尾部, 最后一条应仍是 assistant 的回答"
    assert all(not m.tool_uses() for m in truncated), "截断把消息拍平成文本, 不应再有 tool_use block"

    print("\n  OK: 先做便宜且无损的, 再做昂贵且有损的; 截断是最后手段, 不是默认手段。")


if __name__ == "__main__":
    main()
