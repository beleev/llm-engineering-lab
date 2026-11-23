"""M15 — 接真实模型 (opt-in): 同一个 loop, 把 RuleBasedLLM 换成 ClaudeLLM。

不在 run_all 里。没有 SDK 或没有 ANTHROPIC_API_KEY 时, 只跑离线检查, 然后礼貌退出 —— 不发任何网络请求。
关键设计: harness 对模型的全部依赖就是 LLM 协议的 next(); 工具 / 权限 / hook / 持久化一行不用改。
[1] 离线: 用 toy LLM 跑出一段 transcript, 转成 Messages API 请求体, 检查它满足 API 的结构约束。
[2] 离线: 请求参数按当前代模型构造 (adaptive thinking、effort 在 output_config、无采样参数、无强制 tool_choice、无 prefill)。
[3] 离线: 历史只能追加。对同一段带 thinking 块的 transcript 做 5 种改动, 用 stale_thinking 判定真实 API 上哪些块会失效。
[4] 在线: 真实模型 + 同一个 calculator 工具 + 同一个权限门。
    装了 SDK 且环境里有 ANTHROPIC_API_KEY 时, 这一步会发真实请求并计费。
    tests/test_smoke.py 会扫到本 demo, 所以 CI 里不要设这个变量。

    pip install anthropic && export ANTHROPIC_API_KEY=... && python -m llm_agent.m15_claude_api.demo
"""

from __future__ import annotations

import json

from llm_agent.core import Agent, CalculatorTool, HookManager, HookResult, PermissionGate, RuleBasedLLM, ToolRegistry, validate_transcript
from llm_agent.core.claude_llm import ClaudeLLM, build_request, stale_thinking, to_api_messages
from llm_agent.core.memory import clear_tool_results
from llm_agent.core.schema import Message
from llm_agent.core.utils import banner, kv

MODEL = "claude-opus-5-5"


def make_hooks() -> HookManager:
    hooks = HookManager()
    hooks.register("session_start", lambda source: "Answer in Chinese.")
    hooks.register("user_prompt_submit", lambda prompt: HookResult(additional_context="Show your work."))
    hooks.register("post_tool_use", lambda result: f"[audit] {result.name}")
    return hooks


def history_rows():
    """两轮工具调用的 transcript, 每条 assistant 消息带一个 thinking 块; 对它做 5 种改动。"""
    tools = ToolRegistry([CalculatorTool()])
    agent = Agent(RuleBasedLLM(), tools, PermissionGate("auto"), hooks=make_hooks(), name="history")
    agent.run("计算 17 * 23", verbose=False)
    agent.run("再算 10 / 4", verbose=False)
    # 规则替身不产生 thinking: 给每条 assistant 消息补一个。display 默认 omitted 时 thinking 文字本来就是空的
    agent.messages = [Message("assistant", [{"type": "thinking", "thinking": "", "signature": f"sig-{i}"}] + m.blocks)
                      if m.role == "assistant" else m for i, m in enumerate(agent.messages)]
    schemas = tools.schemas()
    system, messages = to_api_messages(agent._assemble_context(""))
    sent = {"system": system, "tools": schemas, "messages": messages}  # 上一次请求 + 它的回复
    total = sum(b["type"] == "thinking" for m in messages for b in m["content"])
    nxt = Message("user", "再算 6 * 7")
    base = agent._assemble_context("")[:1]  # 每次现拼的 system prompt

    ops = [
        ("追加下一问", base + agent.messages),
        ("清旧工具结果 (m14 第 1 档)", base + clear_tool_results(agent.messages, keep_last=1)),
        ("重拼 system (m04 每问检索记忆)", [Message("system", base[0].text + "\nMemory prefs.md:\n回答尽量简短")] + agent.messages),
    ]
    agent._compact()  # m14 第 2 档: 当前轮之前的历史换成一条摘要, 当前轮原样保留
    ops.append(("摘要旧轮, 保留当前轮 (m14 第 2 档)", agent._assemble_context("")))
    ops.append(("整段换成摘要 (simple compaction)", base + agent.messages[:1]))
    rows = {}
    for name, msgs in ops:
        now = build_request(MODEL, 16000, msgs + [nxt], schemas)
        keep, stale = stale_thinking(sent, now)
        dropped = total - sum(b["type"] == "thinking" for m in now["messages"] for b in m["content"])
        rows[name] = (keep, stale, dropped)
        kv(name, f"前缀保留 {keep}/{len(messages)} 条, thinking 失效 {len(stale)}/{total}, 没再发送 {dropped}")
    return rows, len(messages), total


def main() -> None:
    banner("M15 - Real model adapter (opt-in)")

    print("\n[1] 离线: transcript → Messages API 请求体")
    agent = Agent(RuleBasedLLM(), ToolRegistry([CalculatorTool()]), PermissionGate("auto"), hooks=make_hooks(), name="offline")
    agent.run("计算 17 * 23", verbose=False)
    # _assemble_context 给出的就是"下一次要发给模型的上下文", 拿它去转换
    system, api_messages = to_api_messages(agent._assemble_context(""))
    for item in api_messages:
        print(f"    {item['role']:<9} {json.dumps(item['content'], ensure_ascii=False)[:120]}")
    kv("system", " | ".join(system.split("\n\n")))
    roles = [m["role"] for m in api_messages]
    assert roles == ["user", "assistant", "user", "assistant"], f"角色应严格交替并以 user 开头, 实际: {roles}"
    # 开头的 system 消息进了顶层 system
    assert "Answer in Chinese." in system, "session_start 注入的 system 消息应并进顶层 system 参数"
    # tool_result 在最前, hook 注释排在它后面
    assert api_messages[2]["content"][0]["type"] == "tool_result", "tool_result 应排在该 user 消息的最前面"
    assert "<system-reminder>" in api_messages[2]["content"][1]["text"], (
        "对话中途的 system 消息应转成 user 侧的 <system-reminder> 文本, 排在 tool_result 之后"
    )
    assert api_messages[1]["content"][0]["id"] == api_messages[2]["content"][0]["tool_use_id"], (
        "转换后 tool_use 和 tool_result 的 id 应仍然对得上"
    )

    print("\n[2] 离线: 请求参数")
    context = agent._assemble_context("") + [Message("user", "再算 6 * 7")]
    schemas = agent.tools.schemas()
    req = build_request(MODEL, 16000, context, schemas, effort="low")
    kv("keys", sorted(req))
    kv("thinking", req["thinking"])
    kv("output_config", req["output_config"])
    legacy = build_request("claude-haiku-4-5", 16000, context, schemas)
    kv("haiku-4-5 thinking", legacy["thinking"])
    assert req["thinking"] == {"type": "adaptive"}, "当前代模型应发 adaptive thinking"
    assert req["output_config"] == {"effort": "low"}, "effort 应放在 output_config 里"
    assert "effort" not in req, "effort 不在顶层"
    for key in ("temperature", "top_p", "top_k", "tool_choice"):
        assert key not in req, f"不应发送 {key}"
    assert legacy["thinking"] == {"type": "enabled", "budget_tokens": 8000}, "只有 haiku-4-5 及更老的模型用 budget_tokens"
    for model in ("claude-fable-5-2", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-sonnet-4-6"):
        assert build_request(model, 16000, context, schemas)["thinking"] == {"type": "adaptive"}, f"{model} 应走 adaptive"
    try:
        build_request(MODEL, 16000, context + [Message("assistant", "答案是")], schemas)
        raise AssertionError("末尾是 assistant 消息 (prefill) 应被拒绝")
    except ValueError:
        kv("prefill", "拒绝: 请求必须以 user 消息结尾")

    print("\n[3] 离线: 改历史的 5 种做法, 真实 API 上哪些 thinking 块会失效")
    rows, n_msgs, total = history_rows()
    assert rows["追加下一问"] == (n_msgs, [], 0), "只追加时前缀全保留, 没有 thinking 块失效"
    keep, stale, _ = rows["清旧工具结果 (m14 第 1 档)"]
    assert keep == 2, f"第一个工具结果在第 3 条消息, 前缀只保留 2 条, 实际 {keep}"
    assert len(stale) == 3, f"被清掉的结果之后还有 3 个 thinking 块, 应全部失效, 实际 {len(stale)}"
    assert len(rows["重拼 system (m04 每问检索记忆)"][1]) == total, "system 一变, 全部 thinking 块失效"
    keep, stale, dropped = rows["摘要旧轮, 保留当前轮 (m14 第 2 档)"]
    assert (keep, len(stale), dropped) == (0, 2, 2), f"保留的当前轮有 2 个 thinking 块且全部失效, 旧轮 2 个没再发送, 实际 {(keep, len(stale), dropped)}"
    assert rows["整段换成摘要 (simple compaction)"][1:] == ([], total), "整段换成摘要时不带旧 thinking, 没有失效, 但推理全丢"

    print("\n[4] 在线: 真实模型")
    try:
        llm = ClaudeLLM()
    except RuntimeError as exc:
        print(f"  跳过: {exc}。设置好之后重跑本 demo 即可, 其余模块不受影响。")
        return
    live = Agent(llm, ToolRegistry([CalculatorTool()]), PermissionGate("auto"), max_turns=4, name="claude")
    final = live.run("请用 calculator 工具计算 (17 * 23) + 5, 然后只回答数字。")
    assert validate_transcript(live.messages) == [], "真实模型跑出的 transcript 配对也应合法"
    assert "396" in final, f"(17 * 23) + 5 应等于 396, 实际回答: {final}"
    print("\n  OK: 换模型 = 换一个实现了 next() 的对象。")


if __name__ == "__main__":
    main()
