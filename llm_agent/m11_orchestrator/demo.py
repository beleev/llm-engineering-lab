"""M11 — Orchestrator–Workers: lead 扇出多个并行子 agent, 再综合。

没有它: 一个 agent 串行读完所有资料, 所有原文都堆在同一个上下文里 —— 慢, 而且越读越糊。
关键设计:
  - 扇出不需要新机制: lead 在一个 turn 里发多个 delegate tool_use, loop 的线程池并行执行它们。
  - 每类 worker 有不同的工具集 (researcher 只能检索, calculator 只能算), 上下文互相隔离。
  - 记两本账: 总 token (lead + 所有 worker) 往往更高; lead 的峰值上下文却小得多 —— 多智能体买的是
    并行度和干净的主上下文, 不是省钱。
  - 每个 worker 的 transcript 落盘, 出问题能逐个复盘。
  - delegate 是高风险工具, lead 要有一条 allow 规则才能扇出。worker 跑满 max_turns 时结果是 is_error。
不防: worker 的权限门没有规则也没有 hooks, lead 的 deny 规则管不到 worker (见 core/subagents.py)。
对应: Anthropic multi-agent research system (lead agent + subagents); Claude Code 并行 Task。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    CalculatorTool,
    Decision,
    DelegateTool,
    HookManager,
    PermissionGate,
    PermissionRule,
    RuleBasedLLM,
    TfidfIndex,
    ToolRegistry,
    VectorSearchTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

# 每篇后面拼 25 句 "Background detail. ": 故意把文档撑长, 看原文堆在谁的上下文里
DOCS = {
    "kv_cache": "The kv cache stores past keys and values so decoding is incremental. " + "Background detail. " * 25,
    "lora": "LoRA trains low rank adapters, cutting trainable weights below one percent. " + "Background detail. " * 25,
    "dpo": "DPO aligns a model from preference pairs without a reward model. " + "Background detail. " * 25,
}
PROMPT = "并行调研: 调研 kv cache 如何加速解码; 调研 lora 为什么省显存; 计算 4096 * 32"


def main() -> None:
    banner("M11 - Orchestrator-Workers")
    index = TfidfIndex(DOCS)
    stops = []
    hooks = HookManager()
    hooks.register("subagent_stop", lambda agent_type, summary: stops.append(agent_type))

    with tempfile.TemporaryDirectory(prefix="llm_agent_orch_") as tmp:
        delegate = DelegateTool(
            {
                "researcher": lambda: ToolRegistry([VectorSearchTool(index, k=2)]),
                "calculator": lambda: ToolRegistry([CalculatorTool()]),
            },
            hooks=hooks,
            transcript_dir=Path(tmp),
        )
        gate = PermissionGate("auto", [PermissionRule("delegate", "", Decision.ALLOW, "fan-out is reviewed")])
        lead = Agent(RuleBasedLLM(), ToolRegistry([delegate]), gate, context_budget_chars=4000, name="lead")
        final = lead.run(PROMPT)
        print("\n" + final)

        fan_out = lead.messages[1].tool_uses()  # lead 的第一次回复: 一个 turn 里的全部委托
        assert [b["input"]["agent_type"] for b in fan_out] == ["researcher", "researcher", "calculator"], (
            "三个子任务应在同一个 turn 里扇出: 两个调研交给 researcher, 算式交给 calculator"
        )
        assert len(lead.messages[2].tool_results()) == 3, "三个 worker 的摘要应放在同一条 user 消息里"
        assert validate_transcript(lead.messages) == [], "并行委托的 tool_use / tool_result 应全部配对"
        # 并行执行, 完成的先后不固定, 所以排序后再比
        assert sorted(stops) == ["calculator", "researcher", "researcher"], (
            f"每个 worker 结束时应各触发一次 subagent_stop, 实际: {stops}"
        )
        assert final.startswith("综合 3 个子任务结果"), f"lead 应综合 3 个 worker 的结果, 实际: {final}"
        assert "4096 * 32 = 131072" in final, "calculator worker 的结果应出现在最终回答里"

        print("\n[各 worker]")
        for child in delegate.children:
            used = [b["name"] for m in child["messages"] for b in m.tool_uses()]
            print(f"  {child['path'].name:<28} tools={used} input_tokens={child['usage']['input_tokens']}")
            assert child["path"].exists(), f"worker 的 transcript 应已落盘: {child['path']}"
            assert used == (["calculator"] if child["type"] == "calculator" else ["search_docs"]), (
                f"{child['type']} 应只用自己工具集里的工具, 实际: {used}"
            )

        print("\n[对照] 同一任务交给单个 agent (自己检索 + 自己算)")
        solo = Agent(
            RuleBasedLLM(),
            ToolRegistry([VectorSearchTool(index, k=2), CalculatorTool()]),
            PermissionGate("auto"),
            context_budget_chars=4000,
            name="solo",
        )
        solo.run(PROMPT, verbose=False)

        workers_total = sum(c["usage"]["input_tokens"] for c in delegate.children)
        total = lead.usage["input_tokens"] + workers_total
        print("\n[token 记账]")
        kv("lead 峰值上下文", lead.usage["peak_context_tokens"])
        kv("solo 峰值上下文", solo.usage["peak_context_tokens"])
        kv("lead + worker 总输入", f"{total} (lead {lead.usage['input_tokens']} + worker {workers_total})")
        kv("solo 总输入", solo.usage["input_tokens"])
        # 原文留在 worker 里
        assert lead.usage["peak_context_tokens"] < solo.usage["peak_context_tokens"], (
            "lead 只收摘要, 它的上下文峰值应小于自己读原文的 solo"
        )
        # 代价: 总花费更高
        assert total > solo.usage["input_tokens"], (
            f"lead 加全部 worker 的总输入应高于 solo, 实际: {total} vs {solo.usage['input_tokens']}"
        )

    print("\n  OK: 多智能体 = 用更多总 token, 换并行度和不被原文淹没的 lead 上下文。")


if __name__ == "__main__":
    main()
