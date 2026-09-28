"""M14 — Context Engineering: 把"放什么进上下文"当成工程问题来做。

没有它: 上下文只增不减。每一轮都在为几十轮前的工具输出付费, 模型的注意力被稀释, 最后撑爆窗口。
关键设计 (都已接进 core/agent.py 的 loop):
  1. 工具结果清理  超预算先把旧 tool_result 正文换成占位符 —— 只改发给模型的视图, transcript 不动。
  2. 摘要压缩      还超 → 先触发 pre_compact hook (人指定必留信息) → 模型写摘要 → 真的替换掉历史;
                   JSONL 追加 compact_boundary, 所以 resume 读回来的上下文也变小了 (文件本身只增不减, 供审计)。
  3. 即时检索      上下文里只放"有哪些文件"的索引, 内容用到才读 (just-in-time), 而不是预先全塞进去。
  4. 记忆工具      模型自己往 /memories 写, 新会话再读回来 —— 跨会话的状态不占用任何一轮的上下文。
对应: Anthropic "effective context engineering for AI agents"; Claude API 的 context editing / compaction / memory tool。
差异: 这里的预算按字符算, 不按 token。记忆工具只实现了真实 memory_20250818 六个命令里的四个,
  参数名也做了简化 (见 core/sandbox.py)。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from llm_agent.core import (
    Agent,
    HookManager,
    JsonlSessionStore,
    MemoryTool,
    Message,
    ModelAction,
    PermissionGate,
    ReadFileTool,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    total_chars,
    validate_transcript,
)
from llm_agent.core.utils import banner, estimate_tokens, kv

TOPICS = ["kv cache", "lora", "dpo", "sampling", "rope"]
BUDGET = 900  # 上下文预算 (字符): 装不下全部轮次, 清理和压缩都会被触发
# 每篇 = 一句带编号的关键事实 (FACT-1 … FACT-5) + 8 句重复的背景。背景用来把工具结果撑长
DOCS = {t.replace(" ", "_"): f"{t} key fact: FACT-{i}. " + f"long background about {t}. " * 8 for i, t in enumerate(TOPICS, 1)}


class SpyLLM:
    """记录模型每次实际看到的上下文 —— 上下文工程的第一步是能看见上下文。"""

    def __init__(self) -> None:
        self.inner, self.contexts, self.summaries = RuleBasedLLM(), [], []

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        # 摘要请求不带 tools, 单独记: contexts 只放干活的请求, summaries 放写摘要的请求
        (self.contexts if tools else self.summaries).append(list(messages))
        return self.inner.next(messages, tools)


def long_session(path: Path, compaction: str, hooks: HookManager = None):
    """用 BUDGET 字符的预算连续检索 TOPICS 里的主题, 返回 (agent, 记录了每次上下文的 SpyLLM)。"""
    spy = SpyLLM()
    agent = Agent(
        spy,
        ToolRegistry([SearchDocsTool(DOCS)]),
        PermissionGate("auto"),
        hooks=hooks,
        store=JsonlSessionStore(path),
        context_budget_chars=BUDGET,
        compaction=compaction,
        name=compaction,
    )
    for topic in TOPICS:
        agent.run(f"检索 {topic}", verbose=False)
    return agent, spy


def main() -> None:
    banner("M14 - Context Engineering")

    with tempfile.TemporaryDirectory(prefix="llm_agent_ctx_") as tmp:
        tmp_path = Path(tmp)

        print(f"\n[1] {len(TOPICS)} 轮检索, 预算 {BUDGET} 字符: 清理 → 压缩 逐级触发")
        compact_calls = []
        hooks = HookManager()
        # list.append 返回 None, 所以 `append(...) or "..."` 先记一笔, 再把后面的字符串作为返回值
        hooks.register("pre_compact", lambda messages: compact_calls.append(len(messages)) or "用户只关心 FACT 编号")
        agent, spy = long_session(tmp_path / "summary.jsonl", "summary", hooks)

        assert any("[cleared:" in m.text for ctx in spy.contexts for m in ctx), (
            "第 1 档应生效过: 模型看到的上下文里应出现过清理后的占位符"
        )
        assert agent.compactions >= 1, f"第 2 档应生效过: {BUDGET} 字符装不下 {len(TOPICS)} 轮, 至少压缩一次"
        assert len(compact_calls) == agent.compactions, (
            f"每次压缩前都应触发 pre_compact hook, 实际: hook {len(compact_calls)} 次, 压缩 {agent.compactions} 次"
        )
        summary = agent.messages[0]  # 压缩后, 摘要排在 transcript 的第一条
        print("  " + summary.text.replace("\n", "\n  ")[:600])
        assert summary.name == "compact_summary", f"压缩后的第一条消息应是摘要, 实际: {summary.name}"
        assert "检索 kv cache" in summary.text, "最早的用户目标应留在摘要里"
        assert "用户只关心 FACT 编号" in summary.text, "pre_compact hook 指定的必留信息应留在摘要里"
        # 写摘要也是一次模型调用: 它读的是整段旧历史, token 同样记进 usage
        summary_tokens = sum(estimate_tokens(m.text) for ctx in spy.summaries for m in ctx)
        sent = summary_tokens + sum(estimate_tokens(m.text) for ctx in spy.contexts for m in ctx)
        kv("压缩次数", agent.compactions)
        kv("输入 tokens (其中摘要请求)", f"{agent.usage['input_tokens']} ({summary_tokens})")
        kv("峰值上下文 tokens", agent.usage["peak_context_tokens"])
        assert len(spy.summaries) == agent.compactions, "每次压缩应正好问模型一次"
        assert agent.usage["input_tokens"] == sent, (
            f"usage 应记下发给模型的全部 token (含摘要请求), 实际记了 {agent.usage['input_tokens']}, 发了 {sent}"
        )

        print("\n[2] 压缩真的缩小了'恢复出来的会话', 审计日志一条没少")
        store = JsonlSessionStore(tmp_path / "summary.jsonl")
        resumed, audit = store.load(), store.load_all()
        kv("resume 视图", f"{len(resumed)} 条 / {total_chars(resumed)} 字符")
        kv("审计全量", f"{len(audit)} 条 / {total_chars(audit)} 字符")
        assert total_chars(resumed) < total_chars(audit) * 0.6, (
            f"resume 读回的上下文应不到全量的 60%, 实际: {total_chars(resumed)} vs {total_chars(audit)}"
        )
        # 每轮 4 条: 用户 prompt / tool_use / tool_result / 最终回答
        assert len(audit) == 4 * len(TOPICS), f"审计视图应一条不少: 5 轮 × 4 条, 实际: {len(audit)}"
        assert [m.text for m in resumed] == [m.text for m in agent.messages], (
            "从磁盘恢复出的上下文应等于内存里压缩后的上下文"
        )
        assert validate_transcript(resumed) == [], "压缩不应切断任何 tool_use / tool_result 配对"
        agent_b = Agent(RuleBasedLLM(), ToolRegistry([SearchDocsTool(DOCS)]), PermissionGate("auto"), store=store, load_history=True, name="resumed")
        assert "FACT-2" in agent_b.run("检索 lora", verbose=False), "在压缩过的会话上 resume 后应能继续正常工作"

        print("\n[3] 对照: truncate 策略只裁剪视图 —— 每轮重新裁, 历史和 resume 都不会变小")
        agent_t, _ = long_session(tmp_path / "truncate.jsonl", "truncate")
        store_t = JsonlSessionStore(tmp_path / "truncate.jsonl")
        assert agent_t.compactions == 0, "truncate 策略不做摘要压缩"
        assert len(store_t.load()) == len(store_t.load_all()) == len(agent_t.messages), (
            "truncate 只裁视图: resume 视图、审计视图、内存里的历史应一样长"
        )
        kv("truncate resume 视图", f"{len(store_t.load())} 条 (没有变小)")
        # 更糟的是: 截断把 tool_result 拍平成普通 user 文本, 模型把它当成了新的用户输入, 后几轮直接答非所问
        derailed = [m.text for m in agent_t.messages if m.role == "assistant" and "无需工具" in m.text]
        kv("truncate 下答非所问的轮数", f"{len(derailed)}/{len(TOPICS)}")
        good = [m.text for m in agent.messages if m.role == "assistant" and "无需工具" in m.text]
        assert derailed, "truncate 下应出现答非所问的轮次 (模型没调工具就直接回答)"
        assert not good, f"summary 策略下不应有答非所问的轮次, 实际: {good}"

        print("\n[4] 即时检索 vs 预加载")
        docs_dir = tmp_path / "workspace" / "docs"
        docs_dir.mkdir(parents=True)
        for name, body in DOCS.items():
            (docs_dir / f"{name}.md").write_text(body, encoding="utf-8")
        index = "可读文件:\n" + "\n".join(f"- docs/{name}.md" for name in DOCS)
        preload = "知识库全文:\n" + "\n".join(DOCS.values())
        jit = Agent(RuleBasedLLM(), ToolRegistry([ReadFileTool(tmp_path / "workspace")]), PermissionGate("auto"), system_prompt=index, name="jit")
        final = jit.run("读取文件 docs/lora.md", verbose=False)
        kv("预加载: 每次调用都要带的 tokens", estimate_tokens(preload))
        kv("即时检索: 峰值上下文 tokens", jit.usage["peak_context_tokens"])
        assert "FACT-2" in final, f"即时读取应拿到 lora 文档里的事实, 实际: {final}"
        assert jit.usage["peak_context_tokens"] < estimate_tokens(preload) * 0.5, (
            "只读一篇的峰值上下文应不到预加载全部 5 篇的一半"
        )

        print("\n[5] 记忆工具: 会话 1 写, 全新的会话 2 读")
        memory_root = tmp_path / "agent_home"
        Agent(RuleBasedLLM(), ToolRegistry([MemoryTool(memory_root)]), PermissionGate("auto"), name="s1").run(
            "存入长期记忆: 这个项目用 pytest, 不用 unittest"
        )
        assert (memory_root / "memories" / "notes.md").exists(), "会话 1 的记忆应已写成文件"
        for escape in ("/memories/../escape.md", "/memories_evil/x.md"):  # 围栏立在 memories/ 上, 不是 root 上
            call = ToolCall("memory", {"command": "create", "path": escape, "text": "x"})
            assert not ToolRegistry([MemoryTool(memory_root)]).execute(call).ok, (
                f"写到 memories/ 之外的路径应被拒绝: {escape}"
            )
        assert not (memory_root / "escape.md").exists(), "被拒绝的写入不应在记忆目录之外留下文件"
        session2 = Agent(RuleBasedLLM(), ToolRegistry([MemoryTool(memory_root)]), PermissionGate("auto"), name="s2")
        recalled = session2.run("查看长期记忆")
        # 没有任何历史, 知识来自文件
        assert "pytest" in recalled, f"全新的会话应从文件里读回会话 1 记下的内容, 实际: {recalled}"
        assert len(session2.messages) == 4, (
            f"会话 2 应只有本轮的 4 条消息, 没有继承任何历史, 实际: {len(session2.messages)}"
        )

    print("\n  OK: 上下文是预算 —— 便宜的先清, 贵的再压, 能现取的不预存, 要跨会话的写文件。")


if __name__ == "__main__":
    main()
