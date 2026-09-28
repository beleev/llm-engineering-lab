"""子智能体: 隔离上下文里干活, 只把摘要交回父级。

没有它: 一次调研读 20 篇文档, 全部堆进主上下文 —— 主 agent 很快被无关细节淹没, 每一轮还要为它们重复付 token。
关键设计:
  - delegate 只是一个普通工具: 父模型在同一 turn 发多个 delegate tool_use, agent loop 的并行执行
    自然就成了 orchestrator-workers 扇出, 不需要另一套调度器。
  - 每种 agent_type 有自己的工具集与权限门; 子级拿不到父级的工具, 也看不到父级的对话。
  - 子 transcript 落盘 (可审计), 但不回流父上下文; usage 分开记账, 才能看清"总花费 vs 父上下文占用"。
  - delegate 标成 risk=high: 父级读过不可信数据之后, 污点锁拦下委托, 不能借一个干净的子级绕过去。
    它也标了 untrusted_output: 子级读到的不可信数据会写进摘要, 摘要回到父级时照样包装、置污点。
  - 子级跑满 max_turns 时返回 ok=False (is_error), 父级从结果本身就知道委托没完成。
代价: 总 token 通常更多 (每个子级都要重建上下文) —— 换来的是并行和干净的主上下文。
不防 (子级的权限比看上去松):
  - 子级的权限门是 auto 模式、没有任何规则。父级的 deny 规则没有传下来
  - 子级没有 hooks。父级的 pre_tool_use 拦截管不到子级
  所以子级能做什么, 只取决于 agent_types 给它的工具集和 auto 分类器。
简化: 子级只有 max_turns 这一个上限。没有超时、重试、取消, 也没有总 token 预算;
  这些属于任务调度, 不是隔离上下文要讲的事。
对应: Claude Code 的 Task/Agent 工具与 SubagentStop hook; Anthropic 多智能体 research 系统的 lead + subagents。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_agent.core.agent import MAX_TURNS_REACHED, Agent
from llm_agent.core.hooks import HookManager
from llm_agent.core.llm import LLM
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM
from llm_agent.core.utils import shorten


class DelegateTool(Tool):
    """把一个子任务交给一个全新的子 Agent, 只把它的摘要作为工具结果交回。

    构造参数:
      agent_types        类型名 → 造工具集的函数。每次委托都现造一份
      hooks              父级的 HookManager, 只用来触发 subagent_stop。不会传给子级
      transcript_dir     子 transcript 落盘的目录; 不给就不落盘
      llm_factory        造子级模型的函数
      max_summary_chars  交回父级的摘要最多多少字符
      guardrails         传给子级的护栏
    """

    name = "delegate"
    description = "Run an isolated child agent on a subtask and return only its summary."
    read_only = False  # 子级有自己的权限门、可能会写: 不标成只读, 否则 plan 模式能靠委托绕过
    # 高风险: auto 模式下要有 allow 规则才放行; 父级被污染后, 污点锁会拦下它。
    # 标成 low 的话, 读过注入文本的父级可以把"去执行 shell"写进 task, 交给一个没有污点的子级
    risk = "high"
    untrusted_output = True  # 摘要里可能带着子级读到的不可信数据

    def __init__(
        self,
        agent_types: Dict[str, Callable[[], ToolRegistry]],
        hooks: Optional[HookManager] = None,
        transcript_dir: Optional[Path] = None,
        llm_factory: Callable[[], LLM] = RuleBasedLLM,
        max_summary_chars: int = 200,
        guardrails: Optional[Any] = None,
    ) -> None:
        self.guardrails = guardrails  # 子级同样会读不可信数据、同样会落盘: 护栏要跟着下去
        self.max_summary_chars = max_summary_chars  # 硬上限: 子级再啰嗦, 也淹不了父上下文
        self.agent_types, self.hooks, self.transcript_dir, self.llm_factory = agent_types, hooks, transcript_dir, llm_factory
        self.parameters = {
            "type": "object",
            "properties": {"task": {"type": "string"}, "agent_type": {"type": "string", "enum": sorted(agent_types)}},
            "required": ["task"],
            "additionalProperties": False,
        }
        self.children: List[Dict[str, Any]] = []  # 每个子级的 {type, task, messages, usage, path}
        self._lock = threading.Lock()  # execute 会被线程池并发调用

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """造一个子 Agent 跑 args["task"], 返回截断到 max_summary_chars 的摘要。

        会被父级的线程池并发调用: 每次调用造自己的子 Agent, 只有登记 children 那一步加锁。
        """
        agent_type = args.get("agent_type") or sorted(self.agent_types)[0]  # 没指定就取名字排第一的类型
        with self._lock:  # 锁内占号: 并行委托时编号不会重复
            index = len(self.children)
            record: Dict[str, Any] = {"type": agent_type, "task": args["task"]}
            self.children.append(record)
        path = self.transcript_dir / f"child_{index:02d}_{agent_type}.jsonl" if self.transcript_dir else None

        child = Agent(
            llm=self.llm_factory(),
            tools=self.agent_types[agent_type](),  # 全新的工具集实例: 子级之间也不共享状态
            # 权限固定用 auto。子级没有人可问: default 模式每次都要问, 结果是全部拒绝, 什么也干不了;
            # auto 让低/中风险的工具能跑, 拿不准的 (high) 仍因为没人可问而拒绝。
            # 这扇门没有规则, 也没传 hooks: 父级的 deny 规则和 pre_tool_use hook 都管不到子级。
            # 要补上, 把父级的 rules 和 hooks 传进来。
            permissions=PermissionGate(mode="auto"),
            store=JsonlSessionStore(path) if path else None,
            guardrails=self.guardrails,
            system_prompt=f"You are an isolated {agent_type} subagent. Return a short summary.",
            max_turns=4,
            name=f"child-{index}",
        )
        summary = child.run(args["task"], verbose=False)
        # 完整的子 transcript 留在 record 里给 demo 和审计看, 不进父上下文
        record.update(messages=child.messages, usage=child.usage, path=path)
        if self.hooks:
            self.hooks.on_subagent_stop(agent_type, summary)
        if summary == MAX_TURNS_REACHED:  # 子级没做完: 标成失败, 父级从 is_error 就能看出来, 不用读文本
            reason = f"[{agent_type}] subagent stopped after max_turns={child.max_turns} without a final answer"
            return ToolResult(self.name, reason, ok=False)
        return ToolResult(self.name, f"[{agent_type}] {shorten(summary, self.max_summary_chars)}")
