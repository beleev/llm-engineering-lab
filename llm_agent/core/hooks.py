"""生命周期 hooks: 包在模型循环外面的确定性代码。

没有它: 团队规范只能写进 prompt 里"求"模型遵守 —— 花 token 且不保证执行。
关键设计:
  - hook 只能 拦截 / 改写调用 / 追加上下文, 三者都不会污染工具数据本身:
    追加的文字作为独立 system 消息进 transcript, 不拼进用户 prompt 或 tool_result。
  - pre_tool_use 在权限门之前运行, 改写后的调用仍要过门 (hook 不是提权通道)。
    反过来排 (先过门再跑 hook) 的话, 门批准的是改写前的调用, 实际执行的那个没人审过。
对应: Claude Code hooks 的同名事件 SessionStart / UserPromptSubmit / PreToolUse /
PostToolUse / PreCompact / Stop / SubagentStop。
差异:
  - 这里的 hook 只能拦和改, 不能批准。真实的 PreToolUse hook 可以返回 allow 直接跳过询问
    (deny 规则仍然生效)。
  - 这里的 hook 是进程内的 Python 函数。真实的 hook 是外部命令, 从 stdin 收 JSON。
  - 这里 SessionStart 的来源只有 startup / resume。真实的有四种: startup / resume / clear / compact。
  - 真实的 Stop 还能阻止结束, 这里只做通知。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult

EVENTS = (
    "session_start",  # fn(source: "startup"|"resume") -> str | None
    "user_prompt_submit",  # fn(prompt) -> HookResult
    "pre_tool_use",  # fn(call) -> HookResult
    "post_tool_use",  # fn(result) -> str | None
    "pre_compact",  # fn(messages) -> str | None   压缩时必须保留的要点
    "stop",  # fn(final_text) -> None
    "subagent_stop",  # fn(agent_type, summary) -> None
)


@dataclass
class HookResult:
    """user_prompt_submit 和 pre_tool_use 两种 hook 的返回值。四个字段不是每个事件都认。

    block / reason      两个事件都认: block=True 拦下, reason 是给人和模型看的原因
    updated_call        只有 pre_tool_use 认: 用这个调用替换模型要的那个
    additional_context  只有 user_prompt_submit 认: 追加一条 system 消息。
                        pre_tool_use 的 hook 填了它也会被丢弃
    """

    block: bool = False
    reason: str = ""
    updated_call: Optional[ToolCall] = None
    additional_context: str = ""


class HookManager:
    """按事件名登记 hook 函数。agent 在固定的时机调用 on_* 方法, 由它逐个执行。

    同一事件可以登记多个函数, 按登记顺序执行。hook 函数返回 None 表示没有意见。
    hook 函数抛异常不会被接住, 会直接中断 run()。
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable]] = {event: [] for event in EVENTS}

    def register(self, event: str, fn: Callable) -> None:
        """登记一个 hook。event 必须是 EVENTS 里的名字, 函数签名见 EVENTS 的注释。"""
        if event not in self._hooks:
            raise ValueError(f"unknown hook event: {event}")
        self._hooks[event].append(fn)

    def _collect(self, event: str, *args) -> List[str]:
        """跑完该事件的全部 hook, 收集非空返回值 (转成字符串)。"""
        outputs = (fn(*args) for fn in self._hooks[event])
        return [str(out) for out in outputs if out]

    def on_session_start(self, source: str) -> List[str]:
        """会话开始。source 是 "startup" 或 "resume"。返回要注入上下文的文字, 每个 hook 一段。"""
        return self._collect("session_start", source)

    def on_user_prompt_submit(self, prompt: str) -> HookResult:
        """用户 prompt 进上下文之前。任何一个 hook 拦下就立刻返回, 后面的 hook 不再跑。

        没人拦: 把各 hook 的 additional_context 用换行拼起来返回。updated_call 在这里不起作用。
        """
        extra = []
        for fn in self._hooks["user_prompt_submit"]:
            result = fn(prompt) or HookResult()  # 返回 None = 无意见
            if result.block:
                return result
            if result.additional_context:
                extra.append(result.additional_context)
        return HookResult(additional_context="\n".join(extra))

    def on_pre_tool_use(self, call: ToolCall) -> HookResult:
        """工具调用过权限门之前。任何一个 hook 拦下就立刻返回。

        没人拦: 返回的 updated_call 是最终要执行的调用。没有 hook 改写时它就是传入的 call 本身。
        这里没有"批准"这个选项, 调用之后还要过污点锁和权限门。
        """
        current = call
        for fn in self._hooks["pre_tool_use"]:
            result = fn(current) or HookResult()
            if result.block:
                return result
            if result.updated_call:
                current = result.updated_call  # 链式改写: 后一个 hook 看到前一个的结果
        return HookResult(updated_call=current)

    def on_post_tool_use(self, result: ToolResult) -> str:
        """工具执行之后 (被拦下的调用不触发)。返回的文字由 agent 写成独立的 system 消息。

        hook 拿到的是 ToolResult 对象本身, 改它的字段会影响回填给模型的内容。
        """
        return "\n".join(self._collect("post_tool_use", result))

    def on_pre_compact(self, messages: list) -> str:
        """摘要压缩之前。messages 是将被压缩的旧历史; 返回摘要里必须保留的要点。"""
        return "\n".join(self._collect("pre_compact", messages))

    def on_stop(self, final_text: str) -> None:
        """run() 结束时通知一声。返回值被忽略, 不能阻止结束。"""
        self._collect("stop", final_text)

    def on_subagent_stop(self, agent_type: str, summary: str) -> None:
        """子 agent 跑完时通知父级。summary 是截断前的完整摘要。由 DelegateTool 调用。"""
        self._collect("subagent_stop", agent_type, summary)
