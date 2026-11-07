"""ClaudeLLM: 用真实的 Claude Messages API 实现同一个 LLM 协议 (opt-in)。

默认 demo 不会实例化 ClaudeLLM, 不发网络请求。
m15 和 m19 导入了本文件的 to_api_messages, 它是纯函数, 不需要 SDK 和 key。

没有它: 学员会怀疑 "toy LLM 能跑, 换真模型是不是要重写 loop" —— 不用, 只换这一个对象。
关键设计:
  - transcript 本来就是 Messages API 的 block 格式, 转换只需处理三件事 (to_api_messages, 纯函数, 可离线测试):
    开头的 system 消息 → 顶层 system 参数; 中途 harness 注入的 system 消息 → user 侧 <system-reminder> 文本;
    相邻同角色消息合并, 且 tool_result 必须排在该 user 消息的最前面。
  - 响应的 content blocks 原样存回 transcript (raw_content): thinking block 必须原封不动回传。
  - tool_use.id 用 API 给的; 参数是已解析的 dict, 不要对序列化后的字符串做匹配。
依赖: `pip install anthropic` + 环境变量 ANTHROPIC_API_KEY。模型默认 claude-opus-5
(另有 claude-sonnet-5 / claude-haiku-4-5, 用 LLM_AGENT_MODEL 覆盖)。
未启用服务端 refusal fallbacks (需 beta 端点: betas=["server-side-fallback-2026-07-01"], fallbacks="default"),
这里只识别 stop_reason == "refusal", 返回一句固定文本。
"""

from __future__ import annotations

import os
import time
import random
from contextlib import nullcontext
from llm_agent.core.streaming import MessageStream, StreamError
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import Message, ModelAction, ToolCall


def to_api_messages(messages: List[Message]) -> Tuple[str, List[Dict[str, Any]]]:
    """内部 transcript → (system 字符串, Messages API 的 messages 列表)。

    Message.name 字段在转换时丢掉, 真实 API 的消息里没有它。
    """
    system: List[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system" and not out:  # 第一条 user / assistant 消息之前的 system 消息
            system.append(msg.text)
            continue
        if msg.role == "system":  # 对话中途的 harness 注入: 降级成 user 侧提示, 任何模型都支持
            role, blocks = "user", [{"type": "text", "text": f"<system-reminder>\n{msg.text}\n</system-reminder>"}]
        else:
            role, blocks = msg.role, list(msg.blocks)
        if out and out[-1]["role"] == role:  # 相邻同角色并成一条: tool_result 和随后注入的提示同属一条 user 消息
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})
    for item in out:  # API 要求 tool_result 在 user 消息最前面; sorted 稳定, 其余顺序不变
        item["content"] = sorted(item["content"], key=lambda b: b["type"] != "tool_result")
    return "\n\n".join(system), out


class ClaudeLLM:
    """真实模型适配器。无注入客户端时检查 SDK/key；注入客户端供离线协议验证。"""

    def __init__(self, model: str = "", max_tokens: int = 16000, *, client=None,
                 streaming=True, max_attempts=2, sleep=time.sleep, tracer=None, on_text=None) -> None:
        if max_attempts < 1 or max_tokens < 1:
            raise ValueError('预算必须为正')
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("需要可选依赖: pip install anthropic") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("未设置 ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(max_retries=0)  # 只保留一层重试预算
        self.client = client
        self.streaming = streaming
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.tracer = tracer
        self.on_text = on_text
        self.last_usage = None
        self.last_attempts = []
        self._request_id = 0
        self.model = model or os.environ.get("LLM_AGENT_MODEL", "claude-opus-5")
        self.max_tokens = max_tokens
        self._tools: List[Dict[str, Any]] = []  # 上一次请求带的工具定义, 摘要请求时沿用

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        """发一次逻辑请求；默认流式累积，完整闭合后才交给 Agent 执行。真实客户端会计费。"""
        system, api_messages = to_api_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if "haiku" not in self.model:  # haiku-4-5 不支持 adaptive thinking (它用 budget_tokens), 这里直接不开
            kwargs["thinking"] = {"type": "adaptive"}
        if system:
            kwargs["system"] = system
        # 历史里有 tool_use/tool_result 时 API 要求必须带 tools; 摘要请求传的是 [] → 沿用上一次的定义
        self._tools = tools or self._tools
        if self._tools:
            kwargs["tools"] = self._tools  # Tool.schema() 已经是 {name, description, input_schema}
        self.last_usage = None
        self.last_attempts = []
        self._request_id += 1
        for attempt in range(1, self.max_attempts + 1):
            accumulator = MessageStream(self.on_text)
            stream = None
            span = None
            span_context = self.tracer.span('model_attempt', operation=f'api-{self._request_id}', attempt=attempt) if self.tracer else nullcontext()
            try:
                with span_context as span:
                    if self.streaming:
                        stream = self.client.messages.create(**kwargs, stream=True)
                        for event in stream:
                            accumulator.feed(event)
                        raw = accumulator.finish()
                        stop_reason = accumulator.stop_reason
                        usage = accumulator.usage
                    else:
                        response = self.client.messages.create(**kwargs)
                        raw = [block.model_dump(exclude_none=True) for block in response.content]
                        stop_reason = response.stop_reason
                        usage = response.usage.model_dump(exclude_none=True)
                    self.last_usage = dict(usage)
                    self.last_attempts.append({'attempt': attempt, 'status':'ok', 'usage':dict(usage)})
                    if span is not None:
                        span['usage'] = dict(usage)
                        span['usage_source'] = 'provider'
                break
            except Exception as exc:
                if span is not None:
                    span['usage'] = dict(accumulator.usage)
                    span['usage_source'] = 'provider_partial'
                    span['billing_complete'] = False
                self.last_attempts.append({'attempt':attempt,'status':'error','usage':dict(accumulator.usage),
                                           'billing_complete':False,'error_type':type(exc).__name__})
                status = getattr(exc, 'status_code', None)
                transient = status in (408,429,500,502,503,504,529) or isinstance(exc,(ConnectionError,TimeoutError)) or type(exc).__name__ in ('APIConnectionError','APITimeoutError')
                # 接收过 message_start 的尝试可能已计费/展示部分结果；不自动重发。
                if accumulator.started or not transient or attempt == self.max_attempts:
                    raise
                retry_after = getattr(exc, 'response', None)
                retry_after = getattr(retry_after, 'headers', {}).get('retry-after', '')
                try: delay = min(5., max(0., float(retry_after)))
                except (ValueError,TypeError): delay = min(5., .25 * 2**(attempt-1)) * random.uniform(.8,1.2)
                self.sleep(delay)
            finally:
                if stream is not None and hasattr(stream,'close'):
                    stream.close()

        if stop_reason == "refusal":  # 先看 stop_reason, 再读 content
            return ModelAction.final("[模型拒绝了该请求]")
        text = "\n".join(b["text"] for b in raw if b["type"] == "text")
        calls = [ToolCall(b["name"], b["input"], id=b["id"]) for b in raw if b["type"] == "tool_use"]
        if stop_reason == "max_tokens":  # 可能截在 tool_use 中间: 参数不完整, 绝不执行
            return ModelAction.final(text + "\n[输出在 max_tokens 处被截断]")
        if stop_reason not in ('end_turn','tool_use','stop_sequence','refusal','max_tokens'):
            raise StreamError(f'未支持的停止原因: {stop_reason}')
        if calls and stop_reason != 'tool_use':
            raise StreamError('工具块与停止原因不一致')
        if calls and tools:
            return ModelAction(kind="tool", content=text, tool_calls=calls, raw_content=raw)
        # 没给工具 (如摘要请求) 模型却发了 tool_use: 不执行, 只取文本, 也不存 raw。
        # raw 里的 tool_use 没有对应的 tool_result, 写进 transcript 会让配对不合法
        return ModelAction(kind="final", content=text, raw_content=None if calls else raw)
