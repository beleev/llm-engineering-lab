"""ClaudeLLM: 用真实的 Claude Messages API 实现同一个 LLM 协议 (opt-in)。

默认 demo 不会实例化 ClaudeLLM, 不发网络请求。
m15 和 m19 导入了本文件的 to_api_messages, 它和 build_request、stale_thinking 都是纯函数, 不需要 SDK 和 key。

没有它: 学员会怀疑 "toy LLM 能跑, 换真模型是不是要重写 loop" —— 不用, 只换这一个对象。
关键设计:
  - transcript 本来就是 Messages API 的 block 格式, 转换只需处理三件事 (to_api_messages, 纯函数, 可离线测试):
    开头的 system 消息 → 顶层 system 参数; 中途 harness 注入的 system 消息 → user 侧 <system-reminder> 文本;
    相邻同角色消息合并, 且 tool_result 必须排在该 user 消息的最前面。
  - 请求参数只按当前代模型的约束构造 (build_request, 纯函数):
    thinking 一律 adaptive, 只有 claude-haiku-4-5 及更老的模型用 budget_tokens;
    深浅用可选的 output_config.effort 调 (low / medium / high / xhigh / max), 不在顶层;
    不发 temperature / top_p / top_k, 不发强制 tool_choice, 末尾是 assistant 消息 (prefill) 直接拒绝。
  - 响应先看 stop_reason 再读 content; content blocks 原样存回 transcript (raw_content), thinking block 不改一个字节。
  - 历史只追加: thinking 块的签名绑定产生它时的 system、tools 和之前的全部消息。改旧消息、删旧工具结果、
    本地摘要替换旧轮, 都会让其后的 thinking 块在真实 API 上失效 (stale_thinking 给出离线判定)。
  - tool_use.id 用 API 给的; 参数是已解析的 dict, 不要对序列化后的字符串做匹配。
依赖: `pip install anthropic` + 环境变量 ANTHROPIC_API_KEY。模型默认 claude-opus-5-5
(另有 claude-fable-5-2 / claude-sonnet-5-5 / claude-haiku-5-5 / claude-haiku-4-5, 用 LLM_AGENT_MODEL 覆盖),
effort 用 LLM_AGENT_EFFORT 覆盖, 不设就用模型默认值。
简化: 未启用服务端 refusal fallbacks (需 beta 端点: betas=["server-side-fallback-2026-07-01"], fallbacks="default"),
这里只识别 stop_reason == "refusal", 返回一句固定文本。
简化: stale_thinking 按整条消息比较 (真实签名按 block 位置绑定), 也不模拟 drop_block、服务端 compaction 之后前缀重新起算。
简化: 未开服务端 compaction、context editing、message threads (都要 beta 头), 工具定义也没加 strict: true。
"""

from __future__ import annotations

import os
import re
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


EFFORTS = ("low", "medium", "high", "xhigh", "max")
# 还用 budget_tokens 的老模型: claude-3*, 以及 4.0–4.5 (含 claude-haiku-4-5); 4.6 起的 Opus / Sonnet 和 5.x 都是 adaptive
_LEGACY_THINKING = re.compile(r"claude-(3|(opus|sonnet|haiku)-4(-[0-5])?($|-\d{8}))")


def build_request(model: str, max_tokens: int, messages: List[Message],
                  tools: List[Dict[str, Any]], effort: str = "") -> Dict[str, Any]:
    """内部 transcript → messages.create 的参数。纯函数, demo 离线断言它。"""
    system, api_messages = to_api_messages(messages)
    if not api_messages or api_messages[-1]["role"] != "user":
        # 末尾的 assistant 消息就是 prefill, 4.6 起的模型直接 400
        raise ValueError("请求必须以 user 消息结尾, 不支持 assistant 预填")
    kwargs: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": api_messages}
    if not _LEGACY_THINKING.match(model):
        kwargs["thinking"] = {"type": "adaptive"}  # Opus 5.5 上 disabled 和 budget_tokens 都是 400
    elif max_tokens >= 2048:  # budget_tokens 至少 1024, 且必须小于 max_tokens
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": max_tokens // 2}
    if effort:
        if effort not in EFFORTS:
            raise ValueError(f"effort 只能是 {EFFORTS}, 收到 {effort!r}")
        kwargs["output_config"] = {"effort": effort}  # 在 output_config 里, 不在顶层
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools  # Tool.schema() 已经是 {name, description, input_schema}; tool_choice 不发, 即 auto
    return kwargs


def stale_thinking(sent: Dict[str, Any], now: Dict[str, Any]) -> Tuple[int, List[Tuple[int, int]]]:
    """history-editing check 的离线近似: 这一次请求里, 哪些 thinking 块会被真实 API 判失效。

    sent 是上一次的请求参数, messages 末尾已追加它的回复; now 是这一次的请求参数。
    返回 (原样保留的前缀消息数, now 里失效的 thinking 块 [(消息下标, block 下标)])。
    thinking 块的签名绑定 system、tools 集合和它之前的全部消息; 签名还链着前一个 thinking 块,
    所以从第一条改动的消息起, 之后的 thinking 块全部失效。prompt cache 也只能读到同一个前缀为止。
    """
    def tool_set(req):  # tools 按名字排序后比较: 只换顺序不算改动
        return sorted(req.get("tools") or [], key=lambda t: t["name"])
    old, new = sent["messages"], now["messages"]
    keep = 0
    if sent.get("system") == now.get("system") and tool_set(sent) == tool_set(now):
        while keep < min(len(old), len(new)) and old[keep] == new[keep]:
            keep += 1
    stale = [(i, j) for i, m in enumerate(new[keep:], keep) if m["role"] == "assistant"
             for j, b in enumerate(m["content"]) if b["type"] in ("thinking", "redacted_thinking")]
    return keep, stale


class ClaudeLLM:
    """真实模型适配器。无注入客户端时检查 SDK/key；注入客户端供离线协议验证。"""

    def __init__(self, model: str = "", max_tokens: int = 16000, *, client=None, effort: str = "",
                 streaming=True, max_attempts=2, sleep=time.sleep, tracer=None, on_text=None) -> None:
        if max_attempts < 1 or max_tokens < 1:
            raise ValueError('预算必须为正')
        self.effort = effort or os.environ.get("LLM_AGENT_EFFORT", "")
        if self.effort and self.effort not in EFFORTS:  # 构造时就拒绝, 不等第一次请求
            raise ValueError(f"effort 只能是 {EFFORTS}, 收到 {self.effort!r}")
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
        self.model = model or os.environ.get("LLM_AGENT_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens
        self._tools: List[Dict[str, Any]] = []  # 上一次请求带的工具定义, 摘要请求时沿用

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        """发一次逻辑请求；默认流式累积，完整闭合后才交给 Agent 执行。真实客户端会计费。"""
        # 历史里有 tool_use/tool_result 时 API 要求必须带 tools; 摘要请求传的是 [] → 沿用上一次的定义。
        # 沿用还有一层作用: tools 中途变了, 之前的 thinking 块全部失效
        self._tools = tools or self._tools
        kwargs = build_request(self.model, self.max_tokens, messages, self._tools, self.effort)
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

        # 先看 stop_reason, 再读 content。refusal 是 HTTP 200, 不看它会把半截回答当正常结果
        if stop_reason not in ('end_turn','tool_use','stop_sequence','refusal','max_tokens'):
            raise StreamError(f'未支持的停止原因: {stop_reason}')  # pause_turn 等未实现
        if stop_reason == "refusal":
            return ModelAction.final("[模型拒绝了该请求]")
        text = "\n".join(b["text"] for b in raw if b["type"] == "text")
        calls = [ToolCall(b["name"], b["input"], id=b["id"]) for b in raw if b["type"] == "tool_use"]
        if stop_reason == "max_tokens":  # 可能截在 tool_use 中间: 参数不完整, 绝不执行
            return ModelAction.final(text + "\n[输出在 max_tokens 处被截断]")
        if calls and stop_reason != 'tool_use':
            raise StreamError('工具块与停止原因不一致')
        if calls and tools:
            return ModelAction(kind="tool", content=text, tool_calls=calls, raw_content=raw)
        # 没给工具 (如摘要请求) 模型却发了 tool_use: 不执行, 只取文本, 也不存 raw。
        # raw 里的 tool_use 没有对应的 tool_result, 写进 transcript 会让配对不合法
        return ModelAction(kind="final", content=text, raw_content=None if calls else raw)
