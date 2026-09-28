"""共享数据结构: transcript 用 Claude Messages API 同款 content block。

没有它: 助手 "我要调用哪个工具" 这一步不进 transcript, JSONL 只剩孤零零的工具结果,
既没法审计, 也没法原样喂给真实 API (tool_result 必须紧跟同 id 的 tool_use)。
关键设计: content 要么是 str, 要么是 block 列表:
    assistant: {"type":"tool_use","id","name","input"}
    user     : {"type":"tool_result","tool_use_id","content","is_error"}
一个 assistant turn 可带多个 tool_use (并行工具调用), 其结果必须放进同一条 user 消息。
对应: Anthropic Messages API 的 tool use 消息格式。
差异: 这里多了 system 角色的消息和 name 字段, 真实 API 的 messages 里没有这两样。
  发给真实 API 之前要经 claude_llm.to_api_messages 转换。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

Block = Dict[str, Any]
Content = Union[str, List[Block]]


@dataclass
class Message:
    """transcript 里的一条消息。JSONL 的一行就是一个 Message。

    role     谁说的
    content  纯文本, 或 block 列表 (text / tool_use / tool_result)
    name     harness 注入的 system 消息从哪来; 用户和模型的消息是 None
    """

    role: str  # system / user / assistant
    content: Content
    name: Optional[str] = None  # 仅标记 harness 注入的 system 消息来源 (memory / hook_context ...)

    @property
    def blocks(self) -> List[Block]:
        """统一成 block 列表: 纯文本包成一个 text block, 调用方不用再分两种情况。"""
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        return self.content

    @property
    def text(self) -> str:
        """拍平成纯文本, 供打印 / 计长度 / toy LLM 使用。"""
        parts = []
        for b in self.blocks:
            if b["type"] == "text":
                parts.append(b["text"])
            elif b["type"] == "tool_use":
                parts.append(f"[tool_use {b['name']} {json.dumps(b['input'], ensure_ascii=False)}]")
            elif b["type"] == "tool_result":
                parts.append(str(b["content"]))
        return "\n".join(parts)

    def tool_uses(self) -> List[Block]:
        """这条消息里的全部 tool_use block (模型发起的工具调用)。"""
        return [b for b in self.blocks if b["type"] == "tool_use"]

    def tool_results(self) -> List[Block]:
        """这条消息里的全部 tool_result block (回填给模型的工具结果)。"""
        return [b for b in self.blocks if b["type"] == "tool_result"]

    @property
    def is_user_prompt(self) -> bool:
        """真正的用户输入; 携带 tool_result 的 user 消息不算。"""
        return self.role == "user" and not self.tool_results()

    def to_dict(self) -> Dict[str, Any]:
        """转成可以 json.dumps 的 dict。JSONL 的一行就是它。"""
        return {"role": self.role, "content": self.content, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """to_dict 的逆操作, 读 JSONL 时用。"""
        return cls(role=data["role"], content=data["content"], name=data.get("name"))


@dataclass
class ToolCall:
    """模型发起的一次工具调用。hook 可以把它换成另一个 ToolCall。"""

    name: str  # 工具名, 要和 ToolRegistry 里注册的一致
    args: Dict[str, Any] = field(default_factory=dict)  # 已解析的参数 dict, 执行前按工具的 schema 校验
    reason: str = ""  # 模型自述的调用理由, 只给人看。不发给工具, 也不进 transcript
    id: str = ""  # 真实模型自带 id; toy LLM 留空, 由 agent 分配

    def to_block(self) -> Block:
        """转成 assistant 消息里的 tool_use block。"""
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.args}


@dataclass
class ToolResult:
    """一次工具调用的结果。被拒绝、参数不合法、工具抛异常, 都会变成 ok=False 的结果。"""

    name: str  # 实际执行的工具。hook 改写后可能不是模型要的那个; 被拒绝时是模型要的那个
    output: str  # 回填给模型的文本。失败时是拒绝原因或报错信息
    ok: bool = True  # False 对应 tool_result 的 is_error=True
    tool_use_id: str = ""  # 对应哪个 tool_use。由 agent 填, 永远是模型发出的那个 id

    def to_block(self) -> Block:
        """转成 user 消息里的 tool_result block。name 不在 block 里, 靠 tool_use_id 配对。"""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.output,
            "is_error": not self.ok,  # 失败也要回填, 不能丢: 模型靠它自我纠正
        }


@dataclass
class ModelAction:
    """模型一次回复的结果: 要么要调工具, 要么给出最终回答。LLM.next() 返回它。"""

    kind: str  # "tool" | "final"
    content: str = ""  # 模型说的文本。kind=final 时是最终回答, kind=tool 时是调用前的旁白
    tool_calls: List[ToolCall] = field(default_factory=list)  # 一次回复可带多个调用 (并行)
    raw_content: Optional[List[Block]] = None  # 真实 API 的原始 blocks (含 thinking), 必须原样回传

    @property
    def tool_call(self) -> Optional[ToolCall]:
        """第一个工具调用; 没有则 None。"""
        return self.tool_calls[0] if self.tool_calls else None

    @classmethod
    def tool(cls, *calls: ToolCall, text: str = "") -> "ModelAction":
        """造一个"要调工具"的动作。传几个 ToolCall, 就是一个 turn 里并行发几个。"""
        return cls(kind="tool", content=text, tool_calls=list(calls))

    @classmethod
    def final(cls, text: str) -> "ModelAction":
        """造一个"给出最终回答"的动作。"""
        return cls(kind="final", content=text)


def validate_transcript(messages: List[Message]) -> List[str]:
    """只检查 tool_use / tool_result 的 id 配对, 返回问题列表。

    空列表只说明配对没问题, 还不能直接发给真实 API:
    system 消息和 name 字段要先经 claude_llm.to_api_messages 转换。
    """
    problems = []
    for i, msg in enumerate(messages):
        # 正向: assistant 的每个 tool_use, 下一条 user 消息里要有同 id 的 tool_result (不比顺序)
        uses = [b["id"] for b in msg.tool_uses()] if msg.role == "assistant" else []
        if uses:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            got = [b["tool_use_id"] for b in nxt.tool_results()] if nxt and nxt.role == "user" else []
            if sorted(uses) != sorted(got):
                problems.append(f"msg[{i}] tool_use {uses} 没有在下一条 user 消息里配齐 tool_result {got}")
        # 反向: user 的每个 tool_result, 上一条 assistant 消息里要有同 id 的 tool_use
        if msg.role == "user" and msg.tool_results():
            prev = messages[i - 1] if i else None
            prev_ids = [b["id"] for b in prev.tool_uses()] if prev and prev.role == "assistant" else []
            for b in msg.tool_results():
                if b["tool_use_id"] not in prev_ids:
                    problems.append(f"msg[{i}] 孤儿 tool_result {b['tool_use_id']}")
    return problems
