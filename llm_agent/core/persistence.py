"""append-only JSONL 会话日志。

没有它: 进程一退, 会话全丢; 出了事故也无从复盘"模型当时到底调了什么"。
关键设计:
  - 只追加不改写: 崩溃最多坏最后一行 (load 时跳过); 无 fsync / 文件锁, 这是教学版的上限。
  - 压缩不删历史, 而是追加一条 compact_boundary 记录;
    load() (给 resume 用) 只返回"最后一个 boundary 的摘要 + 保留的尾部 + 之后的消息",
    load_all() (给审计用) 返回全部。于是: 文件只增不减, 恢复出来的上下文却真的变小了。
对应: Claude Code ~/.claude/projects/*.jsonl 会话文件与其中的 compact boundary。
简化: 行格式是本库自己定的 (Message.to_dict 加一种边界行), 只保证本库自己读得回来。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from llm_agent.core.schema import Message


class JsonlSessionStore:
    """一个会话一个 .jsonl 文件。文件里有两种行:

      消息行    {"role", "content", "name"}, 就是 Message.to_dict()
      边界行    {"type": "compact_boundary", "summary": 摘要消息, "kept": 保留的尾部条数}
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        """追加一行。每次都重新打开文件, 写完即关: Python 这一层不留缓冲。没有 fsync, 断电仍可能丢。"""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, message: Message) -> None:
        """追加一条消息。"""
        self._write(message.to_dict())

    def append_compact(self, summary: Message, kept: int) -> None:
        """kept = boundary 之前有多少条尾部消息原样保留 (当前这一轮)。"""
        self._write({"type": "compact_boundary", "summary": summary.to_dict(), "kept": kept})

    def _records(self) -> Iterator[dict]:
        """逐行读出全部记录 (消息行和边界行都有), 跳过空行和坏行。"""
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        if line.strip():
                            yield json.loads(line)
                    except json.JSONDecodeError:  # 崩溃时写了一半的行: 跳过, 不让整个会话无法恢复
                        continue

    def load_all(self) -> List[Message]:
        """给审计用: 文件里的全部消息, 压缩前的旧历史也在。摘要消息不在其中 (它只存在边界行里)。"""
        return [Message.from_dict(r) for r in self._records() if r.get("type") != "compact_boundary"]

    def load(self) -> List[Message]:
        """给 resume 用: 重放文件, 得到压缩之后的上下文。

        结果 = 最后一个边界的摘要 + 它保留的尾部 + 边界之后的消息。没有边界时等于 load_all()。
        """
        view: List[Message] = []
        for r in self._records():
            if r.get("type") == "compact_boundary":
                # 遇到边界: 到目前为止攒下的消息只留最后 kept 条, 前面的换成摘要。
                # kept=0 要单独判断, 因为 view[-0:] 是整个列表
                tail = view[-r["kept"] :] if r["kept"] else []
                view = [Message.from_dict(r["summary"])] + tail
            else:
                view.append(Message.from_dict(r))
        return view
