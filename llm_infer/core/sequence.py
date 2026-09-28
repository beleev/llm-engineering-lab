"""
sequence.py — 单个推理请求的状态 (调度器 / 引擎 / 各 demo 共用)

盯住 num_computed: "KV 已经在 cache 里的 token 数"。vLLM V1 的统一视角 ——
    每步给序列 n 个 token 的算力, 处理 all_ids[num_computed : num_computed+n];
    n>1 是 prefill (或其中一个 chunk), n=1 是 decode; 追平 num_tokens 才采样新 token。
    prefix cache 命中 = num_computed 直接从 hit_len 起步; preempt = num_computed 归 0。

        WAITING ──schedule──→ RUNNING ──finish──→ FINISHED
           ↑                     │
           └──── preempt ────────┘   (block 全还, output_ids 保留, 回来后重算 KV)

对应真实系统: vLLM `Request.num_computed_tokens`, SGLang `Req`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SeqStatus(Enum):
    """序列在调度器里的三种状态, 转移关系见文件头的图。"""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Stage(Enum):
    """序列此刻在 prefill 还是 decode。

    Stage 和下面的 seq.stage 目前没有调用方: 调度器直接比较 num_computed 与 num_tokens。
    """
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Sequence:
    """一条推理请求: 输入的 prompt, 已生成的输出, 以及 KV 算到了哪里。"""
    seq_id: int
    prompt_ids: List[int]
    max_new_tokens: int            # 最多生成几个 token, 到数即结束
    eos_id: int = 2                # 与 CharTokenizer.EOS_ID 一致

    output_ids: List[int] = field(default_factory=list)   # 已生成的 token, 抢占后原样保留
    status: SeqStatus = SeqStatus.WAITING
    num_computed: int = 0          # KV 已就绪的 token 数 (含 prefix cache 命中的)
    num_preempted: int = 0         # 被抢占的次数 (只做统计)

    @property
    def all_ids(self) -> List[int]:
        """prompt + 已生成输出; 模型每步处理的是它的 [num_computed:] 这一段。"""
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_output(self) -> int:
        return len(self.output_ids)

    @property
    def stage(self) -> Stage:
        # 还差不止 1 个 token 的 KV → 在 prefill (含被抢占后的重算); 本属性目前没有调用方
        return Stage.PREFILL if self.num_tokens - self.num_computed > 1 else Stage.DECODE

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def is_finished(self) -> bool:
        """生成够 max_new_tokens 个, 或最后一个输出是 EOS。"""
        # 只数 output_ids: 抢占不会把输出折进 prompt, 所以 max_new_tokens 不会被重置
        return (self.num_output >= self.max_new_tokens
                or (bool(self.output_ids) and self.output_ids[-1] == self.eos_id))

    def reset_for_recompute(self) -> None:
        """preempt: KV 全丢 (num_computed=0), 已生成的 token 原样保留。"""
        self.num_computed = 0
        self.num_preempted += 1
        self.status = SeqStatus.WAITING
