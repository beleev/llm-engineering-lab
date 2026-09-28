"""
pd.py — Prefill/Decode 分离: P 节点只跑 prefill, 把 KV cache 序列化发给 D 节点, D 节点只跑 decode。

瓶颈: prefill 是 compute-bound (一次几千 token), decode 是 memory-bound (一次 1 token)。同卡混跑时,
      一个长 prefill 会让同 batch 所有 decode 请求的这一步 ITL 从 ~decode 耗时 涨到 ~prefill 耗时 (延迟抖动)。
代价: KV 要过网络。字节数 = 2·n_layer·T·D·sizeof(dtype); LLaMA-7B fp16 4k 上下文 = 2.15 GB。
单位: 网络带宽按 **Gbps (bit)** 标, 显存/KV 按 **byte** 算 → bytes/s = Gbps·1e9/8。混淆这个 8 会把传输时间低估 8 倍。
读代码盯住: `KVLink.transfer_ms` 的单位换算; `serialize_kv` 之后 D 节点手里只有 bytes, 没有 P 节点的任何对象引用。
真实系统: DistServe / Splitwise / Mooncake (KV 走 RDMA, 按层流式传); vLLM `KVConnector` (NIXL / LMCache); SGLang PD disaggregation。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from llm_infer.core import TinyLM, ModelConfig

KVCache = List[Tuple[np.ndarray, np.ndarray]]               # 每层一个 (K (T,D), V (T,D))


def kv_nbytes_formula(n_layer: int, n_tokens: int, d_model: int, itemsize: int) -> int:
    """KV cache 的字节数。itemsize: 每个元素几个字节 (fp16 = 2, fp32 = 4)。"""
    return 2 * n_layer * n_tokens * d_model * itemsize       # 2 = K 和 V


def serialize_kv(kv_cache: KVCache) -> Tuple[tuple, str, bytes]:
    """→ (shape, dtype, payload)。payload 是纯 bytes: 过了这一步就和 P 节点的内存没有关系了。"""
    arr = np.stack([np.stack(layer) for layer in kv_cache])  # (n_layer, 2, T, D)
    return arr.shape, arr.dtype.str, arr.tobytes()


def deserialize_kv(shape: tuple, dtype: str, payload: bytes) -> KVCache:
    """serialize_kv 的逆操作: bytes → 每层一个 (K, V)。"""
    arr = np.frombuffer(payload, dtype=dtype).reshape(shape)  # (n_layer, 2, T, D), 只读视图; decode 只会 concat 出新数组, 不会写它
    return [(layer[0], layer[1]) for layer in arr]


@dataclass
class KVLink:
    """P→D 链路的**代价模型** (不 sleep, 不是实测): time = latency + bytes / (Gbps·1e9/8)。"""
    link_gbps: float                 # 标称带宽, 单位 Gbit/s (网卡、IB 都这么标)
    latency_ms: float = 0.0          # 每次传输的固定延迟 (建连、握手), 与字节数无关
    bytes_sent: int = 0              # 累计发出的字节数

    @property
    def bytes_per_s(self) -> float:
        return self.link_gbps * 1e9 / 8                      # bit → byte

    def transfer_ms(self, nbytes: int) -> float:
        """传 nbytes 个字节要多少毫秒 (代价模型算出来的)。"""
        return self.latency_ms + nbytes / self.bytes_per_s * 1e3   # s → ms

    def send(self, payload: bytes) -> Tuple[bytes, float]:
        """→ (原样的 payload, 模型算出的传输耗时 ms)。不真的等待。"""
        self.bytes_sent += len(payload)
        return payload, self.transfer_ms(len(payload))


class PrefillNode:
    """自己加载一份权重; 只做 prefill, 产出首 token (决定 TTFT) + 序列化后的 KV。"""

    def __init__(self, cfg: ModelConfig):
        self.lm = TinyLM(cfg)

    def run(self, prompt_ids: np.ndarray):
        """prompt (T,) → (首 token id, (shape, dtype, payload))。"""
        logits, kv_cache = self.lm.prefill(prompt_ids)       # logits (T, V); kv: n_layer × (K (T,D), V (T,D))
        return int(np.argmax(logits[-1])), serialize_kv(kv_cache)


class DecodeNode:
    """自己加载一份权重 (同 seed → 与 P 节点数值相同); 只做 decode, 从收到的 KV 接着生成。"""

    def __init__(self, cfg: ModelConfig):
        self.lm = TinyLM(cfg)

    def run(self, first_token: int, wire_kv, max_new: int) -> List[int]:
        """wire_kv: (shape, dtype, payload)。返回 max_new 个 token, 第 1 个就是 first_token。"""
        kv_cache = deserialize_kv(*wire_kv)
        out = [first_token]
        for _ in range(max_new - 1):                         # 首 token 已由 P 节点给出
            logits, kv_cache = self.lm.decode_step(out[-1], kv_cache)   # (V,)
            out.append(int(np.argmax(logits)))
        return out
