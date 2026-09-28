"""
flash_decoding.py — Flash-Decoding: decode 阶段沿 KV 长度切 split-K, 各段并行算局部 softmax, 再按 lse 合并

是什么: FlashAttention (m11) 的并行单位是 (batch, head, Q 块)。decode 时每条序列只有 1 个 query,
    Q 块只有 1 个 → 并行单位 = B×H。B=1、H=32 时只有 32 个 thread block, 一张 108 SM 的卡 70% 在闲着;
    每个 block 还要独自把整条 KV (可能几十万 token) 从头读到尾。
做法 (Dao et al. 2023, "Flash-Decoding"): 把 KV 沿长度切成 S 段, 每段一个 thread block 算局部 (O_s, lse_s),
    并行单位变成 B×H×S; 第二个小 kernel 按 lse 加权把 S 段合并 (与 m11.merge_attention 同一个公式, 一次合 S 段):
        lse = log Σ_s exp(lse_s)        O = Σ_s exp(lse_s − lse) · O_s
    数学上与普通 attention 严格相等, 只多了一次 reduce。
读代码盯住: `flash_decoding` 里 splits 循环 (各段互不依赖 = GPU 上并行) 与最后两行 reduce。
对应真实系统: flash-attn 的 `num_splits` (flash_attn_with_kvcache), FlashInfer 的 split-KV decode, vLLM PagedAttention V2
    (按 512 token 分 partition 再 reduce), SGLang triton decode kernel 的 `num_kv_splits`。
"""
from __future__ import annotations
import numpy as np


def partial_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """一段 KV 上的局部 attention。q (...,1,d), K/V (...,Ts,d) → O_s (...,1,d) 已归一化, lse_s (...,1,1)。"""
    S = q @ np.swapaxes(K, -1, -2) / np.sqrt(q.shape[-1])                # (...,1,Ts)
    m = S.max(-1, keepdims=True)
    p = np.exp(S - m)
    l = p.sum(-1, keepdims=True)
    return (p @ V) / l, m + np.log(l)


def flash_decoding(q: np.ndarray, K: np.ndarray, V: np.ndarray, n_splits: int):
    """q (...,1,d), K/V (...,T,d) → (O (...,1,d), lse (...,1,1))。T 不必整除 n_splits (前几段多 1 个 token)。"""
    bounds = np.linspace(0, K.shape[-2], n_splits + 1).round().astype(int)
    parts = [partial_attention(q, K[..., a:b, :], V[..., a:b, :])          # 各段互不依赖: GPU 上是 S 个并行的 thread block
             for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    O_s = np.stack([o for o, _ in parts])                                 # (S,...,1,d)
    lse_s = np.stack([l for _, l in parts])                               # (S,...,1,1)
    lse = np.logaddexp.reduce(lse_s, axis=0)                              # reduce kernel: 全局分母的 log
    return np.sum(np.exp(lse_s - lse) * O_s, axis=0), lse                 # 每段按它占的 softmax 质量加权


# ---------------------------------------------------------------------- #
# 延迟估算模型 (不是实测): decode attention 是纯读 KV 的 memory-bound 操作      #
# ---------------------------------------------------------------------- #
N_SM = 108                    # A100
HBM_GBPS = 2000.0             # 全卡带宽; 假设每个 SM 最多拿到 1/N_SM (只有所有 SM 都在读时才跑满)
KV_BYTES_PER_TOKEN = 512      # 一个 head: K+V 各 d=128 个 fp16
LAUNCH_US = 3.0               # 每个 kernel 的固定开销 (launch + 尾部同步)


def decode_latency_us(B: int, H: int, T: int, S: int) -> tuple[float, float]:
    """→ (一层 decode attention 的估算延迟 μs, SM 利用率)。

    并行单位 B×H×S 个 block, 每个读 T/S 个 token 的 KV; 按 N_SM 一波一波地跑 (wave)。
    S>1 时多一个 reduce kernel: 读回 S 份 (O_s fp32 d 维 + lse)。
    """
    units = B * H * S
    waves = -(-units // N_SM)
    per_sm_bytes_per_us = HBM_GBPS * 1e3 / N_SM                           # GB/s → bytes/μs, 再均分给每个 SM
    t = LAUNCH_US + waves * (T / S) * KV_BYTES_PER_TOKEN / per_sm_bytes_per_us
    if S > 1:
        t += LAUNCH_US + units * (128 + 1) * 4 / (HBM_GBPS * 1e3)
    return t, units / (waves * N_SM)
