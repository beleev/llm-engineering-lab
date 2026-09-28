"""
fp8.py — FP8 (E4M3 / E5M2) 的舍入, 以及 per-tensor / per-channel scale

是什么: 8 bit 浮点。INT8 的格点等间距; FP8 的格点按 2 的幂分段, 每段 2^m 个 → **相对**误差大致恒定
    (E4M3 ≤ 2^-4 = 6.25%, E5M2 ≤ 2^-3 = 12.5%), 代价是大值附近的格子很粗。
        E4M3 (fn): 4 bit 指数 / 3 bit 尾数, bias 7,  最大 448,   最小正规数 2^-6,  最小非正规数 2^-9
        E5M2     : 5 bit 指数 / 2 bit 尾数, bias 15, 最大 57344, 最小正规数 2^-14, 最小非正规数 2^-16
    推理用 E4M3 (精度优先); E5M2 范围大, 主要给训练的梯度用。
为什么还要 scale: FP8 的正规数只覆盖约 2^15 (E4M3) 的动态范围。权重 std ~0.02 直接转 E4M3 会大量落进
    非正规区 (格距固定 2^-9, 相对误差暴涨); 乘一个 scale 把 amax 对齐到 448 就好了。
    scale 的粒度 (per-tensor / per-channel) 对 FP8 的影响远小于对 INT8: 浮点格点本身随数值大小伸缩,
    只有落出正规区的小数才吃亏。
读代码盯住: `fp8_round` 里 `step = 2^(e − m)` —— 每个数的格距由它自己的指数决定。
对应真实系统: H100 / MI300 的 FP8 tensor core; vLLM `quantization="fp8"` (权重 per-channel 或 per-tensor, 激活动态 per-tensor),
    `--kv-cache-dtype fp8_e4m3`; DeepSeek-V3 的 block-wise (128×128) FP8 scale。
"""
from __future__ import annotations
import numpy as np

FORMATS = {"e4m3": (4, 3, 448.0), "e5m2": (5, 2, 57344.0)}               # (指数 bit, 尾数 bit, 最大有限值)


def fp8_round(x: np.ndarray, fmt: str) -> np.ndarray:
    """把 fp32 舍入到最近的 FP8 可表示值 (饱和到 ±max, 不产生 inf)。结果仍用 fp32 存, 只验证数值。"""
    e_bits, m_bits, fmax = FORMATS[fmt]
    e_min = 2 - 2 ** (e_bits - 1)                                         # 最小正规数的指数 = 1 − bias
    with np.errstate(divide="ignore"):                                    # x=0 时 log2 得 -inf, 被下面的 maximum 钳住
        e = np.maximum(np.floor(np.log2(np.abs(x))), e_min)               # 非正规区: 指数钉在 e_min, 格距不再缩小
    step = 2.0 ** (e - m_bits)                                            # 本段格距: 每段 2^m 个格点
    return np.clip(np.round(x / step) * step, -fmax, fmax).astype(np.float32)


def fake_quant_fp8(x: np.ndarray, fmt: str, axis=None) -> np.ndarray:
    """scale = amax / fmax (在 axis 上统计; None → per-tensor), 量化 x/scale 再乘回。"""
    # 1e-12: 整组为 0 时防除零
    scale = np.maximum(np.max(np.abs(x), axis=axis, keepdims=True), 1e-12) / FORMATS[fmt][2]
    return fp8_round(x / scale, fmt) * scale
