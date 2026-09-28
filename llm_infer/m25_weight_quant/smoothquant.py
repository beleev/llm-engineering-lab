"""
smoothquant.py — W8A8: 权重和激活都压到 INT8, 用 SmoothQuant 把激活的离群值"挪"给权重

是什么: weight-only (m08 / GPTQ) 只省显存带宽, matmul 仍是浮点; W8A8 让 matmul 直接走 INT8 tensor core,
    prefill (compute-bound) 也能加速。难点在激活: LLM 激活里少数固定通道比其它大 10~100 倍,
    per-token 量化时每一行都含这些离群值 → scale 被撑大, 普通通道只剩几个格点。
    权重正好相反: 分布平, 很好量化。
做法 (Xiao et al. 2022): 逐输入通道 j 选 s_j, 激活除以 s_j、权重行乘以 s_j, 数学上 XW 不变:
        X W = (X / s) · (s ⊙ W),     s_j = max|X_j|^α / max|W_j|^(1−α)
    α 控制"难度"迁移多少: α=0 什么都不挪给权重 (s 只把权重行归一), α=1 激活每个通道被压平到 1、
    离群值全压到权重上。中间某个 α 两边都好量化。
    1/s 离线折进上一层 (LayerNorm gamma / 前一个 Linear), 推理零开销 —— 与 AWQ 同一个恒等式, 目的不同:
    AWQ 用 s 保护重要权重 (weight-only), SmoothQuant 用 s 压平激活 (W8A8)。
量化器: 对称 INT8; 激活 per-token 动态 scale, 权重 per-output-channel scale (两者都能把 scale 提到 matmul 外面)。
读代码盯住: `smooth_scales` 与 `w8a8_matmul` 里 fake_quant 的两个 axis。
对应真实系统: vLLM / TensorRT-LLM 的 SmoothQuant W8A8; llm-compressor 的 `SmoothQuantModifier` (默认 α=0.5 左右)。
"""
from __future__ import annotations
import numpy as np


def fake_quant_sym(x: np.ndarray, bits: int, axis) -> np.ndarray:
    """对称量化再反量化: 在 axis 上统计 max|x| (keepdims), scale = max|x| / (2^(b-1)−1)。"""
    qmax = 2 ** (bits - 1) - 1                                            # 8 bit → 127
    scale = np.maximum(np.max(np.abs(x), axis=axis, keepdims=True), 1e-8) / qmax   # 1e-8: 整组为 0 时防除零
    return np.clip(np.round(x / scale), -qmax, qmax) * scale


def w8a8_matmul(X: np.ndarray, W: np.ndarray, quant_x: bool = True, quant_w: bool = True) -> np.ndarray:
    """X (N,D_in) per-token, W (D_in,D_out) per-output-channel。两个开关用来把误差拆成"激活贡献 / 权重贡献"。"""
    Xq = fake_quant_sym(X, 8, axis=1) if quant_x else X                   # scale (N,1)
    Wq = fake_quant_sym(W, 8, axis=0) if quant_w else W                   # scale (1,D_out)
    return Xq @ Wq


def smooth_scales(X_calib: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    """s (D_in,) = max|X_j|^α / max|W_j|^(1−α); 激活统计来自校准集, 部署时是常数。"""
    ax = np.max(np.abs(X_calib), axis=0)                                  # (D_in,) 每个输入通道的激活幅度
    aw = np.max(np.abs(W), axis=1)                                        # (D_in,) 对应权重行的幅度
    return (ax ** alpha / np.maximum(aw, 1e-8) ** (1 - alpha)).astype(np.float32)   # 1e-8: 权重行全 0 时防除零
