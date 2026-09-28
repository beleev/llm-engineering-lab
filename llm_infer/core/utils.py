"""
utils.py — 共享小工具

数值算子是纯函数; Timer / banner / kv 只给 demo 计时和打印用。所有模块通用。
"""
from __future__ import annotations
import time
import numpy as np


# --------------------------------------------------------------------- #
# 数值算子                                                              #
# --------------------------------------------------------------------- #

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax: 减最大值再 exp, 防止溢出。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * gamma

    现代 LLM (LLaMA / Qwen) 都用 RMSNorm 替代 LayerNorm:
    - 少一次均值减法, 算得快
    - 实验表明效果几乎无差

    x (..., D), gamma (D,) 或标量 → (..., D)。eps: x 全 0 时防除零。
    """
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish: x * sigmoid(x), LLaMA SwiGLU 的激活函数。"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def causal_mask(t_q: int, t_k: int) -> np.ndarray:
    """生成因果 mask: 形状 (t_q, t_k), 上三角为 -inf, 其余为 0。

    支持 t_q != t_k (decode 时 t_q=1, t_k=context_len)。
    约定: query 位置 i 只能看到 key 位置 j 满足 j <= (t_k - t_q + i)。
    """
    mask = np.zeros((t_q, t_k), dtype=np.float32)
    for i in range(t_q):
        last_visible = t_k - t_q + i          # 第 i 个 query 自己在 K 里的下标, 它后面的都屏蔽
        if last_visible + 1 < t_k:
            mask[i, last_visible + 1:] = -np.inf
    return mask


def dense_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """朴素 attention 基线: softmax(q·Kᵀ/√d + mask)·V, 全库唯一一份。

    q (..., Tq, d), K (..., Tk, d), V (..., Tk, dv) → (..., Tq, dv); 前导维 (如 n_head) 自动广播。
    mask: 加性 (Tq, Tk), 0=可见 / -inf=屏蔽; None → 因果 (query 对齐 K 的尾部)。
    各模块 (paged / flash / TP / 量化 / 树形 mask) 都拿它当"正确答案"对拍。
    """
    if mask is None:
        mask = causal_mask(q.shape[-2], K.shape[-2])
    # 除数 d ** 0.5 是 python float, fp32 输入算出来还是 fp32。
    # 换成 np.sqrt(d) 会得到 np.float64 标量, NumPy 2 下结果被升成 fp64。
    # mask 传 fp64 数组时结果同样是 fp64, 调用方要 fp32 就自己 astype。
    scores = q @ np.swapaxes(K, -1, -2) / q.shape[-1] ** 0.5     # (..., Tq, Tk)
    return softmax(scores + mask, axis=-1) @ V


# --------------------------------------------------------------------- #
# 计时器                                                                #
# --------------------------------------------------------------------- #

class Timer:
    """极简上下文计时器, 用于 demo 里打印对比。

    用法:
        with Timer("some op") as t:
            do_stuff()
        print(t.ms)        # 经过的毫秒数
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self._t0) * 1000.0


# --------------------------------------------------------------------- #
# 打印辅助                                                              #
# --------------------------------------------------------------------- #

def banner(title: str, width: int = 70) -> None:
    """打印一条带标题的分隔线, demo 输出更易读。"""
    pad = (width - len(title) - 2) // 2
    line = "═" * width
    print(f"\n{line}\n{' ' * pad} {title} {' ' * pad}\n{line}")


def kv(key: str, value, indent: int = 2) -> None:
    """key-value 一行打印。"""
    print(f"{' ' * indent}{key:<32} = {value}")
