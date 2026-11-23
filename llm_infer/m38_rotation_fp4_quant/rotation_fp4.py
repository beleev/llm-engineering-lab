"""
rotation_fp4.py — 旋转消除离群值 (QuaRot / SpinQuant / QuIP#) + 推理侧 FP4 格式 (MXFP4 / NVFP4)

是什么: W4A4 (权重和激活都压到 4 bit) 的难点在激活: 少数固定通道比其它大几十倍, 每个 token 的
    量化范围都被它们撑开, 4 bit 只有 15 个格点, 普通通道几乎全被舍成 0。
    SmoothQuant (m25) 把离群值挪给权重; 旋转把离群值"摊平"到所有通道:
        X W = (X Q) (Qᵀ W),   Q 正交 (Q Qᵀ = I)
    Q 取归一化 Hadamard 矩阵时, XQ 的每个通道都是原来所有通道的 ±1/√n 加权和, 一个大通道被均分给 n 个通道,
    每个 token 的 max/RMS 降到 3 左右 (纯高斯的水平)。Qᵀ 离线折进权重, 推理只在激活侧多一次 O(n log n) 的快速 Hadamard 变换。
FP4 格式 (每个元素都是 E2M1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, 区别在 scale):
    MXFP4 (OCP MX 规范): 每 32 个元素共享 1 个 E8M0 scale (纯 2 的幂)          → 4 + 8/32 = 4.25 bit/元素
    NVFP4 (NVIDIA)     : 每 16 个元素 1 个 FP8 E4M3 scale, 外加整张量 1 个 FP32 → 4 + 8/16 = 4.5 bit/元素
    INT4 分组对照       : 每 g 个元素 1 个 fp16 scale, 对称 −7..7                → 4 + 16/g bit/元素
关键数字: demo 里普通通道的 per-token INT4 误差 0.883 → 0.263; 分组越细旋转收益越小, NVFP4 (16 个一组) 旋转后反而略差。
简化: Hadamard 用显式矩阵乘, 真实系统用快速 Hadamard 变换 kernel; 只有一层, QuaRot 在整个网络的残差流、
      down_proj 之前、RoPE 之后都插旋转。
简化: 学习旋转只在一层上用 STE + Cayley 更新优化输出误差; SpinQuant 在整个网络上用 Cayley SGD 优化最终 loss。
简化: 全部是 fake quant (量化后立刻反量化回 fp32 计算), 只验数值, 不说明速度。
读代码盯住: `hadamard` 的递推、`mxfp4` / `nvfp4` 里 scale 那一行、`learn_rotation` 里的 Cayley 更新。
对应真实系统: QuaRot / SpinQuant 的离线旋转 + 在线 Hadamard; GPT-OSS 的 MoE 权重以 MXFP4 发布。
    训练侧的 FP4 见 llm_train/m15_fp4_microscaling (本文件不跨包 import)。
"""
from __future__ import annotations
import numpy as np

from llm_infer.m25_weight_quant.fp8 import fp8_round

E2M1_MAX = 6.0                                     # E2M1 的最大值 (1.5 × 2^2)
E4M3_MAX = 448.0                                   # FP8 E4M3 的最大值, NVFP4 的 block scale 用它
BITS = {                                           # 每元素等效位数 (scale 摊到每个元素上)
    "INT4 g32": 4 + 16 / 32, "INT4 g128": 4 + 16 / 128,
    "MXFP4": 4 + 8 / 32, "NVFP4": 4 + 8 / 16,      # NVFP4 的整张量 FP32 scale 摊下来可以忽略
}


def hadamard(n: int) -> np.ndarray:
    """归一化 Hadamard 矩阵 (n, n), n 是 2 的幂。Sylvester 递推 H_2n = [[H, H], [H, −H]], 除以 √n 后正交。"""
    assert n > 0 and n & (n - 1) == 0, f"Sylvester 构造要求 n 是 2 的幂, 实际 {n}"
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


def random_hadamard(n: int, rs: np.random.RandomState) -> np.ndarray:
    """随机化 Hadamard: diag(±1) · H。仍然正交; 随机符号让离群方向不会恰好和 H 的某一列对齐 (QuIP# / QuaRot 的做法)。"""
    return (rs.choice([-1.0, 1.0], n)[:, None] * hadamard(n)).astype(np.float32)


# --------------------------------------------------------------------- #
# 量化器: 全部 fake quant, 输入输出同形                                  #
# --------------------------------------------------------------------- #

def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    """沿最后一维每 block 个元素切一组 → (..., n/block, block)。"""
    assert x.shape[-1] % block == 0, f"最后一维 {x.shape[-1]} 必须是 block={block} 的整数倍"
    return x.reshape(*x.shape[:-1], x.shape[-1] // block, block)


def int4_sym(x: np.ndarray, group: int | None = None, axis: int = -1) -> np.ndarray:
    """对称 INT4 (−7..7)。group=None → 沿 axis 整行一组 (激活 per-token / 权重 per-channel); 否则沿最后一维每 group 个一组。"""
    if group is None:
        scale = np.maximum(np.abs(x).max(axis=axis, keepdims=True), 1e-12) / 7.0   # 1e-12: 整行为 0 时防除零
        return np.clip(np.round(x / scale), -7, 7) * scale
    b = _blocks(x, group)
    scale = np.maximum(np.abs(b).max(-1, keepdims=True), 1e-12) / 7.0
    return (np.clip(np.round(b / scale), -7, 7) * scale).reshape(x.shape)


def e2m1_round(x: np.ndarray) -> np.ndarray:
    """舍入到最近的 E2M1 值 (就近偶数, 超过 6 饱和)。格距: [0,2) 是 0.5, [2,4) 是 1, [4,6] 是 2。"""
    a = np.abs(x)
    step = np.where(a < 2, 0.5, np.where(a < 4, 1.0, 2.0))
    return np.sign(x) * np.minimum(np.round(a / step) * step, E2M1_MAX)


def mxfp4(x: np.ndarray, block: int = 32) -> np.ndarray:
    """MXFP4: 每 block 个元素一个 E8M0 scale = 2^(⌊log2 amax⌋ − 2), 2 是 E2M1 的最大指数。

    amax / scale 落在 [4, 8), 大于 6 的部分饱和到 6 —— 2 的幂 scale 对不准 amax, 这是 MXFP4 误差的主要来源。
    """
    b = _blocks(x, block)
    amax = np.maximum(np.abs(b).max(-1, keepdims=True), 1e-30)          # 1e-30: 整块为 0 时 log2 不出 -inf
    scale = 2.0 ** (np.floor(np.log2(amax)) - 2)                        # ★ E8M0: 只能是 2 的幂
    return (e2m1_round(b / scale) * scale).reshape(x.shape).astype(np.float32)


def nvfp4(x: np.ndarray, block: int = 16) -> np.ndarray:
    """NVFP4: 每 block 个元素一个 E4M3 scale ≈ amax/6, 外加整张量一个 FP32 scale 把 block scale 搬进 E4M3 的范围。"""
    b = _blocks(x, block)
    amax = np.abs(b).max(-1, keepdims=True)
    s_tensor = max(float(np.abs(x).max()), 1e-30) / (E4M3_MAX * E2M1_MAX)   # 整张量最大的 block scale 恰好对到 448
    scale = fp8_round(amax / E2M1_MAX / s_tensor, "e4m3") * s_tensor     # ★ block scale 本身只有 8 bit (E4M3)
    scale = np.where(scale == 0, 1.0, scale)                            # 整块为 0 时防除零
    return (e2m1_round(b / scale) * scale).reshape(x.shape).astype(np.float32)


FORMATS = {                                        # 名字 → 对最后一维量化的函数
    "INT4 g32": lambda x: int4_sym(x, 32), "INT4 g128": lambda x: int4_sym(x, 128),
    "MXFP4": mxfp4, "NVFP4": nvfp4,
}


def rel(A: np.ndarray, B: np.ndarray) -> float:
    """相对误差 ‖A − B‖ / ‖A‖, A 是基准。"""
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def w4a4_err(X: np.ndarray, W: np.ndarray, Q: np.ndarray | None = None) -> float:
    """W4A4 输出误差: 激活 per-token INT4、权重 per-output-channel INT4 (都是对称 RTN)。Q 给了就先旋转。"""
    Xr, Wr = (X, W) if Q is None else (X @ Q, Q.T @ W)                  # 数学上 (XQ)(QᵀW) = XW
    return rel(X @ W, int4_sym(Xr, axis=1) @ int4_sym(Wr, axis=0))


def incoherence(W: np.ndarray) -> float:
    """QuIP 的非相干度 μ = max|W_ij| · √(mn) / ‖W‖_F。元素幅度均匀时接近 √(2 ln(mn)) 量级, 有尖峰时很大。"""
    return float(np.abs(W).max() * np.sqrt(W.size) / np.linalg.norm(W))


# --------------------------------------------------------------------- #
# 学习旋转 (SpinQuant 的单层简化版)                                      #
# --------------------------------------------------------------------- #

def learn_rotation(X: np.ndarray, W: np.ndarray, Q0: np.ndarray, steps: int = 60, lr: float = 0.02) -> np.ndarray:
    """从 Q0 出发, 用 STE 估计 W4A4 输出误差对 Q 的梯度, 用 Cayley 变换更新, Q 始终正交。

    Ŷ = q(XQ) · q(QᵀW); STE 把量化器 q 的导数当成 1:
        ∂L/∂Q = Xᵀ G Bᵀ + W Gᵀ A,   A = q(XQ), B = q(QᵀW), G = 2(Ŷ − Y) / ‖Y‖²
    Cayley: Ω = ∂L/∂Q · Qᵀ − Q · (∂L/∂Q)ᵀ (反对称), Q ← (I + η/2 Ω)⁻¹ (I − η/2 Ω) Q, 正交性严格保持。
    步长按 ‖Ω‖ 归一化, 只保留校准集误差最低的那一步 (STE 梯度有噪声, 损失不单调)。
    """
    Y, I = X @ W, np.eye(len(Q0), dtype=np.float64)
    Q, best = Q0.astype(np.float64), (np.inf, Q0)
    for _ in range(steps):
        A, B = int4_sym(X @ Q, axis=1), int4_sym(Q.T @ W, axis=0)
        loss = rel(Y, A @ B)
        if loss < best[0]:
            best = (loss, Q.astype(np.float32))
        G = 2 * (A @ B - Y) / np.sum(Y * Y)
        grad = X.T @ G @ B.T + W @ G.T @ A                              # (n, n)
        Om = grad @ Q.T - Q @ grad.T                                    # 反对称: 沿正交矩阵流形的切向
        Om *= lr / max(np.linalg.norm(Om), 1e-12)                       # 固定步长, 只取方向
        Q = np.linalg.solve(I + Om / 2, (I - Om / 2) @ Q)               # ★ Cayley 变换: 正交矩阵乘正交矩阵
    return best[1]
