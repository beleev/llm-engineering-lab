"""
gptq.py — GPTQ: 逐行量化权重, 用 Hessian 逆把每一行的舍入误差摊给还没量化的行

是什么: RTN 每个权重各自四舍五入, 误差互不相干地累加到输出上; GPTQ (Frantar et al. 2022) 把输出误差
    ‖XW − XŴ‖² = tr(ΔWᵀ H ΔW),  H = XᵀX / N   (D_in, D_in)
    当成二次型来最小化: 量化第 i 个输入通道 (W 的第 i 行) 后, 把它的误差 δ_i 按 H 的相关性
    提前"补"到后面的行上 —— 后面的行被量化时, 它们的舍入顺带把前面的误差抵消掉一部分。
解决的瓶颈: 同样 INT4 / 同样存储格式下, 输出误差比 RTN 小 (AWQ 保护重要通道; GPTQ 补偿误差)。
关键公式 (OBS 的逐行形式, 用 H⁻¹ 的上三角 Cholesky 因子 U, H⁻¹ = UᵀU):
    δ_i = (W_i − Q(W_i)) / U_ii
    W_{i+1:} -= U_{i,i+1:}ᵀ · δ_i
    H 近似对角 (输入通道互不相关) 时 U 的非对角元 ≈ 0, GPTQ 退化成 RTN —— 收益全部来自激活通道之间的相关性。
读代码盯住: `gptq_quantize` 循环里的 `W[i + 1:] -= ...` 一行。
约定: W (D_in, D_out), Y = X @ W (与 m08 相同); 论文写法是 W (out, in) 逐列量化, 这里的"行"就是论文的"列"。
对应真实系统: AutoGPTQ / GPTQModel, vLLM `quantization="gptq"` (+ Marlin kernel); 存储格式与 m08 的 group-wise INT4 相同。
"""
from __future__ import annotations
import numpy as np

from llm_infer.m08_quantization.int8_weight import quantize_affine


def gptq_quantize(W: np.ndarray, X_calib: np.ndarray, bits: int = 4, group: int = 32,
                  damp: float = 0.01) -> np.ndarray:
    """→ Ŵ (D_in, D_out) 等效浮点权重。group 的 scale/lo 在走到该组第一行时, 用"已被补偿过"的权重现算。"""
    D_in, D_out = W.shape
    H = (X_calib.T @ X_calib / len(X_calib)).astype(np.float64)          # (D_in,D_in) 二阶统计
    H += damp * np.mean(np.diag(H)) * np.eye(D_in)                        # 阻尼: 校准样本少时 H 接近奇异
    U = np.linalg.cholesky(np.linalg.inv(H)).T                            # 上三角, H⁻¹ = UᵀU
    W = W.astype(np.float64).copy()
    Q = np.zeros_like(W)                                                  # 量化后的权重, 逐行填
    for i in range(D_in):
        if i % group == 0:                                                # 新一组: 按当前 (已补偿的) 权重定格点
            g = quantize_affine(W[i:i + group], bits, axis=0)             # scale/lo: (1, D_out)
            scale, lo = g.scale.astype(np.float64)[0], g.lo.astype(np.float64)[0]
        Q[i] = np.clip(np.round((W[i] - lo) / scale), 0, 2 ** bits - 1) * scale + lo   # 第 i 行舍入到格点
        delta = (W[i] - Q[i]) / U[i, i]                                   # (D_out,)
        W[i + 1:] -= np.outer(U[i, i + 1:], delta)                        # (D_in-i-1, D_out) 误差摊给后面所有行
    # 简化: 逐行 rank-1 更新 O(D_in²·D_out); 真 GPTQ 攒 128 行一批 (lazy batch) 提高访存效率, 数学相同
    return Q.astype(np.float32)
