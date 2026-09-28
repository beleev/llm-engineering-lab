"""
模型合并 — Task Arithmetic (Ilharco et al., 2023) / TIES (Yadav et al., 2023) / DARE (Yu et al., 2024) / SLERP

是什么: 同一个基座 θ₀ 在不同任务上各自微调出 θ_A、θ_B, 不再训练, 直接在**权重空间**里把它们拼成一个模型。
解决什么: 多任务要么一起重训 (要数据、要算力), 要么部署多份权重。合并只要权重, 几秒钟, 不要数据。
核心公式:  任务向量 τ_t = θ_t − θ₀
           Task Arithmetic:  θ = θ₀ + λ·Σ_t τ_t                        (λ = 1/T 就是简单平均)
           TIES:  ① 修剪: 每个 τ_t 只留 |·| 最大的 density 比例, 其余置 0
                  ② 选符号: γ = sign(Σ_t τ̂_t)          (按 "质量" 投票, 不是按人头)
                  ③ 不相交合并: 每个坐标只对 "非零且符号 = γ" 的任务取平均;  θ = θ₀ + λ·τ_m
           DARE:  τ̃_t = m ⊙ τ_t / (1−p),  m ~ Bernoulli(1−p)   (随机丢 p, 剩下的放大, 期望不变), 再做 Task Arithmetic
           SLERP: 两个权重张量摊平成向量, 沿球面插值:  sin((1−t)Ω)/sinΩ · θ_A + sin(tΩ)/sinΩ · θ_B,  Ω = ∠(θ_A, θ_B)
读代码时盯住: `task_vectors()` —— 除 SLERP 外, 所有方法都只是对 τ 做不同的 "逐坐标" 处理再加回 θ₀。
所有函数只处理浮点张量, 输入输出都是 state_dict; 不碰模型结构 (LoRA 也一样: τ 就是合并后的 ΔW = (α/r)BA)。
"""

from typing import Dict, List

import torch

StateDict = Dict[str, torch.Tensor]


def task_vectors(base: StateDict, finetuned: List[StateDict]) -> List[StateDict]:
    """τ_t = θ_t − θ₀, 只取浮点张量 (整型 buffer 没有 "方向" 可言)。"""
    return [{k: ft[k] - v for k, v in base.items() if v.is_floating_point()} for ft in finetuned]


def add_to_base(base: StateDict, delta: StateDict, lam: float = 1.0) -> StateDict:
    """θ₀ + λ·δ; δ 里没有的键 (整型 buffer) 原样取基座的。"""
    return {k: v + lam * delta[k] if k in delta else v.clone() for k, v in base.items()}


def task_arithmetic(base: StateDict, taus: List[StateDict], lam: float) -> StateDict:
    return add_to_base(base, {k: sum(t[k] for t in taus) for k in taus[0]}, lam)


def ties_merge(base: StateDict, taus: List[StateDict], density: float, lam: float = 1.0) -> StateDict:
    """TIES: 修剪 → 选符号 → 不相交平均。density = 每个任务向量保留的比例 (逐张量取 top-k)。"""
    merged = {}
    for k in taus[0]:
        stack = torch.stack([t[k] for t in taus])                         # [T, *shape]
        flat = stack.reshape(len(taus), -1)                               # [T, N]
        n_keep = max(1, int(density * flat.shape[1]))
        # ① 修剪: 每个任务只保留自己 |τ| 的 top-k; 第 k 大的值作为阈值
        thresh = flat.abs().kthvalue(flat.shape[1] - n_keep + 1, dim=1, keepdim=True).values   # [T, 1]
        trimmed = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
        # ② 选符号: 修剪后求和的符号 —— 大幅度的更新票更重
        sign = torch.sign(trimmed.sum(dim=0))                             # [N]
        # ③ 不相交合并: 只对 "非零且与 γ 同号" 的任务取平均; 冲突的那一方直接不参与, 而不是被平均稀释
        agree = (torch.sign(trimmed) == sign) & (trimmed != 0)            # [T, N]
        total = (trimmed * agree).sum(dim=0)
        merged[k] = (total / agree.sum(dim=0).clamp(min=1)).reshape(stack.shape[1:])
    return add_to_base(base, merged, lam)


def dare(taus: List[StateDict], p: float, seed: int = 0) -> List[StateDict]:
    """DARE: 每个坐标以概率 p 置零, 其余 ×1/(1−p) —— E[τ̃] = τ, 但单次实现有方差。返回处理后的任务向量。"""
    g = torch.Generator().manual_seed(seed)
    return [{k: v * (torch.rand(v.shape, generator=g) >= p) / (1 - p) for k, v in t.items()} for t in taus]


def slerp(a: StateDict, b: StateDict, t: float = 0.5, eps: float = 1e-8) -> StateDict:
    """
    对两个完整模型 (不是任务向量) 逐张量球面插值, 与 mergekit 的做法一致。
    两个方向几乎平行时 (sinΩ → 0) 退化为线性插值。
    """
    out = {}
    for k, va in a.items():
        vb = b[k]
        if not va.is_floating_point():
            out[k] = va.clone()
            continue
        cos = torch.dot(va.flatten(), vb.flatten()) / (va.norm() * vb.norm() + eps)
        omega = torch.arccos(cos.clamp(-1, 1))
        if omega.sin() < 1e-4:
            out[k] = (1 - t) * va + t * vb
        else:
            out[k] = (torch.sin((1 - t) * omega) * va + torch.sin(t * omega) * vb) / omega.sin()
    return out


def sign_conflict(taus: List[StateDict]) -> float:
    """两个任务向量里, 双方都非零的坐标中符号相反的比例 —— "干扰" 的一个直接度量。"""
    a = torch.cat([v.flatten() for v in taus[0].values()])
    b = torch.cat([v.flatten() for v in taus[1].values()])
    both = (a != 0) & (b != 0)
    return float(((torch.sign(a) != torch.sign(b)) & both).sum() / both.sum())
