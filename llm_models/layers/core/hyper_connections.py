"""
Hyper-Connections — 把 1 条残差流加宽成 n 条
    HC:  Zhu et al., ByteDance Seed, "Hyper-Connections" (arXiv 2409.19606, ICLR 2025)
    mHC: DeepSeek-AI, "mHC: Manifold-Constrained Hyper-Connections" (arXiv 2512.24880, ICML 2026);
         DeepSeek-V4 用它替换了残差连接 (arXiv 2606.19348)

是什么: Pre-LN 的残差主路只有 1 条, 每层只能 "全读、全写"。HC 把它加宽成 n 条流 x ∈ R^{n×D},
    一层的更新写成 (mHC 式 3)
        x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)
    H_pre [1, n] 是宽度连接, 把 n 条流混成这一层的那一个输入; H_post [1, n] 是深度连接,
    把这一层的输出按不同强度写回 n 条流; H_res [n, n] 决定流之间怎么互相混。
    HC 论文把 (H_pre, H_res) 叫 width connections、(H_post, diag H_res) 叫 depth connections,
    并把三者拼成一个 (n+1)×(n+1) 的矩阵; 普通 Pre-LN 残差就是 n = 1、三个值都为 1 的那一格。
解决什么: 层与层之间的连接强度和有效排列不再写死, 由模型自己学。HC 论文在 OLMo-1B / OLMo-7B /
    OLMoE-1B-7B 上报告 n = 4 的动态 HC 把验证 loss 降 0.02 到 0.03, MoE 上收敛快约 1.8 倍,
    额外参数约 +0.03%、FLOPs 约 +0.2%。
mHC 要修的问题: H_res 不加约束时, 第 j 层到第 k 层的连接是中间各层 H_res 的连乘
    (HC 论文式 6 的 c 系数), 连乘后不再是恒等映射, 信号会被整体放大或衰减。mHC 用
    "最大绝对行和" 衡量这个复合映射的增益 (Amax Gain Magnitude, 前向口径; 反向看最大绝对列和),
    在 27B 上实测 HC 的峰值约 3000, mHC 约 1.6。
mHC 的三个约束 (式 8): H_pre = σ(H̃_pre) 与 H_post = 2σ(H̃_post) 只要求非负, 避免信号相消;
    H_res = Sinkhorn-Knopp(H̃_res) 投到双随机矩阵 (行和与列和都为 1、元素非负) 构成的 Birkhoff 多胞形。
    双随机矩阵的集合对矩阵乘法封闭、谱范数不超过 1, 所以连乘多少层增益都还是 1。
关键数字: mHC 用 n = 4、Sinkhorn 迭代 20 次, 在自家大规模训练上报告 +6.7% 训练时间;
    27B MoE 上最终 loss 比基线低 0.021, 并且没有 HC 在约 12k 步出现的那次 loss 抬升。
本库的做法 (教学约定, 不代表原模型): 两个类都按 mHC 式 7 的 "静态偏置 + 输入相关项" 参数化
    H̃ = α · (RMSNorm(vec(x)) φ) + b, α 初始 0.01、φ 初始 0, 所以起步时只有静态偏置起作用。
    HC 与 mHC 的唯一区别就是 _constrain 里的三个约束函数, 以及静态偏置的初值 (见五个类常量)。
简化: HC 论文的动态权重用 tanh 加可学缩放, 这里统一用 mHC 的线性形式, 两篇的参数化不完全相同。
简化: n 条流是显式的 [B, T, n, D] 张量, 没有 mHC 的 kernel fusion 与混合精度算子,
      访存和显存按 n 倍算; 玩具配置下 φ 的参数占比也远高于论文报告的 +0.03%。
简化: Sinkhorn 固定迭代 20 次、不早停。最后一步是行归一化, 所以行和恰为 1、列和只是近似为 1。
简化: 一个 HyperConnection 包一整个 Pre-LN block, block 内部注意力与 FFN 之间仍是普通残差;
      真实实现把每一处残差连接都换成宽度连接加深度连接, 一层有两个。
读代码时盯住: forward 里的 pre / post / res 三个形状 —— [.., n]、[.., n]、[.., n, n], 对应式 3 的三项。
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.normalization import RMSNorm

# σ(20) 在 float32 下恰好是 1.0。只有 n = 1 时 mHC 需要这个饱和值
LOGIT = 20.0


def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """把打分投影成双随机矩阵 (mHC 式 9)。

    logits [..., n, n] → 同形状, 行和恰为 1、列和近似为 1、元素非负。
    先取 exp 变成正矩阵, 再交替做列归一化和行归一化。减去每个矩阵的最大值只为防 exp 溢出:
    整体乘一个正标量不改变归一化的结果。
    """
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(n_iters):
        m = m / m.sum(dim=-2, keepdim=True)   # T_c: 每列和为 1
        m = m / m.sum(dim=-1, keepdim=True)   # T_r: 每行和为 1
    return m


def composite_gain(res_mats: List[torch.Tensor]) -> Tuple[float, float, float]:
    """一串 H_res 连乘后的 (最大绝对行和, 最大绝对列和, 谱范数)。

    前两项就是 mHC 的 Amax Gain Magnitude: 最大绝对行和对应前向, 最大绝对列和对应反向。
    行和等于把全 1 向量送进复合映射后某条流能拿到的最大总权重, 偏离 1 就说明信号被整体放大或衰减。
    """
    prod = torch.eye(res_mats[0].size(-1), dtype=torch.float64)
    for m in res_mats:
        prod = m.to(torch.float64) @ prod
    return (
        prod.abs().sum(dim=-1).max().item(),
        prod.abs().sum(dim=-2).max().item(),
        torch.linalg.matrix_norm(prod, ord=2).item(),
    )


class HyperConnection(nn.Module):
    """
    一层的宽度连接与深度连接 (HC, 三个矩阵都不加约束)。

        x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)

    静态偏置按 HC 论文初始化: H_pre 是 one-hot e_{l mod n} (第 l 层只读第 l mod n 条流),
    H_post 全 1 (输出写回每条流), H_res 是单位矩阵 (流之间不混)。n 条流起步时数值相同,
    于是整个模型与普通 Pre-LN 残差逐位一致, 训练从这个起点出发 (mHC 的起点换了一种写法, 见子类)。

    Args:
        d_model:     每条流的宽度
        n:           流的条数 (论文的 expansion rate)
        layer_index: 层号, 只用来决定 H_pre 读哪一条流
        alpha:       输入相关项的门控系数初值 (mHC 表 5 用 0.01), 可学
    """

    def __init__(self, d_model: int, n: int, layer_index: int = 0, alpha: float = 0.01):
        super().__init__()
        if n < 1:
            raise ValueError(f"n 必须 >= 1, 当前 {n}")
        self.d_model, self.n = d_model, n
        self.sinkhorn_iters = 20
        self.last_res: Optional[torch.Tensor] = None   # 上一次前向用的 H_res, 供外部统计增益

        pre, post, res = self._init_biases(n, layer_index)
        self.b_pre = nn.Parameter(pre)                                         # [n]
        self.b_post = nn.Parameter(post)                                       # [n]
        self.b_res = nn.Parameter(res)                                         # [n, n]

        # 输入相关项: RMSNorm 后的展平流向量过一个线性投影。φ 初始为 0, 起步时这一项恰为 0。
        # 用裸 Parameter 而不是 nn.Linear, 是为了不被 utils/init.py 的 N(0, 0.02²) 覆盖
        self.norm = RMSNorm(n * d_model)
        self.phi_pre = nn.Parameter(torch.zeros(n * d_model, n))
        self.phi_post = nn.Parameter(torch.zeros(n * d_model, n))
        self.phi_res = nn.Parameter(torch.zeros(n * d_model, n * n))
        self.alpha = nn.Parameter(torch.full((3,), alpha))                     # α_pre / α_post / α_res

    def _init_biases(self, n: int, layer_index: int):
        """HC 的静态偏置 (论文的初始化): 第 l 层只读第 l mod n 条流, 输出写回每条流, 流间不混。"""
        pre = torch.zeros(n)
        pre[layer_index % n] = 1.0
        return pre, torch.ones(n), torch.eye(n)

    def _constrain(self, pre, post, res):
        """HC 不约束三个矩阵, 原样返回。mHC 覆盖这个方法。"""
        return pre, post, res

    def matrices(self, x: torch.Tensor):
        """x [B, T, n, D] → (H_pre [B, T, n], H_post [B, T, n], H_res [B, T, n, n])。"""
        v = self.norm(x.reshape(*x.shape[:-2], self.n * self.d_model))         # [B, T, n·D]
        pre = self.b_pre + self.alpha[0] * (v @ self.phi_pre)                  # [B, T, n]
        post = self.b_post + self.alpha[1] * (v @ self.phi_post)               # [B, T, n]
        res = self.b_res + self.alpha[2] * (v @ self.phi_res).unflatten(-1, (self.n, self.n))
        return self._constrain(pre, post, res)

    def forward(self, x: torch.Tensor, block, **kwargs) -> torch.Tensor:
        """x [B, T, n, D] → [B, T, n, D]。

        block 是一个自带残差主路的 Pre-LN block, 即 block(h) = h + F(h); 式 3 里的 F 只要分支那一项,
        所以这里取 block(inp) − inp。kwargs (mask / rope / position_ids / cache) 原样转给 block。
        """
        pre, post, res = self.matrices(x)
        self.last_res = res.detach()
        inp = (pre.unsqueeze(-1) * x).sum(dim=-2)                              # ★ 宽度连接: n 条流混成 1 个输入
        out = block(inp, **kwargs) - inp                                       # F(inp), [B, T, D]
        mixed = torch.einsum("...ij,...jd->...id", res, x)                     # 流间混合, [B, T, n, D]
        return mixed + post.unsqueeze(-1) * out.unsqueeze(-2)                  # ★ 深度连接: 输出写回 n 条流


class ManifoldHyperConnection(HyperConnection):
    """
    mHC: 把 HC 的三个矩阵投到约束流形上。

    H_pre = σ(H̃_pre) 和 H_post = 2σ(H̃_post) 只要求非负 (H_post 还限在 (0, 2) 内),
    H_res = Sinkhorn-Knopp(H̃_res) 投到双随机矩阵。双随机矩阵对乘法封闭, 所以连乘多少层,
    复合映射的最大绝对行和都还是 1, 恒等映射的性质不会随深度丢掉。

    起步时仍与 Pre-LN 逐位一致, 但走的是另一条路: n 条流数值相同时, 任何行和为 1 的矩阵都把它们
    原样传过去, 所以 H_res 不必是单位矩阵; H_pre 每条流取 1/n、加起来恰为 1, 层输入就等于流的共同值。
    三个偏置都取不饱和的值 —— σ 和 Sinkhorn 在饱和处梯度几乎为 0, 矩阵会一直卡在初值上。
    论文没有写静态偏置的初始化, 这是本库的选法。
    """

    def _init_biases(self, n: int, layer_index: int):
        """σ / 2σ / Sinkhorn 之前的 logit: H_pre 每条流 1/n, H_post 全 1, H_res 偏向单位矩阵但不饱和。"""
        # σ(log(p / (1 − p))) = p = 1/n, n 条加起来是 1; n = 1 时要 σ = 1, 只能用饱和值
        logit = LOGIT if n == 1 else math.log(1.0 / (n - 1))
        return torch.full((n,), logit), torch.zeros(n), 2.0 * torch.eye(n)

    def _constrain(self, pre, post, res):
        return torch.sigmoid(pre), 2 * torch.sigmoid(post), sinkhorn_knopp(res, self.sinkhorn_iters)
