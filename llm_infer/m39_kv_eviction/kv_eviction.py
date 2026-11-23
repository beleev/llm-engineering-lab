"""
kv_eviction.py — 按重要性永久丢弃 KV: H2O / SnapKV / TOVA, 以及它们和 sink + 窗口、Quest 的区别

是什么: 长上下文的 KV cache 放不下时, 只留预算 B 个 token 的 KV, 其余直接删掉。各家的差别只在"按什么打分":
    sink + 窗口 (m16)  : 只看位置, 留开头 n_sink 个和最近 window 个
    H2O                : 累积注意力 (每个 token 被所有 query 看过多少) 选"重击者", 再留最近窗口, 预算两边各一半
    SnapKV             : prompt 末尾观察窗口 (通常就是用户的问题) 的 query 给前文投票, 1D max pool 后取 top-k,
                         prefill 结束时压一次, 生成阶段不再压
    TOVA               : 写满后每来一个 token, 删掉当前 query 注意力最低的那个
    Quest (m22)        : 不删。KV 全部留着, 每步只读打分最高的 k 个 block —— 省的是读取, 不是显存
关键: 淘汰是不可逆的。打分只能根据"到现在为止的 query"判断谁重要, 后面问到已删掉的信息就答不出来。
简化: 单层单头; 用合成的 key / query 方向植入 sink 和"针", 不跑真实模型。
简化: PyramidKV 按层分配预算 (低层多、高层少), 单层设定下无法体现, 只在 README 里讲。
简化: H2O 原文在生成时逐步淘汰; 这里 prefill 结束时按 prompt 内的累积注意力一次选满, 之后每步淘汰 1 个。
读代码盯住: 每个 *_keep 函数返回的是"留下哪些位置"; 没留下的位置之后任何 query 都看不到。
对应论文: H2O (arXiv 2306.14048) / SnapKV (2404.14469) / TOVA (2401.06104) / PyramidKV (2406.02069)。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from llm_infer.core import softmax


@dataclass
class Context:
    """一条合成对话: 前文 (含两根针) + 问题 1 + 回答 1 + 问题 2。每个位置都有自己的 q / k / v。"""
    q: np.ndarray            # (T, d) 每个位置的 query
    k: np.ndarray            # (T, d)
    v: np.ndarray            # (T, d)
    needles: dict            # {"A": 位置数组, "B": 位置数组}
    answers: np.ndarray      # (M, d) 候选答案向量, 第 0 个是 A 的答案, 第 1 个是 B 的答案
    t_q1: int                # 问题 1 最后一个 token 的位置 (prompt 的最后一个位置)
    t_q2: int                # 问题 2 最后一个 token 的位置


def make_context(seed: int, first: str = "A", n_ctx: int = 2032, n_q: int = 16, n_ans: int = 16, d: int = 64,
                 per_needle: int = 2, n_cand: int = 32) -> Context:
    """前文 n_ctx 个 token → 问题 1 (n_q 个) → 回答 1 (n_ans 个) → 问题 2 (n_q 个)。

    - 位置 0 是 sink: key 沿所有 query 共有的方向 u, 吸走普通 query 的大部分注意力 (训练过的 LLM 的典型形态)。
    - 针 A、B 各 per_needle 个 token, key 沿方向 a / b, value 是对应答案向量; 位置随机, 落在前文中段。
    - 问题 1 和回答 1 的 query 带方向 a (在找针 A), 问题 2 的 query 带方向 b。普通 token 的 query 只有 u + 噪声。
    - first="B" 时两个问题对调 (先问 B 再问 A), 前文完全相同。
    """
    rs = np.random.RandomState(seed)
    T = n_ctx + 2 * n_q + n_ans
    u, a, b = np.linalg.qr(rs.randn(d, 3))[0].T * np.sqrt(d)                # 三个正交方向, 模长 √d
    q = rs.randn(T, d) + 1.2 * u                                            # 普通 query: 对 sink 的打分 ≈ 1.2·√d ≈ 9.6
    k = rs.randn(T, d) * 0.2                                                # 普通 key: 普通 query 的打分 ~ N(0, 0.3²)
    v = rs.randn(T, d)
    k[0] = u * 1.0                                                          # sink: 普通 query 打分 ≈ 1.2·√d·1.0 ≈ 9.6
    pos = rs.choice(np.arange(n_ctx // 4, 3 * n_ctx // 4), 2 * per_needle, replace=False)
    needles = {"A": np.sort(pos[:per_needle]), "B": np.sort(pos[per_needle:])}
    answers = np.linalg.qr(rs.randn(d, n_cand))[0].T * 4.0                  # (n_cand, d) 互相正交的候选答案
    for name, direc, ans in (("A", a, answers[0]), ("B", b, answers[1])):
        k[needles[name]] = direc * 0.5 + rs.randn(per_needle, d) * 0.05     # 针的 key 沿 a 或 b; 普通 query 的打分 ~ N(0, 0.5²)
        v[needles[name]] = ans                                              # 针的 value 就是答案
    t_q1, t_ans = n_ctx + n_q - 1, n_ctx + n_q
    d1, d2 = (a, b) if first == "A" else (b, a)
    q[n_ctx:t_ans + n_ans] += 4.5 * d1                                      # 问题 1 + 回答 1 在找第一根针: 对针的打分 ≈ 18
    q[t_ans + n_ans:] += 4.5 * d2                                           # 问题 2 在找另一根
    return Context(q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), needles,
                   answers.astype(np.float32), t_q1, T - 1)


def attn_weights(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """单个 query 对一组 key 的注意力 (n,)。"""
    return softmax(K @ q / np.sqrt(len(q)))


def attend(ctx: Context, t: int, keep: np.ndarray) -> np.ndarray:
    """位置 t 的 query 只在 keep 这些位置上做注意力 → 输出 (d,)。"""
    return attn_weights(ctx.q[t], ctx.k[keep]) @ ctx.v[keep]


def readout(ctx: Context, out: np.ndarray) -> int:
    """输出里含多少比例的各候选答案 → 超过一半的那个候选的编号 (0 = A 的答案, 1 = B 的答案); 都不到一半返回 −1 (答不出)。"""
    frac = ctx.answers @ out / np.sum(ctx.answers[0] ** 2)                  # 候选互相正交、模长相同
    return int(np.argmax(frac)) if frac.max() > 0.5 else -1


# --------------------------------------------------------------------- #
# 淘汰策略: 都返回"留下的位置" (升序)                                    #
# --------------------------------------------------------------------- #

def sink_window_keep(t: int, budget: int, n_sink: int = 4) -> np.ndarray:
    """m16 的规则: 开头 n_sink 个 + 最近 budget − n_sink 个。t = 当前已写入的 token 数。"""
    return np.unique(np.r_[0:min(n_sink, t), max(t - (budget - n_sink), 0):t])


def snapkv_keep(ctx: Context, t_prompt: int, budget: int, window: int = 16, kernel: int = 7) -> np.ndarray:
    """SnapKV: prompt 最后 window 个 query 对前文的注意力求和投票, max pool 后取 top (budget − window), 再加上窗口本身。"""
    prefix = t_prompt - window
    votes = np.zeros(prefix)
    for t in range(prefix, t_prompt):                                       # 观察窗口里的每个 query
        votes += attn_weights(ctx.q[t], ctx.k[:t + 1])[:prefix]             # 因果: 只看得到自己之前的
    pad = kernel // 2
    padded = np.pad(votes, pad, constant_values=-np.inf)
    pooled = np.max(np.lib.stride_tricks.sliding_window_view(padded, kernel), axis=1)   # ★ 1D max pool: 选中的点带上邻居
    top = np.argsort(-pooled, kind="stable")[:budget - window]
    return np.sort(np.r_[top, prefix:t_prompt])


class H2O:
    """累积注意力 + 最近窗口。预算对半分: budget/2 个重击者, budget/2 个最近 token。"""

    def __init__(self, ctx: Context, t_prompt: int, budget: int):
        self.ctx, self.budget, self.recent = ctx, budget, budget // 2
        Q, K = ctx.q[:t_prompt], ctx.k[:t_prompt]
        S = Q @ K.T / np.sqrt(K.shape[1])
        S[np.triu_indices(t_prompt, 1)] = -np.inf                           # 因果 mask
        self.score = softmax(S, axis=-1).sum(0)                             # (t_prompt,) 每个 token 被看过多少
        old = np.arange(t_prompt - self.recent)
        heavy = old[np.argsort(-self.score[old], kind="stable")[:budget - self.recent]]
        self.keep = np.sort(np.r_[heavy, t_prompt - self.recent:t_prompt])
        self.score = dict(zip(range(t_prompt), self.score))

    def step(self, t: int) -> None:
        """写入位置 t: 用它的 query 给现有 cache 加分, 超预算就删掉最近窗口之外分数最低的那个。"""
        self.keep = np.r_[self.keep, t]
        w = attn_weights(self.ctx.q[t], self.ctx.k[self.keep])
        for p, x in zip(self.keep, w):
            self.score[p] = self.score.get(p, 0.0) + x
        if len(self.keep) > self.budget:
            old = self.keep[self.keep < t + 1 - self.recent]
            drop = min(old, key=lambda p: self.score[p])                    # 被删掉的位置从此不再出现
            self.keep = self.keep[self.keep != drop]


def tova_keep(ctx: Context, t_end: int, budget: int) -> np.ndarray:
    """TOVA: 逐 token 写入, 写满后每步删掉当前 query 注意力最低的那个 (从第 0 个位置一路流到 t_end)。"""
    keep = np.arange(min(budget, t_end))
    for t in range(budget, t_end):
        keep = np.r_[keep, t]
        w = attn_weights(ctx.q[t], ctx.k[keep])
        keep = np.delete(keep, int(np.argmin(w)))                           # ★ 只看当前这一个 query
    return keep
