"""
hybrid_cache.py — 混合模型 (线性注意力层 + 全注意力层) 的推理缓存: 定长状态 + 分页 KV

是什么: Qwen3-Next 这类混合模型每 4 层里 3 层是 Gated DeltaNet (线性注意力), 1 层是全注意力。
      全注意力层照常为每个 token 存一行 K/V, 递推层每条序列只有一份定长状态 S。
瓶颈:   KV 是按 token 排的, 可以截断、按块复用; 状态是整段前缀压出来的一个矩阵, 不能从位置 T 的状态
      切出位置 L < T 的状态。前缀缓存和投机解码的回滚都建立在「截断」上, 换成状态就要另想办法。
关键公式 (单头 Gated DeltaNet, 与 llm_models/layers/sparse/linear_attention.py 同一写法):
      S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,   o_t = S_tᵀ q_t,   S ∈ R^{d_k × d_v}
做法:   - 前缀缓存: 每隔 interval 个 token 存一份状态快照, 命中长度向下取整到快照位置 (PrefixStateCache)。
      - 投机解码回滚: KV 截断即可; 状态要么在验证时逐位置留快照, 要么从验证前的快照重算 (speculative_generate)。
      - MTP 头当自投机草稿: 读主模型最后的隐状态和下一个 token 的嵌入, 预测再下一个位置的隐状态 (MTPHead)。
简化:   单头; 没有 MLP、没有短卷积 (真实 Gated DeltaNet 在 q/k/v 前有一层 kernel=4 的因果卷积, 也是要缓存的状态);
      验证时逐 token 喂给模型, 和一次并行前向的结果相同, 但没有并行的速度。
简化:   MTP 头是一个线性映射, 用最小二乘拟合主模型自己的隐状态 (自蒸馏), 不是 DeepSeek-V3 那样带一个
      Transformer 块、和主模型一起训练的模块。随机权重主模型的贪心输出几乎没有规律, 所以 demo 另给主模型
      植入一张「当前 token → 下一个 token」的后继表 (plant), 让接受率有东西可比; 两种主模型的结果都打印。
真实系统: vLLM 的混合模型前缀缓存 (状态存在 block 边界) / SGLang 的 hybrid cache checkpoint / Marconi (arXiv 2411.19379)。
盯住:   HybridCache (states 和 kv 两种东西放在一起) 与 speculative_generate 里三种 rollback 的分支。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from llm_infer.core.tiny_model import apply_rope, precompute_rope
from llm_infer.core.utils import dense_attention, rms_norm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


@dataclass(frozen=True)
class HybridConfig:
    """layout 里 R = Gated DeltaNet 层, A = 全注意力层。RRRA 对应 Qwen3-Next 的 3:1。"""
    vocab: int = 64
    d: int = 32
    d_state: int = 16          # 递推层的 d_k = d_v
    layout: str = "RRRARRRA"
    max_len: int = 512


@dataclass
class HybridCache:
    """一条序列的推理缓存。states 定长, 每层一个 (d_k, d_v) 矩阵; kv 每层随 token 数增长。"""
    states: List[np.ndarray]
    kv: List[Tuple[np.ndarray, np.ndarray]]
    n: int = 0                  # 已经喂进去的 token 数

    def copy_states(self) -> List[np.ndarray]:
        return [s.copy() for s in self.states]

    def truncate_kv(self, n: int) -> None:
        """KV 是按 token 排的, 截掉尾部就等于回到「只看过前 n 个 token」。状态没有这个操作。"""
        self.kv = [(k[:n], v[:n]) for k, v in self.kv]


class HybridLM:
    """随机权重的 numpy 小混合模型。step 是增量路径, forward 是整段重算路径 (注意力层用整块因果矩阵)。

    plant > 0 时在输出头上植入一张「当前 token → 下一个 token」的后继表 (logits 加 plant·onehot(succ[x]))。
    随机权重的贪心输出几乎没有规律, 植入后主模型的输出有一部分可预测, MTP 头才有东西可学。
    """

    def __init__(self, cfg: HybridConfig = HybridConfig(), seed: int = 0, plant: float = 0.0):
        self.cfg = cfg
        rs = np.random.RandomState(seed)
        D, S, V = cfg.d, cfg.d_state, cfg.vocab
        rand = lambda *shape: (rs.randn(*shape) * shape[0] ** -0.5).astype(np.float32)
        self.emb = rand(V, D) * np.float32(V ** 0.5 / D ** 0.5)       # 让嵌入每维的量级约为 1
        self.head = rand(D, V)
        self.layers = []
        for kind in cfg.layout:
            if kind == "R":
                self.layers.append(dict(kind="R", g=np.ones(D, np.float32), wq=rand(D, S), wk=rand(D, S),
                                        wv=rand(D, S), wa=rand(D), wb=rand(D), wo=rand(S, D)))
            else:
                self.layers.append(dict(kind="A", g=np.ones(D, np.float32), wq=rand(D, D), wk=rand(D, D),
                                        wv=rand(D, D), wo=rand(D, D)))
        self.g_f = np.ones(D, np.float32)
        self.bigram = np.zeros((V, V), np.float32)
        self.bigram[np.arange(V), np.random.RandomState(seed + 1).permutation(V)] = plant
        self.cos, self.sin = precompute_rope(D, cfg.max_len)
        self.n_rec = cfg.layout.count("R")
        self.n_attn = cfg.layout.count("A")

    # ---------------- 增量路径 ---------------- #
    def new_cache(self) -> HybridCache:
        S, D = self.cfg.d_state, self.cfg.d
        return HybridCache(states=[np.zeros((S, S), np.float32) for _ in range(self.n_rec)],
                           kv=[(np.zeros((0, D), np.float32), np.zeros((0, D), np.float32)) for _ in range(self.n_attn)])

    def step(self, tok: int, cache: HybridCache) -> Tuple[np.ndarray, np.ndarray]:
        """喂一个 token, 原地更新 cache → (logits (V,), 最后的隐状态 h (D,))。"""
        x = self.emb[tok][None, :]                                    # (1, D)
        ri = ai = 0
        for L in self.layers:
            u = rms_norm(x, L["g"])
            if L["kind"] == "R":
                q, k = _l2norm(u @ L["wq"])[0], _l2norm(u @ L["wk"])[0]
                v = (u @ L["wv"])[0]
                a = _sigmoid(u @ L["wa"] + 2.0)[0]                        # 遗忘门, 偏置 2.0 让它起步接近 0.88
                b = _sigmoid(u @ L["wb"])[0]                              # 写入强度
                S = cache.states[ri]
                S = a * (S - b * np.outer(k, k @ S)) + b * np.outer(k, v)   # ★ 整段前缀被压进这一个矩阵
                cache.states[ri] = S.astype(np.float32)
                x = x + (S.T @ q)[None, :] @ L["wo"]
                ri += 1
            else:
                pos = np.array([cache.n])
                q = apply_rope(u @ L["wq"], self.cos, self.sin, positions=pos)
                k = apply_rope(u @ L["wk"], self.cos, self.sin, positions=pos)
                K, Vv = cache.kv[ai]
                K, Vv = np.concatenate([K, k]), np.concatenate([Vv, u @ L["wv"]])
                cache.kv[ai] = (K, Vv)
                x = x + dense_attention(q, K, Vv) @ L["wo"]             # 1 个 query 看全部历史, mask 自动全可见
                ai += 1
        cache.n += 1
        h = rms_norm(x, self.g_f)[0]
        return self.logits(h, tok), h

    def logits(self, h, tok):
        """输出头: 隐状态乘 head, 再加植入的后继表那一行。MTP 头共用它。"""
        return h @ self.head + self.bigram[tok]

    def feed(self, ids, cache: HybridCache):
        """依次喂一串 token → 每个位置的 (logits, h)。"""
        out = [self.step(int(t), cache) for t in ids]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    # ---------------- 整段重算路径 ---------------- #
    def forward(self, ids) -> Tuple[np.ndarray, np.ndarray]:
        """不带缓存, 从头算整段 → (logits (T, V), h (T, D))。注意力层一次算整块因果矩阵, 递推层按时间扫一遍。"""
        ids = np.asarray(ids)
        T = len(ids)
        x = self.emb[ids]
        for L in self.layers:
            u = rms_norm(x, L["g"])
            if L["kind"] == "R":
                Q, Kk = _l2norm(u @ L["wq"]), _l2norm(u @ L["wk"])
                Vv = u @ L["wv"]
                A, B = _sigmoid(u @ L["wa"] + 2.0), _sigmoid(u @ L["wb"])
                S = np.zeros((self.cfg.d_state,) * 2, np.float32)
                o = np.zeros_like(Vv)
                for t in range(T):
                    S = A[t] * (S - B[t] * np.outer(Kk[t], Kk[t] @ S)) + B[t] * np.outer(Kk[t], Vv[t])
                    o[t] = S.T @ Q[t]
                x = x + o @ L["wo"]
            else:
                q = apply_rope(u @ L["wq"], self.cos, self.sin)
                k = apply_rope(u @ L["wk"], self.cos, self.sin)
                x = x + dense_attention(q, k, u @ L["wv"]) @ L["wo"]  # mask=None → 因果
        h = rms_norm(x, self.g_f)
        return self.logits(h, ids), h

    def greedy(self, prompt, n_new: int) -> List[int]:
        """基线: 逐 token 贪心生成。"""
        cache = self.new_cache()
        logits, _ = self.feed(prompt, cache)
        out = list(prompt) + [int(np.argmax(logits[-1]))]
        while len(out) < len(prompt) + n_new:
            lg, _ = self.step(out[-1], cache)
            out.append(int(np.argmax(lg)))
        return out


# --------------------------------------------------------------------- #
# 前缀缓存: KV 可以截断, 状态只能在快照位置恢复                          #
# --------------------------------------------------------------------- #

@dataclass
class PrefixStateCache:
    """存过的序列: 注意力层的 KV 全量保留, 递推层只在 interval 的整数倍位置存状态快照。"""
    interval: int
    entries: list = field(default_factory=list)       # [(ids, kv, {位置: 状态快照})]

    def insert(self, model: HybridLM, ids) -> None:
        cache, snaps = model.new_cache(), {}
        for i, t in enumerate(ids, start=1):
            model.step(int(t), cache)
            if i % self.interval == 0:
                snaps[i] = cache.copy_states()               # 喂完前 i 个 token 时的状态
        self.entries.append((list(ids), cache.kv, snaps))

    def match(self, model: HybridLM, ids) -> Tuple[int, int, HybridCache]:
        """→ (公共前缀长度 L, 可用命中长度 h, 覆盖前 h 个 token 的 cache)。

        h = 不超过 L 的最大快照位置; 还要留至少 1 个 token 现算, 才有最后一个位置的 logits。
        """
        best = (0, 0, model.new_cache())
        for seq, kv, snaps in self.entries:
            L = 0
            while L < min(len(seq), len(ids)) and seq[L] == ids[L]:
                L += 1
            h = (min(L, len(ids) - 1) // self.interval) * self.interval   # ★ 命中粒度 = 快照间隔
            if h > best[1]:
                c = model.new_cache()
                c.states = [s.copy() for s in snaps[h]]
                c.kv, c.n = [(k[:h], v[:h]) for k, v in kv], h
                best = (L, h, c)
        return best


def snapshot_bytes(n_rec: int, heads: int, d_k: int, d_v: int, conv: int = 0, dtype_bytes: int = 2) -> int:
    """一份「所有递推层的状态」的字节数: 每层 heads·d_k·d_v 的状态矩阵, 加上短卷积缓存的 conv 个元素。"""
    return n_rec * (heads * d_k * d_v + conv) * dtype_bytes


def kv_bytes_per_token(n_attn: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """全注意力层每个 token 的 K + V 字节数。"""
    return n_attn * 2 * kv_heads * head_dim * dtype_bytes


# --------------------------------------------------------------------- #
# MTP 头: 读 (h_t, emb(x_{t+1})) → 预测 h_{t+1}, 再过主模型的输出头      #
# --------------------------------------------------------------------- #

class MTPHead:
    """线性 MTP 头。W 随机时就是一个没训练过的头; fit 用主模型自己的隐状态做最小二乘。"""

    def __init__(self, model: HybridLM, seed: int = 0):
        self.m = model
        D = model.cfg.d
        self.W = (np.random.RandomState(seed).randn(2 * D, D) * (2 * D) ** -0.5).astype(np.float32)

    def _feat(self, h, tok):
        return np.concatenate([h, rms_norm(self.m.emb[tok], self.m.g_f)])

    def fit(self, seqs, ridge: float = 1e-2) -> int:
        """seqs: 主模型自己贪心生成的序列。拟合 h_{t+1} ≈ [h_t ; emb(x_{t+1})] W, 返回样本数。"""
        X, Y = [], []
        for s in seqs:
            _, h = self.m.forward(s)
            X += [self._feat(h[t], s[t + 1]) for t in range(len(s) - 1)]
            Y += [h[t + 1] for t in range(len(s) - 1)]
        X, Y = np.array(X), np.array(Y)
        self.W = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y).astype(np.float32)
        return len(X)

    def draft(self, h, tok, k: int) -> List[int]:
        """从 (h_t, x_{t+1}) 出发链式猜 k 个 token: 每一步把预测出的隐状态当作下一步的 h。"""
        out = []
        for _ in range(k):
            h = self._feat(h, tok) @ self.W                  # 猜的是「喂完 tok 之后」的隐状态
            tok = int(np.argmax(self.m.logits(h, tok)))
            out.append(tok)
        return out


# --------------------------------------------------------------------- #
# 投机解码: 草稿被拒时怎么把状态退回去                                   #
# --------------------------------------------------------------------- #

@dataclass
class SpecStats:
    rounds: int = 0             # 目标模型验证的轮数
    proposed: int = 0           # 草稿 token 总数
    accepted: int = 0           # 被接受的草稿 token 数
    replayed: int = 0           # rollback="recompute" 时为了恢复状态重算的 token 数
    peak_snapshots: int = 0     # 同一时刻最多存了几份状态


def speculative_generate(model: HybridLM, mtp: MTPHead, prompt, n_new: int, k: int,
                         rollback: str) -> Tuple[List[int], SpecStats]:
    """MTP 草稿 + 主模型贪心验证。rollback ∈ {"per_step", "recompute", "truncate_only"}。

    约定 (同 m07): cache 覆盖 out[:-1], out[-1] 还没喂; h 是喂 out[-2] 时得到的隐状态。
    - per_step     : 验证时每喂一个 token 存一份状态, 被拒后取第 n 份。多占 k+1 份状态。
    - recompute    : 只在验证前存一份状态, 被拒后从它把接受的 n+1 个 token 重新喂一遍。多花算力。
    - truncate_only: 只截断 KV, 状态留在「看过全部草稿」的样子 —— 错的, 用来对照。
    """
    st = SpecStats()
    cache = model.new_cache()
    logits, hs = model.feed(prompt[:-1], cache)
    h = hs[-1]
    out = list(prompt)
    while len(out) < len(prompt) + n_new:
        d = mtp.draft(h, out[-1], k)
        base_n = cache.n
        base_states = cache.copy_states() if rollback == "recompute" else None
        snaps = []
        lg, hv = [], []
        for t in [out[-1]] + d:                              # 目标模型验证 k+1 个位置
            l1, h1 = model.step(t, cache)
            lg.append(l1); hv.append(h1)
            if rollback == "per_step":
                snaps.append(cache.copy_states())
        pred = [int(np.argmax(x)) for x in lg]
        n = 0
        while n < k and d[n] == pred[n]:
            n += 1
        keep = base_n + n + 1                                # cache 应覆盖 out[-1] 和接受的 n 个草稿
        cache.truncate_kv(keep)
        if rollback == "per_step":
            cache.states = snaps[n]
            st.peak_snapshots = max(st.peak_snapshots, k + 1)  # k+1 份逐位置快照
        elif rollback == "recompute":
            st.peak_snapshots = max(st.peak_snapshots, 1)      # 只有验证前那一份
            if n < k:                                        # 有草稿被拒, 状态已经吃进了错的 token
                cache.states, cache.n = base_states, base_n
                cache.truncate_kv(base_n)
                model.feed([out[-1]] + d[:n], cache)
                st.replayed += n + 1
        cache.n = keep
        h = hv[n]
        out += d[:n] + [pred[n]]
        st.rounds += 1
        st.proposed += k
        st.accepted += n
    return out[:len(prompt) + n_new], st
