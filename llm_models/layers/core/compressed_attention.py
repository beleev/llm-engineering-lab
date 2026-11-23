"""
压缩注意力 — DeepSeek-V4 的 CSA 与 HCA (arXiv 2606.19348, 2026-04)

是什么: V3.2 的 DSA 用 Lightning Indexer 在 token 上选 top-k, 省的是算力, KV cache 不省。
    V4 换成先把序列压短, 再在压缩后的条目上选:
        CSA (Compressed Sparse Attention):  每 m = 4 个 token 压成 1 条, indexer 在压缩条目上选 top-k
        HCA (Heavily Compressed Attention): 每 m' = 128 个 token 压成 1 条, 不选, 稠密读全部压缩条目
    两种层交错排布, 另外每个 query 都额外读最近 n_win = 128 个未压缩 token (滑动窗口分支)。
压缩器怎么算 (不是池化, 是学出来的加权和): 每个 token 先投出一个 KV 向量 C 与一个压缩打分 Z,
    一个压缩条目是它覆盖的那些 token 的 softmax 加权和, 权重里还加了一组按块内位置学的偏置 B:
        S = softmax(Z + B),   C^{Comp}_i = Σ_j S^a_j ⊙ C^a_j + Σ_j S^b_j ⊙ C^b_j
    CSA 有两路 KV 序列 a 和 b, 块与块之间重叠: 条目 i 覆盖 2m 个 token (自己这一块加前一块);
    HCA 只有一路、不重叠。条目是一个向量, 同时当 key 和 value 用 (shared-KV 的多查询注意力),
    RoPE 只转它的最后 64 维 (本库按比例缩小)。
因果性: 条目 i 要等它覆盖的 token 全部到齐才可见, 也就是位置 (i+1)m − 1 之后才进得了注意力。
关键数字 (V4-Pro / V4-Flash 的配置): m = 4, m' = 128, n_win = 128, CSA top-k 取 1024 / 512 个条目,
    indexer 64 头、每头 128 维, 层数 61 / 43, 头维 c = 512。KV cache 按 RoPE 那 64 维用 BF16、
    其余用 FP8 存, 比纯 BF16 省掉将近一半; 1M 上下文下 V4-Pro 的单 token 推理 FLOPs 是 V3.2 的 27%、
    KV cache 是 10% (V4-Flash 是 10% 和 7%)。原文还给了一句相对比较: 滑窗条目大约是压缩条目的 8 倍大。
    本文件的实现下这个比值恰好是 2m = 8 —— 滑窗条目有各自的 K 和 V, 压缩条目一个向量兼当 K 和 V、
    每 m 个 token 才一条。
本库的做法 (教学约定, 不代表原模型): 一个类加两个开关就覆盖 CSA 和 HCA ——
    sparse 决定要不要 indexer 与 top-k, overlap 决定压缩器是两路重叠还是一路不重叠。
简化: 原文让 indexer 的 query 与主 query 共用同一个低秩 latent; 本文件给 indexer 单独一个投影,
      因为库里的 LightningIndexer 要求 query 和 key 的输入维度相同 (这里 key 是压缩条目)。
简化: 原文式 27 还给每个 head 一个可学的 sink logit, 本文件没有实现 (sink 的机制见 GPT-OSS 一章)。
简化: 滑窗 token 与压缩条目放进同一次 softmax (按原文式 27 的分母形式)。
简化: 条目的 RoPE 位置取它自己那一块的最后一个 token; 原文只说 "转最后 64 维", 没写取哪个位置。
简化: indexer 的 key 每步从缓存的条目重算, 不像主 KV 那样再缓存一份。
简化: 稀疏掩码仍是显式的 [B, T, S] 布尔张量, 没有真的跳过矩阵乘。
简化: 不支持 padding mask。压缩条目会把 pad token 混进加权和, 真实实现要在压缩器里屏蔽它们。
读代码时盯住: compress_entries 返回的 entry_pos —— 条目 i 的 "到齐位置" (i+1)m − 1,
    它同时当 RoPE 的位置和因果可见性的门槛。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import LightningIndexer
from llm_models.layers.core.normalization import RMSNorm


def topk_mask(scores: torch.Tensor, base_mask: torch.Tensor, k: int) -> torch.Tensor:
    """在 base_mask 允许的列里, 每行保留分数最高的 k 列。scores / base_mask [B, T, S] → bool [B, T, S]。

    与 attention.py 的 MultiHeadLatentSparseAttention 同一套做法: 必须先 mask 再 top-k, 否则会选到
    还没到齐的条目; 用 stable sort 而不是 topk, 并列时固定取靠前的列, 带 cache 的解码才和整段前向一致。
    """
    B, T, S = scores.shape
    if k >= S:
        return base_mask
    masked = scores.masked_fill(~base_mask, float("-inf"))
    idx = masked.sort(dim=-1, descending=True, stable=True).indices[..., :k]
    keep = torch.zeros(B, T, S, dtype=torch.bool, device=scores.device)
    keep.scatter_(-1, idx, True)
    return keep & base_mask        # 可见条目不足 k 个的行会选到 -inf, 这里剔掉


class CompressedAttention(nn.Module):
    """
    DeepSeek-V4 的压缩注意力。sparse=True 且 overlap=True 是 CSA, 两者都 False 是 HCA。

    接口与库内注意力层对齐 (可直接装进 DeepSeekBlock):
        forward(q, k=None, v=None, mask=None, rope=None, position_ids=None, cache=None) → [B, T, D]
    mask 参数只为统一签名: 因果性由条目的到齐位置和滑窗的位置差决定, 本层不支持 padding mask。

    Args:
        d_model / num_heads: 模型维度 / query 头数
        latent_dim:          压缩条目的维度 c, 也是每个 query 头的维度
        qk_rope_head_dim:    c 里转 RoPE 的那一段
        block_size:          压缩块大小 m (V4-Pro 的 CSA 取 4, HCA 取 128)
        window:              滑动窗口里未压缩 token 的个数 n_win; 0 表示不要这条分支
        sparse / overlap:    要不要 indexer 与 top-k / 压缩器是两路重叠还是一路不重叠
        sparse_top_k:        每个 query 保留的压缩条目数
        q_lora_rank:         query 的低秩维度, None 表示直接投
        indexer_heads / indexer_head_dim: 交给 LightningIndexer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        latent_dim: int = 64,
        qk_rope_head_dim: int = 16,
        block_size: int = 4,
        window: int = 8,
        sparse: bool = True,
        overlap: bool = True,
        sparse_top_k: int = 8,
        q_lora_rank: Optional[int] = None,
        indexer_heads: int = 2,
        indexer_head_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if not 0 < qk_rope_head_dim < latent_dim:
            raise ValueError(f"qk_rope_head_dim ({qk_rope_head_dim}) 要在 0 和 latent_dim ({latent_dim}) 之间")
        if block_size < 1:
            raise ValueError(f"block_size 必须 >= 1, 当前 {block_size}")

        self.d_model, self.num_heads = d_model, num_heads
        self.c, self.rope_dim = latent_dim, qk_rope_head_dim
        self.m, self.window = block_size, window
        self.sparse, self.overlap = sparse, overlap
        self.sparse_top_k = sparse_top_k
        self.scale = 1.0 / math.sqrt(latent_dim)
        self.dense_warmup = False      # True: 读全部到齐的条目, indexer 只旁听 (对照 V3.2 的预热阶段)
        self.last_index_loss = None

        # query: 可选低秩, 再升到 H 个 c 维的头
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=bias) if q_lora_rank else None
        self.q_up = nn.Linear(q_lora_rank or d_model, num_heads * latent_dim, bias=bias)

        # 压缩器: 每 token 一个 KV 向量加一个压缩打分; CSA 还有第二路
        self.w_kv_a = nn.Linear(d_model, latent_dim, bias=bias)
        self.w_z_a = nn.Linear(d_model, 1, bias=bias)
        self.w_kv_b = nn.Linear(d_model, latent_dim, bias=bias) if overlap else None
        self.w_z_b = nn.Linear(d_model, 1, bias=bias) if overlap else None
        # 块内位置偏置: 初始 0, 所以起步时压缩器就是块内平均池化
        self.block_bias_a = nn.Parameter(torch.zeros(block_size))
        self.block_bias_b = nn.Parameter(torch.zeros(block_size)) if overlap else None

        # 滑窗分支读未压缩 token, K 和 V 各一份 (压缩条目是一个向量兼当两者)
        self.w_win_k = nn.Linear(d_model, latent_dim, bias=bias) if window else None
        self.w_win_v = nn.Linear(d_model, latent_dim, bias=bias) if window else None

        self.q_norm = RMSNorm(latent_dim)
        self.kv_norm = RMSNorm(latent_dim)
        self.w_o = nn.Linear(num_heads * latent_dim, d_model, bias=bias)

        if sparse:
            self.w_iq = nn.Linear(d_model, latent_dim, bias=bias)       # indexer 的 query 投影
            self.indexer = LightningIndexer(
                d_model=latent_dim, num_heads=indexer_heads, head_dim=indexer_head_dim, bias=bias,
            )
        else:
            self.w_iq = self.indexer = None

    # ---- 压缩 ----

    def _src(self, x: torch.Tensor) -> tuple:
        """每 token 的压缩器输入 (C^a, Z^a, C^b, Z^b)。不重叠时后两项是 None。"""
        ca, za = self.w_kv_a(x), self.w_z_a(x).squeeze(-1)              # [B, T, c], [B, T]
        if not self.overlap:
            return ca, za, None, None
        return ca, za, self.w_kv_b(x), self.w_z_b(x).squeeze(-1)

    def compress_entries(self, src: tuple, base: int, next_entry: int):
        """把缓冲区里凑满了的块压成条目。

        src 覆盖绝对位置 [base, base + L), 要求 base 是 m 的整数倍; next_entry 是还没发出的第一条的下标。
        返回 (entries [B, nb_new, c], entry_pos [nb_new], keep_from):
        entry_pos 是每条的到齐位置 (i+1)m − 1, keep_from 是缓冲区该从第几个 token 往后留 ——
        条目 i 的第二路要用块 i−1, 所以最后一整块必须留着。
        """
        ca, za, cb, zb = src
        B, L, c = ca.shape
        m, dev = self.m, ca.device
        nb = L // m
        if nb == 0:
            return ca.new_zeros(B, 0, c), torch.zeros(0, dtype=torch.long, device=dev), 0

        def weighted(cv, zv, blk_bias):
            blocks = cv[:, : nb * m].view(B, nb, m, c)                  # [B, nb, m, c]
            w = F.softmax(zv[:, : nb * m].view(B, nb, m) + blk_bias, dim=-1)
            return (w.unsqueeze(-1) * blocks).sum(dim=2)                # [B, nb, c]

        entries = weighted(ca, za, self.block_bias_a)
        if self.overlap:
            own = weighted(cb, zb, self.block_bias_b)                   # 第 j 项是块 j 自己
            prev = torch.cat([own.new_zeros(B, 1, c), own[:, :-1]], dim=1)   # 右移一格 → 块 j−1
            entries = entries + prev

        first_block = base // m
        # 条目 i 需要块 i−1 也在缓冲区里, 所以 base > 0 时缓冲区的第 0 块只能当 "前一块" 用
        start = max(next_entry - first_block, 0 if base == 0 else 1)
        pos = torch.arange(first_block + start, first_block + nb, device=dev) * m + (m - 1)
        return entries[:, start:], pos, (nb - 1) * m

    # ---- 前向 ----

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None, cache=None):
        """q [B, T, D] → [B, T, D]。只做自注意力, k / v 参数只为统一签名。

        cache 的键: entries / entry_pos (压缩好的条目及其到齐位置)、next_entry、src (还凑不满一块的
        token 的压缩器输入)、src_base (src 第一个 token 的绝对位置)、win_k / win_v (滑窗的未压缩 K/V)。
        """
        x = q
        B, T, _ = x.shape
        dev = x.device
        if position_ids is not None:
            q_pos, past = position_ids, int(position_ids[0])
        else:
            past = int(cache["pos"]) if cache is not None and "pos" in cache else 0
            q_pos = torch.arange(past, past + T, device=dev)

        # 1) 压缩: 缓存里剩下的 token 和本次的 token 一起送进压缩器
        src, base, next_entry = self._src(x), past, past // self.m
        if cache is not None and "src" in cache:
            base, next_entry = int(cache["src_base"]), int(cache["next_entry"])
            src = tuple(None if a is None else torch.cat([a, b], dim=1) for a, b in zip(cache["src"], src))
        new_entries, new_pos, keep_from = self.compress_entries(src, base, next_entry)
        if new_entries.size(1) and rope is not None:                    # 只转最后 rope_dim 段
            head, tail = new_entries[..., : -self.rope_dim], new_entries[..., -self.rope_dim:]
            new_entries = torch.cat([head, rope(tail, position_ids=new_pos)], dim=-1)

        entries, entry_pos = new_entries, new_pos
        if cache is not None:
            if "entries" in cache:
                entries = torch.cat([cache["entries"], entries], dim=1)
                entry_pos = torch.cat([cache["entry_pos"], entry_pos])
            cache["entries"], cache["entry_pos"] = entries, entry_pos
            cache["next_entry"] = next_entry + new_pos.numel()
            cache["src"] = tuple(None if a is None else a[:, keep_from:] for a in src)
            cache["src_base"] = base + keep_from
            cache["pos"] = past + T

        # 2) 条目一个向量兼当 K 和 V; 滑窗分支读未压缩 token, K 和 V 各一份
        ent = self.kv_norm(entries)
        keys, values, key_pos = ent, ent, entry_pos
        if self.window:
            wk, wv = self.w_win_k(x), self.w_win_v(x)
            win_start = past
            if cache is not None:
                if "win_k" in cache:
                    wk = torch.cat([cache["win_k"], wk], dim=1)
                    wv = torch.cat([cache["win_v"], wv], dim=1)
                wk, wv = wk[:, -(self.window + T):], wv[:, -(self.window + T):]
                cache["win_k"], cache["win_v"] = wk, wv
                win_start = past + T - wk.size(1)
            win_pos = torch.arange(win_start, win_start + wk.size(1), device=dev)
            if rope is not None:
                head, tail = wk[..., : -self.rope_dim], wk[..., -self.rope_dim:]
                wk = torch.cat([head, rope(tail, position_ids=win_pos)], dim=-1)
            keys = torch.cat([ent, self.kv_norm(wk)], dim=1)
            values = torch.cat([ent, wv], dim=1)
            key_pos = torch.cat([entry_pos, win_pos])

        # 3) 可见性: 条目要等到齐, 滑窗只看最近 n_win 个
        ne = entries.size(1)
        visible = q_pos[:, None] >= key_pos[None, :]                    # [T, S]
        if self.window:
            near = (q_pos[:, None] - key_pos[None, :]) < self.window
            near[:, :ne] = True                                         # 窗口限制只管滑窗那几列
            visible = visible & near
        visible = visible.unsqueeze(0).expand(B, -1, -1)                # [B, T, S]

        # 4) 稀疏选择只在压缩条目上做, 滑窗分支一直全读
        index_scores = None
        if self.sparse:
            index_scores = self.indexer(self.w_iq(x.detach()), k=entries.detach(), mask=visible[:, :, :ne])
            used = visible[:, :, :ne] if self.dense_warmup else topk_mask(index_scores, visible[:, :, :ne], self.sparse_top_k)
            visible = torch.cat([used, visible[:, :, ne:]], dim=-1)

        # 5) 一次 softmax 读 (选中的压缩条目 + 滑窗 token)
        qh = self.q_norm(self.q_up(self.q_down(x) if self.q_down else x).view(B, T, self.num_heads, self.c))
        if rope is not None:
            head, tail = qh[..., : -self.rope_dim], qh[..., -self.rope_dim:]
            qh = torch.cat([head, rope(tail.transpose(1, 2), position_ids=q_pos).transpose(1, 2)], dim=-1)
        scores = torch.einsum("bthc,bsc->bths", qh, keys) * self.scale   # [B, T, H, S]
        scores = scores.masked_fill(~visible.unsqueeze(2), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bths,bsc->bthc", attn, values).reshape(B, T, self.num_heads * self.c)

        # 6) indexer 的对齐 loss: 让它拟合主注意力在压缩条目上的分布 (与 V3.2 的 DSA 同一套)
        self.last_index_loss = None
        if self.sparse and torch.is_grad_enabled() and ne:
            used = visible[:, :, :ne]
            p = attn[..., :ne].mean(dim=2).detach()                      # [B, T, ne] 老师
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)          # 滑窗分走的那部分不算
            log_q = F.log_softmax(index_scores.masked_fill(~used, float("-inf")), dim=-1)
            log_q = log_q.masked_fill(~used, 0.0)
            self.last_index_loss = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1).mean()
            self.last_index_scores, self.last_attn_target, self.last_used = index_scores.detach(), p, used
        return self.w_o(out)

    def kv_bytes_per_token(self, seq_len: int, rope_bytes: int = 2, other_bytes: int = 1) -> float:
        """每 token 每层的 KV 字节 (代价模型, 不是实测)。

        按原文的混合精度存: RoPE 那一段用 BF16 (2 字节), 其余用 FP8 (1 字节)。压缩条目一个向量兼当
        K 和 V、每 m 个 token 一条; 滑窗条目 K 和 V 各一份, 条数封顶在 n_win、不随长度增长。
        """
        entry = self.rope_dim * rope_bytes + (self.c - self.rope_dim) * other_bytes
        win = min(self.window, seq_len) * 2 * entry
        return (entry * math.ceil(seq_len / self.m) + win) / seq_len
