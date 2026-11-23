"""
BD3-LM — 块扩散语言模型 (Arriola et al., ICLR 2025, "Block Diffusion: Interpolating Between
Autoregressive and Diffusion Language Models", arXiv 2503.09573)

是什么: 把序列切成长度为 block_size 的块, 块与块之间自回归, 块内部用 LLaDA 那样的掩码扩散。
解决了什么: LLaDA 的注意力是全双向的, 任何位置被填上都会改变所有位置的 K/V, 所以没有 KV cache,
           每个去噪步都要整段重算, 生成长度也必须事先给定。块扩散让注意力 "块间因果、块内双向":
           已经定稿的块不再受后面影响, 它们的 K/V 可以缓存; 生成一块接一块, 可以一直往后接。
           block_size = 1 时退化成自回归, block_size = 序列长时退化成 LLaDA。
关键公式:
    p(x) = Π_b p(x^b | x^{<b}),  每个条件分布是一个掩码扩散模型
    loss = Σ_b E_t [ (1/t) · Σ_{i ∈ 块 b 且被遮} −log p(x_i | x_t^b, x^{<b}) ] / L       (每块各抽一个 t)
    训练时一次前向算完所有块: 输入 x_t ⊕ x (加噪序列接干净序列, 长 2L), 掩码分四块:
        [ M_BD   M_OBC ]     M_BD : 加噪块只看本块的加噪 token (块对角)
        [  0     M_BC  ]     M_OBC: 加噪块看之前各块的干净 token (错一块的块因果)
                             M_BC : 干净序列自己块因果, 给后面的块提供 "已定稿的前文"
    裁剪日程: 训练时遮蔽率 t ~ U[β, ω] 而不是 U(0, 1), 1/t 权重不会因 t 很小而爆, loss 估计的方差变小。
本库的做法 (教学约定, 不代表原模型):
    - 零件全部继承 LLaDA (GQA + SwiGLU + RMSNorm + RoPE, [MASK] 是词表最后一个 id, weight tying, embedding 乘 √D)。
    - 块内采样沿用 LLaDA.sample 的低置信度重遮和线性日程。
简化: 论文的 β, ω 是训练中按验证集上 loss 估计的方差网格搜索出来的, 这里固定取一组。
简化: 论文用 DiT 式骨干和自己的注意力 kernel; 这里直接用 LLaMA 零件和一张显式的 bool 掩码。
读代码时盯住: training_mask 的四块, 以及 sample() 里 "草稿块用 cache 的拷贝, 定稿块才写进真 cache"。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from llm_models.models.language_models.llada import LLaDA, LLaDALoss


def block_causal_mask(seq_len: int, block_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """块因果掩码 [1, T, T], True = 可见: 位置 i 能看到所在块及之前所有块。
    block_size = 1 时就是下三角因果掩码; block_size >= T 时全为 True, 即 LLaDA 的全双向。"""
    blk = torch.arange(seq_len, device=device) // block_size   # [T] 每个位置属于第几块
    return (blk[None, :] <= blk[:, None]).unsqueeze(0)         # 列 (key) 的块号 <= 行 (query) 的块号


def training_mask(seq_len: int, block_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """x_t ⊕ x 的训练掩码 [1, 2L, 2L]。前 L 个位置是加噪序列, 后 L 个是干净序列。"""
    blk = torch.arange(seq_len, device=device) // block_size
    q, k = blk[:, None], blk[None, :]
    m_bd = q == k     # 加噪块内部互相看 (块内双向)
    m_obc = k < q     # 加噪块看之前各块的干净 token, 看不到本块的干净答案
    m_bc = k <= q     # 干净序列块因果, 和采样时缓存里的前文算法相同
    top = torch.cat([m_bd, m_obc], dim=1)                      # 加噪 query 的一行: [加噪 key | 干净 key]
    bottom = torch.cat([torch.zeros_like(m_bc), m_bc], dim=1)  # 干净 query 不看加噪序列
    return torch.cat([top, bottom], dim=0).unsqueeze(0)


def block_forward_process(
    x: torch.Tensor,
    mask_id: int,
    block_size: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
    eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    按块加噪: 每个块抽一个遮蔽率 t ~ U[max(t_min, eps), t_max], 块内每个 token 独立以概率 t 换成 [MASK]。

    x [B, L] 干净 token, L 是 block_size 的整数倍。
    Returns: noisy [B, L], masked [B, L] bool, t [B, L // block_size]
    """
    B, L = x.shape
    n = B * (L // block_size)                                  # 全 batch 的块数
    # 分层采样, 与 llada.forward_process 相同: n 个点等间隔铺满 [0, 1), 再线性映射到 [lo, t_max)
    u = (torch.rand(1, generator=generator, device=x.device) + torch.arange(n, device=x.device) / n) % 1
    lo = max(t_min, eps)
    t = (lo + (t_max - lo) * u).view(B, L // block_size)       # [B, nb]
    t_tok = t.repeat_interleave(block_size, dim=1)             # [B, L] 块内所有位置共用本块的 t
    masked = torch.rand(B, L, generator=generator, device=x.device) < t_tok
    return x.masked_fill(masked, mask_id), masked, t


class BlockDiffusionLM(LLaDA):
    """
    BD3-LM (教学版): LLaDA 的全部零件, 加一个 block_size 和一张块因果掩码。

    forward(idx, mask, position_ids, caches) 返回 logits [B, T, V]:
        mask 为 None 时全双向 (等于 LLaDA); caches 是每层一个 dict 的 list, 给了就把本次的 K/V 追加进去。
    elbo_loss(x) 是训练目标; sample(x, steps_per_block) 逐块生成。
    """

    def __init__(self, block_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        mask: Optional[torch.Tensor] = None,            # 可广播到 [B, T, S], True = 可见
        position_ids: Optional[torch.Tensor] = None,    # [T]; None = 0..T-1
        caches: Optional[List[dict]] = None,            # 每层一个 dict, 存 RoPE 之后的 K/V
    ) -> torch.Tensor:
        pos_max = idx.size(1) if position_ids is None else int(position_ids.max()) + 1
        if pos_max > self.max_len:
            raise ValueError(f"位置 {pos_max} 超过 max_len={self.max_len}")
        x = self.token_embedding(idx) * self.d_model ** 0.5     # [B, T, D]; ·√D 是本库约定
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, rope=self.rope, position_ids=position_ids,
                      cache=None if caches is None else caches[i])
        return self.lm_head(self.ln_f(x))                       # [B, T, V]

    def noisy_logits(self, noisy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """训练前向: 输入 noisy ⊕ x (长 2L), 用 training_mask 一次算完, 只返回加噪那一半的 logits [B, L, V]。"""
        L = x.size(1)
        pos = torch.arange(L, device=x.device).repeat(2)        # 加噪 token 和干净 token 共用位置 0..L-1
        full = torch.cat([noisy, x], dim=1)                     # [B, 2L]
        return self(full, mask=training_mask(L, self.block_size, x.device), position_ids=pos)[:, :L]

    def elbo_loss(
        self, x: torch.Tensor, t_min: float = 0.0, t_max: float = 1.0, generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """x [B, L] 干净序列 → 标量 loss。t 的范围 [t_min, t_max] 就是裁剪日程。"""
        B, L = x.shape
        bs = self.block_size
        noisy, masked, t = block_forward_process(x, self.mask_id, bs, t_min, t_max, generator=generator)
        logits = self.noisy_logits(noisy, x)                    # [B, L, V]
        # 把每个块当成一条 "序列" 交给 LLaDALoss: 只在被遮位置算 CE, 乘 1/t, 除以总 token 数 B·L
        nb = L // bs
        return LLaDALoss().compute(
            logits.reshape(B * nb, bs, -1), x.reshape(B * nb, bs),
            masked=masked.reshape(B * nb, bs), t=t.reshape(-1),
        )["total_loss"]

    def _denoise_step(self, blk, logits, n_gen, s, steps):
        """块内一步去噪, 与 LLaDA.sample 的 low_confidence 分支相同。blk / logits 只含当前块。"""
        is_masked = blk == self.mask_id                         # [B, bs]
        logits = logits.clone()
        logits[..., self.mask_id] = float("-inf")               # 永远不生成 [MASK] 本身
        probs = F.softmax(logits, dim=-1)
        x0 = probs.argmax(-1)                                   # [B, bs]
        conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1).masked_fill(~is_masked, float("inf"))
        blk = torch.where(is_masked, x0, blk)                   # 先全部填上
        n_masked = torch.round(n_gen * (1 - s / steps)).long()  # 本步结束后应剩的 [MASK] 数
        rank = conf.argsort(1).argsort(1)                       # 置信度升序名次
        return blk.masked_fill(rank < n_masked[:, None], self.mask_id)

    @torch.inference_mode()
    def sample(
        self, x: torch.Tensor, steps_per_block: int, use_cache: bool = True, return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        逐块生成: 填满 x 里所有 mask_id 的位置。x [B, T], T 是 block_size 的整数倍, 已知 token 原样保留。

        use_cache=True: 前面已定稿的块只算一次 K/V 并缓存; 每个去噪步只把当前块的 block_size 个 token 送进前向。
        use_cache=False: 每个去噪步把 "前文 + 当前块" 整段重算一遍 (块因果掩码), 结果应与带 cache 的相同。
        return_stats=True 时多返回 {"tokens": 送进前向的 token 总数, "calls": 串行前向次数}。
        """
        self.eval()
        x = x.clone()
        B, T = x.shape
        bs = self.block_size
        caches = [dict() for _ in self.layers]
        n_tok = n_calls = 0
        for b0 in range(0, T, bs):
            blk = x[:, b0:b0 + bs].clone()
            pos = torch.arange(b0, b0 + bs, device=x.device)
            n_gen = (blk == self.mask_id).sum(1)                # [B] 本块要生成的个数
            for s in range(1, steps_per_block + 1):
                if not (blk == self.mask_id).any():
                    break
                if use_cache:
                    # ★ 草稿块的 K/V 每步都在变, 不能进真 cache: 传每层 dict 的浅拷贝, 追加只发生在拷贝里
                    logits = self(blk, position_ids=pos, caches=[dict(c) for c in caches])
                    n_tok, n_calls = n_tok + bs, n_calls + 1
                else:
                    ctx = torch.cat([x[:, :b0], blk], dim=1)
                    logits = self(ctx, mask=block_causal_mask(b0 + bs, bs, x.device))[:, b0:]
                    n_tok, n_calls = n_tok + b0 + bs, n_calls + 1
                blk = self._denoise_step(blk, logits, n_gen, s, steps_per_block)
            x[:, b0:b0 + bs] = blk
            if use_cache:
                self(blk, position_ids=pos, caches=caches)      # 定稿块干净地过一遍, K/V 追加进真 cache
                n_tok, n_calls = n_tok + bs, n_calls + 1
        return (x, {"tokens": n_tok, "calls": n_calls}) if return_stats else x
