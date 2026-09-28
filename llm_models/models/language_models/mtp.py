"""
MTP — Multi-Token Prediction (多 token 预测) 模块

论文出处:
    "Better & Faster Large Language Models via Multi-token Prediction"
        (Gloeckle et al., Meta, 2024 — 并行多 head 版)
    DeepSeek-V3 Technical Report (2024) — 串行因果链版 (本实现)

动机:
    标准 LM 的每个位置只预测 t+1 一个 token。MTP 让位置 t 同时预测
    t+1, t+2, ..., t+1+K。这样做有三个作用:
      1. **训练信号更密** — 同一条数据提供 (K+1) 份监督, 数据效率更高
      2. **表征被迫"向前规划"** — hidden state 必须编码更远期的信息,
         缓解 next-token 短视 (teacher forcing 只看一步)
      3. **推理免费拿草稿** — MTP head 对 t+2 的预测可直接作为投机解码
         (speculative decoding) 的 draft; DeepSeek-V3 报告草稿接受率 85%+,
         解码加速约 1.8x (对应 llm_infer/m07_speculative_decoding)

DeepSeek-V3 的串行式 MTP (与 Gloeckle 的并行独立 head 不同, 保留完整因果链):

    主干:    h_i^0 = Backbone(t_<=i)                       → head → 预测 t_{i+1}
    MTP-1:   h_i^1 = Block_1( W_1 [RMSNorm(h_i^0) ; RMSNorm(Emb(t_{i+1}))] )
                                                           → head → 预测 t_{i+2}
    MTP-k:   同构地堆叠, 每深一级多看一个真实 token (teacher forcing)

    要点: Embedding 与 lm_head 与主干**共享**, 每个 MTP 模块只新增
          一个拼接投影 W_k 和一个 Transformer Block, 参数开销极小。

损失:
    L = L_main + λ · mean_k( L_mtp_k )       DeepSeek-V3: λ = 0.3 (前期) / 0.1

推理:
    部署时可以直接丢弃 MTP 模块 (零成本), 或保留用作投机解码 draft head。

本库的做法 (教学约定):
    - MTP 接在 LLaMA 主干 (GQA + SwiGLU) 上, 不是 DeepSeek-V3 的 MLA + MoE 主干。
    - lm_head 与 embedding 共享权重、embedding 乘 √D, 见 models/__init__.py。
    - 投机解码那条路径没有实现: 带 cache 生成时只跑主干。
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class MTPModule(nn.Module):
    """
    单个 MTP 级联模块 (DeepSeek-V3 式)。

    输入上一级的 hidden h^{k-1} 与"下一个真实 token"的 embedding, 输出本级
    hidden h^k (用共享 lm_head 解码即得 t_{i+1+k} 的预测)。

        h^k = Block( W [RMSNorm(h^{k-1}) ; RMSNorm(emb_next)] )

    为什么拼接前要各自 RMSNorm:
        h 与 embedding 的数值尺度不同 (h 经过了多层残差累加), 直接拼接会让
        投影矩阵 W 先花容量学"对齐尺度"; 各自归一化后拼接更稳。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_kv_heads: Optional[int],
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_hidden = RMSNorm(d_model)
        self.norm_emb = RMSNorm(d_model)
        # 拼接 [h; emb] (2d) 投影回 d, 是 MTP 模块唯一的"新零件"
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = PreLNBlock(
            d_model=d_model,
            attn=GroupedQueryAttention(
                d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
            ),
            ffn=SwiGLUFeedForward(d_model, d_ff),
            norm_cls=RMSNorm,
            dropout=dropout,
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        h_prev: torch.Tensor,        # [B, T, D] 上一级 hidden
        emb_next: torch.Tensor,      # [B, T, D] 真实 next-token 的 embedding
        mask: torch.Tensor,          # [B 或 1, T, T] 因果 ∧ padding, 与主干同一张
        rope: RotaryPositionalEncoding,
    ) -> torch.Tensor:               # [B, T, D] 本级 hidden (还没过 final_norm)
        # 两路各自归一化后在最后一维拼接: 2 × [B, T, D] → [B, T, 2D]
        x = torch.cat([self.norm_hidden(h_prev), self.norm_emb(emb_next)], dim=-1)
        x = self.proj(x)                      # [B, T, 2D] -> [B, T, D]
        # 不传 position_ids: 位置按 0..T-1 算。MTP 只在无 cache 的整段前向里跑, 这样是对的
        return self.block(x, mask=mask, rope=rope)


class MTPLLaMA(GenerationMixin, nn.Module):
    """
    LLaMA 主干 + K 级串行 MTP 模块 (教学版)。

    forward 返回 dict:
        {
          "logits":     [B, T, V]          主 head (预测 t+1)
          "mtp_logits": List[[B, T, V]]    第 k 项预测 t+1+k
        }

    接受 attention_mask; 支持 KV cache (generate() 来自 GenerationMixin, 它只取 "logits" 一项)。

    Args:
        vocab_size / d_model / n_heads / num_kv_heads / num_layers / max_len /
        d_ff / dropout: 同 LLaMA。
        mtp_depth: MTP 级联深度 K (DeepSeek-V3 取 1)。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        mtp_depth: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mtp_depth < 1:
            raise ValueError(f"mtp_depth 至少为 1, 当前 {mtp_depth}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.mtp_depth = mtp_depth

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64      # 向上对齐到 64 的倍数, 同 LLaMA

        # ---- 主干: 与 LLaMA 相同的 N 层 stack ----
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=GroupedQueryAttention(
                        d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
                    ),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)

        # ---- 共享输出头: 主干与所有 MTP 模块共用 (也与 embedding 绑定) ----
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # ---- K 个串行 MTP 模块 ----
        self.mtp_modules = nn.ModuleList(
            [
                MTPModule(d_model, n_heads, num_kv_heads, d_ff, dropout)
                for _ in range(mtp_depth)
            ]
        )

        causal = build_causal_mask(max_len, torch.device("cpu"))   # [1, max_len, max_len]
        self.register_buffer("causal_mask", causal, persistent=False)

        init_weights(self)   # weight tying 之后; 初始 CE ≈ ln V

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """取左上角 [1, seq_len, seq_len] 的下三角 mask; 超过缓存大小就现建一张。"""
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [B, T] -> [B, T, D]。主干输入和 MTP 的 emb_next 共用这一个函数 (·√D 是本库约定)。"""
        return self.token_embedding(idx) * math.sqrt(self.d_model)

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T]
        cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        """
        idx [B, T] -> 返回 dict, 两个键:
            "logits":     [B, T, V]            主 head, 位置 i 预测 t_{i+1}
            "mtp_logits": list, 长度 mtp_depth  第 k 项 [B, T, V], 位置 i 预测 t_{i+1+k}
        attention_mask: [B, past+T], 1=有效 0=pad。
        给了 cache (= 生成) 时只跑主干, mtp_logits 为空: MTP 模块要拼接 "未来的真实 token",
        普通自回归解码时它们还不存在 (用作投机解码 draft 是另一条路径)。
        """
        B, T = idx.shape
        past = cache.pos if cache is not None else 0                     # 已缓存的 token 数
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        position_ids = torch.arange(past, past + T, device=idx.device)   # [T] 新 token 的绝对位置
        # 行 past: 是新 token (query), 列 :past+T 是全部历史 (key)
        causal = self._causal_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)   # [B 或 1, T, past+T]

        # ---- 主干前向 ----
        h = self._embed(idx)                                             # [B, T, D]
        for i, layer in enumerate(self.layers):
            h = layer(
                h, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        # h 本身不过 ln_f 就交给 MTP 模块: 每个模块入口有自己的 norm_hidden
        logits_main = self.lm_head(self.ln_f(h))                         # [B, T, V]
        if cache is not None:
            cache.pos += T
            return {"logits": logits_main, "mtp_logits": []}

        # ---- MTP 级联: 第 k 级在位置 i 处拼接真实 token t_{i+k} 的 embedding ----
        mtp_logits: List[torch.Tensor] = []
        for k, module in enumerate(self.mtp_modules, start=1):
            # teacher forcing: idx 左移 k 位; 末尾 k 个位置没有未来 token,
            # 用 0 占位 (这些位置的预测会在 MTPLoss 里被 -100 屏蔽)
            shifted = torch.zeros_like(idx)
            shifted[:, :-k] = idx[:, k:]                                 # 位置 i 放 t_{i+k}
            emb_next = self._embed(shifted)                              # [B, T, D]

            # h 被覆盖: 第 k 级的输出就是第 k+1 级的 h_prev (串行级联)
            h = module(h, emb_next, mask=mask, rope=self.rope)
            mtp_logits.append(self.lm_head(module.final_norm(h)))        # [B, T, V]

        return {"logits": logits_main, "mtp_logits": mtp_logits}


def __getattr__(name: str):
    # MTPLoss 定义在 training/loss.py, 和其它 loss 放在一起。
    # 这里做惰性转发: `from llm_models.models.language_models.mtp import MTPLoss` 可用,
    # 而 training 只在真正取 MTPLoss 时才被 import。
    # llada.py 走另一种写法: 文件顶部直接 import training.loss 的基类。
    # 两种都不会循环 import, 因为 training 不 import models。
    if name == "MTPLoss":
        from llm_models.training.loss import MTPLoss
        return MTPLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
