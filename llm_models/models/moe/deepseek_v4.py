"""
DeepSeek-V4 (arXiv 2606.19348, 2026-04) — 把注意力的 "看多少" 从 token 级搬到压缩块级

相对 V3.2 改了三处, 本文件实现前两处:
    DSA → CSA / HCA 交错 : 先把序列按块压短, 再在压缩条目上选 top-k; 另加一条读未压缩 token 的滑动窗口
    残差 → mHC          : 残差流加宽成 n_hc = 4 条, 流间混合矩阵投到双随机 (residual="mhc", 见 hyper_llama 一章)
    优化器 → Muon        : 本库不涉及 (embedding、预测头、mHC 的静态偏置与门控系数、RMSNorm 仍用 AdamW)
MoE、aux-loss-free 负载均衡、Block 的结构全部沿用 V3: 本文件只换注意力, 外加一个可选的残差模式。

层排布 (原文): V4-Pro 前两层用 HCA, 之后 CSA 与 HCA 交错; V4-Flash 前两层用纯滑动窗口注意力, 之后同样交错。
    原文没有写交错的具体比例。本文件的 pattern 字符串按层指定: "C" 是 CSA, "H" 是 HCA。
关键数字: V4-Pro 1.6T 总参数、49B 激活、61 层、128 个 query 头、头维 512, 预训练 33T token;
    V4-Flash 284B / 13B / 43 层, 32T token。两者都支持 1M 上下文。
    1M 上下文下 V4-Pro 的单 token 推理 FLOPs 是 V3.2 的 27%、KV cache 是 10%; V4-Flash 是 10% 和 7%。
本库约定 (不代表原模型): lm_head 与 embedding 共享权重, embedding 乘 √D, 见 models/__init__.py。
简化: 没有实现 V4-Flash 那两层纯滑动窗口注意力 (pattern 只有 "C" 和 "H")。
简化: forward 不接受带 0 的 attention_mask。压缩条目会把 pad token 混进加权和, 真实实现要在压缩器里屏蔽。
简化: 压缩注意力本身的简化清单在 layers/core/compressed_attention.py 的文件头。
读代码时盯住: __init__ 里按 pattern 逐层构造的 attn_kwargs —— CSA 和 HCA 的区别全在那三个开关上。
"""

from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn

from llm_models.layers.core.compressed_attention import CompressedAttention
from llm_models.layers.core.hyper_connections import HyperConnection, ManifoldHyperConnection
from llm_models.models.moe.deepseekV3 import DeepSeekBlock, DeepSeekV3
from llm_models.utils.generation import KVCache
from llm_models.utils.init import init_weights


class DeepSeekV4Block(DeepSeekBlock):
    """V4 Block: 只把 attention 换成压缩注意力, MoE 与残差结构完全继承 V3。

    额外 kwargs 原样传给 CompressedAttention: block_size, window, sparse, overlap, sparse_top_k,
    q_lora_rank, indexer_heads, indexer_head_dim。
    """

    attn_cls = CompressedAttention


class DeepSeekV4(DeepSeekV3):
    """
    idx → Embed·sqrt(D) → N × DeepSeekV4Block (CSA / HCA 交错) → RMSNorm → lm_head (与 embedding 共享权重)

    forward 返回 (logits, all_routing_info); generate() 来自 GenerationMixin。

    Args:
        pattern:      按层写的注意力类型, 例如 "HHCHCH"; None 表示前两层 HCA、之后 CSA 与 HCA 交错
        csa_block / hca_block: CSA 与 HCA 的压缩块大小 m 与 m' (原文 4 和 128)
        window:       滑动窗口里未压缩 token 的个数 n_win (原文 128)
        csa_top_k:    CSA 每个 query 保留的压缩条目数 (原文 V4-Pro 1024、V4-Flash 512)
        latent_dim:   压缩条目的维度 c, 也是每个 query 头的维度 (原文 512)
        residual:     "standard" (普通 Pre-RMSNorm 残差) | "hc" | "mhc" (加宽成 n_streams 条流)
        其余参数与 DeepSeekV3 相同
    """

    block_cls = DeepSeekV4Block

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 6,
        pattern: Optional[str] = None,
        csa_block: int = 4,
        hca_block: int = 16,
        window: int = 8,
        csa_top_k: int = 8,
        latent_dim: int = 64,
        qk_rope_head_dim: int = 16,
        residual: str = "standard",
        n_streams: int = 4,
        d_ff: Optional[int] = None,
        **kwargs,
    ):
        if pattern is None:
            pattern = ("HH" + "CH" * num_layers)[:num_layers]      # 前两层 HCA, 之后交错
        if len(pattern) != num_layers or set(pattern) - {"C", "H"}:
            raise ValueError(f"pattern 要是 {num_layers} 个 C 或 H, 当前 {pattern!r}")

        # 先建一个 0 层的 V3 骨架 (embedding / RoPE / ln_f / lm_head / 因果 mask 都在里面),
        # 再按 pattern 逐层建 block: 每层的注意力只构造一次, 不会白建
        super().__init__(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=0,
            latent_dim=latent_dim, qk_rope_head_dim=qk_rope_head_dim, d_ff=d_ff, **kwargs,
        )
        self.pattern = pattern
        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)                        # 与 V3 的默认值一致
        common = dict(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            latent_dim=latent_dim, qk_rope_head_dim=qk_rope_head_dim, window=window,
            **{k: v for k, v in kwargs.items() if k in
               ("num_routed_experts", "num_shared_experts", "top_k", "dropout", "use_aux_free_bias")},
        )
        self.layers = nn.ModuleList([
            self.block_cls(
                **common,
                **(dict(block_size=csa_block, sparse=True, overlap=True, sparse_top_k=csa_top_k)
                   if kind == "C" else
                   dict(block_size=hca_block, sparse=False, overlap=False)),
            )
            for kind in pattern
        ])

        self.residual = residual
        self.n_streams = 1 if residual == "standard" else n_streams
        if residual != "standard":
            cls = HyperConnection if residual == "hc" else ManifoldHyperConnection
            self.hyper = nn.ModuleList([cls(d_model, n_streams, i) for i in range(num_layers)])

        init_weights(self)        # 骨架那次调用时还没有层, 建完层要再来一次

    def set_dense_warmup(self, flag: bool) -> None:
        """True: CSA 层读全部到齐的条目, indexer 只旁听学习; False: 真正按 top-k 稀疏。"""
        for layer in self.layers:
            if layer.attn.sparse:
                layer.attn.dense_warmup = flag

    def kv_bytes_per_token(self, seq_len: int) -> Dict[str, float]:
        """每 token 的 KV 字节 (代价模型, 不是实测): 分层类型汇总, 按 RoPE 段 BF16、其余 FP8。"""
        per_layer = [l.attn.kv_bytes_per_token(seq_len) for l in self.layers]
        return {
            "total": sum(per_layer),
            "csa": sum(b for b, k in zip(per_layer, self.pattern) if k == "C"),
            "hca": sum(b for b, k in zip(per_layer, self.pattern) if k == "H"),
        }

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # 只接受全 1 (见文件头的简化)
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """idx [B, T] → (logits [B, T, V], all_routing_info)。"""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("压缩条目会把 pad token 混进加权和, 本模型不接受带 0 的 attention_mask")
        if self.residual == "standard":
            return super().forward(idx, cache=cache)

        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")
        position_ids = torch.arange(past, past + T, device=idx.device)
        mask = self._causal_mask(past + T)[:, past:, :]
        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        all_routing_info: List[Dict[str, torch.Tensor]] = []
        streams = x.unsqueeze(-2).expand(B, T, self.n_streams, self.d_model)
        for i, layer in enumerate(self.layers):
            # block 返回 (x, routing_info), HyperConnection 只要前一项; routing_info 顺手收走
            def call(h, _layer=layer, **kw):
                out, info = _layer(h, **kw)
                all_routing_info.append(info)
                return out
            streams = self.hyper[i](
                streams, call, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T
        return self.lm_head(self.ln_f(streams.mean(dim=-2))), all_routing_info
