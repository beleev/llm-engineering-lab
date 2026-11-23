"""
HyperLLaMA — 把 LLaMA 的残差主路换成 n 条流
    HC:  Zhu et al., ByteDance Seed, "Hyper-Connections" (arXiv 2409.19606, ICLR 2025)
    mHC: DeepSeek-AI, "mHC: Manifold-Constrained Hyper-Connections" (arXiv 2512.24880, ICML 2026)
    DeepSeek-V4 (arXiv 2606.19348) 把残差流加宽成 n_hc × d 并用 mHC 的约束

是什么: 零件、注意力、FFN、归一化全部沿用 LLaMA, 只换残差连接。residual 选三种之一:
    standard —— 普通 Pre-LN, 一条残差流, 走继承来的 LLaMA.forward;
    hc       —— n 条流, 宽度连接 H_pre、深度连接 H_post、流间混合 H_res 都不加约束;
    mhc      —— 同样 n 条流, 三个矩阵分别过 σ、2σ、Sinkhorn-Knopp 投影。
    每层的更新是 x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l), 实现见
    layers/core/hyper_connections.py。
解决什么: 看清 "加宽残差流" 这件事对前向、KV cache 和信号增益各自意味着什么。
    H_res 不加约束时, 第 j 层到第 k 层的有效连接是中间各层 H_res 的连乘, 偏离恒等映射后
    信号被整体放大或衰减; 双随机约束让连乘仍是双随机, 最大绝对行和恒为 1。
关键数字: 三种残差在初始化时逐位相同 (n 条流起步数值一致, H_pre 的权重加起来是 1、H_post 全 1、
    H_res 的行和是 1, 所以流的共同值原样传过去), 差别只在训练之后。
    每 token 的激活显存按 n 倍算; KV cache 与残差宽度无关, 因为注意力只看那一个层输入。
本库的做法 (教学约定, 不代表原模型): 继承 LLaMA, weight tying、embedding 乘 √D、
    N(0, 0.02²) 初始化照旧; HC 的 φ 是裸 Parameter, 不被 init_weights 覆盖。
简化: 读出取 n 条流的均值, 论文是求和。两者只差一个常数因子, 被 ln_f 的 RMSNorm 吸收;
      取均值让 "初始化时与普通 LLaMA 逐位相同" 这条断言成立。
简化: 没有 mHC 的 TileLang 算子融合与混合精度, n 条流是显式的 [B, T, n, D] 张量。
简化: 只实现 mHC 式 7 的动态参数化, HC 论文的静态版 (SHC) 等价于把 α 固定为 0, 没有单独开关。
读代码时盯住: forward 里的 streams —— [B, T, n, D], 每层由 hyper[i] 读一次、写回一次,
    注意力和 FFN 只看那一个 [B, T, D] 的输入, 所以 KV cache 的形状一点没变。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.hyper_connections import HyperConnection, ManifoldHyperConnection
from llm_models.models.language_models.llama import LLaMA
from llm_models.utils.generation import KVCache
from llm_models.utils.masks import combine_causal_and_padding_mask

RESIDUALS = ("standard", "hc", "mhc")


class HyperLLaMA(LLaMA):
    """
    idx -> Embed·sqrt(D) -> 复制成 n 条流 -> N × HyperConnection(PreLNBlock) -> 均值 -> RMSNorm -> lm_head

    forward 返回 Tensor; 接受 attention_mask; 支持 KV cache (generate() 来自 GenerationMixin)。

    Args:
        residual:  "standard" | "hc" | "mhc"
        n_streams: 残差流条数 n (论文的 expansion rate); residual="standard" 时忽略
        其余参数与 LLaMA 相同
    """

    def __init__(self, residual: str = "mhc", n_streams: int = 4, **kwargs):
        super().__init__(**kwargs)
        if residual not in RESIDUALS:
            raise ValueError(f"residual 要是 {RESIDUALS} 之一, 当前 {residual!r}")
        self.residual = residual
        self.n_streams = 1 if residual == "standard" else n_streams
        if residual != "standard":
            cls = HyperConnection if residual == "hc" else ManifoldHyperConnection
            self.hyper = nn.ModuleList(
                [cls(self.d_model, n_streams, i) for i in range(len(self.layers))]
            )

    def res_matrices(self) -> list:
        """上一次前向各层用的 H_res, 取 batch 里第 0 条序列的最后一个位置。供统计连乘增益。"""
        if self.residual == "standard":
            return [torch.ones(1, 1)]
        return [h.last_res[0, -1] for h in self.hyper]

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """idx [B, T] -> logits [B, T, V]。参数含义与 LLaMA.forward 相同。"""
        if self.residual == "standard":
            return super().forward(idx, attention_mask, cache, return_hidden)

        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]
        position_ids = torch.arange(past, past + T, device=idx.device)
        causal = self._causal_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        # 一条流复制成 n 条: 起步时数值相同, 之后由各层的 H_post / H_res 逐渐分化
        streams = x.unsqueeze(-2).expand(B, T, self.n_streams, self.d_model)
        for i, layer in enumerate(self.layers):
            streams = self.hyper[i](
                streams, layer, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T

        h = self.ln_f(streams.mean(dim=-2))                              # [B, T, D]
        return h if return_hidden else self.lm_head(h)
