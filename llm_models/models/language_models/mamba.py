"""
Mamba — 不用 attention 的语言模型 (Gu & Dao, 2023)

是什么: 每层 = Pre-RMSNorm + MambaLayer + 残差; 没有 QKV、没有因果 mask、没有位置编码、没有 FFN。
解决了什么: Transformer 训练 O(T²), 推理时 KV cache 随 T 线性增长;
           Mamba 训练 O(T), 推理每步 O(1) —— 全部历史压在定长状态 (h, conv 窗口) 里。
MambaLayer:  x ─ in_proj ─┬─ main: 因果 depthwise Conv1d → SiLU → SelectiveSSM ─┐
                          └─ gate: SiLU ───────────────────────────────────────⊙─ out_proj
    Conv 提供局部 token 混合 (SSM 前的"短程记忆"), gate 即 GLU 式门控, 顶替了 FFN。
关键数字: d_inner = 2·D, d_state N = 16, d_conv = 4; 每层解码状态 = D_in·N + D_in·(d_conv-1) 个数, 与 T 无关。
与官方实现的差异:
    - SSM 用 Python 循环逐步递推 (layers/sparse/ssm.py), 官方用专用的 CUDA scan kernel。
    - depthwise 卷积用 unfold 手写, 权重仍是 self.conv1d 的, 结果与直接调 conv1d 相同。
    - 不乘 √D、lm_head 与 embedding 共享权重, 这两点沿用官方实现。
读代码时盯住: cache —— {"conv": 最近 d_conv-1 个输入, "h": SSM 状态}; 对照 Transformer 的 KV cache 会越长越大。
左 pad: pad 位置把卷积输入和 SSM 输入置 0 (与 HF 的做法相同)。
    卷积: 真 token 左边本来就补 0, pad 置 0 后看到的窗口一样。
    SSM: h 从 0 出发, 输入为 0 时 h = A_bar·0 + 0 仍是 0, 走到第一个真 token 时状态和不 pad 一样。
    只对左 pad 成立: pad 夹在真 token 中间时, 置 0 挡不住已有状态被 A_bar 衰减。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.sparse.ssm import SelectiveSSM
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights


class MambaLayer(nn.Module):
    """
    conv + selective SSM + gate。d_inner 默认 2·d_model。

    forward: x [B, T, D] -> [B, T, D]。
    cache 是本层自己的 dict, 存两样东西, 大小都与已读长度无关:
        cache["conv"] [B, d_inner, d_conv-1]   卷积还要用的最近几个输入
        cache["h"]    [B, d_inner, N]          SSM 的隐状态 (由 SelectiveSSM 读写)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        d_inner = d_inner or 2 * d_model
        self.d_inner, self.d_conv = d_inner, d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)     # main + gate 一次投出
        # depthwise (groups=d_inner); padding=0, 因果性靠 forward 里手动左填充
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_model=d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cache: Optional[dict] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]。mask: [B, T, 1], 1=真 token 0=左 pad; pad 位置不写进卷积窗口和状态。"""
        # 1) 一次投影, 切成主路和门控路
        x_main, x_gate = self.in_proj(x).chunk(2, dim=-1)              # [B, T, 2·d_inner] → 各 [B, T, d_inner]
        if mask is not None:
            x_main = x_main * mask                                     # pad 的卷积输入 = 0, 和左边补的 0 一样

        # 2) 因果卷积: 左边补上 d_conv-1 个 "过去", 右边不补
        x_main = x_main.transpose(1, 2)                                # [B, d_inner, T] (Conv1d 要通道在前)
        if cache and "conv" in cache:
            x_main = torch.cat([cache["conv"], x_main], dim=2)         # 左边接上一步留下的窗口
        else:
            x_main = F.pad(x_main, (self.d_conv - 1, 0))               # 左填 0 ⇒ 只看过去
        if cache is not None:
            # 留下最后 d_conv-1 个输入给下一步用。要在卷积之前存: 存的是输入, 不是输出
            cache["conv"] = x_main[:, :, x_main.size(2) - (self.d_conv - 1):]  # [B, d_inner, d_conv-1]
        # depthwise conv = 每通道对最近 d_conv 个输入加权求和。等价于 self.conv1d(x_main),
        # 但 CPU 上 grouped conv 是逐组循环 (实测占前向 90% 时间), 所以直接用 unfold 写出来
        # unfold 在时间维上滑窗: [B, d_inner, T+d_conv-1] → [B, d_inner, T, d_conv]
        win = x_main.unfold(2, self.d_conv, 1)
        # weight [d_inner, 1, d_conv]: 每通道一组 d_conv 个系数。与 win 广播相乘,
        # 再对窗口维求和 → [B, d_inner, T]
        x_main = (win * self.conv1d.weight[:, 0, None, :]).sum(-1) + self.conv1d.bias[:, None]
        x_main = F.silu(x_main.transpose(1, 2))                        # [B, T, d_inner]
        if mask is not None:
            x_main = x_main * mask                                     # 卷积 bias 让 pad 处非 0, 再清一次: SSM 输入 0 ⇒ h 不动

        # 3) SSM 走主路, 再被门控路逐元素相乘 (GLU 式门控)
        y = self.ssm(x_main, cache=cache) * F.silu(x_gate)             # [B, T, d_inner]
        return self.out_proj(y)                                        # [B, T, D]


class MambaBlock(nn.Module):
    """x + MambaLayer(RMSNorm(x))。forward: x [B, T, D] -> [B, T, D]; cache 和 mask 原样传给 MambaLayer。"""

    def __init__(self, d_model: int, d_inner: Optional[int] = None, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.layer = MambaLayer(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv)

    def forward(
        self, x: torch.Tensor, cache: Optional[dict] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return x + self.layer(self.norm(x), cache=cache, mask=mask)


class Mamba(GenerationMixin, nn.Module):
    """
    idx -> Embedding -> N × MambaBlock -> RMSNorm -> lm_head (与 embedding 共享权重)

    forward 返回 Tensor; attention_mask 支持左 pad; 支持 generate() (来自 GenerationMixin)。
    """

    max_len = 1 << 30   # 无位置编码, 无上下文上限 (GenerationMixin 用它裁剪前缀)

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        d_inner: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv) for _ in range(num_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight              # weight tying

        init_weights(self)                       # Linear/Embedding -> N(0, 0.02²) ⇒ 初始 CE ≈ ln V
        for blk in self.layers:                  # init_weights 会清零 dt_proj.bias, Δ 的专用初始化要补回来
            blk.layer.ssm.reset_dt()

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T] 1=真 token 0=左 pad
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        idx: [B, T] -> logits [B, T, V], 返回 Tensor。
        cache 里存的是递推状态而不是 K/V, 大小与已读长度无关。
        attention_mask: 只支持左 pad。只取最后 T 列: 更早的位置已经压进 cache 的状态里了。
        """
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            mask = attention_mask[:, -idx.size(1):].to(self.lm_head.weight.dtype).unsqueeze(-1)  # [B, T, 1]
        # 不乘 √D: 沿用官方实现。这里也没有加性位置编码要对齐量级
        x = self.token_embedding(idx)            # [B, T, D]
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache.layers[i] if cache else None, mask=mask)
        if cache is not None:
            cache.pos += idx.size(1)             # 模型自己不用 pos; GenerationMixin 靠它区分 prefill 和 decode
        return self.lm_head(self.ln_f(x))        # [B, T, V]
