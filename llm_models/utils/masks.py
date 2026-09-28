"""
注意力掩码 — "谁能看见谁" 的全部规则都在这里

约定: bool 张量, True = 可见, False = 屏蔽。attention 在 softmax 之前把 False 处的分数填 -inf
      (不能直接把权重置 0: softmax 要对所有位置归一化, 必须先压到 -inf 才会得到真正的 0)。
三类: padding mask (屏蔽 pad key)  |  因果 mask (下三角, 自回归不看未来)
      |  滑动窗口 mask (带状下三角, Mistral/Gemma/GPT-OSS; 可选保留开头 sink token)
坑:   某个 query 行全为 False (左 padding 时的 pad 行) → softmax 一整行 -inf → NaN, 并经残差污染整个 batch。
      两个 combine_* 函数都经过 `_unmask_empty_rows` 兜底: 让这种行只看自己。
KV cache 解码时 mask 不是方阵: 行 = 新 query [past:past+T], 列 = 全部 key [:past+T]。
读代码时盯住: 形状 —— pad [B, 1, S], causal [1, T, S], 广播 AND 后 [B, T, S]。
"""

import torch
from typing import Optional


def get_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding 掩码: seq [B, T] (token id) → [B, 1, T], True = 真 token, False = pad。

    为什么要屏蔽 pad: 不同长度的句子要用 pad 补齐才能拼成 batch。pad 没有语义, 不屏蔽的话
    - pad 会和真实 token 互相影响;
    - softmax 分母里多出 pad 项, 权重分布失真。

    中间的维度 1 留给 query 维: 之后要和因果掩码 [1, T, T] / 注意力分数 [B, H, T, T] 广播 AND。

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        >>> mask = get_pad_mask(seq, pad_idx=0)
        >>> # mask: [[True, True, True, False, False], [True, True, False, False, False]]
    """
    return (seq != pad_idx).unsqueeze(1)   # [B, T] → [B, 1, T]


def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    因果掩码 (Subsequent Mask / Look-ahead Mask): seq [B, T] → [1, T, T] 下三角。

    用于 Decoder / GPT 类自回归模型的自注意力:
    位置 t 只能看到 0..t (包含自己), 看不到 t+1..T-1 (未来)。
    不加的话, 模型预测位置 t 的下一个 token 时能直接看到答案, 训练学不到东西。

    True = 可见 (下三角 + 对角线), False = 屏蔽 (上三角)。
    seq 只用来取长度和 device。最前面的维度 1 和 batch 维广播, 整个 batch 共用一份。

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 4]])
        >>> mask = get_subsequent_mask(seq)
        >>> # mask 形状: [1, 4, 4]
        >>> # [[True,  False, False, False],   # 位置 0 只能看 0
        >>> #  [True,  True,  False, False],   # 位置 1 能看 0,1
        >>> #  [True,  True,  True,  False],   # 位置 2 能看 0,1,2
        >>> #  [True,  True,  True,  True ]]   # 位置 3 能看 0,1,2,3
    """
    sz = seq.size(1)
    return build_causal_mask(sz, seq.device)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    因果掩码, 返回 [1, seq_len, seq_len] 下三角, True = 可见。

    和 get_subsequent_mask 做同一件事, 只是收 seq_len + device 而不是 token 张量。
    手里只有 embedding、没有 token id 时用它 (例如多模态模型的输入是图像 / 音频特征拼出来的)。
    device 要和输入张量一致, 避免 CPU / GPU 之间隐式拷贝。
    """
    # triu(diagonal=1) 取严格上三角 (不含对角线), 也就是 "未来位置" 为 1
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)   # [T, T]
    return (mask == 0).unsqueeze(0)   # == 0 取反得到下三角 bool; [T, T] → [1, T, T]


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    sink_tokens: int = 0,
) -> torch.Tensor:
    """
    滑动窗口 (带状) 因果掩码 — Sliding Window Attention (SWA)。
    返回 [1, seq_len, seq_len], True = 可见, False = 屏蔽。

    全因果掩码下, 位置 t 能看到 [0, t] 共 t+1 个位置: KV cache 随序列长度线性增长,
    注意力计算平方增长。SWA 只留最近 W 个位置:

        可见(t, s) = (s <= t) 且 (s > t - W)

    注意力矩阵从 "下三角" 变成 "带状下三角":
      - 单层感受野限制在 W 内, 但信息可以跨层接力: L 层的理论感受野 ≈ L·W
        (Mistral-7B: 32 层 × 4096 窗口 ≈ 131K)
      - 推理时 KV cache 只需保留最近 W 个位置 (rolling buffer 环形覆写),
        显存从 O(T) 封顶到 O(W)

    sink_tokens > 0 时, 开头 S 个位置永远可见 (StreamingLLM, 2023):
      - softmax 必须把注意力分给 "某些位置", 训练后的模型习惯把多余的注意力
        倒在开头几个 token 上 (attention sink)。
      - 窗口滑过去把它们逐出 cache, 输出分布会崩坏。
      - 保留 4 个 sink 就能在无限流式输入下保持质量。
      - GPT-OSS (2025) 把 sink 做成了每个 head 一个可学习的 logit。

    Args:
        window_size: 窗口大小 W (>= 1); W >= seq_len 时退化为全因果掩码
        sink_tokens: 额外永远可见的开头位置数 S (默认 0 = 纯 SWA)

    Example:
        >>> build_sliding_window_mask(5, window_size=2, device="cpu")[0].int()
        tensor([[1, 0, 0, 0, 0],     # 位置 0 只看自己
                [1, 1, 0, 0, 0],     # 位置 1 看 {0, 1}
                [0, 1, 1, 0, 0],     # 位置 2 看 {1, 2} — 0 滑出窗口
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1]])
    """
    if window_size < 1:
        raise ValueError(f"window_size 至少为 1, 当前 {window_size}")

    i = torch.arange(seq_len, device=device).unsqueeze(1)  # query 位置 [T, 1]
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # key 位置   [1, T]

    # 带状下三角: 因果 (j <= i) 且 在窗口内 (j > i - W)。[T, 1] 与 [1, T] 广播成 [T, T]
    visible = (j <= i) & (j > i - window_size)

    if sink_tokens > 0:
        # 开头 S 个位置对所有"未来"位置永远可见 (仍需满足因果性 j <= i)
        visible = visible | ((j < sink_tokens) & (j <= i))

    return visible.unsqueeze(0)   # [T, T] → [1, T, T]


def _unmask_empty_rows(mask: torch.Tensor) -> torch.Tensor:
    """
    整行全 False 的 query (典型: 左 padding 时的 pad 位置, 因果 ∩ padding 后一个 key 都不剩)
    会让 softmax 对一整行 -inf 归一化 → NaN, 并经残差污染整个 batch 的梯度。
    修法: 让这种行只看自己 (对角线)。它的输出本来就没人用 (pad 是被屏蔽的 key, loss 也忽略它),
    所以不改变任何真实位置的结果, 只是把 NaN 换成有限值。
    """
    T, S = mask.shape[-2:]
    empty = ~mask.any(dim=-1, keepdim=True)                                   # [..., T, 1] 这一行是否全 False
    # "自己" 在非方阵 (KV cache: S = past + T) 里是偏移 S-T 的对角线:
    # 第 t 个 query 的绝对位置是 past + t = (S - T) + t。tril 和 triu 取同一条对角线, 交出来只剩它
    diag = torch.ones(T, S, dtype=torch.bool, device=mask.device).tril(S - T).triu(S - T)   # [T, S]
    return mask | (empty & diag)


def combine_causal_and_padding_mask(
    causal_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    因果掩码 AND padding 掩码: 两个都为 True 才可见。

    用在自回归模型 (GPT / Decoder): 既不能看未来 token, 又要忽略 batch 里的 pad。

    causal_mask:    [1, T, S] 下三角因果掩码 (无 cache 时 S = T)
    attention_mask: [B, S], True / 1 = 真 token; None 表示 batch 里没有 pad (推理或定长输入)
    返回 [B, T, S]; attention_mask 为 None 时原样返回 causal_mask [1, T, S]。
    保证没有全 False 的行 (见 _unmask_empty_rows), 左 padding 也不会产生 NaN。
    """
    if attention_mask is None:
        return causal_mask

    attention_mask = attention_mask.bool()
    # 只屏蔽 key 侧; pad query 的输出本来就会在 loss 里被忽略 (label=-100)。
    # 列数取自 causal_mask: KV cache 解码时它是 [1, T, past+T], attention_mask 相应为 [B, past+T]
    key_mask = attention_mask.unsqueeze(1)             # [B, 1, S]
    return _unmask_empty_rows(key_mask & causal_mask)  # [B, T, S]


def combine_masks(pad_mask: torch.Tensor, subsequent_mask: torch.Tensor) -> torch.Tensor:
    """
    Padding 掩码 AND 因果掩码, 给 Decoder 用: 同时屏蔽 pad 和未来位置。

      pad_mask:        [B, 1, T]   ─┐
                                    ├─ 广播 + 按位 AND → [B, T, T], True = 可见
      subsequent_mask: [1, T, T]   ─┘

    和 combine_causal_and_padding_mask 的差别只在入参: 这里收已经 unsqueeze 好的 pad_mask。

    Example:
        >>> tgt_pad_mask = get_pad_mask(tgt, pad_idx=0)
        >>> tgt_subsequent_mask = get_subsequent_mask(tgt)
        >>> tgt_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)
    """
    return _unmask_empty_rows(pad_mask & subsequent_mask)
