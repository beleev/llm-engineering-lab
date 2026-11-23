"""
Byte Latent Transformer (Pagnoni et al., Meta FAIR 2024, "Byte Latent Transformer: Patches Scale Better Than Tokens")

是什么: 不用 tokenizer 的语言模型。输入是原始 UTF-8 字节, 先按 "下一个字节有多难猜" 把字节切成长短不一的
        patch, 大模型 (全局 latent transformer) 每个 patch 只走一步, 两头各有一个小模型负责字节和 patch 之间的转换。
解决什么: BPE 词表是训练前定死的: 同样一句话, 词表没照顾到的语言要多花几倍 token (llm_train m26),
          拼写、数字、罕见字符也被切得很碎; 而且每个 token 不论难易都花同样的计算。
          BLT 按熵切分: 好猜的字节 (词的后半截、多字节汉字的续字节) 合进同一个 patch, 难猜的位置才开新 patch,
          全局模型的步数按信息量分配, 不按字节数分配。
数据流 (一条长 T 的字节序列):
    1. 熵模型 (小的字节级 LM) 算每个位置的下一字节熵 H(x_t) = H(p(x_t | x_<t)); H(x_t) > θ 处开新 patch。
       H(x_t) 只由 x_<t 算出, 所以切分是因果的, 生成时边走边切。
    2. 局部编码器: byte embedding + hash n-gram embedding → 1 个滑窗因果块 → 交叉注意力池化:
       每个 patch 的 query 用本 patch 字节的 max-pool 初始化, 只对本 patch 的字节做注意力, 得到 patch 向量。
    3. 全局 latent transformer: 在 patch 序列上跑因果块。输入右移一个 patch (开头是可学的 BOS patch),
       所以第 j 个输出 g_j 只汇总 patch 0..j-1。
    4. 局部解码器: 预测落在 patch j 的字节时, 字节隐状态加上 g_j, 再过 1 个滑窗因果块, lm_head 输出 256 个字节的 logits。
关键公式:
    边界:          b_t = 1[H(x_t) > θ],  b_0 = 1;   patch_id_t = Σ_{s ≤ t} b_s − 1
    hash n-gram:   e_t = E_byte[x_t] + Σ_n E_n[hash(x_{t-n+1..t}) mod 桶数]      (默认只有 n = 3)
    bits-per-byte: BPB = Σ_t CE_t / (ln 2 · 字节数)      与分词方式无关, 字节模型和 token 模型可以直接比
与论文的差异:
    简化: 熵模型和主模型分开训练; 熵模型是 1 层的小 LLaMA, 论文约 100M 参数、14 层、512 字节滑窗。
    简化: 只实现全局阈值 H(x_t) > θ; 论文另有近似单调约束 H(x_t) − H(x_{t−1}) > θ_r。
    简化: hash n-gram 默认只取 n = 3, 一张 512 行的表; 论文取 n = 3..8、50 万个 hash。
    简化: 编码器只在最后做一次交叉注意力池化; 论文每个编码器层后都接一次交叉注意力。
    简化: 解码器里每个字节只读一个 patch 向量 g_j。交叉注意力只有一个 key 时 softmax 恒为 1,
          等于对 g_j 做一次线性投影, 这里直接写成投影后相加。
    简化: 全局模型和局部模型同宽 (d_model 相同); 论文的全局模型更宽, 两边用线性层衔接。
    简化: 局部块用很窄的滑窗 (默认 2 字节), 加上 3-gram, 局部视野约 5 字节; 更远的信息只能经全局模型传过去,
          玩具规模下才看得出 patch 切在哪里的影响。
本库约定: Pre-LN、lm_head 与 byte embedding 共享权重、embedding 乘 √D、N(0, 0.02²) 初始化 → 初始 CE ≈ ln 256。
读代码时盯住: forward 里的 `tgt_pid` —— 位置 t 预测 x_{t+1}, 读的是 x_{t+1} 所在 patch 的 g,
              它只汇总更早的 patch, 不会把本 patch 里还没出现的字节泄露进来。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.models.language_models.llama import LLaMA, _make_llama_block
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, build_sliding_window_mask

NUM_BYTES = 256


@torch.no_grad()
def next_byte_entropy(entropy_model: LLaMA, x: torch.Tensor) -> torch.Tensor:
    """
    x [B, T] 字节 → H [B, T], H[:, t] = p(x_t | x_<t) 的熵 (nats)。
    位置 0 没有前文, 记 +inf (恒为边界)。
    """
    entropy_model.eval()
    logp = F.log_softmax(entropy_model(x), dim=-1)          # [B, T, 256], 第 t 行预测 x_{t+1}
    h = -(logp.exp() * logp).sum(-1)                         # [B, T]
    inf = torch.full_like(h[:, :1], float("inf"))
    return torch.cat([inf, h[:, :-1]], dim=1)                # 右移一格: H[:, t+1] 来自第 t 行的预测


def entropy_patch_ids(entropy: torch.Tensor, threshold: float) -> torch.Tensor:
    """H [B, T] → patch_id [B, T] (long)。H > threshold 处开新 patch, 位置 0 恒为边界。"""
    boundary = entropy > threshold
    boundary[:, 0] = True
    return boundary.long().cumsum(1) - 1


def fixed_patch_ids(batch_size: int, seq_len: int, size: int) -> torch.Tensor:
    """固定长度切分 (MegaByte 式): 每 size 个字节一个 patch → [B, T]。"""
    return (torch.arange(seq_len) // size).expand(batch_size, seq_len)


def hash_ngrams(x: torch.Tensor, n: int, buckets: int) -> torch.Tensor:
    """x [B, T] → [B, T]: 以位置 t 结尾的 n 个字节的多项式 hash, 取模到 buckets 个桶。左边不足 n 个时补 0 字节。"""
    padded = F.pad(x, (n - 1, 0))                            # [B, T + n - 1]
    h = torch.zeros_like(x)
    for i in range(n):                                       # 逐字节滚动: h = h·257 + byte, 对大素数取模防溢出
        h = (h * 257 + padded[:, i:i + x.size(1)]) % 2147483647
    return h % buckets


class ByteLatentLM(nn.Module):
    """
    x [B, T] 字节 + patch_ids [B, T] → logits [B, T, 256], 第 t 行预测 x_{t+1}。

    patch_ids 由调用方给 (entropy_patch_ids 或 fixed_patch_ids): 切分规则和模型分开, 同一个模型可以换切分方式对照。
    forward 之后 self.last_num_patches [B] 记录每条序列的 patch 数, 也就是全局模型这次走了几步。

    Args:
        d_model / n_heads: 局部和全局共用
        global_layers:     全局 latent transformer 的层数 (算力的大头)
        local_window:      局部编码器 / 解码器的滑窗宽度 (字节)
        ngrams / buckets:  hash n-gram embedding 的 n 和每张表的桶数
        max_len:           最大字节数 (也是最多的 patch 数)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        global_layers: int = 2,
        local_window: int = 2,
        ngrams=(3,),
        buckets: int = 512,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model, self.max_len, self.local_window = d_model, max_len, local_window
        self.ngrams, self.buckets = tuple(ngrams), buckets

        self.byte_embedding = nn.Embedding(NUM_BYTES, d_model)
        self.hash_embeddings = nn.ModuleList([nn.Embedding(buckets, d_model) for _ in self.ngrams])
        self.rope = RotaryPositionalEncoding(d_model // n_heads, max_len)   # 字节位置和 patch 位置共用一张表

        d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64
        block = lambda: _make_llama_block(d_model, n_heads, None, d_ff, 0.0)  # noqa: E731
        self.encoder = block()                                               # 局部编码器: 1 层滑窗块
        self.pool_norm_q, self.pool_norm_kv = RMSNorm(d_model), RMSNorm(d_model)
        self.pool_attn = GroupedQueryAttention(d_model, n_heads)             # patch query 只看本 patch 的字节
        self.bos_patch = nn.Parameter(torch.zeros(d_model))                  # 全局输入右移后补在开头
        self.global_layers = nn.ModuleList([block() for _ in range(global_layers)])
        self.global_norm = RMSNorm(d_model)
        self.patch_to_byte = nn.Linear(d_model, d_model, bias=False)         # 简化: 单 key 交叉注意力 = 线性投影
        self.decoder = block()                                               # 局部解码器: 1 层滑窗块
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, NUM_BYTES, bias=False)
        self.lm_head.weight = self.byte_embedding.weight                     # weight tying
        init_weights(self)                                                   # bos_patch 是 Parameter 不是 Embedding, 保持 0
        self.last_num_patches: Optional[torch.Tensor] = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """byte embedding + 各 n 的 hash n-gram embedding, 整体乘 √D (本库约定)。x [B, T] → [B, T, D]"""
        e = self.byte_embedding(x)
        for n, table in zip(self.ngrams, self.hash_embeddings):
            e = e + table(hash_ngrams(x, n, self.buckets))
        return e * math.sqrt(self.d_model)

    def forward(self, x: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")
        local_mask = build_sliding_window_mask(T, self.local_window, x.device)   # [1, T, T]

        # ---- 1. 局部编码器: 字节隐状态 h [B, T, D] ----
        h = self.encoder(self.embed(x), mask=local_mask, rope=self.rope)

        # ---- 2. 交叉注意力池化: 每个 patch 一个向量 [B, P, D] ----
        P = int(patch_ids.max()) + 1                                         # batch 内最多的 patch 数, 不足的补在末尾
        self.last_num_patches = patch_ids[:, -1] + 1                         # [B] 每条序列的 patch 数
        member = patch_ids[:, None, :] == torch.arange(P, device=x.device)[None, :, None]   # [B, P, T]
        query = h[:, None].masked_fill(~member[..., None], float("-inf")).amax(2)         # [B, P, D] 本 patch 字节 max-pool
        query = torch.nan_to_num(query, neginf=0.0)                          # 补位 patch 没有字节, 置 0
        # 补位 patch 整行都是 False 会让 softmax 出 NaN: 让它看全部字节, 反正结果不会被读到
        pool_mask = member | ~member.any(-1, keepdim=True)
        patches = query + self.pool_attn(
            q=self.pool_norm_q(query), k=self.pool_norm_kv(h), v=self.pool_norm_kv(h), mask=pool_mask,
        )

        # ---- 3. 全局 latent transformer: 右移一个 patch, g_j 只汇总 patch < j ----
        g = torch.cat([self.bos_patch.expand(B, 1, -1), patches[:, :-1]], dim=1)          # [B, P, D]
        causal = build_causal_mask(P, x.device)                              # [1, P, P]
        for layer in self.global_layers:
            g = layer(g, mask=causal, rope=self.rope)
        g = self.global_norm(g)

        # ---- 4. 局部解码器: 位置 t 预测 x_{t+1}, 读 x_{t+1} 所在 patch 的 g ----
        # 最后一个位置没有 x_{T}, 沿用自己的 patch (只读更早 patch 的汇总, 合法; 训练 loss 不用这一行)
        tgt_pid = torch.cat([patch_ids[:, 1:], patch_ids[:, -1:]], dim=1)   # [B, T]  ★ 不泄露未来的关键
        g_per_byte = g.gather(1, tgt_pid[..., None].expand(-1, -1, self.d_model))         # [B, T, D]
        d = self.decoder(h + self.patch_to_byte(g_per_byte), mask=local_mask, rope=self.rope)
        return self.lm_head(self.ln_f(d))                                    # [B, T, 256]


def bits_per_byte(logits: torch.Tensor, x: torch.Tensor) -> float:
    """logits [B, T, 256] (第 t 行预测 x_{t+1}) 与字节 x [B, T] → 平均每个被预测字节的比特数。"""
    ce = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())  # nats / 字节
    return ce.item() / math.log(2)
