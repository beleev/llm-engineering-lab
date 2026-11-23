"""
条件深度 — Mixture-of-Depths (Raposo et al., Google DeepMind 2024, arXiv 2404.02258)
         与 Mixture-of-Recursions (Bae et al., KAIST / Google DeepMind / Mila 2025, arXiv 2507.10524)

是什么: MoE 在宽度上稀疏 (每个 token 只走 E 个专家里的 k 个); 这两种做法在深度上稀疏 (每个 token 只走一部分层)。
    MoD: 被路由的 block 前有一个路由器, 给每个 token 打一个标量分数 r_i; 整段序列里分数前 k = C·T 名的 token
         过这一层 (注意力和 FFN 都只在它们之间算), 其余 token 直接走残差。过层的 token 输出
             x_i + w(r_i) · (block(x)_i − x_i)
         路由权重乘在输出上, 路由器才在梯度路径上、能被训练。论文最好的配置是隔层路由、容量 C = 12.5%。
         代价: top-k 在整段序列上取, 位置 i 是否入选取决于后面 token 的分数, 采样时还没有后面的 token。
         论文的两种解法: 在路由器输出上加 BCE 辅助损失; 或者另训一个小 MLP 预测器 (输入 stop-gradient,
         不影响主模型) 判断 "这个 token 会不会进 top-k", 采样时用它代替 top-k。本文件实现后一种。
    MoR: 不堆 N 个不同的层, 而是把一个共享的递归块反复用 N_r 次 (Middle-Cycle: 首尾各一个独立层, 中间共享);
         每次递归前路由器决定哪些 token 继续 (expert-choice: 第 r 次只在仍在递归的 token 里取分数前 k_r 名),
         所以每个 token 的递归深度不同, 参数量却与 N_r 基本无关。
         KV cache: 第 r 次递归只有留下的 token 参与注意力, 也只有它们存 K/V (recursion-wise caching)。
关键数字:
    MoD 每 token 的稠密 FLOPs = 不路由的层 + C × 路由的层, 随容量 C 线性下降 (注意力那部分按 C² 下降)。
    MoR 容量日程 k_r = T · (N_r − r) / N_r (r = 0..N_r−1) 时, 缓存的 K/V 条数是每次递归都全过的 (N_r + 1) / (2N_r)。
本库的做法 (教学约定, 不代表原模型): 两个类都继承 LLaMA, 零件、weight tying、embedding 乘 √D、N(0, 0.02²) 初始化照旧。
简化: 路由权重用 w = 2·sigmoid(r) 而不是论文的原始 r。分数为 0 时权重恰为 1, 起步时接近普通 LLaMA, 容量 100% 时可以逐位对照。
简化: MoD 只实现辅助预测器这一种因果解法, 没有路由器上的 BCE 辅助损失。
简化: MoR 的递归块只有 1 层, 只实现 expert-choice 路由和 recursion-wise caching 的计数;
      token-choice 路由 (一开始就给每个 token 定深度) 和 recursive KV sharing (后面的递归复用第一次的 K/V) 没实现。
简化: 没有带 KV cache 的 generate()。跳过某层的 token 在那一层没有 K/V, 生成时要按层分别管理缓存。
读代码时盯住: routed_block 里的 order (入选 token 的原始位置, 同时用作 gather 下标和 RoPE 位置) 和 valid (补位标记)。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.models.language_models.llama import LLaMA
from llm_models.utils.init import init_weights


def topk_select(scores: torch.Tensor, k: int) -> torch.Tensor:
    """scores [B, T] → bool [B, T], 每行分数前 k 名为 True。k 在整段序列上取, 所以不满足因果。"""
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter(1, idx, True)


def routed_block(block: nn.Module, x: torch.Tensor, sel: torch.Tensor, weight: torch.Tensor, rope: nn.Module) -> torch.Tensor:
    """
    只让 sel 为 True 的 token 过 block, 其余 token 原样留在残差上。

    x [B, T, D]; sel [B, T] bool; weight [B, T] 路由权重。每行入选个数可以不同, 不足的用补位填齐到 k = 最大个数。
    入选 token 之间仍按原始位置做因果注意力, RoPE 也用原始位置。
    """
    B, T, D = x.shape
    counts = sel.sum(1)                                              # [B] 每行入选个数
    k = int(counts.max())
    if k == 0:
        return x
    pos = torch.arange(T, device=x.device).expand(B, T)
    # 入选的排在前面且保持原顺序: 入选位置的排序键是 pos, 落选的是 pos + T
    order = torch.where(sel, pos, pos + T).argsort(dim=1)[:, :k]    # [B, k] 原始位置
    valid = torch.arange(k, device=x.device)[None, :] < counts[:, None]   # [B, k] False = 补位
    idx = order.unsqueeze(-1).expand(-1, -1, D)
    xs = x.gather(1, idx)                                            # [B, k, D] ★ 张量只有 k 行, 计算量随 k 走
    eye = torch.eye(k, dtype=torch.bool, device=x.device)
    # order 已按位置排好, 下三角就是原始位置上的因果; 补位的 key 屏蔽, 补位的 query 至少看自己, 防止整行 -inf
    mask = eye.logical_or(valid[:, None, :]) & torch.ones(k, k, dtype=torch.bool, device=x.device).tril()
    out = block(xs, mask=mask, rope=rope, position_ids=order)       # [B, k, D]
    w = weight.gather(1, order).unsqueeze(-1)                        # [B, k, 1]
    new = torch.where(valid.unsqueeze(-1), xs + w * (out - xs), xs)  # 补位写回原值, 等于没动
    return x.scatter(1, idx, new)


def _dense_params(block: nn.Module) -> int:
    """block 里矩阵参数的个数 (不含归一化的向量); 每个 token 过一次约 2× 这么多 FLOPs。"""
    return sum(p.numel() for p in block.parameters() if p.dim() == 2)


class MoDLLaMA(LLaMA):
    """
    LLaMA + Mixture-of-Depths: 第 route_every−1, 2·route_every−1, ... 层 (隔层时是 1, 3, 5, ...) 前各有一个路由器。

    forward(idx, routing="topk" | "predictor") → logits [B, T, V]。
        "topk":      训练用。每个路由层取整段序列的前 k = round(capacity·T) 名; 同时算出预测器的 BCE (self.predictor_loss)。
        "predictor": 采样用。每个 token 由预测器单独判断是否过层, 只依赖前文。
    每次前向后 self.last_sel[层号] 是该层的入选掩码 [B, T], self.last_counts[层号] 是每行过层的 token 数。
    """

    def __init__(self, capacity: float = 0.125, route_every: int = 2, predictor_hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        D = self.d_model
        self.capacity = capacity
        self.routed = [i for i in range(len(self.layers)) if i % route_every == route_every - 1]
        self.routers = nn.ModuleDict({str(i): nn.Linear(D, 1, bias=False) for i in self.routed})
        self.predictors = nn.ModuleDict({
            str(i): nn.Sequential(nn.Linear(D, predictor_hidden), nn.SiLU(), nn.Linear(predictor_hidden, 1))
            for i in self.routed
        })
        init_weights(self.routers)
        init_weights(self.predictors)
        self.last_sel: Dict[int, torch.Tensor] = {}
        self.last_counts: Dict[int, torch.Tensor] = {}
        self.predictor_loss = torch.zeros(())

    def forward(self, idx: torch.Tensor, routing: str = "topk") -> torch.Tensor:
        assert routing in ("topk", "predictor")
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")
        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]
        position_ids = torch.arange(T, device=idx.device)
        mask = self._causal_mask(T)
        aux = []
        for i, layer in enumerate(self.layers):
            if i not in self.routed:
                x = layer(x, mask=mask, rope=self.rope, position_ids=position_ids)
                self.last_counts[i] = torch.full((B,), T, device=idx.device)
                continue
            r = self.routers[str(i)](x).squeeze(-1)                       # [B, T] 路由分数
            pred = self.predictors[str(i)](x.detach()).squeeze(-1)        # [B, T] ★ stop-gradient: 预测器不改主模型
            if routing == "topk":
                sel = topk_select(r, max(1, round(self.capacity * T)))
                aux.append(F.binary_cross_entropy_with_logits(pred, sel.float()))
            else:
                sel = pred > 0                                            # 只看本位置的 x, 而 x 只依赖前文
            x = routed_block(layer, x, sel, 2 * torch.sigmoid(r), self.rope)
            self.last_sel[i], self.last_counts[i] = sel, sel.sum(1)
        self.predictor_loss = torch.stack(aux).mean() if aux else torch.zeros((), device=idx.device)
        return self.lm_head(self.ln_f(x))

    def flops_per_token(self) -> Dict[str, float]:
        """按上一次前向里每层实际过层的 token 数, 估算每个 token 平均的 FLOPs (乘加算 2 次)。"""
        T = int(max(c.max() for c in self.last_counts.values()))         # 不路由的层过的是全部 T 个
        dense = attn = 0.0
        for i, layer in enumerate(self.layers):
            n = self.last_counts[i].float()                               # [B]
            dense += (2 * _dense_params(layer) * n).mean().item()
            attn += (2 * 2 * self.d_model * n * (n + 1) / 2).mean().item()   # QK^T 和 AV, 因果: 第 j 个看 j 个
            if i in self.routed:
                dense += 2 * self.d_model * T                             # 路由器: 每个 token 一次点积
        dense += 2 * self.d_model * self.vocab_size * T                   # lm_head
        return {"dense": dense / T, "attn": attn / T, "total": (dense + attn) / T}


class MoRLLaMA(LLaMA):
    """
    Mixture-of-Recursions (Middle-Cycle, expert-choice): layers[0] 独立, layers[1] 是共享递归块, layers[2] 独立。

    第 r 次递归 (r = 0..N_r−1) 只在仍在递归的 token 里取分数前 k_r = round(T·(N_r − r)/N_r) 名过共享块。
    第 0 次 k_0 = T, 所以每个 token 至少递归 1 次; 一个 token 落选后不再回来 (层级过滤)。
    每次前向后 self.depth [B, T] 是每个 token 的递归次数, self.kv_entries [B] 是共享块里缓存的 K/V 条数。
    """

    def __init__(self, num_recursions: int = 3, **kwargs):
        kwargs["num_layers"] = 3
        super().__init__(**kwargs)
        self.num_recursions = num_recursions
        self.routers = nn.ModuleList(nn.Linear(self.d_model, 1, bias=False) for _ in range(num_recursions))
        init_weights(self.routers)
        self.depth = torch.zeros(0)
        self.kv_entries = torch.zeros(0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")
        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        position_ids = torch.arange(T, device=idx.device)
        mask = self._causal_mask(T)
        first, shared, last = self.layers
        x = first(x, mask=mask, rope=self.rope, position_ids=position_ids)
        active = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        depth = torch.zeros(B, T, dtype=torch.long, device=idx.device)
        N = self.num_recursions
        for r in range(N):
            score = self.routers[r](x).squeeze(-1)                        # [B, T]
            k = max(1, round(T * (N - r) / N))
            sel = topk_select(score.masked_fill(~active, float("-inf")), k)  # 只在仍在递归的 token 里选
            x = routed_block(shared, x, sel, 2 * torch.sigmoid(score), self.rope)  # ★ 同一个 shared, 参数共享
            active, depth = sel, depth + sel.long()
        x = last(x, mask=mask, rope=self.rope, position_ids=position_ids)
        self.depth, self.kv_entries = depth, depth.sum(1)                 # 每次过共享块存一条 K/V
        return self.lm_head(self.ln_f(x))

    def flops_per_token(self) -> Dict[str, float]:
        """按上一次前向的递归深度估算每 token 平均 FLOPs, 口径同 MoDLLaMA.flops_per_token。"""
        B, T = self.depth.shape
        first, shared, last = self.layers
        D = self.d_model
        # 第 r 次递归过共享块的 token 数 = 深度 > r 的 token 数
        ks = [(self.depth > r).sum(1).float() for r in range(self.num_recursions)]   # 每个 [B]
        dense = 2 * (_dense_params(first) + _dense_params(last)) * T
        dense += sum((2 * _dense_params(shared) * k).mean().item() for k in ks)
        dense += 2 * D * T * self.num_recursions + 2 * D * self.vocab_size * T       # 路由器 + lm_head
        attn = 2 * (2 * 2 * D * T * (T + 1) / 2)                                     # 首尾两层
        attn += sum((2 * 2 * D * k * (k + 1) / 2).mean().item() for k in ks)
        return {"dense": dense / T, "attn": attn / T, "total": (dense + attn) / T}
