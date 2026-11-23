#!/usr/bin/env python
"""
DeepSeek-V4 (CSA / HCA) 训练示例 — indexer 在压缩块上选对了没有

任务是植入针的检索: 序列里有 4 对 (KEY_j, v_j) 散落在随机填充里, 末尾依次给出 4 个 KEY_j,
要模型把对应的 v_j 说出来。只有这 4 个答案位算 loss, 其余位置 labels 置 -100。
这样一来 "indexer 该选哪个压缩块" 有了唯一正确答案: v_j 所在的那一块。

CSA 的 top-k 不可导, indexer 拿不到 LM loss 的梯度, 和 V3.2 的 DSA 一样要靠对齐 loss 单独训。
按 V3.2 的顺序分三段:
    A. 稠密预热: dense_warmup=True, CSA 读全部到齐的条目, 只训主模型
    B. indexer 预热: 冻结主模型, 只用 index_loss 训 indexer → KL 下降, 命中针所在块的比例上升
    C. 稀疏训练: dense_warmup=False, CSA 真的只读 top-k 个条目; total = LM + aux + λ·index_loss

召回口径: 对末尾那 4 个 KEY 位置, 看 indexer 的 top-k 里有没有 v_j 所在的那一块。
    随机乱选的期望是 top-k / 可见块数。CSA 的块是重叠的, v_j 也出现在它后一块的第二路里,
    这里只统计它自己那一块, 所以是偏严的口径。
每步抽新数据, 评估在固定的 held-out 批上, 所以召回率是学到的能力, 不是背下了一个 batch。
"""

import math

import torch

from llm_models.models.moe.deepseek_v4 import DeepSeekV4
from llm_models.training import MoELMLoss

V, T, M, TOPK, NEEDLES = 64, 64, 4, 4, 4
KEY0, VAL_MAX, FILL0 = 50, 10, 10          # KEY_j = 50 + j; 值取 0..9; 填充取 10..49
STEPS_A, STEPS_B, STEPS_C = 250, 250, 150
CFG = dict(vocab_size=V, d_model=64, n_heads=4, num_layers=4, pattern="CCCC", max_len=128,
           num_routed_experts=4, num_shared_experts=1, top_k=2, dropout=0.0,
           latent_dim=32, qk_rope_head_dim=16, csa_block=M, hca_block=16,
           window=4, csa_top_k=TOPK, indexer_heads=2, indexer_head_dim=16)
QUERY = [T - 2 * NEEDLES + 2 * j for j in range(NEEDLES)]      # 末尾 KEY_j 所在的位置


def make_batch(batch_size: int, generator: torch.Generator = None):
    """返回 (idx [B, T], labels [B, T], needle_block [B, NEEDLES])。

    填充取 10..49, 前半段每隔 12 个位置放一对 (KEY_j, v_j), 末尾放 4 个 KEY_j 和答案。
    labels 只在 4 个答案位上有值, 其余是 -100。
    """
    g = generator
    idx = torch.randint(FILL0, KEY0, (batch_size, T), generator=g)
    labels = torch.full((batch_size, T), -100)
    vals = torch.randint(0, VAL_MAX, (batch_size, NEEDLES), generator=g)
    offset = torch.randint(0, 8, (batch_size, NEEDLES), generator=g)
    block = torch.zeros(batch_size, NEEDLES, dtype=torch.long)
    for j in range(NEEDLES):
        p = 2 + 12 * j + offset[:, j]                           # KEY_j 的位置, 落在前半段
        rows = torch.arange(batch_size)
        idx[rows, p] = KEY0 + j
        idx[rows, p + 1] = vals[:, j]
        block[:, j] = (p + 1) // M                              # v_j 所在的压缩块
        idx[:, QUERY[j]] = KEY0 + j                             # 末尾再问一次
        idx[rows, QUERY[j] + 1] = vals[:, j]
        labels[rows, QUERY[j]] = vals[:, j]                     # 第 QUERY[j] 行预测下一个 token
    return idx, labels, block


HELD_IDX, HELD_LABELS, HELD_BLOCK = make_batch(64, torch.Generator().manual_seed(1))


def evaluate(model):
    """held-out 上的 (答案位准确率, 命中针自己那一块的比例, 命中它或它后一块的比例)。

    前向要带梯度才会留下 indexer 的打分。CSA 的块重叠, 针所在的 token 既进自己那一块的第一路,
    也进后一块的第二路, 所以第三个数才是机制上完整的口径, 它的随机基线也更高。
    """
    model.eval()
    logits, _ = model(HELD_IDX)
    model.train()
    rows = torch.arange(HELD_IDX.size(0))
    acc = sum((logits[rows, q].argmax(-1) == HELD_LABELS[rows, q]).float().mean().item() for q in QUERY) / NEEDLES
    own, pair = [], []
    for layer in model.layers:
        if not layer.attn.sparse:
            continue
        top = layer.attn.last_index_scores.topk(TOPK, dim=-1).indices       # [B, T, k]
        for j, q in enumerate(QUERY):
            b = HELD_BLOCK[:, j:j + 1]
            own.append((top[rows, q] == b).any(-1).float().mean().item())
            pair.append(((top[rows, q] == b) | (top[rows, q] == b + 1)).any(-1).float().mean().item())
    return acc, sum(own) / len(own), sum(pair) / len(pair)


def main():
    torch.manual_seed(0)
    model = DeepSeekV4(**CFG)
    n_blocks = T // M
    base_own = TOPK / n_blocks
    base_pair = 1 - (n_blocks - TOPK) * (n_blocks - TOPK - 1) / (n_blocks * (n_blocks - 1))
    print(f"层排布 {model.pattern} (这个玩具四层都用 CSA, 好把检索逼到压缩块上; 原文是 CSA 与 HCA 交错); "
          f"压缩比 {M} → {n_blocks} 块, top-k {TOPK}, 滑窗 {CFG['window']}")
    print(f"随机乱选的基线: 命中针自己那一块 {base_own:.3f}, 命中它或它后一块 {base_pair:.3f}")

    idx_params = [p for n, p in model.named_parameters() if "indexer" in n or "w_iq" in n]
    main_params = [p for n, p in model.named_parameters() if not ("indexer" in n or "w_iq" in n)]

    # ---- 0) 只有 LM loss 时, indexer 拿不到梯度 ----
    out = MoELMLoss(index_loss_weight=0.0).compute(model(HELD_IDX), HELD_LABELS)
    grads = torch.autograd.grad(out["lm_loss"], idx_params, allow_unused=True)
    assert all(g is None for g in grads), "top-k 不可导, LM loss 不应给 indexer 梯度"
    lm0 = out["lm_loss"].item()
    acc0, own0, pair0 = evaluate(model)
    print(f"初始 lm_loss {lm0:.4f} (ln V = {math.log(V):.4f}) | LM loss 对 indexer 的梯度全为 None"
          f" | 答案准确率 {acc0:.3f} | 召回 {own0:.3f} / {pair0:.3f}")
    assert abs(lm0 - math.log(V)) < 0.5, "初始 lm_loss 应 ≈ ln V"

    def step(loss_fn, opt):
        i, l, _ = make_batch(16)
        out = loss_fn.compute(model(i), l)
        model.zero_grad()
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return {k: v.item() for k, v in out.items()}

    # ---- A) 稠密预热: CSA 读全部到齐的条目, 只训主模型 ----
    model.set_dense_warmup(True)
    opt = torch.optim.AdamW(main_params, lr=3e-3)
    for _ in range(STEPS_A):
        m = step(MoELMLoss(aux_loss_weight=0.01, index_loss_weight=0.0), opt)
    acc_a, own_a, pair_a = evaluate(model)
    print(f"[A 稠密预热 {STEPS_A} 步]    lm_loss {lm0:.4f} → {m['lm_loss']:.4f} | 答案准确率 {acc0:.3f} → {acc_a:.3f}"
          f" | 召回 {own_a:.3f} / {pair_a:.3f} (indexer 还没训, 仍是随机水平)")
    assert m["lm_loss"] < lm0 - 0.3, "稠密预热后 lm_loss 应明显下降"

    # ---- B) indexer 预热: 冻结主模型, 只用对齐 KL 训 indexer ----
    opt = torch.optim.AdamW(idx_params, lr=3e-3)
    only_index = MoELMLoss(aux_loss_weight=0.0, index_loss_weight=1.0)
    for s in range(STEPS_B):
        i, l, _ = make_batch(16)
        out = only_index.compute(model(i), l)
        model.zero_grad()
        out["index_loss"].backward()                            # 输入和条目都 detach → 只落在 indexer 上
        if s == 0:
            kl0 = out["index_loss"].item()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in idx_params), \
                "index_loss 应给 indexer 的每个参数非零梯度"
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in main_params), \
                "主模型不应从 index_loss 收到梯度"
        opt.step()
    acc_b, own_b, pair_b = evaluate(model)
    print(f"[B indexer 预热 {STEPS_B} 步] KL {kl0:.4f} → {out['index_loss'].item():.4f}"
          f" | 答案准确率 {acc_b:.3f} | 召回 {own_a:.3f} → {own_b:.3f} / {pair_a:.3f} → {pair_b:.3f}")
    assert out["index_loss"].item() < 0.9 * kl0, "indexer 对齐 KL 未下降"
    assert own_b > base_own, "预热后 indexer 命中针自己那一块的比例应高于随机基线"
    assert pair_b > base_pair, "预热后 indexer 命中针所在的两块之一的比例应高于随机基线"

    # ---- C) 稀疏训练: CSA 真的只读 top-k 个条目 ----
    model.set_dense_warmup(False)
    acc_sparse, _, _ = evaluate(model)
    print(f"切到稀疏 (只读 top-{TOPK} 个条目, 不再训练): 答案准确率 {acc_b:.3f} → {acc_sparse:.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    joint = MoELMLoss(aux_loss_weight=0.01, index_loss_weight=1.0)
    for s in range(STEPS_C):
        m = step(joint, opt)
        if s == 0:
            m0 = m
    acc_c, own_c, pair_c = evaluate(model)
    print(f"[C 稀疏训练 {STEPS_C} 步]    lm_loss {m0['lm_loss']:.4f} → {m['lm_loss']:.4f}"
          f" | index_loss {m0['index_loss']:.4f} → {m['index_loss']:.4f} | aux {m['aux_loss']:.3f}"
          f" | 答案准确率 {acc_c:.3f} | 召回 {own_c:.3f} / {pair_c:.3f}")
    assert own_c > base_own, "稀疏训练后 indexer 命中针自己那一块的比例应高于随机基线"
    assert m["lm_loss"] < m0["lm_loss"], "稀疏训练后 lm_loss 应继续下降"


if __name__ == "__main__":
    main()
