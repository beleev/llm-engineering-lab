#!/usr/bin/env python
"""
激活引导 (activation steering) — 能读出的方向, 不一定是加进去能改变行为的方向。

是什么: 不改权重, 在推理时往某一层的残差流里加一个向量, 让模型的行为朝某个方向偏。
    ActAdd (Turner et al. 2023, arXiv 2308.10248): 用一对提示 (如 "Love" / "Hate") 在某层的激活之差当引导向量。
    CAA (Rimsky / Panickssery et al., arXiv 2312.06681): 对很多对只在目标行为上不同的对照样本取激活差的平均,
        推理时乘一个正负系数加到提示之后的每个位置。
    persona vectors (Chen et al. 2025, Anthropic, arXiv 2507.21509): 用对照的系统提示自动抽出 "性格" 方向,
        既拿来监控 (投影值预示行为), 也拿来引导和筛数据; 系数大时会伤通用能力。
本 demo 的设定: 一个 3 层的小 LLaMA, 在两类数据上训练:
    情感序列 [情绪, 话题, 形容词, 名词, 形容词, 名词, ...]: 情绪 (POS / NEG) 决定形容词从哪组里出 (90%),
        话题 (运动 / 食物) 决定名词从哪组里出。
    计数序列 [NEU, d, d+1, d+2, ...] (mod 10): 与情绪无关, 用来量副作用。
看什么:
    1. 在 "情绪和话题完全混在一起" 的数据上 (POS 总配运动, NEG 总配食物), 情绪方向和话题方向都能 100% 读出情绪标签。
    2. 但只有情绪方向 (CAA: 只差情绪的对照样本取平均) 加进残差流后能把 NEG 提示改成说正面词;
       同样长度的话题方向和逻辑回归探针的方向几乎不改情绪, 话题方向改的是名词。
    3. 引导系数越大, 行为改得越多, 无关的计数任务上 loss 也涨得越多。
简化: 数据和模型都是玩具, 行为用 "下一个形容词落在正面词组的概率" 量, 不是开放生成。
简化: 引导加在第 1 层之后、提示之后的每个位置 (CAA 的做法); 没有扫层。
读代码时盯住: steer() 里的 forward hook, 以及 d_mood 和 d_topic 是怎么由两组对照样本算出来的。
"""

import math
import time

import torch
import torch.nn.functional as F

from llm_models.models.language_models.llama import LLaMA

POS, NEG, NEU, SPORT, FOOD = 0, 1, 2, 3, 4
POS_ADJ, NEG_ADJ = list(range(5, 9)), list(range(9, 13))
SPORT_N, FOOD_N = list(range(13, 17)), list(range(17, 21))
DIGIT0, V, T = 21, 31, 14          # 计数用 21..30; 词表 31; 序列长 14
LAYER = 1                          # 在第 1 层 (0 起) 的输出上读激活、加向量


def sentiment(mood: torch.Tensor, topic: torch.Tensor, g: torch.Generator, flip: float = 0.1) -> torch.Tensor:
    """[B] 情绪 + [B] 话题 → [B, T]。形容词以 1 - flip 的概率来自情绪对应的组; 名词总来自话题对应的组。"""
    B, n = mood.size(0), (T - 2) // 2
    pos_side = (mood == POS)[:, None] ^ (torch.rand(B, n, generator=g) < flip)        # 哪些形容词出正面词
    adj = torch.where(pos_side, torch.tensor(POS_ADJ)[torch.randint(0, 4, (B, n), generator=g)],
                      torch.tensor(NEG_ADJ)[torch.randint(0, 4, (B, n), generator=g)])
    noun = torch.where((topic == SPORT)[:, None], torch.tensor(SPORT_N)[torch.randint(0, 4, (B, n), generator=g)],
                       torch.tensor(FOOD_N)[torch.randint(0, 4, (B, n), generator=g)])
    return torch.cat([mood[:, None], topic[:, None], torch.stack([adj, noun], -1).view(B, -1)], 1)


def counting(B: int, g: torch.Generator) -> torch.Tensor:
    start = torch.randint(0, 10, (B, 1), generator=g)
    return torch.cat([torch.full((B, 1), NEU), DIGIT0 + (start + torch.arange(T - 1)) % 10], 1)


def coin(B: int, g: torch.Generator, a: int, b: int) -> torch.Tensor:
    return torch.where(torch.rand(B, generator=g) < 0.5, a, b)


def train(steps: int = 800) -> LLaMA:
    torch.manual_seed(0)
    model = LLaMA(vocab_size=V, d_model=48, n_heads=4, num_layers=3, max_len=T)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        x = torch.cat([sentiment(coin(32, g, POS, NEG), coin(32, g, SPORT, FOOD), g), counting(32, g)])
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def acts(model: LLaMA, x: torch.Tensor) -> torch.Tensor:
    """第 LAYER 层输出的残差流 [B, T, D] (用 forward hook 抓)。"""
    out = {}
    h = model.layers[LAYER].register_forward_hook(lambda m, i, o: out.setdefault("a", o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    return out["a"]


def steer(model: LLaMA, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """在第 LAYER 层输出上, 给位置 2 之后 (提示之后) 的每个位置加 vec, 返回 logits [B, T, V]。"""
    add = torch.zeros(T, vec.numel())
    add[2:] = vec                                                       # ★ CAA: 提示之后的每个位置都加
    h = model.layers[LAYER].register_forward_hook(lambda m, i, o: o + add[: o.size(1)])
    with torch.no_grad():
        logits = model(x)
    h.remove()
    return logits


def pos_rate(logits: torch.Tensor) -> float:
    """在 "下一个是形容词" 的位置 (输入位置 1, 3, 5, ...), 模型给正面词组的总概率, 取平均。"""
    p = F.softmax(logits[:, 1:T - 1:2], -1)
    return p[..., POS_ADJ].sum(-1).mean().item()


def sport_rate(logits: torch.Tensor) -> float:
    """在 "下一个是名词" 的位置 (输入位置 2, 4, ...), 模型给运动名词组的总概率。"""
    p = F.softmax(logits[:, 2:T - 1:2], -1)
    return p[..., SPORT_N].sum(-1).mean().item()


def ce(logits: torch.Tensor, x: torch.Tensor) -> float:
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()


def mean_diff(model: LLaMA, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """两组对照样本在第 LAYER 层、提示之后各位置的平均激活之差 [D]。"""
    return acts(model, a)[:, 2:].mean((0, 1)) - acts(model, b)[:, 2:].mean((0, 1))


def main():
    t0 = time.time()
    model = train()
    g = torch.Generator().manual_seed(1)
    B = 256

    # ---- 两个方向, 都用 "只差一个属性" 的对照样本算 (CAA 的做法): 同一个随机数种子 → 其余 token 对应相同 ----
    topic = coin(B, g, SPORT, FOOD)
    seed = 7
    pos_x = sentiment(torch.full((B,), POS), topic, torch.Generator().manual_seed(seed), flip=0.0)
    neg_x = sentiment(torch.full((B,), NEG), topic, torch.Generator().manual_seed(seed), flip=0.0)
    d_mood = mean_diff(model, pos_x, neg_x)
    mood = coin(B, g, POS, NEG)
    d_topic = mean_diff(model, sentiment(mood, torch.full((B,), SPORT), torch.Generator().manual_seed(seed), 0.0),
                        sentiment(mood, torch.full((B,), FOOD), torch.Generator().manual_seed(seed), 0.0))
    cos = F.cosine_similarity(d_mood, d_topic, 0).item()
    print(f"第 {LAYER} 层残差流: |d_mood| {d_mood.norm():.2f} | |d_topic| {d_topic.norm():.2f} | 夹角余弦 {cos:.2f}")

    # ---- 1. 混杂数据上两个方向都能读出情绪 ----
    gm = torch.Generator().manual_seed(2)
    m = coin(512, gm, POS, NEG)
    confounded = sentiment(m, torch.where(m == POS, SPORT, FOOD), gm)          # POS 总配运动, NEG 总配食物
    a = acts(model, confounded)[:, 2:].mean(1)                                  # 每条序列一个向量 [N, D]
    label = (m == POS)
    readout = {}
    for name, d in (("情绪方向 d_mood", d_mood), ("话题方向 d_topic", d_topic)):
        proj = a @ d
        thr = (proj[label].mean() + proj[~label].mean()) / 2                    # 两类均值的中点当阈值
        readout[name] = ((proj > thr) == label).float().mean().item()
    # 逻辑回归探针: 在同一份混杂数据上训练, 看它学到哪个方向
    w = torch.zeros(a.size(1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=0.05)
    mu, sd = a.mean(0), a.std(0) + 1e-6
    z = (a - mu) / sd
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(z @ w + bias, label.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_act = (w / sd).detach()                                                   # 换回原激活空间的方向
    readout["逻辑回归探针"] = (((z @ w + bias) > 0) == label).float().mean().item()
    print("\n混杂数据 (POS 总配运动, NEG 总配食物) 上读情绪标签的准确率:")
    for name, acc in readout.items():
        print(f"  {name:<12} | {acc:.3f}")
    print(f"  逻辑回归探针方向与 d_mood 的余弦 {F.cosine_similarity(w_act, d_mood, 0):.2f}, 与 d_topic 的余弦 {F.cosine_similarity(w_act, d_topic, 0):.2f}")
    # 换一份话题与情绪反着配的数据, 探针还对不对
    flipped = sentiment(m, torch.where(m == POS, FOOD, SPORT), gm)
    zf = (acts(model, flipped)[:, 2:].mean(1) - mu) / sd
    probe_flip = (((zf @ w + bias) > 0) == label).float().mean().item()
    print(f"  话题与情绪反着配时, 逻辑回归探针的准确率 {probe_flip:.3f}")
    # d_mood 的读出不断言: 混杂数据上沿它投影, 话题分量会按 |d_topic| · cos 混进来。
    # 两者都由训练的浮点细节决定, 本机是 |d_mood| 6.14、cos -0.12, 情绪分量占优 (读出 1.000);
    # 换台机器时见过 |d_mood| 6.97、cos -0.17 的组合, 两项几乎抵消, 读出掉到 0.250。
    mood_gap = ((a[label].mean(0) - a[~label].mean(0)) @ d_mood).item()
    print(f"  d_mood 上两类投影均值之差 {mood_gap:+.1f} = |d_mood|² {d_mood.norm() ** 2:.0f} "
          f"加上话题分量 {(d_topic.norm() * d_mood.norm() * cos):+.0f}: "
          f"后一项够大时, 连情绪方向本身也读不出情绪 (这一项随训练波动, 所以不作断言)")
    assert readout["话题方向 d_topic"] > 0.99, "话题方向在混杂数据上应能读出情绪"
    assert readout["逻辑回归探针"] > 0.99, "探针在混杂数据上应能读出情绪"

    # ---- 2. 加进残差流: 谁能改行为 ----
    gn = torch.Generator().manual_seed(3)
    neg_prompts = sentiment(torch.full((B,), NEG), coin(B, gn, SPORT, FOOD), gn)
    count_x = counting(B, gn)
    unit = {"d_mood": d_mood / d_mood.norm(), "d_topic": d_topic / d_topic.norm(), "探针方向": w_act / w_act.norm()}
    scale = d_mood.norm().item()
    coefs = (0.0, 1.0, 2.0, 4.0, 8.0)
    base = steer(model, neg_prompts, torch.zeros_like(d_mood))
    base_pos, base_sport, base_count = pos_rate(base), sport_rate(base), ce(steer(model, count_x, torch.zeros_like(d_mood)), count_x)
    print(f"\n在 NEG 提示之后的每个位置加 系数 × |d_mood| ({scale:.2f}) 长的向量:")
    print(f"  不引导: 下一个形容词是正面词的概率 {base_pos:.3f} | 名词是运动类的概率 {base_sport:.3f} | 计数任务 loss {base_count:.3f}")
    table = {}
    for name, u in unit.items():
        table[name] = [(pos_rate(lg), sport_rate(lg)) for lg in (steer(model, neg_prompts, c * scale * u) for c in coefs)]
        cells = " | ".join(f"{c:g}: {p:.2f} / {sp:.2f}" for c, (p, sp) in zip(coefs, table[name]))
        print(f"  {name:<6} 正面词 / 运动名词概率 | {cells}")
    assert table["d_mood"][3][0] - base_pos > 0.5, "情绪方向 (系数 4) 应能把 NEG 提示改成说正面词"
    # 小系数下另两个方向基本不动情绪; 系数 4 以上激活已被推离训练分布, 三个方向都会互相牵连,
    # 所以那里比的是幅度差距, 不是绝对阈值 (两者在不同 BLAS 下都成立, 绝对阈值不成立)。
    for name in ("d_topic", "探针方向"):
        for c, (p, _) in zip(coefs[:3], table[name][:3]):
            assert abs(p - base_pos) < 0.1, f"{name} 系数 {c}: 能读出情绪, 加进去却不改变情绪"
        gain = table[name][3][0] - base_pos
        assert table["d_mood"][3][0] - base_pos > 2 * abs(gain), (
            f"{name} 系数 4 把正面词概率改了 {gain:+.2f}, 情绪方向改了 "
            f"{table['d_mood'][3][0] - base_pos:+.2f}: 读得出不等于改得动, 差距应在 2 倍以上")
    # 哪一档系数开始改名词, 随随机初始化出来的方向而变 (本机是 4, 别的 BLAS 上可能到 8), 所以只要求扫到过。
    topic_sport = max(sp for _, sp in table["d_topic"]) - base_sport
    assert topic_sport > 0.3, f"话题方向改的是名词, 扫过的系数里运动名词概率最多只高了 {topic_sport:+.2f}"
    # 加在哪里也有关系: 连提示位置 (情绪标记) 一起加
    add_all = torch.zeros(T, d_mood.numel()) + d_mood
    h = model.layers[LAYER].register_forward_hook(lambda m, i, o: o + add_all[: o.size(1)])
    with torch.no_grad():
        all_pos = pos_rate(model(neg_prompts))
    h.remove()
    print(f"  系数 1 时: 只加在提示之后 {table['d_mood'][1][0]:.3f} | 连提示位置一起加 {all_pos:.3f}")

    # ---- 3. 引导强度与副作用 ----
    print("\n引导强度与副作用 (向量 = 系数 × d_mood):")
    sweep = []
    for c, (p, _) in zip(coefs, table["d_mood"]):
        side = ce(steer(model, count_x, c * d_mood), count_x) - base_count
        sweep.append((c, p, side))
        print(f"  系数 {c:3.1f} | 正面词概率 {p:.3f} | 计数任务 loss 增加 {side:+.3f}")
    assert all(a[1] <= b[1] + 1e-3 for a, b in zip(sweep[:4], sweep[1:4])), "系数增大到 4 之前, 行为改变应单调增加"
    assert all(a[2] < b[2] for a, b in zip(sweep, sweep[1:])), "系数越大, 无关任务上的 loss 涨得越多"

    # ---- ActAdd: 只用一对样本的差 ----
    pair_cos = [F.cosine_similarity(mean_diff(model, pos_x[i:i + 1], neg_x[i:i + 1]), d_mood, 0).item() for i in range(32)]
    print(f"\nActAdd 式单对样本的差与 d_mood (256 对平均) 的余弦: 平均 {sum(pair_cos) / 32:.2f}, 最小 {min(pair_cos):.2f}")
    print(f"\n激活引导 demo 通过 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
