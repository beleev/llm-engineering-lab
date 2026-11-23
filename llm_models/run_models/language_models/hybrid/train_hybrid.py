#!/usr/bin/env python
"""
混合架构对照训练 — 输出门对 attention sink 的影响, 以及几种位置编码排布超出训练长度后的表现

任务是植入针的检索: 序列前半段散落着 4 对 (KEY_j, v_j), 其余是随机填充; 末尾依次给出 4 个 KEY_j,
    只有这 4 个答案位算 loss。KEY 与值每条序列都重抽, 背不下来。
    序列越长, 针与问话之间的距离越远, 所以同一个任务可以直接拿来看超出训练长度之后掉多少。
两组对照 (都是 4 层, d_model = 64, 训练长度 32):
    A. 输出门: gqa × 4 对 gated_gqa × 4, 看第 0 个 token 分走多少注意力质量。
    B. 位置编码排布: 全 RoPE 全注意力 / 全 NoPE 全注意力 / iRoPE (3 层分块 RoPE + 1 层 NoPE)
       / Kimi 式 (3 层 KDA + 1 层 NoPE 的 MLA), 训练长度 32, 评估长度 32 / 64 / 128。
读代码时盯住: sink_mass —— 它读的是各 GQA 系列层上一次前向里第 0 列的平均注意力质量,
    第 0 行不算 (那一行只有自己可看, 必然是 1)。
"""

import math
import time

import torch
import torch.nn.functional as F

from llm_models.models.language_models.hybrid import HybridLM

NKEY, V, T_TRAIN, CHUNK = 4, 20, 32, 16
KEY0, VAL_MAX, FILL0 = 10, 10, 14          # KEY_j = 10 + j; 值取 0..9; 填充取 14..19
STEPS, EVAL_LENS = 1200, (32, 64, 128)
CFG = dict(vocab_size=V, d_model=64, n_heads=4, max_len=128, chunk_size=CHUNK,
           latent_dim=32, qk_rope_head_dim=16, dropout=0.0)
LAYOUTS = {
    "gqa (无输出门)": "gqa gqa gqa gqa",
    "gated_gqa (有输出门)": "gated_gqa gated_gqa gated_gqa gated_gqa",
    "全 NoPE 全注意力": "gqa_nope gqa_nope gqa_nope gqa_nope",
    "iRoPE (分块 RoPE + NoPE)": "gqa_chunk gqa_chunk gqa_chunk gqa_nope",
    "Kimi 式 (KDA + NoPE MLA)": "kda kda kda mla_nope",
}


def make_batch(batch_size: int, seq_len: int, generator: torch.Generator = None):
    """返回 (idx [B, T], labels [B, T])。labels[t] 是位置 t 的 logits 该预测的 token, -100 表示不算。

    针均匀撒在前半段, 问话固定在末尾, 所以序列越长针离问话越远。
    """
    g = generator
    idx = torch.randint(FILL0, FILL0 + 6, (batch_size, seq_len), generator=g)
    labels = torch.full((batch_size, seq_len), -100)
    vals = torch.randint(0, VAL_MAX, (batch_size, NKEY), generator=g)
    span = (seq_len // 2) // NKEY
    rows = torch.arange(batch_size)
    for j in range(NKEY):
        p = 1 + j * span + torch.randint(0, max(1, span - 2), (batch_size,), generator=g)
        idx[rows, p] = KEY0 + j
        idx[rows, p + 1] = vals[:, j]
        q = seq_len - 2 * NKEY + 2 * j                        # 末尾的问话位置
        idx[:, q] = KEY0 + j
        idx[rows, q + 1] = vals[:, j]
        labels[rows, q] = vals[:, j]
    return idx, labels


HELD = {n: make_batch(64, n, torch.Generator().manual_seed(1)) for n in EVAL_LENS}


@torch.no_grad()
def evaluate(model, seq_len: int) -> float:
    """held-out 上 "读出上一次同一个 a 后面那个 b" 的准确率。"""
    model.eval()
    idx, labels = HELD[seq_len]
    logits = model(idx)
    model.train()
    pick = labels != -100
    return (logits.argmax(-1)[pick] == labels[pick]).float().mean().item()


def train(name: str, layout: str, seed: int = 0):
    torch.manual_seed(seed)
    model = HybridLM(layout=layout, **CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss0 = None
    t0 = time.time()
    for step in range(STEPS):
        idx, labels = make_batch(16, T_TRAIN)
        loss = F.cross_entropy(model(idx).flatten(0, 1), labels.flatten(), ignore_index=-100)
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    accs = {n: evaluate(model, n) for n in EVAL_LENS}
    evaluate(model, T_TRAIN)                                           # 让记下的 sink 质量来自训练长度
    sinks = [s for s in model.sink_mass() if s is not None]
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name:<24} | 参数 {params:>7,} | 初始 loss {loss0:.3f} → {loss.item():.3f}"
          f" | 准确率 " + " ".join(f"T={n}: {accs[n]:.3f}" for n in EVAL_LENS)
          + (f" | 第 0 列注意力质量 {sum(sinks) / len(sinks):.4f}" if sinks else "")
          + f" | {time.time() - t0:.0f}s")
    return {"loss0": loss0, "loss": loss.item(), "accs": accs, "sinks": sinks}


def main(seed: int = 0):
    t0 = time.time()
    print(f"植入针检索任务: {NKEY} 对 (KEY, 值) 每条序列重抽, 词表 {V}, 训练长度 {T_TRAIN}, "
          f"分块大小 {CHUNK}; 初始 loss 应 ≈ ln {V} = {math.log(V):.3f}")
    out = {}
    for name, layout in LAYOUTS.items():
        out[name] = train(name, layout, seed)

    plain, gated = out["gqa (无输出门)"], out["gated_gqa (有输出门)"]
    # 均匀分布下第 0 列的期望: 第 t 行有 t+1 个可见 key, 平均 1/(t+1), 第 0 行不计
    uniform = sum(1 / (t + 1) for t in range(1, T_TRAIN)) / (T_TRAIN - 1)
    print(f"\nA 输出门与 attention sink (4 层平均, 在训练长度 {T_TRAIN} 上测; 均匀分布的期望是 {uniform:.4f}):")
    print(f"  无输出门 第 0 列分走 {sum(plain['sinks']) / len(plain['sinks']):.4f}"
          f" | 有输出门 {sum(gated['sinks']) / len(gated['sinks']):.4f}")
    print(f"  逐层 (无门): {[round(s, 4) for s in plain['sinks']]}")
    print(f"  逐层 (有门): {[round(s, 4) for s in gated['sinks']]}")

    print("\nB 超出训练长度后的准确率 (训练长度 32):")
    for name in ("gqa (无输出门)", "全 NoPE 全注意力", "iRoPE (分块 RoPE + NoPE)", "Kimi 式 (KDA + NoPE MLA)"):
        a = out[name]["accs"]
        print(f"  {name:<24} " + "  ".join(f"T={n}: {a[n]:.3f}" for n in EVAL_LENS)
              + f"  | T=128 相对 T=32 {a[128] / max(a[32], 1e-9):.2f}×")

    for name, r in out.items():
        assert abs(r["loss0"] - math.log(V)) < 0.3, f"{name}: 初始 loss 应 ≈ ln V"
        assert r["loss"] < r["loss0"] - 0.3, f"{name}: 训练后 loss 应明显下降"
    assert max(r["accs"][T_TRAIN] for r in out.values()) > 0.9, "至少要有一种排布在训练长度上学会这个任务"
    assert all(0 <= s <= 1 for s in plain["sinks"] + gated["sinks"]), "注意力质量应落在 0 到 1 之间"
    rope, nope = out["gqa (无输出门)"]["accs"], out["全 NoPE 全注意力"]["accs"]
    assert nope[128] / nope[T_TRAIN] > 2 * rope[128] / rope[T_TRAIN], \
        "NoPE 排布在训练长度外的保持率应明显好于全 RoPE"
    print(f"\n混合架构对照训练完成, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
