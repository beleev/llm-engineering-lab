#!/usr/bin/env python
"""
残差宽度对照训练 — 同一个任务上, 1 条残差流 / n 条不约束 / n 条双随机各训一遍。

任务: 序列由 12 组 [a, b, =, (a + b) mod 10] 组成, 长 48, a 和 b 随机。a、b 本身不可预测, "=" 由位置决定,
      只有 "=" 后面的答案需要真的算。每步抽新数据, 评估在 held-out 上。与条件深度那一章用的是同一个任务,
      方便横向比。
三个模型除残差连接外完全相同 (6 层, d_model = 64):
    standard —— 普通 Pre-LN, 1 条残差流。
    hc, n = 4 —— H_pre / H_post / H_res 都不加约束。
    mhc, n = 4 —— H_pre 过 sigmoid、H_post 过 2·sigmoid、H_res 过 Sinkhorn-Knopp 投到双随机。
三者初始化时逐位相同 (见 infer 脚本), 所以初始 loss 一样, 差别全来自训练。
看什么: 训练后各层 H_res 连乘的 Amax 增益。双随机约束下行和恒为 1; 不加约束时它会漂走,
    漂多少由训练自己决定。玩具规模上 6 层漂不出论文在 27B 上看到的那种量级, 照实报告。
读代码时盯住: res_gain —— 它读的是最后一次前向各层实际用过的 H_res, 不是静态偏置。
"""

import math
import time

import torch
import torch.nn.functional as F

from llm_models.layers.core.hyper_connections import composite_gain
from llm_models.models.language_models.hyper_llama import HyperLLaMA

EQ, V, T = 10, 11, 48            # "=" 的 id / 词表 (0-9 加 "=") / 序列长
STEPS, EVAL_EVERY, N = 800, 100, 4
CFG = dict(vocab_size=V, d_model=64, n_heads=4, num_layers=6, max_len=64)


def make_batch(batch_size: int, generator: torch.Generator = None) -> torch.Tensor:
    """[B, T]: 每 4 个 token 一组 [a, b, =, (a + b) % 10]。"""
    a = torch.randint(0, 10, (batch_size, T // 4), generator=generator)
    b = torch.randint(0, 10, (batch_size, T // 4), generator=generator)
    return torch.stack([a, b, torch.full_like(a, EQ), (a + b) % 10], dim=-1).view(batch_size, T)


HELD_OUT = make_batch(256, torch.Generator().manual_seed(1))
ANSWER = (torch.arange(T - 1) % 4 == 2)          # 输入位置是 "=", 目标是答案的那些位置


def lm_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def evaluate(model):
    """held-out 上的 (loss, 答案位准确率)。"""
    model.eval()
    logits = model(HELD_OUT[:, :-1])
    model.train()
    acc = (logits.argmax(-1)[:, ANSWER] == HELD_OUT[:, 1:][:, ANSWER]).float().mean().item()
    return lm_loss(logits, HELD_OUT).item(), acc


def res_gain(model):
    """各层 H_res 连乘后的 (最大绝对行和, 最大绝对列和, 谱范数), 以及各层矩阵本身。

    矩阵取自最后一次前向、第 0 条序列的最后一个位置; 训练前后都在 HELD_OUT 上算, 可以逐元素比。
    """
    mats = [m.clone() for m in model.res_matrices()]
    return composite_gain(mats), mats


def train(name: str, model) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss0, _ = evaluate(model)
    _, res_before = res_gain(model)                   # 训练前的 H_res, 用来看它到底动了多少
    curve, learned_at, t0 = [], None, time.time()
    for step in range(1, STEPS + 1):
        x = make_batch(32)
        loss = lm_loss(model(x[:, :-1]), x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            l, a = evaluate(model)
            curve.append(f"{a:.2f}")
            if learned_at is None and a >= 0.9:
                learned_at = step
    loss, acc = evaluate(model)
    (row, col, spec), res_after = res_gain(model)
    moved = max((a - b).abs().max().item() for a, b in zip(res_after, res_before))
    sec = time.time() - t0
    print(f"  {name:<14} | 参数 {sum(p.numel() for p in model.parameters()):>7,} | 初始 loss {loss0:.3f}"
          f" | held-out loss {loss:.3f} | 答案准确率 {acc:.3f} | 首次 ≥ 0.9 的步数 {learned_at or '未达到'} | {sec:.0f}s")
    print(f"  {'':<14} | 每 {EVAL_EVERY} 步的答案准确率: {' '.join(curve)}")
    print(f"  {'':<14} | 训练后 H_res 连乘 ({CFG['num_layers']} 层): 行和 {row:.6f} 列和 {col:.6f} 谱范数 {spec:.6f}"
          f" | 单层训练中改变最大 {moved:.4f}")
    return {"loss0": loss0, "loss": loss, "acc": acc, "row": row, "col": col, "moved": moved}


def main(seed: int = 0):
    t0 = time.time()
    specs = {
        "standard": dict(residual="standard"),
        f"hc, n = {N}": dict(residual="hc", n_streams=N),
        f"mhc, n = {N}": dict(residual="mhc", n_streams=N),
    }
    # 理论下界 1.127: 47 个目标位置里有 24 个是随机的 a 或 b (各 ln 10), 其余由位置或前文决定
    print(f"任务 [a, b, =, (a+b) mod 10], {CFG['num_layers']} 层, {STEPS} 步, seed {seed}; "
          f"理论下界 (只剩 a、b 的随机性) 1.127, 只学会 '=' 时约 1.715")
    out = {}
    for name, kw in specs.items():
        torch.manual_seed(seed)
        out[name] = train(name, HyperLLaMA(**kw, **CFG))

    base = out["standard"]
    for name, r in out.items():
        assert abs(r["loss0"] - math.log(V)) < 0.2, f"{name}: 初始 loss 应约等于 ln V"
        assert abs(r["loss0"] - base["loss0"]) < 1e-4, f"{name}: 初始化时三种残差应逐位相同, 初始 loss 也应相同"
        assert r["loss"] < r["loss0"] - 0.3, f"{name}: 训练后 held-out loss 应明显下降"
    assert abs(out[f"mhc, n = {N}"]["row"] - 1) < 1e-5, "mhc: 双随机矩阵连乘的行和应恒为 1"
    assert abs(out[f"mhc, n = {N}"]["col"] - 1) < 1e-2, "mhc: 双随机矩阵连乘的列和应近似为 1"
    for key in (f"hc, n = {N}", f"mhc, n = {N}"):
        assert out[key]["moved"] > 1e-3, f"{key}: H_res 应确实被训练改动过, 不是卡在初值上"
    assert abs(out[f"hc, n = {N}"]["row"] - 1) > abs(out[f"mhc, n = {N}"]["row"] - 1), \
        "hc: 不加约束时连乘的行和应比双随机约束下偏离 1 更多"
    print(f"\n残差宽度对照训练完成, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
