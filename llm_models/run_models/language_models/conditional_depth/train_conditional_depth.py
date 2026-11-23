#!/usr/bin/env python
"""
条件深度训练对照 — 每 token 稠密 FLOPs 相同时, LLaMA / MoD / MoR 谁先学会 "两数相加"。

任务: 序列由 12 组 [a, b, =, (a + b) mod 10] 组成, 长 48, a 和 b 随机。a、b 本身不可预测, "=" 由位置决定,
      只有 "=" 后面的答案需要真的算: 把前两个数读过来, 再查一张 100 格的表。
      每步抽新数据, 评估在 held-out 上。这类任务的学习曲线有一段平台, 之后突然学会;
      学会的时刻对种子和配置很敏感, 这里的结论只代表 seed 0 这一次。
三个模型每 token 的稠密 FLOPs 相同 (都是 4 层当量, 脚本里按实际过层的 token 数核对):
    LLaMA-4:        4 层, 每层全部 token 都过。
    MoD-6, C = 1/3: 6 层, 第 1、3、5 层只让 1/3 的 token 过: 3 + 3 × 1/3 = 4 层当量, 参数是 6 层。
    MoR, N_r = 3:   首尾 2 个独立层 + 1 个共享块递归 3 次, 容量日程 1、2/3、1/3: 2 + 2 = 4 层当量, 参数是 3 层。
另跑一个 LLaMA-3 当参照: 它就是 MoD-6 里不路由的那 3 层, FLOPs 少 1/4。
读代码时盯住: MoD 的 loss 里加了 predictor_loss, 它只训练预测器 (输入已 detach), 不改主模型。
"""

import math
import time

import torch
import torch.nn.functional as F

from llm_models.models.language_models.conditional_depth import MoDLLaMA, MoRLLaMA, _dense_params
from llm_models.models.language_models.llama import LLaMA

EQ, V, T = 10, 11, 48            # "=" 的 id / 词表 (0-9 加 "=") / 序列长
STEPS, EVAL_EVERY = 2000, 250
CFG = dict(vocab_size=V, d_model=64, n_heads=4, max_len=64)


def make_batch(batch_size: int, generator: torch.Generator = None) -> torch.Tensor:
    """[B, T]: 每 4 个 token 一组 [a, b, =, (a + b) % 10]。"""
    a = torch.randint(0, 10, (batch_size, T // 4), generator=generator)
    b = torch.randint(0, 10, (batch_size, T // 4), generator=generator)
    return torch.stack([a, b, torch.full_like(a, EQ), (a + b) % 10], dim=-1).view(batch_size, T)


HELD_OUT = make_batch(512, torch.Generator().manual_seed(1))
ANSWER = (torch.arange(T - 1) % 4 == 2)          # 输入位置是 "=", 目标是答案的那些位置


def lm_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def evaluate(model, **kw):
    """held-out 上的 (loss, 答案位准确率)。"""
    model.eval()
    logits = model(HELD_OUT[:, :-1], **kw)
    model.train()
    acc = (logits.argmax(-1)[:, ANSWER] == HELD_OUT[:, 1:][:, ANSWER]).float().mean().item()
    return lm_loss(logits, HELD_OUT).item(), acc


def dense_flops(model) -> float:
    """每 token 稠密 FLOPs, 用上一次前向记录的过层 token 数 (LLaMA 每层都是全部 token)。"""
    if isinstance(model, (MoDLLaMA, MoRLLaMA)):
        return model.flops_per_token()["dense"]
    return sum(2 * _dense_params(layer) for layer in model.layers) + 2 * model.d_model * model.vocab_size


def train(name: str, model) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    curve, learned_at, t0 = [], None, time.time()
    for step in range(1, STEPS + 1):
        x = make_batch(32)
        loss = lm_loss(model(x[:, :-1]), x)
        if isinstance(model, MoDLLaMA):
            loss = loss + model.predictor_loss            # 只训练预测器, 输入已 detach
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
    sec = time.time() - t0
    print(f"  {name:<13} | 参数 {sum(p.numel() for p in model.parameters()):>7,} | 稠密 FLOPs/token {dense_flops(model) / 1e3:5.1f}K"
          f" | held-out loss {loss:.3f} | 答案准确率 {acc:.3f} | 首次 ≥ 0.9 的步数 {learned_at or '未达到'} | {sec:.0f}s")
    print(f"  {'':<13} | 每 {EVAL_EVERY} 步的答案准确率: {' '.join(curve)}")
    return {"loss": loss, "acc": acc, "learned_at": learned_at}


def main(seed: int = 0):
    t0 = time.time()
    makers = {
        "LLaMA-3 (参照)": lambda: LLaMA(num_layers=3, **CFG),
        "LLaMA-4": lambda: LLaMA(num_layers=4, **CFG),
        "MoD-6, C=1/3": lambda: MoDLLaMA(capacity=1 / 3, num_layers=6, **CFG),
        "MoR, N_r=3": lambda: MoRLLaMA(num_recursions=3, **CFG),
    }
    # T-1 个目标里, 除首个 a (不是目标) 外的 a、b 共 23 个不可预测; 只学会 "=" 时 12 个答案也按均匀猜
    floor = (2 * (T // 4) - 1) * math.log(10) / (T - 1)
    print(f"理论下界 (只剩 a、b 的随机性) {floor:.3f}; 只学会 '=' 时约 {(3 * (T // 4) - 1) * math.log(10) / (T - 1):.3f}")
    results, models = {}, {}
    for name, make in makers.items():
        torch.manual_seed(seed)
        model = make()
        init, _ = evaluate(model)
        assert abs(init - math.log(V)) < 0.3, f"{name}: 初始 loss {init:.3f} 应 ≈ ln V = {math.log(V):.3f}"
        results[name], models[name] = train(name, model), model
        assert results[name]["loss"] < init, f"{name}: loss 应下降"
    flops = [dense_flops(m) for name, m in models.items() if name != "LLaMA-3 (参照)"]
    assert max(flops) / min(flops) < 1.02, "三个模型的每 token 稠密 FLOPs 应基本相同"

    # ---- MoD: 采样时换成因果预测器, 代价多大 ----
    mod = models["MoD-6, C=1/3"]
    with torch.no_grad():
        mod.eval()
        mod(HELD_OUT[:, :-1], routing="topk")
        topk_sel = {i: s.clone() for i, s in mod.last_sel.items()}
        mod(HELD_OUT[:, :-1], routing="predictor")
        agree = sum((topk_sel[i] == s).float().mean().item() for i, s in mod.last_sel.items()) / len(topk_sel)
        used = sum(s.float().mean().item() for s in mod.last_sel.values()) / len(topk_sel)
        eq_share = sum(s[:, ANSWER].float().sum().item() / s.float().sum().item() for s in topk_sel.values()) / len(topk_sel)
    loss_pred, acc_pred = evaluate(mod, routing="predictor")
    print(f"\nMoD 预测器与 top-k 的路由一致率 {agree:.3f} | 预测器路由时实际过层比例 {used:.3f} (top-k 固定 {1 / 3:.3f})")
    print(f"MoD 用预测器路由 (采样时可用): held-out loss {loss_pred:.3f} | 答案准确率 {acc_pred:.3f}")
    print(f"MoD top-k 选中的 token 里 '=' 位置占 {eq_share:.3f} (随机选是 {ANSWER.float().mean():.3f})")
    assert agree > 0.5, "预测器应学到 top-k 的大部分决定"

    print(f"\n条件深度训练对照完成 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
