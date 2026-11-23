#!/usr/bin/env python
"""
BD3-LM 训练示例 — 在 LLaDA 用过的 "置换链" 上训练块扩散 LM, 对照全范围日程和裁剪日程, 再和 LLaDA 比生成。

任务: 与 llada/train_llada.py 相同。固定一个 16-环置换 π, 序列 x[i+1] = π(x[i]), 长 16, 块大小 4。
      每步抽新数据和新遮蔽, loss 下降说明学会了 π; 评估都在 held-out 序列上。
看什么:
    1. 初始 loss ≈ ln V, 训练后逼近理论下界 ln P / L。
    2. 裁剪日程 t ~ U[0.45, 0.95] 让 loss 估计的波动变小 (未训练时明显, 训练后差别不大, 照实打印)。
    3. 给首 token 续写: 块扩散带 KV cache, 送进前向的 token 数远少于 LLaDA 每步整段重算。
    4. 生成超过训练长度: 训练只见过 16 个位置, 逐块往后接到 32。
读代码时盯住: elbo_loss 的 t_min / t_max, 以及 sample 返回的 token 计数。
"""

import math
import time

import torch

from llm_models.models.language_models.block_diffusion import BlockDiffusionLM

P, L, BS = 16, 16, 4          # 数据 token 数 / 训练序列长 / 块大小; 词表 = P + 1, 最后一个 id 是 [MASK]
CLIP = (0.45, 0.95)           # 裁剪日程的 [β, ω]


def make_perm(seed: int) -> torch.Tensor:
    """单个 P-环置换, 写法与 train_llada.train 相同。"""
    torch.manual_seed(seed)
    order = torch.randperm(P)
    perm = torch.empty(P, dtype=torch.long)
    perm[order] = order.roll(-1)
    return perm


def make_chain(perm: torch.Tensor, batch_size: int, length: int = L, generator: torch.Generator = None) -> torch.Tensor:
    """x[:, 0] 随机, x[:, i+1] = perm[x[:, i]] → [B, length]。length 可以超过训练长度。"""
    cols = [torch.randint(0, P, (batch_size,), generator=generator)]
    for _ in range(length - 1):
        cols.append(perm[cols[-1]])
    return torch.stack(cols, dim=1)


def make_model(block_size: int = BS) -> BlockDiffusionLM:
    # max_len = 2L: 训练只用到位置 0..L-1, 留出空间测试生成超过训练长度
    return BlockDiffusionLM(block_size=block_size, vocab_size=P + 1, d_model=64, n_heads=4,
                            num_kv_heads=2, num_layers=2, max_len=2 * L)


@torch.no_grad()
def eval_loss(model: BlockDiffusionLM, x: torch.Tensor, seed: int = 123) -> float:
    """固定遮蔽随机数, 全范围 t ~ U(0, 1) → 训练前后、两种日程之间可比的 ELBO 估计。"""
    model.eval()
    return model.elbo_loss(x, generator=torch.Generator().manual_seed(seed)).item()


def train(steps: int = 400, t_min: float = 0.0, t_max: float = 1.0, seed: int = 0, log_interval: int = 100):
    """返回 (model, perm)。t_min / t_max 是训练时遮蔽率的范围。"""
    perm = make_perm(seed)
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        model.train()
        loss = model.elbo_loss(make_chain(perm, 64), t_min, t_max)   # 每步新数据 + 新遮蔽
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_interval and (step == 1 or step % log_interval == 0):
            print(f"  第 {step:4d} 步 | loss {loss.item():.3f}")
    return model, perm


def continuation(model: BlockDiffusionLM, x: torch.Tensor, steps_per_block: int):
    """只给首 token, 逐块生成其余位置; 返回 (生成位置的准确率, 输出, {"tokens", "calls"} 计数)。"""
    noisy = torch.full_like(x, model.mask_id)
    noisy[:, 0] = x[:, 0]
    out, stats = model.sample(noisy, steps_per_block, return_stats=True)
    return (out[:, 1:] == x[:, 1:]).float().mean().item(), out, stats


def main():
    t0 = time.time()
    V = P + 1

    torch.manual_seed(0)
    init_loss = eval_loss(make_model(), torch.randint(0, P, (1024, L)))
    print(f"初始 loss {init_loss:.3f}  vs  ln V = {math.log(V):.3f}")
    assert abs(init_loss - math.log(V)) < 0.3, "初始 loss 应 ≈ ln V (1/t 加权后期望不变)"

    print("全范围日程 t ~ U(0, 1):")
    full, perm = train()
    print(f"裁剪日程 t ~ U[{CLIP[0]}, {CLIP[1]}]:")
    clip, _ = train(t_min=CLIP[0], t_max=CLIP[1])
    held_out = make_chain(perm, 1024, generator=torch.Generator().manual_seed(999))
    floor = math.log(P) / L
    loss_full, loss_clip = eval_loss(full, held_out), eval_loss(clip, held_out)
    print(f"held-out loss (评估都用全范围 t) | 全范围训练 {loss_full:.3f} | 裁剪训练 {loss_clip:.3f} | 下界 ln P / L = {floor:.3f}")
    for name, v in (("全范围", loss_full), ("裁剪", loss_clip)):
        assert v < 0.5 * init_loss, f"{name}日程训练后 loss 未明显下降"
        assert v > floor - 0.05, f"{name}: ELBO 是 NLL 的上界, 不该低于下界"

    # ---- 裁剪日程降方差: 同一批数据, 换 64 次遮蔽随机数, 看 loss 估计的标准差 ----
    # t 很小的块被遮的 token 少, 乘上 1/t 后单次估计忽大忽小; 把 t 限制在 [β, ω] 就去掉了这一段
    batch = held_out[:64]
    torch.manual_seed(0)
    std = {}
    print("同一批 64 条, 换 64 次遮蔽, loss 估计的 均值 ± 标准差:")
    for stage, model in (("未训练", make_model()), ("训练后", clip)):
        model.eval()
        row = []
        for name, (lo, hi) in (("全范围", (0.0, 1.0)), ("裁剪", CLIP)):
            with torch.no_grad():
                d = torch.tensor([model.elbo_loss(batch, lo, hi, torch.Generator().manual_seed(s)).item()
                                  for s in range(64)])
            std[stage, name] = d.std().item()
            row.append(f"{name} {d.mean():.3f} ± {d.std():.3f}")
        print(f"  {stage} | " + " | ".join(row))
    assert std["未训练", "裁剪"] < 0.5 * std["未训练", "全范围"], "未训练时裁剪日程的 loss 估计波动应明显更小"

    # ---- 给首 token 续写: 块扩散 (带 KV cache) vs LLaDA (每步整段重算) ----
    from llm_models.run_models.language_models.llada.train_llada import train as train_llada
    print("对照: 同一任务训练 LLaDA (输出见 train_llada):")
    llada, llada_perm = train_llada(log_interval=10 ** 9)
    llada_x = make_chain(llada_perm, 1024, generator=torch.Generator().manual_seed(999))
    noisy = torch.full_like(llada_x, llada.mask_id)
    noisy[:, 0] = llada_x[:, 0]
    llada_acc = (llada.sample(noisy, L - 1)[:, 1:] == llada_x[:, 1:]).float().mean().item()
    llada_tok = (L - 1) * L                                   # 15 步, 每步整段 16 个位置
    print("\n给首 token 续写 15 个 (held-out 1024 条, chance = 1/16 = 0.062):")
    print(f"  LLaDA, 15 步           | 准确率 {llada_acc:.3f} | 串行前向 {L - 1} 次, 共 {llada_tok} 个 token")
    accs = {}
    for name, model in (("全范围", full), ("裁剪", clip)):
        acc, _, stats = continuation(model, held_out, steps_per_block=BS)
        accs[name] = acc
        print(f"  BD3-LM {name}, 每块 {BS} 步 | 准确率 {acc:.3f} | 串行前向 {stats['calls']} 次, 共 {stats['tokens']} 个 token")
    assert max(accs.values()) > 0.8, "块扩散续写准确率应远高于 chance"
    assert stats["tokens"] < llada_tok, "带 KV cache 的块扩散送进前向的 token 应少于 LLaDA"

    # ---- 生成超过训练长度: 训练只见过位置 0..15, 这里接到 32 ----
    long_x = make_chain(perm, 1024, 2 * L, torch.Generator().manual_seed(5))
    for name, model in (("全范围", full), ("裁剪", clip)):
        _, out, _ = continuation(model, long_x, steps_per_block=BS)
        a_in = (out[:, 1:L] == long_x[:, 1:L]).float().mean().item()
        a_out = (out[:, L:] == long_x[:, L:]).float().mean().item()
        print(f"  生成 32 个 ({name}训练) | 位置 1-15 准确率 {a_in:.3f} | 位置 16-31 准确率 {a_out:.3f}")

    print(f"\nBD3-LM 训练验证通过 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
