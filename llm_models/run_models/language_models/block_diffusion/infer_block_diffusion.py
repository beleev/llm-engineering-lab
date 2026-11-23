#!/usr/bin/env python
"""
BD3-LM 结构验证 — 掩码的两个极端、训练掩码与采样过程一致、KV cache 不改结果、生成长度可以往后接。

独立可跑: 内部先训练 300 步 (复用 train_block_diffusion.train, 约 6 s), 不依赖 checkpoint。
看什么:
    1. block_size = 1 的块因果掩码就是因果掩码; block_size = 序列长时全可见, 前向与 LLaDA 逐位相同。
    2. 训练时 x_t ⊕ x 一次前向算出的加噪块 logits, 与采样时 "前文进 cache, 当前块单独前向" 算的相同。
    3. 带 KV cache 逐块生成, 与每步把前文整段重算的结果相同, 送进前向的 token 少得多。
    4. 先生成 2 块再停, 与一口气生成 4 块的前 2 块相同: 后面的块改不了前面的块。
"""

import torch

from llm_models.models.language_models.block_diffusion import BlockDiffusionLM, block_causal_mask, training_mask
from llm_models.models.language_models.llada import LLaDA
from llm_models.run_models.language_models.block_diffusion.train_block_diffusion import BS, L, P, make_chain, train
from llm_models.utils.masks import build_causal_mask


def show_mask(m: torch.Tensor) -> str:
    """bool 方阵 → 每行一串 '■' (可见) / '·' (屏蔽)。"""
    return "\n".join("  " + "".join("■" if v else "·" for v in row) for row in m.tolist())


def main():
    # ---- 1. 掩码的两个极端 ----
    assert torch.equal(block_causal_mask(L, 1), build_causal_mask(L, torch.device("cpu"))), "block_size=1 应等于因果掩码"
    assert block_causal_mask(L, L).all(), "block_size=L 应全可见 (LLaDA 的全双向)"
    print(f"训练掩码 x_t ⊕ x (L = 8, 块大小 4; 左上 M_BD, 右上 M_OBC, 左下 0, 右下 M_BC):\n{show_mask(training_mask(8, 4)[0])}")

    torch.manual_seed(0)
    full_blk = BlockDiffusionLM(block_size=L, vocab_size=P + 1, d_model=64, n_heads=4, num_kv_heads=2, num_layers=2, max_len=L)
    x = torch.randint(0, P, (8, L))
    with torch.no_grad():
        diff = (full_blk(x, mask=block_causal_mask(L, L)) - LLaDA.forward(full_blk, x)).abs().max().item()
    print(f"\nblock_size = L 时与 LLaDA 前向的最大差 {diff:.1e}")
    assert diff == 0.0, "block_size = L 时应与 LLaDA 前向逐位相同"

    model, perm = train(steps=300, log_interval=0)
    model.eval()
    M = model.mask_id
    held_out = make_chain(perm, 256, generator=torch.Generator().manual_seed(7))

    # ---- 2. 训练掩码 = 采样时的计算 ----
    noisy = held_out.masked_fill(torch.rand(held_out.shape, generator=torch.Generator().manual_seed(1)) < 0.5, M)
    with torch.no_grad():
        train_logits = model.noisy_logits(noisy, held_out)               # [B, L, V] 一次前向
        caches = [dict() for _ in model.layers]
        step_logits = []
        for b0 in range(0, L, BS):
            pos = torch.arange(b0, b0 + BS)
            step_logits.append(model(noisy[:, b0:b0 + BS], position_ids=pos, caches=[dict(c) for c in caches]))
            model(held_out[:, b0:b0 + BS], position_ids=pos, caches=caches)   # 干净块进 cache
    gap = (train_logits - torch.cat(step_logits, 1)).abs().max().item()
    print(f"训练前向 vs 逐块带 cache 前向, 加噪块 logits 最大差 {gap:.1e}")
    assert gap < 1e-4, "训练掩码应与采样时 '前文进 cache, 当前块单独前向' 的计算一致"

    # ---- 3. KV cache 不改结果 ----
    prompt = torch.full_like(held_out, M)
    prompt[:, 0] = held_out[:, 0]                                       # 只给首 token
    out_c, st_c = model.sample(prompt, BS, use_cache=True, return_stats=True)
    out_n, st_n = model.sample(prompt, BS, use_cache=False, return_stats=True)
    print(f"带 cache: 送进前向 {st_c['tokens']} 个 token, 串行 {st_c['calls']} 次 | "
          f"不带 cache: {st_n['tokens']} 个 token, 串行 {st_n['calls']} 次 | 输出相同 {torch.equal(out_c, out_n)}")
    assert torch.equal(out_c, out_n), "带 KV cache 与整段重算的生成结果应相同"
    assert st_c["tokens"] < st_n["tokens"], "KV cache 应减少送进前向的 token 数"
    assert not (out_c == M).any(), "输出里不应有 [MASK]"
    acc = (out_c[:, 1:] == held_out[:, 1:]).float().mean().item()
    print(f"\n真值 : {' '.join(f'{v:2d}' for v in held_out[0].tolist())}")
    print(f"生成 : {' '.join(f'{v:2d}' for v in out_c[0].tolist())}")
    print(f"给首 token 续写 15 个, 每块 {BS} 步: 准确率 {acc:.3f} | chance {1 / P:.3f}")
    assert acc > 0.8, "续写准确率应远高于 chance"

    # ---- 4. 生成长度可以往后接: 后面的块不改前面的块 ----
    short = model.sample(prompt[:, :2 * BS], BS)
    assert torch.equal(short, out_c[:, :2 * BS]), "先生成 2 块的结果应等于生成 4 块时的前 2 块"
    print("先生成 2 块 = 生成 4 块的前 2 块: 成立")
    print("BD3-LM 结构验证通过")


if __name__ == "__main__":
    main()
