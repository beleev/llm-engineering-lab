#!/usr/bin/env python
"""
Transfusion 推理: 验证 "整条序列因果, 同一张图内部双向" 这张掩码, 以及它在前向里真的起作用。

不训练, 用随机初始化的模型做扰动实验 (几秒)。看什么:
    1. 掩码本身: 图像段内互相可见; 图像后的文本看得到整张图; 图像前的文本看不到图。
    2. 前向里的效果: 改图像 patch, 图像前的文本 logits 一位不变, 图像后的文本跟着变;
       改最后一个 patch, 第一个 patch 的输出也变 (段内双向); 换成纯因果掩码后不再变。
    3. 生成接口: 文本 + BOI 后接噪声 patch, Euler 积分出一张 [3, 8, 8] 的图。
"""

import math

import torch

from llm_models.models.multimodal.transfusion import transfusion_mask
from llm_models.run_models.multimodal.transfusion.train_transfusion import (
    N_PATCH, SEQ, VOCAB, build_model, eval_losses, generate, layout, make_batch,
)
from llm_models.utils.masks import build_causal_mask


def show_mask(m: torch.Tensor, names: list) -> None:
    """把 [S, S] 掩码打印成字符画: 行是 query, 列是 key, # = 可见。图像段只画前 4 个 patch。"""
    keep = [i for i, n in enumerate(names) if not n.startswith("p") or int(n[1:]) < 4 or int(n[1:]) == N_PATCH - 1]
    print("        " + " ".join(f"{names[j]:>4}" for j in keep))
    for i in keep:
        print(f"{names[i]:>6}  " + " ".join(f"{'#' if m[i, j] else '.':>4}" for j in keep))


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    # ---------- 1. 掩码结构 ----------
    for kind in ("text_first", "image_first"):
        tokens, image_id, _, _ = layout(kind, 1)
        m = transfusion_mask(image_id)[0]                                    # [S, S]
        img = (image_id[0] >= 0).nonzero().squeeze(1)                         # 图像位置
        before = torch.arange(0, int(img[0]))                                 # 图像前的文本位置
        after = torch.arange(int(img[-1]) + 1, SEQ)                           # 图像后的文本位置
        assert m[img][:, img].all(), f"{kind}: 同一张图的 patch 应互相可见"
        assert m[after][:, img].all(), f"{kind}: 图像后的文本应看得到整张图"
        assert not m[before][:, img].any(), f"{kind}: 图像前的文本不应看到图"
        outside = ~((image_id[0, :, None] >= 0) & (image_id[0, None, :] >= 0))  # 至少一端是文本
        assert torch.equal(m[outside], build_causal_mask(SEQ, m.device)[0][outside]), f"{kind}: 涉及文本的位置应是普通因果"
        n_vis = int(m.sum())
        print(f"{kind:<12} | 可见对 {n_vis} (纯因果 {SEQ * (SEQ + 1) // 2}) | 图前文本看到的 patch {int(m[before][:, img].sum())}"
              f" | 图后文本每个看到的 patch {int(m[after][:, img].sum(1).min()) if len(after) else '-'}")

    batch, _, _ = make_batch("text_first", 1, torch.Generator().manual_seed(3))
    tokens, image_id = batch["tokens"], batch["image_id"]
    names = [VOCAB[t] if i < 4 or i == SEQ - 1 else f"p{i - 4}" for i, t in enumerate(tokens[0].tolist())]
    names = [n.strip("<>") for n in names]
    print("\n文本在前的掩码 (# = 可见, 图像段只画 p0~p3 和 p15):")
    show_mask(transfusion_mask(image_id)[0], names)

    # ---------- 2. 初始损失 ----------
    model = build_model()
    model.eval()
    cap0, tf0, img0 = eval_losses(model)
    print(f"\n未训练 | 看图说话 CE {cap0:.3f} | 图前文本 CE {tf0:.3f} (ln V = {math.log(len(VOCAB)):.3f}) | 图像 MSE {img0:.3f}")
    assert abs(cap0 - math.log(len(VOCAB))) < 0.35, "初始文本 CE 应 ≈ ln V"
    assert 1.5 < img0 < 2.5, "初始速度预测 ≈ 0, 图像 MSE 应 ≈ E[(ε − x_0)²] ≈ 2"

    # ---------- 3. 扰动实验 (随机初始化的模型) ----------
    with torch.inference_mode():
        for kind in ("text_first", "image_first"):
            batch, _, _ = make_batch(kind, 4, g)
            img = (batch["image_id"][0] >= 0).nonzero().squeeze(1)
            a, b = int(img[0]), int(img[-1])
            base = model(**batch)

            other = dict(batch, patches=batch["patches"].clone())
            other["patches"][:, a:b + 1] = torch.randn(4, N_PATCH, other["patches"].size(-1), generator=g)
            out = model(**other)
            d_before = (out["logits"][:, :a] - base["logits"][:, :a]).abs().max().item()
            d_after = (out["logits"][:, b + 1:] - base["logits"][:, b + 1:]).abs().max().item()

            last = dict(batch, patches=batch["patches"].clone())
            last["patches"][:, b] += 1.0                                       # 只改最后一个 patch
            d_first = (model(**last)["velocity"][:, a] - base["velocity"][:, a]).abs().max().item()
            causal = build_causal_mask(SEQ, torch.device("cpu"))               # 对照: 纯因果, 去掉段内双向
            d_first_causal = (model(**last, mask=causal)["velocity"][:, a]
                              - model(**batch, mask=causal)["velocity"][:, a]).abs().max().item()
            print(f"{kind:<12} | 换整张图: 图前文本 logits 变化 {d_before:.1e}, 图后文本 {d_after:.1e}"
                  f" | 改最后一个 patch: 第一个 patch 输出变化 {d_first:.1e}, 纯因果掩码下 {d_first_causal:.1e}")
            assert d_before == 0.0, f"{kind}: 图像前的文本不应受图像影响"
            if b + 1 < SEQ:
                assert d_after > 1e-4, f"{kind}: 图像后的文本应读到图像"
            assert d_first > 1e-4, f"{kind}: 图像段内双向, 后面的 patch 应影响前面的 patch"
            assert d_first_causal == 0.0, f"{kind}: 纯因果掩码下后面的 patch 不应影响前面的 patch"

        # 图像在前时, 改后面的标题不应影响图像的去噪输出
        batch, _, _ = make_batch("image_first", 4, g)
        other = dict(batch, tokens=batch["tokens"].clone())
        other["tokens"][:, -2:] = torch.randint(3, len(VOCAB), (4, 2), generator=g)
        d_img = (model(**other)["velocity"][:, 2:2 + N_PATCH] - model(**batch)["velocity"][:, 2:2 + N_PATCH]).abs().max().item()
        print(f"image_first  | 换图像后的标题: 图像位置的速度预测变化 {d_img:.1e}")
        assert d_img == 0.0, "图像在前时, 后面的文本不应影响图像"

    # ---------- 4. 生成接口 ----------
    out = generate(model, torch.tensor([0, 2]), torch.tensor([1, 2]), steps=4)
    print(f"\n生成接口: 2 条提示 → 图 {list(out.shape)} (未训练, 只检查形状)")
    assert out.shape == (2, 3, 8, 8), "生成图的形状应是 [B, 3, 8, 8]"
    print("\nTransfusion 掩码与数据流检查通过")


if __name__ == "__main__":
    main()
