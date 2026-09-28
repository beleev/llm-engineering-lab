#!/usr/bin/env python
"""
VAR 结构自检 (未训练权重): token 金字塔、block-causal mask、逐级采样
"""

import torch

from llm_models.models.generative.var import ImageTokenizer, VARModel


def main():
    torch.manual_seed(42)
    K, scales = 64, (1, 2, 4)                                              # 码本大小; 三级分辨率, token 数 1 + 4 + 16 = 21

    tokenizer = ImageTokenizer(image_size=16, codebook_size=K, latent_dim=16,
                               base_channels=16, levels=2, scales=scales).eval()
    model = VARModel(tokenizer, d_model=96, n_heads=4, num_layers=2).eval()

    # 1) 图像 → token 金字塔
    idx = tokenizer.encode_to_indices(torch.randn(2, 3, 16, 16))
    print("token 金字塔:", [tuple(i.shape) for i in idx], "| L =", model.num_tokens)
    assert [tuple(i.shape) for i in idx] == [(2, s, s) for s in scales], "token 金字塔各级形状应为 [B, s, s]"

    # 2) mask 长相: 21×21, 分块下三角
    #    序列排布: 位置 0 是级 1, 位置 1..4 是级 2, 位置 5..20 是级 3; m[i, j] = 位置 i 能否看位置 j
    m = model.mask
    assert m.shape == (21, 21), "mask 大小应为 L×L, L = 1 + 4 + 16 = 21"
    assert m[1, 4], "同级双向可见: 级 2 的第一个 token (位置 1) 应看得到同级最后一个 (位置 4)"
    assert not m[1, 5], "细尺度对粗尺度不可见: 级 2 (位置 1) 不应看到级 3 (位置 5)"
    assert m[5, 1], "粗尺度对细尺度可见: 级 3 (位置 5) 应看得到级 2 (位置 1)"
    assert m.sum() == 1 + 4 * 5 + 16 * 21, "可见格子总数: 级 1 的 1 行看 1 个, 级 2 的 4 行各看 5 个, 级 3 的 16 行各看 21 个"
    rows = 6
    print(f"block-causal mask 前 {rows} 行:\n", m[:rows, :8].int())

    # 为了让 token 改动能反映到输入上, 给码本一个非退化的随机值 (默认初始化 ~1/K 太小)
    torch.nn.init.normal_(tokenizer.quantizer.vq.codebook.weight)
    idx = [torch.randint(0, K, (2, s, s)) for s in scales]

    with torch.inference_mode():
        base = model.forward_tokens(idx)                                   # [2, 21, K]

        # 3) 改第 2 级 (位置 1..4) 的一个 token: 级 1、级 2 的 logits 不变, 级 3 必须变
        idx2 = [i.clone() for i in idx]
        idx2[1][:, 0, 0] = (idx2[1][:, 0, 0] + 1) % K
        out = model.forward_tokens(idx2)
        same = (out[:, :5] - base[:, :5]).abs().max().item()
        diff = (out[:, 5:] - base[:, 5:]).abs().max().item()
        print(f"改级2 token → 级1/2 logits 最大变化 {same:.2e}, 级3 最大变化 {diff:.2e}")
        assert same < 1e-6, "改级 2 的 token 不应影响级 1、级 2 的 logits (它只作为级 3 的输入)"
        assert diff > 1e-4, "级 3 以级 2 的 token 为输入: 改了级 2, 级 3 的 logits 应改变"

        # 4) 改最细级 (级 3) 的 token: 它只当 label 不当输入, 所有 logits 不变
        idx3 = [i.clone() for i in idx]
        idx3[2] = (idx3[2] + 1) % K
        assert (model.forward_tokens(idx3) - base).abs().max().item() < 1e-6, \
            "最细一级的 token 只当 label 不当输入, 改它不应影响任何 logits"

        # 5) 直接测 mask 本身: 扰动级 2 的输入特征, 级 1 输出不变, 级 2 自己要变 (同级双向可见)
        feats = tokenizer.quantizer.next_scale_inputs(idx)
        feats2 = [feats[0] + 1.0, feats[1]]                                # feats[0] 是级 2 位置上的输入
        o1, o2 = model._run(feats, 2), model._run(feats2, 2)
        assert (o1[:, :1] - o2[:, :1]).abs().max() < 1e-6, "粗尺度看到了细尺度 → mask 漏了"
        assert (o1[:, 1:5] - o2[:, 1:5]).abs().max() > 1e-4, "扰动级 2 的输入, 级 2 自己的输出应改变"

    # 6) 逐级采样: 词表 = 码本, 不可能采到非法 id
    tokens = model.sample_tokens(batch_size=3, temperature=1.0, top_k=10)
    assert all(0 <= t.min() and t.max() < K for t in tokens), "采样出的 token 应全部落在码本范围 [0, K)"
    img = model.sample(batch_size=3)
    assert img.shape == (3, 3, 16, 16), "采样图像形状应为 [B, 3, 16, 16]"
    assert img.abs().max() <= 1.0, "采样图像的值应落在 [-1, 1]"
    print("采样:", [tuple(t.shape) for t in tokens], "→ 图像", tuple(img.shape))


if __name__ == "__main__":
    main()
