#!/usr/bin/env python
"""
MM-DiT 结构自检: 时间嵌入的量纲、双流联合注意力、Euler 采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.models.generative.mmdit import MMDiT
from llm_models.training.diffusion import EulerFlowSampler, FlowMatchingScheduler


def main():
    torch.manual_seed(42)

    model = MMDiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=128, n_heads=4, num_layers=2,
        text_seq_len=16, text_dim=64,
    ).eval()
    print(f"MM-DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = FlowMatchingScheduler(num_train_timesteps=1000)
    x0 = torch.randn(2, 4, 8, 8)                                       # [B, C, H, W] 干净 latent
    t = torch.tensor([0.1, 0.9])                                       # Flow Matching 的 t ∈ [0, 1]: 0 是数据, 1 是纯噪声
    noised = scheduler.add_noise(x0, t)

    # 1) 时间嵌入量纲: 裸 t∈[0,1] 的 sinusoidal 特征几乎不可分, scheduler ×1000 后才可分
    emb = model.t_embed._sin_embed
    # emb(t) 是 [2, D]; 用 * 拆成两条向量, 算 t=0.1 与 t=0.9 两个嵌入的余弦相似度
    cos_raw = F.cosine_similarity(*emb(t), dim=0).item()
    cos_scaled = F.cosine_similarity(*emb(noised.t_norm), dim=0).item()
    print(f"cos(emb(t={t[0]:.1f}), emb(t={t[1]:.1f})): 裸 t = {cos_raw:.3f} | scheduler 输出的 t_norm = {cos_scaled:.3f}")
    assert cos_raw > 0.95, "不缩放的症状: 两个时间几乎同一个嵌入"
    assert cos_scaled < 0.9, "t ×1000 之后两个时间的嵌入应能分开 (余弦相似度 < 0.9)"
    assert torch.allclose(noised.t_norm, t * 1000), "scheduler 输出的 t_norm 应等于 t × 1000"

    # 2) x_t 与 velocity 的定义
    #    t.view(2, 1, 1, 1): 把 [B] 的 t 变成 [B, 1, 1, 1], 好和 [B, C, H, W] 广播
    assert torch.allclose(noised.noisy, (1 - t.view(2, 1, 1, 1)) * x0 + t.view(2, 1, 1, 1) * noised.noise), \
        "x_t 应等于 (1 - t)·x_0 + t·ε"
    assert torch.allclose(noised.target, noised.noise - x0), "velocity 目标应等于 ε - x_0"

    text_embeds, text_pooled = torch.randn(2, 16, 64), torch.randn(2, 64)  # 逐 token [B, 16, 64] 和整句 [B, 64]
    with torch.inference_mode():
        pred_v = model(noised.noisy, noised.t_norm, text_embeds, text_pooled)
        assert pred_v.shape == x0.shape, "MM-DiT 输出形状应与输入 latent 相同"
        assert pred_v.abs().max() == 0, "零初始化起点: 未训练的 MM-DiT 输出应恒为 0"

        # 3) 模拟训练后 (随机化零初始化层): 文本 token 经联合注意力影响图像输出
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                nn.init.normal_(m.weight, std=0.02)
        base = model(noised.noisy, noised.t_norm, text_embeds, text_pooled)
        d_txt = (model(noised.noisy, noised.t_norm, text_embeds.flip(0), text_pooled) - base).abs().max().item()
        d_t = (model(noised.noisy, noised.t_norm.flip(0), text_embeds, text_pooled) - base).abs().max().item()
        print(f"换文本 token 输出变化 {d_txt:.3e} | 换 t 输出变化 {d_t:.3e}")
        assert d_txt > 1e-5, "文本 token 没有影响图像输出: 联合注意力没通"
        assert d_t > 1e-5, "时间 t 没有影响输出: 条件通路断了"

    # 4) Euler 采样: 积分用 t∈[0,1], 喂模型的 t 与训练同量纲 (1000 → 50)
    seen_t = []

    def denoiser(x, t_model, _labels):
        """把 MM-DiT 包成采样器要的 (x, t, labels) 签名, 顺手记下采样器传进来的 t。"""
        seen_t.append(t_model[0].item())
        return model(x, t_model, text_embeds[:1], text_pooled[:1])

    euler = EulerFlowSampler(num_inference_steps=20)
    x_gen = euler.sample(denoiser, shape=(1, 4, 8, 8), device=torch.device("cpu"))
    assert abs(seen_t[0] - 1000.0) < 1e-3, "第一步喂给模型的 t 应是 1000 (t = 1, 纯噪声)"
    assert abs(seen_t[-1] - 50.0) < 1e-3, "20 步时最后一步喂给模型的 t 应是 50 (t = 1/20)"
    assert len(seen_t) == 20, "20 步 Euler 采样应前向 20 次 (没有 CFG)"
    assert torch.isfinite(x_gen).all(), "Euler 采样结果应全部有限"
    print(f"Euler {euler.num_inference_steps} 步, 模型看到的 t: {seen_t[0]:.0f} → {seen_t[-1]:.0f}")


if __name__ == "__main__":
    main()
