#!/usr/bin/env python
"""
DiT 结构自检: adaLN-Zero 起点、patchify/unpatchify 互逆、条件通路、DDIM + CFG 采样
"""

import torch
import torch.nn as nn

from llm_models.models.generative.dit import DiT
from llm_models.training.diffusion import DDIMSampler, DDPMScheduler


def main():
    torch.manual_seed(42)

    model = DiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=128, n_heads=4, num_layers=4,
        num_classes=10, class_dropout=0.1,
    ).eval()
    print(f"DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    x0 = torch.randn(2, 4, 8, 8)                                       # [B, C, H, W] 干净 latent
    noised = scheduler.add_noise(x0, torch.tensor([100, 900]))         # 两条样本分别加噪到第 100、900 步
    y = torch.tensor([3, 7])                                           # 类别标签

    with torch.inference_mode():
        # 1) adaLN-Zero + FinalLayer 零初始化: 未训练的 DiT 输出恒为 0
        pred = model(noised.noisy, noised.t_norm, y)
        assert pred.shape == x0.shape, "DiT 输出形状应与输入 latent 相同"
        assert pred.abs().max() == 0, "adaLN-Zero + FinalLayer 零初始化: 未训练的 DiT 输出应恒为 0"
        print("未训练 DiT 的输出全 0 (adaLN-Zero 起点) ✔")

        # 2) unpatchify 是 "切 patch" 的逆: [B,C,H,W] → [B,N,p²C] → [B,C,H,W]
        B, C, g, p = 2, 4, model.grid_size, model.patch_size           # g: 每边几个 patch, p: patch 边长
        # 手工切 patch: [B, C, g, p, g, p] → [B, g, g, p, p, C] → [B, g², p²C]
        tokens = x0.view(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)
        assert torch.equal(model.unpatchify(tokens), x0), "unpatchify 应是切 patch 的逆: 切完再拼回去要逐位等于原 latent"

        # 3) 把零初始化的层随机化 (模拟训练后), 条件 t / y 才会真正影响输出
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                nn.init.normal_(m.weight, std=0.02)
        base = model(noised.noisy, noised.t_norm, y)
        # flip(0): 把两条样本的条件对调, 输入 x_t 不动
        d_t = (model(noised.noisy, noised.t_norm.flip(0), y) - base).abs().max().item()
        d_y = (model(noised.noisy, noised.t_norm, y.flip(0)) - base).abs().max().item()
        print(f"换 t 输出变化 {d_t:.3e} | 换 y 输出变化 {d_y:.3e}")
        assert d_t > 1e-4, "时间步 t 没有影响输出: 条件通路断了"
        assert d_y > 1e-4, "类别 y 没有影响输出: 条件通路断了"

    # 4) DDIM + CFG: 记录采样器喂给模型的 t, 必须是 [0, 999] 的原始步数 (没有被再 ×1000)
    seen_t = []

    def spy(x, t, labels):
        """包一层模型: 前向之前先记下采样器传进来的 t。"""
        seen_t.append(t[0].item())
        return model(x, t, labels)

    # clip_x0: ᾱ_999 ≈ 2.4e-9, 不截断的话第一步就把数值放大 ~2 万倍
    sampler = DDIMSampler(scheduler, num_inference_steps=20, clip_x0=3.0)
    raw = DDIMSampler(scheduler, num_inference_steps=20).sample(
        model, shape=(1, 4, 8, 8), device=torch.device("cpu"))
    kwargs = dict(shape=(1, 4, 8, 8), device=torch.device("cpu"),
                  class_labels=torch.tensor([3]), null_class_id=model.null_class_idx)
    # 两次采样用同一个种子: 起始噪声相同, 结果的差别只来自 guidance_scale
    torch.manual_seed(0)
    x_cfg = sampler.sample(spy, guidance_scale=4.0, **kwargs)
    torch.manual_seed(0)
    x_plain = sampler.sample(model, guidance_scale=1.0, **kwargs)
    assert max(seen_t) == 999, "采样器喂给模型的最大 t 应是 999 (原始步数, 没有被再 ×1000)"
    assert min(seen_t) == 0, "采样器喂给模型的最小 t 应是 0"
    assert len(seen_t) == 2 * 20, "CFG 每步前向 2 次 (有条件 + 无条件), 20 步共 40 次"
    assert x_cfg.abs().max() <= 3.0, "clip_x0=3 时采样结果的绝对值不应超过 3"
    assert not torch.allclose(x_cfg, x_plain), "guidance_scale=4 应改变采样结果 (与 scale=1 不同)"
    print(f"|x|max: 不截断 {raw.abs().max():.1f} | clip_x0={sampler.clip_x0:g} → {x_cfg.abs().max():.2f}")
    print(f"DDIM {sampler.num_inference_steps} 步, 模型前向 {len(seen_t)} 次 (CFG ×2), "
          f"t: {max(seen_t):.0f} → {min(seen_t):.0f}")


if __name__ == "__main__":
    main()
