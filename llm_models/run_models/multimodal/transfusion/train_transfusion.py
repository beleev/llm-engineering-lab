#!/usr/bin/env python
"""
Transfusion 训练示例 — 一个模型在 8×8 小图上同时学 "看图说话" 和 "按文本画图"。

数据: 3 种颜色 (red / green / blue) × 3 种形状 (circle 圆环 / square 实心方块 / cross 十字), 位置抖动 ±1 像素,
      亮度在 [0.6, 1] 之间随机, 再加 N(0, 0.05²) 的像素噪声; 2×2 切 patch → 16 个图像位置, 每个 patch 12 维。
两种样本各占一半 (论文是 80% 文本在前、20% 图像在前):
    文本在前: [BOS, 颜色, 形状, BOI, 16 个 patch, EOI]    图像按文本生成, t ~ U(0, 1)
    图像在前: [BOS, BOI, 16 个 patch, EOI, 颜色, 形状]    文本读图, t ~ U(0, 0.5) (论文在这类样本上把噪声限制在 t ≤ 500/1000)
文本损失只算颜色词和形状词两个目标; 图像损失两类样本的图都算。每步抽新数据, 评估都在留出样本上。
判定器: 前景像素 (最大通道 > 0.3) 的平均颜色取最大通道定颜色, 前景形状与 3 种形状在 9 个位置上的模板比 IoU 定形状。
读代码时盯住: make_batch 里两种布局的 image_id 和 lm_mask。
"""

import math
import time

import torch

from llm_models.models.multimodal.transfusion import TransfusionModel, transfusion_loss
from llm_models.training.diffusion import EulerFlowSampler, FlowMatchingScheduler

COLORS, SHAPES = ["red", "green", "blue"], ["circle", "square", "cross"]
VOCAB = ["<bos>", "<boi>", "<eoi>"] + COLORS + SHAPES
BOS, BOI, EOI = 0, 1, 2
C0, S0 = 3, 6                     # 颜色词 / 形状词在词表里的起始 id
N_PATCH, PATCH_DIM = 16, 12       # 8×8 图, 2×2 patch: 16 个 patch, 每个 2·2·3 = 12 维
SEQ = 3 + N_PATCH + 2             # 两种布局都是 21 个位置
IMG_START = {"text_first": 4, "image_first": 2}  # 图像段从第几个位置开始
SCHED = FlowMatchingScheduler()


def shape_mask(shape: int, dy: int, dx: int) -> torch.Tensor:
    """8×8 的前景 0/1 图。dy, dx ∈ {-1, 0, 1} 是位置抖动。"""
    yy, xx = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    if shape == 0:                                   # 圆环: 以格点 (3.5, 3.5) 为心, 半径 1.6~2.8 之间
        r = ((yy - 3.5 - dy) ** 2 + (xx - 3.5 - dx) ** 2).sqrt()
        return ((r > 1.6) & (r < 2.8)).float()
    if shape == 1:                                   # 实心 4×4 方块
        return ((yy >= 2 + dy) & (yy < 6 + dy) & (xx >= 2 + dx) & (xx < 6 + dx)).float()
    cy, cx = 3 + dy, 4 + dx                          # 十字: 两条长 5 的线
    return (((yy == cy) & ((xx - cx).abs() <= 2)) | ((xx == cx) & ((yy - cy).abs() <= 2))).float()


TEMPLATES = torch.stack([torch.stack([shape_mask(s, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
                         for s in range(3)])        # [3 形状, 9 位置, 8, 8]


def make_images(color: torch.Tensor, shape: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """color / shape [B] → 图 [B, 3, 8, 8], 取值约在 [0, 1]。"""
    B = color.size(0)
    pos = torch.randint(0, 9, (B,), generator=g)
    fg = TEMPLATES[shape, pos]                                           # [B, 8, 8]
    bright = 0.6 + 0.4 * torch.rand(B, generator=g)
    img = torch.zeros(B, 3, 8, 8)
    img[torch.arange(B), color] = fg * bright[:, None, None]             # 只亮颜色对应的那个通道
    return img + 0.05 * torch.randn(B, 3, 8, 8, generator=g)


def patchify(img: torch.Tensor) -> torch.Tensor:
    """[B, 3, 8, 8] → [B, 16, 12], 按行扫描 4×4 个 2×2 块。x_0 = 2·img − 1, 背景约为 −1。"""
    B = img.size(0)
    x = (2 * img - 1).view(B, 3, 4, 2, 4, 2).permute(0, 2, 4, 3, 5, 1)  # [B, 4, 4, 2, 2, 3]
    return x.reshape(B, N_PATCH, PATCH_DIM)


def unpatchify(p: torch.Tensor) -> torch.Tensor:
    """patchify 的逆: [B, 16, 12] → [B, 3, 8, 8], 取值换回 [0, 1] 附近。"""
    B = p.size(0)
    x = p.view(B, 4, 4, 2, 2, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, 8, 8)
    return (x + 1) / 2


def classify(img: torch.Tensor):
    """判定器: [B, 3, 8, 8] → (颜色 [B], 形状 [B])。"""
    fg = (img.max(1).values > 0.3).float()                               # [B, 8, 8] 前景
    color = ((img * fg[:, None]).sum((2, 3)) / fg.sum((1, 2)).clamp(min=1)[:, None]).argmax(1)
    inter = (fg[:, None, None] * TEMPLATES[None]).sum((3, 4))           # [B, 3, 9]
    union = (fg[:, None, None] + TEMPLATES[None]).clamp(max=1).sum((3, 4))
    shape = (inter / union).amax(2).argmax(1)                            # 每种形状取 9 个位置里最像的, 再比形状
    return color, shape


def layout(kind: str, B: int):
    """返回 tokens [B, S] (图像位置填 0)、image_id [B, S]、lm_mask [B, S-1]、caption 两个词的位置。"""
    tokens = torch.zeros(B, SEQ, dtype=torch.long)
    image_id = torch.full((B, SEQ), -1)
    a = IMG_START[kind]
    image_id[:, a:a + N_PATCH] = 0
    tokens[:, 0] = BOS
    tokens[:, a - 1], tokens[:, a + N_PATCH] = BOI, EOI
    cap = [1, 2] if kind == "text_first" else [SEQ - 2, SEQ - 1]       # 颜色词、形状词的位置
    lm_mask = torch.zeros(B, SEQ - 1, dtype=torch.bool)
    lm_mask[:, [c - 1 for c in cap]] = True                              # 位置 c-1 预测位置 c 的词
    return tokens, image_id, lm_mask, cap


def make_batch(kind: str, B: int, g: torch.Generator, t: torch.Tensor = None):
    """一种布局的一批样本, 图像已按 flow matching 加噪。返回模型输入、loss 输入和真值。"""
    color, shape = torch.randint(0, 3, (B,), generator=g), torch.randint(0, 3, (B,), generator=g)
    tokens, image_id, lm_mask, cap = layout(kind, B)
    tokens[:, cap[0]], tokens[:, cap[1]] = C0 + color, S0 + shape
    x0 = patchify(make_images(color, shape, g))                          # [B, 16, 12]
    if t is None:                                                        # 图像在前的样本把噪声限制在 t ≤ 0.5
        t = torch.rand(B, generator=g) * (1.0 if kind == "text_first" else 0.5)
    noise = torch.randn(x0.shape, generator=g)
    tb = t[:, None, None]
    a = IMG_START[kind]
    patches = torch.zeros(B, SEQ, PATCH_DIM)
    target = torch.zeros(B, SEQ, PATCH_DIM)
    patches[:, a:a + N_PATCH] = (1 - tb) * x0 + tb * noise               # x_t = (1 − t)·x_0 + t·ε
    target[:, a:a + N_PATCH] = noise - x0                                # v = ε − x_0
    batch = dict(tokens=tokens, patches=patches, image_id=image_id, t=t * SCHED.num_train_timesteps)
    return batch, dict(target_v=target, lm_mask=lm_mask), (color, shape)


def step_loss(model, kind, B, g, t=None):
    batch, lbl, _ = make_batch(kind, B, g, t)
    return transfusion_loss(model(**batch), batch["tokens"], lbl["target_v"], batch["image_id"], lbl["lm_mask"])


@torch.no_grad()
def eval_losses(model):
    """固定留出样本 + 固定噪声: 看图说话 CE (图像在前, t=0)、无图可看的文本 CE、图像去噪 MSE。"""
    model.eval()
    g = torch.Generator().manual_seed(123)
    cap = step_loss(model, "image_first", 512, g, t=torch.zeros(512))["lm_loss"].item()
    tf = step_loss(model, "text_first", 512, g)
    return cap, tf["lm_loss"].item(), tf["image_loss"].item()


def build_model(seed: int = 0) -> TransfusionModel:
    torch.manual_seed(seed)
    return TransfusionModel(vocab_size=len(VOCAB), patch_dim=PATCH_DIM, d_model=64, n_heads=4, num_layers=3, max_len=32)


def train(steps: int = 3000, seed: int = 0, log_interval: int = 500):
    model = build_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        model.train()
        # 两种布局各 32 条; 布局不同, 分两次前向再把 loss 平均
        lt, li = step_loss(model, "text_first", 32, g), step_loss(model, "image_first", 32, g)
        loss = (lt["total_loss"] + li["total_loss"]) / 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % log_interval == 0:
            lm = (lt["lm_loss"].item() + li["lm_loss"].item()) / 2
            im = (lt["image_loss"].item() + li["image_loss"].item()) / 2
            print(f"第 {step:4d} 步 | 文本 CE {lm:.3f} | 图像 MSE {im:.3f}")
    return model


@torch.no_grad()
def caption(model, img: torch.Tensor):
    """看图说话: [BOS, BOI, 干净图, EOI] 之后贪心解码两个词。返回 (颜色 [B], 形状 [B])。"""
    model.eval()
    B = img.size(0)
    tokens, image_id, _, _ = layout("image_first", B)
    patches = torch.zeros(B, SEQ, PATCH_DIM)
    patches[:, 2:2 + N_PATCH] = patchify(img)
    words = []
    for pos in (SEQ - 2, SEQ - 1):           # 后面还没填的位置看不到, 因果掩码保证它们不影响前面
        logits = model(tokens, patches, image_id, torch.zeros(B))["logits"][:, pos - 1]
        tokens[:, pos] = logits.argmax(-1)
        words.append(tokens[:, pos])
    return words[0] - C0, words[1] - S0       # 非颜色 / 形状词会落到范围外, 算答错


@torch.no_grad()
def generate(model, color: torch.Tensor, shape: torch.Tensor, steps: int = 20, seed: int = 0) -> torch.Tensor:
    """按文本画图: [BOS, 颜色, 形状, BOI, 噪声 patch, EOI], Euler 积分 steps 步。返回图 [B, 3, 8, 8]。"""
    model.eval()
    B = color.size(0)
    tokens, image_id, _, _ = layout("text_first", B)
    tokens[:, 1], tokens[:, 2] = C0 + color, S0 + shape

    def denoiser(x, t, _labels):              # EulerFlowSampler 要的签名: (x_t, t, 条件) → velocity
        patches = torch.zeros(B, SEQ, PATCH_DIM)
        patches[:, 4:4 + N_PATCH] = x
        return model(tokens, patches, image_id, t)["velocity"][:, 4:4 + N_PATCH]   # 每步整段前向

    torch.manual_seed(seed)
    x0 = EulerFlowSampler(num_inference_steps=steps).sample(denoiser, (B, N_PATCH, PATCH_DIM), torch.device("cpu"))
    return unpatchify(x0)


def ascii_image(img: torch.Tensor) -> list:
    """一张图 [3, 8, 8] → 8 行字符: 前景写颜色首字母, 背景写 '.'。"""
    fg = img.max(0).values > 0.3
    ch = img.argmax(0)
    return ["".join("RGB"[ch[y, x]] if fg[y, x] else "." for x in range(8)) for y in range(8)]


def main():
    t0 = time.time()
    V = len(VOCAB)

    # ---- 判定器先在真图上验收, 否则后面的命中率没有意义 ----
    g = torch.Generator().manual_seed(999)
    c_true, s_true = torch.randint(0, 3, (600,), generator=g), torch.randint(0, 3, (600,), generator=g)
    real = make_images(c_true, s_true, g)
    c_hat, s_hat = classify(real)
    judge_c, judge_s = (c_hat == c_true).float().mean().item(), (s_hat == s_true).float().mean().item()
    print(f"判定器在留出真图上: 颜色 {judge_c:.3f} | 形状 {judge_s:.3f}")
    assert judge_c > 0.98, "判定器颜色准确率太低, 命中率不可信"
    assert judge_s > 0.98, "判定器形状准确率太低, 命中率不可信"

    fresh = build_model()
    cap0, tf0, img0 = eval_losses(fresh)
    print(f"初始 | 看图说话 CE {cap0:.3f} | 图前文本 CE {tf0:.3f} | 图像 MSE {img0:.3f}  (ln V = {math.log(V):.3f})")
    # 实测比 ln V 高约 0.25: weight tying 让预测位置的 h 接近它自己的 embedding, 初始最偏爱输出 "自己"
    # (logit ≈ 1.3, CE ≈ ln(V − 1 + e^1.3) ≈ 2.46), 同 llada 的 train 脚本
    assert abs(cap0 - math.log(V)) < 0.35, "初始文本 CE 应 ≈ ln V"

    model = train()
    cap1, tf1, img1 = eval_losses(model)
    print(f"训练后 | 看图说话 CE {cap1:.3f} | 图前文本 CE {tf1:.3f} (无图可看的下界 ln 3 = {math.log(3):.3f}) | 图像 MSE {img1:.3f}")
    assert cap1 < 0.5 * cap0, "看图说话的文本损失应明显下降"
    assert img1 < 0.7 * img0, "图像去噪损失应明显下降"
    # 图像前的文本看不到图: 颜色词和形状词各有 3 种等概率取值, CE 降不到 ln 3 以下
    assert tf1 > math.log(3) - 0.05, "图像前的文本不应读到后面的图像"

    # ---- 看图说话: 留出真图 ----
    c_cap, s_cap = caption(model, real)
    acc_c, acc_s = (c_cap == c_true).float().mean().item(), (s_cap == s_true).float().mean().item()
    acc_j = ((c_cap == c_true) & (s_cap == s_true)).float().mean().item()
    print(f"\n看图说话 (留出 {len(c_true)} 张): 颜色 {acc_c:.3f} | 形状 {acc_s:.3f} | 两个都对 {acc_j:.3f}  (随机 0.333 / 0.333 / 0.111)")
    assert acc_c > 1 / 3 + 0.2, "看图说话的颜色准确率应高于随机"
    assert acc_s > 1 / 3 + 0.2, "看图说话的形状准确率应高于随机"
    assert acc_j > 1 / 9 + 0.2, "看图说话两个都对的比例应高于随机"

    # ---- 按文本画图: 9 种组合各 20 张 ----
    c_req = torch.arange(9).repeat_interleave(20) // 3
    s_req = torch.arange(9).repeat_interleave(20) % 3
    gen = generate(model, c_req, s_req)
    c_gen, s_gen = classify(gen)
    hit_c, hit_s = (c_gen == c_req).float().mean().item(), (s_gen == s_req).float().mean().item()
    hit_j = ((c_gen == c_req) & (s_gen == s_req)).float().mean().item()
    print(f"按文本画图 ({len(c_req)} 张, Euler 20 步): 颜色 {hit_c:.3f} | 形状 {hit_s:.3f} | 两个都对 {hit_j:.3f}  (随机 0.333 / 0.333 / 0.111)")
    per_shape = " | ".join(f"{SHAPES[k]} {(s_gen[s_req == k] == k).float().mean().item():.3f}" for k in range(3))
    print(f"  形状命中率按要求的形状拆开: {per_shape}")
    assert hit_c > 1 / 3 + 0.2, "生成图的颜色命中率应高于随机"
    assert hit_s > 1 / 3 + 0.2, "生成图的形状命中率应高于随机"
    assert hit_j > 1 / 9 + 0.2, "生成图两个属性都对的比例应高于随机"

    print("\n生成样例 (R/G/B = 前景颜色, . = 背景):")
    picks = [0, 80, 160]                       # red circle / green square / blue cross
    rows = [ascii_image(gen[i]) for i in picks]
    print("   ".join(f"{COLORS[c_req[i]] + ' ' + SHAPES[s_req[i]]:<12}" for i in picks))
    for y in range(8):
        print("   ".join(f"{r[y]:<12}" for r in rows))

    print(f"\nTransfusion 训练验证通过 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
