#!/usr/bin/env python
"""
Byte Latent Transformer 训练示例 — 同样的全局步数, 按熵切 patch 和按固定长度切 patch 哪个 bits-per-byte 更低。

数据: 合成字节文本。18 个词 (12 个英文词各 7 字节, 6 个中文词各 3 个汉字 = 9 个 UTF-8 字节) 用空格连接,
      下一个词从当前词固定的 3 个后继里均匀抽 (一阶马尔可夫链)。每条样本是长 T 的字节窗口, 起点随机。
      词首字节难猜 (要知道上一个词, 还剩 3 选 1), 词内字节几乎确定, 正好对应 "熵高处开新 patch"。
      植入的结构: 每个词的 3 个后继共用同一个词尾, 只有词首不同, 所以 "上一个词是谁" 只写在它的词首里。
      局部模型的视野 (约 5 字节) 只够看到上一个词的词尾和空格, 这条信息必须经全局模型传过来。
对照: 先训一个小熵模型, 选阈值 θ 让平均 patch 长度 ≈ K, 再和每 K 个字节一刀的固定切分比较。
      两边全局模型的步数 (patch 数) 基本相同, 模型、数据、步数、种子都相同, 只有切分方式不同。
为什么不用 Trainer: 每步要先用冻结的熵模型算切分, 再喂给主模型, Trainer 的 (batch → model → loss) 管线放不下。
读代码时盯住: pick_threshold 怎样让两种切分的全局步数对齐。
"""

import math
import random
import time

import torch
import torch.nn.functional as F

from llm_models.models.language_models.byte_latent import (
    ByteLatentLM,
    bits_per_byte,
    entropy_patch_ids,
    fixed_patch_ids,
    next_byte_entropy,
)
from llm_models.models.language_models.llama import LLaMA

T = 64          # 每条样本的字节数
K = 8           # 固定切分的 patch 长度; 熵切分的阈值按它对齐 (一个词加空格平均约 8.7 字节)
# 词 = 词首 + 词尾。英文词 7 字节; 中文词 3 个汉字, UTF-8 下 9 字节 (三个词首汉字的首字节各不相同)。
HEADS = {"en": ("red", "big", "hot", "old"), "zh": ("大", "老", "中")}
TAILS = {"en": ("wolf", "bear", "lynx"), "zh": ("模型", "字节")}
WORDS = [h + t for lang in HEADS for h in HEADS[lang] for t in TAILS[lang]]
# 刻意植入的结构 (见 README): 每个词的 3 个后继共用同一个词尾, 只在词首上不同。
# 于是 "上一个词是谁" 只写在它的词首里, 光看词尾 (局部模型的视野) 分辨不出来。
_succ_rng = random.Random(0)
SUCC = {}
for _w in WORDS:
    _lang = _succ_rng.choice(["en", "en", "zh"])
    _tail = _succ_rng.choice(TAILS[_lang])
    SUCC[_w] = [h + _tail for h in _succ_rng.sample(HEADS[_lang], 3)]


def make_batch(batch_size: int, rng: random.Random) -> torch.Tensor:
    """按马尔可夫链生成词序列, UTF-8 编码后随机截一段长 T 的字节窗口 → [B, T]。"""
    rows = []
    for _ in range(batch_size):
        words = [rng.choice(WORDS)]
        while len(" ".join(words).encode()) < T + 8:
            words.append(rng.choice(SUCC[words[-1]]))
        off = rng.randrange(8)                               # 窗口起点随机, 开头常常落在词中间
        rows.append(list(" ".join(words).encode()[off:off + T]))
    return torch.tensor(rows)


def train_entropy_model(steps: int = 400, seed: int = 0, log: bool = True) -> LLaMA:
    """熵模型: 1 层、d_model=32 的字节级 LLaMA, 普通下一字节预测。训练后冻结, 只用来切分。"""
    torch.manual_seed(seed)
    model = LLaMA(vocab_size=256, d_model=32, n_heads=2, num_layers=1, max_len=T)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed + 1)
    for step in range(1, steps + 1):
        model.train()
        x = make_batch(32, rng)                              # 每步新数据
        loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (step == 1 or step == steps):
            print(f"熵模型第 {step:4d} 步 | loss {loss.item():.3f}")
    model.eval()
    return model


def mean_patch_len(patch_ids: torch.Tensor) -> float:
    """平均每个 patch 多少字节 = 字节数 / patch 数, 按整个 batch 汇总。"""
    return patch_ids.numel() / (patch_ids[:, -1] + 1).sum().item()


def pick_threshold(entropy: torch.Tensor, target_len: float) -> float:
    """在 0.05 的网格上找让平均 patch 长度最接近 target_len 的阈值 (patch 数随阈值单调, 网格扫一遍就够)。"""
    grid = [i * 0.05 for i in range(1, 61)]
    return min(grid, key=lambda th: abs(mean_patch_len(entropy_patch_ids(entropy, th)) - target_len))


def train_blt(patcher, steps: int = 600, seed: int = 0, log_name: str = "") -> ByteLatentLM:
    """patcher: x [B, T] → patch_ids [B, T]。两种切分用同一个种子、同一串数据。"""
    torch.manual_seed(seed)
    model = ByteLatentLM(d_model=64, n_heads=4, global_layers=2, max_len=T)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed + 2)
    for step in range(1, steps + 1):
        model.train()
        x = make_batch(32, rng)
        loss = F.cross_entropy(model(x, patcher(x))[:, :-1].flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f"  [{log_name}] 第 {step:4d} 步 | loss {loss.item():.3f}")
    return model


@torch.no_grad()
def eval_bpb(model: ByteLatentLM, x: torch.Tensor, patch_ids: torch.Tensor) -> float:
    model.eval()
    return bits_per_byte(model(x, patch_ids), x)


def main():
    t0 = time.time()
    entropy_model = train_entropy_model()
    held_out = make_batch(256, random.Random(999))           # 留出集: 训练从没见过这些窗口

    H = next_byte_entropy(entropy_model, held_out)
    fix_pid = fixed_patch_ids(*held_out.shape, K)
    theta = pick_threshold(H, mean_patch_len(fix_pid))      # 让两种切分的 patch 数 (全局步数) 对齐
    ent_pid = entropy_patch_ids(H, theta)
    ent_len, fix_len = mean_patch_len(ent_pid), mean_patch_len(fix_pid)
    print(f"\n阈值 θ = {theta:.2f}: 熵切分平均 {ent_len:.2f} 字节/patch | 固定切分 {fix_len:.2f} 字节/patch")
    assert abs(ent_len - fix_len) < 0.3, "两种切分的全局步数应基本对齐, 否则比较的不是同等全局计算"

    patch_entropy = lambda x: entropy_patch_ids(next_byte_entropy(entropy_model, x), theta)  # noqa: E731
    patch_fixed = lambda x: fixed_patch_ids(*x.shape, K)                                     # noqa: E731

    # ---- 初始 loss: init_weights 让 logits ≈ 0, CE ≈ ln 256 ----
    torch.manual_seed(0)
    fresh = ByteLatentLM(d_model=64, n_heads=4, global_layers=2, max_len=T)
    init_bpb = eval_bpb(fresh, held_out, ent_pid)
    print(f"初始 {init_bpb * math.log(2):.3f} nats/字节  vs  ln 256 = {math.log(256):.3f}")
    assert abs(init_bpb * math.log(2) - math.log(256)) < 0.3, "初始 CE 应 ≈ ln 256"

    print("\n训练 (同一种子、同一串数据, 只换切分):")
    m_ent = train_blt(patch_entropy, log_name="熵切分")
    m_fix = train_blt(patch_fixed, log_name=f"固定 {K}")
    m_byte = train_blt(lambda x: fixed_patch_ids(*x.shape, 1), log_name="每字节一个 patch")
    bpb_ent = eval_bpb(m_ent, held_out, ent_pid)
    bpb_fix = eval_bpb(m_fix, held_out, fix_pid)
    bpb_byte = eval_bpb(m_byte, held_out, fixed_patch_ids(*held_out.shape, 1))

    print(f"\n留出集 bits-per-byte (256 条 × {T} 字节):")
    print(f"  熵切分          | {ent_len:.2f} 字节/patch | 全局步数/字节 {1 / ent_len:.3f} | BPB {bpb_ent:.3f}")
    print(f"  固定 {K} 字节切分 | {fix_len:.2f} 字节/patch | 全局步数/字节 {1 / fix_len:.3f} | BPB {bpb_fix:.3f}")
    print(f"  每字节一个 patch | 1.00 字节/patch | 全局步数/字节 1.000 | BPB {bpb_byte:.3f}")
    for name, bpb in [("熵切分", bpb_ent), ("固定切分", bpb_fix), ("每字节一个 patch", bpb_byte)]:
        assert bpb < 0.5 * init_bpb, f"{name}: BPB {bpb:.3f} 应明显低于初始 {init_bpb:.3f}"

    print(f"\nByte Latent Transformer 训练验证通过 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
