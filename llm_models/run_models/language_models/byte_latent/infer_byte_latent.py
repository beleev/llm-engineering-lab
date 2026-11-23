#!/usr/bin/env python
"""
Byte Latent Transformer 切分示例 — 熵模型怎样把字节切成 patch, 以及这个切分为什么可以边生成边做。

独立可跑: 内部先训 400 步熵模型 (约 5 s, 复用 train_byte_latent.train_entropy_model), 不依赖 checkpoint。
看什么: 词首字节的熵高、词内字节的熵低, 所以边界大多落在词首; 英文词和中文词字节数不同, 但都大约是一个 patch。
断言: patch 数随阈值单调不增; 全局步数 / 字节数 = 1 / 平均 patch 长度; 改动位置 j 之后的字节不改变 j 及以前的切分,
      也不改变 j 及以前位置的 logits (整个模型因果); 未训练模型的初始 CE ≈ ln 256。
"""

import math
import random

import torch
import torch.nn.functional as F

from llm_models.models.language_models.byte_latent import (
    ByteLatentLM,
    entropy_patch_ids,
    next_byte_entropy,
)
from llm_models.run_models.language_models.byte_latent.train_byte_latent import (
    SUCC,
    T,
    make_batch,
    mean_patch_len,
    train_entropy_model,
)


def show_byte(b: int) -> str:
    """ASCII 字节原样显示, 非 ASCII (UTF-8 多字节汉字的各个字节) 显示成两位十六进制。"""
    return chr(b) if 32 <= b < 127 else f"{b:02X}"


def main():
    entropy_model = train_entropy_model(log=False)
    x = make_batch(256, random.Random(7))                    # [256, T] 留出的字节窗口
    H = next_byte_entropy(entropy_model, x)                  # [256, T], 位置 0 为 +inf

    # ---- 一条从词首开始的样本: 逐字节打印熵和边界 ----
    rng = random.Random(2)
    words = ["redwolf"]
    while len(" ".join(words).encode()) < T:
        words.append(rng.choice(SUCC[words[-1]]))
    sample = torch.tensor([list(" ".join(words).encode()[:T])])
    h = next_byte_entropy(entropy_model, sample)[0]
    theta = 1.2                                              # 高于词尾的歧义熵 (约 0.8), 低于词首的熵 (约 1.6)
    pid = entropy_patch_ids(h[None], theta)[0]
    print(f"样本: {' '.join(words)[:T]!r}")
    print(f"阈值 θ = {theta}: H > θ 处开新 patch, | 是边界")
    line = ""
    for t in range(T):
        line += ("|" if t == 0 or pid[t] != pid[t - 1] else "") + show_byte(int(sample[0, t]))
    print(line)
    print("熵 (nats, 位置 0 无前文记 inf):")
    print("[" + ", ".join("inf" if math.isinf(v) else f"{v:.2f}" for v in h.tolist()) + "]")

    # ---- 1. patch 数随阈值单调不增 ----
    grid = [0.1 * i for i in range(0, 31)]
    counts = [int((entropy_patch_ids(H, th)[:, -1] + 1).sum()) for th in grid]
    print("\n阈值 → 平均字节/patch:")
    print("  " + " | ".join(f"θ={th:.1f}: {x.numel() / c:.2f}" for th, c in zip(grid[::5], counts[::5])))
    assert all(a >= b for a, b in zip(counts, counts[1:])), "阈值越高, 边界越少, patch 数应单调不增"
    assert counts[0] > 2 * counts[-1], "阈值从 0 到 3 应明显改变 patch 数"

    # ---- 2. 全局步数 / 字节数 = 1 / 平均 patch 长度 ----
    pids = entropy_patch_ids(H, theta)
    torch.manual_seed(0)
    model = ByteLatentLM(d_model=64, n_heads=4, global_layers=2, max_len=T)
    with torch.no_grad():
        logits = model(x, pids)
    steps = int(model.last_num_patches.sum())               # 全局模型这次一共走了几步 (所有序列合计)
    ratio = steps / x.numel()
    print(f"\n全局步数 {steps} / 字节数 {x.numel()} = {ratio:.3f}  (平均 {mean_patch_len(pids):.2f} 字节/patch)")
    assert abs(ratio - 1 / mean_patch_len(pids)) < 1e-9, "全局步数 / 字节数应等于 1 / 平均 patch 长度"
    assert ratio < 0.2, "按熵切分后全局模型的步数应远少于字节数"

    # 英文词 7 字节、中文词 9 字节, 但每个词都大约只占一个 patch: 全局步数跟着信息量走, 不跟着字节数走
    for name, lead in [("英文词", lambda b: 97 <= b <= 122), ("中文词", lambda b: b >= 0xE0)]:
        starts = [(i, t) for i in range(x.size(0)) for t in range(1, T - 10)
                  if x[i, t - 1] == 32 and lead(int(x[i, t]))]  # 空格后的第一个字节 = 词首
        n_bytes = 7 if name == "英文词" else 9
        # 词首 t 到词后空格 t+n_bytes 之间开了几个新 patch = pid[t+n_bytes] − pid[t−1]
        per_word = sum(int(pids[i, t + n_bytes] - pids[i, t - 1]) for i, t in starts) / len(starts)
        print(f"  {name}: 每词 {n_bytes} 字节, 加空格 {n_bytes + 1} 字节 | 平均每词 {per_word:.2f} 个 patch ({len(starts)} 个词)")
        assert 0.8 < per_word < 1.5, f"{name}应大约一个词一个 patch"

    # ---- 3. 因果: 改动 j+1 之后的字节, 不改变 j+1 及以前的切分, 也不改变位置 0..j 的 logits ----
    j = 30
    x2 = x.clone()
    x2[:, j + 1:] = torch.randint(0, 256, (x.size(0), T - j - 1), generator=torch.Generator().manual_seed(1))
    pids2 = entropy_patch_ids(next_byte_entropy(entropy_model, x2), theta)
    with torch.no_grad():
        logits2 = model(x2, pids2)
    changed_after = (pids2[:, j + 2:] != pids[:, j + 2:]).any(1).float().mean().item()
    same_pid = bool((pids2[:, :j + 2] == pids[:, :j + 2]).all())
    diff = (logits2[:, :j + 1] - logits[:, :j + 1]).abs().max().item()
    print(f"\n改动位置 {j + 1} 之后的字节: 位置 ≤ {j + 1} 的切分不变 {same_pid} | 之后切分变了的样本比例 {changed_after:.2f}"
          f" | 位置 ≤ {j} 的 logits 最大差 {diff:.1e}")
    assert same_pid, "边界 b_t 只看 H(x_t), 而 H(x_t) 只由 x_<t 算出: 改未来字节不应改变已经切好的 patch"
    assert changed_after > 0.5, "改动之后的字节应当改变之后的切分 (否则这个检查没有区分力)"
    assert diff < 1e-5, "位置 t 只能读 x_{t+1} 所在 patch 之前的全局输出: 改未来字节不应改变过去的 logits"

    # ---- 4. 初始 CE ≈ ln 256 ----
    init_ce = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    print(f"\n初始 CE {init_ce:.3f}  vs  ln 256 = {math.log(256):.3f}")
    assert abs(init_ce - math.log(256)) < 0.3, "init_weights 让初始 logits ≈ 0, CE 应 ≈ ln 256"

    print("\nByte Latent Transformer 切分验证通过")


if __name__ == "__main__":
    main()
