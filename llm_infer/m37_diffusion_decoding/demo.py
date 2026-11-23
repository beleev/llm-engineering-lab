"""
m37 demo — 扩散语言模型的解码: 步数与质量、置信度阈值、近似 KV cache

    [1] 固定每步揭开 k 个: 步数越少, 相关短语拼错的比例越高; 全部独立的位置怎么并行都不错
    [2] 置信度阈值: 独立位置一步揭开大半, 相关短语自动退回一次一个; 阈值太低 (≤ 0.5) 照样拼错
    [3] 近似 KV cache (DualCache 式): 每 r 步刷新一次, 误差和决策不一致率随 r 上升
    [4] 块扩散 (块间因果): 写完的块 K/V 不再变, 缓存与每步完整前向一致

前两节用合成 Oracle (精确条件分布, 错误全来自解码策略); 后两节用随机权重的 3 层双向小模型。

运行: python -m llm_infer.m37_diffusion_decoding.demo
"""
from __future__ import annotations

import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m37_diffusion_decoding.diffusion_decode import (
    BiLM, block_cache_error, decode, dual_cache_error, make_task, trajectory)

N_TASK, L = 200, 32
WORKLOADS = (("全部独立", 0.0), ("混合", 0.15), ("全部相关", 1.0))


def rj(s, w):
    """按显示宽度右对齐 (中文占两格)。"""
    return " " * max(0, w - len(s) - sum(ord(c) > 0x2E80 for c in s)) + s


def evaluate(tasks, mode, **kw):
    """→ (平均步数, 平均每步揭开数, 多 token 短语拼错的比例)。"""
    steps = bad = total = 0
    for o in tasks:
        x, per_step = decode(o, mode, **kw)
        b, t = o.invalid_groups(x)
        steps, bad, total = steps + len(per_step), bad + b, total + t
    return steps / len(tasks), L * len(tasks) / steps, bad / max(total, 1)


def main():
    banner("M37 - 扩散语言模型的并行解码与近似缓存")
    tasks = {name: [make_task(np.random.RandomState(i), L=L, p_group=pg) for i in range(N_TASK)] for name, pg in WORKLOADS}
    kv("每条回答长度 / 条数", f"{L} 个位置 / 每种负载 {N_TASK} 条")
    mixed = [g for o in tasks["混合"] for g in o.groups if g.phrases.shape[1] > 1]
    kv("混合负载: 相关短语", f"{len(mixed) / N_TASK:.1f} 组/条, 占 {sum(g.phrases.shape[1] for g in mixed) / (N_TASK * L):.0%} 的位置")

    print("\n[1] 固定每步揭开 k 个 (置信度最高的 k 个)")
    print("  " + rj("k", 4) + rj("步数", 8) + "".join(rj(n + " 拼错率", 18) for n, _ in WORKLOADS))
    fixed = {}
    for k in (1, 2, 4, 8, 32):
        r = {name: evaluate(tasks[name], "fixed", k=k) for name, _ in WORKLOADS}
        fixed[k] = r
        print(f"  {k:>4}{r['混合'][0]:>8.0f}" + "".join(f"{r[n][2]:>18.1%}" for n, _ in WORKLOADS))
        assert r["全部独立"][2] == 0, f"k={k}: 全部独立的位置, 并行揭开不应拼错"
    assert fixed[1]["混合"][2] == 0, "k=1 一次一个, 每步都在精确条件分布上取值, 不应拼错"
    ks = sorted(fixed)
    for a, b in zip(ks, ks[1:]):
        assert fixed[b]["混合"][2] >= fixed[a]["混合"][2], "步数越少 (k 越大), 拼错率不应下降"
    assert fixed[32]["混合"][2] > 0.3, "一步全揭开时, 相关短语应大量拼错"

    print("\n[2] 置信度阈值 τ (过阈值的全揭开, 都不过就只揭开最高的 1 个)")
    print("  " + rj("τ", 5) + "".join(rj(n + " 每步揭开", 20) + rj("拼错率", 10) for n, _ in WORKLOADS))
    thr = {}
    for tau in (0.3, 0.5, 0.7, 0.9):
        r = {name: evaluate(tasks[name], "threshold", tau=tau) for name, _ in WORKLOADS}
        thr[tau] = r
        print(f"  {tau:>5}" + "".join(f"{r[n][1]:>20.2f}{r[n][2]:>10.1%}" for n, _ in WORKLOADS))
    for tau in (0.7, 0.9):
        for n, _ in WORKLOADS:
            assert thr[tau][n][2] == 0, f"τ={tau} 在「{n}」上不应拼错 (三角短语的置信度约 2/3, 会退回一次一个)"
    assert thr[0.9]["全部独立"][1] >= 8, "独立位置置信度都高, τ=0.9 每步应揭开很多个"
    assert thr[0.9]["全部相关"][1] <= 2, "强相关位置置信度低, τ=0.9 应自动退回每步一两个"
    assert thr[0.5]["全部相关"][2] > 0.1, "τ=0.5 时三角短语的三个位置 (各约 2/3) 会同一步揭开, 应拼错"
    kv("混合负载: 固定 k=2 / τ=0.7", f"步数 {fixed[2]['混合'][0]:.0f} / {thr[0.7]['混合'][0]:.1f}, 拼错率 {fixed[2]['混合'][2]:.1%} / {thr[0.7]['混合'][2]:.1%}")
    assert thr[0.7]["混合"][0] < fixed[2]["混合"][0], "混合负载上 τ=0.7 的步数应少于固定 k=2"
    assert fixed[2]["混合"][2] > thr[0.7]["混合"][2], "且固定 k=2 有拼错, τ=0.7 没有"
    kv("τ=0.7 在独立 / 相关负载上的步数", f"{thr[0.7]['全部独立'][0]:.1f} / {thr[0.7]['全部相关'][0]:.1f}  ← 固定 k 做不到按内容调步数")
    assert thr[0.7]["全部独立"][0] < fixed[2]["全部独立"][0] / 4, "独立负载上阈值策略的步数应远少于固定 k=2"

    print("\n[3] 近似 KV cache: prompt 16 + 回答 32 (块长 8, 每步揭开 1 个), 每 r 步完整刷新一次")
    model = BiLM()
    prompt = np.random.RandomState(3).randint(0, 64, 16)
    states, _ = trajectory(model, prompt, gen=32, block=8)
    print("  " + rj("r", 4) + rj("当前块 logits 平均 |Δ|", 26) + rj("决策不一致率", 16))
    errs = {}
    for r in (1, 2, 4, 8, 16, 32):
        e, d = dual_cache_error(model, states, P=16, block=8, refresh=r)
        errs[r] = (e, d)
        note = "  ← 每块开头刷新 (Fast-dLLM DualCache)" if r == 8 else ""
        print(f"  {r:>4}{e:>26.4f}{d:>16.1%}{note}")
    assert errs[1][0] < 1e-5, "每步都刷新 = 没有缓存, 应与完整前向一致"
    rs = sorted(errs)
    for a, b in zip(rs, rs[1:]):
        assert errs[b][0] > errs[a][0], f"刷新间隔 {b} 的误差应大于 {a}"
    assert errs[32][1] > errs[2][1], "刷新越稀, 揭开决策与精确解码不一致的步越多"

    print("\n[4] 块扩散: 块间因果、块内双向, 写完的块缓存一次")
    worst = block_cache_error(model, prompt, gen=32, block=8)
    kv("缓存 vs 每步完整前向 max|Δlogit|", f"{worst:.1e}")
    assert worst < 1e-5, "块间因果时前面块的 K/V 不受后面影响, 缓存应是精确的"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
