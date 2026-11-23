"""
m35 demo — 混合模型的状态缓存: 前缀快照、投机解码回滚、MTP 自投机

    [1] 带缓存的增量生成 (状态 + KV) 与每步整段重算逐位一致; 状态字节不随长度变, KV 字节随长度涨
    [2] 前缀缓存: 命中长度按快照间隔向下取整; 拿「更长序列的最终状态」冒充中间状态会算错
    [3] 字节账 (估算): 按 Qwen3-Next-80B-A3B 的 config.json, 一份状态快照相当于多少 token 的 KV
    [4] 投机解码回滚: 逐位置快照 / 从快照重算 两种做法都与贪心逐 token 相同, 只截断 KV 会算错
    [5] MTP 草稿接受率: 没训练的头 vs 最小二乘拟合的头; 随机权重主模型 vs 植入后继表的主模型

运行: python -m llm_infer.m35_hybrid_state_cache.demo
"""
from __future__ import annotations

import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m35_hybrid_state_cache.hybrid_cache import (
    HybridLM, MTPHead, PrefixStateCache, kv_bytes_per_token, snapshot_bytes, speculative_generate)

PLANT = 3.0                      # 后继表强度: 主模型的输出有一部分可预测 (见 [5])
K = 3                            # 每轮草稿 token 数 (MTP 头链式用 3 次)
N_NEW = 48


def rj(s, w):
    """按显示宽度右对齐 (中文占两格)。"""
    return " " * max(0, w - len(s) - sum(ord(c) > 0x2E80 for c in s)) + s


def lj(s, w):
    """按显示宽度左对齐。"""
    return s + " " * max(0, w - len(s) - sum(ord(c) > 0x2E80 for c in s))


def prompts(n, length=8, seed=1000):
    return [list(np.random.RandomState(seed + i).randint(0, 64, length)) for i in range(n)]


def fitted_mtp(model, n_seq=40):
    """用主模型自己贪心生成的 40 条序列拟合 MTP 头 (自蒸馏), 和评测用的 prompt 不重叠。"""
    mtp = MTPHead(model)
    n = mtp.fit([model.greedy(p, 56) for p in prompts(n_seq, seed=100)])
    return mtp, n


def main():
    banner("M35 - 混合模型的状态缓存 (Gated DeltaNet + 注意力)")
    m = HybridLM(plant=PLANT)
    S = m.cfg.d_state
    kv("层排布 (R=Gated DeltaNet, A=注意力)", f"{m.cfg.layout}  ({m.n_rec} 个递推层, {m.n_attn} 个注意力层)")
    kv("d_model / 状态矩阵", f"{m.cfg.d} / {S}×{S}, fp32")

    # ---------------- [1] ---------------- #
    print("\n[1] 增量生成 (状态 + KV) vs 每步整段重算")
    p = prompts(1, length=64)[0]
    cache = m.new_cache()
    logits, _ = m.feed(p, cache)
    seq, worst = list(p), 0.0
    sizes = {}
    for _ in range(32):
        seq.append(int(np.argmax(logits[-1])))
        full, _ = m.forward(seq)                                        # 不带缓存, 从头算
        logits, _ = m.feed([seq[-1]], cache)
        worst = max(worst, float(np.abs(full[-1] - logits[-1]).max()))
        if len(seq) in (65, 96):
            sizes[len(seq)] = (sum(s.nbytes for s in cache.states), sum(k.nbytes + v.nbytes for k, v in cache.kv))
    kv("32 步里最大 |Δlogit|", f"{worst:.2e}")
    for n, (sb, kb) in sizes.items():
        kv(f"喂完 {n} 个 token 时", f"状态 {sb:,} B, KV {kb:,} B")
    assert worst < 1e-4, f"带缓存的增量路径应与整段重算一致, 实际最大差 {worst:.2e}"
    assert sizes[65][0] == sizes[96][0], "递推层状态是定长的, 序列变长不应改变它的字节数"
    assert sizes[96][1] > sizes[65][1], "注意力层 KV 按 token 增长"

    # ---------------- [2] ---------------- #
    print("\n[2] 前缀缓存: 命中按快照间隔向下取整")
    a = prompts(1, length=100, seed=7)[0]
    b = a[:77] + prompts(1, length=23, seed=8)[0]                      # 和 a 共享前 77 个 token
    ref, _ = m.forward(b)
    print("  " + "".join(rj(h, w) for h, w in (("间隔", 6), ("公共前缀", 10), ("可用命中", 10), ("快照份数", 10), ("快照字节", 12), ("续算后 max|Δ|", 18))))
    rows = []
    for interval in (1, 8, 16, 32, 64):
        pc = PrefixStateCache(interval)
        pc.insert(m, a)
        L, h, c = pc.match(m, b)
        lg, _ = m.feed(b[h:], c)
        err = float(np.abs(lg - ref[h:]).max())
        n_snap = len(pc.entries[0][2])
        nbytes = n_snap * sum(s.nbytes for s in c.states)
        rows.append((interval, h, nbytes))
        print(f"  {interval:>6}{L:>10}{h:>10}{n_snap:>10}{nbytes:>12,}{err:>18.2e}")
        assert h == (77 // interval) * interval, f"间隔 {interval}: 命中应是 77 向下取整到 {interval} 的倍数"
        assert err < 1e-4, f"间隔 {interval}: 从快照恢复后续算, 应与整段重算一致"
    for (i0, h0, b0), (i1, h1, b1) in zip(rows, rows[1:]):
        assert h1 <= h0 and b1 < b0, "快照越稀, 命中越短、快照越省"
    # 反例: KV 截到 77 可以, 但状态拿的是 a 喂完 100 个 token 的最终状态
    full_a = m.new_cache()
    m.feed(a, full_a)
    wrong = m.new_cache()
    wrong.states, wrong.kv, wrong.n = full_a.copy_states(), [(k[:77], v[:77]) for k, v in full_a.kv], 77
    lg, _ = m.feed(b[77:], wrong)
    err_wrong = float(np.abs(lg - ref[77:]).max())
    kv("反例: KV[:77] + a 的最终状态", f"max|Δlogit| = {err_wrong:.2f}  ← 状态切不出前 77 个 token 的样子")
    assert err_wrong > 0.1, "用更长序列的最终状态冒充第 77 位的状态, 输出应明显偏离"

    # ---------------- [3] ---------------- #
    print("\n[3] 字节账 (估算, 按 Qwen3-Next-80B-A3B 的 config.json, bf16)")
    # 48 层, 每 4 层 1 层全注意力: 12 个注意力层 (2 个 KV 头 × 256 维), 36 个 Gated DeltaNet 层
    # (32 个 value 头, 每头 128×128 的状态; q/k/v 前有 kernel=4 的短卷积, 缓存最近 3 步的 16·128·2 + 32·128 = 8192 维)
    kv_tok = kv_bytes_per_token(n_attn=12, kv_heads=2, head_dim=256)
    snap = snapshot_bytes(n_rec=36, heads=32, d_k=128, d_v=128, conv=3 * 8192)
    full_attn_tok = kv_bytes_per_token(n_attn=48, kv_heads=2, head_dim=256)
    breakeven = snap / kv_tok
    kv("每 token 的 KV (12 个注意力层)", f"{kv_tok:,} B  (若 48 层全是注意力: {full_attn_tok:,} B)")
    kv("一份状态快照 (36 个递推层)", f"{snap / 2**20:.1f} MiB = {breakeven:.0f} 个 token 的 KV")
    T = 32768
    kv(f"{T} token 上下文的 KV", f"{T * kv_tok / 2**20:,.0f} MiB")
    for interval in (512, 2048, 8192):
        n = T // interval
        kv(f"  快照间隔 {interval}: {n} 份快照", f"{n * snap / 2**20:,.0f} MiB")
        assert (n * snap > T * kv_tok) == (interval < breakeven), \
            f"间隔 {interval}: 快照比 KV 还大 当且仅当 间隔小于 {breakeven:.0f} token"
    assert full_attn_tok == 4 * kv_tok, "3:1 混合后每 token KV 应是全注意力的 1/4"

    # ---------------- [4] ---------------- #
    print(f"\n[4] 投机解码回滚 (MTP 草稿, 每轮 {K} 个, 3 条 prompt × {N_NEW} 个新 token)")
    mtp, _ = fitted_mtp(m)
    ps = prompts(3)
    refs = [m.greedy(x, N_NEW) for x in ps]
    print("  " + lj("做法", 14) + "".join(rj(h, w) for h, w in (("与贪心相同", 12), ("接受率", 10), ("多存状态份数", 14), ("重算 token", 12))))
    for mode, label in (("per_step", "逐位置快照"), ("recompute", "从快照重算"), ("truncate_only", "只截断 KV")):
        same, acc, prop, replay, peak = 0, 0, 0, 0, 0
        for x, r in zip(ps, refs):
            out, st = speculative_generate(m, mtp, x, N_NEW, K, mode)
            same += out == r
            acc, prop, replay, peak = acc + st.accepted, prop + st.proposed, replay + st.replayed, max(peak, st.peak_snapshots)
        print(f"  {lj(label, 14)}{same:>10}/3{acc / prop:>10.1%}{peak:>14}{replay:>12}")
        if mode == "truncate_only":
            assert same < 3, "只截断 KV、不恢复状态, 输出应偏离贪心"
        else:
            assert same == 3, f"{label}: 输出应与贪心逐 token 相同"
        if mode == "per_step":
            assert peak == K + 1, "逐位置快照要多存 K+1 份状态"
        if mode == "recompute":
            assert replay > 0, "有草稿被拒时, 应从快照把接受的 token 重喂一遍"

    # ---------------- [5] ---------------- #
    print("\n[5] MTP 草稿接受率 (5 条 prompt, 每轮 3 个草稿)")
    print("  " + lj("主模型", 20) + lj("MTP 头", 10) + rj("接受率", 8) + rj("每次验证产出", 14))
    res = {}
    for plant in (0.0, PLANT):
        mm = HybridLM(plant=plant)
        trained, n_fit = fitted_mtp(mm)
        for name, head in (("没训练", MTPHead(mm)), ("拟合", trained)):
            acc = prop = rounds = 0
            for x in prompts(5):
                out, st = speculative_generate(mm, head, x, N_NEW, K, "per_step")
                assert out == mm.greedy(x, N_NEW), "接受率再低, 输出也应与贪心相同"
                acc, prop, rounds = acc + st.accepted, prop + st.proposed, rounds + st.rounds
            res[plant, name] = acc / prop
            label = "随机权重" if plant == 0 else f"植入后继表 ({plant:g})"
            print(f"  {lj(label, 20)}{lj(name, 10)}{acc / prop:>8.1%}{(acc + rounds) / rounds:>10.2f} tok")
    kv("拟合样本数", f"{n_fit} 个 (h_t, x_t+1) → h_t+1")
    for plant in (0.0, PLANT):
        assert res[plant, "拟合"] > res[plant, "没训练"], "拟合过的 MTP 头接受率应高于没训练的"
    assert res[0.0, "拟合"] < 0.2, "随机权重主模型的贪心输出几乎没有规律, 线性 MTP 头学不到多少"
    assert res[PLANT, "拟合"] > 0.5, "主模型输出有可预测结构时, 拟合的 MTP 头应接受过半"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
