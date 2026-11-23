"""
m39 demo — KV 淘汰: 同样的 KV 预算, 谁留得住针, 谁答得了第二个问题

    合成对话: 2032 token 前文 (位置 0 是 sink, 中段埋两根针 A、B) → 问题 1 (找 A) → 回答 1 → 问题 2 (找 B)。
    [1] 预算 256 token (prompt 的 12.5%) 下比较: 完整 cache / sink + 窗口 / TOVA / H2O / SnapKV / Quest (m22)
    [2] 20 个种子汇总: 问题 1、问题 2 各答对几次; 淘汰策略答对与否完全取决于针是否碰巧还在
    [3] 不可逆: 前文不变, 两个问题对调, 淘汰策略的成败跟着提问顺序走; Quest 存的是全部 KV, 读的只是一部分
数据是人工合成的单层单头注意力, 不是真实模型; 结论说明机制, 不说明真实模型上的准确率。

运行: python -m llm_infer.m39_kv_eviction.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m22_sparse_attention.sparse_attention import quest_upper_bound, select_blocks
from llm_infer.m39_kv_eviction.kv_eviction import (
    H2O, attend, make_context, readout, sink_window_keep, snapkv_keep, tova_keep,
)

BUDGET = 256                     # KV 预算 (token 数)
BS = 16                          # Quest 的 block 大小; BUDGET / BS = 每步读几个 block
SEEDS = range(20)
EVICT = ("sink + 窗口", "TOVA", "H2O", "SnapKV")      # 会删 KV 的策略
POLICIES = ["完整 cache", "sink + 窗口", "TOVA", "H2O", "SnapKV", "Quest (不删, 只少读)"]


def run(seed: int, first: str = "A"):
    """一个种子上跑全部策略 → (ctx, 各策略留下的位置, {策略: (问题1 针质量, 问题1 答对, 问题2 针质量, 问题2 答对, 存的 KV, 每步读的 KV)})。"""
    ctx = make_context(seed, first)
    order = ("A", "B") if first == "A" else ("B", "A")                      # 问题 1、问题 2 各在找哪根针
    t1, t2 = ctx.t_q1 + 1, ctx.t_q2 + 1                                     # 两次提问时 cache 里写了多少 token
    keeps = {}                                                              # 策略 → (问题 1 时留下的位置, 问题 2 时)
    keeps["完整 cache"] = (np.arange(t1), np.arange(t2))
    keeps["sink + 窗口"] = (sink_window_keep(t1, BUDGET), sink_window_keep(t2, BUDGET))
    keeps["TOVA"] = (tova_keep(ctx, t1, BUDGET), tova_keep(ctx, t2, BUDGET))
    h2o = H2O(ctx, t1, BUDGET)
    k1 = h2o.keep.copy()
    for t in range(t1, t2):                                                 # 回答 1 和问题 2 逐 token 写入, 每步淘汰 1 个
        h2o.step(t)
    keeps["H2O"] = (k1, h2o.keep)
    snap = snapkv_keep(ctx, t1, BUDGET)                                     # prefill 结束时压一次
    keeps["SnapKV"] = (snap, np.r_[snap, t1:t2])                            # 生成阶段写入的 token 不再压
    quest = []
    for t in (t1 - 1, t2 - 1):                                              # Quest: 每个 query 现选 block, KV 全留
        blocks = select_blocks(quest_upper_bound(ctx.q[t], ctx.k[:t + 1], BS), BUDGET // BS)
        quest.append((blocks[:, None] * BS + np.arange(BS)).reshape(-1))
    keeps["Quest (不删, 只少读)"] = tuple(quest)

    res = {}
    for name, (ka, kb) in keeps.items():
        row = []
        for t, keep, nd in ((t1 - 1, ka, order[0]), (t2 - 1, kb, order[1])):
            needle, target = ctx.needles[nd], "AB".index(nd)                # 候选 0 是 A 的答案, 1 是 B 的
            full = np.arange(t + 1)
            w = np.exp(ctx.k[full] @ ctx.q[t] / 8.0)                         # 完整注意力 (未归一化), √d = 8
            mass = w[np.intersect1d(keep, needle)].sum() / w[keep].sum()   # 留下的 cache 里, 注意力落在针上的比例
            row += [mass, readout(ctx, attend(ctx, t, keep)) == target]
        stored = t2 if name.startswith(("完整", "Quest")) else len(kb)
        res[name] = (*row, stored, len(kb))
    return ctx, keeps, res


def main():
    banner("M39 - KV 淘汰: H2O / SnapKV / TOVA vs sink + 窗口 vs Quest")
    ctx, keeps, res = run(0)
    T = ctx.t_q2 + 1
    kv("对话长度 / 预算", f"前文 2032 + 问题 1 16 + 回答 1 16 + 问题 2 16 = {T} token; 预算 {BUDGET} (prompt 的 {BUDGET / (ctx.t_q1 + 1):.1%})")
    kv("针 A / 针 B 的位置 (seed 0)", f"{ctx.needles['A'].tolist()} / {ctx.needles['B'].tolist()}")

    print(f"\n[1] seed 0: 针质量 = 留下的 cache 里注意力落在针上的比例; ✓ = 输出里过半是正确答案")
    print(f"  {'策略':<20} {'问题1 针质量':>10} {'答':>3} {'问题2 针质量':>10} {'答':>3} {'存的 KV':>7} {'每步读':>6}")
    for name, (m1, a1, m2, a2, stored, read) in res.items():
        print(f"  {name:<20} {m1:>13.3f} {'✓' if a1 else '✗':>3} {m2:>13.3f} {'✓' if a2 else '✗':>3} {stored:>8} {read:>7}")
    lucky = len(np.intersect1d(keeps["SnapKV"][0], ctx.needles["B"]))
    kv("seed 0 里 SnapKV 留下的针 B token", f"{lucky} / {len(ctx.needles['B'])}  (没人问过它, 留下全凭运气, 见 [2])")

    print(f"\n[2] {len(SEEDS)} 个种子 (针的位置和噪声都换): 答对次数 / 针还在 cache 里的次数")
    tally = {p: np.zeros(4, int) for p in POLICIES}                         # 问题1 答对, 针A 还在, 问题2 答对, 针B 还在
    for seed in SEEDS:
        c, ks, rs_ = run(seed)
        for name, (ka, kb) in ks.items():
            alive = [len(np.intersect1d(ka, c.needles["A"])) > 0, len(np.intersect1d(kb, c.needles["B"])) > 0]
            r = rs_[name]
            tally[name] += [r[1], alive[0], r[3], alive[1]]
            if name in EVICT:
                assert r[1] == alive[0] and r[3] == alive[1], \
                    f"seed {seed} {name}: 淘汰策略答对与否应完全取决于针是否还在 cache 里"
    n = len(SEEDS)
    for name, (c1, s1, c2, s2) in tally.items():
        print(f"  {name:<20} 问题1 {c1:>2}/{n} (针A 在 {s1:>2})   问题2 {c2:>2}/{n} (针B 在 {s2:>2})")
    assert list(tally["完整 cache"][[0, 2]]) == [n, n], "完整 cache 两个问题都应答对"
    assert list(tally["Quest (不删, 只少读)"][[0, 2]]) == [n, n], "Quest 不删 KV, 每个 query 现选 block, 两个问题都应答对"
    for name in ("H2O", "SnapKV"):
        assert tally[name][0] == n, f"{name} 的打分里有问题 1 的 query, 针 A 应每次都留下"
        assert tally[name][2] <= 0.3 * n, f"{name}: 针 B 没被问过, 只能碰巧留下, 问题 2 答对应不超过 30%"
    assert tally["sink + 窗口"].sum() == 0, "sink + 窗口只看位置, 前文中段的针一根也留不住"
    assert tally["TOVA"][0] <= 0.3 * n, "TOVA 在问题 1 到来之前就按普通 query 的注意力淘汰, 针 A 应大多已被删掉"

    print("\n[3] 不可逆: 前文不变, 只把两个问题对调 (先问 B 再问 A)")
    swap = {name: np.zeros(2, int) for name in ("H2O", "SnapKV")}
    for seed in SEEDS:
        r = run(seed, first="B")[2]
        for name in swap:
            swap[name] += [r[name][1], r[name][3]]
    for name, (cb, ca) in swap.items():
        print(f"  {name:<8} 先问 B: 答对 {cb:>2}/{n}   再问 A: 答对 {ca:>2}/{n}")
        assert cb == n, f"{name}: 先问 B 时, 针 B 应每次都留下 —— 打分本身没问题"
        assert ca <= 0.3 * n, f"{name}: 后问 A 时, 针 A 已在压缩时被删, 答对应不超过 30%"
    full, quest = res["完整 cache"], res["Quest (不删, 只少读)"]
    kv("Quest: 存的 KV / 每步读的 KV", f"{quest[4]} / {quest[5]}  (显存不省, 省的是带宽)")
    assert quest[4] == full[4] and quest[5] == BUDGET, "Quest 存全部 KV, 每步只读预算内的 block"
    for name in ("sink + 窗口", "TOVA", "H2O"):
        assert res[name][4] <= BUDGET, f"{name}: cache 应始终不超过预算 {BUDGET}"
    kv("SnapKV 问题 2 时存的 KV", f"{res['SnapKV'][4]}  (prompt 压到 {BUDGET}, 生成阶段写入的 32 个不压)")

    print("\n  ✓ 同样的预算: 按问题 1 打分的淘汰策略答得了问题 1, 问题 2 只能靠针碰巧被留下; 按位置淘汰的两个都答不了")
    print("  ✓ Quest 不删 KV, 两个问题都答得了, 代价是显存一点没省")


if __name__ == "__main__":
    main()
