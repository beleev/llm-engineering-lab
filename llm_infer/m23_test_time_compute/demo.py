"""
m23 demo — test-time compute: 同一个 sampler, 四种花 token 的方式, 画 "正确率 vs 生成 token 数"

    [0] 任务: K=4 步算术链, 30% 的题有一个"看错运算符"的陷阱步 (模型的众数就是错的)
    [1] 并行采样: best-of-N + ORM vs 多数投票; assert 多花 token 换正确率, 且多数投票在陷阱题上饱和
    [2] PRM 引导的逐步 beam search: 小预算下高于同 token 数的 best-of-N, 大预算打平 (PRM 有噪声)
    [3] 串行长思考 + budget forcing: 截断 → 0; 追加 "Wait" → 涨, 但涨不过陷阱

运行: python -m llm_infer.m23_test_time_compute.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m23_test_time_compute import tts
from llm_infer.m23_test_time_compute.tts import (
    best_of_n, make_problems, majority_vote, prm_beam_search, rollout, think,
)

N_PROB, K, TRAP_FRAC = 200, 4, 0.3                        # 题数, 每题步数, 带陷阱的题的比例
NS = [1, 2, 4, 8, 16, 32, 64]                             # [1] 里并行采样的条数 N
BEAMS = [(1, 2), (2, 2), (2, 4), (4, 4), (4, 8)]          # (width, expand)
BUDGETS = [2, 4, 6, 8, 12, 16, 24, 32, 64]                # [3] 里 budget forcing 的 token 预算 B


def acc(answers, probs, mask=None) -> float:
    """正确率。mask (N_PROB,) bool: 只统计其中为 True 的题。"""
    ok = np.array([a == p.answer for a, p in zip(answers, probs)])
    return float(ok[mask].mean() if mask is not None else ok.mean())


def at_n(pick, probs, pool, n) -> np.ndarray:
    """(N_PROB,) 每题的 @N 正确率: 把 64 条切成 64//n 组互不重叠的 n 条, 每组挑一个答案, 取平均 (比只看前 n 条方差小)。"""
    return np.array([np.mean([pick(p, c[g * n:(g + 1) * n]) == p.answer for g in range(len(c) // n)])
                     for p, c in zip(probs, pool)])


def main():
    banner("M23 - Test-Time Compute (best-of-N / 多数投票 / PRM beam / budget forcing)")
    rng = np.random.RandomState(0)
    probs = make_problems(N_PROB, K, TRAP_FRAC, rng)
    trap = np.array([p.trap >= 0 for p in probs])

    p0 = probs[int(np.argmax(trap))]
    print(f"\n[0] {N_PROB} 道题, 每道 K={K} 步, {trap.mean():.0%} 带陷阱步。例: x0={p0.x0}, "
          f"ops={' '.join(f'{o}{a}' for o, a in p0.ops)}, 答案 {p0.answer}, 陷阱在第 {p0.trap} 步")
    greedy = [rollout(p, rng, temperature=0.0)[-1] for p in probs]
    kv("greedy (T=0) 正确率", f"{acc(greedy, probs):.3f}  ← 恰好 = 无陷阱题比例: 玩具里随机错只来自采样")
    assert abs(acc(greedy, probs) - (1 - trap.mean())) < 1e-9, \
        "greedy 应恰好答对所有无陷阱题、答错所有陷阱题"

    print(f"\n[1] 并行采样 T=1, 每题采 {NS[-1]} 条, 切成互不重叠的 N 条一组算 @N; token = N·K")
    pool = [[rollout(p, rng) for _ in range(NS[-1])] for p in probs]
    bon_n = lambda n: at_n(lambda p, cs: best_of_n(p, cs, rng), probs, pool, n)
    rows = {}
    for n in NS:
        maj = at_n(lambda p, cs: majority_vote(cs), probs, pool, n)
        oracle = at_n(lambda p, cs: p.answer if any(c[-1] == p.answer for c in cs) else None, probs, pool, n)
        rows[n] = dict(bon=bon_n(n).mean(), maj=maj.mean(), maj_trap=maj[trap].mean(),
                       maj_clean=maj[~trap].mean(), pass_n=oracle.mean())
    print(f"  {'N':>4}{'token':>7} | {'pass@N':>7}{'BoN+ORM':>9}{'多数投票':>9} | {'投票(无陷阱)':>11}{'投票(陷阱)':>10}")
    for n, r in rows.items():
        print(f"  {n:>4}{n * K:>7} | {r['pass_n']:>7.3f}{r['bon']:>9.3f}{r['maj']:>11.3f} | "
              f"{r['maj_clean']:>13.3f}{r['maj_trap']:>12.3f}")
    assert rows[64]["bon"] > rows[1]["bon"] + 0.4, "best-of-N: 多花 token 换正确率"
    assert rows[64]["maj"] - rows[16]["maj"] < 0.03, "多数投票 N≥16 后饱和"
    assert rows[64]["maj_trap"] < 0.15, "陷阱题上投票收敛到错的众数, 正确率应很低"
    assert rows[64]["maj_clean"] > 0.95, "无陷阱题上投票应几乎全对; 所以饱和点 = 无陷阱题比例"
    assert rows[64]["bon"] > rows[64]["maj"] + 0.15, "外部判分器能越过系统性偏差, 投票不能"

    print(f"\n[2] PRM 引导的逐步 beam search (width × expand 个候选/步); 对照同 token 数的 best-of-N")
    beam_rows = []
    for w, e in BEAMS:
        res = [prm_beam_search(p, w, e, rng) for p in probs]
        tokens = np.mean([t for _, t in res])
        n = int(round(tokens / K))                        # 同 token 数的 BoN: N = tokens / K
        beam_rows.append((w, e, tokens, acc([a for a, _ in res], probs), n, bon_n(n).mean()))
    print(f"  {'width×expand':>13}{'token':>7}{'PRM beam':>10} | {'BoN N':>6}{'BoN+ORM':>9}")
    for w, e, t, a, n, b in beam_rows:
        print(f"  {f'{w}×{e}':>13}{t:>7.0f}{a:>10.3f} | {n:>6}{b:>9.3f}")
    for w, e, t, a, n, b in beam_rows[:3]:
        assert a > b, f"{w}×{e}: 小预算下 PRM beam ({a:.3f}) 应 > 同 token 数的 best-of-N ({b:.3f})"
    # 对照: 同样的 beam 形状 (BEAMS[3]), 换成无噪声的 PRM。临时改模块全局量 SIGMA, 算完立刻改回
    w3, e3 = BEAMS[3]
    tts.SIGMA, sigma = 0.0, tts.SIGMA
    exact = acc([prm_beam_search(p, w3, e3, rng)[0] for p in probs], probs)
    tts.SIGMA = sigma
    kv(f"{w3}×{e3}, PRM 判分无噪声 (σ=0)", f"{exact:.3f}  vs σ={sigma}: {beam_rows[3][3]:.3f}")
    assert exact > beam_rows[3][3] + 0.1, \
        f"PRM 无噪声时 {w3}×{e3} beam 的正确率应明显更高: {exact:.3f} vs {beam_rows[3][3]:.3f}"
    print("  小预算: 错步一出现就被剪掉, 不再为它的后续步付费 → beam 赢;")
    print("  大预算: 打平 —— 候选一多, 带噪声的 PRM 总会给某个错步打出高分 (σ=0 时就没有这个问题)。")

    print(f"\n[3] 串行长思考: 写完链后自查, 发现粗心错就重写; budget forcing 把 token 数钉在 B")
    # 每道题固定一个种子, 各预算共用: 首稿完全相同, 正确率的差别只来自预算
    nat = [think(p, np.random.RandomState(j)) for j, p in enumerate(probs)]
    kv("不加干预 (模型自己决定何时停)", f"平均 {np.mean([t for _, t in nat]):.1f} token, 正确率 {acc([a for a, _ in nat], probs):.3f}")
    print(f"  {'B':>4} | {'正确率':>6}{'无陷阱':>8}{'陷阱':>8}")
    bf = {}
    for b in BUDGETS:
        ans = [think(p, np.random.RandomState(j), min_tokens=b, max_tokens=b)[0] for j, p in enumerate(probs)]
        bf[b] = (acc(ans, probs), acc(ans, probs, ~trap), acc(ans, probs, trap))
        print(f"  {b:>4} | {bf[b][0]:>9.3f}{bf[b][1]:>9.3f}{bf[b][2]:>8.3f}")
    assert bf[2][0] < 0.1 < bf[K][0], "预算 < K: 链没写完就被截断, 答的是中间值 (只有后两步恰好抵消时才碰对)"
    assert bf[64][0] > bf[K][0] + 0.2, "追加 'Wait' 延长思考 → 正确率上涨"
    assert bf[64][2] < bf[64][1] - 0.4, "自查看不见自己的误解: 陷阱题远落后"
    print("  结论: 投票收敛到模型自己的众数, 自查看不见自己的误解; 只有外部判分器 (ORM/PRM) 能越过系统性偏差。")


if __name__ == "__main__":
    main()
