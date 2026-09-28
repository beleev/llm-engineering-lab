"""
m10 demo — 采样策略: 先看直方图建立直觉, 再用断言钉死每个策略承诺的性质。

[1] 固定 logits (V=10) 上各策略 10000 次采样的频次
[2] 性质断言 (200 组随机 logits): top-k 恰好 k 个 / top-p 质量 ≥p 且最小 / min-p 阈值 / T→0 == greedy
[3] Gumbel-max 经验分布 vs softmax 的 total-variation 距离
[4] repetition penalty 让被罚 token 概率下降 (正、负 logit 都成立)

运行: python -m llm_infer.m10_sampling.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, softmax
from llm_infer.m10_sampling.samplers import (
    greedy, temperature_sample, top_k_filter, top_p_filter, min_p_filter,
    repetition_penalty, gumbel_max, sample, SamplingParams,
)

V = 10
LOGITS = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0])
N = 10000
TV_TOL = 0.02        # N=10000, V=10 时采样噪声造成的 TV 约 0.01
N_CASES, V_RAND = 200, 50            # [2] 随机 logits 的组数和词表大小
PENALTY, HISTORY = 1.5, [0, 0, 9]    # [4] repetition penalty 的系数和已生成的 token


def histogram(fn) -> np.ndarray:
    """把采样函数 fn 调 N 次, 统计每个 token id 出现的次数。"""
    return np.bincount([fn() for _ in range(N)], minlength=V)          # (V,)


def check_filter_properties(rng) -> None:
    """[2] 在 N_CASES 组随机 logits 上验证每个 filter 承诺的性质。"""
    for _ in range(N_CASES):
        logits = rng.normal(0, 2, size=V_RAND)
        logits[:3] = logits[0]                                         # 故意制造并列, 专门考 top-k
        probs = softmax(logits)
        k, p, mp = int(rng.integers(1, V_RAND)), float(rng.uniform(0.05, 0.99)), float(rng.uniform(0.01, 0.5))

        kept = np.isfinite(top_k_filter(logits, k))                    # (V,) bool: 没被置 -inf 的就是留下的
        assert kept.sum() == k, f"top-k 应恰好留 {k} 个 (有并列值时也一样), 实际 {kept.sum()}"
        assert logits[kept].min() >= logits[~kept].max(), "top-k 留下的应是最大的 k 个"

        kept = np.isfinite(top_p_filter(logits, p))
        mass = probs[kept].sum()
        # -1e-12: cumsum 的浮点舍入余量
        assert mass >= p - 1e-12, f"top-p 留下的总概率 {mass:.6f} 应 ≥ p={p:.6f}"
        assert mass - probs[kept].min() < p, "top-p 最小性: 再去掉留下的最小者, 总概率就应 < p"
        # initial=0.0: 全部留下时 ~kept 为空, max 需要一个初值
        assert probs[kept].min() >= probs[~kept].max(initial=0.0), "top-p 留下的应是概率最大的那批"

        kept = np.isfinite(min_p_filter(logits, mp))
        assert np.array_equal(kept, probs >= mp * probs.max()), "min-p 留下的集合应恰好是 {p_i ≥ min_p·p_max}"

        assert sample(logits, SamplingParams(temperature=0.0), rng=rng) == greedy(logits), \
            "temperature=0 应等于 greedy"
        logits[:3] += [0.3, 0.2, 0.1]                                  # 拆掉并列, 否则 T→0 时并列 token 本来就该平分
        assert sample(logits, SamplingParams(temperature=1e-4), rng=rng) == greedy(logits), \
            "temperature → 0 时采样应退化为 greedy"


def main():
    banner("M10 - Sampling Strategies")
    rng = np.random.RandomState(0)
    probs = softmax(LOGITS)

    print(f"\n[1] 采样频次 (每行 {N} 次, seed=0); 第一行是 softmax 概率 ×{N}")
    rows = {
        "softmax(logits)·N": (probs * N).round().astype(int),
        "greedy": histogram(lambda: greedy(LOGITS)),
        "temp=1.0": histogram(lambda: temperature_sample(LOGITS, 1.0, rng)),
        "temp=0.3 (更尖)": histogram(lambda: temperature_sample(LOGITS, 0.3, rng)),
        "temp=2.0 (更平)": histogram(lambda: temperature_sample(LOGITS, 2.0, rng)),
        "top_k=3": histogram(lambda: sample(LOGITS, SamplingParams(top_k=3), rng=rng)),
        "top_p=0.5": histogram(lambda: sample(LOGITS, SamplingParams(top_p=0.5), rng=rng)),
        "min_p=0.1": histogram(lambda: sample(LOGITS, SamplingParams(min_p=0.1), rng=rng)),
    }
    print(f"  {'token id':<20}" + "".join(f"{i:>6}" for i in range(V)))
    for name, h in rows.items():
        print(f"  {name:<20}" + "".join(f"{c:>6}" for c in h))
    assert rows["greedy"][0] == N, "greedy 每次都应选 logit 最大的 token 0"
    assert rows["top_k=3"][3:].sum() == 0, "top_k=3: 被砍的 token 3~9 一次都不应出现"
    # 下面两条的概率是 LOGITS 取当前值时的 softmax
    assert rows["top_p=0.5"][2:].sum() == 0, \
        "top_p=0.5: p0=0.416 < 0.5, p0+p1=0.668 ≥ 0.5, 只留 token 0 和 1"
    assert rows["min_p=0.1"][4:].sum() == 0, \
        "min_p=0.1: 阈值 0.1·0.416=0.0416; p3=0.056 留下, p4=0.034 起全砍"

    print(f"\n[2] 性质断言: {N_CASES} 组随机 logits (V={V_RAND}, 含并列值)")
    check_filter_properties(np.random.default_rng(1))
    print("  top-k 恰好 k 个有限值 / top-p 质量≥p 且去掉最小者就<p / min-p 集合 == {p_i ≥ min_p·p_max} / T=0 与 T=1e-4 == greedy  ✓")

    print("\n[3] Gumbel-max 与 softmax 同分布? total-variation = ½·Σ|经验频率 − p|")
    tv_gumbel = 0.5 * np.abs(histogram(lambda: gumbel_max(probs, rng)) / N - probs).sum()
    tv_multi = 0.5 * np.abs(rows["temp=1.0"] / N - probs).sum()
    kv("TV(gumbel-max, softmax)", f"{tv_gumbel:.4f}")
    kv("TV(multinomial, softmax)", f"{tv_multi:.4f}  ← 同样 N 下的采样噪声水平, 作参照")
    assert tv_gumbel < TV_TOL, f"Gumbel-max 的经验分布应与 softmax 一致: TV={tv_gumbel:.4f}, 上限 {TV_TOL}"
    assert tv_multi < TV_TOL, f"参照组 multinomial 的 TV={tv_multi:.4f} 也应低于 {TV_TOL}"

    hit = sorted(set(HISTORY))                                        # 被罚的 token (去重)
    print(f"\n[4] repetition penalty={PENALTY}, history={HISTORY} ("
          + ", ".join(f"token {t} logit={LOGITS[t]:+.1f}" for t in hit) + ")")
    pen = softmax(repetition_penalty(LOGITS, HISTORY, PENALTY))
    for t in hit:
        kv(f"token {t} 概率", f"{probs[t]:.4f} → {pen[t]:.4f}")
        assert pen[t] < probs[t], f"token {t} 被罚后概率应下降 (logit 正负都一样)"
    wrong = LOGITS.copy(); wrong[hit] /= PENALTY                       # 常见错误写法: 不分正负一律除
    neg = hit[-1]                                                     # logit 为负的那个被罚 token
    kv(f"错误写法 (一律除) token {neg} 概率", f"{probs[neg]:.4f} → {softmax(wrong)[neg]:.4f}  ← 反而变大")
    assert LOGITS[neg] < 0 and softmax(wrong)[neg] > probs[neg], "反例: 负 logit 一律除以 penalty, 概率应反而变大"


if __name__ == "__main__":
    main()
