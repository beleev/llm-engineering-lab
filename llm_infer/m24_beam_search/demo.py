"""
m24 demo — beam search vs greedy vs 采样 (core 的 TinyLM, 与 m19 同一配置)

    [1] 固定长度 20 token: 序列 log 概率随 width 上升, 但越宽越重复; width=1 ≡ greedy; 窄 beam 可能输给 greedy
    [2] 多样性: 同一个 prompt, beam 的 8 条候选几乎一样, 8 条采样各不相同
    [3] 带 EOS: 不加长度惩罚时 beam 偏爱短输出 (几乎空回复); length penalty α 把长度拉回来

运行: python -m llm_infer.m24_beam_search.demo
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

from llm_infer.core import ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m24_beam_search.beam import beam_search, decode, log_softmax

N_PROMPT, MAX_NEW = 10, 20            # prompt 个数, 每个 prompt 生成几个 token
WIDTHS = [1, 2, 4, 8, 16, 32]         # 要比较的 beam 宽度
N_CAND = 8                            # [2][3] 用的 beam 宽度, 也是每个 prompt 的采样条数; 必须在 WIDTHS 里
EOS, EOS_BIAS = 2, 3.0                # EOS 的 token id; [3] 里加在 EOS logit 上的常数
EOS_MAX_NEW = 24                      # [3] 里每条输出最多生成几个 token


def rep2(toks) -> float:
    """重复率 = 1 − 不同 bigram 数 / bigram 总数。0 = 没有重复的两字组合。"""
    bigrams = list(zip(toks, toks[1:]))
    return 1 - len(set(bigrams)) / len(bigrams)


class EosBiased:
    """给 EOS 的 logit 加一个常数, 模拟一个会停的模型。

    随机权重的 TinyLM 从没学过何时结束: 默认配置下 P(EOS) 不到 1%/步。
    接口与 TinyLM 相同 (prefill / decode_step), 可以直接传给 beam_search 和 decode。
    """

    def __init__(self, lm: TinyLM, bias: float):
        self.lm, self.bias = lm, bias

    def prefill(self, ids):
        logits, kv = self.lm.prefill(ids)
        logits = logits.copy(); logits[:, EOS] += self.bias      # copy: 不改模型返回的原数组
        return logits, kv

    def decode_step(self, t, kv):
        logits, kv = self.lm.decode_step(t, kv)
        logits = logits.copy(); logits[EOS] += self.bias
        return logits, kv


def mean_p_eos(lm, prompts, n: int) -> float:
    """沿 greedy 路径走 n 步 (遇到 EOS 也不停), 返回每步 P(EOS) 的均值: 模型"想停"的程度。"""
    ps = []
    for p in prompts:
        logits, kv = lm.prefill(np.asarray(p))
        lp = log_softmax(logits[-1])                              # 最后一个位置的分布 = 第 1 个新 token
        for _ in range(n):
            ps.append(np.exp(lp[EOS]))
            logits, kv = lm.decode_step(int(np.argmax(lp)), kv)
            lp = log_softmax(logits)
    return float(np.mean(ps))


def main():
    banner("M24 - Beam Search (序列 log 概率 / 重复退化 / 长度惩罚)")
    lm = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128))
    rs = np.random.RandomState(0)
    prompts = [rs.randint(3, 128, size=6) for _ in range(N_PROMPT)]

    print(f"\n[1] {N_PROMPT} 个 prompt × {MAX_NEW} token, 不停在 EOS; 分数 = 整条序列的 log P (越大越好)")
    greedy = [decode(lm, p, MAX_NEW) for p in prompts]
    g_lp = np.array([s for s, _ in greedy])
    samples = [decode(lm, p, MAX_NEW, rng=rs) for p in prompts for _ in range(N_CAND)]
    print(f"  {'':<12}{'logP/token':>11}{'输给 greedy 的 prompt':>22}{'重复率 rep-2':>14}")
    print(f"  {'采样 T=1':<12}{np.mean([s for s, _ in samples]) / MAX_NEW:>11.3f}{'—':>18}"
          f"{np.mean([rep2(t) for _, t in samples]):>17.3f}")
    print(f"  {'greedy':<12}{g_lp.mean() / MAX_NEW:>11.3f}{'—':>18}{np.mean([rep2(t) for _, t in greedy]):>17.3f}")
    rows, outs = {}, {}
    for w in WIDTHS:
        outs[w] = [beam_search(lm, p, w, MAX_NEW) for p in prompts]
        best = [o[0] for o in outs[w]]
        lp = np.array([s for s, _ in best])
        rows[w] = (lp, np.mean([rep2(t) for _, t in best]))
        if w == 1:
            assert all(b[1] == g[1] for b, g in zip(best, greedy)), \
                "width=1 的 beam search 应与 greedy 逐 token 相同"
        print(f"  {f'beam w={w}':<12}{lp.mean() / MAX_NEW:>11.3f}{int((lp < g_lp - 1e-4).sum()):>18}{rows[w][1]:>17.3f}")
    for w, (lp, _) in rows.items():
        assert lp.mean() >= g_lp.mean() - 1e-4, f"w={w}: 平均序列 log 概率应 ≥ greedy (1e-4 是浮点余量)"
    assert (rows[WIDTHS[-1]][0] >= g_lp - 1e-4).all(), "最宽的 beam 在每个 prompt 上都 ≥ greedy"
    losers = {w: int((lp < g_lp - 1e-4).sum()) for w, (lp, _) in rows.items()}
    assert any(losers.values()), "窄 beam 不是精确搜索: 至少一个 prompt 上输给 greedy"
    assert rows[WIDTHS[-1]][1] > rows[1][1] + 0.05, "越宽 → 找到的序列越重复"
    kv("窄 beam 输给 greedy", f"{losers}  ← 中途把 greedy 那条剪掉了, 后面没补回来")

    print(f"\n[2] 多样性: 同一个 prompt 的 {N_CAND} 条输出, 两两之间平均有几个位置不同 (满分 {MAX_NEW}, {N_PROMPT} 个 prompt 平均)")
    # ham: 一组序列两两之间的平均汉明距离 (有几个位置的 token 不同)
    ham = lambda seqs: np.mean([sum(a != b for a, b in zip(x, y)) for x, y in combinations(seqs, 2)])
    d_beam = np.mean([ham([t for _, t in o[:N_CAND]]) for o in outs[N_CAND]])
    d_samp = np.mean([ham([t for _, t in samples[N_CAND * i:N_CAND * (i + 1)]]) for i in range(N_PROMPT)])
    kv(f"beam w={N_CAND} 的 {N_CAND} 条候选", f"{d_beam:.1f} 个位置不同")
    kv(f"{N_CAND} 条 T=1 采样", f"{d_samp:.1f} 个位置不同")
    assert d_beam < d_samp / 3, "beam 的候选共享大段前缀, 只在少数位置上不同"

    biased = EosBiased(lm, EOS_BIAS)
    p_raw, p_bias = mean_p_eos(lm, prompts, EOS_MAX_NEW), mean_p_eos(biased, prompts, EOS_MAX_NEW)
    print(f"\n[3] 带 EOS: EOS logit +{EOS_BIAS}, 每步 P(EOS) 均值 {p_raw:.1%} → {p_bias:.1%} "
          f"(greedy 路径 {EOS_MAX_NEW} 步); max_new={EOS_MAX_NEW}, 看输出长度")
    assert p_raw < 0.01 < p_bias, f"加偏置前模型几乎不停 (<1%/步), 加偏置后才会停: {p_raw:.2%} → {p_bias:.2%}"
    lens = {"greedy": np.mean([len(decode(biased, p, EOS_MAX_NEW, eos_id=EOS)[1]) for p in prompts]),
            "采样 T=1": np.mean([len(decode(biased, p, EOS_MAX_NEW, rng=rs, eos_id=EOS)[1])
                               for p in prompts for _ in range(4)])}
    beam_len = {a: np.mean([len(beam_search(biased, p, N_CAND, EOS_MAX_NEW, a, EOS)[0][1]) for p in prompts])
                for a in (0.0, 0.5, 1.0)}                         # 长度惩罚 α → 平均输出长度
    lens.update({f"beam w={N_CAND}, α={a}": n for a, n in beam_len.items()})
    for name, n in lens.items():
        kv(name, f"平均 {n:.1f} token")
    assert beam_len[0.0] < lens["greedy"] / 3, "不加长度惩罚: 累计 log 概率越短越高 → 近乎空回复"
    assert beam_len[1.0] > beam_len[0.0] + 8, "α=1 (按平均 log 概率选) 把长度拉回来"
    print("  结论: beam 找的是 '最可能' 的序列, 而最可能的往往是短的、重复的、千篇一律的 —— 不适合开放式对话。")


if __name__ == "__main__":
    main()
