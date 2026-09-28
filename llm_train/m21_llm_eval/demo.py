"""
M21 — 通用 LLM 评测: perplexity · 污染检测 · pass@k · LLM-as-judge 位置偏差

是什么: 训完之后怎么打分, 以及四个最常见的 "分数会骗人" 的地方。
    perplexity : PPL = exp(逐 token 交叉熵的均值)。按 token 平均还是按文档平均、用哪个分词器, 数字都会变。
    污染检测   : 测试题出现在训练语料里, 分数就是背出来的。GPT-3 / Llama 报告用 n-gram 重叠标出 "脏" 题。
    pass@k     : 每题采 n 个样本、c 个对, 无偏估计 1 − C(n−c, k)/C(n, k)。直接代 1 − (1 − c/n)^k 会系统性低估。
    judge 偏差 : LLM 当裁判时偏爱第一个位置; 交换 A/B 各判一次, 两次一致才算数, 偏差被抵消。
关键公式: CE = −(1/T)·Σ log p(x_t | x_<t);  PPL = e^CE;  bits/字符 = CE·T / (字符数·ln 2)
          pass@k 无偏 = E_c[1 − C(n−c,k)/C(n,k)],  数值稳定写法 1 − Π_{i=n−c+1..n} (1 − k/i)
读代码盯住: `pass_at_k` 那一行乘积, `ngram_flags` 的 "任一 n-gram 命中即判脏", `judge_swap` 的一致性规则。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv, make_rng

JUDGE_BIAS = 0.8                                                         # 模拟裁判给第一个位置加的分


# ---- perplexity -------------------------------------------------------- #

def sample_markov(rs, P, n):
    """按转移矩阵 P [V, V] 采一条长 n 的序列 [n]。P[i, j] = 前一个词是 i 时, 下一个是 j 的概率。"""
    x = [rs.randint(len(P))]
    for _ in range(n - 1):
        x.append(rs.choice(len(P), p=P[x[-1]]))
    return np.array(x)


def token_nll(P_model, x):
    """逐 token 负对数似然 (第一个 token 按均匀计)。均值就是交叉熵 CE, exp(CE) 就是 PPL。"""
    # P_model[前一个, 后一个] → [n-1]; 前面拼上第一个 token 的 log V → [n]
    return np.r_[np.log(len(P_model)), -np.log(P_model[x[:-1], x[1:]])]


# ---- 污染检测 ---------------------------------------------------------- #

def ngrams(words, n):
    """词列表里所有连续 n 个词组成的集合。"""
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_flags(tests, train_docs, n):
    """GPT-3 的做法: 测试题的任一 n-gram 在训练集出现过 → 判为污染。"""
    bank = set().union(*(ngrams(d, n) for d in train_docs))              # 训练集里出现过的全部 n-gram
    return np.array([bool(ngrams(t, n) & bank) for t in tests])          # [题数] bool


# ---- pass@k ------------------------------------------------------------ #

def pass_at_k(n, c, k):
    """Codex 论文的无偏估计: 从 n 个里无放回挑 k 个, 至少一个对的概率。"""
    if n - c < k:
        return 1.0                                                       # 错的不到 k 个: 挑 k 个必然挑到对的
    # C(n−c, k) / C(n, k) 约分后就是这 c 项的乘积; 不算阶乘, 大 n 也不溢出
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ---- judge ------------------------------------------------------------- #

def rule_judge(q_first, q_second, rs, bias=JUDGE_BIAS, noise=0.5):
    """模拟 LLM 裁判: 看质量差, 但给第一个位置加 bias 分。返回 True = 选第一个。"""
    return q_first - q_second + bias + noise * rs.randn(*np.shape(q_first)) > 0


def judge_swap(qa, qb, rs):
    """A/B 各当一次第一个; 两次都选 A → A 赢 (+1), 都选 B → B 赢 (−1), 不一致 → 平局 (0)。"""
    a_first = rule_judge(qa, qb, rs)                                     # True = 选 A
    b_first = rule_judge(qb, qa, rs)                                     # True = 选 B
    return np.where(a_first & ~b_first, 1, np.where(~a_first & b_first, -1, 0))


def main() -> None:
    banner("M21 - LLM Evaluation")
    rs = make_rng(21)

    # ---- 1) perplexity ----
    V = 20
    print(f"\n[1] Perplexity: 语料由一个已知的 {V} 词二元马尔可夫链生成")
    P_true = rs.dirichlet(np.full(V, 0.2), size=V)                       # 稀疏的转移矩阵 → 熵低, 可预测
    train = sample_markov(rs, P_true, 20000)
    test_docs = [sample_markov(rs, P_true, n) for n in rs.randint(5, 400, size=40)]
    counts = np.full((V, V), 0.1)                                        # 每格先放 0.1: 没见过的转移概率也不为 0, 免得 log(0)
    np.add.at(counts, (train[:-1], train[1:]), 1)                        # 同一个 (前, 后) 对出现多次要累加 → 用 add.at
    models = {
        "均匀分布": np.full((V, V), 1 / V),
        "unigram": np.tile(np.bincount(train, minlength=V) / len(train), (V, 1)),
        "bigram (训练集计数)": counts / counts.sum(1, keepdims=True),
        "真实分布": P_true,
    }
    ppl = {}
    test = np.concatenate(test_docs)                                     # 40 篇测试文档拼成一条
    for name, Pm in models.items():
        nll = np.concatenate([token_nll(Pm, d) for d in test_docs])      # 每篇单独算 (首 token 按均匀计), 再拼起来
        ppl[name] = np.exp(nll.mean())
        kv(f"{name} CE / PPL", f"{nll.mean():.4f} nats / {ppl[name]:.3f}")
    assert abs(ppl["均匀分布"] - V) < 1e-9, "均匀分布的 PPL 恰好是词表大小: PPL = '等效在几个词里瞎猜'"
    assert ppl["真实分布"] < ppl["bigram (训练集计数)"] < ppl["unigram"] < ppl["均匀分布"], \
        "模型越接近真实分布 PPL 越低: 真实分布 < bigram < unigram < 均匀分布"

    per_doc = [token_nll(models["bigram (训练集计数)"], d) for d in test_docs]
    corpus_ppl = np.exp(np.concatenate(per_doc).mean())
    doc_avg_ppl = np.mean([np.exp(n.mean()) for n in per_doc])
    kv("按 token 平均 / 按文档平均的 PPL", f"{corpus_ppl:.3f} / {doc_avg_ppl:.3f}  (后者被短文档拉高)")
    assert doc_avg_ppl > corpus_ppl, "Jensen: 先 exp 再平均 ≥ 先平均再 exp; 两种口径不能混比"
    total_nll = np.concatenate(per_doc).sum()
    for tokens_per_char in (1.0, 0.5):                                   # 同一段文本, 分词器把 2 个字符并成 1 个 token
        n_tok = len(test) * tokens_per_char
        kv(f"  若 {tokens_per_char:.1f} token/字符: PPL / bits-per-char",
           f"{np.exp(total_nll / n_tok):7.3f} / {total_nll / len(test) / np.log(2):.4f}")
    kv("结论", "同一模型同一文本, token 更少 PPL 就更高; 跨分词器只能比 bits-per-char / bits-per-byte")

    # ---- 2) 污染检测 ----
    # n_test 道题, 每道 q_len 个词; 每种泄漏方式 n_leak 道; 轻改换 n_swap 个词; 重度改写每 every 个词换 1 个
    n_test, q_len, n_leak, n_swap, every = 200, 30, 30, 2, 3
    print(f"\n[2] 污染检测: {n_test} 道测试题; 训练语料混入 {n_leak} 道原题、{n_leak} 道轻改 (换 {n_swap} 个词)、"
          f"{n_leak} 道重度改写 (每 {every} 个词换 1 个)")
    vocab = [f"w{i}" for i in range(400)]
    zipf = 1.0 / np.arange(1, 401)
    gen = lambda n: [vocab[i] for i in rs.choice(400, size=n, p=zipf / zipf.sum())]
    tests = [gen(q_len) for _ in range(n_test)]
    leak = rs.permutation(n_test)                                        # 打乱后前 3·n_leak 道是泄漏题, 每 n_leak 道一种方式
    kind = np.array(["clean"] * n_test, dtype=object)
    kind[leak[:n_leak]], kind[leak[n_leak:2 * n_leak]], kind[leak[2 * n_leak:3 * n_leak]] = "verbatim", "light", "heavy"
    train_docs = [gen(rs.randint(50, 300)) for _ in range(2000)]
    for i in leak[:3 * n_leak]:
        t = list(tests[i])
        if kind[i] == "light":
            for j in rs.choice(q_len, n_swap, replace=False):
                t[j] = vocab[rs.randint(400)]
        elif kind[i] == "heavy":
            t = [vocab[rs.randint(400)] if j % every == every - 1 else w for j, w in enumerate(t)]
        train_docs.append(gen(rs.randint(20, 100)) + t + gen(rs.randint(20, 100)))   # 埋在正常文档中间
    flags = ngram_flags(tests, train_docs, n=8)
    for kd in ("verbatim", "light", "heavy", "clean"):
        kv(f"  {kd:<8} 被标为污染的比例", f"{flags[kind == kd].mean():.2f}")
    assert flags[kind == "verbatim"].all(), "逐字泄漏的题必须全部被标出"
    assert flags[kind == "light"].mean() > 0.9, "只换 2 个词的题, 90% 以上应被标出"
    assert flags[kind == "clean"].mean() < 0.02, "干净题几乎不被误标 (8-gram 在随机文本里撞上的概率很低)"
    assert flags[kind == "heavy"].mean() < 0.2, "每 3 词换 1 个 → 没有完整的 8-gram 幸存, n-gram 检测失效"

    # 一个 "见过就会" 的模型: 见过的题 90% 答对, 没见过的 4 选 1 瞎猜
    correct = rs.rand(n_test) < np.where(kind == "clean", 0.25, 0.9)
    kv(f"报告分 (全部 {n_test} 题)", f"{correct.mean():.3f}")
    kv("去掉被标记的题后", f"{correct[~flags].mean():.3f}  (重度改写的题漏网, 仍在虚高)")
    kv("真实能力 (只算 clean)", f"{correct[kind == 'clean'].mean():.3f}")
    assert correct.mean() > correct[~flags].mean() > correct[kind == "clean"].mean(), \
        "分数排序: 全部题 > 去掉被标记的题 > 只算干净题; 漏网的污染题让分数仍然虚高"

    # ---- 3) pass@k ----
    n, R, n_prob, (ba, bb) = 20, 400, 300, (0.5, 2.0)                   # 每题样本数, 重复次数, 题数, Beta 参数
    print(f"\n[3] pass@k: {n_prob} 道题, 每题真实通过率 p ~ Beta({ba:g}, {bb:g}), 每题采 n = {n} 个样本, 重复 {R} 次")
    p = rs.beta(ba, bb, size=n_prob)
    c = rs.binomial(n, p, size=(R, len(p)))                              # [重复, 题]
    for k in (1, 5, 10):
        truth = np.mean(1 - (1 - p) ** k)                                # 用真实通过率 p 算的 pass@k
        table = np.array([pass_at_k(n, ci, k) for ci in range(n + 1)])   # 按 c 查表
        unb = table[c].mean(1)                                           # 每次重复的数据集级估计
        naive = (1 - (1 - c / n) ** k).mean(1)                           # 朴素: 把 c/n 当 p 直接代进去
        se = unb.std() / np.sqrt(R)                                      # R 次重复的均值的标准误
        kv(f"k={k:<2} 真值 / 无偏 / 朴素",
           f"{truth:.4f} / {unb.mean():.4f} (偏差 {unb.mean() - truth:+.4f}) / {naive.mean():.4f} (偏差 {naive.mean() - truth:+.4f})")
        assert abs(unb.mean() - truth) < 4 * se + 1e-12, "无偏估计的均值落在真值的 MC 误差内"
        if k > 1:
            assert naive.mean() - truth < -10 * se, "朴素估计系统性偏低 (1−(1−p̂)^k 对 p̂ 是凹的)"
        else:
            assert np.allclose(unb, naive), "k=1 时两者都是 c/n"

    # ---- 4) LLM-as-judge 位置偏差 ----
    n_pair = 2000
    print(f"\n[4] LLM-as-judge: {n_pair} 对回答, 裁判 = 质量差 + {JUDGE_BIAS:g} 的 '第一个位置' 加分 + 噪声")
    qa, qb = rs.randn(n_pair), rs.randn(n_pair)                              # A 是先给的那个 (随机决定, 与质量无关)
    a_better = qa > qb
    pick_a = rule_judge(qa, qb, rs)
    kv("单次判: 选第一个的比例", f"{pick_a.mean():.3f}  (真实 A 更好的比例 {a_better.mean():.3f})")
    acc_when = {s: (pick_a == a_better)[a_better == (s == "A")].mean() for s in ("A", "B")}
    kv("单次判: 准确率 (好的在 A / 在 B)", f"{acc_when['A']:.3f} / {acc_when['B']:.3f}")
    v = judge_swap(qa, qb, rs)                                           # [n_pair] 取值 +1 / −1 / 0
    decisive = v != 0                                                    # 两次判决一致, 给出了结论
    ok = (v == 1) == a_better                                            # 结论与真实质量一致
    acc_swap = {s: ok[decisive & (a_better == (s == "A"))].mean() for s in ("A", "B")}
    kv("交换判: 平局 (两次不一致) 比例", f"{1 - decisive.mean():.3f}  (代价: 2 倍调用, 接近的对判不出)")
    kv("交换判: A 胜比例 (非平局中)", f"{(v[decisive] == 1).mean():.3f}")
    kv("交换判: 准确率 (好的在 A / 在 B)", f"{acc_swap['A']:.3f} / {acc_swap['B']:.3f}")
    assert pick_a.mean() > 0.65 and acc_when["A"] - acc_when["B"] > 0.3, "单次判严重偏向第一个位置"
    assert abs(acc_swap["A"] - acc_swap["B"]) < 0.05, "交换 + 一致性: 位置偏差被抵消"
    assert ok[decisive].mean() > (pick_a == a_better).mean(), "给出结论的那部分更准"

    print("\n  OK: PPL = e^CE 且口径敏感; n-gram 查得到原题查不到改写; pass@k 用无偏估计; 裁判交换顺序消位置偏差。")


if __name__ == "__main__":
    main()
