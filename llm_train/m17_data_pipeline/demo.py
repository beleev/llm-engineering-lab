"""
M17 — 预训练数据流水线: 去重 → 质量过滤 → 配比采样

是什么: 网页语料进训练前的三道工序。
    去重    : 精确哈希只能抓逐字相同的副本; 改了几个词的转载要用 MinHash + LSH 近似去重。
    质量过滤: Gopher / C4 式的启发式规则 (长度、平均词长、符号比例、停用词、重复 n-gram)。
    配比采样: 各来源按权重混合; 数据源温度配比 p_i ∝ n_i^(1/T) 把小语料 (wiki / math) 上采样。
解决的瓶颈: 重复数据让模型背诵、浪费算力; 垃圾页拉低质量; 按原始比例采样时小而精的语料几乎见不到。
关键公式: MinHash 两签名某一位相等的概率 = Jaccard(A, B)
          LSH 分 b 段每段 r 行: 被选为候选的概率 = 1 − (1 − s^r)^b, 阈值约 (1/b)^(1/r)
读代码盯住: `minhash_signatures` 的一行 (a·x + b) mod p, `lsh_dedup` 的分桶 → 校验 → 并查集, 和 `temperature_weights`。
"""
from __future__ import annotations

import zlib
from collections import Counter

import numpy as np

from llm_train.core import banner, kv, make_rng

STOP = ["the", "of", "and", "to", "in", "is", "that", "it", "for", "on", "with", "as"]
PRIME = (1 << 31) - 1                                                    # Mersenne 素数; x < 2^31, a < 2^31 → a·x 不溢出 int64


def make_vocab(rs, n=400):
    """造 n 个随机小写字母串当词表, 每个词长 3~8。"""
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    return ["".join(rs.choice(letters, rs.randint(3, 9))) for _ in range(n)]


def gen_words(rs, vocab, n):
    """Zipf 分布的词 + 30% 停用词, 像一段 "正常的自然语言"。"""
    p = 1.0 / np.arange(1, len(vocab) + 1)                               # Zipf: 第 k 个词的概率 ∝ 1/k
    ids = rs.choice(len(vocab), size=n, p=p / p.sum())
    return [STOP[rs.randint(len(STOP))] if rs.rand() < 0.3 else vocab[i] for i in ids]


def perturb(rs, words, vocab, rate):
    """模拟转载: 每个词以 rate 概率被替换, 0.5·rate 被删除, 0.5·rate 在后面插入一个新词。

    三种改动互斥, 合计约 2·rate 的词被改动。
    """
    out = []
    for w in words:
        u = rs.rand()                                                    # [0, 1) 均匀; 落在哪一段决定怎么改
        if u < rate:
            out.append(vocab[rs.randint(len(vocab))])                   # 替换
        elif u < 1.5 * rate:
            continue                                                     # 删除
        else:
            out.append(w)
            if u > 1 - 0.5 * rate:
                out.append(vocab[rs.randint(len(vocab))])               # 插入
    return out


# ---- 去重 --------------------------------------------------------------- #

def shingles(words, n=5):
    """词级 5-gram → 32 位整数集合。crc32 而不是 hash(): 后者每个进程加盐, 不可复现。"""
    return {zlib.crc32(" ".join(words[i:i + n]).encode()) % PRIME for i in range(len(words) - n + 1)}


def jaccard(a, b):
    """两个集合的 Jaccard 相似度 = 交集大小 / 并集大小。"""
    return len(a & b) / len(a | b)


def minhash_signatures(sets, num_perm, rs):
    """签名 [num_docs, num_perm]: 第 j 位 = min_x (a_j·x + b_j) mod p, 即随机置换下集合的最小元素。"""
    # 每个哈希函数一对系数 (a_j, b_j), 各 [num_perm]。a 从 1 起: a = 0 会把所有 x 映到同一个值
    a = rs.randint(1, PRIME, size=num_perm).astype(np.int64)
    b = rs.randint(0, PRIME, size=num_perm).astype(np.int64)
    sig = np.empty((len(sets), num_perm), dtype=np.int64)
    for i, s in enumerate(sets):
        x = np.fromiter(s, dtype=np.int64)[:, None]                     # [|s|, 1]
        sig[i] = ((a * x + b) % PRIME).min(axis=0)                      # [|s|, num_perm] → 沿元素取最小 → [num_perm]
    return sig


def union_find_removed(n, pairs):
    """把判为重复的对连成簇, 每簇只留下标最小的那篇, 返回被删的下标集合。"""
    parent = list(range(n))                                              # parent[i] == i 表示 i 是簇的根

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]                                # 路径压缩: 顺手挂到祖父上, 树越查越矮
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        parent[max(ri, rj)] = min(ri, rj)                                # 下标小的当根 → 根就是每簇要留下的那篇
    return {i for i in range(n) if find(i) != i}


def exact_dedup(docs):
    """整篇文本算一个 crc32, 和见过的相同就删。返回被删的下标集合。"""
    seen, removed = set(), set()
    for i, words in enumerate(docs):
        h = zlib.crc32(" ".join(words).encode())
        (removed.add(i) if h in seen else seen.add(h))
    return removed


def lsh_dedup(docs, bands, rows, threshold, rs):
    """MinHash 签名切成 bands 段 × rows 行; 任一段完全相同即成候选, 再用签名估的 Jaccard 校验。

    返回 (被删的下标集合, 候选对数)。
    """
    sig = minhash_signatures([shingles(w) for w in docs], bands * rows, rs)   # [num_docs, bands·rows]
    pairs = set()
    for band in range(bands):
        buckets = {}                                                     # 这一段的 rows 个哈希值 (转成 bytes) → 文档下标列表
        for i, key in enumerate(map(bytes, sig[:, band * rows:(band + 1) * rows])):
            buckets.setdefault(key, []).append(i)
        for ids in buckets.values():
            pairs.update((ids[0], j) for j in ids[1:])                  # 同桶都连到桶首, 并查集会补全传递关系
    # 校验: 两个签名里相等的位占多少, 就是 Jaccard 的估计值 (文件头第一条公式)
    confirmed = [(i, j) for i, j in pairs if np.mean(sig[i] == sig[j]) >= threshold]
    return union_find_removed(len(docs), confirmed), len(pairs)


def score_dedup(removed, group):
    """group[i] 是第 i 篇的来源 id; 同组下标最小的那篇算原件, 其余算重复。

    返回 (召回 = 删对的 / 该删的, 误杀篇数 = 删了但不该删的)。
    """
    first = {}                                                           # 来源 id → 该组第一篇的下标
    for i, g in enumerate(group):
        first.setdefault(g, i)
    dups = {i for i, g in enumerate(group) if first[g] != i}
    return len(removed & dups) / len(dups), len(removed - dups)


# ---- 质量过滤 ----------------------------------------------------------- #

SYMBOLS = set("#$%{}[]<>|=;*@~^")


def quality_reasons(words):
    """返回命中的规则名列表; 空列表 = 通过。阈值取自 Gopher / C4 的量级, 按玩具语料缩放。"""
    n = len(words)
    reasons = []
    if not 30 <= n <= 1000:
        reasons.append("长度")
    if not 3 <= np.mean([len(w) for w in words]) <= 10:
        reasons.append("平均词长")
    if sum(any(c in SYMBOLS for c in w) for w in words) / n > 0.1:
        reasons.append("符号比例")
    if sum(w in STOP for w in words) < 2:
        reasons.append("停用词")
    if n >= 3:
        top = Counter(tuple(words[i:i + 3]) for i in range(n - 2)).most_common(1)[0][1]
        if top * 3 / n > 0.2:                                            # 最常见的 3-gram 占掉全文 20% 以上的词
            reasons.append("重复3-gram")
    return reasons


def make_quality_corpus(rs, vocab):
    """造三份语料: 200 篇正常文本, 5 类垃圾各 40 篇, 20 篇代码。返回 (good, bad 字典, code)。"""
    good = [gen_words(rs, vocab, rs.randint(60, 200)) for _ in range(200)]
    bad = {
        "过短": [gen_words(rs, vocab, rs.randint(5, 25)) for _ in range(40)],
        "符号刷屏": [sum(([w, "###", "$$$"] for w in gen_words(rs, vocab, 40)), []) for _ in range(40)],
        "重复刷词": [("click here to buy now " * 20).split() for _ in range(40)],
        "乱码长串": [["".join(rs.choice(list("qxzjkv0123456789"), rs.randint(14, 24))) for _ in range(80)]
                   for _ in range(40)],
        "关键词堆砌": [[vocab[i] for i in rs.randint(0, len(vocab), 80)] for _ in range(40)],
    }
    code = [("def f ( x ) : return { x [ 0 ] } ; if x == y : z = x * 2 ; " * 4).split() for _ in range(20)]
    return good, bad, code


# ---- 配比采样 ----------------------------------------------------------- #

def temperature_weights(tokens, T):
    """p_i ∝ n_i^(1/T): T=1 按原始比例, T→∞ 趋于均匀。"""
    w = np.asarray(tokens, dtype=float) ** (1.0 / T)
    return w / w.sum()


def credit_sampler(p, n):
    """确定性配比: 每步给各源加 p_i 的额度, 选额度最大的源, 扣 1。偏差不会累积。"""
    credit, out = np.zeros(len(p)), np.empty(n, dtype=int)               # 各源的额度; 第 t 步选中的源
    for t in range(n):
        credit += p
        out[t] = credit.argmax()
        credit[out[t]] -= 1
    return out


def main() -> None:
    banner("M17 - Pretraining Data Pipeline")
    rs = make_rng(17)
    vocab = make_vocab(rs)

    # ---- 1) 去重: 精确哈希 vs MinHash-LSH ----
    n_orig, n_copy, n_near, n_tmpl, rate = 300, 30, 90, 40, 0.02         # rate: perturb 的替换概率
    print(f"\n[1] 去重: {n_orig} 篇原件 + {n_copy} 份逐字副本 + {n_near} 份轻改转载 (每词约 {2 * rate:.0%} 被改)"
          f" + {n_tmpl} 篇共享模板的不同页面")
    originals = [gen_words(rs, vocab, rs.randint(80, 160)) for _ in range(n_orig)]
    docs, group = list(originals), list(range(n_orig))                   # group[i]: 第 i 篇出自哪篇原件
    for g in rs.choice(n_orig, n_copy, replace=False):
        docs.append(list(originals[g])); group.append(g)
    for g in rs.choice(n_orig, n_near, replace=False):
        docs.append(perturb(rs, originals[g], vocab, rate)); group.append(g)
    template = gen_words(rs, vocab, 60)                                  # 同一站点的页头页脚: 内容不同, 算 **不同** 文档
    for k in range(n_tmpl):
        docs.append(template + gen_words(rs, vocab, 60)); group.append(1000 + k)
    order = rs.permutation(len(docs))                                    # 打乱: 原件不一定排在前面
    docs, group = [docs[i] for i in order], [group[i] for i in order]

    sets = [shingles(w) for w in docs]
    # near: 同源但文本不同的文档对 (轻改转载) 的真实 Jaccard; tmpl_j: 模板页两两之间的
    near = [jaccard(sets[i], sets[j]) for i in range(len(docs)) for j in range(i)
            if group[i] == group[j] and docs[i] != docs[j]]
    tmpl = [i for i, g in enumerate(group) if g >= 1000]
    tmpl_j = [jaccard(sets[i], sets[j]) for i in tmpl for j in tmpl if i < j]
    kv("轻改转载与原件的真实 Jaccard", f"均值 {np.mean(near):.2f}, 最低 {min(near):.2f}")
    kv("模板页之间的真实 Jaccard", f"均值 {np.mean(tmpl_j):.2f}")

    bands, rows, thr = 32, 4, 0.5
    kv("LSH 参数", f"{bands} 段 × {rows} 行 = {bands * rows} 个哈希; S 曲线阈值 ≈ {(1 / bands) ** (1 / rows):.2f}")
    sims = (0.2, 0.4, 0.6, 0.8)
    kv(f"P(成候选) @ s={'/'.join(map(str, sims))}", [round(1 - (1 - s**rows) ** bands, 3) for s in sims])
    ex_removed = exact_dedup(docs)
    lsh_removed, n_cand = lsh_dedup(docs, bands, rows, thr, make_rng(170))
    ex_recall, ex_fk = score_dedup(ex_removed, group)
    lsh_recall, lsh_fk = score_dedup(lsh_removed, group)
    n_dup = len(docs) - len(set(group))                                  # 总篇数 − 不同来源数
    kv("应删的重复篇数", n_dup)
    kv("精确哈希: 召回 / 误杀篇数", f"{ex_recall:.3f} / {ex_fk}")
    kv("MinHash-LSH: 召回 / 误杀篇数", f"{lsh_recall:.3f} / {lsh_fk}   (候选对 {n_cand}, 校验阈值 Ĵ ≥ {thr})")
    assert ex_recall < 0.3 and ex_fk == 0, "精确哈希只抓逐字副本 (30/120, 外加恰好一个词都没改的转载)"
    assert lsh_recall > 0.95, "近似去重抓住绝大多数轻改转载"
    assert lsh_fk <= 2, "Jaccard ≈ 0.3 的模板页大多不会被当成重复"

    # ---- 2) 质量过滤 ----
    good, bad, code = make_quality_corpus(rs, vocab)
    print(f"\n[2] 启发式质量过滤: {len(good)} 篇正常文本 + {len(bad)} 类各 {len(next(iter(bad.values())))} 篇垃圾"
          f" + {len(code)} 篇代码")
    kept_good = sum(not quality_reasons(w) for w in good) / len(good)
    kv("正常文本保留率", f"{kept_good:.3f}")
    for name, ds in bad.items():
        hits = Counter(r for w in ds for r in quality_reasons(w))
        caught = sum(bool(quality_reasons(w)) for w in ds) / len(ds)
        kv(f"  {name} 被拦", f"{caught:.2f}  命中规则 {dict(hits)}")
    bad_caught = sum(bool(quality_reasons(w)) for ds in bad.values() for w in ds) / sum(map(len, bad.values()))
    code_killed = sum(bool(quality_reasons(w)) for w in code) / len(code)
    kv("垃圾总拦截率", f"{bad_caught:.3f}")
    kv("代码被误杀率", f"{code_killed:.2f}  (命中 {sorted(set(r for w in code for r in quality_reasons(w)))})")
    assert kept_good >= 0.95, "正常文本的保留率应不低于 95%"
    assert bad_caught >= 0.95, "垃圾的总拦截率应不低于 95%"
    assert code_killed == 1.0, "为网页调的规则会把正常代码当垃圾 —— 所以代码要单独走一条流水线"

    # ---- 3) 配比采样 ----
    names = ["web", "code", "books", "wiki", "math"]
    tokens = np.array([800, 150, 40, 8, 2]) * 1e6
    budget = 500e6
    print(f"\n[3] 配比: {len(names)} 个来源, 训练预算 {budget / 1e6:.0f}M token (总库存 {tokens.sum() / 1e6:.0f}M)")
    q = tokens / tokens.sum()                                            # 原始占比 (T=1 时的采样概率)
    print(f"  {'T':>4}" + "".join(f"{n:>20}" for n in names))
    ups = {}
    for T in (1, 2, 5):
        p = temperature_weights(tokens, T)
        ups[T] = p / q                                                   # 上采样倍数: >1 被多采, <1 被少采
        cells = "".join(f"{pi:>7.3f} ×{u:>5.2f} {pi * budget / n:>4.1f}ep" for pi, u, n in zip(p, ups[T], tokens))
        print(f"  {T:>4}{cells}")
    print("        (每格: 占比  ×上采样倍数  ep = 该来源在预算内被过几遍)")
    assert np.allclose(ups[1], 1.0), "T=1 就是按原始比例采样, 上采样倍数全是 1"
    assert ups[5][-1] > ups[2][-1] > 5, "T 越大, 最小的 math 被上采样越多"
    assert ups[5][0] < ups[2][0] < 1, "代价: 大语料 web 被下采样"

    T_mix = 2                                                            # 下面比较两种抽样方式时用的温度
    p = temperature_weights(tokens, T_mix)
    n_seq, batch = 64 * 1500, 64                                         # 1500 个 batch, 每个 64 条
    iid = rs.choice(len(p), size=n_seq, p=p)
    det = credit_sampler(p, n_seq)
    err = {k: np.abs(np.bincount(s, minlength=len(p)) / n_seq - p).max() for k, s in (("iid", iid), ("credit", det))}
    per_batch = {k: max(np.abs(np.bincount(s[i:i + batch], minlength=len(p)) / batch - p).max()
                        for i in range(0, n_seq, batch)) for k, s in (("iid", iid), ("credit", det))}
    kv(f"全局配比误差 max|p̂−p| (T={T_mix})", f"iid 抽样 {err['iid']:.4f} / 额度调度 {err['credit']:.6f}")
    kv(f"单个 batch({batch}) 内最坏误差", f"iid 抽样 {per_batch['iid']:.3f} / 额度调度 {per_batch['credit']:.3f}")
    assert err["iid"] < 0.005, "iid 抽样的全局配比误差应低于 0.005"
    assert err["credit"] < 1e-4, "额度调度的全局配比误差应低于 1e-4"
    assert per_batch["credit"] < 2 / batch < per_batch["iid"], "确定性调度: 前缀计数误差 < 1 篇, 任一 batch (两前缀之差) 误差 < 2/batch"

    print("\n  OK: MinHash-LSH 抓住精确哈希漏掉的转载; 规则过滤拦下垃圾但误杀代码; 温度采样把小语料上采样。")


if __name__ == "__main__":
    main()
