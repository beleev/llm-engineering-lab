"""
tts.py — test-time compute: 模型不变, 推理时多花 token 换正确率。

瓶颈: 单次采样的正确率被两类错误压住 —— 随机的"粗心"错 (换个样本就可能对) 和系统性的"误解" (模型的众数就是错的)。
      多采样 + 投票 / 模型自查只能消掉前者; 要越过后者得有**外部**判分器 (ORM / PRM)。
关键数字: 一条 K 步的链每步答对 p → 整条答对 p^K; p=0.75, K=4 → 0.32。
      pass@N = 1 − (1 − p^K)^N 涨得很快, 但前提是有东西能把对的那条挑出来。
读代码盯住: 四种方法只差在 "token 花在哪、靠什么挑":
      best_of_n 并行 N 条靠 ORM 看终点; majority_vote 并行 N 条靠众数; prm_beam_search 每步靠 PRM 剪枝; think 串行自查。
真实系统: self-consistency (Wang 2022); best-of-N + ORM (Cobbe 2021); PRM 引导搜索 (Math-Shepherd / PRM800K);
      o1 / DeepSeek-R1 的长思考; s1 的 budget forcing (到上限就截断, 想停时追加 "Wait")。

这里的"模型"是带噪声的程序化 sampler: 每步给 6 个候选值一组 logits, 用 m10 的 `sample` 抽一个;
ORM / PRM 是"真值 + 高斯噪声"的程序化判分器 —— 真实系统里它们是训练出来的 (见 README)。
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from llm_infer.m10_sampling.samplers import SamplingParams, sample

OPS = {"+": lambda v, a: v + a, "-": lambda v, a: v - a, "*": lambda v, a: v * a}
CONFUSE = {"+": "*", "-": "+", "*": "+"}        # 系统性误解: 把运算符看错
OFFSETS = (0, 1, -1, 10, -10)                   # 候选 0 = 正确值, 其余 = 粗心错 (个位 / 十位差 1)
LOGIT_OK, LOGIT_SLIP, LOGIT_TRAP = 3.0, 0.3, 3.5
# T=1 时: 普通步答对 e^3/(e^3+5·e^0.3) = 0.75; 陷阱步 "看错运算符" 的 logit 3.5 > 正确的 3.0 → 模型的众数就是错的
SIGMA = 0.5                                     # ORM / PRM 判分噪声 (真值 0/1 上加 N(0, σ²))
DETECT = 0.5                                    # 自查时发现一处粗心错的概率 (看不见自己的误解)
STOP = 0.6                                      # 模型每次"想停"的概率 (没有 budget forcing 时, 过早自信)


@dataclass(frozen=True)
class Problem:
    x0: int
    ops: Tuple[Tuple[str, int], ...]            # K 步, 每步 (运算符, 操作数)
    trap: int                                   # 陷阱步下标; -1 = 这道题没有陷阱

    def truth(self, v: int, i: int) -> int:
        """第 i 步在上一值 v 之后的正确值 (局部正确, PRM 判的就是它)。"""
        op, a = self.ops[i]
        return OPS[op](v, a)

    @property
    def answer(self) -> int:
        v = self.x0
        for i in range(len(self.ops)):
            v = self.truth(v, i)
        return v


def make_problems(n: int, k: int, trap_frac: float, rng: np.random.RandomState) -> List[Problem]:
    out = []
    for _ in range(n):
        ops = tuple((str(rng.choice(list(OPS))), int(rng.randint(3, 10))) for _ in range(k))  # a≥3: v·a ≠ v+a
        out.append(Problem(int(rng.randint(2, 10)), ops, int(rng.randint(k)) if rng.rand() < trap_frac else -1))
    return out


def step_dist(prob: Problem, i: int, v: int) -> Tuple[np.ndarray, np.ndarray]:
    """第 i 步的候选值 (6,) 与 logits (6,): 正确值 + 4 种粗心错 + 看错运算符。"""
    op, a = prob.ops[i]
    c = OPS[op](v, a)
    values = np.array([c + d for d in OFFSETS] + [OPS[CONFUSE[op]](v, a)])
    logits = np.array([LOGIT_OK] + [LOGIT_SLIP] * 4 + [LOGIT_TRAP if i == prob.trap else LOGIT_SLIP])
    return values, logits


def sample_step(prob: Problem, i: int, v: int, rng, temperature: float = 1.0) -> int:
    """模型写第 i 步 = 1 个 token: 在 6 个候选上用 m10 的 sample 抽一个。temperature=0 即 greedy。"""
    values, logits = step_dist(prob, i, v)
    return int(values[sample(logits, SamplingParams(temperature=temperature), rng=rng)])


def rollout(prob: Problem, rng, temperature: float = 1.0) -> List[int]:
    """一条完整的链 [v1..vK]。错误会传下去: 后面每步都在错的值上"正确地"算。"""
    chain: List[int] = []
    for i in range(len(prob.ops)):
        chain.append(sample_step(prob, i, chain[-1] if chain else prob.x0, rng, temperature))
    return chain


def prm(prob: Problem, i: int, v_prev: int, v: int, rng) -> float:
    """过程奖励: 这一步在上一值之后算得对不对 (+噪声)。程序化判分, 真实 PRM 是训练出来的。"""
    return float(v == prob.truth(v_prev, i)) + rng.normal(0, SIGMA)


def orm(prob: Problem, final: int, rng) -> float:
    """结果奖励: 只看最终答案 (+噪声)。"""
    return float(final == prob.answer) + rng.normal(0, SIGMA)


def best_of_n(prob: Problem, chains: List[List[int]], rng) -> int:
    return chains[int(np.argmax([orm(prob, c[-1], rng) for c in chains]))][-1]


def majority_vote(chains: List[List[int]]) -> int:
    """self-consistency: 最终答案取众数 (并列取先出现的)。不需要任何判分器。"""
    return Counter(c[-1] for c in chains).most_common(1)[0][0]


def prm_beam_search(prob: Problem, width: int, expand: int, rng) -> Tuple[int, int]:
    """每个 beam 采 expand 个下一步, PRM 给每步打分, 按累计分留 top-width。返回 (答案, 生成 token 数)。

    错的步在它出现的那一步就被剪掉, 不会再为它的后续步付费 —— 这是它比 best-of-N 省 token 的原因。
    """
    beams: List[Tuple[float, List[int]]] = [(0.0, [])]
    tokens = 0
    for i in range(len(prob.ops)):
        cand = []
        for score, chain in beams:
            v = chain[-1] if chain else prob.x0
            for _ in range(expand):
                nxt = sample_step(prob, i, v, rng)
                tokens += 1
                cand.append((score + prm(prob, i, v, nxt, rng), chain + [nxt]))
        cand.sort(key=lambda c: -c[0])
        beams = cand[:width]
    return beams[0][1][-1], tokens


def believes_ok(prob: Problem, i: int, v_prev: int, v: int) -> bool:
    """模型自查时认为这一步对吗: 真对的认对; 陷阱步上它自己的误解也认对 (看不见系统性错误)。"""
    op, a = prob.ops[i]
    return v == prob.truth(v_prev, i) or (i == prob.trap and v == OPS[CONFUSE[op]](v_prev, a))


def think(prob: Problem, rng, min_tokens: int = 0, max_tokens: int = 10 ** 9) -> Tuple[int, int]:
    """串行长思考: 先写完 K 步, 再一遍遍自查; 发现粗心错就从那一步重写。返回 (答案, token 数)。

    每检查一步记 1 个 token, 重写一步记 1 个 token。budget forcing (s1):
      - 到 max_tokens 立刻截断, 拿最近一条写完的链作答; 第一遍都没写完 → 只能答中间值;
      - 模型想停 (概率 STOP) 但还没到 min_tokens → 追加 "Wait", 再查一遍。
    """
    k, chain, tokens = len(prob.ops), [], 0            # chain = 最近一条写完的链

    def write_from(i: int) -> bool:                     # 从第 i 步重写到结尾; 预算不够返回 False
        nonlocal chain, tokens
        draft = chain[:i]
        while len(draft) < k:
            if tokens >= max_tokens:
                chain = chain or draft                  # 第一遍没写完: 以中间值作答
                return False
            draft.append(sample_step(prob, len(draft), draft[-1] if draft else prob.x0, rng))
            tokens += 1
        chain = draft
        return True

    done = write_from(0)
    while done and tokens < max_tokens and (tokens < min_tokens or rng.rand() >= STOP):
        for i in range(k):                              # 一遍自查, 找到第一处就重写并结束这一遍
            if tokens >= max_tokens:
                break
            tokens += 1
            v_prev = chain[i - 1] if i else prob.x0
            if not believes_ok(prob, i, v_prev, chain[i]) and rng.rand() < DETECT:
                done = write_from(i)
                break
    return (chain[-1] if chain else prob.x0), tokens
