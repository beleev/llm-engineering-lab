"""
beam.py — beam search 解码: 每步保留累计 log 概率最高的 width 条前缀。

瓶颈: greedy 每步只看眼前, 第一步选了次优 token 后面再也回不来; 精确求 argmax_y log P(y|x) 要搜 V^T 条序列。
      beam search 是折中: 每步 width·V 个候选里留 width 条, 代价 ≈ width 倍的 decode。
关键数字: width=1 ≡ greedy; 它不是精确搜索 —— 窄 beam 可能在中途剪掉 greedy 那条, 最后反而输给 greedy。
读代码盯住: 分数是**累计** log 概率 (负数, 越长越小), 所以带 EOS 时短序列天然占便宜;
      length penalty 在**选最终答案**时把分数除以 len^α 来抵消。
真实系统: HF `generate(num_beams=, length_penalty=)` (BeamSearchScorer), vLLM `BeamSearchParams`;
      用在翻译 / 摘要 / ASR 这类"答案基本唯一"的任务, 开放式对话默认都是采样。
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import numpy as np

from llm_infer.m10_sampling.samplers import SamplingParams, sample


def log_softmax(x: np.ndarray) -> np.ndarray:
    """logits (V,) → log 概率 (V,)。先减最大值再 exp, 防溢出。"""
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def beam_search(lm, prompt: Sequence[int], width: int, max_new: int,
                alpha: float = 0.0, eos_id: Optional[int] = None) -> List[Tuple[float, List[int]]]:
    """返回所有候选 [(累计 log 概率, tokens)], 按 score / len^alpha 降序; [0] 即 beam search 的输出。

    每步: 每条活 beam 取 top-(width+1) 个 token 作候选 (多 1 个, 保证有 EOS 时仍能凑满 width 条活 beam),
    全体候选按累计 log 概率排序; 以 EOS 结尾且排进前 width 名的收进 finished, 其余依次补满 width 条活 beam。
    """
    logits, kv = lm.prefill(np.asarray(prompt))
    beams = [(0.0, [], kv, log_softmax(logits[-1]))]      # (累计 logp, tokens, KV, 下一步 logp (V,))
    finished: List[Tuple[float, List[int]]] = []
    for _ in range(max_new):
        cand = [(s + lp[t], toks + [int(t)], kv)
                for s, toks, kv, lp in beams for t in np.argsort(-lp)[: width + 1]]
        cand.sort(key=lambda c: -c[0])                    # 按累计 log 概率从高到低
        beams = []
        for rank, (s, toks, kv) in enumerate(cand):
            if toks[-1] == eos_id:
                if rank < width:                          # 排在 width 名之外的 EOS 候选直接丢掉
                    finished.append((s, toks))
                continue
            logits, kv2 = lm.decode_step(toks[-1], kv)    # 每条活 beam 各自一份 KV (真实系统用 m02 的 ref_count 共享前缀块)
            beams.append((s, toks, kv2, log_softmax(logits)))
            if len(beams) == width:
                break
    finished += [(s, toks) for s, toks, _, _ in beams]    # 到 max_new 还没结束的也算候选
    # 长度惩罚只在这里起作用: alpha=0 按累计 log 概率排, alpha=1 按每 token 平均 log 概率排
    return sorted(finished, key=lambda f: -f[0] / len(f[1]) ** alpha)


def decode(lm, prompt: Sequence[int], max_new: int, rng=None,
           eos_id: Optional[int] = None) -> Tuple[float, List[int]]:
    """逐 token 解码: rng=None 即 greedy, 否则 T=1 采样 (m10 的 sample)。返回 (累计 log 概率, tokens)。"""
    logits, kv = lm.prefill(np.asarray(prompt))
    lp = log_softmax(logits[-1])
    score, toks = 0.0, []
    for _ in range(max_new):
        t = int(np.argmax(lp)) if rng is None else sample(lp, SamplingParams(temperature=1.0), rng=rng)
        score, toks = score + lp[t], toks + [t]
        if t == eos_id:
            break
        logits, kv = lm.decode_step(t, kv)
        lp = log_softmax(logits)
    return score, toks
