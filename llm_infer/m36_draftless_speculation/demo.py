"""
m36 demo — 不用草稿模型的投机: prompt lookup / 跨请求 n-gram 缓存

    [1] 无损: 真实 TinyLM 当 target, prompt lookup 投机的输出与 target 逐 token greedy 完全相同
    [2] 三类负载 × 两种查法 (target 是合成的 oracle, 它的 greedy 输出是事先写好的参考答案):
        代码编辑 (输出 ≈ 输入的文件, 改几处) / RAG 摘录 (输出里夹着整段原文) / 开放生成 (输出与输入无关)
    [3] 同一类请求反复出现 (输出彼此相似, 但不在当前 prompt 里): 只有跨请求的缓存查得到
验证和接受规则全部用 m07 的 speculative_decode (greedy); 草稿模型的 forward 次数恒为 0。

运行: python -m llm_infer.m36_draftless_speculation.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m07_speculative_decoding.speculative import speculative_decode
from llm_infer.m36_draftless_speculation.draftless import OracleTarget, PromptLookupDrafter, SuffixCacheDrafter

V = 1000          # oracle 负载的词表大小: 足够大, 随机文本里 n-gram 偶然撞上的概率低
K = 8             # 每轮最多提几个草稿 token


def code_edit(rs):
    """代码编辑: prompt = 指令 + 一个 300 token 的文件; 输出 = 同一个文件, 随机改 12 处。"""
    doc = rs.randint(0, V, 300).tolist()
    out = list(doc)
    for i in rs.choice(300, 12, replace=False):
        out[i] = int(rs.randint(0, V))
    return rs.randint(0, V, 20).tolist() + doc, out


def rag_excerpt(rs):
    """RAG 摘录: prompt = 3 段各 120 token 的检索结果 + 问题; 输出 = 4 次 (8 个新 token + 从原文抄 25 个)。"""
    passages = [rs.randint(0, V, 120).tolist() for _ in range(3)]
    out = []
    for _ in range(4):
        p = passages[rs.randint(3)]
        s = int(rs.randint(0, 120 - 25))
        out += rs.randint(0, V, 8).tolist() + p[s:s + 25]
    return sum(passages, []) + rs.randint(0, V, 16).tolist(), out


def open_gen(rs):
    """开放生成: prompt 200 token, 输出 132 个与 prompt 无关的新 token。"""
    return rs.randint(0, V, 200).tolist(), rs.randint(0, V, 132).tolist()


def agent_call(rs, template):
    """同一类请求: prompt 是每次都不同的观察结果; 输出 = 固定的 40 token 调用模板, 其中 3 个参数位每次不同。"""
    out = list(template)
    for i in (6, 19, 31):
        out[i] = int(rs.randint(0, V))
    return rs.randint(0, V, 150).tolist(), out


def spec(prompt, ref, drafter):
    """跑一次投机 → (输出是否等于参考答案, target forward 次数, 草稿接受率, 提了多少草稿 token)。"""
    target = OracleTarget(prompt, ref, V)
    proposed0 = drafter.proposed
    out, calls, accepts = speculative_decode(target, drafter, prompt, len(ref), K)
    proposed = drafter.proposed - proposed0
    return out == list(prompt) + list(ref), calls, sum(accepts) / max(proposed, 1), proposed


def main():
    banner("M36 - 不用草稿模型的投机解码 (prompt lookup / n-gram 缓存)")

    print("\n[1] 无损: 真实 TinyLM (随机权重) 当 target")
    lm = TinyLM(ModelConfig(vocab_size=128, max_seq_len=256))
    prompt = np.random.RandomState(0).randint(0, 128, 40)
    ref = lm.generate_greedy(prompt, 64)
    d = PromptLookupDrafter()
    out, calls, acc = speculative_decode(lm, d, prompt, 64, K)
    kv("输出 == target 逐 token greedy", out == ref)
    kv("target forward: 逐 token / 投机", f"64 / {calls}  (每轮接受 {np.mean(acc):.2f} 个, 草稿 {d.proposed} 个)")
    assert out == ref, "prompt lookup 投机的 greedy 输出必须与 target 逐 token greedy 完全相同"
    assert d.calls == 0, "prompt lookup 没有草稿模型, draft forward 次数应为 0"

    print(f"\n[2] 三类负载 (oracle target, K={K}, 每类 5 条请求取平均; 逐 token 需要的 target forward = 输出长度)")
    print(f"  {'负载':<10} {'查法':<14} {'草稿':>5} {'接受率':>6} {'forward 逐token→投机':>18} {'每次 forward 产出':>14}")
    rs = np.random.RandomState(1)
    summary = {}
    for wname, make in (("代码编辑", code_edit), ("RAG 摘录", rag_excerpt), ("开放生成", open_gen)):
        reqs = [make(rs) for _ in range(5)]
        for dname, drafter in (("prompt lookup", PromptLookupDrafter()), ("n-gram 缓存", SuffixCacheDrafter())):
            rows = []
            for p, r in reqs:
                if isinstance(drafter, SuffixCacheDrafter):
                    drafter.start_request()
                ok, calls, _, prop = spec(p, r, drafter)
                assert ok, f"{wname} / {dname}: 输出必须与 target 的 greedy 输出完全相同"
                rows.append((len(r), calls, prop))
                if isinstance(drafter, SuffixCacheDrafter):
                    drafter.finish_request(r)
            n_out, n_calls, prop = (sum(x[i] for x in rows) for i in range(3))
            # 每轮产出 = 接受数 + 1, 所以接受的草稿总数 = 输出数 − 1 次 prefill 产出 − 验证轮数
            rate = (n_out - len(rows) - (n_calls - len(rows))) / max(prop, 1)
            summary[wname, dname] = (rate, n_out / n_calls, prop)
            print(f"  {wname:<10} {dname:<14} {prop:>6} {rate:>7.1%} {n_out:>11} → {n_calls:<5} {n_out / n_calls:>12.2f}x")
            assert drafter.calls == 0, f"{dname}: 没有草稿模型, draft forward 次数应为 0"
    for w in ("代码编辑", "RAG 摘录"):
        rate, speed = summary[w, "prompt lookup"][:2]
        assert rate > 0.4, f"{w}: 输出大段照抄输入, prompt lookup 的接受率应超过 40%, 实际 {rate:.1%}"
        assert speed > 2.5, f"{w}: 每次 target forward 应产出 2.5 个 token 以上, 实际 {speed:.2f}"
    waste = summary["开放生成", "prompt lookup"][2]
    kv("开放生成: prompt lookup 白验的草稿 token", f"{waste}  (decode 带宽受限时几乎免费, 大 batch 算力受限时是浪费)")
    rate, speed = summary["开放生成", "prompt lookup"][:2]
    assert rate < 0.05, f"开放生成: 输出不在输入里, 接受率应接近 0, 实际 {rate:.1%}"
    assert speed < 1.1, f"开放生成: 几乎退化成逐 token decode, 每次 forward 产出应 < 1.1, 实际 {speed:.2f}"

    print("\n[3] 同一类请求反复出现 8 次: 输出是 40 token 的固定模板 (3 个参数位每次不同), 模板不在 prompt 里")
    template = rs.randint(0, V, 40).tolist()
    reqs = [agent_call(rs, template) for _ in range(8)]
    pl, sc = PromptLookupDrafter(), SuffixCacheDrafter()
    res = {"prompt lookup": [], "n-gram 缓存": []}
    for p, r in reqs:
        res["prompt lookup"].append(spec(p, r, pl))
        sc.start_request()
        res["n-gram 缓存"].append(spec(p, r, sc))
        sc.finish_request(r)
    for name, rows in res.items():
        assert all(r[0] for r in rows), f"{name}: 输出必须与 target 的 greedy 输出完全相同"
        print(f"  {name:<14} 第 1 次请求 forward {rows[0][1]:>2}, 之后平均 {np.mean([r[1] for r in rows[1:]]):5.1f}"
              f"   接受率 (第 2~8 次) {np.mean([r[2] for r in rows[1:]]):.1%}")
    pl_rate = np.mean([r[2] for r in res["prompt lookup"][1:]])
    sc_rate = np.mean([r[2] for r in res["n-gram 缓存"][1:]])
    assert sc_rate > 0.5, f"跨请求缓存记住了模板, 第 2 次以后的接受率应超过 50%, 实际 {sc_rate:.1%}"
    assert pl_rate < 0.1, f"模板不在当前 prompt 里, prompt lookup 的接受率应低于 10%, 实际 {pl_rate:.1%}"
    assert res["n-gram 缓存"][0][1] >= res["n-gram 缓存"][-1][1] * 2, "第 1 次请求缓存是空的, forward 次数应是后面的 2 倍以上"

    print("\n  ✓ 草稿来自查表, 没有草稿模型: 照抄输入的负载每次 forward 产出 2.5 个以上, 开放生成退化成逐 token")
    print("  ✓ 无损: 所有负载上输出都与 target 的 greedy 输出逐 token 相同")


if __name__ == "__main__":
    main()
