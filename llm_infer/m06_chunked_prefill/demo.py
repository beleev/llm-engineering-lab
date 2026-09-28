"""
m06 demo — Chunked Prefill: (1) 数值上与整段 prefill 等价; (2) 与 decode 混批后 TBT 尖刺消失

运行: python -m llm_infer.m06_chunked_prefill.demo
[1] 用真 TinyLM 对拍分块与整段 prefill; [2] 用 m03 的真调度器 + 代价模型 (CostModel, 非实测) 算延迟。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import TinyLM, ModelConfig
from llm_infer.core.utils import banner, kv
from llm_infer.m06_chunked_prefill.chunked_prefill import chunked_prefill, simulate, CostModel


def main():
    banner("M06 - Chunked Prefill (Sarathi)")
    lm = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128))
    T = 64
    prompt = np.random.RandomState(0).randint(3, 128, size=T)
    logits_full, kv_full = lm.prefill(prompt)

    print(f"\n[1] 分块 vs 整段 prefill 的数值差 (T={T})")
    print(f"  {'chunk':>6}  {'logits max|Δ|':>14}  {'KV max|Δ|':>12}  {'attn 分数矩阵峰值':>18}")
    for chunk in [8, 16, 32, 64]:
        logits_c, kv_c = chunked_prefill(lm, prompt, chunk)
        # 分块只返回最后一块的 logits, 所以拿整段 logits 的末尾同样多行来比
        d_logits = np.max(np.abs(logits_full[-len(logits_c):] - logits_c))
        d_kv = max(np.max(np.abs(a[0] - b[0])) for a, b in zip(kv_full, kv_c))   # 各层 K 的最大差
        peak = chunk * T                                   # 最后一块: chunk 行 × T 列
        print(f"  {chunk:>6}  {d_logits:>14.2e}  {d_kv:>12.2e}  {peak:>10} (整段 {T * T})")
        assert d_logits < 1e-4, f"chunk={chunk}: 分块 prefill 的 logits 应等于整段, 实际差 {d_logits:.2e}"
        assert d_kv < 1e-4, f"chunk={chunk}: 分块 prefill 的 KV 应等于整段, 实际差 {d_kv:.2e}"

    n_dec, long_at, long_len = 4, 10, 1024                 # decode 用户数; 长 prompt 的到达 step 和长度
    print(f"\n[2] 负载: {n_dec} 条请求在 decode, 第 {long_at} 步来了一条 {long_len}-token 的长 prompt")
    cost = CostModel()
    kv("代价模型 (非实测)", f"step_ms = {cost.fixed_ms} + {cost.per_token_ms} × batch_tokens")
    arrivals = [(0, 16, 40)] * n_dec + [(long_at, long_len, 8)]   # (到达 step, prompt_len, max_new)
    res = {}
    for name, chunked, budget in [("prefill 优先 (不分块)", False, 2048), ("分块混批 B=128", True, 128),
                                  ("分块混批 B=512", True, 512)]:
        r = res[name] = simulate(chunked, budget, arrivals, cost)
        tbt = np.concatenate([r["tbt"][s] for s in range(n_dec)])      # 只看那几条 decode 用户
        r["max_tbt"], r["p50_tbt"], r["long_ttft"] = tbt.max(), np.median(tbt), r["ttft"][n_dec]   # 长请求排在最后
        print(f"  {name:<22} 每步最大 token={max(r['step_tokens']):>5}  TBT p50={r['p50_tbt']:6.1f}ms  "
              f"max={r['max_tbt']:6.1f}ms  长请求 TTFT={r['long_ttft']:6.1f}ms  总耗时={r['total_ms']:.0f}ms")

    base, c128, c512 = res["prefill 优先 (不分块)"], res["分块混批 B=128"], res["分块混批 B=512"]
    print(f"\n  decode 用户最大卡顿: {base['max_tbt']:.0f}ms → {c128['max_tbt']:.0f}ms "
          f"({base['max_tbt'] / c128['max_tbt']:.1f}x); 代价: 长请求 TTFT {base['long_ttft']:.0f}ms → {c128['long_ttft']:.0f}ms")
    assert max(c128["step_tokens"]) <= 128, "token 预算 128 就是每步 token 数的上限"
    assert max(c512["step_tokens"]) <= 512, "token 预算 512 就是每步 token 数的上限"
    assert c128["max_tbt"] < c512["max_tbt"] < base["max_tbt"], "预算越小, decode 用户的最大卡顿应越小"
    # +1e-9: 浮点累加的舍入余量
    assert c128["max_tbt"] <= cost.step_ms(128) + 1e-9, "TBT 应被 step_ms(token 预算) 封顶"
    assert c128["long_ttft"] > base["long_ttft"], "代价: 分块后长请求自己的 TTFT 应变长"
    print("  ✓ 分块数值等价; TBT 上限 = step_ms(token 预算); 预算是 TBT 与 TTFT 之间的旋钮")


if __name__ == "__main__":
    main()
