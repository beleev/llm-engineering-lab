"""
m18 demo — MHA / GQA / MQA / MLA 的 KV cache 账本 + 可运行 decode

    [1] 真实模型配置的每 token KV 字节数, 以及 40 GB 能装下多少 batch×context
    [2] 四种变体: 带 cache 的增量 decode == 全量重算; 实测 cache.nbytes == 公式
    [3] GQA 代码路径 == 独立写的逐头 MHA 基线 (含 n_kv = n_head)
    [4] MLA 的 absorb 技巧: q·(C W_UK)ᵀ == (q W_UKᵀ)·Cᵀ, 两种形式输出一致

运行: python -m llm_infer.m18_kv_attention_variants.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m18_kv_attention_variants.attention_variants import (
    GQALayer, MLALayer, mha_reference, run_stack, cache_nbytes,
    kv_bytes_per_token, mla_bytes_per_token,
)

BUDGET = 40 * 2**30       # 留给 KV 的显存 (不含权重/激活), 可改
CTX = 8192                # 算 "能装多少 batch" 时每条请求的上下文长度
# [1] 的真实模型配置: (名字, 注意力类型, n_kv 头数 (MLA 为 d_c), head_dim (MLA 为 d_rope), 层数)
MODELS = [
    ("LLaMA-2-7B", "MHA", 32, 128, 32),
    ("LLaMA-3-8B", "GQA", 8, 128, 32),
    ("(假想) 7B", "MQA", 1, 128, 32),
    ("(假想) DS-V3", "MHA", 128, 128, 61),
    ("DeepSeek-V3", "MLA", 512, 64, 61),
]


def footprint_table():
    """[1] 把真实模型的配置代进公式, 打印每 token KV 字节数和 KV 预算下的容量。"""
    print(f"\n[1] 每 token KV 字节 (fp16) 与 {BUDGET / 2**30:.0f} GiB KV 预算下的容量")
    rows = []
    for name, kind, a, b, n_layer in MODELS:
        pad = 13 - len(name) - sum(ord(c) > 0x2E80 for c in name)   # 中文字符占两格, 按显示宽度补齐
        if kind == "MLA":
            rows.append((f"{name}{' ' * pad}{kind} {f'{a}+{b}':>9}, {n_layer}L", mla_bytes_per_token(a, b, n_layer)))
        else:
            rows.append((f"{name}{' ' * pad}{kind} {f'{a}kv×{b}':>9}, {n_layer}L", kv_bytes_per_token(a, b, n_layer)))
    print(f"  {'配置':<32}{'KiB/token':>10}{'最多 token':>10}{f'batch@{CTX}':>12}")   # 中文表头按显示宽度少补几格
    for name, b in rows:
        print(f"  {name:<34}{b / 1024:>10.1f}{BUDGET // b:>12,}{BUDGET // b // CTX:>12}")
    assert rows[0][1] == 4 * rows[1][1] == 32 * rows[2][1], \
        "KV 字节数应正比于 n_kv: MHA (32) = 4 × GQA (8) = 32 × MQA (1)"
    kv("DS-V3: MLA 相对同尺寸 MHA 省", f"{rows[3][1] / rows[4][1]:.1f}x")
    return rows


def main():
    banner("M18 - 各种 attention 变体的 KV cache (MHA/GQA/MQA/MLA)")
    footprint_table()

    D, H, dh, L, T0, T = 64, 8, 16, 3, 12, 20                      # T0: prefill 长度; T: 总长度
    rs = np.random.RandomState(0)
    x = rs.randn(T, D).astype(np.float32)
    mk_gqa = lambda n_kv: [GQALayer(rs, D, H, n_kv, dh) for _ in range(L)]
    d_c, d_rope = 24, 8
    # 每项: (各层, 每 token cache 字节数的公式值 (fp32 所以 nbytes=4), forward 的额外参数)
    mla_name = f"MLA (d_c={d_c},d_rope={d_rope})"
    variants = {f"{kind} (n_kv={n_kv})": (mk_gqa(n_kv), kv_bytes_per_token(n_kv, dh, L, 4), {})
                for kind, n_kv in (("MHA", H), ("GQA", 2), ("MQA", 1))}
    variants[mla_name] = ([MLALayer(rs, D, H, dh, d_rope, dh, d_c) for _ in range(L)],
                          mla_bytes_per_token(d_c, d_rope, L, 4), {})
    variants["MLA absorb"] = (variants[mla_name][0], variants[mla_name][1], {"absorb": True})

    print(f"\n[2] 增量 decode vs 全量重算 (D={D}, {H} 头×{dh}, {L} 层, prefill {T0} + decode {T - T0}, fp32)")
    print(f"  {'变体':<22}{'max|Δ|':>10}{'cache 字节':>11}{'公式值':>7}")
    for name, (layers, bpt, kw) in variants.items():
        full, _ = run_stack(layers, x, **kw)                       # (T, D) 无 cache 一次算完
        out, caches = run_stack(layers, x[:T0], **kw)              # prefill
        outs = [out]
        for t in range(T0, T):                                     # 每步只喂 1 个 token
            o, caches = run_stack(layers, x[t:t + 1], caches, **kw)
            outs.append(o)
        diff = np.abs(np.concatenate(outs) - full).max()
        print(f"  {name:<24}{diff:>10.2e}{cache_nbytes(caches):>13,}{bpt * T:>10,}")
        assert diff < 1e-5, f"{name}: 带 cache 的增量 decode 应等于全量重算, 实际差 {diff:.2e}"
        assert cache_nbytes(caches) == bpt * T, \
            f"{name}: 实测 cache 字节数 {cache_nbytes(caches)} 应等于公式值 {bpt * T}"

    print("\n[3] GQA 广播路径 vs 逐头 MHA 基线 (KV 头权重复制给组内每个 query 头)")
    for n_kv in (8, 2, 1):
        layer = GQALayer(rs, D, H, n_kv, dh)
        diff = np.abs(layer.forward(x, layer.new_cache())[0] - mha_reference(x, layer)).max()
        kv(f"n_kv={n_kv}  max|Δ|", f"{diff:.2e}")
        assert diff < 1e-5, f"n_kv={n_kv}: GQA 的广播路径应等于逐头 MHA 基线, 实际差 {diff:.2e}"

    print("\n[4] MLA absorb: 把 W_UK 吸进 q, K 就是 cache 本身 (≡ head_dim=d_c+d_rope 的 MQA)")
    mla = variants["MLA absorb"][0][0]
    q = rs.randn(H, 1, dh).astype(np.float32)                      # (H, 1, d_nope)
    C = rs.randn(T, d_c).astype(np.float32)                        # (T, d_c) latent cache
    w_uk = mla.w_uk.reshape(d_c, H, dh).transpose(1, 2, 0)         # (H, d_nope, d_c)
    s_naive = q @ np.swapaxes(C @ w_uk.transpose(0, 2, 1), -1, -2)  # q·(C W_UK)ᵀ : 先物化 (H, T, d_nope) 的 K
    s_absorb = (q @ w_uk) @ C.T                                    # (q W_UKᵀ)·Cᵀ: 只碰 (T, d_c)
    d_score = np.abs(s_naive - s_absorb).max()
    a, _ = run_stack(variants["MLA absorb"][0], x)
    b, _ = run_stack(variants["MLA absorb"][0], x, absorb=True)
    kv("score max|Δ|", f"{d_score:.2e}")
    kv("整网输出 naive vs absorb max|Δ|", f"{np.abs(a - b).max():.2e}")
    kv("naive 每步物化的 per-head K/V", f"{T * H * (dh + d_rope + dh) * 4:,} B/层 (absorb: 0)")
    assert d_score < 1e-4, f"absorb 前后的注意力分数应相等: q·(C W_UK)ᵀ == (q W_UKᵀ)·Cᵀ, 实际差 {d_score:.2e}"
    assert np.abs(a - b).max() < 1e-5, "MLA 的朴素形式与 absorb 形式, 整网输出应一致"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
