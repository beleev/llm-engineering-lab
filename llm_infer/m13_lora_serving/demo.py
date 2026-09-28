"""
m13 demo — Multi-LoRA serving: 5 条请求、3 个 adapter, 同一个 batch 一次算完。

[1] 零初始化的 B → 新 adapter 的输出与底模逐位相同 (LoRA 的初始化约定)
[2] BGMV == SGMV == 逐请求循环 == 合并权重参考 (max-abs-diff)
[3] 显存: N 份合并权重 vs 1 份底模 + N 个 adapter
[4] 底模 gemm 次数与耗时: 逐请求 vs 同 batch

运行: python -m llm_infer.m13_lora_serving.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, Timer
from llm_infer.m13_lora_serving.lora import (
    init_adapter, merged_forward, loop_forward, bgmv_forward, sgmv_forward,
)


def max_diff(outs_a, outs_b) -> float:
    """两组逐请求输出 (列表, 每项 (T_i, d_out)) 之间的最大绝对差。"""
    return max(float(np.abs(a - b).max()) for a, b in zip(outs_a, outs_b))


def main():
    banner("M13 - Multi-LoRA Serving (Punica BGMV / SGMV)")
    rs = np.random.RandomState(0)
    d_in, d_out, r, n_adapter = 64, 96, 4, 3            # d_in ≠ d_out: A/B 的形状写反会直接报错而不是悄悄算错
    W = rs.randn(d_out, d_in).astype(np.float32) * 0.05  # (d_out, d_in), nn.Linear 布局
    adapters = [init_adapter(d_in, d_out, r, seed=s) for s in range(n_adapter)]
    lens, ids = [3, 5, 2, 4, 1], [0, 1, 0, 2, 1]        # 每条请求的 token 数 / adapter id
    xs = [rs.randn(t, d_in).astype(np.float32) for t in lens]

    print("\n[1] 初始化约定: A (r, d_in) 随机, B (d_out, r) 全零, ΔW = B·A")
    kv("A.shape / B.shape / ΔW.shape", f"{adapters[0].A.shape} / {adapters[0].B.shape} / {adapters[0].delta_w().shape}")
    base = [x @ W.T for x in xs]
    base_cat = np.concatenate(xs) @ W.T                  # (15, d_out) 与 bgmv 内部同一次 gemm, 所以可以要求逐位相等
    d0 = float(np.abs(np.concatenate(bgmv_forward(xs, ids, W, adapters)) - base_cat).max())
    kv("B=0 时 |LoRA 输出 - 底模输出|", d0)
    assert d0 == 0.0, f"B 全零时 ΔW = 0, LoRA 输出应与底模逐位相同, 实际差 {d0}"
    for ad in adapters:                                  # 模拟 "训练之后": B 不再是 0
        ad.B = rs.randn(d_out, r).astype(np.float32) * 0.1

    print(f"\n[2] 数值等价: {len(xs)} 条请求, lens={lens}, adapter_ids={ids}")
    ref = merged_forward(xs, ids, W, adapters)
    results = {"loop (逐请求)": loop_forward, "BGMV (逐 token gather)": bgmv_forward, "SGMV (按 adapter 分段)": sgmv_forward}
    for name, fn in results.items():
        d = max_diff(fn(xs, ids, W, adapters), ref)
        kv(f"{name} vs 合并权重", f"{d:.2e}")
        assert d < 1e-5, f"{name} 应等于合并权重的参考答案, 实际差 {d:.2e}"
    effect = max_diff(ref, base)
    kv("(对照) LoRA 对输出的改变量", f"{effect:.2e}")
    assert effect > 1e-2, "LoRA 对输出的改变量太小, 上面的等价性比较等于在比一堆 0"
    wrong = max_diff(loop_forward(xs, [0] * len(xs), W, adapters), ref)
    kv("(对照) 全部错用 adapter 0 的误差", f"{wrong:.2e}")
    assert wrong > 1e-2, "全部错用 adapter 0 时误差应明显大于 0, 否则路由错误测不出来"

    print("\n[3] 显存 (float32 字节)")
    per_adapter = sum(ad.A.nbytes + ad.B.nbytes for ad in adapters) // n_adapter
    kv("底模 W", f"{W.nbytes} B   单个 adapter: {per_adapter} B = r·(d_in+d_out)·4 (W 的 {per_adapter / W.nbytes:.1%})")
    assert per_adapter == r * (d_in + d_out) * 4, \
        f"单个 adapter 应占 r·(d_in+d_out)·4 字节 (fp32), 实际 {per_adapter}"
    for n in (3, 1000):
        merged, shared = n * W.nbytes, W.nbytes + n * per_adapter
        kv(f"{n} 个 adapter: 合并 / 不合并", f"{merged / 1e6:.2f} MB / {shared / 1e6:.2f} MB  ({merged / shared:.1f}x)")
    big, r_big = 4096, 16                                # 真实尺寸的一个例子: 7B 级的 d, 常用的 rank
    kv(f"d={big}, r={r_big} 时 adapter/W", f"{r_big * 2 * big / big**2:.2%}")

    n_req = 500
    print(f"\n[4] {n_req} 条请求 (每条 1 token = decode), numpy/CPU 实测, 仅作趋势参考")
    xs_big = [rs.randn(1, d_in).astype(np.float32) for _ in range(n_req)]
    ids_big = rs.randint(0, n_adapter, size=n_req).tolist()
    timing = {}
    for name, fn in results.items():
        with Timer() as t:
            for _ in range(10):
                out = fn(xs_big, ids_big, W, adapters)
        timing[name] = t.ms / 10
        assert max_diff(out, merged_forward(xs_big, ids_big, W, adapters)) < 1e-5, \
            f"{name}: 500 条请求的大 batch 上也应等于合并权重的参考答案"
    kv("底模 gemm 次数 loop / batched", f"{n_req} / 1")
    for name, ms in timing.items():
        kv(name, f"{ms:.2f} ms")
    # 这条依赖墙钟计时, 机器很忙时可能偶发失败, 重跑即可
    assert timing["SGMV (按 adapter 分段)"] < timing["loop (逐请求)"], \
        "同 batch 的 SGMV 应比逐请求循环快 (底模 gemm 1 次 vs 500 次)"


if __name__ == "__main__":
    main()
