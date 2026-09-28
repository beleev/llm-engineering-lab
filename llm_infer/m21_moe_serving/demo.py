"""
m21 demo — MoE 专家并行 + EPLB 冗余专家

    [1] EP dispatch/combine 的输出 == 逐 token 循环的朴素基线
    [2] Zipf 倾斜路由 → 各 rank 负载不均 (max/mean)
    [3] EPLB: 用历史 batch 的专家负载统计做副本+放置, 在新 batch 上比较 max/mean; 输出不变

运行: python -m llm_infer.m21_moe_serving.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m21_moe_serving.moe import (
    make_layer, dense_reference, ep_forward, contiguous_placement, eplb_placement,
)

E, R, K, T, D = 16, 4, 2, 2048, 32     # 专家数, rank 数, top-k, token 数, 隐藏维
N_REDUNDANT = 4            # 每 rank 多 1 个 slot


def imbalance(load):
    """最满的 rank 是平均负载的多少倍。1.0 = 完全均衡。"""
    return load.max() / load.mean()


def main():
    banner("M21 - MoE 服务: 专家并行 (EP) + EPLB")
    layer = make_layer(D=D, E=E, top_k=K, zipf_s=1.0)
    rs = np.random.RandomState(1)
    x_hist = rs.randn(T, D).astype(np.float32)       # 历史 batch: 只用来统计专家负载
    x = rs.randn(T, D).astype(np.float32)            # 新 batch: 真正评测
    kv("专家 / rank / top-k / token 数", f"{E} / {R} / {K} / {T}")

    print("\n[1] EP dispatch/combine vs 朴素逐 token 循环")
    ref = dense_reference(layer, x)
    base = contiguous_placement(E, R)
    out_base, load_base = ep_forward(layer, x, base, R)
    kv("max|Δ|", f"{np.abs(out_base - ref).max():.2e}")
    assert np.abs(out_base - ref).max() < 1e-5, "EP dispatch/combine 的输出应等于逐 token 循环的基线"
    assert load_base.sum() == T * K, "每个 (token, k) 分配应恰好被一个 rank 处理, 各 rank 负载之和 = T·k"

    print(f"\n[2] Zipf 倾斜路由下的负载 (连续放置: 专家 0-{E // R - 1} 在 rank0, ...)")
    expert_load = np.bincount(layer.route(x)[0].reshape(-1), minlength=E)   # (E,) 每个专家被选中的次数
    kv("每专家 token 数", expert_load.tolist())
    kv("每 rank token 数", load_base.tolist())
    kv("max/mean", f"{imbalance(load_base):.2f}x")
    print("  为什么看 max/mean: EP 每层 combine 都要等所有 rank, 一步耗时 = 最慢 (最满) 的 rank;")
    print(f"  max/mean={imbalance(load_base):.2f} ⇒ 平均 rank 利用率只有 {1 / imbalance(load_base):.0%}, 其余时间在等最慢的 rank。")

    print(f"\n[3] EPLB (负载统计来自历史 batch, 在新 batch 上评测; 冗余 slot = {N_REDUNDANT})")
    hist_load = np.bincount(layer.route(x_hist)[0].reshape(-1), minlength=E)
    results = {"连续放置": imbalance(load_base)}
    for name, n_red in (("greedy 放置, 无副本", 0), (f"EPLB +{N_REDUNDANT} 副本", N_REDUNDANT)):
        pl = eplb_placement(hist_load, R, n_red)
        out, load = ep_forward(layer, x, pl, R)
        results[name] = imbalance(load)
        n_rep = np.bincount(pl.slot_expert, minlength=E)
        print(f"  {name:<20} rank 负载={load.tolist()}  max/mean={imbalance(load):.2f}x  "
              f"副本数>1 的专家={ {int(e): int(n) for e, n in enumerate(n_rep) if n > 1} }")
        assert np.abs(out - ref).max() < 1e-5, f"{name}: 副本权重相同, 输出应与放置和复制方式无关"
        assert load.sum() == T * K, f"{name}: 各 rank 负载之和应为 T·k, 没有分配被漏掉或重复处理"
    vals = list(results.values())
    kv("max/mean: 连续放置 → greedy → EPLB", " → ".join(f"{v:.2f}x" for v in vals))
    kv("模拟 step 时间 (∝ 最满 rank 的负载) 缩短", f"{1 - vals[2] / vals[0]:.0%}")
    assert vals[2] < vals[1] < vals[0], f"max/mean 应为 EPLB < greedy 放置 < 连续放置, 实际 {vals}"
    assert vals[2] < 1.1, f"EPLB 应把 max/mean 压到 1.1 以内, 实际 {vals[2]:.2f}"
    hot = expert_load.max() / (T * K / R)
    kv("最热专家 / 平均 rank 负载", f"{hot:.2f}  (>1 ⇒ 不复制就不可能均衡)" if hot > 1 else f"{hot:.2f}")

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
