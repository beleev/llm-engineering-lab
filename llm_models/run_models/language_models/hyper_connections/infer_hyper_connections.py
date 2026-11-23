#!/usr/bin/env python
"""
Hyper-Connections 结构验证 — 退化性、Sinkhorn 投影的性质、L 层连乘的增益、显存与 cache 开销。不训练, 几秒跑完。

看什么:
    1. 初始化时三种残差逐位相同: n 条流起步数值一致, H_pre 的权重加起来是 1、H_post 全 1、H_res 的行和是 1,
       所以 hc 和 mhc 的 logits 与普通 LLaMA 相等, 初始 loss 也相同。n = 1 时更是字面上的 Pre-LN 残差。
    2. Sinkhorn-Knopp 投影后行和恰为 1 (最后一步是行归一化)、列和近似为 1、元素非负; 谱范数不超过 1。
    3. 把 H_res 换成偏离恒等的矩阵, 看 L 层连乘后的最大绝对行和 (mHC 的 Amax Gain Magnitude):
       不加约束时随深度指数增长, 双随机约束下恒为 1。这一段是确定性模拟, 不是训练出来的矩阵。
    4. 加宽残差流只放大残差处的激活, 不改 KV cache: 逐 token 增量生成与整段前向一致。
"""

import math

import torch

from llm_models.layers.core.hyper_connections import composite_gain, sinkhorn_knopp
from llm_models.models.language_models.hyper_llama import HyperLLaMA
from llm_models.models.language_models.llama import LLaMA
from llm_models.utils.generation import KVCache

V, T, N = 11, 32, 4
CFG = dict(vocab_size=V, d_model=64, n_heads=4, num_layers=4, max_len=64)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(0)
    x = torch.randint(0, V, (16, T))
    llama = LLaMA(**CFG)

    # ---- 1. 初始化时与普通 Pre-LN 残差逐位相同 ----
    print("初始化时与普通 LLaMA 的 logits 最大差 (n 条流起步数值相同, H_res 的行和为 1, 所以原样传过去):")
    with torch.no_grad():
        base = llama(x)
    base_loss = torch.nn.functional.cross_entropy(base[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    for residual in ("hc", "mhc"):
        for n in (1, N):
            model = HyperLLaMA(residual=residual, n_streams=n, **CFG)
            model.load_state_dict(llama.state_dict(), strict=False)   # 共用 embedding / 各层 / ln_f
            with torch.no_grad():
                out = model(x)
            diff = (out - base).abs().max().item()
            loss = torch.nn.functional.cross_entropy(out[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
            print(f"  {residual:<4} n = {n} | 最大差 {diff:.1e} | 初始 loss {loss:.4f} (ln V = {math.log(V):.4f})"
                  f" | 参数 {n_params(model):,} (LLaMA {n_params(llama):,})")
            assert diff < 1e-6, f"{residual} n={n}: 初始化时应与普通 Pre-LN 残差逐位相同"
            assert abs(loss - base_loss) < 1e-4, f"{residual} n={n}: 初始 loss 应与普通 LLaMA 相同"
    assert abs(base_loss - math.log(V)) < 0.15, "普通 LLaMA 的初始 loss 应约等于 ln V"

    # ---- 2. Sinkhorn-Knopp 投影的性质 ----
    raw = torch.randn(6, N, N) * 1.5
    ds = sinkhorn_knopp(raw, n_iters=20)
    row_err = (ds.sum(-1) - 1).abs().max().item()
    col_err = (ds.sum(-2) - 1).abs().max().item()
    spec = max(torch.linalg.matrix_norm(m, ord=2).item() for m in ds)
    print(f"\nSinkhorn-Knopp 20 次迭代 (6 个随机 {N}×{N} 打分): 行和偏差 {row_err:.1e} | 列和偏差 {col_err:.1e}"
          f" | 最小元素 {ds.min().item():.1e} | 最大谱范数 {spec:.6f}")
    assert row_err < 1e-6, "最后一步是行归一化, 行和应恰为 1"
    assert col_err < 1e-3, "20 次迭代后列和应近似为 1"
    assert ds.min().item() >= 0, "双随机矩阵的元素必须非负"
    assert spec <= 1 + 1e-4, "双随机矩阵的谱范数不超过 1"

    # ---- 3. L 层 H_res 连乘的增益 ----
    torch.manual_seed(1)
    drift = torch.eye(N) + 0.3 * torch.randn(24, N, N)     # 模拟训练后偏离恒等的 H_res
    print("\nL 层 H_res 连乘后的 Amax 增益 (同一组偏离恒等的矩阵, 确定性模拟, 不是训练出来的):")
    gains = {}
    for depth in (1, 4, 8, 16, 24):
        hc = composite_gain(list(drift[:depth]))
        mhc = composite_gain([sinkhorn_knopp(m) for m in drift[:depth]])
        gains[depth] = (hc, mhc)
        print(f"  L = {depth:2d} | 不约束 (HC) 行和 {hc[0]:11.3f} 列和 {hc[1]:11.3f} 谱范数 {hc[2]:11.3f}"
              f" | 双随机 (mHC) 行和 {mhc[0]:.6f} 列和 {mhc[1]:.6f} 谱范数 {mhc[2]:.6f}")
    for depth, (hc, mhc) in gains.items():
        assert abs(mhc[0] - 1) < 1e-5, f"L = {depth}: 双随机矩阵连乘的行和应恒为 1"
        assert abs(mhc[1] - 1) < 1e-2, f"L = {depth}: 双随机矩阵连乘的列和应近似为 1"
        assert mhc[2] <= 1 + 1e-2, f"L = {depth}: 双随机矩阵连乘的谱范数应不超过 1"
    hc_rows = [gains[d][0][0] for d in (1, 4, 8, 16, 24)]
    assert all(a < b for a, b in zip(hc_rows, hc_rows[1:])), "不加约束时, 连乘的行和应随深度单调增长"
    assert hc_rows[-1] > 3 * hc_rows[0], "不加约束时, 24 层连乘的行和应比单层明显更大"

    # ---- 4. 增量生成与整段前向一致 ----
    print("\n逐 token 增量生成 vs 整段前向 (加宽残差流不改 KV cache 的形状):")
    for residual in ("hc", "mhc"):
        model = HyperLLaMA(residual=residual, n_streams=N, **CFG).eval()
        with torch.no_grad():
            full = model(x)
            cache = KVCache(len(model.layers))
            step = torch.cat([model(x[:, t: t + 1], cache=cache) for t in range(T)], dim=1)
        diff = (full - step).abs().max().item()
        entries = sum(c["k"].numel() + c["v"].numel() for c in cache.layers)
        print(f"  {residual:<4} | 最大差 {diff:.1e} | KV cache 元素数 {entries:,}"
              f" | 残差处激活元素数 ×{N} (每 token {N * model.d_model} 个)")
        assert diff < 1e-4, f"{residual}: 增量生成应与整段前向一致"
    print("Hyper-Connections 结构验证通过")


if __name__ == "__main__":
    main()
