#!/usr/bin/env python
"""
Mamba 推理: 验证 "递推解码 == 整段前向", 以及长生成数值稳定

盯住: cache 里的状态形状不随已读长度变化 (Transformer 的 KV cache 会变长)。
"""

import math
import time

import torch

from llm_models.layers.sparse.ssm import SelectiveSSM
from llm_models.models.language_models.mamba import Mamba
from llm_models.utils.generation import KVCache


def main():
    torch.manual_seed(42)
    V, L = 1000, 4                                                     # 词表大小, 层数
    model = Mamba(vocab_size=V, d_model=128, num_layers=L, d_state=16, d_conv=4).eval()
    print(f"Mamba Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    idx = torch.randint(0, V, (2, 64))

    with torch.inference_mode():
        # 1) 整段前向 vs prefill 8 个 token + 逐 token 递推: logits 必须一致
        full = model(idx)                                              # [2, 64, V]
        cache = KVCache(L)                                             # 复用 KVCache 这个容器, 里面装的是 SSM 状态
        outs = [model(idx[:, :8], cache=cache)]
        outs += [model(idx[:, t:t + 1], cache=cache) for t in range(8, 64)]
        diff = (full - torch.cat(outs, dim=1)).abs().max().item()
        state = {k: tuple(v.shape) for k, v in cache.layers[0].items()}
    print(f"递推 vs 整段 logits 最大差: {diff:.2e} | 读完 {cache.pos} 个 token 后每层状态: {state}")
    assert diff < 1e-4, "递推解码 (prefill + 逐 token) 的 logits 应与整段前向一致"
    # conv: 卷积要用的最近 d_conv-1 = 3 步输入; h: SSM 隐状态, 最后一维是 d_state = 16
    assert state == {"conv": (2, 256, 3), "h": (2, 256, 16)}, "每层状态的形状应固定, 与已读长度 T=64 无关"

    # 2) 因果性: 改最后一个 token, 前面位置的 logits 不变 (没有 mask, 靠递推方向天然因果)
    idx2 = idx.clone(); idx2[:, -1] = (idx2[:, -1] + 1) % V
    with torch.inference_mode():
        assert torch.allclose(model(idx2)[:, :-1], full[:, :-1], atol=1e-6), \
            "Mamba 应天然因果: 改最后一个 token 却影响了前面位置的 logits"

    # 3) 长生成: 每步 logits 有限。不需要 nan_to_num 兜底, 稳定性由 A<0, Δ>0 保证
    #    在 lm_head 上挂 hook: 每次前向记一笔 "这次输出是否全部有限"
    seen = []
    hook = model.lm_head.register_forward_hook(lambda m, i, o: seen.append(torch.isfinite(o).all().item()))
    n_gen = 300
    t0 = time.time(); g_cache = model.generate(idx[:, :5], n_gen, temperature=0); t_cache = time.time() - t0
    t0 = time.time(); g_naive = model.generate(idx[:, :5], n_gen, temperature=0, use_cache=False); t_naive = time.time() - t0
    hook.remove()
    assert all(seen), "生成过程中出现非有限 logits"
    assert len(seen) == 2 * n_gen, "两次生成各 n_gen 个 token, lm_head 应被调用 2·n_gen 次"
    assert torch.equal(g_cache, g_naive), "递推生成与每步重算的贪心输出应逐 token 相同"
    model.generate(idx[:, :5], 50, temperature=0.8, top_k=20)          # 采样路径也不应报错
    print(f"生成 {n_gen} token: 递推 {t_cache:.2f}s vs 每步重算 {t_naive:.2f}s ({t_naive / t_cache:.0f}×), "
          f"{len(seen)} 步 logits 全部有限")

    # 左 pad: pad 位置的卷积输入和 SSM 输入置 0, 真 token 的输出与不 pad 一致 (不打印)
    P = 3
    padded = torch.cat([torch.zeros(2, P, dtype=torch.long), idx[:, :8]], dim=1)  # [B, P+8], pad id = 0
    mask = torch.cat([torch.zeros(2, P), torch.ones(2, 8)], dim=1)                # [B, P+8], 0 = pad
    with torch.inference_mode():
        d_pad = (model(padded, attention_mask=mask)[:, P:] - full[:, :8]).abs().max().item()
    assert d_pad < 1e-4, f"左 pad 不应改变真实位置的 logits: {d_pad}"
    # 批量生成: 一条左 pad、一条不 pad, 走 cache 递推, 每条都要和单独生成逐 token 相同
    batch = torch.stack([torch.cat([torch.zeros(P, dtype=torch.long), idx[0, :5]]), idx[1, :P + 5]])
    bmask = torch.ones(2, P + 5)
    bmask[0, :P] = 0                                                   # 只有第 0 条左边 3 个是 pad
    g_batch = model.generate(batch, 20, temperature=0, attention_mask=bmask)
    g_solo = model.generate(idx[:1, :5], 20, temperature=0)
    assert torch.equal(g_batch[0, P:], g_solo[0]), "左 pad 的批量生成应与单独生成逐 token 相同"

    # 4) 反例: 同一个递推 h = exp(Δ·a)·h + Δ·x, 只把 a 的符号翻过来
    def first_overflow(a: float, delta: float = 0.1, T: int = 400) -> int:
        """标量递推 h ← exp(Δ·a)·h + Δ·x (x 恒为 1); 返回第一次溢出的步数, 不溢出返回 -1。"""
        h = torch.zeros(())
        for t in range(T):
            h = math.exp(delta * a) * h + delta * 1.0
            if not torch.isfinite(h):
                return t
        return -1

    a, steps = 16.0, 400
    t_neg, t_pos = first_overflow(-a, T=steps), first_overflow(+a, T=steps)
    print(f"稳定性: a=-{a:.0f} ⇒ {steps} 步不溢出; a=+{a:.0f} ⇒ 第 {t_pos} 步溢出为 inf "
          f"(A=-exp(A_log) 就是为了排除这种情况)")
    # a < 0: 每步乘 exp(Δ·a) < 1, h 收敛; a > 0: 每步乘 exp(1.6) ≈ 5, 指数爆炸
    assert t_neg == -1, "a < 0 时递推 400 步都不应溢出"
    assert 0 < t_pos < 100, "a > 0 时递推应在 100 步内溢出为 inf"
    ssm = SelectiveSSM(64, 16)
    with torch.no_grad():
        ssm.A_log.normal_(0, 5)                                        # 任意折腾 A_log, A 依然恒负
        assert torch.isfinite(ssm(torch.randn(1, 400, 64))).all(), \
            "A = -exp(A_log) 恒负: 无论 A_log 取什么值, 400 步递推的输出都应有限"


if __name__ == "__main__":
    main()
