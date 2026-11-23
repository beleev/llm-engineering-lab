#!/usr/bin/env python
"""
条件深度结构验证 — MoD 的计算量、等价性、因果性, MoR 的参数量和 KV 条数。不训练, 几秒跑完。

看什么:
    1. MoD 容量 100% 且路由权重为 1 时, 前向与普通 LLaMA 相同; 容量不足 100% 时, 只算入选的 k 行,
       结果与 "整段都算、掩码只让入选 token 互相看见、落选的输出丢掉" 相同 (gather / scatter 没有搬错位置)。
    2. 每 token 稠密 FLOPs 随容量 C 线性下降; 注意力那部分按 C² 下降。
    3. 整段序列取 top-k 不满足因果: 只改后 16 个 token, 前 48 个 token 的路由也会变; 换成预测器后不再变。
    4. MoR 的参数量几乎不随递归次数变, 同样展开深度的 LLaMA 每多一层多一个 block 的参数;
       expert-choice 的容量日程下, 缓存的 K/V 条数是每次递归都全过的 (N_r + 1) / (2N_r)。
"""

import torch

from llm_models.models.language_models.conditional_depth import MoDLLaMA, MoRLLaMA, routed_block
from llm_models.models.language_models.llama import LLaMA

V, T = 11, 64
CFG = dict(vocab_size=V, d_model=64, n_heads=4, max_len=T)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(0)
    x = torch.randint(0, V, (8, T))

    # ---- 1. 容量 100% 时等于 LLaMA ----
    llama = LLaMA(num_layers=4, **CFG)
    mod = MoDLLaMA(capacity=1.0, num_layers=4, **CFG)
    mod.load_state_dict(llama.state_dict(), strict=False)            # 共用 embedding / 各层 / ln_f
    for r in mod.routers.values():
        torch.nn.init.zeros_(r.weight)                               # 分数 0 → 权重 2·sigmoid(0) = 1
    with torch.no_grad():
        diff = (mod(x) - llama(x)).abs().max().item()
    print(f"MoD 路由层 {mod.routed}, 容量 100%, 路由权重 1: 与 LLaMA 的 logits 最大差 {diff:.1e}")
    assert diff < 1e-5, "容量 100% 且权重为 1 时应与 LLaMA 前向一致"

    h = torch.randn(8, T, CFG["d_model"])
    sel = torch.rand(8, T, generator=torch.Generator().manual_seed(2)) < 0.3     # 每行入选个数不同, 要补位
    w = torch.rand(8, T) + 0.5
    block = llama.layers[1]
    with torch.no_grad():
        fast = routed_block(block, h, sel, w, llama.rope)
        see = torch.ones(T, T, dtype=torch.bool).tril() & (sel[:, None, :] | torch.eye(T, dtype=torch.bool))
        full = block(h, mask=see, rope=llama.rope, position_ids=torch.arange(T))  # 整段都算
        ref = torch.where(sel[..., None], h + w[..., None] * (full - h), h)
    gap = (fast - ref).abs().max().item()
    print(f"只算入选的 {int(sel.sum(1).max())} 行 vs 整段算再屏蔽: 最大差 {gap:.1e}")
    assert gap < 1e-5, "gather 后只算入选 token, 应与整段计算再屏蔽落选 token 一致"

    # ---- 2. FLOPs 随容量线性下降 ----
    print(f"\n每 token FLOPs (4 层, 第 1、3 层路由, T = {T}; 按实际过层的 token 数计):")
    caps = [1.0, 0.75, 0.5, 0.25, 0.125]
    rows = []
    for c in caps:
        mod.capacity = c
        with torch.no_grad():
            mod(x)
        f = mod.flops_per_token()
        rows.append(f)
        print(f"  容量 {c:5.3f} | 稠密 {f['dense'] / 1e3:6.1f}K | 注意力 {f['attn'] / 1e3:5.1f}K | 合计 {f['total'] / 1e3:6.1f}K"
              f" | 相对 100% {f['total'] / rows[0]['total']:.3f}")
    slope = (rows[0]["dense"] - rows[2]["dense"]) / (caps[0] - caps[2])
    for c, f in zip(caps, rows):
        assert abs(f["dense"] - (rows[0]["dense"] - slope * (1 - c))) < 1e-6 * rows[0]["dense"], "稠密 FLOPs 应随容量线性下降"
    assert all(a["total"] > b["total"] for a, b in zip(rows, rows[1:])), "总 FLOPs 应随容量单调下降"

    # ---- 3. top-k 路由不满足因果, 预测器满足 ----
    mod = MoDLLaMA(capacity=0.125, num_layers=4, **CFG)
    future = x.clone()
    future[:, 48:] = torch.randint(0, V, (8, T - 48), generator=torch.Generator().manual_seed(1))  # 只改后 16 个
    print("\n只改后 16 个 token, 前 48 个 token 的路由决定变了多少 (8 条序列, 第 1、3 层合计):")
    changed = {}
    for routing in ("topk", "predictor"):
        with torch.no_grad():
            mod(x, routing=routing)
            before = {i: s[:, :48].clone() for i, s in mod.last_sel.items()}
            mod(future, routing=routing)
        changed[routing] = sum(int((before[i] != s[:, :48]).sum()) for i, s in mod.last_sel.items())
        print(f"  {routing:<9} | 前 48 个位置里路由翻转 {changed[routing]} 次")
    assert changed["topk"] > 0, "整段 top-k: 后面的 token 应能改变前面 token 的路由"
    assert changed["predictor"] == 0, "预测器路由只看前文, 改后面的 token 不应改变前面的路由"

    # ---- 4. MoR: 参数量与展开深度, KV 条数 ----
    print("\nMoR (首尾各 1 层独立, 中间 1 层共享递归 N_r 次) vs 同样展开深度的 LLaMA:")
    base = None
    for N in (1, 2, 3, 4):
        mor = MoRLLaMA(num_recursions=N, **CFG)
        plain = LLaMA(num_layers=2 + N, **CFG)
        with torch.no_grad():
            mor(x[:, :48])
        ratio = mor.kv_entries.float().mean().item() / (N * 48)
        print(f"  N_r = {N} | 展开深度 {2 + N} | MoR 参数 {n_params(mor):,} | LLaMA 参数 {n_params(plain):,}"
              f" | 共享块 K/V 条数 / 全过 {ratio:.3f} (公式 {(N + 1) / (2 * N):.3f})")
        assert abs(ratio - (N + 1) / (2 * N)) < 1e-9, "expert-choice 容量日程下 KV 条数应为 (N_r+1)/(2N_r)"
        if base is None:
            base = (n_params(mor), n_params(plain))
        else:
            assert n_params(mor) - base[0] == (N - 1) * CFG["d_model"], "MoR 每多递归一次只多一个路由器 (D 个参数)"
            assert n_params(plain) - base[1] == (N - 1) * sum(p.numel() for p in plain.layers[0].parameters()), "LLaMA 每多一层多一个 block"
    print(f"  一个 token 的递归次数 (N_r = 4, 第 0 条序列前 16 个): {mor.depth[0, :16].tolist()}")
    print("条件深度结构验证通过")


if __name__ == "__main__":
    main()
