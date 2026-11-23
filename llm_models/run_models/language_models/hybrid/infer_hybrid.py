#!/usr/bin/env python
"""
混合架构零件验证 — 退化关系、缓存算式、分块掩码、增量生成。不训练, 几秒跑完。

看什么:
    1. KDA 的逐通道遗忘门全取同一个值时, 与 Gated DeltaNet 逐位相同: KDA 是 GDN 的严格推广,
       两者的区别只在门的粒度。
    2. GatedAttention 关掉输出门时与 GroupedQueryAttention 逐位相同, 所以 "有门 / 无门" 的对照
       是同一条代码路径。
    3. 四种排布在长度 T 下的缓存元素数与算式逐个相符: 全注意力层随 T 线性长, 线性层是定长状态。
    4. 分块注意力层每个 query 只看自己那一块, NoPE 全注意力层看全长。
    5. 每种排布的逐 token 增量生成都与整段前向一致。
"""

import torch

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.gated_attention import GatedAttention
from llm_models.layers.sparse.linear_attention import GatedDeltaNet, KimiDeltaAttention
from llm_models.models.language_models.hybrid import HybridLM
from llm_models.utils.generation import KVCache

V, T, D, H, CHUNK = 64, 32, 64, 4, 8
CFG = dict(vocab_size=V, d_model=D, n_heads=H, max_len=128, chunk_size=CHUNK,
           latent_dim=32, qk_rope_head_dim=16, dropout=0.0)
LAYOUTS = {
    "Kimi Linear / K3 (3:1)": "kda kda kda mla_nope",
    "Qwen3-Next / Qwen3.5 (3:1)": "gdn gdn gdn gated_gqa",
    "Llama 4 iRoPE (3:1)": "gqa_chunk gqa_chunk gqa_chunk gqa_nope",
    "全注意力基线": "gqa gqa gqa gqa",
}


def main():
    torch.manual_seed(0)
    x = torch.randint(0, V, (2, T))

    # ---- 1. KDA 的通道门全相同时等于 GDN ----
    gdn, kda = GatedDeltaNet(D, H), KimiDeltaAttention(D, H)
    kda.load_state_dict({n: p for n, p in gdn.state_dict().items() if "gate_alpha" not in n}, strict=False)
    with torch.no_grad():     # 把 GDN 的每头标量门复制到 KDA 的每个通道
        kda.gate_alpha.weight.copy_(gdn.gate_alpha.weight.repeat_interleave(kda.d_head, dim=0))
        kda.gate_alpha.bias.copy_(gdn.gate_alpha.bias.repeat_interleave(kda.d_head))
    h = torch.randn(2, T, D)
    with torch.no_grad():
        diff = (gdn(h) - kda(h)).abs().max().item()
        channels = torch.sigmoid(kda.gate_alpha(h)).view(2, T, H, kda.d_head)
        spread = (channels.amax(-1) - channels.amin(-1)).max().item()
    print(f"KDA 通道门全相同时 vs GatedDeltaNet: 最大差 {diff:.1e} | 同一头内各通道门的最大跨度 {spread:.1e}")
    assert diff < 1e-5, "α 各通道取同一个值时, KDA 应与 GatedDeltaNet 逐位相同"
    assert spread < 1e-6, "这一组权重下同一头内各通道的门应完全相同"
    kda2 = KimiDeltaAttention(D, H)                      # 默认初始化: 各通道的门互不相同
    with torch.no_grad():
        ch = torch.sigmoid(kda2.gate_alpha(h)).view(2, T, H, kda2.d_head)
    print(f"默认初始化的 KDA: 同一头内各通道门的最大跨度 {(ch.amax(-1) - ch.amin(-1)).max().item():.3f}")
    assert (ch.amax(-1) - ch.amin(-1)).max().item() > 0.01, "默认初始化下各通道的门应各不相同"

    # ---- 2. 关掉输出门就是 GQA ----
    gqa = GroupedQueryAttention(D, H, num_kv_heads=2)
    plain = GatedAttention(D, H, use_gate=False, num_kv_heads=2)
    plain.load_state_dict(gqa.state_dict(), strict=False)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    with torch.no_grad():
        diff = (gqa(h, mask=causal) - plain(h, mask=causal)).abs().max().item()
    print(f"\nGatedAttention(use_gate=False) vs GroupedQueryAttention: 最大差 {diff:.1e}"
          f" | 第 0 列分走的注意力质量 {plain.last_sink_mass:.4f} (初始化时接近均匀)")
    assert diff < 1e-6, "关掉输出门时应与 GroupedQueryAttention 逐位相同"

    # ---- 3. 缓存元素数与算式 ----
    print("\n一条序列的缓存元素数 (实测 vs 算式):")
    for name, layout in LAYOUTS.items():
        model = HybridLM(layout=layout, **CFG).eval()
        for seq in (32, 64):
            cache = KVCache(len(model.layers))
            with torch.no_grad():
                model(x[:1, :1], cache=cache)                 # 先建好各层的 cache 键
                for t in range(1, seq):
                    model(x[:1, t % T: t % T + 1], cache=cache)
            got = sum(sum(v.numel() for v in c.values()) for c in cache.layers)
            want = model.cache_elements(seq)
            print(f"  {name:<28} T = {seq:3d} | 实测 {got:6d} | 算式 {want['total']:6d}"
                  f" (随 T 增长 {want['grows_with_t']}, 定长状态 {want['fixed']})")
            assert got == want["total"], f"{layout} 在 T = {seq} 时的缓存元素数应与算式相符"

    # ---- 4. 分块掩码 ----
    model = HybridLM(layout=LAYOUTS["Llama 4 iRoPE (3:1)"], **CFG).eval()
    pos = torch.arange(T)
    chunk_seen = ((pos[:, None] // CHUNK) == (pos[None, :] // CHUNK)) & torch.ones(T, T, dtype=torch.bool).tril()
    full_seen = torch.ones(T, T, dtype=torch.bool).tril()
    print(f"\niRoPE 排布 {model.layout}: 分块大小 {CHUNK}")
    print(f"  分块 RoPE 层: 位置 {T - 1} 能看到 {int(chunk_seen[-1].sum())} 个 key (块内), 全长是 {T}")
    print(f"  NoPE 全注意力层: 位置 {T - 1} 能看到 {int(full_seen[-1].sum())} 个 key")
    assert int(chunk_seen[-1].sum()) <= CHUNK, "分块层每个 query 看到的 key 数不应超过块大小"
    assert not chunk_seen[CHUNK, CHUNK - 1], "跨块的 key 对分块层应不可见"
    assert int(full_seen[-1].sum()) == T, "NoPE 全注意力层应看到全部历史"

    # ---- 5. 增量生成与整段前向一致 ----
    print("\n逐 token 增量生成 vs 整段前向:")
    for name, layout in LAYOUTS.items():
        torch.manual_seed(0)
        model = HybridLM(layout=layout, **CFG).eval()
        with torch.no_grad():
            full = model(x)
            cache = KVCache(len(model.layers))
            step = torch.cat([model(x[:, t:t + 1], cache=cache) for t in range(T)], dim=1)
        diff = (full - step).abs().max().item()
        print(f"  {name:<28} {layout:<40} 最大差 {diff:.1e}")
        assert diff < 1e-4, f"{layout}: 增量生成应与整段前向一致"
    print("混合架构零件验证通过")


if __name__ == "__main__":
    main()
