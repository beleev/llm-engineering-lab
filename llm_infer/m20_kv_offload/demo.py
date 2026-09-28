"""
m20 demo — 分层 KV 卸载 (GPU → CPU → 磁盘), 所有时间来自代价模型而非实测

    [1] 多轮对话负载 (8 用户轮流发言, GPU 层装不下 → 抖动): GPU-only / +CPU / +CPU+disk 的逐层命中率与 TTFT
    [2] 交叉点: 命中前缀多短时, 从磁盘加载不如重算
    [3] 慢链路: "命中就加载" 反而拖慢 TTFT; 带 "加载 vs 重算" 判断的策略永远不差于全量重算

运行: python -m llm_infer.m20_kv_offload.demo
"""
from __future__ import annotations

from llm_infer.core.utils import banner, kv
from llm_infer.m20_kv_offload.tiered_cache import Tier, CostModel, TieredKVCache, chat_workload

COST = CostModel(block_size=16, kv_bytes_per_token=131072, prefill_tok_per_s=8000.0)
GPU = Tier("GPU", capacity_blocks=64, bandwidth_GBps=float("inf"), latency_ms=0.0)
CPU = Tier("CPU", capacity_blocks=192, bandwidth_GBps=25.0, latency_ms=0.2)      # PCIe4 x16
DISK = Tier("disk", capacity_blocks=4096, bandwidth_GBps=3.0, latency_ms=2.0)    # NVMe
SLOW = Tier("remote", capacity_blocks=4096, bandwidth_GBps=0.5, latency_ms=20.0)  # 跨机房 / 机械盘
N_USERS, N_TURNS = 8, 6                  # [1] 的负载: 几个用户轮流发言, 每人几轮


def run(tiers, reqs, **kw):
    """用给定的几层存储把整个负载回放一遍 → (累计 Stats, 每个请求的模拟 TTFT)。"""
    cache = TieredKVCache(tiers, COST, **kw)
    ttfts = []
    for _, prompt, with_reply in reqs:
        ttfts.append(cache.serve(prompt))
        cache.store(with_reply)                  # decode 产生的回复 KV 也登记进 cache
    s = cache.stats
    assert sum(s.hit) + s.miss == s.total, "token 守恒: 每个 prompt token 要么命中某一层, 要么 miss"
    assert all(len(t) <= cfg.capacity_blocks for t, cfg in zip(cache.tiers, tiers)), \
        "每一层存的 block 数都不能超过它的容量"
    return s, ttfts


def main():
    banner("M20 - 分层 KV 卸载 (GPU→CPU→disk)")
    print("  以下时间全部来自代价模型 (cost model), 不是实测。模型参数:")
    kv("block_size / 每 token KV 字节", f"{COST.block_size} / {COST.kv_bytes_per_token:,} (LLaMA-3-8B fp16)")
    kv("重算 (prefill) 速度", f"{COST.prefill_tok_per_s:.0f} tok/s → {COST.recompute_ms(COST.block_size):.2f} ms/block")
    for t in (GPU, CPU, DISK, SLOW):
        kv(f"存储层 {t.name}", f"容量 {t.capacity_blocks} blocks, 带宽 {t.bandwidth_GBps} GB/s, "
                              f"延迟 {t.latency_ms} ms → {COST.load_ms(t, 1) - t.latency_ms:.3f} ms/block")

    reqs = chat_workload(n_users=N_USERS, n_turns=N_TURNS)
    total_blocks = sum(len(r[2]) for r in reqs[-N_USERS:]) // COST.block_size   # 最后一轮每人一条, 即各自的完整历史
    print(f"\n[1] 多轮对话: {N_USERS} 个用户 × {N_TURNS} 轮, 最终历史共 ≈{total_blocks} blocks "
          f"(GPU 只放得下 {GPU.capacity_blocks})")
    # 中文字符占两格, 表头的格式宽度按显示宽度少补几格
    print(f"  {'配置':<14}{'GPU命中':>7}{'CPU命中':>7}{'disk命中':>7}{'miss':>8}{'平均 TTFT(模拟)':>16}")
    results = {}
    for name, tiers in (("仅 GPU", [GPU]), ("GPU+CPU", [GPU, CPU]), ("GPU+CPU+disk", [GPU, CPU, DISK])):
        s, _ = run(tiers, reqs)
        rates = [h / s.total for h in s.hit] + [0.0] * (3 - len(s.hit))
        results[name] = s.ttft_ms / s.n_req
        w = 16 - sum(ord(c) > 0x2E80 for c in name)
        print(f"  {name:<{w}}{rates[0]:>9.1%}{rates[1]:>9.1%}{rates[2]:>9.1%}{s.miss / s.total:>8.1%}{results[name]:>15.2f} ms")
    no_cache = sum(COST.recompute_ms(len(r[1])) for r in reqs) / len(reqs)
    kv("无 cache (全量重算)", f"{no_cache:.2f} ms")
    assert results["GPU+CPU+disk"] < results["GPU+CPU"] < results["仅 GPU"] < no_cache, \
        "每多一层存储, 平均 TTFT 都应更低; 任何一种都应低于全量重算"

    print("\n[2] 交叉点: 从 disk 加载 n 个命中 block vs 重算它们")
    per_blk_load = COST.load_ms(DISK, 1) - DISK.latency_ms        # 去掉固定延迟后, 每个 block 的加载时间
    # 令 latency + n·load = n·recompute, 解出交叉点 n*: 命中块数超过它, 加载才划算
    n_star = DISK.latency_ms / (COST.recompute_ms(COST.block_size) - per_blk_load)
    for n in (1, 2, 4, 8, 32):
        l, r = COST.load_ms(DISK, n), COST.recompute_ms(n * COST.block_size)
        print(f"  n={n:<3} 加载={l:6.2f} ms  重算={r:6.2f} ms  → {'加载' if l <= r else '重算'}")
        assert (l <= r) == (n >= n_star), f"n={n}: 加载更快 当且仅当 命中块数 ≥ 交叉点 n*={n_star:.2f}"
    kv("n* = 固定延迟 / (每 block 重算 − 每 block 加载)", f"{n_star:.2f} blocks")

    print(f"\n[3] 慢链路 ({SLOW.name} {SLOW.bandwidth_GBps:g} GB/s: 每 block 加载比重算还慢)")
    base, _ = run([GPU, CPU], reqs)
    naive, _ = run([GPU, CPU, SLOW], reqs, always_load=True)
    smart, smart_ttfts = run([GPU, CPU, SLOW], reqs)
    kv("GPU+CPU (无慢层)", f"{base.ttft_ms / base.n_req:.2f} ms")
    kv(f"+{SLOW.name}, 命中就加载", f"{naive.ttft_ms / naive.n_req:.2f} ms   ← 加了一层缓存反而更慢")
    kv(f"+{SLOW.name}, 加载/重算取小", f"{smart.ttft_ms / smart.n_req:.2f} ms  (命中但选择重算 {smart.recomputed_hit} tokens)")
    # +1e-9: 浮点累加的舍入余量
    assert naive.ttft_ms > base.ttft_ms, "慢链路上命中就加载, 总 TTFT 应比不加这一层还高"
    assert smart.ttft_ms <= base.ttft_ms + 1e-9, "加载和重算取快的那个, 总 TTFT 不应高于不加这一层"
    assert smart.recomputed_hit > 0, "应有命中慢层但选择重算的 token, 否则这一节没测到判断逻辑"
    for (_, prompt, _), t in zip(reqs, smart_ttfts):
        assert t <= COST.recompute_ms(len(prompt)) + 1e-9, "逐请求: 带判断的策略不应比全量重算慢"

    print("\n  全部断言通过")


if __name__ == "__main__":
    main()
