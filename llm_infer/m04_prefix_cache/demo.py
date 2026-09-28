"""
m04 demo — Prefix Cache: 命中 / 分叉 / 释放后仍可命中 / 被覆盖才失效

运行: python -m llm_infer.m04_prefix_cache.demo
看什么: [1] 冷启动全 miss; [2] 共享前缀命中; [3] 第 2 块分叉后全断;
        [4] 释放后仍可命中; [5] pool 被占满后索引同步失效。
这里只演示索引与 block 生命周期; "命中的 token 真的不再做前向" 在 full_engine demo 里对账。
"""
from __future__ import annotations
from llm_infer.core.utils import banner, kv
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m04_prefix_cache.prefix_cache import PrefixCache


def admit(bm: BlockManager, cache: PrefixCache, seq_id: int, ids):
    """一条请求进场的完整流程: 查前缀 → 命中的 share、其余新分配 → (假装算完 KV) → 登记。"""
    hits, n_hit = cache.match_prefix(ids)
    table = bm.allocate(seq_id, len(ids), shared=hits)
    cache.register(ids, table, num_computed=len(ids))
    return hits, n_hit, table


def main():
    banner("M04 - Prefix Cache (block 链式 hash)")
    bm = BlockManager(num_blocks=10, block_size=4)
    cache = PrefixCache(bm)

    SYSTEM = [10, 11, 12, 13, 20, 21, 22, 23]            # 8 token = 2 个完整 block
    USER_A = SYSTEM + [30, 31, 32, 33, 40, 41]
    USER_B = SYSTEM + [50, 51, 52, 53, 60, 61]
    USER_C = [10, 11, 12, 13, 99, 99, 99, 99, 5]          # 第 2 块就分叉

    print("\n[1] A 冷启动 → 全 miss")
    hits_a, n_a, table_a = admit(bm, cache, 1, USER_A)
    kv("A 命中 token / 页表", f"{n_a} / {table_a}")
    assert n_a == 0, f"索引为空时不应命中, 实际命中 {n_a} 个 token"

    print("\n[2] B 共享 SYSTEM → 命中前 2 块, 物理 block 与 A 相同")
    hits_b, n_b, table_b = admit(bm, cache, 2, USER_B)
    kv("B 命中 token / 页表", f"{n_b} / {table_b}")
    kv("共享 block 的 ref_count", [bm.ref_count[b] for b in hits_b])
    assert n_b == 8, f"B 应命中 SYSTEM 的 8 个 token (2 个完整 block), 实际 {n_b}"
    assert table_b[:2] == table_a[:2], "B 的前 2 页应是 A 的同一批物理 block, KV 只存一份"
    assert bm.ref_count[table_a[0]] == 2, "共享 block 被 A、B 各引用一次, ref_count 应为 2"

    print("\n[3] C 只有第 1 块相同 → 命中 4 token (链式 hash: 第 2 块内容不同, 后面全断)")
    hits_c, n_c, _ = admit(bm, cache, 3, USER_C)
    kv("C 命中 token", n_c)
    assert n_c == 4, f"C 只有第 1 块与 A 相同, 应命中 4 个 token, 实际 {n_c}"

    naive = sum(bm.blocks_needed(len(u)) for u in (USER_A, USER_B, USER_C))
    kv("占用 block: 无缓存 vs 实际", f"{naive} vs {bm.used_blocks()}")
    assert bm.used_blocks() == naive - 3, \
        f"B 省 2 块, C 省 1 块, 应占 {naive - 3} 块, 实际 {bm.used_blocks()}"

    print("\n[4] A、B、C 全部结束 → block 回到 free_list, 但内容和索引还在 → 新请求 D 照样命中")
    for sid in (1, 2, 3):
        bm.free(sid)
    assert bm.used_blocks() == 0, "三条序列都释放后不应还有 block 被占用"
    hits_d, n_d, table_d = admit(bm, cache, 4, USER_A)
    kv("D (= A 的 prompt) 命中 token", n_d)
    # A 的 prompt 14 个 token: 前 3 个完整 block (12 token) 可命中, 第 4 块没写满不进索引
    assert n_d == 12, f"block 释放后内容和索引还在, D 应命中 12 个 token, 实际 {n_d}"
    assert table_d[:3] == table_a[:3], "D 命中的应是 A 当初写的那 3 个物理 block"
    assert all(bm.ref_count[b] == 1 for b in table_d), "复活的 block 引用计数应从 0 回到 1"
    bm.free(4)

    print("\n[5] 来一条占满整个 pool 的请求 → 旧 block 被覆盖, 索引同步失效 (不会命中到脏数据)")
    big = list(range(1000, 1000 + 40))                    # 40 token = 10 block, 且 id ≥ 256
    _, n_big, _ = admit(bm, cache, 5, big)
    hits_e, n_e = cache.match_prefix(USER_A)
    kv("pool 被占满后 A 的 prompt 命中", n_e)
    kv("索引里的 block 数", cache.stats()["indexed_blocks"])
    assert n_big == 0, f"big 的 token 从没出现过, 不应命中, 实际 {n_big}"
    assert n_e == 0, f"A 的 block 已被 big 覆盖, 再查应全 miss, 实际命中 {n_e}"
    assert all(bm.ref_count[b] > 0 or b in bm.free_list for b in cache.block_to_hash), \
        "索引里的每个 block 要么在用, 要么躺在 free_list 里等复活"
    assert set(cache.hash_to_block.values()) == set(cache.block_to_hash), \
        "hash_to_block 与 block_to_hash 两张表必须指向同一批 block"
    print("  ✓ 缓存项寿命 = block 内容寿命: 释放 ≠ 失效, 覆盖才失效 (free_list 顺序即 LRU)")


if __name__ == "__main__":
    main()
