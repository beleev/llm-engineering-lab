"""
m05 demo — Radix Tree 前缀缓存: 手工走一遍 split, 再用随机负载对拍。

[1] 两条序列插入, 看边在分叉点被 split, slots 同步切开
[2] 随机 300 条请求: match 长度 == 暴力最长公共前缀; 命中的 slots 里存的确实是这些 token 的 KV
[3] 同一负载下 radix (任意长度) vs block-hash 粒度 (只命中整 block, m04 的做法) 的命中 token 数
[4] 上锁 + LRU 驱逐: 锁住的路径一个都不丢, 槽位守恒 (树内 token + 空闲槽 == pool 大小)

运行: python -m llm_infer.m05_radix_cache.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m05_radix_cache.radix_tree import RadixCache, _lcp

POOL_SIZE = 4096      # 模拟 KV pool 的槽位总数 (真实系统里每槽 = 1 个 token 的各层 K/V)
BLOCK = 4             # 对比用的 block-hash 粒度
N_REQ = 300           # [2][3] 随机负载的请求数
N_LOCK = 5            # [4] 锁住几条路径
EVICT_N = 100         # [4] 第一次按 LRU 要驱逐多少个 token


def walkthrough():
    """[1] 手工插两条序列, 逐个检查 split、match 长度和 ref_count 继承。"""
    rc = RadixCache()
    print("\n[1] insert A=[1..8] (slots 100..107), 再 insert B=[1,2,3,4,9,9] (slots 200..205)")
    rc.insert([1, 2, 3, 4, 5, 6, 7, 8], list(range(100, 108)))
    dup = rc.insert([1, 2, 3, 4, 9, 9], list(range(200, 206)))
    print(rc.pretty())
    kv("B 与树重复的前缀长度", f"{dup}  → slots 200..203 是重复 KV, 调用方应释放")
    assert dup == 4, f"B 的前 4 个 token 已在树里, insert 应返回 4, 实际 {dup}"
    assert rc.total_tokens() == 10, f"公共前缀只存一份: 树内应有 8 + 2 = 10 个 token, 实际 {rc.total_tokens()}"
    for q, want in (([1, 2, 3, 4, 5, 6], 6), ([1, 2, 3, 9], 3), ([1, 2], 2), ([42], 0)):
        _, slots = rc.match(q)
        print(f"  match {q!s:<22} → 命中 {len(slots)} 个, slots={slots}")
        assert len(slots) == want, f"match {q} 应命中 {want} 个 token, 实际 {len(slots)}"
    # split 继承 ref_count: 先锁住 B, 再用一个更短的 match 把 B 路径上的边切开, 解锁后必须全部归零
    node_b, _ = rc.match([1, 2, 3, 4, 9, 9])
    rc.lock_path(node_b)
    rc.match([1, 7])                                     # 把 [1,2] 切成 [1] + [2], 新的 mid 必须继承 ref=1
    # B 的路径此时被切成 5 段, 各段首槽位是 100 / 101 / 102 / 103 / 204
    assert all(x.ref_count == 1 for x in rc._nodes() if x.slots[0] in (100, 101, 102, 103, 204)), \
        "B 路径上的每段边 (含 split 新切出的 mid) ref_count 都应为 1"
    rc.unlock_path(node_b)
    assert all(x.ref_count == 0 for x in rc._nodes()), "解锁后全树 ref_count 应归零; split 不继承就会剩下负数"
    print("  锁住 B 后再 split 其路径, 解锁后 ref_count 全部归零  ✓")


def random_workload():
    """[2][3] 随机负载: 每条请求都拿暴力 LCP 对拍, 同时统计 radix 与 block-hash 两种粒度的命中数。"""
    rng = np.random.default_rng(0)
    rc = RadixCache()
    free = list(range(POOL_SIZE))
    slot_tok = {}                                        # slot → 它存的是哪个 token 的 KV (用来验证 slots 没错位)
    seen: list[list[int]] = []
    hit_radix = hit_block = total = 0

    for _ in range(N_REQ):
        # 模拟多轮对话: 取某条旧请求的随机前缀 + 新的随机后缀
        base = seen[rng.integers(len(seen))] if seen and rng.random() < 0.8 else []
        toks = base[:rng.integers(0, len(base) + 1)] + rng.integers(0, 50, size=rng.integers(1, 12)).tolist()

        brute = max((_lcp(s, toks) for s in seen), default=0)   # 暴力基线: 与所有历史序列逐条比
        node, slots = rc.match(toks)
        n = len(slots)
        assert n == brute, f"radix 命中长度 {n} 应等于暴力最长公共前缀 {brute}"
        assert [slot_tok[s] for s in slots] == toks[:n], "命中的槽位里存的应是这段前缀的 KV, slots 错位了"

        new = [free.pop() for _ in toks[n:]]             # 只给未命中的后缀分配槽位 (= 只 prefill 这部分)
        slot_tok.update(zip(new, toks[n:]))
        assert rc.insert(toks, slots + new) == n, "insert 报告的已有前缀长度应与刚才 match 的一致"
        assert rc.total_tokens() + len(free) == POOL_SIZE, "槽位守恒: 树内 token + 空闲槽 应等于 pool 大小"

        seen.append(toks)
        total += len(toks)
        hit_radix += n
        hit_block += n // BLOCK * BLOCK                  # block hash 只能命中整 block
    return rc, free, seen, slot_tok, (total, hit_radix, hit_block)


def main():
    banner("M05 - Radix Cache (SGLang RadixAttention)")
    walkthrough()

    rc, free, seen, slot_tok, (total, hit_radix, hit_block) = random_workload()
    print(f"\n[2] 随机 {N_REQ} 条请求 (seed=0): match 长度 == 暴力 LCP, slots 内容 == 前缀 token  "
          f"✓ ({len(seen)}/{N_REQ} 断言通过)")
    kv("树内 token / 空闲槽 / pool", f"{rc.total_tokens()} / {len(free)} / {POOL_SIZE}")

    print(f"\n[3] 同一负载的命中 token 数 (总 token {total})")
    kv("radix (任意长度)", f"{hit_radix}  ({hit_radix / total:.1%})")
    kv(f"block-hash 粒度 (block={BLOCK})", f"{hit_block}  ({hit_block / total:.1%})")
    assert hit_radix > hit_block, \
        f"radix 能命中任意长度, 应多于只命中整 block 的 block-hash: {hit_radix} vs {hit_block}"

    print(f"\n[4] 锁住 {N_LOCK} 条路径后驱逐")
    rng = np.random.default_rng(1)
    locked = [seen[i] for i in rng.choice(len(seen), size=N_LOCK, replace=False)]
    locked_nodes, locked_slots = [], set()
    for toks in locked:
        node, slots = rc.match(toks)
        rc.lock_path(node)
        locked_nodes.append(node)
        locked_slots |= set(slots)

    before = rc.total_tokens()
    freed = rc.evict(EVICT_N)                            # 先按 LRU 要 EVICT_N 个
    kv(f"evict({EVICT_N}) 实际释放", f"{len(freed)} (整叶子驱逐, 所以 ≥{EVICT_N})")
    freed += rc.evict(10**9)                             # 再把能驱逐的全驱逐
    free += freed
    kv("驱逐后 树内 token", f"{before} → {rc.total_tokens()}  (剩下的 = {N_LOCK} 条锁定路径的并集 {len(locked_slots)})")
    assert not locked_slots & set(freed), "锁住的槽位不能被驱逐"
    assert all(x.ref_count > 0 for x in rc._nodes()), "能驱逐的都驱逐后, 剩下的节点应全在锁定路径上"
    assert rc.total_tokens() == len(locked_slots), \
        f"树内剩余 token 应等于锁定路径的并集 {len(locked_slots)}, 实际 {rc.total_tokens()}"
    for toks in locked:
        assert len(rc.match(toks)[1]) == len(toks), "锁定的序列在驱逐后仍应整条命中"
    assert rc.total_tokens() + len(free) == POOL_SIZE, "槽位守恒: 树内 token + 空闲槽 应等于 pool 大小"
    assert len(set(free)) == len(free), "空闲槽里有重复: 同一个槽位被释放了两次"

    for node in locked_nodes:
        rc.unlock_path(node)
    free += rc.evict(10**9)
    kv("解锁后全部驱逐: 树内 token / 空闲槽", f"{rc.total_tokens()} / {len(free)}")
    assert rc.total_tokens() == 0, f"解锁后应能驱逐干净, 树内还剩 {rc.total_tokens()} 个 token"
    assert sorted(free) == list(range(POOL_SIZE)), "全部驱逐后, pool 的每个槽位应恰好回到空闲列表一次"


if __name__ == "__main__":
    main()
