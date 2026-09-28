"""
m02 demo — BlockManager 记账 + 分页 KV 上的 attention 与连续 KV 逐元素一致

运行: python -m llm_infer.m02_paged_attention.demo
看什么: block_table 如何增长/归还; pool 用满时的 MemoryError (m03 抢占的触发信号);
        [5][6] decode 与多 token prefill 两种形状下 paged == dense。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, dense_attention
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m02_paged_attention.paged_attention import (
    gather_kv, write_kv, paged_attention,
)


def main():
    banner("M02 - Paged Attention: 显存的虚拟内存")

    # --- 1) 多序列并发分配 ---------------------------------------- #
    bm = BlockManager(num_blocks=8, block_size=4)
    print(f"\n[1] 多序列并发分配 (pool={bm.num_blocks} blocks, block_size={bm.block_size})")
    bm.allocate(seq_id=101, n_tokens=10)  # ceil(10/4)=3 blocks
    bm.allocate(seq_id=102, n_tokens=6)   # ceil(6/4) =2 blocks
    bm.allocate(seq_id=103, n_tokens=3)   # 1 block
    print(f"  seq 101 block_table: {bm.block_table(101)}")
    print(f"  seq 102 block_table: {bm.block_table(102)}")
    print(f"  seq 103 block_table: {bm.block_table(103)}")
    print(f"  pool 状态:           {bm.stats()}")

    # --- 2) 序列结束, block 自动归还 ----------------------------- #
    print("\n[2] seq 102 结束, 它的 block 立即被 free_list 吃掉")
    bm.free(102)
    print(f"  pool 状态:           {bm.stats()}")
    print(f"  free_list:           {list(bm.free_list)}")

    # --- 3) 序列追加 token, 必要时分配新 block -------------------- #
    cur = 12                                     # 3 个 block 刚好装满的长度
    print(f"\n[3] seq 101 增长到 {cur + 1} token, 触发新 block 分配")
    # 3 个 block 装满 12 个 token, 第 13 个要第 4 个 block (跳过 11、12, 只演示跨 block 的那一步)
    blk = bm.append(seq_id=101, current_len=cur)
    print(f"  新分配的 block:      {blk}")
    print(f"  seq 101 block_table: {bm.block_table(101)}")

    # --- 4) 池子用满, 申请失败 ----------------------------------- #
    print("\n[4] 申请超出 pool 容量")
    try:
        bm.allocate(seq_id=999, n_tokens=100)
    except MemoryError as e:
        print(f"  MemoryError: {e}")
    print(f"  → 这就是 m03 scheduler 触发 preempt 的信号")

    # --- 5) 数值正确性: paged vs 连续 KV ------------------------- #
    banner("[5] 数值验证: paged_attention 与连续 KV 计算结果一致")
    rs = np.random.RandomState(0)
    D, T = 8, 11
    Q = rs.randn(1, D).astype(np.float32)        # 一个 query (decode 场景)
    K_full = rs.randn(T, D).astype(np.float32)
    V_full = rs.randn(T, D).astype(np.float32)

    # 把 K_full / V_full 装进 pool
    bm2 = BlockManager(num_blocks=10, block_size=4)
    bm2.allocate(seq_id=1, n_tokens=T)
    block_size = bm2.block_size
    k_pool = np.zeros((bm2.num_blocks, block_size, D), dtype=np.float32)
    v_pool = np.zeros_like(k_pool)
    table = bm2.block_table(1)
    write_kv(k_pool, v_pool, table, np.arange(T), K_full, V_full)   # 11 个 token 散进 3 个不一定相邻的 block

    # paged 路径
    out_paged = paged_attention(Q, k_pool, v_pool, table, ctx_len=T)
    # 连续路径 (m01 风格)
    out_dense = dense_attention(Q, K_full, V_full)

    diff = np.max(np.abs(out_paged - out_dense))
    kv("max |paged - dense|", f"{diff:.2e}")
    assert diff < 1e-6, f"decode (T_q=1): 分页 KV 上的 attention 必须等于连续 KV, 实际差 {diff:.2e}"

    # chunked prefill 形状: 最后 5 个 token 一起当 query, 因果 mask 对齐尾部
    Q5 = rs.randn(5, D).astype(np.float32)
    diff5 = np.max(np.abs(paged_attention(Q5, k_pool, v_pool, table, ctx_len=T)
                          - dense_attention(Q5, K_full, V_full)))
    kv("[6] T_q=5 (prefill chunk) max diff", f"{diff5:.2e}")
    assert diff5 < 1e-6, f"prefill chunk (T_q=5): 分页 KV 上的 attention 必须等于连续 KV, 实际差 {diff5:.2e}"
    K_back, _ = gather_kv(k_pool, v_pool, table, T)
    assert np.array_equal(K_back, K_full), "写进 pool 再按页表读回的 K 必须与原数组逐元素相同"
    waste = len(table) * block_size - T                  # 已分配的槽位数 - 实际 token 数
    kv("内碎片 (末页空槽)", f"{waste} / {len(table) * block_size} slot, 上界 block_size-1={block_size - 1}")
    assert waste < block_size, f"内碎片只出现在最后一页, 应小于 block_size={block_size}, 实际 {waste}"
    print("  ✓ 数值一致, 分页对外语义透明")


if __name__ == "__main__":
    main()
