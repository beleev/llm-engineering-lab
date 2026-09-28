"""
m19 demo — 树形投机 vs 链式投机 (同一个 draft 模型)
    [1] 小例子: 打印 tree mask; assert 树形验证的 logits / gather 后的 KV == 顺序 forward
    [2] 生成: assert 输出 == target greedy; 每次 target 调用的产出 tree ≥ chain
    [3] 打印一棵真实的树, 标出被接受的路径

运行: python -m llm_infer.m19_tree_speculation.demo
"""
from __future__ import annotations

import numpy as np

from llm_infer.core import ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m07_speculative_decoding.speculative import ModelDrafter, make_draft, speculative_decode
from llm_infer.m19_tree_speculation.tree_spec import (
    TreeDrafter, gather_kv, render_tree, tree_mask, tree_shape, tree_speculative_decode)

NOISE = 0.1              # draft = target 权重加这么大比例的噪声 (与 m07 同款)
SMALL_WIDTHS = [2, 1]    # [1] 小例子的树形: 第 1 层 2 个分支, 第 2 层每个分支 1 个
WIDTHS = [3, 2, 1]       # [2][3] 用的树形


def small_example(target: TinyLM, draft: TinyLM) -> None:
    """[1] 5 个节点的小树: 打印 mask, 再验证树形 forward 与顺序 forward 给出同样的 logits 和 KV。"""
    ctx = [1, 5, 10]
    n_ctx, n_nodes = len(ctx), len(tree_shape(SMALL_WIDTHS)[0])
    print(f"\n[1] 小例子: widths={SMALL_WIDTHS} → {n_nodes} 个节点, 上下文 {n_ctx} 个 token")
    logits, cache = target.forward(ctx)
    out = ctx + [int(np.argmax(logits[-1]))]
    drafter = TreeDrafter(draft, SMALL_WIDTHS)
    tokens = drafter.propose(out)
    parents, depth = drafter.parents, drafter.depth
    kv("parents (BFS 序)", parents.tolist())
    kv(f"depth → RoPE 位置 = {n_ctx} + depth", (n_ctx + depth).tolist())
    mask = tree_mask(drafter.anc, n_ctx=n_ctx)            # (n_nodes, n_ctx+n_nodes)
    print(f"  tree mask (■=可见, ·=屏蔽; 前 {n_ctx} 列是上下文, 后 {n_nodes} 列是树节点):")
    for i, row in enumerate(mask):
        cells = ["■" if v == 0 else "·" for v in row]
        print(f"    节点{i} (tok {tokens[i]:>3}, 父 {parents[i]:>2})  {' '.join(cells[:n_ctx])} | {' '.join(cells[n_ctx:])}")

    t_logits, t_kv = target.forward(tokens, cache, positions=n_ctx + depth, mask=mask)  # (n_nodes, V)
    # 任取一条根→叶路径, 顺序 forward 应给出同样的 logits 与 KV
    leaf = len(parents) - 1
    branch = np.flatnonzero(drafter.anc[leaf])            # 根→叶的节点号
    seq_logits, seq_kv = target.forward(tokens[branch], cache)
    d_logit = np.abs(t_logits[branch] - seq_logits).max()
    g_kv = gather_kv(t_kv, np.concatenate([np.arange(n_ctx), n_ctx + branch]))
    d_kv = max(np.abs(a - b).max() for g, s in zip(g_kv, seq_kv) for a, b in zip(g, s))   # 逐层、K 和 V 都比
    kv(f"路径 {branch.tolist()}: 树形 vs 顺序 max|Δlogits|", f"{d_logit:.1e}")
    kv("gather 出的 KV vs 顺序 KV max|Δ|", f"{d_kv:.1e}")
    assert d_logit < 1e-4, f"树形验证在这条路径上的 logits 应等于顺序 forward, 实际差 {d_logit:.1e}"
    assert d_kv < 1e-4, f"gather 出的 KV 应等于顺序 forward 的 KV, 实际差 {d_kv:.1e}"


def main() -> None:
    banner("M19 - Tree Speculation (tree attention mask 一次验整棵树)")
    target = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128))
    draft = make_draft(target, noise=NOISE)               # 与 m07 同款: 权重加噪, 模拟蒸馏 draft
    small_example(target, draft)

    widths = WIDTHS
    n_nodes = len(tree_shape(widths)[0])
    n_prompts, max_new = 12, 32
    rs = np.random.RandomState(0)
    prompts = [rs.randint(target.cfg.vocab_size, size=6) for _ in range(n_prompts)]
    refs = [target.generate_greedy(p, max_new) for p in prompts]

    print(f"\n[2] {n_prompts} 个 prompt × {max_new} token, draft = target 权重加噪 {NOISE:.0%} "
          f"(baseline target 调用 = {n_prompts * max_new})")
    rows, best = {}, None
    parents, _ = tree_shape(widths)
    spine = [0]                                           # 树的 top-1 脊 = 同状态下链式 draft 会猜的那条
    while (kids := np.flatnonzero(parents == spine[-1])).size:
        spine.append(int(kids[0]))

    for name, K in [(f"chain K={len(widths)} (同 draft 调用数)", len(widths)),
                    (f"chain K={n_nodes - 1} (同验证 token 数)", n_nodes - 1)]:
        calls = dcalls = 0
        acc = []
        for p, ref in zip(prompts, refs):
            d = ModelDrafter(draft)
            out, c, a = speculative_decode(target, d, p, max_new, K)
            assert out == ref, f"{name}: 链式投机输出必须与 target greedy 逐 token 相同"
            calls, dcalls, acc = calls + c, dcalls + d.calls, acc + a
        # (target 调用数, 每轮平均接受数, 每轮验证的 token 数, 每轮 draft 调用数)
        rows[name] = (calls, np.mean(acc), K + 1, dcalls / len(acc))

    calls = dcalls = 0
    acc, spine_acc = [], []
    for p, ref in zip(prompts, refs):
        d = TreeDrafter(draft, widths)
        out, c, trace = tree_speculative_decode(target, d, p, max_new)
        assert out == ref, "树形投机输出必须与 target greedy 逐 token 相同"
        assert c == 1 + len(trace), "target 调用数应为 1 次 prefill + 每轮 1 次验证"
        calls, dcalls = calls + c, dcalls + d.calls
        for r in trace:
            n_spine = 0                                   # 同一状态下, 只验 top-1 脊能接受几个
            while n_spine < len(widths) and r["tokens"][spine[n_spine + 1]] == r["t_pred"][spine[n_spine]]:
                n_spine += 1
            acc.append(len(r["path"]) - 1)
            spine_acc.append(n_spine)
            # 挑一轮给 [3] 展示: 接受路径最长, 且走的不是 top-1 脊
            if best is None or (len(r["path"]) > len(best["path"]) and r["path"] != spine[:len(r["path"])]):
                best = r
    tree_name = f"tree {widths} ({n_nodes} 节点)"
    rows[tree_name] = (calls, np.mean(acc), n_nodes, dcalls / len(acc))

    print(f"  {'':<32} target调用  每轮接受  token/target调用  验证token/轮  draft调用/轮")
    for name, (c, a, v, dc) in rows.items():
        print(f"  {name:<32} {c:>8}  {a:>8.2f}  {n_prompts * max_new / c:>14.2f}  {v:>10}  {dc:>10.1f}")
    kv("同一状态下: 树 vs 它自己的 top-1 脊", f"{np.mean(acc):.2f} vs {np.mean(spine_acc):.2f} (每轮接受)")

    chain = rows[f"chain K={len(widths)} (同 draft 调用数)"]
    assert all(t >= s for t, s in zip(acc, spine_acc)), "树包含链: 逐轮接受数不可能更少"
    assert rows[tree_name][1] >= chain[1], \
        f"draft 调用数相同时, 树形每轮接受数不应少于链式: {rows[tree_name][1]:.2f} vs {chain[1]:.2f}"
    assert rows[tree_name][0] <= chain[0], \
        f"draft 调用数相同时, 树形的 target 调用数不应多于链式: {rows[tree_name][0]} vs {chain[0]}"

    print("\n[3] 一棵真实的树 (✓ = 被 target 接受的路径; 这一轮 top-1 脊会在半路断掉):")
    print("    " + render_tree(best["tokens"], parents, best["path"]).replace("\n", "\n    "))
    last = best["path"][-1]
    kv("本轮产出", f"{len(best['path']) - 1} 个接受 + 1 个 target 给的 token {best['t_pred'][last]}")
    print(f"\n  代价: 每轮验证 {n_nodes} 个 token 而不是 {len(widths) + 1} 个 —— "
          "decode 带宽受限时几乎免费, 大 batch 算力受限时不免费。")


if __name__ == "__main__":
    main()
