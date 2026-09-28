"""
M11 — 专家并行 (Expert Parallel, MoE)

是什么: E 个专家 FFN 分给 D 张卡 (每卡 E/D 个)。token 所在的卡 ≠ 它选中的专家所在的卡, 于是:
    all-to-all #1 dispatch: token 发往专家所在卡 → 本地专家计算 → all-to-all #2 combine: 结果原路寄回
解决的瓶颈: 显存 (MoE 参数大头在专家上)。新瓶颈是通信 + **负载均衡**: all-to-all 的量由路由结果决定,
           最热的卡决定 step 时间; 超过 capacity 的 token 被丢弃 (只走残差)。
关键公式: capacity = ⌈cf · N_tok / E⌉;   Switch aux loss  L = E · Σ_e f_e · P_e   (f: 实际占比, P: 平均概率)
          DeepSeek-V3 aux-loss-free: 选专家用 s_e + b_e, 过载 b_e -= γ, 欠载 b_e += γ; gate 权重仍用原始 s_e
读代码盯住: `send_idx[src][dst]` —— 行号留在源卡不上网线, combine 回来的块靠它按原顺序写回。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_to_all, banner, comm, kv, make_rng, max_abs_diff, relu, softmax


def expert_forward(weights, x):
    """一个专家 = 一个两层 FFN。weights = (w1 [d, d_ff], w2 [d_ff, d]); x [n, d] → [n, d]。"""
    w1, w2 = weights
    return relu(x @ w1) @ w2                                  # [n, d] -> [n, d_ff] -> [n, d]


def route(x, router_w, bias=None):
    """top-1 路由。bias 只影响 '选谁', 不影响 gate 权重 (aux-loss-free 的关键)。

    x [N, d], router_w [d, E], bias [E] → (每个 token 选中的专家 [N], 它的 gate 权重 [N], 概率 [N, E])。
    """
    probs = softmax(x @ router_w)                             # [N, d] @ [d, E] → [N, E]
    expert_of = (probs if bias is None else probs + bias).argmax(axis=1)    # [N]
    return expert_of, probs[np.arange(len(x)), expert_of], probs            # gate = 选中那个专家的原始概率


def moe_forward_ep(x_shards, expert_shards, gate_shards, experts, capacity=None):
    """x_shards: D × [n_local, d]。返回 (D × [n_local, d], 发送矩阵, 丢弃数)。

    expert_shards: D × [n_local], 每个 token 选中的专家 id。gate_shards: D × [n_local], 它的 gate 权重。
    """
    world, per_dev = len(x_shards), len(experts) // len(x_shards)       # 卡数 D, 每卡的专家数 E/D

    # ---- all-to-all #1 dispatch: 每张卡按目标卡把本地 token 分成 D 堆 ----
    # send_idx[s][d]: 卡 s 上要发往卡 d 的 token 的行号。专家 id // per_dev = 专家所在的卡
    send_idx = [[np.where(expert_shards[s] // per_dev == d)[0] for d in range(world)] for s in range(world)]
    recv_tok = all_to_all([[x_shards[s][i] for i in row] for s, row in enumerate(send_idx)])   # [dst][src]: [n_sd, d]
    recv_eid = all_to_all([[expert_shards[s][i] for i in row] for s, row in enumerate(send_idx)])  # 专家 id 随行

    # ---- 本地专家计算: 卡 dev 只持有专家 [dev·per_dev, (dev+1)·per_dev) ----
    dropped = 0
    back = [[None] * world for _ in range(world)]
    for dev in range(world):
        toks = np.concatenate(recv_tok[dev])                  # [n_recv, d], 按 src 顺序拼
        eids = np.concatenate(recv_eid[dev])                  # [n_recv], 每个 token 要找的专家
        out = np.zeros_like(toks)                             # 被丢弃的 token 输出 0 (外面还有残差连接)
        for e in range(dev * per_dev, (dev + 1) * per_dev):
            rows = np.where(eids == e)[0]                     # 这张卡收到的 token 里, 归专家 e 的行号
            if capacity is not None and len(rows) > capacity:
                dropped += len(rows) - capacity
                rows = rows[:capacity]                        # 先到先得, 超出容量的丢弃
            out[rows] = expert_forward(experts[e], toks[rows])
        # np.split 要的是切点: 各 src 那一堆长度的累加; 最后一个切点是末尾, 不要
        sizes = np.cumsum([len(t) for t in recv_tok[dev]])[:-1]
        back[dev] = np.split(out, sizes)                      # 按 src 切回去, 顺序与收到时一致

    # ---- all-to-all #2 combine: 原路寄回, 源卡按自己留着的行号写回并乘 gate ----
    returned = all_to_all(back)                               # [src][dev]: 与 send_idx[src][dev] 行行对齐
    outs = []
    for s in range(world):
        y = np.zeros_like(x_shards[s])                        # [n_local, d]
        for d in range(world):
            y[send_idx[s][d]] = returned[s][d]
        outs.append(y * gate_shards[s][:, None])              # [n_local, d] × [n_local, 1]
    matrix = np.array([[len(i) for i in row] for row in send_idx])   # [D, D]: 卡 s 发给卡 d 几个 token
    return outs, matrix, dropped


def train_router_aux(x, router_w, steps, lr):
    """只用 Switch aux loss 训练 router —— 刻意隔离它的作用; 真实训练是 L_task + α·L_aux (α≈0.01)。"""
    E, w = router_w.shape[1], router_w.copy()                 # w [d, E]: router 权重的副本, 不改原件
    for _ in range(steps):
        probs = softmax(x @ w)                                # [N, E]
        f = np.bincount(probs.argmax(1), minlength=E) / len(x)        # [E] 实际占比; argmax 不可导, f 当常数
        # L = E·Σ_e f_e·P_e, P_e 是 N 个 token 概率的平均 → 对每个 token 的概率求导得 E·f_e / N
        grad_p = np.tile(f, (len(x), 1)) * E / len(x)                 # [N, E] dL/dP
        grad_logits = probs * (grad_p - (probs * grad_p).sum(1, keepdims=True))   # [N, E] 过 softmax 的 Jacobian
        w -= lr * x.T @ grad_logits                           # [d, N] @ [N, E] → [d, E]
    return w


def balance_bias(x, router_w, steps, gamma):
    """aux-loss-free: 不加 loss、不产生干扰梯度, 每步按负载符号微调一个 [E] 的 bias。"""
    E = router_w.shape[1]
    bias = np.zeros(E)
    for _ in range(steps):
        load = np.bincount(route(x, router_w, bias)[0], minlength=E)   # [E] 每个专家分到几个 token
        bias += gamma * np.sign(load.mean() - load)           # 过载 → 降, 欠载 → 升
    return bias


def main() -> None:
    banner("M11 - Expert Parallel (MoE all-to-all)")

    rs = make_rng(7)
    world, E, N, d, d_ff = 4, 8, 64, 8, 16
    # world 卡数, E 专家数, N token 数, d 模型维, d_ff 专家 FFN 的中间维
    x = rs.randn(N, d)                                                # [N, d]
    experts = [(rs.randn(d, d_ff) * 0.3, rs.randn(d_ff, d) * 0.3) for _ in range(E)]
    router_w = rs.randn(d, E) + np.linspace(1.2, -1.2, E)[None, :]    # 故意倾斜: 偏爱编号小的专家
    capacity = int(np.ceil(1.25 * N / E))                             # capacity factor 1.25 → 10

    def run(expert_of, gate, cap=None):
        comm.reset()
        shards = lambda a: np.split(a, world)                         # [N, ...] -> D × [N/D, ...]
        outs, matrix, dropped = moe_forward_ep(shards(x), shards(expert_of), shards(gate), experts, cap)
        return np.concatenate(outs), matrix, dropped, comm.total

    # ---- 1) 正确性: EP (dispatch + combine 都真的走 all-to-all) == 单卡逐专家计算 ----
    # dispatch 把 token 和专家 id 分两次发, 加上 combine 一共 3 次 all_to_all 调用
    expert_of, gate, _ = route(x, router_w)
    out_ep, matrix, _, wire = run(expert_of, gate)
    dense = np.zeros_like(x)
    for e in range(E):
        sel = expert_of == e                                          # [N] bool: 哪些 token 选了专家 e
        dense[sel] = expert_forward(experts[e], x[sel]) * gate[sel, None]
    print("\n[1] dispatch 发送矩阵 (行 = 源卡, 列 = 目标卡)")
    for row in matrix:
        print("      " + " ".join(f"{v:>3}" for v in row))
    kv("max |dense - EP|", f"{max_abs_diff(dense, out_ep):.1e}")
    kv("每卡收到的 token", f"{matrix.sum(0).tolist()}  (均匀应为 {N // world})")
    kv("通信量 (dispatch + combine)", f"{wire:.0f} B/rank")
    assert max_abs_diff(dense, out_ep) == 0.0, "EP 的输出必须与单卡逐专家计算逐位相同"
    assert comm.calls["all_to_all"] == 3, "dispatch 发 token 和专家 id (2 次), combine 1 次"

    # ---- 2) 三种路由的负载对比 ----
    w_aux = train_router_aux(x, router_w, steps=60, lr=0.5)
    bias = balance_bias(x, router_w, steps=200, gamma=0.01)
    print(f"\n[2] 负载均衡 (capacity = {capacity})")
    print(f"      {'':<22}{'每专家 token 数':<36}{'max/mean':>9}{'丢弃':>6}{'最热卡':>7}")
    stats = {}
    for name, (eo, g) in {
        "倾斜 router": (expert_of, gate),
        "aux loss 60 步": route(x, w_aux)[:2],
        "aux-loss-free bias": route(x, router_w, bias)[:2],
    }.items():
        counts = np.bincount(eo, minlength=E)
        _, m, dropped, _ = run(eo, g, capacity)
        stats[name] = (counts.max() / counts.mean(), dropped)
        print(f"      {name:<22}{str(counts.tolist()):<36}{stats[name][0]:>8.2f}x{dropped:>6}{m.sum(0).max():>7}")

    base_imb, base_drop = stats["倾斜 router"]
    assert base_drop > 0, "倾斜的 router 应当丢 token, 否则下面的对比没有意义"
    for name in ("aux loss 60 步", "aux-loss-free bias"):
        assert stats[name][0] < base_imb, f"{name}: max/mean 应低于倾斜 router"
        assert stats[name][1] < base_drop, f"{name}: 丢弃的 token 应少于倾斜 router"
    # bias 法不改 router 权重 → gate 概率与原始完全相同, 只是 "选谁" 变了
    assert np.array_equal(route(x, router_w, bias)[2], route(x, router_w)[2]), \
        "加 bias 不能改变 router 算出的概率"

    print("\n  OK: EP 输出与单卡逐位相同; 两种均衡手段都压低了 max/mean 和丢 token 数。")


if __name__ == "__main__":
    main()
