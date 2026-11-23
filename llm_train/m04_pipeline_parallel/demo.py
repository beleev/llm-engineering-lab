"""
M04 — 流水线并行: GPipe / 1F1B / Interleaved 1F1B / Zero Bubble / DualPipe

是什么: 模型按层切成 PP 个 stage, batch 切成 M 个 micro-batch 在 stage 间流动。
解决的瓶颈: 整个模型放不下 (层间切, 只传边界激活, 通信最省); 新问题是 **气泡** 和 **在途激活**。
关键公式: 气泡占比 = (PP-1) / (v·M + PP-1)          v = 每卡的虚拟 stage 数 (非交错时 v=1)
          在途激活峰值 (stage s): GPipe = M,  1F1B = PP - s   ← 1F1B 省的是显存, 不是时间
          Zero Bubble (ZB-H1): 反向拆成只算输入梯度的 B 和只算权重梯度的 W, 用 W 去填气泡,
          每卡空闲 = (PP-1)·(t_f + t_b - t_w), 三者耗时相等时正好是 1F1B 的 1/3
读代码盯住: `device_order` 里的 warmup —— 它就是 "最多允许几个 micro-batch 只 F 没 B";
            `simulate_zb` 里的 mem_cap —— 它把在途激活压回 1F1B 的峰值 PP。
说明: 反向 ≈ 2× 前向, 所以前三种调度里 B 占 2 个时间格 (t_b 可调); 拆开后 B 和 W 各占 1 格, 总量不变。
简化: DualPipe 的双向流水线不做事件模拟, 只按 DeepSeek-V3 技术报告的代价公式算 (`dualpipe_cost`)。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from llm_train.core import banner, kv

Op = Tuple[str, int, int]          # ("F"|"B", micro-batch, chunk);  chunk = 该卡上第几个虚拟 stage


def device_order(kind: str, s: int, PP: int, M: int, v: int = 1) -> List[Op]:
    """stage s 这张卡按什么顺序执行自己的 M·v 个 F 和 M·v 个 B。"""
    # F 的顺序: 每 PP 个 micro-batch 一组, 组内先把 chunk 0 跑完再跑 chunk 1 (Megatron 交错顺序)
    fwd = [("F", g + i, c) for g in range(0, M, PP) for c in range(v) for i in range(min(PP, M - g))]
    bwd = [("B", g + i, c) for g in range(0, M, PP) for c in reversed(range(v)) for i in range(min(PP, M - g))]
    if kind == "gpipe":
        return fwd + bwd                                  # 全部 F 完才开始 B → 在途 = M
    # 1F1B: 先做 warmup 个 F, 之后严格 F/B 交替 → 在途恒为 warmup+1
    # v == 1: warmup = PP-1-s, 即本卡后面还有几个 stage; 在途峰值 warmup+1 = PP-s, 对上文件头的公式
    # v > 1 (交错): 用的是 Megatron-LM 的公式, 两项相加
    #   (v-1)·PP    fwd 里前 (v-1)·PP 个都是前 v-1 个 chunk 的 F, 做完才轮到最后一个 chunk
    #   (PP-1-s)·2  最后一个 chunk 上还能再做几个 F; 系数 2 本文件不推导
    warmup = (PP - 1 - s) if v == 1 else (PP - 1 - s) * 2 + (v - 1) * PP
    warmup = min(warmup, len(fwd))                        # M 很小时 F 总数不够 warmup, 截到全部 F
    order, f_i, b_i = list(fwd[:warmup]), warmup, 0       # f_i / b_i: 下一个要排的 F / B 的下标
    while b_i < len(bwd):
        if f_i < len(fwd):
            order.append(fwd[f_i]); f_i += 1
        order.append(bwd[b_i]); b_i += 1
    return order


def simulate(kind: str, PP: int, M: int, v: int = 1, t_f: int = 1, t_b: int = 2):
    """每张卡按固定顺序执行; 一个 op 在 "卡空闲 且 依赖完成" 的最早时刻开始。

    依赖: F(m, 虚拟stage k) 等 F(m, k-1);  B(m, k) 等 B(m, k+1);  最后一个 stage 的 B 等自己的 F。
    虚拟 stage k = chunk·PP + s, 共 PP·v 个, 第 k 个住在卡 k % PP 上。
    """
    if v > 1:
        assert M % PP == 0, "交错调度要求 M 是 PP 的整数倍"
    orders = [device_order(kind, s, PP, M, v) for s in range(PP)]
    end: Dict[Tuple[str, int, int], int] = {}             # (F/B, m, 虚拟stage) -> 完成时刻
    free, ptr = [0] * PP, [0] * PP                        # 卡 s 最早空闲的时刻 / 执行到顺序表第几个
    rows: List[List[str]] = [[] for _ in range(PP)]       # 卡 s 的时间线, 一格一个字符串, "." = 空闲
    live, peak = [0] * PP, [0] * PP                       # 卡 s 此刻在途的激活份数 / 它的历史最大值
    last = PP * v - 1                                     # 最后一个虚拟 stage 的编号

    while any(p < len(o) for p, o in zip(ptr, orders)):
        progressed = False
        for s in range(PP):
            if ptr[s] == len(orders[s]):
                continue
            op, m, c = orders[s][ptr[s]]
            k = c * PP + s                                # 这个 op 所在的虚拟 stage 编号
            # 三种依赖, 对应 docstring 里的三条; 第 0 个虚拟 stage 的 F 没有依赖
            dep = ("F", m, k - 1) if op == "F" and k > 0 else \
                  ("B", m, k + 1) if op == "B" and k < last else \
                  ("F", m, k) if op == "B" else None
            if dep is not None and dep not in end:
                continue                                  # 依赖还没被调度, 这张卡先等
            start = max(free[s], end[dep] if dep else 0)  # 卡空闲和依赖完成, 两个时刻取晚的
            dur = t_f if op == "F" else t_b
            # 先用 "." 把空闲的格子补到 start, 再占 dur 格
            rows[s] += ["."] * (start - len(rows[s])) + [f"{op}{m}"] * dur
            end[(op, m, k)] = free[s] = start + dur
            live[s] += 1 if op == "F" else -1             # F 产生一份待反向的激活, B 释放一份
            peak[s] = max(peak[s], live[s])
            ptr[s] += 1
            progressed = True
        assert progressed, "调度死锁"

    total = max(free)                                     # 最后一张卡收工的时刻 = 总时长
    table = [r + ["."] * (total - len(r)) for r in rows]  # 每行补齐到同样长, 方便打印
    busy = M * v * (t_f + t_b)                            # 每张卡的有效工作量
    return table, peak, 1.0 - busy / total                # 气泡占比 = 空闲格 / 总格


def print_table(table) -> None:
    """一行一张卡, 一列一个时间格。F3 = micro-batch 3 的前向, B3 = 它的反向, "." = 空闲。"""
    for i, row in enumerate(table):
        print(f"  s{i}: " + " ".join(f"{c:>2}" for c in row))


# ---- Zero Bubble: 把反向拆成 B 和 W ------------------------------------ #
# B(m, s) 只算输入梯度 dX, 传给上一个 stage; W(m, s) 只算权重梯度 dW, 谁都不等。
# 所以 W 可以往后挪, 填进这张卡原本空着的格子; 代价是它要用的那份输入激活不能提前释放。

def simulate_zb(PP: int, M: int, t_f: int = 1, t_b: int = 1, t_w: int = 1, mem_cap: int | None = None):
    """ZB-H1: F/B 的顺序和 1F1B 一模一样, W 只在这张卡没别的事可做时插进去。

    mem_cap = 一张卡同时允许几个 micro-batch 处于 "F 做完、W 没做" 的状态。
    默认取 PP, 也就是 1F1B 里 stage 0 的在途激活峰值 —— 这是 ZB-H1 与 1F1B 同显存的那个口径。
    依赖: F(m, s) 等 F(m, s-1);  B(m, s) 等 B(m, s+1) (最后一个 stage 等自己的 F);  W(m, s) 等 B(m, s)。
    """
    # 论文把 ZB-H1 的每卡显存写成 (p-i+1)·M_B + (i-1)·M_W: 在途总份数每张卡都是 p,
    # 但靠后的 stage 里有一部分已经做完 B、只等 W, 这部分只要 M_W (比 M_B 小),
    # 所以峰值仍由 stage 0 决定, 等于 1F1B 的 p·M_B。本文件只数份数, 不区分 M_B 和 M_W。
    cap = PP if mem_cap is None else mem_cap
    orders = [[(op, m) for op, m, _ in device_order("1f1b", s, PP, M)] for s in range(PP)]
    end: Dict[Tuple[str, int, int], int] = {}             # (F/B/W, m, stage) -> 完成时刻
    free, ptr = [0] * PP, [0] * PP                        # 卡 s 最早空闲的时刻 / 执行到顺序表第几个
    pending: List[List[int]] = [[] for _ in range(PP)]    # 卡 s 上 B 做完、W 还没做的 micro-batch
    rows: List[List[str]] = [[] for _ in range(PP)]
    live, peak = [0] * PP, [0] * PP                       # 在途 = F 做完但 W 没做的份数
    t = 0
    while any(ptr[s] < len(orders[s]) or pending[s] for s in range(PP)):
        for s in range(PP):
            if free[s] > t:                               # 这张卡还在忙
                continue
            op = None
            if ptr[s] < len(orders[s]):
                kind, m = orders[s][ptr[s]]
                dep = ("F", m, s - 1) if kind == "F" and s > 0 else \
                      ("B", m, s + 1) if kind == "B" and s < PP - 1 else \
                      ("F", m, s) if kind == "B" else None
                full = kind == "F" and live[s] >= cap     # ★ 显存满了: 先还一个 W 再收新的 F
                if not full and (dep is None or end.get(dep, 1 << 30) <= t):
                    op = (kind, m)
                    ptr[s] += 1
            if op is None and pending[s]:                 # ★ 没别的事可做 (或显存满) 就补一个 W
                op = ("W", pending[s].pop(0))
            if op is None:
                continue
            kind, m = op
            dur = {"F": t_f, "B": t_b, "W": t_w}[kind]
            rows[s] += ["."] * (t - len(rows[s])) + [f"{kind}{m}"] * dur
            end[(kind, m, s)] = free[s] = t + dur
            if kind == "F":
                live[s] += 1
            elif kind == "B":
                pending[s].append(m)                      # B 不释放激活, W 才释放
            else:
                live[s] -= 1
            peak[s] = max(peak[s], live[s])
        t += 1
        assert t <= 4 * M * (t_f + t_b + t_w) + 4 * PP, "调度没有收敛"

    total = max(free)
    table = [r + ["."] * (total - len(r)) for r in rows]
    busy = M * (t_f + t_b + t_w)
    # 依赖合法性: 三条依赖逐个 micro-batch 核对一遍
    for m in range(M):
        for s in range(PP):
            assert end[("W", m, s)] - t_w >= end[("B", m, s)], f"W({m},{s}) 必须排在自己的 B 之后"
            if s < PP - 1:
                assert end[("B", m, s)] - t_b >= end[("B", m, s + 1)], f"B({m},{s}) 必须等下一个 stage 的 B"
            if s > 0:
                assert end[("F", m, s)] - t_f >= end[("F", m, s - 1)], f"F({m},{s}) 必须等上一个 stage 的 F"
    return table, peak, 1.0 - busy / total, total


def dualpipe_cost(PP: int, t_f: int = 1, t_b: int = 1, t_w: int = 1, t_fb: int | None = None):
    """按 DeepSeek-V3 技术报告的代价公式算三种调度的每卡空闲、参数份数和在途激活。

    报告里的符号: F = 一次前向, B = 一次 **完整** 反向 (= 本文件的 t_b + t_w), W = 只算权重梯度,
    F&B = 一对前向与反向合并执行的块 (计算掩盖 all-to-all 与 PP 通信, 所以可以比 F + B 短)。
    报告的 ZB1P 行换成本文件的符号就是 (PP-1)(t_f + t_b - t_w), 也就是 ZB-H1, 不是 ZB-H2。
    这是代价模型, 不是事件模拟: DualPipe 两端同时喂 micro-batch, 时间表要双向排, 本文件不做。
    """
    F, B, W = t_f, t_b + t_w, t_w
    FB = F + B if t_fb is None else t_fb
    assert PP % 2 == 0, "DualPipe 要求 stage 数和 micro-batch 数都能被 2 整除"
    return {
        #  名字            每卡空闲                       参数份数  在途激活 (micro-batch 份数)
        "1F1B": ((PP - 1) * (F + B), 1, PP),
        "ZB1P": ((PP - 1) * (F + B - 2 * W), 1, PP),
        "DualPipe": ((PP // 2 - 1) * (FB + B - 3 * W), 2, PP + 1),
    }


def main() -> None:
    banner("M04 - Pipeline Parallel: GPipe / 1F1B / Interleaved")

    PP, M, v, t_b = 3, 6, 2, 2                             # t_b: 反向占几个时间格 (前向占 1 格)
    results = {}
    for name, kind, vv in [("GPipe", "gpipe", 1), ("1F1B", "1f1b", 1), (f"Interleaved v={v}", "1f1b", v)]:
        table, peak, bubble = simulate(kind, PP, M, vv, t_b=t_b)
        results[name] = (peak, bubble)
        unit = "1 格 = 1 个 stage 的前向" if vv == 1 else f"1 格 = 1/{vv} 个 stage 的前向 (时间轴被放大 {vv} 倍)"
        print(f"\n[{name}]  PP={PP}, M={M}; {unit}; B 占 {t_b} 格")
        print_table(table)

    print()
    for name, (peak, bubble) in results.items():
        formula = (PP - 1) / ((v if "Inter" in name else 1) * M + PP - 1)
        kv(f"{name} 气泡 (实测 / 公式)", f"{bubble:.1%} / {formula:.1%}")
    for name, (peak, _) in results.items():
        unit = " (单位: 1/v 个 stage 的激活)" if "Inter" in name else ""
        kv(f"{name} 在途激活峰值 / stage", f"{peak}{unit}")

    # ---- 精确断言: (PP=3, M=4) 和 (PP=4, M=8) 两组配置 ----
    for pp, m in [(3, 4), (4, 8)]:
        _, peak_g, bub_g = simulate("gpipe", pp, m)
        _, peak_1, bub_1 = simulate("1f1b", pp, m)
        assert peak_g == [m] * pp, "GPipe 每个 stage 的在途激活峰值都是 M"
        assert peak_1 == [pp - s for s in range(pp)], \
            f"1F1B 在 stage s 的在途激活峰值应是 PP - s, 实得 {peak_1}"  # [3,2,1] / [4,3,2,1]
        assert abs(bub_g - (pp - 1) / (m + pp - 1)) < 1e-12, "GPipe 气泡占比必须等于 (PP-1)/(M+PP-1)"
        assert abs(bub_1 - bub_g) < 1e-12, "1F1B 不减少气泡, 只减少在途激活"
        kv(f"PP={pp}, M={m}: 1F1B 峰值", f"{peak_1}  (GPipe {peak_g})")
    pp, m = 4, 8
    _, peak_i, bub_i = simulate("1f1b", pp, m, v=v)
    bub_1 = (pp - 1) / (m + pp - 1)                        # 1F1B 的气泡, 与 GPipe 相同
    assert abs(bub_i - (pp - 1) / (v * m + pp - 1)) < 1e-12, "交错的气泡占比必须等于 (PP-1)/(v·M+PP-1)"
    assert bub_i < bub_1, "交错的气泡必须小于 1F1B 的 (PP-1)/(M+PP-1)"
    # 1F1B 的最大在途峰值是 PP 份整 stage 激活 (stage 0); 交错的单位是 1/v 个 stage, 除以 v 折算
    assert max(peak_i) / v > pp, "交错的代价: 在途激活 (折算成整 stage) 比 1F1B 多"
    kv(f"PP={pp}, M={m}: 气泡 1F1B → 交错(v={v})", f"{bub_1:.1%} → {bub_i:.1%}")
    kv(f"PP={pp}, M={m}: 交错在途峰值", f"{peak_i} 份 1/{v}-stage 激活 = {max(peak_i) / v:.1f} 个 stage (1F1B 为 {pp})")

    print("\n  OK: GPipe 与 1F1B 气泡相同、显存不同; 交错把气泡再除以 v, 代价是 v 倍的 P2P 次数和更多在途激活。")

    # ---- Zero Bubble: 反向拆成 B 和 W, 用 W 填气泡 ----
    pp, m = 4, 8
    zb_table, zb_peak, zb_bub, zb_total = simulate_zb(pp, m)
    print(f"\n[Zero Bubble ZB-H1]  PP={pp}, M={m}; 1 格 = 1 个 stage 的前向; B 和 W 各占 1 格")
    print_table(zb_table)
    _, peak_1, bub_1 = simulate("1f1b", pp, m, t_b=2)      # 同样的总计算量: 完整反向 = B + W = 2 格
    total_1 = round(m * 3 / (1 - bub_1))
    print()
    kv("1F1B 总时长 / 每卡空闲", f"{total_1} 格 / {total_1 - 3 * m} 格")
    kv("ZB-H1 总时长 / 每卡空闲", f"{zb_total} 格 / {zb_total - 3 * m} 格")
    kv("气泡占比 1F1B → ZB-H1", f"{bub_1:.1%} → {zb_bub:.1%}  (比值 {zb_bub / bub_1:.2f})")
    kv("在途激活峰值 1F1B / ZB-H1", f"{peak_1} / {zb_peak}  (峰值都是 {pp})")

    # 每卡空闲 = (PP-1)·(t_f + t_b - t_w): 换几组 (PP, M) 和几组耗时都成立 (要求 t_w ≤ t_b)
    for pp2, m2 in [(2, 4), (3, 6), (4, 8), (4, 12), (8, 16)]:
        for t_f2, t_b2, t_w2 in [(1, 1, 1), (1, 2, 1), (2, 2, 1), (2, 3, 1)]:
            _, peak2, _, total2 = simulate_zb(pp2, m2, t_f2, t_b2, t_w2)
            idle = total2 - m2 * (t_f2 + t_b2 + t_w2)
            assert idle == (pp2 - 1) * (t_f2 + t_b2 - t_w2), \
                f"PP={pp2}, M={m2}, 耗时 {t_f2}/{t_b2}/{t_w2}: 每卡空闲应为 (PP-1)(t_f+t_b-t_w), 实得 {idle}"
            assert peak2 == [pp2] * pp2, f"ZB-H1 每张卡的在途激活峰值都应是 PP, 实得 {peak2}"
    # 三者耗时相等时, 1F1B 的每卡空闲是 (PP-1)·3, ZB-H1 是 (PP-1)·1
    assert zb_total - 3 * m == pp - 1, "t_f = t_b = t_w 时 ZB-H1 的每卡空闲应是 PP-1"
    assert total_1 - 3 * m == 3 * (pp - 1), "同一口径下 1F1B 的每卡空闲是 3(PP-1)"
    assert max(zb_peak) == max(peak_1) == pp, "ZB-H1 与 1F1B 的在途峰值相同 (都是 PP 份)"
    assert zb_peak[-1] > peak_1[-1], "代价: 靠后的 stage 在 ZB-H1 下在途的份数比 1F1B 多"

    # ---- DualPipe: 不做事件模拟, 按 DeepSeek-V3 技术报告的代价公式算 ----
    print()
    for pp2 in [4, 8, 16]:
        cost = dualpipe_cost(pp2, t_f=1, t_b=1, t_w=1, t_fb=2)
        kv(f"PP={pp2} 每卡空闲 / 参数份数 / 在途激活 (代价模型)",
           "; ".join(f"{n} {b}/{p}x/{a}" for n, (b, p, a) in cost.items()))
    cost = dualpipe_cost(8, t_f=1, t_b=1, t_w=1, t_fb=2)
    assert cost["1F1B"][0] > cost["ZB1P"][0] > cost["DualPipe"][0], \
        f"代价模型里空闲应依次减小, 实得 {[c[0] for c in cost.values()]}"
    assert cost["ZB1P"][0] == cost["1F1B"][0] / 3, "t_f = t_b = t_w 时 ZB1P 的空闲是 1F1B 的 1/3"
    assert cost["DualPipe"][1] == 2 * cost["ZB1P"][1], "DualPipe 两端同时跑, 每卡要放两份参数"
    assert dualpipe_cost(8)["ZB1P"][0] == (8 - 1) * (1 + 2 - 2), "ZB1P 的公式是 (PP-1)(F+B-2W)"

    print("\n  OK: 拆出 W 以后, 事件模拟的每卡空闲从 (PP-1)(t_f+t_b+t_w) 降到 (PP-1)(t_f+t_b-t_w),")
    print("      在途激活峰值仍是 PP; DualPipe 的数字出自报告的代价公式, 本文件没有排它的双向时间表。")


if __name__ == "__main__":
    main()
