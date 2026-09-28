"""
M07 — 激活重算 (Activation / Gradient Checkpointing)

是什么: 前向只保存每段 (segment) 的入口激活; 反向走到哪一段, 就从入口重跑那段前向, 用完即弃。
解决的瓶颈: 显存。L 层全存要 O(L) 份激活; 分成长度 k 的段后:
    峰值 ≈ L/k (段边界, 常驻) + k (当前段重算出来的, 瞬时)   →   k = √L 时最小, O(√L)
代价: 前向几乎多算一遍 → 计算 +25%~33% (fwd : bwd ≈ 1 : 2, 总量 3 → 接近 4), 梯度逐位不变。
读代码盯住: `live` 集合 —— 每保存/释放一个激活都经过它, `peak` 是它的历史最大值 (含段内瞬时激活)。
"""
from __future__ import annotations

import math

import numpy as np

from llm_train.core import banner, kv, max_abs_diff


class Tracker:
    """激活账本: 同时数 '此刻活着几份激活' 和 '前向层调用了几次'。"""

    def __init__(self) -> None:
        self.live, self.peak, self.fwd_calls = set(), 0, 0

    def keep(self, i: int) -> None:
        """记下 "第 i 层的输入激活现在占着显存", 顺手更新峰值。"""
        self.live.add(i)
        self.peak = max(self.peak, len(self.live))

    def free(self, i: int) -> None:
        self.live.discard(i)


def layer(h, w, tr: Tracker):
    """一层前向 tanh(h @ w)。h [B, H], w [H, H]。每调一次, 账本上的前向次数加 1。"""
    tr.fwd_calls += 1
    return np.tanh(h @ w)                                  # [B, H] -> [B, H]


def backward_layer(d_out, h_in, h_out, w):
    """一层反向。d_out [B, H] 是 loss 对本层输出的梯度。返回 (dW [H, H], 传给上一层的梯度 [B, H])。"""
    d_pre = d_out * (1 - h_out * h_out)                    # 需要 h_out (tanh') 和 h_in (dW) → 两者都得活着
    return h_in.T @ d_pre, d_pre @ w.T                     # [H, B] @ [B, H] → [H, H];  [B, H] @ [H, H] → [B, H]


def run(x, y, weights, segment: int):
    """segment=1 即 '每层都存' 的普通反向; segment=k 即每 k 层设一个 checkpoint。"""
    L, tr = len(weights), Tracker()                        # L: 层数
    acts = {0: x}                                          # acts[i] = 第 i 层的输入 (= 第 i-1 层的输出)
    tr.keep(0)

    # ---- 前向: 只在段边界保存 ----
    h = x
    for i in range(L):
        h = layer(h, weights[i], tr)
        if (i + 1) % segment == 0 or i + 1 == L:           # 段的出口, 或整个网络的出口
            acts[i + 1] = h
            tr.keep(i + 1)

    d = 2.0 * (h - y) / y.size                             # [B, H] MSE 对输出求导, 反向的起点
    grads = [None] * L

    # ---- 反向: 逐段 "重算 → 反向 → 释放" ----
    for start in reversed(range(0, L, segment)):
        end = min(start + segment, L)                      # 这一段是第 start .. end-1 层
        h = acts[start]                                    # 段入口: 前向时存下来的 checkpoint
        for i in range(start, end - 1):                    # 重算段内部激活; 段出口 acts[end] 本来就存着
            h = layer(h, weights[i], tr)
            acts[i + 1] = h
            tr.keep(i + 1)                                 # 瞬时激活也占显存, 必须计入峰值
        for i in reversed(range(start, end)):
            grads[i], d = backward_layer(d, acts[i], acts[i + 1], weights[i])
            tr.free(i + 1)                                 # 第 i 层反向完, 它的输出就没用了
            del acts[i + 1]
    return grads, tr


def main() -> None:
    banner("M07 - Activation Checkpointing")

    rs = np.random.RandomState(7)
    L, H = 16, 32
    x = rs.randn(4, H).astype(np.float32)                  # [B=4, H]
    y = rs.randn(4, H).astype(np.float32)                  # [B=4, H]
    # 除以 √H: h @ w 每个元素的方差约等于 h 的均方, 不随层数放大
    weights = [(rs.randn(H, H) / np.sqrt(H)).astype(np.float32) for _ in range(L)]

    base_grads, base = run(x, y, weights, segment=1)

    print(f"  L = {L} 层; 每份激活 {x.nbytes} B\n")
    print(f"  {'段长 k':>6}{'峰值激活份数':>12}{'前向调用':>8}{'额外前向':>8}   max|Δgrad|")
    peaks, extra = {}, {}                                  # segment k → 峰值激活份数 / 额外前向次数
    for k in (1, 2, 4, 8, 16):
        grads, tr = run(x, y, weights, segment=k)
        diff = max(max_abs_diff(a, b) for a, b in zip(grads, base_grads))
        peaks[k], extra[k] = tr.peak, tr.fwd_calls - L
        note = "  ← 全存基线" if k == 1 else "  ← √L" if k * k == L else ""
        print(f"  {k:>8}{tr.peak:>14}{tr.fwd_calls:>12}{tr.fwd_calls - L:>11}   {diff:.1e}{note}")
        assert diff == 0.0, "重算的是同一串浮点运算, 梯度必须逐位相同"
        assert tr.peak == L // k + k, "峰值 = 段边界数 (L/k + 输入) + 段内瞬时 (k - 1)"
        assert tr.fwd_calls == L + (L - L // k), "每段除出口外的层都重算一次"

    k_opt = math.isqrt(L)                                  # √L
    print()
    kv(f"峰值显存 (k=1 → k={k_opt})", f"{peaks[1] * x.nbytes} → {peaks[k_opt] * x.nbytes} B  ({peaks[1] / peaks[k_opt]:.1f}x)")
    assert min(peaks, key=peaks.get) == k_opt, "k = √L 最省"
    assert base.peak == L + 1, "全存基线的峰值 = L 层的输出 + 1 份输入"

    # 总计算按 "前向当量" 算: 反向约是前向的 2 倍, 所以全存基线 = 3L
    total0, total = 3 * L, 3 * L + extra[k_opt]
    print(f"\n  OK: 梯度与全存基线逐位相同; 峰值激活 {peaks[1]} → {peaks[k_opt]} 份, 代价是 {extra[k_opt]} 次额外前向"
          f" (总计算 {total0} → {total} 个前向当量, +{total / total0 - 1:.0%})。")


if __name__ == "__main__":
    main()
