"""
afd.py — Attention–FFN 分离部署 (AFD) 的代价模型: A:F 实例比、ping-pong micro-batch、什么时候不划算

是什么: P/D 分离 (m15) 把 prefill 和 decode 拆到不同实例; AFD 再把 decode 的每一层拆成两半:
    注意力实例只跑 attention, 显存几乎全给 KV; FFN 实例只放专家权重。两边按各自的瓶颈独立扩缩:
        注意力: 每步要把 batch × 上下文 的 KV 读一遍, 受显存带宽限制, 随 batch × 上下文增长
        专家 FFN: 每步要把专家权重读一遍, batch 小时同样受带宽限制; 汇聚的 token 够多才变成算力瓶颈、把算力吃满
    一个全局 batch 切成 m 个 micro-batch, 在两边之间来回传 (ping-pong), 一个 micro-batch 在 FFN 上算时,
    另一个在注意力上算, 第三个在网络上, 通信就被藏起来了。
代价模型 (每层、每个 micro-batch, 全部是估算, 不是实测):
    t_a = (W_attn + b·S·kv_bytes) / BW                        注意力实例, b = 该实例这一份 micro-batch 的序列数
    t_f = max(E/n_f · P_e / BW,  2·P_e·N·k/n_f / FLOPS)       FFN 实例: 读权重 vs 算, 取慢的; N = 这一份的 token 总数
    t_c = max(b·k, N·k/n_f) · D · 2 bytes / NET               一个方向的 all-to-all, 取收发两端更慢的一端
    一层的周期 T = max(t_a + t_f + 2·t_c, m·t_a, m·t_f, m·t_c)  第一项是一个 micro-batch 走一圈, 后三项是各部件的忙碌时间
    TPOT ≈ 层数 · T;  每卡吞吐 = 全局 batch / TPOT / (n_a + n_f)
    MegaScale-Infer 的条件: t_a ≈ t_f, 且 m ≥ 2·(1 + t_c / t_f) 时通信完全藏住。
同卡部署 (对照): 每卡既放 1/N 的专家又跑注意力, EP all-to-all 假设被双 micro-batch 完全藏住 (对它偏乐观):
    T = t_a(b) + t_f(N·b 个 token 摊到 N 卡)
简化: 每个专家负载均衡、所有专家都被激活; 不算 shared expert、dense 层、router、LM head 和 prefill。
简化: 模型参数仿 DeepSeek-V3 / Step-3 的量级但注意力统一写成 GQA 式 KV, 硬件数字取 H800 类的公开规格量级; 全部是假设。
读代码盯住: layer_cycle 里的三项 max, 以及 ffn_time 里"读权重"和"算"哪个更慢。
对应真实系统: MegaScale-Infer (ByteDance, ping-pong + M2N 通信库), Step-3 的 AFD (StepFun, StepMesh),
    2602.09721 (AFD 的挑战), FastAFD (rack 级 NVLink 上何时划算)。
"""
from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HW:
    """一张卡。数字是 H800 类的量级, 全部是假设。"""
    bw: float = 3.35e12          # 显存带宽 B/s
    flops: float = 1.98e15       # FP8 稠密算力 FLOP/s
    net: float = 50e9            # 每卡跨机网络 B/s (400 Gb/s)
    mem: float = 72e9            # 每卡可用显存 B (80 GB 留 8 GB 余量)


@dataclass(frozen=True)
class MoE:
    """一个假想的 MoE 模型 (每层相同)。权重和 KV 都按 FP8 1 字节算。"""
    name: str
    layers: int = 58
    d: int = 7168                # 隐藏维
    experts: int = 256           # 路由专家数 E
    top_k: int = 8
    expert_params: float = 3 * 7168 * 2048    # 每个专家 3 个矩阵
    attn_params: float = 0.19e9  # 每层注意力权重
    kv_bytes: float = 2048       # 每 token 每层 KV 字节数 (GQA 8 个 KV 头 × 128 维 × K、V)


def attn_time(m: MoE, hw: HW, b: float, ctx: int) -> float:
    """一层注意力: 读注意力权重 + 读 b 条序列各 ctx 个 token 的 KV。"""
    return (m.attn_params + b * ctx * m.kv_bytes) / hw.bw


def ffn_time(m: MoE, hw: HW, tokens: float, n_f: int) -> float:
    """一层专家 FFN, tokens 个 token 摊到 n_f 张卡: 读本卡专家权重 和 算 取慢的。"""
    read = m.experts / n_f * m.expert_params / hw.bw
    compute = 2 * m.expert_params * tokens * m.top_k / n_f / hw.flops
    return max(read, compute)                                               # ★ batch 不够大时是 read 说了算


def comm_time(m: MoE, hw: HW, b: float, tokens: float, n_f: int) -> float:
    """一个方向的 all-to-all: 注意力卡发 b·k 份隐藏向量, FFN 卡收 tokens·k/n_f 份, 取慢的一端 (按 2 字节算)。"""
    return max(b * m.top_k, tokens * m.top_k / n_f) * m.d * 2 / hw.net


def layer_cycle(t_a: float, t_f: float, t_c: float, n_mb: int) -> float:
    """ping-pong: 一个 micro-batch 走一圈要 t_a + t_f + 2t_c; n_mb 个轮流用注意力、FFN 和网卡, 各忙 n_mb·t。取最慢的。"""
    return max(t_a + t_f + 2 * t_c, n_mb * t_a, n_mb * t_f, n_mb * t_c)


def afd(m: MoE, hw: HW, n_a: int, n_f: int, ctx: int, n_mb: int = 3, b: float | None = None) -> dict:
    """AFD 部署的一步 decode。b = 每张注意力卡的序列数, 默认塞满显存。返回各项时间 (秒) 与吞吐。"""
    b_max = (hw.mem - m.layers * m.attn_params) / (ctx * m.kv_bytes * m.layers)   # 注意力卡: 显存全给 KV
    b = b_max if b is None else min(b, b_max)
    fits = m.layers * m.experts * m.expert_params / n_f <= hw.mem                  # FFN 卡放得下全部专家吗
    tokens = n_a * b / n_mb                                                 # 一个 micro-batch 的 token 总数
    t_a, t_f = attn_time(m, hw, b / n_mb, ctx), ffn_time(m, hw, tokens, n_f)
    t_c = comm_time(m, hw, b / n_mb, tokens, n_f)
    tpot = m.layers * layer_cycle(t_a, t_f, t_c, n_mb)
    return dict(b=b, t_a=t_a, t_f=t_f, t_c=t_c, tpot=tpot, tgs=n_a * b / tpot / (n_a + n_f), fits=fits,
                hidden=n_mb * max(t_a, t_f) >= max(t_a + t_f + 2 * t_c, n_mb * t_c))


def colocated(m: MoE, hw: HW, n: int, ctx: int, n_mb: int = 2, b: float | None = None) -> dict:
    """同卡部署 (大 EP): n 张卡, 每卡放 1/n 的专家 + 全部注意力权重, 剩下的显存给 KV。

    双 micro-batch 重叠 (DeepSeek 式): 一份在算 (注意力 + 专家), 另一份在做 all-to-all, 一层取两者中慢的。
    """
    weights = m.layers * (m.attn_params + m.experts * m.expert_params / n)
    b_max = max(hw.mem - weights, 0) / (ctx * m.kv_bytes * m.layers)
    b = b_max if b is None else min(b, b_max)
    compute = attn_time(m, hw, b / n_mb, ctx) + ffn_time(m, hw, n * b / n_mb, n)
    t_c = comm_time(m, hw, b / n_mb, n * b / n_mb, n)                       # dispatch 与 combine 各一次
    tpot = m.layers * n_mb * max(compute, 2 * t_c)
    return dict(b=b, tpot=tpot, tgs=b / tpot if b > 0 else 0.0)


def with_sla(fn, sla: float, *args, **kw) -> dict:
    """在 TPOT ≤ sla 的约束下, 把每卡 batch 从满显存往下二分, 取满足 SLA 的最大 batch。"""
    r = fn(*args, **kw)
    if r["tpot"] <= sla:
        return r
    lo, hi = 0.0, r["b"]
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(*args, b=mid, **kw)["tpot"] <= sla else (lo, mid)
    return fn(*args, b=lo, **kw)


FINE = MoE("细粒度专家 (256 选 8)")                                         # DeepSeek-V3 式
COARSE = replace(FINE, name="粗粒度专家 (48 选 3)", experts=48, top_k=3, expert_params=3 * 7168 * 5120)   # Step-3 式
