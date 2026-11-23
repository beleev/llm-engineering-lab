"""
容错训练的几件原语: 集群 MTBF、最优 checkpoint 间隔、goodput 模拟、慢节点、SDC、弹性缩容。

是什么: 千卡任务里 "挂掉" 不是异常而是常态, 这个文件把 "多久存一次盘" 从拍脑袋变成算出来的。
解决的瓶颈: 不是显存也不是算力, 是有效训练时间占比 (goodput) ——
            存得太勤, 时间花在落盘上; 存得太稀, 每次挂掉要重跑很长一段。
关键公式: 整体 MTBF  M_sys = M_gpu / N                 N 张卡各自独立, 故障率相加
          Young 1974  tau = sqrt(2 · delta · M)         delta = 一次落盘阻塞多久
          Daly 2006   tau = sqrt(2 · delta · (M + R)) - delta    R = 重启恢复耗时
          goodput     有效训练时间 / 墙钟时间
简化: 故障按泊松过程到达 (单一指数分布), 真实集群有批次效应和相关故障。
简化: 异步 checkpoint 只建模 "阻塞时间变短", 没有建模后台落盘还没写完就又挂了的情况。
读代码盯住: `simulate_goodput` 里挂掉之后 progress 归零 —— 丢的是上一个 checkpoint 之后的全部进度,
            包括落盘过程中挂掉的那次 (checkpoint 没写完就不算存上)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from llm_train.core import flatten_tree

ArrayDict = Dict[str, np.ndarray]           # 参数树 / 梯度树: 名字 → 数组


# ---- 1) 卡越多, 挂得越勤 ----------------------------------------------- #

def cluster_mtbf(gpu_mtbf: float, n_gpu: int) -> float:
    """N 张独立的卡, 任意一张挂就算整个任务挂: 整体 MTBF = 单卡 MTBF / N。

    推导: 单卡故障率 1/M_gpu, N 张卡的故障率相加得 N/M_gpu, 倒过来就是 M_gpu/N。
    """
    return gpu_mtbf / n_gpu


# ---- 2) 最优 checkpoint 间隔 ------------------------------------------- #

def young_interval(dump: float, mtbf: float) -> float:
    """Young 1974 的一阶近似。dump = 一次落盘阻塞的时间, mtbf = 整体 MTBF, 单位一致即可。"""
    return math.sqrt(2.0 * dump * mtbf)


def daly_interval(dump: float, mtbf: float, restart: float = 0.0) -> float:
    """Daly 2006 的一阶式: 把重启耗时 R 算进 MTBF, 再减掉落盘本身占的那一段。

    delta >= 2(M+R) 时这个式子给出的间隔没有意义 (落盘比平均无故障时间还长),
    此时每个 checkpoint 之间只放一步, 也就是能存多稀就多稀。
    """
    if dump >= 2.0 * (mtbf + restart):
        return float("inf")
    return math.sqrt(2.0 * dump * (mtbf + restart)) - dump


def simulate_goodput(interval: float, dump: float, mtbf: float, restart: float,
                     horizon: float, rng: np.random.RandomState, trials: int = 200) -> float:
    """蒙特卡洛: 要完成 horizon 秒的有效训练, 平均要花多少墙钟时间, goodput 是多少。

    一轮的流程: 跑 interval 秒有效工作 → 花 dump 秒落盘 → 重复, 直到有效工作攒够 horizon。
    故障按 Exp(1/mtbf) 到达; 挂了就付 restart 秒, 然后从上一个 checkpoint 重跑。
    """
    walls = []
    for _ in range(trials):
        done, wall = 0.0, 0.0
        while done < horizon:
            segment = min(interval, horizon - done) + dump      # 这一轮要连续跑完的时长
            progress = 0.0                                      # 本轮已经跑掉的部分
            while True:
                to_fail = rng.exponential(mtbf)
                if to_fail >= segment - progress:               # 这一轮撑到了落盘结束
                    wall += segment - progress
                    done += segment - dump
                    break
                wall += to_fail + restart                       # 挂在半路: 本轮进度全丢
                progress = 0.0
        walls.append(wall)
    return horizon / float(np.mean(walls))


def goodput_model(interval: float, dump: float, mtbf: float, restart: float) -> float:
    """同一件事的解析解, 不用抽样: goodput = tau / (M · e^(R/M) · (e^((tau+delta)/M) - 1))。

    读法: 分母是 "为了保住 tau 秒有效进度, 平均要花多少墙钟时间" ——
    e^((tau+delta)/M) - 1 是这一段里期望重试的次数, e^(R/M) 把每次重启的代价摊进去。
    与 simulate_goodput 在四组参数上差都在 0.3 个百分点内 (demo 里有断言)。
    """
    return interval / (mtbf * math.exp(restart / mtbf) * (math.exp((interval + dump) / mtbf) - 1.0))


def sweep_interval(intervals: List[float], dump: float, mtbf: float, restart: float,
                   horizon: float, seed: int = 0, trials: int = 200) -> List[Tuple[float, float]]:
    """扫一串 checkpoint 间隔, 每个都重开一个随机源, 返回 [(间隔, goodput), ...]。"""
    return [(i, simulate_goodput(i, dump, mtbf, restart, horizon,
                                 np.random.RandomState(seed), trials)) for i in intervals]


# ---- 3) 慢节点: 同步训练的每步时间是 N 个里的最大值 ---------------------- #

def straggler_step_time(n_rank: int, rng: np.random.RandomState, base: float = 1.0,
                        tail: float = 0.1, samples: int = 4000) -> Dict[str, float]:
    """每个 rank 的单步耗时 = base + Exp(tail); 同步训练一步要等最慢的那个。

    返回单 rank 的平均耗时、N 个取最大之后的平均耗时, 以及被慢节点拖掉的比例。
    """
    times = base + rng.exponential(tail, size=(samples, n_rank))    # [samples, n_rank]
    one = float(times[:, 0].mean())
    worst = float(times.max(axis=1).mean())
    return {"one": one, "step": worst, "loss": 1.0 - one / worst}


# ---- 4) 静默数据损坏 (SDC): 副本间对校验和, 再确定性重算 ----------------- #

def grad_checksum(tree: ArrayDict) -> float:
    """梯度的校验和。按 key 顺序拼平再用 float64 求和, 同一份输入每次给同一个数。"""
    return float(np.sum(flatten_tree(tree).astype(np.float64)))


def flip_bit(x: np.ndarray, index: int, bit: int) -> np.ndarray:
    """把 x 拍平后第 index 个 float32 的第 bit 位翻过来, 模拟一次位翻转。

    bit 从 0 (尾数最低位) 到 31 (符号位)。bit 越高, 数值变化越大, 也越容易被发现。
    """
    out = np.array(x, dtype=np.float32, copy=True)
    raw = out.reshape(-1).view(np.uint32)
    raw[index] ^= np.uint32(1) << np.uint32(bit)
    return out


def sdc_report(clean: ArrayDict, dirty: ArrayDict, tol: float) -> Dict[str, object]:
    """两个副本的梯度对比: 逐位是否相同、校验和差多少、在阈值 tol 下能不能发现。"""
    a, b = flatten_tree(clean).astype(np.float64), flatten_tree(dirty).astype(np.float64)
    gap = float(np.max(np.abs(a - b)))
    return {
        "bitwise": bool(np.array_equal(flatten_tree(clean), flatten_tree(dirty))),
        "checksum_gap": abs(grad_checksum(clean) - grad_checksum(dirty)),
        "max_gap": gap,
        "caught": gap > tol,
    }


# ---- 5) 弹性缩容: 少几张卡, 重切数据和优化器状态 ------------------------ #

def shard_sizes(total: int, n_rank: int, micro: int) -> List[int]:
    """把 total 个样本分给 n_rank 张卡, 每张卡再拆成若干个 micro 大小的 micro-batch。

    要求能整除: 全局 batch 不变是等价性的前提, 除不开就得改 micro 或者丢样本。
    """
    assert total % (n_rank * micro) == 0, f"{total} 个样本分不均给 {n_rank} 张卡 × {micro}"
    per_rank = total // n_rank
    return [micro] * (per_rank // micro) * n_rank


def accumulate_grads(model, x: np.ndarray, y: np.ndarray, n_rank: int, micro: int) -> ArrayDict:
    """n_rank 张卡、每卡若干个 micro-batch 的梯度累积 + 跨卡平均, 得到全局平均梯度。

    各份样本数相同, 所以 "先各自取均值再平均" 等于 "整批取均值" (m01 的加权规则在这里退化掉)。
    """
    total = x.shape[0]
    shard_sizes(total, n_rank, micro)                            # 只为检查整除关系
    per_rank = total // n_rank
    rank_grads = []
    for r in range(n_rank):
        xs, ys = x[r * per_rank:(r + 1) * per_rank], y[r * per_rank:(r + 1) * per_rank]
        acc = None
        n_micro = per_rank // micro
        for i in range(n_micro):
            _, g = model.loss_and_grads(xs[i * micro:(i + 1) * micro], ys[i * micro:(i + 1) * micro])
            acc = {k: v / n_micro for k, v in g.items()} if acc is None else \
                {k: acc[k] + v / n_micro for k, v in g.items()}
        rank_grads.append(acc)
    return {k: sum(g[k] for g in rank_grads) / n_rank for k in rank_grads[0]}
