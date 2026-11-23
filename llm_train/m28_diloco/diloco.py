"""
DiLoCo: 本地多步 + 外层动量, 把同步次数降到 1/H。

是什么: 每个 worker 用自己的数据连着跑 H 步内层优化器, 然后只同步一次参数;
        外层把 "旧参数 - 新的平均参数" 当成一个梯度 (伪梯度), 用带 Nesterov 动量的 SGD 走一步。
解决的瓶颈: DDP 每步都要 all-reduce 一遍梯度, 跨数据中心的带宽和延迟撑不住。
关键公式: 伪梯度  Delta_t = theta_{t-1} - mean_i(theta_i^{(t)})     旧参数减新参数, 方向与梯度一致
          外层    buf <- mu·buf + Delta;  Nesterov 取 Delta + mu·buf;  theta <- theta - lr_out·step
          同步次数 = 总步数 / H, 通信量同比例下降
退化情形: H = 1、内层用朴素 SGD、外层 lr = 1 且无动量时, 外层这一步正好还原成 DDP 的那一步。
简化: 单进程模拟, worker 之间不是真的异步; 不建模掉队 worker 和跨数据中心的延迟。
读代码盯住: `outer_step` 里 Nesterov 的 `delta + mu * buf` —— 外层的有效步长是 lr_out/(1-mu),
            H 很小时这个放大会把每一步都走过头, 所以论文里 H 取几百。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from llm_train.core import LinearModel, ToyDataStream, adam_update, average_grad_trees

ArrayDict = Dict[str, np.ndarray]

D_IN, D_OUT, BATCH = 16, 4, 8            # 玩具线性回归: y = x @ W + b, 16 维进 4 维出


def worker_streams(world: int, seed: int = 1000) -> List[ToyDataStream]:
    """world 个 worker 共用同一个 true_W, 各自从数据流的不同位置取样本。

    cursor 拉开 10 万是为了让各 worker 的样本不重叠 —— 这是 DiLoCo 的设定 (数据分片, 分布相同)。
    """
    out = []
    for r in range(world):
        stream = ToyDataStream(D_IN, D_OUT, BATCH, seed=seed)
        stream.cursor = r * 100_000
        out.append(stream)
    return out


def holdout(batches: int = 16, seed: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """留出集: 同一个 true_W, cursor 放在所有 worker 之后, 谁都没训过。"""
    stream = ToyDataStream(D_IN, D_OUT, BATCH, seed=seed)
    stream.cursor = 9_000_000
    xs, ys = zip(*(stream.next_batch() for _ in range(batches)))
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(params: ArrayDict, x: np.ndarray, y: np.ndarray) -> float:
    """留出集 MSE。全 0 预测的 MSE 约等于 y 的方差 (4.08), 可以当作坏到底的参照。"""
    return float(np.mean((x @ params["W"] + params["b"] - y) ** 2))


def inner_adam(model: LinearModel, stream: ToyDataStream, state: dict, step: int, lr: float) -> None:
    """worker 本地走一步 AdamW。state 里每个参数存 m、v 两份, 跨外层轮次不清空。"""
    x, y = stream.next_batch()
    _, grads = model.loss_and_grads(x, y)
    for name, value in model.params().items():
        adam_update(value, grads[name], state[name][0], state[name][1], step, lr)


def run_ddp(steps: int, lr: float, world: int = 4, seed: int = 5) -> ArrayDict:
    """基线: 每步 all-reduce 一次梯度, 再走一步 AdamW。"""
    model = LinearModel.init(D_IN, D_OUT, seed=seed)
    state = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in model.params().items()}
    streams = worker_streams(world)
    for step in range(1, steps + 1):
        local = []
        for stream in streams:
            x, y = stream.next_batch()
            local.append(model.loss_and_grads(x, y)[1])
        grads = average_grad_trees(local)                      # ★ 每步一次 all-reduce
        for name, value in model.params().items():
            adam_update(value, grads[name], state[name][0], state[name][1], step, lr)
    return {k: v.copy() for k, v in model.params().items()}


def outer_step(params: ArrayDict, avg: ArrayDict, buf: ArrayDict, lr_out: float,
               momentum: float, nesterov: bool = True) -> None:
    """外层 Nesterov 动量 SGD, 原地改 params。

    Nesterov 把动量再往前推一步 (delta + mu·buf), 所以外层的有效步长约为 lr_out/(1-mu)。
    """
    for name in params:
        delta = params[name] - avg[name]                       # ★ 伪梯度: 旧参数 - 新的平均参数
        buf[name][...] = momentum * buf[name] + delta
        update = delta + momentum * buf[name] if nesterov else buf[name]
        params[name] -= lr_out * update


def run_diloco(steps: int, inner_steps: int, lr_in: float, lr_out: float = 0.7,
               momentum: float = 0.9, nesterov: bool = True, world: int = 4, seed: int = 5,
               inner: str = "adam") -> ArrayDict:
    """DiLoCo: 每 inner_steps 步同步一次。inner="sgd" 时内层换成朴素 SGD, 用来做退化检查。"""
    assert steps % inner_steps == 0, "总步数要能被 H 整除, 否则最后一轮不完整"
    outer = LinearModel.init(D_IN, D_OUT, seed=seed)
    buf = {k: np.zeros_like(v) for k, v in outer.params().items()}
    streams = worker_streams(world)
    workers = [LinearModel.init(D_IN, D_OUT, seed=seed) for _ in range(world)]
    states = [{k: [np.zeros_like(v), np.zeros_like(v)] for k, v in outer.params().items()}
              for _ in range(world)]
    step = 0
    for _ in range(steps // inner_steps):
        for worker in workers:
            worker.load_params(outer.params())                 # 每轮从外层参数重新出发
        for _ in range(inner_steps):
            step += 1
            for r, worker in enumerate(workers):
                if inner == "adam":
                    inner_adam(worker, streams[r], states[r], step, lr_in)
                else:
                    x, y = streams[r].next_batch()
                    _, grads = worker.loss_and_grads(x, y)
                    worker.apply_grads(grads, lr_in)
        avg = average_grad_trees([w.params() for w in workers])   # ★ 每 H 步一次 all-reduce
        outer_step(outer.params(), avg, buf, lr_out, momentum, nesterov)
    return {k: v.copy() for k, v in outer.params().items()}


def run_streaming_diloco(steps: int, inner_steps: int, lr_in: float, groups: int = 2,
                         lr_out: float = 0.7, momentum: float = 0.9, world: int = 4,
                         seed: int = 5) -> ArrayDict:
    """Streaming DiLoCo: 参数分成 groups 份错峰同步, 每份仍然每 H 步同步一次。

    每 H/groups 步同步一份: 总通信量不变, 但单次同步只发 1/groups 的参数,
    峰值带宽同比例降下来, 而且同步和本地计算更容易叠在一起。
    """
    assert inner_steps % groups == 0, "H 要能被分片数整除, 每份才轮得到一次"
    assert steps % inner_steps == 0, "总步数要能被 H 整除"
    # 按输出维轮转分片: 第 g 份是 W 的第 g, g+G, ... 列和 b 的同样几个位置
    shards = [np.arange(g, D_OUT, groups) for g in range(groups)]
    outer = LinearModel.init(D_IN, D_OUT, seed=seed)
    buf = {k: np.zeros_like(v) for k, v in outer.params().items()}
    streams = worker_streams(world)
    workers = [LinearModel.init(D_IN, D_OUT, seed=seed) for _ in range(world)]
    states = [{k: [np.zeros_like(v), np.zeros_like(v)] for k, v in outer.params().items()}
              for _ in range(world)]
    step = 0
    for event in range(steps // (inner_steps // groups)):
        for _ in range(inner_steps // groups):                 # 两次同步之间只跑 H/G 步
            step += 1
            for r, worker in enumerate(workers):
                inner_adam(worker, streams[r], states[r], step, lr_in)
        cols = shards[event % groups]                          # ★ 本次只同步这一份参数
        avg = average_grad_trees([{"W": w.params()["W"][:, cols], "b": w.params()["b"][cols]}
                                  for w in workers])
        for name, index in [("W", (slice(None), cols)), ("b", (cols,))]:
            delta = outer.params()[name][index] - avg[name]
            buf[name][index] = momentum * buf[name][index] + delta
            outer.params()[name][index] -= lr_out * (delta + momentum * buf[name][index])
            for worker in workers:                             # 同步过的那份拉回外层的值
                worker.params()[name][index] = outer.params()[name][index]
    return {k: v.copy() for k, v in outer.params().items()}
