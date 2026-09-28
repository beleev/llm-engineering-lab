# M09 — 通信原语

运行: `python -m llm_train.m09_collectives.demo`

## 直觉
N 张卡各拿一块数据。四个原语只在两件事上不同:
- 要不要把各卡的数据加起来;
- 每张卡最后拿回整份, 还是只拿一片。

逐个看:
- **all-reduce**: 加。每张卡拿回整份总和。
- **reduce-scatter**: 加。每张卡只拿总和的一片。
- **all-gather**: 不加。把各卡手里的片拼成整份, 每张卡一份。
- **all-to-all**: 不加, 互换。卡 r 给卡 s 准备的那块, 最后到了卡 s 手里。

先 reduce-scatter 再 all-gather, 结果就是一次 all-reduce。

下表的 S 是一份完整张量的字节数, N 是卡数, rank 指一张卡。

| 原语 | 每 rank 输入 → 输出 | 谁在用 |
|---|---|---|
| all-reduce | S → S (总和) | DDP 梯度, TP 激活 |
| reduce-scatter | S → S/N (总和的第 r 片) | ZeRO-2/3 梯度 |
| all-gather | S/N → S | ZeRO-3/FSDP 参数 |
| all-to-all | N 块 → N 块 (转置) | MoE 路由, Ulysses |

## 核心公式 (ring, 每 rank 发送字节)
all-reduce `2(N-1)/N·S` = reduce-scatter `(N-1)/N·S` + all-gather `(N-1)/N·S`。
朴素 "reduce 到 rank 0 再 broadcast": rank 0 吞吐 `2(N-1)·S`, 随 N 线性增长。

## 运行后应该看到什么 (N=4, S=32B)
```
all_reduce = 48 B   reduce_scatter = 24 B   all_gather = 24 B   ring_all_reduce = 48 B   all_to_all = 12 B
朴素 reduce→broadcast 的 rank 0 = 192 B  (4x, 且随 N 线性增长)
all_to_all: rank 1 发出 / 收到 = [10, 11, 12, 13] / [1, 11, 21, 31]
```
`ring_all_reduce_sum` 真的一步步传 chunk, 断言它实测的字节数 == 公式, 结果 == 直接求和。

## core 里的通信计数器
所有原语都往 `llm_train.core.comm` 记账 (`comm.reset()` / `comm.total` / `comm.summary()`), m02/m03/m05/m11/m12/m16/full_loop 的通信量都来自它。

## 与真实系统的差距
- 只数字节 (带宽项), 不数延迟项: ring 要 2(N-1) 步, 小消息/大 N 时 NCCL 会换 tree 或 NVLS。
- 没有拓扑: 机内 NVLink 900GB/s vs 机间 IB 50GB/s 差一个数量级, 这决定了 TP 放机内、DP/PP 放机间。
- 没有通信-计算重叠, 而这是真实系统里隐藏通信时间的主要手段。

## 常见误区
- "卡越多 all-reduce 越慢" —— 带宽项几乎不变, 变的是延迟项。
- "ZeRO 通信一定比 DDP 多" —— 分 stage 看:
  - ZeRO-2 与 DDP 相同: reduce-scatter + all-gather 正好是一次 all-reduce。
  - ZeRO-1 看实现: DeepSpeed 用 reduce-scatter, 与 DDP 持平; 本库 m05 用 all-reduce + all-gather, 是 1.5× (864 B vs 576 B)。
  - ZeRO-3 也是 1.5×。

## 自测题
1. N=8 的 ring all-reduce 要几步? **答: 2(N-1) = 14 步。**
2. all-to-all 为什么对角块不计通信? **答: 发给自己的数据留在本地。**
3. reduce-scatter 后再 all-gather 得到什么? **答: 与 all-reduce 完全相同的结果。**
