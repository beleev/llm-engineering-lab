# m22 · 把 DP、TP、PP 放到同一张预算表

## 直觉

把参数切到更多卡上会省显存，却可能多出大量小消息。只算参数能不能装下，回答不了训练会不会更快。

## 核心原理

rank 编号为 `(dp_index * PP + pp_index) * TP + tp_index`；DP×TP×PP 必须等于 world，hidden/TP、layers/PP、batch/(DP×micro) 都要整除。固定 1B 参数、24 层、hidden=2048、序列 1024、全局 batch=64、micro=4 和 8 卡。

每卡训练状态为 `16P/(TP×PP)` 字节：BF16 参数/梯度各 2，FP32 主权重及 Adam m/v 各 4。不用 ZeRO；DP 副本没有分摊状态。激活采用简化的 `2×S×H×micro×(L/PP)×在途微批数/TP`，不含所有临时张量。

通信 = 消息数 × 启动延迟 α + 传输字节/带宽 β。DP 用 ring all-reduce，TP 每层前后向合计四次 collective，PP 用相邻阶段发送。这里只有 DP 梯度通信可与计算重叠；TP/PP 是依赖项，不能全部藏起来。

计算/访存时间取 `max(FLOP/峰值算力, bytes/HBM带宽)`；算术强度为 FLOP/byte，Roofline 上限为 `min(峰值算力, 强度×HBM带宽)`。再加未隐藏通信与 `(1+(PP-1)/微批数)` 气泡倍率。MFU 是有效模型 FLOP / (时间×设备峰值)，这里是估算 MFU，不是 profiler 测量值。

## 运行

```bash
python -m llm_train.m22_parallel_budget.demo
```

## 运行后应该看到什么

默认 DP2×TP2×PP2 每卡状态约 3.725 GiB、步时间约 708 ms；DP1×TP4×PP2 状态约 1.863 GiB，却需要约 975 ms。TP 消息数增加抵消了内存节省。把 α 从 8 μs 改到 100 μs 更慢。DP2×TP3×PP2 被拒绝，不能为非法配置编出收益。

## 边界与自测

没有启动真实多进程，也没有拓扑争用、NVLink/跨节点区别、算子效率、重算和 allocator 余量。这张表用于淘汰明显不可行配置，真正容量需实测峰值显存、时间和通信轨迹。为什么同样总卡数，TP 越大反而越慢？把本地计算与必须等待的通信拆开解释。
