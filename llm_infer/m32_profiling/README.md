# 性能实测: 模型估算与 profiler 怎样对照

## 直觉

这组实验用两种写法算同一个线性层:
- x@w+b 先做矩阵乘再加偏置 bias。
- torch.addmm 用一个函数把两步一起做。

按公式数, 两种写法的逻辑 FLOPs (浮点乘加运算的次数) 都是 1056768, 纸面上一模一样。但 PyTorch 自带的 profiler (记录每个底层算子调用和内存分配的工具) 看到的不一样:
- 两版各做 1 次矩阵乘。
- ATen 调用 (PyTorch 底层算子库 ATen 的函数调用次数) 是 6 对 7。
- CPU 上分配的内存是 65536 对 32768 字节, 先乘后加的写法多分配了一份中间结果。

FLOPs 相同不代表执行时间相同, 算子调用、内存分配和 CPU 一侧的调度开销 (host 开销) 都要实际测出来。

## 核心原理

测耗时分三步, 而且要分开做:
1. 空跑 3 次 (warmup), 让只在第一次调用时发生的初始化开销过去。
2. 独立计时 7 次, 报最小值、中位数和最大值。
3. 单独跑一次 profiler, 因为 profiler 记录事件本身会拖慢执行, 不能用它那次的时间当耗时。

同时记下 Python 和 PyTorch 版本、设备、线程数 (本例固定 1 个线程) 和重复次数, 不然换一台机器结果就没法比。耗时和机器强相关, 所以只留在命令行输出的 JSON 里; 网页只回放不随机器变化的事件计数。

读 profiler 的数字, 要先弄清它数的是什么:
- CPU 分配字节: 所有「申请内存」事件的大小加起来 (eager 65536, fused 32768), 不是同一时刻最多占了多少内存; 一个 128×64 的 float32 结果正好是 32768 字节, 先乘后加多出来的就是一份中间结果。
- ATen 调用数: 把所有 aten:: 开头的事件都算上, 包括一个函数内部再调用的其它 aten 函数, 它不等于 GPU 上实际启动的 kernel (显卡上执行的一段计算程序) 个数; fused 版本反而多 1 次 (6 对 7), 也说明调用次数不能直接拿来比快慢。
- self 时间: profiler 报的 self 时间只算这个事件自己、不含它内部调用的子事件, 嵌套事件的时间、异步执行的 CPU 和 GPU 时间都不能直接相加。

选做的 CUDA 分支在计时前后都调用 synchronize (等 GPU 把已经排队的计算全部做完)。GPU 算子是异步的: CPU 把任务交出去就返回, 不等结果, 不同步的话计时器停下时 kernel 可能还没跑完。这个分支还会报告显存峰值 (峰值 allocated)。如果涉及 CPU 和 GPU 之间的异步拷贝, 还需要锁页内存 (pinned memory)、正确的 stream (GPU 上的一条执行队列) 和它们之间的依赖管理。本次验收只跑了 CPU 分支, 它只验证测量流程是对的, 不能推断 GPU 上会快多少。

## 运行

```bash
python -m llm_infer.m32_profiling.demo
```

需要 Python ≥ 3.10; 使用现有 torch 依赖。默认 CPU、本机夹具, 无模型 API 调用。

## 运行后应该看到什么

### matmul 后加 bias

| 读数 | 实际结果 |
| --- | --- |
| 矩阵乘调用 | 1 |
| ATen 调用 | 6 |
| 逻辑 FLOPs | 1056768 |
| CPU 分配字节 | 65536 |


### addmm

| 读数 | 实际结果 |
| --- | --- |
| 矩阵乘调用 | 1 |
| ATen 调用 | 7 |
| 逻辑 FLOPs | 1056768 |
| CPU 分配字节 | 32768 |


## 与真实系统的差距

CPU 分配字节是正向分配事件之和，不是峰值驻留内存。profiler 会扰动耗时，计时与采样分开；CPU self 时间排除子事件但不能直接与异步 GPU 总时间相加。CUDA 选做会同步后计时并记录峰值显存；本次默认只测 CPU。

## 怎样验证

固定同一组张量 (x 是 128×64, w 是 64×64, bias 有 64 个数)、单线程和随机种子, 比较「matmul 后加 bias」和「addmm」两种写法。判定看两件事:
- 两版输出在 1e-5 的容差内一致。
- 矩阵乘调用、ATen 调用、逻辑 FLOPs、CPU 分配字节四个计数是否和预期相同。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

两版 FLOPs 一样, 能不能就此认定耗时一样?

<details><summary>展开解析</summary>

不能。FLOPs 只数乘法和加法做了多少次, 不数函数调用了几次、内存分配了几次、数据在内存里搬了多少。本例两版 FLOPs 都是 1056768, 但先乘后加的写法要把矩阵乘的结果先存成一个中间张量, 再读出来加 bias, 于是多分配了 32768 字节; ATen 调用数也不同 (6 对 7)。这些开销在 FLOPs 公式里都看不到。该怎么测:
- 要比耗时, 就先 warmup 再独立计时多次。
- 想知道差在哪, 再用 profiler 看每个算子。

</details>

## 参考资料

- [PyTorch Profiler 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Stanford CS336 2025：Systems 作业包含 profiling 与 benchmark](https://cs336.stanford.edu/spring2025/)

## 本机计时与选做 GPU

```bash
python -m llm_infer.m32_profiling.demo --trace /tmp/llm-profile
python -m llm_infer.m32_profiling.demo --device cuda
```

第一条输出 eager/fused 各自的环境、warmup、重复次数、min/median/max 毫秒与 profiler 数据，并生成两份 Chrome trace。第二条仅在 CUDA 可用时执行；CPU-only 环境明确报错。本次验收没有 GPU 实测。
