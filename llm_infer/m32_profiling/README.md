# 性能实测：模型估算与 profiler 怎样对照

## 直觉

FLOPs 相同不代表执行时间相同。真正的算子调用、分配与 host 开销需要测出来。

## 核心原理

先 warmup，再独立计时，再采 profiler。记录 Python/PyTorch、设备、线程数和重复次数；网页展示稳定事件计数，机器相关耗时在 CLI 的 JSON 中保留。

正向分配字节之和不等于峰值内存，ATen 调用也不等于 GPU kernel 数。self 时间排除了子调用，不能把嵌套事件时间或异步 CPU/GPU 时间直接相加。

CUDA 选做在计时前后 synchronize，并报告峰值 allocated；异步拷贝还需 pinned memory、正确 stream 和依赖管理。CPU 分支验证测量流程，不推断 GPU 加速比。

## 运行

```bash
python -m llm_infer.m32_profiling.demo
```

需要 Python ≥ 3.10；使用现有 torch 依赖。默认 CPU、本机夹具，无模型 API 调用。

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

CPU profiler 示例记录 warmup、重复测量和环境，解释总时间与算子时间为何不同。；GPU 选做再讲同步、异步拷贝、峰值显存及 kernel trace，不拿 CPU 或模拟结果宣称 GPU 加速比。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

两版 FLOPs 相同，能否仅据此认定耗时相同？

<details><summary>展开解析</summary>

CUDA 选做在计时前后 synchronize，并报告峰值 allocated；异步拷贝还需 pinned memory、正确 stream 和依赖管理。CPU 分支验证测量流程，不推断 GPU 加速比。

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
