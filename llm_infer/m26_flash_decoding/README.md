# M26 — Flash-Decoding: decode 只有 1 个 query, 就沿 KV 长度切开并行

## 直觉
FlashAttention (m11) 的并行单位是 (batch, head, Q 块)。prefill 时 Q 有几千行, 切出来的块多得是;
decode 时每条序列只有 **1 个 query**, Q 块只有一个, 并行单位只剩 B×H。
B=1、H=32 的长对话: 32 个 thread block 跑在 108 个 SM 上, **70% 的 SM 在闲着**,
而每个 block 还得独自把 32k token 的 KV 从头读到尾。

Flash-Decoding 的治法: Q 切不动, 就切 KV。把长度 T 切成 S 段, 每段一个 block 算局部 softmax,
并行单位变成 B×H×S; 最后一个小 kernel 用 log-sum-exp 把 S 段合起来。
合并公式就是 m11 的 `merge_attention`, 只是一次合 S 段 —— 所以结果与普通 attention **严格相等**。

## 核心数据结构或公式
`flash_decoding.py:partial_attention` —— 一段 KV 上的局部结果 (已归一化的 `O_s` 与分母的 log `lse_s`):
```
S_s = q·K_sᵀ/√d        lse_s = logsumexp(S_s)        O_s = softmax(S_s)·V_s
```
`flash_decoding.py:flash_decoding` —— 两步:
```
① split: 对 s = 1..S 并行 → (O_s, lse_s)                 # 各段互不依赖, GPU 上是 S 个 block
② reduce: lse = log Σ_s exp(lse_s)
          O   = Σ_s exp(lse_s − lse) · O_s               # 每段按它占的 softmax 质量加权
```
`flash_decoding.py:decode_latency_us` —— **估算**模型, 不是实测:
```
units = B·H·S                      waves = ⌈units / N_SM⌉
t     = launch + waves · (T/S) · KV_bytes_per_token / (HBM_bw / N_SM)
      + [S>1] (launch + units·(d+1)·4 B / HBM_bw)         # reduce kernel: 读回 S 份 O_s(fp32) 与 lse
加速比上限 = N_SM / (B·H)  (B·H < N_SM 时; 最多把闲着的 SM 全用上)
```
参数: A100, 108 SM, 2000 GB/s, 每个 SM 最多拿到 1/108 的带宽; head_dim 128, fp16 → 512 B/token/head; 每个 kernel 3 μs 固定开销。

## 运行后应该看到什么
```bash
python -m llm_infer.m26_flash_decoding.demo     # < 1 s
```
```
[1] 13 组 (B,H,T,S) 含 S 不整除 T、S=T: 最坏误差 2.06e-07   (O 对 core.dense_attention, lse 对朴素 logsumexp)
[2] B=1, H=32, T=32,768 (估算)
     S  units  waves  SM 利用率   延迟
     1     32      1       30%   909.0 μs
     2     64      1       59%   459.0 μs
     4    128      2       59%   459.0 μs      ← 128 个 block 要跑 2 波, 比 S=2 没快
    16    512      5       95%   289.2 μs
    64   2048     19      100%   275.5 μs      最优 3.30x (上限 108/32 = 3.38x)
[3] 最优 S 相对 S=1 的加速比 (估算, H=32)
     B        T=512        T=4,096       T=32,768      T=131,072
     1   1.63x (S=16)   2.90x (S=64)   3.30x (S=64)   3.35x (S=64)
     4   1.34x (S=16)   1.63x (S=16)   1.68x (S=16)   1.68x (S=16)
    16   1.00x (S=4)    1.05x (S=4)    1.05x (S=4)    1.05x (S=128)
    64   1.00x (S=1)    1.00x (S=1)    1.00x (S=32)   1.00x (S=32)
```
断言: 数值误差 < 1e-5; S=1 时 SM 利用率 < 30%; B=1 长上下文加速 > 2.5x 且不超过 N_SM/(B·H);
B=1 时加速比随 T 单调不减, T=512 的加速 < 0.6×T=128k 的; 每个 T 上 B=4 的加速都小于 B=1; B=64 全部 < 1.1x。

## 与真实系统的差距 (诚实边界)
- **[2][3] 的所有延迟都是代价模型算出来的**, numpy 里 split 循环是串行的, 反而更慢。
- 模型假设每个 SM 最多拿到 1/N_SM 的 HBM 带宽。真卡上单个 SM 能拿到的带宽比这高, 少数 SM 也能吃掉一部分带宽,
  所以真实加速比低于这里的数。Flash-Decoding 原文 (2023 博客) 报告超长序列上最高约 8x, 基线和硬件都不同, 不能与这里的数直接对照。
- 没有模拟 wave 的尾部效应、L2 cache、reduce 与主 kernel 的重叠; 真实实现 (FlashInfer / flash-attn) 用启发式选 S,
  也会把 reduce 融进主 kernel 的最后一个 block。
- 这里 H=32 是 MHA。GQA (如 8 个 KV head) 下多个 query head 共读一份 KV, 并行单位更少, split-K 更有用。
- 只讲 decode attention 这一个算子; 整步 decode 还包括 GEMM 读权重, 那部分不受影响, 所以端到端加速远小于 attention 本身的加速。

## 常见误区
- "Flash-Decoding 是近似" —— 不是, lse 加权合并与普通 softmax 数学相等 (最坏误差 2e-7)。
- "S 越大越好" —— S=4 与 S=2 一样慢 (128 个 block 跑 2 波); S 太大 reduce 开销上升。要让 B·H·S 接近 N_SM 的整数倍。
- "所有 decode 都该开 split-K" —— B·H ≥ N_SM 时 SM 已经满了, B=64 加速 1.00x, 多一个 kernel 反而亏。
- "局部结果直接平均就行" —— 每段的 softmax 分母不同, 必须按 exp(lse_s − lse) 加权 (m11 demo [4] 的反例)。

## 自测题
1. B=2, H=8 (GQA 后的 KV head 数), 108 个 SM。按本模型, split-K 的加速比上限是多少?
   **答**: N_SM/(B·H) = 108/16 = 6.75x。并行单位越少, split-K 越有用。
2. 为什么 reduce 只需要每段的 O_s 和 lse_s, 而不需要每段的全部分数?
   **答**: 全局 softmax = 各段分母之和; lse_s 就是第 s 段分母的 log, O_s 是该段归一化后的输出, 两者足以重建全局加权和。
3. vLLM PagedAttention V2 按 512 token 切 partition。对 T=300 的请求它会做什么?
   **答**: 只有一个 partition, 等价于不切 (V1); 短序列上 split 只会多一个 reduce 的开销。
