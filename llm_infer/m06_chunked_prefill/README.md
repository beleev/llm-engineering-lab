# M06 — Chunked Prefill: 把长 prompt 切块, 和 decode 混在一个 batch 里

## 直觉
prefill 是算力瓶颈 (一次吃几千 token), decode 是访存瓶颈 (每步每条 1 token)。prefill 优先的调度器来一条长 prompt,
所有正在 decode 的用户就得停下来等它整段算完 —— 表现为输出"卡一下"。
这里看两个指标: TBT 是同一条请求相邻两个输出 token 的间隔, "卡一下" 就是 TBT 出现尖刺; TTFT 是请求到达到吐出第一个 token 的时间。
Sarathi 的做法: 每步固定 token 预算 B, 先给每条 decode 1 个 token, 剩下的预算切一块 prefill 塞进同一个 batch。

## 核心数据结构或公式
```
每步 batch = [decode × k 条 (各 1 token)] + [prefill chunk (≤ B - k token)]
step_ms ≈ fixed_ms + per_token_ms × batch_tokens        → TBT 上限 = step_ms(B)
分块等价性: 第 j 块的 Q (chunk, D) 对 K[0 : 已缓存+chunk] 做因果 attention, 前面 token 的 KV 不依赖后面 → 结果与整段相同
attention 分数矩阵峰值: T×T → chunk×T
```
实现: `chunked_prefill()` 就是对 `TinyLM.forward(ids_chunk, kv)` 的循环; 混批调度是 m03 `Scheduler(chunked_prefill=True)` 本身。

## 运行后应该看到什么
```bash
python -m llm_infer.m06_chunked_prefill.demo
```
```
[1] 分块 vs 整段 prefill 的数值差 (T=64)
   chunk   logits max|Δ|     KV max|Δ|         attn 分数矩阵峰值
       8        1.67e-06      2.86e-06         512 (整段 4096)
      16        0.00e+00      0.00e+00        1024 (整段 4096)
      32        0.00e+00      0.00e+00        2048 (整段 4096)
      64        0.00e+00      0.00e+00        4096 (整段 4096)

[2] 负载: 4 条请求在 decode, 第 10 步来了一条 1024-token 的长 prompt
  代价模型 (非实测)                       = step_ms = 20.0 + 0.25 × batch_tokens
  prefill 优先 (不分块)       每步最大 token= 1024  TBT p50=  21.0ms  max= 297.2ms  长请求 TTFT= 276.0ms  总耗时=1133ms
  分块混批 B=128             每步最大 token=  128  TBT p50=  21.0ms  max=  52.0ms  长请求 TTFT= 445.0ms  总耗时=1113ms
  分块混批 B=512             每步最大 token=  512  TBT p50=  21.0ms  max= 148.0ms  长请求 TTFT= 319.0ms  总耗时=1113ms

  decode 用户最大卡顿: 297ms → 52ms (5.7x); 代价: 长请求 TTFT 276ms → 445ms
```
- [1] 只有 chunk=8 有 ~1e-6 的差, 来自 fp32 matmul 分块求和的顺序; 其余为 0。
- [2] 的毫秒数都来自**代价模型**, 不是实测。B 越小, decode 的 TBT 尖刺越低, 长请求自己的 TTFT 越长。

## 与真实系统的差距
- 延迟来自线性代价模型; 真实 prefill 还有 attention 的 O(T²) 项, 且每个 chunk 要重读前面所有 KV (chunk 越小总开销越大)
- vLLM V1 默认开启, 旋钮是 `max_num_batched_tokens`; SGLang 是 `chunked_prefill_size`; 混批依赖 varlen attention kernel
- 分块与 P/D 分离 (m15) 是同一问题的两种解法: 一个在时间上错开, 一个在空间上隔离

## 常见误区
- "分块是为了省显存" —— 激活峰值确实降了, 但主要目的是**压 TBT 尖刺**
- "分块白赚" —— 长请求自己的 TTFT 变长 (276 → 445 ms), B 是 TBT 与 TTFT 之间的旋钮
- "chunk 之间要重算前面的 token" —— 不用, 前面 chunk 的 KV 已在 cache 里, 只是 Q 分批算

## 自测题
1. B=256, 当前 10 条在 decode, 这一步 prefill chunk 最多几个 token? **答**: 246。
2. 为什么分块 prefill 的结果与整段完全相同? **答**: 因果 mask 下 token i 的输出只依赖 0..i; 分块只是把 Q 的行分批算, K/V 仍是完整前缀。
3. 总耗时为什么分块反而略短 (1113 vs 1133 ms)? **答**: 代价模型里每步有固定开销, 混批把 prefill 搭进本来就要跑的 decode 步, 少了独立的 prefill 步; 真实系统里小 chunk 重读 KV 的开销会抵消一部分。
