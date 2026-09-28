# M27 — 多副本路由: 把请求送到"缓存里已有它前缀"的副本, 但别把一个副本压垮

## 直觉
一个模型起 8 个引擎副本, 每个副本有自己的前缀缓存 (m04 / m05), **副本之间不共享 KV**。
多轮对话第 k 轮的 prompt = system prompt + 前 k−1 轮全部历史。轮询和最少负载不看内容,
下一轮多半落到别的副本, 那里只有 system prompt, 整段历史得重新 prefill。

前缀感知路由问每个副本"你缓存里有这个请求多长的前缀", 送给最长的那个。命中率一下子上去了,
但新问题来了: 所有新对话都带着同一个热门 system prompt, 只有最先缓存它的副本能命中 → 全挤过去,
其它副本闲着, 热点副本队列越排越长。治法: **负载阈值兜底** —— 最佳副本比最闲副本积压多出阈值, 就放弃缓存走最少负载;
被分流过去的副本随后也缓存了热点前缀, 热点自然复制开。

## 核心数据结构或公式
- `router.py:Replica` —— 一个副本: `cache` (直接复用 `m05_radix_cache.radix_tree.RadixCache`)、`backlog_s` (未做完的活, 路由器眼里的负载)。
- `router.py:prefix_len` —— **只读**的最长前缀匹配。路由探测 8 个副本, 不能像 `RadixCache.match` 那样 split 树、刷新 LRU
  (探测 ≠ 使用, 否则没被选中的副本的 LRU 也被搅乱)。
- `router.py:route` —— 三个策略:
```
round_robin : k mod N
least_load  : argmin backlog              (平手轮转)
prefix      : best = argmax (prefix_len, −backlog)
              if backlog[best] − min backlog > threshold: 改走 least_load      ← 兜底
```
- `router.py:simulate` —— 逐请求: 时间推进消化积压 → 路由 → `cache.match` 算命中 → 记 TTFT → 积压 += 活 → 插入 prompt+回复 → 超容量就 `evict`。
```
命中 h = min(命中长度, len(prompt) − 1)          (至少算最后 1 个 token 才有 logits)
活     = (len(prompt) − h) / prefill_tok_s + len(reply) / decode_tok_s
TTFT   = 到达时积压 + (len(prompt) − h) / prefill_tok_s          ← 估算
负载均衡度 = max(各副本累计的活) / mean
```
- `router.py:chat_stream` —— 200 段对话 × 3~8 轮, 4 个 512-token system prompt, 流行度 ∝ 1/rank (头部 48%), 轮间思考 Exp(15 s)。

## 运行后应该看到什么
```bash
python -m llm_infer.m27_multi_replica_routing.demo     # ≈ 2.5 s
```
```
1084 个请求, prompt 共 1,133,527 tokens; 8 副本, 每个缓存 24,000 tokens; prefill 1000 tok/s, decode 100 tok/s (估算)
[1] policy                  命中率    miss  max/mean  TTFT 均值  TTFT p90 (秒, 估算)
    round_robin              59.3%   40.7%     1.03      0.507     1.090
    least_load               59.3%   40.7%     1.02      0.469     0.999
    prefix (最长前缀匹配)     93.9%    6.1%     4.18    105.954   300.605
[2] 各副本分到的活: 52% 32% 0% 0% 0% 16% 0% 0%    真正在干活的副本 3 / 8
[3] prefix, thr=4s           93.4%    6.6%     1.45      0.689     2.102
    prefix, thr=2s           90.5%    9.5%     1.15      0.461     1.486
    prefix, thr=1s           83.6%   16.4%     1.19      0.314     0.912
    prefix, thr=0.5s         77.9%   22.1%     1.12      0.284     0.823   ← TTFT 最优
    prefix, thr=0.2s         75.7%   24.3%     1.07      0.285     0.830
    prefix, thr=0.05s        72.9%   27.1%     1.12      0.308     0.908
```
断言: 前缀感知命中率比轮询/最少负载高 25 个百分点以上, miss 不到最少负载的 1/4; 纯前缀感知 max/mean > 3、
至多 4 个副本在干活、TTFT > 50× 最少负载; thr=1s 时 max/mean < 1.5、命中率比最少负载高 15 个百分点以上、TTFT < 0.8× 最少负载;
阈值越紧命中率越低、越均衡; TTFT 最优阈值落在扫描范围中间。

## 与真实系统的差距 (诚实边界)
- **TTFT 和负载都来自代价模型**: 每个副本是一条 FIFO 工作队列, prefill 与 decode 按固定速度串行消化。
  真实引擎做 continuous batching, decode 不会整段挡住后来请求的 prefill; 过载表现为 batch 变大、TPOT 变差、KV 显存打满后排队。
  纯前缀感知 "TTFT 106 s" 是队列模型在过载 (热点副本利用率 > 1) 时的发散, 只说明"会爆", 数值没有意义。
- 参数刻意取得偏慢 (decode 100 tok/s/副本), 让 8 个副本有 30~40% 的利用率, 排队效应才看得见; 利用率很低时三种策略的 TTFT 差别只剩 prefill 本身。
- 命中率是真实跑 `RadixCache` 得到的, 但"轮询 59%"主要来自 system prompt (512 token, 每个副本很快都缓存了); system prompt 越长, 轮询和前缀感知的差距越小。
- 路由器直接读每个副本的真实 radix tree。真实 router (SGLang sgl-router) 在路由器里维护**近似**树 (只记录发出去的请求, 不知道副本的驱逐),
  也有纯哈希方案 (对前 N 个 token 做一致性哈希): 无状态、便宜, 但若 N 只覆盖 system prompt, 热点前缀会被整桶送到一个副本, 与 [2] 同病。
- 负载 = 积压秒数。真实系统用排队请求数、运行中 token 数或 KV 使用率。
- 前缀缓存容量对所有策略相同 (24k token / 副本); 没有跨副本的 KV 迁移 (Mooncake 那种共享 KV 池能让"缓存在哪"与"在哪算"解耦)。

## 常见误区
- "前缀缓存命中率越高, TTFT 就越低" —— 纯前缀感知命中率最高 (93.9%), TTFT 最差; 命中省下的 prefill 抵不过排队。
- "最少负载 = 最均衡 = 最快" —— 均衡 (1.02) 但 41% 的 prompt token 在重算, TTFT 比 thr=1s 高 49%。
- "阈值越紧越好" —— thr=0.05s 已经在往最少负载退化 (命中率 72.9%, TTFT 回升到 0.308)。
- "路由时顺手用 cache.match 探测就行" —— match 会 split 树、刷新 LRU; 探测 8 个副本就把 8 棵树的 LRU 全搅了。用只读的 `prefix_len`。

## 自测题
1. 为什么纯前缀感知下, 4 个 system prompt 最多只让 4 个副本干活?
   **答**: 新对话在已缓存它 system prompt 的副本上命中 512 token, 在其它副本命中 0, 永远选前者; 后续轮次的历史也只在那里。
   每个 system prompt 只在它第一次出现时 (全体平手) 选过一次副本 —— 两个 prompt 还可能落到同一个副本 (本例 3/8)。
2. 加了阈值后, 为什么热点 system prompt 最终会出现在多个副本上?
   **答**: 过载时请求被分流到最闲副本, 该副本处理后插入缓存; 之后新对话在两个副本上都命中 512 token, 平手按负载选, 热点就被复制开了。
3. 把 system prompt 从 512 拉长到 4096 token, 轮询的命中率会怎样? 前缀感知的相对优势呢?
   **答**: 轮询命中率上升 (每个副本都缓存了长 system prompt, 占 prompt 的比重变大); 历史部分占比下降, 前缀感知的相对优势变小。
