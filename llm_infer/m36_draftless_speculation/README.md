# M36 — 不用草稿模型的投机: 输出照抄输入时, 草稿查表就有

[llm_infer 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/infer/tree-speculation)

## 直觉

m07 的投机解码要一个 draft 模型: 它要占显存、每轮要串行跑 K 次, 还得和 target 对齐。可很多负载的输出大段照抄输入: 代码编辑把整个文件改几处再吐回来, RAG 回答里夹着整段检索原文, 多轮对话复述上文。还有一些负载的输出和以前的输出高度相似, 比如 Agent 反复发同一类工具调用。
这时"下一段大概率是什么"直接在文本里查就知道。取已生成文本的最后几个 token, 在 prompt 里找到同样的片段, 把它后面那段抄过来当草稿, 再交给 target 按 m07 的规则一次验完。草稿的成本是一次 CPU 上的字符串匹配, 查不到就不提草稿, 退化成普通 decode。

## 核心原理

### 核心数据结构或公式

```
PromptLookupDrafter.propose(out, K)
    for n = max_n .. min_n:                       默认 3 .. 1
        tail = out[−n:]
        从后往前找 s, 使 out[s : s+n] == tail     最近一次出现
        找到 → 草稿 = out[s+n : s+n+K]
    都找不到 → 空草稿 (本轮就是普通 decode)

SuffixCacheDrafter (SuffixDecoding 的简化)
    计数表: n-gram (n ≤ 6) → 下一个 token 的出现次数, 内容 = 以前请求的输出 + 当前请求的全部文本
    p = 当前末尾能匹配上的最长 n-gram 长度
    草稿长度上限 = min(K, α·p), α = 2              匹配越长, 敢猜得越远 (论文的 MAX_SPEC = α·p)
    沿 "出现次数最多的下一个 token" 往下走, 经验概率 < 0.3 就停

验证: m07 的 speculative_decode, greedy 规则 accept_greedy
    target 一次 forward 验 [out[−1], d_0 .. d_{K−1}], 接受最长一致前缀, 再白送 1 个纠错或 bonus token
```

| 方法 | 草稿从哪来 | 本模块 |
|---|---|---|
| Prompt lookup decoding (2023) / vLLM 的 method="ngram" | 当前 prompt + 已生成文本里的 n-gram 匹配 | `PromptLookupDrafter` |
| REST (2023) | 外部检索库 (后缀数组), 取最长后缀匹配的若干续写组成 Trie, 树注意力一次验 | 未实现 |
| Lookahead decoding (2024) | 同一次 forward 里用 Jacobi 迭代生成 n-gram 池, 另一支验证池里的候选 | 未实现 |
| SuffixDecoding (2024) | 以前请求输出的全局后缀树 + 当前请求的后缀树, 按频率长出草稿树, 只占 CPU 内存 | `SuffixCacheDrafter` (链, 非树) |

target 是合成的 `OracleTarget`: 上下文与 prompt + 参考答案的前 j 个 token 一致时, 它的 greedy 下一个 token 就是参考答案的第 j 个。它只用来量接受率; 无损性另在真实 TinyLM 上验证。

## 运行

在仓库根目录执行：

```bash
python -m llm_infer.m36_draftless_speculation.demo
```

## 运行后应该看到什么

```bash
python -m llm_infer.m36_draftless_speculation.demo     # < 1 s
```
```
══════════════════════════════════════════════════════════════════════

[1] 无损: 真实 TinyLM (随机权重) 当 target
  输出 == target 逐 token greedy      = True
  target forward: 逐 token / 投机     = 64 / 52  (每轮接受 0.24 个, 草稿 182 个)

[2] 三类负载 (oracle target, K=8, 每类 5 条请求取平均; 逐 token 需要的 target forward = 输出长度)
  负载         查法                草稿    接受率  forward 逐token→投机  每次 forward 产出
  代码编辑       prompt lookup    1698   72.7%        1500 → 265           5.66x
  代码编辑       n-gram 缓存        1516   79.0%        1500 → 303           4.95x
  RAG 摘录     prompt lookup     960   45.2%         660 → 226           2.92x
  RAG 摘录     n-gram 缓存         686   60.3%         660 → 246           2.68x
  开放生成       prompt lookup    1178    0.0%         660 → 660           1.00x
  开放生成       n-gram 缓存         523    0.0%         660 → 660           1.00x
  开放生成: prompt lookup 白验的草稿 token  = 1178  (decode 带宽受限时几乎免费, 大 batch 算力受限时是浪费)

[3] 同一类请求反复出现 8 次: 输出是 40 token 的固定模板 (3 个参数位每次不同), 模板不在 prompt 里
  prompt lookup  第 1 次请求 forward 40, 之后平均  40.0   接受率 (第 2~8 次) 0.0%
  n-gram 缓存      第 1 次请求 forward 40, 之后平均  13.0   接受率 (第 2~8 次) 83.7%

  ✓ 草稿来自查表, 没有草稿模型: 照抄输入的负载每次 forward 产出 2.5 个以上, 开放生成退化成逐 token
  ✓ 无损: 所有负载上输出都与 target 的 greedy 输出逐 token 相同
```
断言:
- [1] 真实 TinyLM 当 target 时, prompt lookup 投机的输出与 target 逐 token greedy 完全相同; 草稿模型 forward 次数为 0。
- [2] 所有负载、两种查法, 输出都与 oracle 的参考答案逐 token 相同; 代码编辑和 RAG 摘录上 prompt lookup 的接受率 > 40%、每次 target forward 产出 > 2.5 个 token; 开放生成接受率 < 5%、产出 < 1.1。
- [3] 模板不在 prompt 里时 prompt lookup 的接受率 < 10%; 跨请求缓存从第 2 次起接受率 > 50%, 第 1 次的 forward 次数是后面的 2 倍以上。

**怎么读**: "每次 forward 产出"是输出 token 数除以 target forward 次数, 也就是按 target 调用数算的加速上限。decode 受带宽限制时, 一次 forward 多验 8 个位置几乎不加时间, 这个数接近真实加速; batch 大到算力受限时, 被拒的草稿位置 (开放生成里白验了 1178 个) 就是真实开销。
RAG 摘录的接受率 (45.2%) 比代码编辑低, 是因为每段原文抄完后 prompt lookup 还会接着猜原文的下文, 这 8 个草稿全被拒。

## 与真实系统的差距

- target 是事先写好参考答案的 oracle, 不是真实模型; 接受率完全由负载里"照抄"的比例决定, 这正是这类方法的性质, 但具体数字不能外推到真实模型和真实数据。
- 加速按 target forward 次数算, 没有计 CPU 查表的时间, 也没有计一次 forward 多验几个位置的额外耗时。
- 论文和项目的数字 (各自摘要或 README): prompt lookup decoding 在摘要、文档问答、代码编辑这类基于输入的任务上报告 2~4 倍; REST 在 7B / 13B 模型上 1.62~2.36 倍; Lookahead decoding 在 MT-bench 上最高 1.8 倍, 多卡代码补全最高 4 倍; SuffixDecoding 在 Agent 负载 (SWE-Bench、Text-to-SQL) 上最高 5.3 倍。
- SuffixDecoding 用真正的后缀树, 按频率长出一棵草稿树再用树注意力验证 (m19); 这里只出一条链, 用 n ≤ 6 的计数表代替后缀树。
- 只有 greedy。采样时 n-gram 草稿没有草稿分布 p_d, 要按草稿分布是 one-hot 处理接受概率, 这里没有实现。

## 常见误区

- "没有草稿模型就不是投机解码" —— 投机解码只要求草稿便宜、验证无损。草稿可以来自小模型 (m07)、特征层 (m17)、或者直接查文本。
- "prompt lookup 在任何负载上都有加速" —— 开放生成里接受率是 0, 每次 forward 产出 1.00 个, 和逐 token 一样。
- "草稿猜错会让输出变差" —— 不会。greedy 下接受规则保证输出与 target 逐 token 相同 ([1] 在真实 TinyLM 上断言过)。猜错只浪费验证的计算。
- "只查当前 prompt 就够了" —— 输出和以前的请求相似、但不在当前 prompt 里时 ([3]), 只有跨请求的缓存查得到, 接受率 0.0% vs 83.7%。

## 自测题

1. 代码编辑负载里输出 300 个 token, prompt lookup 用了多少次 target forward 才算"5.66x"?
   **答**: 5 条请求共 1500 个输出 token, 用了 265 次 forward, 1500 / 265 = 5.66。逐 token 需要 1500 次。
2. 为什么 SuffixCacheDrafter 的草稿长度上限要随匹配长度 p 伸缩?
   **答**: 只匹配上 1 个 token 时, 后面跟什么分歧很大, 猜长了大多被拒; 匹配上 6 个 token 时, 这段文本大概率是在复读, 可以放心猜远。
3. 开放生成里 prompt lookup 提了 1178 个草稿全被拒, 为什么 forward 次数没有增加?
   **答**: 每轮验证都会白送 1 个纠错 token, 所以最坏情况是每次 forward 产出 1 个, 和逐 token 一样; 被拒的草稿只是在同一次 forward 里多算了几个位置。
