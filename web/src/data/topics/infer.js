// 阶段 5 · llm_infer 章节内容。
// 本文件持有本阶段全部 16 章的完整 page 对象 (models.js 不再保留副本)。
// 贯穿全阶段的一条主线: decode 是带宽瓶颈, 不是算力瓶颈 ——
// 所以几乎每一项优化都在省 KV 的"搬运"和"存储", 而不是省乘法。
const I = 'llm_infer/'

export default {
  stage: 'infer',
  chapters: [
    { route: 'infer-prefix-radix', label: '前缀复用 · hash 与 radix', hint: '链式 block hash、radix tree、命中即跳过前向' },
    { route: 'infer-structured-output', label: 'token 级语法约束', hint: 'FSM × 多字符词表、预编译 mask 表' },
    { route: 'infer-quant-awq', label: 'INT4 · AWQ · KIVI', hint: 'group-wise、激活感知缩放、K 按通道 V 按 token' },
    { route: 'infer-kv-footprint', label: 'KV 体积 · MHA→MLA', hint: 'bytes/token 公式、latent cache decode' },
    { route: 'infer-attention-sinks', label: 'Attention sinks', hint: 'StreamingLLM: sink + 窗口 + 位置重编号' },
    { route: 'infer-tree-speculation', label: 'EAGLE 与树形投机', hint: '特征级 draft、tree attention mask、最长接受路径' },
    { route: 'infer-kv-offload', label: '分层 KV offload', hint: 'GPU→CPU→disk, 加载 vs 重算' },
    { route: 'infer-moe-serving', label: 'MoE serving · EPLB', hint: 'EP dispatch/combine、冗余专家' },
    { route: 'infer-sparse-decode', label: '稀疏注意力 decode', hint: 'block 打分 → top-k, 只读一小部分 KV' },
    { route: 'infer-test-time-compute', after: 'infer-decode-control', label: '推理时计算 · best-of-N 与长思考', hint: '投票、ORM/PRM 判分、PRM beam、budget forcing' },
    { route: 'infer-multi-replica-routing', label: '多副本路由', hint: '前缀感知路由 + 负载阈值兜底' },
  ],
  pages: {
    // ---------------- 主线五章 ---------------- //
    'infer-kv-memory': {
      widgets: ['AttnMaskLab'],
      title: 'KV 与缓存内存 · 推理优化从"别重算"开始',
      subtitle: '读完你能算出一张卡同时放得下多少条序列, 并说清分页到底买到了什么。',
      tldr: '- KV cache: 因果 mask 下旧 token 的 K/V 不会因新 token 到来而改变, 存下来就别再算。累计过一遍模型的 token 数从 688 掉到 37 (18.6×)。\n- 分页: 省下的算力换成了显存压力, 于是把 KV 切成定长 block 用页表管。分页不让 attention 变快, 它买的是显存利用率。',
      question: '有了 KV cache 之后, decode 每步是不是就 O(1) 了?',
      code: 'llm_infer/m01_kv_cache · llm_infer/m02_paged_attention (前缀复用另见「前缀复用 · hash 与 radix」)',
      points: [
        {
          title: '只有 K/V 值得存',
          body: '因果 mask 保证第 $i$ 个 token 的每层 hidden 只依赖 $\\le i$ 的 token, 后面来什么都不会改它。\n- prefill: 把每层的 K/V 存下来。\n- decode: 每步只算新 token 的 q/k/v, 再追加。\nq 不用存: 历史 token 的 q 只在它自己那一步用过一次。',
        },
        {
          title: 'cache 没省掉的那部分',
          key: true,
          body: '新 token 的 query 仍要和全部 t 个 key 做点积。cache 省的是旧 token 的 K/V 投影和 MLP 重算, 不是 attention 本身。\n所以长上下文 decode 依然是 O(t) 的访存。后面的 GQA/MLA、KV 量化、稀疏读取, 都在解决这件事。',
        },
        {
          title: '分页买的是显存, 不是速度',
          body: 'KV 切成定长 block, 每条序列一张页表: 位置 pos 的 KV 在 pool[table[pos // bs], pos % bs]。\n- 代价: 多一层间接寻址, 单次 attention 反而略慢。\n- 收益: 不再按 max_length 预留连续显存, 浪费只剩每条序列末页那几个空槽。',
        },
      ],
      links: [
        { from: 'sample.py 逐 token 生成', to: 'KV cache', body: '同样是自回归, 但不再每步把整段 prefix 重跑一遍。' },
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '显存块够不够, 决定这一步能不能接新请求; 不够就抢占。' },
        { from: 'free_list 归还顺序', to: 'PrefixCache 的 LRU', body: 'block 被 free 只是 ref_count 归零进队列, 内容留到被覆盖为止。前缀复用靠的就是这一点。' },
      ],
      sourceRows: [
        { concept: '对照组', code: 'm01_kv_cache/demo.py:generate_with_cache', takeaway: '有 cache 与无 cache 的输出 ids 必须逐个相同 (logits 差 2.80e-06); 累计过模型的 token 688 → 37。' },
        { concept: 'KV 字节公式', code: '2 · n_layer · T · D · sizeof(dtype)', takeaway: 'demo 断言公式算的 75776 B == numpy 实际 nbytes。代入 LLaMA-7B fp16 = 0.50 MiB/token, T=4096 就是 2.00 GiB 一条。' },
        { concept: '地址翻译', code: 'm02_paged_attention/paged_attention.py:paged_attention', takeaway: 'pos → (block_table[pos // bs], pos % bs), 和操作系统的虚拟内存一模一样; 与连续 KV 的结果 $\\max|\\Delta|$ = 0.00e+00。' },
        { concept: '碎片上界', code: 'BlockManager.stats', takeaway: '每条序列最多浪费 bs−1 个槽 (demo 实测 1/12, 上界 3); 按 max_len 连续预留的方案, vLLM 论文测得只有 20~40% 装着真数据。' },
        { concept: '要不到块', code: 'MemoryError: need 25 new blocks, only 3 free', takeaway: '这个异常就是调度器触发抢占的信号。' },
      ],
      snippetTitle: 'KV cache 路径',
      snippet: `logits, kv_cache = lm.prefill(prompt_ids)
next_id = argmax(logits[-1])

for _ in range(max_new - 1):
    logits, kv_cache = lm.decode_step(next_id, kv_cache)   # 只喂 1 个 token
    next_id = argmax(logits)`,
      source: [`${I}m01_kv_cache/demo.py:generate_with_cache`, `${I}m02_paged_attention/paged_attention.py:paged_attention`],
      run: 'python -m llm_infer.m01_kv_cache.demo',
    },

    'infer-scheduler': {
      title: '调度与 prefill · 每一步都重新组 batch',
      subtitle: '读完你能说清"每步 token 预算"这一个旋钮, 怎么同时拧动 TBT 和 TTFT。',
      tldr: '- 静态 batch: 要等最长的那条。\n- 连续批: 调度粒度从 "一个请求" 缩到 "一步前向", 每步重问一遍谁进谁出。\n一条长 prompt 会把所有正在 decode 的用户卡住, 所以再加两道:\n- chunked prefill: 长 prompt 切块混进 decode 的 batch, 最大卡顿 297 → 52 ms。\n- P/D 分离: 干脆把两类负载放到不同节点。',
      question: '队首请求因为 block 不够而进不来, 这一步调度器该干什么?',
      code: 'llm_infer/m03_continuous_batching · m06_chunked_prefill · m15_pd_disaggregation',
      points: [
        {
          title: 'batch = [(seq, n)]',
          key: true,
          body: '一个列表说清了全部: n>1 是 prefill (或它的一个 chunk), n=1 是 decode, 两者可以出现在同一步。没有"prefill 步"和"decode 步"之分, 只有"这一步谁算几个 token"。',
        },
        {
          title: '永远不能返回空 batch',
          body: '队首拿不到 block 时如果直接返回空: running 不前进 → 没有序列结束 → block 永远不释放 → 队首永远进不来。\n修法只有一行: prefill 进不来就落到 decode, 也就是 _admit(budget) or _schedule_running(budget) 的那个 or。\n抢占也只踢最年轻的, 保证最老的那条总能前进。',
        },
        {
          title: 'token 预算就是 TBT 上限',
          body: '代价模型 step_ms = 20 + 0.25 × 本步 token 数。每步算多少 token, 直接封顶用户两个 token 之间的间隔。\nB=128 时:\n- decode 用户最大卡顿: 297 → 52 ms\n- 那条长请求自己的 TTFT: 276 → 445 ms\n不是白赚。',
        },
      ],
      links: [
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '接不接这条请求, 先问 block 够不够。' },
        { from: 'max_batch_tokens', to: '每步耗时 → TBT', body: '预算是 TBT 与 TTFT 之间的那个旋钮, 调小压卡顿、调大保首 token。' },
        { from: 'KVLink.send', to: 'P/D 分离', body: 'prefill 节点把 KV 按 byte 打包发给 decode 节点; 链路按 bit (Gbps) 标, 少除一个 8 就低估 8 倍。' },
      ],
      sourceRows: [
        { concept: '每步重组 batch', code: 'm03_continuous_batching/scheduler.py:Scheduler.schedule', takeaway: 'decode 优先时先给每条 running 1 个 token, 剩余预算切给 prefill chunk。' },
        { concept: '抢占', code: 'Scheduler._preempt', takeaway: 'block 全还、num_computed 归 0、回 waiting 队首; 已生成的文本原样保留, 回来时连同它一起重算 KV。' },
        { concept: 'TBT 代价模型', code: 'm06_chunked_prefill/chunked_prefill.py:simulate', takeaway: 'step_ms = 固定开销 + 每 token 开销 × batch token 数, 是模型不是实测; 预算 B 决定 TBT 上限。' },
        { concept: '分块不改结果', code: 'chunked_prefill(lm, ids, chunk)', takeaway: 'chunk ≥ 16 时 logits 与整段 prefill 逐位相同 (chunk=8 差 2.86e-06, 只来自分块求和顺序); 分数矩阵峰值 4096 → 512。' },
        { concept: '干扰有多大', code: 'm15_pd_disaggregation/pd.py:KVLink', takeaway: '同卡混跑时被长 prefill 插队的那一步, decode 延迟 0.253 → 4.68 ms (18× 抖动); 分离后回到 0.253 ms。' },
      ],
      snippetTitle: '调度骨架 (decode 优先 + 分块 prefill)',
      snippet: `def schedule(self):
    budget = cfg.max_batch_tokens
    batch = self._schedule_running(budget)   # 每条 running 先拿 1 个 token
    budget -= sum(n for _, n in batch)       # block 不够 → 抢占最年轻的
    if not self._just_preempted:
        batch += self._admit(budget)         # 剩余预算切给 prefill chunk
    return batch                             # [(seq, n_tokens)]`,
      source: [`${I}m03_continuous_batching/scheduler.py:schedule`, `${I}m15_pd_disaggregation/pd.py:KVLink`],
      run: 'python -m llm_infer.m03_continuous_batching.demo',
    },

    'infer-decode-control': {
      widgets: ['SoftmaxTempLab'],
      title: '解码加速与采样 · 更少 target 调用, 更可控的分布',
      subtitle: '读完你能解释"小模型先猜"为什么不让输出变差, 以及采样参数顺序为什么会改结果。',
      tldr: 'decode 每步只出 1 个 token, 却要把整份权重读一遍, 算力大量闲着。\n投机解码: 便宜的 draft 连猜 $K$ 个, target 一次 forward 验 $K+1$ 个槽位, 输出分布一点不变。\n- greedy: 逐位比 argmax。\n- 采样: 用 rejection sampling。\n采样这一侧是 logits 上的一串后处理, 顺序会改结果。',
      question: '为什么投机解码在采样模式下仍然"无损": 输出分布和只用 target 采样完全一样?',
      code: 'llm_infer/m07_speculative_decoding · m10_sampling (EAGLE / 树形投机、语法约束、attention sinks 各有独立章节)',
      points: [
        {
          title: '一次验 K+1 个槽位',
          body: 'target 喂 [out[-1], $d_0, \\dots, d_{K-1}$], 因果 mask 让第 $i$ 行只看到 $d_{<i}$。接受最长的正确前缀, 全对再白送 1 个 bonus token。\n加速比取决于 draft:\n- 与 target 完全一致: 48 个 token 只要 11 次 target 调用 (4.36×)\n- 蒸馏式 draft: 1.85×\n- 随便找个小模型: 1.12×',
        },
        {
          title: 'rejection sampling 让它无损',
          key: true,
          body: '- 接受: 以 $\\min(1, p/q)$ 接受 draft 给的 token。\n- 拒绝: 从归一化残差 $\\max(0, p-q)$ 重采样, 并停止。\n两步合起来恰好还是 $p$, 所以 draft 再差也只影响速度。\ndemo 跑了 $\\chi^2$ 检验: 正确规则 $\\le 13.6$ (临界值 37.7); 故意改成 "全收 draft", 立刻涨到 585。',
        },
        {
          title: '采样是一串顺序敏感的 filter',
          body: '顺序: rep penalty → temperature → top-k → top-p → min-p → Gumbel-max。\n- 温度在前: 同样的 p 会留下更多 token。\n- repetition penalty: 对负 logit 要乘, 不是除。一律除会把 token 9 的概率从 0.0076 抬到 0.0144, 反而鼓励重复。\n各框架顺序不完全一致, 同一组参数跨框架结果可能不同。',
        },
        {
          title: '搜索式解码: beam 找的是"最可能"',
          body: 'greedy 每步只看眼前, 选错一步就回不来。beam 每步在 $w \\times V$ 个候选里留累计 log 概率最高的 $w$ 条前缀, 代价约 $w$ 倍 decode。\nTinyLM 上 10 个 prompt × 20 token:\n- logP/token: greedy −2.620, $w=32$ 到 −2.255。\n- 重复率 rep-2: 0.453 → 0.563。越"可能"越复读。\n- 多样性: 同一 prompt 的 8 条 beam 候选平均只差 3.5/20 个位置, 8 条采样差 19.7。\n趋势不单调: $w=4$ 的 rep-2 是 0.447, 比 greedy 还低。随机权重模型本身就爱复读。',
        },
        {
          title: 'beam 不是精确搜索, 还偏爱短句',
          body: '- 会输给 greedy: $w=2$ 在 2/10 个 prompt 上输。greedy 那条前缀某一步排到第 3, 被剪掉就回不来。只有平均意义上 $\\ge$ greedy。\n- 偏爱短句: 分数是负数累加, 多一个 token 只会更低。给 EOS logit +3 后, $\\alpha=0$ 的输出平均只有 1.3 token。\n- 长度惩罚: 按 $\\text{score}/\\text{len}^\\alpha$ 选。$\\alpha=0.5$ 仍是 1.3, $\\alpha=1$ 才到 15.7 (greedy 11.4)。\n怎么选:\n- 用 beam: 翻译、摘要, 好答案集中在一个峰附近。\n- 用采样: 开放式对话, 好回答很多且分散。',
        },
      ],
      links: [
        { from: 'sample.py 的 temperature/top-k', to: 'm10_sampling', body: '基础采样长成一整套服务端参数。' },
        { from: 'accept_greedy / accept_sampling', to: 'EAGLE · 树形投机', body: '后面两章只换 drafter 和验证形状, 接受规则原样复用。' },
        { from: 'truncate_kv', to: 'KV 回滚', body: '被拒 draft 的 KV 直接截掉, 从不重新 prefill: 因果性保证前缀的 KV 与后面无关。' },
      ],
      sourceRows: [
        { concept: 'greedy 接受', code: 'm07_speculative_decoding/speculative.py:accept_greedy', takeaway: 'target 的 argmax 与 draft token 逐位比对, 第一个不一致处停下, 返回 target 自己的 token。' },
        { concept: '采样接受', code: 'm07_speculative_decoding/speculative.py:accept_sampling', takeaway: '$u < p_t/p_d$ 就接受; 否则从 $\\max(0, p_t - p_d)$ 归一化后重采样。全接受时 bonus 直接采自 target。' },
        { concept: '期望产出', code: '(1 − α^(K+1)) / (1 − α)', takeaway: '每次 target 调用的期望 token 数; $\\alpha$ 是每 token 接受率。$K$ 再大也被 $1/(1-\\alpha)$ 封顶, 这正是树形投机的动机。' },
        { concept: '采样顺序', code: 'm10_sampling/samplers.py:sample', takeaway: '六步链条里每一步都是"把被砍的置 -inf", 所以可以任意串联。' },
        { concept: 'Gumbel-max 是精确的', code: 'argmax(log p + G), G = −log(−log U)', takeaway: '与 multinomial 同分布, TV 0.0064 vs 参照噪声 0.0066; 只有逐元素运算 + 一次 argmax, 整个 batch 一个 kernel。' },
        { concept: 'beam search', code: 'm24_beam_search/beam.py:beam_search', takeaway: '每条活 beam 取 top-$(w+1)$ 个 token 作候选, 保证有 EOS 时仍能凑满 $w$ 条; 结束时按 $\\text{score}/\\text{len}^\\alpha$ 排序。$w=1$ 与 greedy 逐 token 相同。' },
        { concept: '造一个会停的模型', code: 'm24_beam_search/demo.py:EosBiased', takeaway: '随机权重模型 P(EOS) 约 0.5%/步, 从不停。EOS logit +3 后约 5%/步, 才看得到 beam 偏爱短句。' },
      ],
      snippetTitle: 'rejection sampling 接受规则',
      snippet: `def accept_sampling(d_tokens, d_probs, t_probs, rng):
    for i, tok in enumerate(d_tokens):
        if rng.random() < t_probs[i, tok] / d_probs[i, tok]:
            continue                                  # 以 min(1, p/q) 接受
        residual = maximum(t_probs[i] - d_probs[i], 0)
        return i, sample(residual / residual.sum())   # 拒绝: 从残差重采样并停止
    return len(d_tokens), sample(t_probs[-1])         # 全接受: bonus 采自 target`,
      source: [`${I}m07_speculative_decoding/speculative.py:accept_sampling`, `${I}m10_sampling/samplers.py:sample`, `${I}m24_beam_search/beam.py:beam_search`],
      run: 'python -m llm_infer.m07_speculative_decoding.demo',
    },

    'infer-compute': {
      title: '算子与调度开销 · 一样的数学, 少搬几趟',
      subtitle: '读完你能说清 FlashAttention 快在哪一步, 以及 CUDA Graph 省的是谁的时间。',
      tldr: '- FlashAttention: Q 和 K/V 都切块, 用 online softmax 增量维护最大值、分母和输出, 从不把 $T \\times T$ 的分数矩阵写进显存。工作集恒为 64×64, $T=4096$ 时比 $T^2$ 小 4096 倍, 结果与朴素实现只差 1.78e-06。\n- CUDA Graph: 一步几百次 kernel 提交压成 1 次。\n- 推理 TP: 每层权重切给多张卡, 一个 block 只需 2 次 all-reduce。',
      question: 'FlashAttention 的 FLOPs 一点没少, 凭什么还能快?',
      code: 'llm_infer/m09_tensor_parallel · m11_flash_attention · m12_cuda_graph (量化见「INT4 · AWQ · KIVI」)',
      points: [
        {
          title: 'softmax 的分母可以增量维护',
          key: true,
          body: '这是全部技巧。每个 query 行只带三个运行量: 最大值 $m$、分母 $l$、输出 $O$。\n来一块新分数就更新它们。出现更大的 max 时, 把旧的和整体乘 $\\exp(m_{\\text{old}} - m_{\\text{new}})$ 换基准。\n于是每个 tile 只进一次 SRAM, 从不物化 $T \\times T$。快不是因为算得少, 是因为搬得少。',
        },
        {
          title: 'LSE 是可以拼接的凭证',
          body: '$\\mathrm{lse} = m + \\log l$。\n- 用两段各自的 $(O, \\mathrm{lse})$ 合成: 精确得到全量结果 (误差 2.68e-07)。\n- 直接取平均: 错到 0.212, 因为两段的 softmax 质量不相等。\nring attention、chunked prefill、跨卡分段 KV 全靠这一个原语。',
        },
        {
          title: 'CUDA Graph 省的是 host',
          body: '固定形状的 decode 捕获一次, 之后每步只 replay 一次。kernel 本身一点没变快。\n本仓库的 launch 开销是显式注入的参数: 0 / 10 / 50 / 200 µs 分别对应 1.3× / 6.4× / 12.2× / 15.0×。这些加速比是代价模型的产物, 不是实测。\nbatch 越大, 单 kernel 越久, launch 占比越小。所以大 prefill 通常回退 eager。',
        },
        {
          title: 'decode 时 FlashAttention 喂不饱 GPU',
          body: 'FlashAttention 的并行单位是 (batch, head, Q 块)。\n- prefill: Q 有几千行, 块多得是。\n- decode: 每条序列只有 1 个 query, 并行单位只剩 $B \\times H$。\nB=1、H=32 的长对话: 32 个 block 跑在 108 个 SM 上, SM 利用率 30%。每个 block 还要独自把 32k token 的 KV 从头读到尾。',
        },
        {
          title: 'Flash-Decoding: Q 切不动就切 KV',
          body: '把 KV 长度切成 $S$ 段, 每段一个 block 算局部 $(O_s, \\mathrm{lse}_s)$, 再按 $\\exp(\\mathrm{lse}_s - \\mathrm{lse})$ 加权合并。就是上面的 LSE 拼接, 一次合 $S$ 段, 13 组配置最坏误差 2.06e-07。\n代价模型 (A100 参数, 不是实测) 下:\n- B=1、T=32k: S=64 时 909.0 → 275.5 μs, 3.30× (上限 $108/32 = 3.38$×)。\n- S=4 和 S=2 一样慢 (459.0 μs): 128 个 block 要跑 2 波。\n- B=16 最多 1.05×, B=64 为 1.00×: SM 本来就满了。\n边界:\n- 带宽: 真卡上单个 SM 能拿到的带宽比 1/108 高, 真实加速更小。\n- 权重 GEMM: 整步 decode 里读权重的 GEMM 不受影响。',
        },
      ],
      links: [
        { from: '朴素 attention', to: 'flash_attention', body: '数学上严格相等 (30 组配置最坏 1.78e-06), 省的是对 $N \\times N$ 中间矩阵的反复读写。' },
        { from: 'merge_attention(O, lse)', to: 'chunked prefill / ring attention', body: '能分段算完再合并, prefill 才切得动、KV 才跨得了卡。' },
        { from: 'tp_block', to: '多卡推理', body: '列切→行切配对, 中间结果天然是切开的, 只在子层末尾 all-reduce 一次。' },
      ],
      sourceRows: [
        { concept: 'online softmax', code: 'm11_flash_attention/flash_attention.py:flash_attention', takeaway: '$m$ / $l$ / $O$ 随 KV block 增量更新; 外层 Q 块、内层 K/V 块 (FA-2), Q 块的状态常驻 SRAM 只写回一次。' },
        { concept: 'causal 整块跳过', code: 'flash_attention(..., causal=True)', takeaway: 'K 块最早的 key 晚于 Q 块最晚的 query 就整块不算。$T=4096$ 时 2016/4096 块被跳过 (49.2%), 省计算靠跳块不靠 mask。' },
        { concept: '分段合并', code: 'm11_flash_attention/flash_attention.py:merge_attention', takeaway: '按 lse 加权合并两段输出; 只需传 $(O, \\mathrm{lse})$, 不传 $T \\times T$。' },
        { concept: '图只认地址', code: 'm12_cuda_graph/graph.py:CudaGraph', takeaway: 'host 提交 16 → 1 次, 输出逐位相同。static_input 必须拷贝写入; 重新绑定名字的写法算的还是旧数据 (demo 复现了这个 bug)。' },
        { concept: 'TP 切在 head 边界', code: 'm09_tensor_parallel/parallel_linear.py:tp_block', takeaway: '每 block 2 次 all-reduce, 载荷恒为 $T \\times D$ (8192 B) 与 tp 无关; 切进单个 head 内部会错 5.63e-01, 因为 softmax 不能跨 rank 拆。' },
        { concept: '一段 KV 的局部结果', code: 'm26_flash_decoding/flash_decoding.py:partial_attention', takeaway: '返回已归一化的 $O_s$ 和分母的 log $\\mathrm{lse}_s$, 各段互不依赖, GPU 上是 $S$ 个并行 block。' },
        { concept: 'split-K + reduce', code: 'm26_flash_decoding/flash_decoding.py:flash_decoding', takeaway: '$\\mathrm{lse} = \\log \\sum_s \\exp(\\mathrm{lse}_s)$, $O = \\sum_s \\exp(\\mathrm{lse}_s - \\mathrm{lse}) \\cdot O_s$。T 不必整除 S。' },
        { concept: '延迟估算', code: 'm26_flash_decoding/flash_decoding.py:decode_latency_us', takeaway: 'units = B·H·S, waves = ⌈units / 108⌉, 每波读 T/S 个 token 的 KV, 每个 SM 最多拿 1/108 的带宽。纯代价模型。' },
      ],
      snippetTitle: 'online softmax 的三个运行量',
      snippet: `m_new = maximum(m, max(S_block))
P_b = exp(S_block - m_new)
l = exp(m - m_new) * l + sum(P_b)       # 旧分母换基准
O = exp(m - m_new) * O + P_b @ V_block  # 旧输出同样换基准
m = m_new
# 收尾: lse = m + log(l); out = O / l`,
      source: [
        `${I}m11_flash_attention/flash_attention.py:flash_attention`,
        `${I}m11_flash_attention/flash_attention.py:merge_attention`,
        `${I}m12_cuda_graph/graph.py:CudaGraph`,
        `${I}m26_flash_decoding/flash_decoding.py:flash_decoding`,
        `${I}m26_flash_decoding/flash_decoding.py:decode_latency_us`,
      ],
      run: 'python -m llm_infer.m11_flash_attention.demo',
    },

    'infer-engine': {
      title: 'mini-vLLM 引擎 · 把前面 22 章接成一个循环',
      subtitle: '读完你能在 vLLM 的 step() 里认出每一行对应前面哪一章。',
      tldr: 'Engine.step 永远是同四件事: 调度 → 前向 → 采样 → 后处理。\nKV 不挂在序列上, 而是写进全局分页 pool, 所以前缀命中的 block 真的跳过前向。demo 实测:\n- 首轮: 305 个待算 token = 233 实算 + 72 命中。\n- 同一批 prompt 再来一遍: 只实算 105。\n- 9 个 block 的小池: 逼出 4 次抢占, 输出仍与朴素 greedy 逐 token 相同。',
      question: '为什么说推理引擎首先是调度器和资源管理器, 其次才是 model.forward 的包装?',
      code: 'llm_infer/full_engine/engine.py · llm_infer/full_engine/model_runner.py · llm_infer/m13_lora_serving/lora.py',
      points: [
        {
          title: '真分页 KV pool',
          body: 'KV 写进 pool[layer] (num_blocks, block_size, D), attention 经 block_table 读回。\n所以 "命中" 是真的没算: ModelRunner.run 只对 ids[start_pos:] 做 Q/K/V。\nstart_pos 之前的 KV 直接从 pool 里取, 来源有三: 前缀命中、上一个 chunk、之前的 decode 步。',
        },
        {
          title: '难的只有调度那一步',
          key: true,
          body: '前向、采样、后处理都是固定动作。调度要同时管住三件事:\n- block 够不够: 不够就抢占最年轻的。\n- 队首会不会饿死: 进不来必须落到 decode。\n- 这一步算多少 token: 用预算封顶每步耗时。\n引擎的复杂度全在这里, 主循环只有四行。',
        },
        {
          title: '账必须对得上',
          body: 'prefix_hit_tokens + tokens_computed 必须等于全部需要 KV 的 token 数, 抢占后的重算也如实计入。\n这条 assert 是 "省下的算力是真的" 的唯一证据。命中了却照样整段 prefill 的引擎, 账一对就露馅。',
        },
      ],
      links: [
        { from: 'Engine.add_request', to: 'Scheduler.waiting', body: 'prompt encode 后进入调度系统, 每条请求绑定自己的 SamplingParams。' },
        { from: 'Scheduler._admit', to: 'PrefixCache.match_prefix', body: '命中的 token 不占预算、不用算, seq.num_computed 直接从 n_hit 起步。' },
        { from: 'ModelRunner.run', to: 'm02.paged_attention', body: 'q 只有新 token 那几行, K/V 是经页表读回的整段前缀 (含别人算好的公共部分)。' },
        { from: 'Multi-LoRA (m13)', to: '同一 batch 多个 adapter', body: '底模那次 gemm 全 batch 共享, 每个 token 按自己的 adapter id 取 A/B。1000 个 adapter: 合并权重要 24.58 MB, 不合并只要 2.58 MB。' },
      ],
      sourceRows: [
        { concept: '一步', code: 'full_engine/engine.py:Engine.step', takeaway: 'schedule → 逐序列 run → 追平了才 sample → postprocess, 一共四行。' },
        { concept: '只算新 token', code: 'full_engine/model_runner.py:ModelRunner.run', takeaway: 'run(ids, block_table, start_pos): Q 只有 len(ids) − start_pos 行, K/V 是完整的 len(ids) 行。' },
        { concept: '中途不采样', code: 'done_prefill = num_computed + n == num_tokens', takeaway: 'prefill chunk 最后一个位置预测的是 prompt 里已知的下一个 token, 没有新 token 可采, 所以返回 None。' },
        { concept: '抢占', code: 'm03_continuous_batching/scheduler.py:Scheduler._preempt', takeaway: '还 block + 回队首, output_ids 原样保留; 只踢最年轻的, 所以最老的序列永远能前进, 不会活锁。' },
        { concept: '统计', code: 'Engine.report_stats', takeaway: 'steps / tokens_computed / prefix_hit_tokens / preempt / pool, 一眼看完这一轮的资源账。' },
        { concept: 'fuzz', code: 'demo [5]: 30 组随机配置', takeaway: '214 条请求、34 次抢占, 全部与 generate_greedy 逐 token 相同, 无活锁、无 block 泄漏。' },
      ],
      snippetTitle: 'Engine.step 的四件事',
      snippet: `def step(self):
    batch = self.scheduler.schedule()            # ① 调度: [(seq, n)], prefill chunk 与 decode 混批
    tokens = []
    for seq, n in batch:
        ids = seq.all_ids[:seq.num_computed + n]
        logits = self.runner.run(ids, block_table(seq), seq.num_computed)   # ② 前向: 只算 n 个
        caught_up = seq.num_computed + n == seq.num_tokens
        tokens.append(sample(logits) if caught_up else None)                # ③ 采样
    return self.scheduler.postprocess(batch, tokens)                        # ④ 后处理`,
      source: [`${I}full_engine/engine.py:step`, `${I}full_engine/model_runner.py:run`],
      run: 'python -m llm_infer.full_engine.demo',
    },

    // ---------------- 深入九章 ---------------- //
    'infer-prefix-radix': {
      title: '前缀复用 · 从 block hash 到 radix tree',
      subtitle: '多轮对话、few-shot、共享 system prompt: 请求之间的公共前缀只该算一次。',
      tldr: 'KV 只依赖它之前的 token, 所以相同前缀的 KV 可以跨请求复用。\n- m04 链式 block hash: 只能命中整块。\n- m05 radix tree: 能命中任意长度, 代价是要处理边分裂。\n容量满了从叶子按 LRU 驱逐。',
      question: '驱逐时为什么只能从叶子开始, 不能直接扔掉最久没用的中间节点?',
      code: 'llm_infer/m04_prefix_cache/prefix_cache.py · llm_infer/m05_radix_cache/radix_tree.py',
      points: [
        {
          title: '一个 hash 标识一整段前缀',
          key: true,
          body: '$h_i = H(h_{i-1} \\,\\|\\, \\text{block}_i)$。\n父 hash 参与计算, 所以同一段 token 接在不同前缀后面不会误命中。原因: attention 让位置 $i$ 的 KV 取决于它前面的全部 token。\n查表 O(块数), 但只能命中 $\\lfloor n/\\text{bs} \\rfloor \\cdot \\text{bs}$ 个 token。',
        },
        {
          title: 'radix tree 换到 token 粒度',
          body: '边上存一段 token 和等长的 KV 槽位, 新请求沿树走到最长公共前缀。\n分叉落在边中间时, _split 把 (tokens, slots) 同步切成两半, 新的中间节点继承 ref_count。\n前 100 个 token 相同、bs=16 时: 链式 hash 命中 96 个, radix 命中全部 100 个。',
        },
        {
          title: '只驱逐叶子',
          body: '中间节点的 KV 被它所有后代依赖, 先扔它, 后代就全部作废: 前缀必须从根连续走下来。所以只挑 ref_count=0 的叶子, 按 last_access 从旧到新。',
        },
      ],
      links: [
        { from: 'BlockManager.ref_count', to: 'PrefixCache', body: '共享 block 靠引用计数, 最后一个使用者释放后才可回收。' },
        { from: 'RadixCache.match_prefix', to: 'Scheduler._admit', body: '命中的 token 不占 token 预算, num_computed 从 n_hit 起步。' },
        { from: 'radix tree', to: '分层 KV offload', body: '被 GPU 挤出去的前缀不丢弃, 降级到 CPU / 磁盘。' },
      ],
      sourceRows: [
        { concept: '链式 hash', code: 'm04_prefix_cache/prefix_cache.py:_block_hash', takeaway: 'parent_hash 参与计算, 所以一个 hash 唯一标识"到第 i 块为止"的整段前缀。' },
        { concept: '至少留一个', code: 'match_prefix 只匹配 (len−1)//bs 块', takeaway: '前缀全命中也要真算最后一个 token, 否则拿不到 logits, 首 token 无从采起。cache 里存的是 KV, 不是 logits。' },
        { concept: '最长前缀匹配', code: 'm05_radix_cache/radix_tree.py:RadixCache.match_prefix', takeaway: '按子边的首 token 索引孩子, 再沿边逐 token 比较。' },
        { concept: '边分裂', code: 'RadixCache._split', takeaway: 'tokens 与 slots 必须同步切; 新中间节点继承原节点的 ref_count。' },
        { concept: '驱逐', code: 'RadixCache.evict', takeaway: '只看叶子, 按 last_access 从旧到新。' },
      ],
      snippetTitle: 'radix 匹配 + 分裂',
      snippet: `def match_prefix(self, ids):
    node, i, slots = self.root, 0, []
    while i < len(ids) and ids[i] in node.children:
        child = node.children[ids[i]]
        n = lcp(child.edge_tokens, ids[i:])      # 沿这条边能走多远
        slots += child.slots[:n]
        if n < len(child.edge_tokens):
            child = self._split(child, n)        # 分叉落在边中间 → 切成两段
        node, i = child, i + n
    return slots                                 # 这些 KV 槽位直接复用, 不用前向`,
      source: [`${I}m05_radix_cache/radix_tree.py:RadixCache`, `${I}m04_prefix_cache/prefix_cache.py:PrefixCache`],
      run: 'python -m llm_infer.m05_radix_cache.demo',
    },

    'infer-structured-output': {
      title: 'token 级语法约束 · 把 FSM 预编译成 mask 表',
      subtitle: '读完你能解释为什么"合法字符"和"合法 token"是两回事。',
      tldr: '字符级 FSM 能保证语法合法, 但模型吐的是多字符 token (如 「":」、「true」)。\n- 离线: 对每个 (状态, token) 试走一遍。整段字符都走得通才算合法, 同时记下落点状态。\n- 在线: 每步只查一行表, 把非法 token 的 logit 置 −inf。',
      question: '为什么不能每步在线对 V 个 token 逐字符试走一遍 FSM?',
      code: 'llm_infer/m14_structured_output/grammar.py',
      points: [
        {
          title: '合法性是 token 级的, 不是字符级的',
          key: true,
          body: '真实词表里 「":」 是一个 token, 一口气跨过 "结束 key" 和 "进入 value" 两个状态。\n判定只有一条标准: 这段字符从状态 s 出发能不能全部走通。\nnext_state[s, t] 记下落点, −1 表示走不通; mask_table = next_state ≥ 0。',
        },
        {
          title: '为什么必须离线编译',
          body: '在线现算是 O(V·len) 的纯 CPU 开销, 卡在 GPU 前向和采样中间, 词表十几万时直接吃掉 decode 延迟。FSM 状态数有限, 离线各跑一遍就够, 在线只查一行, O(1)。',
        },
        {
          title: '还要能收尾',
          body: 'need[s, t] = 选了 t 之后最少还要几个 token 才能结束。\n剩余长度不够时, 提前屏蔽那些 "收不了尾" 的 token。这保证输出一定是完整可解析的 JSON, 而不是被长度上限截断的半截。',
        },
      ],
      links: [
        { from: 'JsonFSM (字符级)', to: 'compile_char_dfa', body: '先把字符级 FSM 枚举成 DFA 转移表。' },
        { from: 'token_row', to: 'compile_token_table', body: '对每个状态跑一遍"逐 token 逐字符试走", 存成 (S, V) 表。' },
        { from: 'mask_table[s]', to: 'm10 sample', body: 'mask 也只是一个把 logit 置 -inf 的 filter, 插在同一条采样链上, 之后的温度 / top-p 照常。' },
      ],
      sourceRows: [
        { concept: '词表', code: 'm14_structured_output/grammar.py:build_vocab', takeaway: '单字符 + 类 BPE 多字符片段, 特意含跨语法边界的 \'":\' 、\'e"\' 和永远非法的垃圾 token。' },
        { concept: '试走', code: 'm14_structured_output/grammar.py:token_row', takeaway: 'EOS 只在接受态合法; 其它 token 逐字符走转移表, 中途走不通就整个 token 作废。' },
        { concept: '预编译', code: 'm14_structured_output/grammar.py:compile_token_table', takeaway: 'next_state / mask_table / need 三张表, 一次算好反复用。' },
        { concept: '光靠 prompt 不够', code: 'logits[~mask] = -inf', takeaway: 'prompt 只能提高概率; 只有在 logits 上把非法 token 置 −inf, 概率才严格为 0。' },
        { concept: '与 xgrammar 的差距', code: 'CFG / 下推自动机', takeaway: '有栈的语法只能预编译"与栈无关"的那部分 token, 其余运行时再查。' },
      ],
      snippetTitle: '离线编译 + 在线查表',
      snippet: `# 离线: 每个状态 × 每个 token 试走一遍
for s in states:
    for t, piece in enumerate(vocab):
        cur = s
        for ch in piece:                  # 多字符 token 可以连跨几个状态
            cur = trans[cur].get(ch, -1)
            if cur < 0: break
        next_state[s, t] = cur            # -1 = 非法

# 在线: 每步 O(1)
logits[~mask_table[state]] = -inf
tok = sample(logits)
state = next_state[state, tok]`,
      source: [`${I}m14_structured_output/grammar.py:token_row`, `${I}m14_structured_output/grammar.py:compile_token_table`],
      run: 'python -m llm_infer.m14_structured_output.demo',
    },

    'infer-quant-awq': {
      title: 'INT4 · AWQ · KIVI · 比特越低, "怎么分组"越要紧',
      subtitle: 'decode 是带宽受限的: 权重和 KV 越小, 每步要搬的字节越少。',
      tldr: '- group-wise INT4: 每 $g$ 个权重共用一组 scale/zero, 离群值只污染自己那一组。\n- AWQ: 输出误差 $= \\sum_i x_i \\cdot \\Delta W_i$。量化前把激活大的输入通道放大 $s_i$、激活同步缩小, 数学等价, 误差从 0.0759 降到 0.0273。\n- KV 量化: K 有固定的离群通道, 按通道分组; V 没有, 按 token 分组。',
      question: '量化要最小化的为什么是 $\\|XW - X\\hat{W}\\|$ 而不是 $\\|W - \\hat{W}\\|$?',
      code: 'llm_infer/m08_quantization/{int8_weight.py,int4_awq.py,kv_quant.py}',
      points: [
        {
          title: '分组把离群值关起来',
          body: '一组一套 scale/zero, 一个离群值只撑大自己这一组的 range。$g$ 越小越准, 但 scale/zero 的额外开销越大: $g=128$ 时是 $4 + 32/128 = 4.25$ bit/权重, $g=32$ 时约 4.5 bit。',
        },
        {
          title: '重要的是输出误差, 不是权重误差',
          key: true,
          body: '模型在意的是 $X\\hat{W}$ 和 $XW$ 差多少。误差 $= \\sum_i x_i \\cdot \\Delta W_i$: 同样的舍入误差, 乘上大激活后对输出的影响就大。\nAWQ 用恒等式 $XW = (X/s) \\cdot (s \\odot W)$: 把激活大的那些行放大 $s$ 倍, 该行相对舍入误差缩小约 $s$ 倍。\n代价是撑大了同组的 range, 所以 $\\alpha$ 要在校准集上网格搜 (demo 最优 $\\alpha=0.4$, $\\alpha=0$ 就是 RTN)。',
        },
        {
          title: 'K 和 V 的离群结构不一样',
          body: 'K 的少数通道在所有 token 上都是大值:\n- 按 token 分组: 每一行的 scale 都被它撑大, 误差 0.18。\n- 按通道分组: 把它关在自己组里, 误差 0.026。\nV 没有这种固定离群通道。按 token 分组, 还能在每个新 token 写入时独立量化, 不用回头改旧的。',
        },
        {
          title: 'GPTQ 和 AWQ 分工: 一个补偿, 一个保护',
          body: '两者都只量化权重, 存储格式和上面的 group-wise INT4 完全相同, 推理 kernel 不变。治的病不同:\n- AWQ: 少数重要输入通道的误差被大激活放大。量化前把这些行放大, 让它们分到更细的格点。\n- GPTQ: RTN 每个权重各自舍入, 误差在输出上累加。量化一行, 就把它的误差按 $H^{-1}$ 摊给后面还没量化的行, $H = X^\\top X / N$。\n实测 (INT4 g32, held-out 激活):\n- 相关激活: RTN 0.0776 → GPTQ 0.0369。\n- 再加 ×30 离群通道: RTN 0.0749, AWQ 0.0258, GPTQ 0.0293, 先 AWQ 再 GPTQ 0.0113。\n叠加比单用都好, 说明两者治的是不同的病。',
        },
        {
          title: 'GPTQ 的收益全靠激活相关性',
          body: '激活通道互不相关时 $H$ 是对角的, 补偿项全为 0, GPTQ 退化成 RTN。\n校准样本少时, 它会把采样噪声当成相关性去补偿:\n- N=256 (对 256 维输入): held-out 误差是 RTN 的 1.56×。\n- N=512: 1.17×; N=2048: 1.03×; N=8192: 1.01×, 才打平。\n本 demo 用"rank-32 公共成分 + 噪声"合成相关激活。真实 LLM 激活确实强相关, 但幅度不是这个数。',
        },
        {
          title: 'SmoothQuant 把离群值挪给权重',
          body: 'weight-only 省显存和 decode 带宽。想让 matmul 本身走 INT8 / FP8 tensor core (prefill 也快), 激活也得量化, 而激活有离群通道。\nSmoothQuant 用和 AWQ 同一个恒等式, 目的相反: $s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$, 激活除以 $s$、权重乘 $s$。\n- $\\alpha=0.5$: W8A8 误差 0.0278 → 0.0070, 降 4.0×。\n- $\\alpha=1$: 离群值全压给权重, 反而涨到 0.0297。',
        },
        {
          title: 'FP8 换一种格点',
          body: 'FP8 的格距随数值伸缩, 相对误差恒定:\n- per-channel scale: 对 INT8 改善 7.71×, 对 E4M3 只有 1.02×。\n- 激活 per-tensor、带 ×50 离群值: E4M3 0.0360, INT8 0.0606。\n- 没有离群值: INT8 per-channel (0.0070) 比 E4M3 (0.0251) 还准。\n这一节和上一节全是 fake quant, 不说明速度。',
        },
      ],
      links: [
        { from: 'quantize_int8 (per-channel)', to: 'quantize_groupwise', body: '从每通道一组, 缩小到每 $g$ 个权重一组。' },
        { from: 'mean|x_i|', to: 's_i = mean|x_i|^α', body: '只用校准激活的逐通道统计量, 不需要反向传播, 所以 AWQ 很便宜。' },
        { from: 'KV 量化', to: 'KV 体积一章', body: '量化减的是每个元素的字节数, GQA/MLA 减的是元素个数, 两者相乘。' },
      ],
      sourceRows: [
        { concept: '分组量化', code: 'm08_quantization/int4_awq.py:quantize_groupwise', takeaway: 'reshape 成 (D_in/g, g, D_out), 在 g 那一维上统计 min/max。' },
        { concept: '优化目标', code: 'm08_quantization/int4_awq.py:output_err', takeaway: '$\\|XW - X\\hat{W}\\|_F / \\|XW\\|_F$, 拿校准激活算出来的相对输出误差。' },
        { concept: 'AWQ 搜索', code: 'm08_quantization/int4_awq.py:awq_quantize', takeaway: '$\\hat{W} = Q(s \\odot W) / s$; $1/s$ 离线折进上一层的权重, 推理时零额外开销。' },
        { concept: 'KV 三种分组', code: 'm08_quantization/kv_quant.py:quantize_kv', takeaway: '三种 scheme 的实现只差 reshape 与 axis, 误差却差一个数量级。' },
        { concept: 'GPTQ', code: 'm25_weight_quant/gptq.py:gptq_quantize', takeaway: '$U = \\mathrm{chol}(H^{-1})^\\top$; 第 $i$ 行量化后 $\\delta = (W_i - Q_i)/U_{ii}$, 后面的行减去 $U_{i,i+1:}^\\top \\otimes \\delta$。每组的 scale 用已补偿过的权重现算。' },
        { concept: '平滑因子', code: 'm25_weight_quant/smoothquant.py:smooth_scales', takeaway: '$s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$; 激活统计来自校准集, 部署时是常数, 可折进上一层。' },
        { concept: 'W8A8', code: 'm25_weight_quant/smoothquant.py:w8a8_matmul', takeaway: '激活 per-token、权重 per-output-channel 的对称 INT8; 两个 scale 都能提到 matmul 外面。两个开关把误差拆成激活贡献和权重贡献。' },
        { concept: 'FP8 舍入', code: 'm25_weight_quant/fp8.py:fp8_round', takeaway: '格距 $= 2^{\\lfloor \\log_2|x| \\rfloor - m}$, 指数下限钉在最小正规数, 超过 max 饱和。E4M3 正规区相对误差 $\\le 2^{-4}$, E5M2 $\\le 2^{-3}$。' },
      ],
      snippetTitle: 'AWQ: 量化前先缩放',
      snippet: `act = mean(abs(X_calib), axis=0)          # (D_in,) 每个输入通道的激活幅度
for alpha in arange(0, 1.01, 0.1):        # α = 0 就是 RTN
    s = act ** alpha
    s = s / sqrt(s.max() * s.min())       # 几何中心归一, 不整体放大 W
    W_hat = quant_dequant(W * s[:, None]) / s[:, None]
    err[alpha] = norm(X @ W - X @ W_hat) / norm(X @ W)
best = argmin(err)`,
      source: [`${I}m08_quantization/int4_awq.py:awq_quantize`, `${I}m08_quantization/kv_quant.py:quantize_kv`, `${I}m25_weight_quant/gptq.py:gptq_quantize`, `${I}m25_weight_quant/smoothquant.py:smooth_scales`],
      run: 'python -m llm_infer.m08_quantization.demo',
    },

    'infer-kv-footprint': {
      title: 'KV 体积 · MHA / MQA / GQA / MLA',
      subtitle: '同一套 attention 数学, 四种"cache 里存什么", 直接决定一张卡能同时服务多少条序列。',
      tldr: '每 token KV 字节 $= 2 \\cdot n_{kv} \\cdot d_{\\text{head}} \\cdot n_{\\text{layer}} \\cdot \\text{bytes}$; MLA 只缓存 latent, 是 $(d_c + d_{\\text{rope}}) \\cdot n_{\\text{layer}} \\cdot \\text{bytes}$。LLaMA-2-7B (MHA) 512 KiB → LLaMA-3-8B (GQA-8) 128 KiB → DeepSeek-V3 (MLA) 68.6 KiB, 比同尺寸 MHA 省 56.9×。',
      question: 'MLA 的 latent 为什么不能带 RoPE, 而要另外留一份解耦的 RoPE key?',
      code: 'llm_infer/m18_kv_attention_variants/attention_variants.py',
      points: [
        {
          title: '一个参数 n_kv 分出前三种',
          key: true,
          body: '- MHA: $n_{kv} = n_{\\text{head}}$\n- GQA: $1 < n_{kv} < n_{\\text{head}}$\n- MQA: $n_{kv} = 1$\n- MLA: 另一条路, 改存 latent。\n每组 query 头靠广播共享 KV, kernel 里不真复制。\n并发上限 = 显存预算 ÷ 每 token 字节 ÷ 上下文长度。结构选择在推理侧直接变成吞吐。',
        },
        {
          title: 'MLA 的 cache 里只有 latent',
          body: '只存 (T, d_c) 和 (T, d_rope) 两份。per-head 的 K/V 在 attention 时才由 $C W_{UK}$ / $C W_{UV}$ 现场还原, 从不进 cache。所以公式里没有那个 "2·", K 和 V 共用同一个 latent。',
        },
        {
          title: 'absorb 形式',
          body: '$q \\cdot (C W_{UK})^\\top = (q W_{UK}^\\top) \\cdot C^\\top$: 把 $W_{UK}$ 乘到 $q$ 上, K 就是 cache 本身。\n于是 MLA 的 decode 等价于一个 $\\text{head\\_dim} = d_c + d_{\\text{rope}}$ 的 MQA。\nlatent 一旦带上 RoPE, 位置相关的旋转就夹在 $W_{UK}$ 前面, 这个吸收做不成了。',
        },
      ],
      links: [
        { from: '阶段 2 注意力演进', to: 'KV bytes/token', body: '当年为省参数做的结构选择, 在推理侧变成显存账。' },
        { from: 'bytes/token', to: 'BlockManager.num_blocks', body: '显存 ÷ 每 token 字节 = 能放多少 token = 并发上限。' },
        { from: 'MLA absorb', to: 'FlashMLA', body: '真实 kernel 走吸收形式, 全程不物化 per-head K/V。' },
      ],
      sourceRows: [
        { concept: 'GQA 公式', code: 'm18_kv_attention_variants/attention_variants.py:kv_bytes_per_token', takeaway: '开头那个 2 = K 和 V 两份。' },
        { concept: 'MLA 公式', code: 'm18_kv_attention_variants/attention_variants.py:mla_bytes_per_token', takeaway: '没有 "2·": K/V 共用同一个 latent。' },
        { concept: 'latent decode', code: 'MLALayer.forward', takeaway: 'latent 不加 RoPE, 位置信息全放在另一份解耦的 RoPE key 里。' },
        { concept: '对拍', code: 'cache_nbytes', takeaway: 'demo 断言实际 cache 字节数 == 公式值。' },
      ],
      snippetTitle: '两条公式',
      snippet: `def kv_bytes_per_token(n_kv, d_head, n_layer, nbytes=2):
    return 2 * n_kv * d_head * n_layer * nbytes     # 2 = K 和 V

def mla_bytes_per_token(d_c, d_rope, n_layer, nbytes=2):
    return (d_c + d_rope) * n_layer * nbytes        # K/V 共用 latent

# LLaMA-2-7B  MHA 32kv×128×32L → 512 KiB
# LLaMA-3-8B  GQA  8kv×128×32L → 128 KiB
# DeepSeek-V3 MLA (512+64)×61L → 68.6 KiB
max_seqs = kv_budget_bytes // (bytes_per_token * context_len)`,
      source: [`${I}m18_kv_attention_variants/attention_variants.py:kv_bytes_per_token`, `${I}m18_kv_attention_variants/attention_variants.py:mla_bytes_per_token`],
      run: 'python -m llm_infer.m18_kv_attention_variants.demo',
    },

    'infer-attention-sinks': {
      title: 'Attention sinks · 流式长上下文的有界 KV',
      subtitle: '纯滑动窗口一旦把开头几个 token 挤出去, 模型立刻崩, 因为 softmax 必须把那个 1 分给某个人。',
      tldr: 'softmax 的权重和恒为 1。当前 token 没什么可看时, 多余的注意力倒在开头几个 token 上, 它们被训练成了垃圾桶 (sink)。\nStreamingLLM 的做法:\n- 永远保留开头 n_sink 个 + 最近 window 个。\n- 位置按 cache 槽位重新编号, 所以 K 必须存未旋转的版本。',
      question: 'SinkCache 里的 K 为什么要存 pre-RoPE 的, 和普通 KV cache 正好相反?',
      code: 'llm_infer/m16_attention_sinks/sink_cache.py',
      points: [
        {
          title: 'sink 是 softmax 的排污口',
          key: true,
          body: '开头的 token 对所有后续位置都可见, 于是被学成了 "注意力没处放时就倒这儿"。\n逐出它们: softmax 的分母骤变, 其余权重被迫整体膨胀。这是训练时从未见过的分布, 输出立刻漂掉。\n和语义无关, 换成别的开头 token 也一样。',
        },
        {
          title: '有界 cache',
          body: 'cache 条目恒 ≤ n_sink + window。超预算时逐出槽位 n_sink, 也就是窗口里最老的那个, sink 那几个永远不动。',
        },
        {
          title: '位置按槽位重编号',
          body: '位置取 cache 槽位 $0, \\dots, L-1$, 而不是它在数据流里的绝对位置。这样永远不会超过训练长度。\n代价是每步要把 cache 里的 K 整体重转一遍 (O(L·D)), 所以 K 只能存旋转前的。',
        },
      ],
      links: [
        { from: 'KV cache (m01)', to: 'SinkCache', body: '从 O(T) 无限增长变成 O(n_sink + window) 有界。' },
        { from: 'RoPE', to: 'positions = arange(L)', body: '相对位置只看槽位差, 中间 token 被逐出后"距离"就被压缩了。' },
        { from: 'select_blocks 的 forced', to: '稀疏 decode', body: '稀疏选块同样强制保留第 0 块, 是同一个 sink。' },
      ],
      sourceRows: [
        { concept: '逐出规则', code: 'm16_attention_sinks/sink_cache.py:SinkCache.append', takeaway: 'keep = [0:n_sink] + [n_sink+1:], 永远去掉槽位 n_sink。' },
        { concept: '重编号', code: 'm16_attention_sinks/sink_cache.py:stream_step', takeaway: 'pos = arange(L); K 每步按当前槽位现转, query 放在最后一个槽位。' },
        { concept: '不是长上下文', code: 'window 之外的 token', takeaway: '被逐出的内容彻底丢失。sink 保证的是"不崩", 不是"记得"; 要记得就得保留 KV (offload) 或者去检索。' },
        { concept: '随机权重上看不到', code: 'TinyLM 的 attention', takeaway: '未训练模型的注意力近乎均匀, 没有 sink 现象。这一章的效应要在植入 sink 的合成数据上才成立。' },
      ],
      snippetTitle: 'SinkCache 一步',
      snippet: `def append(self, k_raw, v):                 # k_raw: 未旋转的 K
    K = concat([self.k, k_raw]); V = concat([self.v, v])
    if len(K) > self.n_sink + self.window:
        keep = r_[0:self.n_sink, self.n_sink + 1:len(K)]   # 去掉窗口里最老的
        K, V = K[keep], V[keep]
    self.k, self.v = K, V

pos = arange(len(cache))                    # 位置 = 槽位, 不是绝对位置
K = apply_rope(cache.k, positions=pos)      # 每步整体重转
q = apply_rope(q, positions=pos[-1:])`,
      source: [`${I}m16_attention_sinks/sink_cache.py:SinkCache`, `${I}m16_attention_sinks/sink_cache.py:stream_step`],
      run: 'python -m llm_infer.m16_attention_sinks.demo',
    },

    'infer-tree-speculation': {
      title: 'EAGLE 与树形投机 · 让一次 target forward 验更多',
      subtitle: '链式 draft 第一个猜错, 后面 $K-1$ 个全废。两条改进路线: 猜得更准, 或者一次验更多分支。',
      tldr: '- EAGLE: draft 吃 target 白送的 hidden state、共享 target 的 lm_head, 接受率更高。\n- 树形投机: 每个节点留下 draft 的 top-k 候选, 长成一棵树。target 用 tree attention mask 一次验完, 接受与 target greedy 一致的最长路径。\n同样的验证预算: 链 $K=15$ 每次 1.86 token, 树 [3,2,1] 2.58。',
      question: '同一深度的兄弟节点为什么共享同一个 RoPE 位置?',
      code: 'llm_infer/m17_eagle_speculative/eagle.py · llm_infer/m19_tree_speculation/tree_spec.py',
      points: [
        {
          title: '特征级 draft',
          body: '$\\hat{h}_{t+1} = \\mathrm{Draft}(h_t, \\mathrm{emb}(x_{t+1}))$, 再过 target 自己的 lm_head。第 1 步的 $h$ 是 target 验证时白送的真特征, 之后用 draft 自己的 $\\hat{h}$。误差逐步累积, 所以越靠后的槽位越难接受。',
        },
        {
          title: 'tree mask',
          key: true,
          body: 'anc[i, j] = "j 是 i 的祖先或就是 i"。\n- 每个节点: 看到的恰好是从根到自己那一条链, 等价于把每条路径各跑一次顺序 decode。\n- 兄弟之间: 互不可见, 同深度共享同一个位置。\n一次 forward 就把整棵树验完。',
        },
        {
          title: '算力换延迟',
          body: 'widths=[3,2,1] 长出 16 个节点, 只要 1 次 target 调用 + 3 次 draft 调用。\ndecode 是带宽受限的, 多验几个 token 几乎不花额外时间。但大 batch 下算力吃紧时, 这笔账就不再免费。',
        },
      ],
      links: [
        { from: 'm07 accept_greedy', to: 'accept_tree', body: '从"沿一条链比对"变成"从根往下走, 命中哪个孩子就进哪个"。' },
        { from: 'TinyLM.forward(return_hidden)', to: 'EagleDrafter.propose', body: '验证那次 forward 白送了下一轮 draft 要用的特征。' },
        { from: 'gather_kv', to: 'KV 回滚', body: '只留被接受路径上那几行 KV; K 已是 post-RoPE, 挑行就行。' },
      ],
      sourceRows: [
        { concept: '树形状', code: 'm19_tree_speculation/tree_spec.py:tree_shape', takeaway: 'BFS 编号, 父先于子; [3,2,1] → 1+3+6+6 = 16 个节点。' },
        { concept: '树 mask', code: 'm19_tree_speculation/tree_spec.py:tree_mask', takeaway: '上下文全可见, 树内只看祖先和自己; positions = n_ctx + depth。' },
        { concept: '最长路径', code: 'm19_tree_speculation/tree_spec.py:accept_tree', takeaway: '同一父节点的 top-k 候选互不相同, 所以每层至多命中一个孩子。' },
        { concept: 'EAGLE drafter', code: 'm17_eagle_speculative/eagle.py:EagleDrafter', takeaway: '只换 drafter, 验证循环与接受规则完全复用 m07.speculative_decode。' },
        { concept: '实测', code: 'm19 demo', takeaway: '同样的验证 token 数: chain $K=15$ 每轮接受 1.40 (1.86 token/次 target 调用), tree [3,2,1] 1.80 (2.58)。' },
      ],
      snippetTitle: '树 mask + 接受最长路径',
      snippet: `anc = eye(n, dtype=bool)
for i in range(1, n):
    anc[i] |= anc[parents[i]]             # 祖先闭包 (BFS 序: 父先于子)
mask = where(anc, 0, -inf)                # 节点只看祖先 + 自己
logits = target.forward(tokens, kv, positions=n_ctx + depth, mask=mask)

t_pred, path = logits.argmax(-1), [0]
while True:
    kids = where(parents == path[-1])
    hit = [k for k in kids if tokens[k] == t_pred[path[-1]]]
    if not hit: break                     # 断了: t_pred[path[-1]] 就是纠错 token
    path.append(hit[0])`,
      source: [`${I}m19_tree_speculation/tree_spec.py:accept_tree`, `${I}m19_tree_speculation/tree_spec.py:tree_mask`, `${I}m17_eagle_speculative/eagle.py:EagleDrafter`],
      run: 'python -m llm_infer.m19_tree_speculation.demo',
    },

    'infer-kv-offload': {
      title: '分层 KV offload · GPU → CPU → 磁盘',
      subtitle: '多轮对话每一轮的 prompt 都是整段历史, GPU 装不下所有用户的历史。被挤出去的 KV 别扔, 降级存起来。',
      tldr: '1 个 token 的 KV (128 KiB):\n- 走 PCIe 25 GB/s 搬: 约 5 µs\n- 重算: 约 125 µs\n搬比算便宜 25 倍。但每次加载还有一笔固定延迟, 链路一慢, 命中反而比重算更贵。\n所以每一层都要单独判断 "加载还是重算"。\n以下时间全部来自代价模型, 不是实测。',
      question: '为什么"多加一层缓存"有时反而让 TTFT 变差?',
      code: 'llm_infer/m20_kv_offload/tiered_cache.py',
      points: [
        {
          title: '降级, 不是丢弃',
          body: '每一层一个 LRU。GPU 层溢出的最旧 block 降到 CPU, CPU 溢出降到磁盘; 再次命中时提升回 GPU 层。',
        },
        {
          title: '链断了后面就用不了',
          body: '沿 hash 链逐块查找, 在第一个全层都 miss 的块处停下。后面的块即使还在也不可用: 第 4 块的 hash 标识的是"前 4 块的整段前缀", 前缀 KV 不全, 后面接不上。',
        },
        {
          title: '交叉点 n*',
          key: true,
          body: '- load = 固定延迟 + n × 每块搬运\n- recompute = n × 每块重算\n两条线交在 $n^*$ = latency / (每块重算 − 每块加载)。demo 里磁盘层的 $n^* = 1.54$ 块: 只命中 1 块时重算更快。\n所以策略必须是 "两者取小", 而不是 "命中就加载"。',
        },
      ],
      links: [
        { from: 'm04 PrefixCache', to: 'TieredKVCache', body: '同样的链式 block hash, 只是多了几层存储。' },
        { from: 'Tier.bandwidth_GBps', to: 'load_ms', body: '这里带宽单位是 GB/s (byte); 对照 P/D 分离那一章的 Gbps (bit), 差一个 8。' },
        { from: 'use_load', to: 'TTFT', body: '接一条 0.5 GB/s 的慢远端: "命中就加载"要 68.9 ms, "取小"回到 34.8 ms。' },
      ],
      sourceRows: [
        { concept: '代价模型', code: 'm20_kv_offload/tiered_cache.py:CostModel', takeaway: 'load = latency + bytes / bandwidth; recompute = tokens / 8000 tok/s。' },
        { concept: '降级', code: 'TieredKVCache._insert', takeaway: '溢出的最旧 block 递归插入下一层, 而不是直接删掉。' },
        { concept: '逐层决策', code: 'm20_kv_offload/tiered_cache.py:TieredKVCache.serve', takeaway: 'use_load = (load ≤ recompute), 每层各判断一次。' },
        { concept: 'demo 结果 (模拟)', code: '8 users × 6 turns', takeaway: '平均 TTFT: 只有 GPU 51.5 → 加 CPU 层 34.8 → 再加磁盘层 19.5 ms。' },
      ],
      snippetTitle: 'serve: 逐层"加载 vs 重算"',
      snippet: `where = self.lookup(block_hashes(prompt))     # 每个命中块在哪一层; 首个 miss 处停
ttft = recompute_ms(len(prompt) - len(where) * bs)
for tier, nb in zip(tiers, hits_per_tier):
    load = tier.latency_ms + nb * block_bytes / tier.bandwidth   # 固定延迟 + 搬运
    recompute = recompute_ms(nb * bs)
    ttft += min(load, recompute)              # 慢链路 / 命中太少 → 重算更快
self.store(prompt)                            # 命中的提升回 GPU 层, 溢出的逐层降级`,
      source: [`${I}m20_kv_offload/tiered_cache.py:serve`, `${I}m20_kv_offload/tiered_cache.py:CostModel`],
      run: 'python -m llm_infer.m20_kv_offload.demo',
    },

    'infer-moe-serving': {
      title: 'MoE serving · 专家并行与 EPLB',
      subtitle: '专家分布在多张卡上, 每层都要同步, 所以一步的耗时等于最忙那张卡的耗时。',
      tldr: '- dispatch: token 经 all-to-all 发到专家所在的 rank。\n- combine: 算完再发回, 按 gate 加权。\n路由一倾斜, 热专家所在 rank 的负载能到均值的 2.95×, 平均利用率只剩 34%。\nEPLB 给热专家加冗余副本、把它的 token 拆开: max/mean 压到 1.02×, 输出逐位不变。',
      question: '为什么只靠"重新摆放专家"不够, 必须复制热专家?',
      code: 'llm_infer/m21_moe_serving/moe.py',
      points: [
        {
          title: '看 max, 不看 mean',
          key: true,
          body: 'EP 每层 combine 都要等齐所有 rank, 所以一步的耗时由最忙的那张卡决定。max/mean = 2.95 意味着平均 rank 只有 1/2.95 ≈ 34% 的时间在干活, 其余都在等。',
        },
        {
          title: '两步贪心',
          body: '① 反复给"每副本负载"最大的那个专家再加一个副本。② 把所有 slot 按负载从大到小, 依次放到当前最轻且还有空位的 rank 上。',
        },
        {
          title: '不可分的热点',
          body: '最热专家的负载 / 平均 rank 负载 = 1.42 > 1: 它一个就超过一张卡该分到的量。\n单个专家不可分, 怎么摆都不可能均衡 (贪心无副本只能做到 1.48×)。只能复制它, 并拆分它的 token。',
        },
      ],
      links: [
        { from: '阶段 2 MoE 路由', to: 'MoELayer.route', body: '同一个 top-k 路由, 这里只关心它在多卡上造成的负载分布。' },
        { from: '阶段 3 EP all-to-all', to: 'ep_forward', body: '训练可以用辅助 loss 压倾斜; 推理时路由已经定死, 只能靠放置。' },
        { from: 'eplb_placement', to: 'slot_of', body: '有副本的专家把自己的 token 轮流分给各个副本。' },
      ],
      sourceRows: [
        { concept: '路由', code: 'm21_moe_serving/moe.py:MoELayer.route', takeaway: 'gate 只在被选中的 k 个 logit 上做 softmax。' },
        { concept: 'dispatch/combine', code: 'm21_moe_serving/moe.py:ep_forward', takeaway: 'rank_load[r] = 发到 rank r 的 (token, k) 分配数。' },
        { concept: 'EPLB', code: 'm21_moe_serving/moe.py:eplb_placement', takeaway: 'n_rep[argmax(load / n_rep)] += 1, 一次加一个副本。' },
        { concept: '输出不变', code: 'max|Δ| = 0', takeaway: 'EPLB 只改"逻辑专家 → 物理 slot"的映射, 路由和 gate 都没动, 所以输出逐位相同。' },
        { concept: 'demo 结果', code: '16 专家 / 4 rank / top-2', takeaway: 'max/mean: 连续摆放 2.95× → 贪心无副本 1.48× → EPLB 加 4 个副本 1.02×。' },
      ],
      snippetTitle: 'EPLB 两步贪心',
      snippet: `n_rep = ones(E)
for _ in range(n_redundant):
    n_rep[argmax(expert_load / n_rep)] += 1     # ① 最热的"每副本负载"再加一个副本

slot_expert = repeat(arange(E), n_rep)
slot_load = (expert_load / n_rep)[slot_expert]  # 副本间均分 token
for s in argsort(-slot_load):                   # ② 从重到轻
    r = argmin(where(rank_free > 0, rank_load, inf))
    slot_rank[s] = r; rank_load[r] += slot_load[s]; rank_free[r] -= 1`,
      source: [`${I}m21_moe_serving/moe.py:eplb_placement`, `${I}m21_moe_serving/moe.py:ep_forward`],
      run: 'python -m llm_infer.m21_moe_serving.demo',
    },

    'infer-sparse-decode': {
      title: '稀疏注意力 decode · 只读 top-k 个 KV block',
      subtitle: '长上下文 decode 是访存瓶颈: 每步要把全部 KV 读一遍, 而注意力质量集中在很少的 token 上。',
      tldr: 'KV 切成 block, 每块常驻一份很小的摘要:\n- Quest: 逐维 min/max\n- NSA/DSA: 压缩 key 或低维 indexer\n用 q 给摘要打分选 top-k 块, 只对它们做 attention。\n- needle 负载: 读 3.1% 的 KV, 误差就降到 0.023。\n- 随机权重模型: 注意力弥散, 这个前提根本不成立。',
      question: 'Quest 为什么用"上界"打分, 而不是直接用 block 的均值?',
      code: 'llm_infer/m22_sparse_attention/sparse_attention.py',
      points: [
        {
          title: '廉价的上界打分',
          body: 'Quest: $\\sum_i \\max(q_i \\cdot \\text{kmin}_i,\\ q_i \\cdot \\text{kmax}_i) \\ge$ 该 block 内任何一个 $q \\cdot k$。\n上界保证含有 "针" 的块不会被低估而漏掉。代价是偏松 (demo 平均松 6.7×)。不漏比估得准重要。',
        },
        {
          title: '两块必须保留',
          body: '第 0 块 (attention sink) 和最后一块 (最近的 token) 永远选中, 并且算在 k 之内。丢掉第 0 块, softmax 的分母就崩了。',
        },
        {
          title: '前提是注意力足够尖',
          key: true,
          body: '收益完全来自 "质量集中在少数 block"。\n- k 从 4 加到 8 (读 1.6% → 3.1% 的 KV): 误差从 0.83 断崖掉到 0.023。针全进 top-k 的那一刻, 其余 KV 就无关紧要了。\n- 随机权重的 TinyLM: 注意力近乎均匀, Quest 打分并不优于随机选块。\n这个方法的收益来自数据分布, 不是算法本身。',
        },
      ],
      links: [
        { from: 'm02 KV block', to: 'block 摘要', body: '分页 KV 天然按 block 组织, 摘要大约是 KV 的 1/16 ~ 1/8。' },
        { from: 'attention sink', to: 'select_blocks 的 forced', body: '首块必须留, 否则 softmax 分母崩。' },
        { from: '阶段 2 DSA', to: 'lightning indexer', body: 'DeepSeek-V3.2 用训练出来的低维 indexer 按 token 选; 这里只有免训练的选块。' },
      ],
      sourceRows: [
        { concept: 'Quest 上界', code: 'm22_sparse_attention/sparse_attention.py:quest_upper_bound', takeaway: '$q_i$ 为正取 kmax, 为负取 kmin, 逐维求和。' },
        { concept: '选块', code: 'm22_sparse_attention/sparse_attention.py:select_blocks', takeaway: 'forced = {0, nb−1}, 其余按分数取 top-(k−2)。' },
        { concept: '评测', code: 'm22_sparse_attention/sparse_attention.py:evaluate', takeaway: '相对 L2 误差 + 选中 block 覆盖的真实注意力质量 (recall), 两个指标一起看。' },
        { concept: 'demo 结果', code: 'T=4096, 256 blocks', takeaway: 'k=4 (1.6%) 误差 0.83 → k=8 (3.1%) 误差 0.023; 同样读 3.1%, 随机选块误差 1.06。' },
      ],
      snippetTitle: '打分 → 选块 → 只读选中的 KV',
      snippet: `Kb = K.reshape(nb, bs, d)
kmin, kmax = Kb.min(1), Kb.max(1)                   # 常驻显存的摘要 (nb, d)
scores = maximum(q * kmin, q * kmax).sum(-1)        # ≥ block 内任何 q·k

forced = {0, nb - 1}                                # sink 块 + 最近块
rest = [b for b in argsort(-scores) if b not in forced]
blocks = sorted(list(forced) + rest[:k - 2])

idx = (blocks[:, None] * bs + arange(bs)).ravel()   # 只 gather k·bs 个 token
out = softmax(q @ K[idx].T / sqrt(d)) @ V[idx]`,
      source: [`${I}m22_sparse_attention/sparse_attention.py:quest_upper_bound`, `${I}m22_sparse_attention/sparse_attention.py:select_blocks`],
      run: 'python -m llm_infer.m22_sparse_attention.demo',
    },

    'infer-test-time-compute': {
      title: '推理时计算 · 模型不变, 多花 token 换正确率',
      subtitle: '读完你能说清 best-of-N、多数投票、PRM beam、budget forcing 各靠什么挑答案, 以及哪一种错它们都消不掉。',
      tldr: '模型不换, 推理时多花 token 也能涨正确率。三条路:\n- 并行: 采 $N$ 条再挑。best-of-N 让 ORM 看终点, 多数投票取最终答案的众数。\n- 按步搜索: PRM 给每一步打分, 错步当场剪掉。\n- 串行: 写完再自查, budget forcing 控制想多久。\n多采样只能消掉随机错。模型的众数本身就错时, 要越过它得靠外部判分器。',
      question: '多采几条再投票, 为什么有的题反而越投越错?',
      code: 'llm_infer/m23_test_time_compute/{tts.py,demo.py} (每一步的采样复用 m10 的 sample)',
      points: [
        {
          title: '一步错, 整条错',
          body: '$K$ 步的推理题, 每步答对 $p$, 整条答对只有 $p^K$。$p=0.75$、$K=4$ 时只剩 0.316。一步错了, 后面每步都在错的值上"正确地"算。\n换更大的模型很贵。另一条路是推理时多花 token: 同一个模型, 多写几条、多查几遍。\n玩具任务: 200 道 4 步算术题, T=1 时单条正确率 0.275。',
        },
        {
          title: '投票消不掉系统性的错',
          key: true,
          body: '错分两种:\n- 粗心错: 换个样本就可能对。多采几条, 众数会落在正确答案上。\n- 误解: 模型的众数本身就是错的。32.5% 的题带一个陷阱步, "看错运算符"的 logit 3.5 高过正确的 3.0。\n- 多数投票: 只用模型自己的分布, 在 0.71 饱和 (N=16 0.706, N=64 0.710)。陷阱题上 N 越大越稳定地错: 0.254 → 0.108。\n- best-of-N: 让外部判分器 (ORM) 挑, 能从少数派里认出正确答案, N=64 到 0.970。',
        },
        {
          title: 'PRM 在错步出现时就剪掉',
          body: 'best-of-N 要为错链的全部 $K$ 步付费, 到终点才判。PRM beam 每步给候选打分, 错步当场剪掉, 省下的 token 给别的候选。\n- 28 token: PRM beam 0.865, 同 token 的 best-of-7 0.747。\n- 52 token: 0.830 vs 0.849, 打平甚至略输。\n大预算输在判分噪声: 候选一多, 总有错步被噪声抬高。PRM 无噪声 ($\\sigma=0$) 时同配置到 0.970。',
        },
        {
          title: 'budget forcing: 截断就崩, 延长才涨',
          body: '到 $B$ 个 token 强行截断; 模型想停但没到 $B$, 就追加 "Wait" 再查一遍。\n- B=2: 0.055。第一遍都没写完, 只能交中间值。\n- B=4 到 8: 都是 0.240。一次重写要能把后面几步写完, 预算不够就白查。\n- B=64: 0.815。无陷阱题 1.000, 陷阱题只有 0.431。\n不干预时模型平均只想 6.5 token 就停, 正确率 0.305。自查看不见自己的误解, 预算再多也改不掉。',
        },
        {
          title: '诚实边界',
          body: '- greedy 太好: 玩具里随机错只来自采样噪声, greedy (T=0) 就有 0.675, 恰好等于无陷阱题比例。真实模型的 greedy 也犯随机错, 不要读成"greedy 就够了"。\n- 判分器是程序化的: ORM / PRM = 真值 + $\\mathcal{N}(0, 0.5^2)$ 噪声。真实系统里它们是训练出来的, 也有自己的盲区。\n- token 账偏乐观: 没算判分器的前向。真实 PRM beam 每步要给 width × expand 个候选各跑一次 PRM。\n- 自查规则写死: 发现率 0.5、想停概率 0.6。o1 / R1 的长思考是 RL 训出来的。',
        },
      ],
      links: [
        { from: 'm10 sample', to: 'sample_step', body: '模型的每一步 = 在 6 个候选上用 m10 的 sample 抽一个 token; T=0 即 greedy。' },
        { from: '奖励模型 (阶段 4)', to: 'orm / prm', body: 'ORM 用"最终答案对不对"做标签训练打分头; PRM 要逐步标签, 人工标 (PRM800K) 或 Monte-Carlo 自动标 (Math-Shepherd)。' },
        { from: 'prm_beam_search', to: 'beam search (解码一章)', body: '同样是每步只留 top-width, 排序依据从累计 log 概率换成了 PRM 累计分。' },
      ],
      sourceRows: [
        { concept: '模型的一步', code: 'm23_test_time_compute/tts.py:step_dist', takeaway: '6 个候选: 正确值 + 4 种粗心错 (±1, ±10) + 看错运算符。logits 为 [3.0, 0.3×4, 0.3 或 3.5], 陷阱步上"看错"那一项是 3.5。' },
        { concept: 'best-of-N', code: 'm23_test_time_compute/tts.py:best_of_n', takeaway: 'N 条链各让 ORM 打一次分, 取最高那条的最终答案。' },
        { concept: 'self-consistency', code: 'm23_test_time_compute/tts.py:majority_vote', takeaway: '最终答案取众数, 不需要判分器, 但要求答案能比较相等。开放式回答没法直接投票。' },
        { concept: 'PRM beam', code: 'm23_test_time_compute/tts.py:prm_beam_search', takeaway: '每条 beam 采 expand 个下一步, 按 PRM 累计分留 width 条。token = expand + (K−1)·width·expand。' },
        { concept: '长思考', code: 'm23_test_time_compute/tts.py:think', takeaway: '写完再一遍遍自查, 发现错就从那一步重写。min_tokens = max_tokens = B 就是 budget forcing。' },
        { concept: '自查的盲区', code: 'm23_test_time_compute/tts.py:believes_ok', takeaway: '真对的认对; 陷阱步上自己的误解也认对。粗心错以 0.5 的概率被发现。' },
      ],
      snippetTitle: 'PRM beam: 错步在出现的那一步就被剪掉',
      snippet: `beams = [(0.0, [])]
for i in range(K):
    cand = []
    for score, chain in beams:
        v = chain[-1] if chain else x0
        for _ in range(expand):
            nxt = sample_step(prob, i, v)              # 1 个 token
            cand.append((score + prm(prob, i, v, nxt), chain + [nxt]))
    beams = sorted(cand, key=lambda c: -c[0])[:width] # 只留 PRM 累计分最高的 width 条
answer = beams[0][1][-1]`,
      source: [
        `${I}m23_test_time_compute/tts.py:best_of_n`,
        `${I}m23_test_time_compute/tts.py:majority_vote`,
        `${I}m23_test_time_compute/tts.py:prm_beam_search`,
        `${I}m23_test_time_compute/tts.py:think`,
      ],
      run: 'python -m llm_infer.m23_test_time_compute.demo',
    },

    'infer-multi-replica-routing': {
      title: '多副本路由 · 前缀缓存与负载的拉锯',
      subtitle: '读完你能解释命中率最高的路由为什么 TTFT 反而最差, 以及负载阈值怎么取舍两者。',
      tldr: '8 个引擎副本各有自己的前缀缓存, 副本之间不共享 KV。\n- 轮询 / 最少负载: 不看内容, 多轮对话的下一轮多半落到别的副本, 历史整段重新 prefill。\n- 前缀感知: 送到缓存里前缀最长的副本, 命中率 59.3% → 93.9%。但热门 system prompt 会把流量全吸到一个副本上。\n治法: 最佳副本比最闲副本多积压超过阈值, 就让位给最闲的。',
      question: '多个副本各有前缀缓存, 请求该发给谁?',
      code: 'llm_infer/m27_multi_replica_routing/{router.py,demo.py} (每个副本复用 m05 的 RadixCache)',
      points: [
        {
          title: '不看内容, 历史就白算',
          body: '多轮对话第 $k$ 轮的 prompt = system prompt + 前 $k-1$ 轮全部历史。\n轮询和最少负载下, 下一轮多半落到别的副本。那里只缓存了 system prompt, 40.7% 的 prompt token 要重新 prefill。\n前缀感知路由问每个副本"你缓存里有这个请求多长的前缀", 送给最长的。miss 降到 6.1%。',
        },
        {
          title: '命中率最高, TTFT 最差',
          body: '1084 个请求共用 4 个 system prompt, 流行度 48% / 24% / 16% / 12%。\n新对话只在最先缓存它 system prompt 的副本上命中 512 token, 于是永远选它:\n- 8 个副本只有 3 个在干活, 占比 52% / 32% / 16%。\n- 负载 max/mean 4.18。\n- 平均 TTFT 105.954 s (最少负载 0.469 s)。这是队列模型过载发散的结果, 只说明"会爆"。',
        },
        {
          title: '负载阈值兜底',
          key: true,
          body: '最佳副本的积压比最闲副本多出阈值, 就放弃缓存, 改走最少负载。被分流过去的副本随后也缓存了热点前缀, 热点自然复制开。\nthr=1s: 命中 83.6%, max/mean 1.19, TTFT 0.314 s, 比最少负载低 33%。\n阈值是个旋钮:\n- 太松 (4s): 命中 93.4%, 但 max/mean 1.45, TTFT 0.689 s。\n- 太紧 (0.05s): 命中跌到 72.9%, TTFT 回升到 0.308 s, 在往最少负载退化。\n本例 TTFT 最优在 0.5s (0.284 s)。',
        },
        {
          title: '探测必须只读',
          body: '路由要探测全部 8 个副本。RadixCache.match 会 split 树、刷新 LRU: 拿它探测, 没被选中的 7 个副本的 LRU 也被搅乱。\n所以单独写一个只读的 prefix_len, 走法相同, 不改树。探测不等于使用。',
        },
        {
          title: '诚实边界',
          body: '- TTFT 来自代价模型: 每副本一条 FIFO 队列, prefill 1000 tok/s、decode 100 tok/s, 刻意调慢到 30~40% 利用率, 排队才看得见。真实引擎做 continuous batching, 过载表现为 TPOT 变差。\n- 命中率是真跑的: 每副本 24k token 的 RadixCache。轮询的 59.3% 主要来自 512 token 的 system prompt。\n- 路由器看得太清楚: 这里直接读副本的真实 radix tree。SGLang 的 router 维护近似树, 不知道副本的驱逐。',
        },
      ],
      links: [
        { from: 'm05 RadixCache', to: 'Replica.cache', body: '每个副本直接复用 radix cache, 命中长度是真跑出来的。' },
        { from: 'match (会改树)', to: 'prefix_len (只读)', body: '同一种走法, 去掉 split 和 LRU 刷新, 探测才不会扰动没被选中的副本。' },
        { from: 'EPLB (MoE serving)', to: '负载阈值', body: '同一个病: 热点把负载压在一处。EPLB 复制热专家; 这里靠分流, 让热点前缀在多个副本上各缓存一份。' },
      ],
      sourceRows: [
        { concept: '只读探测', code: 'm27_multi_replica_routing/router.py:prefix_len', takeaway: '与 RadixCache.match 相同的走法, 但不 split、不刷新 last_used。' },
        { concept: '三种策略', code: 'm27_multi_replica_routing/router.py:route', takeaway: 'prefix: best = argmax(prefix_len, −backlog); backlog[best] − min backlog > threshold 就改走最少负载。平手按轮转打散。' },
        { concept: '模拟', code: 'm27_multi_replica_routing/router.py:simulate', takeaway: 'TTFT = 到达时积压 + miss 部分的 prefill。命中至少留最后 1 个 token 重算, 才有 logits。' },
        { concept: '请求流', code: 'm27_multi_replica_routing/router.py:chat_stream', takeaway: '200 段对话 × 3~8 轮, 4 个 512-token system prompt, 流行度 ∝ 1/rank, 轮间思考 Exp(15 s)。' },
      ],
      snippetTitle: '前缀优先, 积压超阈值就让位',
      snippet: `def route(replicas, tokens, threshold_s):
    loads = [r.backlog_s for r in replicas]
    least = argmin(loads)
    best = argmax(prefix_len(r.cache, tokens) for r in replicas)  # 只读探测, 平手选负载低的
    if loads[best] - loads[least] > threshold_s:
        return least                                              # 热点副本排太长: 放弃缓存
    return best`,
      source: [
        `${I}m27_multi_replica_routing/router.py:route`,
        `${I}m27_multi_replica_routing/router.py:prefix_len`,
        `${I}m27_multi_replica_routing/router.py:simulate`,
      ],
      run: 'python -m llm_infer.m27_multi_replica_routing.demo',
    },
  },
}
