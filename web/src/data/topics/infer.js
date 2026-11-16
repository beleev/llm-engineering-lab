import { canonicalizeRelated } from '../../utils/consolidate.js'
// 阶段 5 · llm_infer 章节内容。
// 本文件持有本阶段全部 16 章的完整 page 对象。
// 贯穿全阶段的一条主线: decode 是带宽瓶颈, 不是算力瓶颈 ——
// 所以几乎每一项优化都在省 KV 的"搬运"和"存储", 而不是省乘法。
const I = 'llm_infer/'

const topic = {
  stage: 'infer',
  chapters: [
    { route: 'infer-prefix-radix', label: '前缀复用 · hash 与 radix', hint: '公共前缀的 KV 怎么只算一次, 让后来的请求直接用' },
    { route: 'infer-structured-output', label: 'token 级语法约束', hint: '让模型只能吐出合法 JSON, 而且每步几乎不花时间' },
    { route: 'infer-quant-awq', label: 'INT4 · AWQ · KIVI', hint: '把权重和 KV 压到 4 bit 还不失真, 关键在怎么分组' },
    { route: 'infer-kv-footprint', label: 'KV 体积 · MHA→MLA', hint: '算清每个 token 的 KV 占几个字节, 它决定并发上限' },
    { route: 'infer-attention-sinks', label: 'Attention sinks', hint: '对话无限长时, 怎么让 KV cache 不涨还不崩' },
    { route: 'infer-tree-speculation', label: 'EAGLE 与树形投机', hint: '让 draft 猜得更准, 再让 target 一次验一整棵树' },
    { route: 'infer-kv-offload', label: '分层 KV offload', hint: 'GPU 装不下的历史 KV 往哪放, 搬回来还是重算' },
    { route: 'infer-moe-serving', label: 'MoE serving · EPLB', hint: 'MoE 分到多张卡后, 热门专家压垮一张卡怎么办' },
    { route: 'infer-sparse-decode', label: '稀疏注意力 decode', hint: '每步只读一小部分 KV, 还不漏掉关键 token' },
    { route: 'infer-test-time-compute', after: 'infer-decode-control', label: '推理时计算 · best-of-N 与长思考', hint: '模型不换, 多花 token 怎么换正确率, 哪种错换不掉' },
    { route: 'infer-multi-replica-routing', label: '多副本路由', hint: '请求发给哪个副本: 既要缓存命中, 又不能压垮一台' },
  ],
  pages: {
    // ---------------- 主线五章 ---------------- //
    'infer-kv-memory': {
      widgets: [],
      title: 'KV 与缓存内存 · 推理优化从"别重算"开始',
      subtitle: '读完你能说清 KV cache 存的是什么、省掉了什么、没省掉什么, 能用一个公式算出一张卡同时放得下多少条序列, 并说清分页到底买到了什么。',
      tldr: '- KV cache: 生成每个新 token 时, 前面 token 的 K 和 V 都不会变, 所以算一次存起来, 之后只算新 token 的。demo 里生成同一段话, 累计送进模型的 token 从 688 个降到 37 个 (少了 18.6 倍)。\n- 分页: 不重算了, 显存就成了瓶颈, 因为每条请求的 cache 随长度线性增长、事先不知道多长。分页把 cache 切成固定大小的块按需分配。它不让 attention 更快, 它让同一张卡能装下更多请求。',
      question: '有了 KV cache 之后, 生成每个新 token 是不是就变成固定开销 (O(1)), 和上下文有多长没关系了?',
      evolution: {
        title: '从整块预留到按页记账',
        subtitle: '根问题: KV cache 省掉了重算, 代价是按 token 线性吃显存, 而每条请求会写多长事先不知道。',
        steps: [
          { name: '按 max_len 连续预留', pain: '(原点) LLaMA-7B fp16 每个 token 的 KV 要 0.5 MiB, 4K 上下文一条请求就是 2 GiB', fix: '每条序列一开始就占满 max_len 的连续显存; 只写 100 个 token 也占着 4096 个位置' },
          { name: '按需拼接', pain: '预留的位置大多空着, vLLM 论文测得只有 20~40% 装着真数据', fix: '用多少长多少: 每步把新 K/V 拼接 (concat) 到数组尾部 (m01 的写法); 拼接要整块拷贝 O(T), 序列结束还留下大小不一的空洞' },
          { name: '分页 + 页表', year: 2023, pain: '拷贝随长度变贵, 序列结束后留下的空洞大小不一, 填不上', fix: '切成定长 block, 每条序列一张页表把位置映射到物理块; 每条最多空 bs−1 个槽' },
          { name: '引用计数 + 惰性回收', pain: '多条请求带同一段 system prompt, 各存一份一模一样的 KV', fix: '每块记一个引用计数 ref_count, 多条请求可共用同一物理 block; free 只把块放回队列, 内容留到被覆盖, 前缀复用靠这一点' },
        ],
      },
      code: 'llm_infer/m01_kv_cache · llm_infer/m02_paged_attention (前缀复用另见「前缀复用 · hash 与 radix」)',
      points: [
        {
          title: '只有 K/V 值得存',
          body: '模型生成一句话时是一个 token 一个 token 往外吐的。每吐一个新 token, 都要把前面所有 token 再过一遍注意力。KV cache 就是把这一遍里可以复用的部分存起来。\n先说哪部分可以复用。注意力层会给每个 token 算三个向量: q (query, 它要去问什么)、k (key, 它能被谁问到)、v (value, 被问到时交出什么)。语言模型用因果 mask, 意思是第 $i$ 个 token 只能看到它前面的 token, 看不到后面的。所以不管后面再来多少新 token, 旧 token 在每一层算出来的 k 和 v 都不会变。不变的东西算一次存起来就够了, 这就是 KV cache 里的 K 和 V。\n- prefill: 第一次把整段 prompt 送进模型, 顺手把每一层的 k、v 都存下来。\n- decode: 之后每生成一个 token, 只算这一个新 token 的 q、k、v, 把它的 k、v 追加到 cache 末尾。\nq 为什么不存: 一个 token 的 q 只在它自己那一步用来向前面的 token 提问, 之后再也用不上。',
        },
        {
          title: 'cache 没省掉的那部分',
          key: true,
          body: '有了 cache, 生成一个新 token 仍然不是固定开销: 新 token 的 query 还是要和前面全部 t 个 key 逐个做点积, 这一步的工作量随上下文长度 t 线性增长。\n先看 decode 一步到底要做哪些事。没有 cache 时, 每生成一个 token, 要把前面全部 t 个 token 重新送进模型: 每一层都要给这 t 个 token 重新算 q、k、v (这叫 K/V 投影, 就是几次矩阵乘法), 过 attention, 再过 MLP (每层后面那个两层的前馈网络)。有了 cache, 旧 token 的 k、v 直接从 cache 里读, 它们的投影和 MLP 都不用再算了, 每层只需要给新 token 这一个算。\n没省掉的是 attention 本身: 新 token 要决定看前面哪些 token, 就必须拿自己的 q 和 cache 里全部 t 个 k 各做一次点积, 再按权重把 t 个 v 加起来。t 越长, 这一步要读的 cache 越多。所以长上下文的 decode 仍然是 O(t) 的: 每生成一个 token, 都要把整个 cache 从显存里读一遍, 只为算这一个 token, 瓶颈在访存而不在算力。\n后面几章的 GQA/MLA (让 cache 本身变小)、KV 量化 (每个数用更少的字节存)、稀疏读取 (每步只读 cache 的一部分), 都在解决这件事。',
        },
        {
          title: '分页买的是显存, 不是速度',
          body: '有了 cache, 显存就成了新问题: 每条请求最后会写多长, 开始时不知道。最省事的做法是按最大长度 max_length 给每条请求预留一整段连续显存, 但大多数请求根本写不满, 预留的位置多数空着。分页 (PagedAttention) 借的是操作系统虚拟内存的办法。\n- 怎么切: 把 KV 按固定长度切成 block (demo 里一个 block 放 bs 个 token), 所有 block 放进一个公用的池子。\n- 怎么找: 每条请求有一张页表 block_table, 记着它的第几页放在池子里哪个 block。要取位置 pos 的 KV, 先算它在第 pos // bs 页, 查页表得到物理 block, 再取这个 block 里第 pos % bs 个槽。写成代码就是 pool[table[pos // bs], pos % bs]。\n- 代价: 多了一次查表, 单次 attention 反而略慢。\n- 收益: 请求写多长就要多少 block, 不再按 max_length 预留。浪费只剩每条请求最后一个 block 里没填满的几个槽。\n所以分页买到的是显存利用率: 同一张卡能同时放更多请求, batch 更大, 吞吐才上得去。',
        },
      ],
      links: [
        { from: 'sample.py 逐 token 生成', to: 'KV cache', body: '左边是训练阶段的 sample.py, 它也是一个 token 一个 token 地生成 (自回归), 但每生成一个新 token 都把整段前文重新送进模型跑一遍。右边的 KV cache 做的是同一件事, 只是把前文每层的 K、V 存下来, 每步只算新 token 那一个。' },
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: 'BlockManager 是管理 KV block 池子的记账员, can_allocate 回答 "现在还有没有足够的空闲 block 给这条请求"。调度器 (Scheduler) 每一步接新请求 (_admit) 之前都先问它; 空闲块不够就不接, 甚至要把正在跑的某条请求踢出去腾地方, 这叫抢占。' },
        { from: 'free_list 归还顺序', to: 'PrefixCache 的 LRU', body: 'free_list 是空闲 block 的队列。一条请求结束时, 它的 block 只是引用计数 ref_count 归零、排到队尾, 里面的 K/V 内容并没有被清掉, 要等这个 block 被重新分配出去、被新内容覆盖才真正消失。前缀复用 (PrefixCache) 靠的就是这一点: 队列的顺序天然就是 "最久没用的先被覆盖", 也就是 LRU。' },
      ],
      sourceRows: [
        { concept: '对照组', code: 'm01_kv_cache/demo.py:generate_with_cache', takeaway: 'demo 用同一个 prompt 跑两遍: 一遍每步重算, 一遍用 cache。两遍生成的 token id 必须逐个相同, 每步 logits (模型对下一个 token 的打分) 最大只差 2.80e-06, 这是 fp32 的舍入误差。累计送进模型的 token 数从 688 个降到 37 个。' },
        { concept: 'KV 字节公式', code: '2 · n_layer · T · D · sizeof(dtype)', takeaway: '2 是 K 和 V 各一份, n_layer 是层数, T 是 cache 里的 token 数, D 是每个 token 的 K (或 V) 向量维度, sizeof(dtype) 是每个数占几个字节。demo 用 4 层、D=64、T=37、fp32 (4 字节) 代入算出 75776 B, 断言它等于 numpy 数组实际的 nbytes。代入 LLaMA-7B fp16 (32 层、D=4096、2 字节), 每个 token 的 KV 是 0.50 MiB, T=4096 的一条请求就是 2.00 GiB。' },
        { concept: '地址翻译', code: 'm02_paged_attention/paged_attention.py:paged_attention', takeaway: 'paged_attention 先把要找的位置 pos 翻译成 (block_table[pos // bs], pos % bs): 先查页表找到物理 block, 再找块内的槽。这和操作系统把虚拟地址翻译成物理地址是同一个办法。demo 断言分页 KV 上算出的 attention 和连续 KV 算出的逐元素相同, $\\max|\\Delta|$ = 0.00e+00。' },
        { concept: '碎片上界', code: 'BlockManager.stats', takeaway: 'stats 报告池子里有多少 block 在用、多少空闲。因为只有每条序列的最后一个 block 可能没填满, 每条序列最多浪费 bs−1 个槽; demo 里 11 个 token 占了 3 个 block 共 12 个槽, 实测浪费 1/12, 上界是 3。对比之下, 按 max_len 连续预留的方案, vLLM 论文测得只有 20~40% 的 KV 显存装着真数据。' },
        { concept: '要不到块', code: 'MemoryError: 需要 25 个新 block, 只剩 3 个空闲', takeaway: 'demo 最后申请一条 100 个 token 的请求, 需要 25 个新 block, 池子只剩 3 个, BlockManager 抛出这个 MemoryError。调度器收到它就知道显存不够了, 这就是触发抢占 (把某条正在跑的请求暂停、腾出 block) 的信号。' },
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
      subtitle: '读完你能说清调度器每一步怎么决定谁上场, 以及 "每步 token 预算" 这一个旋钮, 怎么同时拧动 TBT (同一条请求相邻两个输出 token 的间隔) 和 TTFT (从请求到达到吐出第一个 token 的时间)。',
      tldr: 'GPU 要把很多请求攒成一批 (batch) 一起算才划算。本章讲这一批怎么组。\n- 静态 batch: 凑齐一批一起跑, 要等最长的那条结束才散。短请求陪跑, 新请求在外面干等。\n- 连续批 (continuous batching): 调度粒度从 "一个请求" 缩到 "一步前向"。每做一步前向就重问一遍谁进谁出: 结束的先退出, 能进来的马上补上。\n新问题出在 prefill 上。prefill 是第一次把整段 prompt 送进模型, 顺手把每个 token 在每层算出的 K、V 向量存下来, 这份存货叫 KV cache; 之后的 decode 每步只生成一个新 token, 要反复读它。一条长 prompt 的 prefill 一步算完, 所有正在 decode 的用户都得停下来等它。所以再加两道:\n- chunked prefill (分块 prefill): 把长 prompt 切成小块, 每步只算一块, 和 decode 混进同一个 batch。demo 的代价模型里, decode 用户的最大卡顿从 297 ms 降到 52 ms。\n- P/D 分离: 干脆把 prefill 和 decode 两类负载放到不同的机器 (节点) 上, 代价是 KV cache 要走网络传过去。',
      question: '排在最前面的新请求因为放 KV 的显存 block 不够进不来, 这一步调度器该干什么? 能不能干脆什么都不做, 等下一步?',
      evolution: {
        title: '调度粒度从请求缩到一步',
        subtitle: '根问题: GPU 要攒一批一起算才划算, 请求却长短不一、随时到达, prefill 和 decode 还抢同一份算力。',
        steps: [
          { name: '静态 batch', pain: '(原点) 一次前向只算一条太亏, GPU 要一批一起算', fix: '凑齐一批, 跑到最长那条结束才散; 短请求陪跑, 新请求在门外干等' },
          { name: '连续批 (Orca)', year: 2022, pain: '短请求结束后槽位空转, 新请求要等整批结束', fix: '每步重问谁进谁出; block 不够就抢占最年轻的 (收回它的 KV 显存, 让它回队首以后重算), 队首进不来就先跑 decode' },
          { name: '分块 prefill (Sarathi)', year: 2023, pain: '一条 1024-token 的 prompt 一步算完, decode 用户卡 297 ms', fix: '每步算的 token 数设上限 (预算), 长 prompt 切块和 decode 混批: 卡顿降到 52 ms, 代价是它自己的 TTFT 从 276 涨到 445 ms' },
          { name: 'P/D 分离 (DistServe)', year: 2024, pain: '分块只是在时间上错开, prefill 和 decode 仍在同一张卡上抢算力', fix: 'prefill 和 decode 分到两组节点, KV 走网络传: 7B 模型 4K 上下文的 KV 有 2 GiB, 400 Gbps 链路传 42.9 ms' },
        ],
      },
      code: 'llm_infer/m03_continuous_batching · m06_chunked_prefill · m15_pd_disaggregation',
      points: [
        {
          title: 'batch 只记谁算几个 token',
          key: true,
          body: '调度器每一步返回一个 batch, 它就是一个列表 [(seq, n), ...]: 每一项是一条请求 seq, 加上这一步给它算几个 token n。一个列表说清了全部:\n- n>1: 这条请求在做 prefill, 或者是 prefill 的一个 chunk (长 prompt 切开后的一块)。\n- n=1: 这条请求在 decode, 这一步只生成 1 个新 token。\n两者可以出现在同一步。所以没有 "prefill 步" 和 "decode 步" 之分, 只有 "这一步谁算几个 token"。\n每条请求靠 num_computed 记进度, 它表示这条请求已经有 KV 的 token 数。每步算完, num_computed 加上 n; 追平整条序列的长度 num_tokens 时, 这一步的输出才是一个真正的新 token, 才去采样。\nm03 demo 的打印里, P5 表示 prefill 5 个 token, D 表示 decode 1 个。宽松池子下, step 4 时 s1 完成退出, step 5 新请求 s4 立刻以 P5 补位, 5 条请求共 11 步跑完。',
        },
        {
          title: '永远不能返回空 batch',
          body: '调度器有两条队列: waiting 放还没开始的请求, 先来先到; running 放正在跑的请求。KV cache 按固定长度切成 block 存放, 每接一条新请求, 都要先从 block 池里分到足够的 block。\n假设队首那条请求要 6 个 block, 池子只剩 2 个空闲, running 里有 3 条在 decode。如果调度器这时直接返回一个空 batch, 会发生一串连锁反应:\n- running 里的请求这一步没被安排, 不前进。\n- 没有请求前进, 就没有请求结束。\n- 没有请求结束, block 就永远不释放。\n- block 不释放, 队首永远进不来。\n引擎每一步都在空转, 永远跑不完, 这叫活锁。\n修法只有一行, 在不分块、prefill 优先的那条路径上: _admit(budget) or _schedule_running(budget)。意思是先试着接新请求做 prefill (_admit); 一个都接不进来、返回空列表时, 就转去给 running 里的请求排 decode (_schedule_running)。prefill 进不来就落到 decode。分块的那条路径本来就先排 decode、再用剩下的预算接新请求, 见下方骨架。\n抢占也服从同一个原则。抢占是 block 不够时把一条正在跑的请求暂时踢出去、腾出显存。调度器只踢最年轻的, 保证最老的那条总能前进。m03 demo 用只有 7 个 block 的紧池子验证: 发生 2 次抢占, 17 步跑完, 全程没有一步是空 batch。',
        },
        {
          title: 'token 预算就是 TBT 上限',
          body: '每一步算的 token 数有个上限, 叫 token 预算。这个预算直接决定了用户等下一个 token 最多要等多久。\n先说两个指标。TBT (time between tokens) 是同一条请求相邻两个输出 token 的间隔, 用户看到的 "输出卡一下" 就是 TBT 突然变大。TTFT (time to first token) 是从请求到达到吐出第一个 token 的时间。\nm06 用一个代价模型估算每步耗时: step_ms = 20 + 0.25 × 本步 token 数。这个式子在说: 每一步有 20 ms 固定开销 (读一遍权重、启动 kernel), 每多算 1 个 token 再加 0.25 ms。它是模型, 不是实测。按这个模型, 每步算多少 token 有上限, 每步耗时就有上限, 用户两个 token 之间的间隔也就被封顶。\ndemo 的场景: 4 条序列正在 decode, 每步只算 4 个 token, 一步 21 ms。这时来了一条 1024-token 的长 prompt。\n- prefill 优先, 整段一步算完: 这一步要算 1024 个 token, 光这一步就是 276 ms, decode 用户全都停下来等, 最大卡顿 297 ms。那条长请求自己的 TTFT 是 276 ms。\n- 分块, 每步预算 128 token: 每步先给 4 条 decode 各 1 个 token, 剩下的预算切一块 prefill 混进去。一步最多 128 个 token, 耗时最多 52 ms, decode 用户最大卡顿就是 52 ms。代价是长 prompt 要分很多步才算完, 它的 TTFT 涨到 445 ms。\n- 预算 512 时介于两者之间: 最大卡顿 148 ms, 长请求 TTFT 319 ms。\n分块不是白赚: 卡顿降下来, 代价是那条长请求的首 token 更晚。预算越小, 卡顿越小, 长请求的首 token 越晚。',
        },
      ],
      links: [
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: 'BlockManager 管着 KV 池里所有 block 的分配和回收。调度器 _admit 接新请求前, 先调 can_allocate 问一句: 这条请求需要的 block (前缀命中、可以共用的 block 不用新分), 池子里还够不够。不够就不接, 让它继续在队首等。' },
        { from: 'max_batch_tokens', to: '每步耗时 → TBT', body: 'max_batch_tokens 就是每步的 token 预算。它封顶每步耗时, 也就封顶了 TBT。它是 TBT 与 TTFT 之间的那个旋钮: 调小, decode 用户卡顿小, 但长 prompt 的首 token 更晚; 调大, 长 prompt 的首 token 早, 但 decode 用户卡得更久。vLLM 里对应的参数叫 max_num_batched_tokens。' },
        { from: 'KVLink.send', to: 'P/D 分离', body: 'P/D 分离里, prefill 节点算完把 KV cache 按 byte 打包发给 decode 节点, KVLink.send 按带宽算出传输要多久。坑在单位: 网卡带宽按 bit 标 (Gbps, 每秒 10 亿 bit), KV 按 byte 算, 1 byte 是 8 bit, 少除一个 8 就把传输时间低估 8 倍。分离不会让单条请求更快, 还多了一次传输; 收益是消除干扰, 在同样的延迟要求下每张卡能承载更多请求, 这个指标叫 goodput (满足延迟要求的有效吞吐)。' },
      ],
      sourceRows: [
        { concept: '每步重组 batch', code: 'm03_continuous_batching/scheduler.py:Scheduler.schedule', takeaway: 'schedule 每一步重新组 batch, 有两种策略, 由 chunked_prefill 开关决定。关掉时是 prefill 优先: 先接新请求整段 prefill, 接不进来才排 decode。打开时是 decode 优先: 先给 running 里每条请求它还欠的 token (decode 欠 1 个, 没算完的 prefill 接着算下一块), 剩余预算切给新请求的 prefill chunk。刚抢占过的那一步不接新请求, 免得腾出来的 block 马上又被占掉。' },
        { concept: '抢占', code: 'Scheduler._preempt', takeaway: '这里是 recompute (重算) 式抢占: block 全还给池子, num_computed 归 0, 放回 waiting 队首。已生成的文本 output_ids 原样保留, 回来时连同 prompt 一起重新 prefill, 把 KV 重算一遍。m03 demo 里 s3 被抢占后重新进来是 P4: 2 个 prompt token 加 2 个已生成的 token 一起重算。' },
        { concept: 'TBT 代价模型', code: 'm06_chunked_prefill/chunked_prefill.py:simulate', takeaway: 'simulate 用 m03 的真调度器回放一段负载, 但每步耗时不是测出来的, 是代价模型算的: step_ms = 固定开销 + 每 token 开销 × batch token 数 (demo 里是 20 ms 和 0.25 ms)。它记下每个输出 token 产出的时刻, 相邻两个相减是 TBT, 第一个减去到达时刻是 TTFT。因为每步 token 数被预算封顶, 每步的 token 预算决定 TBT 上限。' },
        { concept: '分块不改结果', code: 'chunked_prefill(lm, ids, chunk)', takeaway: '把 prompt 每 chunk 个 token 一块送进模型, 前面块的 KV 留在 cache 里, 后面的块只算自己这几行的 Q, 再对已缓存的加本块的 K/V 做 attention。因果 mask 让每个 token 只看自己和前面的 token, 前面 token 的 KV 不依赖后面, 所以分块和整段一次算完结果相同。demo 用 64 个 token 对拍: chunk ≥ 16 时 logits 与 KV 都和整段 prefill 逐位相同; chunk=8 有极小的差 (logits 1.67e-06, KV 2.86e-06), 只来自分块求和顺序。attention 分数矩阵的峰值也从 64 × 64 = 4096 个数降到 8 × 64 = 512 个。' },
        { concept: '干扰有多大', code: 'm15_pd_disaggregation/pd.py:KVLink', takeaway: 'm15 在 TinyLM 上用墙钟实测: 同卡混跑时, 被一次 400 个 token 的长 prefill 插队的那一步, decode 用户要等 decode 一步加整段 prefill 的时间, 延迟从 0.253 涨到 4.68 ms (18× 抖动)。分离后 prefill 在别的节点上, 回到 0.253 ms。m15 把这个间隔叫 ITL, 和 TBT 是同一个量。毫秒数随机器变化。KVLink 则是另一半账: 分离后 KV 要过网络, 它按字节数除以每秒字节数算传输时间。' },
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
      widgets: [],
      title: '解码加速与采样 · 更少 target 调用, 更可控的分布',
      subtitle: '读完你能解释 "小模型先猜、大模型验证" 为什么不让输出变差, 采样参数的执行顺序为什么会改结果, 以及 beam search 找到的 "最可能" 为什么不等于最好。',
      tldr: '生成阶段 (decode) 每一步只产出 1 个 token, 却要把整份模型权重从显存读一遍, 时间都花在读权重上, 算力大量闲着。\n投机解码 (speculative decoding) 利用这一点: 让一个便宜的 draft (小模型) 先连猜 $K$ 个 token, 再让 target (原来的大模型) 一次 forward 把这 $K+1$ 个位置一起验证, 猜对几个就收几个。输出分布和只用 target 完全一样, draft 只影响速度。\n- greedy 模式: 逐位比较 draft 的 token 和 target 的 argmax。\n- 采样模式: 用 rejection sampling (按 target 和 draft 给这个 token 的概率之比决定收不收) 保证分布不变。\n另一侧是采样本身: 温度、top-k、top-p 这些参数都是 logits 上的一串后处理, 执行顺序不同结果就不同。',
      question: '小模型猜得再差, 为什么最后的输出分布还能和只用大模型采样时一模一样?',
      evolution: {
        title: '读一遍权重, 多收几个 token',
        subtitle: '根问题: decode 每步只出 1 个 token, 却要把整份权重从显存读一遍, 算力大半闲着。',
        steps: [
          { name: '逐 token decode', pain: '(原点) decode 受带宽限制: 读一遍权重只算出 1 个 token', fix: '一步一个 token; 延迟 = 生成长度 × 读一遍权重的时间' },
          { name: '分块并行解码', year: 2018, pain: '一次 forward 本可以并行验很多位置, 却只产出 1 个 token', fix: '给模型加几个头一次猜后面 K 个, 再并行验证; 要改模型结构, 还要训练这些头' },
          { name: '独立 draft + greedy 验证', year: 2022, pain: '改 target 结构、重训额外的头, 部署成本高', fix: '小模型连猜 K 个, target 一次验 K+1 个; 加速 1.12× 到 4.36×, 取决于 draft' },
          { name: 'rejection sampling', pain: '逐位比 argmax 只适用 greedy; 采样时全收 draft, 分布就成了 draft 的', fix: '以 $\\min(1, p/q)$ 接受 (p、q 是 target 和 draft 给这个 token 的概率), 拒绝就从 $\\max(0, p-q)$ 重采样: 输出分布严格等于 target 的' },
        ],
      },
      code: 'llm_infer/m07_speculative_decoding · m10_sampling (EAGLE / 树形投机、语法约束、attention sinks 各有独立章节)',
      points: [
        {
          title: '一次验 K+1 个槽位',
          body: '先说 draft 和 target 怎么配合。target 是我们真正要用的模型, 它的 KV cache 里存着到目前为止输出的所有 token (除了最后一个 out[-1], 它已经确定但还没喂进去)。draft 是便宜的小模型, 从同样的上下文往后连猜 $K$ 个 token $d_0, \\dots, d_{K-1}$。\n验证只需要 target 跑一次 forward: 一次喂进 [out[-1], $d_0, \\dots, d_{K-1}$] 这 $K+1$ 个 token。因果 mask 保证第 $i$ 行只看得到 $d_{<i}$, 所以这一次 forward 得到的 $K+1$ 行 logits, 和一个一个顺序 decode 得到的完全一样。第 0 行说的是 out[-1] 之后该是什么, 拿它和 $d_0$ 比; 第 1 行和 $d_1$ 比; 以此类推。\n接受规则 (greedy): 从前往后比, 接受最长的正确前缀, 第一个猜错的位置用 target 自己的 argmax 替换掉, 然后停。全部猜对时, 第 $K$ 行 logits 就是 "看完全部 $K$ 个 draft 之后的下一个 token", 已经算出来了, 白送 1 个 bonus token。所以每次 target 调用至少产出 1 个 token (不会比不用 draft 更差), 最多 $K+1$ 个。\n加速比取决于 draft 猜得准不准。demo 用 $K=4$ 生成 48 个 token, 不用 draft 要调用 target 48 次:\n- draft 和 target 完全一致 (上限): 只要 11 次 target 调用 (4.36×), 每轮接受 4.00/4\n- 蒸馏式 draft (用 target 权重加 10% 噪声模拟): 26 次 (1.85×), 每轮接受 0.88/4\n- 随便找个小模型 (独立随机的 1 层模型): 43 次 (1.12×), 每轮接受 0.14/4\n几种 draft 的输出都和 target 单独 greedy 逐 token 相同。这里的加速只按 target 调用数算, 没计 draft 自己的开销。',
        },
        {
          title: 'rejection sampling 让它无损',
          key: true,
          body: '采样模式下不能再用 "逐位比 argmax": 采样本来就允许选非最大概率的 token, 要保证的是输出 token 的分布和只用 target 采样时一样。\n记 $p$ = target 给这个 token 的概率, $q$ = draft 给的概率。draft 抽出一个 token 之后:\n- 接受: 以 $\\min(1, p/q)$ 的概率接受它。target 觉得它比 draft 觉得的更可能 ($p \\ge q$) 就一定收; target 觉得没那么可能, 就只按 $p/q$ 的比例收一部分。\n- 拒绝: 不收时, 从归一化后的残差分布 $\\max(0, p-q)$ 里重新抽一个 token (只在 target 比 draft 更看好的那些 token 里抽), 然后停止, 后面的 draft token 全部作废。\n为什么两步合起来恰好还是 $p$:\n- 直接接受: 一个 token 被 draft 抽中又被接受的概率是 $q\\cdot\\min(1, p/q) = \\min(p, q)$。\n- 拒绝后补上: 离 $p$ 还差的那截正好是 $\\max(0, p-q)$, 由重采样补齐。\n两部分加起来就是 $p$。所以 draft 再差也只影响速度, 不影响输出分布。\ndemo 用一个 16 个 token 的小词表, 枚举所有前缀算出 target 的精确分布, 再各采样 2500 条对照, 用 $\\chi^2$ 检验 (衡量采样频率和精确分布差多远, 越大越不像): 正确规则的 $\\chi^2 \\le 13.6$, 低于临界值 37.7, 和 target 直接采样的噪声同量级; 故意改成 "全收 draft" 的错误规则, 立刻涨到 585。',
        },
        {
          title: '采样是一串顺序敏感的 filter',
          body: '模型最后给出 V 个 logit (每个 token 一个分数), 采样就是从里面挑一个。m10 的 sample 把这件事做成一串 filter: 每个 filter 都是 logits 进、logits 出, 被砍掉的 token 置成 −inf (softmax 之后概率为 0), 所以可以任意串联。顺序是: rep penalty → temperature → top-k → top-p → min-p → Gumbel-max。\n- repetition penalty: 已经出现过的 token 扣分。正 logit 除以 penalty, 负 logit 乘以 penalty, 两种情况都是往更小走。要是不分正负一律除, 负 logit 除以大于 1 的数反而变大: demo 里 token 9 (logit −1) 的概率从 0.0076 抬到 0.0144, 反而鼓励重复。\n- temperature: logits 除以 $T$ 再 softmax。$T$ 越大分布越平, 越小越尖; $T$ = 0 就是 greedy。\n- top-k: 只留分数最大的 k 个。\n- top-p: 按概率从大到小累加, 留下累计概率刚够 p 的最小集合。\n- min-p: 只留概率不低于 min_p 乘最大概率的 token, 模型越确定砍得越狠。\n- Gumbel-max: 最后真正抽一个 token 的办法, 见下面表格。\n温度排在 top-p 前面, 所以 top-p 看到的是缩放后的分布: 同样的 top-p, $T$ 越大留下的 token 越多, $T \\lt 1$ 则相反。各框架的顺序不完全一致, 同一组参数跨框架结果可能不同。',
        },
        {
          title: '搜索式解码: beam 找的是"最可能"',
          body: 'greedy 每步只选当下概率最高的 token, 选错一步就回不来。beam search 每步在 $w \\times V$ 个候选 (现有的 $w$ 条前缀各接 V 个 token) 里留下累计 log 概率最高的 $w$ 条前缀, 最后输出得分最高的那条; 代价约是 $w$ 倍的 decode。\nTinyLM (随机权重的小模型) 上 10 个 prompt × 20 token:\n- 平均每 token 的 log 概率 (logP/token, 越接近 0 越 "可能"): greedy −2.620, $w=32$ 到 −2.255。beam 确实找到了更可能的序列。\n- 重复率 rep-2 (重复出现的两字组合占比): 0.453 → 0.563。越 "可能" 越复读。\n- 多样性: 同一 prompt 的 8 条 beam 候选两两平均只差 3.5/20 个位置, 8 条采样差 19.7。beam 的候选共享大段前缀。\n趋势不单调: $w=4$ 的 rep-2 是 0.447, 比 greedy 还低。随机权重模型本身就爱复读, greedy 的 rep-2 已经 0.453; 真实模型上 "beam 越宽越退化" 更明显。',
        },
        {
          title: 'beam 不是精确搜索, 还偏爱短句',
          body: '- 会输给 greedy: $w=2$ 在 10 个 prompt 里有 2 个输给 greedy。原因是某一步 greedy 那条前缀的累计分排到第 3, 被剪掉就回不来, 而留下的两条后面走低了。所以 beam 只在平均意义上 $\\ge$ greedy, 不是定理。\n- 偏爱短句: 序列分数是每个 token 的 log 概率累加, 都是负数, 多一个 token 只会更低。demo 给 EOS 的 logit +3 造出一个会停的模型, 不加长度惩罚 ($\\alpha=0$) 时 beam 的输出平均只有 1.3 个 token, 几乎是空回复。\n- 长度惩罚: 最后按 $\\text{score}/\\text{len}^\\alpha$ 选答案。$\\alpha=0$ 是原始累加分; $\\alpha=1$ 是平均每 token 的 log 概率。demo 里 $\\alpha=0.5$ 仍是 1.3 个 token, $\\alpha=1$ 才到 15.7 (greedy 是 11.4)。\n怎么选:\n- 用 beam: 翻译、摘要这类任务, 好答案集中在一个概率峰附近, 找峰就对。\n- 用采样: 开放式对话, 好回答很多且分散, 找峰只会得到最平庸、最重复的那条。',
        },
      ],
      links: [
        { from: 'sample.py 的 temperature/top-k', to: 'm10_sampling', body: '训练阶段 sample.py 里只有温度和 top-k 两个旋钮。到了服务端, 它长成了一整套参数 (temperature、top_k、top_p、min_p、repetition_penalty), m10 的 SamplingParams 就是这套参数, 每个都是采样链上的一个 filter。' },
        { from: 'accept_greedy / accept_sampling', to: 'EAGLE · 树形投机', body: '这两个函数是投机解码的接受规则: 给定 draft 的 token 和 target 的 logits, 决定收几个。后面两章只换 drafter (EAGLE 让 draft 看 target 的特征) 和验证的形状 (树形一次验多条分支), 接受规则原样复用。' },
        { from: 'truncate_kv', to: 'KV 回滚', body: '验证那次 forward 会把 K+1 个 token 的 KV 都写进 cache。被拒绝的 draft token 的 KV 直接用 truncate_kv 截掉, 从不重新 prefill: 因果性保证前面 token 的 KV 和后面的 token 无关, 截掉后面的不影响前面的。' },
      ],
      sourceRows: [
        { concept: 'greedy 接受', code: 'm07_speculative_decoding/speculative.py:accept_greedy', takeaway: '拿 target 每一行的 argmax 和对应的 draft token 逐位比对, 第一个不一致处停下, 那个位置返回 target 自己的 argmax 作为纠错 token。' },
        { concept: '采样接受', code: 'm07_speculative_decoding/speculative.py:accept_sampling', takeaway: '抽一个均匀随机数 $u$, $u < p_t/p_d$ 就接受 ($p_t$、$p_d$ 是 target 和 draft 给这个 token 的概率); 否则从 $\\max(0, p_t - p_d)$ 归一化后重采样并停止。K 个全接受时, bonus token 直接从 target 最后一行的分布里采。' },
        { concept: '期望产出', code: '(1 − α^(K+1)) / (1 − α)', takeaway: '每次 target 调用平均能产出几个 token, $\\alpha$ 是每个 token 被接受的概率。链式接受一断全断, 第 k 个 token 被接受的概率约是 $\\alpha^k$, 所以 $K$ 再大, 产出也被 $1/(1-\\alpha)$ 封顶。这正是树形投机的动机: 预算花在宽度上比花在深度上值。' },
        { concept: '采样顺序', code: 'm10_sampling/samplers.py:sample', takeaway: '按 rep_penalty → /T → top_k → top_p → min_p → Gumbel-max 的顺序调用六个函数。每一步都是 "把被砍的 token 置 −inf", 不改其它 token 的 logit, 所以可以任意串联。' },
        { concept: 'Gumbel-max 是精确的', code: 'argmax(log p + G), G = −log(−log U)', takeaway: '给每个 token 的 log 概率加一个 Gumbel 噪声 $G$ ($U$ 是均匀随机数), 再取 argmax, 得到的 token 和按 multinomial 直接抽是同一个分布。demo 用 TV (总变差距离, 衡量两个分布差多远) 检验: Gumbel-max 0.0064, multinomial 自己的采样噪声 0.0066, 同一水平。好处是只有逐元素运算加一次 argmax, 没有累加和二分搜索, 整个 batch 能用一个 kernel 做完。' },
        { concept: 'beam search', code: 'm24_beam_search/beam.py:beam_search', takeaway: '每条活着的 beam 取 top-$(w+1)$ 个 token 作候选, 多取一个是为了有候选以 EOS 结束时仍能凑满 $w$ 条活 beam; 结束时所有候选按 $\\text{score}/\\text{len}^\\alpha$ 排序, 取第一条。demo 断言 $w=1$ 的输出与 greedy 逐 token 相同。' },
        { concept: '造一个会停的模型', code: 'm24_beam_search/demo.py:EosBiased', takeaway: '随机权重的模型没学过何时结束, 每步 P(EOS) 约 0.5%, 从不停。EosBiased 给 EOS 的 logit 加 3, 之后每步 P(EOS) 约 8.3% (沿 greedy 路径走 24 步的均值), 才看得到 beam 偏爱短句的现象。' },
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
      subtitle: '读完你能说清 FlashAttention 快在哪一步 (不是少算, 是少搬)、Flash-Decoding 为什么要沿 KV 切、CUDA Graph 省的是谁的时间, 以及张量并行一层为什么只同步 2 次。',
      tldr: '这一章的几件事都不改数学, 结果和朴素实现逐元素相等, 省的是搬运和等待:\n- FlashAttention: 朴素 attention 要把 $T \\times T$ 的分数矩阵写进显存再读回来。它把 Q 和 K/V 都切成小块, 用 online softmax (边算边更新最大值、分母和输出) 把各块结果精确地接起来, 从不把 $T \\times T$ 的矩阵写进显存。\n- Flash-Decoding: decode 时每条序列只有 1 个 query, 切不动 Q 就沿 KV 长度切, 让更多 GPU 计算单元同时干活。\n- CUDA Graph: 一步要向 GPU 提交几百个 kernel (GPU 上跑的小程序), 每次提交 CPU 都要花时间; 录一次之后每步只提交 1 次。\n- 推理 TP (张量并行): 每层权重切给多张卡, 一层 (一个 Transformer block) 只需 2 次 all-reduce (各卡把结果加起来的集体通信)。',
      question: 'FlashAttention 的 FLOPs 一点没少, 凭什么还能快?',
      evolution: {
        title: '一层层拆掉等待',
        subtitle: '根问题: GPU 算得快、搬得慢, 还要等 host 一个个提交 kernel; 耗时常卡在搬运和提交上, 不在乘法上。',
        steps: [
          { name: '朴素 attention', pain: '(原点) GPU 显存 (HBM) 大而慢, 片上缓存 (SRAM) 小而快, 耗时看搬了多少字节', fix: '分数和概率两张 $T \\times T$ 矩阵写回 HBM 再读; $T=4096$ 时每头 16M 个元素' },
          { name: 'FlashAttention', year: 2022, pain: '慢在反复搬 $T \\times T$ 的中间矩阵, 不在乘法', fix: 'Q、K/V 分块进 SRAM, online softmax 增量维护最大值和分母, 不物化 $T \\times T$' },
          { name: 'Flash-Decoding', year: 2023, pain: 'decode 只 1 个 query: B=1、H=32 只能派出 32 个线程块, GPU 的 108 个计算单元 (SM) 只用上 30%', fix: '沿 KV 长度切 S 段并行算, 再按 LSE 合并; 代价模型下 S=64 快 3.30×' },
          { name: 'CUDA Graph', pain: 'kernel 够快了, host 每提交一个还要约 10 µs, 一步几百个', fix: '整步 kernel 连同显存地址录一次, 每步 replay 1 次; 形状须固定, 大 prefill 走 eager' },
        ],
      },
      code: 'llm_infer/m09_tensor_parallel · m11_flash_attention · m12_cuda_graph (量化见「INT4 · AWQ · KIVI」)',
      points: [
        {
          title: 'softmax 的分母可以增量维护',
          key: true,
          body: '先说朴素 attention 慢在哪。它先算分数矩阵 $S = QK^\\top$ (每个 query 对每个 key 一个分数, 共 $T \\times T$ 个数), 写进显存; 再读回来做 softmax 得到概率矩阵 $P$, 又写进显存; 再读回来乘 $V$。GPU 的显存 (HBM) 大但读写慢, 片上缓存 (SRAM) 快但很小。$T=4096$ 时每个头的分数矩阵有 16M 个元素, 来回搬这两张大矩阵的时间远超乘法本身。\nFlashAttention 把 Q、K、V 都切成小块, 每次只在 SRAM 里算一小块 (b_q 行 query 对 b_k 行 key) 的分数。难点是 softmax 看起来需要整行的全局分母 (所有 key 的 exp 之和), 一块一块算拿不到。解法: 每个 query 行只带三个运行量: 到目前为止见过的最大分数 $m$、分母 $l$、输出 $O$。来一块新分数就更新它们: 如果这一块里出现了更大的最大值, 先把旧的 $l$ 和 $O$ 整体乘 $\\exp(m_{\\text{old}} - m_{\\text{new}})$ 换成新基准, 再把这一块的贡献加进去。这就是 online softmax, 也是全部技巧。\n于是每个 tile (小块) 只进一次 SRAM, 从不把 $T \\times T$ 的矩阵写进显存。快不是因为算得少 (FLOPs 一点没变), 是因为搬得少。临时工作集恒为 64×64 (一块 b_q × b_k 的分数), 与 $T$ 无关, $T=4096$ 时比 $T^2$ 小 4096 倍。demo 对 30 组配置和朴素实现对拍, 最坏只差 1.78e-06, 是浮点舍入。',
        },
        {
          title: 'LSE 是可以拼接的凭证',
          body: '$\\mathrm{lse} = m + \\log l$, 也就是这一行 softmax 分母的对数 (log-sum-exp)。有了它, 两段 K/V 上各自算出的部分结果就能精确合并成全量结果: 新的 lse 是两个 lse 的 logaddexp, 输出按 $\\exp(\\mathrm{lse}_i - \\mathrm{lse})$ 加权求和。这个权重就是每段占的 softmax 质量。\n- 用两段各自的 $(O, \\mathrm{lse})$ 合成: demo 和全量计算只差 2.68e-07。\n- 直接取两段输出的平均: 错到 0.212, 因为两段的 softmax 分母不相等, 质量大的那段应该占更大的权重。\nFlashAttention 的内层循环本身就在反复做这个合并。ring attention (K/V 分在多张卡上各算各的)、chunked prefill (prompt 分块算)、跨卡分段 KV, 全靠这一个原语: 只传 $(O, \\mathrm{lse})$, 不传 $T \\times T$。',
        },
        {
          title: 'CUDA Graph 省的是 host',
          body: 'decode 每步只算 1 个 token, 每个 kernel (GPU 上跑的一个小程序, 比如一次矩阵乘) 在 GPU 上几十 µs 就跑完。但 host (CPU 这一侧) 每提交一个 kernel 要经过 Python、框架、驱动, 也要约 10 µs; 一步几百个 kernel, GPU 大部分时间在等 CPU 提交。\nCUDA Graph 把一次 forward 提交的 kernel 序列 (连同参数和显存地址) 录下来, 之后每步 host 只发 1 次 replay。kernel 本身一点没变快, 省的是 host 的提交时间。所以形状必须固定: 录的是显存地址, 输入要拷进固定的 static_input 缓冲区, 不能换一个新数组。\n本仓库没有 GPU, launch 开销是显式注入的参数: 0 / 10 / 50 / 200 µs 分别对应加速比 1.3× / 6.4× / 12.2× / 15.0×。开销为 0 时几乎不加速, 正说明省的只是提交。这些加速比是代价模型的产物, 不是实测。\nbatch 越大, 单个 kernel 算得越久, launch 占比越小。所以 vLLM 这类系统只对 decode 用 graph, 大 prefill 通常回退 eager (逐个提交)。',
        },
        {
          title: 'decode 时 FlashAttention 喂不饱 GPU',
          body: 'FlashAttention 的并行单位是 (batch, head, Q 块): 每一个组合派一个线程块去算。\n- 线程块: GPU 派给一个 SM 去跑的一份活。它和 KV 分页的 block 是两回事。\n- SM: streaming multiprocessor, GPU 上的计算单元。A100 有 108 个, 代价模型里每个同一时刻跑一个线程块。\n- prefill: Q 有几千行, 切出来的 Q 块多得是, 线程块够多。\n- decode: 每条序列只有 1 个 query, Q 块只有一个, 并行单位只剩 $B \\times H$。\nB=1、H=32 的长对话 (batch 里 1 条序列, 32 个头) 只派得出 32 个线程块, 108 个 SM 只用上 30%。每个线程块还要独自把 32k token 的 KV 从头读到尾。',
        },
        {
          title: '推理 TP: 一层只同步 2 次',
          body: 'LLaMA-70B 的 fp16 权重 140 GB, 一张 80 GB 的卡装不下。张量并行 (tensor parallel, TP) 把每一层的权重矩阵切成几份, 每张卡只存、只算自己那份; 输入激活 x 每张卡都有完整副本。\n切法有两种: 列切 (按输出维切), 各卡的输出拼起来就是全量, 不用通信; 行切 (按输入维切), 各卡算出的是部分和, 要加起来, 这一次相加就是 all-reduce (每张卡把自己的结果发出去, 所有卡都拿到总和)。先列切、后行切配对, 中间结果天然是切开的, 整个子层只在末尾 all-reduce 一次:\n- attention: Q/K/V 列切, 每卡拿到若干个完整的头, 各自 softmax。O 投影行切, 第 1 次 all-reduce。\n- MLP: gate/up 列切, down 行切, 第 2 次 all-reduce。\n每次 all-reduce 的载荷恒为 $T \\times D$ 个激活, 与卡数无关; 卡越多, 每卡算得越少, 通信占比反而上升。\n列切必须落在 head 边界: softmax 在每个头内部归一化, 一个头切到两张卡上各自 softmax 就错了。demo 把单头硬切成 4 份, 结果错 5.63e-01; 按头切则和不切分的基线只差 1e-7 量级。',
        },
        {
          title: 'Flash-Decoding: Q 切不动就切 KV',
          body: 'Q 只有一行, 切不出并行, 那就把 KV 长度切成 $S$ 段: 每段派一个线程块, 算这一段上的局部结果 $(O_s, \\mathrm{lse}_s)$; 最后一个小 kernel 按 $\\exp(\\mathrm{lse}_s - \\mathrm{lse})$ 加权把 $S$ 段合并。这就是上面的 LSE 拼接, 只是一次合 $S$ 段, 结果严格相等: demo 13 组配置 (含 $S$ 不整除 $T$ 的) 最坏误差 2.06e-07。并行单位从 $B \\times H$ 变成 $B \\times H \\times S$。\n代价模型 (A100 参数: 108 个 SM, 显存带宽 2000 GB/s, 每个 SM 最多拿 1/108 的带宽; 不是实测) 下:\n- B=1、T=32k: $S=64$ 时 909.0 → 275.5 μs, 3.30× (上限是把闲着的 SM 全用上, $108/32 = 3.38$×)。\n- $S=4$ 和 $S=2$ 一样慢 (459.0 μs): 128 个线程块超过 108 个 SM, 要跑 2 波。\n- B=16 最多 1.05×, B=64 为 1.00×: 线程块本来就够, SM 已经满了, 切了只多一个合并 kernel。\n边界:\n- 带宽: 真卡上单个 SM 能拿到的带宽比 1/108 高, 少数 SM 也能吃掉一部分带宽, 真实加速更小。\n- 权重 GEMM: 这里只讲 attention 一个算子。整步 decode 里读权重的矩阵乘 (GEMM) 不受影响, 端到端加速远小于 attention 本身。',
        },
      ],
      links: [
        { from: '朴素 attention', to: 'flash_attention', body: '两者数学上严格相等 (30 组配置最坏差 1.78e-06, 浮点舍入)。省的是对 $N \\times N$ 中间矩阵 (分数和概率) 的反复读写, 不是乘法次数。' },
        { from: 'merge_attention(O, lse)', to: 'chunked prefill / ring attention', body: 'merge_attention 让两段 K/V 上各自算完的结果能精确合并。有了它, prefill 才能把 prompt 切成块分批算 (chunked prefill), KV 才能分在多张卡上各算各的再合 (ring attention)。' },
        { from: 'tp_block', to: '多卡推理', body: 'tp_block 是一个 Transformer block 的张量并行版本: 列切接行切配对, 中间结果天然是切开的, 只在 attention 和 MLP 两个子层的末尾各 all-reduce 一次。多卡推理就是把每一层都这样切。' },
      ],
      sourceRows: [
        { concept: 'online softmax', code: 'm11_flash_attention/flash_attention.py:flash_attention', takeaway: '每个 query 行的 $m$ (最大值) / $l$ (分母) / $O$ (输出) 随 K/V 块增量更新。循环顺序是外层 Q 块、内层 K/V 块 (FlashAttention-2 的顺序), 一个 Q 块的状态在内层循环里常驻 SRAM, 算完只写回一次。' },
        { concept: 'causal 整块跳过', code: 'flash_attention(..., causal=True)', takeaway: '因果 mask 下, 如果一个 K 块里最早的 key 都晚于 Q 块里最晚的 query, 这一块全被 mask, 连矩阵乘都不做, 整块跳过。$T=4096$、块大小 64 时 4096 块里跳过 2016 块 (49.2%); 省计算靠跳块, 不靠 mask 本身。' },
        { concept: '分段合并', code: 'm11_flash_attention/flash_attention.py:merge_attention', takeaway: '按 lse 加权合并两段输出: 先算合并后的 lse, 再把两段的 $O$ 各乘 $\\exp(\\mathrm{lse}_i - \\mathrm{lse})$ 相加。只需传 $(O, \\mathrm{lse})$, 不传 $T \\times T$。' },
        { concept: '图只认地址', code: 'm12_cuda_graph/graph.py:CudaGraph', takeaway: '录一次之后 host 提交从 16 次变 1 次, 输出和逐个提交逐位相同。录的是数组的地址, 所以新输入必须拷贝写进 static_input (static_input[...] = x); 写成 static_input = x 只是给名字重新绑定, 图读的还是旧数组, demo 复现了这个 bug。' },
        { concept: 'TP 切在 head 边界', code: 'm09_tensor_parallel/parallel_linear.py:tp_block', takeaway: '每个 Transformer block 做 2 次 all-reduce, 载荷恒为 $T \\times D$ 个激活 (demo 里 8192 B), 与 tp 无关。切进单个 head 内部会错 5.63e-01, 因为 softmax 要在一个头内归一化, 不能跨 rank 拆开算。' },
        { concept: '一段 KV 的局部结果', code: 'm26_flash_decoding/flash_decoding.py:partial_attention', takeaway: '对一段 KV 算局部 attention, 返回已归一化的输出 $O_s$ 和这一段分母的对数 $\\mathrm{lse}_s$。各段互不依赖, GPU 上是 $S$ 个并行线程块。' },
        { concept: 'split-K + reduce', code: 'm26_flash_decoding/flash_decoding.py:flash_decoding', takeaway: '合并公式: $\\mathrm{lse} = \\log \\sum_s \\exp(\\mathrm{lse}_s)$, $O = \\sum_s \\exp(\\mathrm{lse}_s - \\mathrm{lse}) \\cdot O_s$, 每段按它占的 softmax 质量加权。T 不必整除 S, 最后一段短一点也行。' },
        { concept: '延迟估算', code: 'm26_flash_decoding/flash_decoding.py:decode_latency_us', takeaway: '纯代价模型。线程块数 units = B·H·S; 108 个 SM 一次跑 108 个, 要跑 waves = ⌈units / 108⌉ 波; 每波每个线程块读 T/S 个 token 的 KV, 每个 SM 最多拿到 1/108 的显存带宽。S>1 时再加一个合并 kernel 的开销。' },
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
      title: 'mini-vLLM 引擎 · 把分页 KV、调度、前缀缓存、采样接成一个循环',
      subtitle: '读完你能看懂一个推理引擎每一步在做什么, 并在 vLLM 的 step() 里认出每一行对应前面哪一章。',
      tldr: '引擎的主循环就是反复调用 Engine.step, 每一步固定做四件事: 先调度 (决定这一步哪些请求上场、各算几个 token), 再前向 (把这些 token 送进模型算一遍), 再采样 (挑出下一个 token), 最后后处理 (记进度、登记可复用的前缀、把结束请求占的显存还回去)。\n所有请求的 KV cache (每个 token 在每层算出的 K、V 两组向量, 存起来供后面的 token 复用) 都写进一个全局的分页池子, 不挂在单条请求自己身上。分页的意思是: 池子切成固定大小的 block, 请求写到哪就按需分给它几个 block。所以别的请求算好的公共前缀, 新请求能直接读来用, 这部分 token 真的不做前向。demo 实测:\n- 首轮 5 条请求: 一共 305 个 token 需要 KV, 其中 233 个真算, 72 个靠前缀命中直接复用。\n- 同一批 prompt 再来一遍: 只需真算 105 个。\n- 池子只有 9 个 block 时: 显存不够, 逼出 4 次抢占 (把一条请求暂时踢出去腾显存, 之后再重算它)。输出仍和最朴素的 greedy 生成 (一条一条跑, 每步取概率最大的 token) 逐 token 相同。',
      question: '推理引擎不就是给 model.forward 包一层循环吗? 为什么说它首先是调度器和资源管理器?',
      evolution: {
        title: '省下的算力要对得上账',
        subtitle: '根问题: 请求随时来随时走, 长度和采样参数各不相同; 引擎要在一张卡上同时推进它们, 还不能重复算。',
        steps: [
          { name: '逐请求 generate', pain: '(原点) 请求随时到、长度不一, 每条还带自己的采样参数', fix: '每条请求一个 for 循环跑到底; 同一时刻只服务一条, GPU 大半闲着' },
          { name: '批调度 + 私有 KV', pain: '读一遍权重只喂一条序列', fix: '调度器每步组 batch; 但 KV 挂在各自序列上, 前缀命中了也拿不到别人算好的 KV' },
          { name: '全局分页 pool', pain: '查到前缀命中也拿不到别人的 KV, 命中只是记了一笔账, 公共前缀照样整段 prefill', fix: 'KV 写进所有请求共用的 pool, 前向只算还没有 KV 的 token (start_pos 之后): 首轮 305 个待算只实算 233' },
          { name: '资源对账', pain: '命中了却照样重算的引擎, 输出完全正确, 从结果上看不出来', fix: 'assert 命中 + 实算 = 全部要 KV 的 token, 抢占重算也计入; 30 组随机配置 (fuzz) 的输出全对上' },
        ],
      },
      code: 'llm_infer/full_engine/engine.py · llm_infer/full_engine/model_runner.py · llm_infer/m13_lora_serving/lora.py',
      points: [
        {
          title: 'KV 真的存在共用池子里',
          body: '这条讲 KV 存在哪, 以及为什么前缀命中能真的省掉计算。\n先分清两个阶段: prefill 是第一次把整段 prompt 送进模型, 一次算很多 token; decode 是之后每一步只生成一个新 token。两个阶段算出的 K、V 都要存进 KV cache。\n- 存在哪: 这个引擎不让每条请求各自保管 KV, 而是开一个全局的大池子。ModelRunner 里每层一份 pool[layer], 形状是 (num_blocks, block_size, D): 池子切成 num_blocks 个 block, 每个 block 装 block_size 个 token 的向量, D 是向量的维度。\n- 怎么找: 每条请求有一张页表 block_table, 记着它的第几页放在池子里哪个 block。attention 计算时按页表把整段前缀的 K、V 读回来。\n- 只算新的: ModelRunner.run 收到一个 start_pos, 表示前 start_pos 个 token 的 KV 已经在池子里了。它只对 ids[start_pos:] 这几个新 token 算 Q/K/V, 先把新的 K、V 写进池子, 再让它们对池子里的整段前缀做 attention。\nstart_pos 之前的 KV 直接从 pool 里取, 来源有三: 前缀命中 (别的请求算好的公共前缀)、上一个 chunk (长 prompt 分块算时, 前一块刚写进去的)、之前的 decode 步 (一个个追加进去的)。三种都直接读, 不重算。所以 "命中" 是真的没算, 不只是记了一笔账。\nREADME 自测题里的例子: prompt 共 45 个 token, 前 24 个前缀命中, 预算充足时第一步 start_pos=24, Q 只有 21 行, K/V 经页表读回全部 45 行。',
        },
        {
          title: '难的只有调度那一步',
          key: true,
          body: '引擎主循环 Engine.step 的四步里, 前向、采样、后处理都是固定动作, 给什么算什么。真正要做决定的只有调度: 这一步让哪些请求上场, 每条各算几个 token。\n调度器每一步都重新挑一遍: 结束的请求马上退出, 新请求有空位就补进来, 这叫连续批 (continuous batching)。它要同时管住三件事:\n- block 够不够: 每条请求往下算都要占新的 block。池子不够时, 调度器抢占最年轻 (最晚进来) 的那条: 收回它的全部 block, 让它回到等待队列 waiting 的队首, 以后再重算它的 KV。\n- 队首会不会饿死: waiting 队首如果因为 block 不够进不来, 这一步也不能空着, 必须落到 decode, 让已经在跑的请求继续生成。否则没有请求结束、没有 block 被归还, 队首永远进不来, 引擎原地空转, 这种情况叫活锁。\n- 这一步算多少 token: 每步有一个 token 预算 (demo [1] 是 48), 这一步所有请求的 token 加起来不能超过它。一步算的 token 越多这一步越慢, 所以预算封顶了每步耗时。\ndemo [1] 开着 chunked prefill (分块 prefill, 把长 prompt 切成几块分几步算), 打印里 P 表示 prefill 几个 token, D 表示 decode 1 个。step 2 是 s0:D s1:P39 s2:P8: s0 在 decode, s1 和 s2 各算一块 prompt, 合计 1 + 39 + 8 = 48, 正好用满预算。\n引擎的复杂度全在调度里, 主循环本身只有四行。',
        },
        {
          title: '账必须对得上',
          body: '引擎记两个数, 用来证明前缀复用真的省了计算:\n- tokens_computed: ModelRunner 每做一次前向, 就加上这次真算了几个 token。\n- prefix_hit_tokens: 调度器接入新请求时, 前缀命中了几个 token 就加几, 这些 token 的 KV 直接从池子里拿。\nprefix_hit_tokens + tokens_computed 必须等于全部需要 KV 的 token 数。需要 KV 的 token 数这样算: 每条请求的 prompt 加上生成的输出, 再减去最后采出的那一个 (它后面不再生成, 不用为它算 KV)。发生抢占时, 被踢出的请求回来要把 KV 重算一遍, 这部分重算也计入。README 和 report_stats 的注释只写了没有抢占时的等式, demo 也只在没有抢占的 [1] 里 assert 这笔账。\n为什么要用 assert 卡死这笔账: 一个命中了却照样整段 prefill 的引擎, 输出完全正确, 从结果上看不出任何问题。只有对账能发现它: 它的实算数会多出命中的那部分, 账一对就露馅。所以这条 assert 是 "省下的算力是真的" 的唯一证据。\ndemo [1] 里需要 KV 的 305 个 token, 正好等于实算的 233 个加命中的 72 个。72 怎么来的: 5 条请求共用同一段 system prompt, block 大小是 8, 能整块复用的是前 3 个 block 共 24 个 token。s2、s3、s4 各命中 24 个; s1 和 s0 在同一步进来, 当时 s0 还没登记前缀, 所以 s1 没命中。',
        },
      ],
      links: [
        { from: 'Engine.add_request', to: 'Scheduler.waiting', body: 'add_request 先把 prompt 文本编码 (encode) 成 token id, 再把这条请求放进调度器等待队列 waiting 的队尾。同时引擎给它记下自己的 SamplingParams (温度、top_k、top_p 这些控制采样随机程度的参数), 所以同一个 batch 里每条请求可以用不同的采样方式。demo [4] 就是同一个 prompt 配三套参数。' },
        { from: 'Scheduler._admit', to: 'PrefixCache.match_prefix', body: '调度器 _admit 从等待队列接新请求时, 先调 match_prefix 查这条请求开头有多少个 token 的 KV 已经在池子里, 记为 n_hit。这些 token 不用算, 所以不占这一步的 token 预算; 记录 "KV 已就绪的 token 数" 的 seq.num_computed 直接从 n_hit 起步, 前向只从这里往后算。' },
        { from: 'ModelRunner.run', to: 'm02.paged_attention', body: 'run 只给新 token 算 q (query, 每个 token 拿去和前文逐个比对的向量), 所以 q 只有新 token 那几行。m02 的 paged_attention 再按页表从池子里读回整段前缀的 K/V, 其中也包括别的请求算好的公共部分, 新 token 的 q 和它们一起做 attention。' },
        { from: 'Multi-LoRA (m13)', to: '同一 batch 多个 adapter', body: 'LoRA 是一种微调方式: 不改底模权重, 只额外训练两个很瘦的小矩阵 A 和 B, 合起来叫一个 adapter。很多客户各带一个 adapter 时, m13 不把 adapter 合并进底模: 底模那次 gemm (通用矩阵乘) 全 batch 共享、只算一次, 每个 token 再按自己的 adapter id 取对应的 A/B 补算一小段。m13 demo 的小例子里, 1000 个 adapter 如果各合并出一份完整权重要 24.58 MB, 不合并 (一份底模加 1000 对 A/B) 只要 2.58 MB。' },
      ],
      sourceRows: [
        { concept: '一步 (主循环)', code: 'full_engine/engine.py:Engine.step', takeaway: '先调 scheduler.schedule() 拿到这一步的 batch, 它是一张 [(seq, n)] 清单: 哪条请求、算几个 token。然后对每条请求调一次 runner.run 做前向; 只有 KV 已经追平整条序列的请求才调 sample 采新 token。最后调 postprocess 推进进度、登记前缀、归还结束请求的 block。主干一共四行。' },
        { concept: '只算新 token', code: 'full_engine/model_runner.py:ModelRunner.run', takeaway: 'run(ids, block_table, start_pos) 三个参数: ids 是到本步末尾为止的全部 token, block_table 是这条请求的页表, start_pos 是已有 KV 的 token 数。Q 只有 len(ids) − start_pos 行, 正好是新 token 的个数; K/V 经页表读回, 是完整的 len(ids) 行。返回最后一个位置的 logits (模型对下一个 token 的打分)。' },
        { concept: '中途不采样', code: 'done_prefill = num_computed + n == num_tokens', takeaway: '长 prompt 分块算时, 中间某一个 prefill chunk 的最后一个位置, 模型预测的是 prompt 里本来就有的下一个 token, 没有新 token 可采, 所以这一步给它返回 None。只有这一步算完后整条序列的 KV 都追平了 (num_computed + n 等于 num_tokens), done_prefill 才为真, 才去采样。' },
        { concept: '抢占', code: 'm03_continuous_batching/scheduler.py:Scheduler._preempt', takeaway: '抢占是显存不够时的退路。_preempt 把一条正在跑的请求的 block 全部还给池子, 把它放回 waiting 队首; 已生成的 output_ids 原样保留, 回来后连同 prompt 一起重算 KV。只踢最年轻的, 所以最老的序列永远能往前走, 不会活锁。' },
        { concept: '统计', code: 'Engine.report_stats', takeaway: '返回五项: steps (跑了几步)、tokens_computed (真做前向的 token 数)、prefix_hit_tokens (靠前缀命中跳过的 token 数)、preempt (抢占次数)、pool (block 池的占用情况)。一眼看完这一轮的资源账。' },
        { concept: 'fuzz 随机测试', code: 'demo [5]: 30 组随机配置', takeaway: 'fuzz 是用大量随机配置去撞 bug 的测试。这里随机抽 30 组 block 大小、池子大小、token 预算、是否分块、是否开前缀缓存, 词表里还有 id ≥ 256 的 token, 每组把同一批请求跑两轮。共 214 条请求、触发 34 次抢占, 输出全部与朴素的 generate_greedy 逐 token 相同。没有活锁 (没超过 3000 步上限), 也没有 block 泄漏 (跑完后所有 block 空闲、引用计数全为 0)。' },
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
      subtitle: '读完你能说清多轮对话、few-shot、共享 system prompt 这些公共前缀的 KV 怎么只算一次就给所有请求用, 以及 block hash 和 radix tree 两种查法能命中多少 token 差在哪。',
      tldr: '一个 token 的 K 和 V 只取决于它前面的 token。所以两条请求只要开头一段 token 完全相同 (比如同一段 system prompt), 这一段的 KV 就可以只算一次, 第二条请求直接拿来用。本章讲两种找 "相同前缀" 的办法:\n- m04 链式 block hash: 给每个写满的 KV block 算一个 hash 当索引。只能整块整块地命中: 前缀相同 100 个 token、块大小 16 时只能复用 96 个。\n- m05 radix tree: 把见过的 token 序列存成一棵树, 沿树走到分叉处为止。能命中任意长度, 100 个全中; 代价是分叉落在边中间时要把边切开 (split)。\n容量满了, 只能从树的叶子开始、按最久没用的先删 (LRU) 驱逐。',
      question: '驱逐时为什么只能从叶子开始, 不能直接扔掉最久没用的中间节点?',
      evolution: {
        title: '命中要准, 还要细',
        subtitle: '根问题: 很多请求共用一段前缀 (system prompt、多轮历史), 它的 KV 只取决于前缀本身, 却被每条请求各算一遍。',
        steps: [
          { name: '各自 prefill', pain: '(原点) 同一段 system prompt, 每来一条请求就重算一遍 KV', fix: '不复用, 实现最简单; 公共前缀有 N 条请求就算 N 遍' },
          { name: '按块内容查表', pain: '公共前缀的 KV 被一遍遍重算', fix: '写满的 block 按自身 token 算 hash 建索引; 同一段 token 接在不同前缀后面会误命中' },
          { name: '链式 block hash', pain: '位置 $i$ 的 KV 取决于它前面全部 token, 只 hash 块本身会拿错 KV', fix: '父 hash 参与计算, 一个 hash 管一整段前缀; 只能整块命中: 100 个相同 token 命中 96' },
          { name: 'radix tree (SGLang)', year: 2023, pain: '不满一块的尾巴永远命不中, 只能重算', fix: '边上存一段 token 和等长槽位, 走到最长公共前缀, 100 个全中; 分叉落在边中间就 split' },
        ],
      },
      code: 'llm_infer/m04_prefix_cache/prefix_cache.py · llm_infer/m05_radix_cache/radix_tree.py',
      points: [
        {
          title: '一个 hash 标识一整段前缀',
          key: true,
          body: '$h_i = H(h_{i-1} \\,\\|\\, \\text{block}_i)$。这个式子在说: 第 $i$ 个 block 的 hash, 是把上一个 block 的 hash $h_{i-1}$ 和这个 block 自己的 token 拼在一起再做 hash。所以 $h_i$ 不只代表这一块 bs 个 token, 它代表 "从开头到第 $i$ 块为止" 的整段前缀。\n为什么要这样链起来: attention 让位置 $i$ 的 K 和 V 取决于它前面的全部 token。同样一段 token, 接在不同的前文后面, 算出来的 KV 是不一样的。如果只按块自身的内容做 hash, 两条前文不同的请求会查到同一个 block, 拿到错误的 KV。父 hash 参与计算之后, 前文只要差一个 token, 后面每一块的 hash 都不同, 不会误命中。\n查表的流程: 新请求从第一块开始逐块算 hash、去索引里找, 找到就把那个物理 block 直接挂进自己的页表, 找不到就停下, 后面的全部自己算。查表开销是 O(块数)。demo 里 B 和 A 共享 8 个 token 的 system prompt, B 命中前 2 块, 页表前两页和 A 指向同一批物理 block; C 只有第 1 块和 A 相同, 命中 4 个 token 之后链就断了。\n局限是只能整块命中: 一个 block 要写满了才会登记进索引, 所以公共前缀有 $n$ 个 token 时, 只能命中 $\\lfloor n/\\text{bs} \\rfloor \\cdot \\text{bs}$ 个, 不满一块的尾巴永远复用不到。',
        },
        {
          title: 'radix tree 把命中粒度细到单个 token',
          body: 'radix tree (基数树) 把所有见过的 token 序列存成一棵树: 每条边上存一段 token (不是单个 token), 旁边存一份等长的 KV 槽位列表 slots, 记着这段里每个 token 的 KV 放在池子里的哪一格。\n新请求从根出发, 先按第一个 token 挑子边, 再沿着边逐个 token 比对, 走到对不上为止。走过的那段就是它和历史请求的最长公共前缀, 这一段的 KV 直接复用, 只需要算剩下的后缀。\n分叉可能落在一条边的中间。demo [1] 里树中先有 [1..8], 再插入 [1,2,3,4,9,9], 第 5 个 token 处分开, 于是 [1..8] 这条边被切成 [1,2,3,4] + [5,6,7,8], 新分支 [9,9] 只占两个新槽位。做这件事的是 _split: tokens 和 slots 要在同一个位置同步切成两半; 新切出来的中间节点要继承原节点的 ref_count (有几条正在跑的请求锁着这条路径), 不然锁着下游的请求解锁时会把它减成负数, 或者它会被误驱逐。\n粒度的差别: 前 100 个 token 相同、bs=16 时, 链式 hash 只能命中 6 个整块 96 个 token, radix 命中全部 100 个。demo [3] 的随机负载上, 总 token 3189, radix 命中 1430 (44.8%), 按 block 粒度只命中 1068 (33.5%); 这两个比例只对这份合成负载成立。',
        },
        {
          title: '只驱逐叶子',
          body: 'cache 满了要腾地方时, 只能删树的叶子, 不能删中间节点。\n原因: 一条前缀的 KV 要能用, 必须从根开始连续走下来每一段都在。中间节点的 KV 被它所有后代共用, 先删它, 后代存的 KV 就都失去了前缀, 留着也用不了, 等于全部作废。\n所以 evict 只挑 ref_count 为 0 (没有正在跑的请求锁着它) 的叶子, 按最近一次被访问的时间 (last_access, 代码里的字段叫 last_used) 从旧到新删。一个父节点要等它的子节点都删光、自己变成叶子之后, 才会轮到它。demo [4] 里锁住 5 条路径再驱逐, 树里的 token 从 1759 降到 49, 剩下的正好是这 5 条锁定路径的并集; 解锁后全部驱逐, 回到 0。',
        },
      ],
      links: [
        { from: 'BlockManager.ref_count', to: 'PrefixCache', body: 'BlockManager 给每个物理 block 记一个引用计数 ref_count, 表示有几条请求在用它。PrefixCache 让多条请求共用同一个 block, 靠的就是这个计数: 命中一次加 1, 请求结束减 1, 减到 0 才算没人用、可以回收。' },
        { from: 'RadixCache.match_prefix', to: 'Scheduler._admit', body: 'match_prefix 返回命中了多少个 token (n_hit)。调度器接新请求时, 这 n_hit 个 token 的 KV 已经有了, 不需要再算, 所以不占这一步的 token 预算; 请求的 "已算 token 数" num_computed 直接从 n_hit 起步, 只 prefill 剩下的部分。' },
        { from: 'radix tree', to: '分层 KV offload', body: 'radix tree 的驱逐只是把前缀从 GPU 显存里挤出去。后面分层 offload 那一章不丢弃被挤出的前缀, 而是把它降级到 CPU 内存或磁盘, 再被命中时搬回来。' },
      ],
      sourceRows: [
        { concept: '链式 hash', code: 'm04_prefix_cache/prefix_cache.py:_block_hash', takeaway: '_block_hash 先把父 block 的 hash (parent_hash) 喂进 SHA1, 再喂本 block 的 token。第一个 block 没有父, 用固定的 "<root>" 起头。所以一个 hash 唯一标识 "从开头到第 i 块为止" 的整段前缀, 而不只是这一块。' },
        { concept: '至少留一个', code: 'match_prefix 只匹配 (len−1)//bs 块', takeaway: 'match_prefix 故意把最后一个 token 留出来不命中。因为前缀全命中之后, 还得真算最后一个 token 才能得到 logits (模型对下一个 token 的打分), 否则第一个输出 token 无从采起。cache 里存的是 KV, 不是 logits。' },
        { concept: '最长前缀匹配', code: 'm05_radix_cache/radix_tree.py:RadixCache.match_prefix', takeaway: '从根出发, 用查询的下一个 token 作键在 children 里找到对应的子边, 再沿这条边逐个 token 比较; 比到不同就停, 走到边尾就进下一层。demo 用 300 条随机请求对拍, 每次命中长度都等于和全部历史序列暴力算出的最长公共前缀。' },
        { concept: '边分裂', code: 'RadixCache._split', takeaway: '分叉落在边中间时, _split 把这条边切成两半。edge_tokens 和 slots 是等长的两个列表, 必须在同一个位置切; 新切出来的中间节点要继承原节点的 ref_count, 否则锁着这条路径的请求解锁时会减出负数。' },
        { concept: '驱逐', code: 'RadixCache.evict', takeaway: '只在叶子里挑, 且只挑 ref_count 为 0 的, 按 last_used 最旧的先删。每次要 O(N) 扫一遍全树, 代码里注释写明了这个上限; SGLang 用按访问时间排序的小顶堆。' },
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
      subtitle: '读完你能解释为什么 "下一个字符合法" 和 "下一个 token 合法" 是两回事, 以及为什么合法 token 表要提前算好。',
      tldr: '要让模型输出一定能解析的 JSON, 不靠在 prompt 里恳求, 而是在采样前把会违反语法的 token 的 logit 置成 −inf, 让它们的概率严格为 0。哪些 token 合法由一个有限状态机 (FSM) 决定: 它记着 "现在写到 JSON 的哪一步了", 每吃一个字符就换一个状态。\n难点是模型吐的不是单个字符, 而是多字符 token (比如 「":」、「true」), 一个 token 要在字符级状态机上连走好几步。\n- 离线: 对每个 (状态, token) 组合试走一遍。整段字符都走得通才算合法, 同时记下走完后落在哪个状态。\n- 在线: 每生成一个 token 只查一行表, 把非法 token 的 logit 置 −inf。\ndemo 里查表每步 0.086 µs, 在线现算 18.4 µs, 差 215 倍。',
      question: '每生成一个 token 时, 现场对词表里全部 V 个 token 逐字符试走一遍状态机, 为什么不行?',
      evolution: {
        title: '从请模型守规矩到管住 logits',
        subtitle: '根问题: 下游程序要的是一定能解析的 JSON, 模型的每个 token 却是采样出来的。',
        steps: [
          { name: '在 prompt 里要求格式', pain: '(原点) 每个 token 都是采样的, 写错的概率从来不是 0', fix: '给格式说明和示例; 只能抬高合法输出的概率, 输出一长迟早错一次' },
          { name: '字符级 FSM mask', pain: '概率不为 0, 解析失败只能整段重采', fix: '按状态机把非法字符的 logit 置 −inf, 概率严格为 0; 可模型吐的是多字符 token' },
          { name: 'token 级在线试走', pain: '「":」一个 token 连跨两个状态, 字符级的合法集合对不上 token', fix: '每步对 V 个 token 逐字符试走 FSM; 这是 O(V·len) 的 CPU 活, 夹在两次 GPU 前向之间' },
          { name: '离线编译 mask 表', year: 2023, pain: '词表十几万时, 在线试走吃掉 decode 延迟', fix: 'FSM 状态有限, 离线把每个 (状态, token) 试走一遍存成表; 在线每步只查一行' },
        ],
      },
      code: 'llm_infer/m14_structured_output/grammar.py',
      points: [
        {
          title: '合法性是 token 级的, 不是字符级的',
          key: true,
          body: '字符级的状态机很好写: 当前状态下允许哪些字符, 一查就知道。但模型的词表是多字符的: 真实词表里 「":」 是一个 token, 它一口气跨过 "结束 key" 和 "进入 value" 两个状态; 「true」 是一个 token, 要在状态机上连走 4 步。所以 "下一个字符合法" 推不出 "这个 token 合法": 首字符合法但中间某个字符走不通的 token, 照样非法。\n判定只有一条标准: 这个 token 的整段字符从状态 s 出发能不能全部走通。demo 用两张表记结果: next_state[s, t] 记下 token t 从状态 s 走完后落在哪个状态, 走不通就记 −1; mask_table = next_state ≥ 0, 就是每个状态下哪些 token 合法的布尔表。EOS (结束符) 只在接受态 (JSON 已经完整) 合法。\n每一步的流程: 取 mask_table[s] 这一行, 把非法 token 的 logit 置 −inf, 采样得到 token, 再用 next_state[s, tok] 更新状态。demo 用一个 96 个 token 的词表 (48 个多字符), 状态机有 101 个状态, 表共 9696 项, 其中 21.0% 合法。一个随机权重的 TinyLM 加上这张 mask 采样 100 次, 100 次都是合法 JSON; 同一个模型不加 mask, 100 次里 0 次合法。mask 保证的只是语法, 内容照样是乱码。',
        },
        {
          title: '为什么必须离线编译',
          body: '在线现算的意思是: 每生成一个 token 之前, 把词表里全部 V 个 token 从当前状态逐字符试走一遍, 开销是 O(V·len) (len 是 token 的平均字符数)。这是纯 CPU 活, 而且卡在 GPU 算完 logits 和采样之间, GPU 只能等着。词表十几万时, 这一步直接吃掉 decode 延迟。\n能离线算的原因是状态机的状态数有限 (demo 里 101 个)。每个状态各把全部 token 试走一遍, 存成 (状态数 × 词表大小) 的表, 之后每步只取一行, O(1)。demo 测得在线现算 18.4 µs/step, 查表 0.086 µs/step, 快 215 倍; 预编译一共花 7.9 ms, 几百步就摊平了。真实系统的表是状态数 × 128k 的 bitmask, 编译要几百 ms 到几秒, 所以按 schema 缓存。',
        },
        {
          title: '还要能收尾',
          body: '只用 mask 还有一个漏洞: 输出长度到了 max_tokens 上限时会被直接截断, 得到的是 "合法的前缀" 而不是完整 JSON。demo [4] 把 max_tokens 压到 12, 只查 mask 的模式 100 次里只有 33 次是完整 JSON, 其余像 \'{ "smfmpr": true, "\' 这样半截停下。\n第三张表 need[s, t] 解决这个问题: 它记的是选了 token t 之后, 最少还要几个 token 才能以 EOS 结束 (在 token 图上算最短路)。每一步把 mask 再收紧一层: 只放行 need 不超过 "剩余 token 数减 1" 的 token, 剩余长度不够时提前屏蔽那些 "收不了尾" 的 token。这样输出一定是完整可解析的 JSON, 不会是被长度上限截断的半截: budget 模式 100/100。代价是结尾被硬掰, 内容可能不是模型想说的; vLLM、OpenAI 这类系统默认不强制收尾, 撞上 max_tokens 就返回 finish_reason="length" 让调用方重试。',
        },
      ],
      links: [
        { from: 'JsonFSM (字符级)', to: 'compile_char_dfa', body: 'JsonFSM 是手写的字符级状态机, 回答 "当前状态允许哪些字符" 和 "吃了这个字符后到哪个状态"。compile_char_dfa 用 BFS 把它所有可达状态枚举出来, 存成显式的转移表 trans[s] = {字符: 下一状态}, 后面试走 token 时直接查这张表。' },
        { from: 'token_row', to: 'compile_token_table', body: 'token_row 对一个状态 s 做一次 "逐 token 逐字符试走", 得到这个状态下每个 token 的落点 (走不通是 −1)。compile_token_table 对全部 S 个状态各调用一次, 拼成 (S, V) 的 next_state 表, 再派生出 mask_table 和 need。' },
        { from: 'mask_table[s]', to: 'm10 sample', body: '语法 mask 也只是一个把非法 token 的 logit 置 -inf 的 filter, 和 m10 采样链上的 top-k、top-p 是同一类东西, 插在同一条链上; mask 之后的温度、top-p 照常执行。' },
      ],
      sourceRows: [
        { concept: '词表', code: 'm14_structured_output/grammar.py:build_vocab', takeaway: '造一个演示用的词表: 单个字符, 加上类似 BPE 的多字符片段。特意放进了跨语法边界的 \'":\' 和 \'e"\' (一个 token 跨两个状态) 和永远非法的垃圾 token, 用来检验 token 级判定。' },
        { concept: '试走', code: 'm14_structured_output/grammar.py:token_row', takeaway: '对一个状态逐个 token 试走: EOS 只在接受态合法; 其它 token 按字符顺序走转移表, 中途任何一个字符走不通, 整个 token 作废 (记 −1)。' },
        { concept: '预编译', code: 'm14_structured_output/grammar.py:compile_token_table', takeaway: '一次算出 next_state (落点状态)、mask_table (合法与否)、need (选了之后最少还要几个 token 才能结束) 三张表, 之后每一步只查表, 反复用。' },
        { concept: '光靠 prompt 不够', code: 'logits[~mask] = -inf', takeaway: 'prompt 里写格式要求只能提高合法输出的概率, 输出一长迟早错一次; 只有在 logits 上把非法 token 置 −inf, 它的概率才严格为 0。demo 里不加 mask 的合法率是 0/100。' },
        { concept: '与 xgrammar 的差距', code: 'CFG / 下推自动机', takeaway: '本章的 grammar 没有嵌套, 是正则语言, 一个有限状态机就够。真实 JSON Schema 有嵌套 (对象里套对象), 要用带栈的下推自动机; 栈的内容无限多, 没法全部预编译, 所以 xgrammar 只预编译 "与栈无关" 的那部分 token, 其余运行时再查。' },
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
      subtitle: '读完你能说清 INT4 为什么要分组、AWQ 为什么先缩放再量化, 以及 K 和 V 为什么分组方向不同。',
      tldr: '量化就是把浮点数吸附到少数几个等间距的格点上 (INT4 只有 16 个), 一组数共用一个刻度 (scale)。decode 每步要把权重和 KV 从显存读一遍, 字节少一半, 时间就少一半。\n- group-wise INT4: 每 $g$ 个权重共用一组 scale/zero。组里混进一个特别大的数 (离群值), 整组的刻度都被撑粗; 分组之后离群值只污染自己那一组。\n- AWQ: 模型在意的是输出误差 $= \\sum_i x_i \\cdot \\Delta W_i$, 同样的舍入误差乘上大激活才要命。量化前把激活大的输入通道的权重放大 $s_i$ 倍、激活同步缩小, 数学等价; 误差从 0.0759 降到 0.0273。\n- KV 量化: K 有几个固定的离群通道 (某几维在所有 token 上都很大), 按通道分组把它们关起来; V 没有, 按 token 分组。',
      question: '量化时要让谁的误差最小: 权重本身 ($\\|W - \\hat{W}\\|$), 还是乘上激活后的输出 ($\\|XW - X\\hat{W}\\|$)? 为什么是后者?',
      evolution: {
        title: '从权重误差走到输出误差',
        subtitle: '根问题: decode 每步要把权重和 KV 从显存读一遍, 耗时跟着字节数走。',
        steps: [
          { name: 'FP16 权重', pain: '(原点) decode 受带宽限制, 每步搬多少字节就花多少时间', fix: '每个权重 16 bit; 7B 模型光权重就 14 GB, 每出一个 token 读一遍' },
          { name: 'per-channel INT8', pain: '16 bit 太胖, 带宽全花在读权重上', fix: '每个输出通道一套 scale, 舍入到最近的格点 (RTN); 字节减半' },
          { name: 'group-wise INT4', pain: '压到 4 bit 只剩 16 个格点, 一个离群值就撑大整个通道的格距', fix: '每 $g$ 个权重一组 scale/zero, 离群值只拖累本组; $g=128$ 时 4.25 bit/权重' },
          { name: 'AWQ', year: 2023, pain: 'RTN 只盯权重误差, 可它乘上大激活才是输出误差', fix: '激活大的输入通道先放大 $s$ 倍再量化, $1/s$ 折进上一层; 输出误差 0.0759 → 0.0273' },
          { name: 'KIVI', year: 2024, pain: '权重压完, 长上下文下 KV 成了大头, 而 K 有固定的离群通道', fix: 'K 按通道分组把离群通道关起来, V 按 token 分组; K 的误差 0.18 → 0.026' },
        ],
      },
      code: 'llm_infer/m08_quantization/{int8_weight.py,int4_awq.py,kv_quant.py}',
      points: [
        {
          title: '分组把离群值关起来',
          body: '先说量化怎么做: 一组数里找出最小值和最大值, 把这段范围 (range) 等分成 $2^{\\text{bits}}-1$ 格, 每个数舍入到最近的格点, 只存格点编号; 用的时候乘回刻度 (scale) 再加偏移 (zero) 还原。格距 = range / 格数, 所以范围越宽, 每个数的误差越大。INT4 只有 16 个格点, 组里混进一个离群值 (比别的数大几十倍), 整组的格距都被它撑大, 其它数全被舍得面目全非。\ngroup-wise 的做法是缩小 "组": 每 $g$ 个相邻权重一组, 各自一套 scale/zero。一个离群值只撑大自己这一组的 range, 别的组不受影响。demo 里同样 INT4, per-channel (整个输出通道一组) 的输出误差 0.1058, group=32 降到 0.0759。\n$g$ 越小越准, 但 scale/zero 的额外开销越大。每组存一个 fp16 scale 和一个 fp16 zero, 共 32 bit:\n- $g=128$: $4 + 32/128 = 4.25$ bit/权重。\n- $g=32$: $4 + 32/32 = 5.00$ bit/权重。',
        },
        {
          title: '重要的是输出误差, 不是权重误差',
          key: true,
          body: '模型在意的不是权重 $W$ 本身被改了多少, 而是这一层的输出 $X\\hat{W}$ 和原来的 $XW$ 差多少 ($X$ 是输入激活)。输出误差 $= \\sum_i x_i \\cdot \\Delta W_i$: 把第 $i$ 个输入通道的激活 $x_i$ 乘上这一行权重的舍入误差 $\\Delta W_i$, 再对所有通道求和。这个式子在说: 同样大小的舍入误差, 乘上大激活后对输出的影响就大。LLM 的激活恰好有少数固定通道在所有 token 上都很大, 这些通道对应的权重行最该保护。\nAWQ (activation-aware weight quantization) 用恒等式 $XW = (X/s) \\cdot (s \\odot W)$: 把激活大的那些权重行放大 $s$ 倍, 激活对应的通道同时缩小 $s$ 倍, 乘积不变。放大后的行在同样的格距下, 相对舍入误差缩小约 $s$ 倍。$1/s$ 可以离线折进上一层的权重, 推理时零额外开销。\n$s_i$ 取校准激活的逐通道平均幅度 $\\text{mean}|x_i|$ 的 $\\alpha$ 次方。代价是放大的行撑大了同组的 range, 同组其它行变粗, 所以 $\\alpha$ 要在校准集上网格搜。demo 里 $\\alpha=0$ 就是 RTN (round-to-nearest, 不缩放直接舍入) 0.0757, 最优 $\\alpha=0.4$ 到 0.0279, $\\alpha=1$ 反而 0.1173 比 RTN 还差; 在没见过的 held-out 激活上, AWQ 的输出误差 0.0273, 对比 RTN group=32 的 0.0759。',
        },
        {
          title: 'K 和 V 的离群结构不一样',
          body: 'KV cache 也能量化, 分组方向要看数据的离群结构。K 的少数通道 (demo 里是第 3、17、40 维) 在所有 token 上都是大值:\n- 按 token 分组 (一个 token 的整行一组): 每一行都含有这几个离群通道, 每一行的 scale 都被撑大, INT4 下 K 的相对误差 0.18。\n- 按通道分组 (每 32 个 token 内的一列一组, KIVI 的做法): 离群通道被关在自己的组里, 误差 0.026。\nV 没有这种固定的离群通道, 按 token 分组就够 (INT4 误差 0.0894, 两种分法一样)。按 token 分组还有一个好处: 每个新 token 写入时可以独立量化, 不用回头改旧的; 按通道分组要攒满 32 个 token 才有 min/max, 真实 KIVI 把最近不满一组的 token 留在 fp16 里。',
        },
        {
          title: 'GPTQ 和 AWQ 分工: 一个补偿, 一个保护',
          body: '两者都只量化权重, 存储格式和上面的 group-wise INT4 完全相同, 推理 kernel 不变。治的病不同:\n- AWQ: 少数重要输入通道的误差被大激活放大。量化前把这些行放大, 让它们分到更细的格点。这是保护。\n- GPTQ: RTN 每个权重各自舍入, 误差在输出上累加。GPTQ 一行一行量化, 量化完一行就算出它的误差, 按 $H^{-1}$ 摊给后面还没量化的行, 让它们提前往反方向偏一点, 抵消这一行的误差。$H = X^\\top X / N$ 是校准激活的协方差 (记录输入通道两两之间的相关性), 相关的通道才能互相补偿。这是补偿。\n实测 (INT4 g32, held-out 激活):\n- 相关激活: RTN 0.0776 → GPTQ 0.0369。\n- 再加 ×30 离群通道: RTN 0.0749, AWQ 0.0258, GPTQ 0.0293, 先 AWQ 再 GPTQ 0.0113。\n叠加比单用都好, 说明两者治的是不同的病。真实部署常把两者视为二选一, 叠加是教学组合。',
        },
        {
          title: 'GPTQ 的收益全靠激活相关性',
          body: 'GPTQ 的补偿靠的是输入通道之间的相关性: 这一行多舍了一点, 和它相关的下一行就少舍一点, 输出上互相抵消。激活通道互不相关时 $H$ 是对角矩阵, 补偿项全为 0, GPTQ 退化成 RTN, 一点收益也没有。\n校准样本少时更糟, 它会把采样噪声当成相关性去补偿, 在没见过的数据上反而更差:\n- N=256 (对 256 维输入): held-out 误差是 RTN 的 1.56×。\n- N=512: 1.17×; N=2048: 1.03×; N=8192: 1.01×, 才打平。\n本 demo 用 "rank-32 公共成分 + 噪声" 合成相关激活, 才显出 0.0776 → 0.0369 的效果。真实 LLM 激活确实强相关, 但幅度不是这个数。',
        },
        {
          title: 'SmoothQuant 把离群值挪给权重',
          body: '前面几种都是 weight-only: 只量化权重, 激活仍是浮点, 省的是显存和 decode 带宽。想让矩阵乘本身走 INT8 / FP8 tensor core (prefill 也变快), 激活也得量化, 而激活有离群通道, 一个刻度被它撑得很粗。\nSmoothQuant 用和 AWQ 同一个恒等式, 目的相反: 激活除以 $s$、权重乘 $s$, 把离群值从激活挪到权重上。$s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$, $\\alpha$ 控制挪多少: $\\alpha=0$ 一点不挪, $\\alpha=1$ 全挪给权重。\n- $\\alpha=0.5$: W8A8 (权重和激活都 INT8) 的误差从不平滑的 0.0278 降到 0.0070, 降 4.0×。\n- $\\alpha=1$: 离群值全压给权重, 权重那边的误差变大, 总误差反而涨到 0.0297, 比不平滑还差。',
        },
        {
          title: 'FP8 换一种格点',
          body: 'INT8 的格点等间距, 大数小数用同一个格距, 小数的相对误差很大。FP8 是浮点格式, 格距随数值大小伸缩, 相对误差恒定, 所以它不靠细粒度 scale, 也更扛离群值。E4M3 是 FP8 的一种: 4 位指数、3 位尾数, 正规区相对误差不超过 $2^{-4}$ = 6.25%。\n- 不需要 per-channel: 列幅度悬殊的权重, 从 per-tensor 换成 per-channel scale, INT8 误差改善 7.71×, E4M3 只改善 1.02×, 说明它本来就不怕。\n- 扛离群值: 激活 per-tensor、带 ×50 离群值时, E4M3 0.0360, INT8 0.0606。\n- 不是处处更好: 没有离群值时, INT8 per-channel (0.0070) 比 E4M3 (0.0251) 还准, 均匀格点在没有离群值的分布上更划算。\n这一节和上一节全是 fake quant: 量化后立刻反量化回浮点再算, 只验数值, 不说明速度。',
        },
      ],
      links: [
        { from: 'quantize_int8 (per-channel)', to: 'quantize_groupwise', body: 'quantize_int8 每个输出通道一套 scale (一整列共用一个刻度)。quantize_groupwise 把组缩小到每 $g$ 个相邻权重一套 scale/zero, 离群值的影响范围从一整列缩到 $g$ 个数。' },
        { from: 'mean|x_i|', to: 's_i = mean|x_i|^α', body: 'AWQ 的缩放因子只用校准激活的逐通道平均幅度 mean|x_i| 这一个统计量, 取 $\\alpha$ 次方。不需要反向传播, 不需要训练, 所以 AWQ 很便宜。' },
        { from: 'KV 量化', to: 'KV 体积一章', body: '量化减的是每个元素占的字节数 (fp16 的 2 字节变成 INT4 的半字节), GQA/MLA 减的是元素个数 (存几组头)。两者相乘, 才是每 token 的 KV 字节数。' },
      ],
      sourceRows: [
        { concept: '分组量化', code: 'm08_quantization/int4_awq.py:quantize_groupwise', takeaway: '把 (D_in, D_out) 的权重 reshape 成 (D_in/g, g, D_out), 在 g 那一维上统计 min/max, 得到每组一个 scale 和 zero。' },
        { concept: '优化目标', code: 'm08_quantization/int4_awq.py:output_err', takeaway: '$\\|XW - X\\hat{W}\\|_F / \\|XW\\|_F$: 拿校准激活 X 算出量化前后输出的相对差。本章表格里的 "输出误差" 都是这个数。' },
        { concept: 'AWQ 搜索', code: 'm08_quantization/int4_awq.py:awq_quantize', takeaway: '对 $\\alpha$ 从 0 到 1 网格搜索, 每个 $\\alpha$ 算一次 $\\hat{W} = Q(s \\odot W) / s$ 的输出误差取最小。$1/s$ 离线折进上一层的权重, 推理时零额外开销。' },
        { concept: 'KV 三种分组', code: 'm08_quantization/kv_quant.py:quantize_kv', takeaway: 'per-tensor、per-token、per-channel 三种 scheme 的实现只差 reshape 的形状和统计 min/max 的 axis, K 的误差却差一个数量级 (INT4 下 0.18 对 0.026)。' },
        { concept: 'GPTQ', code: 'm25_weight_quant/gptq.py:gptq_quantize', takeaway: '先算 $U = \\mathrm{chol}(H^{-1})^\\top$ (上三角)。第 $i$ 行量化后算误差 $\\delta = (W_i - Q_i)/U_{ii}$, 后面还没量化的行减去 $U_{i,i+1:}^\\top \\otimes \\delta$, 把误差摊出去。每组的 scale 用已经补偿过的权重现算。' },
        { concept: '平滑因子', code: 'm25_weight_quant/smoothquant.py:smooth_scales', takeaway: '$s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$, 每个输入通道一个。激活统计来自校准集, 部署时 $s$ 是常数, 可折进上一层。' },
        { concept: 'W8A8', code: 'm25_weight_quant/smoothquant.py:w8a8_matmul', takeaway: '激活按 token (每行一个 scale)、权重按输出通道 (每列一个 scale) 做对称 INT8; 两个 scale 都能提到矩阵乘外面, 中间就是纯整数乘法。demo 用两个开关分别只量化 X 或只量化 W, 把误差拆成激活贡献和权重贡献。' },
        { concept: 'FP8 舍入', code: 'm25_weight_quant/fp8.py:fp8_round', takeaway: '格距 $= 2^{\\lfloor \\log_2|x| \\rfloor - m}$ ($m$ 是尾数位数), 数越大格距越大; 指数下限钉在最小正规数, 超过 max 饱和。E4M3 正规区相对误差 $\\le 2^{-4}$, E5M2 $\\le 2^{-3}$。' },
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
      subtitle: '读完你能算出 MHA / GQA / MLA 每个 token 的 KV 占几个字节, 并说清这个数怎么决定一张卡能同时服务多少条序列。',
      tldr: '每个 token 的 KV 要占多少字节, 取决于 cache 里要存几组 K/V 头:\n- LLaMA-2-7B (MHA, 每个注意力头各存一组 K/V): 512 KiB\n- LLaMA-3-8B (GQA-8, 32 个 query 头分 8 组, 每组共用一对 K/V): 128 KiB\n- DeepSeek-V3 (MLA, 不存 K/V 本身, 只存一个压缩过的 latent 向量): 68.6 KiB\n一张卡能同时服务多少条序列, 就是留给 KV 的显存除以这个数、再除以上下文长度。MLA 比同尺寸全存 K/V 的 MHA 省 56.9 倍。',
      question: 'MLA 已经把 K/V 压成了一个 latent, 为什么位置信息 (RoPE) 不能一起压进去, 还要另外存一份解耦的 RoPE key?',
      evolution: {
        title: 'cache 里存得越来越少',
        subtitle: '根问题: 一张卡能同时服务几条序列 = KV 显存 ÷ 每 token 字节 ÷ 上下文长度。',
        steps: [
          { name: 'MHA 全存', pain: '(原点) 并发上限由每 token 的 KV 字节数决定', fix: '每个头各存一对 K/V; LLaMA-2-7B 每 token 512 KiB' },
          { name: 'GQA', pain: '512 KiB/token, 4K 上下文一条序列就 2 GiB', fix: '8 组 query 头各共用一对 KV 头: LLaMA-3-8B 128 KiB; 砍到 1 组就是 MQA' },
          { name: 'MLA 存 latent', year: 2024, pain: 'KV 头数砍到底, cache 仍与 $n_{kv} \\cdot d_{\\text{head}}$ 成正比', fix: '只存一个维度很小的压缩向量 (低秩 latent $C$), 用时乘 $W_{UK}$、$W_{UV}$ 现场还原每个头的 K/V; 代价是每步还原 T 行' },
          { name: 'absorb', pain: '每步把 T 行 latent 升维成每头 K/V, 计算随上下文涨', fix: '把 $W_{UK}$ 乘到 q 上, cache 本身就是 K: decode 等价于一个大 head_dim 的 MQA' },
          { name: '解耦 RoPE key', pain: 'latent 一带 RoPE, 旋转夹在 $W_{UK}$ 前面, 吸收做不成', fix: 'latent 不旋转, 另存一份各头共享的 RoPE key: DeepSeek-V3 每 token 68.6 KiB' },
        ],
      },
      code: 'llm_infer/m18_kv_attention_variants/attention_variants.py',
      points: [
        {
          title: '一个参数 n_kv 分出前三种',
          key: true,
          body: '每个 token 的 KV 占多少字节, 由公式 $2 \\cdot n_{kv} \\cdot d_{\\text{head}} \\cdot n_{\\text{layer}} \\cdot \\text{bytes}$ 算出。这个式子在说: K 和 V 各一份 (开头的 2), 每份有 $n_{kv}$ 个头, 每个头是 $d_{\\text{head}}$ 个数, 每一层都要存一套 (乘 $n_{\\text{layer}}$), 每个数占 bytes 个字节 (fp16 是 2)。\n$n_{kv}$ 是 cache 里实际存的 K/V 头数, 它把前三种结构区分开。注意力层里有 $n_{\\text{head}}$ 个 query 头, 每个 query 头都要有一组 K/V 来对照; 区别只在这些 K/V 是各头一份, 还是几个头合用一份:\n- MHA (multi-head attention): 每个 query 头各有自己的 K/V, $n_{kv} = n_{\\text{head}}$。\n- GQA (grouped-query attention): query 头分成几组, 每组共用一对 K/V 头, $1 < n_{kv} < n_{\\text{head}}$。\n- MQA (multi-query attention): 所有 query 头共用同一对 K/V, $n_{kv} = 1$。\n- MLA: 另一条路, 不存 K/V, 改存 latent, 见下一条。\n共用 K/V 的几个 query 头在代码里靠广播 (broadcast) 读同一份数据, kernel 里不真的把 K/V 复制成 $n_{\\text{head}}/n_{kv}$ 份。\ndemo 把真实模型配置代进公式: LLaMA-2-7B (MHA, 32 个 KV 头) 每 token 512 KiB; LLaMA-3-8B (GQA, 8 个 KV 头) 128 KiB, 正好是 1/4。\n为什么这个数重要: 并发上限 = 显存预算 ÷ 每 token 字节 ÷ 上下文长度。demo 假设 40 GiB 全给 KV、每条请求 8192 个 token, LLaMA-2-7B 只能同时放 10 条, LLaMA-3-8B 能放 40 条。当年训练时为省参数做的结构选择, 在推理侧直接变成吞吐。',
        },
        {
          title: 'MLA 的 cache 里只有 latent',
          body: 'MLA (multi-head latent attention, DeepSeek-V2/V3 用的结构) 的 cache 里不存任何一个头的 K 或 V。它存两样东西: 一个每 token $d_c$ 维的压缩向量 $C$ (叫 latent, 形状 (T, d_c)), 和一份所有头共享、只负责位置信息的 RoPE key (形状 (T, d_rope))。所以每 token 的字节数是 $(d_c + d_{\\text{rope}}) \\cdot n_{\\text{layer}} \\cdot \\text{bytes}$。\n每个头的 K 和 V 要用时才现场还原: 用 $C$ 乘一个上投影矩阵 $W_{UK}$ 得到这个头的 K, 乘 $W_{UV}$ 得到 V。这两个矩阵是训练出来的模型参数, 不是事后对 cache 做的压缩。还原出的 K/V 只在 attention 这一步用, 从不进 cache。\n对比上一条的公式, 这里没有开头那个 "2·": K 和 V 共用同一个 latent, 不用分开存两份。DeepSeek-V3 每层 $d_c$ = 512、$d_{\\text{rope}}$ = 64, 61 层, 每 token 68.6 KiB; 如果它用 MHA (128 个头) 全存, 要 3904 KiB, 所以省 56.9 倍。demo 在一个小模型上断言: 用 latent cache 做增量 decode 和不用 cache 全量重算的输出一致 (max|Δ| 1.88e-06), 实际 cache 字节数等于公式值 (7680)。',
        },
        {
          title: 'absorb 形式: K 不用还原',
          body: '每步还原 K 有个问题: cache 里有 T 行 latent, 每生成一个 token 都要把这 T 行各乘一次 $W_{UK}$ 变成每个头的 K, 这笔计算随上下文长度一直涨。absorb (吸收) 技巧把它省掉了。\n注意力分数本来是 $q \\cdot (C W_{UK})^\\top$: 先把 latent 还原成 K, 再和 q 点积。矩阵乘法满足结合律, 它等于 $(q W_{UK}^\\top) \\cdot C^\\top$: 先把 $W_{UK}$ 乘到 q 上 (q 只有一行, 这一步很便宜), 再直接和 latent $C$ 点积。这样 K 就不用还原了, cache 里的 latent 本身就当 K 用。\n于是 MLA 的 decode 等价于一个 $\\text{head\\_dim} = d_c + d_{\\text{rope}}$ 的 MQA: 所有 query 头共用同一份 "K" (latent 拼上 RoPE key), 只是这份 K 的维度比较大 (DeepSeek-V3 是 512 + 64 = 576)。demo 断言两种算法的注意力分数相差 2.86e-06, 整网输出相差 1.43e-06。\n这也解释了为什么 latent 不能带 RoPE: RoPE 是按位置对向量做一次旋转, 如果这个旋转加在还原后的 K 上, 它就夹在 q 和 $W_{UK}$ 之间 (式子变成 $q R (C W_{UK})^\\top$), $W_{UK}$ 没法提前乘到 q 上, 吸收做不成。所以位置信息单独放进那份 RoPE key, latent 保持不旋转。',
        },
      ],
      links: [
        { from: '阶段 2 注意力演进', to: 'KV bytes/token', body: '左边是训练阶段讲的注意力结构演进 (MHA 到 GQA 到 MLA), 当年换结构是为了省参数和训练算力。右边是推理侧的账: 同样这个结构选择, 决定了每个 token 的 KV 要占几个字节, 也就决定了显存能放几条请求。' },
        { from: 'bytes/token', to: 'BlockManager.num_blocks', body: '每 token 字节数是分母: 留给 KV 的显存 ÷ 每 token 字节 = 池子里一共能放多少个 token 的 KV。BlockManager 启动时就按这个数决定开多少个 block (num_blocks); 这个数再除以上下文长度, 就是并发上限。' },
        { from: 'MLA absorb', to: 'FlashMLA', body: '本章 demo 的 absorb 分支只是验证两种算法结果一样。真实的 kernel FlashMLA 全程按吸收形式算, 直接在 576 维的 latent 上做 attention, 从头到尾不把每个头的 K/V 还原出来 (不物化)。' },
      ],
      sourceRows: [
        { concept: 'GQA 公式', code: 'm18_kv_attention_variants/attention_variants.py:kv_bytes_per_token', takeaway: '就是 2 · n_kv · d_head · n_layer · bytes 这个式子。开头那个 2 是 K 和 V 两份。demo 断言 MHA (n_kv=32) 的结果等于 4 倍 GQA (n_kv=8)、32 倍 MQA (n_kv=1): 字节数正比于 n_kv。' },
        { concept: 'MLA 公式', code: 'm18_kv_attention_variants/attention_variants.py:mla_bytes_per_token', takeaway: '是 (d_c + d_rope) · n_layer · bytes。没有 "2·", 因为 K 和 V 共用同一个 latent; d_rope 那一份是所有头共享的 RoPE key。' },
        { concept: 'latent decode', code: 'MLALayer.forward', takeaway: 'latent c = x W_DKV 直接进 cache, 不加 RoPE; 位置信息全放在另一份 k_rope = RoPE(x W_KR) 里, 所有头共享。每个头的 K 是 [c W_UK_h ; k_rope] 拼起来的, V 是 c W_UV_h, 都在 attention 时现算, 不进 cache。' },
        { concept: '对拍', code: 'cache_nbytes', takeaway: '把各层 cache 数组的实际字节数加起来。demo 对 MHA/GQA/MQA/MLA 四种变体各跑一遍 20 个 token 的增量 decode, 断言实际字节数等于公式值 (61440 / 15360 / 7680 / 7680), 并且输出和全量重算一致。' },
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
      subtitle: '读完你能解释滑动窗口一挤掉开头几个 token 模型为什么立刻崩, 以及 StreamingLLM 怎么把 KV cache 压到固定大小还不崩。',
      tldr: '流式对话没有尽头, KV cache 不能无限涨。最直接的办法是只留最近一段 (滑动窗口), 但真实模型一这么做就崩, 原因在 softmax: 注意力权重加起来必须等于 1, 当前 token 没什么值得看的时候, 多出来的注意力就倒在开头几个 token 上, 它们被训练成了垃圾桶 (attention sink)。把它们挤出去, 剩下的权重会被整体放大, 输出就乱了。\nStreamingLLM 的做法:\n- 永远保留开头 n_sink 个 token, 再加最近 window 个, cache 大小恒定。\n- 位置不用绝对位置, 按 cache 里的槽位重新编号, 所以永远不会超过训练长度; 代价是 K 必须存旋转 (RoPE) 之前的版本, 每步现转。',
      question: '普通 KV cache 存的是加过位置信息 (RoPE) 之后的 K, SinkCache 为什么反过来要存加 RoPE 之前的?',
      evolution: {
        title: '有界, 还不能崩',
        subtitle: '根问题: 流式对话没有尽头, KV 和位置编号都随 token 一直涨。',
        steps: [
          { name: '全量 KV cache', pain: '(原点) 对话不停, KV 线性增长, 位置迟早超过训练长度', fix: '全存; 显存迟早爆, 超出训练长度的位置模型从没见过' },
          { name: '滑动窗口', pain: '显存和位置都没有上界', fix: '只留最近 window 个 token; 开头几个一被挤出, 输出立刻漂掉' },
          { name: 'sink + 窗口 (StreamingLLM)', year: 2023, pain: 'softmax 权重和恒为 1, 开头 token 被训成倒注意力的地方, 逐出后分母骤变', fix: '开头 n_sink 个永不逐出, 其余走滑动窗口; cache 恒 ≤ n_sink + window' },
          { name: '按槽位重编号', pain: '绝对位置还在涨; K 存的是旋转后的, 位置已烙死改不了', fix: '位置取 cache 槽位 0..L−1, K 存旋转前的, 每步整体重转一遍, 代价 O(L·D)' },
        ],
      },
      code: 'llm_infer/m16_attention_sinks/sink_cache.py',
      points: [
        {
          title: 'sink 是 softmax 的排污口',
          key: true,
          body: '开头几个 token 之所以变成 sink, 是训练的副产品。先说 softmax: 注意力给每个历史 token 一个权重, 这些权重由 softmax 算出, 加起来恒等于 1。当前 token 有时候并不需要看任何历史 (比如正在输出一个固定搭配), 但权重不能全是 0, 总得倒给谁。开头的几个 token 对所有后续位置都可见 (因果 mask 下任何位置都能看到它们), 于是模型在训练中学会了 "注意力没处放时就倒这儿"。这就是 attention sink。\n把它们逐出会发生什么: softmax 的分母是所有 exp(分数) 的和。如果 sink 那一项占了分母的 90%, 删掉它, 其余每个权重都会被放大 10 倍。这是一种训练时从没见过的权重分布, 输出立刻漂掉。\n这和这几个 token 的语义无关, 换成别的开头 token (比如换行符) 也一样; 重要的是它们的位置和它们吸走的注意力。\ndemo [B] 人工植入了一个 sink (第 0 个 key 和所有 query 强对齐): 流长 256 时开头 4 个位置吃掉 93.0% 的注意力; 同样 36 个槽位的预算, 留着 sink 的输出相对误差 19.40%, 纯滑动窗口 103.06%。demo [A] 用随机权重的 TinyLM 就看不到这个现象, 见下面表格最后一行。',
        },
        {
          title: '有界 cache',
          body: 'SinkCache 里的条目数永远不超过 n_sink + window。每来一个新 token 就追加到末尾; 超出预算时, 逐出槽位 n_sink 那一个, 它正好是窗口里最老的 token (槽位 0 到 n_sink−1 是 sink, 永远不动)。\ndemo [A] 用 1024 个 token 的流, 预算 64 = sink 4 + 窗口 60, 断言 cache 条目峰值恰好是 64; 前 64 步还没开始逐出时, 输出和完整 cache 逐位一致。被逐出的 token 就彻底丢了: 这个办法保证的是流不崩, 不是记得很久以前的内容。',
        },
        {
          title: '位置按槽位重编号',
          body: 'RoPE 给每个 token 的 K 和 q 按位置做一次旋转, 注意力只看两个位置的差。普通 cache 用 token 在数据流里的绝对位置, 流到第 100 万个 token 时位置就是 999999, 早就超出训练长度, 模型从没见过。\nSinkCache 改用 cache 槽位当位置: 当前 cache 里有 L 条, 位置就是 $0, \\dots, L-1$, 新 token 的 query 放在最后一个槽位 L−1。这样位置永远小于 n_sink + window, RoPE 表只要 64 行就够。demo 断言: 用 64 行的 RoPE 表, 绝对位置 1023 直接 IndexError, 而槽位重编号的策略跑完 1024 步全程正常。\n代价: 每逐出一个 token, 它后面所有 token 的槽位都前移一格, 位置变了, 之前旋转过的 K 就不对了。所以 K 只能存旋转前的版本, 每步把 cache 里全部 L 行 K 按当前槽位重新旋转一遍, 开销 O(L·D)。普通 cache 存的是旋转后的 K, 位置已经烙死在里面, 改不了。',
        },
      ],
      links: [
        { from: 'KV cache (m01)', to: 'SinkCache', body: 'm01 的 cache 每来一个 token 就多一行, 随流长 T 无限增长 (O(T))。SinkCache 只是在它上面加了一条逐出规则, 条目数固定在 n_sink + window, 流多长都不再涨。' },
        { from: 'RoPE', to: 'positions = arange(L)', body: 'RoPE 让注意力只看两个 token 的位置差。重编号后位置就是槽位号, 所以 "距离" 看的是槽位差: 中间有 token 被逐出后, 两边的 token 在 cache 里挨得更近了, 模型眼里它们的距离就被压缩了。' },
        { from: 'select_blocks 的 forced', to: '稀疏 decode', body: '后面稀疏 decode 那一章每步只读 cache 的一部分 block, 它的 select_blocks 有个 forced 参数, 强制永远保留第 0 块。这个第 0 块就是本章的 sink, 道理相同: 丢了它 softmax 的分母就不对了。' },
      ],
      sourceRows: [
        { concept: '逐出规则', code: 'm16_attention_sinks/sink_cache.py:SinkCache.append', takeaway: '新 token 追加到末尾后, 条目一旦超过 n_sink + window, 就只保留 keep = [0:n_sink] + [n_sink+1:], 也就是永远去掉槽位 n_sink 那一个 (窗口里最老的), 前 n_sink 个 sink 原地不动。' },
        { concept: '重编号', code: 'm16_attention_sinks/sink_cache.py:stream_step', takeaway: '每一步都执行 pos = arange(L): 位置就是当前槽位 0 到 L−1。cache 里存的是没旋转的 K, 每步按这组位置现场旋转一遍; 新 token 的 query 放在最后一个槽位。' },
        { concept: '不是长上下文', code: 'window 之外的 token', takeaway: '被逐出的内容彻底丢失, 模型再也看不到它。sink 保证的是 "不崩", 不是 "记得"; 要记得, 就得把 KV 保留下来 (offload 到 CPU 或磁盘) 或者靠检索把内容重新塞回 prompt。' },
        { concept: '随机权重上看不到', code: 'TinyLM 的 attention', takeaway: 'demo [A] 用没训练过的 TinyLM, 它开头 4 个 token 的注意力占比和均匀分布差不多 (T=1024 时逐层 0.0008 到 0.0062, 均匀值 0.0039), 没有 sink 现象, 所以 sink+窗口 只在 3 个权重 seed 里的 2 个赢过纯窗口, demo 不对这个做断言。这一章的效应要在 [B] 植入 sink 的合成数据上才成立; 真实 LLM 的 sink 是预训练出来的。' },
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
      subtitle: '读完你能说清链式 draft 为什么第一个猜错后面 $K-1$ 个全废, 以及 EAGLE (猜得更准) 和树形投机 (一次验更多分支) 各怎么补这个漏洞。',
      tldr: '上一章的投机解码里, draft 连猜 $K$ 个 token 是一条链: 第一个猜错, 后面 $K-1$ 个全废。本章两个办法:\n- EAGLE: 让 draft 猜得更准。target 验证时每个位置都顺手算出了 hidden state (最后一层的特征向量), 里面已经写着 "接下来想说什么"; EAGLE 的 draft 吃这个特征, 并且共享 target 的 lm_head (把特征变成每个 token 分数的那一层), 接受率更高。\n- 树形投机: 一次验更多分支。每个位置不只押 draft 的第一候选, 留下 top-k 个候选, 长成一棵树; target 用 tree attention mask 一次 forward 验完整棵树, 接受和 target greedy 一致的最长路径。\n同样验证 16 个 token: 链 $K=15$ 每次 target 调用产出 2.13 个 token, 树 [3,2,1] 产出 2.58 个; 同样只调 3 次 draft 的链 $K=3$ 只有 1.86 个。',
      question: '树里同一深度的兄弟节点为什么共享同一个 RoPE 位置, 而不是依次加一?',
      evolution: {
        title: '让一次验证多收几个 token',
        subtitle: '根问题: target 一次 forward 能并行验很多槽位, 产出却取决于从头连续猜对几个。',
        steps: [
          { name: '链式独立 draft', pain: '(原点) 一次 target forward 的产出 = 连续猜对的个数 + 1', fix: '小模型连猜 K 个一次验; 它只看 token id, 猜不准, 第一个错后面全废' },
          { name: 'EAGLE', year: 2024, pain: 'target 验证时算出的 hidden state 被扔掉, draft 只能看 token id', fix: 'draft 吃 target 的 hidden state, 在特征空间自回归, 共用 target 的 lm_head' },
          { name: '树形验证', pain: '猜得再准也是一条链, 一步错后面全废; 第 2、3 候选常常是对的', fix: '每个节点留 top-k 孩子, tree mask 一次验完: 同样验 16 个 token, 链 K=15 每次 2.13 token, 树 2.58 (链 K=3 只有 1.86)' },
        ],
      },
      code: 'llm_infer/m17_eagle_speculative/eagle.py · llm_infer/m19_tree_speculation/tree_spec.py',
      points: [
        {
          title: '特征级 draft: 看特征, 不只看 token',
          body: 'm07 的独立 draft 只看得到 token id。可 target 每次验证都顺手算出了每个位置的 hidden state $h$ (最后一层的特征向量), 里面已经写着 "接下来想说什么", 扔掉太浪费。\nEAGLE 让 draft 在特征空间里自回归: $\\hat{h}_{t+1} = \\mathrm{Draft}(h_t, \\mathrm{emb}(x_{t+1}))$。这个式子在说: 把上一个位置的特征 $h_t$ 和刚确定的 token $x_{t+1}$ 的 embedding 拼在一起, 预测下一个位置的特征 $\\hat{h}_{t+1}$; 再把它过 target 自己的 lm_head (把特征变成每个 token 分数的那一层), 取 argmax 就是 draft token。为什么要喂 $x_{t+1}$: $h_t$ 只决定下一个 token 的分布, 采样或纠错后实际选了哪个是额外信息, 不喂它 draft 消不掉这个不确定性。\n第 1 步用的 $h$ 是 target 验证那次 forward 白送的真特征; 第 2 步到第 $K$ 步只能用 draft 自己算出的 $\\hat{h}$, 误差逐步累积, 所以越靠后的槽位越难被接受。draft 本体很小 (demo 里是一个线性映射, 真实 EAGLE 是一层 decoder), 不用另养一个小模型。\ndemo 把同样的训练数据、同样的验证循环, 给两种 draft 各跑一遍: 32 个 prompt × 24 个新 token, $K=4$。只喂 token embedding 的 draft 每轮接受 0.36/4, target 调用 588 次 (1.31×); 多喂一个特征 $h$ 的 EAGLE 式 draft 每轮接受 0.60/4, 505 次 (1.52×)。两种 draft 的输出都与 target 单独 greedy 逐 token 相同。数字远低于论文的约 80% 接受率, 因为这里的 draft 只是线性映射, target 也是随机权重。',
        },
        {
          title: 'tree mask: 让 target 一次验完整棵树',
          key: true,
          body: '树里每个节点是一个 draft token, 根是已经确定的 out[-1]; widths=[3,2,1] 的意思是根留 3 个孩子, 每个孩子再留 2 个, 再各留 1 个。从根到任一节点的路径, 就是一条候选的续写序列。\n把整棵树的 token 一次喂给 target 时, 要保证每个节点只能看到自己那条路径上的 token, 看不到别的分支。tree mask 就是干这个的: 先算祖先矩阵 anc[i, j] = "j 是 i 的祖先或就是 i", mask 里上下文列全部可见, 树内的列只有 anc 为真的才可见, 其它置 −inf。\n- 每个节点: 看到的恰好是从根到自己那一条链, 所以它那一行的 logits 和把这条路径顺序 decode 一遍完全一样。demo [1] 断言某条路径上树形与顺序 forward 的 logits 差 9.5e-07。\n- 兄弟之间: 互不可见, 因为它们是同一个位置的不同候选。所以同深度的节点共享同一个 RoPE 位置 n_ctx + depth, 不能依次加一, 否则被接受路径的位置和顺序 decode 对不上。\n一次 forward 就把整棵树验完, 得到每个节点之后 target 认为该接什么。接受时从根出发: 当前节点的 target argmax 命中哪个孩子就走进去, 走不动时那个 argmax 就是纠错或 bonus token。',
        },
        {
          title: '算力换延迟',
          body: '树的账: widths=[3,2,1] 长出 1+3+6+6 = 16 个节点, 一轮只要 1 次 target 调用加 3 次 draft 调用 (每层一次), 但验证的 token 数是 16 个, 是链 $K=3$ 的 4 倍。\ndemo [2] 用 12 个 prompt × 32 token、draft 是 target 权重加 10% 噪声: 链 $K=3$ (同样 3 次 draft 调用) 每轮接受 0.98 个, 每次 target 调用产出 1.86 个 token; 链 $K=15$ (同样验证 16 个 token) 每轮接受 1.40, 产出 2.13; 树每轮接受 1.80, 产出 2.58。同样 16 个验证 token, 花在 "宽" 上比花在 "深" 上值: 链的第 k 个 token 被接受的概率约按 $\\alpha^k$ 衰减, 后面的槽位几乎都浪费, 树把预算花在前几层的备选上。同一状态下, 树接受 1.80, 只看它自己的 top-1 那条脊只有 1.09。\n为什么多验 12 个 token 几乎不花钱: decode 是带宽受限的, 时间花在读权重上, 多算十几个 token 的矩阵乘几乎不加延迟。但 batch 大、算力吃紧时, 这笔账就不再免费, 真实系统会按负载缩小树。',
        },
      ],
      links: [
        { from: 'm07 accept_greedy', to: 'accept_tree', body: 'm07 的 accept_greedy 沿一条链逐位比对 draft token 和 target 的 argmax。accept_tree 做的是同一件事, 只是从 "沿一条链比" 变成 "从根往下走, target 的 argmax 命中哪个孩子就进哪个", 走不动时停下。' },
        { from: 'TinyLM.forward(return_hidden)', to: 'EagleDrafter.propose', body: 'forward 加上 return_hidden 就会把每个位置最后一层的特征一起返回。验证那次 forward 白送了被接受位置的真特征, EagleDrafter.propose 拿它当下一轮 draft 的起点, 不用额外跑 target。' },
        { from: 'gather_kv', to: 'KV 回滚', body: '验证那次 forward 让 KV 多了 16 行 (整棵树)。gather_kv 只留被接受路径上那几行, 其余丢掉。K 存的是 RoPE 之后的值, 路径上的 depth 是 0、1、2 连续的, 正好等于顺序 decode 的位置, 所以挑行就行, 不用重算。' },
      ],
      sourceRows: [
        { concept: '树形状', code: 'm19_tree_speculation/tree_spec.py:tree_shape', takeaway: '按 BFS (一层一层) 给节点编号, 父节点编号总在子节点前面, 返回每个节点的 parents 和 depth。[3,2,1] 长出 1+3+6+6 = 16 个节点。' },
        { concept: '树 mask', code: 'm19_tree_speculation/tree_spec.py:tree_mask', takeaway: '上下文的列全可见; 树内的列只对祖先和自己可见, 兄弟和堂兄弟互相看不见。RoPE 位置 positions = n_ctx + depth, 同深度共享。' },
        { concept: '最长路径', code: 'm19_tree_speculation/tree_spec.py:accept_tree', takeaway: '从根往下走, target 的 argmax 命中哪个孩子就进哪个。同一父节点的 top-k 候选互不相同, 所以每层至多命中一个孩子, 路径唯一。' },
        { concept: 'EAGLE drafter', code: 'm17_eagle_speculative/eagle.py:EagleDrafter', takeaway: '只实现 drafter 协议 propose(out, K, temperature, rng, hidden) → (tokens, probs); 验证循环与接受规则完全复用 m07.speculative_decode。' },
        { concept: '实测', code: 'm19 demo', takeaway: '同样验证 16 个 token: chain $K=15$ 每轮接受 1.40 (2.13 token/次 target 调用), tree [3,2,1] 1.80 (2.58)。同样 3 次 draft 调用: chain $K=3$ 每轮接受 0.98 (1.86)。' },
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
      subtitle: '读完你能说清被 GPU 挤出去的历史 KV 怎么降级到 CPU 和磁盘, 以及每一层为什么要单独算一笔 "搬回来还是重算" 的账。',
      tldr: '多轮对话每一轮的 prompt 都带着整段历史。历史的 KV 如果还在 GPU 上就能直接复用; 但用户一多 GPU 装不下, 被挤出去的 KV 以前就直接丢了, 下一轮只能从头重算。本章的办法是不丢, 降级到 CPU 内存、再到磁盘, 用到时搬回来。\n搬回来值不值, 算一笔账 (1 个 token 的 KV 按 128 KiB 算):\n- 走 PCIe 25 GB/s 从 CPU 搬回 GPU: 约 5 µs\n- 重新算一遍: 约 125 µs\n搬比算便宜 25 倍。但每次加载还有一笔固定延迟, 链路一慢, 命中反而比重算更贵。所以每一层都要单独判断 "加载还是重算"。\n以下时间全部来自代价模型, 不是实测。',
      question: '多加一层缓存, 怎么会反而让首 token 延迟 (TTFT) 变差?',
      evolution: {
        title: '装不下就降级, 降级还要算账',
        subtitle: '根问题: 多轮对话每一轮的 prompt 都带着整段历史, 重算它费算力, 存着它占显存。',
        steps: [
          { name: 'GPU 前缀缓存', pain: '(原点) 历史不复用, 每轮都要把整段历史重新 prefill', fix: '历史 KV 留在 GPU 上复用; 用户一多, LRU 把别人的历史挤掉, 下一轮从头 prefill' },
          { name: '降级到 CPU / 磁盘', pain: '被挤掉的历史直接丢, 下一轮整段重算', fix: '溢出的 block 降到更大更慢的一层, 命中再搬回; 1 token 搬约 5 µs, 重算约 125 µs' },
          { name: '逐层取小', pain: '每次加载还有固定延迟, 链路慢、命中少时比重算还贵', fix: '每层各算加载和重算, 取小; 接一条 0.5 GB/s 慢远端时 TTFT 68.9 → 34.8 ms (代价模型)' },
        ],
      },
      code: 'llm_infer/m20_kv_offload/tiered_cache.py',
      points: [
        {
          title: '降级, 不是丢弃',
          body: 'TieredKVCache 把存储分成几层 (demo 里是 GPU、CPU、磁盘), 每一层都是一个 LRU 队列 (最近最少使用: 满了就先踢最久没碰过的)。\n- GPU 层满了, 最旧的 block 不是删掉, 而是放进 CPU 层; CPU 层满了, 最旧的放进磁盘层; 磁盘也满了才真正丢弃。\n- 某个 block 在 CPU 或磁盘层被再次命中, 就把它从那一层拿走, 提升回 GPU 层的最新端。\n每层容量在 demo 里是 GPU 64 个 block、CPU 192 个、磁盘 4096 个, 一个 block 装 16 个 token。demo [1] 模拟 8 个用户各聊 6 轮, 最后所有人的历史加起来约 442 个 block, GPU 只放得下 64 个: 只有 GPU 时 86.7% 的 prompt token 都 miss, 平均 TTFT (首 token 延迟) 51.5 ms; 加 CPU 层后有 29.5% 落在 CPU 层命中, TTFT 34.8 ms; 再加磁盘层又接住 41.4%, 降到 19.5 ms。剩下 15.7% 的 miss 是每轮的新消息和第一轮, 任何缓存都救不了。',
        },
        {
          title: '链断了后面就用不了',
          body: '这一章的索引和 m04 一样是链式 block hash: 第 i 块的 hash 是把前一块的 hash 和本块的 token 一起算出来的, 所以它标识的是 "前 i 块的整段前缀", 不只是这一块。\n查找时沿着这条链逐块找: 每一块先看 GPU 层有没有, 再看 CPU 层、磁盘层; 在第一个所有层都没有的块处停下。后面的块即使还躺在某一层也不可用: 比如第 3 块 miss、第 4 块命中, 第 4 块的 KV 是在前 3 块的基础上算出来的, 前缀的 KV 不全, 它接不上, 只能连同后面的一起重算。\n写入时 demo 还做了一件事: 把一条链从尾到头倒序 touch, 让链尾比链头先被驱逐。否则 LRU 会先踢掉第 0 块, 整条链立刻全部失效。这是在用 LRU 近似 radix tree 的 "先驱逐叶子"。',
        },
        {
          title: '交叉点 n*: 命中几块以上才值得搬',
          key: true,
          body: '搬回来还是重算, 要看命中了几个 block。代价模型里两条线:\n- load = 固定延迟 latency + n × 每块搬运时间。固定延迟是发起一次传输的开销 (demo 里 CPU 层 0.2 ms, 磁盘层 2 ms), 不管搬几块都要付一次; 每块搬运时间是块的字节数除以带宽 (磁盘 3 GB/s 时每块 0.699 ms)。\n- recompute = n × 每块重算时间。重算按 prefill 速度 8000 token/s 算, 一块 16 个 token 就是 2.00 ms。\n两条线交在 $n^*$ = latency / (每块重算 − 每块加载)。这个式子在说: 固定延迟要靠每一块省下来的那点时间慢慢摊平, 命中的块数超过 $n^*$ 以后加载才划算。demo [2] 里磁盘层的 $n^* = 1.54$ 块: 只命中 1 块时加载 2.70 ms、重算 2.00 ms, 重算更快; 命中 2 块时加载 3.40 ms、重算 4.00 ms, 加载更快。\n如果某一层每块加载时间比每块重算还长, 比如 demo [3] 那条 0.5 GB/s、20 ms 延迟的慢远端 (每块 4.194 ms), 命中再多也该重算。所以策略必须是 "两者取小", 而不是 "命中就加载": 盲目加载让 TTFT 从 34.81 ms 涨到 68.91 ms, 取小之后回到 34.81 ms。',
        },
      ],
      links: [
        { from: 'm04 PrefixCache', to: 'TieredKVCache', body: 'm04 的 PrefixCache 用链式 block hash 给 GPU 上的 KV block 建索引, 只有一层。TieredKVCache 用同样的链式 hash, 只是一个 hash 对应的 block 可能落在 GPU、CPU、磁盘中的任意一层, 查找时要逐层找。' },
        { from: 'Tier.bandwidth_GBps', to: 'load_ms', body: 'load_ms 用 Tier 的带宽算搬一个 block 要多久: 字节数除以带宽。注意这里带宽的单位是 GB/s, 按字节算; 后面 P/D 分离那一章用的是网络带宽 Gbps, 按比特算, 两者差一个 8 倍。' },
        { from: 'use_load', to: 'TTFT', body: 'use_load 是 serve 里每一层的那个判断: 加载时间不超过重算时间才加载。demo [3] 接一条 0.5 GB/s 的慢远端: "命中就加载" 把平均 TTFT 推到 68.9 ms, 比不加这一层 (34.8 ms) 还慢; 加上取小的判断后回到 34.8 ms, 其中命中慢层但选择重算的有 9456 个 token。' },
      ],
      sourceRows: [
        { concept: '代价模型', code: 'm20_kv_offload/tiered_cache.py:CostModel', takeaway: '只有两个公式: load = latency + bytes / bandwidth (固定延迟加上传输时间), recompute = tokens / 8000 tok/s (按 prefill 速度算重算时间)。每 token 131072 B 按 LLaMA-3-8B fp16 取。所有时间都由这两个式子算出, 不是测出来的。' },
        { concept: '降级', code: 'TieredKVCache._insert', takeaway: '往某一层放 block, 这一层满了就把最旧的 block 拿出来递归插入下一层, 而不是直接删掉; 只有最后一层满了才真正丢弃。' },
        { concept: '逐层决策', code: 'm20_kv_offload/tiered_cache.py:TieredKVCache.serve', takeaway: '处理一条 prompt 时, 对每一层命中的那批 block 各算一次 load 和 recompute, use_load = (load ≤ recompute); 为真才搬, 否则当 miss 重算。TTFT 是 miss 部分的重算时间, 加上各层取小后的时间之和。' },
        { concept: 'demo 结果 (模拟)', code: '8 个用户 × 6 轮', takeaway: '8 个用户各聊 6 轮, 每轮的 prompt 是整段历史加新消息。平均 TTFT: 只有 GPU 51.5 ms, 加 CPU 层 34.8 ms, 再加磁盘层 19.5 ms; 完全不缓存、全量重算是 59.42 ms。demo 断言每多一层 TTFT 都更低。' },
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
      subtitle: '读完你能说清 MoE 模型分到多张卡上服务时, 速度为什么只取决于最忙的那张卡, 以及 EPLB 为什么必须复制热门专家, 只重新摆放不够。',
      tldr: 'MoE (mixture of experts, 混合专家) 模型的一层里有很多个专家, 每个专家是一个小的前馈网络 (FFN), 每个 token 只挑其中 k 个来算。专家太多一张卡放不下, 就分到多张卡上, 这叫专家并行 (EP, expert parallelism)。每层要来回搬两趟 token:\n- dispatch (分发): 每个 token 被发到它选中的专家所在的 rank (一张卡算一个 rank)。所有卡同时互相发, 这种通信叫 all-to-all。\n- combine (汇总): 专家算完, 结果发回 token 原来的位置, 按 gate (路由给每个专家的权重) 加权求和。\ncombine 要等所有卡都算完, 所以一步的耗时取决于最忙的那张卡。路由一倾斜 (偏向少数热门专家), demo 里热专家所在 rank 的负载是平均的 2.95 倍, 平均每张卡只有 34% 的时间在干活。\nEPLB (expert parallelism load balancer, 专家并行负载均衡) 给热门专家多放几个冗余副本, 把发给它的 token 拆给各副本: 最忙与平均之比 (max/mean) 压到 1.02 倍, 模型输出逐位不变。',
      question: '把专家换个摆法不就均衡了吗? 为什么非得把热门专家复制几份?',
      evolution: {
        title: '从摆放到复制',
        subtitle: '根问题: 专家分在多张卡上, 每层都要等所有卡算完, 一步的耗时等于最忙那张卡。',
        steps: [
          { name: '按编号均分专家', pain: '(原点) 专家太多一张卡放不下, 每 token 只用其中 k 个', fix: '专家分到各 rank, token 经 all-to-all 去找; 路由一斜, 最忙 rank 达均值 2.95×' },
          { name: '训练时加均衡 loss', pain: '热 rank 在算, 其余在等, 平均利用率只剩 34%', fix: '训练时用辅助 loss 压倾斜; 推理时路由已经定死, 服务侧只能改放置' },
          { name: '按负载贪心摆放', pain: '推理侧改不了路由, 只能改专家放在哪张卡', fix: '重专家依次放到最轻的 rank: 2.95× → 1.48×; 最热专家一个就是平均 rank 负载的 1.42 倍' },
          { name: 'EPLB 冗余专家', pain: '单个专家不可分, 它一个就超过一张卡该分的量', fix: '给热专家加副本、把它的 token 拆开再摆: 加 4 个副本到 1.02×, 输出逐位不变' },
        ],
      },
      code: 'llm_infer/m21_moe_serving/moe.py',
      points: [
        {
          title: '看最忙的卡, 不看平均',
          key: true,
          body: '衡量 MoE 多卡服务的负载, 要看最忙那张卡 (max), 不看平均 (mean)。\n原因在 combine。一个 token 选了 k 个专家, 这 k 份结果可能来自不同的 rank, 要全部到齐才能加权求和, 下一层才能开始。所以 EP 每一层都要等齐所有 rank, 一步的耗时由最忙的那张卡决定, 其它卡算完了也只能等。\ndemo 的设置: 16 个专家、4 个 rank、每个 token 选 2 个专家 (top-2)、2048 个 token。router 故意加了偏置, 让编号小的专家更热 (专家 e 被选中的先验正比于 1/(e+1), 叫 Zipf 偏置)。按编号连续摆放 (专家 0 到 3 在 rank 0, 依此类推) 时, 4 个 rank 收到的分配数是 3016、851、125、104。平均 1024, 最大 3016, max/mean 是 2.95。\nmax/mean = 2.95 意味着平均每个 rank 只有 1/2.95 ≈ 34% 的时间在干活, 其余都在等最忙的那个。\n边界: 这里的负载就是每个 rank 处理的 (token, 专家) 分配数, 是代价模型。真实 grouped GEMM 的耗时和 token 数不严格成正比; demo 用数组分组模拟 all-to-all, 没有通信开销。',
        },
        {
          title: 'EPLB 的两步贪心',
          body: 'EPLB 分两步, 每一步都是贪心: 每次只做眼下最划算的那一个选择。先说两个词: slot 是卡上放一份专家权重的位置; 副本是把同一个专家的权重再放一份到另一个 slot 上。\n- 加副本: 每个专家的 "每副本负载" 是它的 token 数除以它现有的副本数。反复挑这个值最大的专家, 给它再加一个副本, 共加 n_redundant 次。demo 多出 4 个冗余 slot (每个 rank 多 1 个), 结果专家 0 有 4 份、专家 1 有 2 份。\n- 摆放: 把所有 slot 按每副本负载从大到小排好, 依次放到当前负载最轻、而且还有空位的 rank 上。每个 rank 的 slot 数相同。\n有副本的专家, 发给它的 token 轮流分给各副本。\n负载统计来自另一个历史 batch, 评测在新的 batch 上做。这模拟了真实系统的做法: 摆放要搬专家权重, 必须在 batch 到来之前按历史热度定好。只摆放、不加副本时 max/mean 是 1.48; 加 4 个副本后是 1.02, 4 个 rank 的负载是 1025、1033、992、1046。按最忙 rank 的负载估算, 一步的耗时比连续摆放缩短 65%。',
        },
        {
          title: '一个专家就超过一张卡的量',
          body: '只换摆法不够, 因为最热的那个专家一个就超过了一张卡该分到的量。\n算一下: 最热的专家 0 收到 1451 个分配。平均每个 rank 该分到 2048 × 2 / 4 = 1024 个。最热专家的负载除以平均 rank 负载是 1451 / 1024 = 1.42, 大于 1。\n单个专家不可分: 它的权重只能整个放在一个 slot 里, 不能切开放到两张卡上。所以不管怎么摆, 放专家 0 的那张卡至少有 1451 个分配, 怎么摆都不可能均衡。贪心无副本实际只能做到 1.48 倍。\n唯一的办法是复制它, 并拆分它的 token: 把专家 0 的权重多放几份到别的卡上, 发给它的 token 分给这些副本。副本权重完全相同, 一个 token 走哪个副本算出来都一样, 所以复制不改变模型输出。',
        },
      ],
      links: [
        { from: '阶段 2 MoE 路由', to: 'MoELayer.route', body: '阶段 2 讲过 MoE 的 top-k 路由: router 给每个 token 对每个专家打分, 挑分最高的 k 个。MoELayer.route 是同一个路由。这一章不关心它选得好不好, 只关心它选出来的结果在多张卡上造成怎样的负载分布。' },
        { from: '阶段 3 EP all-to-all', to: 'ep_forward', body: '阶段 3 讲过训练时专家并行的 all-to-all 通信, ep_forward 用数组分组模拟同样的 dispatch 和 combine。区别在治倾斜的手段: 训练可以加一个辅助 loss, 让 router 学得更均匀; 推理时模型已经训完, 路由已经定死, 只能靠调整专家放在哪张卡上。' },
        { from: 'eplb_placement', to: 'slot_of', body: 'eplb_placement 决定每个 slot 放哪个专家、放在哪个 rank。ep_forward 里的 slot_of 再决定每一条 (token, 专家) 分配具体去哪个 slot: 有副本的专家把自己的 token 轮流分给各个副本, 第 1 条给第 1 个副本, 第 2 条给第 2 个, 依此类推。' },
      ],
      sourceRows: [
        { concept: '路由', code: 'm21_moe_serving/moe.py:MoELayer.route', takeaway: 'router 给每个 token 对每个专家算一个分数 (logit), 取最高的 k 个。gate 是这 k 个专家的权重: 只在被选中的 k 个 logit 上做 softmax, 加起来等于 1, 不是对全部专家做 softmax。README 注明各模型的做法略有不同。' },
        { concept: 'dispatch/combine', code: 'm21_moe_serving/moe.py:ep_forward', takeaway: '每个 token 选 k 个专家, 就产生 k 条 (token, k) 分配, 2048 个 token、top-2 一共 4096 条。ep_forward 先按 slot_of 把每条分配送到对应 rank (dispatch), 每个 rank 对自己的每个 slot 批量算一次专家 FFN, 再把结果按 gate 加权加回 token 原位置 (combine)。rank_load[r] 就是发到 rank r 的 (token, k) 分配数, 各 rank 加起来恰好是 4096, demo 用 assert 检查。' },
        { concept: 'EPLB 加副本', code: 'm21_moe_serving/moe.py:eplb_placement', takeaway: '核心一行是 n_rep[argmax(load / n_rep)] += 1。这一行在说: n_rep 是每个专家现有的副本数, load / n_rep 是每个副本平均分到的 token 数; 找出这个值最大的专家, 给它加一个副本。一次加一个副本, 循环 n_redundant 次。之后再按每副本负载从大到小把 slot 依次放到最轻且有空位的 rank。' },
        { concept: '输出不变', code: 'max|Δ| = 0', takeaway: 'EPLB 只改 "逻辑专家放在哪个物理 slot" 这张映射表。路由选哪几个专家、gate 权重是多少都没动, 副本的权重又和原专家完全相同, 所以输出逐位相同。连续摆放、贪心摆放、EPLB 三种放法和逐 token 循环的基线相比, 最大差都是 0。' },
        { concept: 'demo 结果', code: '16 专家 / 4 rank / top-2', takeaway: '16 个专家、4 个 rank、每个 token 选 2 个专家、2048 个 token。最忙 rank 与平均之比 max/mean: 按编号连续摆放是 2.95×, 按负载贪心摆放但不加副本是 1.48×, EPLB 加 4 个副本后是 1.02×。' },
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
      subtitle: '读完你能说清 Quest 怎么用 block 摘要只读一小部分 KV 还不漏掉针, 以及这个办法在什么模型上不成立。',
      tldr: '长上下文下 decode 每生成一个 token 都要把全部 KV 读一遍, 可训练好的模型注意力高度集中, 绝大部分权重落在少数 token 上, 大部分 KV 读来几乎不起作用。本章的办法: 把 KV 切成 block, 每块常驻一份很小的摘要, 不读 KV 就先猜哪些块重要:\n- Quest: 摘要是每块 key 的逐维 min/max。\n- NSA/DSA: 摘要是压缩过的 key, 或者一个训练出来的低维 indexer。\n用当前 query 给摘要打分, 选分最高的 top-k 块, 只对它们做 attention。\n- needle (大海捞针: 少数几个 key 和 query 强对齐, 其余是噪声) 负载: 读 3.1% 的 KV, 误差就降到 0.023。\n- 随机权重模型: 注意力弥散, 没有可利用的集中性, 这个前提根本不成立。',
      question: 'Quest 为什么用"上界"打分, 而不是直接用 block 的均值?',
      evolution: {
        title: '少读 KV, 别漏掉针',
        subtitle: '根问题: 长上下文 decode 每步要把全部 KV 读一遍, 而注意力质量集中在很少的 token 上。',
        steps: [
          { name: '全量读 KV', pain: '(原点) decode 受带宽限制, 每步读的 KV 字节随上下文线性涨', fix: '每步读全部 KV, 结果精确; 大部分字节读来只分到很小的注意力' },
          { name: '固定位置 (sink + 窗口)', pain: '全读太贵, 大部分 KV 读了也几乎不起作用', fix: '只读开头几个和最近一段, 不看 query; 远处的关键 token (针) 落在窗口外就丢' },
          { name: '精确 top-k token', pain: '重要 token 可能在任何位置, 得看 query 才知道', fix: '先算全部 $q \\cdot k$ 再只读 top-k 的 V; 选得准, 但打分本身就读了全部 K' },
          { name: 'Quest 块摘要', year: 2024, pain: '打分要读全部 K, 省下的只有 V', fix: '每块常驻 min/max 摘要, q 算上界选 top-k 块: 读 3.1% 的 KV, 误差 0.023' },
        ],
      },
      code: 'llm_infer/m22_sparse_attention/sparse_attention.py',
      points: [
        {
          title: '廉价的上界打分',
          body: 'Quest 给每个 block 存两个向量: 这一块里所有 key 的逐维最小值 kmin 和最大值 kmax。打分时看 query 的每一维 $i$: 如果 $q_i$ 为正, 这一维的乘积最大是 $q_i \\cdot \\text{kmax}_i$; 为负则是 $q_i \\cdot \\text{kmin}_i$。逐维取大再求和, 得到 $\\sum_i \\max(q_i \\cdot \\text{kmin}_i,\\ q_i \\cdot \\text{kmax}_i)$, 它一定 $\\ge$ 这个 block 内任何一个 $q \\cdot k$。\n用上界打分的好处: 含有 "针" (和 query 强对齐的 key) 的块, 分数一定不会被低估, 所以不会漏掉。代价是偏松: demo 里上界平均是真实最大值的 6.7× (在各向同性的噪声上很松), 排序信息主要来自结构化的 key。不漏比估得准重要。demo 另外试了用块内 key 的均值打分 (NSA 压缩分支的最简替身), needle 负载上两者几乎一样。',
        },
        {
          title: '两块必须保留',
          body: 'select_blocks 有一个 forced 集合: 第 0 块 (attention sink, 上一章讲过的那几个承接多余注意力的开头 token) 和最后一块 (最近的 token) 永远选中, 并且算在 k 之内, 其余按分数取 top-(k−2)。丢掉第 0 块, softmax 的分母就少了最大的一项, 其余权重整体放大, 输出就崩了。\n这也是 demo 里 k=4 时召回只有 59.1% 的原因: 首尾占了 2 个名额, 只剩 2 个名额给 4 个 needle 块。',
        },
        {
          title: '前提是注意力足够尖',
          key: true,
          body: '稀疏 decode 的收益完全来自数据: 注意力质量要集中在少数 block 上。demo 的 needle 负载 (T=4096 切成 256 块, 4 个 needle 块各含 2 个和 q 对齐的 key):\n- k 从 4 加到 8 (读 1.6% → 3.1% 的 KV): 相对误差从 0.83 断崖掉到 0.023, 召回 (选中的块覆盖的真实注意力质量) 从 59.1% 到 97.7%。针全部进了 top-k 的那一刻, 其余 KV 就无关紧要了。\n- 同样读 3.1%, 随机选块误差 1.06; 随机读到 50% 误差仍然大于 1。\n换成随机权重的 TinyLM 就不成立: 它的注意力近乎均匀, 召回约等于读取比例, Quest 打分只比随机好几个点, 误差甚至不如随机 (k=8 时 0.62 对 0.51)。真实系统的 NSA、DSA 还把选块的 indexer 训练出来, 比 min/max 启发式准得多; 但归根结底, 稀疏性来自训练出来的注意力集中性, 不是算法本身带来的。',
        },
      ],
      links: [
        { from: 'm02 KV block', to: 'block 摘要', body: '分页 KV 本来就按 block 组织, 摘要直接挂在 block 上。每块两个向量 (kmin、kmax), 对比每块 bs 个 K 加 bs 个 V, 摘要大约只有 KV 的 1/16 ~ 1/8。' },
        { from: 'attention sink', to: 'select_blocks 的 forced', body: '上一章的 attention sink 说开头几个 token 承接着多余的注意力, 丢了 softmax 分母就崩。这里 select_blocks 的 forced 集合把第 0 块永远留下, 是同一个道理。' },
        { from: '阶段 2 DSA', to: 'lightning indexer', body: 'DeepSeek-V3.2 的 DSA 用一个训练出来的低维 indexer 按 token 选 top-k (2048 个), 不是按块。这里只有免训练的 min/max 选块, 是它的最简替身。' },
      ],
      sourceRows: [
        { concept: 'Quest 上界', code: 'm22_sparse_attention/sparse_attention.py:quest_upper_bound', takeaway: '对 query 的每一维, $q_i$ 为正取 kmax, 为负取 kmin, 乘起来逐维求和。demo 断言这个上界在两种数据上都 ≥ 块内真实的最大 $q \\cdot k$。' },
        { concept: '选块', code: 'm22_sparse_attention/sparse_attention.py:select_blocks', takeaway: 'forced = {0, nb−1} (第 0 块和最后一块) 永远选中, 其余按分数取 top-(k−2), 一共 k 块。' },
        { concept: '评测', code: 'm22_sparse_attention/sparse_attention.py:evaluate', takeaway: '两个指标一起看: 稀疏输出相对全量 attention 的相对 L2 误差, 和召回 (recall, 选中的块覆盖了多少真实注意力概率质量)。误差看结果差多少, 召回看漏没漏。' },
        { concept: 'demo 结果', code: 'T=4096, 256 blocks', takeaway: 'T=4096 切成 256 块, k=4 (读 1.6%) 误差 0.83 → k=8 (读 3.1%) 误差 0.023; 同样读 3.1%, 随机选块误差 1.06。' },
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
      subtitle: '读完你能说清 "多采几条再挑"、"按步搜索"、"写完再自查" 这几种花 token 的方式各靠什么挑答案, 以及哪一种错它们都消不掉。',
      tldr: '模型不换, 推理时多花 token 也能涨正确率。办法有三类: 同一道题多采几条再挑一条 (best-of-N、多数投票), 每一步多采几个候选、边走边剪 (PRM beam), 或者写完再自查几遍 (budget forcing)。\n关键是靠什么挑。错误分两种: 粗心错是随机的, 换个样本就可能对, 多采样就能消掉; 误解是系统性的, 模型自己最常给的答案 (众数) 本身就错, 投票和自查只会更稳定地错。要越过它, 得靠外部判分器: demo 的陷阱题上, 多数投票 N 越大越错 (0.254 降到 0.108), best-of-N 加判分器到 N=64 能到 0.970。',
      question: '多采几条再投票, 为什么有的题反而越投越错?',
      evolution: {
        title: '挑答案的人越来越懂行',
        subtitle: '根问题: 多步推理一步错整条错, 换更大的模型又很贵。',
        steps: [
          { name: '单条生成', pain: '(原点) 每步对 $p$, 整条只对 $p^K$: $p=0.75$、$K=4$ 时 0.316', fix: '写一条就交; 玩具任务上 T=1 单条正确率 0.275' },
          { name: '多数投票', year: 2022, pain: '一步粗心整条错, 换个样本就可能对', fix: '采 N 条取最终答案的众数; 在 0.71 饱和, 陷阱题上 N 越大越稳定地错' },
          { name: 'best-of-N + ORM', pain: '众数本身错时投票只会更稳地错: 陷阱题 0.254 → 0.108', fix: '外部判分器看终点, 从 N 条里挑: N=64 到 0.970; 错链的 K 步 token 照付' },
          { name: 'PRM beam', pain: 'best-of-N 到终点才判, 错链的每一步都白花 token', fix: '给每一步打分的判分器 (PRM) 逐步打分, 错步当场剪掉: 28 token 时 0.865, 同预算 best-of-7 只有 0.747' },
        ],
      },
      code: 'llm_infer/m23_test_time_compute/{tts.py,demo.py} (每一步的采样复用 m10 的 sample)',
      points: [
        {
          title: '一步错, 整条错',
          body: '一道要推 $K$ 步的题, 每步答对的概率是 $p$, 整条都对的概率只有 $p^K$。$p=0.75$、$K=4$ 时只剩 0.316。原因是一步错了, 后面每一步都在错的中间值上 "正确地" 继续算, 最后的答案一定错。\n换更大的模型很贵。另一条路是推理时多花 token: 同一个模型, 多写几条、多查几遍, 再想办法挑出对的那条。\ndemo 的玩具任务: 200 道 4 步算术题 (一个起始数, 连做 4 次加减乘), 模型每一步从 6 个候选里采样一个: 正确值、4 种粗心错 (差 ±1 或 ±10)、看错运算符。温度 T=1 时单条正确率 0.275。',
        },
        {
          title: '三条花 token 的路',
          body: '- 并行: 同一道题采 $N$ 条完整的链, 再从里面挑一条。挑法有两种: best-of-N 让 ORM (outcome reward model, 只看最终答案打分的判分器) 给每条打分取最高; 多数投票 (self-consistency) 不要判分器, 直接取出现次数最多的最终答案。\n- 按步搜索: 每一步多采几个候选, 让 PRM (process reward model, 给每一步打分的判分器) 当场打分, 分低的错步立刻剪掉, 只留分高的继续往下走。\n- 串行: 先写完一条, 再一遍遍自查, 发现错就从那一步重写。budget forcing 控制想多久: 到预算上限就强行截断, 想停但没到上限就追加一个 "Wait" 让它再查一遍。\n这一章的判分器是程序化的 (真值加噪声), 真实系统里 ORM 和 PRM 都是训练出来的模型。',
        },
        {
          title: '投票消不掉系统性的错',
          key: true,
          body: '错分两种:\n- 粗心错: 随机的, 换个样本就可能对。多采几条, 众数会落在正确答案上。\n- 误解: 系统性的, 模型自己最常给的答案 (众数) 本身就是错的。demo 里 32.5% 的题带一个陷阱步: 在那一步, "看错运算符" 这个候选的 logit 是 3.5, 高过正确值的 3.0, 所以模型大概率看错, 而且每次都这么错。\n两种挑法的差别就在第二种错上:\n- 多数投票: 只用模型自己的分布, 采样越多, 样本的众数越接近分布的众数。无陷阱题上它随 N 一路涨到 1.000; 整体在 0.71 饱和 (N=16 时 0.706, N=64 时 0.710); 陷阱题上 N 越大越稳定地错, 从 N=16 的 0.254 掉到 N=64 的 0.108。\n- best-of-N: 让外部判分器 ORM 挑。它不看哪个答案最多, 只看哪条的最终答案对, 所以能从少数派里认出正确答案, N=64 到 0.970。\n有完美判分器时 best-of-N 的上限是 pass@N (N 条里至少一条对的概率), N=64 时 1.000; demo 的 ORM 带噪声, 所以 0.970 低于它。',
        },
        {
          title: 'PRM 在错步出现时就剪掉',
          body: 'best-of-N 到终点才判: 一条链第 1 步就错了, 后面 $K-1$ 步的 token 照样要花。PRM beam 每一步都给候选打分, 错步当场剪掉, 省下的 token 给别的候选。\ndemo 用 width × expand 描述一次搜索: 每一步保留 width 条链, 每条链各采 expand 个下一步, 再按 PRM 累计分留下 width 条。花的 token 数是 expand + (K−1) × width × expand。\n- 28 token (width 2 × expand 4): PRM beam 0.865, 花同样 token 的 best-of-7 只有 0.747。\n- 52 token (4 × 4): PRM beam 0.830, best-of-13 是 0.849, 打平甚至略输。\nbeam 加宽后自己也从 0.865 掉到 0.830。大预算输在判分噪声: 候选一多, 总有错步被噪声抬高、混进来。把 PRM 的噪声去掉 ($\\sigma=0$), 同样 4 × 4 的配置到 0.970。真实系统里 PRM 每一步都要跑一次前向, 这笔账 demo 没算。',
        },
        {
          title: 'budget forcing: 截断就崩, 延长才涨',
          body: '记思考预算为 $B$ 个 token (这一章的 $B$ 是预算, 不是 batch)。模型写完一条链后会想停 (demo 里每次以 0.6 的概率想停); 到了 $B$ 就强行截断, 以当前中间值作答; 想停但还没到 $B$, 就追加 "Wait", 让它再自查一遍, 发现错就从那一步重写。\n- B=2: 0.055。第一遍都没写完就被截断, 只能交中间值。\n- B=4 到 8: 都是 0.240。一次重写要能把后面几步全写完才算数, 预算不够, 查出错也改不完。\n- B=16: 0.460; B=64: 0.815。无陷阱题到 1.000, 陷阱题只有 0.431。\n不干预时模型平均只想 6.5 个 token 就停, 正确率 0.305。陷阱题上涨不上去, 是因为自查看不见自己的误解: demo 的自查规则里, 粗心错以 0.5 的概率被发现, 陷阱步上的误解永远被认为是对的, 预算再多也改不掉。',
        },
        {
          title: '玩具和真实系统差在哪',
          body: '- greedy 太好: 玩具里随机错只来自采样噪声, 所以不采样、每步取最大概率 (T=0) 就有 0.675, 恰好等于无陷阱题的比例。\n- 别读成 "greedy 就够了": 真实模型的 greedy 也犯随机错, 长思考和采样的收益比这里更实在。\n- 判分器是程序化的: ORM / PRM = 真值 + $\\mathcal{N}(0, 0.5^2)$ 的高斯噪声。真实系统里它们是训练出来的, 也有自己的盲区, 遇上盲区同样越不过去。\n- token 账偏乐观: 没算判分器的前向。真实 PRM beam 每步要给 width × expand 个候选各跑一次 PRM。\n- 自查规则写死: 发现粗心错的概率 0.5、想停的概率 0.6, 对误解全盲。o1 / R1 的长思考是用强化学习训出来的, 自查能力和何时停都不是常数。',
        },
      ],
      links: [
        { from: 'm10 sample', to: 'sample_step', body: '这一章的 "模型" 不是神经网络: 每一步给 6 个候选各一个 logit, 用 m10 的 sample 从里面抽一个 token。T=0 就是 greedy, 取 logit 最大的那个。' },
        { from: '奖励模型 (阶段 4)', to: 'orm / prm', body: '阶段 4 讲的奖励模型是训练出来给回答打分的。ORM 用 "最终答案对不对" 做标签训练打分头; PRM 要每一步的标签, 可以人工标 (PRM800K), 也可以 Monte-Carlo 自动标 (Math-Shepherd: 从这一步往后采 n 条, 能走到正确答案的比例就是这一步的分)。本章用 "真值加噪声" 代替它们。' },
        { from: 'prm_beam_search', to: 'beam search (解码一章)', body: '两者骨架一样: 每步只留 top-width 条。区别在排序依据: 解码那一章的 beam 按累计 log 概率排 (模型自己觉得哪条最可能), 这里按 PRM 的累计分排 (外部判分器觉得哪条走得对)。' },
      ],
      sourceRows: [
        { concept: '模型的一步', code: 'm23_test_time_compute/tts.py:step_dist', takeaway: '每一步 6 个候选: 正确值、4 种粗心错 (±1, ±10)、看错运算符。logits 是 [3.0, 0.3×4, 0.3 或 3.5]: 普通步上正确值 3.0 最高, 每步答对的概率 0.75; 陷阱步上 "看错运算符" 那一项是 3.5, 比正确值还高, 模型的众数就是错的。' },
        { concept: 'best-of-N', code: 'm23_test_time_compute/tts.py:best_of_n', takeaway: 'N 条链各让 ORM 打一次分, 取分最高那条的最终答案。ORM 是真值加噪声, 所以偶尔会挑错。' },
        { concept: 'self-consistency', code: 'm23_test_time_compute/tts.py:majority_vote', takeaway: '最终答案取出现次数最多的, 不需要判分器, 但要求答案能比较相等 (数学题可以)。开放式回答没法直接投票。' },
        { concept: 'PRM beam', code: 'm23_test_time_compute/tts.py:prm_beam_search', takeaway: '每条 beam 采 expand 个下一步, 全部候选按 PRM 累计分排序留 width 条。token 数 = expand + (K−1)·width·expand。' },
        { concept: '长思考', code: 'm23_test_time_compute/tts.py:think', takeaway: '写完一条链后一遍遍自查, 发现错就从那一步重写。min_tokens = max_tokens = B 就是 budget forcing: 不到 B 不许停, 到 B 必须停。' },
        { concept: '自查的盲区', code: 'm23_test_time_compute/tts.py:believes_ok', takeaway: '真对的一步会被认对; 陷阱步上自己的误解也会被认对 (永远发现不了); 粗心错以 0.5 的概率被发现。所以自查只能修粗心错。' },
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
      subtitle: '读完你能解释: 把请求发给缓存命中最多的副本, 为什么首 token 延迟 (TTFT) 反而最差, 以及负载阈值这个旋钮怎么在命中率和负载均衡之间取舍。',
      tldr: '一个模型扛不住流量时, 会起好几份一模一样的推理引擎, 每一份叫一个副本 (replica)。demo 起了 8 个副本, 每个副本有自己的前缀缓存: 把算过的 prompt 开头那段的 KV (模型为每个 token 算出的 K、V 向量, 生成时要反复用) 存起来, 下次遇到同样的开头直接复用, 不用重算。副本之间不共享 KV。\n- 前缀感知路由: 先问每个副本 "你缓存里有这个请求多长的开头", 把请求送到最长的那个。前缀命中率从 59.3% 升到 93.9%。\n- 新问题: 热门 system prompt 只在最先缓存它的副本上能命中, 新对话全被吸到那里。8 个副本只有 3 个在干活, 估算的平均 TTFT (从请求到达到吐出第一个 token 的时间) 从 0.469 s 爆到 105.954 s。\n- 负载阈值兜底: 最佳副本比最闲副本多积压的活超过一个阈值, 就让位给最闲的。阈值 0.5 s 时平均 TTFT 0.284 s, 本例最优。',
      question: '起了 8 个副本, 每个都有自己的前缀缓存, 新来的请求该发给谁?',
      evolution: {
        title: '先看内容, 再看负载',
        subtitle: '根问题: 一个副本扛不住流量要起多个, 各副本的前缀缓存彼此不共享。',
        steps: [
          { name: '轮询 / 最少负载', pain: '(原点) 8 个副本各有前缀缓存, 彼此不共享 KV', fix: '轮流发, 或发给积压最少的, 都不看内容; 下一轮多半换副本, 40.7% 的 prompt token 要重新 prefill' },
          { name: '前缀感知路由', pain: '历史 KV 在上一个副本上, 换副本就得整段重算', fix: '问每个副本能命中多长前缀, 送给最长的: 命中率 59.3% → 93.9%' },
          { name: '前缀优先 + 负载阈值', pain: '热门 system prompt 把新对话全吸到一个副本, 最忙副本的活是平均的 4.18 倍', fix: '最佳比最闲多积压超过阈值就让位; 阈值 0.5 s 时 TTFT 0.284 s, 热点前缀自然复制开' },
        ],
      },
      code: 'llm_infer/m27_multi_replica_routing/{router.py,demo.py} (每个副本复用 m05 的 RadixCache)',
      points: [
        {
          title: '不看内容, 历史就白算',
          body: '多轮对话的每一轮, 都要把前面的对话全部带上重新发一遍。第 $k$ 轮的 prompt 是 system prompt 加上前 $k-1$ 轮的全部历史, 再加这一轮的新消息。同一段对话越往后, prompt 越长, 而开头一大段和上一轮完全相同。\n上一轮在哪个副本上算, 这段历史的 KV 就只缓存在哪个副本上。\n- 轮询 (round_robin): 第 k 个请求发给第 k mod 8 个副本, 轮着来。\n- 最少负载 (least_load): 发给手上积压的活最少的副本。\n这两种都不看内容, 下一轮多半落到别的副本。那里只缓存了大家共用的 system prompt, 历史要整段重新 prefill (把这段 prompt 重新送进模型, 算一遍 KV)。demo 里两种策略都有 40.7% 的 prompt token 要重新 prefill。\n前缀感知路由问每个副本 "你缓存里有这个请求多长的前缀", 送给最长的。下一轮会回到上一轮的副本, 历史直接命中, miss (需要重算的比例) 降到 6.1%。',
        },
        {
          title: '命中率最高, TTFT 最差',
          body: '纯前缀感知的命中率最高, TTFT 却最差: 命中省下的 prefill, 抵不过排队的时间。\ndemo 的请求流: 1084 个请求共用 4 个 system prompt, 每个 512 个 token, 流行度分别是 48% / 24% / 16% / 12%, 最热的那个占将近一半。\n一段新对话到来时, 只有它的 system prompt 可能已被缓存。只有最先缓存这个 system prompt 的副本能命中 512 个 token, 别的副本命中 0, 于是前缀感知路由永远选它。后续每一轮的历史也只在那个副本上, 也永远选它。结果:\n- 8 个副本只有 3 个在干活, 分到的活占比 52% / 32% / 16%, 其余 5 个闲着。4 个 system prompt 只落到 3 个副本上, 因为有两个落到了同一个副本。\n- 负载 max/mean 4.18: 最忙副本累计的活是平均的 4.18 倍。\n- 平均 TTFT 105.954 s, 而最少负载只有 0.469 s。热点副本的活超过了它的处理能力, 队列越排越长。这是队列模型过载发散的结果, 只说明 "会爆", 数值本身没有意义。',
        },
        {
          title: '负载阈值兜底',
          key: true,
          body: '负载阈值兜底, 是在前缀感知路由上加一条规则: 先照常找出命中最长的最佳副本; 如果它的积压比最闲副本多出一个阈值, 就放弃缓存, 改走最少负载。积压 (backlog) 是副本手上还没做完的活, 按秒计, 路由器把它当作负载。\n这条规则有个顺带的好处: 被分流过去的副本处理完请求, 也把热点 system prompt 存进了自己的缓存。之后新对话在两个副本上都能命中这 512 个 token, 平手时按负载选, 热点前缀就自然复制到了多个副本上。\n阈值是个旋钮: 太松等于没兜底, 太紧就几乎总在放弃缓存。demo 从松到紧扫了一遍, 每行依次是命中率 / max/mean / 平均 TTFT:\n- 4s (太松): 93.4% / 1.45 / 0.689 s。\n- 1s: 83.6% / 1.19 / 0.314 s, 比最少负载 (0.469 s) 低 33%。\n- 0.5s: 77.9% / 1.12 / 0.284 s, 本例 TTFT 最优。\n- 0.05s (太紧): 72.9% / 1.12 / 0.308 s, 比最优回升, 在往最少负载退化。\n阈值越紧, 命中率越低。TTFT 最好的点落在中间: 太松输给排队, 太紧输给重算。',
        },
        {
          title: '探测必须只读',
          body: '每来一个请求, 路由器都要问一遍全部 8 个副本 "你能命中多长", 这叫探测。探测只是问问, 最后只有 1 个副本真的处理这个请求。\n每个副本的前缀缓存是 m05 的 RadixCache, 一棵 radix tree (基数树: 每条边存一段 token, 从根往下走过的路就是一段前缀)。它查最长前缀用的 match 会顺手改树:\n- split: 匹配停在一条边的中间时, 把这条边切成两段。\n- 刷新 LRU: 把沿途节点的最近使用时间 last_used 更新成现在。缓存满了要删东西时, 按 LRU (least recently used, 最久没用的先删) 从叶子删起, 删谁就看这个时间。\n如果拿 match 去探测, 没被选中的 7 个副本也会被记成 "刚用过", 它们的 LRU 顺序就被搅乱了, 不再反映真实的使用情况。\n所以 router.py 单独写了一个只读的 prefix_len: 走法和 match 相同, 但不 split、不刷新 last_used, 不改树。选定副本之后, simulate 才对那一个副本真正调用 match。探测不等于使用。',
        },
        {
          title: '哪些数是真跑的, 哪些是模型算的',
          body: '这一页的数字分两类, 读的时候要分开看。\n- TTFT 来自代价模型: 每个副本被简化成一条 FIFO (先进先出) 工作队列。一个请求的活是 miss 部分的 prefill 加上回复的 decode, prefill 按 1000 tok/s、decode 按 100 tok/s 的固定速度做完。TTFT 估算为到达时队列里的积压, 加上本请求 miss 部分的 prefill 时间。\n- 参数刻意调慢: 这样副本利用率在 30~40%, 排队才看得见。利用率很低时, 三种策略的 TTFT 差别只剩 prefill 本身。\n- 真实引擎: 做 continuous batching (连续批, 每一步都重新组 batch, 新请求随时插进来), decode 不会整段挡住后来请求的 prefill。过载表现为 batch 变大、TPOT (time per output token, 每出一个 token 的耗时) 变差。\n- 命中率是真跑的: 每个副本是一个容量 24k token 的真实 RadixCache, 命中长度是实际匹配出来的。轮询的 59.3% 主要来自 512 token 的 system prompt, 每个副本很快都缓存了它。\n- 路由器看得太清楚: 这里直接读副本的真实 radix tree。SGLang 的 router 自己维护一棵近似树, 只记录发出去的请求, 不知道副本的驱逐。',
        },
      ],
      links: [
        { from: 'm05 RadixCache', to: 'Replica.cache', body: 'm05 的 RadixCache 是一棵按 token 存前缀的基数树。这里每个副本的 cache 直接复用它, 没有另写一份, 所以命中长度是真跑出来的; 容量满了也按 m05 的规则从叶子按 LRU 驱逐。' },
        { from: 'match (会改树)', to: 'prefix_len (只读)', body: 'prefix_len 照搬 match 的走法: 从根出发按 token 往下走, 走到对不上为止。区别是去掉了 split (切边) 和 LRU 刷新, 这样探测 8 个副本时, 不会扰动那 7 个没被选中的副本。' },
        { from: 'EPLB (MoE serving)', to: '负载阈值', body: '两边是同一个病: 热点把负载压在一处。MoE serving 那章的 EPLB 给热门专家多放几个副本, 把它的 token 拆开; 这里靠负载阈值把请求分流出去, 让热点前缀在多个副本上各缓存一份。' },
      ],
      sourceRows: [
        { concept: '只读探测', code: 'm27_multi_replica_routing/router.py:prefix_len', takeaway: '从树根出发, 用下一个 token 找对应的子边, 沿边逐个 token 比较, 对不上就返回已匹配的长度。和 RadixCache.match 走法相同, 但不 split 边、不刷新 last_used (LRU 用的最近使用时间), 树一点都不改。' },
        { concept: '三种策略', code: 'm27_multi_replica_routing/router.py:route', takeaway: 'round_robin 按请求序号轮流发; least_load 发给积压最少的。prefix 先算 best = argmax(prefix_len, −backlog): 先比命中长度, 一样长就选积压少的。再检查 backlog[best] − min backlog > threshold: 这个式子在说最佳副本比最闲副本多积压的活超过了阈值, 成立就改走最少负载。连积压也一样的平手, 按请求序号轮转打散, 免得空闲时全压给 0 号副本。threshold 默认是无穷大, 这时永远不会改走, 就是纯前缀感知、永不放弃缓存。' },
        { concept: '模拟', code: 'm27_multi_replica_routing/router.py:simulate', takeaway: '逐个请求回放: 先让各副本按时间流逝消化积压, 再路由, 再对选中的副本调 match 算命中, 记下 TTFT, 把这个请求的活加进积压, 最后把 prompt 加回复插进缓存, 超过容量就驱逐。TTFT 估算为到达时积压加上 miss 部分的 prefill。命中至少留最后 1 个 token 重算, 才有 logits (模型对下一个 token 的打分), 才能采出第一个输出 token。' },
        { concept: '请求流', code: 'm27_multi_replica_routing/router.py:chat_stream', takeaway: '生成合成的多轮对话请求流: 200 段对话, 每段 3~8 轮, 第一轮在 400 s 内随机到达。共 4 个 system prompt, 各 512 个 token, 流行度 ∝ 1/rank: 被选中的概率和排名成反比, 第 1 名是第 4 名的 4 倍。每轮的 prompt 是 system prompt 加全部历史加新消息。两轮之间的思考时间服从均值 15 s 的指数分布 Exp(15 s)。' },
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

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "infer-scheduler": [
    {
      "title": "等待队列要有上限",
      "body": "调度器每一步从等待队列里挑请求。请求来得比处理得快时, 队列就会越排越长。这条讲队列该不该设上限。\nm28 用代价模型模拟一个服务: 请求按固定的随机分布到达, 其中 25% 是长请求; 同时最多 4 条请求在跑 (4 个活跃槽); 每条请求有 1 秒截止时间, 到点还没完成就算超时。到达率从每秒 8 个扫到 96 个, 比两种队列:\n- 有界队列: 最多让 8 个请求排队, 再来的直接拒绝。\n- 无界队列: 来多少排多少, 从不主动拒绝。\n到达率 48 个/秒时差别最明显。这里的 SLO (服务等级目标) 指同时满足三条: 首 token 不晚于 200 ms、相邻 token 间隔不超过 6 ms、1 秒内完成。每秒满足 SLO 的请求数也叫 goodput (好吞吐)。\n- 有界: 队列最长 8 个, 拒绝 131 个; 每秒 16.5 个请求满足 SLO。\n- 无界: 队列最长排到 52 个, 超时 93 个; 每秒满足 SLO 的只剩 1.55 个。\n无限排队并没有让服务多处理请求, 只是把「立刻被拒」变成了「排很久再超时」。\n每个请求最后只落进一种终态: 拒绝、超时、取消 (每 17 个请求里有 1 个在 100 ms 后主动取消)、完成, 四种分开统计。每种终态都会把这条请求占的 KV 页 (分页 KV 的 block) 还回去, 模拟结束时残留页为 0。\n毫秒数来自代价模型: 一步耗时 2 + 0.5×活跃请求数 ms。它只用来比较两种队列, 不能拿来配置真实 GPU 的容量。",
      "source": "llm_infer/m28_service_capacity/demo.py:simulate"
    }
  ],
  "infer-multi-replica-routing": [
    {
      "title": "不要只报平均吞吐",
      "body": "看一个服务扛不扛得住, 只报平均吞吐会被骗。m28 (一个模拟排队的代价模型, 同时最多 4 条请求在跑) 在到达率 48 个/秒时给了一个反例:\n- 无界队列 (从不拒绝, 来多少排多少) 每秒完成 34.7 个请求, 看着比有界队列 (最多排 8 个, 再来就拒绝) 的 34.2 还高。\n- 可它的 TTFT p99 是 955 ms: 把完成请求的首 token 延迟从小到大排, 排在第 99% 的那个要等 955 ms。真正满足 SLO (服务等级目标) 的请求每秒只有 1.55 个, 有界队列是 16.5 个。\n读这几个指标要注意:\n- TTFT 从请求到达算起, 排队的时间也算在里面。ITL (inter-token latency) 看的是相邻两个输出 token 之间隔了多久。\n- p95/p99 只在完成了的请求上统计。被拒绝、超时、取消的请求都不进统计。所以分位数必须和这三类未完成请求的比例一起报, 否则就是幸存者偏差: 慢的都被丢掉了, 剩下的自然显得快。\n- goodput (好吞吐): 同时满足 TTFT ≤ 200 ms、最大 ITL ≤ 6 ms、1 秒内完成这三条的请求数, 除以观测时间。观测时间包括到达停止后把队列排空的那一段, 不能只拿到达期做分母。",
      "source": "llm_infer/m28_service_capacity/demo.py:experiment"
    }
  ],
  "infer-engine": [
    {
      "title": "机制对了, 还要看负载下能不能用",
      "body": "上面的资源账证明引擎在机制上是对的。上了负载, 还要再对一本账: 每个请求最后去了哪。这里有三个容易混的数:\n- 模型吞吐: 每秒生成多少 token。\n- 接收量: 有多少请求被收进来, 没被拒绝。\n- 好吞吐 (goodput): 每秒有多少请求在延迟目标内完成。\nm28 用代价模型模拟到达率 96 个/秒、有界队列 (最多排 8 个, 再来就拒绝) 的情形: 到达 956 个 = 完成 364 + 拒绝 568 + 超时 0 + 取消 24, 结束时 KV 页全部还回, 占页为 0。四种终态加起来等于到达数, 页也没漏, 这本账才算对上。\n同一档到达率换成无界队列 (从不拒绝): 首 token 延迟 TTFT 的 p99 (从小到大排在第 99% 的那个) 是 960 ms, 超时 693 个。有界队列的 p99 是 455 ms, 看着好得多, 但那是拒掉 568 个请求换来的。\n所以拒绝率提高、p99 变好看, 不能就此说服务变好了, 可能只是把慢请求丢掉了。要把 offered load (到达的请求量)、完成数、goodput 和全部终态放在一起看。",
      "source": "llm_infer/m28_service_capacity/demo.py:simulate"
    }
  ]
}
for (const [route, rows] of Object.entries(supplements)) {
  const page = topic.pages[route]
  if (!page) throw new Error(`缺少补充章节: ${route}`)
  for (const { title, body, source } of rows) {
    page.points.push({ title, body })
    if (source) page.source = [...new Set([].concat(page.source || [], source))]
  }
}

// 审查后的主讲职责与前置链接。
topic.pages["infer-compute"].points.push({"title": "FLOPs 一样, 耗时不一定一样", "body": "本章几件事都是「数学一样, 快慢不同」。这条用 m32 把它实测出来: FLOPs (浮点运算次数) 相同, 不代表执行时间相同。\nm32 用两种写法算同一件事, 矩阵乘再加偏置:\n- x@w+b: 先做矩阵乘, 再单独做一次加法。\n- torch.addmm: 一次调用把乘和加都做完。\n两者的逻辑 FLOPs 都是 1,056,768。但 profiler (记录每个算子调用、内存分配和耗时的工具) 显示: 两版各有 1 次矩阵乘; ATen 调用 (PyTorch 底层算子库被调用的次数) 是 6 次对 7 次; CPU 上分配的内存是 65,536 字节对 32,768 字节, x@w+b 多分配了一份中间结果。算子调用、内存分配和 host 开销, 都要测出来才知道。\n测法分三步: 先空跑 3 次 warmup (不计时), 再独立计时 7 次, 最后单独采一次 profiler。profiler 要和计时分开, 因为它本身会拖慢被测的代码。同时记下 Python/PyTorch 版本、设备、线程数 (这里是 1) 和重复次数。\n网页只展示稳定的事件计数; 毫秒数跟机器有关, 留在命令行 (CLI) 输出的 JSON 里。\n两个读数各有边界: 分配字节是所有分配事件加起来的总量, 不是同一时刻占用的峰值; ATen 调用数也不等于 GPU 上实际跑的 kernel 数。"})
topic.pages["infer-scheduler"].related = [{"route": "infer-multi-replica-routing", "label": "到达率上升后, 排队会发生什么", "reason": "本章只说明等待队列要设上限。请求越来越多时, 分位数和 goodput 该怎么读、为什么不能只看平均吞吐, 在多副本路由那一章。"}]
topic.pages["infer-engine"].related = [{"route": "infer-multi-replica-routing", "label": "到达率上升后, 排队会发生什么", "reason": "本章只要求每个请求的终态对上账。指标该怎么读 (分位数为什么只算完成的请求、goodput 怎么定义), 在那一章。"}, {"route": "finetune-runs", "label": "模型交付与独立进程加载", "reason": "引擎要加载的不只是权重文件, 还有词表、模板、生成参数这些约定, 都记在交付清单 manifest 里。那一章演示新进程怎么按清单加载, 错配时怎么拒绝。"}]
topic.pages["infer-decode-control"].related = [{"route": "basic-optim-sample", "label": "复习 softmax 温度", "reason": "同一个温度公式: 基础页演示它怎么改变分布, 本章讲它在采样里怎么用。"}]
topic.pages["infer-kv-memory"].related = [{"route": "attention", "label": "复习注意力掩码", "reason": "掩码 (mask) 决定每个 token 能看见哪些 token。本章只讲 K/V 怎么存; 拖滑杆看掩码怎么变的实验在注意力那一章。"}]
export default topic

topic.pages["infer-scheduler"].points.push({"title": "多租户: 总吞吐会掩盖谁在等", "body": "同一个推理服务常常被几个调用方共用, 每个调用方叫一个租户。只看全局吞吐, 看不出是哪个租户在挨饿。\nm33 用离散的服务量单位 tick 来模拟 (tick 是服务量, 不是 GPU 毫秒), 跑一条固定的到达轨迹:\n- A 租户连续发 60 个长请求, 每个要服务 5 tick。\n- B 租户每 15 tick 发 1 个短请求, 每个要服务 2 tick, 一共 4 个。\n- 请求从到达起 40 tick 内没完成就算超时。\n四种调度策略跑同一条轨迹:\n- FIFO (先到先服务): B 只完成 1 个, 其余 3 个超时。\n- 固定优先级 (A 优先): B 一个都没完成, 从来没被服务过。\n- 公平轮转: B 4 个全完成, 第一次拿到服务前的等待是 0; A 的完成数从 9 降到 8。\n- 公平 + 配额 (每个租户最多排 8 个): A 的队列峰值从 40 降到 8, 代价是拒绝了 41 个。\n只看 A 或总完成数, 前三种策略几乎一样 (A 完成 9 / 9 / 8)。要按租户分别看等待、超时和 goodput (这里指端到端延迟不超过 10 tick 的完成数, 除以观测时长), 才看得到 B 在挨饿。\n公平轮转的做法: 两个租户轮流拿一个等成本的服务量子 (每次服务 1 tick)。长请求一次没跑完, 就回到自己租户队列的最前面, 下一轮再接着跑, 不会挡住另一个租户拿到它的那一份。\n边界: 真实请求每个 token 的代价不一样, 要结合 prefill/decode、KV 占用和预算来估; 一条有限的轨迹也证明不了任意负载下都不会饥饿。"})
topic.pages["infer-scheduler"].source.push("llm_infer/m33_tenant_scheduling/demo.py:experiment")

topic.pages["infer-prefix-radix"].points.push({"title": "相同 token 不一定能复用 KV", "body": "前面的复用都只比 token。但 KV 是模型算出来的, 它的身份不只是 token, 还包括解释这些 token 的模型、adapter (挂在底模上的微调小模块, 比如 LoRA) 和输入约定。缓存键 (查缓存用的键) 里少了哪一样, 都可能复用错。\nm34 先让可信身份 A 把 [1,2,3,4] 在 model-v1 / adapter-v1 下算出的 KV 写进缓存, 再换条件请求同一串 token:\n- 跨租户共用 (缓存键里不带租户的共享域 scope): B 命中了 A 写的条目。命中更快, 而这个快慢本身就泄露了信息: B 能由此推断别的租户用过这个前缀。\n- 隔离命名空间 (键里带上 scope): B 未命中, 缓存条目变成 2 条。同一个域里的 A 再来, 照样命中。\n- 忽略版本 (键里不带模型和 adapter 版本): 换了 adapter 仍然命中, 返回的是旧 adapter 下算的值, 和当前模型约定对不上, 而且没有任何报错。\n- 模型或 adapter 变化 (键里带版本): 未命中, 重新算。\n- 伪造租户声明: B 拿自己的凭证, 却声称自己是 A。网关按凭证判定身份, 声明和凭证对不上, 直接拒绝。\n所以共享域要由可信的身份网关从凭证推出来 (demo 用 HMAC 这种带密钥的哈希, 从租户身份生成命名空间)。客户端自己声明的租户或 salt (加进缓存键的一段随机串) 不能覆盖鉴权结果。同域可以复用, 跨域要有明确的命名空间隔离。\n边界: salt 只隔离缓存键, 不隔离进程、网络和显存。命中代价 1、未命中代价 5 是合成的数, 不是测出来的时延, 也不是攻击成功率。"})
topic.pages["infer-prefix-radix"].source.push("llm_infer/m34_cache_identity/demo.py:experiment")

canonicalizeRelated(topic)
