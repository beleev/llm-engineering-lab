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
    { route: 'infer-quant-awq', label: 'INT4 · AWQ · KIVI', hint: '把权重和 KV 压到 4 bit 还不失真, 要看怎么分组' },
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
      subtitle: '自回归解码每吐出一个新词都要参考全部前文。KV cache 省去了历史位置的重复计算, PagedAttention 则用按页动态分配消除了预留连续显存带来的大块浪费。',
      tldr: '- KV cache: 因果掩码下历史 token 在各层的 K 与 V 向量恒定不变, 算完一次缓存起来后每步解码只需送入最新 1 个 token (demo 生成同段文本累计送入的 token 数从 688 降至 37, 缩减 18.6 倍)。\n- PagedAttention (分页管理): 请求生成长度事先未知, 按最大长度整块预留会浪费大半显存; 将 KV 切成定长物理块并经页表按需映射, 能把显存碎片压到每条序列最后一个未满块之内, 直接抬高单卡并发上限。',
      question: '开启 KV cache 之后, 每多生成一个新 token 的开销变成与上下文长度无关的常数 O(1) 了吗?',
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
          title: '因果掩码让历史位置的 K 与 V 算一次即可长期复用',
          body: '自回归生成分为两个阶段: 首次处理整段输入称为 prefill, 随后逐词追加称为 decode。注意力层在每个位置计算查询向量 $q$、键向量 $k$ 与值向量 $v$: 由于因果掩码严格阻断了历史位置读取未来词元, 不管后续再续写多少步, 前文每个位置在各层算出的 $k$ 和 $v$ 都完全不变。\n因此 prefill 阶段顺手把各层 $k, v$ 存入缓存, 后续每步 decode 只需计算最新 1 个 token 的 $q, k, v$, 将新 $k, v$ 追加到缓存尾部即可; 而历史位置的 $q$ 只在它自己提问的那一步使用过一次, 无需保留。',
        },
        {
          title: 'KV cache 省掉了历史投影与 MLP, 但没省掉 O(t) 的缓存读取',
          key: true,
          body: '加上 KV cache 后单步 decode 依然不是常数开销, 而是随当前上下文长度 $t$ 线性增长的访存密集型操作。\n对比省掉与没省掉的两笔账: 没有缓存时每步都要重跑全部 $t$ 个历史位置的 QKV 线性投影与两层 MLP; 有了缓存后这些大矩阵乘只需对最新 1 个 token 计算。然而在注意力内部, 最新位置的单个 $q$ 仍必须与缓存里全部 $t$ 个历史 $k$ 逐一点积, 再按权重加权全部 $t$ 个 $v$——每吐出 1 个词就要把整份历史 KV 从显存完整搬进计算核一遍, 这使长文本解码牢牢卡在显存带宽而非算力峰值上, 也引出了后续章节压缩 KV 体积、量化位宽与稀疏读取的全部优化。',
        },
        {
          title: '分页 KV 买的不是单步计算提速, 而是高显存利用率与大并发',
          body: '每条请求最终会输出多少个词在结束前无法预知: 若按最大长度 `max_len` 预先划拨连续显存, 大半空间常年闲置 (vLLM 统计真实有效占比仅 20%~40%); 若每步动态拼接扩容又会引发频繁整块拷贝与外部碎片。\nPagedAttention 借鉴操作系统虚拟内存, 将全局 KV 空间切成固定容纳 `bs` 个词元的物理块 (`block`), 每条请求维护一张轻量页表 `block_table`, 按 `pool[table[pos // bs], pos % bs]` 二级寻址。虽然多了一次页表间接寻址, 但外部碎片彻底归零, 内部浪费严格封顶在每条序列末尾块的 `bs - 1` 个空槽内——省出的显存可同时容纳数倍并发请求, 从而大幅拉升整体吞吐。',
        },
      ],
      links: [
        { from: 'sample.py 逐 token 生成', to: 'KV cache', body: '阶段 1 的 `sample.py` 每步重跑整段前文导致总复杂度达 $O(T^3)$; 引入 KV cache 后历史前文在每层不再重算投影与 MLP, 生成结果逐 token 完全一致。' },
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '块管理器实时清点全局空闲物理块数量; 调度器接纳新请求或推进解码前先向其询价, 空闲块不足时暂缓准入或触发抢占回收。' },
        { from: 'free_list 归还顺序', to: 'PrefixCache 的 LRU', body: '请求完成释放物理块时仅将引用计数 `ref_count` 减至 0 并挂回空闲队尾, 块内 KV 数据保留到被下次覆写前依然可被后续同前缀请求直接命中复用。' },
      ],
      sourceRows: [
        { concept: '对照组', code: 'm01_kv_cache/demo.py:generate_with_cache', takeaway: '同一输入分别运行每步全量重算与增量缓存两条路径: 输出 token 序列逐位相等, 末位 logits 最大浮点差仅 `2.80e-06`, 累计前向词元数从 688 压至 37。' },
        { concept: 'KV 字节公式', code: '2 · n_layer · T · D · sizeof(dtype)', takeaway: '核算 K 与 V 双份、层数、序列长、隐藏维与数据类型字节乘积: 代入 LLaMA-7B fp16 (`32 层, D=4096`) 可知每 token 占 `0.50 MiB`, 单条 4K 请求即占 `2.00 GiB`。' },
        { concept: '地址翻译', code: 'm02_paged_attention/paged_attention.py:paged_attention', takeaway: '将逻辑位置 `pos` 映射为物理块号 `block_table[pos // bs]` 与块内偏移 `pos % bs`, 与连续存储注意力的最大误差为 `0.00e+00`。' },
        { concept: '碎片上界', code: 'BlockManager.stats', takeaway: '仅每条活跃序列的最后一个物理块可能未填满: demo 中 11 个词元占 3 块 (12 槽), 仅余 1 个空闲槽位。' },
        { concept: '要不到块', code: 'MemoryError: 需要 25 个新 block, 只剩 3 个空闲', takeaway: '当申请块数超出空闲池余量时显式抛出 `MemoryError`, 为上层调度器提供执行抢占或排队等待的精确信号。' },
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
      subtitle: '线上请求长短不一且随时到达: 连续批处理 (continuous batching) 将组批粒度缩至单步前向, 分块 prefill 与 P/D 分离则消除长输入对正在生成用户的插队卡顿。',
      tldr: '- 连续批处理: 每走完一步前向就即时让已结束的请求退场、让等待队列的新请求补位, 消除静态凑批的短请求陪跑与门外干等。\n- 消除长 prefill 干扰: 一条上千词的长 prompt 若独占一步算完, 会让同卡所有正在流式吐字的用户停顿数百毫秒; 分块 prefill (chunked prefill) 用单步 token 预算将长输入切块与 decode 混批执行 (模拟最大卡顿从 `297 ms` 压至 `52 ms`), P/D 分离则直接将 prefill 与 decode 拆到两组独立机器上。',
      question: '当排在等待队首的新请求因空闲 KV 块不足暂时进不来时, 这一步调度器该怎么排? 如果直接返回空 batch 等下一步会发生什么?',
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
          title: '统一调度抽象: 每步只需决定「哪条序列算几个 token」',
          key: true,
          body: '现代推理调度器不再区分固定的「prefill 轮」和「decode 轮」, 每一步返回的调度单都是统一的二元组列表 `[(seq, n), ...]`: `n == 1` 代表该序列正在做单步 decode, `n > 1` 代表该序列正在做整段或分块 prefill, 两者可以在同一次模型前向中混合执行。\n每条请求通过 `num_computed` 记录已写入缓存的前缀长度: 每步算完加上 `n`, 只有当 `num_computed` 追平序列当前总长度 `num_tokens` 时才真正到达句尾并触发下一个新词采样 (m03 中 `s1` 在第 4 步结束退场, 新请求 `s4` 在第 5 步无缝补位)。',
        },
        {
          title: '有在途请求时绝不返回空 batch, 防止资源死锁与活锁',
          body: '调度器同时维护等待队列 `waiting` 与运行队列 `running`。设想 `waiting` 队首的大请求需要 6 个物理块而池中只剩 2 个, 此时 `running` 里还有 3 条请求在正常解码: 如果调度器因队首进不来而直接返回空列表, 在途请求就不会前进、永远无法结束归还物理块, 引擎便陷入原地空转的活锁!\n因此在非分块的 prefill 优先分支中必须回退执行 `_admit(budget) or _schedule_running(budget)` (接不进新请求就继续推进在途解码); 若推进在途解码时也缺新块, 则按后进先出踢出最晚入场的一条请求释放其物理块 (重算式抢占), 确保最老请求必定跑完。m03 在仅 7 个块的紧缩池下触发 2 次抢占、17 步全部跑通且零空转。',
        },
        {
          title: '单步 token 预算是在流式吐字间隔 (TBT) 与首字延迟 (TTFT) 间权衡的旋钮',
          body: '用户体感由两个延迟指标刻画: 首字延迟 TTFT (提交请求到看见第 1 个字) 与吐字间隔 TBT / ITL (流式生成时相邻两个字之间的停顿)。在单步耗时近似满足 `20 ms + 0.25 ms × 本步总 token 数` 的代价模型下, 4 条请求正常解码每步仅需 `21 ms`, 此时突然插入一条 1024-token 的长请求:\n- 整段插队 prefill: 该步一口气涌入 1024 个 token, 在途解码用户被迫跟着停顿 `297 ms`, 长请求自身 TTFT 为 `276 ms`。\n- 分块 prefill (`budget = 128`): 每步先分给 4 条解码各 1 个配额, 剩余 124 个配额切一块长 prompt 混跑; 单步总负载封顶在 128 token, 解码用户最大卡顿压回 `52 ms`, 代价是长请求需分多步算完、自身 TTFT 升至 `445 ms` (`budget = 512` 则折中在卡顿 `148 ms`、TTFT `319 ms`)。',
        },
      ],
      links: [
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '准入检查扣除掉前缀缓存已命中的共享块后核算净增块需求, 池中余量不足时把新请求留在队首等待而不破坏已运行请求。' },
        { from: 'max_batch_tokens', to: '每步耗时 → TBT', body: '单步最大 token 预算 (对应 vLLM 的 `max_num_batched_tokens`) 直接封顶单次前向时长: 调小优先保流式顺滑度 (低 TBT), 调大优先保长文本首字响应 (低 TTFT)。' },
        { from: 'KVLink.send', to: 'P/D 分离', body: 'P/D 分离把算力密集的 prefill 与带宽密集的 decode 物理隔离到不同节点以根除同卡抢占, 代价是跨网传输 KV (计算耗时时务必区分网卡 `Gbps` 比特率与 KV 字节数的 8 倍换算关系)。' },
      ],
      sourceRows: [
        { concept: '每步重组 batch', code: 'm03_continuous_batching/scheduler.py:Scheduler.schedule', takeaway: '支持关闭分块时的 prefill 优先回退与开启分块时的 decode 优先混批; 刚执行过抢占的当步暂停接纳新请求以防刚腾出的块被新请求抢走。' },
        { concept: '抢占', code: 'Scheduler._preempt', takeaway: '采用重算 (recompute) 策略回收最年轻请求的全部物理块并放回 `waiting` 队首, 已生成的 `output_ids` 保留待下次连同 prompt 一并重算 KV。' },
        { concept: 'TBT 代价模型', code: 'm06_chunked_prefill/chunked_prefill.py:simulate', takeaway: '驱动真实调度器重放请求轨迹并按线性耗时公式统计每条请求的 TTFT 与最大 TBT。' },
        { concept: '分块不改结果', code: 'chunked_prefill(lm, ids, chunk)', takeaway: '因果掩码保证分块依次前向与整段一次前向数学等价 (`chunk >= 16` 逐位相等, `chunk = 8` 仅差 `1.67e-06` 浮点求和序误差), 同时将瞬时注意力矩阵峰值缩小 `T / chunk` 倍。' },
        { concept: '同卡混跑抖动实测', code: 'm15_pd_disaggregation/pd.py:KVLink', takeaway: 'TinyLM 真机墙钟测量显示, 被 400-token 长 prefill 插队那一步解码延迟从 `0.253 ms` 跳升至 `4.68 ms` (18 倍抖动), 拆至独立节点后恢复平稳。' },
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
      subtitle: '自回归解码受显存带宽限制, 每读一遍大模型权重只出 1 个词; 投机解码通过「小模型连猜、大模型并行校验」在不改变输出分布的前提下减少大模型调用次数, 后处理采样链与束搜索则塑造最终输出行为。',
      tldr: '- 投机解码 (Speculative Decoding): 用轻量草稿模型 (`draft`) 连续试猜 $K$ 个词, 再交由目标大模型 (`target`) 单次前向并行校验全部 $K+1$ 个位置; 贪心模式下逐位核对 argmax, 随机采样模式下通过拒绝采样 (`min(1, p/q)` 接受加残差分布重采) 严格保持输出分布与纯大模型一模一样。\n- 采样与搜索控制: 重复惩罚、温度、top-k、top-p 与 min-p 是一条顺序敏感的 logits 过滤链; 束搜索 (beam search) 寻找高概率序列但天然偏向短句且易陷入高频词复读。',
      question: '就算草稿小模型猜得很离谱, 拒绝采样凭什么能从数学上保证最终吐出的每个词与单独运行大模型采样服从完全相同的概率分布?',
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
          title: '利用空闲算力: 单次大模型前向并行检验 K+1 个位置',
          body: '单序列解码时 GPU 时间绝大部分花在从显存搬运整套大模型权重上, 喂 1 个词和一次喂 5 个词耗时相差无几。投机解码先让轻量 `draft` 连猜 $K$ 个候选词 $d_0, \\dots, d_{K-1}$, 再拼上已确认的末尾词一次送入 `target` 算出 $K+1$ 行 logits (因果掩码保证这与串行跑 $K+1$ 步完全等价)。\n在贪心解码下从左往右逐位核对: 接受最长匹配前缀, 在首个猜错处直接换上 `target` 自己的 `argmax` 纠错词; 若 $K$ 个全猜对, 第 $K$ 行还白送 1 个新词, 因此单次 `target` 调用保底产出 1 个词、至多产出 $K+1$ 个词。m07 生成 48 个词 (`K=4`): 完美草稿仅需 11 次调用 (`4.36×`), 加 10% 噪声的蒸馏草稿需 26 次 (`1.85×`), 完全无关的随机小模型需 43 次 (`1.12×`), 且所有输出与单跑 `target` 逐位一致。',
        },
        {
          title: '拒绝采样与残差重采保证随机解码严格无损',
          key: true,
          body: '开启温度随机采样时不能再简单比对 `argmax`, 而需使用修正拒绝采样。设当前位置 `target` 概率为 $p$、`draft` 概率为 $q$, 对 `draft` 抽出的候选词:\n- 按概率 $\\min(1, p/q)$ 接受: 若大模型比小模型更认可该词 ($p \\ge q$) 则 100% 收下, 否则按比例概率打折接受, 这一步贡献的采样概率恰好是 $q \\cdot \\min(1, p/q) = \\min(p, q)$。\n- 一旦拒绝, 立即作废后续草稿词, 从归一化的正残差分布 $\\max(0, p - q)$ 中重抽一个纠错词补齐缺口。\n直接收下与拒绝重采两路概率相加恒等于 $\\min(p, q) + (p - \\min(p, q)) = p$! 草稿模型好坏只决定接受率高低 (跑得快慢), 绝不污染输出分布。m07 在 16 词表上采样 2,500 条做卡方检验: 修正拒绝采样的 $\\chi^2 \\le 13.6$ (远低于显著性阈值 `37.7`), 而盲目全收草稿则暴涨至 `585`。',
        },
        {
          title: '采样算子链顺序敏感, 重复惩罚需区分正负 logit 符号',
          body: 'm10 将词表打分后处理组织为可串联的原地掩码链 `rep_penalty -> temperature -> top_k -> top_p -> min_p -> Gumbel-max`, 被淘汰的词置为 `-inf`:\n- 正负异向的重复惩罚: 对已出现词降权时, 正 logit 需除以 `penalty > 1`, 而负 logit 必须乘以 `penalty` 往更负推; 若不分正负一律做除法, `-1.0 / 2.0 = -0.5` 反而把负分往上抬高, 变相鼓励复读 (demo 中概率从 `0.0076` 反升至 `0.0144`)。\n- 温度与截断的先后顺序: 温度排在 `top-p` 之前意味着高温拉平分布后会让更多长尾词挤进累计概率 `p` 的门槛内; 不同推理框架若调换两者顺序, 相同超参也会采出不同风格。',
        },
        {
          title: '束搜索 (Beam Search) 追逐最高似然, 却不等于最佳生成文本',
          body: '束搜索每步保留累计对数概率最高的 $w$ 条活跃前缀。在 TinyLM 上将束宽从贪心 (`w=1`) 扩至 `w=32`, 单词平均对数概率确实从 `-2.620` 升至 `-2.255`, 但双词重复率 `rep-2` 也从 `0.453` 升至 `0.563`, 且同组 8 条候选彼此高度同质 (20 个位置平均仅差 3.5 处, 随机采样则差 19.7 处)。',
        },
        {
          title: '局部剪枝的非精确性与偏向短句的长度惩罚校准',
          body: '使用束搜索时还需避开两个常见误区:\n- 并非全局最优搜索: 早期某一步暂时排在第 $w+1$ 名的前缀一旦被剪枝便无法复活, 因此偶尔会出现 `w=2` 最终总分反输给 `w=1` 贪心的情况 (10 条测试中发生 2 次)。\n- 短句偏置与长度归一化: 每多走一步都在累加负的 $\\log p$, 不做修正 (`α=0`) 时一旦模型学会输出 EOS 便倾向于只吐 1~2 个词草草收尾 (demo 均值仅 `1.3` 词); 按 $\\text{score} / \\text{len}^\\alpha$ (`α=1.0` 即平均每词对数概率) 重排后句长才恢复至正常的 `15.7` 词。工程上机器翻译等强约束任务适合束搜索, 开放对话与创作则普遍采用核采样 (`top-p` / `min-p`)。',
        },
      ],
      links: [
        { from: 'sample.py 的 temperature/top-k', to: 'm10_sampling', body: '将阶段 1 的基础采样扩展为服务端标准 `SamplingParams` 过滤链, 并用无分支并行的 Gumbel-max 技巧替代串行前缀和二分查找。' },
        { from: 'accept_greedy / accept_sampling', to: 'EAGLE · 树形投机', body: '本章确立的无损验证与拒绝重采规则在后续章节原样复用, EAGLE 与树形投机只升级草稿预测器准确度与单次验证拓扑。' },
        { from: 'truncate_kv', to: 'KV 回滚', body: '并行验证时写进缓存但未获接受的尾部草稿 KV, 直接修改长度游标截断丢弃即可, 因果掩码保证已接受前缀无需重算。' },
      ],
      sourceRows: [
        { concept: 'greedy 接受', code: 'm07_speculative_decoding/speculative.py:accept_greedy', takeaway: '从左至右比对草稿词与大模型 `argmax`, 在首个失配处直接采用大模型预测词作为纠错输出。' },
        { concept: '采样接受', code: 'm07_speculative_decoding/speculative.py:accept_sampling', takeaway: '当均匀随机数 $u < p_t / p_d$ 时接受, 否则从正残差 $\\max(0, p_t - p_d)$ 归一化分布中重采并终止本轮草稿。' },
        { concept: '期望产出', code: '(1 − α^(K+1)) / (1 − α)', takeaway: '单词接受率为 $\\alpha$ 时链式投机的单轮期望产出上界为 $1/(1-\\alpha)$, 深度过大收益递减, 为树形横向扩展埋下伏笔。' },
        { concept: '采样顺序', code: 'm10_sampling/samplers.py:sample', takeaway: '依次串联重复惩罚、温度缩放、top-k、top-p、min-p 与 Gumbel-max 抽样。' },
        { concept: 'Gumbel-max 等价抽样', code: 'argmax(log p + G), G = −log(−log U)', takeaway: '向对数概率注入标准 Gumbel 噪声后取最大值, 与多项式抽样严格同分布 (`TV = 0.0064`), 且纯逐元素加归约极其契合 GPU 批处理。' },
        { concept: 'beam search', code: 'm24_beam_search/beam.py:beam_search', takeaway: '每步扩展并保留得分前 $w$ 条活跃序列, 结束时按长度惩罚系数 $\\alpha$ 归一化选优 (`w=1` 与贪心逐位一致)。' },
        { concept: 'EOS 偏置对照', code: 'm24_beam_search/demo.py:EosBiased', takeaway: '抬高随机模型的结束符倾向以复现无长度惩罚时束搜索提前收尾输出超短句的现象。' },
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
      subtitle: '不改动一行数学公式的前提下压榨硬件效率: FlashAttention 消除显存中的 $T \\times T$ 注意力矩阵读写, Flash-Decoding 在单步解码时沿 KV 维度切分跑满计算单元, CUDA Graph 消除 CPU 逐个提交算子的气泡, 张量并行则将单层多卡同步压至两次。',
      tldr: '四项底层系统优化的输出均与朴素实现严格等价, 节省的全是数据搬运与调度空等:\n- FlashAttention: 将 Q 与 K/V 分块装入高速片上缓存 SRAM, 借 online softmax 流式重缩放分母, 避免在慢速显存 HBM 中反复读写巨大的 $T \\times T$ 中间矩阵。\n- Flash-Decoding: 解码阶段只有单个 query 导致并行度不足, 改为沿历史 KV 长度切分 $S$ 段并行计算再按对数分母 (`LSE`) 精确合并。\n- CUDA Graph: 将解码单步数百个碎小算子录制为一张静态执行图, CPU 每步只需下发 1 次重放指令。\n- 推理张量并行 (TP): 按注意力头边界做「列切接行切」配对, 单个 Transformer 层仅需 2 次 all-reduce。',
      question: 'FlashAttention 的浮点乘加次数 (FLOPs) 一点都没比朴素注意力少, 为什么实际运行却快得多且显存大减?',
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
          title: 'FlashAttention: 片上流式更新分母, 省掉 T×T 显存读写',
          key: true,
          body: 'GPU 片上缓存 SRAM 带宽极高但容量极小, 主显存 HBM 容量大但带宽低一个数量级。朴素注意力先算完整的 $S = QK^\\top$ 写回 HBM, 再读出求 softmax 写回 $P$, 最后再读出乘 $V$——当 $T=4096$ 时单头矩阵就有 1,600 万个元素, 耗时全卡在反复读写 HBM 上。\nFlashAttention 将 Q、K、V 切成小块进 SRAM, 为每行 query 维护当前最大值 $m$、指数和分母 $l$ 与未归一输出 $O$ 三个标量/行向量: 每扫过一块新 K/V, 若最大值刷出新高 $m_{\\text{new}}$, 就把旧的 $l$ 和 $O$ 乘上 $\\exp(m_{\\text{old}} - m_{\\text{new}})$ 平滑换底后再累加当前块贡献。全程不往 HBM 物化 $T \\times T$ 矩阵, 临时空间恒为小块尺寸 (`64×64`), 30 组配置对拍与朴素注意力最大误差仅 `1.78e-06`。',
        },
        {
          title: 'LSE (对数分母) 是跨块、跨步与跨卡精确拼接注意力的通用凭证',
          body: '记录每段局部注意力的归一化输出 $O_i$ 与对数分母 $\\mathrm{lse}_i = m_i + \\log l_i$ 后, 任意多段独立的 K/V 计算结果都能无损合成为全局注意力: 合并后的总 $\\mathrm{lse} = \\mathrm{logaddexp}(\\mathrm{lse}_1, \\mathrm{lse}_2)$, 总输出即为按各段概率质量占比 $\\exp(\\mathrm{lse}_i - \\mathrm{lse})$ 加权之和。\n实测按 LSE 加权合并与全量一次计算误差仅 `2.68e-07` (若误作简单算术平均则错达 `0.212`)。分块 prefill、跨卡 Ring Attention 与下文的 Flash-Decoding 全都建立在这一合并算子之上。',
        },
        {
          title: 'CUDA Graph 消除小批量解码时 CPU 下发算子的启动气泡',
          body: '解码单步只处理几个 token, 单个 GPU kernel 往往几十微秒就跑完了, 而 CPU 侧经 Python 与驱动下发一次 kernel 就要花约 `10 µs`——一步几百个算子下来, GPU 大半时间都在空闲等待 CPU 派活。\nCUDA Graph 先试跑一遍把整条算子序列及绑定的固定显存地址录制成图, 后续每步只需 CPU 发 1 次 `replay` 指令 (代价是输入输出张量形状与缓冲区地址必须固定, 新数据需原地拷入 `static_input[...] = x`, 故常用于定形 decode 而大动态 prefill 仍走常规即时下发)。',
        },
        {
          title: '单步解码 Q 只有 1 行时, 标准 FlashAttention 派不满 GPU 计算单元',
          body: '标准 FlashAttention 按 `(batch, head, Q_block)` 派发并行线程块: prefill 阶段几千行 Q 能切出海量块喂饱整张卡; 但到了单步 decode 阶段每个请求只有 1 行 query (`Q_block = 1`), 例如 `B=1, H=32` 的长上下文对话总共只能派出 32 个线程块, 在拥有 108 个流多处理器 (SM) 的 A100 上七成计算单元处于闲置状态, 工作的 32 个块还得各自从头到尾硬啃几万词的 KV。',
        },
        {
          title: '推理张量并行 (TP): 沿注意力头边界列行配对, 每层仅两次同步',
          body: '单卡装不下超大模型权重时, 推理 TP 与训练期 Megatron TP 一样采用「先列切、后行切」配对: 注意力 Q/K/V 按列切分让每张卡独立算完整的若干个头, O 投影按行切分后做第 1 次 all-reduce; MLP 的 gate/up 按列切、down 按行切后做第 2 次 all-reduce, 整层合计仅同步 2 次且单次通信量恒为 $T \\times D$。\n需特别注意列切必须严格落在注意力头边界上: 若把单个头拆到多张卡上各自求局部 softmax, 归一化分母失真会导致输出偏差高达 `5.63e-01`。',
        },
        {
          title: 'Flash-Decoding: 沿 KV 历史长度切分 split-K 跑满空闲 SM',
          body: '既然解码时单行 Q 切不出并行度, Flash-Decoding (split-K) 改为把长历史 KV 均匀切成 $S$ 段, 派出 $B \\times H \\times S$ 个线程块并行算出各段的 $(O_s, \\mathrm{lse}_s)$, 最后补一个小归约核按 LSE 精确合并 (13 组配置含非整除边界对拍最坏误差仅 `2.06e-07`)。\n在 A100 108-SM 解析代价模型下: `B=1, T=32k` 取 $S=64$ 可将注意力耗时从 `909.0 µs` 压至 `275.5 µs` (提速 `3.30×`, 逼近 $108/32 = 3.38\\times$ 的空闲单元填满上限); 当批大小升至 `B=64` 本就喂满 SM 时则自动无需再切。',
        },
      ],
      links: [
        { from: '朴素 attention', to: 'flash_attention', body: '数学输出一致 (`max|Δ| = 1.78e-06`), 通过片上分块与因果上三角整块跳过 (T=4096 时跳过 49.2% 无效块) 同时省下 HBM 带宽与一半无效乘加。' },
        { from: 'merge_attention(O, lse)', to: 'chunked prefill / ring attention', body: '只要各子段保留自身的对数分母 `lse`, 时间分块或跨卡分片的注意力都能无损还原为全局精确值。' },
        { from: 'tp_block', to: '多卡推理', body: '单层内两次 all-reduce 将多卡延迟敏感通信压缩到极致, 适合单机内高速 NVLink 互联部署超大模型。' },
      ],
      sourceRows: [
        { concept: 'online softmax', code: 'm11_flash_attention/flash_attention.py:flash_attention', takeaway: '采用外层遍历 Q 块、内层遍历 K/V 块的 FlashAttention-2 循环序, 让累加器常驻片上直到整行算完只写回一次。' },
        { concept: 'causal 整块跳过', code: 'flash_attention(..., causal=True)', takeaway: '若当前 K 块起始位置已晚于 Q 块最末位置则整块直接跳过, 省去近一半上三角无效矩阵乘。' },
        { concept: '分段合并', code: 'm11_flash_attention/flash_attention.py:merge_attention', takeaway: '利用 `logaddexp` 合成总对数分母并重加权合并各段局部输出。' },
        { concept: '图只认地址', code: 'm12_cuda_graph/graph.py:CudaGraph', takeaway: '将 16 次独立算子下发合并为单次图重放; 演示了重新绑定变量名而不原地拷贝会导致图反复读取旧缓冲区的经典坑。' },
        { concept: 'TP 切在 head 边界', code: 'm09_tensor_parallel/parallel_linear.py:tp_block', takeaway: '校验按整数个注意力头切分与单卡基线对齐, 并展示切碎单个头引发的归一化错误。' },
        { concept: '一段 KV 的局部结果', code: 'm26_flash_decoding/flash_decoding.py:partial_attention', takeaway: '独立计算单个 KV 切片上的局部归一化输出 $O_s$ 与对数分母 $\\mathrm{lse}_s$。' },
        { concept: 'split-K + reduce', code: 'm26_flash_decoding/flash_decoding.py:flash_decoding', takeaway: '支持任意非整除长度切分 $S$ 段并行计算并在尾核通过 LSE 无损归约。' },
        { concept: '延迟估算', code: 'm26_flash_decoding/flash_decoding.py:decode_latency_us', takeaway: '建模 SM 波次利用率 (`waves = ceil(B*H*S / 108)`) 与尾核合并开销, 量化小批量长上下文下的提速空间。' },
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
      subtitle: '将前面各章独立的内存池、连续批调度、前缀匹配与采样链组装成完整的推理服务主循环, 并用严格的 token 对账断言验证前缀命中确实跳过了重复前向计算。',
      tldr: '`Engine.step` 每轮循环依次执行「调度组批 → 增量前向 → 到位采样 → 资源回收与前缀登记」四步:\n- 全局共享分页池: 所有请求的 KV 统一存放在按页寻址的全局 `pool` 中, 新请求命中的公共前缀直接挂载既有物理块并从 `start_pos` 之后开始算前向。\n- 端到端实测对账: 首轮 5 条共享系统提示词的请求共需 305 个 KV 位置, 实际仅前向计算 233 个、前缀直接复用 72 个; 将池容量压到 9 个块逼出 4 次抢占重算, 输出依然与逐条贪心基线逐词一致。',
      question: '如果一个推理引擎只在统计数字上记了「前缀命中」, 前向计算时却照样把整条 prompt 重算了一遍, 输出文本完全正确——该用什么断言一眼抓出这种假复用?',
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
          title: '全局 KV 物理块池让跨请求前缀共享真正跳过前向计算',
          body: '若每条请求各自持有私有 KV 数组, 就算查出两段输入前缀相同也无法直接借用对方的内存。`ModelRunner` 在引擎启动时按层开辟全局统一池 `pool[layer]` (形状 `(num_blocks, block_size, D)`), 所有请求的页表都指向同一个池中的物理块。\n每次执行 `run(ids, block_table, start_pos)` 时, 模型只为尚未计算的切片 `ids[start_pos:]` 计算 Q/K/V 并写入池中, 随后经页表把 `0 .. len(ids)` 的完整历史与当前块一并取出做注意力——无论 `start_pos` 之前的 KV 来自跨请求前缀命中、上一轮分块 prefill 还是历史 decode 步, 全都走同一套零重算读取路径。',
        },
        {
          title: '主循环四步里, 复杂状态转移全收敛在调度器一步',
          key: true,
          body: '`Engine.step` 的主干非常紧凑: 前向执行、末尾采样与后处理回收都是按单行事的确定操作, 真正的控制中心在首行 `scheduler.schedule()`——它必须在每一步同时平衡物理块余量 (不足时按 LIFO 抢占最晚入场的请求)、防活锁推进 (队首大请求进不来时照常排在途 decode) 以及单步最大 token 预算 (如 demo `[1]` 预算为 48 时第 2 步排出 `s0:D + s1:P39 + s2:P8 = 48` 恰好跑满预算)。',
        },
        {
          title: '用「实算 token + 命中 token == 应有 KV 总数」断言真省算力',
          body: '检验前缀缓存是不是「真跳过了计算而非只打印漂亮日志」, 光对输出文本是查不出来的——重算一遍同样会输出一模一样的正确文本! 因此引擎显式维护两本计数器: `tokens_computed` 统计每次实际送进模型前向的行数, `prefix_hit_tokens` 统计准入时直接挂载复用的前缀长度。\n无抢占场景下两者之和必须严格等于所有完成请求需要建 KV 的词元总数 (demo `[1]` 中 `233 实算 + 72 命中 == 305`); 此外随机抽取 30 组块大小、池容量与分块开关的 fuzz 压力测试共跑完 214 条请求、经历 34 次抢占重算, 全部输出与朴素贪心逐位一致且结束时引用计数归零无泄漏。',
        },
      ],
      links: [
        { from: 'Engine.add_request', to: 'Scheduler.waiting', body: '将输入文本编码为整数序列排入等待队尾, 同时为每条请求独立绑定各自的 `SamplingParams`, 支持同批混跑不同温度与采样策略。' },
        { from: 'Scheduler._admit', to: 'PrefixCache.match_prefix', body: '新请求入场前先查公共前缀命中长度 `n_hit`, 将 `seq.num_computed` 直接拨到 `n_hit`, 命中的前缀块不消耗当步前向 token 预算。' },
        { from: 'ModelRunner.run', to: 'm02.paged_attention', body: '仅构造新增 `len(ids) - start_pos` 行的 Q 向量, 配合页表读取全长 K/V 完成增量注意力。' },
        { from: 'Multi-LoRA (m13)', to: '同一 batch 多个 adapter', body: '多租户挂载不同 LoRA 时无需合并出上千份大模型: 共享基座矩阵乘全批合算一次, 再按请求各自的 `adapter_id` 路由叠加轻量 `B(Ax)` 旁路 (1,000 个 adapter 显存从合并的 `24.58 MB` 压至 `2.58 MB`)。' },
      ],
      sourceRows: [
        { concept: '一步 (主循环)', code: 'full_engine/engine.py:Engine.step', takeaway: '四行代码串联组批调度、增量模型前向、追平序列后的条件采样与状态回收登记。' },
        { concept: '只算新 token', code: 'full_engine/model_runner.py:ModelRunner.run', takeaway: '仅对 `ids[start_pos:]` 执行嵌入与各层投影, 写回物理块池后返回末尾位置的词表打分。' },
        { concept: '中途不采样', code: 'done_prefill = num_computed + n == num_tokens', takeaway: '长 prompt 处于中间分块 prefill 步时还差后续已知输入未读完, 跳过采样返回 `None`, 直至最后一块追平全长才采出首个生成词。' },
        { concept: '抢占', code: 'm03_continuous_batching/scheduler.py:Scheduler._preempt', takeaway: '物理块吃紧时回收最年轻在途请求的块并退回队首等待重算, 保证最早入场的请求必定推进完成。' },
        { concept: '统计对账', code: 'Engine.report_stats', takeaway: '汇总步数、实际前向词元数、前缀命中词元数、抢占次数与块池余量供自动化断言核验。' },
        { concept: 'fuzz 随机测试', code: 'demo [5]: 30 组随机配置', takeaway: '随机组合池大小、块尺寸、预算与缓存开关运行两轮共 214 条请求, 验证零活锁、零内存块泄漏且结果逐词一致。' },
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
      subtitle: '共享系统提示词、少样本示例与多轮历史让不同请求开头存在大量相同词元。链式块哈希以整块为单位快速查表复用, 基数树 (Radix Tree) 则把匹配粒度推进到任意单个词元并按叶子 LRU 安全驱逐。',
      tldr: '因果注意力下前缀词元的 KV 完全由该前缀自身决定, 跨请求查找最长公共前缀主要有两种索引结构:\n- 链式块哈希 (vLLM 风格, m04): 将父块哈希与当前块词元一并做哈希, 只能按填满的整块命中 (如公共前缀 100 词、块长 16 时复用前 6 块共 96 词)。\n- 基数树 RadixCache (SGLang 风格, m05): 将历史词元序列与对应 KV 槽位压缩存储在前缀树边上, 分叉落在边中段时动态切边 (`split`), 100 个相同词元可悉数命中; 容量满时仅从无在途引用的叶子节点按 LRU 向上驱逐。',
      question: '基数树缓存空间不足需要淘汰旧条目时, 为什么只能从叶子节点删起, 而不能直接删掉最久没访问过的中间节点?',
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
          title: '链式块哈希必须纳入父块哈希, 唯一标识从头算起的完整前缀',
          key: true,
          body: '第 $i$ 个块的哈希值必须递推计算 $h_i = H(h_{i-1} \\,\\|\\, \\text{block}_i)$: 因为自注意力中同一句话接在不同前文后面算出的 K/V 完全不同, 若只对当前块内的几个词单独算哈希, 前文不同的两条请求就会张冠李戴拿错 KV; 把父哈希串进来后, 前面哪怕只差 1 个词, 后续所有块的哈希值都会彻底改变。\n新请求从首块起逐块查表挂载物理块, 直至首次未命中为止; 由于未写满的尾块不入表, 长度为 $n$ 的公共前缀最多只能命中 $\\lfloor n / \\text{bs} \\rfloor \\cdot \\text{bs}$ 个词元。',
        },
        {
          title: 'Radix Tree 压缩边存储与动态切边, 把复用粒度细化到单 token',
          body: '基数树每条有向边直接挂载一串词元 `edge_tokens` 及等长的物理槽位列表 `slots`。新请求从树根沿边逐词比对到首个分叉点即为最长公共前缀; 若分叉恰好落在某条现有边的中间 (如已有 `[1..8]`, 新来 `[1,2,3,4,9,9]`), `_split` 操作会将原边在位置 4 处同步拆成 `[1..4]` 与 `[5..8]` 两段, 并让新增的中间父节点原样继承下游子节点的在途引用计数 `ref_count`。\n在块大小 16、共享前缀 100 词时, 块哈希只能复用 96 词, 基数树则能复用全部 100 词 (不过若整条 prompt 100% 命中, 实现仍会刻意留出末尾最后 1 个词送入前向以算出首步采样所需的 logits)。',
        },
        {
          title: '驱逐只能从零引用的叶子开始逐层向上剥离',
          body: '当缓存池满需要腾挪槽位时, 绝不能直接抠掉某个中间节点的 KV: 因为所有子孙节点的 KV 都是建立在祖先前缀完整存在的前提下使用的, 一旦中间某一段缺失, 其下整棵子树保存的后缀 KV 都会因前缀断裂而彻底报废。\n因此 `evict` 严格只挑选 `ref_count == 0` (当前无活跃请求锁定) 的叶子节点按 `last_used` 最久未使用顺序删除; 只有当某个内部节点的全部孩子都被删空、自身退化为新叶子后, 下一轮才允许回收它 (m05 锁定 5 条活跃路径后执行驱逐, 树容量从 1,759 精准收缩至这 5 条路径并集的 49 个词元)。',
        },
      ],
      links: [
        { from: 'BlockManager.ref_count', to: 'PrefixCache', body: '多个并发请求共享同一物理块依靠 `ref_count` 追踪在途使用者数量, 仅当计数归零时该块才允许被驱逐覆写。' },
        { from: 'RadixCache.match_prefix', to: 'Scheduler._admit', body: '返回的命中前缀槽位直接装入新序列页表并将已计算游标拨至 `n_hit`, 只需为剩余后缀调度 prefill 算力。' },
        { from: 'radix tree', to: '分层 KV offload', body: '内存驱逐并非只能一扔了之; 后续「分层 KV offload」将从 GPU 挤出的冷前缀逐级下沉到主机内存与本地磁盘中暂存。' },
      ],
      sourceRows: [
        { concept: '链式 hash', code: 'm04_prefix_cache/prefix_cache.py:_block_hash', takeaway: '以固定根种子 `"<root>"` 起步将 `parent_hash` 与当前块词元序列一并送入 SHA1, 确保哈希绑定完整历史前缀。' },
        { concept: '预留末词算 logits', code: 'match_prefix 只匹配 (len−1)//bs 块', takeaway: '缓存只保存 K/V 而不保存输出层 logits, 故至少保留最后 1 个输入词元做真实前向以触发首词生成。' },
        { concept: '最长前缀匹配', code: 'm05_radix_cache/radix_tree.py:RadixCache.match_prefix', takeaway: '沿树匹配并在 300 条随机序列对拍中验证每次命中长度严格等于暴力扫描全历史的真实最长公共前缀。' },
        { concept: '边分裂', code: 'RadixCache._split', takeaway: '在边内部分叉点同步切开词元数组与槽位数组, 新插入的中间节点严格继承原节点的 `ref_count` 以防解锁减成负数。' },
        { concept: '叶子 LRU 驱逐', code: 'RadixCache.evict', takeaway: '仅在 `ref_count == 0` 的叶子集合中挑选 `last_used` 最早的节点回收槽位。' },
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
      subtitle: '在提示词里反复叮嘱输出 JSON 仍无法保证百分百可解析。结构化输出在采样前通过有限状态机将破坏语法的词元 logit 直接置为 −inf; 由于子词词元常一次跨越多个字符状态, 必须将状态转移离线预编译成词元级掩码表。',
      tldr: '约束解码将格式合规率从概率问题变成硬保证:\n- 词元级跨越: 模型输出的是像 `\":` 或 `true` 这样的多字符子词, 必须在字符级有限状态机 (FSM) 上把整个词元的全部字符连续试走通才能判定合法。\n- 离线预编译加速: 若解码每步现场在 CPU 上对十几万词表逐词试走状态机, 会严重拖慢 GPU 前向流水线; 事先离线枚举 `(状态, token)` 编译为二维查找表后, 在线每步仅需 `O(1)` 取一行掩码 (m14 实测单步从 `18.4 µs` 缩至 `0.086 µs`, 提速 215 倍, 随机模型 100 次采样 JSON 合法率从 `0%` 升至 `100%`)。',
      question: '为什么「当前允许输入哪些字符」不等于「当前允许采样哪些 token」? 如果解码撞上 max_tokens 长度上限被截成半截 JSON 该怎么防?',
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
          title: '多字符子词会连跨多个语法状态, 合法性必须整词试走判定',
          key: true,
          body: '字符级 JSON 状态机很容易定义当前允许接哪些单字符, 但 BPE 词表中的词元往往包含多个字符——例如单个词元 `\":` 会一口气走完「闭合键名引号」与「输入键值冒号」两步转移, 而像 `a\"x` 这样首字符合法、第三字符违法的词元只有走完整个串才能暴露错误。\n因此判定词元 $t$ 在状态 $s$ 下是否合法, 必须将 $t$ 拆成字符流从 $s$ 出发依次查转移表: 全程走通则将终点记入 `next_state[s, t]`, 任一字符卡壳则记 `-1`, 结束符 EOS 仅在完整闭合的接受态放行。在随机权重的 TinyLM 上施加该掩码后, 100 次采样从原本 `0/100` 可解析变为 `100/100` 合法 JSON (注意语法掩码只保外壳结构合法, 语义合理性仍靠模型自身能力)。',
        },
        {
          title: '将 O(V·len) 的逐词遍历前置到离线编译, 在线解码只做 O(1) 查表',
          body: '如果在每步生成夹缝里让 CPU 现场对十余万词表做逐字符试走, 这笔串行开销会直接卡死 GPU 解码吞吐。鉴于确定性有限状态机的状态数有限 (demo 为 101 个状态、96 词表共 9,696 格), 完全可以在加载 Schema 时一次性遍历所有 `(状态, token)` 填好 `mask_table` 与 `next_state` 两张静态表: 在线每步只需按当前状态号取出一行布尔掩码把非法位刷成 `-inf`, 耗时从现场试走的 `18.4 µs/step` 骤降至 `0.086 µs/step` (快 215 倍)。',
        },
        {
          title: '最短收尾步数表 (need) 防止触达长度上限时留下半截残文',
          body: '单纯保证每步转移合法还有一个边界隐患: 若 `max_tokens` 预算只剩 2 步而模型还在往外开新字段, 撞线截断就会留下一串尚未闭合引号与花括号的合法前缀 (demo 把预算压到 12 步时仅 `33/100` 能完整闭合)。\n在离线建表阶段同时用图最短路算出第三张表 `need[s, t]` (代表在状态 $s$ 选了词元 $t$ 后至少还需多少个词元才能抵达 EOS), 在线每步额外屏蔽掉 `need[s, t] > 剩余步数 - 1` 的分支, 就能引导序列在预算耗尽前优雅闭合 (`100/100` 完整 JSON; 工业 API 则常选择不强掰结尾, 直接返回 `finish_reason="length"` 交调用方决定)。',
        },
      ],
      links: [
        { from: 'JsonFSM (字符级)', to: 'compile_char_dfa', body: '通过广度优先搜索将手写字符规则展开为显式整数状态转移字典 `trans[s][ch]`, 供离线词元试走高速查询。' },
        { from: 'token_row', to: 'compile_token_table', body: '对全部可达状态逐行运行词元试走, 一次生成 `next_state`、`mask_table` 与最短闭合距离表 `need`。' },
        { from: 'mask_table[s]', to: 'm10 sample', body: '语法掩码本质上只是插在采样过滤链最前端的一道 `-inf` 过滤器, 放行后的合法候选照常经过后续温度与 `top-p` 抽样。' },
      ],
      sourceRows: [
        { concept: '多字符词表', code: 'm14_structured_output/grammar.py:build_vocab', takeaway: '构造包含跨语法边界组合词元 (`\":`, `e\"`) 与永不合法噪声词元的测试词表, 检验多步连跨判定。' },
        { concept: '单状态试走', code: 'm14_structured_output/grammar.py:token_row', takeaway: '从状态 $s$ 出发逐字符推进词元切片, 任一字符失配即标为 `-1`, EOS 仅在接受态开放。' },
        { concept: '三表预编译', code: 'm14_structured_output/grammar.py:compile_token_table', takeaway: '离线产出落点状态表、合法布尔掩码表与到终态最短步数表, 供在线解码 `O(1)` 索引。' },
        { concept: '硬约束保证', code: 'logits[~mask] = -inf', takeaway: '将违规词元的未归一化分数置为负无穷使其采样概率严格归零, 消除长文本采样偶发破格。' },
        { concept: '上下文无关文法扩展', code: 'CFG / 下推自动机', takeaway: '真实任意深度嵌套 JSON 属于上下文无关文法需配合下推栈, XGrammar 等引擎通过分离「栈无关预编译掩码」与少量「栈相关运行时检查」兼顾嵌套表达力与极低延迟。' },
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
      subtitle: '解码吞吐受制于权重与 KV 缓存的访存字节数。压到 4-bit 只剩 16 个离散格点后, 细粒度分组负责隔离局部离群值, AWQ 与 GPTQ 从激活分布出发保护输出精度, KIVI 与 SmoothQuant 则分别针对 K/V 缓存和全整数矩阵乘对症处理离群通道。',
      tldr: '推理端主流低比特量化方法按优化目标清晰分工:\n- Group-wise INT4 与 AWQ (权重量化): 每 $g$ 个相邻权重共享一套刻度以圈住离群值; AWQ 注意到「乘上大激活的权重行对输出误差影响最大」, 量化前将显著输入通道对应的权重行放大 $s$ 倍再把 $1/s$ 折入前一层, 将留出集输出相对误差从 RTN 的 `0.0759` 压至 `0.0273`。\n- KIVI (KV 缓存量化): K 存在少数跨所有位置恒定偏大的离群特征通道, 故按通道方向分组量化 (`0.18 -> 0.026`); V 无固定离群通道, 按 token 方向分组即可流式追加。\n- GPTQ / SmoothQuant: GPTQ 利用校准激活协方差逐行把已量化误差补偿给后续未量化行; SmoothQuant 则把激活离群值平滑转移部分给权重以解锁 W8A8 整数算子加速。',
      question: '给线性层权重量化时, 为什么最小化层输出误差 $\\|XW - X\\hat{W}\\|$ 比直接最小化权重自身误差 $\\|W - \\hat{W}\\|$ 有效得多? 为什么 K 和 V 缓存要沿着互相垂直的两个方向分组?',
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
          title: 'Group-wise 细粒度分组将离群值的破坏半径关进小块内',
          body: '均匀整数格点量化的步长等于 `(max - min) / (2^bits - 1)`: 降到只剩 16 个格点的 INT4 时, 整列里只要混进一个幅值大几十倍的离群值, 整列的刻度就会被拉得极粗, 其余中小权重全挤在两三个格点上。\nGroup-wise INT4 将同一列内每 $g$ 个连续元素划为一组独立统计 `scale` 与 `zero` (每组多花两个 fp16 即 32 bit 元数据: `g=128` 折合 `4.25 bit/参数`, `g=32` 折合 `5.00 bit/参数`), 让离群值只影响本组内的 $g$ 个元素, 使直接就近舍入 (RTN) 的输出相对误差从整通道分组的 `0.1058` 降至 `0.0759`。',
        },
        {
          title: 'AWQ: 按激活幅度先放大关键权重行再量化, 保护层输出',
          key: true,
          body: '网络下一层真正接收的是矩阵乘输出 $X\\hat W$, 其误差展开为 $\\sum_i x_i \\Delta W_i$: 同样大小的权重舍入误差 $\\Delta W_i$, 落在激活幅值 $|x_i|$ 常年偏大的显著输入通道上时对输出的破坏要严重几十倍。\nAWQ (Activation-aware Weight Quantization) 利用等价变换 $XW = (X / s)(s \\odot W)$, 按校准集逐通道平均激活幅度设置缩放向量 $s_i = (\\mathrm{mean}|x_i|)^\\alpha$: 量化前把重要权重行放大 $s_i$ 倍使其享有更精细的相对刻度, 同时将反向缩小系数 $1/s_i$ 提前离线吸收进上一层的归一化或投影权重中, 线上推理零额外计算。由于把单行放太大又会反过来撑宽同组量程, 指数 $\\alpha \\in [0, 1]$ 只需做一次一维网格搜索 (`α=0.4` 最优, 留出集输出相对误差从 RTN 的 `0.0759` 降至 `0.0273`)。',
        },
        {
          title: 'KIVI: 顺着 K 的通道离群特征按列切, 顺着 V 的流式写入按行切',
          body: '长文本下 KV 缓存反超权重成为显存大头, 而 K 与 V 张量的数值分布截然不同:\n- K 张量存在少数固定的离群通道 (所有词元在同一特征维上数值都极大): 若按 token 整行分组, 每一行都会撞上这几个大值导致误差高达 `0.18`; 改为按特征通道在每 32 个时间步内纵向分组 (per-channel), 离群维被单独关进自己的小组, INT4 误差骤降至 `0.026` (未凑满一组的近期新词暂留 fp16 缓冲区)。\n- V 张量无固定通道尖峰, 直接按每个新词独立横向分组 (per-token, 误差 `0.0894`), 生成一个词即可当场量化归档。',
        },
        {
          title: 'GPTQ 与 AWQ 治的是不同误差源: 逐行协方差补偿 vs 显著行保护',
          body: 'GPTQ 与 AWQ 产出的底层 INT4 存储格式完全一致, 但入手角度正交:\n- AWQ 在量化前通过通道重缩放保护大激活对应的敏感行。\n- GPTQ 则利用校准集输入协方差 $H = X^\\top X / N$ 的逆矩阵 Cholesky 分解, 每量化完第 $i$ 行就把该行产生的舍入误差投影分摊给后面尚未量化的相关行去反向抵消。\n在同时存在通道相关性与 30 倍离群通道的留出集测试中: RTN 为 `0.0749`, 单用 AWQ 为 `0.0258`, 单用 GPTQ 为 `0.0293`, 先平滑放大再协方差补偿两者叠加更可压至 `0.0113`。',
        },
        {
          title: 'GPTQ 依赖充足校准样本估计真实相关性, 样本过少反而过拟合噪声',
          body: 'GPTQ 的跨行误差抵消完全建立在输入各维真实线性相关的前提上: 若输入各维彼此独立 ($H$ 为对角阵), 补偿项恒为零, GPTQ 原地退化回 RTN; 更需警惕的是当校准集样本数 $N$ 过小时 (如 256 维特征仅给 $N=256$ 条样本), 样本协方差矩阵接近奇异, GPTQ 会把偶然采样噪声当成真相关性大力补偿, 导致留出集误差反升至 RTN 的 `1.56` 倍 (扩充至 `N=2048~8192` 后才趋于稳定)。',
        },
        {
          title: 'SmoothQuant 迁移激活离群值解锁 W8A8, FP8 用浮点指数天然抗尖峰',
          body: '前几种方法只压权重而保留浮点激活 (weight-only), 加速的是带宽受限的小批量 decode; 若想让算力受限的长文本 prefill 也跑进高速整数/低精度 Tensor Core, 就必须把激活一并量化:\n- SmoothQuant (W8A8): 激活含剧烈离群通道难以直接做整张量 INT8 量化; 通过双向缩放因子 $s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$ (`α=0.5`) 把一半难度平滑转嫁给更容易量化的权重, 使 W8A8 输出误差从 `0.0278` 降至 `0.0070`。\n- FP8 (E4M3): 浮点格点间距随数值自身大小按比例伸缩, 正规数区间相对误差恒不超过 $2^{-4} = 6.25\\%$, 面对 50 倍激活尖峰时无需繁琐逐通道校准即可将整张量量化误差控制在 `0.0360` (同条件 INT8 为 `0.0606`)。',
        },
      ],
      links: [
        { from: 'quantize_int8 (per-channel)', to: 'quantize_groupwise', body: '从整列共享一套刻度细化为每 $g$ 个元素共享一套刻度, 以每权重额外 `32/g` 比特的元数据换取 4-bit 可用精度。' },
        { from: 'mean|x_i|', to: 's_i = mean|x_i|^α', body: 'AWQ 仅统计前向校准集各输入维的平均绝对值即可跑完一维网格搜索, 全程无需反向传播或梯度微调, 数分钟内即可量化百亿模型。' },
        { from: 'KV 量化', to: 'KV 体积一章', body: 'KIVI 把单元素存储从 2 字节压到 0.5 字节, 下一章 GQA / MLA 则从架构源头砍掉每 token 需要存储的元素个数, 两者可正交乘积叠加。' },
      ],
      sourceRows: [
        { concept: '分组量化', code: 'm08_quantization/int4_awq.py:quantize_groupwise', takeaway: '将权重重排为 `(D_in / g, g, D_out)` 沿组内轴求极值, 算出每组独立的缩放与零点偏移。' },
        { concept: '输出相对误差', code: 'm08_quantization/int4_awq.py:output_err', takeaway: '统一以矩阵乘输出相对 Frobenius 误差 $\\|XW - X\\hat W\\|_F / \\|XW\\|_F$ 作为各量化算法的公平标尺。' },
        { concept: 'AWQ 搜索', code: 'm08_quantization/int4_awq.py:awq_quantize', takeaway: '在 `[0, 1]` 上网格搜索最优幂指数 $\\alpha$, 几何中心归一化缩放向量以保护显著输入通道。' },
        { concept: 'KV 三种分组', code: 'm08_quantization/kv_quant.py:quantize_kv', takeaway: '仅改变张量重排与归约轴即可切换整张量、按词元与按通道三种方案, 清晰复现 K 缓存近 7 倍的误差落差。' },
        { concept: 'GPTQ', code: 'm25_weight_quant/gptq.py:gptq_quantize', takeaway: '基于逆协方差的 Cholesky 上三角因子逐行量化并把当前行残差即时补偿到未量化后缀行。' },
        { concept: '平滑因子', code: 'm25_weight_quant/smoothquant.py:smooth_scales', takeaway: '按激活与权重列极值幂次之比构造对角迁移因子, 部署时离线吸收进前序层。' },
        { concept: 'W8A8', code: 'm25_weight_quant/smoothquant.py:w8a8_matmul', takeaway: '激活按行、权重按列做对称 INT8 量化, 缩放因子外提后核心乘加全走整数运算。' },
        { concept: 'FP8 舍入', code: 'm25_weight_quant/fp8.py:fp8_round', takeaway: '模拟对数变步长的 E4M3 / E5M2 浮点格点吸附与上下界饱和行为。' },
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
      subtitle: '单张 GPU 能同时服务多少条长对话, 直接受限于每生成一个词要往缓存里存多少字节。从全头独立存储的 MHA, 到分组共享的 GQA, 再到低秩潜向量加矩阵吸收的 MLA, 缓存体积缩小了一个数量级以上。',
      tldr: '对比主流注意力架构在 fp16 下每个 token 占用的 KV 缓存字节数:\n- LLaMA-2-7B (MHA, 32 个 KV 头各自独立): `512 KiB / token`。\n- LLaMA-3-8B (GQA-8, 32 个查询头分 8 组共享 KV): `128 KiB / token` (缩至 1/4, 同显存并发上限翻 4 倍)。\n- DeepSeek-V3 (MLA, 缓存 512 维共享压缩潜向量加 64 维解耦旋转位置键): `68.6 KiB / token` (相比同规模 128 头 MHA 缩减 `56.9` 倍, 并通过矩阵乘法结合律吸收上投影免除逐步还原开销)。',
      question: 'MLA 既然已经把 K 和 V 压进了一个低维潜向量里, 为什么不能直接对潜向量施加 RoPE 旋转, 非要额外单列一份解耦的共享 RoPE key?',
      evolution: {
        title: 'cache 里存得越来越少',
        subtitle: '根问题: 一张卡能同时服务几条序列 = KV 显存 ÷ 每 token 字节 ÷ 上下文长度。',
        steps: [
          { name: 'MHA 全存', pain: '(原点) 并发上限由每 token 的 KV 字节数决定', fix: '每个头各存一对 K/V; LLaMA-2-7B 每 token 512 KiB' },
          { name: 'GQA', pain: '512 KiB/token, 4K 上下文一条序列就 2 GiB', fix: '8 组 query 头各共用一对 KV 头: LLaMA-3-8B 128 KiB; 砍到 1 组就是 MQA' },
          { name: 'MLA 存 latent', year: 2024, pain: 'KV 头数砍到底, cache 仍与 $n_{kv} \\cdot d_{\\text{head}}$ 成正比', fix: '只存一个维度很小的压缩向量 (低秩 latent $C$), 用时乘 $W_{UK}$、$W_{UV}$ 现场还原每个头的 K/V; 代价是每步还原 T 行' },
          { name: 'absorb', pain: '每步把 T 行 latent 升维成每头 K/V, 计算随上下文涨', fix: '把 $W_{UK}$ 乘到 q 上, cache 直接当 K 用: decode 等价于一个大 head_dim 的 MQA' },
          { name: '解耦 RoPE key', pain: 'latent 一带 RoPE, 旋转夹在 $W_{UK}$ 前面, 吸收做不成', fix: 'latent 不旋转, 另存一份各头共享的 RoPE key: DeepSeek-V3 每 token 68.6 KiB' },
        ],
      },
      code: 'llm_infer/m18_kv_attention_variants/attention_variants.py',
      points: [
        {
          title: 'MHA / GQA / MQA 的单词缓存账统一由 KV 头数 n_kv 决定',
          key: true,
          body: '前三种结构的单词缓存体积共用一条精确公式: `2 × n_kv × d_head × n_layer × dtype_bytes` (开头的 `2` 代表 K 与 V 各存一份)。三者的唯一差别在于 $n_{\\text{head}}$ 个查询头如何分摊 $n_{kv}$ 组键值头:\n- MHA ($n_{kv} = n_{\\text{head}}$): 每头各管各的, LLaMA-2-7B (`32 头, d=128, 32 层`) 每词占 `512 KiB`。\n- GQA ($1 < n_{kv} < n_{\\text{head}}$): 同组查询头通过张量广播共享同一对 K/V, LLaMA-3-8B (`n_kv=8`) 降至 `128 KiB`; 固定 40 GiB 缓存池跑 8K 上下文时, 最大并发请求数直接从 10 条跃升至 40 条。\n- MQA ($n_{kv} = 1$): 全员共享唯一一对 K/V, 体积压至极限但大容量模型表达力稍受损。',
        },
        {
          title: 'MLA 不缓存显式 K/V, 改存联合低秩潜向量与共享位置键',
          body: 'DeepSeek-V2/V3 的多头潜在注意力 (MLA) 跳出了「按头存 K/V」的框架, 缓存里只放两样紧凑向量: 一份 K 与 V 联合共享的低维压缩潜向量 $C \\in \\mathbb{R}^{d_c}$, 外加一份所有头共享的解耦位置键 $k_{\\text{rope}} \\in \\mathbb{R}^{d_{\\text{rope}}}$。\n单词字节公式随之省去了开头的因子 `2`, 变为 `(d_c + d_rope) × n_layer × dtype_bytes`: DeepSeek-V3 (`d_c=512, d_rope=64, 61 层`, fp16) 每词仅占 `68.6 KiB`, 比同层同头数 MHA (`3,904 KiB`) 整整缩小了 `56.9` 倍; 小模型增量解码实测缓存字节与公式逐位相等且输出误差仅 `1.88e-06`。',
        },
        {
          title: '矩阵吸收 (absorb) 省掉逐步还原计算, 也解释了为何 RoPE 必须解耦',
          body: '若每次解码都先把历史 $T$ 行潜向量乘上投影矩阵 $W_{UK}$ 还原出各头的显式 K, 解码计算量就会随序列变长而暴涨。利用矩阵乘法结合律改写内积: $q (C W_{UK})^\\top = (q W_{UK}^\\top) C^\\top$——由于解码时新词 $q$ 只有 1 行, 先把 $W_{UK}$ 吸收乘到单行 $q$ 上几乎不花时间, 随后即可直接拿缓存里的潜向量 $C$ 当成共享单头 MQA 的键做点积 ($W_{UV}$ 同理可吸收到输出投影后)!\n这一步推导也回答了关键结构疑问: 为什么潜向量 $C$ 不能直接带位置旋转? 一旦把随历史位置 $t$ 变化的旋转矩阵 $R_t$ 作用在还原后的键上, 式子就变成 $q R_m (R_t C_t W_{UK})^\\top = q R_{m-t} W_{UK}^\\top C_t^\\top$, 中间横插了一个随相对位置 $m-t$ 不断变化的旋转阵 $R_{m-t}$, $W_{UK}$ 便再也无法提取吸收进单行 $q$ 里了! 因此必须保持内容潜向量 $C$ 不旋转, 把位置信息单独剥离到极窄的共享 $k_{\\text{rope}}$ 分支上。',
        },
      ],
      links: [
        { from: '阶段 2 注意力演进', to: 'KV bytes/token', body: '阶段 2 从模型架构与参数量视角拆解 MHA → GQA → MLA, 本章则在推理服务端将其直接换算为单卡并发槽位数。' },
        { from: 'bytes/token', to: 'BlockManager.num_blocks', body: '引擎启动时用预留给缓存的显存预算除以 `bytes_per_token × block_size`, 即可确定全局分页池的总块数。' },
        { from: 'MLA absorb', to: 'FlashMLA', body: '生产级算子如 FlashMLA 直接在 `512 + 64 = 576` 维吸收态潜空间上做分块流式注意力, 全程无需在显存中展开多头 K/V。' },
      ],
      sourceRows: [
        { concept: 'GQA 公式', code: 'm18_kv_attention_variants/attention_variants.py:kv_bytes_per_token', takeaway: '计算 `2 * n_kv * d_head * n_layer * nbytes`, 验证字节开销与 `n_kv` 严格成正比。' },
        { concept: 'MLA 公式', code: 'm18_kv_attention_variants/attention_variants.py:mla_bytes_per_token', takeaway: '计算 `(d_c + d_rope) * n_layer * nbytes`, K 与 V 共享潜向量故消去二倍系数。' },
        { concept: '解耦前向与吸收解码', code: 'MLALayer.forward', takeaway: '验证显式升维还原路径与矩阵吸收路径的注意力分数最大误差仅 `2.86e-06`、整网输出误差仅 `1.43e-06`。' },
        { concept: '真实数组字节核对', code: 'cache_nbytes', takeaway: '对 MHA / GQA / MQA / MLA 四套缓存数组的真实 `.nbytes` 求和 (`61440 / 15360 / 7680 / 7680`), 断言与解析公式分毫不差。' },
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
      subtitle: '无限持续的流式语音或长程对话不能让 KV 缓存无上限膨胀; 然而简单采用滑动窗口一旦把句首最初几个词挤出缓存, 模型输出便会瞬间崩溃。StreamingLLM 通过永久钉住句首汇聚词并按缓存槽位重编位置实现恒定内存流式生成。',
      tldr: '- 崩溃根因 (Attention Sink): Softmax 强制注意力权重之和恒为 1; 当前位置不需要参考历史信息时, 预训练模型习惯将多余注意力倾倒在所有位置都看得见的句首前几个词上。一旦滑窗把这几个词逐出, 分母骤降会导致剩余权重整体失真放大。\n- StreamingLLM 解法: 始终保留开头 `n_sink` 个词外加最近 `window` 个词使缓存大小恒定; 同时存储未施加 RoPE 旋转的原始 K 并按当前缓存槽位 `0 .. L-1` 动态重旋转, 确保位置编码永不越过预训练长度上限。',
      question: '常规 KV cache 存的都是已经乘过 RoPE 旋转矩阵的 K, 为什么流式 SinkCache 必须反过来存未旋转前的原始 K?',
      evolution: {
        title: '有界, 还不能崩',
        subtitle: '根问题: 流式对话没有尽头, KV 和位置编号都随 token 一直涨。',
        steps: [
          { name: '全量 KV cache', pain: '(原点) 对话不停, KV 线性增长, 位置迟早超过训练长度', fix: '全存; 显存迟早不够, 超出训练长度的位置模型从没见过' },
          { name: '滑动窗口', pain: '显存和位置都没有上界', fix: '只留最近 window 个 token; 开头几个一被挤出, 输出立刻漂掉' },
          { name: 'sink + 窗口 (StreamingLLM)', year: 2023, pain: 'softmax 权重和恒为 1, 开头 token 被训成倒注意力的地方, 逐出后分母骤变', fix: '开头 n_sink 个永不逐出, 其余走滑动窗口; cache 恒 ≤ n_sink + window' },
          { name: '按槽位重编号', pain: '绝对位置还在涨; K 存的是旋转后的, 位置已烙死改不了', fix: '位置取 cache 槽位 0..L−1, K 存旋转前的, 每步整体重转一遍, 代价 O(L·D)' },
        ],
      },
      code: 'llm_infer/m16_attention_sinks/sink_cache.py',
      points: [
        {
          title: '句首词充当了 Softmax 归一化分母的「泄洪口」',
          key: true,
          body: '自注意力中每行经过 softmax 后所有位置的权重之和必须等于 1。当模型正在输出固定搭配或标点、并不需要从上下文提取实质信息时, 多余的概率质量总得找个固定锚点安放; 在下三角因果掩码下, 只有序列最开头的三四个词对后续所有位置永远可见, 于是预训练自然把它们塑造成了吸收冗余注意力的汇聚点 (attention sink, 与开头具体是什么词义基本无关)。\n一旦纯滑动窗口把开头这几个词挤掉, 原本占分母大头的指数项突然消失, 窗口内普通词的权重瞬间被放大数倍甚至十倍, 激活分布彻底失真。在植入 sink 特性的合成流测试 (`demo [B]`, 前 4 词吸走 93.0% 注意力) 中, 同为 36 槽预算下保留 sink 的相对误差为 `19.40%`, 纯滑窗则直接崩至 `103.06%`。',
        },
        {
          title: '固定大小环形淘汰: 钉死前 n_sink 格, 滚动替换中间最老格',
          body: '`SinkCache` 将容量硬性锁死在 `n_sink + window` (如 `4 + 60 = 64`): 新词追加到队尾后若总长超限, 永远精准剔除下标 `n_sink` 处的元素 (即滑动窗口里最老的那一个词), 而下标 `0 .. n_sink-1` 纹丝不动。\n需明确它的能力边界: `SinkCache` 解决的是「无限流式生成不爆显存、不崩溃乱码」, 而非「无限长程记忆」——一旦某个词滑出窗口被丢弃, 模型就再也读不到它的具体内容了 (长程精确召回需搭配外部检索或分层 KV offload)。',
        },
        {
          title: '缓存未旋转的原始 K, 按当前槽位重编相对位置以防坐标越界',
          body: '流式跑上百万词时除了显存限制, 还会撞上第二个硬墙: 绝对位置下标早已远远超出 RoPE 预训练表长 (例如 64 行的位置表在跑到第 1023 步时直接抛出 `IndexError`)。\n`SinkCache` 改为按当前缓存槽位实时分配位置 `0, 1, ..., L-1`, 让最新查询词永远坐在槽位 `L-1`, 相对距离始终落在预训练熟悉的短窗口内。正因为每淘汰一个旧词都会让窗口内所有词的槽位号整体左移一格, 如果缓存里存的是已经烙死旧绝对位置的旋转后 K 就无法改写了; 因此 `SinkCache` 必须保存未加 RoPE 的原始 K, 每步对有界长度 $L$ 的缓存整体现算一次旋转 (因 $L$ 很小且恒定, $O(L \\cdot D)$ 开销极轻)。',
        },
      ],
      links: [
        { from: 'KV cache (m01)', to: 'SinkCache', body: '在标准线性增长缓存上增加「保留头部汇聚槽 + 淘汰窗口首槽 + 槽位动态旋转」三项改造, 将空间复杂度从 $O(T)$ 降为常数 $O(1)$。' },
        { from: 'RoPE', to: 'positions = arange(L)', body: '利用 RoPE 仅依赖相对位置差的性质按槽位连续重编号, 确保长流运行百万步也不会踏入未训练的外推位置区。' },
        { from: 'select_blocks 的 forced', to: '稀疏 decode', body: '后续「稀疏注意力 decode」在按相关性挑块时同样把第 0 块列入强制必选集合 (`forced`), 背后正是同一个保护 softmax 分母的物理原因。' },
      ],
      sourceRows: [
        { concept: '定向逐出', code: 'm16_attention_sinks/sink_cache.py:SinkCache.append', takeaway: '超限时切片拼接 `[0:n_sink]` 与 `[n_sink+1:]`, 始终淘汰窗口最老槽位而不动头部汇聚词。' },
        { concept: '槽位重旋转', code: 'm16_attention_sinks/sink_cache.py:stream_step', takeaway: '每步生成 `pos = arange(L)` 对原始未旋转 K 与当前 Q 现场施加 RoPE, 64 行位置表即可平稳跑完 1,024 步长流。' },
        { concept: '流式稳定与长程记忆区分', code: 'window 之外的 token', takeaway: '清晰界定流式防崩与大海捞针记忆的区别: 滑出窗口的信息已被物理丢弃。' },
        { concept: '未经训练的随机权重对照', code: 'TinyLM 的 attention', takeaway: '诚实展示未训练随机权重模型注意力近乎均匀分布 (`~1/T`) 故无自发 sink 现象, 证明 attention sink 是大规模自回归预训练涌现的结构特征。' },
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
      subtitle: '单链投机解码一旦在第一个草稿词处猜偏, 后面跟着猜的 K−1 个词便全盘作废。EAGLE 通过接入大模型顶层隐状态让草稿猜得更准, 树形投机则用祖先注意力掩码在单次大模型前向里同时验证多条候选分支。',
      tldr: '沿着两个正交方向放大单次大模型验证的平均产出词数:\n- EAGLE (特征层自回归草稿): 草稿头不再只看离散 token id, 而是直接读取大模型验证时顺手算出的顶层特征向量 $h_t$ 并复用大模型的 `lm_head`, 显著提升单步命中率。\n- 树形投机 (Tree Speculation): 每个节点保留 `top-k` 个备选分支张成候选树, 构造只看自身祖先的二维 `tree_mask` 并让同层兄弟节点共享同一个 RoPE 深度位置, 一次大模型前向即可验完整棵树 (同样验证 16 个词元时, 单次产出从长链 `K=15` 的 `2.13` 词升至树 `[3,2,1]` 的 `2.58` 词)。',
      question: '把一棵候选树的 16 个节点展平一次喂给大模型验证时, 为什么处于同一深度的兄弟节点必须共用同一个 RoPE 位置下标, 而不能按数组顺序依次加一?',
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
          title: 'EAGLE: 复用大模型顶层隐状态与输出头, 在特征空间递推草稿',
          body: '独立小模型草稿只能根据离散词元编号从头重猜, 白白丢掉了大模型上一轮验证前向已经算出的高维隐状态 $h_t$。EAGLE 用一个极轻量的单层草稿网络接收「上一位置特征 $h_t$ + 刚确认词元的嵌入 $\\mathrm{emb}(x_{t+1})$」, 直接回归下一位置特征 $\\hat h_{t+1}$ 并原样乘大模型自己的 `lm_head` 选词 (必须同时拼入已选词嵌入 $x_{t+1}$ 以消除采样分支的不确定性)。\nm17 控制其余条件完全相同对比 (`K=4`): 仅吃词嵌入的草稿每轮接受 `0.36/4` 词 (`1.31×`), 接入顶层特征 $h$ 后每轮接受升至 `0.60/4` 词 (`1.52×`), 输出与大模型贪心逐词一致。',
        },
        {
          title: '树形注意力掩码与同深度共享 RoPE, 单次前向并行验完整棵候选树',
          key: true,
          body: '链式草稿中哪怕第 1 候选以微弱差距猜错、第 2 候选其实是对的, 整条链也只能当场中断。树形投机在每层保留多个高分分支 (如 `widths=[3, 2, 1]` 共展开 `1 + 3 + 6 + 6 = 16` 个节点), 从根到任意节点的路径各代表一条平行宇宙续写。\n要把这 16 个节点打包进同一次大模型前向且保证每条分支的打分与单独串行解码丝毫不差, 必须同时配齐两项设置:\n- 祖先闭包掩码 (`tree_mask`): 节点 $i$ 只能看见历史前缀以及树内自己的直系祖先节点 (`anc[i, j] == True`), 同层兄弟与旁系分支全部置 `-inf` 严格隔离。\n- 同深度共享位置下标: 同一深度的兄弟节点竞争的是序列里的同一个未来槽位, 因此它们的 RoPE 位置必须统一设为 `n_ctx + depth` (若按展平数组下标递增, 被选中路径上的位置编码就会错位); 实测树路径 logits 与串行解码最大误差仅 `9.5e-07`。',
        },
        {
          title: '用带宽受限期富余的矩阵乘算力, 把预算花在「前几层的宽」而非「单链的深」上',
          body: '单链第 $k$ 步存活概率按 $\\alpha^k$ 指数衰减, 拉长单链尾部几乎全是废算力; 将同样的 16 个验证槽位从单链 `K=15` 改为树 `[3, 2, 1]` 后, 单次大模型调用的平均产出从 `2.13` 词提升至 `2.58` 词 (同样调 3 次草稿的单链 `K=3` 仅为 `1.86` 词)。\n在小批量解码阶段, 耗时卡在搬运权重而非算力上, 一次前向验 4 个词还是验 16 个词墙钟时间几乎不变; 当线上并发升高、GPU 算力趋于饱和时, 引擎再动态收缩树宽即可。',
        },
      ],
      links: [
        { from: 'm07 accept_greedy', to: 'accept_tree', body: '将链式从左到右比对推广为从树根向下贪心寻路: 当前节点的 `target argmax` 匹配哪个孩子就步入哪个孩子, 失配处即得纠错或奖励词。' },
        { from: 'TinyLM.forward(return_hidden)', to: 'EagleDrafter.propose', body: '大模型验证前向顺手返回已接受路径的真实顶层隐状态, 直接作为下一轮 EAGLE 草稿递推的零成本起点。' },
        { from: 'gather_kv', to: 'KV 回滚', body: '验证完 16 个树节点后, `gather_kv` 按最终胜出路径的下标挑出对应的几行 K/V 紧凑留存 (因其深度恰好连续对应 `0, 1, 2...`, 无需重算旋转), 其余分支槽位直接释放。' },
      ],
      sourceRows: [
        { concept: '树拓扑展开', code: 'm19_tree_speculation/tree_spec.py:tree_shape', takeaway: '按广度优先顺序生成各节点的父指针数组 `parents` 与深度数组 `depth` (`[3,2,1]` 对应 16 个节点)。' },
        { concept: '祖先掩码与深度位置', code: 'm19_tree_speculation/tree_spec.py:tree_mask', takeaway: '递推计算布尔祖先闭包矩阵屏蔽旁系分支, 并令位置编码按 `n_ctx + depth` 同层共享。' },
        { concept: '最长匹配路径', code: 'm19_tree_speculation/tree_spec.py:accept_tree', takeaway: '由于同一父节点的 `top-k` 子词互异, 每层至多命中唯一子节点, 自顶向下确定性走出最长精确匹配链。' },
        { concept: 'EAGLE 草稿器', code: 'm17_eagle_speculative/eagle.py:EagleDrafter', takeaway: '实现统一 `propose` 接口融合顶层特征与词嵌入递推预测, 无缝接入 `m07` 验证循环。' },
        { concept: '同槽位效率对比', code: 'm19 demo', takeaway: '固定单次验证 16 个节点: 树 `[3,2,1]` 单轮产出 `2.58` 词, 显著高于单链 `K=15` 的 `2.13` 词与同草稿步数单链 `K=3` 的 `1.86` 词。' },
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
      subtitle: '多用户多轮对话的历史 KV 很容易撑爆有限的 GPU 显存。与其把被淘汰的冷块直接丢弃导致下一轮整段重算, 分层缓存将其逐级下沉至主机内存与本地磁盘, 并在命中时逐层权衡「搬回耗时」与「重算耗时」。',
      tldr: '分层 KV 缓存构建了 `GPU -> CPU -> 磁盘` 三级 LRU 存储池:\n- 逐级降级复用: 单个词元的 KV (以 128 KiB 计) 从 PCIe 内存搬回约需 `5 µs`, 而重算 prefill 约需 `125 µs`; 在 8 用户 6 轮对话模拟中, 接入 CPU 与磁盘层将平均首字延迟 TTFT 从单 GPU 缓存的 `51.5 ms` 依次压低至 `34.8 ms` 与 `19.5 ms`。\n- 逐层盈亏决策: 慢速存储层带有固定寻道/网络启动延迟, 当命中块数低于盈亏平衡点 $n^*$ 或远端带宽极低时, 盲目搬回反而比重算更慢 (`68.9 ms`), 因此每层必须独立取 `min(load, recompute)`。',
      question: '给推理服务多挂一层远端大容量 KV 缓存、缓存命中率明明涨了, 为什么首字延迟 (TTFT) 有时反而变慢了一倍?',
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
          title: '三级 LRU 瀑布流: 挤出 GPU 不丢弃, 再次命中提回热层',
          body: '`TieredKVCache` 按容量从小到大串联 GPU (`64` 块)、CPU (`192` 块) 与本地磁盘 (`4,096` 块) 三级队列: GPU 满时将最久未用的块下沉到 CPU, CPU 满时再下沉到磁盘; 一旦某块在低速层被后续对话重新命中, 则立即将其抽回 GPU 热端。\n在 8 名并发用户各聊 6 轮 (总历史约 442 块, 远超 GPU 的 64 块上限) 的模拟负载中, 纯 GPU 缓存因频繁互踢导致 `86.7%` 词元未命中 (平均 TTFT `51.5 ms`), 加上 CPU 与磁盘两级承接后未命中率只剩每轮新增提问的 `15.7%`, 平均 TTFT 降至 `19.5 ms`。',
        },
        {
          title: '前缀链连续性约束与「从尾到头逆序刷新」技巧',
          body: '分层查找沿用链式块哈希, 从第 0 块往后逐块跨层检索并在首个全层缺失处立即停止——即便第 4 块还留在磁盘里, 一旦第 3 块丢了, 前缀缺失也会导致第 4 块及之后的缓存无法单独拼接使用。\n为避免平面 LRU 错误地先把第 0 块淘汰导致整条链雪崩失效, 写入整段历史时特意采用「从链尾向链头逆序 touch」, 确保块 0 的访问时间戳最新、淘汰总是从链尾叶子端先开始。',
        },
        {
          title: '跨过固定延迟盈亏点 n* 才值得加载, 慢速层必须走 min(load, recompute)',
          key: true,
          body: '从低速层搬回 $n$ 个块耗时为 $\\text{load}(n) = \\text{latency} + n \\cdot t_{\\text{xfer}}$, 而直接在 GPU 上重算耗时为 $\\text{recompute}(n) = n \\cdot t_{\\text{comp}}$。联立可得盈亏平衡块数 $n^* = \\text{latency} / (t_{\\text{comp}} - t_{\\text{xfer}})$:\n- 固定延迟摊销门槛: demo 中磁盘层启动延迟 `2.0 ms`、每块传输 `0.699 ms`、每块重算 `2.00 ms`, 算出 $n^* = 1.54$ 块——只命中 1 块时搬回要 `2.70 ms` 反而不如直接重算 (`2.00 ms`), 命中 2 块及以上 (`3.40 ms < 4.00 ms`) 搬回才开始赚时间。\n- 慢链路自动熔断: 若挂上一层带宽仅 `0.5 GB/s` 的慢速远端存储 (单块搬回 `4.194 ms` 比单块重算还慢), 「逢命中必搬回」的朴素策略会把平均 TTFT 从 `34.81 ms` 恶化到 `68.91 ms`; 在每层独立判断 `min(load, recompute)` 后则自动放弃慢速加载改走重算, 稳稳守住 `34.81 ms`。',
        },
      ],
      links: [
        { from: 'm04 PrefixCache', to: 'TieredKVCache', body: '复用完全相同的链式块哈希键设计, 将单层 GPU 字典扩展为跨多级存储介质的分层检索与升降级管线。' },
        { from: 'Tier.bandwidth_GBps', to: 'load_ms', body: '本地存储带宽通常以字节每秒 (`GB/s`) 标称, 直接用块字节数相除即可 (注意与网卡比特率 `Gbps` 区分)。' },
        { from: 'use_load', to: 'TTFT', body: '`serve` 对每层命中的块数实时对比搬运与重算耗时, 保证挂载任意慢速外部存储都绝不会让延迟出现负优化。' },
      ],
      sourceRows: [
        { concept: '代价模型', code: 'm20_kv_offload/tiered_cache.py:CostModel', takeaway: '按 LLaMA-3-8B fp16 (`128 KiB/token`) 与 `8,000 tok/s` prefill 吞吐建模各级加载耗时与重算耗时。' },
        { concept: '瀑布式降级', code: 'TieredKVCache._insert', takeaway: '本层满载时弹出 LRU 队首旧块递归推入下一级大容量层, 直至末层满载才物理丢弃。' },
        { concept: '逐层性价比裁剪', code: 'm20_kv_offload/tiered_cache.py:TieredKVCache.serve', takeaway: '按层汇总命中块数并执行 `min(load, recompute)`, 自动跳过得不偿失的零星命中或慢速远端读取。' },
        { concept: '多轮会话模拟', code: '8 个用户 × 6 轮', takeaway: '验证三级缓存相比全重算 (`59.42 ms`) 与纯 GPU 缓存 (`51.5 ms`) 逐级将平均 TTFT 压低至 `19.5 ms`。' },
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
      subtitle: '在线推理阶段路由权重已经定型, 无法再靠加训练损失压平热门专家。当最热专家单枪匹马就超过了单卡平均负载时, 仅调换专家位置也无济于事, EPLB 通过为热门专家部署冗余副本并均分流量消除多卡木桶效应。',
      tldr: '多卡专家并行 (EP) 每一层的 `combine` 归约都必须等最慢的那张卡算完, 单步延迟完全由 `max_rank_load` 决定:\n- 倾斜痛点: 在服从 Zipf 偏置的路由流量下 (16 专家分 4 卡, top-2), 顺号摆放导致最忙卡负载高达均值的 `2.95` 倍, 集群平均有效利用率仅剩 `34%`; 即便按历史热度贪心重排位置, 因最热的 0 号专家独自就占了单卡均值的 `1.42` 倍, `max/mean` 仍卡在 `1.48` 倍下不去。\n- EPLB (专家并行负载均衡): 预留少量冗余槽位复制最热专家的权重并轮询分流其输入词元, 仅加 4 个冗余副本即将 `max/mean` 压平至 `1.02` 倍, 且模型输出逐位不变。',
      question: '既然可以把重负载专家和轻负载专家搭配摆放到不同 GPU 上, 为什么有时不管怎么重新排列组合都还是均衡不了, 非得把同一个热门专家复制好几份?',
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
          title: '全卡同步归约使单层延迟完全受制于最忙 rank',
          key: true,
          body: 'MoE 推理层通过 `dispatch` (all-to-all 发往目标卡) 与 `combine` (all-to-all 收回原位按门控加权) 包夹本地专家计算。由于每个词元选中的 top-2 专家往往跨在不同 GPU 上, 必须等全部 4 张卡都算完才能启动 `combine` 进入下一层。\nm21 在 2,048 个词元、16 专家分 4 卡、带 Zipf 热度倾斜的场景下实测: 顺号连续摆放时四张卡分到的 `(token, expert)` 任务数悬殊为 `[3016, 851, 125, 104]` (均值 `1,024`, `max/mean = 2.95`), 相当于其余三张卡近三分之二的时间都在干等 rank 0。',
        },
        {
          title: 'EPLB 两阶段贪心: 先按单副本负载削峰加副本, 再按轻重装箱摆放',
          body: 'DeepSeek EPLB 基于近期历史批次统计出的各专家热度, 分两步更新物理槽位映射 (并在新到达的未见测试批上评测):\n- 贪心分配冗余副本: 每次找出当前「总负载 / 已有副本数」最高的瓶颈专家为其追加 1 个副本, 循环 `n_redundant` 次 (demo 每卡多留 1 个槽共加 4 个冗余槽后, 极热的 0 号专家扩为 4 副本、1 号扩为 2 副本)。\n- 降序装箱放置: 将所有副本槽位按各自均摊后的预期负载从重到轻排序, 依次放入当前累计负载最轻且尚有空槽的 rank 上; 线上路由命中多副本专家时直接轮询打散流量, 使四卡实际负载拉平至 `[1025, 1033, 992, 1046]` (`max/mean = 1.02`, 峰值负载削减 65%)。',
        },
        {
          title: '单专家超过单卡均摊水位时, 不复制拆分就存在数学下界死结',
          body: '为什么只重排位置而不加副本 (`max/mean = 1.48`) 压不到 1.0 附近? 算一笔硬账就清楚了: 本例 4 张卡总任务数为 `2048 × 2 = 4096`, 理想单卡均值为 `1,024`; 然而 0 号热门专家自己一个人就收到了 `1,451` 个任务 (`1451 / 1024 = 1.42 > 1`)!\n只要单个专家权重不拆份, 无论把它挪到哪张卡上, 那张卡保底就要独自咽下这 `1,451` 份计算; 唯有复制出多个权重一模一样的副本分驻多卡并均分其输入词元, 才能在不改变任何一行模型数学输出 (`max|Δ| == 0`) 的前提下打破单专家瓶颈。',
        },
      ],
      links: [
        { from: '阶段 2 MoE 路由', to: 'MoELayer.route', body: '复用完全相同的 `top-k` 门控打分逻辑, 视角从模型表达力转向线上多卡流量的长尾倾斜治理。' },
        { from: '阶段 3 EP all-to-all', to: 'ep_forward', body: '训练期可借助辅助损失或偏置微调改变路由器选人倾向; 推理期模型权重已冻结只读, 只能靠服务引擎动态重映射物理槽位。' },
        { from: 'eplb_placement', to: 'slot_of', body: '`eplb_placement` 按历史统计周期性更新槽位排布表, `slot_of` 在每次前向时将热门专家的请求均匀轮转分发至各物理副本。' },
      ],
      sourceRows: [
        { concept: '路由门控', code: 'm21_moe_serving/moe.py:MoELayer.route', takeaway: '计算专家打分取 `top-k` 并在选中子集上做局部 softmax 归一化权重。' },
        { concept: '分发与归约记账', code: 'm21_moe_serving/moe.py:ep_forward', takeaway: '模拟跨 rank 分发、按物理槽批量计算与加权归约, 精确清点各 rank 承受的计算任务数 (总和恒为 `4,096`)。' },
        { concept: 'EPLB 削峰装箱', code: 'm21_moe_serving/moe.py:eplb_placement', takeaway: '核心迭代 `n_rep[argmax(load / n_rep)] += 1` 精准削低最高单副本负载, 随后按最长处理时间优先 (LPT) 贪心装入各卡。' },
        { concept: '逐位等价验证', code: 'max|Δ| = 0', takeaway: '所有副本权重与原专家完全相同, 连续摆放、贪心重排与 EPLB 三种方案的输出与单卡逐词基线差值严格为 `0`。' },
        { concept: '倾斜改善对比', code: '16 专家 / 4 rank / top-2', takeaway: '在未见测试批上将 `max/mean` 从连续摆放的 `2.95×`、无副本重排的 `1.48×` 一举压平至 `1.02×`。' },
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
      subtitle: '长文本解码每步全量读取几万词的 KV 带宽代价高昂, 而训练成熟的大模型在任一步往往只将绝大部分注意力集中在寥寥几个关键块上。Quest 利用常驻的块级逐维极值摘要估算内积上界, 每步只挑出高分块与首尾关键块读取。',
      tldr: '动态稀疏解码在不丢失远距离关键信息 (区别于盲目丢弃窗口外的 SinkCache) 的前提下大幅削减单步 KV 访存量:\n- 块级极值摘要 (Quest): 为每个 KV 物理块常驻逐维最小值 `kmin` 与最大值 `kmax` 向量, 根据查询向量 $q$ 各维正负号算出块内最大可能点积的严格上界, 只挑出首块 (sink)、末块 (局部前文) 与上界最高的 `top-(k-2)` 块读入计算。\n- 成立前提明确: 在注意力高度集中的大海捞针负载上仅读 `3.1%` (`8/256` 块) 的 KV 即可捕获 `97.7%` 注意力质量、输出相对误差降至 `0.023`; 但在注意力均匀弥散的未训练随机模型上稀疏近似前提并不成立。',
      question: '不把块里每个 token 的 K 读出来, Quest 凭什么能保证藏着关键答案「针」的那个块一定不会被漏评低分?',
      evolution: {
        title: '少读 KV, 别漏掉针',
        subtitle: '根问题: 长上下文 decode 每步要把全部 KV 读一遍, 而注意力质量集中在很少的 token 上。',
        steps: [
          { name: '全量读 KV', pain: '(原点) decode 受带宽限制, 每步读的 KV 字节随上下文线性涨', fix: '每步读全部 KV, 结果精确; 大部分字节读来只分到很小的注意力' },
          { name: '固定位置 (sink + 窗口)', pain: '全读太贵, 大部分 KV 读了也几乎不起作用', fix: '只读开头几个和最近一段, 不看 query; 远处的关键 token (针) 落在窗口外就丢' },
          { name: '精确 top-k token', pain: '重要 token 可能在任何位置, 得看 query 才知道', fix: '先算全部 $q \\cdot k$ 再只读 top-k 的 V; 选得准, 但打分时就得读全部 K' },
          { name: 'Quest 块摘要', year: 2024, pain: '打分要读全部 K, 省下的只有 V', fix: '每块常驻 min/max 摘要, q 算上界选 top-k 块: 读 3.1% 的 KV, 误差 0.023' },
        ],
      },
      code: 'llm_infer/m22_sparse_attention/sparse_attention.py',
      points: [
        {
          title: '逐维符号匹配极值构造严格内积上界, 确保高相关「针」块零漏检',
          body: '若为了挑重点先把全部历史 K 读一遍算点积, 那就只省了一半 V 的带宽。Quest 为每个大小为 `bs` 的物理块额外维护两个极小的包络向量 `kmin` 和 `kmax` (体积仅为原块 KV 的 `1/bs`, 如 `bs=16` 时仅占 1/16)。\n当新查询向量 $q$ 到来时, 逐维检查符号: 若 $q_i > 0$ 则乘 `kmax[i]`, 若 $q_i < 0$ 则乘 `kmin[i]`, 求和得到 $\\sum_i \\max(q_i \\cdot \\text{kmin}_i, q_i \\cdot \\text{kmax}_i)$——从代数上它必定大于等于该块内任意单个词元的真实点积 $q \\cdot k$! 尽管在随机背景噪声上该上界偏松, 但它保证了真正与 $q$ 强对齐的目标「针」块分数绝不会被低估漏选。',
        },
        {
          title: '强制保底第 0 块 (Sink) 与末尾最近块, 守住分母基准与局部连贯',
          body: '`select_blocks` 始终将第 0 块 (承载多余注意力的句首 sink) 与最后一块 `nb - 1` (紧挨当前词的局部上下文) 放入必选集合 `forced`, 剩余 `k - 2` 个名额再按上界得分从高到低择优录取。正因首尾固定占去 2 席, 当文本藏有 4 个「针」块而预算只给 `k=4` 时最多只能捞进 2 个针块 (注意力质量召回 `59.1%`), 扩至 `k=8` 后 4 个针块悉数入围。',
        },
        {
          title: '稀疏收益完全仰赖真实预训练注意力的尖峰稀疏性',
          key: true,
          body: 'm22 将 `T=4096` 切成 256 块并在 4 个远距块中植入强对齐「针」信号做对照:\n- 在尖峰集中的大海捞针分布上: 读取块数从 `k=4` (`1.6%`) 增至 `k=8` (`3.1%`) 时, 注意力质量召回率跃升至 `97.7%`, 输出相对误差从 `0.83` 断崖式跌落至 `0.023` (同预算随机抽块误差高达 `1.06`)。\n- 在未经训练的随机权重 TinyLM 上: 注意力均匀摊薄在全部 256 块上, 读 3.1% 的块就只能召回约 3% 的质量, 任何选块算法都无稀疏红利可挖——这也解释了为什么 NSA 与 DeepSeek DSA 进一步将选块索引器纳入端到端训练以学习更紧凑的语义筛选。',
        },
      ],
      links: [
        { from: 'm02 KV block', to: 'block 摘要', body: '直接复用 PagedAttention 的定长物理块边界, 每写满一个块顺手归约一行 `min/max` 摘要即可, 无需改动底层分页池结构。' },
        { from: 'attention sink', to: 'select_blocks 的 forced', body: '第 0 块若被稀疏筛选漏掉同样会引发上一章分析的 softmax 分母坍塌, 故无条件钉死在必读列表中。' },
        { from: '阶段 2 DSA', to: 'lightning indexer', body: 'Quest 是无需重训模型的推理期免训练块筛选启发式; 阶段 2 介绍的 DSA 则在训练期联合学习轻量低维索引器实现更高精度的按词筛选。' },
      ],
      sourceRows: [
        { concept: 'Quest 严格上界', code: 'm22_sparse_attention/sparse_attention.py:quest_upper_bound', takeaway: '计算 $\\sum_i \\max(q_i \\cdot \\text{kmin}_i, q_i \\cdot \\text{kmax}_i)$, 并在断言中验证其恒不小于块内真实最大点积。' },
        { concept: '保底加排序选块', code: 'm22_sparse_attention/sparse_attention.py:select_blocks', takeaway: '固定锁定 `{0, nb - 1}` 首尾两块, 其余槽位按上界分数降序补齐至 $k$ 块。' },
        { concept: '双指标联合评估', code: 'm22_sparse_attention/sparse_attention.py:evaluate', takeaway: '同时报告输出向量的相对 L2 误差与选中块覆盖的真实 softmax 概率质量召回率 (`recall`)。' },
        { concept: '大海捞针实测', code: 'T=4096, 256 blocks', takeaway: '仅读取 `8 / 256 = 3.1%` 的历史 KV 即将相对误差压至 `0.023`, 显著优于同比例随机抽样的 `1.06`。' },
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
      subtitle: '不重新训练更大的基座模型, 在测试阶段额外投入解码 token 预算同样能显著推高复杂推理题的准确率。并行投票、外部判分器选优 (best-of-N / PRM beam) 与串行长思考自查各自擅长消除不同性质的错误。',
      tldr: '测试时计算扩展 (Test-Time Compute Scaling) 主要有三条路线, 能推多高取决于错误的性质:\n- 消除随机粗心错: 多数投票 (`self-consistency`) 与强制延长自查 (`budget forcing`) 无需外部判分器即可抹平偶发失误, 但在模型存在系统性误解的陷阱题上 (模型自身众数就是错的), 采样越多投票反而错得越稳 (`0.254 -> 0.108`)。\n- 跨越系统性误解: 必须引入具备独立判别力的外部验证器——终点选优 `best-of-N + ORM` (`N=64`) 将总准确率推至 `0.970`, 按步剪枝的 `PRM beam` 则在紧预算 (`28 token`) 下以 `0.865` 显著跑赢同开销的 `best-of-7` (`0.747`)。',
      question: '对同一道数学题并行采样几十条推理链再做多数投票, 为什么在有些陷阱题上采样条数 N 越大, 最终答对率反而越低?',
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
          title: '多步推理的指数衰减困境与三条算力换准确率路径',
          body: '一道需要连续走对 $K=4$ 步的算术题, 哪怕单步正确率有 $p=0.75$, 一次连贯全对的概率也只剩 $0.75^4 = 0.316$ (m23 温度 1 下单次采样准确率为 `0.275`)。不换模型权重时, 可以把额外 token 预算花在三个方向上:\n- 并行独立采样后聚合: 采 $N$ 条完整解答, 用无外部监督的多数投票 (`majority_vote`) 或结果判分器挑最高分 (`best-of-N + ORM`)。\n- 树状按步剪枝搜索 (`PRM beam`): 每步扩展多个候选并由过程判分器 PRM 当场裁掉错步, 把后续算力省给有希望的分支。\n- 串行反思拉长思考链 (`budget forcing`): 模型想提前停笔时追加 `"Wait"` 提示词促使其回溯检查并从出错步重写。',
        },
        {
          title: '区分偶发粗心与系统性误解: 多数投票收敛于模型众数而非真值',
          key: true,
          body: 'm23 在 200 道 4 步题中为 `32.5%` 的题目植入了「陷阱步」(在该步看错运算符的错误候选 logit 为 `3.5`, 高于正确答案的 `3.0`), 清晰划出两种错误的边界:\n- 无陷阱常规题 (偶发采样粗心): 正确答案本就是最高概率众数, 多数投票随 $N$ 增大顺利收敛至 `1.000`。\n- 陷阱题 (系统性认知盲区): 模型分布自身的众数落在错误答案上! 此时大数定律反而帮了倒忙——样本 $N$ 越加多, 投票结果就越笃定地锁死那个高频错解, 陷阱题准确率从 `N=16` 的 `0.254` 一路跌至 `N=64` 的 `0.108`, 拖累全局天花板死死卡在 `0.710`。\n要捅破这层天花板, 必须借助能认出少数派正确解的外部判分器: `best-of-N + ORM` 在 `N=64` 时将总准确率拉升至 `0.970`。',
        },
        {
          title: '中小预算下 PRM 按步剪枝效率更高, 大预算下受判分器自身噪声制约',
          body: '`best-of-N` 即使第 1 步就算错了也得老老实实把后面 3 步废话写完才判分; `PRM beam` 则在错步露头的第一时间将其剪除: 在 `28 token` 预算下 (`width=2, expand=4`), `PRM beam` 取得 `0.865` 准确率, 大幅领先花同样 token 的 `best-of-7` (`0.747`)。\n有趣的是将预算扩至 `52 token` (`4×4`) 时 `PRM beam` 略回落至 `0.830`, 这是因为带高斯噪声的 PRM 在候选变多时更容易把某条撞上正噪声尖峰的错步放进束内 (将判分噪声置零后立即升至 `0.970`)。',
        },
        {
          title: '串行预算控制 (Budget Forcing): 过早截断直接崩盘, 自查同样受限于认知盲区',
          body: '限制串行思考长度为固定预算 $B$ 时: 若 $B=2$ 连首遍 4 步推导都写不完就被腰斩, 准确率仅 `0.055`; 放宽至 $B=16$ 和 $B=64$ 后, 反复追加 `"Wait"` 触发自查重算能把无陷阱题推至 `1.000`、总分推至 `0.815`, 显著高于任其自然停笔 (`平均 6.5 token`, 准确率 `0.305`)。但由于模型自查同样看不破自己深信不疑的运算符陷阱, 陷阱子集仍停在 `0.431` (真实推理模型如 DeepSeek-R1 则需通过大规模强化学习内化更强的自我纠错策略)。',
        },
      ],
      links: [
        { from: 'm10 sample', to: 'sample_step', body: '每一步从 6 个离散算术动作候选的 logits 中直接调用 `m10` 标准温度采样函数生成下一步。' },
        { from: '奖励模型 (阶段 4)', to: 'orm / prm', body: '阶段 4 讲解如何从数据中训练出 ORM 与 PRM 标量头, 本章则在推理端对比它们引导搜索与选解的算力转化效率。' },
        { from: 'prm_beam_search', to: 'beam search (解码一章)', body: '搜索骨架完全一致, 区别仅在于排序依据从「模型自回归对数似然」换成了「外部过程奖励模型的累计得分」。' },
      ],
      sourceRows: [
        { concept: '粗心与陷阱建模', code: 'm23_test_time_compute/tts.py:step_dist', takeaway: '常规步令正确项 logit 最高 (`3.0 > 0.3`), 陷阱步令误读算符项反超 (`3.5 > 3.0`), 复现偶发失误与系统性偏见两类错误。' },
        { concept: 'Best-of-N', code: 'm23_test_time_compute/tts.py:best_of_n', takeaway: '独立采样 $N$ 条完整推理链, 交由带噪结果验证器 ORM 打分选出最高分答案。' },
        { concept: '自洽多数投票', code: 'm23_test_time_compute/tts.py:majority_vote', takeaway: '无需外部验证器直接对 $N$ 条链的最终答案做众数表决, 适用于答案可精确判等的客观题。' },
        { concept: 'PRM 束搜索', code: 'm23_test_time_compute/tts.py:prm_beam_search', takeaway: '每步对活跃束各采样 `expand` 个后继并按 PRM 累计分截断保留前 `width` 条, 提前止损错误分支。' },
        { concept: '强制延长思考', code: 'm23_test_time_compute/tts.py:think', takeaway: '模拟未达最小 token 预算前拦截结束意图、注入反思标记回溯重算的串行扩展机制。' },
        { concept: '自我校验边界', code: 'm23_test_time_compute/tts.py:believes_ok', takeaway: '刻画自我反思的固有局限: 能以一定概率揪出随机粗心错, 却对自身固有的系统性误解视而不见。' },
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
      subtitle: '单实例推理引擎抗不住高并发流量时需要横向部署多个独立副本, 而各副本内存中的基数树前缀缓存彼此并不共享。网关路由必须在「送去缓存最热的那台省重算」与「分流去空闲机器防排队」之间动态平衡。',
      tldr: '在 8 个各自维护独立 `RadixCache` 的服务副本前对比三种网关分发策略:\n- 轮询 / 最少积压路由: 完全无视请求文本内容, 多轮对话下一轮大概率换机器重算整段历史 (`40.7%` 词元未命中, 平均首字延迟 TTFT `0.469 s`)。\n- 纯前缀感知路由: 永远发往匹配前缀最长的副本, 命中率虽飙升至 `93.9%`, 却因热门系统提示词产生强磁吸效应把半数流量死死吸在一台机器上排长队, 导致平均 TTFT 雪崩恶化至 `105.954 s`!\n- 前缀优先 + 积压差阈值兜底: 当命中最佳副本比全场最闲副本多积压超过阈值 (`0.5 s`) 时主动让位分流, 既消除过载排队又顺水推舟把热点前缀自然复制到多台机器上, 取得全场最低 TTFT `0.284 s`。',
      question: '为什么把每条新请求都精准发给前缀缓存命中率最高的那台服务器, 集群整体命中率涨到了 94%, 用户的首字响应延迟却慢了两百多倍?',
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
          title: '无状态轮询打断多轮会话亲和性, 四成历史被迫反复重算',
          body: '多轮对话每开启新一轮都要把系统提示词与前序所有问答历史完整重传一遍, 而前几轮算好的历史 KV 只躺在上一次处理它的那个副本显存里。传统的轮询 (`round_robin`) 与纯最少积压 (`least_load`) 路由只看序号或队列长度、不看请求文本, 导致同一会话后续轮次在 8 个副本间随机漂移, 多达 `40.7%` 的输入词元每次都要从头做 prefill。',
        },
        {
          title: '纯前缀贪心引发热点磁吸雪崩: 命中率最高, 排队延迟却最惨',
          body: 'm27 模拟了共享 4 套热门系统提示词 (最热一套占全部流量的 `48%`) 的 1,084 条多轮对话请求。纯前缀感知路由每次挑能匹配最长前缀的副本: 第一条带最热系统提示词的请求落到某台副本缓存下这 512 个词后, 后续所有使用同一提示词的新老会话探测一圈发现只有这一台能命中 512 词、其余全为 0, 于是源源不断全往这一台机器上挤!\n结果 8 台机器里 5 台彻底空转、最忙一台独自吞下 `52%` 流量 (`max/mean = 4.18`), 省下的几毫秒 prefill 在百秒级的排队积压面前杯水车薪, 平均 TTFT 飙升至 `105.954 s`。',
        },
        {
          title: '积压差值阈值兜底: 超载主动分流, 顺带完成热点前缀多副本扩散',
          key: true,
          body: '解法只需在前缀路由后补一行负载差判断: 先找出前缀匹配最长的副本 `best` 与当前积压最少的副本 `least`, 若 `backlog[best] - backlog[least] > threshold_s`, 则果断放弃本次缓存命中改投 `least`。\n这一分流动作还产生了极佳的自愈副产物: `least` 副本接下并跑完这条请求后, 它的本地缓存里也顺手存下了那段热门系统提示词! 下次再来同类新对话时就有两台机器同时命中并按谁更空闲自动均摊, 热点前缀随流量自然扩散到多台副本。扫描阈值可见清晰的 U 形甜点: `4.0 s` 过松 (`TTFT 0.689 s`), `0.05 s` 过紧退化向最少负载 (`0.308 s`), 中间 `0.5 s` 兼顾 `77.9%` 命中率与 `1.12` 均衡度, 拿下最优平均 TTFT `0.284 s`。',
        },
        {
          title: '路由探测必须严格只读, 避免污染未选中副本的 LRU 淘汰序',
          body: '每到一条请求网关都要向全部 8 个副本询价匹配长度, 但最终只有 1 个副本真正承接该请求。直接调用 `m05` 的 `RadixCache.match_prefix` 会顺手切分树边并把沿途节点的 `last_used` 时间戳刷成当前时刻——若拿它去做全员探测, 陪跑的 7 台机器缓存里的旧节点也会被误标为「刚刚活跃」, 彻底搅乱真实的 LRU 淘汰顺序! 因此 `router.py` 专门实现了零副作用的只读探测函数 `prefix_len`, 仅在最终敲定目标副本后才对那一台执行真实匹配与写入。',
        },
      ],
      links: [
        { from: 'm05 RadixCache', to: 'Replica.cache', body: '每个服务副本内部直接实例化 `m05` 的真实基数树缓存 (容量 24K 词元), 精确复现跨请求前缀共享与叶子 LRU 淘汰行为。' },
        { from: 'match (会改树)', to: 'prefix_len (只读)', body: '分离探测期只读遍历与执行期有状态写入, 确保路由广播询价不干扰落选副本的缓存生命周期。' },
        { from: 'EPLB (MoE serving)', to: '负载阈值', body: '无论卡间专家并行还是实例间副本路由, 都面临 Zipf 热点吸干单点算力的共性挑战: EPLB 靠显式复制热门专家权重分流, 多副本路由则靠积压阈值把热门前缀自然复制到多个实例缓存中。' },
      ],
      sourceRows: [
        { concept: '无副作用探测', code: 'm27_multi_replica_routing/router.py:prefix_len', takeaway: '沿基数树只读统计最长公共前缀词元数, 不触发切边 (`_split`) 也不刷新 `last_used` 时间戳。' },
        { concept: '带阈值前缀路由', code: 'm27_multi_replica_routing/router.py:route', takeaway: '先按 `(匹配长度, -当前积压)` 选出首选副本, 若其相对最闲副本的超额积压越过 `threshold_s` 则回退至最闲副本。' },
        { concept: '多副本离散回放', code: 'm27_multi_replica_routing/router.py:simulate', takeaway: '推进请求到达时钟、消化各实例在途积压、执行路由与基数树真实更新并统计全局命中率与平均 TTFT。' },
        { concept: 'Zipf 多轮会话流', code: 'm27_multi_replica_routing/router.py:chat_stream', takeaway: '合成 200 段 3~8 轮对话流, 4 套系统提示词热度按 `1/rank` 倾斜分布, 真实还原线上冷热不均场景。' },
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
      "title": "有界等待队列与准入控制: 宁可快速拒单, 也不让全队排到超时",
      "body": "当外部请求到达速率超过服务最大处理能力时, 无限制放请求进等待队列并不会提高吞吐, 只会把「立即返回繁忙」变成「苦等良久后集体超时作废」。`m28` 在 4 活跃槽、单请求 1 秒截止期 (SLO 要求 `TTFT <= 200 ms, ITL <= 6 ms, 1s 内完成`) 下对比到达率 `48 req/s` 时的两种队列:\n- 有界队列 (最多排队 8 条, 超额立即拒绝): 拒绝 131 条, 每秒按质按时达标的 `goodput` 为 `16.5 req/s`。\n- 无界队列 (来者不拒): 队列一路堆积到 52 条, 93 条超时报废, 真正满足延迟目标的 `goodput` 暴跌十倍仅剩 `1.55 req/s`!\n同时无论请求最终落入完成、拒绝、超时还是客户端主动取消哪一种终态, 块管理器都必须确保其占用的物理 KV 页悉数归还 (模拟结束时残留泄漏页恒为 `0`)。",
      "source": "llm_infer/m28_service_capacity/demo.py:simulate"
    }
  ],
  "infer-multi-replica-routing": [
    {
      "title": "服务端指标防坑: 脱离失败率看 p99 与脱离延迟看平均吞吐都会失真",
      "body": "评估线上推理容量时单报一个平均吞吐或单报一个完成请求的 `p99` 极易产生误导。在 `m28` 到达率 `48 req/s` 的反例中, 无界队列每秒完成 `34.7` 条看似略高于有界队列的 `34.2` 条, 但它的首字延迟 `TTFT p99` 高达 `955 ms`, 真正达标的 `goodput` 只有 `1.55` 对 `16.5`。\n阅读延迟分位数时还必须警惕幸存者偏差: `p95 / p99` 仅在成功跑完的请求子集上统计, 被准入拒绝、半途超时或取消的请求并不在内——如果一个服务靠拒掉六成慢请求换来漂亮的 `p99`, 单看延迟分位数就会得出完全相反的结论; 正确做法是将到达流量 (`offered load`)、四种终态计数与达标好吞吐 (`goodput`) 并排呈现。",
      "source": "llm_infer/m28_service_capacity/demo.py:experiment"
    }
  ],
  "infer-engine": [
    {
      "title": "高负载终态对账: 区分原始 token 吞吐、准入量与达标 Goodput",
      "body": "在单机引擎内部核对完 KV token 守恒账之后, 推上高压流量还要核对第二本请求终态账。`m28` 在 `96 req/s` 过载流量下验证有界队列: 总到达 `956 = 完成 364 + 拒绝 568 + 超时 0 + 取消 24`, 四类互斥终态严格加总等于总到达数且无一页 KV 泄漏; 同流量若换用无界队列则 `TTFT p99` 冲至 `960 ms`、超时高达 `693` 条。把「每秒吐多少 token」、「收进门多少请求」和「在 SLO 延迟线内交付多少请求 (goodput)」拆开核算, 才能看清引擎在真实生产水位下的有效容量。",
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
topic.pages["infer-compute"].points.push({"title": "算子融合实测: 逻辑 FLOPs 完全相同, 算子派发与临时内存翻倍", "body": "理论浮点运算次数相同绝不代表实际运行开销一致。`m32` 用 PyTorch 对比计算同一线性变换的两种写法——分步 `x @ w + b` 与融合算子 `torch.addmm`:\n两者的逻辑乘加次数同为 `1,056,768` FLOPs, 但开启独立 profiler 追踪底层事件可见: 分步写法触发了 7 次底层 ATen 算子调用并分配了 `65,536` 字节内存 (为未加偏置前的矩阵乘结果额外开辟了一份临时张量), 而融合写法仅触发 6 次调用、分配 `32,768` 字节内存。基准测试流程严格执行「先预热 warmup → 独立多次计时 → 最后单独采集一次 profiler」, 避免性能追踪器自身的插桩开销污染耗时读数。"})
topic.pages["infer-scheduler"].related = [{"route": "infer-multi-replica-routing", "label": "到达率上升后, 排队会发生什么", "reason": "本章只说明等待队列要设上限。请求越来越多时, 分位数和 goodput 该怎么读、为什么不能只看平均吞吐, 在多副本路由那一章。"}]
topic.pages["infer-engine"].related = [{"route": "infer-multi-replica-routing", "label": "到达率上升后, 排队会发生什么", "reason": "本章只要求每个请求的终态对上账。指标该怎么读 (分位数为什么只算完成的请求、goodput 怎么定义), 在那一章。"}, {"route": "finetune-runs", "label": "模型交付与独立进程加载", "reason": "引擎要加载的不只是权重文件, 还有词表、模板、生成参数这些约定, 都记在交付清单 manifest 里。那一章演示新进程怎么按清单加载, 错配时怎么拒绝。"}]
topic.pages["infer-decode-control"].related = [{"route": "basic-optim-sample", "label": "复习 softmax 温度", "reason": "同一个温度公式: 基础页演示它怎么改变分布, 本章讲它在采样里怎么用。"}]
topic.pages["infer-kv-memory"].related = [{"route": "attention", "label": "复习注意力掩码", "reason": "掩码 (mask) 决定每个 token 能看见哪些 token。本章只讲 K/V 怎么存; 拖滑杆看掩码怎么变的实验在注意力那一章。"}]
export default topic

topic.pages["infer-scheduler"].points.push({"title": "多租户公平调度: 全局高吞吐可能掩盖单租户饥饿", "body": "多个业务租户共用同一套推理集群时, 单看全局总完成数往往发现不了低频短请求租户正在被大户饿死。`m33` 模拟租户 A 一口气灌入 60 个长请求 (每条占 5 tick) 而租户 B 每隔 15 tick 才发 1 个短请求 (每条占 2 tick, 共 4 个, 40 tick 超时):\n- 先到先服务 (FIFO) 或 A 高优: 全局看似完成了 9 条请求, 但租户 B 的短请求被 A 的长龙死死堵在门外, 完成数仅 `1/4` 甚至 `0/4`。\n- 等成本量子公平轮转 (`Fair Round-Robin`): 两个活跃租户每次轮流获取 1 tick 服务时间片, 未跑完的长请求退回本租户队首等待下一轮, 使租户 B 的 4 条短请求实现零首发等待、`4/4` 全部按时完成 (A 完成数仅从 9 微降至 8); 再叠加上单租户排队配额即可同时防住单一大户撑爆全局内存队列。"})
topic.pages["infer-scheduler"].source.push("llm_infer/m33_tenant_scheduling/demo.py:experiment")

topic.pages["infer-prefix-radix"].points.push({"title": "前缀缓存的隔离边界: 缓存键必须绑定可信租户域与模型/Adapter 版本", "body": "两段请求哪怕词元序列一字不差, 它们的 KV 缓存也绝不能无条件跨请求复用——KV 的真实语义同时绑定了「谁有权限访问」与「由哪一版模型及 LoRA 算子算出」。`m34` 完整验证了两类跨界错误:\n- 跨租户侧信道泄露: 若缓存键未包含由可信网关根据凭证签发的租户隔离域 (`HMAC scope`), 租户 B 发送试探前缀即可命中租户 A 留下的缓存, 并通过首字响应显著变快推断出其他租户刚刚问过什么敏感内容 (且隔离域必须由服务端鉴权生成, 绝不能采信客户端自报的租户声明)。\n- 跨 Adapter / 模型版本静默串味: 同一基座切换了不同版本的 LoRA adapter 后若缓存键漏带 adapter 版本标识, 引擎会静默返回旧 adapter 算出的错误 KV 且不报任何异常。将可信租户命名空间、模型摘要与 adapter 身份一并纳入缓存键后方可安全启用前缀复用。"})
topic.pages["infer-prefix-radix"].source.push("llm_infer/m34_cache_identity/demo.py:experiment")

canonicalizeRelated(topic)
