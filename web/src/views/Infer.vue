<template>
  <div>
    <h1 class="page-title">推理与部署优化 · 从能生成到高吞吐服务</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_infer/" label="llm_infer/" tiny /> 聚焦于将训练完毕的自回归模型部署为线上推理服务。模型依然按 token 逐步生成, 但核心工程目标从「收敛训练损失」转向了「压低首 token 延迟、提升并发吞吐、控制显存峰值与约束输出结构」。
    </p>

    <ChapterIntro
      :tldr="tldr"
      question="为什么训练阶段的主要开销在反向传播, 而线上推理的瓶颈却往往落在 KV cache 容量、动态调度与显存带宽上?"
      :goals="[
        '厘清 KV cache、分页管理与前缀缓存各自消除的冗余开销',
        '理解连续批处理与分块 prefill 提升 GPU 利用率的机制及其延迟权衡',
        '将本仓库 mini-vLLM 的服务主循环与真实 vLLM / SGLang 架构逐一对应',
      ]"
      :codes="[
        { path: 'llm_infer/core/' },
        { path: 'llm_infer/full_engine/' },
        { path: 'llm_infer/m01_kv_cache/' },
        { path: 'llm_infer/m03_continuous_batching/' },
      ]"
      :prereq="prevChapter"
      :next-step="{ name: 'infer-kv-memory', label: '下一章 · KV 与缓存内存' }"
    />

    <StageContents stage-id="infer" />

    <section class="section">
      <h2>1. 推理优化的递进关系</h2>
      <div class="lead-group">
        <ul class="pts">
          <li><b>消除历史重算:</b> 将每步重新前向整段上下文的朴素解码, 改为仅计算当前新 token 并从 KV cache 读取历史键值。</li>
          <li><b>消除显存碎片:</b> 将连续预留的 KV cache 显存池切分为固定大小的物理块, 按需动态分配并在请求结束后立即回收。</li>
          <li><b>消除批内空转:</b> 从静态批处理改为逐步重组的连续批处理 (Continuous Batching), 允许新请求随时插入、已完成请求即时退出。</li>
        </ul>
        <p>后续的前缀缓存复用、投机解码、权重与 KV 量化以及结构化解码约束, 均挂载在这条核心服务主循环之上。</p>
      </div>
      <EvolutionChain
        title="从朴素 generate 到 mini-vLLM"
        subtitle="每次演进均保持模型数学语义不变: 精确路径下输出 logits 一致, 仅重构计算与显存调度方式。"
        :steps="inferChain"
      />
    </section>

    <section class="section">
      <h2>2. 第一性瓶颈 · 逐步重算与 KV cache</h2>
      <div class="lead-group">
        <ul class="pts">
          <li><b>无缓存路径:</b> 第 t 步解码时需要将原始 prompt 与已生成的 t 个 token 拼接后整段重新执行 prefill。</li>
          <li><b>KV cache 路径:</b> 输入阶段仅执行一次 prefill, 后续 decode 阶段每步只为单个新 token 计算并追加键值向量。</li>
        </ul>
        <p>在 m01 的基准演示中, 累计流经模型前向的 token 总数从 688 降至 37 (压缩 18.6×), 且生成的输出 token 序列逐位置完全一致。</p>
      </div>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>朴素路径 <span class="tag">重复 prefill</span></h3>
          <pre class="code">{{ noCacheCode }}</pre>
          <p class="hint">
            单步前向开销随已生成序列长度线性增长, 累计复杂度呈平方级上升。演示脚本会将该路径与缓存路径的输出序列逐项比对, 验证优化前后生成结果严格等价。
          </p>
        </div>
        <div class="card">
          <h3>增量路径 <span class="tag">KV cache</span></h3>
          <pre class="code">{{ cacheCode }}</pre>
          <p class="hint">
            首步 prefill 缓存各层的历史 K/V 张量, 此后每次调用 <code class="inline">decode_step</code> 仅输入单个新 token。
            该机制省去了历史 token 的线性投影与前馈网络重算, 但当前步的新 query 仍需与全部 t 个历史 key 计算注意力内积。
            对应实现见 <RepoLink path="llm_infer/m01_kv_cache/demo.py" label="llm_infer/m01_kv_cache/demo.py" tiny />。
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 服务端核心原语 · 分页内存、连续调度与前缀复用</h2>
      <div class="lead-group">
        <p>基础 KV cache 仅能加速单条请求的解码过程。进入多并发线上服务后, 引擎还必须同时应对请求输出长度未知导致的显存碎片、长短请求交错到达的吞吐波动, 以及大量请求共享相同 system prompt 带来的重复计算。</p>
        <p>以下三个核心组件共同构成了 vLLM 与 SGLang 等现代推理引擎的系统骨架。</p>
      </div>
      <div class="grid grid-3">
        <div v-for="p in servingPrimitives" :key="p.name" class="card primitive-card">
          <h3>{{ p.name }} <span class="tag">{{ p.tag }}</span></h3>
          <p class="desc">{{ p.desc }}</p>
          <pre class="code">{{ p.code }}</pre>
          <p class="hint"><CodeRef :value="p.file" base="llm_infer/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 算子加速、模型压缩与解码约束</h2>
      <div class="lead-group">
        <p>在分页缓存与连续调度搭起的服务骨架之上, 其余子模块分别围绕计算加速、访存压缩、采样控制与语法约束展开。工程组合时需严格区分两类正确性边界:</p>
        <ul class="pts">
          <li><b>数学等价优化 (贪心或同随机源下输出逐 token 一致):</b> 分页 KV 管理、连续批调度、前缀缓存复用与投机解码。</li>
          <li><b>有损近似压缩 (需结合任务监控精度退化):</b> 权重与 KV 量化、Attention Sink 滑窗截断以及稀疏注意力。</li>
        </ul>
      </div>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="infer-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>优化对象</th>
              <th>和主线的关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in inferModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. full_engine · 把模块接成服务主循环</h2>
      <div class="lead-group">
        <p>
          <RepoLink path="llm_infer/full_engine/engine.py" label="full_engine/engine.py" tiny /> 虽不是完整的工业级 vLLM, 但把连续批调度 (m03)、真分页 KV 池 (m02)、链式前缀复用 (m04)、分块 prefill (m03 的开关, 原理见 m06) 以及抢占与采样 (m10) 串在了同一条服务主循环上。
        </p>
        <p>哪怕在只有 9 个物理块的小池子里触发了 4 次抢占重算, 它的贪心输出依然与朴素单条生成逐 token 完全一致。</p>
      </div>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>Engine.step <span class="tag">混合 batch</span></h3>
          <pre class="code">{{ engineStepCode }}</pre>
          <p class="hint">
            引擎内部不再区分独立的「prefill 步」与「decode 步」: 同一批次内可同时混跑长度大于 1 的 prefill 分块与单 token 的 decode 请求, 仅当序列将输入 prompt 全部前向完毕后才对末尾 logits 执行采样。
          </p>
        </div>
        <div class="card">
          <h3>接纳新请求时发生什么 <span class="tag">资源账本</span></h3>
          <pre class="code">{{ prefillCode }}</pre>
          <p class="hint">
            命中前缀缓存的 token 直接跳过前向传播, 其历史键值已驻留在全局分页池中, 只需通过页表映射读取。在演示脚本的资源统计中, 305 个待处理 token 精确拆分为 233 个实际前向 token 与 72 个前缀命中 token。
          </p>
        </div>
      </div>
      <div class="card compare-card">
        <table class="infer-table">
          <thead>
            <tr>
              <th>真实服务概念</th>
              <th>本仓库对应实现</th>
              <th>读代码时抓住什么</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in engineMap" :key="r.concept">
              <td class="axis">{{ r.concept }}</td>
              <td class="mono small"><CodeRef :value="r.file" base="llm_infer/" tiny /></td>
              <td>{{ r.focus }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '回顾微调与偏好对齐如何产出待部署的模型与适配器' }"
      :next="{ name: 'infer-kv-memory', label: '下一章 · KV 与缓存内存', hint: '从消除历史重算与分页显存池开始拆解推理优化' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { inferModules, learningPath } from '@/data/models.js'

// 上一章从 learningPath 取, 不手写编号 (别的阶段加章后手写的 "4.4" 会过期)
const prevItem = learningPath[learningPath.findIndex((x) => x.route === 'infer') - 1]
const prevChapter = { name: prevItem.route, label: `上一章 · ${prevItem.label}` }

// 模块数从数据里数, 不写死 (inferModules 里除了 mNN 还有一行 full)
const moduleCount = inferModules.filter((m) => /^m\d+$/.test(m.id)).length
const tldr = `线上推理分为两段截然不同的计算负载: prefill 一次并行算完输入 prompt, 受算力上限约束; decode 则逐个生成新 token, 每步都要把模型权重与历史 KV cache 从显存完整搬运一遍, 瓶颈主要卡在显存带宽与容量。
本阶段 ${moduleCount} 个模块围绕这条主循环层层展开:
- 管住 KV 显存与并发调度: 用 KV cache 省掉历史重算, 用 PagedAttention 分页消除显存碎片, 用 Prefix Cache 复用公共前缀, 再配合 Continuous Batching 与 Chunked Prefill 喂饱 GPU。
- 压低单步延迟与访存字节: 通过投机解码一次验证多个草稿 token, 借助权重量化、KV 量化、FlashAttention 与 CUDA Graph 减少读写与启动开销。
- 控制输出质量与多租户扩展: 涵盖采样策略、JSON/CFG 语法约束、测试时搜索, 以及张量并行、多 LoRA 和多副本路由。`

const inferChain = [
  {
    name: 'No Cache',
    year: 'baseline',
    pain: '每步解码都重新前向整段已知前缀, 单步开销随生成长度线性增加。',
    fix: '作为 m01 基准对照路径, 用于逐 token 校验后续缓存优化的数值等价性。',
    color: 'var(--text-dim)',
  },
  {
    name: 'KV Cache',
    year: 'm01',
    pain: '注意力层中历史 token 的 K/V 投影与前馈激活在每步解码中被重复计算。',
    fix: 'prefill 阶段缓存各层 K/V, decode_step 每步仅计算并追加单个新 token。',
    color: 'var(--left)',
  },
  {
    name: 'Paged KV',
    year: 'm02',
    pain: '输出长度事先未知, 按最大长度连续预留 KV 显存会造成内部浪费与外部碎片。',
    fix: '将 KV 显存池切分为固定大小的物理 block, 每条请求通过页表 block_table 按需映射物理块。',
    color: 'var(--accent)',
  },
  {
    name: 'Scheduler',
    year: 'm03',
    pain: '并发请求到达时刻与生成长度各异, 静态批处理需等待最长序列结束而浪费算力。',
    fix: '按步动态重组执行批次, 支持分块 prefill 与单步 decode 混跑, 显存不足时触发抢占。',
    color: 'var(--eye)',
  },
  {
    name: 'Reuse',
    year: 'm04-m06',
    pain: '多条请求共享的 system prompt 被反复计算, 且超长输入易独占单步执行时间。',
    fix: '通过链式哈希或 Radix 树跨请求复用前缀 KV 块, 并以 chunked prefill 摊平长输入延迟。',
    color: 'var(--right)',
  },
  {
    name: 'Engine',
    year: 'full',
    pain: '孤立的缓存、调度与采样算子尚不能直接对外提供并发流式推理能力。',
    fix: '在 Engine 中串联 add_request、step 与 generate, 闭环构成 mini-vLLM 服务主循环。',
    color: 'var(--warn)',
  },
]

const noCacheCode = `ids = list(prompt_ids)
for _ in range(max_new):
    logits, _ = lm.prefill(np.array(ids))
    next_id = argmax(logits[-1])
    ids.append(next_id)`

const cacheCode = `logits, kv_cache = lm.prefill(prompt_ids)
next_id = argmax(logits[-1])
ids.append(next_id)

for _ in range(max_new - 1):
    logits, kv_cache = lm.decode_step(next_id, kv_cache)
    next_id = argmax(logits)
    ids.append(next_id)`

const servingPrimitives = [
  {
    name: 'BlockManager',
    tag: 'PagedAttention',
    desc: '将全局 KV 显存池划分为固定大小的物理 block, 为每条序列维护独立的逻辑到物理块映射表 (block_table), 支持按需分配与跨请求引用计数共享。',
    file: 'llm_infer/m02_paged_attention/block_manager.py',
    code: `for blk in shared:          # 前缀命中的 block
    share_block(blk)        # ref_count += 1
n_new = blocks_needed(n_tokens) - len(shared)
table = list(shared) + [pop_free() for _ in range(n_new)]
block_tables[seq_id] = table`,
  },
  {
    name: 'Scheduler',
    tag: 'continuous',
    desc: '每步动态重组形如 [(seq, n)] 的混合执行批次: 优先为运行中序列分配单步 decode 槽位, 剩余 token 预算切分给等待队列做分块 prefill; 物理块不足时触发抢占回收。',
    file: 'llm_infer/m03_continuous_batching/scheduler.py',
    code: `budget = max_batch_tokens
batch = schedule_running(budget)   # decode: n = 1
budget -= sum(n for _, n in batch)
if not just_preempted:
    batch += admit(budget)         # prefill chunk
return batch                       # [(seq, n)]`,
  },
  {
    name: 'PrefixCache',
    tag: 'reuse',
    desc: '对每个写满的逻辑块结合前驱块摘要计算链式哈希。新请求到达时按块匹配前缀哈希, 命中的公共前缀直接复用全局池中的物理 block 并递增引用计数, 免去重复 prefill。',
    file: 'llm_infer/m04_prefix_cache/prefix_cache.py',
    code: `for i in range((len(ids) - 1) // bs):   # 至少留 1 个 token 真算
    parent = sha1(parent + block_i)     # 链式 hash
    blk = hash_to_block.get(parent)
    if blk is None: break
    hits.append(blk)`,
  },
]

const engineStepCode = `def step(self):
    batch = self.scheduler.schedule()   # [(seq, n)]
    tokens = []
    for seq, n in batch:
        end = seq.num_computed + n
        logits = self.runner.run(
            seq.all_ids[:end], table(seq), seq.num_computed)
        done = end == seq.num_tokens    # 追平了才采样
        tokens.append(sample(logits) if done else None)
    return self.scheduler.postprocess(batch, tokens)`

const prefillCode = `# Scheduler._admit: 接纳新请求
hits, n_hit = prefix_cache.match_prefix(ids)
n = min(len(ids) - n_hit, budget)   # 只吃一个 chunk
if not bm.can_allocate(len(ids), hits):
    break
bm.allocate(seq_id, len(ids), hits) # 命中块直接进页表
seq.num_computed = n_hit            # 命中的不用前向
batch.append((seq, n))`

const engineMap = [
  { concept: '请求接入入口', file: 'Engine.add_request', focus: '将输入 prompt 编码后排入 waiting 队列, 并为每条请求绑定独立的 SamplingParams' },
  { concept: '逐步动态组批', file: 'm03_continuous_batching/scheduler.py', focus: 'schedule() 返回 [(seq, n)] 混合批次: 优先保障运行中请求的 decode, 剩余 token 预算分配给 prefill chunk, 物理块不足时执行重算式抢占' },
  { concept: '真分页前向执行', file: 'full_engine/model_runner.py', focus: 'run() 仅前向计算增量区间 ids[start_pos:], 历史上下文 KV 通过 block_table 从全局物理池按页读取' },
  { concept: '物理块引用账本', file: 'BlockManager', focus: '通过 allocate(shared=...)、ensure_capacity、free 与 share_block 维护引用计数, 空闲队列顺序直接对应 LRU 淘汰顺序' },
  { concept: '公共前缀复用', file: 'PrefixCache', focus: '对写满的块计算链式哈希, 命中的前缀块直接共享物理引用, 未命中后缀继续分配新块并登记哈希' },
  { concept: '末尾采样管线', file: 'm10_sampling/samplers.py', focus: '在末尾 logits 上依次施加重复惩罚 (repetition penalty)、温度缩放以及 top-k / top-p / min-p 截断过滤' },
]
</script>

<style scoped>
table.infer-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.infer-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.infer-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.infer-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.infer-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.primitive-card pre.code {
  min-height: 142px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.compare-card {
  margin-top: 16px;
  padding: 0;
  overflow-x: auto;
}
</style>
