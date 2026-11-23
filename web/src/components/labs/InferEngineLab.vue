<!--
  mini-vLLM 引擎主循环实验台 (llm_infer/full_engine)。
  一步永远是同四件事: 调度 → 前向 → 采样 → 后处理。难的只有调度那一步:
  它要同时管住 KV 显存 (block 够不够)、公平 (谁先进) 和延迟 (每步算几个 token)。
  调度和 block 记账与 m03 Scheduler / m02 BlockManager / m04 PrefixCache 逐步同构, demo 打印的数字在这里调得出来。
  带 ⚠ 的开关演示一类真实存在的活锁: 队首拿不到 block 时如果不落到 decode, 整条时间线就冻住。
-->
<template>
  <LabFrame
    title="引擎主循环 — 三条队列、一个 block 池、一个调度分支, 看调度器每一步怎么选"
    sub="- 模拟什么: 一个微型 vLLM 推理引擎的完整事件主循环。场景包含 5 条共享同一段 30 token 系统提示词 (system prompt) 的用户请求; 开启前缀缓存时, 后续请求可直接命中并复用前 3 个物理 KV 块而无需重复计算。注: 单步耗时沿用线性代价模型估算 (每步 20 ms 固定开销加上 0.25 ms 乘以本步处理的 token 数), 并非硬件实测。引擎主循环在每个迭代步固定依次执行四个阶段:
  1. 调度决策: 决定当前步将哪些序列选入批次以及每条序列推进几个 token。这是引擎最核心的控制环节, 需同时协调物理 KV 显存管理 (将 KV cache 切分为每块容纳 8 个 token 的定长物理 block 放入全局共享池, 当池内空闲块不足以支撑活跃序列继续增长时必须触发抢占, 即驱逐一条活跃请求退回等待队列)、先来先服务公平性以及单步最大 token 预算限制。
  2. 模型前向计算。
  3. 输出 token 采样。
  4. 状态后处理: 检测已生成完毕的序列并将其占用的独占 KV block 归还至空闲池。
- 图上是什么: 面板自上而下分为三个视图区:
  - 上方三行状态队列: waiting 队列存放尚未准入或被抢占退回的请求; running 队列存放当前驻留显存的活跃序列, 卡片上的「已算/总长」标明当前已缓存 KV 的 token 数与序列目标当前总长, 带彩色底纹表示该序列在本步获得了执行配额; finished 队列存放已完成的序列并标注其累计输出长度。
  - 中间状态栏: 显示本步调度器命中的执行分支 (decode、prefill、混批、抢占或空转) 及其触发原因。
  - 下方物理块网格: 展示全局 KV block 池的实时分配状态, 每格代表 1 个物理块。彩色格标注当前独占该块的请求编号; 标有 ∷ 的方格代表被多条请求共同引用的共享前缀块; 标有 · 的方格代表引用计数已归零但哈希内容仍保留在缓存池中的可复用冷块 (后续相同前缀到达时仍可直接激活命中); 虚线空框代表纯空闲块。
- 可以做什么: 顶排提供三种调度模式按钮, 下方配有特性开关、容量滑杆与步进播放器:
  - 点击「decode 优先 + 分块 prefill」: 每步优先为 running 队列中每条活跃序列各分配 1 个 decode 配额, 再用剩余预算切分准入新请求的 prefill。
  - 点击「prefill 优先」: 优先尝试为 waiting 队首的新请求分配整段 prefill, 仅当无新请求可准入时才回退执行活跃队列的 decode。
  - 点击带 ⚠ 标记的缺陷模式开关: 载入一版故意省去回退分支的错误调度逻辑 (当 waiting 队首因缺块无法准入时直接返回空批次), 用于复现系统活锁现象 (进程未崩溃但所有请求永久停滞)。
  - 切换「前缀缓存 开/关」: 控制多条请求间的公共 system prompt 是否共享物理块并跳过重复计算。
  - 拖动「KV block 池」滑杆: 调节全局物理块总容量 (26 块为分块模式无抢占运行的临界下界, 低于该值将触发显存抢占)。
  - 拖动「max_batch_tokens」与「到达速率」滑杆: 调节单步最大计算配额与外部请求到达密度。
  - 使用播放器逐步单步追踪; 点击队列中的 R 卡片可高亮查看该请求占用的物理块分布及端到端延迟。
- 应该看到: 在默认配置下, 5 条请求平滑交错执行分块 prefill 与 decode, 块池中稳定驻留标有 ∷ 的公共前缀块, 全程共 34 步完成所有请求; 逐步调小块池容量后系统开始触发「抢占」, 最晚到达的活跃请求被驱逐回 waiting 队首并释放其私有块, 待后续重获准入时需重新计算丢失的上下文; 若开启 ⚠ 缺陷开关后再缩减块池, 时间线将在某一步之后陷入永久「空转」。右侧四项读数含义如下:
  - 吞吐: 有效输出的总 token 数除以总耗时 (陷入活锁时跌为 0)。
  - 最大 TBT: 所有请求生命周期内相邻两次生成之间的最长等待间隔。
  - 抢占次数: 发生显存驱逐重排的累计次数, 大于 0 时标红预警。
  - 实算 + 命中: 实际送入前向计算的 token 总数与前缀缓存命中跳过的 token 总数之和; 无抢占时二者之和严格等于「需要」(全部请求所需的总 KV 长度), 超出部分即为被抢占序列退回重入时重复消耗的算力开销。"
    module="llm_infer/full_engine"
    run="python -m llm_infer.full_engine.demo"
    :challenge="{
      ask: '开启带 ⚠ 标记的缺陷调度开关 (即 waiting 队首无法分配足够 block 时不回退执行 decode), 其余参数保持默认, 随后将 KV block 池容量从 26 块起逐格下调: 容量降至几块时引擎时间线会彻底停滞?',
      answer: '降至 22 块时陷入停滞。\n维持 4 条并发序列完整跑满全程共需 $3 + 4 \\times 5 = 23$ 个物理块: 其中 3 块存放多请求共享的 30 token 系统提示词前缀 (每块容纳 8 个 token, 前 3 块哈希内容完全一致因而跨请求共用), 每条请求各自的独立提问加上 20 个生成输出 token 额外独占 5 块。当池容量缩减至 22 块 (仅差 1 块) 时, 引擎首先驱逐最晚进入的活跃序列 R3 触发抢占, 随后立即陷入闭环死结:\n1. 被抢占的 R3 退回 waiting 队列头部等待重新准入, 但池中剩余空闲块不足以支撑其完成重入分配; 缺陷版 prefill 分支准入失败后未做回退, 直接向主循环返回空 batch。\n2. running 队列中仍在驻留的其余活跃序列因得不到任何前向步配额而停止推进, 无法生成至终止符。\n3. 没有任何活跃序列结束退出, 池中已被占用的物理 block 便永远无法释放归还。\n4. 物理块不释放, 阻塞在 waiting 队首的 R3 便永远无法获批进入。\n这一进程存活但吞吐归零的互锁状态即系统活锁 (livelock)。\n修复方案仅需补齐一行回退逻辑: 当新请求或重入请求准入失败时立即回退调度活跃队列的 decode, 对应 Python 引擎代码中 _admit(budget) or _schedule_running(budget) 的短路或分支。该回退机制确保驻留时间最早的活跃序列始终能够向前推进直至完结并释放物理块, 从而打破资源互锁。\n相比之下, 单纯达到最大并发条数上限 (例如 max_batch_seqs 设为 4) 并不会引发活锁: 此时池内物理块依然充裕, running 队列正常推进 decode, 只要任一序列生成完毕腾出并发名额, 队首请求即可顺利准入。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: chunked }" @click="setMode('chunk')">decode 优先 + 分块 prefill</button>
        <button type="button" :class="{ active: !chunked && !livelock }" @click="setMode('prefill')">prefill 优先 (进不来就落到 decode)</button>
        <button type="button" class="danger-btn" :class="{ active: livelock }" @click="setMode('bug')">⚠ 队首拿不到 block 时不落到 decode</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: prefixOn }" @click="prefixOn = !prefixOn">
          前缀缓存 {{ prefixOn ? '开' : '关' }}
        </button>
        <span class="tip">5 条请求共享同一段 30 token 的 system prompt; R4 第 14 个 token 采到 EOS, 其余生成满 20 个</span>
      </div>
      <LabSlider v-model="pool" label="KV block 池" :min="MIN_POOL" :max="64" unit=" 块" />
      <LabSlider v-model="budget" label="max_batch_tokens" :min="8" :max="96" :step="8" />
      <LabSlider v-model="rate" label="到达速率" :min="0.2" :max="5" :step="0.2" unit=" 条/步" />
      <StepPlayer :stepper="stepper" :label="`step ${f.t + 1} / ${sim.frames.length}`" />
    </template>

    <div class="engine">
      <div v-for="q in queues" :key="q.name" class="qrow">
        <span class="qlab mono">{{ q.name }}</span>
        <button
          v-for="c in q.items" :key="c.id" type="button" class="chip"
          :class="{ active: sel === c.id, dimmed: sel >= 0 && sel !== c.id }"
          :style="{ borderColor: color(c.id), background: c.n ? fill(c.id) : 'transparent' }"
          @click="sel = sel === c.id ? -1 : c.id"
        >R{{ c.id }}<em v-if="c.txt">{{ c.txt }}</em></button>
        <span v-if="!q.items.length" class="tip">—</span>
      </div>

      <p class="branch">
        <b class="tag" :class="branchCls">{{ f.branch }}</b>
        <span v-if="sim.dead && stepper.step.value >= sim.frames.length - 1" class="tag dead">活锁</span>
        {{ f.why }}
      </p>

      <div class="cells" :style="{ gridTemplateColumns: `repeat(${Math.min(pool, 20)}, 20px)` }">
        <span
          v-for="(c, i) in f.cells" :key="i" class="cell"
          :class="{ dim: sel >= 0 && !c.ids.includes(sel), free: !c.ids.length, shared: c.ids.length > 1 }"
          :style="c.ids.length === 1 ? { background: fill(c.ids[0]), borderColor: color(c.ids[0]) } : null"
          :title="c.title"
        >{{ c.txt }}</span>
      </div>
      <p class="lab-note info">
        <template v-if="selSeq">
          R{{ selSeq.id }}: prompt {{ selSeq.prompt }} token, 第 {{ selSeq.arrive + 1 }} 步到达 ·
          TTFT {{ selSeq.ttft === null ? '未出首 token' : selSeq.ttft.toFixed(0) + ' ms' }} ·
          最大 TBT {{ selSeq.tbt ? selSeq.tbt.toFixed(0) + ' ms' : '—' }} ·
          已出 {{ selSeq.out }}/{{ OUT[selSeq.id] }} token
        </template>
        <template v-else>
          点队列里的 R 按钮, 看那条请求占了哪些 block、它的 TTFT (首 token 延迟) 和最大 TBT (相邻两个 token 之间最长的等待) 是多少。
          ∷ 是几条请求共享的前缀 block; · 是已释放但内容还在的 block, 同样的前缀再来仍能命中。
        </template>
      </p>
    </div>

    <template #stats>
      <div class="kv"><span>吞吐 (输出 token 数 ÷ 总耗时, tok/s)</span><b :class="{ bad: sim.dead }">{{ sim.thr.toFixed(1) }}</b></div>
      <div class="kv">
        <span>最大 TBT (相邻 token 最长等待)</span>
        <b v-if="!sim.maxTbt">—</b>
        <b v-else :class="sim.maxTbt > 120 ? 'bad' : 'good'">{{ sim.maxTbt.toFixed(0) }} ms</b>
      </div>
      <div class="kv"><span>抢占次数</span><b :class="sim.preempts ? 'bad' : 'good'">{{ sim.preempts }}</b></div>
      <div class="kv">
        <span>实算 + 前缀命中的 token (需要 {{ sim.need }})</span>
        <b :class="sim.comp + sim.hit === sim.need ? 'good' : 'bad'">{{ sim.comp }} + {{ sim.hit }}</b>
      </div>
      <p v-if="sim.dead" class="lab-note">活锁判定: 调度器连续 6 步返回空批次且系统无任何外部状态变更可能——物理 block 仅在活跃序列完结时释放, 而此时没有任何活跃序列能够获得计算配额向前推进。在真实线上服务中, 该故障表现为进程存活、无报错退出, 但所有客户端请求均无限期挂起。</p>
      <p v-else-if="sim.comp + sim.hit !== sim.need" class="lab-note">「实算 + 命中」超出「需要」总量的差额, 源于被抢占序列后续重获准入时对整段历史序列的二次遍历: 其中已被覆盖驱逐的块需重新执行前向计算, 尚未被覆写的残留冷块则直接缓存命中。抢占机制不会丢失已生成的输出 token 文本, 损耗的仅是重算耗费的冗余算力。</p>
      <div class="lab-note">
        <p>单步延迟估算沿用「分块预填充」一节的线性代价模型: 每步固定开销 20 ms 加上 0.25 ms 乘以本步处理的 token 总数, 并非硬件实测读数。</p>
        <p>对照本节 demo 脚本基准配置 (单步预算 48, 到达速率不低于 2, 开启前缀缓存):</p>
        <ul class="pts">
          <li>默认的分块模式, 池 26 到 64 块: 305 个待算 token = 233 实算 + 72 命中, 共 34 步。</li>
          <li>池拖到 9 块: prefill 优先 82 步, 分块 81 步, 都抢占 4 次。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const BS = 8                       // block_size, 与 demo 一致
const SYS = 30                     // 共享 system prompt 的 token 数
const MAX_SEQS = 4                 // max_batch_seqs
const QLEN = [15, 12, 14, 13, 12]  // 每条请求各自的问句长度
const OUT = [20, 20, 20, 20, 14]   // demo 实测的输出长度: max_new = 20, R4 提前采到 EOS
const FIXED_MS = 20, PER_TOK_MS = 0.25
const blocks = (n) => Math.ceil(n / BS)
// 池至少装得下最长那条请求跑满 (9 块); 再小 Python 的 add_request 直接 assert
const MIN_POOL = blocks(SYS + Math.max(...QLEN) + Math.max(...OUT))
const PAL = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)']
const color = (i) => PAL[i % PAL.length]
const fill = (i) => `color-mix(in srgb, ${color(i)} 40%, transparent)`

const pool = ref(26)   // 26 是默认模式的临界点: 往下拖一格就开始抢占
const budget = ref(48)
const rate = ref(2)
const prefixOn = ref(true)
const chunked = ref(true)
const livelock = ref(false)
const sel = ref(-1)

const setMode = (m) => { chunked.value = m === 'chunk'; livelock.value = m === 'bug' }

const sim = computed(() => {
  const P = Math.floor(SYS / BS)   // 内容相同的完整前缀 block 数; 第 4 块起混进了各自的问句
  const NB = pool.value, B = budget.value, prefix = prefixOn.value
  const seqs = QLEN.map((q, i) => ({
    id: i, arrive: Math.floor(i / rate.value), prompt: SYS + q, len: SYS + q,
    computed: 0, out: 0, ttft: null, last: 0, tbt: 0, done: false, table: null,
  }))
  const waiting = [], running = [], finished = []

  // ---- m02 BlockManager + m04 PrefixCache 的记账 ----
  const refc = Array(NB).fill(0), keyOf = Array(NB).fill(null), index = new Map()
  const free = Array.from({ length: NB }, (_, i) => i)   // 队首最先被复用 = LRU
  const key = (s, i) => (i < P ? `S${i}` : `${s.id}:${i}`)   // 链式 hash 的替身: 内容相同才同 key
  const match = (s) => {
    const hits = []
    for (let i = 0; prefix && i < Math.floor((s.len - 1) / BS); i++) {   // 至少留 1 个 token 真算
      const b = index.get(key(s, i))
      if (b === undefined) break
      hits.push(b)
    }
    return hits
  }
  const register = (s) => {
    for (let i = 0; prefix && i < Math.floor(s.computed / BS); i++) {
      const k = key(s, i), b = s.table[i]
      if (!index.has(k) && keyOf[b] === null) { index.set(k, b); keyOf[b] = k }
    }
  }
  const popFree = () => {
    const b = free.shift()
    if (keyOf[b] !== null) { index.delete(keyOf[b]); keyOf[b] = null }   // 内容要被覆盖了, 缓存项跟着失效
    refc[b] = 1
    return b
  }
  const grow = (s, n) => { while (s.table.length < blocks(n)) s.table.push(popFree()) }
  const release = (s) => {
    for (const b of [...s.table].reverse()) if (--refc[b] === 0) free.push(b)   // 内容留着, 只是可以被覆盖
    s.table = null
    running.splice(running.indexOf(s), 1)
  }

  let preempts = 0, comp = 0, hit = 0, ms = 0, idle = 0, dead = false
  const frames = [], startMs = []

  for (let t = 0; t < 400; t++) {
    startMs[t] = ms
    for (const s of seqs) if (s.arrive === t) waiting.push(s)
    const batch = []
    let bud = B, just = false, why = '', blocked = ''

    // 给 running 里每条序列它还欠的 token; block 不够就抢占最年轻的
    const schedRunning = () => {
      const todo = [...running]
      while (todo.length && bud > 0) {
        const s = todo.shift()
        const n = Math.min(s.len - s.computed, bud)
        let ok = true
        while (ok && free.length < blocks(s.computed + n) - s.table.length) {
          const v = todo.length ? todo.pop() : s
          release(v)
          v.computed = 0                       // ★ recompute 式抢占: 还 block, 已生成的文本留着, KV 之后重算
          waiting.unshift(v)
          preempts++; just = true
          why = `R${s.id} 要新 block, 池里没有空闲 → 踢掉最年轻的 R${v.id}: 还 block、num_computed 归 0、回 waiting 队首`
          ok = v !== s
        }
        if (!ok) continue
        grow(s, s.computed + n)
        batch.push([s, n]); bud -= n
      }
    }
    // 从 waiting 队首 FCFS 接新请求; 命中前缀的 token 不占预算、不用算
    const admit = () => {
      while (waiting.length && running.length < MAX_SEQS && bud > 0) {
        const s = waiting[0]
        const hits = match(s), h = hits.length * BS
        let n = s.len - h
        if (chunked.value) n = Math.min(n, bud)
        else if (n > bud && batch.length) break
        // 命中的 block 如果正躺在 free 里, 复活它同样吃掉一个空闲名额
        const need = blocks(s.len) - hits.length + hits.filter((b) => refc[b] === 0).length
        if (free.length < need) {
          blocked = `队首 R${s.id} 还要 ${need} 个 block, 池里只剩 ${free.length} 个`
          break
        }
        waiting.shift()
        for (const b of hits) { if (refc[b] === 0) free.splice(free.indexOf(b), 1); refc[b]++ }
        s.table = [...hits]
        grow(s, s.len)
        s.computed = h; hit += h
        running.push(s); batch.push([s, n]); bud -= n
      }
    }

    if (chunked.value) {
      schedRunning()
      if (!just) admit()                       // 刚抢占过就别立刻把腾出的 block 又占掉
    } else {
      admit()
      // ★ 这个 fallback 就是活锁修复。带 ⚠ 的开关只在「拿不到 block」时把它关掉
      if (!batch.length && !(livelock.value && blocked)) schedRunning()
    }

    // 前向 + 采样 + 后处理
    const nTok = batch.reduce((a, [, n]) => a + n, 0)
    const stalled = running.length - batch.length      // 在 running 里、这一步却没轮到的
    ms += nTok ? FIXED_MS + PER_TOK_MS * nTok : 0
    comp += nTok
    for (const [s, n] of batch) {
      s.computed += n
      register(s)
      if (s.computed < s.len) continue                    // prefill 还没追平, 这一步没有 logits 可采
      s.out++; s.len++
      if (s.ttft === null) s.ttft = ms - startMs[s.arrive]
      else s.tbt = Math.max(s.tbt, ms - s.last)
      s.last = ms
      if (s.out >= OUT[s.id]) { s.done = true; release(s); finished.push(s) }
    }

    const kinds = batch.map(([, n]) => (n === 1 ? 'D' : 'P'))
    const branch = !batch.length ? '空转' : just ? '抢占' : kinds.every((k) => k === 'D') ? 'decode'
      : kinds.every((k) => k === 'P') ? 'prefill' : '混批'
    if (!why) {
      const pre = batch.filter(([, n]) => n > 1)
      const who = `R${pre.map(([s]) => s.id).join(' R')}`, toks = pre.map(([, n]) => n)
      const stuck = blocked || (running.length >= MAX_SEQS ? `max_batch_seqs=${MAX_SEQS} 的并发名额满了` : '本步 token 预算不够再接一条')
      why = branch === '空转'
        ? (waiting.length ? `${stuck} → 队首进不来; 这一步没有落到 decode, running 原地不动` : '队列是空的, 下一条请求还没到')
        : (branch === 'decode' ? 'running 各拿 1 个 token'
          : branch === 'prefill' ? `${who} 做 prefill, 算 ${toks.join(' / ')} 个 token${stalled ? `; running 里另外 ${stalled} 条这一步停顿` : ''}`
            : `running 先各拿 1 个, 剩下的 ${toks.reduce((a, n) => a + n, 0)} 个预算切给 ${who} 的 prefill chunk`)
          + (waiting.length ? `; waiting 里还压着 ${waiting.length} 条: ${stuck}` : '')
    }
    const cells = refc.map((r, b) => {
      const ids = running.filter((s) => s.table.includes(b)).map((s) => s.id)
      const cached = !r && keyOf[b] !== null
      return {
        ids, txt: r > 1 ? '∷' : r ? ids[0] : cached ? '·' : '',
        title: r > 1 ? `共享前缀 block (ref_count = ${r})` : r ? `R${ids[0]} 的 block`
          : cached ? '已释放, 内容还在: 同样的前缀再来仍能命中' : '空闲 block',
      }
    })

    const inBatch = Object.fromEntries(batch.map(([s, n]) => [s.id, n]))
    frames.push({
      t, branch, why, cells,
      waiting: waiting.map((s) => ({ id: s.id, n: 0, txt: s.computed === 0 && s.out ? ` 重算 ${s.len}` : '' })),
      running: running.map((s) => ({ id: s.id, n: inBatch[s.id] || 0, txt: ` ${s.computed}/${s.len}` })),
      finished: finished.map((s) => ({ id: s.id, n: 0, txt: ` ${s.out} tok` })),
    })

    if (!batch.length && waiting.length) { if (++idle > 5) { dead = true; break } } else idle = 0
    if (!waiting.length && !running.length && seqs.every((s) => s.arrive <= t)) break
  }

  const totalOut = seqs.reduce((a, s) => a + s.out, 0)
  return {
    frames, dead, preempts, comp, hit, seqs,
    need: seqs.reduce((a, s) => a + s.len - 1, 0),
    thr: dead || ms === 0 ? 0 : (totalOut / ms) * 1000,   // 卡死之后再也不会有 token 出来, 稳态吞吐就是 0
    maxTbt: Math.max(0, ...seqs.map((s) => s.tbt)),
  }
})

const stepper = useStepper(() => sim.value.frames.length, { interval: 420 })
const f = computed(() => sim.value.frames[Math.min(stepper.step.value, sim.value.frames.length - 1)])
const queues = computed(() => [
  { name: 'waiting', items: f.value.waiting },
  { name: 'running', items: f.value.running },
  { name: 'finished', items: f.value.finished },
])
const selSeq = computed(() => sim.value.seqs.find((s) => s.id === sel.value))
const branchCls = computed(() => ({ 抢占: 'warn', 空转: 'dead', decode: 'ok', prefill: 'ok', 混批: 'ok' }[f.value.branch]))
// 换参数后回到第 0 步, 从头看这组配置怎么演化
watch(sim, () => { stepper.pause(); stepper.step.value = 0 }, { immediate: true })
</script>

<style scoped>
.engine { display: flex; flex-direction: column; gap: 8px; }
.qrow { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; min-height: 30px; }
.qlab { width: 62px; flex: none; font-size: 11px; color: var(--text-dim); }
.chip { min-height: 26px; padding: 2px 8px; border-width: 2px; font-size: 12px; }
.chip em { font-style: normal; font-size: 10px; color: var(--text-dim); margin-left: 3px; }
.chip.active { outline: 2px solid var(--accent); outline-offset: 1px; }
.chip.dimmed { opacity: 0.4; }
.tip { font-size: 11px; color: var(--text-dim); }
.branch { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; font-size: 12px; color: var(--text-muted); line-height: 1.7; margin: 4px 0 2px; }
.branch .tag { font-size: 11px; }
.tag.ok { border-color: var(--left); color: var(--left); }
.tag.warn { border-color: var(--warn); color: var(--warn); }
.tag.dead { border-color: var(--danger); color: var(--danger); }
.cell.free { border-style: dashed; opacity: 0.5; }
.cell.shared { border-style: dashed; border-color: var(--text-dim); }
.danger-btn.active { background: var(--danger); border-color: var(--danger); color: var(--bg); }
.info { margin-top: 6px; min-height: 3.2em; }
</style>
