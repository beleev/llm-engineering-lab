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
    sub="- 模拟什么: 一个 mini-vLLM 推理引擎的主循环。5 条请求共享同一段 30 token 的 system prompt, 前缀缓存开着时, 后来者的前 3 块 KV 直接命中, 不用再算。耗时用代价模型 (每步 20 ms + 0.25 ms × token 数), 不是实测。每一步永远是同四件事:
        1. 调度 (决定这一步算谁、算几个 token)。难的只有调度, 它要同时管住 KV 显存 (KV cache 切成固定大小的 block 放在一个池子里, 每条请求每 8 个 token 占 1 块; 池子不够就得抢占, 也就是踢掉一条请求让它回去重排)、公平 (先来先进) 和延迟 (每步的 token 预算)。
        2. 前向。
        3. 采样。
        4. 后处理 (结束的请求把 KV block 还回池子)。
      - 图上是什么: 从上到下三部分:
        - 上面三行: 三条队列。waiting 是排队等进入的请求; running 是正在算的, 芯片上的「已算/总长」是这条请求已经有 KV 的 token 数和当前总长度, 有底色表示这一步轮到它; finished 是已结束的, 标着输出了几个 token。
        - 中间一行: 这一步调度器走的分支 (decode / prefill / 混批 / 抢占 / 空转) 和原因。
        - 下面的格子: KV block 池, 每格一块。有颜色的写着占用它的请求编号, ∷ 是几条请求共享的前缀块, · 是已释放但内容还在的块 (同样的前缀再来仍能命中), 虚线空格是空闲。
      - 可以做什么: 第一排是三个调度按钮, 下面是开关、滑杆和播放器:
        - 「decode 优先 + 分块 prefill」先给 running 里的每条 1 个 token, 剩下的预算切给新请求的 prefill。
        - 「prefill 优先」先让新请求整段 prefill, 进不来再落到 decode。
        - 带 ⚠ 的开关是故意写错的版本, 队首拿不到 block 时直接返回空 batch, 用来演示活锁 (服务没崩, 但所有请求都不动了)。
        - 「前缀缓存 开/关」决定共享的 system prompt 能不能命中。
        - 拖「KV block 池」改池子有几块 (26 块是分块模式的临界点, 再小就开始抢占)。
        - 拖「max_batch_tokens」改每步最多算几个 token。
        - 拖「到达速率」改请求来得多快。
        - 用播放器逐步看; 点队列里的 R 按钮看那条请求占了哪些块、延迟是多少。
      - 应该看到: 默认参数下 5 条请求交错着 prefill 和 decode, 池子里出现 ∷ 的共享块, 全部跑完 34 步。把池子往下拖, 开始出现「抢占」: 最年轻的请求被踢回 waiting 队首, 它的块被还回去, 之后要重算。开 ⚠ 开关再拖池子, 时间线会在某一步之后永远「空转」。右侧读数:
        - 吞吐: 输出 token 数除以总耗时 (活锁时是 0)。
        - 最大 TBT: 所有请求里相邻两个 token 之间最长的等待。
        - 抢占次数: 有抢占就红。
        - 实算 + 命中: 实际做了前向的 token 数加上前缀命中跳过的 token 数, 两者之和应该等于「需要」(所有请求需要 KV 的 token 总数), 多出来的部分是被抢占的请求回来重走的。"
    module="llm_infer/full_engine"
    run="python -m llm_infer.full_engine.demo"
    :challenge="{
      ask: '先点带 ⚠ 的开关 (队首拿不到 block 时不落到 decode), 其余保持默认, 再把 block 池从 26 块一格一格往下拖。拖到几块时间线冻住?',
      answer: '22 块。\n4 条并发跑满要 $3 + 4 \\times 5 = 23$ 块: 3 块是共享的 system prompt 前缀 (30 个 token, 每块 8 个, 前 3 块内容完全相同, 几条请求共用), 每条请求自己的问句加 20 个输出另占 5 块。池子少一块, 就得踢掉最年轻的 R3 (抢占)。接下来:\n1. R3 回到 waiting 队首, 拿不到 block; prefill 分支选不出任何序列, 返回空 batch。\n2. running 里的请求一个 token 也不前进, 就没有序列结束。\n3. 没有序列结束, block 就永远不释放。\n4. block 不释放, 队首就永远进不来。\n这种状态叫活锁。\n解法只有一行: prefill 进不来时落到 decode, 对应 Python 里 `_admit(budget) or _schedule_running(budget)` 的那个 or。它保证最老的那条序列总能前进: 序列跑完才会释放 block, 释放了队首才有名额。\n并发名额满了 (max_batch_seqs = 4) 不会冻住: 那时 block 够用, running 照常 decode, 有一条结束队首就能进。',
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
      <p v-if="sim.dead" class="lab-note">
        活锁: 连续 6 步选不出任何序列, 之后也不会变: block 只在序列结束时释放, 而没有序列能前进。
        真实引擎里这表现为「服务没崩, 但所有请求都不动了」。
      </p>
      <p v-else-if="sim.comp + sim.hit !== sim.need" class="lab-note">
        实算加命中超过「需要」的部分, 是被抢占的序列回来重走的 token: 一部分重算, 一部分命中自己留在池里的 block。文本没丢 (已生成的 token 还在), 丢的是算力。
      </p>
      <div class="lab-note">
        <p>耗时用的是 chunked prefill 那一节的代价模型: 每步 20 ms 固定开销 + 0.25 ms × 本步 token 数, 不是实测。</p>
        <p>对照 demo (预算 48, 到达速率 2 以上, 前缀缓存开):</p>
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
