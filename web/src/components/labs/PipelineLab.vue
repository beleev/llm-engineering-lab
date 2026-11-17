<!--
  流水线并行调度实验台 (对应 llm_train/m04_pipeline_parallel)。
  这个文件同时是"怎么写一个 lab"的范例:
    LabFrame 负责版式, LabSlider 负责滑杆, useStepper + StepPlayer 负责时间轴,
    纯计算放在 computed 里, 模板只管画。
-->
<template>
  <LabFrame
    title="流水线并行调度 — GPipe 和 1F1B 的时间表: 气泡在哪, 显存差在哪"
    sub="- 模拟什么: 流水线并行 (PP) 是把模型按层切成 PP 段 (每段叫一个 stage), 各放一张卡; 一个 batch 切成 M 个 micro-batch 依次流过。每个 micro-batch 要先从卡 0 到卡 PP−1 做完前向 (F), 再从卡 PP−1 倒回卡 0 做反向 (B)。一张卡在等上游的前向或下游的反向时就是空转, 这段空等叫气泡。已经前向、还没反向的 micro-batch, 它的激活必须留在显存里等反向用, 这叫在途激活。GPipe 和 1F1B 是两种安排 F 和 B 先后顺序的调度。
      - 图上是什么: 每一行是一张卡 (stage), 每一格是一个时间槽。
        - 紫格是前向, 绿格是反向, 格子里的字如 F3 表示第 3 个 micro-batch 的前向 (一个 B 占的格子比 F 多, 按「B 耗时」设定)。
        - 空格就是气泡, 这张卡在等。
        - 当前时间槽的格子有高亮框, 之后的格子变淡。
        - 图下面一行字说明当前时间槽每张卡在做什么、几张卡在等。
      - 可以做什么: 控件从上到下:
        - 「GPipe」: 先把 M 个 micro-batch 的前向全做完, 再做全部反向。
        - 「1F1B」: 每张卡先做几个前向预热, 然后一前一后交替。
        - 拖「流水线级数 PP」改卡数。
        - 拖「micro-batch 数 M」改 micro-batch 个数。
        - 拖「B 耗时」改反向比前向慢几倍 (真实训练约 2 倍)。
        - 用播放控件按时间槽走; 改参数后台子停在时间表末尾, 先看整体再回放。
      - 应该看到: 两种调度的总时间槽数一样, 时间表开头和结尾的三角形空白就是气泡, M 越大三角形占比越小。差别在显存。右侧读数:
        - 「各卡激活峰值」: 每张卡同时攥着的在途激活最多有几份。GPipe 下卡 0 是 M 份, 1F1B 下是 PP − s 份 (卡 0 为 PP), 和 M 无关。
        - 「当前在途激活」: 播放到当前时间槽时每张卡手里的份数。
        - 「气泡比例」: 空格数占总格数的比例, 公式 (PP−1)/(PP−1+M), 两种调度相同。
        - 「总时间槽」: 跑完一个 batch 要多久。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: '把 micro-batch 数 M 从 4 拖到 16, 先猜:\n- 气泡比例怎么变?\n- 卡 0 的激活峰值怎么变?\n- 两种调度在这两个数上有什么不同?',
      answer: '答案:\n- 气泡比例: 两种调度都从 3/7 ≈ 42.9% 降到 3/19 ≈ 15.8% (PP = 4 时)。\n- 卡 0 的激活峰值: GPipe 从 4 涨到 16, 1F1B 一直是 4。\n为什么:\n- 气泡比例是 $(PP-1)/(PP-1+M)$: 时间表开头有 PP−1 个槽在等第一个 micro-batch 流到底, 结尾也有 PP−1 个槽在等最后一个流回来, 中间 M 个 micro-batch 把流水线填满。M 越大, 填满的部分占比越高。两种调度的总时间槽数相同, 所以气泡一样。\n- GPipe: 先做完全部 M 个前向, 这 M 份激活都得攥着等反向, 卡 0 的峰值 = M。M 拖到 16 就是 16 份。\n- 1F1B: 卡 $s$ 上「已前向未反向」的 micro-batch 数不许超过 $PP - s$, 超过就先做一个反向把激活释放掉。卡 0 的峰值恒为 PP = 4, 和 M 无关。\n所以 1F1B 省的是显存, 不是时间。它的价值是: 在途激活不随 M 涨, 于是可以把 M 开大, 间接把气泡压小。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'gpipe' }" @click="mode = 'gpipe'">GPipe (先全 F 再全 B)</button>
        <button type="button" :class="{ active: mode === '1f1b' }" @click="mode = '1f1b'">1F1B (一前一后交替)</button>
      </div>
      <LabSlider v-model="pp" label="流水线级数 PP" :min="2" :max="6" />
      <LabSlider v-model="m" label="micro-batch 数 M" :min="1" :max="16" />
      <LabSlider v-model="bCost" label="B 耗时 (F = 1)" :min="1" :max="3" unit="×" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `48px repeat(${sched.T}, 22px)` }">
      <template v-for="(row, s) in sched.grid" :key="s">
        <span class="row-label mono">卡 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t"
          class="cell"
          :class="[c && (c.type === 'F' ? 'on' : 'ok'), { dim: t > stepper.step.value, now: t === stepper.step.value }]"
          :title="c ? `${c.type === 'F' ? '前向' : '反向'} 第 ${c.m} 个 micro-batch, 卡 ${s}, 时间槽 ${t}` : '气泡: 这张卡在等上游的前向或下游的反向'"
        >{{ c && c.head ? c.type + c.m : '' }}</span>
      </template>
    </div>
    <p class="lab-note" style="margin-top: 10px;">{{ nowText }}</p>

    <template #stats>
      <div class="kv"><span>总时间槽 (跑完一个 batch 要多久)</span><b>{{ sched.T }}</b></div>
      <div class="kv"><span>气泡比例 (空格 / 总格, 公式 (PP−1)/(PP−1+M))</span><b>{{ (sched.bubble * 100).toFixed(1) }}%</b></div>
      <div class="kv">
        <span>各卡激活峰值 (同时攥着几份在途激活, 卡 0 在前)</span>
        <b :class="Math.max(...sched.peaks) > pp ? 'bad' : 'good'">[{{ sched.peaks.join(', ') }}]</b>
      </div>
      <div class="kv"><span>当前时间槽各卡的在途激活</span><b>[{{ liveNow.join(', ') }}]</b></div>
      <div class="lab-note">
        <p><Tex text="1F1B 的关键只有一行: 卡 $s$ 上「已前向未反向」的 micro-batch 数不许超过 $PP - s$。到了上限就先做一个反向, 把那份激活释放掉, 再做下一个前向。" /></p>
        <p>少了这个上限, 调度就退化成 GPipe, 卡 0 的激活峰值回到 M。有了它, 峰值封顶在 PP, 于是 M 可以开大, 气泡跟着变小。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { range } from '@/utils/labmath.js'

const mode = ref('1f1b')
const pp = ref(4)
const m = ref(8)
const bCost = ref(2)

// 离散事件模拟: 每个时间槽, 每张空闲的卡挑一个"依赖已满足"的任务开工
const sched = computed(() => {
  const PP = pp.value, M = m.value, FB = { F: 1, B: bCost.value }
  const fEnd = range(PP).map(() => Array(M).fill(Infinity)) // fEnd[s][k]: F_k 在卡 s 上的完成时刻
  const bEnd = range(PP).map(() => Array(M).fill(Infinity))
  const fNext = Array(PP).fill(0), bNext = Array(PP).fill(0) // 每张卡下一个要做的 F / B 编号
  const busyUntil = Array(PP).fill(0)
  const grid = range(PP).map(() => [])

  for (let t = 0; bNext.some((b) => b < M) && t < 400; t++) {
    for (let s = 0; s < PP; s++) {
      if (busyUntil[s] > t) continue
      const k = bNext[s], j = fNext[s]
      // B_k 的依赖: 下游卡已反传完 (最后一张卡则是自己的 F_k 做完)
      const bReady = k < M && (s === PP - 1 ? fEnd[s][k] : bEnd[s + 1][k]) <= t
      // F_j 的依赖: 上游卡已前向完
      const fReady = j < M && (s === 0 || fEnd[s - 1][j] <= t)
      // ★ 1F1B 的全部秘密: 在途 (已 F 未 B) 不超过 PP - s; GPipe 没有这个上限, 但要求 F 全做完才开始 B
      const inflight = fNext[s] - bNext[s]
      const canF = fReady && (mode.value === 'gpipe' || inflight < PP - s)
      const canB = bReady && (mode.value === '1f1b' || fNext[s] === M)

      const type = canB ? 'B' : canF ? 'F' : null
      if (!type) continue
      const idx = type === 'B' ? k : j
      for (let d = 0; d < FB[type]; d++) grid[s][t + d] = { type, m: idx, head: d === 0 }
      busyUntil[s] = t + FB[type]
      if (type === 'B') { bEnd[s][idx] = busyUntil[s]; bNext[s]++ } else { fEnd[s][idx] = busyUntil[s]; fNext[s]++ }
    }
  }

  const T = Math.max(...grid.map((r) => r.length))
  grid.forEach((r) => { for (let t = 0; t < T; t++) r[t] = r[t] || null })
  const work = M * (FB.F + FB.B)
  // live[s][t]: 时刻 t 卡 s 上攥着的激活份数 = 已开始的 F − 已完成的 B
  const live = range(PP).map((s) => range(T).map((t) =>
    fEnd[s].filter((e) => e - FB.F <= t).length - bEnd[s].filter((e) => e <= t).length))
  return { grid, T, bubble: 1 - work / T, peaks: live.map((r) => Math.max(...r)), live }
})

const stepper = useStepper(() => sched.value.T, { interval: 350 })
const liveNow = computed(() => sched.value.live.map((r) => r[stepper.step.value] ?? 0))
// 当前时间槽每张卡在干什么: F3 = 第 3 个 micro-batch 的前向, 气泡 = 这张卡在等
const nowText = computed(() => {
  const t = stepper.step.value
  const cards = sched.value.grid.map((row, s) => `卡 ${s} ${row[t] ? row[t].type + row[t].m : '气泡'}`)
  const idle = sched.value.grid.filter((row) => !row[t]).length
  return `t = ${t}: ${cards.join(', ')}。${idle ? `${idle} 张卡在等上游的前向或下游的反向。` : '所有卡都在干活。'}`
})
// 换参数后直接跳到末尾, 先看到完整时间表, 再按播放重看过程
watch(sched, (s) => { stepper.pause(); stepper.step.value = s.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
</style>
