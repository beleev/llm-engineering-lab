<!--
  流水线并行调度实验台 (对应 llm_train/m04_pipeline_parallel)。
  这个文件同时是"怎么写一个 lab"的范例:
    LabFrame 负责版式, LabSlider 负责滑杆, useStepper + StepPlayer 负责时间轴,
    纯计算放在 computed 里, 模板只管画。
-->
<template>
  <LabFrame
    title="流水线并行调度 — GPipe 和 1F1B 的时间表: 气泡在哪, 显存差在哪"
    sub="- 模拟什么: 流水线并行 (PP) 将深层模型按层切分为 PP 个连续阶段 (stage), 依次部署在 PP 张计算卡上, 并将单个大 batch 拆分为 M 个 micro-batch 顺次送入流水线。每个 micro-batch 需先从卡 0 流向卡 $\text{PP} - 1$ 完成前向计算 (F), 再从卡 $\text{PP} - 1$ 逆向回传至卡 0 完成反向梯度计算 (B)。当某张卡等待上游传来的前向输出或下游传回的反向梯度而处于空闲状态时, 这段闲置时间槽称为流水线气泡 (bubble); 而已完成前向计算、尚未执行对应反向计算的 micro-batch, 必须在显存中持续保留其中间激活张量以供反向求导使用, 称为在途激活。GPipe 与 1F1B 是两种组织前向 F 与反向 B 执行次序的核心流水线调度策略。
- 图上是什么: 时空甘特图中每一行代表一张计算卡 (stage), 每一列代表一个离散时间槽:
  - 紫色块代表前向计算 F, 绿色块代表反向计算 B, 格内标注如 F3 表示第 3 个 micro-batch 的前向步骤 (单个反向步占用的时间槽宽度由「B 耗时」倍率决定)。
  - 空白时间槽即代表流水线气泡, 此时对应设备处于空闲等待状态。
  - 带高亮边框的列标示当前播放到的时间槽, 后续未执行的时间槽以半透明淡色显示。
  - 图表底部状态栏实时列出当前时间槽内各卡正在执行的任务及处于等待状态的卡数。
- 可以做什么: 自上而下依次操作各控件:
  - 「GPipe」: 待全部 M 个 micro-batch 的前向计算悉数完成后, 再集中执行全部反向计算。
  - 「1F1B」: 各卡先执行若干个前向步骤完成流水线预热填充, 随后进入一前向一反向交替执行的稳定阶段。
  - 拖动「流水线级数 PP」: 调节切分的流水线卡数。
  - 拖动「micro-batch 数 M」: 调节单批次拆分的微批次数量。
  - 拖动「B 耗时」: 调节单步反向相对于单步前向的耗时倍率 (真实训练中通常约为 2 倍)。
  - 播放控件: 按时间槽逐步推进观察在途激活变化; 修改参数后视图默认停在时间表末尾以便先总览全局排布再回放细节。
- 应该看到: 两种调度方案的总时间槽长度完全一致, 时间表首尾两端的三角形空白区即为流水线启停气泡, M 越大气泡占总时长的比例越低; 两种方案的核心差异体现在显存占用上。右侧读数含义如下:
  - 「各卡激活峰值」: 整个批次执行期间各张卡同一时刻最多驻留的在途激活份数。GPipe 模式下卡 0 峰值为 M 份; 1F1B 模式下第 s 张卡峰值为 $\text{PP} - s$ 份 (卡 0 峰值为 PP 份), 与 M 无关。
  - 「当前在途激活」: 播放至当前时间槽时各张卡显存中实际保留的激活份数。
  - 「气泡比例」: 空白时间槽占总设备时间槽的比例, 理论公式为 $(\text{PP} - 1) / (\text{PP} - 1 + M)$, 两种调度完全相同。
  - 「总时间槽」: 完成整个 batch 前向与反向计算所需的总时间槽数。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: '将 micro-batch 数 M 从 4 拖至 16, 请先思考以下三个问题:\n- 流水线气泡比例会如何变化?\n- 卡 0 的在途激活峰值会如何变化?\n- GPipe 与 1F1B 两种调度策略在上述两项指标上有何异同?',
      answer: '答案:\n- 气泡比例: 在 $\\text{PP} = 4$ 下, 两种调度的气泡比例均从 $3/7 \\approx 42.9\\%$ 降至 $3/19 \\approx 15.8\\%$。\n- 卡 0 的激活峰值: GPipe 从 4 份增至 16 份, 而 1F1B 始终保持为 4 份不变。\n机制分析:\n- 气泡比例由公式 $(\\text{PP} - 1) / (\\text{PP} - 1 + M)$ 决定: 流水线启动阶段需经历 $\\text{PP} - 1$ 个时间槽等待首个 micro-batch 流至末卡, 收尾阶段同样需经历 $\\text{PP} - 1$ 个时间槽等待末个 micro-batch 回传至首卡, 中间稳定段则由 M 个 micro-batch 填满。M 越大, 满载稳定段占比越高; 由于两种调度的总时间槽长度严格相等, 其气泡比例完全一致。\n- GPipe: 必须先执行完全部 M 个 micro-batch 的前向步骤, 这 M 份中间激活在整个前向阶段都必须驻留在显存中等待后续反向求导, 因此卡 0 的激活峰值等于 M, 当 M 调至 16 时即占用 16 份激活显存。\n- 1F1B: 严格限制第 $s$ 张卡上「已完成前向但尚未反向」的在途 micro-batch 数不超过 $\\text{PP} - s$, 一旦达到上限便立即执行一次反向计算以释放对应激活显存, 再接纳新的前向任务。因此卡 0 的激活峰值恒定为 $\\text{PP} = 4$, 与 M 的大小无关。\n由此可见, 1F1B 直接节省的是激活显存而非单步绝对耗时; 其核心工程价值在于解耦了在途激活显存与 micro-batch 数 M 的线性绑定, 使训练系统得以在不触发显存溢出的前提下将 M 开大, 从而间接压低流水线气泡占比。',
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
        <p><Tex text="1F1B 调度的核心约束在于控制各卡在途上限: 第 $s$ 张卡上「已完成前向但尚未反向」的 micro-batch 数不得超过 $\text{PP} - s$; 一旦触及该上限, 必须优先执行一次反向计算并立即释放该份激活显存, 随后再启动下一个前向步骤。" /></p>
        <p>若去掉该在途上限约束, 流水线调度便退化为 GPipe, 卡 0 的在途激活峰值重新随 M 线性增长; 引入该约束后激活峰值封顶于 PP, 从而允许大幅调高 M 以压缩气泡占比。</p>
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
