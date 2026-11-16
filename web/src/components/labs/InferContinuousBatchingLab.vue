<!-- 连续批处理实验台 (llm_infer/m03): 同一条请求轨迹, 静态 batch vs iteration-level 调度。 -->
<template>
  <LabFrame
    title="连续批处理 — 静态 batch 整批进出, 连续批每一步重组"
    sub="- 模拟什么: 推理引擎的 batch 调度。GPU 一次前向可以同时推进好几条请求 (一个 batch); 每条请求先用 1 步做 prefill (算完整条 prompt), 再逐步 decode (每步生成 1 个 token), 长度各不相同。静态 batch 是「整批进、整批出」: 凑齐一批一起开跑, 要等这一批里最长的那条结束, 才能接下一批。连续批 (continuous batching, 也叫 iteration-level scheduling) 把调度粒度缩到一步: 每步结束都检查谁做完了, 空出来的槽位立刻给排队的下一条。这里 12 条请求、同样的到达时刻, 只换调度策略。
      - 图上是什么: 每一行是 batch 里的一个槽位 (同时最多跑 B 条), 每一列是一步前向, 从左到右是时间。有颜色的格子表示这一步这个槽位在跑哪条请求, 格里的数字是请求编号 (只标在它的第一格); 虚线格表示这条请求已经结束, 但槽位还被占着空转, 在等同一批里最长的那条 (只有静态 batch 会出现)。空白格是槽位空闲。网格下面一行字是选中请求的明细: 什么时候到达、需要几步、什么时候开始和结束、排队了几步。
      - 可以做什么: 点「静态 batch」或「连续批」切换策略; 点「换一组请求」换一组随机的到达时刻和长度; 拖「到达率」改请求来得多密 (每步平均来几条, 每条请求的长度不变); 拖「batch 槽位数 B」改同时最多跑几条; 用播放器逐步播放; 点下面的 R 按钮或悬停格子, 高亮一条请求的全部格子。
      - 应该看到: 静态 batch 的网格里有大片虚线格和空白格: 短请求早就结束了, 槽位却要等最长的那条; 后到的请求在外面排队。切到连续批, 虚线格消失, 格子紧紧排在一起, 总步数变少。右侧读数都是「静态 → 连续」: 「槽位利用率」是有颜色的实心格占全部格子的比例; 「平均延迟」是每条请求从到达到结束平均等了几步; 「全部完成」是 12 条请求全部跑完用了几步; 「吞吐」是平均每步算了几个 token。连续批那一侧更好显示绿色。"
    module="llm_infer/m03"
    run="python -m llm_infer.m03_continuous_batching.demo"
    :challenge="{
      ask: '把到达率拖到最大 (请求一开始就全到齐), 静态 batch 还会比连续批差吗? 差在哪?',
      answer: '还是差。请求全到齐时, 静态 batch 不用再等人, 但同一批里各条请求的输出长度不同:\n- 静态 batch: 短请求结束后, 它的槽位要一直空转到这一批最长的那条结束 (网格里的虚线格), 下一批才能开始。槽位利用率掉下来, 后面几批的请求都在排队。\n- 连续批: 每一步结束后立刻把空出来的槽位补上排队的请求, 利用率接近 100%, 平均延迟也低。\n所以差距来自「同一批里输出长度参差不齐」, 而不是到达时间: 只要请求长度不一样, 静态 batch 就一定会空转。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'static' }" @click="mode = 'static'">静态 batch (整批进, 整批出)</button>
        <button type="button" :class="{ active: mode === 'cont' }" @click="mode = 'cont'">连续批 (每步重组)</button>
        <button type="button" @click="seed++">换一组请求</button>
      </div>
      <LabSlider v-model="rate" label="到达率 (请求/步)" :min="0.1" :max="3" :step="0.1" />
      <LabSlider v-model="slots" label="batch 槽位数 B" :min="2" :max="6" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
      <div class="row reqs">
        <button
          v-for="r in reqs" :key="r.id" type="button" class="req" :class="{ active: sel === r.id }"
          :style="{ borderColor: color(r.id) }" @click="sel = sel === r.id ? -1 : r.id"
        >R{{ r.id }}</button>
      </div>
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `44px repeat(${cur.T}, 20px)` }">
      <template v-for="(row, s) in cur.grid" :key="s">
        <span class="row-label mono">槽 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t" class="cell"
          :class="{ dim: t > stepper.step.value || (sel >= 0 && c?.id !== sel), held: c?.held, now: t === stepper.step.value }"
          :style="c ? { background: c.held ? 'transparent' : fill(c.id), borderColor: color(c.id) } : null"
          :title="c ? `R${c.id}${c.held ? ' 已结束, 槽位空转' : ''} · t=${t}` : '空闲'"
          @mouseenter="pick(c)" @click="pick(c)"
        >{{ c && c.head ? c.id : '' }}</span>
      </template>
    </div>
    <p class="lab-note info">
      <template v-if="selReq">R{{ selReq.id }}: 第 {{ selReq.arrive }} 步到达, 需要 {{ selReq.dur }} 步 (1 步 prefill + {{ selReq.dur - 1 }} 步 decode);
        第 {{ cur.start[selReq.id] }} 步开始, 第 {{ cur.finish[selReq.id] }} 步结束, 从到达到结束共 {{ cur.finish[selReq.id] - selReq.arrive }} 步
        (其中排队 {{ cur.start[selReq.id] - selReq.arrive }} 步)</template>
      <template v-else>悬停或点格子, 或点上面的 R 按钮, 高亮一条请求的全部格子, 这里会显示它的到达、开始、结束时间。</template>
    </p>

    <template #stats>
      <div class="kv head"><span></span><span class="mono">静态 → 连续</span></div>
      <div class="kv"><span>槽位利用率 (实心格 ÷ 全部格)</span><b :class="cmp(sims.cont.util, sims.static.util)">{{ pct(sims.static.util) }} → {{ pct(sims.cont.util) }}</b></div>
      <div class="kv"><span>平均延迟 (从到达到结束, 步)</span><b :class="cmp(sims.static.lat, sims.cont.lat)">{{ sims.static.lat.toFixed(1) }} → {{ sims.cont.lat.toFixed(1) }}</b></div>
      <div class="kv"><span>12 条全部完成用了几步</span><b>{{ sims.static.T }} → {{ sims.cont.T }}</b></div>
      <div class="kv"><span>吞吐 (平均每步算几个 token)</span><b>{{ sims.static.thr.toFixed(2) }} → {{ sims.cont.thr.toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>连续批只改了一处: 调度粒度从「一个 batch」变成「一步」。每步结束都检查谁做完了, 空出来的槽位立刻给 waiting 队列 (排队中的请求) 的队首。</p>
        <p>真实引擎里的「槽位」不是固定数量的位子, 而是 KV block 够不够、每步 token 预算够不够 (见引擎主循环那一章的实验台)。</p>
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
import { mulberry32, range, sum } from '@/utils/labmath.js'

const N = 12
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)']
const color = (id) => PALETTE[id % PALETTE.length]
const fill = (id) => `color-mix(in srgb, ${color(id)} ${id % 12 < 6 ? 55 : 28}%, transparent)`
const pct = (x) => (x * 100).toFixed(0) + '%'

const mode = ref('cont')
const rate = ref(1)
const slots = ref(4)
const seed = ref(1)
const sel = ref(-1)

// 随机数只跟 seed 走: 拖到达率时每条请求的长度不变, 只是到达时刻整体伸缩
const draws = computed(() => {
  const rand = mulberry32(seed.value * 7919)
  return range(N).map(() => ({ gap: -Math.log(1 - rand()), dur: 2 + Math.floor(rand() ** 2 * 13) })) // 长度偏斜: 多数短, 少数很长
})
const reqs = computed(() => {
  let t = 0
  return draws.value.map((d, id) => { t += d.gap / rate.value; return { id, arrive: Math.floor(t), dur: d.dur } })
})

const simulate = (kind) => {
  const B = slots.value, R = reqs.value
  const grid = range(B).map(() => []), slotOf = Array(B).fill(null) // slotOf[s] = { id, left }
  const start = {}, finish = {}
  let next = 0, batchEnd = 0
  for (let t = 0; Object.keys(finish).length < N && t < 400; t++) {
    const free = range(B).filter((s) => !slotOf[s])
    // ★ 唯一的区别: 静态 batch 要等整批都结束才接人; 连续批每一步都把空槽补上
    const mayAdmit = kind === 'cont' || (free.length === B && t >= batchEnd)
    if (mayAdmit) for (const s of free) {
      if (next >= N || R[next].arrive > t) break
      slotOf[s] = { id: R[next].id, left: R[next].dur, head: true }
      start[R[next].id] = t
      next++
    }
    if (kind === 'static' && mayAdmit) batchEnd = t + Math.max(0, ...slotOf.filter(Boolean).map((x) => x.left))
    for (let s = 0; s < B; s++) {
      const x = slotOf[s]
      if (!x) { grid[s][t] = null; continue }
      grid[s][t] = { id: x.id, head: x.head, held: x.left <= 0 }
      x.head = false
      if (--x.left === 0) finish[x.id] = t + 1
      if (x.left <= 0 && (kind === 'cont' || t + 1 >= batchEnd)) slotOf[s] = null
    }
  }
  const T = grid[0].length
  const busy = sum(R.map((r) => r.dur))
  return { grid, T, start, finish, util: busy / (B * T), thr: busy / T, lat: sum(R.map((r) => finish[r.id] - r.arrive)) / N }
}
const sims = computed(() => ({ static: simulate('static'), cont: simulate('cont') }))
const cur = computed(() => sims.value[mode.value])
const selReq = computed(() => reqs.value.find((r) => r.id === sel.value))
// 颜色跟数值走: 连续批那一侧更好才绿, 更差才红, 打平不上色
const cmp = (better, worse) => (better > worse ? 'good' : better < worse ? 'bad' : '')
const pick = (c) => { if (c) sel.value = c.id }

const stepper = useStepper(() => cur.value.T, { interval: 300 })
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.cell { min-width: 20px; }
.cell.held { border-style: dashed; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.reqs .req { min-height: 28px; padding: 2px 8px; border-width: 2px; }
.info { margin-top: 10px; min-height: 3.4em; }
.kv.head { border: 0; padding: 0; font-size: 11px; color: var(--text-dim); }
</style>
