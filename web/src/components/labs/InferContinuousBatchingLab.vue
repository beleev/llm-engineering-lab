<!-- 连续批处理实验台 (llm_infer/m03): 同一条请求轨迹, 静态 batch vs iteration-level 调度。 -->
<template>
  <LabFrame
    title="连续批处理 — 静态 batch 整批进出, 连续批每一步重组"
    sub="- 模拟什么: 推理服务引擎的多请求批处理调度。GPU 单次前向可并行推进多条请求组成的 batch; 每条请求先占用 1 个迭代步完成 prefill, 随后逐个迭代步执行 decode (每步生成 1 个 token), 且各请求所需的生成总步数长短不一。实验固定 12 条请求及其到达时间序列, 对比两种批调度粒度:
  - 静态 batch: 采用「整批进入、整批退出」机制, 凑齐一批请求后同步启动, 必须等待同批内生成长度最长的请求彻底结束, 整批槽位才统一释放并接纳下一批请求。
  - 连续批 (continuous batching, 亦称 iteration-level scheduling 迭代级调度): 将调度决策粒度细化至单个迭代步, 每步前向结束后立即回收已生成完毕的请求槽位, 并当即填入等待队列中的新请求。
- 图上是什么: 主体网格中每一行代表 batch 内的一个并发槽位 (上限为 B 条), 每一列代表一个前向迭代步, 横轴从左至右对应时间推进。网格包含三类状态格, 下方附有单条请求的时间明细:
  - 彩色实心格: 表示该槽位在当前步正在有效执行某条请求, 请求编号标注于其起始首格。
  - 虚线边框格: 表示该槽位上的短请求已提前生成完毕, 但因等待同批内最长的请求收尾而被迫留置闲置 (仅在静态 batch 模式下出现)。
  - 空白格: 当前无活跃请求占用的空闲槽位。
  - 网格下方状态栏: 展示当前选中请求的到达步、所需总步数、实际启动步与完成步以及累计排队等待步数。
- 可以做什么: 按面板从上至下支持以下操作:
  - 点击「静态 batch」或「连续批」切换批处理调度策略。
  - 点击「换一组请求」随机生成一组新的请求到达时刻与输出长度分布。
  - 拖动「到达率」滑杆调节请求到达的密集程度 (即每步平均新到达请求数, 各请求自身的生成长度保持不变)。
  - 拖动「batch 槽位数 B」滑杆修改引擎允许并行执行的最大并发槽位数。
  - 使用播放器逐步回放槽位填充与回收过程。
  - 点击底部的 R 编号按钮或悬停网格方格, 高亮追踪单条请求在整个生命周期内占用的时间格。
- 应该看到: 静态 batch 网格中散布大量虚线闲置格与尾部空白格: 短请求虽早已完成生成, 槽位却被同批长请求阻塞而无法复用, 后续已到达的新请求只能在队列外等待; 切换至连续批后, 虚线闲置格彻底消失, 有效计算格紧密拼接, 跑完全部请求所需的总步数显著缩短。右侧四项核心读数均按「静态基线对比连续批」并列展示, 连续批表现更优时标绿:
  - 槽位利用率: 彩色有效计算格占运行期间全部槽位时间格的比例。
  - 平均延迟: 每条请求从到达系统至生成完毕平均经历的总步数。
  - 全部完成: 处理完 12 条请求所需的总迭代步数。
  - 吞吐: 运行期间平均每个迭代步有效生成的 token 数量。"
    module="llm_infer/m03"
    run="python -m llm_infer.m03_continuous_batching.demo"
    :challenge="{
      ask: '将到达率拖至最大 (所有请求在初始时刻已全部到达就绪), 静态 batch 的表现仍会劣于连续批吗? 二者的差距源于何处?',
      answer: '仍然显著劣于连续批。当所有请求在初始时刻全部就绪时, 静态 batch 虽然消除了凑批等待外部到达的空窗, 但同一批次内各条请求的输出生成长度依然参差不齐:\n- 静态 batch 的同步屏障损耗: 批内短请求提前输出结束符后, 其占用的并发槽位无法提前释放, 必须持续闲置等待同批内最长的那条请求生成完毕 (对应网格中的虚线格), 下一批排队请求才能进入执行。由此导致有效槽位利用率大幅回落, 后续批次的请求在等待队列中承受长尾阻塞。\n- 连续批的即时复用优势: 每个迭代步结束后立即将已完成请求的槽位释放并补入等待队列头部的请求, 在请求充足时可将槽位利用率维持在接近 100% 的水平, 平均排队与端到端延迟同步大幅降低。\n二者的核心性能差距源于同批请求间生成长度的异质性, 而非请求到达时刻的疏密: 只要各请求的输出长度不一致, 静态批处理就必然产生同步等待空泡。',
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
        <p>连续批处理的核心改动仅在于将调度决策粒度从「整个批次生命周期」缩减至「单个前向迭代步」: 每步执行完毕后即时回收已终止序列的槽位, 并立即接纳 waiting 等待队列头部的请求进入下一迭代步。</p>
        <p>生产级推理引擎中的并发上限并非由静态固定的槽位数决定, 而是受物理 KV block 池剩余容量与单步最大 token 预算双重动态约束 (详见「推理引擎主循环」一章的实验台)。</p>
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
