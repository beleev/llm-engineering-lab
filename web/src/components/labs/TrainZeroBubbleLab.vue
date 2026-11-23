<!--
  Zero Bubble 调度实验台 (对应 llm_train/m04_pipeline_parallel:simulate_zb)。
  只讲一件事: 把反向拆成 B (算输入梯度) 和 W (算权重梯度) 之后, W 可以往后挪去填气泡。
  右下角的 DualPipe 一栏是 DeepSeek-V3 报告的代价公式, 不是时间表模拟。
-->
<template>
  <LabFrame
    title="Zero Bubble — 把反向拆成两半, 用 W 去填气泡"
    sub="- 模拟什么: 流水线并行把模型按层切成 PP 个 stage, 一个 batch 拆成 M 个 micro-batch 依次流过。1F1B 调度里每张卡轮流做一个前向和一个完整反向, 首尾仍然有空等的时间格 (气泡)。Zero Bubble (ZB-H1) 把完整反向拆成两步: B 只算对输入的梯度, 要立刻传给上一个 stage; W 只算对权重的梯度, 谁都不等它。于是 W 可以往后挪, 填进这张卡本来空着的格子。
- 图上是什么: 一行一张卡 (stage 0 在最上), 一列一个时间格:
  - 紫色是前向 F, 绿色是 B, 黄色是 W, 空白是气泡 (这张卡在等)。
  - 格里的数字是 micro-batch 编号, 比如 B3 是第 3 个 micro-batch 的 B。
  - 带高亮边框的那一列是当前播放到的时间格, 后面还没执行的格子显示得淡一些。
  - 1F1B 模式下不拆反向, 所以只有 F 和 B 两种颜色, 那个 B 占 t_b + 1 格。
- 可以做什么: 控件从上到下:
  - 「1F1B (反向不拆)」: 完整反向当成一个不可分的操作, 占 t_b + t_w 格。
  - 「ZB-H1 (拆成 B + W)」: B 和 W 分开排, W 在卡空闲时插空, 同时限制每张卡在途不超过 PP 份。
  - 拖动「stage 数 PP」「micro-batch 数 M」改流水线形状。
  - 拖动「B 占几格」改算输入梯度的耗时 (F 和 W 固定 1 格; 论文实测 $T_W \lt T_F \lt T_B$)。
  - 播放控件: 按时间格推进, 看 W 是怎么被塞进空格的。
- 应该看到: 切到 ZB-H1 时时间表整体变短, 省下来的正好是收尾那一片三角形空白。右侧读数:
  - 「总时间格」: 跑完一个 batch 要多少格。ZB-H1 应该比 1F1B 少。
  - 「每卡空闲」: 一张卡在整条时间线上空着的格数。1F1B 是 $(\text{PP}-1)(t_f+t_b+t_w)$, ZB-H1 是 $(\text{PP}-1)(t_f+t_b-t_w)$; 三者都是 1 格时后者正好是前者的 1/3。
  - 「气泡占比」: 空闲格 ÷ 总格数。
  - 「在途峰值」: 每张卡同时攥着几份还没做完 W 的激活。两种调度的最大值都是 PP, 但 ZB-H1 下靠后的 stage 攥得更多。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: '先看 PP=4、M=8、B 占 1 格: ZB-H1 的每卡空闲是 3 格, 1F1B 是 9 格, 正好三分之一。现在把「B 占几格」拖到 3, 再预测:\n- 两种调度的每卡空闲各变成多少?\n- 「ZB-H1 的空闲是 1F1B 的 1/3」还成立吗?\n- 两种调度的在途峰值为什么一个是 [4,3,2,1]、一个是 [4,4,4,4]?',
      answer: '答案: 1F1B 15 格、ZB-H1 9 格, 比值变成 3/5, 不再是 1/3。\n逐步看:\n1. 两条公式分别是 $(\\text{PP}-1)(t_f+t_b+t_w)$ 和 $(\\text{PP}-1)(t_f+t_b-t_w)$。B 占 3 格时括号里是 $1+3+1=5$ 与 $1+3-1=3$, 乘上 $\\text{PP}-1=3$ 得 15 和 9。\n2. 比值 $(t_f+t_b-t_w)/(t_f+t_b+t_w)$ 只在三者耗时相等时等于 1/3。论文为了讲清楚先假设 $T_F=T_B=T_W$, 所以「降到 1/3」这句话带着这个前提; 它实测的比例是 $T_W \\lt T_F \\lt T_B$ 且 $T_B+T_W=2T_F$, 真实收益小于 1/3 这个理想值。\n3. 在途峰值的差别来自 W 什么时候做。1F1B 里反向一做完就把激活释放了, 第 s 张卡同时攥着 $\\text{PP}-s$ 份; ZB-H1 把 W 推迟到空闲时才做, 而激活要等 W 用完才能放, 所以每张卡都攥满 PP 份。\n4. 这不是显存变大了 PP/(PP−s) 倍: 做完 B 只等 W 的那部分, 需要的中间量比还没反向的那部分小 (论文记作 $M_W \\lt M_B$)。卡 0 上 PP 份全是大的那种, 所以峰值仍由卡 0 决定, 和 1F1B 相等。\n再往下一步: 允许峰值升到 $(2\\text{PP}-1)$ 份时 (ZB-H2), 空闲可以压到 $(\\text{PP}-1)(t_f+t_b-2t_w)$, 三者耗时相等时正好为 0。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === '1f1b' }" @click="mode = '1f1b'">1F1B (反向不拆)</button>
        <button type="button" :class="{ active: mode === 'zb' }" @click="mode = 'zb'">ZB-H1 (拆成 B + W)</button>
      </div>
      <LabSlider v-model="pp" label="stage 数 PP" :min="2" :max="6" />
      <LabSlider v-model="m" label="micro-batch 数 M" :min="2" :max="12" />
      <LabSlider v-model="tb" label="B 占几格 (F = W = 1)" :min="1" :max="3" unit=" 格" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `44px repeat(${sched.total}, 22px)` }">
      <template v-for="(row, s) in sched.grid" :key="s">
        <span class="row-label mono">卡 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t"
          class="cell" :class="[c && KIND[c.kind], { dim: t > stepper.step.value, now: t === stepper.step.value }]"
          :title="c ? `${LABEL[c.kind]} 第 ${c.m} 个 micro-batch, 卡 ${s}, 时间格 ${t}` : '气泡: 这张卡在等上游的 F 或下游的 B'"
        >{{ c && c.head ? c.kind + c.m : '' }}</span>
      </template>
    </div>
    <p class="lab-note" style="margin-top: 10px;">{{ nowText }}</p>

    <template #stats>
      <div class="kv"><span>总时间格 (跑完一个 batch; 1F1B {{ base.total }} / ZB-H1 {{ zb.total }})</span><b>{{ sched.total }}</b></div>
      <div class="kv"><span>每卡空闲 (实测 / 公式)</span><b>{{ sched.idle }} / {{ formula }}</b></div>
      <div class="kv"><span>气泡占比 (空闲 ÷ 总格数, 低于 15% 算压住了)</span><b :class="sched.bubble < 0.15 ? 'good' : 'bad'">{{ (sched.bubble * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>在途峰值 (还没做完 W 的份数, 卡 0 在前)</span><b :class="Math.max(...sched.peak) <= pp ? 'good' : 'bad'">[{{ sched.peak.join(', ') }}]</b></div>
      <div class="lab-note">
        <p>★ 1F1B 的空闲是 (PP−1)(t_f+t_b+t_w), ZB-H1 是 (PP−1)(t_f+t_b−t_w): W 从加号挪到了减号, 因为它不再占着首尾那段必须空等的时间。</p>
        <p>本例 1F1B {{ base.total }} 格、ZB-H1 {{ zb.total }} 格, 省 {{ base.total - zb.total }} 格。</p>
        <p>DualPipe (DeepSeek-V3 报告的代价公式, 不随左侧的 M 变):
          PP = {{ ppEven }} 时每卡空闲 1F1B {{ dual['1F1B'][0] }} 格、ZB1P {{ dual.ZB1P[0] }} 格、DualPipe {{ dual.DualPipe[0] }} 格;
          代价是每卡存 2 份参数、在途激活 PP+1 份。它两端同时喂 micro-batch, 时间表要双向排, 左边这张图不画它。</p>
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
import { range } from '@/utils/labmath.js'

const KIND = { F: 'on', B: 'ok', W: 'hot' }
const LABEL = { F: '前向', B: '算输入梯度', W: '算权重梯度' }
const mode = ref('zb')
const pp = ref(4)
const m = ref(8)
const tb = ref(1)

// 和 Python 的 simulate_zb 同一套规则: 1F1B 的 F/B 顺序不变, W 在卡空闲时插空, 在途不超过 PP
function simulate (PP, M, tf, tb, twCost, split) {
  const dur = { F: tf, B: split ? tb : tb + twCost, W: twCost }
  const orders = range(PP).map((s) => {            // 每张卡的 F/B 顺序: warmup 个 F, 之后严格交替
    const warm = Math.min(PP - 1 - s, M)
    const o = range(warm).map((i) => ['F', i])
    let f = warm
    for (let b = 0; b < M; b++) { if (f < M) o.push(['F', f++]); o.push(['B', b]) }
    return o
  })
  const end = new Map(), free = Array(PP).fill(0), ptr = Array(PP).fill(0)
  const pending = range(PP).map(() => []), grid = range(PP).map(() => [])
  const live = Array(PP).fill(0), peak = Array(PP).fill(0)
  for (let t = 0; t < 600; t++) {
    if (ptr.every((p, s) => p === orders[s].length && !pending[s].length)) break
    for (let s = 0; s < PP; s++) {
      if (free[s] > t) continue
      let op = null
      if (ptr[s] < orders[s].length) {
        const [kind, mb] = orders[s][ptr[s]]
        const dep = kind === 'F' ? (s > 0 ? `F${mb},${s - 1}` : null)
          : s < PP - 1 ? `B${mb},${s + 1}` : `F${mb},${s}`
        const full = kind === 'F' && live[s] >= PP        // ★ 显存满了: 先还一个 W 再收新的 F
        if (!full && (dep === null || (end.has(dep) && end.get(dep) <= t))) { op = [kind, mb]; ptr[s]++ }
      }
      if (!op && pending[s].length) op = ['W', pending[s].shift()]   // ★ 没别的事可做就补一个 W
      if (!op) continue
      const [kind, mb] = op
      for (let d = 0; d < dur[kind]; d++) grid[s][t + d] = { kind, m: mb, head: d === 0 }
      free[s] = t + dur[kind]
      end.set(`${kind}${mb},${s}`, free[s])
      if (kind === 'F') live[s]++
      else if (kind === 'B') { if (split) pending[s].push(mb); else live[s]-- }
      else live[s]--
      peak[s] = Math.max(peak[s], live[s])
    }
  }
  const total = Math.max(...free)
  grid.forEach((r) => { for (let t = 0; t < total; t++) r[t] = r[t] || null })
  const busy = M * (tf + tb + twCost)
  return { grid, total, peak, idle: total - busy, bubble: 1 - busy / total }
}

const base = computed(() => simulate(pp.value, m.value, 1, tb.value, 1, false))
const zb = computed(() => simulate(pp.value, m.value, 1, tb.value, 1, true))
const sched = computed(() => (mode.value === 'zb' ? zb.value : base.value))
const formula = computed(() => (pp.value - 1) * (1 + tb.value + (mode.value === 'zb' ? -1 : 1)))
// DeepSeek-V3 表里的符号: B 是完整反向 = 1 + t_w, W = t_w, F&B 取 F + B (通信被掩盖, 计算量不变)
const ppEven = computed(() => (pp.value % 2 === 0 ? pp.value : pp.value + 1))
const dual = computed(() => {
  const P = ppEven.value, F = 1, B = tb.value + 1, W = 1
  return { '1F1B': [(P - 1) * (F + B)], ZB1P: [(P - 1) * (F + B - 2 * W)], DualPipe: [(P / 2 - 1) * (F + B + B - 3 * W)] }
})

const stepper = useStepper(() => sched.value.total, { interval: 320 })
const nowText = computed(() => {
  const t = stepper.step.value
  const cards = sched.value.grid.map((row, s) => `卡 ${s} ${row[t] ? row[t].kind + row[t].m : '气泡'}`)
  const idle = sched.value.grid.filter((row) => !row[t]).length
  return `t = ${t}: ${cards.join(', ')}。${idle ? `${idle} 张卡在等。` : '所有卡都在干活。'}`
})
watch(sched, (s) => { stepper.pause(); stepper.step.value = s.total - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
</style>
