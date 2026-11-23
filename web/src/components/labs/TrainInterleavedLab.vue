<!--
  交错 1F1B (对应 llm_train/m04_pipeline_parallel:device_order + simulate, 逐行移植)。
  只讲一件事: 每张卡拿 v 个不相邻的小 stage, micro-batch 在卡之间绕 v 圈 —— 气泡除以 v, 代价是 v 倍的 P2P 和更多在途激活。
-->
<template>
  <LabFrame
    title="交错 1F1B — 每张卡拿 v 个不相邻的小段, 把流水线气泡再除以 v"
    sub="- 模拟什么: 流水线并行 (PP) 将模型按层切分为 PP 个连续阶段 (stage) 并分别放置在不同卡上, 单个 batch 拆分为 M 个 micro-batch 依次流经各卡。流水线启动与收尾阶段不可避免存在设备空闲等待, 空闲时间占比称为流水线气泡。普通 1F1B 调度下, 每张卡先执行若干步前向传播完成热身 (warmup), 随后严格按一次前向 (F)、一次反向 (B) 交替执行。交错 1F1B 进一步将每张卡负责的层拆分为 $v$ 个不相邻的子阶段 (虚拟 stage), 单个 micro-batch 需在所有卡之间循环流转 $v$ 轮。由于每个子阶段的计算量缩减为原来的 $1/v$, 流水线填充与排空阶段的空闲时间也同步缩减至 $1/v$, 气泡占比公式由 $(PP-1)/(M+PP-1)$ 降至 $(PP-1)/(vM+PP-1)$。其代价是跨卡点对点传输次数增至约 $v$ 倍, 同时在途激活 (已完成前向但尚未执行反向传播的中间激活) 的缓存压力也会上升。本实验台按照 Megatron 交错调度规则逐时间格生成完整执行时间表。
- 图上是什么: 每一行代表一张卡, 横轴从左至右为时间轴, 单格宽度对应 $1/v$ 个整 stage 的前向耗时, 单次反向 B 占用 2 格宽度。
  - 蓝色格表示前向 F, 绿色格表示反向 B, 格内如 F3、B3 的标签标明当前处理的 micro-batch 编号。
  - 同色系的深浅区分当前卡上的第几个虚拟 stage, 颜色越深代表模型层位置越靠后。
  - 空白格即为流水线气泡, 表示该卡在当前时刻处于空闲等待状态。
- 可以做什么: 控件从上到下依次为:
  - 拖动「流水线级数 PP」调整参与流水的卡数。
  - 拖动「M」调整 micro-batch 总数 (交错调度要求 M 为 PP 的整数倍, 因此滑杆按倍数步进)。
  - 拖动「每卡虚拟 stage 数 v」, 当 v 取 1 时退化为普通 1F1B 调度。
  - 点击「跟踪 micro-batch」中的编号, 或悬停、点击时间表中的任意格子, 同一 micro-batch 的完整执行轨迹将高亮显示而其余变暗, 便于观察其如何在各卡间循环绕行 $v$ 轮。
  - 使用播放器沿时间轴逐步推进, 尚未到达的时刻呈暗色显示。
- 应该看到: 在 PP 为 4、M 为 8 的配置下, 将 v 从 1 调至 2 时, 右侧「气泡」占比从 27.3% 降至 15.8%, 排程模拟值与理论公式完全一致。气泡缩小的同时, 「折算成整 stage」与「P2P 次数」两项指标均有所上升, 直观体现了交错调度的资源权衡。右侧其余读数含义如下:
  - 「在途激活峰值」: 各卡在同一时刻最多持有的子阶段激活份数, 每份单位为 $1/v$ 个整 stage。
  - 「折算成整 stage (卡 0)」: 将卡 0 的峰值份数除以 $v$ 折算为整 stage 等效激活量, 并与括号内普通 1F1B 的基准值对比, 高于 v 为 1 时标红。
  - 「每个 micro-batch 的 P2P 次数」: 单个 micro-batch 在卡间传递激活与梯度的总次数, 等于 $2(PP \cdot v - 1)$, 高于 v 为 1 时标红。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: '固定 PP 为 4、M 为 8。将虚拟 stage 数 v 从 1 调至 2 时, 气泡占比会从 27.3% 降至多少? 卡 0 的在途激活峰值 (折算成整 stage) 是上升还是下降?',
      answer: '结论:\n- 气泡占比降至 15.8%。\n- 卡 0 折算为整 stage 的在途激活峰值不降反升, 从 4 个整 stage 增至 5.5 个。\n机制解析:\n- 气泡占比: 代入公式 $(PP-1)/(v \\cdot M + PP - 1)$, 当 v 为 1 时等于 $3/11$ 即 27.3%, 当 v 为 2 时等于 $3/19$ 即 15.8%。物理机制在于流水线气泡的绝对时长正比于单个子阶段的计算耗时 (流水线需等待首个 micro-batch 走完全部阶段后才能进入满载稳态)。交错调度将每个 stage 拆分为 $v$ 个虚拟子阶段, 单段计算量缩减为 $1/v$, 填充与排空阶段的空闲时长随之缩减至 $1/v$, 而总有效计算量保持不变, 因此分母中的 $M$ 等效放大为 $vM$。\n- 在途激活峰值: 读数中卡 0 的在途峰值为 11 份, 但每份仅包含半个 stage 的层激活, 折算为整 stage 后等于 11 乘 1/2 即 5.5 个, 高于普通 1F1B 下的 4 个。这是因为交错调度的热身 (warmup) 阶段更长: 卡 s 需先连续完成 $(PP-1-s) \\cdot 2 + (v-1) \\cdot PP$ 次前向传播后才进入一 F 一 B 稳态交替, 热身前向次数越多, 同时驻留在显存中等待反向传播的激活份数就越多。\n- 跨卡点对点传输: 单个 micro-batch 的 P2P 通信次数从 $2(PP-1)$ 即 6 次增至 $2(PP \\cdot v - 1)$ 即 14 次, 约为原来的 $v$ 倍。本实验台将单次传输耗时简化记为 0, 实际集群中的净加速比还需扣除这部分通信开销。\n因此 Megatron 通常仅在 micro-batch 数 M 已受显存限制无法继续增大、同时气泡占比仍然偏高时才启用交错 1F1B; 若显存允许直接增大 M, 优先增大 M 更为经济。',
    }"
  >
    <template #controls>
      <LabSlider v-model="pp" label="流水线级数 PP" :min="2" :max="4" />
      <LabSlider v-model="groups" label="M (须为 PP 的倍数)" :min="1" :max="3" :format="(g) => g * pp" />
      <LabSlider v-model="v" label="每卡虚拟 stage 数 v" :min="1" :max="4" :format="(x) => (x === 1 ? '1 (普通 1F1B)' : x)" />
      <div class="row">
        <span class="lbl">跟踪 micro-batch:</span>
        <button v-for="m in M" :key="m" type="button" class="mb mono" :class="{ active: pinM === m - 1 }" :aria-pressed="pinM === m - 1"
                @click="pinM = pinM === m - 1 ? -1 : m - 1">{{ m - 1 }}</button>
      </div>
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
    </template>

    <div class="cells sched" :style="{ gridTemplateColumns: `40px repeat(${sim.T}, 18px)` }" @mouseleave="hoverM = -1">
      <template v-for="(row, s) in sim.grid" :key="s">
        <span class="row-label mono">卡 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t" class="cell"
          :class="[c && (c.op === 'F' ? 'f' : 'b'), { dim: t > stepper.step.value || (traceM >= 0 && (!c || c.m !== traceM)), trace: c && c.m === traceM }]"
          :style="c ? { '--k': 0.25 + 0.6 * (c.chunk + 1) / v } : null"
          :title="c ? `${c.op}${c.m} · 卡${s} 的第 ${c.chunk} 段 (虚拟 stage ${c.chunk * pp + s})` : '气泡'"
          @mouseenter="hoverM = c ? c.m : -1" @click="pinM = c && pinM !== c.m ? c.m : -1"
        >{{ c && c.head ? c.op + c.m : '' }}</span>
      </template>
    </div>

    <template #stats>
      <!-- 颜色都是和 v=1 (普通 1F1B) 的数值比出来的 -->
      <div class="kv"><span><Tex text="气泡 (卡空等的时间占比): 模拟 / 公式 $\frac{PP-1}{v \cdot M + PP - 1}$" /></span><b :class="{ good: sim.bubble < base.bubble - 1e-9 }">{{ (sim.bubble * 100).toFixed(1) }}% / {{ (formula * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>各卡在途激活峰值 (同时攥着几份, 每份是 1/{{ v }} 个 stage)</span><b>[{{ sim.peak.join(', ') }}]</b></div>
      <div class="kv"><span>折算成整 stage (卡 0, 除以 v 后和普通 1F1B 比)</span><b :class="{ bad: sim.peak[0] / v > base.peak }">{{ (sim.peak[0] / v).toFixed(1) }} <small>(v=1 时 {{ base.peak }})</small></b></div>
      <div class="kv"><span>每个 micro-batch 的 P2P 次数 (跨卡传激活和梯度的次数)</span><b :class="{ bad: 2 * (pp * v - 1) > base.p2p }">{{ 2 * (pp * v - 1) }} <small>(v=1 时 {{ base.p2p }})</small></b></div>
      <p class="lab-note">
        <Tex text="时间轴每 1 格对应 $1/v$ 个 stage 的前向时长, 反向传播 B 的耗时约为前向的 2 倍, 因此占用 2 格。★ 交错调度与普通 1F1B 的唯一区别在于热身 (warmup) 阶段更长: 卡 s 先连续执行" />
        <code class="inline">(PP−1−s)·2 + (v−1)·PP</code> 个 F, 之后才开始一 F 一 B 交替。热身越长, 攥着没反向的 micro-batch 越多, 在途激活就越多。
      </p>
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

const pp = ref(4), groups = ref(2), v = ref(2)
const hoverM = ref(-1), pinM = ref(-1)                   // 悬停的 / 点击固定的 micro-batch
const traceM = computed(() => (hoverM.value >= 0 ? hoverM.value : pinM.value))
const M = computed(() => groups.value * pp.value)
watch(M, (m) => { if (pinM.value >= m) pinM.value = -1 })
const TF = 1, TB = 2

// 卡 s 按什么顺序执行自己的 M·v 个 F 和 M·v 个 B (Megatron 交错顺序)
const deviceOrder = (s, PP, Mn, V) => {
  const fwd = [], bwd = []
  for (let g = 0; g < Mn; g += PP) {
    for (let c = 0; c < V; c++) for (let i = 0; i < Math.min(PP, Mn - g); i++) fwd.push(['F', g + i, c])
    for (let c = V - 1; c >= 0; c--) for (let i = 0; i < Math.min(PP, Mn - g); i++) bwd.push(['B', g + i, c])
  }
  // ★ warmup = 允许多少个 micro-batch "只 F 没 B"; 之后严格一 F 一 B
  const warmup = Math.min(V === 1 ? PP - 1 - s : (PP - 1 - s) * 2 + (V - 1) * PP, fwd.length)
  const order = fwd.slice(0, warmup)
  for (let f = warmup, b = 0; b < bwd.length; b++) { if (f < fwd.length) order.push(fwd[f++]); order.push(bwd[b]) }
  return order
}

const sim = computed(() => {
  const PP = pp.value, Mn = M.value, V = v.value, last = PP * V - 1
  const orders = range(PP).map((s) => deviceOrder(s, PP, Mn, V))
  const end = new Map(), free = Array(PP).fill(0), ptr = Array(PP).fill(0)
  const grid = range(PP).map(() => []), live = Array(PP).fill(0), peak = Array(PP).fill(0)
  const key = (op, m, k) => `${op}:${m}:${k}`
  for (let guard = 0; ptr.some((p, s) => p < orders[s].length) && guard < 5000; guard++) {
    for (let s = 0; s < PP; s++) {
      if (ptr[s] === orders[s].length) continue
      const [op, m, c] = orders[s][ptr[s]], k = c * PP + s // 虚拟 stage k 住在卡 k % PP 上
      const dep = op === 'F' ? (k > 0 ? key('F', m, k - 1) : null) : k < last ? key('B', m, k + 1) : key('F', m, k)
      if (dep && !end.has(dep)) continue // 依赖还没被调度, 这张卡先等
      const t0 = Math.max(free[s], dep ? end.get(dep) : 0), dur = op === 'F' ? TF : TB
      for (let d = 0; d < dur; d++) grid[s][t0 + d] = { op, m, chunk: c, head: d === 0 }
      free[s] = t0 + dur
      end.set(key(op, m, k), free[s])
      live[s] += op === 'F' ? 1 : -1
      peak[s] = Math.max(peak[s], live[s])
      ptr[s]++
    }
  }
  const T = Math.max(...free)
  grid.forEach((r) => { for (let t = 0; t < T; t++) r[t] = r[t] || null })
  return { grid, T, peak, bubble: 1 - (Mn * V * (TF + TB)) / T }
})
const formula = computed(() => (pp.value - 1) / (v.value * M.value + pp.value - 1))
// v=1 (普通 1F1B) 的三个数, 读数的颜色拿它当基准
const base = computed(() => ({ bubble: (pp.value - 1) / (M.value + pp.value - 1), peak: Math.min(pp.value, M.value), p2p: 2 * (pp.value - 1) }))

const stepper = useStepper(() => sim.value.T, { interval: 220 })
watch(sim, (s) => { stepper.pause(); stepper.step.value = s.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.sched .cell { min-width: 18px; font-size: 9px; color: var(--text); cursor: pointer; }
.lbl { font-size: 12px; color: var(--text-muted); }
.mb { min-width: 30px; min-height: 28px; padding: 2px 6px; }
.cell.f { background: color-mix(in srgb, var(--accent) calc(var(--k) * 100%), transparent); border-color: var(--accent); }
.cell.b { background: color-mix(in srgb, var(--left) calc(var(--k) * 100%), transparent); border-color: var(--left); }
.cell.trace { outline: 2px solid var(--warn); outline-offset: -1px; }
small { font-size: 10px; color: var(--text-dim); font-weight: 400; }
</style>
