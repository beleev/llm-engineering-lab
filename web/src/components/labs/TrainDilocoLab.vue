<!--
  DiLoCo 实验台 (对应 llm_train/m28_diloco:run_diloco)。
  只讲一件事: worker 各跑 H 步再同步一次, 外层把 "旧参数 - 新平均参数" 当梯度走一步动量 SGD。
  左边是前端现算的 2 维玩具, 右下角的表是 m28 在 16 维回归上跑出来的实测值, 不随滑杆变化。
-->
<template>
  <LabFrame
    title="DiLoCo — 本地跑 H 步再同步一次, 通信少 H 倍"
    sub="- 模拟什么: 跨数据中心训练时, DDP 每走一步就要 all-reduce 一遍梯度, 带宽和延迟撑不住。DiLoCo 让每个 worker 用自己的数据连着跑 H 步本地优化, 只在每 H 步末尾同步一次参数: 先把各 worker 的参数取平均, 再把「同步前的旧参数减去这个平均值」当成一个梯度 (论文叫伪梯度) 交给外层的 Nesterov 动量 SGD 走一步。同步次数因此降到 1/H。
- 图上是什么: 一个 2 维参数平面, 两个坐标轴就是两个参数:
  - 灰色椭圆是损失的等高线, 中心的十字是最优点。各 worker 的梯度在真梯度上加了自己那一份噪声, 对应各自不同的数据分片。
  - 紫色粗线是外层参数的轨迹, 每个圆点是一次同步。
  - 细的绿色线是当前这一轮里各 worker 的本地轨迹, 末端的小方块是它们各自跑完 H 步到达的位置。
  - 黄色虚线连着「本地平均位置」和「外层走完一步之后的位置」, 它的长度就是外层步长放大的效果。
  - 起点上的大圆点可以拖动。
- 可以做什么: 控件从上到下:
  - 「DDP (每步同步)」: 等价于 H=1 且外层退化成直接取平均。
  - 「DiLoCo」: 用论文的外层超参 (lr 0.7, 动量 0.9)。
  - 拖动「本地步数 H」: 总步数固定 240 步, H 越大同步越少。
  - 拖动「外层学习率」和「外层动量」: 看外层步长怎么被动量放大。
  - 播放控件: 一轮一轮地看同步前的发散和同步后的收束。
  - 拖图上的大圆点换起点。
- 应该看到: H 调大时紫色轨迹上的点变少 (同步次数少), 每两点之间的绿色本地轨迹变长、彼此散得更开。H 很小时外层动量会让紫色轨迹冲过最优点来回震荡 —— 外层的有效步长约为 lr/(1−动量), 动量 0.9 就是放大 10 倍。右侧读数:\n  - 「同步次数 / 通信量」: 240/H 次, 通信量按同步次数计, 与 DDP 的比值直接显示。
  - 「当前 loss」: 外层参数在等高线上的损失值。
  - 「本地漂移」: 同步前各 worker 与它们平均位置的最大距离, H 越大漂得越远。"
    module="llm_train/m28"
    run="python -m llm_train.m28_diloco.demo"
    :challenge="{
      ask: '把「本地步数 H」拖到 1, 保持外层 lr 0.7、动量 0.9, 先预测:\n- 紫色轨迹会更快到最优点, 还是反而冲过去?\n- 为什么 H=1 不等于 DDP?\n- 要让 H=1 真的退回 DDP, 外层的两个旋钮该设成多少?',
      answer: '答案: 会冲过去来回震荡; H=1 不等于 DDP, 因为外层又加了一层带动量的更新。\n逐步看:\n1. H=1 时每个 worker 只走一步, 本地平均位置就是 DDP 走一步会到的地方。如果外层照搬这个位置就结束, 那确实等于 DDP。\n2. 但外层拿到的伪梯度 $\\Delta$ 正好是这一步的位移, 然后它用 lr 0.7、动量 0.9 再走一次。Nesterov 的有效步长约为 $\\text{lr}/(1-\\mu)=0.7/0.1=7$, 也就是把这一步放大了 7 倍。步长大 7 倍的梯度下降在这个病态椭圆上会震荡。\n3. m28 在 16 维回归上量到的就是这件事: 同样 960 步, H=16 的留出集 MSE 是 3.6e-10, H=1 是 1.2 (比全 0 预测的 4.46 只好一点)。\n4. 要退回 DDP, 外层 lr 设 1、动量设 0: 这时 $\\theta \\leftarrow \\theta - 1 \\cdot (\\theta - \\bar\\theta) = \\bar\\theta$, 外层就是一个恒等映射。m28 的断言测的正是这一条, 和手写 DDP 的差只有 1.8e-07 (fp32 的求和顺序差异)。\n反过来 H 太大也不行: m28 里 H=240 (只同步 4 次) 的 MSE 回升到 2.9e-03。所以 H 有一个中间的好区间, 论文在 1.5 亿参数、8 个 worker 上取 500。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'ddp' }" @click="mode = 'ddp'">DDP (每步同步)</button>
        <button type="button" :class="{ active: mode === 'diloco' }" @click="mode = 'diloco'">DiLoCo (外层动量)</button>
      </div>
      <LabSlider v-model="hIdx" label="本地步数 H" :min="0" :max="5" :step="1" :format="(v) => HS[v] + ' 步'" />
      <LabSlider v-model="lrOut" label="外层学习率" :min="0.1" :max="1.2" :step="0.1" />
      <LabSlider v-model="mu" label="外层动量" :min="0" :max="0.95" :step="0.05" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value + 1} / ${run.rounds.length} 次同步`" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${VW} ${VH}`" role="group" aria-label="DiLoCo 在二维参数平面上的轨迹, 起点圆点可拖">
      <ellipse v-for="r in [0.3, 0.6, 1.0, 1.5]" :key="r" :cx="sx(0)" :cy="sy(0)"
        :rx="(sx(r / Math.sqrt(A0)) - sx(0))" :ry="(sy(0) - sy(r / Math.sqrt(A1)))" class="iso" />
      <line :x1="sx(-0.12)" :x2="sx(0.12)" :y1="sy(0)" :y2="sy(0)" class="star" />
      <line :x1="sx(0)" :x2="sx(0)" :y1="sy(-0.12)" :y2="sy(0.12)" class="star" />
      <polyline v-for="(path, i) in cur.locals" :key="'l' + i" :points="path.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')" class="local" />
      <rect v-for="(path, i) in cur.locals" :key="'e' + i" :x="sx(path[path.length - 1][0]) - 3" :y="sy(path[path.length - 1][1]) - 3" width="6" height="6" class="end" />
      <line :x1="sx(cur.avg[0])" :x2="sx(cur.after[0])" :y1="sy(cur.avg[1])" :y2="sy(cur.after[1])" class="outer" />
      <polyline :points="run.rounds.slice(0, stepper.step.value + 2).map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')" class="traj" />
      <circle v-for="(p, i) in run.rounds.slice(0, stepper.step.value + 2)" :key="'p' + i" :cx="sx(p[0])" :cy="sy(p[1])" r="2.5" class="node" />
      <g class="draggable" tabindex="0" role="slider" aria-label="起点的第一个参数" :aria-valuenow="start0.toFixed(2)"
        @pointerdown="startDrag($event, { svg, onMove: ({ x, y }) => { start0 = clamp(sxInv(x), -2.4, 2.4); start1 = clamp(syInv(y), -2.4, 2.4) } })"
        @keydown.right.prevent.stop="start0 = clamp(start0 + 0.1, -2.4, 2.4)" @keydown.left.prevent.stop="start0 = clamp(start0 - 0.1, -2.4, 2.4)">
        <circle :cx="sx(start0)" :cy="sy(start1)" r="7" class="dot" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>同步次数 (总 240 步)</span><b>{{ run.rounds.length - 1 }}</b></div>
      <div class="kv"><span>通信量 (与 DDP 每步同步相比)</span><b :class="H > 1 ? 'good' : 'bad'">1 / {{ H }}</b></div>
      <div class="kv"><span>最后一次同步后的 loss</span><b :class="finalLoss < startLoss / 100 ? 'good' : 'bad'">{{ finalLoss.toExponential(2) }}</b></div>
      <div class="kv"><span>本轮同步前的本地漂移 (最远的 worker 离平均有多远)</span><b>{{ cur.drift.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p><Tex text="★ 外层那一步: $\Delta = \theta_{\text{旧}} - \bar\theta$, 然后 $\theta \leftarrow \theta - \mathrm{lr}\cdot(\Delta + \mu \cdot \mathrm{buf})$。把外层 lr 设 1、动量设 0 时这一步化简成 $\theta \leftarrow \bar\theta$, DiLoCo 就退回成 DDP。" /></p>
        <p>m28 实测 (16 维回归, 960 步, 4 个 worker, 内层 AdamW lr 0.02, 外层 lr 0.7 动量 0.9; 不随左侧变化):</p>
        <ul class="pts">
          <li><b>DDP:</b> 留出集 MSE 9.9e-14, 同步 960 次。</li>
          <li><b>H=16 / 48:</b> 3.6e-10 / 1.4e-11, 通信少 16 / 48 倍。</li>
          <li><b>H=1 / 4:</b> 1.2 / 1.3, 被外层动量放大后走坏了。</li>
          <li><b>H=120 / 240:</b> 2.7e-04 / 2.9e-03, 同步太少跟不上。</li>
        </ul>
        <p>论文在 1.5 亿参数、C4、8 个 worker 上取 H=500, 通信次数是每步同步的 1/500, 困惑度与每步同步持平。Streaming DiLoCo 进一步把参数分片错峰同步: 总通信量不变, 单次只发其中一片, 峰值带宽降到 1/片数。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32 } from '@/utils/labmath.js'

const HS = [1, 4, 12, 24, 60, 120]        // 都能整除 240
const TOTAL = 240, WORLD = 4
const A0 = 1.0, A1 = 0.12                 // 病态二次曲面: 两个方向的曲率差约 8 倍
const LR_IN = 0.25

const mode = ref('diloco')
const hIdx = ref(2), lrOut = ref(0.7), mu = ref(0.9)
const start0 = ref(-2.0), start1 = ref(1.8)
const svg = ref(null)
const { start: startDrag } = useDrag()

const H = computed(() => HS[hIdx.value])
const loss = (p) => 0.5 * (A0 * p[0] ** 2 + A1 * p[1] ** 2)
// DiLoCo 的一轮: 各 worker 从外层参数出发跑 H 步带噪声的 SGD, 取平均, 外层再走一步
const run = computed(() => {
  const h = mode.value === 'ddp' ? 1 : H.value
  const lr = mode.value === 'ddp' ? 1 : lrOut.value
  const m = mode.value === 'ddp' ? 0 : mu.value
  let theta = [start0.value, start1.value]
  const buf = [0, 0]
  const rounds = [theta.slice()], detail = []
  for (let r = 0; r < TOTAL / h; r++) {
    const rand = mulberry32(r * 97 + 13)
    const locals = []
    for (let w = 0; w < WORLD; w++) {
      const path = [theta.slice()]
      let p = theta.slice()
      for (let s = 0; s < h; s++) {
        // 各 worker 的梯度 = 真梯度 + 自己那一份噪声 (对应不同的数据分片)
        const n0 = (rand() - 0.5) * 0.5, n1 = (rand() - 0.5) * 0.5
        p = [p[0] - LR_IN * (A0 * p[0] + n0), p[1] - LR_IN * (A1 * p[1] + n1)]
        path.push(p)
      }
      locals.push(path)
    }
    const ends = locals.map((pa) => pa[pa.length - 1])
    const avg = [0, 1].map((k) => ends.reduce((a, e) => a + e[k], 0) / WORLD)
    const before = theta.slice()
    for (const k of [0, 1]) {
      const delta = theta[k] - avg[k]                    // ★ 伪梯度: 旧参数 - 新的平均参数
      buf[k] = m * buf[k] + delta
      theta[k] -= lr * (delta + m * buf[k])
    }
    detail.push({
      locals, avg, after: theta.slice(), before,
      drift: Math.max(...ends.map((e) => Math.hypot(e[0] - avg[0], e[1] - avg[1]))),
    })
    rounds.push(theta.slice())
    if (!Number.isFinite(theta[0]) || Math.abs(theta[0]) > 1e3) break
  }
  return { rounds, detail }
})

const stepper = useStepper(() => run.value.detail.length, { interval: 260 })
const cur = computed(() => run.value.detail[Math.min(stepper.step.value, run.value.detail.length - 1)])
const startLoss = computed(() => loss([start0.value, start1.value]))
const finalLoss = computed(() => loss(run.value.rounds[run.value.rounds.length - 1]))

const VW = 520, VH = 260
const sx = (v) => VW / 2 + (v / 2.6) * (VW / 2 - 16)
const sy = (v) => VH / 2 - (v / 2.6) * (VH / 2 - 14)
const sxInv = (x) => ((x - VW / 2) / (VW / 2 - 16)) * 2.6
const syInv = (y) => ((VH / 2 - y) / (VH / 2 - 14)) * 2.6
watch([run, mode], () => { stepper.pause(); stepper.step.value = run.value.detail.length - 1 }, { immediate: true })
</script>

<style scoped>
.iso { fill: none; stroke: var(--border-strong); }
.star { stroke: var(--text-muted); stroke-width: 2; }
.local { fill: none; stroke: var(--left); stroke-width: 1; opacity: 0.8; }
.end { fill: var(--left); }
.outer { stroke: var(--warn); stroke-width: 1.5; stroke-dasharray: 3 3; }
.traj { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.node { fill: var(--accent); }
.dot { fill: var(--eye); stroke: var(--bg); stroke-width: 2; }
</style>
