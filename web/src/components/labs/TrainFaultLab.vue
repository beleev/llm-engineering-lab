<!--
  checkpoint 间隔与 goodput 实验台 (对应 llm_train/m27_fault_tolerance:goodput_model)。
  只讲一件事: 存得太勤亏在落盘上, 存得太稀亏在重跑上, 最优间隔由落盘耗时和整体 MTBF 定。
  曲线用解析式现算; m27 的蒙特卡洛在同一组参数上与它差不到 0.4 个百分点。
-->
<template>
  <LabFrame
    title="checkpoint 间隔 — 卡越多挂得越勤, 多久存一次盘才最划算"
    sub="- 模拟什么: 千卡训练任务里挂掉是常态。N 张卡各自独立出故障时, 任意一张挂掉整个任务就得停, 所以整体 MTBF (mean time between failures, 平均无故障间隔) 等于单卡 MTBF 除以 N。每隔一段时间把参数、优化器状态和数据游标落盘一次 (checkpoint), 挂掉之后从上一份 checkpoint 重跑。这个台子算的是 goodput: 有效训练时间占墙钟时间的比例。
- 图上是什么: 横轴是两次 checkpoint 之间的纯计算时间 (对数刻度, 分钟), 纵轴是 goodput。
  - 实线是解析模型算出的 goodput 曲线, 呈单峰: 左边太勤, 时间花在落盘上; 右边太稀, 每次挂掉要重跑很长一段。
  - 绿色竖虚线是 Young 1974 的一阶最优间隔 $\tau=\sqrt{2\delta M}$, 黄色竖虚线是 Daly 的修正式 $\tau=\sqrt{2\delta(M+R)}-\delta$。
  - 可拖动的大圆点是你当前选的间隔。
  - 顶部横虚线是这条曲线的峰值 goodput。
- 可以做什么: 控件从上到下:
  - 「同步落盘」: 阻塞 10 分钟把状态写进共享存储, 这段时间谁都不能算。
  - 「异步快照」: 先花 30 秒把状态拷进内存, 落盘在后台做, 阻塞时间只剩快照那一下。
  - 拖动「卡数 N」: 单卡 MTBF 固定 10 万小时, 整体 MTBF 随 N 变短。
  - 拖动「重启恢复 R」: 从挂掉到真正开始算之间的开销 (重新拉起进程、读 checkpoint)。
  - 「放到 Young 间隔」「放到 Daly 间隔」: 把圆点放到两条公式给出的位置。
  - 在图上左右拖圆点 (或用左右方向键) 试别的间隔。
- 应该看到: 卡数从 1024 拖到 16384 时整体 MTBF 从 97.7 小时掉到 6.1 小时, 曲线整体下移、峰值左移。切到异步快照后峰值明显抬高。右侧读数:
  - 「整体 MTBF」: 单卡 MTBF ÷ N。
  - 「Young / Daly 间隔」: 两条公式的结果, 通常很接近。
  - 「当前 goodput」: 你选的间隔下的有效训练时间占比; 和峰值差不到 1 个百分点时显示绿色。
  - 「两笔开销」: 落盘占比 $\delta/\tau$ 与重跑占比 $\tau/(2M)$。Young 的最优间隔正是让这两个数相等的那一点。"
    module="llm_train/m27"
    run="python -m llm_train.m27_fault_tolerance.demo"
    :challenge="{
      ask: '把卡数从 1024 拖到 16384 (整体 MTBF 从 97.7 小时降到 6.1 小时), 保持同步落盘 10 分钟, 先预测:\n- 最优间隔会变长还是变短, 大约多少倍?\n- 峰值 goodput 会掉到多少?\n- 换成异步快照 (阻塞 30 秒) 能把它救回多少?',
      answer: '答案: 最优间隔缩短约 4 倍 (Young 间隔从 342 分钟到 86 分钟), 峰值 goodput 从 94.1% 掉到 76.3%; 换异步快照能回到 92.3%。\n逐步看:\n1. Young 的式子是 $\\tau=\\sqrt{2\\delta M}$: $\\delta$ 不变、$M$ 变成 1/16 时, $\\tau$ 变成 $1/\\sqrt{16}=1/4$。所以 342 分钟变 86 分钟, 不是变 1/16。\n2. 把 $\\tau$ 代回两笔开销: 落盘占比 $\\delta/\\tau$ 和重跑占比 $\\tau/(2M)$ 都正比于 $\\sqrt{\\delta/M}$, 所以 $M$ 缩到 1/16 时两笔各涨 4 倍。原来一共亏 5.9%, 现在亏 23.7%。\n3. 想把这笔亏压回去, 只有两条路: 让 $M$ 变大 (更可靠的硬件、更快的故障隔离), 或者让 $\\delta$ 变小。$\\delta$ 从 10 分钟降到 30 秒是 1/20, $\\sqrt{1/20}\\approx 0.22$, 两笔开销同比缩到 22%。\n4. 这也解释了异步 checkpoint 为什么值得做: 它不改变总共要写多少字节, 只把阻塞的那一段挪到后台, 而公式里只有阻塞时间说话。\n边界: 曲线假设故障按泊松过程到达, 且后台落盘总能在下次故障前写完。真实集群有相关故障 (一个机架、一次固件升级同时带走很多卡), 这时实际 goodput 比曲线低。Llama 3 405B 的 54 天预训练快照里有 466 次中断 (47 次计划内、419 次意外), 报告的有效训练时间高于 90%, 靠的是自动化恢复而不是更长的间隔。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: dumpMin === 10 }" @click="dumpMin = 10">同步落盘 (阻塞 10 分钟)</button>
        <button type="button" :class="{ active: dumpMin === 0.5 }" @click="dumpMin = 0.5">异步快照 (阻塞 30 秒)</button>
      </div>
      <LabSlider v-model="lgN" label="卡数 N" :min="3" :max="15" :step="1" :format="(v) => 2 ** v + ' 张'" />
      <LabSlider v-model="restartMin" label="重启恢复 R" :min="0" :max="30" :step="1" unit=" 分钟" />
      <div class="row">
        <button type="button" @click="lgTau = Math.log10(young)">放到 Young 间隔</button>
        <button type="button" @click="lgTau = Math.log10(daly)">放到 Daly 间隔</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="goodput 随 checkpoint 间隔变化的单峰曲线, 曲线上的圆点可拖">
      <g v-for="g in [0.2, 0.4, 0.6, 0.8, 1.0]" :key="'y' + g">
        <line :x1="X0" :x2="W - 10" :y1="py(g)" :y2="py(g)" class="grid" />
        <text :x="X0 - 5" :y="py(g) + 3" class="tick" text-anchor="end">{{ (g * 100).toFixed(0) }}%</text>
      </g>
      <g v-for="t in [5, 15, 60, 240, 960]" :key="'x' + t">
        <text :x="px(Math.log10(t))" :y="H - 6" class="tick" text-anchor="middle">{{ t }}分</text>
      </g>
      <line :x1="X0" :x2="W - 10" :y1="py(peak.gp)" :y2="py(peak.gp)" class="floor" />
      <text :x="W - 12" :y="py(peak.gp) - 4" class="tick" text-anchor="end">峰值 {{ (peak.gp * 100).toFixed(1) }}%</text>
      <polyline :points="curve" class="curve" />
      <line :x1="px(Math.log10(young))" :x2="px(Math.log10(young))" :y1="py(0.02)" :y2="py(peak.gp)" class="mark young" />
      <line :x1="px(Math.log10(daly))" :x2="px(Math.log10(daly))" :y1="py(0.02)" :y2="py(peak.gp)" class="mark daly" />
      <text :x="px(Math.log10(young))" y="14" class="tick" text-anchor="middle">Young</text>
      <text :x="px(Math.log10(daly))" y="26" class="tick" text-anchor="middle">Daly</text>
      <g class="draggable" tabindex="0" role="slider" aria-label="checkpoint 间隔 (分钟)" :aria-valuenow="Math.round(tau)"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (lgTau = clamp(pxInv(x), LO, HI)) })"
        @keydown.right.prevent.stop="lgTau = clamp(lgTau + 0.04, LO, HI)" @keydown.left.prevent.stop="lgTau = clamp(lgTau - 0.04, LO, HI)">
        <circle :cx="px(lgTau)" :cy="py(goodput(tau))" r="7" class="dot" />
        <text :x="px(lgTau) + (lgTau > 2.2 ? -10 : 10)" :y="py(goodput(tau)) - 10" class="hl" :text-anchor="lgTau > 2.2 ? 'end' : 'start'">{{ Math.round(tau) }} 分钟</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>整体 MTBF (单卡 10 万小时 ÷ N)</span><b :class="mtbfHours > 24 ? 'good' : 'bad'">{{ mtbfHours.toFixed(1) }} 小时</b></div>
      <div class="kv"><span>Young 间隔 / Daly 间隔</span><b>{{ young.toFixed(0) }} / {{ daly.toFixed(0) }} 分钟</b></div>
      <div class="kv"><span>当前 goodput (峰值 {{ (peak.gp * 100).toFixed(1) }}% 在 {{ peak.tau.toFixed(0) }} 分钟)</span><b :class="peak.gp - goodput(tau) < 0.01 ? 'good' : 'bad'">{{ (goodput(tau) * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>两笔开销: 落盘 δ/τ / 重跑 τ/(2M)</span><b>{{ (dumpMin / tau * 100).toFixed(1) }}% / {{ (tau / (2 * mtbfHours * 60) * 100).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p><Tex text="★ 最优点的来历: 落盘占掉 $\delta/\tau$, 挂掉重跑平均丢半段也就是 $\tau/(2M)$。两者相加在 $\tau=\sqrt{2\delta M}$ 处最小, 而且那一点上两笔正好相等, 把圆点放到 Young 间隔就能看到右侧两个数一样。" /></p>
        <p><Tex text="曲线用的是解析式 $\mathrm{goodput} = \tau / (M e^{R/M}(e^{(\tau+\delta)/M}-1))$。m27 的蒙特卡洛在八组参数上与它差最多 0.39 个百分点, 所以这条曲线和 Python 算的是同一件事。" /></p>
        <p>m27 固定实测读数 (不随左侧变化): 整体 MTBF 6 小时、重启 10 分钟时, 同步落盘 10 分钟的最优 goodput 为 75.9% (Daly 间隔 76 分钟), 异步快照 30 秒为 92.2% (Daly 间隔 19 分钟)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const GPU_MTBF_H = 100_000            // 单卡 MTBF, 小时
const LO = Math.log10(3), HI = Math.log10(1600)   // 间隔范围 3 到 1600 分钟
const lgN = ref(14), dumpMin = ref(10), restartMin = ref(10), lgTau = ref(Math.log10(90))
const svg = ref(null)
const { start } = useDrag()

const mtbfHours = computed(() => GPU_MTBF_H / 2 ** lgN.value)
const tau = computed(() => 10 ** lgTau.value)
// 全部换成分钟; goodput = tau / (M·e^(R/M)·(e^((tau+delta)/M) - 1)), 与 m27 的 goodput_model 同式
const goodput = (t) => {
  const M = mtbfHours.value * 60
  return t / (M * Math.exp(restartMin.value / M) * (Math.exp((t + dumpMin.value) / M) - 1))
}
const young = computed(() => Math.sqrt(2 * dumpMin.value * mtbfHours.value * 60))
const daly = computed(() => Math.sqrt(2 * dumpMin.value * (mtbfHours.value * 60 + restartMin.value)) - dumpMin.value)
// 在同一个对数网格上取峰值, 和曲线看到的是同一批点
const grid = computed(() => range(241).map((i) => 10 ** (LO + (i / 240) * (HI - LO))))
const peak = computed(() => grid.value.reduce((best, t) => (goodput(t) > best.gp ? { tau: t, gp: goodput(t) } : best), { tau: 0, gp: 0 }))

const W = 560, H = 250, X0 = 40
const px = (lg) => X0 + ((lg - LO) / (HI - LO)) * (W - X0 - 14)
const pxInv = (x) => LO + ((x - X0) / (W - X0 - 14)) * (HI - LO)
const py = (v) => 14 + (1 - clamp(v, 0, 1)) * (H - 38)
const curve = computed(() => grid.value.map((t) => `${px(Math.log10(t))},${py(goodput(t))}`).join(' '))
</script>

<style scoped>
.grid { stroke: var(--border); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.floor { stroke: var(--text-dim); stroke-dasharray: 2 4; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.mark { stroke-dasharray: 4 3; }
.mark.young { stroke: var(--left); }
.mark.daly { stroke: var(--warn); }
.dot { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.hl { font-size: 11px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; }
</style>
