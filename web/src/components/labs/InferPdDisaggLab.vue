<!-- P/D 分离实验台 (llm_infer/m15): 同卡混跑的 ITL 尖刺 vs 分离后要付的 KV 传输。全部时间来自代价模型。 -->
<template>
  <LabFrame
    title="P/D 分离 — 把 prefill 搬到另一台机器, 用一次 KV 传输换掉 decode 卡顿"
    sub="- 模拟什么: 推理分两段: prefill 把整段 prompt 一次算完 (几千 token, 吃算力), decode 之后每步只出 1 个 token (吃带宽)。几个用户正在 decode 时来了一条长 prompt, 同一张卡要先把它的 prefill 算完, 这一步里所有人的下一个 token 都得等, 用户看到的是输出突然卡住一下 (ITL 尖刺; ITL 是相邻两个输出 token 之间的间隔)。P/D 分离的做法: 一组节点 (P) 只做 prefill, 另一组 (D) 只做 decode; P 算完把这条请求的 KV cache 通过网络发给 D, D 接着往下生成。代价是 KV 要过一次网络。所有时间来自代价模型: decode 每步 25 ms, prefill 8000 tok/s, 不是实测。
      - 图上是什么: 五条时间线, 横轴是时间。第一条「同卡 GPU」: 一串绿色小格是 decode 步, 黄色一段是插进来的长 prompt 的 prefill, 这段时间里 decode 格子停了。下面四条是分离部署: 「D 节点」的 decode 格子从头到尾不断; 「P 节点」上只有那段 prefill; 「链路」上红色一段是 KV 传输; 「新请求 @ D」是这条长 prompt 的后续 token 开始在 D 节点上 decode 的时刻。竖虚线是长 prompt 到达的时刻。
      - 可以做什么: 顶上两个按钮选模型, 决定每个 token 的 KV 有多少字节 (LLaMA-3-8B 用 GQA, 128 KiB; LLaMA-2-7B 用 MHA, 512 KiB)。拖「prompt 长度」改这条长 prompt 有多少 token, prefill 时间和 KV 字节数都随它线性变。拖「链路带宽」或点预设 (10 GbE 到 NVLink) 改 P 到 D 之间的网速, 下面一行把 Gbps 换算成 GB/s 给你看。拖「链路固定延迟」加一笔和字节数无关的固定开销。按下「把 Gbps 当成 GB/s」, 右侧多出一行, 显示少除一个 8 会把传输时间算错多少。悬停或点时间线上的任一段看它的时长。
      - 应该看到: 右侧「最大 ITL: 同卡 → 分离」: 同卡那个数等于整段 prefill 加一步 decode, prompt 越长越大; 分离后恒为 25 ms, 就是一步 decode。「传 X GB KV, 第 2 个 token 推迟」是传输花的时间, 它推迟的是新请求的第 2 个 token (首 token 在 P 节点算完 prefill 时就产出了, 不受影响); 传输比它换掉的那次 prefill 卡顿短就标绿, 长就标红。「盈亏带宽」是让传输时间恰好等于那次 prefill 卡顿的网速, 当前链路比它快就划算。把链路拖到 10 Gbps 再拖长 prompt, 能看到红色传输段变得比黄色 prefill 段还长。"
    module="llm_infer/m15"
    run="python -m llm_infer.m15_pd_disaggregation.demo"
    :challenge="{
      ask: '链路延迟设为 0, 把 prompt 长度从 1K 拖到 16K, 盈亏带宽怎么变? 为什么?',
      answer: '不变, 一直是 8.4 Gbps 左右 (LLaMA-3-8B)。\n盈亏带宽的定义: 传输时间恰好等于它换掉的那次 prefill 卡顿时的网速。两边都和 prompt 长度成正比:\n- KV 字节数 = token 数 × 每 token 的 KV 字节 (128 KiB)。\n- prefill 时间 = token 数 ÷ 8000 tok/s。\n传输时间 = KV 字节数 ÷ 带宽。令它等于 prefill 时间, 两边的 token 数约掉: 盈亏带宽 = 每 token KV 字节 × prefill 速度 = 128 KiB × 8000 tok/s ≈ 1.05 GB/s。换成网卡的单位要乘 8: ≈ 8.4 Gbps。换成 LLaMA-2-7B (512 KiB/token) 就是 4 倍, 约 33.6 Gbps。普通的 100 GbE 都远高于这个数, 所以搬 KV 几乎总比让所有人等 prefill 划算。\n两个容易错的地方:\n- 单位: 网卡和 InfiniBand 都按 Gbps (bit 每秒) 标, KV 按 byte 算, 要除以 8 才是 GB/s。少除一个 8, 传输时间就被低估 8 倍, 按下「把 Gbps 当成 GB/s」能看到这个差距。\n- 推迟的是谁: 传输推迟的是新请求的第 2 个 token, 不是首 token (TTFT)。首 token 是 P 节点 prefill 的直接产物, 可以马上回给用户。\n链路固定延迟不为 0 时, 盈亏带宽会随 prompt 变短而升高: 固定延迟在总传输时间里占的比例变大了, 要靠更快的链路把剩下的部分压短。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(m, i) in MODELS" :key="m.name" type="button" :class="{ active: model === i }" @click="model = i">{{ m.name }} · {{ m.kib }} KiB/token</button>
      </div>
      <LabSlider v-model="kTok" label="prompt 长度" :min="1" :max="16" unit="K tok" />
      <LabSlider v-model="bwIdx" label="链路带宽" :min="0" :max="LINKS.length - 1" :format="() => `${gbps} Gbps`" />
      <div class="row">
        <span class="unit mono">= {{ gbps }} × 10⁹ ÷ 8 = <b>{{ (gbps / 8).toFixed(2) }} GB/s</b></span>
        <button v-for="p in PRESETS" :key="p.name" type="button" :class="{ active: gbps === p.gbps }" @click="bwIdx = LINKS.indexOf(p.gbps)">{{ p.name }}</button>
      </div>
      <LabSlider v-model="latency" label="链路固定延迟" :min="0" :max="20" unit=" ms" />
      <div class="row">
        <button type="button" :class="{ active: wrongUnit }" :aria-pressed="wrongUnit" @click="wrongUnit = !wrongUnit">把 Gbps 当成 GB/s (常见错误)</button>
      </div>
    </template>

    <svg class="tl" :viewBox="`0 0 ${W} 190`" role="group" aria-label="同卡与分离两种部署的时间线">
      <g v-for="(lane, li) in lanes" :key="lane.name">
        <text x="0" :y="lane.y + 15" class="lbl">{{ lane.name }}</text>
        <!-- decode 格子成百上千, 不进 Tab 序列; prefill 和传输这几段可以聚焦 -->
        <rect
          v-for="(s, i) in lane.segs" :key="i" :x="X0 + s.t0 * sx" :y="lane.y" :width="Math.max(1, (s.t1 - s.t0) * sx - 1)" height="22" rx="2"
          :class="['seg', s.kind, { hov: shown === `${li}-${i}` }]"
          :tabindex="s.steps ? undefined : 0" :role="s.steps ? undefined : 'button'" :aria-label="`${s.label} ${(s.t1 - s.t0).toFixed(1)} ms`"
          @mouseenter="hov = `${li}-${i}`" @mouseleave="hov = ''" @focus="hov = `${li}-${i}`" @blur="hov = ''"
          @click="pin = pin === `${li}-${i}` ? '' : `${li}-${i}`" @keydown.enter="pin = pin === `${li}-${i}` ? '' : `${li}-${i}`"
        />
      </g>
      <line :x1="X0 + arrive * sx" :x2="X0 + arrive * sx" y1="0" y2="168" class="arr" />
      <text :x="X0 + arrive * sx + 4" y="182" class="lbl">长 prompt 到达</text>
    </svg>
    <p class="lab-note hovline">{{ hovText }}</p>
    <p class="legend"><span class="sw dec" />已有用户的 decode 步 (每步 25 ms) <span class="sw pre" />长 prompt 的 prefill <span class="sw xfer" />KV 传输 <span class="sw neo" />新请求在 D 节点的 decode</p>

    <template #stats>
      <div class="kv"><span>最大 ITL (相邻两个输出 token 的最长间隔): 同卡 → 分离</span><b :class="STEP < coloItl ? 'good' : ''">{{ fmtMs(coloItl) }} → {{ fmtMs(STEP) }}</b></div>
      <div class="kv"><span>传 {{ kvGB.toFixed(2) }} GB 的 KV 要多久, 新请求的第 2 个 token 推迟</span><b :class="xfer < prefillMs ? 'good' : 'bad'">+{{ fmtMs(xfer) }}</b></div>
      <div v-if="wrongUnit" class="kv"><span>错算的传输时间 (忘了 ÷8)</span><b :class="xferWrong < xfer ? 'bad' : ''">{{ fmtMs(xferWrong) }}</b></div>
      <div class="kv"><span>盈亏带宽 (传输时间恰好等于那次 prefill 卡顿)</span><b>{{ isFinite(breakEven) ? breakEven.toFixed(1) + ' Gbps' : '∞' }}</b></div>
      <div class="lab-note">
        <p>同卡的最大 ITL = 整段 prefill ({{ fmtMs(prefillMs) }}) + 一步 decode (25 ms): 上一个 token 出来后, 要等 prefill 算完、再算一步 decode, 下一个 token 才出来。分离后 D 节点不受打扰, 最大 ITL 就是一步 decode。</p>
        <p>盈亏带宽: 传输耗时恰好等于它换掉的那次 prefill 卡顿 ({{ fmtMs(prefillMs) }}) 时的网速。当前链路是它的 {{ (gbps / breakEven).toFixed(1) }}×, 大于 1 就说明搬 KV 比让大家等 prefill 划算。</p>
        <p>TTFT (首 token 延迟) 两种部署相同: 首 token 在 P 节点 prefill 结束时就产出了, 传输只推迟第 2 个 token。</p>
        <p v-if="wrongUnit">忘了 ÷8 会把传输时间低估 8 倍: 错算 {{ fmtMs(xferWrong) }} vs 真实 {{ fmtMs(xfer) }}。网卡和 InfiniBand 都按 bit 标速, KV 按 byte 算。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range } from '@/utils/labmath.js'

const W = 720, X0 = 104, STEP = 25, PREFILL_TPS = 8000, WARM = 3, USERS_AFTER = 5
const MODELS = [{ name: 'LLaMA-3-8B (GQA)', kib: 128 }, { name: 'LLaMA-2-7B (MHA)', kib: 512 }]
const LINKS = [10, 25, 50, 100, 200, 400, 800, 3600]
const PRESETS = [{ name: '10 GbE', gbps: 10 }, { name: '100 GbE', gbps: 100 }, { name: '400G IB', gbps: 400 }, { name: 'NVLink ≈3600', gbps: 3600 }]
const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : ms.toFixed(ms < 10 ? 1 : 0) + ' ms')

const model = ref(0)
const kTok = ref(4)
const bwIdx = ref(3)
const latency = ref(1)
const wrongUnit = ref(false)
const hov = ref(''), pin = ref('')
const shown = computed(() => hov.value || pin.value)   // 悬停优先; 点一下固定, 触屏和键盘也能看

const gbps = computed(() => LINKS[bwIdx.value])
const kvBytes = computed(() => kTok.value * 1024 * MODELS[model.value].kib * 1024)
const kvGB = computed(() => kvBytes.value / 1e9)
const prefillMs = computed(() => (kTok.value * 1024 / PREFILL_TPS) * 1e3)
// ★ 单位: 带宽标的是 bit/s, KV 是 byte → bytes/s = Gbps · 1e9 / 8 (与 m15 KVLink.transfer_ms 相同)
const xfer = computed(() => latency.value + (kvBytes.value / (gbps.value * 1e9 / 8)) * 1e3)
const xferWrong = computed(() => latency.value + (kvBytes.value / (gbps.value * 1e9)) * 1e3)
const coloItl = computed(() => prefillMs.value + STEP) // 上一个 token → (整段 prefill) → 下一个 decode 步结束
const breakEven = computed(() => (prefillMs.value > latency.value ? (kvBytes.value * 8) / ((prefillMs.value - latency.value) / 1e3) / 1e9 : Infinity))

const arrive = WARM * STEP
const pEnd = computed(() => arrive + prefillMs.value)
const xEnd = computed(() => pEnd.value + xfer.value)
const nD = computed(() => Math.ceil((xEnd.value + USERS_AFTER * STEP) / STEP))   // D 节点那一行最长, 它定横轴
const sx = computed(() => (W - X0) / (nD.value * STEP))
const lanes = computed(() => {
  const name = (kind) => (kind === 'neo' ? '新请求 decode 步' : 'decode 步')
  // 一格不到 3px 就看不清也点不中: 合并成一条色带
  const dec = (t0, n, kind = 'dec') => (STEP * sx.value < 3
    ? [{ t0, t1: t0 + n * STEP, kind, steps: n, label: `${n} 个 ${name(kind)}, 每步 ${STEP} ms, 合计` }]
    : range(n).map((i) => ({ t0: t0 + i * STEP, t1: t0 + (i + 1) * STEP, kind, steps: 1, label: name(kind) })))
  return [
    { name: '同卡 GPU', y: 6, segs: [...dec(0, WARM), { t0: arrive, t1: pEnd.value, kind: 'pre', label: 'prefill (所有 decode 停住)' }, ...dec(pEnd.value, USERS_AFTER)] },
    { name: '分离 · D 节点', y: 50, segs: dec(0, nD.value) },
    { name: '分离 · P 节点', y: 80, segs: [{ t0: arrive, t1: pEnd.value, kind: 'pre', label: 'prefill → 首 token' }] },
    { name: '分离 · 链路', y: 110, segs: [{ t0: pEnd.value, t1: xEnd.value, kind: 'xfer', label: `KV 传输 ${kvGB.value.toFixed(2)} GB` }] },
    { name: '新请求 @ D', y: 140, segs: dec(xEnd.value, 3, 'neo') },
  ]
})
// 悬停文字从当前 lanes 现算: 拖滑杆后不会留下过期的数字
const hovText = computed(() => {
  const [li, i] = shown.value.split('-').map(Number), lane = lanes.value[li], s = lane?.segs[i]
  return s ? `${lane.name} · ${s.label}: ${(s.t1 - s.t0).toFixed(1)} ms` : '悬停或点时间线上的任一段, 这里显示它是什么、用了多少毫秒。'
})
</script>

<style scoped>
.tl { min-width: 560px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.seg { stroke: var(--bg-card); outline: none; }
.seg.hov, .seg:focus-visible { stroke: var(--text); stroke-width: 1.5; }
.dec { fill: color-mix(in srgb, var(--left) 60%, transparent); background: var(--left); }
.pre { fill: var(--warn); background: var(--warn); }
.xfer { fill: var(--danger); background: var(--danger); }
.neo { fill: var(--accent); background: var(--accent); }
.arr { stroke: var(--text-dim); stroke-dasharray: 3 3; }
.lbl { font-size: 12px; fill: var(--text-muted); }
.unit { font-size: 12px; color: var(--text-muted); }
.unit b { color: var(--accent); }
.hovline { margin-top: 8px; min-height: 1.7em; }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
</style>
