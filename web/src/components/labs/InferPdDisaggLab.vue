<!-- P/D 分离实验台 (llm_infer/m15): 同卡混跑的 ITL 尖刺 vs 分离后要付的 KV 传输。全部时间来自代价模型。 -->
<template>
  <LabFrame
    title="P/D 分离 — 把 prefill 搬到另一台机器, 用一次 KV 传输换掉 decode 卡顿"
    sub="- 模拟什么: 大模型推理包含特征迥异的两个阶段: prefill 阶段一次性并行计算整段 prompt (常达数千 token, 属于算力受限阶段), decode 阶段每步仅自回归生成 1 个 token (属于访存带宽受限阶段)。在同卡混合部署下, 若多条请求正在 decode 时插入一条长 prompt, GPU 必须先完成该长请求的 prefill, 导致同批次所有活跃请求的下一个输出 token 被迫阻塞等待, 用户侧表现为流式输出突然停顿 (即 ITL 延迟毛刺; ITL 为相邻两个输出 token 之间的时间间隔)。本实验所有耗时均基于代价模型计算 (decode 每步固定 25 ms, prefill 吞吐率 8000 tok/s), 并非真机实测。P/D 分离架构的解决思路如下:
  - P 节点: 专职执行 prefill 计算, 完成后将该请求生成的 KV cache 经网络发送至 D 节点。
  - D 节点: 专职执行 decode 迭代, 接收 KV cache 后继续自回归生成后续 token。
  - 工程代价: 整段 prompt 的 KV cache 需要跨节点传输一次。
- 图上是什么: 五条对齐的时间轴, 横轴为时间, 竖直虚线标示长 prompt 到达的时刻; 第一条对应同卡混部基线, 下方四条对应 P/D 分离部署:
  - 「同卡 GPU」: 连续绿色小格代表正常 decode 步, 中间插入的黄色区段代表长 prompt 的 prefill 计算, 期间所有 decode 步被迫中断。
  - 「D 节点」: 绿色 decode 步全程均匀推进、不受新到达长请求的干扰。
  - 「P 节点」: 仅执行该长 prompt 的黄色 prefill 区段。
  - 「链路」: 红色区段代表 KV cache 跨网络传输的耗时。
  - 「新请求 @ D」: 标示该长请求在 KV 传输完毕后于 D 节点开始 decode 的时刻。
- 可以做什么: 按界面从上到下依次操作:
  - 点击顶部两个模型预设按钮切换单 token KV 体积 (LLaMA-3-8B 采用 GQA, 每 token 占 128 KiB; LLaMA-2-7B 采用 MHA, 每 token 占 512 KiB)。
  - 拖动「prompt 长度」调整新到达长请求的 token 数量, prefill 耗时与待传 KV 字节数均随其线性变化。
  - 拖动「链路带宽」或点击网络预设 (从 10 GbE 到 NVLink) 调整 P 节点至 D 节点的互联速率, 下方同步显示 Gbps 换算为 GB/s 后的有效字节带宽。
  - 拖动「链路固定延迟」叠加与数据量无关的单次通信固定开销。
  - 勾选「把 Gbps 当成 GB/s」, 右侧会新增一行对比读数, 展示漏除以 8 时对传输耗时的低估程度。
  - 悬停或点击时间轴上的任意区段查看具体耗时。
- 应该看到: 右侧「最大 ITL: 同卡 → 分离」读数中, 同卡混部的最大 ITL 等于整段 prefill 耗时加上一步 decode, 随 prompt 变长而线性增大; 分离部署下则恒定为一步 decode 的 25 ms。将链路带宽拖至 10 Gbps 并拉长 prompt 时, 可见红色 KV 传输区段长于黄色 prefill 区段。其余两项读数含义如下:
  - 「传 X GB KV, 第 2 个 token 推迟」表示网络传输耗时, 该耗时仅推迟新请求第 2 个输出 token 的生成时刻 (首 token 已在 P 节点完成 prefill 时直接产出, 不受传输影响); 当传输耗时短于同卡混部下造成的 prefill 阻塞时间时标绿, 反之标红。
  - 「盈亏带宽」表示使 KV 传输耗时恰好等于该次 prefill 计算耗时的临界网络速率, 当前链路带宽高于该值时跨网传输在时间账上占优。"
    module="llm_infer/m15"
    run="python -m llm_infer.m15_pd_disaggregation.demo"
    :challenge="{
      ask: '将「链路固定延迟」设为 0, 并将「prompt 长度」从 1K 拖至 16K, 「盈亏带宽」会如何变化? 原因是什么?',
      answer: '盈亏带宽保持不变, 在 LLaMA-3-8B 预设下恒定约为 8.4 Gbps。\n盈亏带宽定义为使 KV 传输耗时恰好等于同卡 prefill 阻塞耗时所需的临界带宽。在零固定延迟下, 传输数据量与 prefill 耗时均与 prompt 长度成严格正比:\n- 待传 KV 字节数等于 token 数乘以单 token KV 体积 (128 KiB)。\n- prefill 计算耗时等于 token 数除以吞吐率 8000 tok/s。\n传输耗时等于 KV 字节数除以链路带宽。令传输耗时等于 prefill 耗时, 等式两侧的 token 数直接约去, 得到盈亏带宽等于单 token KV 字节数乘以 prefill 吞吐率, 即 $128\\text{ KiB} \\times 8000\\text{ tok/s} \\approx 1.05\\text{ GB/s}$, 乘以 8 换算为网络比特速率即约为 8.4 Gbps。若切换为 LLaMA-2-7B (512 KiB/token), 单 token 体积增大至 4 倍, 盈亏带宽相应升至约 33.6 Gbps。常规数据中心 100 GbE 网卡的实际带宽已远高于上述阈值, 因此跨网传输 KV 几乎总是比同卡阻塞全体 decode 请求更划算。\n工程核算中有两处常见误区需要注意:\n- 速率单位换算: 网卡与 InfiniBand 带宽通常按 Gbps (每秒比特数) 标称, 而 KV 体积按字节计算, 必须除以 8 才能得到 GB/s。若漏除以 8, 传输耗时会低估 8 倍, 勾选「把 Gbps 当成 GB/s」即可直观对比这一误差。\n- 受延迟影响的指标: KV 传输推迟的是新请求第 2 个输出 token 的生成时刻, 而非首 token 延迟 (TTFT)。首 token 在 P 节点完成 prefill 时便已直接产出并可立即返回客户端。\n当链路固定延迟不为 0 时, 盈亏带宽会随 prompt 变短而升高: 此时固定延迟在总传输时间中的占比上升, 需要更高的传输带宽压缩数据发送耗时才能与 prefill 耗时持平。',
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
        <p>同卡的最大 ITL = 整段 prefill ({{ fmtMs(prefillMs) }}) + 一步 decode (25 ms)。上一个 token 出来后:</p>
        <ol class="pts">
          <li>要等 prefill 算完。</li>
          <li>再算一步 decode。</li>
          <li>下一个 token 才出来。</li>
        </ol>
        <p>采用 P/D 分离后, D 节点的自回归迭代不受新请求 prefill 干扰, 最大 ITL 恒等于单步 decode 耗时。</p>
        <p>盈亏带宽: 传输耗时恰好等于它换掉的那次 prefill 卡顿 ({{ fmtMs(prefillMs) }}) 时的网速。当前链路是它的 {{ (gbps / breakEven).toFixed(1) }}×, 大于 1 就说明搬 KV 比让大家等 prefill 划算。</p>
        <p>两种部署架构下的 TTFT (首 token 延迟) 相同: 首 token 在 P 节点完成 prefill 时即已生成, 跨网传输 KV 仅推迟第 2 个输出 token。</p>
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
