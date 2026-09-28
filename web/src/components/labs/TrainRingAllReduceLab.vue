<!--
  Ring all-reduce 步进器 (对应 llm_train/core/collectives.py:ring_all_reduce_sum)。
  只讲一件事: 张量切 N 块沿环传, 每卡只发 2(N-1)/N·S 字节, 与 N 几乎无关。
-->
<template>
  <LabFrame
    title="Ring all-reduce — 每张卡每步只给右邻居发一块"
    sub="每一行是一张卡, 每一列是张量的一块 (chunk)。格子里的数字 = 这一块已经累加了几张卡的梯度。
      - 前 $N-1$ 步: 边传边加 (reduce-scatter)。
      - 后 $N-1$ 步: 把加好的块再传一圈 (all-gather)。
      点列头追踪某一块的旅程, 悬停或点击格子看它含哪些卡的贡献。"
    module="llm_train/core"
    run="python -m llm_train.m09_collectives.demo"
    :challenge="{
      ask: '把卡数从 4 拖到 8, 每张卡要发送的总字节数会翻倍吗? 对比「全发给 0 号卡再广播」呢?',
      answer: '不会。\n- ring: 每卡发 $2(N-1)$ 块, 每块 $S/N$, 总量 $2(N-1)/N \\cdot S$。$N=4$ 是 $1.5S$, $N=8$ 是 $1.75S$, 上限 $2S$。\n- 朴素做法: 0 号卡要收 $(N-1) \\cdot S$ 再发 $(N-1) \\cdot S$, 它的网卡随 $N$ 线性变成瓶颈。\nring 把流量均摊到每条链路上。代价是 $2(N-1)$ 步的延迟, 所以小张量要先攒成 bucket 再发。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="卡数 N" :min="2" :max="8" />
      <LabSlider v-model="sizeMB" label="张量大小 S" :min="64" :max="2048" :step="64" unit=" MB" />
      <StepPlayer :stepper="stepper" :label="phaseLabel" />
    </template>

    <!-- 格子: 悬停临时看, 点击固定, 方向键移动固定的格子 -->
    <div class="cells ring" :style="{ gridTemplateColumns: `52px repeat(${n}, minmax(34px, 1fr))` }"
         tabindex="0" role="group" aria-label="卡 × 块 的累加进度表, 方向键移动选中的格子"
         @keydown.left.prevent="move(0, -1)" @keydown.right.prevent="move(0, 1)" @keydown.up.prevent="move(-1, 0)" @keydown.down.prevent="move(1, 0)">
      <span />
      <button
        v-for="c in n" :key="'h' + c" type="button" class="col-head mono"
        :class="{ active: trace === c - 1 }" @click="trace = trace === c - 1 ? -1 : c - 1"
      >块 {{ c - 1 }}</button>
      <template v-for="(row, r) in frame.state" :key="r">
        <span class="row-label mono">卡 {{ r }}</span>
        <span
          v-for="(mask, c) in row" :key="c"
          class="cell"
          :class="[cellClass(r, c, mask), { dim: trace >= 0 && trace !== c, sel: cur && cur.r === r && cur.c === c }]"
          :style="{ background: heat(bits(mask) / n * 0.8, bits(mask) === n ? 'var(--left)' : 'var(--accent)') }"
          :title="`卡 ${r} 的块 ${c}: 含卡 {${members(mask).join(',')}} 的梯度`"
          @mouseenter="hover = { r, c }" @mouseleave="hover = null" @click="pin = { r, c }"
        >{{ bits(mask) }}/{{ n }}</span>
      </template>
    </div>
    <p class="lab-note msg" aria-live="polite">
      <template v-if="frame.msgs.length">本步: {{ frame.msgs.map((m) => `卡${m.src}→卡${m.dst} 发块${m.c}`).join(' · ') }}</template>
      <template v-else>初始: 每张卡只有自己的梯度 (每块 1/{{ n }})。按「▶ 播放」, 或点「下一步 ›」一步一步走。</template>
      <br />
      <span v-if="cur">卡 {{ cur.r }} 的块 {{ cur.c }} 已含卡 { {{ members(frame.state[cur.r][cur.c]).join(', ') }} } 的贡献。</span>
    </p>

    <template #stats>
      <div class="kv"><span>阶段</span><b>{{ frame.k <= n - 1 ? (frame.k === 0 ? '—' : 'reduce-scatter') : 'all-gather' }}</b></div>
      <div class="kv"><span>每卡已发送</span><b>{{ fmtMB(frame.k * sizeMB / n) }}</b></div>
      <div class="kv"><span><Tex text="总量: ring 每卡 $\frac{2(N-1)}{N} S$ / 朴素做法的 0 号卡 $2(N-1) S$" /></span><b class="pair">{{ fmtMB(2 * (n - 1) / n * sizeMB) }} / {{ fmtMB(2 * (n - 1) * sizeMB) }}</b></div>
      <div class="kv"><span>已拿到完整和的格子</span><b :class="{ good: doneCells === n * n }">{{ doneCells }} / {{ n * n }}</b></div>
      <div class="lab-note">
        <p><Tex text="★ 第 $t$ 步卡 $r$ 发的是块 $(r - t) \bmod N$: 每块沿环走一圈, 走到哪加到哪。" /></p>
        <p><Tex text="reduce-scatter 结束时, 卡 $r$ 恰好握着块 $(r+1) \bmod N$ 的全局和。" />这一半单独拿出来, 就是 ZeRO-2 的梯度同步。</p>
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
import { heat, range } from '@/utils/labmath.js'

const n = ref(4)
const sizeMB = ref(512)
const trace = ref(-1)
const hover = ref(null), pin = ref(null)     // 悬停的 / 点击固定的格子 { r, c }
const cur = computed(() => hover.value || pin.value)
const move = (dr, dc) => {
  const p = cur.value || { r: 0, c: 0 }, hi = n.value - 1
  pin.value = { r: Math.min(hi, Math.max(0, p.r + dr)), c: Math.min(hi, Math.max(0, p.c + dc)) }
  hover.value = null
}

const bits = (m) => { let k = 0; for (; m; m >>= 1) k += m & 1; return k }
const members = (m) => range(8).filter((i) => m & (1 << i))
const fmtMB = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + ' GB' : mb.toFixed(0) + ' MB')

// 逐帧状态: state[r][c] 是一个位掩码, 记录卡 r 的块 c 已经累加了哪些卡的梯度。与 Python 实现同一套下标。
const frames = computed(() => {
  const N = n.value
  let state = range(N).map((r) => range(N).map(() => 1 << r))
  const out = [{ k: 0, state, msgs: [] }]
  for (const phase of ['reduce', 'gather']) {
    for (let t = 0; t < N - 1; t++) {
      const shift = phase === 'reduce' ? 0 : 1
      // ★ 同一步内所有卡同时发: 先取快照再写, 否则后发的卡会读到本步刚收到的数据
      const msgs = range(N).map((r) => ({ src: r, dst: (r + 1) % N, c: (((r + shift - t) % N) + N) % N }))
      const next = state.map((row) => [...row])
      for (const { src, dst, c } of msgs) next[dst][c] = phase === 'reduce' ? state[dst][c] | state[src][c] : state[src][c]
      state = next
      out.push({ k: out.length, state, msgs, phase })
    }
  }
  return out
})

const stepper = useStepper(() => frames.value.length, { interval: 800 })
const frame = computed(() => frames.value[Math.min(stepper.step.value, frames.value.length - 1)])
const doneCells = computed(() => frame.value.state.flat().filter((m) => bits(m) === n.value).length)
const phaseLabel = computed(() => `第 ${frame.value.k} / ${2 * (n.value - 1)} 步`)
watch(n, () => { stepper.reset(); trace.value = -1; hover.value = null; pin.value = null })

const cellClass = (r, c, mask) => {
  const m = frame.value.msgs
  if (m.some((x) => x.dst === r && x.c === c)) return 'recv'
  if (m.some((x) => x.src === r && x.c === c)) return 'send'
  return bits(mask) === n.value ? 'full' : ''
}
</script>

<style scoped>
.ring .cell { height: 30px; font-size: 11px; color: var(--text); cursor: pointer; }
.cell.sel { box-shadow: inset 0 0 0 2px var(--text); }
.pair { font-size: 13px !important; }
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.col-head { font-size: 11px; min-height: 26px; padding: 2px 4px; }
.cell.send { outline: 2px dashed var(--warn); outline-offset: 1px; }
.cell.recv { outline: 2px solid var(--warn); outline-offset: 1px; }
.cell.full { border-color: var(--left); }
.msg { margin-top: 12px; min-height: 3.4em; }
</style>
