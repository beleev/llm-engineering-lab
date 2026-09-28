<!--
  推理时计算实验台 · 串行长思考 (对应 llm_infer/m23_test_time_compute/tts.py:think / believes_ok)。
  一件事: budget forcing 把 token 数钉在 B —— 截断就崩, 延长才涨, 但自查看不见自己的误解。
-->
<template>
  <LabFrame
    title="Budget forcing — 想多久由预算 B 决定"
    sub="模型先写完 4 步, 再一步步自查, 发现粗心错就从那一步重写。每写一步、查一步都算 1 个 token。
      到 B 个 token 立刻截断; 模型想停但没到 B, 就追加 Wait 再查一遍。
      - 图 ①: 200 道题的正确率随 B 的变化 (Python 实测)。点圆点或拖滑杆改 B。
      - 图 ②: 一道题在预算 B 下的 token 流水 (前端按同一规则现场模拟)。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: 'B 从 4 加到 8, 预算翻倍, 正确率会涨多少?',
      answer: '一点没涨: B=4、6、8 都是 0.240。\n在图 ② 里把 B 设成 8, 多换几次种子, 重写没有一次写完。\n4 个 token 刚好写完第一稿。之后查到第 $i$ 步 (从 1 数) 发现错, 要从这一步重写到结尾, 还得再花 $5-i$ 个 token。\nB=8 时查完 4 步预算就用光, 重写总是写不完, 只能交第一稿。\n预算要大到「查 + 重写」能完整走一轮才有用: B=16 到 0.460, B=64 到 0.815。\n但陷阱题在 B=64 只有 0.431: 自查认为自己的误解是对的, 预算再多也改不掉。',
    }"
  >
    <template #controls>
      <LabSlider v-model="bi" label="预算 B (token)" :min="0" :max="8" :format="(i) => BUDGETS[i]" />
      <div class="row">
        <button type="button" :class="{ active: trap }" :aria-pressed="trap" @click="trap = !trap">{{ trap ? '这道题有陷阱 (第 2 步)' : '这道题没有陷阱' }}</button>
        <button type="button" @click="seed++">换一个种子</button>
      </div>
    </template>

    <p class="blk">① 实测, 不随种子和陷阱开关变化: 200 道题的正确率随 B</p>
    <svg viewBox="0 0 560 200" role="group" :aria-label="`正确率随预算 B 的曲线, 当前 B=${B}: 总体 ${ALL[bi]}`">
      <line v-for="v in [0, 0.5, 1]" :key="v" x1="40" x2="550" :y1="cy(v)" :y2="cy(v)" class="grid" />
      <text v-for="v in [0, 0.5, 1]" :key="`t${v}`" x="34" :y="cy(v) + 3" class="tick end">{{ v }}</text>
      <line x1="40" x2="550" :y1="cy(NAT.acc)" :y2="cy(NAT.acc)" class="nat" />
      <text x="548" :y="cy(NAT.acc) - 4" class="tick end">不干预: {{ NAT.tok }} token, {{ NAT.acc }}</text>
      <g v-for="s in SERIES" :key="s.key">
        <polyline :points="s.data.map((v, i) => `${cx(i)},${cy(v)}`).join(' ')" class="line" :class="s.key" />
        <g v-for="(v, i) in s.data" :key="i" class="pt" tabindex="0" role="button" :aria-label="`${s.name}, B=${BUDGETS[i]}: ${v}`" :aria-pressed="i === bi" @click="bi = i" @keydown.enter="bi = i">
          <circle :cx="cx(i)" :cy="cy(v)" r="9" class="hit" />
          <circle :cx="cx(i)" :cy="cy(v)" :r="i === bi ? 4.5 : 3" :class="s.key" />
        </g>
      </g>
      <line :x1="cx(bi)" :x2="cx(bi)" y1="10" y2="170" class="cursor" />
      <text v-for="(b, i) in BUDGETS" :key="`b${i}`" :x="cx(i)" y="184" class="tick">{{ b }}</text>
      <text v-for="(s, k) in SERIES" :key="`l${k}`" :x="60 + k * 90" y="198" class="tick start" :class="s.key">● {{ s.name }}</text>
    </svg>

    <p class="blk">② 前端现算: 这道题在预算 B = {{ B }} 下的 token 流水</p>
    <div class="trace" aria-label="token 流水">
      <span v-for="(e, i) in run.log" :key="i" class="tok mono" :class="e.cls" :title="e.title">{{ e.label }}</span>
      <span v-for="i in Math.max(0, B - run.log.length)" :key="`e${i}`" class="tok empty" />
    </div>
    <p class="legend mono">
      W<i>k</i> = 写第 k 步 · C<i>k</i> = 查第 k 步 ·
      <span class="tok ok">W</span> 算对 <span class="tok bad">W</span> 粗心错 <span class="tok mis">W</span> 误解 (自查看不见)
      <span class="tok chk">C</span> 查过, 认为没问题 <span class="tok found">C</span> 查出错, 从这一步重写
      <span class="tok empty" /> 没用掉的预算
    </p>

    <template #stats>
      <div class="kv"><span>B = {{ B }}: 总体正确率</span><b :class="ALL[bi] > 0.5 ? 'good' : ALL[bi] < 0.1 ? 'bad' : ''">{{ ALL[bi].toFixed(3) }}</b></div>
      <div class="kv"><span>无陷阱 / 陷阱题</span><b>{{ OK[bi].toFixed(3) }} / <span :class="{ bad: OK[bi] - TRAP[bi] > 0.5 }">{{ TRAP[bi].toFixed(3) }}</span></b></div>
      <div class="kv"><span>② 这道题交的答案</span><b :class="run.right ? 'good' : 'bad'">{{ run.right ? '对' : run.done ? '错 (写完了, 但带错)' : '错 (第一稿没写完)' }}</b></div>
      <div class="kv"><span>② 写 / 查 / 重写次数</span><b>{{ run.writes }} / {{ run.checks }} / {{ run.rewrites }}</b></div>
      <div class="lab-note">
        <p>图 ① 的数字来自 demo 的 [3] 段 (200 道题, min_tokens = max_tokens = B)。陷阱题比无陷阱题低 0.5 以上时标红。</p>
        <p>图 ② 的规则与 think / believes_ok 相同。每步 6 个候选按 logits [3.0, 0.3×4, 0.3 或 3.5] 采样。粗心错以 0.5 的概率被发现, 陷阱步上的误解永远被认为是对的。</p>
        <p>只模拟对错, 不模拟数值碰巧相等。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, softmax } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m23_test_time_compute.demo 的 [3] 段
const BUDGETS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
const ALL = [0.055, 0.24, 0.24, 0.24, 0.345, 0.46, 0.615, 0.72, 0.815]
const OK = [0.059, 0.259, 0.259, 0.259, 0.4, 0.533, 0.741, 0.867, 1.0]
const TRAP = [0.046, 0.2, 0.2, 0.2, 0.231, 0.308, 0.354, 0.415, 0.431]
const NAT = { tok: 6.5, acc: 0.305 }
const SERIES = [
  { key: 'all', name: '总体', data: ALL },
  { key: 'ok', name: '无陷阱', data: OK },
  { key: 'trap', name: '陷阱题', data: TRAP },
]
const K = 4, TRAP_STEP = 1, DETECT = 0.5

const bi = ref(3), trap = ref(true), seed = ref(1)
const B = computed(() => BUDGETS[bi.value])
const cx = (i) => 60 + i * 60
const cy = (v) => 170 - v * 150

// 一步的结果: 0 = 对, 1..4 = 粗心错, 5 = 看错运算符 (陷阱步上 logit 3.5, 是众数)
const probs = (i) => softmax([3, 0.3, 0.3, 0.3, 0.3, trap.value && i === TRAP_STEP ? 3.5 : 0.3])
const pick = (ps, r) => { let u = r(), k = 0; while (k < 5 && (u -= ps[k]) > 0) k++; return k }
// 自查: 对的认对; 陷阱步上自己的误解也认对 (= believes_ok)
const believes = (i, o) => o === 0 || (trap.value && i === TRAP_STEP && o === 5)

// ★ 与 think(min_tokens=B, max_tokens=B) 同一套控制流, 只记每一步是对是错
const run = computed(() => {
  const r = mulberry32(seed.value * 7919), max = B.value, log = []   // 种子不含 B: 小 B 的流水是大 B 的前缀
  let chain = [], tokens = 0, writes = 0, checks = 0, rewrites = 0
  const cls = (i, o) => (o === 0 ? 'ok' : believes(i, o) ? 'mis' : 'bad')
  const writeFrom = (i) => {
    const draft = chain.slice(0, i)
    while (draft.length < K) {
      if (tokens >= max) { if (!chain.length) chain = draft; return false }
      const j = draft.length, o = pick(probs(j), r)
      draft.push(o); tokens++; writes++
      log.push({ label: `W${j + 1}`, cls: cls(j, o), title: `写第 ${j + 1} 步` })
    }
    chain = draft
    return true
  }
  let done = writeFrom(0)
  while (done && tokens < max) {       // 没到 B 就不许停 (Wait)
    for (let i = 0; i < K; i++) {
      if (tokens >= max) break
      tokens++; checks++
      const found = !believes(i, chain[i]) && r() < DETECT
      log.push({ label: `C${i + 1}`, cls: found ? 'found' : 'chk', title: found ? `查第 ${i + 1} 步: 发现错, 重写` : `查第 ${i + 1} 步: 认为没问题` })
      if (found) { rewrites++; done = writeFrom(i); break }
    }
  }
  const right = chain.length === K && chain.every((o) => o === 0)
  return { log, right, done: chain.length === K, writes, checks, rewrites }
})
</script>

<style scoped>
svg { min-width: 540px; }
.grid { stroke: var(--border); }
.nat { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.start { text-anchor: start; }
polyline.line { fill: none; stroke-width: 2; }
.all { stroke: var(--accent); fill: var(--accent); }
.ok { stroke: var(--left); fill: var(--left); }
.trap { stroke: var(--danger); fill: var(--danger); }
.pt { cursor: pointer; }
.blk { font-size: 12px; color: var(--text-muted); margin: 10px 0 4px; }
.hit { fill: transparent; stroke: none; }
.cursor { stroke: var(--warn); stroke-dasharray: 3 3; }
.trace { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 10px; }
.tok { min-width: 30px; padding: 2px 4px; font-size: 11px; text-align: center; border-radius: 3px; border: 1px solid var(--border); }
.tok.ok { background: color-mix(in srgb, var(--left) 30%, transparent); }
.tok.bad { background: color-mix(in srgb, var(--danger) 30%, transparent); }
.tok.mis { background: color-mix(in srgb, var(--warn) 40%, transparent); }
.tok.chk { color: var(--text-muted); }
.tok.found { border-color: var(--danger); color: var(--danger); }
.tok.empty { border-style: dashed; }
.legend { font-size: 11px; color: var(--text-muted); margin: 6px 0 0; line-height: 2; }
.legend .tok { display: inline-block; min-width: 22px; }
.lab-stats .bad { color: var(--danger); }
</style>
