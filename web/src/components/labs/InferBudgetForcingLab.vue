<!--
  推理时计算实验台 · 串行长思考 (对应 llm_infer/m23_test_time_compute/tts.py:think / believes_ok)。
  一件事: budget forcing 把 token 数钉在 B —— 截断就崩, 延长才涨, 但自查看不见自己的误解。
-->
<template>
  <LabFrame
    title="Budget forcing — 想多久由 token 预算 B 决定, 预算不够一次重写就白给"
    sub="- 模拟什么: 串行的「长思考」: 模型先把 4 步算术写完 (第一稿), 再一步步自查, 查出粗心错就从那一步往后重写。budget forcing 是控制它想多久的办法: 给一个 token 预算 B, 写满 B 个 token 立刻截断交卷; 模型想停但还没到 B, 就追加一个 Wait 逼它再查一遍。这里每写一步、每查一步都算 1 个 token。自查有个盲区: 粗心错 (随机的) 有 0.5 的概率被查出来, 但陷阱步上的误解 (看错运算符) 模型自己永远认为是对的。
      - 图上是什么: 上下两张图:
        - 图 ①: Python 实测, 横轴是预算 B, 纵轴是 200 道题的正确率。紫线是总体, 绿线是没有陷阱的题, 红线是带陷阱的题; 灰色虚线是不干预 (想停就停) 的成绩, 平均用 6.5 个 token, 正确率 0.305。
        - 图 ②: 前端按同一套规则现场模拟的一道题的 token 流水, 从左到右每格一个 token。
        - 图 ② 里 W1 到 W4: 写第 1 到 4 步, 绿色写对了, 红色是粗心错, 黄色是误解 (自查看不见)。
        - 图 ② 里 C1 到 C4: 查第几步, 灰色表示查过认为没问题, 红框表示查出错、接下来从这一步重写。
        - 图 ② 里虚线空格: 没用掉的预算。
      - 可以做什么: 三种操作:
        - 拖「预算 B」或点图 ① 的圆点改 B, 图 ② 的流水会跟着截断或延长。
        - 点「这道题有陷阱 / 没有陷阱」切换图 ② 这道题的第 2 步是不是陷阱步。
        - 点「换一个种子」换一道题 (换一组随机抽样)。
      - 应该看到: 图 ① 的正确率随 B 分三段, 右侧读数分两组:
        - B = 2: 第一稿都写不完, 正确率只有 0.055。
        - B = 4、6、8: 都停在 0.240, 因为一次重写要把后面几步写完, 预算不够就用不上。
        - B 到 16 以上: 才明显上涨, B = 64 时无陷阱题到 1.000, 陷阱题只有 0.431。
        - 右侧前两行: 当前 B 下的总体、无陷阱、陷阱题正确率 (陷阱题比无陷阱题低 0.5 以上标红)。
        - 右侧后两行: 图 ② 这道题最终交的答案对不对, 以及它写、查、重写了几次。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: 'B 从 4 加到 8, 预算翻倍, 正确率会涨多少?',
      answer: '一点没涨: B = 4、6、8 的正确率都是 0.240。\n可以在图 ② 里把 B 设成 8, 多点几次「换一个种子」: 只要查出了错, 重写就没有一次能写完。\n原因是账算不过来:\n1. 4 个 token 刚好写完第一稿 (W1 到 W4)。\n2. 之后开始自查, 查到第 $i$ 步 (从 1 数) 发现错, 要从这一步重写到第 4 步, 还得再花 $5-i$ 个 token。\n3. B = 8 时查完 4 步预算刚好用光; 哪怕中途查出错, 重写也写不完, 最后只能交第一稿, 和 B = 4 一样。\n预算要大到「查一遍 + 重写一遍」能完整走完才有用: B = 16 到 0.460, B = 64 到 0.815。\n但陷阱题在 B = 64 也只有 0.431: 模型自查时认为自己的误解是对的, 预算再多也改不掉。自查只能消随机错, 消不掉系统性的错。',
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
      W<i>k</i> 是写第 k 步, C<i>k</i> 是查第 k 步 ·
      <span class="tok ok">W</span> 算对 <span class="tok bad">W</span> 粗心错 <span class="tok mis">W</span> 误解 (自查看不见)
      <span class="tok chk">C</span> 查过, 认为没问题 <span class="tok found">C</span> 查出错, 从这一步重写
      <span class="tok empty" /> 没用掉的预算
    </p>

    <template #stats>
      <div class="kv"><span>B = {{ B }}: 200 道题的总体正确率</span><b :class="ALL[bi] > 0.5 ? 'good' : ALL[bi] < 0.1 ? 'bad' : ''">{{ ALL[bi].toFixed(3) }}</b></div>
      <div class="kv"><span>无陷阱题 / 陷阱题的正确率</span><b>{{ OK[bi].toFixed(3) }} / <span :class="{ bad: OK[bi] - TRAP[bi] > 0.5 }">{{ TRAP[bi].toFixed(3) }}</span></b></div>
      <div class="kv"><span>② 这道题最终交的答案</span><b :class="run.right ? 'good' : 'bad'">{{ run.right ? '对' : run.done ? '错 (写完了, 但带错)' : '错 (第一稿没写完)' }}</b></div>
      <div class="kv"><span>② 这道题写 / 查 / 重写的次数</span><b>{{ run.writes }} / {{ run.checks }} / {{ run.rewrites }}</b></div>
      <div class="lab-note">
        <p>图 ① 的数字来自 demo 的 [3] 段 (200 道题, min_tokens = max_tokens = B, 也就是不多不少正好用 B 个 token)。陷阱题比无陷阱题低 0.5 以上时标红。</p>
        <p>图 ② 的规则与 Python 的 think / believes_ok 相同:</p>
        <ul class="pts">
          <li>每步 6 个候选按 logits [3.0, 0.3×4, 0.3 或 3.5] 采样 (依次是正确、4 种粗心错、误解; 陷阱步上误解的 logit 是 3.5, 比正确的 3.0 还高)。</li>
          <li>粗心错以 0.5 的概率被查出, 陷阱步上的误解永远被认为是对的。</li>
        </ul>
        <p>只模拟每一步对或错, 不模拟数值碰巧相等。</p>
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
