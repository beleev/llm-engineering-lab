<!--
  推理时计算实验台 · 串行长思考 (对应 llm_infer/m23_test_time_compute/tts.py:think / believes_ok)。
  一件事: budget forcing 把 token 数钉在 B —— 截断就崩, 延长才涨, 但自查看不见自己的误解。
-->
<template>
  <LabFrame
    title="Budget forcing — 想多久由 token 预算 B 决定, 预算不够一次重写就白给"
    sub="- 模拟什么: 串行测试时扩展 (长思考链自省修正): 模型先写出完整的 4 步算术初稿, 随后逐步自我核查, 一旦检出粗心错误便从出错步骤起向后全部重写。budget forcing (预算强制) 通过硬性控制生成的 token 数量来调节思考深度: 给定总预算 B, 生成满 B 个 token 时立即截断并提取当前答案; 若模型在消耗满 B 之前试图提前终止, 则强制追加反思触发词 Wait 促使其继续核查一轮。本模拟将每写一步与每查一步均计为消耗 1 个 token, 且设定自查存在能力盲区: 随机粗心错误有 0.5 的概率被自查检出, 而陷阱步上的系统性误解 (看错运算符) 在模型自查时永远被误判为正确。
- 图上是什么: 视图由上下两张图表组成:
  - 图 ①: 来自 Python 脚本的 200 道题实测曲线, 横轴为 token 预算 B, 纵轴为解题准确率。紫线代表全集总体准确率, 绿线代表无陷阱题子集, 红线代表含陷阱题子集; 灰色虚线标出不施加干预 (允许模型自主决定何时停止) 时的基线表现, 平均消耗 6.5 个 token, 总体准确率为 0.305。
  - 图 ②: 前端遵循相同状态转移规则实时生成的单题 token 执行流水线, 从左至右每格代表 1 个 token。
  - 图 ② 中的 W1 至 W4: 表示书写第 1 至第 4 步计算结果, 绿色代表计算正确, 红色代表随机粗心错误, 黄色代表系统性误解 (自查无法检出)。
  - 图 ② 中的 C1 至 C4: 表示核查对应步骤, 灰色代表核查通过, 红框代表检出错误并触发从该步开始的后续重写。
  - 图 ② 中的虚线空格: 尚未消耗的剩余 token 预算。
- 可以做什么: 支持三项交互操作:
  - 拖动「预算 B」滑杆或点击图 ① 曲线上的圆点以修改预算上限 B, 图 ② 的单题执行流水线会同步截断或向后延展。
  - 点击「这道题有陷阱 / 没有陷阱」切换图 ② 当前演示题目的第 2 步是否包含系统性陷阱。
  - 点击「换一个种子」重新随机抽取一道题目的生成与核查轨迹。
- 应该看到: 图 ① 的准确率随预算 B 呈现明显的三阶段阶跃特征, 右侧面板分两组展示统计读数:
  - 当 $B = 2$ 时: 初稿尚未来得及写完即被截断, 总体准确率仅为 0.055。
  - 当 B 取 4、6、8 时: 总体准确率全部停滞在 0.240, 因为一旦触发重写必须将后续步骤全部补齐才能更新最终答案, 预算不足时半截重写无法生效。
  - 当 B 增至 16 及以上时: 准确率显著攀升, 在 $B = 64$ 时无陷阱题子集达到 1.000, 而含陷阱题子集仅停留在 0.431。
  - 右侧前两行: 当前预算 B 下的总体、无陷阱题与陷阱题准确率 (当陷阱题准确率低于无陷阱题超过 0.5 时标红提示)。
  - 右侧后两行: 图 ② 当前单题最终提交答案的正误状态, 以及全程累计执行的书写、核查与重写次数。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: '将 token 预算 B 从 4 增加至 8 (预算翻倍), 总体准确率会提升多少?',
      answer: '完全没有提升: 当 B 取 4、6、8 时, 总体准确率恒为 0.240。\n可在图 ② 中将预算 B 设为 8 并多次点击「换一个种子」验证: 凡是在自查阶段检出错误的轨迹, 后续触发的重写均因预算耗尽而无法写完最后一步。\n这源于串行核查与重写的最小步数开销约束:\n1. 完成首轮初稿书写 (W1 至 W4) 恰好消耗前 4 个 token。\n2. 第 5 个 token 起进入逐步自查, 若核查至第 $i$ 步 (从 1 开始计数) 检出粗心错误, 则至少还需消耗 $i$ 个核查 token 加上从第 $i$ 步重写至第 4 步所需的 $5-i$ 个书写 token, 合计额外需要 5 个 token 才能产出完整的新终稿。\n3. 因此当 $B = 8$ 时, 剩余的 4 个 token 最多只够完成一轮无错自查; 一旦中途检出错误并启动重写, 必然在写到第 4 步之前耗尽预算而被截断, 系统只能回退提交未修正的第一稿, 最终准确率与 $B = 4$ 完全一致。\n只有当预算充裕到足以完整容纳「核查 + 全链路重写」闭环时准确率才会跃升: $B = 16$ 时升至 0.460, $B = 64$ 时进一步升至 0.815。\n然而含陷阱题子集即使在 $B = 64$ 下准确率也仅为 0.431: 模型在自查时与初稿共享相同的认知盲区, 始终将看错运算符的误解判为正确, 继续追加串行思考预算也无法自我纠正。纯模型自查仅能消除随机失误, 无法消除系统性认知偏差。',
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
        <p>图 ① 实测数据取自本节 demo 脚本的 [3] 段 (共 200 道题, 将 min_tokens 与 max_tokens 同时设为 B, 即强制每题精确消耗 B 个 token)。当含陷阱题准确率比无陷阱题低 0.5 以上时读数标红。</p>
        <p>图 ② 的状态转移规则与 Python 脚本中的 think 及 believes_ok 函数完全一致:</p>
        <ul class="pts">
          <li>每步 6 个候选按 logits [3.0, 0.3×4, 0.3 或 3.5] 采样 (依次是正确、4 种粗心错、误解; 陷阱步上误解的 logit 是 3.5, 比正确的 3.0 还高)。</li>
          <li>粗心错以 0.5 的概率被查出, 陷阱步上的误解永远被认为是对的。</li>
        </ul>
        <p>仅模拟各推理步骤的逻辑正误状态, 不模拟错误运算在特定数值下巧合得出相同答案的边界情形。</p>
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
