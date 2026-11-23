<!-- KV 淘汰实验台 (llm_infer/m39): make_context + sink_window_keep / tova_keep / H2O / snapkv_keep + m22 Quest 的缩小版 JS 移植。 -->
<template>
  <LabFrame
    title="KV 淘汰 — 同样的预算, 删掉的 KV 后面就找不回来"
    sub="- 模拟什么: 一段 136 个 token 的对话: 112 个前文 (第 0 个是 attention sink, 中段埋着两根针 A、B) → 问题 1 (找 A) → 回答 → 问题 2 (找 B)。KV cache 只能存预算内的 token, 淘汰策略按不同规则决定留谁: sink + 窗口只看位置; TOVA 每写入一个 token 删掉当前 query 最不关注的那个; H2O 按累积注意力留一半、最近的留一半; SnapKV 用问题 1 的 query 投票, prefill 结束时压一次。Quest 不删, 只是每步只读打分最高的几个 block。
- 图上是什么: 每格一个 token。S = sink, A / B = 针, 1 / 答 / 2 = 问题 1、回答、问题 2。实心 = 存在 cache 里, 淡色 = 已被删掉, 绿框 = Quest 这一步读的 block, 空格 = 还没写入。
- 可以做什么: 选策略; 切换「问题 1 时 / 问题 2 时」看两个时刻的 cache; 拖「KV 预算」; 选「放针 A / 放针 B」后点前文里的格子挪针; 「换一组」重新采样注意力噪声。
- 应该看到: 默认预算 16 时, H2O 和 SnapKV 在问题 1 时留着针 A, 到问题 2 时针 B 已经不在 (SnapKV 偶尔碰巧留下); sink + 窗口两根针都留不住, 除非把针挪进最近的窗口; Quest 两个问题都能读到针, 但存的 KV 是全部 136 个。右侧前两行是两个时刻 cache 里有没有对应的针, 有就答得对。"
    module="llm_infer/m39"
    run="python -m llm_infer.m39_kv_eviction.demo"
    :challenge="{
      ask: '选 SnapKV, 先在「问题 1 时」确认针 A 被留下, 再切到「问题 2 时」。针 B 还在吗? 把预算拖到 64 再看一次。',
      answer: '预算 16 时针 B 大概率已经被删, 预算 64 时它被留下的机会变大, 但仍然靠运气。\nSnapKV 只在 prefill 结束时压一次, 打分来自 prompt 末尾的观察窗口, 也就是问题 1 的 query。问题 1 在找针 A, 所以针 A 的票数远高于其它位置, 一定被留下; 针 B 和普通 token 得到的票差不多, 能不能留下只取决于它是否落在被留下的那一部分里。预算越大, 留下的比例越高, 碰巧留住针 B 的概率也越高, 但这和问题 2 无关。\nPython demo 在 2080 个 token、预算 256 下跑了 20 个种子: SnapKV 问题 1 全对, 问题 2 只对 3 次; 把两个问题对调后先问的那个 20/20 全对。打分本身没问题, 失败只来自删掉的 KV 回不来。Quest 不删, 两个问题都 20/20, 代价是显存一点没省。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in POLICIES" :key="p" type="button" :class="{ active: policy === p }" @click="policy = p">{{ p }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: stage === 0 }" @click="stage = 0">问题 1 时</button>
        <button type="button" :class="{ active: stage === 1 }" @click="stage = 1">问题 2 时</button>
        <span class="sep">点格子:</span>
        <button type="button" :class="{ active: place === 'A' }" @click="place = 'A'">放针 A</button>
        <button type="button" :class="{ active: place === 'B' }" @click="place = 'B'">放针 B</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="budget" label="KV 预算 (token)" :min="8" :max="64" :step="4" />
    </template>

    <div class="cells grid">
      <button
        v-for="t in T" :key="t" type="button" class="cell tk" :class="cls(t - 1)"
        :aria-label="`位置 ${t - 1} ${label(t - 1)} ${state(t - 1)}`" @click="move(t - 1)"
      >{{ label(t - 1) }}</button>
    </div>

    <template #stats>
      <div class="kv"><span>问题 1 时 cache 里有针 A (答得对)</span><b :class="okA ? 'good' : 'bad'">{{ okA ? '有 ✓' : '没有 ✗' }}</b></div>
      <div class="kv"><span>问题 2 时 cache 里有针 B (答得对)</span><b :class="okB ? 'good' : 'bad'">{{ okB ? '有 ✓' : '没有 ✗' }}</b></div>
      <div class="kv"><span>{{ stage ? '问题 2' : '问题 1' }} 时存的 KV / 每步读的 KV</span><b :class="stored > budget ? 'bad' : 'good'">{{ stored }} / {{ read }}</b></div>
      <div class="lab-note">
        <p>Python demo (2080 token, 预算 256, 20 个种子; 实测, 不随左侧变化) 答对次数 问题 1 / 问题 2:</p>
        <ul class="pts">
          <li v-for="r in PY" :key="r[0]"><b>{{ r[0] }}:</b> {{ r[1] }}</li>
        </ul>
        <p>淘汰策略答对与否完全取决于针是否还在 cache 里; demo 逐个种子断言了这一点。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, softmax } from '@/utils/labmath.js'

const D = 16, NCTX = 112, NQ = 8, NANS = 8, T = NCTX + 2 * NQ + NANS, T1 = NCTX + NQ, BS = 4, SQ = Math.sqrt(D)
const POLICIES = ['sink + 窗口', 'TOVA', 'H2O', 'SnapKV', 'Quest']
const PY = [['完整 cache', '20 / 20'], ['sink + 窗口', '0 / 0'], ['TOVA', '0 / 0'], ['H2O', '20 / 0'], ['SnapKV', '20 / 3 (碰巧)'], ['Quest', '20 / 20 (存全部 KV)']]

const policy = ref('H2O'), stage = ref(0), place = ref('A'), budget = ref(16), seed = ref(1)
const pos = ref({ A: 40, B: 76 })
const move = (t) => { if (t > 0 && t < NCTX && t !== pos.value[place.value === 'A' ? 'B' : 'A']) pos.value = { ...pos.value, [place.value]: t } }

// make_context 的缩小版: 方向 u (sink) / a (针 A) / b (针 B) 取坐标轴, 模长 √d
const ctx = computed(() => {
  const r = mulberry32(seed.value * 7919), axis = (k) => range(D).map((i) => (i === k ? SQ : 0))
  const [u, a, b] = [axis(0), axis(1), axis(2)]
  const q = range(T).map(() => range(D).map((i) => randn(r) + 1.2 * u[i]))
  const k = range(T).map(() => range(D).map(() => randn(r) * 0.2))
  k[0] = u
  k[pos.value.A] = a.map((x) => x * 0.5)
  k[pos.value.B] = b.map((x) => x * 0.5)
  for (let t = NCTX; t < T; t++) q[t] = q[t].map((x, i) => x + 3 * (t < T1 + NANS ? a[i] : b[i]))   // 问题 1 + 回答找 A, 问题 2 找 B
  return { q, k }
})
const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0)
const attn = (t, keep) => softmax(keep.map((j) => dot(ctx.value.q[t], ctx.value.k[j]) / SQ))
const argmin = (w) => w.reduce((m, x, i) => (x < w[m] ? i : m), 0)

function keeps(name, B) {                                 // → [问题 1 时留下的位置, 问题 2 时]
  if (name === 'sink + 窗口') return [T1, T].map((t) => range(t).filter((j) => j < 4 || j >= t - (B - 4)))
  if (name === 'TOVA') return [T1, T].map((tEnd) => {
    let keep = range(B)
    for (let t = B; t < tEnd; t++) { keep = [...keep, t]; keep.splice(argmin(attn(t, keep)), 1) }   // ★ 只看当前 query
    return keep
  })
  if (name === 'H2O') {
    const score = new Array(T).fill(0), rec = B >> 1
    for (let t = 0; t < T1; t++) attn(t, range(t + 1)).forEach((w, j) => { score[j] += w })
    const heavy = range(T1 - rec).sort((x, y) => score[y] - score[x] || x - y).slice(0, B - rec)
    let keep = [...heavy, ...range(rec).map((i) => T1 - rec + i)]
    const k1 = keep.slice()
    for (let t = T1; t < T; t++) {                        // 回答和问题 2 逐个写入, 每步删 1 个
      keep = [...keep, t]
      attn(t, keep).forEach((w, i) => { score[keep[i]] += w })
      const old = keep.filter((p) => p < t + 1 - rec)
      if (keep.length > B) { const d = old.reduce((m, p) => (score[p] < score[m] ? p : m), old[0]); keep = keep.filter((p) => p !== d) }
    }
    return [k1, keep]
  }
  if (name === 'SnapKV') {
    const W = Math.min(NQ, B), pre = T1 - W, votes = new Array(pre).fill(0)
    for (let t = pre; t < T1; t++) attn(t, range(t + 1)).slice(0, pre).forEach((w, j) => { votes[j] += w })
    const pooled = votes.map((_, j) => Math.max(...votes.slice(Math.max(0, j - 3), j + 4)))   // max pool, kernel 7
    const snap = [...range(pre).sort((x, y) => pooled[y] - pooled[x] || x - y).slice(0, B - W), ...range(W).map((i) => pre + i)]
    return [snap, [...snap, ...range(T - T1).map((i) => T1 + i)]]          // 生成阶段写入的不压
  }
  return [T1, T].map((len) => range(len))                 // Quest: 不删
}
function questRead(t, B) {                                // m22: 逐维 max(q·kmin, q·kmax) 上界, 首尾块强制保留
  const nb = (t + 1) / BS, q = ctx.value.q[t], K = ctx.value.k
  const ub = range(nb).map((bk) => range(D).reduce((s, i) => {
    const col = range(BS).map((j) => K[bk * BS + j][i])
    return s + Math.max(q[i] * Math.min(...col), q[i] * Math.max(...col))
  }, 0))
  const rest = range(nb).filter((x) => x !== 0 && x !== nb - 1).sort((x, y) => ub[y] - ub[x] || x - y)
  return [0, nb - 1, ...rest.slice(0, Math.max(B / BS - 2, 0))].flatMap((bk) => range(BS).map((j) => bk * BS + j))
}

const res = computed(() => {
  const [k1, k2] = keeps(policy.value, budget.value)
  const quest = policy.value === 'Quest'
  const r1 = quest ? questRead(T1 - 1, budget.value) : k1, r2 = quest ? questRead(T - 1, budget.value) : k2
  return { kept: [new Set(k1), new Set(k2)], read: [new Set(r1), new Set(r2)] }
})
const okA = computed(() => res.value.read[0].has(pos.value.A))
const okB = computed(() => res.value.read[1].has(pos.value.B))
const stored = computed(() => res.value.kept[stage.value].size)
const read = computed(() => res.value.read[stage.value].size)

const label = (t) => (t === 0 ? 'S' : t === pos.value.A ? 'A' : t === pos.value.B ? 'B' : t >= T1 + NANS ? '2' : t >= T1 ? '答' : t >= NCTX ? '1' : '')
const written = (t) => t < (stage.value ? T : T1)
const state = (t) => (!written(t) ? '还没写入' : res.value.kept[stage.value].has(t) ? '在 cache 里' : '已删除')
const cls = (t) => ({
  on: written(t) && res.value.kept[stage.value].has(t), dim: written(t) && !res.value.kept[stage.value].has(t),
  future: !written(t), rd: policy.value === 'Quest' && res.value.read[stage.value].has(t), needle: t === pos.value.A || t === pos.value.B,
})
</script>

<style scoped>
.grid { grid-template-columns: repeat(17, minmax(0, 1fr)); max-width: 560px; }
.tk { min-width: 0; min-height: 0; height: 26px; padding: 0; font-size: 10px; color: var(--text); transition: none; }
.tk.future { opacity: 0.15; }
.tk.rd { outline: 2px solid var(--left); outline-offset: -2px; }
.tk.needle { font-weight: 700; border: 2px solid var(--warn); }
.sep { color: var(--text-muted); margin-left: 8px; }
</style>
