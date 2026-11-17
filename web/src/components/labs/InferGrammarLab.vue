<!--
  token 级语法约束实验台 (对应 llm_infer/m14_structured_output/grammar.py:token_row / compile_token_table)。
  一件事: 真实 token 是多字符的, 一个 token 可能一口气跨过几个语法状态 —— 所以要把 "状态 × token → 落点" 离线编译成表。
-->
<template>
  <LabFrame
    title="结构化输出 — 用状态机限制每一步能选哪些 token, 并把它预编译成表"
    sub='- 模拟什么: 让模型只能输出合法 JSON 的办法: 每一步采样前把会破坏语法的 token 的 logit 置 −inf (概率归零)。哪些 token 合法由一个 FSM (有限状态机, 这里是字符级 DFA) 决定: 它记着「JSON 写到哪了」, 每吃一个字符就换到下一个状态。难点是模型的词表是多字符 token ({" 、": 、true 都是一个 token), 一个 token 要在字符级状态机上连走几步, 整段字符都走得通才算合法。每步对全词表逐字符试走太慢, 所以离线把「每个状态 × 每个 token 走到哪」算成一张表, 在线每步只查一行。这里的迷你语法: {"key":value, …}, value 是字符串、数字、true、false 或 null。
      - 图上是什么: 四部分:
        - 上面的状态图: 把 DFA 的状态归并成 10 个方框, 紫框是当前状态, 箭头是允许的转移, 虚线箭头是回边 (逗号之后回去等下一个 key)。悬停一个 token 时, 它一口气经过的状态会亮黄框并标上顺序号。
        - 状态图下面一行字: 说明悬停的 token 为什么合法或非法。
        - 再下面: 整个词表 (每个 token 一个按钮)。灰掉的在当前状态下非法, 绿边框的是合法的多字符 token, 划掉的是故意放进去的垃圾 token (比如 }} 和换行符), 在任何状态都非法。
        - 最上面「输出」: 目前拼出来的字符串, 走到 EOS 时会显示 JSON.parse 的结果。
      - 可以做什么: 按钮和悬停:
        - 点一个合法 token 让它前进一步。
        - 「撤销」退一步, 「重置」清空。
        - 「随机合法一步」从合法 token 里随机挑一个。
        - 「只用单字符词表」把多字符 token 藏起来, 模拟一个只有单字符 token 的词表, 看拼同一个串要多几步。
        - 悬停任何 token (包括灰掉的) 看它会走到哪, 或者卡在哪。
      - 应该看到: 词表里大部分 token 一直是灰的, 每个状态下只有少数几个合法; 多字符 token 常常一步跨过两三个状态。右侧读数:
        - 「当前状态下合法的 token / 词表大小」。
        - 「上一个 token 跨过的状态数」大于 1 时绿色, 说明一个 token 走了好几步。
        - 「已用 token / 已出字符」比较两者就能看出多字符词表省了多少步。
        - 「预编译表 S × V」是离线要算的表有多大 (状态数 × 词表大小)。'
    module="llm_infer/m14"
    run="python -m llm_infer.m14_structured_output.demo"
    :challenge="{
      ask: '重置后, 用最少的 token 拼出 {&quot;id&quot;:42}。需要几个?',
      answer: '5 个 token: {&quot; → id → &quot;: → 42 → }, 再加一个 EOS 收尾。其中 &quot;: 一口气跨了 2 个状态 (先闭合 key 的引号, 再吃掉冒号)。\n勾上「只用单字符词表」拼同一个串: { &quot; i d &quot; : 4 2 } 要 9 个 token, 再加 EOS。\n一个 token 合不合法, 看的是「整段字符都能沿 DFA 走通」, 不是只看第一个字符:\n- true: 从 value? 状态走 4 步, 落到「, 或 }」。\n- tru: 走 3 步停在字面量中间, 也合法, 但之后只能接 e。\n- }}: 第一个 } 走到结束态, 第二个 } 在结束态无路可走, 所以它对所有状态都是 −1 (非法)。\n这张 S × V 的表离线算一次 (每个状态对每个 token 试走一遍), 在线每步只查一行拿 mask, 再查一次落点。否则每步要对全词表逐字符试走, 这是纯 CPU 开销, 卡在 GPU 前向和采样之间。demo 实测: 在线现算 18.4 μs/步, 查表 0.086 μs/步 (215×)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :disabled="!hist.length" @click="hist.pop()">撤销</button>
        <button type="button" :disabled="!hist.length" @click="hist.splice(0)">重置</button>
        <button type="button" :disabled="finished" @click="randomStep">随机合法一步</button>
        <button type="button" :class="{ active: singleOnly }" :aria-pressed="singleOnly" @click="singleOnly = !singleOnly; hist.splice(0)">只用单字符词表</button>
      </div>
      <p class="out mono" aria-live="polite">输出: <b>{{ output || '∅' }}</b><span v-if="finished" class="okmsg"> ⏹ EOS · JSON.parse ✓ → {{ parsed }}</span></p>
    </template>

    <svg viewBox="0 0 560 165" role="img" :aria-label="`语法状态图, 当前状态 ${GROUPS[curGroup].label}`">
      <defs>
        <marker id="gram-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" class="head" />
        </marker>
      </defs>
      <line v-for="(e, k) in ARROWS" :key="k" :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2" class="edge" :class="{ back: e.back }" marker-end="url(#gram-arrow)" />
      <g v-for="(g, k) in GROUPS" :key="k" :class="['st', { cur: k === curGroup, via: hoverPath.includes(k) }]">
        <rect :x="g.x - 30" :y="g.y - 13" width="60" height="26" rx="6" />
        <text :x="g.x" :y="g.y + 4">{{ g.label }}</text>
        <text v-if="hoverPath.includes(k)" :x="g.x + 26" :y="g.y - 8" class="ord">{{ hoverPath.indexOf(k) + 1 }}</text>
      </g>
    </svg>
    <p class="hint mono">{{ hoverMsg }}</p>

    <div class="vocab">
      <button v-for="t in shown" :key="t" type="button" class="tokbtn mono" :class="{ multi: VOCAB[t].length > 1 && t !== EOS_ID, junk: JUNK.includes(VOCAB[t]) }"
        :aria-disabled="!ok(t)" @click="ok(t) && hist.push(t)"
        @mouseenter="hover = t" @mouseleave="hover = -1" @focus="hover = t" @blur="hover = -1">{{ show(VOCAB[t]) }}</button>
    </div>

    <template #stats>
      <div class="kv"><span>当前状态下合法的 token / 词表大小</span><b>{{ finished ? 0 : legal.length }} / {{ shown.length }}</b></div>
      <div class="kv"><span>上一个 token 跨过的状态数</span><b :class="lastCross > 1 ? 'good' : ''">{{ lastCross }}</b></div>
      <div class="kv"><span>已用 token / 已出字符</span><b>{{ hist.length }} / {{ output.length }}</b></div>
      <div class="kv"><span>预编译表 S × V (状态数 × 词表大小)</span><b>{{ STATES.length }} × {{ VOCAB.length }}</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>在线每步:</b> 查表一行拿 mask, 再查一次落点状态, <Tex text="$O(1)$" />。</li>
          <li><b>不预编译:</b> 每步对全词表逐字符试走, 这个小词表是 {{ totalChars }} 次状态转移。真实词表 <Tex text="$V \approx 10^5$" /> 时是几十万次, 纯 CPU, 夹在两次 GPU 前向之间。</li>
        </ul>
        <p>EOS 只在结束态合法, 所以输出一定是完整 JSON, 不靠模型自觉。mask 只保证语法合法, 不保证内容对。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import { mulberry32 } from '@/utils/labmath.js'

// ---- 字符级 DFA: step(状态名, 字符) → 状态名 | null。字面量 / 长度上限都需要自己的状态 ----
const LIT = { t: 'true', f: 'false', n: 'null' }
const az = (c) => c >= 'a' && c <= 'z', dg = (c) => c >= '0' && c <= '9'
const step = (s, c) => {
  const [kind, a, b] = s.split(':')
  if (kind === 'open') return c === '{' ? 'keyq' : null
  if (kind === 'keyq') return c === '"' ? 'key:0' : null
  if (kind === 'key' || kind === 'str') {                       // [a-z]{1,6} 再闭引号
    if (az(c) && +a < 6) return `${kind}:${+a + 1}`
    return c === '"' && +a > 0 ? (kind === 'key' ? 'colon' : 'after') : null
  }
  if (kind === 'colon') return c === ':' ? 'val' : null
  if (kind === 'val') return c === '"' ? 'str:0' : c >= '1' && c <= '9' ? 'num:1' : LIT[c] ? `lit:${LIT[c]}:1` : null
  if (kind === 'lit') return c !== a[+b] ? null : +b + 1 === a.length ? 'after' : `lit:${a}:${+b + 1}`
  if (kind === 'num' && dg(c)) return +a < 4 ? `num:${+a + 1}` : null
  if (kind === 'num' || kind === 'after') return c === ',' ? 'keyq' : c === '}' ? 'done' : null
  return null                                                   // done: 只有 EOS
}
const SINGLE = [...'{}":,adefgilmnrstu0124'], JUNK = ['[', '}}', '\n', "'", '::']
const MULTI = ['{"', '":', '",', '"}', '":"', '","', 'e":', 'd":', '1}', '2}', 'true', 'false', 'null', 'tru', 'name', 'age', 'id', '42', '10']
const VOCAB = ['<EOS>', ...SINGLE, ...MULTI, ...JUNK], EOS_ID = 0
const STATES = (() => { const seen = ['open']; for (const s of seen) for (const c of SINGLE) { const n = step(s, c); if (n && !seen.includes(n)) seen.push(n) } return seen })()
const DONE = STATES.indexOf('done')

// ★ token_row: 从状态 s 逐字符试走 token; 全走通 → 记落点, 否则 −1。对每个状态各跑一遍 = 预编译
const walk = (s, piece) => { const path = []; let cur = STATES[s]; for (const c of piece) { cur = step(cur, c); if (!cur) return null; path.push(STATES.indexOf(cur)) } return path }
const table = STATES.map((_, s) => VOCAB.map((p, t) => (t === EOS_ID ? (s === DONE ? DONE : -1) : walk(s, p)?.at(-1) ?? -1)))
const totalChars = VOCAB.slice(1).reduce((a, p) => a + p.length, 0)

// ---- 状态图: 把 DFA 状态归并成 10 个展示状态 ----
const GROUPS = [
  { id: 'open', label: '等 {', x: 40, y: 30 }, { id: 'keyq', label: '等 key "', x: 130, y: 30 }, { id: 'key', label: 'key 中', x: 220, y: 30 },
  { id: 'colon', label: '等 :', x: 310, y: 30 }, { id: 'val', label: 'value?', x: 400, y: 30 }, { id: 'str', label: '字符串中', x: 515, y: 30 },
  { id: 'num', label: '数字中', x: 515, y: 85 }, { id: 'lit', label: '字面量中', x: 515, y: 140 }, { id: 'after', label: ', 或 }', x: 310, y: 140 }, { id: 'done', label: '结束', x: 130, y: 140 },
]
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [4, 6], [4, 7], [5, 8], [7, 8], [6, 8], [8, 1], [8, 9], [6, 9]]
// 箭头停在目标框的边上 (框半宽 30, 半高 13); 回边 = 指向更早状态的边, 画虚线
const ARROWS = EDGES.map(([a, b]) => {
  const p = GROUPS[a], q = GROUPS[b], dx = q.x - p.x, dy = q.y - p.y
  const t = 1 - Math.min(dx ? 32 / Math.abs(dx) : 1, dy ? 15 / Math.abs(dy) : 1)
  return { x1: p.x, y1: p.y, x2: p.x + dx * t, y2: p.y + dy * t, back: b < a }
})
const groupOf = (s) => GROUPS.findIndex((g) => g.id === STATES[s].split(':')[0])
const dedupe = (xs) => xs.filter((x, i) => x !== xs[i - 1])

const hist = reactive([]), singleOnly = ref(false), hover = ref(-1)
let rand = mulberry32(7)
const states = computed(() => hist.reduce((acc, t) => [...acc, table[acc.at(-1)][t]], [0]))   // 每步查表前进
const state = computed(() => states.value.at(-1))
const finished = computed(() => hist.at(-1) === EOS_ID)
const curGroup = computed(() => groupOf(state.value))
const shown = computed(() => VOCAB.map((_, t) => t).filter((t) => !singleOnly.value || VOCAB[t].length === 1 || t === EOS_ID))
const legal = computed(() => shown.value.filter((t) => table[state.value][t] >= 0))
const output = computed(() => hist.filter((t) => t !== EOS_ID).map((t) => VOCAB[t]).join(''))
const parsed = computed(() => { try { return JSON.stringify(JSON.parse(output.value)) } catch { return '✗ 解析失败' } })
const crossed = (s, t) => (t === EOS_ID ? [] : dedupe((walk(s, VOCAB[t]) || []).map(groupOf)).filter((g, i) => i > 0 || g !== groupOf(s)))
const lastCross = computed(() => (hist.length ? crossed(states.value.at(-2), hist.at(-1)).length : 0))
const hoverPath = computed(() => (hover.value > 0 && !finished.value ? crossed(state.value, hover.value) : []))
const ok = (t) => !finished.value && table[state.value][t] >= 0   // aria-disabled 而非 disabled: 非法 token 也要能悬停看原因
const show = (p) => (p === '\n' ? '\\n' : p)
const hoverMsg = computed(() => {
  if (hover.value < 0) return `当前状态: ${GROUPS[curGroup.value].label} (DFA 状态 #${state.value} ${STATES[state.value]})`
  const p = show(VOCAB[hover.value])
  if (!ok(hover.value)) return `「${p}」在当前状态走不通 → next_state = −1 → logit 置 −inf`
  if (hover.value === EOS_ID) return 'EOS: 只在结束态合法'
  return hoverPath.value.length > 1 ? `「${p}」一口气跨了 ${hoverPath.value.length} 个状态: ${hoverPath.value.map((g) => GROUPS[g].label).join(' → ')}`
    : `「${p}」→ ${hoverPath.value.length ? GROUPS[hoverPath.value[0]].label : '留在 ' + GROUPS[curGroup.value].label}`
})
const randomStep = () => { const c = legal.value; if (c.length) hist.push(c[Math.floor(rand() * c.length)]) }
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 500px; touch-action: pan-x pan-y; }
.out { font-size: 13px; color: var(--text-muted); word-break: break-all; }
.out b { color: var(--text); font-weight: 500; }
.okmsg { color: var(--left); }
.edge { stroke: var(--text-dim); stroke-width: 1; }
.edge.back { stroke-dasharray: 5 3; }
.head { fill: var(--text-dim); }
.st rect { fill: var(--bg-elev); stroke: var(--border-strong); }
.st text { font-size: 10px; fill: var(--text-muted); text-anchor: middle; }
.st.via rect { stroke: var(--warn); stroke-width: 2; fill: color-mix(in srgb, var(--warn) 18%, var(--bg-elev)); }
.st.cur rect { stroke: var(--accent); stroke-width: 2.5; fill: var(--accent-soft); }
.st.cur text, .st.via text { fill: var(--text); }
.st .ord { font-size: 9px; fill: var(--warn); font-weight: 700; }
.hint { font-size: 11px; color: var(--text-muted); min-height: 2.6em; line-height: 1.5; margin: 4px 0 8px; }
.vocab { display: flex; flex-wrap: wrap; gap: 5px; }
.tokbtn { min-width: 34px; min-height: 30px; padding: 2px 8px; font-size: 12px; }
.tokbtn.multi:not([aria-disabled='true']) { border-color: var(--left); }
.tokbtn[aria-disabled='true'] { opacity: 0.28; cursor: not-allowed; }
.tokbtn.junk { text-decoration: line-through; }
</style>
