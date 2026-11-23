<!--
  token 级语法约束实验台 (对应 llm_infer/m14_structured_output/grammar.py:token_row / compile_token_table)。
  一件事: 真实 token 是多字符的, 一个 token 可能一口气跨过几个语法状态 —— 所以要把 "状态 × token → 落点" 离线编译成表。
-->
<template>
  <LabFrame
    title="结构化输出 — 用状态机限制每一步能选哪些 token, 并把它预编译成表"
    sub="- 模拟什么: 受限解码 (语法约束生成) 确保模型输出严格符合合法 JSON 语法的核心机制是在每个采样步前将一切会破坏语法的非法 token 的 logit 置为 −inf (使其采样概率归零)。合法 token 集合由有限状态机 FSM (此处采用字符级确定性有限自动机 DFA) 动态判定: 自动机记录当前 JSON 结构所处的语法位置, 每读入一个字符便按规则转移至下一状态。工程难点在于现代分词器的词表包含大量多字符子词 token (例如 {&quot;、&quot;: 与 true 各自均为单个 token), 单个 token 需在字符级 DFA 上连续推进多步, 唯有整段字符序列每一步转移均合法才允许放行。若在每个解码步对全词表在线逐字符模拟转移将产生高昂的 CPU 延迟, 因此标准做法是在推理前离线预编译一张「状态数 × 词表大小」的全局跳转表, 在线解码每步仅需按当前状态行做一次 O(1) 查表。本实验支持的受限子集语法为扁平对象 {&quot;key&quot;:value, …}, 其中 value 可取字符串、数值、true、false 或 null。
- 图上是什么: 视图由四个功能区域组成:
  - 上方状态转移图: 将底层字符级 DFA 归并为 10 个可视化语法状态框, 紫色边框标识当前所处状态, 实线箭头表示合法正向转移, 虚线箭头表示循环回边 (读取逗号后回退等待下一个键名)。鼠标悬停任意候选 token 时, 该 token 连续跨越的各个状态框会依次高亮为黄框并标出途经序号。
  - 状态图下方诊断栏: 实时解释当前悬停 token 被判为合法放行或非法拦截的具体状态转移路径。
  - 下方词表候选池: 将全部词表 token 渲染为交互按钮。置灰按钮表示在当前语法状态下被掩码屏蔽的非法 token; 带绿色边框的按钮标识当前合法的多字符复合 token; 带删除线的按钮为故意混入词表的恒非法干扰项 (如连续右花括号 }} 与裸换行符), 在任意语法状态下均无法走通。
  - 顶部「输出」预览栏: 实时拼接已选出的输出字符串, 到达结束符 EOS 时自动调用 JSON.parse 校验并展示解析结果。
- 可以做什么: 支持按钮点击推进与悬停探查:
  - 点击任意未置灰的合法 token 按钮向后推进一个解码步。
  - 点击「撤销」回退一步, 或点击「重置」清空当前生成状态。
  - 点击「随机合法一步」从当前状态允许的合法子集中随机采样一个 token。
  - 切换「只用单字符词表」开关屏蔽全部多字符复合 token, 对比纯单字符词表拼出相同 JSON 串所需的解码步数膨胀。
  - 悬停词表中的任意 token (含已被置灰屏蔽的非法项), 预览其在状态机上的落点或中途受阻位置。
- 应该看到: 在任意解码步下词表中绝大多数 token 均处于置灰屏蔽状态, 仅极少数符合当前语法期待的候选被放行; 选用多字符复合 token 时常能单步连续跨越两至三个语法状态。右侧四项读数含义如下:
  - 「当前状态下合法的 token / 词表大小」: 当前步未被掩码屏蔽的候选数量占总词表的比例。
  - 「上一个 token 跨过的状态数」: 大于 1 时标绿, 直观体现多字符子词在字符级自动机上的多步跳跃效应。
  - 「已用 token / 已出字符」: 对比模型前向解码步数与产出的有效字符总数, 衡量多字符子词节省的自回归步数。
  - 「预编译表 S × V」: 离线预计算状态跳转与合法性掩码矩阵的元素规模 (DFA 状态总数乘以词表大小)。"
    module="llm_infer/m14"
    run="python -m llm_infer.m14_structured_output.demo"
    :challenge="{
      ask: '点击「重置」后, 使用最少数量的 token 拼出 {&quot;id&quot;:42}, 共需要几个 token?',
      answer: '最少仅需 5 个正文 token: 依次选取 {&quot;、id、&quot;:、42 与 }, 最后再加 1 个 EOS 结束符收尾。其中多字符复合 token &quot;: 单步连续跨越了 2 个语法状态 (先闭合键名的右引号, 紧接着读入键值分隔冒号)。\n若勾选「只用单字符词表」拼接同一字符串, 则需逐字符依次生成 {、&quot;、i、d、&quot;、:、4、2、} 共 9 个正文 token, 外加末尾 EOS。\n判定多字符 token 是否合法的充要条件是其包含的完整字符序列能够从当前状态起在字符级 DFA 上全程无阻地走完, 而非仅检查首字符:\n- 例如 true: 从等待键值的 value? 状态出发连续合法转移 4 步, 准确落入等待「, 或 }」的后继状态。\n- 例如前缀子词 tru: 连续合法推进 3 步后停留在布尔字面量内部中间态, 同样判定为合法, 但下一步仅允许接续字符 e。\n- 例如非法组合 }}: 第一个 } 虽可闭合对象进入终止态, 但第二个 } 在终止态下无任何合法出边, 因此该 token 在全局所有状态下的跳转目标均记为 −1 (恒非法屏蔽)。\n这张规模为 S × V 的跳转与掩码表只需针对目标语法离线预编译一次 (遍历每个状态对全词表试走一遍), 在线解码每步仅需按当前状态索引读取一行布尔掩码并查表更新下一状态。若省去预编译而在每步对数万规模词表逐字符在线试走, 这部分纯 CPU 串行开销将直接阻塞在每次 GPU 前向计算与采样之间。本节 demo 脚本实测对比: 每步在线现算耗时 18.4 μs/步, 预编译查表仅需 0.086 μs/步, 加速达 215×。',
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
        <p>结束符 EOS 仅在自动机到达完整闭合的终止态时才解除屏蔽, 因此解码终止时输出必定为结构完整的合法 JSON, 无需依赖模型自身学会何时闭合括号。需要注意, logit 掩码仅保证输出的语法与结构合规, 不保证字段取值在事实或语义上正确。</p>
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
