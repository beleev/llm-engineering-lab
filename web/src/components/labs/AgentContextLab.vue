<!--
  上下文预算 (对应 llm_agent/core/memory.py 的三档瘦身 + agent.py:_assemble_context / _compact)。
  只讲一件事: 同样是"塞不下了", 截断 / 清工具结果 / 模型摘要 丢掉的信息完全不同。
-->
<template>
  <LabFrame
    title="上下文预算 — 塞不下的时候, 丢什么?"
    sub="- 模拟什么: agent 每一轮都要把整段对话重新发给模型, 而模型的上下文窗口有上限 (这里用一个可拖的「预算」代替)。会话越长越塞不下, 到时候必须丢东西。这个台子模拟一段 14 条消息的编程会话 (用户下任务、模型调工具、工具返回大段结果), 里面埋了 6 个之后还会用到的事实 (目标、文件行号、中途的决定、失败的测试用例等), 比较四种瘦身策略各自丢掉了什么。
      - 图上是什么: 三部分:
        - 顶部的堆叠条: 当前发给模型的上下文由哪几块组成, system 提示、skills 目录、memory、摘要、对话文本、工具结果; 红色虚线是预算线, 条超过虚线就是塞不下。
        - 下面 14 行: 每一条消息。左边序号, 中间的条是它压缩后的大小 (黄色是工具结果, 其它是对话), 右边数字是「压缩后 ← 原来」。点一条消息, 下方显示它的原文和「模型现在看到」的样子。
        - 右侧列表: 6 个事实的命运。✔ 还在上下文里, ↻ 正文被清掉但 tool_use 还在、再调一次工具就能取回, ✘ 已经丢失、取不回来。
      - 可以做什么: 五个策略按钮 (第一个是「什么都不做」) 和下面几个控件:
        - 「硬截断」: 反例, 头尾各留一点, 中间每条只留开头 8 tok 的纯文本。
        - 「清旧工具结果」: 只把旧的工具返回正文换成 [cleared: N chars] 占位, 保留最近几个。
        - 「模型摘要」: 再调一次模型, 把当前用户轮之前的历史写成一条 220 tok 的摘要。
        - 「级联」: 真实 loop 的做法, 先清, 还不够再摘要。
        - 拖「上下文预算」: 改窗口大小 (全部消息加起来 12260 tok, 预算比它小才会触发瘦身)。
        - 拖「保留最近几个工具结果」: 改清理时留几个不清。
        - 「PreCompact hook」: 只在摘要触发时可用, 它告诉摘要器必须保留文件位置和失败用例。
        - 「memory 工具」: 只在中途的决定被丢掉时可用, 它把决定写进上下文之外的 /memories 文件。
      - 应该看到: 几组对照和右侧读数:
        - 预算 6000、级联: 只清了 5 个旧工具结果, 2 个事实变成 ↻ 可重取, 没有丢失, 额外模型调用 0。
        - 切到「硬截断」: 3 个事实 ✘ 丢失, 而且「tool_use/result 配对」变成「拍平成文本」, 模型连当时调过什么工具都不知道了。
        - 切到「模型摘要」: 目标和决定保住了, 但行号、数值这类细节丢了 (✘ 2 个), 打开 PreCompact hook 才能保住, 代价是「额外模型调用」1 次。
        - 右侧「压缩后总量 / 预算」: 红色表示仍然塞不下。"
    module="llm_agent/m14"
    run="python -m llm_agent.m14_context_engineering.demo"
    :challenge="{
      ask: '预算 6000。先猜「硬截断」和「清旧工具结果」哪个留下的事实多。再想想: 被清掉的工具结果里的事实, 和被截断丢掉的事实, 有什么区别?',
      answer: '- 清旧工具结果留下的多: 6 个事实里 4 个原样保留, 2 个变成「可重取」, 没有丢失。被清掉的只是工具返回的正文, 模型发出的 tool_use 块 (调了什么工具、什么参数) 原样保留, 所以模型知道当时读的是哪个文件, 再调一次工具就能把内容取回来。这种做法叫 just-in-time 检索: 上下文里只留「指针」, 不留「内容」。\n- 硬截断只留下 3 个: 它丢的是对话本身, 比如 msg[5] 里中途做的决定 (用方案 B), 这种东西没有任何工具可以再取回来。它还把 tool_use 和 tool_result 拍平成纯文本, 配对结构没了, 模型连自己调过什么工具都不知道。\n- 两者的区别就在「能不能重新拿到」: 工具结果是环境里的事实, 随时可以再读; 对话里的决定只存在于这段历史里, 丢了就丢了。\n- 模型摘要是第三条路: 最省空间, 但有损、要多花一次模型调用; 行号、数值这类细节摘要器通常不会保留, 要靠 PreCompact hook 点名。\n真实的 loop 是级联: 先清 (零成本), 不够再摘要。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="s in STRATS" :key="s.id" type="button" :class="{ active: strat === s.id }" @click="strat = s.id">{{ s.label }}</button>
      </div>
      <LabSlider v-model="budget" label="上下文预算" :min="2000" :max="14000" :step="500" unit=" tok" />
      <LabSlider v-model="keep" label="保留最近几个工具结果" :min="0" :max="3" />
      <div class="row">
        <button type="button" :class="{ active: hook }" :aria-pressed="hook" :disabled="!view.summarized" @click="hook = !hook">{{ hook ? '☑' : '☐' }} PreCompact hook: 摘要必须保留文件位置与失败用例</button>
        <button type="button" :class="{ active: memory }" :aria-pressed="memory" :disabled="!memUseful" @click="memory = !memory">{{ memory ? '☑' : '☐' }} memory 工具: 把决定写进 /memories</button>
      </div>
      <ul v-if="!view.summarized || !memUseful" class="pts off">
        <li v-if="!view.summarized"><b>PreCompact hook 置灰:</b> 它只在摘要前触发, 当前没有触发摘要。切到「模型摘要」, 或在级联下把预算拖小。</li>
        <li v-if="!memUseful"><b>memory 工具置灰:</b> 它救的是丢掉的决定, 当前决定 (msg[5]) 没丢。切到「硬截断」再试。</li>
      </ul>
    </template>

    <div class="stack" :title="`预算 ${budget}`">
      <div v-for="s in segs" :key="s.k" class="seg" :class="s.k" :style="{ width: s.v / scale * 100 + '%' }" :title="`${s.label} ${s.v}`" />
      <div class="mark" :style="{ left: budget / scale * 100 + '%' }"><span class="mono">预算</span></div>
    </div>
    <p class="legend"><span v-for="s in segs" :key="s.k"><i :class="s.k" />{{ s.label }} {{ fmtNum(s.v) }}</span></p>

    <div class="msgs">
      <button v-for="m in view.msgs" :key="m.i" type="button" class="m" :class="[m.kind, m.change, { sel: sel === m.i }]" @click="sel = m.i">
        <span class="mono idx">{{ m.i }}</span>
        <span class="bar" :style="{ width: Math.max(2, m.after / 2600 * 100) + '%' }" />
        <span class="mono tk">{{ m.after }}<em v-if="m.after !== m.tok"> ← {{ m.tok }}</em></span>
      </button>
    </div>
    <p class="detail"><b class="mono">msg[{{ sel }}] {{ cur.role }}</b> {{ cur.text }}<br /><span class="now">模型现在看到: {{ cur.seen }}</span></p>

    <template #stats>
      <div class="kv"><span>压缩后总量 / 预算</span><b :class="view.total > budget ? 'bad' : 'good'">{{ fmtNum(view.total) }} / {{ fmtNum(budget) }}</b></div>
      <div class="kv"><span>6 个事实: 保留 · 可重取 · 丢失</span><b :class="count.lost ? 'bad' : 'good'">{{ count.kept }} · {{ count.refetch }} · {{ count.lost }}</b></div>
      <div class="kv"><span>tool_use/result 配对</span><b :class="view.paired ? 'good' : 'bad'">{{ view.paired ? '完好' : '拍平成文本' }}</b></div>
      <div class="kv"><span>额外模型调用 (写摘要花的)</span><b>{{ view.calls }}</b></div>
      <ul class="facts">
        <li v-for="f in facts" :key="f.id" :class="f.status"><button type="button" @click="sel = f.msg">{{ ICON[f.status] }} {{ f.label }} <span class="mono">msg[{{ f.msg }}]</span></button></li>
      </ul>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, sum } from '@/utils/labmath.js'

const STRATS = [{ id: 'none', label: '什么都不做' }, { id: 'truncate', label: '硬截断 (反例)' }, { id: 'clear', label: '清旧工具结果' }, { id: 'summary', label: '模型摘要' }, { id: 'cascade', label: '级联: 先清, 不够再摘要 (真实 loop)' }]
const ICON = { kept: '✔', refetch: '↻', lost: '✘' }
const SYSTEM = 800, SKILLS = 360, MEM = 60, SUMMARY = 220, CLEARED = 6, GIST = 8 // token; 32 字符 ≈ 8 token

// kind: u=用户输入 a=assistant(text/tool_use) r=tool_result。 fact.off = 事实距消息开头多少 token
const MSGS = [
  { role: 'user', kind: 'u', tok: 40, text: '目标: 修复登录超时 bug, 不要改 public API。' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use grep("timeout")' },
  { role: 'user', kind: 'r', tok: 1800, text: 'tool_result: grep 输出 1800 tok …… 超时常量在 auth/session.py:42 ……' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("auth/session.py")' },
  { role: 'user', kind: 'r', tok: 2600, text: 'tool_result: session.py 全文 2600 tok' },
  { role: 'assistant', kind: 'a', tok: 80, text: '看完 session 与 refresh 流程、比较两种做法之后, 决定: 用方案 B (续期 token), 不改表结构。 tool_use edit(…)' },
  { role: 'user', kind: 'r', tok: 300, text: 'tool_result: edit ok' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use run_tests()' },
  { role: 'user', kind: 'r', tok: 2200, text: 'tool_result: 测试日志 2200 tok …… test_refresh 失败: 期望 3600 实际 1800 ……' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("tests/test_refresh.py")' },
  { role: 'user', kind: 'r', tok: 2400, text: 'tool_result: test_refresh.py 全文 2400 tok' },
  { role: 'user', kind: 'u', tok: 30, text: '顺便把超时改成可配置的。' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("settings.toml")' },
  { role: 'user', kind: 'r', tok: 1500, text: 'tool_result: settings.toml 1500 tok …… 配置项放在 [auth] 段 ……' },
]
const FACTS = [
  { id: 'A', label: '目标与约束', msg: 0, off: 0 }, { id: 'B', label: '超时常量的文件:行号', msg: 2, off: 900, detail: true },
  { id: 'C', label: '中途的决定 (方案 B)', msg: 5, off: 20, decision: true }, { id: 'D', label: '失败用例与数值', msg: 8, off: 1200, detail: true },
  { id: 'E', label: '用户最新指令', msg: 11, off: 0 }, { id: 'F', label: '配置项位置', msg: 13, off: 300 },
]
const LAST_USER = 11 // 当前用户轮的起点: 摘要只压它之前的历史, 当前轮原样保留

const strat = ref('cascade'), budget = ref(6000), keep = ref(1), hook = ref(false), memory = ref(false), sel = ref(5)
const base = computed(() => SYSTEM + SKILLS + (memory.value ? MEM : 0))
const full = computed(() => base.value + sum(MSGS.map((m) => m.tok)))

const view = computed(() => {
  const room = budget.value - base.value
  let msgs = MSGS.map((m, i) => ({ ...m, i, after: m.tok, change: '', seen: '原样' }))
  let paired = true, calls = 0, summarized = false
  const total = () => sum(msgs.map((m) => m.after))
  // memory.py:clear_tool_results —— 只保留最近 keep 个结果的正文; tool_use 块和配对结构不动
  const clear = () => {
    const rs = msgs.filter((m) => m.kind === 'r' && m.change !== 'gone')
    rs.slice(0, Math.max(0, rs.length - keep.value)).forEach((m) => Object.assign(m, { after: CLEARED, change: 'cleared', seen: `[cleared: ${m.tok * 4} chars] (tool_use 还在, 可以重新调用)` }))
  }
  // agent.py:_compact —— 当前用户轮之前的历史换成一条模型写的摘要
  const compact = () => {
    msgs.forEach((m) => { if (m.i < LAST_USER) Object.assign(m, { after: m.i === 0 ? SUMMARY : 0, change: m.i === 0 ? 'summary' : 'gone', seen: m.i === 0 ? '[compact summary of 11 messages] 目标 / 已完成 / 关键结果 / 待办' : '(已并入摘要)' }) })
    calls = 1; summarized = true
  }
  if (full.value > budget.value) {
    if (strat.value === 'truncate') {
      // memory.py:truncate_messages —— 头 2 + 尾 2 各留 edge, 中间每条只留 32 字符
      const edge = Math.max(10, Math.floor(room / 6))
      msgs.forEach((m, i) => {
        const keepTok = i < 2 || i >= msgs.length - 2 ? Math.min(m.tok, edge) : Math.min(m.tok, GIST)
        Object.assign(m, { after: keepTok, change: keepTok < m.tok ? 'cut' : '', seen: keepTok < m.tok ? `只剩开头 ${keepTok} tok 的纯文本` : '拍平成纯文本' })
      })
      paired = false
    } else if (strat.value === 'clear') clear()
    else if (strat.value === 'summary') { compact(); clear() }
    else if (strat.value === 'cascade') { clear(); if (base.value + total() > budget.value) { compact(); clear() } } // ★ 由便宜到贵
  }
  return { msgs, total: base.value + total(), paired, calls, summarized }
})

const statusOf = (f, mem) => {
  const m = view.value.msgs[f.msg]
  let status = 'kept'
  if (m.change === 'cut') status = f.off < m.after ? 'kept' : 'lost'
  else if (m.change === 'cleared') status = 'refetch'
  else if (m.change === 'gone') status = f.detail ? (hook.value ? 'kept' : 'lost') : 'kept' // 摘要留得住目标和决定, 留不住行号和数值
  return status === 'lost' && f.decision && mem ? 'kept' : status // 写进 /memories 的东西在窗口之外, 任何压缩都碰不到
}
const facts = computed(() => FACTS.map((f) => ({ ...f, status: statusOf(f, memory.value) })))
// 两个开关只在有东西可救时才起作用: hook 要等摘要触发, memory 要等决定被丢
const memUseful = computed(() => FACTS.some((f) => f.decision && statusOf(f, false) === 'lost'))
const count = computed(() => Object.fromEntries(['kept', 'refetch', 'lost'].map((s) => [s, facts.value.filter((f) => f.status === s).length])))

const segs = computed(() => {
  const by = (pred) => sum(view.value.msgs.filter(pred).map((m) => m.after))
  return [
    { k: 'sys', label: 'system', v: SYSTEM }, { k: 'skl', label: 'skills 目录', v: SKILLS }, { k: 'mem', label: 'memory', v: memory.value ? MEM : 0 },
    { k: 'sum', label: '摘要', v: by((m) => m.change === 'summary') }, { k: 'txt', label: '对话', v: by((m) => m.kind !== 'r' && m.change !== 'summary') }, { k: 'res', label: '工具结果', v: by((m) => m.kind === 'r') },
  ].filter((s) => s.v > 0)
})
const scale = computed(() => Math.max(full.value, budget.value) * 1.02)
const cur = computed(() => view.value.msgs[sel.value])
</script>

<style scoped>
.stack { position: relative; display: flex; height: 22px; border: 1px solid var(--border); border-radius: 4px; margin-top: 14px; }
.seg { height: 100%; }
.sys, i.sys { background: var(--text-dim); } .skl, i.skl { background: var(--accent); } .mem, i.mem { background: var(--right); }
.sum, i.sum { background: var(--left); } .txt, i.txt { background: var(--eye); } .res, i.res { background: var(--warn); }
.mark { position: absolute; top: -12px; bottom: -4px; border-left: 2px dashed var(--danger); }
.mark span { position: absolute; top: -4px; left: 4px; font-size: 10px; color: var(--danger); white-space: nowrap; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11px; color: var(--text-dim); margin: 6px 0 12px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
.msgs { display: flex; flex-direction: column; gap: 3px; }
.m { display: grid; grid-template-columns: 22px 1fr 110px; gap: 8px; align-items: center; min-height: 0; padding: 2px 6px; border: 1px solid transparent; border-radius: 4px; background: none; color: var(--text-muted); }
.m .idx { font-size: 10px; color: var(--text-dim); }
.m .bar { height: 12px; border-radius: 2px; background: var(--eye); }
.m.r .bar { background: var(--warn); } .m.summary .bar { background: var(--left); }
.m.cleared .bar, .m.cut .bar { opacity: 0.45; } .m.gone { opacity: 0.35; }
.m .tk { font-size: 10px; text-align: right; } .m .tk em { font-style: normal; color: var(--text-dim); }
.m.sel { border-color: var(--accent); }
.detail { margin-top: 10px; font-size: 12px; color: var(--text-muted); line-height: 1.7; min-height: 64px; }
.detail b { color: var(--text); font-weight: 500; } .now { color: var(--accent); }
.facts { list-style: none; display: flex; flex-direction: column; gap: 2px; }
.facts button { min-height: 0; padding: 2px 4px; border: 0; background: none; font-size: 12px; text-align: left; color: inherit; }
.facts .kept { color: var(--left); } .facts .refetch { color: var(--warn); } .facts .lost { color: var(--danger); }
.facts .mono { font-size: 10px; color: var(--text-dim); }
.off { font-size: 12px; color: var(--text-muted); }
</style>
