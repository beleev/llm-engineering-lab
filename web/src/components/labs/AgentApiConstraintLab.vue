<!--
  history-editing check (对应 llm_agent/core/claude_llm.py 的 stale_thinking 与 m15_claude_api/demo.py 的 history_rows)。
  只讲一件事: 当前代 Claude 模型的 thinking 块绑定它之前的全部前缀, harness 改了旧历史, 其后的块就失效。
  transcript 与 demo [3] 相同: 两道算术题, 8 条 API 消息, 4 个 thinking 块。
-->
<template>
  <LabFrame
    title="只追加 — 改一次旧历史, 真实 API 上哪些 thinking 块会失效"
    sub="- 模拟什么: Claude API 对 harness 的 history-editing check (历史编辑检查)。Opus 5.5、Sonnet 5.5、Fable 5.2 等当前代模型返回的每个 thinking 块都带签名, 签名绑定产生它时的 system、tools 集合和它之前的全部消息; 下一次请求发回这个块时, API 核对这段前缀是否逐字节不变, 变了就判失效。本台的 transcript 与 m15 demo [3] 相同: 规则替身跑两道算术题得到 8 条 API 消息, 4 条 assistant 消息各带一个 thinking 块。
- 图上是什么: 每行是下一次请求里的一条消息, 从上到下按顺序排列:
  - 左侧色条: 绿色表示这条消息落在原样保留的前缀里 (prompt cache 也能读到这里), 黄色表示内容被改过, 灰色表示在改动之后, 删除线表示这一次没有再发送。
  - 行末的 thinking 标记: 有效、失效 (请求会被拒或块被丢弃)、没再发送三种状态。
  - 顶部的 system 行: system 或 tools 一变, 后面所有消息的前缀都跟着变。
- 可以做什么: 第一排按钮选择 harness 对历史做的改动:
  - 「追加下一问」: 只在末尾加一条 user 消息。
  - 「清旧工具结果」: m14 第 1 档, 把第一个工具结果换成占位符。
  - 「重拼 system」: m04 按每个问题现拼记忆进顶层 system。
  - 「摘要旧轮, 保留当前轮」: m14 第 2 档的本地摘要。
  - 「整段换成摘要」: 只发一条摘要和新问题, 不带任何旧 thinking。
  - 「服务端 compaction」: 回复里多一个 compaction 块, 客户端原样追加。
  - 直接点击任意一条消息: 模拟改写这一条 (例如重写一段提醒文字)。
  - 第二排切换不匹配时的处理: 默认 error (请求 400), 或 drop_block (丢弃失效块后继续)。
- 应该看到: 只有追加类的改动让 4 个块全部有效。右侧读数:
  - 「有效 thinking 块」: 前缀没变、下一次还能被模型读到的块数。
  - 「失效 thinking 块」: 发回去会触发检查失败的块数; 清旧工具结果时为 3, 重拼 system 时为 4。
  - 「没再发送」: 被摘要换掉的块数, 推理随之丢失。
  - 「可读缓存的前缀」: 与上一次请求逐条相同的消息数, 和 thinking 有效性用的是同一条规则。"
    module="llm_agent/m15"
    run="python -m llm_agent.m15_claude_api.demo"
    :challenge="{
      ask: '选「摘要旧轮, 保留当前轮」。当前轮的两条 assistant 消息一个字节都没改, 先预测: 它们的 thinking 块有效还是失效? 换成「整段换成摘要」又怎样?',
      answer: '保留当前轮时, 当前轮的 2 个 thinking 块全部失效, 旧轮的 2 个没再发送; 整段换成摘要时没有失效的块, 但 4 个块都没再发送。\n- 签名绑定的是前缀, 不是块本身: 当前轮的块产生时, 它前面是完整的 8 条历史; 摘要换掉旧轮后前缀变了, 块的内容再完整也通不过检查。\n- 本库把摘要放进顶层 system, system 一变前缀从第 0 条起就不同, 所以「可读缓存的前缀」也归零。\n- 整段换成摘要不带任何旧 thinking, 没有可以失效的块, 请求能正常发出, 代价是旧推理全部丢失, 模型只能从摘要文字里接着做。\n- 想压缩又保住推理, 要把压缩交给服务端 compaction 或 context editing: 它们改的是服务端的副本, 客户端发出的历史仍只追加, 不算编辑。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="o in OPS" :key="o.id" type="button" :class="{ active: op === o.id }" @click="op = o.id">{{ o.label }}</button>
      </div>
      <div class="row">
        <span class="tip">前缀不匹配时:</span>
        <button v-for="b in ['error', 'drop_block']" :key="b" type="button" :class="{ active: behavior === b }" @click="behavior = b">{{ b }}</button>
      </div>
    </template>

    <div class="msgs">
      <div class="msg head" :class="view.head ? 'changed' : 'kept'">
        <span class="mono idx">sys</span><span class="txt">{{ view.head ? 'system 变了: ' + HEADS[op] : 'system + tools 与上一次相同' }}</span>
      </div>
      <button v-for="r in view.rows" :key="r.i" type="button" class="msg" :class="r.cls"
        :aria-label="`第 ${r.i + 1} 条 ${r.role}, ${r.text}, 点击改写这一条`" @click="edit(r.i)">
        <span class="mono idx">{{ r.i + 1 }}</span>
        <span class="role mono">{{ r.role }}</span>
        <span class="txt">{{ r.text }}</span>
        <span v-if="r.think" class="think mono" :class="r.think">{{ LABEL[r.think] }}</span>
      </button>
      <div class="msg after"><span class="mono idx">+</span><span class="role mono">user</span><span class="txt">再算 6 * 7 (这一次新追加)</span></div>
    </div>
    <p class="detail">{{ verdict }}</p>

    <template #stats>
      <div class="kv"><span>有效 thinking 块</span><b :class="view.valid === TOTAL ? 'good' : 'bad'">{{ view.valid }}/{{ TOTAL }}</b></div>
      <div class="kv"><span>失效 thinking 块</span><b :class="view.stale ? 'bad' : 'good'">{{ view.stale }}</b></div>
      <div class="kv"><span>没再发送</span><b :class="view.dropped ? 'bad' : 'good'">{{ view.dropped }}</b></div>
      <div class="kv"><span>可读缓存的前缀 (条)</span><b :class="view.keep === MSGS.length ? 'good' : 'bad'">{{ view.keep }}/{{ MSGS.length }}</b></div>
      <p class="lab-note">5 种按钮改动的读数与 m15 demo [3] 的输出逐项一致 (服务端 compaction 与点击改写是本台补充的情形)。规则是 stale_thinking 的近似: 按整条消息比较, 不模拟「从最前面按顺序删掉 thinking 块仍合法」这类例外, 也没有对真实 API 运行过。error 是 2026-08-31 及之后创建的账号在 Opus 5.5 / Sonnet 5.5 上的默认行为, Fable 5.2 对所有账号都是如此。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

// 与 m15 demo [3] 的 8 条 API 消息相同 (hook 提醒已并进相邻的 user 消息)
const MSGS = [
  { role: 'user', text: '计算 17 * 23' },
  { role: 'assistant', text: 'tool_use calculator 17 * 23', think: true },
  { role: 'user', text: 'tool_result 17 * 23 = 391' },
  { role: 'assistant', text: '回答 391', think: true },
  { role: 'user', text: '再算 10 / 4' },
  { role: 'assistant', text: 'tool_use calculator 10 / 4', think: true },
  { role: 'user', text: 'tool_result 10 / 4 = 2.5' },
  { role: 'assistant', text: '回答 2.5', think: true },
]
const TOTAL = MSGS.filter((m) => m.think).length
const OPS = [
  { id: 'append', label: '追加下一问' }, { id: 'clear', label: '清旧工具结果 (m14 第 1 档)' },
  { id: 'system', label: '重拼 system (m04)' }, { id: 'keeptail', label: '摘要旧轮, 保留当前轮 (m14 第 2 档)' },
  { id: 'simple', label: '整段换成摘要' }, { id: 'server', label: '服务端 compaction' },
]
const HEADS = { system: '拼进了这一问检索到的记忆', keeptail: '旧轮摘要并进了顶层 system', simple: '旧轮摘要并进了顶层 system' }
const LABEL = { valid: 'thinking 有效', stale: 'thinking 失效', dropped: 'thinking 没再发送' }
const op = ref('clear')
const editAt = ref(-1) // 读者点中的那条消息; 只在 op === 'edit' 时有意义
const behavior = ref('error')
function edit(i) { op.value = 'edit'; editAt.value = i }

// 每种改动: system 是否变了、哪些原消息还发、哪一条被改过
function plan(id) {
  const all = MSGS.map((_, i) => i)
  if (id === 'clear') return { head: false, sent: all, changed: 2 } // 第一个工具结果换成占位符
  if (id === 'system') return { head: true, sent: all, changed: -1 }
  if (id === 'keeptail') return { head: true, sent: all.slice(4), changed: -1 } // 从最后一条用户 prompt 起保留
  if (id === 'simple') return { head: true, sent: [], changed: -1 }
  if (id === 'edit') return { head: false, sent: all, changed: editAt.value }
  return { head: false, sent: all, changed: -1 } // append / server: 客户端只追加
}

const view = computed(() => {
  const p = plan(op.value)
  // ★ 与 stale_thinking 同一条规则: system 变了前缀为 0; 否则保留到第一条被改的消息之前, 其后的 thinking 全部失效
  const keep = p.head ? 0 : p.changed >= 0 ? p.changed : MSGS.length
  const rows = MSGS.map((m, i) => {
    const sent = p.sent.includes(i)
    const think = !m.think ? '' : !sent ? 'dropped' : i < keep ? 'valid' : 'stale'
    const cls = !sent ? 'gone' : i === p.changed ? 'changed' : i < keep ? 'kept' : 'after'
    const text = i !== p.changed ? m.text : op.value === 'clear' ? '[cleared: 13 chars]' : m.text + ' (已改写)'
    return { i, role: m.role, text, think, cls }
  })
  const count = (t) => rows.filter((r) => r.think === t).length
  return { head: p.head, rows, keep, valid: count('valid'), stale: count('stale'), dropped: count('dropped') }
})

const verdict = computed(() => {
  const v = view.value
  const tail = op.value === 'server' ? ' 服务端在自己的副本上用 compaction 块替换旧内容, 客户端发出的历史仍只追加, 不算编辑。' : ''
  if (!v.stale) return `请求正常发出: ${v.valid} 个 thinking 块有效${v.dropped ? `, ${v.dropped} 个随摘要丢失, 模型只能从摘要文字接着做` : ''}。${tail}`
  return behavior.value === 'error'
    ? `请求被拒: 400 invalid_request_error, 第一个失效块在第 ${v.rows.find((r) => r.think === 'stale').i + 1} 条消息。重试同一请求体仍会失败, 恢复办法是剥掉全部 thinking 块重发, 或改用 drop_block。`
    : `请求继续: API 丢弃第一个失效块及其后全部 ${v.stale} 个 thinking 块, 模型在没有这些推理的情况下重新规划。`
})
</script>

<style scoped>
.msgs { display: flex; flex-direction: column; gap: 3px; }
.msg { display: flex; align-items: center; gap: 8px; padding: 3px 8px; border: 1px solid var(--border); border-left: 4px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text); font-size: 12px; text-align: left; min-height: 0; }
.msg.kept { border-left-color: var(--left); }
.msg.changed { border-left-color: var(--warn); background: var(--accent-soft); }
.msg.after { border-left-color: var(--text-dim); }
.msg.gone { opacity: 0.55; }
.msg.gone .txt { text-decoration: line-through; }
.idx { width: 26px; flex-shrink: 0; color: var(--text-dim); font-size: 11px; }
.role { width: 64px; flex-shrink: 0; color: var(--text-muted); font-size: 11px; }
.txt { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.think { flex-shrink: 0; font-size: 11px; padding: 1px 6px; border-radius: var(--radius-sm); }
.think.valid { color: var(--left); border: 1px solid var(--left); }
.think.stale { color: var(--danger); border: 1px solid var(--danger); }
.think.dropped { color: var(--text-dim); border: 1px dashed var(--text-dim); }
.detail { margin-top: 8px; font-size: 12px; color: var(--text-muted); line-height: 1.7; }
.tip { font-size: 12px; color: var(--text-muted); }
</style>
