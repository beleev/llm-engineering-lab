<!--
  Agent loop 步进器 (对应 llm_agent/core/agent.py 的 Agent.run)。
  只讲一件事: loop 就是"往 messages 里追加 content block", 而每次问模型都要重发整个列表。
-->
<template>
  <LabFrame
    title="Agent loop — 消息列表怎么一块一块长出来, 每问一次模型为什么都要重发全部历史"
    sub="- 模拟什么: 最小 agent 执行两步复合任务「查一下 kv cache 文档, 再算 16*64」的完整过程。agent 的核心仅是一个循环 (loop):
  1. 将截至当前的完整对话历史 (即 messages 列表) 发送给模型。
  2. 模型要么返回纯文本作为最终答复并结束循环, 要么返回包含目标工具名与参数的 tool_use 请求。
  3. 外围确定性代码 (harness) 实际执行对应工具, 将返回值封装为 tool_result 追加至 messages 末尾, 随后再次将完整列表发送给模型。
  4. 上述一次请求与回填构成一轮 (turn), 循环持续至模型给出最终回答或触及轮数硬上限 max_turns。模型自身不保留跨请求状态, 每次推理仅依赖传入的 messages 列表。
- 图上是什么: 从上到下直接对应内存中的 messages 列表结构:
  - 每个外层方框: 代表一条消息。左上角标注消息角色, user 表示传给模型的外部输入, assistant 表示模型输出; 标有「第 N 次模型调用的输出」的方框对应第 N 次请求返回的内容。
  - 方框内的每个子格: 代表一个 content block, 类型包括普通文本、tool_use 或 tool_result。
  - tool_use: 模型发起的工具调用块, 以紫色边框标识并带有唯一 id。
  - tool_result: 工具执行结果块, 以绿色边框标识并通过 tool_use_id 关联对应的调用请求; 红色文字表示标记了 is_error 为 true 的执行错误。
  - 鼠标悬停交互: 将指针停留在任意 tool_use 或 tool_result 上时, 与之配对的另一端区块会以黄色边框高亮。
- 可以做什么: 操作两个功能开关、一根滑杆与底部播放器:
  - 「并行工具调用」: 开启后, 模型在首次回复中同时发出查文档与算术两个 tool_use 块, 两个执行结果合并封装在同一条 user 消息中回填, 全程减少一次模型调用。
  - 「工具出错」: 开启后, 模型首次调用算术工具时传错参数名 (误传为 expression, 而工具 Schema 要求 expr), 工具返回带 is_error 标记的报错信息, 模型在下一轮根据报错自行修正参数。
  - 「max_turns」: 循环最大轮数硬上限, 默认值 6 足以完成任务; 若调小至 1 或 2, 可观察任务未完成时 loop 如何被强制截断。
  - 底部播放器: 按 content block 粒度逐步展开消息列表。点击「下一步」推进一格, 拖动「时间轴」可跳转至任意步骤, 右侧面板仅统计当前已展开部分的数据。
- 应该看到: 消息列表的增长规律与右侧读数变化:
  - 列表单调追加: 每个工具轮次均以「assistant 发出 tool_use、紧随一条 user 回填 tool_result」成对出现; 尽管 tool_result 由本地工具生成, 在协议上仍归入 user 角色消息, 因为对模型而言它属于外部环境提供的观测输入。
  - 调用次数变化: 默认串行配置下模型共被调用 3 次 (查文档、算术、生成最终答复), 开启并行后降至 2 次, 开启工具出错后增至 4 次。
  - 右侧「模型调用次数」: 截至当前回放位置已向模型发起的请求总数。
  - 「当前上下文」: 当前已展开的全部 block 加上 system prompt 的 token 估算值 (按中文每字 1 token、其余每 4 个字符 1 token 粗略折算, 其中 system prompt 占 8 token)。
  - 「累计 input tokens」: 历次模型调用传入上下文长度的累加和。由于每轮均需重发完整历史, 该数值远大于当前上下文长度, 超过当前上下文 2 倍时标红提示。
  - 「结束方式」: 显示循环是以 final (模型正常收尾) 还是触及 max_turns 强制终止。"
    module="llm_agent/m01"
    run="python -m llm_agent.m01_agent_loop.demo"
    :challenge="{
      ask: '关闭「并行工具调用」、开启「工具出错」, 并将播放进度拖至末尾。先预测: 模型总共被调用几次? 「累计 input tokens」约为「当前上下文」的多少倍?',
      answer: '共调用 4 次; 累计 input 为 258 token, 最终上下文为 116 token, 前者超过后者的 2 倍 (2 倍多一点)。\n四次模型调用的具体过程依次为:\n1. 第 1 次调用: 模型发起 search_docs 检索文档。\n2. 第 2 次调用: 模型发起参数名错误的 calculator 调用 (误用 expression 而非 expr)。执行层捕获错误后未抛出未处理异常, 而是封装一条带 is_error 标记的 tool_result 追加至消息列表。\n3. 第 3 次调用: 模型读取报错详情后, 将参数名修正为 expr 重新发起调用。\n4. 第 4 次调用: 模型获取计算结果 1024, 输出最终文本答复。\n由于每次推理均需重发 system prompt 与截至当时的全部历史消息, 累计 input 等于 4 次调用各自上下文长度的总和 (每次请求均比上一次更长), 而非末次上下文大小。随着交互轮数增加, 每轮重发的历史前缀不断变长, 累计 token 消耗近似按轮数的平方增长, 这正是长程 agent 会话成本高昂的原因; 并行工具调用与上下文压缩均旨在压低这一累加开销。\n在出错轮次中, 将工具执行失败转化为结构化的 tool_result 回填给模型, 使模型得以利用上下文自主修正参数并继续循环; 若执行层直接抛出异常中断进程, 模型便失去了自愈纠错的机会。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: parallel }" @click="parallel = !parallel">并行工具调用: {{ parallel ? '开' : '关' }}</button>
        <button type="button" :class="{ active: toolError }" @click="toolError = !toolError">工具出错 (参数名写错): {{ toolError ? '开' : '关' }}</button>
      </div>
      <LabSlider v-model="maxTurns" label="max_turns (轮数上限)" :min="1" :max="6" />
      <StepPlayer :stepper="stepper" :label="`已揭开 block ${stepper.step.value + 1}/${frames.length}`" />
    </template>

    <div class="msgs">
      <div v-for="(m, i) in visible" :key="i" class="msg" :class="m.role">
        <span class="role mono">{{ m.role }}<em v-if="m.call"> · 第 {{ m.call }} 次模型调用的输出</em></span>
        <button
          v-for="(b, j) in m.blocks" :key="j" type="button" class="block mono"
          :class="[b.type, { err: b.is_error, linked: hoverId && (b.id === hoverId || b.tool_use_id === hoverId) }]"
          @mouseenter="hoverId = b.id || b.tool_use_id || ''" @mouseleave="hoverId = ''"
          @focus="hoverId = b.id || b.tool_use_id || ''" @blur="hoverId = ''"
        >{{ show(b) }}</button>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>模型调用次数 (列表被发给模型几次)</span><b>{{ now.calls }}</b></div>
      <div class="kv"><span>当前上下文 (列表 + system prompt)</span><b>{{ now.ctx }} tok</b></div>
      <div class="kv"><span>累计 input tokens (每次调用时的上下文之和)</span><b :class="now.input > 2 * now.ctx ? 'bad' : ''">{{ now.input }}</b></div>
      <div class="kv"><span>结束方式</span><b :class="sim.forced ? 'bad' : 'good'">{{ done ? (sim.forced ? 'max_turns 强停' : 'final') : '进行中' }}</b></div>
      <div class="lab-note">
        <p>tool_use 与 tool_result 在消息序列中的配对约束:</p>
        <ul class="pts">
          <li>tool_result 放在 <b>user</b> 消息里, 靠 tool_use_id 指回 assistant 那条 tool_use。</li>
          <li>每个 tool_use 都必须有一个配对的 tool_result, 真实 API 缺一个就直接拒绝整段历史。</li>
          <li>同一轮发出的多个调用, 它们的结果必须放进同一条 user 消息。</li>
        </ul>
        <p>累计 input 显著大于当前上下文, 根源在于模型自身无状态: 每次调用都必须重发截至该时刻的完整消息列表, 累计值即历次发送长度之和。对话历史越长, 单轮边际成本越高; 当「累计」超过「当前」的 2 倍时读数标红提示。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const parallel = ref(false)
const toolError = ref(false)
const maxTurns = ref(6)
const hoverId = ref('')

const SYSTEM = 'You are a small teaching agent.'
const DOC = '[0.62] kv_cache: the kv cache is the model memory of past keys and values, decode becomes incremental.'
// 与 core/utils.py:estimate_tokens 同一个粗估: 中文 1 字/token, 其余 4 字符/token
const est = (s) => { const cjk = (s.match(/[一-鿿]/g) || []).length; return cjk + Math.floor((s.length - cjk + 3) / 4) }
const flat = (b) => (b.type === 'text' ? b.text : b.type === 'tool_use' ? `[tool_use ${b.name} ${JSON.stringify(b.input)}]` : String(b.content))
const show = (b) => (b.type === 'text' ? b.text
  : b.type === 'tool_use' ? `tool_use  id=${b.id}  ${b.name}(${JSON.stringify(b.input)})`
    : `tool_result  tool_use_id=${b.tool_use_id}  is_error=${b.is_error}  ${b.content}`)

// 工具执行: 参数先过 JSON Schema 校验, 失败也回填成 is_error 结果 (tools.py: ToolRegistry.execute)
const execute = (name, input) => {
  if (name === 'search_docs') return { content: DOC, is_error: false }
  if (!('expr' in input)) return { content: "INVALID_ARGS: missing required 'expr'; unexpected 'expression'", is_error: true }
  return { content: '16*64 = 1024', is_error: false }
}

const sim = computed(() => {
  // 脚本化的"模型": 每个元素是一次模型调用想发出的 tool_use 列表
  const search = { name: 'search_docs', input: { query: 'kv cache' } }
  const bad = { name: 'calculator', input: { expression: '16*64' } }
  const good = { name: 'calculator', input: { expr: '16*64' } }
  const calc = toolError.value ? [[bad], [good]] : [[good]]
  const plan = parallel.value ? [[search, ...calc[0]], ...calc.slice(1)] : [[search], ...calc]

  const msgs = [{ role: 'user', blocks: [{ type: 'text', text: '查一下 kv cache 文档, 再算 16*64' }] }]
  const ctxAt = [] // 每次模型调用时的上下文 token 数
  const ctx = () => est(SYSTEM) + msgs.reduce((s, m) => s + m.blocks.reduce((t, b) => t + est(flat(b)), 0), 0)
  let id = 0, forced = true
  // ★ 这就是 Agent.run: for turn in range(max_turns): 问模型 → final 则返回, 否则执行工具并回填
  for (let turn = 0; turn < maxTurns.value; turn++) {
    ctxAt.push(ctx())
    const uses = plan[turn]
    if (!uses) {
      msgs.push({ role: 'assistant', call: ctxAt.length, blocks: [{ type: 'text', text: 'kv cache 保存历史 K/V 让 decode 增量进行; 16×64 = 1024。' }] })
      forced = false
      break
    }
    const blocks = uses.map((u) => ({ type: 'tool_use', id: `toolu_${String(++id).padStart(4, '0')}`, ...u }))
    msgs.push({ role: 'assistant', call: ctxAt.length, blocks })
    msgs.push({ role: 'user', blocks: blocks.map((b) => ({ type: 'tool_result', tool_use_id: b.id, ...execute(b.name, b.input) })) })
  }
  if (forced) msgs.push({ role: 'assistant', blocks: [{ type: 'text', text: 'stopped: max_turns reached' }] })
  return { msgs, ctxAt, forced, final: ctx() }
})

// 帧 = 逐个 block 揭开
const frames = computed(() => sim.value.msgs.flatMap((m, i) => m.blocks.map((_, j) => [i, j])))
const stepper = useStepper(() => frames.value.length, { interval: 900 })
watch(sim, () => { stepper.pause(); stepper.step.value = frames.value.length - 1 }, { immediate: true })

const visible = computed(() => {
  const [mi, bi] = frames.value[stepper.step.value] || [0, 0]
  return sim.value.msgs.slice(0, mi + 1).map((m, i) => (i < mi ? m : { ...m, blocks: m.blocks.slice(0, bi + 1) }))
})
const done = computed(() => stepper.step.value >= frames.value.length - 1)
const now = computed(() => {
  const calls = visible.value.filter((m) => m.call).length
  const ctxs = sim.value.ctxAt.slice(0, calls)
  const ctx = est(SYSTEM) + visible.value.reduce((s, m) => s + m.blocks.reduce((t, b) => t + est(flat(b)), 0), 0)
  return { calls, ctx, input: ctxs.reduce((a, b) => a + b, 0) }
})
</script>

<style scoped>
.msgs { display: flex; flex-direction: column; gap: 8px; min-height: 320px; }
.msg { border: 1px solid var(--border); border-left-width: 3px; border-radius: var(--radius-sm); padding: 6px 8px; display: flex; flex-direction: column; gap: 4px; }
.msg.user { border-left-color: var(--eye); }
.msg.assistant { border-left-color: var(--accent); }
.role { font-size: 11px; color: var(--text-dim); }
.role em { font-style: normal; color: var(--accent); }
.block { text-align: left; font-size: 11px; line-height: 1.5; min-height: 0; padding: 4px 6px; white-space: pre-wrap; word-break: break-all; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 4px; color: var(--text); cursor: default; }
.block.tool_use { border-color: var(--accent); }
.block.tool_result { border-color: var(--left); }
.block.err { border-color: var(--danger); color: var(--danger); }
.block.linked { outline: 2px solid var(--warn); outline-offset: 1px; }
</style>
