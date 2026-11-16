<!--
  Agent loop 步进器 (对应 llm_agent/core/agent.py 的 Agent.run)。
  只讲一件事: loop 就是"往 messages 里追加 content block", 而每次问模型都要重发整个列表。
-->
<template>
  <LabFrame
    title="Agent loop — 消息列表怎么一块一块长出来, 每问一次模型为什么都要重发全部历史"
    sub="- 模拟什么: 一个最小的 agent 在做一个两步任务, 用户说「查一下 kv cache 文档, 再算 16*64」。agent 的核心只是一个循环 (loop): 把到目前为止的整段对话 (一个叫 messages 的列表) 发给模型; 模型要么回一段文字收工, 要么回一个 tool_use, 意思是「请帮我调这个工具, 参数是这些」; 模型外面那层普通代码 (harness) 真的去执行工具, 把结果包成 tool_result 追加到列表末尾, 再把整个列表重新发给模型。这样一来一回算一轮 (turn), 直到模型给出最终回答, 或者轮数撞到上限 max_turns。模型本身没有记忆, 它每次看到的就是这个列表。
- 图上是什么: 从上到下就是 messages 列表本身。每个方框是一条消息, 左上角写着它的角色: user 是发给模型的输入, assistant 是模型的输出; 写着「第 N 次模型调用的输出」的方框, 就是第 N 次把列表发给模型后它回的内容。方框里每个小格是一个 content block: 普通文字; tool_use (模型发出的调用, 紫边, 带一个 id); tool_result (工具的执行结果, 绿边, 带 tool_use_id 指回它回答的是哪一次调用; 红字的是 is_error=true 的失败结果)。把鼠标停在任意 tool_use 或 tool_result 上, 和它配对的那一格会被黄框圈出来。
- 可以做什么: 「并行工具调用」打开后, 模型在第一次回复里就把查文档和算术两个 tool_use 一起发出, 两个结果放进同一条 user 消息, 整个任务少问一次模型。「工具出错」打开后, 模型第一次算术把参数名写错 (写成 expression, 工具要的是 expr), 工具返回一条 is_error 的结果, 模型下一轮自己改对。「max_turns」是循环的硬上限, 默认 6 够用; 调小到 1 或 2, 任务做不完, 看 loop 怎么被强行停下。下面的播放器把列表按 block 一格一格揭开: 点「下一步」多露出一格, 拖「时间轴」跳到任意一步, 右侧读数只统计已经露出来的部分。
- 应该看到: 列表只增不减, 每一轮都是「assistant 发 tool_use, 紧跟一条 user 回 tool_result」成对出现; tool_result 虽然是工具产生的, 却放在 user 消息里, 因为对模型来说它是外界送进来的输入。默认设置下模型被调用 3 次 (查文档, 算术, 收工), 开并行变 2 次, 开工具出错变 4 次。右侧「模型调用次数」是列表到现在被发给模型几次; 「当前上下文」是现在列表里全部 block 加 system prompt 的 token 粗估 (中文 1 字算 1 token, 其他 4 个字符算 1 token, system prompt 只有 8 token); 「累计 input tokens」是每次调用时上下文大小的总和, 因为每次都重发全部历史, 它比当前上下文大得多, 超过当前上下文 2 倍就标红; 「结束方式」是 final (模型自己收工) 还是 max_turns 强停。"
    module="llm_agent/m01"
    run="python -m llm_agent.m01_agent_loop.demo"
    :challenge="{
      ask: '关掉「并行」、打开「工具出错」, 把播放拖到最后。先猜: 模型一共被调用几次? 「累计 input tokens」大约是「当前上下文」的几倍?',
      answer: '4 次, 累计 input 258 对最终上下文 116, 2 倍多一点。\n一步步数: 第 1 次调用模型发出 search_docs; 第 2 次它发出参数名写错的 calculator (expression 而不是 expr); 工具没有抛异常, 而是返回一条 is_error=true 的 tool_result, 列表照常往下长; 第 3 次模型看到错误, 改成 expr 重发; 第 4 次拿到 1024, 写出最终回答。\n每一次调用都要把 system prompt 加全部历史重新发一遍, 所以累计 input 是 4 次调用时上下文的总和 (每次比上一次长), 而不是最后一次的大小。轮数越多, 每轮重发的历史越长, 这个总和近似按轮数的平方增长: 长会话贵, 以及为什么要有并行工具调用和上下文压缩, 根子就在这里。\n再看错误那一轮: 工具报错变成一条普通的 tool_result 回填给模型, 模型自己改参数, loop 没有被炸掉。如果工具一报错就抛异常终止, 模型就没有机会纠正。',
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
        <p>tool_result 放在 <b>user</b> 消息里, 靠 tool_use_id 指回 assistant 那条 tool_use; 每个 tool_use 都必须有一个配对的 tool_result, 真实 API 缺一个就直接拒绝整段历史。同一轮发出的多个调用, 它们的结果必须放进同一条 user 消息。</p>
        <p>累计 input 比当前上下文大, 是因为模型没有记忆: 每次调用都把到那时为止的整个列表重发一遍, 这个数就是每一次发出去的量加起来。历史越长, 每多一轮越贵; 「累计」超过「当前」的 2 倍时标红提醒。</p>
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
