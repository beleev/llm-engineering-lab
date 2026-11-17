<!--
  Plan 模式 (对应 llm_agent/core/tools.py:TodoWriteTool / ExitPlanModeTool 与 permissions.py:_evaluate_one 的 plan 分支)。
  只讲一件事: "批准之前只许看不许动"是权限门强制的, 不是 prompt 里求模型自觉。
-->
<template>
  <LabFrame
    title="Plan 模式 — 计划批准之前「只许看不许动」, 是权限门强制的, 不是模型自觉的"
    sub="- 模拟什么: 多步任务里让人在动手之前有否决权的机制, 流程是:
  1. 模型先用 todo_write 工具把计划写成一份显式的待办列表。
  2. 再用 exit_plan_mode 工具把计划提交给用户。
  3. 用户批准之前, 权限门处于 plan 模式, 只放行声明了 read_only=true 的只读工具, 任何写操作一律拒绝。这个限制由门强制: 模型不交计划直接写, 门照样不开。
  4. 用户批准后, exit_plan_mode 把门切到 accept_edits 模式, 写操作才解锁。
- 图上是什么: 左边是计划, 右边是工具:
  - 左边: 模型用 todo_write 写下的最新一版计划 (两项, 状态是待办或已完成), 下面一行是 exit_plan_mode 这次提交得到的答复。
  - 右边 7 个工具: 每行写着工具名、它自己声明的 read_only 和风险等级 risk, 最右边是它在当前模式下会被裁决成 allow 还是 deny, 左边绿条是放行, 红条是拒绝。
  - 点任意一个工具: 下方画出它的裁决路径, 先查 deny 规则, 再查 ask / allow 规则 (plan 模式下直接跳过), 再按模式兜底, 最后是结论; 红色的那一格就是把它拒掉的那一步。
- 可以做什么: 第一排扮演用户, 第二排是三个开关:
  - 「plan 模式 · 计划待批」: 提交后还没答复。
  - 「用户拒绝计划」: 门留在 plan 模式。
  - 「用户批准」: 门切到 accept_edits。
  - 「配一条 allow write_note 规则」: 演示一条早先配好的放行规则在 plan 模式下为什么不管用。
  - 「把 delegate 标成只读 (漏洞)」: 把委托子 agent 的工具错标成只读, 看写入怎样绕过 plan 模式。
  - 「有人审批高风险工具」: 只在批准之后有意义, 决定 shell、delegate 这类高风险工具是问人批准, 还是无人可问直接拒。
- 应该看到: 三种情形和右侧读数:
  - plan 模式下: 只有 4 个只读工具放行, write_note、delegate、shell 全拒, 右侧「放行 / 拒绝」是 4 / 3。打开 allow write_note 规则也不变, 因为 plan 模式查完 deny 规则就跳出, ask / allow 规则都不看。
  - 批准之后: write_note 放行, 「直接写进的笔记 / 已获批」变成 1 / 1。计划里只有一条写笔记, 写一条在授权之内, 绿色。
  - 把 delegate 标成只读再回到 plan 模式: delegate 被放行, 子 agent 用它自己那扇门写了笔记, 「经子 agent 写进的笔记 / 已获批」变成 1 / 0 标红: 「批准前零写入」被一层委托绕开了。
  - 「gate.mode」: 门现在处于哪个模式。"
    module="llm_agent/m10"
    run="python -m llm_agent.m10_planning.demo"
    :challenge="{
      ask: '打开「配一条 allow write_note 规则」, 停在 plan 模式: write_note 会被放行吗? 再打开「把 delegate 标成只读」: 计划还没批准, 右侧为什么已经有一条笔记被写进去了?',
      answer: '第一问: 不放行。\n权限门的 _evaluate_one 按 deny、ask、allow 的顺序查规则, 但在 plan 模式下:\n1. 处理完 deny 规则就 break, ask 和 allow 都不看。\n2. 然后按「只读才放行」兜底; write_note 不是只读, 拒绝。\n点右边的 write_note 看裁决路径, 「plan 模式 → break」那一格是红的。这样设计是因为 plan 模式的承诺是「批准之前零写入」, 不能被一条早先配好的 allow 绕开; deny 规则倒是照常生效, 连只读工具也能被它拒掉。\n第二问: 洞在 read_only 这个标记上。\n- read_only 是写工具的人自己在类属性里声明的, harness 没法验证, 标错就是漏洞。\n- delegate 一旦被标成只读, plan 模式就放行它; 子 agent 跑的是自己那扇 auto 模式的门, 和父级的 plan 模式无关, 里面的 write_note 是 medium 风险, 直接放行。笔记就这样在计划批准之前写进去了, 右侧「经子 agent 写进的笔记 / 已获批」是 1 / 0。\n所以 core/subagents.py 里 DelegateTool 显式写了 read_only = False 并按高风险处理, 注释里说明了原因。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="tip">用户的答复:</span>
        <button v-for="s in STAGES" :key="s.id" type="button" :class="{ active: stage === s.id }" @click="stage = s.id">{{ s.label }}</button>
      </div>
      <div class="row">
        <span class="tip">开关:</span>
        <button type="button" :class="{ active: allowRule }" :aria-pressed="allowRule" @click="allowRule = !allowRule">{{ allowRule ? '☑' : '☐' }} 配一条 allow write_note 规则</button>
        <button type="button" :class="{ active: fakeRO }" :aria-pressed="fakeRO" @click="fakeRO = !fakeRO">{{ fakeRO ? '☑' : '☐' }} 把 delegate 标成只读 (漏洞)</button>
        <button type="button" :class="{ active: human }" :aria-pressed="human" :disabled="mode === 'plan'" @click="human = !human">{{ human ? '☑' : '☐' }} 有人审批高风险工具</button>
        <span v-if="mode === 'plan'" class="tip">plan 模式不问人, 写操作直接拒。「有人审批」批准计划后才起作用。</span>
      </div>
    </template>

    <div class="panes">
      <section class="todo">
        <h4>todo_write 写下的最新一版计划 <em>每次整表覆写, 不做增量修改</em></h4>
        <div v-for="(t, i) in todos" :key="i" class="todo-row">
          <span class="pill" :class="t.status">{{ STATUS[t.status] }}</span><span>{{ t.content }}</span>
        </div>
        <p class="plan mono">exit_plan_mode(plan="1. search_docs 2. write_note") → {{ stageNote }}</p>
      </section>

      <section class="tools">
        <h4>工具 (点一个看它的裁决路径) <em>当前 gate.mode = {{ mode }}</em></h4>
        <button
          v-for="t in TOOLS" :key="t.name" type="button" class="trow" :class="[judge(t).ok ? 'ok' : 'deny', { sel: sel === t.name }]"
          @click="sel = t.name"
        >
          <span class="mono tn">{{ t.name }}</span>
          <span class="meta mono">read_only={{ ro(t) }} · risk={{ t.risk }}</span>
          <span class="mono vd">{{ judge(t).ok ? 'allow' : 'deny' }}</span>
        </button>
      </section>
    </div>

    <div class="trace">
      <b class="mono">{{ sel }}</b> 的裁决路径:
      <span v-for="(s, i) in trace" :key="i" class="tr" :class="s.state">{{ s.label }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>gate.mode (门现在的模式)</span><b>{{ mode }}</b></div>
      <div class="kv"><span>放行 / 拒绝 (7 个工具里)</span><b>{{ counts.ok }} / {{ counts.no }}</b></div>
      <div class="kv"><span>直接写进的笔记 / 已获批的条数</span><b :class="notes > approved ? 'bad' : 'good'">{{ notes }} / {{ approved }}</b></div>
      <div class="kv"><span>经子 agent 写进的笔记 / 已获批的条数</span><b :class="viaChild > 0 ? 'bad' : 'good'">{{ viaChild }} / 0</b></div>
      <div class="lab-note">
        <p>写进的笔记比已获批的多就标红: 那是在用户点头之前就发生的写入。</p>
        <p>todo_write 标成只读, 是因为它只改 agent 自己的计划状态, 不碰外部世界。否则连「先列个计划」这一步都会被 plan 模式拒掉。</p>
        <p>模式的翻转发生在 ExitPlanModeTool.execute 里: 人点了同意, 它才去改 gate.mode。模型没有别的路径能自己改模式, 所以批准绕不过去。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

const STAGES = [
  { id: 'plan', label: 'plan 模式 · 计划待批' },
  { id: 'rejected', label: '用户拒绝计划' },
  { id: 'approved', label: '用户批准 → accept_edits' },
]
const STATUS = { pending: '待办', in_progress: '进行中', completed: '已完成' }
// 元数据取自 core/tools.py 和 core/subagents.py 的类属性
const TOOLS = [
  { name: 'todo_write', ro: true, risk: 'low' },
  { name: 'exit_plan_mode', ro: true, risk: 'low' },
  { name: 'search_docs', ro: true, risk: 'low' },
  { name: 'read_notes', ro: true, risk: 'low' },
  { name: 'write_note', ro: false, risk: 'medium' },
  { name: 'delegate', ro: false, risk: 'high' }, // 在 core/subagents.py: 委托出去的子级能做什么父级管不到, 按高风险处理
  { name: 'shell', ro: false, risk: 'high' },
]

const stage = ref('plan')
const allowRule = ref(false)
const fakeRO = ref(false)
const human = ref(false)
const sel = ref('write_note')

const mode = computed(() => (stage.value === 'approved' ? 'accept_edits' : 'plan'))
const ro = (t) => (t.name === 'delegate' && fakeRO.value ? true : t.ro)

// ★ 照搬 permissions.py:_evaluate_one 的分支顺序
const judge = (t) => {
  if (mode.value === 'plan') {
    // deny 规则先看 (本实验没配 deny 规则), 然后直接 break —— ask / allow 规则一概不查
    return ro(t)
      ? { ok: true, why: 'plan: read-only tool' }
      : { ok: false, why: 'plan: plan mode is read-only until the plan is approved' }
  }
  if (allowRule.value && t.name === 'write_note') return { ok: true, why: 'rule: allow write_note' }
  if (t.risk === 'high') {
    return human.value
      ? { ok: true, why: 'human: high-risk tool still needs approval; approved' }
      : { ok: false, why: 'human: 无人可问, fail closed' }
  }
  return { ok: true, why: 'accept_edits: low/medium risk' }
}

const trace = computed(() => {
  const t = TOOLS.find((x) => x.name === sel.value) || TOOLS[0]
  const plan = mode.value === 'plan'
  const hitAllow = allowRule.value && t.name === 'write_note'
  return [
    { label: 'deny 规则: 未命中', state: 'pass' },
    plan
      ? { label: 'plan 模式 → break, 不查 ask / allow 规则', state: hitAllow ? 'kill' : 'pass' }
      : { label: hitAllow ? 'allow 规则: 命中 write_note' : 'ask / allow 规则: 未命中', state: hitAllow ? 'ok' : 'pass' },
    { label: plan ? `模式兜底: read_only=${ro(t)}` : `模式兜底: accept_edits, risk=${t.risk}`, state: 'pass' },
    { label: judge(t).why, state: judge(t).ok ? 'ok' : 'kill' },
  ]
})

const counts = computed(() => {
  const ok = TOOLS.filter((t) => judge(t).ok).length
  return { ok, no: TOOLS.length - ok }
})
const todos = computed(() => {
  const s = stage.value === 'approved' ? 'completed' : 'pending'
  return [
    { content: 'search_docs: 需要外部知识', status: s },
    { content: 'write_note: 把结论落成笔记', status: s },
  ]
})
const stageNote = computed(() => ({
  plan: '等待用户答复; 在此之前写操作一律被拒',
  rejected: 'plan rejected by user; stay in plan mode (ok=False)',
  approved: 'plan approved; mode -> accept_edits',
}[stage.value]))

// 计划里只有一条 write_note: 批准后写下的这 1 条在授权之内, 超出已获批条数才算越权
const approved = computed(() => (stage.value === 'approved' ? 1 : 0))
const notes = computed(() => (judge(TOOLS.find((t) => t.name === 'write_note')).ok ? 1 : 0))
// 委托被错标成只读 → plan 模式放行 delegate → 子 agent 用自己的 auto 门写下了笔记
const viaChild = computed(() => (mode.value === 'plan' && fakeRO.value ? 1 : 0))
</script>

<style scoped>
.panes { display: grid; grid-template-columns: minmax(230px, 1fr) minmax(280px, 1.2fr); gap: 14px; min-width: 520px; }
.panes h4 { font-size: 12px; color: var(--text-muted); font-weight: 500; margin-bottom: 8px; }
.panes h4 em { font-style: normal; color: var(--text-dim); font-size: 11px; }
.todo-row { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); padding: 5px 0; border-bottom: 1px dashed var(--border); }
.pill { font-size: 10px; padding: 1px 6px; border-radius: 10px; border: 1px solid var(--border-strong); color: var(--text-dim); white-space: nowrap; }
.pill.completed { border-color: var(--left); color: var(--left); }
.plan { margin-top: 10px; font-size: 10.5px; line-height: 1.6; color: var(--text-dim); background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; padding: 6px; }
.trow { display: grid; grid-template-columns: 110px 1fr 48px; gap: 8px; align-items: center; width: 100%; text-align: left; font-size: 11px; padding: 6px; margin-bottom: 4px; min-height: 0; background: var(--bg-elev); border: 1px solid var(--border); border-left-width: 3px; border-radius: 3px; color: var(--text-muted); }
.trow.ok { border-left-color: var(--left); }
.trow.deny { border-left-color: var(--danger); }
.trow.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
.trow .tn { color: var(--text); }
.trow .meta { font-size: 10px; color: var(--text-dim); }
.trow .vd { text-align: right; }
.trow.ok .vd { color: var(--left); }
.trow.deny .vd { color: var(--danger); }
.trace { margin-top: 12px; font-size: 12px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; line-height: 1.7; }
.tr { font-size: 11px; padding: 2px 7px; border-radius: 3px; border: 1px solid var(--border); }
.tr.pass { color: var(--text-dim); }
.tr.ok { border-color: var(--left); color: var(--left); }
.tr.kill { border-color: var(--danger); color: var(--danger); }
.tip { font-size: 12px; color: var(--text-muted); }
@media (max-width: 1200px) { .panes { grid-template-columns: 1fr; min-width: 0; } }
</style>
