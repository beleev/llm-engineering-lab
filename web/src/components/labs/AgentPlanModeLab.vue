<!--
  Plan 模式 (对应 llm_agent/core/tools.py:TodoWriteTool / ExitPlanModeTool 与 permissions.py:_evaluate_one 的 plan 分支)。
  只讲一件事: "批准之前只许看不许动"是权限门强制的, 不是 prompt 里求模型自觉。
-->
<template>
  <LabFrame
    title="Plan 模式 — 计划批准之前「只许看不许动」, 是权限门强制的, 不是模型自觉的"
    sub="- 模拟什么: 在多步复杂任务中赋予人类事前否决权的 plan 模式机制, 标准工作流分为四步:
  1. 模型先调用 todo_write 工具将执行计划整理为显式的结构化待办列表。
  2. 模型调用 exit_plan_mode 工具将待办计划正式提交给用户审阅。
  3. 在用户点击批准前, 权限门始终锁定在 plan 模式, 仅放行声明了 read_only 为 true 的只读工具, 一律拒绝任何外部写操作。该约束由权限门硬性实施: 即便模型跳过计划直接发起写调用, 权限门同样予以拦截。
  4. 用户批准计划后, exit_plan_mode 内部将权限门切换至 accept_edits 模式, 写操作权限随之解锁。
- 图上是什么: 左侧展示计划状态面板, 右侧列出全部可用工具:
  - 左侧面板: 显示模型通过 todo_write 记录的最新计划列表 (共两项, 状态分为待办或已完成), 下方显示本次 exit_plan_mode 提交所收到的用户审批结果。
  - 右侧 7 个工具: 每行依次列出工具名称、该工具静态声明的 read_only 属性与风险等级 risk, 最右侧显示其在当前模式下的裁决结论 (allow 或 deny), 左侧绿条代表放行、红条代表拒绝。
  - 点击任意工具行: 底部展开该工具的完整裁决路径, 依次经过 deny 规则检查、ask / allow 规则检查 (plan 模式下直接跳过)、模式兜底策略及最终结论; 标红节点即为触发拦截的具体步骤。
- 可以做什么: 第一排模拟用户的三种审批决策, 第二排提供三个实验开关:
  - 「plan 模式 · 计划待批」: 计划已提交但尚未收到人工答复。
  - 「用户拒绝计划」: 审批未通过, 权限门继续停留在 plan 模式。
  - 「用户批准」: 审批通过, 权限门切换至 accept_edits 模式。
  - 「配一条 allow write_note 规则」: 验证预先配置的白名单放行规则在 plan 模式下为何不会生效。
  - 「把 delegate 标成只读 (漏洞)」: 模拟将委托子 agent 的工具错误声明为只读属性, 观察写操作如何借由子进程绕过父级 plan 模式。
  - 「有人审批高风险工具」: 仅在计划获批后产生影响, 决定 shell 与 delegate 等高风险工具是经人工确认放行, 还是因无人值守而直接拒绝。
- 应该看到: 三种典型状态的对比表现与右侧读数:
  - 处于 plan 模式时: 仅 4 个只读工具获准放行, write_note、delegate 与 shell 全部被拒, 右侧「放行 / 拒绝」读数为 4 / 3。即使开启 allow write_note 规则结果依然不变, 因为 plan 模式在检查完 deny 规则后立即跳出规则链, 完全忽略 ask 与 allow 配置。
  - 用户批准计划后: write_note 解锁放行, 「直接写进的笔记 / 已获批」变为 1 / 1。由于获批计划中包含一项写笔记任务, 该次写入处于授权范围内, 读数标绿。
  - 开启「把 delegate 标成只读 (漏洞)」并切回 plan 模式: delegate 被父级权限门当作只读工具放行, 而被拉起的子 agent 使用自身独立的权限门执行了写笔记操作, 导致「经子 agent 写进的笔记 / 已获批」变为 1 / 0 并标红, 即「批准前零写入」承诺被一层跨智能体委托击穿。
  - 「gate.mode」: 实时显示当前权限门所处的工作模式。"
    module="llm_agent/m10"
    run="python -m llm_agent.m10_planning.demo"
    :challenge="{
      ask: '开启「配一条 allow write_note 规则」并保持在 plan 模式下: write_note 会被放行吗? 随后再开启「把 delegate 标成只读 (漏洞)」: 在计划尚未获批的情况下, 为什么右侧已记录到一条实际写入的笔记?',
      answer: '第一问: 不会被放行。\n权限门内部的 _evaluate_one 默认按 deny、ask、allow 顺序依次匹配规则, 但在 plan 模式下存在特殊短路逻辑:\n1. 检查完 deny 规则后立即执行 break 跳出规则链, 完全跳过后续的 ask 与 allow 规则。\n2. 直接进入「仅放行只读工具」的模式兜底判断; 由于 write_note 未声明为只读, 调用被直接拒绝。\n点击右侧的 write_note 查看下方裁决路径, 可见「plan 模式 → break」节点标红。该设计旨在严格兑现 plan 模式「人工批准前零副作用写入」的安全承诺, 防止被会话早期遗留的 allow 白名单意外穿透; 而最上游的 deny 规则依然正常生效, 仍可用于封禁特定只读工具。\n第二问: 漏洞出在工具的 read_only 静态元数据声明与子智能体权限隔离失配上。\n- read_only 属性由工具开发者在类定义中手工标注, 外围 harness 无法在运行前静态证明工具代码是否确实无副作用, 一旦误标即构成权限逃逸漏洞。\n- 当 delegate 被误标为只读后, 父级 plan 模式会将其直接放行; 而被派生的子 agent 实例绑定的是自身独立的 auto 模式权限门, 未继承父级的 plan 模式约束, 子环境内风险等级为 medium 的 write_note 随即获准执行。由此导致计划获批前便发生了外部写入, 右侧「经子 agent 写进的笔记 / 已获批」显示为 1 / 0。\n因此, 在源码 core/subagents.py 中, DelegateTool 显式声明了 read_only 为 False 并将其风险等级定为高风险, 以阻断跨层级委托越权。',
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
        <p>当实际写入的笔记数大于已获批计划中的写入配额时读数标红, 表明系统在获得用户授权前已产生越权写入。</p>
        <p>todo_write 被声明为只读工具, 是因为该操作仅更新 agent 内存中的待办计划状态而不触碰外部环境; 否则连「起草待审批计划」这一前置动作本身也会被 plan 模式拦截。</p>
        <p>权限模式的切换逻辑封装在 ExitPlanModeTool.execute 内部: 仅当人工审批返回同意结果时, 代码才会修改 gate.mode。模型不存在其他可自行变更权限模式的调用路径, 从而保证了人工审批环节不可绕过。</p>
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
