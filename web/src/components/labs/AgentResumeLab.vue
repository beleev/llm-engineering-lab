<!--
  Resume 与隔离 (对应 llm_agent/core/persistence.py:JsonlSessionStore、agent.py 的 resume 分支、subagents.py:DelegateTool)。
  只讲一件事: 磁盘上有, 不等于回到上下文里 —— resume 只带回消息, 权限和子 transcript 都不回来。
  界面分两个区块: ① resume (m06) ② 子 agent 隔离 (m07)。两块各有自己的控件, 互不影响。
-->
<template>
  <LabFrame
    title="Resume — 进程重启后从磁盘恢复会话, 哪些东西回来了, 哪些没有"
    sub="- 模拟什么: 会话持久化日志与上下文恢复 (resume) 的状态边界。agent 运行期的核心对话状态即内存中的 messages 列表; 为支持进程退出后跨会话续跑, 系统每生成一条新消息便向磁盘 JSONL 文件 (每行存储一条独立 JSON 记录) 末尾顺序追加一行, 全程只追加不原地修改。resume 即由新进程按行读回该文件以重建 messages 列表。本实验演示两类典型的「磁盘已落盘记录、但主上下文并不加载」的工程隔离设计:
  - ① 主会话崩溃恢复: 会话 A 完成一次文档检索后退出, 新进程对其执行 resume, 观察损坏日志行的容错修复机制, 以及为何上一进程的权限审批状态不会随消息一并恢复。
  - ② 子智能体日志隔离: lead 通过 delegate 派生的子 agent 同样将完整交互落盘以供离线审计, 但这批长对话记录绝不会进入父级 lead 的上下文窗口。
- 图上是什么: 界面划分为 ① 与 ② 两个实验区块:
  - ① 左栏: 对应磁盘文件 session.jsonl 的逐行视图, 每行渲染为一个按钮。绿框表示结构完整可正常加载的日志行, 黄框表示存在异常的记录 (如未配对的悬空调用或截断的残缺 JSON)。点击任意一行可在下方查看原始 JSON 文本及恢复阶段的处置方式。
  - ① 右栏: 展示新进程重建后的内存列表 agent.messages。带数字编号的条目源自磁盘日志恢复, 标有「+」号的条目则由新进程在启动阶段动态追加 (包括自动补齐的错误占位结果、重复注入的 session_start 消息及新会话首条用户输入)。
  - ① 右栏底部状态行: 显示新会话中执行写操作 write_note 时经过当前权限门的裁决结果, 绿色代表放行, 红色代表拒绝。
  - ② 双横条对比: 上方黄色横条表示子 agent 完整 transcript 的 token 规模, 下方绿色横条表示父级 lead 的峰值上下文长度。
- 可以做什么: 在 ① 区操作三个故障注入开关与新会话权限门配置, 在 ② 区拖动阅读量滑杆:
  - 「崩在 tool_use 和 tool_result 之间」: 模拟进程在写出调用请求后、工具返回前意外崩溃, 在日志末尾遗留未闭合的悬空调用。
  - 「最后一行只写了一半」: 模拟进程被强制终止时末尾行未刷完缓冲区而产生的截断坏行。
  - 「resume 重放 session_start」: 模拟常见实现缺陷, 即每次恢复旧会话时未做去重便再次注入启动钩子生成的 system 消息。
  - 「新会话的权限门」: 切换新进程实例绑定的权限门策略, 包括 default 且无人可问、default 且有人审批, 以及 auto 模式。
  - ② 滑杆「子 agent 读几份文档」: 增加子智能体的文档阅读量, 观察子级落盘日志增长时父级 lead 上下文是否保持恒定。
- 应该看到: 两个区块的容错隔离表现与右侧读数:
  - ① 默认无故障状态: 磁盘全部 5 行日志完整加载, 「恢复出的消息」与「磁盘行数」相等。
  - ① 同时开启前两个崩溃故障开关: 磁盘文件共 7 行, 新进程成功恢复前 6 条合法记录 (自动跳过末尾截断的第 7 行), 并针对第 6 行悬空的 tool_use 自动补齐 1 条带错误标记的占位 tool_result, 原因是在真实模型 API 协议下任何未配对的 tool_use 都会触发请求校验报错; 右侧「跳过的坏行 / 补的占位结果」读数显示为 1 / 1。
  - ① 权限不跨进程继承: 尽管原会话曾批准放行过 write_note, 当新会话选用「default · 无人可问」时该操作仍会被拒绝; 这是因为权限配置并不持久化在 JSONL 消息流中, 操作授权属于当前宿主运行环境的实时决策, 而非可由历史数据文件自行赋予的状态。
  - ② 子智能体上下文隔离: 调大子 agent 阅读份数时黄色横条持续增长, 而代表 lead 的绿色横条始终固定在 107 token, 因为父级上下文仅接收硬截断至 200 字符的最终摘要。
  - ② 「子 agent 峰值 / lead 峰值」: 当 lead 峰值低于子级峰值时读数标绿, 直观体现上下文隔离效果。"
    module="llm_agent/m06 · m07"
    run="python -m llm_agent.m06_persistence_resume.demo"
    :challenge="{
      ask: '将三个故障开关全部开启, 此时磁盘日志共有 7 行。执行 resume 后能从磁盘恢复出几条历史消息? 随后将新会话的权限门切换至「default · 无人可问」: 在上一会话中曾获放行的 write_note, 为何在新会话中会被拒绝?',
      answer: '从磁盘实际恢复出 6 条历史消息, 另加新进程启动时自动补齐的 1 条占位结果 (若计入反例重放的钩子消息与新提问则一并追加在后)。\n- 第 1 至第 5 行: 属于原会话 A 正常落盘的完整记录, 原样加载至内存。\n- 第 7 行 (截断坏行): 因进程强退导致 JSON 语法残缺, json.loads 解析时抛出 JSONDecodeError, load() 捕获异常后直接丢弃该行, 避免单行尾部损坏导致整段历史会话无法恢复。\n- 第 6 行 (悬空调用): 进程崩溃发生在写出 tool_use 之后、回写 tool_result 之前, 使有效日志末尾停留在未闭合的工具请求上。Agent.__init__ 检测到末条消息为未配对的 tool_use 时, 会自动构造一条标记 is_error 的占位 tool_result 并同步追加写回磁盘 JSONL:\n  - 必须补齐占位块, 是因为直接向真实 Messages API 发送包含未闭合 tool_use 的消息序列会立即触发 HTTP 400 协议校验错误。\n  - 必须同步写回磁盘, 是因为若仅在内存中修补, 磁盘日志中的悬空记录一旦被后续新消息压入中间位置, 下一次仅检查末尾的恢复逻辑便无法再定位修复。\n- 「resume 重放 session_start」属于典型的幂等性缺陷反例: 当恢复出的历史记录中已存在启动钩子消息时不应重复注入, 否则每次重启恢复都会令系统提示膨胀一条。\nwrite_note 在新会话中被拒绝, 是因为 PermissionGate 实例状态并不序列化进 JSONL 文件, 磁盘每行仅记录 role、content 与 name 三个对话字段。操作授权本质上属于当前宿主运行环境的安全策略, 而非可由外部数据文件自行声明恢复的状态; 否则篡改本地会话文件即可实现越权提权。上一会话的放行仅代表对当时上下文情境的单次许可, 进入新进程后必须由当前权限门重新裁决。\n基于同一边界原则, 挂载的 hooks、已安装的 skills、工具在外部系统留下的副作用以及子 agent 的独立 transcript 均不会随主会话 JSONL 一并恢复。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="tip">① 故障:</span>
        <button type="button" :class="{ active: dangling }" :aria-pressed="dangling" @click="dangling = !dangling">{{ dangling ? '☑' : '☐' }} 崩在 tool_use 和 tool_result 之间</button>
        <button type="button" :class="{ active: halfLine }" :aria-pressed="halfLine" @click="halfLine = !halfLine">{{ halfLine ? '☑' : '☐' }} 最后一行只写了一半</button>
        <button type="button" :class="{ active: replay }" :aria-pressed="replay" @click="replay = !replay">{{ replay ? '☑' : '☐' }} resume 重放 session_start</button>
      </div>
      <div class="row">
        <span class="tip">① 新会话的权限门:</span>
        <button v-for="g in GATES" :key="g.id" type="button" class="mono" :class="{ active: gate === g.id }" @click="gate = g.id">{{ g.label }}</button>
      </div>
    </template>

    <h4 class="part">① Resume (m06): 磁盘上的行, 哪些回到了新会话</h4>
    <div class="cols">
      <section>
        <h4>磁盘 · session.jsonl <em>只追加, 从不就地改写</em></h4>
        <button
          v-for="r in disk" :key="r.id" type="button" class="line mono" :class="[r.fate, { sel: sel === r.id }]"
          @click="sel = sel === r.id ? -1 : r.id"
        ><span class="no">{{ r.id }}</span>{{ r.text }}</button>
      </section>
      <section>
        <h4>新进程 · resume 出来的 agent.messages</h4>
        <div v-for="(m, i) in resumed" :key="i" class="line mono" :class="{ dup: m.dup, fresh: m.fresh }">
          <span class="no">{{ m.dup || m.fresh ? '+' : i + 1 }}</span>{{ m.text }}
        </div>
        <p class="tip">有编号的是从磁盘恢复的 {{ restored }} 条; 「+」是新进程自己加的。</p>
        <div class="line mono verdict" :class="verdict.ok ? 'ok' : 'deny'">
          <span class="no">门</span>write_note → {{ verdict.text }}
        </div>
      </section>
    </div>

    <p class="detail">{{ detail.tip }}</p>
    <pre v-if="detail.raw" class="raw mono">{{ detail.raw }}</pre>

    <div class="iso">
      <h4 class="part">② 子 agent 隔离 (m07): 和上面的开关无关</h4>
      <LabSlider v-model="docs" label="子 agent 读几份文档" :min="1" :max="5" />
      <h4>磁盘 · child_00_researcher.jsonl <em>落盘可审计, 但不属于 lead 的上下文</em></h4>
      <div class="barrow">
        <span class="bar child" :style="{ width: childTok / SCALE * 100 + '%' }" />
        <span class="mono lbl">{{ childTok }} tok · {{ 3 + docs }} 条消息</span>
      </div>
      <h4>lead 的峰值上下文 <em>子级只交回 1 条摘要, 在 harness 侧硬截断到 200 字符</em></h4>
      <div class="barrow">
        <span class="bar lead" :style="{ width: LEAD / SCALE * 100 + '%' }" />
        <span class="mono lbl">{{ LEAD }} tok · 含 1 条摘要</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>① 磁盘行数</span><b>{{ disk.length }}</b></div>
      <div class="kv"><span>① 从磁盘恢复出的消息 (少于行数就标红)</span><b :class="restored < disk.length ? 'bad' : ''">{{ restored }}</b></div>
      <div class="kv"><span>① 跳过的坏行 / 补的占位结果</span><b>{{ +halfLine }} / {{ +dangling }}</b></div>
      <div class="kv"><span>② 子 agent 峰值 / lead 峰值 (lead 小就标绿)</span><b :class="LEAD < childTok ? 'good' : 'bad'">{{ childTok }} / {{ LEAD }} tok</b></div>
      <div class="lab-note">
        <p><b>resume 回来了的:</b></p>
        <ul class="pts">
          <li>消息本身 (含 tool_use / tool_result 的配对)。</li>
          <li>tool_use id 的续号起点 (新调用从 toolu_0002 往下编, 不和旧的撞号)。</li>
        </ul>
        <p><b>没回来的:</b></p>
        <ul class="pts">
          <li>权限模式和上次的审批结果: JSONL 里没有这个字段, 新会话自己建门。</li>
          <li>hooks 和 skills: 由新进程自己装。</li>
          <li>工具在外面留下的改动 (写过的文件、发过的请求): 存储管不着, resume 不会回滚也不会重放。</li>
          <li>子 agent 的 transcript: 它在磁盘上可以审计, 但从来不属于 lead 的上下文。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const GATES = [
  { id: 'default', label: 'default · 无人可问' },
  { id: 'human', label: 'default · 有人审批' },
  { id: 'auto', label: 'auto' },
]
const replay = ref(false)
const halfLine = ref(false)
const dangling = ref(false)
const gate = ref('default')
const docs = ref(1)
const sel = ref(-1)

// 会话 A 写下的 5 行, 与 m06 demo 的 [JSONL 原文] 逐行对应 (text 为缩写, raw 为原行)
const BASE = [
  { role: 'system', text: 'system  name=session_start  "Policy: answer in Chinese."', raw: '{"role":"system","content":"Policy: answer in Chinese.","name":"session_start"}' },
  { role: 'user', text: 'user  "搜索 agent loop"', raw: '{"role":"user","content":"搜索 agent loop","name":null}' },
  { role: 'assistant', use: true, text: 'assistant  [tool_use toolu_0001 search_docs]', raw: '{"role":"assistant","content":[{"type":"tool_use","id":"toolu_0001","name":"search_docs","input":{"query":"搜索 agent loop"}}],"name":null}' },
  { role: 'user', text: 'user  [tool_result toolu_0001 "agent_loop: …"]', raw: '{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_0001","content":"agent_loop: Agent loop uses messages, tools, permissions and persistence."}],"name":null}' },
  { role: 'assistant', text: 'assistant  "基于工具结果完成：agent_loop…"', raw: '{"role":"assistant","content":"基于工具结果完成：\\nsearch_docs: agent_loop: Agent loop uses messages, tools, permissions and persistence.","name":null}' },
]

const disk = computed(() => {
  const rows = BASE.map((r, i) => ({ ...r, id: i + 1, fate: 'restored' }))
  // 崩在"写下 tool_use"之后、"写下 tool_result"之前: 日志里留下一个没有结果的调用
  if (dangling.value) rows.push({ id: rows.length + 1, role: 'assistant', use: true, fate: 'patched', text: 'assistant  [tool_use toolu_0002 write_note]  ← 崩在这里', raw: '{"role":"assistant","content":[{"type":"tool_use","id":"toolu_0002","name":"write_note","input":{"text":"…"}}],"name":null}' })
  // 进程被 kill 时正写到一半: 这一行不是合法 JSON
  if (halfLine.value) rows.push({ id: rows.length + 1, role: '?', broken: true, fate: 'skipped', text: '(半行) {"role":"user","content":[{"type":"tool_re', raw: '{"role":"user","content":[{"type":"tool_re' })
  return rows
})

const resumed = computed(() => {
  // load(): json.loads 抛 JSONDecodeError 的行跳过, 其余照单全收
  const kept = disk.value.filter((r) => !r.broken)
  const out = kept.map((r) => ({ text: r.text }))
  // ★ Agent.__init__: 最后一条若是悬空的 tool_use, 补一条 is_error 的占位结果并写回 JSONL, transcript 才合法
  if (dangling.value) out.push({ fresh: true, text: 'user  [tool_result toolu_0002 is_error "interrupted: …"]  ← 补上并写回 JSONL' })
  // session_start: 历史里已经有了就不再注入, 否则每次 resume 都会重复注入同一段
  if (replay.value) out.push({ dup: true, text: 'system  name=session_start  ← 又注入了一遍' })
  out.push({ fresh: true, text: 'user  "把刚才的结果写进笔记"  ← 新会话的第一句' })
  return out
})
// 真正从磁盘恢复出来的条数 (不算重放的 session_start、补上的占位结果和新会话的第一句)
const restored = computed(() => resumed.value.length - 1 - (replay.value ? 1 : 0) - (dangling.value ? 1 : 0))

const verdict = computed(() => {
  if (gate.value === 'auto') return { ok: true, text: 'allow (auto: bounded local write): medium 风险直接放行' }
  if (gate.value === 'human') return { ok: true, text: 'allow (human: 重新问了一次, 人批准了)' }
  return { ok: false, text: 'deny (human: 无人可问, fail closed)。上次的"同意"不跟着 resume 回来' }
})

const FATE = {
  restored: '恢复进新会话的上下文。',
  patched: '恢复, 并补上结果。这是崩在 tool_use 与 tool_result 之间留下的悬空调用: 每个 tool_use 都必须有配对的 tool_result。新进程给它补一条 is_error 的占位结果, 同时写回 JSONL, 第二次 resume 读到的也是配好对的。',
  skipped: '被跳过。json.loads 在这一行抛 JSONDecodeError, load() 吞掉异常继续读下一行。坏一行不该让整个会话无法恢复。',
}
const detail = computed(() => {
  const r = disk.value.find((x) => x.id === sel.value)
  if (!r) return { tip: '点左边任意一行, 看它的原始 JSON 和 resume 时的下场。append-only 的意思是: 这些行写下去就永不改变。出事故时能逐行复盘"模型当时要求了什么、harness 放行了什么"。', raw: '' }
  return { tip: `第 ${r.id} 行 (${r.role}): ${FATE[r.fate]}`, raw: r.raw }
})

// 子 agent: 自己的工具集、自己的上下文; 父级只拿一条硬截断到 200 字符的摘要
// docs = 1 时与 m07 demo 一致: child 峰值 279 tok, parent 峰值 107 tok。
// 摘要有上限, 所以子级读得再多 lead 也不涨: lead 比子级小 = 隔离在起作用 (标绿)
const LEAD = 107
const SCALE = 59 + 5 * 220
const childTok = computed(() => 59 + docs.value * 220)
</script>

<style scoped>
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; min-width: 520px; }
.cols > section { min-width: 0; }
.iso { margin-top: 18px; padding-top: 12px; border-top: 1px dashed var(--border-strong); min-width: 300px; }
h4.part { font-size: 13px; color: var(--text); margin: 0 0 10px; }
h4 { font-size: 12px; color: var(--text-muted); font-weight: 500; margin: 0 0 6px; }
h4 em { font-style: normal; color: var(--text-dim); font-size: 11px; }
.iso h4:not(.part) { margin-top: 10px; }
.line { display: block; width: 100%; text-align: left; font-size: 10.5px; line-height: 1.5; padding: 4px 6px; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; color: var(--text-muted); min-height: 0; }
.line .no { display: inline-block; width: 20px; color: var(--text-dim); }
.line.restored { border-left: 3px solid var(--left); }
.line.patched { border-left: 3px solid var(--warn); color: var(--warn); }
.line.skipped { border-left: 3px solid var(--warn); color: var(--warn); }
.line.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
.line.dup { border-color: var(--warn); color: var(--warn); }
.line.fresh { border-color: var(--eye); color: var(--text); }
.line.verdict { white-space: normal; margin-top: 6px; }
.line.verdict.ok { border-color: var(--left); color: var(--text); }
.line.verdict.deny { border-color: var(--danger); color: var(--danger); }
.tip { font-size: 12px; color: var(--text-muted); }
.detail { margin-top: 10px; font-size: 12.5px; color: var(--text-muted); line-height: 1.7; }
.raw { margin-top: 6px; font-size: 10.5px; line-height: 1.6; color: var(--text-dim); background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; padding: 6px 8px; white-space: pre-wrap; word-break: break-all; }
.barrow { display: flex; align-items: center; gap: 8px; }
.barrow .lbl { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
.bar { height: 22px; min-width: 6px; border-radius: 3px; display: block; }
.bar.child { background: color-mix(in srgb, var(--warn) 30%, transparent); border: 1px solid var(--warn); }
.bar.lead { background: color-mix(in srgb, var(--left) 30%, transparent); border: 1px solid var(--left); }
@media (max-width: 1200px) { .cols { grid-template-columns: 1fr; min-width: 0; } }
</style>
