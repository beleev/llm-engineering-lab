<!--
  权限门追踪器 (对应 llm_agent/core/permissions.py 的 PermissionGate)。
  只讲一件事: 裁决顺序 deny → ask → allow → 模式兜底; 以及字符串黑名单为什么天生很弱。
-->
<template>
  <LabFrame
    title="权限门 — 模型发出的一次工具调用, 是怎么被裁决成放行、问人或拒绝的"
    sub="- 模拟什么: 权限门 (PermissionGate) 的多层裁决机制与黑名单边界。模型发起的每次工具调用在进入执行层前均需经过权限门审查, 裁决结果分为 ALLOW (放行)、DENY (拒绝) 与 ASK (转交人工审批) 三种。单条权限规则由「工具名 + 参数通配模式」(glob 语法, 星号匹配任意字符序列) 构成。实验同时揭示字符串黑名单的固有局限: deny 规则本质上是在枚举危险命令的字面写法, 而等价的攻击变体无穷无尽。权限门内部遵循严格固定的裁决优先级:
  1. 优先检查 deny 规则。
  2. 其次检查 ask 规则。
  3. 再次检查 allow 规则。
  4. 若上述规则均未命中, 则交由当前激活的「模式」执行兜底裁决 (例如 default 模式兜底为询问人工, plan 模式仅放行只读工具)。
- 图上是什么: 中央区域按从上到下顺序绘制本次调用穿过权限门的四站决策轨迹, 依次为 deny 规则、ask 规则、allow 规则与模式兜底:
  - 紫色高亮站点: 最终作出本次裁决结论的生效层级。
  - 灰字站点: 已执行匹配检查但未命中的层级。
  - 半透明站点: 因上游已得出最终结论或当前模式规定不予参考 (如 plan 模式直接跳过 ask 与 allow 规则) 而未执行的层级。
  - 复合命令拆解: 当 shell 命令包含 &amp;&amp;、||、;、| 或 &amp; 等控制连接符时, 解析器会先将其拆分为多个独立子命令段并在上方逐段展示裁决结果, 下方主轨迹仅绘制决定整体结论的关键子段 (即首个未被放行的子段)。
- 可以做什么: 按界面从上到下操作五组控件:
  - 「模式」: 切换无显式规则命中时的六种兜底策略。
  - plan: 仅放行声明为只读的工具。
  - default: 转交人工审批。
  - accept_edits: 自动放行低中风险操作, 高风险操作转交人工审批。
  - auto: 调用启发式特征分类器识别高危词汇。
  - dont_ask: 仅放行显式命中 allow 规则的调用, 其余一律拒绝。
  - bypass_permissions: 兜底自动放行所有操作, 但最上游的 deny 规则依然强制生效。
  - 「规则」: 五条预设规则支持独立开关, 红框代表 deny, 黄框代表 ask, 绿框代表 allow。
  - 「匹配方式」: 在 permissions.py 实现的归一化匹配 (先转小写、压缩多余空白、将相邻短选项合并并按字母序重排, 例如将 RM  -r -f 统一规范化为 rm -fr, 对规则与输入命令同步执行) 与朴素 glob (原样字面匹配, 作为对照反例) 之间切换。
  - 「有人审批」开关: 控制触发 ASK 裁决时是由人工点击批准, 还是处于无人值守环境 (无人可问时按 fail closed 原则一律拒绝)。
  - 「预设调用」: 提供 12 条典型测试调用, 点击即可一键填入下方输入框。
  - 「自己输入」: 支持手动切换目标工具并编辑参数文本, 右侧同步显示该工具静态声明的风险等级 risk。
- 应该看到: 两种字符串匹配策略的拦截差异与右侧审计读数:
  - 默认测试调用为 shell 工具执行 rm -fr /: 在归一化匹配下成功命中 deny 规则, 第一站即被直接拒绝。
  - 切换至「朴素 glob」: 由于规则字面写作 rm -rf 而输入命令为 rm -fr, 字符顺序差异导致黑名单漏判, 请求继续下沉至 default 模式触发人工询问。
  - 右侧「最终裁决」: 本次工具调用的整体放行或拒绝结论。
  - 「谁做的决定」: 明确记录作出裁决的责任主体, 包括 rule (显式规则)、human (人工审批)、auto (分类器)、parser (语法解析失败拦截) 或具体兜底模式名称, 满足安全审计溯源要求。
  - 「匹配时看到的文本」: 展示规则引擎实际参与比对的字符串形式, 归一化模式下显示经小写转换与短选项重排后的规范文本。
  - 「deny 规则: 朴素 / 归一化」: 对同一输入同步运行两种匹配器, 结论不一致时标红提示, 直观暴露可绕过朴素字面匹配的变形写法。
  - 点击预设中的 find / -delete: 两种字符串匹配均发生漏报, 但该命令同样会清空磁盘文件, 印证黑名单无法穷举所有等价破坏指令。"
    module="llm_agent/m03"
    run="python -m llm_agent.m03_permissions.demo"
    :challenge="{
      ask: '在归一化匹配开启时, 「rm -r -f /」与「RM  -RF /」均会被成功拦截。你能否构造出一条同样会清空磁盘、却能直接穿过该条 deny 规则的命令? (可尝试预设列表中的 find / -delete)',
      answer: '可以构造, 且等价变体有无穷多种: 例如改用长选项 rm --recursive --force /、换用其他命令 find / -delete, 或通过 python -c 调用 shutil.rmtree 等。\ndeny 规则 *rm -rf* 本质上只是针对命令文本的字符串模式匹配。参数归一化仅能封堵成本最低的表层变形 (如大小写混写、多余空格、短选项拆分与重排); 一旦更换可执行程序名称、改用长选项或借用脚本解释器, 归一化后的字符串便不再命中原模式。由于黑名单试图枚举所有「危险写法」, 而危险语义的语法表达空间是无限的, deny 规则只能作为低成本的辅助防线用于拦截常见误操作。\n真正可靠的安全边界需要三层机制协同构建:\n- 默认拒绝的白名单机制 (allowlist): 未经显式授权的操作一律不予自动放行。default 模式在未命中规则时强制询问人工、无人值守时按 fail closed 拒绝, 正是这一原则的体现。\n- 结构化语法解析: 将命令解析为语法树或参数结构后再做语义审查, 而非直接比对原始字符串。\n- 操作系统级沙箱隔离: 在内核层直接限制子进程可访问的文件系统目录与网络套接字。\n在本实验中, find / -delete 默认并未被实际执行: 在 default 与 auto 模式下它会下沉至人工审批环节, 无人值守时被 fail closed 策略拦截。真正阻断它的是默认拒绝机制, 而非 deny 黑名单识别出了删除风险; 一旦将模式切换至 bypass_permissions, 该命令便会被直接放行。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="tip">模式:</span>
        <button v-for="m in MODES" :key="m" type="button" class="mono" :class="{ active: mode === m }" @click="mode = m">{{ m }}</button>
      </div>
      <div class="row">
        <span class="tip">规则:</span>
        <button v-for="(r, i) in rules" :key="i" type="button" class="rule mono" :class="[r.decision, { off: !r.on }]" :aria-pressed="r.on" @click="r.on = !r.on">
          {{ r.on ? '☑' : '☐' }} {{ r.decision }} · {{ r.tool }}{{ r.pattern ? ` "${r.pattern}"` : '' }}
        </button>
      </div>
      <div class="row">
        <span class="tip">匹配方式:</span>
        <button type="button" :class="{ active: normalized }" @click="normalized = true">归一化匹配 (permissions.py)</button>
        <button type="button" :class="{ active: !normalized }" @click="normalized = false">朴素 glob (反例)</button>
        <button type="button" :class="{ active: human }" :aria-pressed="human" @click="human = !human">有人审批: {{ human ? '会批准' : '无人可问' }}</button>
      </div>
      <div class="row">
        <span class="tip">预设调用:</span>
        <button v-for="p in PRESETS" :key="p.arg" type="button" class="mono" :class="{ active: tool === p.tool && arg === p.arg }" @click="tool = p.tool; arg = p.arg">{{ p.arg }}</button>
      </div>
      <div class="row call">
        <span class="tip">自己输入:</span>
        <select v-model="tool" class="mono" aria-label="工具"><option v-for="(t, n) in TOOLS" :key="n" :value="n">{{ n }}</option></select>
        <input v-model="arg" type="text" class="mono arg" spellcheck="false" aria-label="参数" />
        <span class="risk mono">risk={{ TOOLS[tool].risk }}</span>
      </div>
    </template>

    <div v-for="(part, pi) in result.parts" :key="pi" class="part">
      <p v-if="result.parts.length > 1" class="mono seg" :class="{ decisive: pi === result.idx }">第 {{ pi + 1 }} 段: {{ part.text }} → {{ part.verdict }}</p>
      <ol v-if="pi === result.idx" class="trace">
        <li v-for="s in part.stages" :key="s.name" :class="s.state">
          <span class="stage mono">{{ s.name }}</span>
          <span class="what">{{ s.detail }}</span>
        </li>
      </ol>
    </div>

    <template #stats>
      <div class="kv"><span>最终裁决</span><b :class="final.cls">{{ final.text }}</b></div>
      <div class="kv"><span>谁做的决定 (rule 规则 / human 问人 / auto 分类器 / parser 解析器 / 模式名)</span><b>{{ decisive.source }}</b></div>
      <div class="kv"><span>匹配时看到的文本 (规则实际拿去比的字符串)</span><b class="small">{{ decisive.seen }}</b></div>
      <div class="kv"><span>deny 规则能拦住吗: 朴素 / 归一化</span><b :class="denyHit.naive === denyHit.norm ? '' : 'bad'">{{ denyHit.naive ? '命中' : '漏过' }} / {{ denyHit.norm ? '命中' : '漏过' }}</b></div>
      <div class="lab-note">
        <p>本次裁决的理由: {{ decisive.reason }}。</p>
        <p>复合命令 (用 &amp;&amp; || ; | &amp; 连起来的) 会被拆开逐段评估, 任一段不过则整体不过。否则 <code class="inline">echo hi &amp;&amp; rm -rf /</code> 整串能匹配 allow "echo *", 蹭着 echo 放行。</p>
        <p>拆分时还有三条规矩:</p>
        <ul class="pts">
          <li><b>认引号:</b> <code class="inline">echo 'a; b'</code> 里的分号在引号内, 算一段。</li>
          <li><b>重定向:</b> 带输出重定向 (&gt; &gt;&gt;) 的段不享受 allow 规则, 因为 <code class="inline">echo x &gt; ~/.bashrc</code> 整串匹配 allow "echo *", 实际却在改文件。</li>
          <li><b>引号不配对:</b> 引号没配对的命令解析不了, 直接拒绝。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

const MODES = ['plan', 'default', 'accept_edits', 'auto', 'dont_ask', 'bypass_permissions']
const TOOLS = {
  shell: { risk: 'high', readOnly: false },
  write_note: { risk: 'medium', readOnly: false },
  read_file: { risk: 'low', readOnly: true },
  search_docs: { risk: 'low', readOnly: true },
  mcp__weather__get_weather: { risk: 'high', readOnly: false },
}
const PRESETS = [
  { tool: 'shell', arg: 'rm -rf /' }, { tool: 'shell', arg: 'rm -fr /' }, { tool: 'shell', arg: 'RM  -r -f /' },
  { tool: 'shell', arg: 'find / -delete' }, { tool: 'shell', arg: 'echo hi && rm -rf /' }, { tool: 'shell', arg: 'echo $(cat ~/.ssh/id_rsa)' }, { tool: 'shell', arg: 'git push origin main' },
  { tool: 'shell', arg: "echo 'a; b'" }, { tool: 'shell', arg: 'echo x > ~/.bashrc' },
  { tool: 'write_note', arg: 'todo: fix login' }, { tool: 'read_file', arg: '.env' }, { tool: 'mcp__weather__get_weather', arg: 'Beijing' },
]
const rules = reactive([
  { tool: 'shell', pattern: '*rm -rf*', decision: 'deny', on: true },
  { tool: 'shell', pattern: 'git push*', decision: 'ask', on: true },
  { tool: 'shell', pattern: 'echo *', decision: 'allow', on: true },
  { tool: 'write_note', pattern: '', decision: 'allow', on: false },
  { tool: 'mcp__weather__*', pattern: '', decision: 'allow', on: false },
])
const mode = ref('default'), normalized = ref(true), human = ref(false)
const tool = ref('shell'), arg = ref('rm -fr /')

// fnmatch: 只需要 * 和 ?
const glob = (text, pat) => new RegExp('^' + pat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '[\\s\\S]*').replace(/\?/g, '.') + '$').test(text)
// ★ 与 permissions.py:normalize_command 逐行对应: 小写、压空白、相邻短 flag 合并并排序 (-r -f → -fr)
const FLAG = /^(\*?)-([a-z]+)(\*?)$/
const sortSet = (s) => [...new Set(s)].sort().join('')
const normalize = (text) => {
  const out = []
  for (const tok of text.toLowerCase().split(/\s+/).filter(Boolean)) {
    const m = tok.match(FLAG), prev = out.length ? out[out.length - 1].match(FLAG) : null
    if (m && prev && !prev[3] && !m[1]) out[out.length - 1] = `${prev[1]}-${sortSet(prev[2] + m[2])}${m[3]}`
    else if (m) out.push(`${m[1]}-${sortSet(m[2])}${m[3]}`)
    else out.push(tok)
  }
  return out.join(' ')
}
const matches = (rule, name, text, norm) => glob(name, rule.tool)
  && (!rule.pattern || (norm ? glob(normalize(text), normalize(rule.pattern)) : glob(text, rule.pattern)))

const DANGER = ['rm ', 'sudo', 'curl ', 'wget ', 'ssh ', 'chmod ', '>', 'token', 'secret'] // 同 permissions.py:_SHELL_DANGER
const SENSITIVE = ['.env', 'secret', 'id_rsa', '.ssh', 'credentials']
// dont_ask 从不问人: 该问的 (命中 ask 规则) 直接拒绝, 与 permissions.py:_ask 相同
const ask = (reason) => (mode.value === 'dont_ask'
  ? { verdict: 'DENY', source: 'dont_ask', reason: `${reason}; dont_ask 不问人, 直接拒绝` }
  : { verdict: 'ASK', source: 'human', reason })
const fallback = (name, text) => {
  const t = TOOLS[name], m = mode.value
  if (m === 'plan') return t.readOnly ? { verdict: 'ALLOW', source: 'plan', reason: '只读工具, plan 模式放行' } : { verdict: 'DENY', source: 'plan', reason: 'plan 模式只读: 计划获批前任何写操作都拒绝' }
  if (m === 'dont_ask') return { verdict: 'DENY', source: m, reason: '没有 allow 规则预先批准它: dont_ask 只放行 allow 规则命中的调用' }
  if (m === 'bypass_permissions') return { verdict: 'ALLOW', source: m, reason: '该模式对未命中规则的调用一律放行 (deny 规则仍生效)' }
  if (m === 'accept_edits') return t.risk === 'high' ? ask('accept_edits 只自动放行低/中风险, 高风险仍问人') : { verdict: 'ALLOW', source: m, reason: '低/中风险自动放行' }
  if (m === 'auto') {
    if (name === 'shell' && DANGER.some((x) => (normalize(text) + ' ').includes(x))) return { verdict: 'DENY', source: 'auto', reason: 'auto 分类器看到危险 shell 片段' }
    if (name === 'read_file' && SENSITIVE.some((x) => text.toLowerCase().includes(x))) return { verdict: 'DENY', source: 'auto', reason: 'auto 分类器: 敏感路径' }
    return t.risk === 'high' ? ask('auto 分类器对高风险工具拿不准, 回退问人') : { verdict: 'ALLOW', source: 'auto', reason: `${t.risk} 风险工具, auto 放行` }
  }
  return ask('default 模式: 没有规则命中就问人')
}

// ★ 对应 permissions.py:split_command: 按引号切词, 引号里的 ; & | > 都是普通字符。
// 这些字符 (引号外) 连着的一串算一个操作符, 如 && >>。引号没配对返回 null
// 简化: 不处理反斜杠转义 (shlex 会处理)
const OPS = ';&|<>()\n'
const lex = (cmd) => {
  const toks = []
  let cur = null, i = 0
  const push = () => { if (cur !== null) toks.push(cur); cur = null }
  while (i < cmd.length) {
    const c = cmd[i]
    if (c === "'" || c === '"') {
      const j = cmd.indexOf(c, i + 1)
      if (j < 0) return null
      cur = (cur ?? '') + cmd.slice(i + 1, j)
      i = j + 1
    } else if (c === ' ' || c === '\t' || c === '\r') { push(); i++ }
    else if (OPS.includes(c)) {
      push()
      let j = i
      while (j < cmd.length && OPS.includes(cmd[j])) j++
      toks.push(cmd.slice(i, j))
      i = j
    } else { cur = (cur ?? '') + c; i++ }
  }
  push()
  return toks
}
// 返回 [{ text, redirect }]: 一段是去掉引号后的词用空格拼起来; redirect = 这一段有 > 或 >>
const splitCommand = (cmd) => {
  const toks = lex(cmd)
  if (!toks) return null
  const segs = []
  let words = [], redirect = false
  for (const t of toks) {
    const isOp = t.length > 0 && [...t].every((c) => OPS.includes(c))
    if (isOp && !/[<>]/.test(t) && /[;&|\n]/.test(t)) { // && || ; | & 换行: 一段结束
      if (words.length) segs.push({ text: words.join(' '), redirect })
      words = []
      redirect = false
      continue
    }
    words.push(t)
    redirect = redirect || (isOp && t.includes('>'))
  }
  if (words.length) segs.push({ text: words.join(' '), redirect })
  return segs
}

// 对应 PermissionGate._evaluate_one: 顺序就是优先级。noAllow = 这一段不享受 allow 规则的原因
const evaluateOne = (name, text, noAllow = '') => {
  const stages = []
  let out = null
  for (const d of ['deny', 'ask', 'allow']) {
    const skip = out ? '已有结论, 不再检查'
      : d !== 'deny' && mode.value === 'plan' ? 'plan 模式下 ask/allow 规则不生效 (allow 也不能放行写操作)'
        : d === 'ask' && mode.value === 'bypass_permissions' ? 'bypass_permissions 跳过 ask 规则'
          : d === 'allow' && noAllow ? noAllow : ''
    const hit = skip ? null : rules.find((r) => r.on && r.decision === d && matches(r, name, text, normalized.value))
    if (hit) out = d === 'ask' ? ask(`命中 ask 规则 ${hit.tool} "${hit.pattern}"`) : { verdict: d.toUpperCase(), source: 'rule', reason: `命中 ${d} 规则 ${hit.tool} "${hit.pattern}"` }
    const n = rules.filter((r) => r.on && r.decision === d).length
    stages.push({ name: `${d} 规则`, state: hit ? 'hit' : skip ? 'skip' : 'miss', detail: hit ? `命中: ${hit.tool} "${hit.pattern}"` : skip || `${n} 条规则, 无命中 → 继续` })
  }
  const fb = out ? null : fallback(name, text)
  stages.push({ name: `模式兜底 (${mode.value})`, state: fb ? 'hit' : 'skip', detail: fb ? fb.reason : '已有结论, 不再检查' })
  return { ...(out || fb), stages, text, seen: normalized.value ? normalize(text) : text }
}

const result = computed(() => {
  if (tool.value !== 'shell') return { parts: [evaluateOne(tool.value, arg.value)], idx: 0 }
  const segs = splitCommand(arg.value)
  if (!segs) { // 看不懂的命令不能放行 (fail closed)
    const stages = [{ name: 'split_command', state: 'hit', detail: '引号没配对, shlex 解析失败 → 直接拒绝, 不查规则' }]
    return { parts: [{ verdict: 'DENY', source: 'parser', reason: '引号没配对, 解析不了的命令一律拒绝', stages, text: arg.value, seen: arg.value }], idx: 0 }
  }
  // 命令替换查原始字符串: 切词会把 $( 拆成两个记号
  const subst = /\$\(|`/.test(arg.value) ? '命令替换 $(…) / `…` 里能藏任何东西: 不享受 allow 规则' : ''
  const redir = '输出重定向 (> >>): 整串匹配 allow 规则, 实际却在写文件, 不享受 allow 规则'
  const parts = (segs.length ? segs : [{ text: arg.value, redirect: false }])
    .map((s) => evaluateOne('shell', s.text, subst || (s.redirect ? redir : '')))
  const bad = parts.findIndex((p) => p.verdict !== 'ALLOW' && !(p.verdict === 'ASK' && human.value))
  return { parts, idx: bad < 0 ? 0 : bad }
})
const decisive = computed(() => result.value.parts[result.value.idx])
const final = computed(() => {
  const v = decisive.value.verdict
  if (v === 'ASK') return human.value ? { text: 'ASK → 人批准', cls: 'good' } : { text: 'ASK → 无人 = DENY', cls: 'bad' }
  return { text: v, cls: v === 'ALLOW' ? 'good' : 'bad' }
})
const denyHit = computed(() => {
  const hit = (norm) => result.value.parts.some((p) => rules.some((r) => r.on && r.decision === 'deny' && matches(r, tool.value, p.text, norm)))
  return { naive: hit(false), norm: hit(true) }
})
</script>

<style scoped>
.rule.off { opacity: 0.45; }
.rule.deny { border-color: var(--danger); }
.rule.ask { border-color: var(--warn); }
.rule.allow { border-color: var(--left); }
.call select { max-width: 150px; }
.call .arg { flex: 1; min-width: 160px; }
.risk { font-size: 12px; color: var(--accent); }
.tip { font-size: 12px; color: var(--text-muted); min-width: 64px; }
.call select, .arg { min-height: 32px; background: var(--bg-elev); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0 8px; font-size: 12px; max-width: 100%; min-width: 0; }
.seg { font-size: 12px; color: var(--text-dim); margin: 6px 0; word-break: break-all; }
.seg.decisive { color: var(--text); }
.trace { list-style: none; display: flex; flex-direction: column; gap: 6px; }
.trace li { display: grid; grid-template-columns: 150px 1fr; gap: 10px; align-items: center; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; font-size: 12px; }
.trace li.miss { color: var(--text-muted); }
.trace li.skip { opacity: 0.4; }
.trace li.hit { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
.stage { font-size: 12px; }
.small { font-size: 12px !important; word-break: break-all; text-align: right; }
@media (max-width: 600px) { .trace li { grid-template-columns: 1fr; gap: 2px; } }
</style>
