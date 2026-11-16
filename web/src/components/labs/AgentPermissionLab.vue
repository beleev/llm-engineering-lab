<!--
  权限门追踪器 (对应 llm_agent/core/permissions.py 的 PermissionGate)。
  只讲一件事: 裁决顺序 deny → ask → allow → 模式兜底; 以及字符串黑名单为什么天生很弱。
-->
<template>
  <LabFrame
    title="权限门 — 模型发出的一次工具调用, 是怎么被裁决成放行、问人或拒绝的"
    sub="- 模拟什么: 权限门 (PermissionGate): 模型每发出一个工具调用, 执行之前都要过这道门, 门给出三种裁决之一: ALLOW 放行, DENY 拒绝, ASK 问人。裁决顺序固定: 先查 deny 规则, 再查 ask 规则, 再查 allow 规则, 都没命中就按当前「模式」兜底 (比如 default 模式是问人, plan 模式是只读工具才放行)。一条规则就是「工具名 + 参数的通配模式」(glob, 星号匹配任意字符)。这个台子的另一半是演示字符串黑名单有多弱: deny 规则在枚举「坏命令的写法」, 而等价的写法无穷多。
- 图上是什么: 中间是这次调用穿过门的轨迹, 一行一站: deny 规则、ask 规则、allow 规则、模式兜底。紫色高亮的那一站是做出决定的地方; 灰字的站是查了但没命中; 半透明的站是被跳过没查 (已经有结论了, 或者 plan 模式下根本不看 ask / allow 规则)。如果命令里有 && || ; | & 这类连接符, 它会先被拆成几段, 上方逐段列出各段的裁决, 轨迹只画「决定整体结果」的那一段 (第一个不放行的段)。
- 可以做什么: 五排控件从上到下。「模式」是没有规则命中时的兜底策略, 六种: plan 只放行只读工具; default 问人; accept_edits 低中风险放行、高风险问人; auto 用一个粗糙分类器看危险词; dont_ask 只放行 allow 规则命中的, 其余直接拒; bypass_permissions 一律放行, 但 deny 规则仍生效。「规则」五条可以单独开关, 红边是 deny, 黄边是 ask, 绿边是 allow。「匹配方式」在 permissions.py 的归一化匹配 (先小写、压空格、把相邻的短 flag 合并并排序, 比如 RM  -r -f 变成 rm -fr, 规则和命令都这么处理) 和朴素 glob (原样字符串匹配, 当反例) 之间切换; 旁边的「有人审批」决定 ASK 的结果是人批准, 还是无人可问 (无人可问时按拒绝处理, 叫 fail closed)。「预设调用」是 12 条准备好的调用, 点一下就填进下面。「自己输入」可以换工具、改参数, 右边显示这个工具声明的风险等级 risk。
- 应该看到: 默认是 shell 的 rm -fr /: 归一化匹配下 deny 规则命中, 第一站就拒。切到「朴素 glob」, 规则写的是 rm -rf, 命令写的是 rm -fr, 字面不同就漏过, 落到 default 模式问人。右侧「最终裁决」是整体结果; 「谁做的决定」是 rule (规则)、human (问人)、auto (分类器)、parser (命令解析不了) 或某个模式名, 审计时要能回答这一问; 「匹配时看到的文本」是规则实际拿去比的字符串, 归一化模式下是处理过的小写版本; 「deny 规则: 朴素 / 归一化」把同一条命令按两种匹配各试一遍, 结果不一致时标红, 那就是朴素匹配被绕过的写法。再点预设里的 find / -delete: 两种匹配都漏过, 它同样能删光磁盘, 这就是黑名单的极限。"
    module="llm_agent/m03"
    run="python -m llm_agent.m03_permissions.demo"
    :challenge="{
      ask: '归一化匹配下, 「rm -r -f /」「RM  -RF /」都被拦住了。你能想出一条同样删光磁盘、却仍然穿过 deny 规则的命令吗? (试试预设里的 find / -delete)',
      answer: '能, 而且有无穷多条: rm --recursive --force /, find / -delete, python -c 后面跟一句 shutil.rmtree, 等等。\n一步步看: deny 规则 *rm -rf* 是一个字符串模式。归一化只堵住最廉价的绕过 (大小写、多余空格、flag 的顺序和合并); 换一个命令名, 或者换一种写 flag 的方式, 字符串就不一样了。黑名单是在枚举「坏」, 而坏的写法是无穷的, 所以 deny 规则只能当最后一道便宜的网, 用来拦常见的误操作。\n真正的安全边界要靠三样叠起来:\n- 默认拒绝的 allowlist: 没有明确放行的就不放行。default 模式没命中规则就问人、无人可问就拒, 就是这个意思。\n- 把命令解析成结构再检查, 而不是比字符串。\n- OS 级沙箱, 限制进程能碰的文件系统和网络。\n注意 find / -delete 在这个台子里并没有被放行: default 和 auto 模式下它落到「问人」, 无人可问时 fail closed 拒绝。拦住它的是默认拒绝, 不是 deny 规则认出了危险; 把模式切到 bypass_permissions, 它就直接通过了。',
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
        <p>拆分认引号: <code class="inline">echo 'a; b'</code> 里的分号在引号内, 算一段。带输出重定向 (&gt; &gt;&gt;) 的段不享受 allow 规则, 因为 <code class="inline">echo x &gt; ~/.bashrc</code> 整串匹配 allow "echo *", 实际却在改文件。引号没配对的命令解析不了, 直接拒绝。</p>
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
