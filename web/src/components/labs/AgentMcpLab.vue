<!--
  MCP 握手时序 (对应 llm_agent/core/mcp.py 的 MCPClient / MCPTool 与 m09_mcp/server.py)。
  只讲一件事: MCP 就是 stdio 上逐行的 JSON-RPC 2.0; 外部工具进来以后, 和内置工具走同一个校验 + 权限门。
-->
<template>
  <LabFrame
    title="MCP — 住在另一个进程里的工具是怎么接进 agent 的, 一次调用走了哪些报文"
    sub="- 模拟什么: MCP (Model Context Protocol) 跨进程工具接入机制。agent 主进程 (作为 MCP client) 将外部 server 程序作为子进程启动, 通过标准输入输出管道 (stdio) 按行收发 JSON-RPC 2.0 报文 (协议规定请求携带唯一 id, 响应必须回传相同 id)。在模型视角下, MCP 外部工具与本地内置工具完全同构, 均表现为「工具名称 + 参数 Schema」, 接入后同样经过统一的参数校验与权限门管控。本实验模拟一个提供 get_weather 与 add 两个工具的精简天气 server, 标准接入流程分为三步:
  1. 协议握手协商 (initialize)。
  2. 拉取可用工具清单及其参数定义 (tools/list)。
  3. 发起具体工具执行请求 (tools/call)。
- 图上是什么: 左右双栏跨进程时序图, 左侧代表 agent 主进程, 右侧代表 server 子进程:
  - 靠左带 ──▶ 的紫边条: agent 写入管道发往 server 的单行 JSON 报文 (请求或通知)。
  - 靠右带 ◀── 的绿边条: server 从管道返回给 agent 的单行 JSON 响应报文。
  - 带 ● 的虚线条: 仅在 agent 进程内存态执行、不产生跨进程通信的内部步骤 (包括拉起子进程、模型输出 tool_use、权限门裁决、本地参数校验与结果回填)。
  - 状态颜色: 红色代表发生异常或拦截的步骤, 灰色代表尚未播放到的步骤, 黄色外框高亮当前所在步骤。
  - 时序图下方: 展示当前步骤的机制说明与原始 JSON 报文载荷。
- 可以做什么: 第一排切换四个通信场景, 第二排控制权限开关, 底部使用播放器逐步查看:
  - 「正常调用」: 参数 city 传入合法字符串 Beijing。
  - 「参数类型错」: 参数 city 误传为数值 123。
  - 「工具内部出错」: 参数 city 传入 Atlantis, 触发 server 端执行异常。
  - 「server 不支持的方法」: client 请求调用该 server 未实现的 resources/list 方法。
  - 第二排规则开关: 控制权限门是否配置 allow mcp__weather__* 放行规则。未配置该规则时, 系统将外部 MCP 工具一律视作高风险操作, 需转交人工审批, 在无人值守环境下默认拒绝。
  - 底部播放器: 按时序逐步推进, 或直接点击图上任意条目查看对应时刻的报文内容。
- 应该看到: 各异常场景的拦截位置对照与右侧读数:
  - 正常调用: 完整走通全部 12 步, 经由 stdio 管道发送的带 id 请求仅有 initialize、tools/list 与 tools/call 三条, 另加一条不含 id 的 notifications/initialized 握手确认通知。
  - 「参数类型错」: 流程在本地 validate_args 步骤即告终止。agent 利用 tools/list 阶段缓存的 inputSchema 在本地完成类型校验, 直接将非法参数转化为带 is_error 标记的 tool_result 回填模型, server 端未收到任何 tools/call 报文。
  - 关闭 allow 规则: 流程在更上游的权限门步骤即被拦截, 同样不产生跨进程调用报文。
  - 右侧「已发 JSON-RPC 请求」: 统计已写入管道且带有 id 的正式请求数。
  - 「到达 server 的 tools/call」: 统计实际发往子进程的工具执行请求数, 为 0 时标绿, 表示非法或未授权调用已在本地边界被拦截。
  - 「stdio 上已走的 JSON 行」: 分别统计发送与接收的报文总行数 (含无 id 通知)。
  - 「结局」: 概括当前场景的最终处理结果。"
    module="llm_agent/m09"
    run="python -m llm_agent.m09_mcp.demo"
    :challenge="{
      ask: '选择「参数类型错」场景 (参数 city 传入数值 123)。先预测: 本次运行会产生几条 tools/call 报文? server 子进程会收到这个错误参数吗?',
      answer: '产生 0 条 tools/call 报文, server 子进程不会收到该错误参数。\ntools/list 响应中每个工具均附带一份 inputSchema (即 JSON Schema, 明确声明 city 字段必须为 string 类型)。MCP client 将其直接注册为本地工具校验规则:\n1. 在发起跨进程调用前先执行本地 validate_args。\n2. 检测到数值 123 不符合 string 类型约束, 校验立即失败。\n3. client 在本地直接构造错误码为 INVALID_ARGS 且标记 is_error 的 tool_result 回填给模型, 供模型在下一轮修正参数。\n整个调用链停留在本地 validate_args 阶段, 右侧「到达 server 的 tools/call」读数为 0。\n若进一步关闭 allow 放行规则, 拦截位置还会继续前移:\n1. 权限门检索后发现无任何显式规则命中。\n2. 系统对外部 MCP 工具一律按 high 高风险等级管控 (不采信外部 server 自报的只读声明), 要求人工介入审批。\n3. 当前环境无人值守, 按 fail closed (故障闭合) 原则直接拒绝执行, 同样不向管道发送任何 tools/call 报文。\n上述两道防线均部署在跨进程发包之前: MCP 协议仅解决「如何以标准格式接入外部工具」, 而不解决「外部工具是否可信」, 安全边界仍由 agent 自身的本地参数校验与权限门强制实施。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="s in SCENES" :key="s.id" type="button" :class="{ active: scene === s.id }" @click="scene = s.id">{{ s.label }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: allowRule }" :aria-pressed="allowRule" @click="allowRule = !allowRule">{{ allowRule ? '☑' : '☐' }} allow 规则 mcp__weather__*</button>
      </div>
      <StepPlayer :stepper="stepper" />
    </template>

    <div class="seq">
      <div class="heads"><span>agent 进程 · MCPClient</span><span>server 子进程 · stdio</span></div>
      <button
        v-for="(s, i) in steps" :key="i" type="button" class="step" :class="[s.dir, s.state, { dim: i > stepper.step.value, now: i === stepper.step.value }]"
        @click="stepper.pause(); stepper.step.value = i"
      >
        <span class="mono title">{{ s.dir === 'req' ? '──▶ ' : s.dir === 'res' ? '◀── ' : '● ' }}{{ s.title }}</span>
      </button>
    </div>
    <div class="payload">
      <p class="note">{{ cur.note }}</p>
      <pre v-if="cur.json" class="mono">{{ JSON.stringify(cur.json, null, 1) }}</pre>
    </div>

    <template #stats>
      <div class="kv"><span>已发 JSON-RPC 请求 (带 id 的)</span><b>{{ sent.requests }}</b></div>
      <div class="kv"><span>到达 server 的 tools/call (0 = 本地就拦住了)</span><b :class="sent.calls ? '' : 'good'">{{ sent.calls }}</b></div>
      <div class="kv"><span>stdio 上已走的 JSON 行 (发 / 收)</span><b>{{ sent.out }} / {{ sent.back }}</b></div>
      <div class="kv"><span>结局</span><b :class="outcome.cls">{{ outcome.text }}</b></div>
      <div class="lab-note">
        <p>模型看到的工具名是 <code class="inline">mcp__weather__get_weather</code>, 不是 server 自己叫的 get_weather。前缀 <code class="inline">mcp__server__tool</code> 有两个用途:</p>
        <ul class="pts">
          <li>防止两个 server 的工具和内置工具重名。</li>
          <li>让权限规则可以按 server 粒度写, 一条 allow mcp__weather__* 就管住这个 server 的全部工具。</li>
        </ul>
        <p>发往 server 的 tools/call 报文会自动剥离本地命名空间前缀, 还原为该 server 原生定义的工具名称。</p>
        <p>右侧读数仅统计播放器当前已回放的步骤; 将时间轴向前回拖时, 各项统计值也会同步回退。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const SCENES = [
  { id: 'ok', label: '正常调用', name: 'get_weather', args: { city: 'Beijing' } },
  { id: 'badargs', label: '参数类型错', name: 'get_weather', args: { city: 123 } },
  { id: 'toolerr', label: '工具内部出错', name: 'get_weather', args: { city: 'Atlantis' } },
  { id: 'nomethod', label: 'server 不支持的方法', name: 'get_weather', args: { city: 'Beijing' } },
]
const scene = ref('ok')
const allowRule = ref(true)

const TOOLS = [
  { name: 'get_weather', description: 'Return a (fake) weather report for a city.', inputSchema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'], additionalProperties: false } },
  { name: 'add', description: 'Add two numbers.', inputSchema: { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'], additionalProperties: false } },
]
const rpc = (id, method, params) => ({ jsonrpc: '2.0', id, method, params })

const steps = computed(() => {
  const sc = SCENES.find((s) => s.id === scene.value)
  const full = `mcp__weather__${sc.name}`
  const st = [
    { dir: 'local', title: 'spawn server 子进程', note: 'Popen(argv, stdin=PIPE, stdout=PIPE): 只拉起调用方给的 argv, 不经过 shell。之后 stdout 就是协议通道, server 的日志只能写 stderr。' },
    { dir: 'req', title: 'initialize', json: rpc(1, 'initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'llm_agent', version: '0.1' } }), note: '2025-06-18 教学配置的握手第一步; 2026-07-28 已改为逐请求元数据, 见本章版本对照实验。每条请求带递增的 id, 响应必须带同一个 id。' },
    { dir: 'res', title: 'result: serverInfo + capabilities', json: { jsonrpc: '2.0', id: 1, result: { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'toy-weather', version: '0.1' } } }, note: 'server 声明自己支持 tools 能力。' },
    { dir: 'req', title: 'notifications/initialized', json: { jsonrpc: '2.0', method: 'notifications/initialized' }, note: '没有 id = 通知 (notification): 规范要求 server 不回复。' },
    { dir: 'req', title: 'tools/list', json: rpc(2, 'tools/list', {}), note: '问 server: 你有哪些工具?' },
    { dir: 'res', title: 'result: tools[2]', json: { jsonrpc: '2.0', id: 2, result: { tools: TOOLS } }, note: `每个工具自带 JSON Schema。client 把它们包成 MCPTool 注册进 ToolRegistry, 名字变成 ${full.replace(sc.name, '*')}; risk 一律按 high (server 自报的只读注解不可信)。` },
  ]
  if (scene.value === 'nomethod') {
    st.push({ dir: 'req', title: 'resources/list', json: rpc(3, 'resources/list', {}), note: 'client 试探一个这个 server 没实现的方法。' })
    st.push({ dir: 'res', state: 'bad', title: 'error -32601', json: { jsonrpc: '2.0', id: 3, error: { code: -32601, message: 'Method not found' } }, note: '协议层错误用 error 字段 (没有 result)。它和「工具执行失败」是两回事, 可以对比「工具内部出错」场景。' })
    return st
  }
  st.push({ dir: 'local', title: `模型发出 tool_use ${full}`, json: { type: 'tool_use', id: 'toolu_0001', name: full, input: sc.args }, note: '对模型来说, 它和内置工具没有任何区别: 都是工具列表里的一个 name + input_schema。' })
  const allowed = allowRule.value
  st.push({ dir: 'local', state: allowed ? 'ok' : 'bad', title: `权限门: ${allowed ? 'ALLOW (rule)' : 'ASK → 无人 = DENY'}`, note: allowed ? '命中 allow mcp__weather__* → 放行。' : '没有规则命中; MCPTool.risk = high → 问人; 无人可问 → fail closed。报文不会发出。' })
  if (!allowed) return st
  // ★ inputSchema 在本地先校验: 坏参数到不了 server
  const bad = typeof sc.args.city === 'number'
  st.push({ dir: 'local', state: bad ? 'bad' : 'ok', title: `validate_args: ${bad ? 'INVALID_ARGS' : '通过'}`, json: bad ? { type: 'tool_result', tool_use_id: 'toolu_0001', content: "INVALID_ARGS: 'city' should be string, got int", is_error: true } : null, note: bad ? '用 tools/list 拿到的 inputSchema 在本地校验, 错误直接回填给模型, 让它下一轮改参数。' : '参数符合 inputSchema。' })
  if (bad) return st
  st.push({ dir: 'req', title: 'tools/call', json: rpc(3, 'tools/call', { name: sc.name, arguments: sc.args }), note: `name 是 "${sc.name}": 前缀 mcp__weather__ 已被剥掉。` })
  const err = scene.value === 'toolerr'
  const text = err ? "tool error: KeyError('Atlantis')" : 'Beijing: sunny, 24C, light wind'
  st.push({ dir: 'res', state: err ? 'bad' : 'ok', title: `result: isError=${err}`, json: { jsonrpc: '2.0', id: 3, result: { content: [{ type: 'text', text }], isError: err } }, note: err ? '工具失败是「正常结果」: 仍然走 result, 只是 isError=true。协议没坏, 模型可以据此换个做法。(教学 server 对任何城市都会回答, 这里的异常是示意。)' : '结果是 content block 列表, client 取出其中的 text。' })
  st.push({ dir: 'local', title: err ? 'tool_result 回填 (is_error)' : 'tool_result 回填 (标记为不可信数据)', json: { type: 'tool_result', tool_use_id: 'toolu_0001', content: err ? text : `<untrusted_data>\n${text}\n</untrusted_data>`, is_error: err }, note: 'MCP 工具的输出来自第三方: untrusted_output = True, 开了护栏就会被包进 <untrusted_data>, 见「护栏」一章。' })
  return st
})

const stepper = useStepper(() => steps.value.length, { interval: 1100 })
watch(steps, (s) => { stepper.pause(); stepper.step.value = s.length - 1 }, { immediate: true })
const cur = computed(() => steps.value[stepper.step.value] || steps.value[0])
const sent = computed(() => {
  const past = steps.value.slice(0, stepper.step.value + 1)
  const seen = past.filter((s) => s.dir === 'req' && s.json.id)
  // 发 = 请求 + 通知 (通知没有 id, 也占一行); 收 = 响应
  return { requests: seen.length, calls: seen.filter((s) => s.json.method === 'tools/call').length, out: past.filter((s) => s.dir === 'req').length, back: past.filter((s) => s.dir === 'res').length }
})
const outcome = computed(() => {
  const last = steps.value[steps.value.length - 1]
  if (scene.value === 'nomethod') return { text: '协议错误 -32601', cls: 'bad' }
  if (!allowRule.value) return { text: '权限门拒绝, 未发报文', cls: 'good' }
  if (last.title.startsWith('validate')) return { text: '本地拦下坏参数', cls: 'good' }
  return scene.value === 'toolerr' ? { text: 'isError 结果回填', cls: '' } : { text: '成功', cls: 'good' }
})
</script>

<style scoped>
.seq { display: flex; flex-direction: column; gap: 4px; }
.heads { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); padding: 0 4px 4px; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.step { min-height: 0; padding: 5px 10px; width: 78%; text-align: left; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text); }
.step.req { border-color: var(--accent); }
.step.res { align-self: flex-end; text-align: right; border-color: var(--left); }
.step.local { width: 60%; border-style: dashed; }
.step.bad { border-color: var(--danger); color: var(--danger); }
.step.dim { opacity: 0.3; }
.step.now { outline: 2px solid var(--warn); outline-offset: 1px; }
.title { font-size: 11px; word-break: break-all; }
.payload { margin-top: 12px; min-height: 150px; }
.note { font-size: 12px; color: var(--text-muted); line-height: 1.7; }
.payload pre { margin-top: 6px; font-size: 11px; line-height: 1.45; background: var(--code-bg); color: var(--code-text); border-radius: var(--radius-sm); padding: 8px 10px; overflow-x: auto; max-height: 260px; }
.small { font-size: 12px !important; word-break: break-all; text-align: right; }
@media (max-width: 600px) { .step, .step.local { width: 92%; } }
</style>
