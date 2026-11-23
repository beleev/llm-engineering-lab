<template>
  <div>
    <h1 class="page-title">Agent 应用层 · 把模型接成可行动系统</h1>
    <div class="page-subtitle lead-group">
      <p>
        把大模型接成可靠运行的 Agent, 核心工程不在那十几行循环, 而在外围的控制面: 工具参数怎么校验、高危命令怎么拦截、长对话怎么压缩、中断后怎么恢复。
      </p>
      <p>
        为了把这些控制机制单独拆出来验证, <RepoLink path="llm_agent/" label="llm_agent/" tiny /> 默认使用确定性的 <code class="inline">RuleBasedLLM</code> 代替真实 API, 让每个模块都能用 <code class="inline">assert</code> 精确检验行为; 权限模式、hook 事件、<code class="inline">mcp__server__tool</code> 命名与消息结构均对齐 Claude Code。
      </p>
    </div>

    <ChapterIntro
      :tldr="tldr"
      question="为什么具备工具调用能力的大模型本身, 尚不能直接构成可靠的生产级 Agent 系统?"
      :goals="[
        '掌握 Agent 主循环的最小消息闭环: messages、tool_use 与 tool_result 的配对流转',
        '厘清权限门、上下文预算管理、生命周期 hooks 与持久化存储各自承担的安全与可靠性边界',
        '理解生产级 Agent harness 如何将这些确定性控制组件正交组装在同一执行循环外围',
      ]"
      :codes="[
        { path: 'llm_agent/core/' },
        { path: 'llm_agent/full_loop/' },
        { path: 'llm_agent/m01_agent_loop/' },
        { path: 'llm_agent/m02_tool_use/' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <StageContents stage-id="agent" />

    <section class="section">
      <h2>1. 应用层依赖链</h2>
      <div class="lead-group">
        <p>从「单轮生成文本」到「稳定把长任务做完」, 中间多出来的全是确定性的控制层: 先跑通最小循环, 再逐层叠上工具、权限、上下文管理、扩展接口、状态持久化与子智能体隔离。</p>
      </div>
      <EvolutionChain
        title="从 completion 到 mini Agent harness"
        subtitle="根问题: 单次文本生成无法与外部环境交互或跨越长周期任务保持受控状态, 如何围绕模型构建确定性的执行与安全边界。"
        :steps="agentChain"
      />
    </section>

    <section class="section">
      <h2>2. Agent harness 的六个核心状态账本</h2>
      <p class="lead">
        阅读 Agent 工程源码时, 核心不在于单次模型推理调用, 而在于定位以下六个状态与策略账本的维护位置及其交互顺序。
      </p>
      <div class="grid grid-3">
        <div v-for="p in primitives" :key="p.name" class="card primitive-card">
          <h3>{{ p.name }} <span class="tag">{{ p.tag }}</span></h3>
          <p class="desc">{{ p.desc }}</p>
          <pre class="code">{{ p.code }}</pre>
          <p class="hint"><CodeRef :value="p.file" base="llm_agent/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 模型与 harness 的职责边界</h2>
      <div class="lead-group">
        <p>本代码库用极简的确定性替身 <code class="inline">RuleBasedLLM</code> 隔离外部 API 波动, 以凸显系统内部的职责切分:</p>
        <ul class="pts">
          <li><b>模型层:</b> 负责根据当前对话上下文决策「下一步请求哪个动作或给出最终回复」, 本质上是概率性推理。</li>
          <li><b>harness 层:</b> 负责裁决「该动作是否允许执行、参数是否合法、执行结果如何记入状态账本」, 全程由确定性代码保障。</li>
        </ul>
        <p>任何涉及权限控制、命令拦截与密钥保护的安全边界, 都必须落实在确定性的 harness 层, 绝不能依赖模型自觉遵守提示词约束。</p>
      </div>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>模型负责选择动作 <span class="tag">概率性决策</span></h3>
          <pre class="code">{{ modelSide }}</pre>
          <p class="hint">
            替换为真实大模型 API 时, 此处即对应原生 tool calling 返回的结构化调用请求。下游的工具参数校验、权限门评估与结果落盘代码无需任何改动, 因为主循环对模型接口的唯一契约就是 <code class="inline">next()</code>。
          </p>
        </div>
        <div class="card">
          <h3>harness 负责执行边界 <span class="tag">确定性控制</span></h3>
          <pre class="code">{{ harnessSide }}</pre>
          <p class="hint">
            系统能否可靠拦截高危动作、控制上下文长度、在中断后精确恢复会话以及在事后逐行审计复盘, 完全取决于这段外围控制流。
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 模块索引 · 运行与验证方式</h2>
      <div class="lead-group">
        <p>所有子模块均无需配置外部模型密钥即可离线运行验证, 支持以下两种执行方式:</p>
        <ul class="pts">
          <li>单独运行任意子模块的 <code class="inline">demo.py</code>, 逐步观察对应控制机制的输入、拦截与状态变化。</li>
          <li>
            执行 <code class="inline">python -m llm_agent.run_all</code>
            按演进依赖顺序一次性跑通全部模块与断言校验 (<RepoLink path="llm_agent/run_all.py" label="llm_agent/run_all.py" tiny />)。
          </li>
        </ul>
      </div>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="agent-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>要点</th>
              <th>与生产级 Agent 的对应关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in agentModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. full_loop · 对齐 Claude Code 架构的最小集成闭环</h2>
      <p class="lead">
        <RepoLink path="llm_agent/full_loop/demo.py" label="full_loop/demo.py" tiny /> 将本阶段全部控制机制组装进同一个 <code class="inline">Agent</code> 实例:
        向量检索、笔记写入、按需加载的 Skill、文档抓取、Shell 命令、子智能体委托 (<code class="inline">delegate</code>) 与外部 MCP 工具统一接入同一条权限、审计与安全护栏执行流水线。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>系统组装入口 <span class="tag">main</span></h3>
          <pre class="code">{{ fullLoopCode }}</pre>
        </div>
        <div class="card">
          <h3>集成验证场景 <span class="tag">五个端到端用例</span></h3>
          <div class="run-list">
            <div v-for="r in runRows" :key="r.title" class="run-row">
              <span class="pill">{{ r.step }}</span>
              <div>
                <strong>{{ r.title }}</strong>
                <p>{{ r.body }}</p>
              </div>
            </div>
          </div>
          <p class="hint">
            运行命令: <code class="inline">python -m llm_agent.full_loop.demo</code>
            (<RepoLink path="llm_agent/full_loop/demo.py" label="源码" tiny />)。
          </p>
        </div>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '推理服务负责高效产出 token, Agent harness 负责安全编排外部动作' }"
      :next="{ ...nextChapter, hint: '从十几行代码的最小消息与工具调用闭环出发' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { agentModules, learningPath } from '@/data/models.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'agent')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 最后一个模块的编号从数据里取, 不写死 (agentModules 里除了 mNN 还有一行 full)
const lastModule = agentModules.filter((m) => /^m\d+$/.test(m.id)).at(-1).id
const tldr = `Agent 系统的运行核心是一个多轮交互主循环:
1. 模型读取当前对话历史 messages 与可用工具 Schema, 决定发起工具调用或输出最终回复。
2. 宿主程序校验并执行工具调用, 将结构化的执行结果追加回对话历史。
3. 再次将更新后的历史交由模型推理, 循环往复直至任务收敛。
这一核心循环本身仅有十余行代码, 从 m01 到 ${lastModule} 主干结构保持不变; 系统的安全性与工程可靠性取决于环绕在主循环外围的六道确定性边界:
- 工具注册表: 限定模型可选择的合法动作集合并校验入参结构。
- 权限门: 在工具实际执行前按规则优先级与权限模式完成放行或拦截裁决。
- 上下文与记忆管理: 按需检索外部记忆文件, 并在历史逼近窗口上限时清理陈旧结果或生成摘要压缩。
- 生命周期 hooks: 在工具调用前后等固定事件点注入确定性的审计、拦截与参数改写逻辑。
- JSONL 会话持久化: 以仅追加方式记录完整对话轨迹, 支持进程中断后恢复上下文。
- 子智能体隔离: 将高噪声子任务委托给拥有独立上下文的子 Agent 执行, 仅向父级返回精简摘要。`

const agentChain = [
  {
    name: 'Loop',
    year: 'm01',
    pain: '单轮文本生成接口仅能一次性返回静态回复, 无法将外部工具的执行结果回填至下一轮推理。',
    fix: '建立最小消息闭环: 模型根据 messages 输出 ModelAction, 宿主执行工具封装为 ToolResult 并追加回 messages 驱动下一轮迭代。',
    color: 'var(--left)',
  },
  {
    name: 'Tools',
    year: 'm02',
    pain: '语言模型本身无法直接读取实时外部状态, 也不具备执行本地计算或系统调用的能力。',
    fix: '通过结构化 Tool Schema 向模型声明可用能力与参数约束, 经入参校验后由确定性的本地代码完成 execute 执行。',
    color: 'var(--accent)',
  },
  {
    name: 'Permission',
    year: 'm03',
    pain: '接入文件写入与命令执行等具有副作用的工具后, 模型误判或提示词注入会直接带来破坏风险。',
    fix: '工具执行前强制经过权限门评估: 显式规则按 deny、ask、allow 优先级依次裁决, 未命中规则的调用由六种权限模式兜底控制。',
    color: 'var(--warn)',
  },
  {
    name: 'Context',
    year: 'm04',
    pain: '多轮工具调用返回的大段输出与长期记忆不断累积, 极易超出模型上下文窗口预算。',
    fix: '采用三级上下文治理: 外部记忆文件按需检索注入, 超预算时优先清理陈旧工具输出, 仍超限时触发模型摘要压缩。',
    color: 'var(--eye)',
  },
  {
    name: 'Extensible',
    year: 'm05',
    pain: '将全部业务规范、审计策略与外部工具说明硬编码进系统 prompt 会持续推高每次推理开销。',
    fix: '分层解耦扩展能力: Hooks 在生命周期事件点运行确定性策略, Skills 按需加载领域操作指南, MCP 通过标准协议接入外部进程工具。',
    color: 'var(--right)',
  },
  {
    name: 'State + Team',
    year: 'm06-m07',
    pain: '长周期任务需要支持故障中断恢复, 同时复杂子任务的中间检索过程容易污染主对话上下文。',
    fix: '以仅追加 JSONL 持久化会话 transcript 实现状态恢复; 通过独立上下文运行子智能体并仅向父级返回精简结果摘要。',
    color: 'var(--left)',
  },
]

const primitives = [
  {
    name: 'Transcript',
    tag: 'messages',
    desc: '采用与 Messages API 一致的 content block 消息结构: assistant 消息发出带唯一 id 的 tool_use 请求, 执行结果封装为携带相同 tool_use_id 的 tool_result 放入下一条 user 消息完成配对。',
    file: 'llm_agent/core/schema.py',
    code: `Message("user", prompt)
Message("assistant", [{"type": "tool_use", "id": "toolu_0001", ...}])
Message("user", [{"type": "tool_result", "tool_use_id": "toolu_0001", ...}])`,
  },
  {
    name: 'Tool Pool',
    tag: 'actions',
    desc: '工具注册表维护模型可调用的全部合法动作集合, 统一导出参数 Schema 并作为工具参数校验与分发执行的唯一入口。',
    file: 'llm_agent/core/tools.py',
    code: `tools = ToolRegistry([
    CalculatorTool(),
    SearchDocsTool(DOCS),
    WriteNoteTool(notes),
])`,
  },
  {
    name: 'Permission',
    tag: 'guard',
    desc: '权限门在工具实际执行前按 deny、ask、allow 规则优先级及权限模式兜底策略完成裁决, 且评估对象为经 pre_tool_use 钩子改写后的最终调用参数。',
    file: 'llm_agent/core/permissions.py',
    code: `final = pre_hook.updated_call or call
outcome = permissions.evaluate(final, tool)  # 评估改写后的调用
if not outcome.allowed:
    return ToolResult(call.name, "DENIED", ok=False)`,
  },
  {
    name: 'Memory',
    tag: 'context',
    desc: '长期记忆以普通 .md 文件形式存储并按当前查询相关性动态检索注入上下文; 当字符总量超出预算时, 先清理早期轮次的工具返回内容, 仍超限时再调用模型生成对话摘要。',
    file: 'llm_agent/core/memory.py',
    code: `base += memory_messages(memory, prompt)
view = clear_tool_results(messages, keep_last=1)
if total_chars(view) > budget: compact()   # 模型写摘要`,
  },
  {
    name: 'Hooks',
    tag: 'policy',
    desc: '通过在 pre_tool_use、post_tool_use 等固定生命周期节点挂载确定性回调函数, 将安全拦截、参数改写与日志审计从模型 prompt 中彻底剥离。',
    file: 'llm_agent/core/hooks.py',
    code: `hooks.register("pre_tool_use", block_secret_shell)
hooks.register("post_tool_use", audit_tool_result)`,
  },
  {
    name: 'Persistence',
    tag: 'resume',
    desc: '采用仅追加写入的 JSONL 文件持久化每一条会话消息; 重启恢复时仅加载历史对话记录, 不自动继承上一会话临时授予的高风险权限。',
    file: 'llm_agent/core/persistence.py',
    code: `store.append(message)
messages = store.load()   # load_history=True`,
  },
]

const modelSide = `prompt = latest_user(messages)
if "计算" in prompt:
    return ModelAction.tool(
        ToolCall("calculator", {"expr": expr})
    )

if last_message_is_tool_result:
    return ModelAction.final(summary)`

const harnessSide = `action = llm.next(context, tools.schemas())
final = hooks.on_pre_tool_use(call).updated_call or call
outcome = permissions.evaluate(final, tool)
if outcome.allowed:
    result = tools.execute(final)      # 先 validate_args
else:
    result = ToolResult(name, "DENIED", ok=False)

messages.append(Message("user", [result.to_block()]))
store.append(message)`

const fullLoopCode = `tools = ToolRegistry([
    VectorSearchTool(index), WriteNoteTool(notes), SkillTool(skills),
    FetchDocTool(PAGES), ShellTool(), delegate, *mcp_tools(mcp),
])

agent = Agent(
    llm=RuleBasedLLM(),
    tools=tools,
    permissions=PermissionGate(mode="auto", rules=rules),
    hooks=hooks, memory=memory, store=store,
    guardrails=Guardrails(),
)`

const runRows = [
  { step: '1', title: '按需加载 Skill、文档检索与笔记写入', body: '先按需加载目标 SKILL.md 的操作指南正文, 再执行 TF-IDF 检索并将汇总结果写入笔记工具。' },
  { step: '2', title: '委托子智能体独立执行调研', body: '父 Agent 通过 delegate 启动独立上下文的子 Agent 完成多步检索, 父级会话仅接收最终 summary 摘要。' },
  { step: '3', title: '调用真实子进程提供的 MCP 工具', body: '通过 stdio JSON-RPC 协议拉起本地子进程服务并注册 mcp__weather__* 工具, 经显式 allow 规则匹配放行。' },
  { step: '4', title: '归一化拦截变体高危命令', body: '面对调换参数顺序的 rm -fr 命令, 权限层在命令归一化后准确命中 deny 规则并拒绝执行。' },
  { step: '5', title: '防御外部文档提示词注入与密钥泄露', body: '抓取到的不可信外部内容自动打上来源标记并对夹带的密钥脱敏, 同时触发会话污点规则锁死后续高风险工具。' },
]
</script>

<style scoped>
table.agent-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.agent-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.agent-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.agent-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.agent-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.run-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.run-row {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 12px;
  align-items: start;
  padding: 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
.run-row strong {
  display: block;
  font-size: 13px;
  margin-bottom: 2px;
}
.run-row p {
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}
</style>
