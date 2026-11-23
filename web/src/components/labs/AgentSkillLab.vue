<!--
  Skill 渐进式披露 (对应 llm_agent/core/skills.py 的 SkillRegistry / SkillTool)。
  只讲一件事: 常驻上下文的只有每个 skill 一行 description, 正文在被触发时才加载。
-->
<template>
  <LabFrame
    title="Skills — 装了一堆任务手册, 渐进式披露让常驻上下文只剩每本一行"
    sub="- 模拟什么: agent 技能手册 (skill) 的按需加载与渐进式披露 (progressive disclosure) 机制。每个 skill 对应一份 SKILL.md 文档, 首部元数据声明简短的 name 与一行功能概述 description, 主体正文则详述具体领域的标准操作规程。若将已安装的全部手册正文一并静态拼接到 system prompt 中, 每轮模型请求都必须为当前完全无关的手册支付高昂的上下文成本。渐进式披露将其拆分为两级按需加载:
  1. 仅将每个已安装 skill 的单行 name 与 description (下称技能目录) 常驻于基础上下文。
  2. 模型阅读目录后若判断某项 skill 与当前用户请求匹配, 主动发起 skill 工具调用, 此时对应手册的完整正文才作为 tool_result 动态载入当前上下文。
- 图上是什么: 技能卡片网格、工具调用轨迹与双横条容量对比:
  - 卡片网格: 每个方格代表一个已安装的 skill, 顶部显示名称, 下方显示单行 description; 前 6 个为具备具体业务语义的示例技能, 其余为模拟团队扩展规模的占位手册。方格变为绿色表示该技能的正文已载入当前上下文, 并标注其占用的正文 token 数。
  - 网格下方等宽文本: 实时显示模型按需发起的 skill 工具调用序列。
  - 底部两根分段横条: 对比两种加载策略占用的上下文容量。紫色段代表常驻目录开销 (全部 skill 单行概述的长度总和), 黄色段代表已载入的手册正文开销; 上方横条对应「全量常驻」(静态注入所有手册正文), 下方横条对应「渐进披露」(仅计入当前已触发的手册正文)。
- 可以做什么: 调节三根规模滑杆并配合预设请求按钮或手动点击测试:
  - 滑杆「已安装 skill 数」: 调整环境中注册挂载的技能总数。
  - 滑杆「每个正文长度」: 设定单份技能手册正文的平均 token 数。
  - 滑杆「description 长度」: 设定目录中单行功能概述的平均 token 数。
  - 四个预设请求按钮: 点击任意一句用户提问, 模拟模型依据目录关键词匹配并自动加载相关 skill 的过程 (末句「今天天气怎么样」不匹配任何专业手册, 保持零正文加载)。
  - 点击任意技能方格: 手动切换单个 skill 正文的加载或卸载状态。
  - 「清空已加载」: 一键卸载全部已载入的正文内容。
- 应该看到: 常驻开销的数量级差异与右侧读数:
  - 默认安装 12 个 skill 且单篇正文为 3000 token 时: 全量常驻高达 36.4K token, 而未触发任何技能时渐进披露仅占 360 token (即 12 行目录开销), 两者相差 101 倍。由于 agent 在多轮循环中每轮均需重发完整上下文, 这一容量差额会在每一次模型调用中重复产生计费差距。
  - 点击复合请求「审一下这个 SQL 迁移, 再写发布说明」: 自动按需加载 2 个相关技能后, 渐进披露总占用升至 6.4K token, 依然显著低于全量常驻基线。
  - 右侧「全量常驻」与「渐进披露」: 分别对应底部两根对比横条的总 token 数。
  - 「节省倍数」: 全量常驻容量相对于当前渐进披露容量的比值。
  - 「全量占 200K 窗口」: 显示静态塞入全部手册正文将占去 200K token 上下文窗口的百分比, 超过半数时标红警示。"
    module="llm_agent/m05"
    run="python -m llm_agent.m05_extensibility.demo"
    :challenge="{
      ask: '假设环境中安装了 40 个 skill, 每个技能的正文长度为 3000 token。请按默认目录长度计算「全量常驻」所需的上下文 token 数, 以及「渐进披露」在仅触发 1 个技能时的 token 数。节省下来的这部分开销是在每一轮模型请求中都会重复节省, 还是仅节省一次?',
      answer: '- 全量常驻开销: $40 \\times (30 + 3000) = 121200$, 约为 121K token, 尚未开始处理具体业务便已占去 200K 上下文窗口的六成容量。\n- 渐进披露开销 (仅触发 1 个技能时): $40 \\times 30 + 3000 = 4200$, 约为 4.2K token, 两者相差约 29 倍。\n将前两根滑杆分别调至 40 个与 3000 token, 再点击预设请求「把这份数据导出成 PDF」(仅命中 pdf-report 单个技能), 右侧面板即精确显示上述两组读数。\n这部分差额在每一轮模型调用中都会持续节省: 由于 agent 循环每次推理均需重发完整上下文历史, 全量常驻方案每推进一轮都必须重复传输全部 121K token, 而渐进披露方案每轮仅传输轻量目录加上当前实际激活的少量正文。\n该机制的工程代价在于召回完全依赖目录层的单行 description: 若功能概述撰写得含糊笼统、无法与用户意图准确对齐, 模型便不会主动发起加载调用, 内部手册正文写得再详尽也无法发挥作用。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="已安装 skill 数" :min="1" :max="40" />
      <LabSlider v-model="body" label="每个正文长度 (一份手册)" :min="200" :max="5000" :step="100" unit=" tok" />
      <LabSlider v-model="desc" label="description 长度 (目录里每行)" :min="10" :max="100" :step="5" unit=" tok" />
      <div class="row">
        <button v-for="p in PROMPTS" :key="p.text" type="button" :class="{ active: prompt === p.text }" @click="ask(p)">{{ p.text }}</button>
        <button type="button" @click="loaded = []; prompt = ''">清空已加载</button>
      </div>
    </template>

    <div class="skills">
      <button
        v-for="s in skills" :key="s.name" type="button" class="skill" :class="{ loaded: loaded.includes(s.name) }"
        :aria-pressed="loaded.includes(s.name)" :title="s.desc" @click="toggle(s.name)"
      >
        <span class="mono nm">{{ s.name }}</span>
        <span class="ds">{{ loaded.includes(s.name) ? `正文已加载 +${body}` : s.desc }}</span>
      </button>
    </div>
    <p v-if="trace" class="mono trace">{{ trace }}</p>

    <div class="bars">
      <div v-for="b in bars" :key="b.label" class="bar-row">
        <span class="lbl">{{ b.label }}</span>
        <div class="track">
          <div class="seg cat" :style="{ width: b.cat / scale * 100 + '%' }" />
          <div class="seg bod" :style="{ width: b.bod / scale * 100 + '%' }" />
        </div>
        <span class="mono num">{{ fmtNum(b.cat + b.bod) }}</span>
      </div>
      <p class="legend"><i class="cat" /> 目录 (name + description) <i class="bod" /> 正文</p>
    </div>

    <template #stats>
      <div class="kv"><span>全量常驻 (所有正文都塞进上下文)</span><b :class="allIn > progressive ? 'bad' : ''">{{ fmtNum(allIn) }} tok</b></div>
      <div class="kv"><span>渐进披露 (目录 + 已加载的正文)</span><b :class="progressive < allIn ? 'good' : ''">{{ fmtNum(progressive) }} tok</b></div>
      <div class="kv"><span>节省倍数 (全量 / 渐进)</span><b>{{ (allIn / progressive).toFixed(1) }}×</b></div>
      <div class="kv"><span>全量常驻占 200K 窗口的比例</span><b :class="allIn > 100000 ? 'bad' : ''">{{ (allIn / 2000).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p>渐进式披露架构通常划分为三个层级, 每一层均遵循按需读取原则:</p>
        <ol class="pts">
          <li>启动时只读每个 SKILL.md 开头的 frontmatter (name 和 description), 正文当场丢掉, 只有这一层常驻上下文。</li>
          <li>模型调用 <code class="inline">skill</code> 工具时, 正文才作为 tool_result 进上下文。</li>
          <li>正文里引用的脚本和附件, 用到才读 (本 demo 没实现这一层)。</li>
        </ol>
        <p>skill 正文属于用户或团队主动安装的受信任操作指令, 模型应当遵照执行; 这与工具从外部网络抓取的网页文档性质截然不同, 后者属于不可信外部数据。两者的安全边界取决于信任来源, 而非文本格式。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, range } from '@/utils/labmath.js'

const NAMED = [
  { name: 'pdf-report', desc: '生成 / 填写 PDF 报告', keys: ['pdf'] },
  { name: 'debug-agent', desc: '排查 agent 没有结果', keys: ['排查', '没有结果'] },
  { name: 'release-notes', desc: '从 git log 写发布说明', keys: ['发布说明', 'release'] },
  { name: 'sql-review', desc: '审查 SQL 迁移脚本', keys: ['sql', '迁移'] },
  { name: 'code-review', desc: '按团队清单做代码评审', keys: ['评审', 'review'] },
  { name: 'deploy', desc: '按流程发布到 staging', keys: ['发布到', '部署'] },
]
const PROMPTS = [
  { text: '排查为什么 agent 没有结果' },
  { text: '把这份数据导出成 PDF' },
  { text: '审一下这个 SQL 迁移, 再写发布说明' },
  { text: '今天天气怎么样' },
]

const n = ref(12), body = ref(3000), desc = ref(30)
const loaded = ref([]), prompt = ref(''), trace = ref('')

const skills = computed(() => range(n.value).map((i) => NAMED[i] || { name: `skill-${i + 1}`, desc: '别的团队手册', keys: [] }))
// 装的 skill 变少时, 已加载但被卸掉的要清理
watch(skills, (list) => { loaded.value = loaded.value.filter((x) => list.some((s) => s.name === x)) })

// 模型只看得到目录: 用 description 关键词命中模拟"模型判断相关" → 调 skill 工具
const ask = (p) => {
  prompt.value = p.text
  const hits = skills.value.filter((s) => s.keys.some((k) => p.text.toLowerCase().includes(k)))
  loaded.value = hits.map((s) => s.name)
  trace.value = hits.length
    ? hits.map((s) => `tool_use skill({"name":"${s.name}"}) → tool_result: 正文 ${body.value} tok`).join('\n')
    : '目录里没有相关 skill → 不加载任何正文, 只付目录的钱'
}
const toggle = (name) => {
  loaded.value = loaded.value.includes(name) ? loaded.value.filter((x) => x !== name) : [...loaded.value, name]
  trace.value = ''
}

// ★ 全量: N×(desc+body) 每轮都付; 渐进: N×desc 常驻 + 只为用到的正文付费
const catalog = computed(() => n.value * desc.value)
const allIn = computed(() => n.value * (desc.value + body.value))
const progressive = computed(() => catalog.value + loaded.value.length * body.value)
const scale = computed(() => Math.max(allIn.value, 1))
const bars = computed(() => [
  { label: '全量常驻', cat: catalog.value, bod: n.value * body.value },
  { label: '渐进披露', cat: catalog.value, bod: loaded.value.length * body.value },
])
</script>

<style scoped>
.skills { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 6px; }
.skill { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; min-height: 0; padding: 6px 8px; text-align: left; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text); }
.skill .nm { font-size: 11px; }
.skill .ds { font-size: 10px; color: var(--text-dim); line-height: 1.4; }
.skill.loaded { border-color: var(--left); background: color-mix(in srgb, var(--left) 18%, transparent); }
.skill.loaded .ds { color: var(--left); }
.trace { font-size: 11px; color: var(--text-muted); margin-top: 10px; white-space: pre-wrap; word-break: break-all; }
.bars { margin-top: 14px; display: flex; flex-direction: column; gap: 6px; }
.bar-row { display: grid; grid-template-columns: 64px 1fr 56px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); }
.track { height: 16px; display: flex; border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }
.seg.cat, .legend i.cat { background: var(--accent); }
.seg.bod, .legend i.bod { background: var(--warn); }
.num { text-align: right; color: var(--text); }
.legend { font-size: 11px; color: var(--text-dim); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; vertical-align: -1px; }
</style>
