<!-- Python 运行证据回放。配置切换与事件步进不是在浏览器里调用模型或执行工具。 -->
<template>
  <LabFrame v-if="record" class="agent-coverage-lab agent-experiment"
    :title="record.title" :sub="description" :module="`llm_agent/${record.module}`"
    :run="`python -m llm_agent.${record.module}.demo`" :challenge="challenge">
    <template #controls>
      <label :for="`${record.module}-mode`">实验条件
        <select :id="`${record.module}-mode`" v-model="mode" aria-label="实验条件">
          <option v-for="entry in record.cases" :key="entry.mode" :value="entry.mode">
            {{ names[entry.mode] || entry.mode }}
          </option>
        </select>
      </label>
      <label class="comparison-toggle"><input v-model="compare" type="checkbox" /> 展开各条件的终态对照</label>
    </template>
    <div class="coverage-timeline" aria-label="执行事件">
      <button v-for="(event, i) in current.trace" :key="`${mode}-${i}`" type="button"
        :aria-pressed="stepper.step.value === i"
        :class="{ active: stepper.step.value === i, good: event.ok, bad: !event.ok }"
        @click="stepper.step.value = i">
        <span class="step-number">{{ i + 1 }}</span>{{ event.title }}
      </button>
    </div>
    <StepPlayer :stepper="stepper" />
    <div class="coverage-detail" aria-live="polite">
      <h4>{{ selected.title }}</h4>
      <Prose :text="selected.detail" />
      <p :class="selected.ok ? 'good' : 'bad'">{{ selected.ok ? '这一步满足了检查条件' : '这一步暴露了失败或适用边界' }}</p>
    </div>
    <div v-if="compare" class="coverage-comparison">
      <h4>各条件跑完后的终态读数 (Python 实际运行记录)</h4>
      <div v-for="entry in record.cases" :key="entry.mode" class="comparison-row">
        <button type="button" :aria-pressed="mode === entry.mode" @click="mode = entry.mode">{{ names[entry.mode] || entry.mode }}</button>
        <dl><div v-for="metric in entry.metrics" :key="metric.label"><dt>{{ metric.label }}</dt><dd>{{ display(metric.value) }}</dd></div></dl>
      </div>
    </div>
    <template #stats>
      <div v-for="metric in current.metrics" :key="metric.label" class="kv">
        <span>{{ metric.label }}</span><b>{{ display(metric.value) }}</b>
      </div>
      <p class="lab-note">右侧指标面板展示的是当前所选实验条件完整运行结束后的终态读数, 不随时间轴游标移动而改变; 时间轴仅用于逐步检视执行过程证据, 切换实验条件才会更新整组终态读数。</p>
      <p class="lab-note">记录来源: {{ record.module }}/demo.py<br />源码 sha256 前 12 位 (用来核对记录和当前源码一致): <code>{{ record.sha256.slice(0, 12) }}</code></p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import Prose from '@/components/Prose.vue'
import { useStepper } from '@/composables/useStepper.js'
import evidence from '@/data/agent-coverage-evidence.json'

const route = useRoute()
const record = computed(() => evidence[route.name])
const mode = ref('')
const compare = ref(false)
const current = computed(() => record.value?.cases.find((row) => row.mode === mode.value) || record.value?.cases[0])
const stepper = useStepper(() => current.value?.trace.length || 1)
const selected = computed(() => current.value?.trace[stepper.step.value] || { title: '', detail: '', ok: false })
watch(record, (value) => { mode.value = value?.cases[0].mode || ''; compare.value = false; stepper.reset() }, { immediate: true })
watch(mode, () => stepper.reset())
const description = "- 模拟什么: 本回放台不在浏览器端实时运行语言模型或外部工具, 而是忠实回放本章 Python demo 在真实执行后导出的结构化轨迹数据。demo 针对同一任务在不同对照配置下各运行一遍, 记录每一步的中间事件与最终评测读数; 面板标题对应 demo 名称, 每项「实验条件」对应代码中的一组配置参数。\n- 图上是什么: 自上而下分为三个区域:\n  - 编号事件时间轴: 按执行先后排列所选条件下的每一步事件按钮; 绿框表示该步骤通过了预期检查条件, 红框表示该步骤触发了故障现象或能力边界 (反例基线配置下的红框正是实验旨在复现的失效模式, 而非程序 bug)。\n  - 单步证据面板: 展示当前选中事件步骤的详细输入、状态变化与校验说明。\n  - 跨条件终态对照表: 勾选展开开关后显示在底部, 将全部实验条件运行结束后的终态指标并排陈列以便横向比对。\n- 可以做什么: 支持三类检视操作:\n  - 切换「实验条件」下拉框: 更换对照配置, 事件序列与右侧终态读数随之整组切换。\n  - 操作事件按钮与播放器: 点击任意编号按钮, 或使用播放器的「上一步 / 下一步」及进度轴逐步核查中间证据。\n  - 勾选「展开各条件的终态对照」: 展开全量实验条件的终态读数矩阵, 点击表中任意一行即可快速跳转至对应条件。\n- 应该看到: 右侧读数固定显示当前所选条件完整运行后的终态结果, 不随时间轴位置变化; 各指标的具体物理含义以本章正文要点说明为准。建议按以下两步定位核心证据:\n  1. 先展开跨条件终态对照表, 找出在不同配置之间发生显著差异的结果指标。\n  2. 再回到单步事件时间轴从头向后排查, 定位首次出现绿框与红框分化的步骤, 该分叉点即为本章核心机制生效或缺失的关键位置。"
const challenge = computed(() => ({
  ask: `先预测: ${record.value?.title.split(' · ')[0]}中, 改变实验条件会让哪一步的证据先变化?`,
  answer: "定位关键分叉步骤可按两步进行:\n1. 勾选「展开各条件的终态对照」, 横向比对哪些终态指标在基线条件与改进条件之间存在差异。\n2. 切回事件时间轴自第 1 步向后逐一对照, 首个出现绿框与红框状态分化的事件即为机制起效的分界点。\n解读回放证据时需注意三项原则:\n- 终态读数相同不代表执行过程等价: 两种配置可能经由不同中间路径到达同一终态 (如盲目猜中与合规查询), 此时必须结合逐步事件日志加以区分。\n- 反例配置的红框属于预期对照现象: 诸如「训练前」「无特权信息」「全部平铺」等基线条件专门用于复现缺失目标机制时的失效位置, 并非实验代码故障。\n- 结论受限于离线夹具范围: 回放记录仅在 demo 给定的固定任务与参数规模下成立, 具体的适用前提与未覆盖场景详见本章正文的「实验边界」说明。",
}))
const names = {
  exact: '精确扫描', tree: '投影树叶子', before: '训练前', 'no-hint': '无特权信息', privileged: '有特权信息', flat: '全部平铺', discovery: '按需发现', poisoned: '描述版本被改', hierarchy: '树状汇总', graph: '关系遍历',
  'one-shot': '一次修改', repair: '失败后修复', budget: '预算耗尽', complete: '正常完成',
  'cancel-before': '提交前取消', 'cancel-after': '提交后取消', 'file-only': '只看文件冲突', revision: '版本检查', invariant: '跨文件约束',
  baseline: '朴素基线', 'no-clarify': '关闭澄清', full: '完整机制', untrained: '训练前', sft: 'SFT 后', rl: '在线 RL 后',
  global: '全局规则提案', scoped: '按作用域处理', rollback: '发布后回滚', half: '半双工', duplex: '全双工',
  'no-generation': '不检查代际', stale: '旧截图坐标', refresh: '刷新观察', verify: '动作后核对',
  'open-loop': '标称开环', feedback: '闭环反馈', robust: '反馈与最坏情况限制', base: '原始知识',
  'domain-only': '只训新域', replay: '混合旧域回放', naive: '错误规则', validated: '规则验收', rejected: '拒绝越界表达式',
  breadth: '广度搜索', prior: '按先验搜索', selected: '训练回放选择', exploit: '利用合作方', mutual: '双方背叛',
  reciprocal: '双方互惠', workflow: '固定流程', agent: '自主执行', hybrid: '混合路由', forget: '完整删除',
}
function display(value) {
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (value === null) return '未产生'
  if (Array.isArray(value)) return value.length ? value.map(display).join(' · ') : '无'
  if (typeof value === 'object') return Object.entries(value).map(([key, val]) => `${key}: ${display(val)}`).join(' / ')
  return String(value)
}
</script>

<style scoped>
.coverage-timeline { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; }
.coverage-timeline button { text-align: left; padding: .6rem .8rem; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); color: var(--text); max-width: 100%; overflow-wrap: anywhere; }
.coverage-timeline button.good { border-color: var(--left); }
.coverage-timeline button.bad { border-color: var(--danger); }
.coverage-timeline button.active { outline: 2px solid var(--accent); outline-offset: 2px; }
.step-number { font-weight: 700; margin-right: .5rem; }
.coverage-detail { padding: .8rem; margin-top: .8rem; border-left: 3px solid var(--accent); background: var(--bg-card); overflow-wrap: anywhere; }
.good { color: var(--left); }
.bad { color: var(--danger); }
.coverage-comparison { margin-top: 1rem; }
.comparison-row { padding: .6rem 0; border-top: 1px solid var(--border); }
.comparison-row dl { margin: .5rem 0; }
.comparison-row dl > div { display: flex; flex-wrap: wrap; gap: .5rem; }
.comparison-row dt { color: var(--text-muted); }
.comparison-row dd { margin: 0; overflow-wrap: anywhere; }
.lab-stats .kv { align-items: flex-start; flex-wrap: wrap; }
.lab-stats .kv b { overflow-wrap: anywhere; max-width: 100%; white-space: normal; }
.comparison-toggle { display: flex; align-items: center; gap: .4rem; }
select { max-width: 100%; }
</style>
