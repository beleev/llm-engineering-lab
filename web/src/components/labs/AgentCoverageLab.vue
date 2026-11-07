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
      <p :class="selected.ok ? 'good' : 'bad'">{{ selected.ok ? '本步满足检查条件' : '本步暴露失败或适用边界' }}</p>
    </div>
    <div v-if="compare" class="coverage-comparison">
      <h4>所有配置的终态 · Python 已运行记录</h4>
      <div v-for="entry in record.cases" :key="entry.mode" class="comparison-row">
        <button type="button" :aria-pressed="mode === entry.mode" @click="mode = entry.mode">{{ names[entry.mode] || entry.mode }}</button>
        <dl><div v-for="metric in entry.metrics" :key="metric.label"><dt>{{ metric.label }}</dt><dd>{{ display(metric.value) }}</dd></div></dl>
      </div>
    </div>
    <template #stats>
      <div v-for="metric in current.metrics" :key="metric.label" class="kv">
        <span>{{ metric.label }}</span><b>{{ display(metric.value) }}</b>
      </div>
      <p class="lab-note">这些读数是所选配置的完整运行结果，不随时间轴改变。时间轴用于逐步查看证据，切换配置才会换一组读数。</p>
      <p class="lab-note">来源：{{ record.module }}/demo.py<br />源码摘要：<code>{{ record.sha256.slice(0, 12) }}</code></p>
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
const description = '选择已运行的对照条件，再点击事件或播放时间轴，找出第一次偏离目标的位置。这里回放 Python 实际输出，不在浏览器执行训练、代码或外部工具。'
const challenge = computed(() => ({
  ask: `先预测：${record.value?.title.split(' · ')[0]}中，改变实验条件会让哪一步的证据先变化？`,
  answer: '展开终态对照，再逐步检查事件。结果相同也可能采用了不同的路径；反例配置出现失败是实验设计的一部分。适用条件与未覆盖能力见本章“实验边界”。',
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
