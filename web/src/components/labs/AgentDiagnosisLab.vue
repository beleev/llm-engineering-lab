<!-- 对照实验: m13_evals。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="哪一步先错了 — 同一句「已完成」, 五种轨迹"
    sub="- 模拟什么: Agent 故障诊断中不同评测判分口径的漏检差异。目标任务要求向存储写入 alpha 与 beta 两条笔记后回复「已完成」。本实验台回放 Python 教学 Agent 真实执行导出的五条典型轨迹, 其中前四条的最终自然语言回复完全相同, 均为「已完成」:
  - 真正完成: 依次合规写入 alpha 与 beta 后汇报完成。
  - 提前结束: 仅写入 alpha 便提前宣告完成。
  - 多写了别的对象: 除目标笔记外越界写入了范围外的 extra。
  - 工具执行失败: 写入工具抛出异常未落盘, 模型忽略错误照常回复完成。
  - 模型不可用: 基础设施故障导致未能产出任何回复。
- 图上是什么: 上方为轨迹步骤序列按钮, 下方为诊断详情面板:
  - 步骤按钮行: 按时间顺序列出当前轨迹的每一步动作, 其中 alpha / beta / extra 代表对笔记存储的写入调用, final 代表模型输出最终回复; 在「全过程」口径下, 首次偏离预期规范的步骤会标出红色边框。
  - 诊断详情面板: 展示模型最终回答、存储终态内容 (在「环境终态」或「全过程」口径下可见)、首次出错步骤编号及归因类别 (false_completion 虚假完成 / scope 范围越界 / tool 工具异常 / infrastructure 基础设施故障); 点击首错步骤时还会额外展示故障发生前的环境快照。
- 可以做什么: 通过顶部下拉框与按钮交互排查故障:
  - 「轨迹」下拉框: 在五条真实导出轨迹之间切换; 点击时间轴任意步骤可查看该步执行详情。
  - 「判分口径」切换三种评测粒度:
    - 「最终回答」: 仅核对模型最后一句文本是否为「已完成」。
    - 「环境终态」: 直接核查笔记存储最终是否恰好包含 alpha 与 beta 且无多余条目。
    - 「全过程」: 在核查环境终态的基础上, 逐步审计执行轨迹中是否存在越界写入与未处理的工具报错, 并定位首错位置。
- 应该看到: 对比典型故障轨迹在不同口径下的判分翻转与右侧读数:
  - 「提前结束」轨迹: 在「最终回答」口径下被误判为通过, 切换至「环境终态」后立即被识别为失败 (存储中仅有 alpha)。
  - 「多写了别的对象」轨迹: 在「环境终态」下因检出多余条目 extra 而判为失败; 切换至「全过程」口径后进一步精确定位首错发生在第 2 步 (写入 extra 的动作), 归因类别为 scope。
  - 右侧「当前口径下的判分」「工具调用次数」与「首错步骤」: 分别给出当前口径的通过结论、轨迹累计调用工具次数及全过程审计锁定的首个出错步骤; 修复系统时应针对首错步骤排查根因, 而非被后续连锁反应误导。"
    module="llm_agent/m13_evals"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: '「提前结束」与「多写了别的对象」两条轨迹最终都回复了「已完成」。如果评测器只检查这句最终回答, 能否区分哪条轨迹真正按要求完成了任务?',
      answer: '不能。两条故障轨迹与正常完成轨迹的最终回复文本完全一致, 在「最终回答」口径下均会被误判为通过。\n- 「环境终态」口径可拦截结果偏差: 直接读取外部笔记存储后可见, 「提前结束」轨迹仅写入了 alpha 而漏掉 beta; 「多写了别的对象」轨迹虽包含 alpha 与 beta, 但额外污染了范围外的 extra 条目。两条轨迹均被准确判为失败, 且终态差异直接揭示了漏做与越界两类不同问题。\n- 「全过程」口径可进一步锁定根因位置: 对轨迹逐步审计后, 「提前结束」的首错被定位于第 2 步 (本应继续写 beta 却提前输出完成声明, 归类为 false_completion); 「多写了别的对象」的首错同样定位于第 2 步 (发起越界写入 extra, 归类为 scope)。\n- 首错前快照的调试价值: 点击首错步骤即可获取故障发生前一刻的上下文与环境快照; 将该状态冻结为单步回归用例专门验证「下一步动作选择」, 远比每次从头重跑整条多轮轨迹高效。',
    }"
  >
    <template #controls>
      <label
        >轨迹
        <select v-model="chosen">
          <option v-for="(c, i) in cases" :key="c.name" :value="i">{{ labels[c.name] }}</option>
        </select></label
      >
      <label
        >判分口径
        <select v-model="grader">
          <option value="text">最终回答</option>
          <option value="state">环境终态</option>
          <option value="trace">全过程</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="动作时间轴">
      <button
        v-for="(action, i) in row.actions"
        :key="i"
        type="button"
        :class="{ active: step === i, bad: grader === 'trace' && row.failure?.step === i + 1 }"
        @click="step = i"
      >
        {{ i + 1 }} · {{ action }}
      </button>
    </div>
    <div class="detail" aria-live="polite">
      <p>最终回答: {{ row.final || '模型不可用, 没有回答' }}</p>
      <p v-if="grader !== 'text'">笔记存储最后剩下: {{ Object.keys(row.state).join('、') || '空' }}</p>
      <p v-if="grader === 'trace'">首错: {{ row.failure ? `第 ${row.failure.step} 步 · 类别 ${row.failure.category}` : '无' }}</p>
      <p v-if="grader === 'trace' && row.failure && step + 1 === row.failure.step">
        首错前的环境快照 (出错那一步开始之前笔记存储里有什么): {{ row.before_error }}
      </p>
      <p v-else>当前选中第 {{ Math.min(step + 1, row.actions.length) }} 步。红框标的是首错, 不代表后面没有别的错误。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>当前口径下的判分</span><b :class="passed ? 'good' : 'bad'">{{ passed ? '通过' : '失败' }}</b>
      </div>
      <div class="kv">
        <span>工具调用次数</span><b>{{ row.tool_calls }}</b>
      </div>
      <div class="kv">
        <span>首错步骤 (全过程检查找到的)</span><b>{{ row.failure?.step || '—' }}</b>
      </div>
      <p class="lab-note">这五条轨迹均由 Python 教学 Agent 在固定夹具题目上实际运行导出; 面板中的通过与失败结论仅针对该单道受控任务, 不代表线上通用模型的统计成功率。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import cases from '@/data/agent-diagnostics.json'
const chosen = ref(1),
  grader = ref('text'),
  step = ref(0)
const labels = {
  correct: '真正完成',
  premature: '提前结束',
  'extra-write': '多写了别的对象',
  'tool-failure': '工具执行失败',
  outage: '模型不可用',
}
const row = computed(() => cases[chosen.value])
const passed = computed(() =>
  grader.value === 'text'
    ? row.value.final === '已完成'
    : grader.value === 'state'
      ? Object.keys(row.value.state).sort().join(',') === 'alpha,beta'
      : row.value.passed,
)
</script>
