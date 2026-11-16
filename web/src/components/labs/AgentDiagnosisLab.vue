<!-- 对照实验: m13_evals。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="哪一步先错了 — 同一句「已完成」, 五种轨迹"
    sub="- 模拟什么: 评测一个 agent 时, 只看它最后那句话够不够。这里的任务是写两条笔记 alpha 和 beta, 然后回答「已完成」。demo 用 Python 真实跑了五条轨迹: 真正完成; 提前结束 (只写了 alpha 就说完成); 多写了别的对象 (多写了一条范围外的笔记 extra); 工具执行失败 (写入报错, 模型照样说完成); 模型不可用 (根本没有回答)。前四条的最终回答一字不差, 都是「已完成」。
      - 图上是什么: 一排按钮是这条轨迹的每一步动作: alpha / beta / extra 是对笔记存储的一次写入, final 是模型给出最终回答。选「全过程」口径时, 首错那一步会标红框。下面的面板显示最终回答、笔记存储最后剩下什么 (选「环境终态」或「全过程」时显示)、首错在第几步和属于哪一类 (false_completion 假完成 / scope 越界 / tool 工具失败 / infrastructure 基础设施); 点到首错那一步时, 还会显示出错之前环境是什么样。
      - 可以做什么: 「轨迹」下拉框切换五条轨迹。「判分口径」切换三种判法: 「最终回答」只看最后那句话是不是「已完成」; 「环境终态」去笔记存储里数 alpha 和 beta 是否都在、有没有多余的; 「全过程」除了终态, 还逐步检查有没有越界写入和工具报错。点任一步, 看那一刻的说明。
      - 应该看到: 「提前结束」这条在「最终回答」口径下是通过, 切到「环境终态」立刻变成失败 (笔记里只有 alpha)。「多写了别的对象」在「环境终态」下 alpha 和 beta 都在, 但多出一个 extra, 判失败; 「全过程」口径还能告诉你首错在第 2 步 (写 extra 的那一步), 类别是 scope。右侧「当前口径下的判分」是这条轨迹过没过; 「工具调用次数」是这条轨迹一共调了几次工具; 「首错步骤」是全过程检查发现的第一个错在第几步, 之后的步骤可能还有错, 但修复应该从这一步开始。"
    module="llm_agent/m13_evals"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: '提前结束和越界写入都回答「已完成」。只看这句话能分出谁做对了吗?',
      answer:
        '不能, 两条轨迹的最终回答一字不差, 在「最终回答」口径下都通过。\n- 「环境终态」口径去读笔记存储: 提前结束的那条只有 alpha, 缺 beta; 越界写入的那条 alpha、beta 都在, 但多了一条范围外的 extra。两条都判失败, 而且失败原因不同。\n- 「全过程」口径再往前追: 提前结束的首错是第 2 步 (该写 beta 的时候直接说完成了, 类别 false_completion); 越界写入的首错也是第 2 步 (写了 extra, 类别 scope)。\n- 首错前的环境快照 (点到那一步能看到) 告诉你该从哪里改: 把那一步之前的消息和环境冻结下来, 单独检查「下一步该做什么」, 比整条轨迹重跑便宜得多。',
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
      <p class="lab-note">这五条轨迹是 Python 教学 Agent 实际执行后导出的记录, 题目是固定的。通过与否说的是这一道题, 不是线上模型的成功率。</p>
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
