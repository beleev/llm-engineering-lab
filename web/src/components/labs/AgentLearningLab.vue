<!-- 对照实验: full_learning_loop。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="持续改进 — 学会了新规则以后, 旧任务还会做吗"
    sub="- 模拟什么: 依赖「外部记忆」持续演进的 agent 机制。系统不更新模型权重, 而是将每次任务验收后获得的环境反馈 (如「项目 A 用 TypeScript」) 写入外部记忆库, 后续遇到同类任务时先检索记忆再作答。实验包含 9 个同构查询任务: 给定人员 (Alice 或 Bob) 与项目 (A 或 B), 回答应使用的编程语言。9 个任务按时间顺序划分为四个阶段:
  1. 学习期: L1 与 L2, 作答后接收验收反馈并写入记忆库。
  2. 迁移期: T1 至 T3, 仅读取记忆库而不写入, 检验已学规则能否泛化至不同查询组合。
  3. 规则变化期: C1 与 C2, 项目 A 的规范变更为 Rust, 但新规范仅在 C1 作答完毕后才通过验收反馈给出。
  4. 旧能力保持期: R1 与 R2, 重新测试既有任务, 检验局部新规则是否破坏了原有记忆。
- 图上是什么: 上方为任务序列按钮, 下方为单题追踪面板:
  - 一排 9 个按钮: 按时间顺序对应 9 个评测任务。每个按钮显示任务编号、所属阶段及 agent 当时给出的回答, 绿框表示回答符合验收标准, 红框表示回答错误。
  - 下方面板: 点击任意任务按钮后, 展示该题的模型回答、验收标准、作答前是否命中记忆 (记忆激活)、验收后是否写入记忆, 以及严格固定的事件时序: act (作答)、grade (验收) 与 update (更新记忆), 确保环境反馈始终在作答之后到达。
- 可以做什么: 通过「更新策略」下拉框切换三种记忆读写机制:
  - 「静态规则」: 忽略所有反馈, 始终按初始默认规则作答 (Alice 默认使用 Python, 其余人员返回 unknown)。
  - 「直接追加记忆」: 将每次反馈顺序追加至列表末尾, 查询时直接返回该人员名下的首条记忆记录。
  - 「按范围更新版本」: 按「实体 + 作用范围」分立存储条目 (如 alice:A 与 alice:* 互不覆盖), 检索时优先匹配具体项目范围, 未命中时再回退至全局范围。
- 应该看到: 不同记忆策略的准确率对照与右侧读数:
  - 「直接追加记忆」: 学习期写入的首条记录 (项目 A 用 TypeScript) 被无差别套用于 Alice 的后续全部查询, 导致项目 B 与变更后的 Rust 规则全部答错, 保持阶段准确率为 0%。
  - 「按范围更新版本」: 四个阶段的准确率依次为 50%、100%、50%、100%。学习期与规则变化期各出现一次预期内的首题失误, 原因均是新规则在该题作答后才下发, 紧随其后的下一题即恢复正确; 迁移期与保持期均达到 100%, 证明局部更新仅作用于项目 A 而未污染项目 B。
  - 右侧各阶段读数: 对应阶段的答题准确率。
  - 「负迁移任务」: 统计学习期结束后调用了记忆却答错的题数; 由于包含规则刚发生变更、新反馈尚未写入的 C1, 版本化策略在此项上也会计为 1。"
    module="llm_agent/full_learning_loop"
    run="python -m llm_agent.full_learning_loop.demo"
    :challenge="{
      ask: '当项目 A 的语言规范从 TypeScript 变更为 Rust 时, 规则变化期的第一题 (C1) 为什么应当出现一次失败?',
      answer: '因为变更后的新规则仅在 C1 作答完毕并完成验收之后, 才会作为环境反馈提供给系统。\nC1 前后的严格事件时序如下:\n1. 在 C1 作答前, 记忆库中关于项目 A 的有效记录仍为学习期 L1 写入的 TypeScript。\n2. agent 检索记忆后按既有记录回答 TypeScript, 而此时外部验收标准已切换为 Rust, 因此被判定为答错。\n3. 验收结束后, 新反馈「项目 A 用 Rust」才写入记忆库, 从而使紧随其后的 C2 恢复正确作答。\n若系统在 C1 当题即答出 Rust, 则意味着评测管线在作答前泄露了未来时刻的反馈标签 (数据穿越)。\n在随后的旧能力保持阶段:\n- 切换至「按范围更新版本」: R1 查询项目 B 时仍准确返回 Python, 因为 Rust 更新仅写入了 alice:A 这一局部范围, 未覆盖 alice:* 全局默认值。\n- 切换至「直接追加记忆」: R1 与 R2 均回答错误, 因为查询逻辑始终机械读取列表首条记录。\n规则变更首题失误后立即恢复, 配合既有范围任务不受干扰, 两个现象结合才能完整验证系统同时具备「适应环境变化」与「保持旧有能力」两项特性。',
    }"
  >
    <template #controls>
      <label
        >更新策略
        <select v-model="mode">
          <option value="static">静态规则</option>
          <option value="append">直接追加记忆</option>
          <option value="versioned">按范围更新版本</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="跨任务时间轴">
      <button
        v-for="(row, i) in result.rows"
        :key="row.id"
        type="button"
        :class="{ active: selected === i, good: row.passed, bad: !row.passed }"
        @click="selected = i"
      >
        {{ row.id }} · {{ names[row.phase] }}<br />{{ row.action }}
      </button>
    </div>
    <div class="detail">
      <p>{{ current.id }}: agent 先回答 {{ current.action }}, 再按验收标准 {{ current.expected }} 判对错。</p>
      <p>记忆激活 (回答时查到了记忆): {{ current.activated ? '是' : '否' }}; 验收后更新 (反馈写进了记忆): {{ current.updated ? '是' : '否' }}。</p>
      <p>本题事件顺序: {{ result.order.filter((e) => e.startsWith(current.id + ':')).join(' → ') }}</p>
      <p>迁移与保持阶段只读记忆, 不写入。记忆更新只在学习期和明确的规则变化事件之后开放, 这样测试题的答案不会边测边泄漏进记忆。</p>
    </div>
    <template #stats>
      <div v-for="(score, phase) in result.scores" :key="phase" class="kv">
        <span>{{ names[phase] }}阶段通过率</span
        ><b :class="score === 1 ? 'good' : score === 0 ? 'bad' : ''">{{ (score * 100).toFixed(0) }}%</b>
      </div>
      <div class="lab-note">
        <p>本实验基于 9 个固定评测任务演示外部记忆演化机制, 全程不更新模型权重, 仅改变 agent 检索的外部记忆状态。</p>
        <p>负迁移任务 {{ result.negative_transfer }} 个: 指学习期之后、回答时用了记忆却答错的题。这个数包含规则刚变、还没来得及反馈的那一次 (C1), 所以「按范围更新版本」也会有 1; 「直接追加记忆」有 5, 是第一条记忆被套到了所有任务上。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import { learning } from '@/utils/agent-reliability.js'
const mode = ref('append'),
  selected = ref(5)
const names = { learning: '学习', transfer: '迁移', change: '规则变化', retention: '旧能力保持' }
const result = computed(() => learning(mode.value))
const current = computed(() => result.value.rows[selected.value])
</script>
