<!-- 对照实验: full_learning_loop。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="持续改进 — 学会了新规则以后, 旧任务还会做吗"
    sub="- 模拟什么: 一个靠「外部记忆」持续改进的 agent: 它不改模型权重, 而是把每次任务验收后拿到的反馈 (比如「项目 A 用 TypeScript」) 记进一份记忆, 下次遇到类似任务先查记忆再回答。这里的任务都是同一种: 给一个人 (Alice 或 Bob) 和一个项目 (A 或 B), 回答「该用什么语言」。9 个任务按时间顺序分四个阶段:
  1. 学习期: L1, L2, 答完能拿到反馈并写进记忆。
  2. 迁移: T1 到 T3, 只读记忆, 考学到的东西能不能用到别的组合上。
  3. 规则变化: C1, C2, 项目 A 改用 Rust, 新规则只在 C1 答完之后才作为反馈出现。
  4. 旧能力保持: R1, R2, 回头考旧任务有没有被新规则弄坏。
- 图上是什么: 按钮和下面的面板:
  - 一排 9 个按钮: 就是 9 个任务, 从左到右按时间排。每个按钮上是任务编号、所属阶段和 agent 当时给出的回答, 绿边是回答和验收标准一致, 红边是答错。
  - 下面的面板: 点一个按钮, 面板显示这道题的回答、验收标准、回答时有没有查到记忆 (记忆激活)、验收后有没有写入记忆, 以及这道题的事件顺序: act 是回答, grade 是验收, update 是更新记忆; 顺序固定, 反馈永远在回答之后出现。
- 可以做什么: 「更新策略」下拉框换三种记忆更新方式:
  - 「静态规则」: 不记任何反馈, 永远按默认规则答 (Alice 用 Python, 其他人回答 unknown)。
  - 「直接追加记忆」: 把每条反馈追加到一个列表, 回答时取这个人的第一条记忆。
  - 「按范围更新版本」: 把反馈按「对象 + 范围」存成不同条目 (alice:A 和 alice:* 是两条), 查询时先找项目范围的, 没有再退回全局范围的。
- 应该看到: 两种策略的对照和右侧读数:
  - 「直接追加」: 在学习期拿到的第一条记忆 (项目 A 用 TypeScript) 之后被套到了 Alice 的所有任务上, 项目 B 和后来的 Rust 规则全答错, 保持阶段 0%。
  - 「按范围更新版本」: 四个阶段是 50%、100%、50%、100%。学习期和规则变化期各错一次, 都是因为新规则在那一题答完才出现, 下一题就对了; 迁移和保持都是 100%, 说明新规则只改了项目 A, 没有碰坏项目 B。
  - 右侧每一项: 该阶段答对的比例。
  - 「负迁移任务」: 学习期之后、用了记忆却答错的题数, 它包含规则刚变、还没来得及反馈的那一次, 所以版本化策略也会有 1。"
    module="llm_agent/full_learning_loop"
    run="python -m llm_agent.full_learning_loop.demo"
    :challenge="{
      ask: '项目 A 从 TypeScript 改成 Rust 的第一题 (C1), 为什么应该失败一次?',
      answer:
        '因为新规则只在这一题答完之后才作为反馈出现。\nC1 前后的顺序:\n1. C1 之前, 记忆里项目 A 的记录还是 TypeScript (学习期 L1 的反馈)。\n2. agent 查记忆答 TypeScript, 验收标准却已经变成 Rust, 判错。\n3. 验收之后反馈「项目 A 用 Rust」才写进记忆, 所以下一题 C2 答对了。\n如果 C1 就答出 Rust, 只能说明它在回答前偷看了未来的反馈, 实验就漏了答案。\n保持阶段:\n- 切到「按范围更新版本」: R1 问项目 B 仍答 Python, 因为 Rust 这条反馈只写进了 alice:A 这个范围, 没有覆盖全局。\n- 切到「直接追加记忆」: R1 和 R2 都错, 因为它永远取第一条记忆。\n先错一次再恢复, 加上旧任务不受影响, 这两件事合起来才能分别说明「适应了变化」和「保留了旧能力」。',
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
        <p>9 个固定任务的外部记忆实验, 没有训练模型权重, 改的只是 agent 查的那份记忆。</p>
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
