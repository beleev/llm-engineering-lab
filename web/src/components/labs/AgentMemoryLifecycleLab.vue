<!-- 对照实验: m20_memory_lifecycle。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="记忆更新 — 记住了新偏好, 为什么别的项目也跟着错"
    sub="- 模拟什么: agent 长期记忆在接收新事实后的生命周期更新与作用域隔离。输入证据仅包含两条: 「Alice 通常用 Python」(全局默认范围, 适用于其名下所有项目) 与「Alice 只在项目 A 用 TypeScript」(仅对项目 A 生效的局部例外)。系统随后需回答三个测试查询: Alice 在项目 A 使用什么语言、Alice 在项目 B 使用什么语言, 以及 Bob 在项目 A 使用什么语言 (记忆库无任何关于 Bob 的证据, 标准答案应为 unknown, 严禁凭空捏造)。
- 图上是什么: 上方为三个查询按钮, 下方为证据核对面板:
  - 三个按钮: 分别对应三个测试查询, 首行显示目标实体与项目名称, 次行显示当前记忆策略给出的回答; 绿框表示回答与标准结果一致, 红框表示回答错误。
  - 下方面板: 点击任意按钮后, 对照列出两条原始输入证据、当前查询条件及预期标准答案。
- 可以做什么: 通过「更新策略」下拉框切换三种记忆存储与检索规则:
  - 「追加后取首条」: 将每条新事实顺序追加至列表末尾, 检索时始终返回该实体名下的首条记录, 导致后续局部更新无法生效。
  - 「最新覆盖全局」: 用最新接收到的事实直接覆盖该实体的唯一记录, 忽略其限定作用域, 使项目 A 的局部例外覆盖全局设定。
  - 「按范围审核」: 将每条事实按实体 (entity) 与作用范围 (scope) 独立建档存储, 检索时优先匹配具体项目范围, 未命中时再回退至全局范围。
- 应该看到: 前两种粗放策略各自仅答对 2 题且失误方向相反, 而「按范围审核」3 题全部答对。具体对照与右侧读数如下:
  - 「追加后取首条」: 将项目 A 误答为 Python (局部新规则被首条旧记录遮蔽)。
  - 「最新覆盖全局」: 将项目 B 误答为 TypeScript (局部例外污染了全局默认记忆)。
  - 右侧「正确查询」: 统计当前策略答对的查询总数。
  - 「跨范围错误」: 专门监测项目 B 查询结果, 读数为 1 表示仅对项目 A 成立的局部事实被错误外推至项目 B。
  - 「无证据时编造」: 专门监测无证据实体 Bob 的查询结果, 三种策略均规范返回 unknown, 因此读数保持为 0; 缺乏证据时必须拒答, 该底线不应随记忆更新策略而改变。"
    module="llm_agent/m20_memory_lifecycle"
    run="python -m llm_agent.m20_memory_lifecycle.demo"
    :challenge="{
      ask: '若将仅针对项目 A 的新偏好 (TypeScript) 直接覆盖写入 Alice 的全局记忆记录, 查询项目 B 时会产生什么后果?',
      answer: '查询项目 B 时会错误地返回 TypeScript。\n「最新覆盖全局」策略在写入时仅保留了「Alice 使用 TypeScript」这一结论, 丢失了「仅限项目 A」的作用范围约束; 后续查询项目 B 时检索到该全局记录便直接返回 TypeScript, 而项目 B 实际上从未发生变更, 正确答案仍应为全局默认的 Python。\n切换至「按范围审核」策略后, 两条证据被分别存储为全局部件 alice/* (值为 Python) 与项目部件 alice/A (值为 TypeScript):\n- 查询项目 A 时: 优先命中作用域更具体的 A 范围条目, 返回 TypeScript。\n- 查询项目 B 时: 未查到 B 范围专属条目, 自动回退至 alice/* 全局记录, 返回 Python。\n由此保证每条新事实仅更新其证据所能支撑的作用范围。\n此外, 记忆库在三种策略下均无 Bob 的相关证据, 系统必须保持返回 unknown, 绝不能跨实体借用 Alice 的偏好进行推测。',
    }"
  >
    <template #controls>
      <label
        >更新策略
        <select v-model="mode">
          <option value="append-first">追加后取首条</option>
          <option value="overwrite">最新覆盖全局</option>
          <option value="versioned">按范围审核</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="选择查询对象">
      <button
        v-for="(q, i) in queries"
        :key="q"
        type="button"
        :class="{ active: selected === i, good: answers[i] === expected[i], bad: answers[i] !== expected[i] }"
        @click="selected = i"
      >
        {{ q }}<br />{{ answers[i] }}
      </button>
    </div>
    <div class="detail">
      <p>原始证据 1: Alice / 全局 (所有项目) / Python</p>
      <p>原始证据 2: Alice / 项目 A / TypeScript</p>
      <p>当前查询: {{ queries[selected] }}; 正确结果: {{ expected[selected] }}</p>
      <p>「按范围审核」给每条记忆保留 entity (对象)、scope (范围) 和 source_id (它来自哪条证据); 撤销项目 A 的例外之后, 查询会自动回退到全局的 Python。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>正确查询 (3 问答对几个)</span><b :class="correct === 3 ? 'good' : 'bad'">{{ correct }} / 3</b>
      </div>
      <div class="kv">
        <span>跨范围错误 (项目 A 的例外被用到了项目 B)</span
        ><b :class="answers[1] === expected[1] ? 'good' : 'bad'">{{ Number(answers[1] !== expected[1]) }}</b>
      </div>
      <div class="kv">
        <span>无证据时编造 (Bob 没有答 unknown)</span
        ><b :class="answers[2] === 'unknown' ? 'good' : 'bad'">{{ Number(answers[2] !== 'unknown') }}</b>
      </div>
      <p class="lab-note">本实验仅评测外部记忆的更新与检索逻辑, 不测试生成模型对检索结果的服从度。完整版 demo 同时覆盖写入前审核拦截 (缺失来源引用、丢失作用范围或篡改原始取值的候选条目一律禁止发布)、版本回滚与索引重建。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import { memoryAnswers } from '@/utils/agent-reliability.js'
const mode = ref('overwrite'),
  selected = ref(1)
const queries = ['Alice / 项目 A', 'Alice / 项目 B', 'Bob / 项目 A'],
  expected = ['TypeScript', 'Python', 'unknown']
const answers = computed(() => memoryAnswers(mode.value))
const correct = computed(() => answers.value.filter((a, i) => a === expected[i]).length)
</script>
