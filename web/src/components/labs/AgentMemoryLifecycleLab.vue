<!-- 对照实验: m20_memory_lifecycle。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="记忆更新 — 记住了新偏好, 为什么别的项目也跟着错"
    sub="- 模拟什么: agent 的长期记忆在收到新事实之后该怎么更新。证据只有两条: 「Alice 通常用 Python」(全局范围, 对她所有项目都成立) 和「Alice 只在项目 A 用 TypeScript」(只对项目 A 成立的例外)。之后 agent 要回答三个查询: Alice 在项目 A 用什么, Alice 在项目 B 用什么, Bob 在项目 A 用什么 (Bob 没有任何证据, 正确答案是 unknown, 不能编)。
- 图上是什么: 三个按钮就是三个查询, 上面一行是查的对象和项目, 下面一行是当前策略给出的回答; 绿边是回答和正确结果一致, 红边是答错。点一个按钮, 下面的面板列出两条原始证据、当前查询和正确结果。
- 可以做什么: 「更新策略」下拉框换三种记忆的存法和查法。「追加后取首条」: 每条新事实追加到列表末尾, 查询时取这个人的第一条, 新事实永远用不上。「最新覆盖全局」: 新事实直接覆盖这个人的记录, 不管它只对哪个项目成立, 项目 A 的例外被当成了全局。「按范围审核」: 每条事实带着对象 (entity) 和范围 (scope) 分开存, 查询时先找项目范围的, 没有再退回全局。
- 应该看到: 前两种策略各答对 2 个, 错的方向相反: 「追加后取首条」把项目 A 答成 Python (新例外没生效), 「最新覆盖全局」把项目 B 答成 TypeScript (例外污染了全局)。「按范围审核」3 个全对。右侧「正确查询」是答对的个数; 「跨范围错误」只看项目 B 这一问, 为 1 时说明只对项目 A 成立的事实被用到了项目 B 上; 「无证据时编造」只看 Bob 这一问, 三种策略都老实答 unknown, 所以都是 0, 它提醒的是: 没证据就该说不知道, 这一点不能随更新策略变。"
    module="llm_agent/m20_memory_lifecycle"
    run="python -m llm_agent.m20_memory_lifecycle.demo"
    :challenge="{
      ask: '把项目 A 的新偏好 (TypeScript) 直接覆盖到 Alice 的全局记录, 项目 B 会怎样?',
      answer:
        'B 会错误地变成 TypeScript。\n一步步看: 「最新覆盖全局」只记下「Alice 用 TypeScript」, 丢掉了「只在项目 A」这个范围; 查项目 B 时找到的就是这条, 于是答 TypeScript, 可是 B 从来没改过, 正确答案仍是全局的 Python。\n切到「按范围审核」: 两条证据分别存成 alice/* = Python 和 alice/A = TypeScript。查项目 A 先命中 A 这个范围的例外, 答 TypeScript; 查项目 B 没有 B 范围的记录, 退回全局, 答 Python。新事实只替换了它能证明的那一部分。\nBob 在三种策略下都没有证据, 必须保留 unknown, 不能拿 Alice 的记忆去猜。',
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
      <p class="lab-note">这里只测记忆的更新与检索规则, 不测模型会不会服从记忆。完整 demo 还验证审核拒绝 (没有来源、丢了范围、改了原值的候选都不能发布)、回滚和索引重建。</p>
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
