<!-- 对照实验: m24_agentic_rag。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="相关, 不代表足够回答 — 证据链缺一环就不能答"
    sub="- 模拟什么: 检索增强 Agent (RAG) 回答「指定产品的上传大小限制是多少」时的证据链完备性校验。外部知识库包含 4 条微型文档: 一条项目别名映射记录 (说明代号 Comet 指向产品 Atlas v2)、Atlas v1 与 v2 各自的上传限额 (分别为 10 MB 与 20 MB), 以及另一产品 Orion v2 的上传限额 (90 MB)。实验旨在区分「召回字面相关片段」与「凑齐足以支撑答案的闭环证据链」两件事, 涵盖三类典型问题:
  - 单跳可直接作答: 直接询问目标实体与版本 Atlas v2。
  - 需两跳桥接作答: 先查明别名 Comet 对应的真实产品与版本, 再检索 Atlas v2 的上传限额。
  - 库内无证据支撑: 询问知识库中不存在的未知产品 Nova。
- 图上是什么: 上方为知识库文档卡片按钮, 下方为文档详情与检索轨迹面板:
  - 4 个文档卡片按钮: 对应知识库中的全部 4 条记录, 按钮状态标示该文档在本次作答中是否被纳入引用证据链, 被引用时高亮绿色边框。
  - 文档详情与查询列表面板: 点击任意文档卡片可查看其正文原文及绑定的对象 (entity) 与版本 (version) 元数据; 面板下方按顺序列出本次运行实际发出的每一条检索查询语句。
- 可以做什么: 组合以下四项控制参数进行对照实验:
  - 「问题」下拉框: 在 Atlas v2 (单跳可达)、Comet (需两跳桥接) 与 Nova (无证据) 三道测试题之间切换。
  - 「检索方式」选择「无上下文的一跳」: 基线做法, 仅对不含元数据的正文建索引, 单次检索只取相似度最高的一条记录。
  - 选择「加对象/版本的一跳」: 建索引时为每块文档前置拼接其所属对象与版本, 召回后按实体与版本精确过滤, 但最多只发起一跳检索。
  - 选择「按证据继续检索」: agentic 多跳策略, 首跳命中别名记录后自动用解析出的目标实体与版本构造并发起第二跳检索。
  - 拖动「查询预算」滑杆: 限定单道问题允许发起的检索调用次数上限。
- 应该看到: 观察各组合下的作答表现与右侧读数:
  - 默认询问 Comet 且选用「加对象/版本的一跳」: 首跳虽成功命中 alias-comet 并获知 Comet 对应 Atlas v2, 但因未发起第二跳查询数值, 右侧「回答」如实返回「证据不足」。
  - 切换至「按证据继续检索」: 系统自动追加第二条查询 atlas v2 upload limit, 同时引用 alias-comet 与 atlas-v2 两条证据, 准确输出 20 MB。
  - 将查询预算收紧为 1: 因第二跳被预算截断, 回答再次退回「证据不足」。
  - 询问 Atlas v2 但选用「无上下文的一跳」: 因三条包含 upload limit 的正文词面高度雷同且缺少版本前缀区分, 系统无法锁定 v2 限额块而拒绝作答。
  - 询问 Nova: 在任意检索策略下均返回「证据不足」, 绝不挪用其他产品的限额数值拼凑答案。
  - 右侧「查询次数 / 预算」与「引用证据」: 分别显示实际消耗的检索次数与支撑最终答案的文档条数, 两跳别名问题必须完整引用 2 条证据方可通过核验。"
    module="llm_agent/m24_agentic_rag"
    run="python -m llm_agent.m24_agentic_rag.demo"
    :challenge="{
      ask: '假设知识库中已有 Atlas v2 上传限制为 20 MB 的记录, 但缺失说明 Comet 对应 Atlas v2 的别名文档, 系统能否直接回答关于 Comet 上传限制的提问?',
      answer: '不能。尽管 20 MB 这一数值对 Atlas v2 本身完全正确, 但缺失了证明「Comet 即 Atlas v2」的桥接证据, 强行输出该数值等价于将未核实归属的数据安在未知对象上。\n- 多跳问答须具备完整引用闭环: demo 中的独立验证器会严格检查所引用的来源集合是否同时覆盖别名映射、目标实体版本与限额数值, 缺少任意一环均判为证据不成立:\n  - alias-comet: 提供实体桥接依据, 证明项目别名 Comet 指向产品 Atlas v2。\n  - atlas-v2: 提供目标属性依据, 证明 Atlas v2 的上传大小限制为 20 MB。\n- 按需检索而非固定多跳: 当用户直接指明实体与版本 (如直接询问 Atlas v2) 时, 单跳检索即可构成完整证据链, 无需冗余发起第二跳; 仅当首跳证据存在逻辑缺口时才继续追查。\n- 缺证据时的安全退守: 一旦查询预算耗尽或库中缺失关键桥接文档, 唯一合规的输出是返回「证据不足」并指明缺失环节, 绝不能挪用 Orion 或 v1 的数值充当答案。',
    }"
  >
    <template #controls>
      <label
        >问题
        <select v-model="question">
          <option :value="0">Atlas v2 的上传限制是多少? (一跳就够)</option>
          <option :value="1">Comet 的上传限制是多少? (要两跳)</option>
          <option :value="2">Nova 的上传限制是多少? (没有证据)</option>
        </select></label
      >
      <label
        >检索方式
        <select v-model="mode">
          <option value="plain">无上下文的一跳</option>
          <option value="contextual">加对象/版本的一跳</option>
          <option value="agentic">按证据继续检索</option>
        </select></label
      >
      <LabSlider v-model="budget" label="查询预算" :min="0" :max="3" />
    </template>
    <div class="trace" aria-label="证据文档">
      <button
        v-for="(doc, i) in ragDocs"
        :key="doc.id"
        type="button"
        :class="{ active: selected === i, good: result.sources.includes(doc.id) }"
        @click="selected = i"
      >
        {{ doc.id }}<br />{{ result.sources.includes(doc.id) ? '已引用' : '未引用' }}
      </button>
    </div>
    <div class="detail">
      <p>{{ ragDocs[selected].body }}</p>
      <p>这条文档描述的对象: {{ ragDocs[selected].entity }}; 版本: {{ ragDocs[selected].version }}</p>
      <p v-for="(query, i) in result.queries" :key="i">实际发出的查询 {{ i + 1 }}: {{ query }}</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>回答 (证据链完整才给数字)</span><b>{{ result.answer === null ? '证据不足' : result.answer + ' MB' }}</b>
      </div>
      <div class="kv">
        <span>查询次数 / 预算</span><b>{{ result.queries.length }} / {{ budget }}</b>
      </div>
      <div class="kv">
        <span>引用证据 (答案依赖几条文档)</span><b>{{ result.sources.length }}</b>
      </div>
      <p class="lab-note">本实验台的底层检索沿用 Python demo 的 TF-IDF 词面匹配, 第二跳改写逻辑由确定性规则驱动 (命中别名记录后自动按目标实体与版本发起二次查询), 未引入稠密神经 embedding 或语言模型自主查询规划。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { evidenceChain, ragDocs } from '@/utils/agent-reliability.js'
const question = ref(1),
  mode = ref('contextual'),
  budget = ref(2),
  selected = ref(0)
const result = computed(() => evidenceChain(question.value, mode.value, budget.value))
</script>
