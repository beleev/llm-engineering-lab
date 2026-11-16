<!-- 对照实验: m24_agentic_rag。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="相关, 不代表足够回答 — 证据链缺一环就不能答"
    sub="- 模拟什么: 一个带检索的 agent (RAG) 回答「某产品的上传限制是多少」。知识库里只有 4 条小文档: 一条别名记录 (Comet 这个项目名其实指 Atlas v2)、Atlas v1 和 v2 各自的上传限制 (10 MB 和 20 MB)、Orion v2 的上传限制 (90 MB)。有的问题一次检索就能答 (直接问 Atlas v2); 有的要两跳 (先查 Comet 是什么, 再查 Atlas v2 的限制); 有的根本没有证据 (Nova 在库里不存在)。台子演示的是: 检索到「相关」的文档, 和凑齐「足以回答」的证据链, 是两回事。
      - 图上是什么: 一排按钮是知识库里的 4 条文档, 按钮上写着它这次有没有被引用进证据链, 绿框就是被引用了。点一条, 下面的面板显示它的原文、它描述的对象 (entity) 和版本 (version), 再下面列出这次实际发出的每一条检索查询。
      - 可以做什么: 「问题」下拉框换三个问题: Atlas v2 (一跳就够)、Comet (要两跳)、Nova (无证据)。「检索方式」有三种: 「无上下文的一跳」是最朴素的做法, 文档块里只有正文, 只取相似度最高的一条; 「加对象/版本的一跳」是建索引时给每块文档补上它的对象和版本, 检索后再按对象和版本筛选, 但只查一次; 「按证据继续检索」是 agentic 的做法, 第一跳查到别名就用别名指向的对象和版本再发一次查询。拖「查询预算」限制最多发几次查询。
      - 应该看到: 默认问 Comet、加对象/版本的一跳: 第一跳找到了别名记录 alias-comet, 知道 Comet 是 Atlas v2, 但没有第二跳, 右侧「回答」是「证据不足」。切到「按证据继续检索」: 第二条查询变成 atlas v2 upload limit, 引用 alias-comet 和 atlas-v2 两条证据, 回答 20 MB。把预算拖到 1, 又回到证据不足。问 Atlas v2 用「无上下文的一跳」也答不出: 三条写着 upload limit 的块长得几乎一样, 分不出哪条是 v2 的。问 Nova 在任何方式下都是证据不足, 不会拿别的产品的数字凑数。右侧「查询次数 / 预算」是实际发出的查询数和上限; 「引用证据」是最终答案依赖的文档条数, 两跳问题要有 2 条才算完整。"
    module="llm_agent/m24_agentic_rag"
    run="python -m llm_agent.m24_agentic_rag.demo"
    :challenge="{
      ask: '知道 Atlas v2 的限制是 20 MB, 却没有 Comet 的别名资料, 能回答 Comet 的问题吗?',
      answer: '不能。20 MB 这个数字本身是对的, 但「Comet 就是 Atlas v2」这一环没有证据, 答出来等于把一个数字安到了一个没核实过的对象上; 对象错了, 数字再对也没用。\n- 两跳问题要两段证据: 别名记录 alias-comet 证明 Comet 指向 Atlas v2; atlas-v2 这条证明 Atlas v2 的上传限制是 20 MB。demo 里的验证器会检查引用的来源是否同时覆盖对象、版本和答案, 缺任何一段都判不通过。\n- 问题本身已经写明对象和版本时 (直接问 Atlas v2), 一次查询就够, 不必强制多跳; 多跳是为了补齐证据链, 不是为了多查几次。\n- 预算用完还缺证据, 正确的输出是「证据不足」并说明缺哪一段, 而不是拿 Orion 或 v1 的数字凑一个答案。',
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
      <p class="lab-note">检索沿用 Python demo 的 TF-IDF 词面匹配, 第二跳由确定性规则决定 (查到别名就按别名指向的对象和版本再查一次)。没有神经 embedding, 也没有让模型自由规划查询。</p>
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
