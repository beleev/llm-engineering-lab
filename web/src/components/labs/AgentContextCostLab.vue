<!-- 对照实验: m23_context_cache。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="压短了, 就一定省钱吗 — 摘要、缓存和动态状态一起算账"
    sub="- 模拟什么: 多轮 Agent 会话中的上下文组织与前缀缓存计费账单, 同时耦合三项机制:
  - 上下文拼接结构: 每轮发送给模型的输入由四部分拼成, 包括固定策略说明 policy (200 tok)、逐轮累积的对话历史 (每轮新增用户消息 20 tok 与工具返回结果 200 tok)、每轮更新的动态状态 (记忆与进度, 40 tok) 以及本轮用户输入 (20 tok)。
  - 前缀缓存计费: 与上一轮请求从头完全一致的连续前缀按「读缓存」计价 (每 tok 0.1), 自首个发生变动的文本块起及其后全部内容按「写缓存」计价 (每 tok 1.25)。
  - 历史摘要压缩: 当历史块数达到阈值时, 额外调用一次模型将全部历史压缩为单条摘要块 (50 tok); 该次摘要调用本身需读取整段历史并附加固定 50 的输出费用。
- 图上是什么: 上方为轮次时间轴按钮, 下方为单轮明细与约束保留面板:
  - 轮次按钮: 对应会话的每一轮, 按钮数值显示该轮实际送入模型的上下文总长度。
  - 单轮明细与进度条: 点击任一轮次, 面板展示该轮的输入构成与单轮费用, 进度条反映该轮上下文长度相对于全会话峰值的比例。
  - 成本对照行: 并排对比「相同参数但不开启摘要」的基线总费用与当前策略总费用。
  - 底部审计与约束行: 显示当前摘要保留了多少条关键限定条件, 以及底层未被摘要覆盖删除的原始审计块总数。
- 可以做什么: 调节以下四项控件观察账单与信息保留的变化:
  - 「动态状态位置」: 选择「最前部」会将每轮变化的动态状态块置于上下文最开头; 选择「历史之后」则将其移至稳定的 policy 与历史块之后。
  - 拖动「会话轮数」: 改变会话持续的总轮次数。
  - 「摘要阈值」: 设定历史累积满多少块时触发摘要压缩, 选择「不摘要」则全程保留完整历史。
  - 取消勾选「摘要保留范围与禁止事项」: 模拟有损摘要仅保留编程语言偏好 TypeScript, 丢弃项目作用范围与禁止删除两条安全边界。
- 应该看到: 所有读数均由本地费用模型按相对计价单位算出, 而非真实云端账单:
  - 默认配置 (6 轮、历史之后、阈值 4): 模拟总费用为 3202.5 (其中摘要开销占 1327.5), 低于不摘要时的 4925, 峰值上下文从 1360 降至 530。
  - 将动态状态移至「最前部」: 每轮首块内容变动导致后续前缀缓存全部失效, 不摘要时的 6 轮总费用升至 6075。
  - 将会话轮数设为 2、摘要阈值设为 2: 不摘要总费用为 695, 开启摘要反而升至 840, 短会话无法摊平摘要调用的固定开销。
  - 取消勾选保留约束: 账单费用分毫未变, 但右侧「约束保留」由 3/3 降至 1/3。
  - 右侧「模拟总费用」: 低于不摘要基线时标绿, 高于基线时标红; 「其中摘要费用」单列摘要调用成本, 「峰值上下文」显示单次请求的最大 tok 数。"
    module="llm_agent/m23_context_cache"
    run="python -m llm_agent.m23_context_cache.demo"
    :challenge="{
      ask: '将会话轮数调为 2、摘要阈值调为 2。把历史上下文压缩变短之后, 为什么会话总费用反而比不压缩更高?',
      answer: '因为生成摘要本身需要额外支付一次模型调用费用, 且压缩后的摘要块会改写后续轮次的前缀并触发缓存重写。\n- 摘要调用的直接成本: 第 2 轮前触发摘要时, 虽有首块 policy 可命中读缓存, 但两块新增历史与摘要指令均需按写缓存计价, 叠加固定 50 的输出费用后, 仅摘要调用本身便消耗 357.5。\n- 前缀缓存失效成本: 原历史序列被替换为全新的 summary 块后, 与此前缓存中的历史前缀不再匹配, 下一轮请求必须按写缓存单价重新写入该摘要块。\n- 收益与开销权衡: 压缩节省的仅是后续轮次少读的历史 token; 在 2 轮短会话中历史尚未充分累积, 节省额不足以抵消摘要开销, 导致总费用由不摘要的 695 升至 840; 当会话延长至 6 轮时, 累积节省额超过摘要成本, 总费用才由 4925 降至 3202.5。\n- 语义保真独立于账单: 完整实验同时表明, 保留项目作用范围与「禁止删除」约束的摘要同丢失这些限定的摘要费用完全相等, 评估压缩策略不能只看账单数字。',
    }"
  >
    <template #controls>
      <label
        >动态状态位置
        <select v-model="placement">
          <option value="front">最前部</option>
          <option value="tail">历史之后</option>
        </select></label
      >
      <LabSlider v-model="turns" label="会话轮数" :min="2" :max="10" />
      <label
        >摘要阈值
        <select v-model="threshold">
          <option :value="0">不摘要</option>
          <option :value="2">2 个历史块</option>
          <option :value="4">4 个历史块</option>
          <option :value="6">6 个历史块</option>
        </select></label
      >
      <label><input v-model="keepScope" type="checkbox" />摘要保留范围与禁止事项</label>
    </template>
    <div class="trace" aria-label="每轮请求">
      <button
        v-for="row in result.rows"
        :key="row.turn"
        type="button"
        :class="{ active: selected === row.turn - 1 }"
        @click="selected = row.turn - 1"
      >
        第 {{ row.turn }} 轮<br />{{ row.input }} tok
      </button>
    </div>
    <div class="detail">
      <p>这一轮送进模型 {{ frame.input }} tok, 这一次请求的费用 {{ frame.cost.toFixed(1) }}。</p>
      <div class="meter"><span :style="{ width: (frame.input / result.peak) * 100 + '%' }" /></div>
      <p>同样的状态位置但不摘要, 总价 {{ baseline.cost.toFixed(1) }}; 当前设置的总价 {{ result.cost.toFixed(1) }}。</p>
      <p>任务里有三条约束: 项目范围 A、语言 TypeScript、禁止删除。摘要保住了 {{ result.retained }} 条; 原始审计记录仍有 {{ result.audit_blocks }} 块, 摘要不删它们。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>模拟总费用 (比不摘要便宜就是绿)</span
        ><b :class="result.cost < baseline.cost ? 'good' : result.cost > baseline.cost ? 'bad' : ''">{{
          result.cost.toFixed(1)
        }}</b>
      </div>
      <div class="kv">
        <span>其中摘要调用花掉的</span><b>{{ result.summary_cost.toFixed(1) }}</b>
      </div>
      <div class="kv">
        <span>约束保留 (摘要留住了几条)</span><b :class="result.retained === 3 ? 'good' : 'bad'">{{ result.retained }} / 3</b>
      </div>
      <div class="kv">
        <span>峰值上下文 (最大的一次请求)</span><b>{{ result.peak }} tok</b>
      </div>
      <div class="lab-note">
        <p>计费规则直接复用 m19 前缀缓存章节的三档代价模型:</p>
        <ul class="pts">
          <li>命中缓存的前缀每 tok 算 0.1。</li>
          <li>没命中的每 tok 算 1.25。</li>
          <li>每次摘要再固定加 50 (摘要输出 50 tok, 每 tok 算 1)。</li>
        </ul>
        <p>文本块都是固定大小的示意值。</p>
        <p>上述读数并非云厂商实时 API 报价, 亦非真实 tokenizer 切分统计的精确 token 数, 仅用于在同一计价标尺下横向比较不同上下文管理策略的相对成本。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { contextCache } from '@/utils/agent-reliability.js'
const turns = ref(6),
  placement = ref('tail'),
  threshold = ref(4),
  keepScope = ref(true),
  selected = ref(0)
const result = computed(() => contextCache(turns.value, placement.value, threshold.value, keepScope.value))
const baseline = computed(() => contextCache(turns.value, placement.value, 0, true))
const frame = computed(() => result.value.rows[Math.min(selected.value, result.value.rows.length - 1)])
</script>
