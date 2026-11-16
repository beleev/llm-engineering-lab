<!-- 对照实验: m23_context_cache。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="压短了, 就一定省钱吗 — 摘要、缓存和动态状态一起算账"
    sub="- 模拟什么: 一个多轮 agent 会话的 token 账单。每轮送给模型的上下文由几块拼成: 固定的策略说明 policy (200 tok)、历史 (每轮一条用户消息 20 tok 加一条工具结果 200 tok, 越积越多)、动态状态 (记忆和进度, 40 tok)、本轮用户输入 (20 tok)。模型供应商按前缀缓存计费: 和上一次请求开头完全相同的那一段按「读缓存」算 (每 tok 0.1), 从第一个不同的块开始全部按「写缓存」算 (每 tok 1.25)。历史太长时可以请模型把历史压成一条摘要 (50 tok), 但摘要这次调用本身也要花钱: 要把整段历史读一遍, 再加固定 50 的输出费。
      - 图上是什么: 一排按钮是会话的每一轮, 按钮上是这一轮实际送进模型的上下文大小。点一轮, 下面的面板给出这一轮的输入和费用, 进度条是它相对峰值的大小; 再下面比较「同样设置但不摘要」的总价和当前总价; 最后一行是摘要保住了几条约束, 以及原始审计记录 (不被摘要删掉的原文块) 还有几块。
      - 可以做什么: 「动态状态位置」选「最前部」就是把每轮都在变的状态块放在上下文开头, 选「历史之后」放在历史后面。拖「会话轮数」改一共跑几轮。「摘要阈值」是历史积累到几块就触发一次摘要, 「不摘要」就是一直累积。勾掉「摘要保留范围与禁止事项」, 摘要就只保住语言偏好 TypeScript, 丢掉项目范围和禁止删除这两条约束。
      - 应该看到: 默认 6 轮、历史之后、阈值 4: 总费用 3202.5, 其中摘要 1327.5, 不摘要是 4925, 峰值上下文从 1360 降到 530。把动态状态改到「最前部」: 每轮开头就变, 后面整段历史都缓存不上, 不摘要时总价涨到 6075。把轮数调成 2、阈值调成 2: 不摘要 695, 摘要反而 840, 短会话摊不薄摘要的开销。勾掉保留约束: 费用一分不少, 但「约束保留」从 3/3 变成 1/3。右侧「模拟总费用」比不摘要便宜就是绿色, 更贵就是红色; 「其中摘要费用」是摘要调用花掉的部分; 「峰值上下文」是最大的一次请求有多少 tok。所有数字都是费用模型算出来的单位, 不是真实账单。"
    module="llm_agent/m23_context_cache"
    run="python -m llm_agent.m23_context_cache.demo"
    :challenge="{
      ask: '把轮数调成 2, 摘要阈值调成 2。输入变短后, 为什么总费用可能变高?',
      answer:
        '因为摘要自己也是一次模型调用, 而且它改写了后面每一轮的前缀。\n- 摘要调用要把整段历史读一遍: 第 2 轮触发摘要时, policy 能走读缓存, 但两块历史和摘要指令都按写缓存计价, 再加固定 50 的输出费, 这一次就花了 357.5。\n- 摘要之后历史变成一条新的 summary 块, 它和之前缓存里的任何前缀都对不上, 下一轮又要重新写一遍缓存。\n- 省下来的只有「少送的历史 token」。会话只有 2 轮时根本没攒下多少历史, 省的不够抵摘要的开销: 不摘要 695, 摘要 840。6 轮时历史攒得够多, 摘要才划算 (4925 对 3202.5)。\n完整实验还检查另一件事: 两份摘要费用一样, 但一份保住了项目范围和「禁止删除」这两条约束, 另一份只剩语言偏好。费用相同, 丢掉的可能正是最关键的约束。',
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
        <p>费用模型沿用 m19 (prompt caching 那一章): 命中缓存的前缀每 tok 算 0.1, 没命中的每 tok 算 1.25, 每次摘要再固定加 50 (摘要输出 50 tok, 每 tok 算 1)。文本块都是固定大小的示意值。</p>
        <p>这些数字不是实际 API 价格, 也不是 tokenizer 量出来的 token 数, 只用来比较不同策略之间的相对高低。</p>
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
