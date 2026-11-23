<!-- 回放来自 Python 实际执行; 沿用 LabFrame, 与现有章节和目录合并。sub 是所有知识实验共用的读法说明, 每组实验自己的解释在 challenge 里。 -->
<template>
  <LabFrame v-for="entry in entries" :key="entry.id" :title="entry.title"
            sub="- 模拟什么: 本面板离线回放一组真实 Python 实验脚本的运行结果。顶部标题点明该组对照实验所要验证的核心问题, 标题旁标注对应实验源码路径; 所有实验条件均已在 Python 环境中实际运行完毕, 前端仅读取预先固化的输出夹具数据, 不在浏览器内训练模型, 也不发起任何外部 API 调用。
- 图上是什么: 界面分为左侧执行日志栏与右侧核心指标栏:
  - 左栏主体: 当前选定实验条件下按执行顺序打印的运行记录, 逐段呈现具体操作步骤与中间输出 (其中的原始 token 序列或 JSON 结构均按原样保留)。
  - 左栏底部带竖线注记: 部分实验附有补充说明, 交代该实验的简化前提、规模边界与适用局限。
  - 左栏源码折叠区: 可点击展开对应实验入口函数 experiment 的完整 Python 源码。
  - 右栏读数面板: 汇总当前条件下的量化评估指标, 名称与代码中的统计字段一致, 「是 / 否」代表对应断言或性质是否成立。
- 可以做什么: 顶部一排按钮对应本组实验的不同对照条件 (默认选中第一项); 点击切换任意条件按钮, 左栏运行记录与右栏评估读数会同步更新为对应条件下的实测结果。
- 应该看到: 切换不同对照条件时重点观察右栏同一核心指标的变化幅度与方向, 该差异正是本组实验旨在揭示的机制结论。建议按以下顺序阅读验证:
  1. 先阅读底部「试一试」栏提出的问题。
  2. 根据原理自行推演预期结果。
  3. 切换顶部各对照条件核验实测读数。
  4. 展开参考解析, 逐条对照实验设计逻辑与关键数值的物理含义。"
            :module="entry.module" :run="entry.run" :challenge="entry.challenge" class="completion-lab" :data-experiment="entry.id">
    <template #controls>
      <div class="row" role="group" :aria-label="entry.title + ': 选择实验条件'">
        <button v-for="condition in entry.conditions" :key="condition.id" type="button"
                :class="{ active: current(entry).id === condition.id }" :aria-pressed="current(entry).id === condition.id"
                @click="selected[entry.id] = condition.id">{{ condition.label }}</button>
      </div>
    </template>
    <div class="replay-details">
      <p v-for="(step, i) in current(entry).steps" :key="i" class="step">{{ step }}</p>
      <p v-if="current(entry).note" class="lab-note">{{ current(entry).note }}</p>
      <SourceSnippet :key="entry.module" :src="entry.module + ':experiment'" />
    </div>
    <template #stats>
      <div aria-live="polite" aria-atomic="true">
        <div v-for="(value, key) in current(entry).metrics" :key="key" class="kv">
          <span>{{ key }}</span><b>{{ format(value) }}</b>
        </div>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive } from 'vue'
import { useRoute } from 'vue-router'
import LabFrame from '@/components/lab/LabFrame.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
import records from '@/data/knowledge-completion.json'
const route = useRoute()
const selected = reactive({})
const entries = computed(() => records.entries.filter(entry => entry.routes.includes(route.name)))
const current = entry => entry.conditions.find(condition => condition.id === selected[entry.id]) || entry.conditions[0]
const format = value => Array.isArray(value) ? value.join(', ') : typeof value === 'boolean' ? (value ? '是' : '否') : value
</script>

<style scoped>
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.replay-details { min-width: 0; }
.step { white-space: pre-wrap; overflow-wrap: anywhere; margin-bottom: 10px; font-size: 13px; }
.kv { gap: 12px; align-items: baseline; }
.kv > span, .kv > b { overflow-wrap: anywhere; min-width: 0; }
.kv > b { max-width: 65%; }
</style>
