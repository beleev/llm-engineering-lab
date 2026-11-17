<!-- 回放来自 Python 实际执行; 沿用 LabFrame, 与现有章节和目录合并。sub 是所有知识实验共用的读法说明, 每组实验自己的解释在 challenge 里。 -->
<template>
  <LabFrame v-for="entry in entries" :key="entry.id" :title="entry.title"
            sub="- 模拟什么: 这里回放的是一组真实 Python 实验的结果。标题是这组实验要回答的问题, 标题旁的路径是实验代码; 每种条件都已经在 Python 里实际跑过, 网页只读出预先存好的结果, 不在浏览器里训练模型, 也不调用任何 API。
- 图上是什么: 左栏是记录, 右栏是读数:
  - 左栏: 当前条件下实验按顺序打印的执行记录, 每一段说一步做了什么、得到什么 (有的是原始的 token 列表或 JSON, 照原样保留)。
  - 记录下面左侧带竖线的小字: 有些实验才有这段说明, 写着实验的前提和局限。
  - 左栏最下面: 可以展开实验函数 experiment 的源码。
  - 右栏: 这个条件下的读数, 名字就是实验里的指标名, 「是 / 否」表示这一项成立与否。
- 可以做什么: 上面一排按钮是这组实验的几种条件, 默认选中第一个; 点另一个按钮, 左栏的记录和右栏的读数都会换成那个条件下的结果。
- 应该看到: 切换条件时盯住右栏同一个读数怎么变, 它就是这组实验要比较的东西。建议的读法:
  1. 先读最下面「试一试」的问题。
  2. 自己猜一个答案。
  3. 再切换条件验证。
  4. 展开答案, 它会一步步解释这组实验做了什么、关键读数是什么意思。"
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
