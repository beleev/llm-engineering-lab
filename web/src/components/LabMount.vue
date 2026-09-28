<!--
  实验台挂载点。labs/ 目录下的 .vue 自动注册 (文件名即 lab 名), 新增 lab 不需要改任何注册表。
  某章挂哪些 lab: topicPages[route].widgets 与 data/labMap.js 两处合并。
  每个 lab 是单独的分包, 按需下载。下载中显示占位框 (占住高度, 页面不会跳), 下载失败显示提示。
-->
<template>
  <component :is="lab" v-for="(lab, i) in list" :key="names[i]" />
</template>

<script setup>
import { computed, defineAsyncComponent, h } from 'vue'
import { useRoute } from 'vue-router'
import { labMap } from '@/data/labMap.js'
import { topicPages } from '@/data/models.js'

// 占位框的样式在 main.css 的 Lab kit 一节 (.lab-placeholder)
const Loading = { render: () => h('div', { class: 'card lab lab-placeholder', 'aria-busy': 'true' }, '实验台加载中…') }
const Failed = { render: () => h('div', { class: 'card lab lab-placeholder failed', role: 'alert' }, '实验台加载失败。刷新页面重试; 下面的正文不受影响。') }

const modules = import.meta.glob('@/components/labs/*.vue')
const LABS = Object.fromEntries(
  Object.entries(modules).map(([file, loader]) => [
    file.split('/').pop().replace('.vue', ''),
    // delay: 200 ms 内下载完就不闪占位框
    defineAsyncComponent({ loader, loadingComponent: Loading, errorComponent: Failed, delay: 200 }),
  ]),
)

const props = defineProps({ names: { type: Array, default: null } })
const route = useRoute()
const names = computed(() => {
  const wanted = props.names
    || [...(topicPages[route.name]?.widgets || []), ...(labMap[route.name] || [])]
  return [...new Set(wanted)].filter((n) => LABS[n])
})
const list = computed(() => names.value.map((n) => LABS[n]))
defineExpose({ count: computed(() => list.value.length) })
</script>
