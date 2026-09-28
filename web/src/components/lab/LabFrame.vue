<!--
  实验台外框 —— 所有 lab 统一用它, 保证标题 / 控件 / 图 / 读数 / 挑战题的版式一致。
  插槽: controls(滑杆按钮) · default(可视化主体) · stats(右侧读数) · footer
  challenge: 先让读者预测再动手, 比直接看答案记得牢。
  challenge.ask 和 answer 都走 Prose: 问题里有几问就用 \n 分段或写成 "- " 要点, 折叠着也是分行显示的。
-->
<template>
  <section class="lab card">
    <header class="lab-head">
      <h3>{{ title }} <span v-if="module" class="tag mono">{{ module }}</span></h3>
      <Prose v-if="sub" class="lab-sub" :text="sub" />
    </header>

    <div v-if="$slots.controls" class="lab-controls"><slot name="controls" /></div>

    <div class="lab-body" :class="{ single: !$slots.stats }">
      <div class="lab-viz"><slot /></div>
      <aside v-if="$slots.stats" class="lab-stats"><slot name="stats" /></aside>
    </div>

    <details v-if="challenge" class="lab-challenge">
      <summary>
        <span class="ask-k">🎯 试一试</span>
        <span class="ask-hint">点这里看答案</span>
        <Prose class="ask" :text="challenge.ask" />
      </summary>
      <Prose class="answer" :text="challenge.answer" />
    </details>
    <p v-if="run" class="lab-run">对应可运行代码: <code class="inline">{{ run }}</code></p>
    <slot name="footer" />
  </section>
</template>

<script setup>
import Prose from '@/components/Prose.vue'

defineProps({
  title: { type: String, required: true },
  sub: { type: String, default: '' },
  module: { type: String, default: '' },     // 例如 "llm_infer/m02"
  run: { type: String, default: '' },        // 例如 "python -m llm_infer.m02_paged_attention.demo"
  challenge: { type: Object, default: null }, // { ask, answer }
})
</script>
