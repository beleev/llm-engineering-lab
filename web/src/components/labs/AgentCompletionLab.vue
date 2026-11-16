<!-- 对照实验: m21_completion_verification。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="模型说「完成」之后, 再看一眼环境 — 验收方式决定你信不信它"
    sub="- 模拟什么: 任务要求 agent 写三条笔记 alpha、beta、gamma。这里的模型是一个规则模型: 第一轮只写了 alpha 和 beta 两条, 然后就回答「已完成」。真实系统里模型的完成声明经常和环境里的事实对不上。这个台子比较三种验收方式对同一个声明的反应: 直接相信、看回答文字、还是去读环境 (笔记真的在吗)。
      - 图上是什么: 一排按钮是这次运行的时间轴, 每个按钮一件事: 写入了哪几条、模型说了什么、验收结果是什么。点一个按钮, 下面的面板显示那一刻笔记存储里实际有几条, 进度条满格就是 3 条。原本就存在的笔记 existing 不算在 3 条里, 也不该被动过。
      - 可以做什么: 「验收方式」选「直接接受 final」就是模型说完成就结束; 「回答包含「完成」」是检查回答文字里有没有这个词, 在这个 demo 里和直接接受效果一样; 「读取环境验收」是去笔记存储里数 alpha、beta、gamma 是不是都在, 缺了就把缺失项交回给 agent 再做一轮 (修复)。「最多修复次数」是环境验收失败后允许再试几轮 (0 到 2)。勾上「模型始终只说完成」, 模型每轮都什么也不写, 只嘴上说完成, 用来看验收器会不会被拖垮。
      - 应该看到: 前两种验收方式下, 右侧「系统宣告通过」是「是」, 但「环境实际正确」是「否」: 系统以为做完了, 其实只写了 2 条。切到「读取环境验收」且修复次数至少 1: 第一轮验收失败, 第二轮模型补写 gamma, 验收通过, 两项都变成「是」, 「模型调用」从 2 次涨到 4 次 (每轮两次: 一次写笔记、一次宣告完成)。修复次数拖到 0, 验收失败后直接报未完成。勾上「模型始终只说完成」, 每轮验收都失败, 修复次数用完后仍然是未完成, 验收器不会自己替模型补笔记。「模型调用」后面的 4 是这次运行的总调用上限, 用完就停。"
    module="llm_agent/m21_completion_verification"
    run="python -m llm_agent.m21_completion_verification.demo"
    :challenge="{
      ask: '允许修复一次, 就一定能成功吗?',
      answer:
        '不一定。修复只是把缺口交回给 agent 再试一轮, 能不能补上要看模型和工具。\n- 模型可以每一轮都只说「完成」什么也不写 (勾上「模型始终只说完成」就能看到): 验收每轮都失败, 修复次数用完, 结果仍是未完成。\n- 工具也可能一直失败, 笔记写不进去。\n- 验收器的职责只有一个: 只读环境, 指出缺了什么。它不能自己替模型把 gamma 写进去再宣称通过, 否则验收器就变成了代写工具, 「通过」这个结论就不再说明 agent 真的做到了。\n- 修复预算 (这里最多 2 次, 总调用上限 4 次) 用完后必须如实报告未完成, 不能无限重试拖下去。',
    }"
  >
    <template #controls>
      <label
        >验收方式
        <select v-model="mode">
          <option value="direct">直接接受 final</option>
          <option value="text">回答包含「完成」</option>
          <option value="verified">读取环境验收</option>
        </select></label
      >
      <LabSlider v-model="repairs" label="最多修复次数" :min="0" :max="2" />
      <label><input v-model="stubborn" type="checkbox" />模型始终只说完成</label>
    </template>
    <div class="trace" aria-label="执行与验收时间轴">
      <button
        v-for="(frame, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        {{ i + 1 }} · {{ frame.text }}
      </button>
    </div>
    <div class="detail" aria-live="polite">
      <p>{{ frame.text }}</p>
      <p>这一时刻实际写入 {{ frame.count }} / 3 条。</p>
      <div class="meter"><span :style="{ width: (frame.count / 3) * 100 + '%' }" /></div>
      <p>原本就有的笔记 existing 一直保留, 验收还会检查它没被改动。这次运行的模型调用总上限是 4 次。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>系统宣告通过 (验收器怎么判)</span><b>{{ result.passed ? '是' : '否' }}</b>
      </div>
      <div class="kv">
        <span>环境实际正确 (三条笔记都在)</span><b :class="result.actual ? 'good' : 'bad'">{{ result.actual ? '是' : '否' }}</b>
      </div>
      <div class="kv">
        <span>模型调用 / 总上限</span><b>{{ result.calls }} / 4</b>
      </div>
      <p class="lab-note">时间轴每次按确定性规则重算, 没有随机性。验收检查本身只读, 不写笔记; agent 结束时的 Stop Hook 仍然只负责发通知, 验收由外部驱动器来做。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { completion } from '@/utils/agent-reliability.js'
const mode = ref('direct'),
  repairs = ref(1),
  stubborn = ref(false),
  selected = ref(0)
const result = computed(() => completion(mode.value, repairs.value, stubborn.value))
const frame = computed(() => result.value.frames[Math.min(selected.value, result.value.frames.length - 1)])
</script>
