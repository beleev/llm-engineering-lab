<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="最快返回的, 可能是错的 — 什么时候才能取消剩下的任务"
    sub="- 模拟什么: 一个 agent 同时派出三个子任务去做同一件事 (真实系统里可能是三个并行的工具调用或三个子 agent), 想拿到第一个可用的结果就把其余的取消, 省下算力。问题是「第一个返回的」不一定「合格」。这里用逻辑时钟模拟: 时间 t 从 1 走到 6, 每个还在跑的任务每个时刻消耗 1 个工作单位。三个任务:
        - fast-wrong: t=1 完成, 但结果不合格。
        - correct: t=3 完成, 合格。
        - slow: t=6 完成, 合格。
      - 图上是什么: 一排按钮是逻辑时钟的每一个时刻, 按钮上写着到这一刻为止累计消耗了多少工作量。点一个时刻, 下面的面板列出这一刻三个任务各自的状态: running (还在跑)、completed (已完成)、canceled (被取消)。
      - 可以做什么: 两个控件:
        - 「收尾策略」选「等待全部」: 三个任务都跑完才收工。
        - 「收尾策略」选「首个合格结果后取消」: 一旦有一个结果通过验收, 当场取消还在跑的任务。
        - 拖「总工作量上限」: 给整体加一个预算, 累计工作量到达上限后, 还没完成的任务全部被取消。
      - 应该看到: 两种策略的对照和右侧读数:
        - 「等待全部」: 总工作量 10, 最后工作时刻 6, 合格结果 2 个。
        - 切到「首个合格结果后取消」: t=1 时 fast-wrong 虽然返回了, 但它不合格, 不能触发取消, 另外两个继续跑; t=3 时 correct 返回并通过验收, 这时才取消 slow, 总工作量变成 7, 最后工作时刻变成 3, 合格结果 1 个。
        - 右侧「合格结果」: 通过验收的结果个数, 0 个标红。
        - 右侧「总工作量」: 三个任务一共消耗的工作单位。
        - 右侧「最后工作时刻」: 最后一次有任务在干活的时刻, 也就是整件事实际花了多久。
        - 把上限拖小: 任务会在预算耗尽时被取消, 合格结果可能变成 0。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '第 1 格已经有人返回了, 为什么还要继续工作?',
      answer:
        '因为那个结果没有通过验收。\n- t=1 返回的是 fast-wrong: 它最快, 但验收不合格。取消别的任务的前提是「已经有一个可用的结果」, 不合格的结果不算数, 所以 correct 和 slow 要继续跑。\n- t=3 时 correct 返回并通过验收, 这时才有资格取消 slow。总工作量从 10 降到 7, 最后工作时刻从 6 提前到 3。\n- 取消不是立刻生效: 真实系统里取消是一条请求, 要等执行器确认后才保证不再发新动作; 从发出请求到收到确认之间已经发生的副作用 (已经写出去的文件、已经扣的款) 不会自动回滚。本实验让执行器当场确认, 所以看不到这段空窗。',
    }"
  >
    <template #controls>
      <label
        >收尾策略
        <select v-model="mode">
          <option value="all">等待全部</option>
          <option value="first-valid">首个合格结果后取消</option>
        </select></label
      >
      <LabSlider v-model="budget" label="总工作量上限" :min="0" :max="12" />
    </template>
    <div class="trace" aria-label="逻辑时钟">
      <button
        v-for="(frame, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        t={{ frame.tick }}<br />累计工作 {{ frame.work }}
      </button>
    </div>
    <div class="detail">
      <p v-for="(state, job) in frame.states" :key="job">{{ job }}: {{ state }}</p>
      <p>取消的顺序是:</p>
      <ol class="pts">
        <li>先发出取消请求。</li>
        <li>等执行器确认。</li>
        <li>确认之后才保证不再发新动作。</li>
      </ol>
      <p>被取消的任务如果之后还送回一个迟到的结果, 接收方要先核对任务版本 (generation), 旧版本的结果不能混进新任务。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>合格结果 (通过验收的个数)</span><b :class="result.accepted.length ? 'good' : 'bad'">{{ result.accepted.length }}</b>
      </div>
      <div class="kv">
        <span>总工作量 (三个任务一共消耗)</span><b>{{ result.work }}</b>
      </div>
      <div class="kv">
        <span>最后工作时刻 (整件事用了多久)</span><b>{{ result.elapsed }}</b>
      </div>
      <p class="lab-note">时间 t 和工作量都是逻辑单位: 一个时刻就是循环走一轮, 一个工作单位就是一个任务跑了一轮。没有真实线程、网络延迟或重试退避。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { schedule } from '@/utils/agent-reliability.js'
const mode = ref('all'),
  budget = ref(12),
  selected = ref(0)
const result = computed(() => schedule(mode.value, budget.value))
const frame = computed(() => result.value.frames[Math.min(selected.value, result.value.frames.length - 1)])
</script>
