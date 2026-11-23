<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="最快返回的, 可能是错的 — 什么时候才能取消剩下的任务"
    sub="- 模拟什么: 单个 Agent 并发派发三个子任务求解同一问题 (对应真实系统中的三路并行工具调用或三个子 Agent), 目标是在拿到首个可用结果后立即取消其余分支以节省算力。核心难点在于「最先返回」并不等于「验收合格」。实验采用逻辑时钟模拟: 时刻 t 从 1 推进至 6, 每个处于运行态的任务在每个时刻消耗 1 个工作单位。三个子任务的参数如下:
  - fast-wrong: 在 t=1 完成, 但结果验收不合格。
  - correct: 在 t=3 完成, 结果验收合格。
  - slow: 在 t=6 完成, 结果验收合格。
- 图上是什么: 上方为一排逻辑时刻按钮, 下方为对应时刻的任务状态面板:
  - 时刻按钮: 按逻辑时间排列, 按钮上标注截至该时刻累计消耗的工作量。
  - 状态面板: 点击任一时刻后, 下方面板列出该时刻三个子任务各自所处的状态, 包括 running (运行中)、completed (已完成) 与 canceled (已取消)。
- 可以做什么: 调节以下三项控制参数:
  - 「收尾策略」选择「等待全部」: 必须等全部三个子任务自然结束后才汇总收尾。
  - 「收尾策略」选择「首个合格结果后取消」: 一旦有任意子任务的返回结果通过验收, 立即向其余仍在运行的任务发出取消指令。
  - 拖动「总工作量上限」: 为整次调度设置算力预算, 累计工作量触及上限时立即取消所有未完成任务。
- 应该看到: 对比两种策略的运行过程与右侧读数:
  - 「等待全部」策略: 总工作量为 10, 最后工作时刻为 6, 合格结果数为 2 个。
  - 切换至「首个合格结果后取消」: t=1 时 fast-wrong 虽率先返回但未通过验收, 不会触发取消, 其余两路继续运行; t=3 时 correct 完成并通过验收, 系统随即取消 slow, 总工作量降至 7, 最后工作时刻提前至 3, 合格结果数为 1 个。
  - 右侧「合格结果」: 最终通过验收的结果数量, 为 0 时标红警示。
  - 右侧「总工作量」: 三个子任务累计消耗的工作单位之和。
  - 右侧「最后工作时刻」: 最后一次存在运行中任务的逻辑时刻, 反映端到端等待时长。
  - 调低「总工作量上限」: 任务会在预算耗尽时被提前取消, 合格结果数可能降为 0。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '在「首个合格结果后取消」策略下, 时刻 1 已经有子任务返回结果, 为什么其余两个任务仍需继续运行?',
      answer: '因为时刻 1 返回的结果未能通过验收, 不满足触发取消的前提条件。\n- t=1 返回的是 fast-wrong: 该分支虽然耗时最短, 但输出未通过质量校验。提前取消其余分支的前提是已获得至少一个可用结果, 未通过验收的输出不能采信, 因此 correct 与 slow 必须继续运行。\n- t=3 时 correct 完成并通过验收: 此时系统才具备合法依据取消仍在运行的 slow, 使总工作量从 10 降至 7, 最后工作时刻从 6 提前至 3。\n- 取消的异步边界: 在真实分布式或异步运行时中, 取消是一条协商请求, 须待执行器确认后才能保证不再发出新动作; 请求发出至确认送达期间已落盘的外部副作用 (如写入文件或扣款) 不会自动回滚。本实验台简化为执行器当场同步确认, 因此未展示该确认空窗。',
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
      <p class="lab-note">时间 t 与工作量均为离散逻辑单位: 每推进一个时刻代表调度循环迭代一轮, 每个工作单位代表单个运行中任务执行一轮。本实验未引入操作系统线程、真实网络延迟或重试退避。</p>
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
