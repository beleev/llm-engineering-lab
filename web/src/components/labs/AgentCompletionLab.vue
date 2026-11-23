<!-- 对照实验: m21_completion_verification。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="模型说「完成」之后, 再看一眼环境 — 验收方式决定你信不信它"
    sub="- 模拟什么: 任务要求 Agent 向存储中写入 alpha、beta、gamma 三条笔记。实验使用确定性规则模型复现线上常见故障: 模型在第一轮仅写入 alpha 与 beta 两条记录, 随后便直接输出「已完成」。本实验台对比三种完成验收策略面对同一虚假完成声明时的表现: 直接采信结束信号、检查回复字面关键词, 以及直接读取外部环境状态进行核验。
- 图上是什么: 上方为执行事件时间轴, 下方为环境状态快照面板:
  - 时间轴按钮: 按执行顺序排列本次运行的每一步事件, 记录写入了哪些键名、模型返回了什么声明以及验收器的判定结论。
  - 环境快照面板: 点击任一时间步, 面板即展示该时刻笔记存储中的真实条目与完成进度条 (满格为全部 3 条目标笔记)。存储中原有的历史笔记 existing 不计入目标进度, 也不应在运行中被覆盖或删除。
- 可以做什么: 通过以下控件切换验收与故障条件:
  - 「验收方式」选择「直接接受 final」: 只要模型发出结束响应即判定任务完成。
  - 选择「回答包含「完成」」: 检查模型最终回复文本中是否包含指定字样, 在本实验设定下与直接接受表现一致。
  - 选择「读取环境验收」: 直接检查笔记存储中 alpha、beta、gamma 三项是否齐全; 若有缺失, 则将缺失项拼入反馈交回 Agent 继续执行下一轮修复。
  - 调节「最多修复次数」: 设定环境验收未通过后允许追加执行的修复轮数上限 (0 至 2 次)。
  - 勾选「模型始终只说完成」: 令模型在每一轮均不调用写入工具而仅声称已完成, 用于检验模型持续失职时验收闭环的兜底行为。
- 应该看到: 对比不同策略的运行结果与右侧读数:
  - 前两种字面验收方式: 右侧「系统宣告通过」显示为「是」, 但「环境实际正确」为「否」, 系统在仅写入 2 条笔记的情况下发生静默漏检。
  - 切换至「读取环境验收」且修复次数至少设为 1: 第一轮验收拦截缺失并触发第二轮补写 gamma, 复核通过后两项指标均变为「是」, 「模型调用」从 2 次增至 4 次 (每轮含一次工具调用与一次完成宣告)。
  - 将修复次数设为 0: 首轮环境验收失败后立即如实报告任务未完成。
  - 勾选「模型始终只说完成」: 每轮环境验收均告失败, 耗尽修复预算后系统如实宣告未完成, 验收器绝不越俎代庖替模型补写数据。
  - 「模型调用」斜杠后的 4: 代表单次运行允许的模型调用硬上限, 触顶后强制停止。"
    module="llm_agent/m21_completion_verification"
    run="python -m llm_agent.m21_completion_verification.demo"
    :challenge="{
      ask: '开启基于环境状态的完成验收并允许至少一次修复重试后, 任务就一定能成功完成吗?',
      answer: '不一定。修复机制仅负责将环境核查发现的缺失项反馈给 Agent 再尝试一轮, 最终能否达成目标仍取决于模型决策与工具可用性。\n- 模型持续失职: 若模型在后续修复轮次中依然不调用工具而仅重复输出「完成」 (勾选「模型始终只说完成」即可复现), 每轮环境验收均会拦截失败, 直至耗尽修复配额后宣告未完成。\n- 外部工具故障: 若底层存储接口持续报错无法落盘, 重试同样无法补齐缺失状态。\n- 验收器的只读边界: 验收器的唯一职责是以只读方式核查外部环境并报告差异, 严禁代替模型写入缺失的 gamma 后宣称通过; 否则验收器便退化为隐式业务逻辑, 破坏评测与运行监控的真实性。\n- 有界预算兜底: 修复次数 (本实验上限为 2 次, 模型调用总上限为 4 次) 耗尽后必须明确返回失败状态, 防止陷入无限重试循环。',
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
      <p class="lab-note">时间轴事件在切换参数时按确定性规则实时重算, 不含随机采样。环境验收函数严格保持只读而不修改笔记存储; Agent 循环内部的 Stop Hook 仍仅承担结束通知职责, 完成度核验与修复重入均由外层驱动器编排。</p>
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
