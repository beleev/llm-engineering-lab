<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="故障恢复 — 提交之后响应丢了, 要不要再写一次"
    sub="- 模拟什么: agent 调用一个有副作用的工具 (往账本写一笔 10 元)。请求发到了服务端, 服务端也记了账, 但响应在回来的路上丢了 (客户端只看到超时)。客户端此时只知道「不知道结果」: 既不能说没执行, 也不能说成功了。这个台子比较三种处理超时的办法, 看哪种会把钱扣两次。
- 图上是什么: 按钮和下面的面板:
  - 一排按钮: 这次提交从头到尾的事件, 按发生顺序编号, 比如「服务端已写入 10 元」「响应丢失」「同一个幂等键, 返回原收据」。
  - 下面的面板: 点一个按钮, 面板显示这一步的说明和客户端最终的状态: confirmed 是确认成功, unknown 是无法确认, denied 是被授权拒绝。
- 可以做什么: 「重试方式」下拉框选三种做法, 另有两个控件:
  - 「盲目重试」: 超时就原样再发一次。
  - 「同一幂等键」: 每次重试都带同一个幂等键 (idempotency key, 一个标识「这是同一笔操作」的 id), 服务端认出来就直接返回上次的收据, 不再写第二次。
  - 「先查询收据」: 超时后先问服务端这笔到底成没成, 再决定要不要重发。
  - 「提交次数上限」: 最多提交几次, 拖到 1 看只提交一次会停在什么状态。
  - 取消勾选「有写入权限」: 看授权被拒时重试会不会把拒绝变成放行。
- 应该看到: 几组对照和右侧读数:
  - 「盲目重试」: 写入 2 次, 实际金额 20 元, 右侧标红。第一次已经成功, 只是响应丢了, 第二次就重复扣款。
  - 「同一幂等键」和「先查询收据」: 都只写 1 次 10 元, 区别在「提交 / 查询」。幂等键是提交 2 次、查询 0 次, 第二次提交被服务端认出来返回原收据; 查询收据是提交 1 次、查询 1 次。
  - 把上限拖到 1: 盲目重试停在 unknown, 这是对的: 没确认过, 就不能当成功。
  - 取消「有写入权限」: 三种方式都只发 1 次请求就停, 状态 denied, 授权拒绝不能靠重试绕过。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '一次超时, 能证明服务端没执行吗?',
      answer:
        '不能。\n请求、执行、响应是三个环节, 超时只说明响应没在规定时间内回来。本实验里服务端第一次就把 10 元记进账本了, 丢的只是响应。这时候「盲目重试」会再扣一次, 右侧写入次数变 2、金额变 20 元。\n避免重复写入有两种办法, 这里都演示了:\n- 服务端幂等键: 客户端每次重试带同一个键, 服务端发现这个键已经处理过, 直接返回原来的收据, 不再写第二次。键必须和参数绑在一起, 同一个键换金额应该被拒绝。\n- 先查询状态: 超时后先问服务端这笔成没成, 确认了就不再提交。\n另外, 授权被拒不是超时: 取消「有写入权限」后三种方式都只发一次请求就停, 重试不能把拒绝变成放行。',
    }"
  >
    <template #controls>
      <label
        >重试方式
        <select v-model="mode">
          <option value="blind">盲目重试</option>
          <option value="idempotent">同一幂等键</option>
          <option value="query">先查询收据</option>
        </select></label
      >
      <LabSlider v-model="attempts" label="提交次数上限" :min="1" :max="3" />
      <label><input v-model="allowed" type="checkbox" />有写入权限</label>
    </template>
    <div class="trace" aria-label="提交与确认过程">
      <button
        v-for="(text, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        {{ i + 1 }} · {{ text }}
      </button>
    </div>
    <div class="detail">
      <p>{{ result.frames[Math.min(selected, result.frames.length - 1)] }}</p>
      <p>客户端状态: {{ result.status }} (confirmed 已确认成功, unknown 无法确认, denied 被授权拒绝)</p>
      <p>unknown 表示无法确认: 它既不等于没执行, 也不等于已经成功; 下一步要么去查, 要么带幂等键重试。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>服务端写入次数 (超过 1 就是重复扣款)</span><b :class="result.writes > 1 ? 'bad' : 'good'">{{ result.writes }} 次</b>
      </div>
      <div class="kv">
        <span>账本实际金额 (应该是 10 元)</span><b :class="result.amount > 10 ? 'bad' : ''">{{ result.amount }} 元</b>
      </div>
      <div class="kv">
        <span>提交次数 / 查询次数</span><b>{{ result.requests }} / {{ result.queries }}</b>
      </div>
      <p class="lab-note">内存账本模拟。幂等记录和副作用放在同一个服务端里, 所以能保证不写两次; 这不代表跨服务的事务已经做到 exactly-once (恰好执行一次), 真实系统还要处理退避、并发和网络分区。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { retryWrite } from '@/utils/agent-reliability.js'
const mode = ref('blind'),
  attempts = ref(2),
  allowed = ref(true),
  selected = ref(0)
const result = computed(() => retryWrite(mode.value, allowed.value, attempts.value))
</script>
