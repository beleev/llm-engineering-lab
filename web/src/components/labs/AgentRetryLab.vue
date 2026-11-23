<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="故障恢复 — 提交之后响应丢了, 要不要再写一次"
    sub="- 模拟什么: agent 调用具有外部副作用的工具 (向远端账本写入一笔 10 元扣款) 时的网络超时与幂等重试机制。场景设定为首次请求已成功送达服务端并完成记账落盘, 但返回响应在网络传输途中丢失, 客户端仅观测到等待超时。此时客户端处于结果未知的不确定状态: 既无法断言服务端未执行, 也无法确认记账已完成。本实验对比三种超时恢复策略, 检验何种做法会引发重复扣款事故。
- 图上是什么: 上方为事件流时间轴按钮, 下方为状态解析面板:
  - 一排事件按钮: 按实际发生顺序编号展示本次调用经历的各个环节, 如「服务端已写入 10 元」「响应丢失」「同一个幂等键, 返回原收据」等。
  - 下方面板: 点击任意事件按钮可查看该环节的机制说明及客户端最终收敛状态: confirmed 表示已拿到凭据确认成功, unknown 表示状态悬而未决无法确认, denied 表示被权限策略明确拒绝。
- 可以做什么: 通过「重试方式」下拉框切换三种超时处理策略, 并配合两个边界控件进行测试:
  - 「盲目重试」: 捕获超时异常后不附加去重标识, 直接将原始写请求重发一次。
  - 「同一幂等键」: 首次发送与后续重试均携带相同的幂等键 (idempotency key, 用于唯一标识同一笔业务意图的请求 ID), 服务端识别出已处理过的幂等键后直接回传原执行收据, 不再重复执行记账。
  - 「先查询收据」: 遇到超时后暂停重发, 先调用只读查询接口核实服务端是否已生成对应交易记录, 再决定后续动作。
  - 滑杆「提交次数上限」: 限制客户端最多允许发起的写提交次数, 调小至 1 可观察仅允许单次提交且遭遇超时时的最终收敛状态。
  - 复选框「有写入权限」: 取消勾选后模拟权限校验不通过, 验证重试机制是否会错误地反复冲击权限门。
- 应该看到: 三种策略的账本副作用对照与右侧统计读数:
  - 「盲目重试」: 账本实际写入 2 次, 累计扣款金额达 20 元, 右侧读数标红告警。由于首次请求已在服务端生效、丢失的仅为回执报文, 无幂等保护的重试直接造成了重复扣款。
  - 「同一幂等键」与「先查询收据」: 两者均保证账本仅写入 1 次、实际金额准确保持在 10 元, 差异体现在右侧「提交 / 查询」调用计数上。幂等键策略记为提交 2 次、查询 0 次 (第二次提交被服务端幂等表拦截并原样返回历史收据); 先查询策略则记为提交 1 次、查询 1 次。
  - 将「提交次数上限」调至 1: 遭遇首次响应丢失后客户端不再重试, 最终状态停留在 unknown; 在未拿到确定性回执前如实报告状态未知而非假定成功或失败, 符合容错语义要求。
  - 取消勾选「有写入权限」: 三种策略均在发出 1 次请求收到明确拒绝后立即终止, 最终状态记为 denied, 证明确定性的授权拒绝不属于瞬态网络故障, 绝不应触发重试。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '客户端观测到一次请求超时, 能否据此证明服务端尚未执行该写操作?',
      answer: '不能证明。\n一次远程调用包含请求发送、服务端执行与响应回传三个独立阶段, 客户端超时仅表明在截止时间内未收到回执报文, 无法区分故障发生在请求到达前、正在执行中还是响应返回途中。在本实验场景下, 服务端在首次收到请求时便已将 10 元扣款写入账本, 丢失的仅仅是回程响应。此时若执行「盲目重试」, 服务端会将其当作一笔全新的扣款指令再次入账, 导致右侧实际写入次数变为 2、累计扣款金额翻倍为 20 元。\n消除超时重试引发的重复副作用通常有两种标准方案:\n- 服务端幂等键校验: 客户端为同一业务动作生成唯一幂等键并在历次重试中保持不变; 服务端检测到该键已有成功处理记录时直接返回历史收据, 跳过二次写入。工程实现上幂等键必须与请求参数哈希强绑定, 若携带相同幂等键却篡改扣款金额应直接报错拒绝。\n- 先查询后决策: 捕获超时后先调用只读状态查询接口核对交易收据, 一旦确认服务端已落盘即直接采纳结果, 不再发起重复提交。\n此外必须严格区分瞬态超时与确定性拒绝: 取消勾选「有写入权限」后, 三种策略均在单次请求收到权限拒绝后立即停止; 权限拦截不会因重试而改变结论。',
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
      <p class="lab-note">本实验基于单进程内存账本模拟。由于幂等键登记与账本写入处于同一服务端原子上下文内, 能够严格避免重复入账; 但这并不等同于跨外部系统的分布式事务已实现 exactly-once (恰好执行一次), 生产环境仍需额外处理指数退避、并发竞态与网络分区等问题。</p>
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
