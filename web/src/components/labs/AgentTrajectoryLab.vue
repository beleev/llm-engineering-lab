<!-- 对照实验: m25_trajectory_learning。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="轨迹学习 — 答案碰巧猜对的轨迹, 能拿去当训练数据吗"
    sub="- 模拟什么: 利用 agent 采样生成的执行轨迹 (trajectory, 即单次任务中的完整动作序列与最终答复) 构建微调或强化学习训练数据时的验收与奖励设计。验收器对轨迹中的每一步依次审查三项约束: 证据来源 (最终答案是否具备前序检索依据)、操作授权 (调用的动作是否处于许可范围内) 与步数预算 (总动作步数是否未超上限)。实验展示针对同一查询任务「某个配额是多少」(标准答案为 10) 采集到的 5 条典型候选轨迹:
  - verified: 先调用合法工具 lookup 查得证据后再准确作答。
  - lucky-guess: 未执行任何检索, 一步直接盲猜输出 10。
  - forbidden: 调用未授权动作 steal 违规获取数据后输出 10。
  - over-budget: 连续重复查询三次后才给出答案。
  - wrong: 正常执行了检索但最终将数值误答为 11。
- 图上是什么: 上方为五条候选轨迹切换按钮, 下方为逐步验收明细面板:
  - 五个按钮: 分别对应五条轨迹, 次行显示在当前奖励口径下计算出的标量奖励分; 绿色外框标识通过全部结果与过程审查、具备资格纳入训练集的合规轨迹 (该合规边框由客观验收准则决定, 不随奖励口径切换而改变)。
  - 下方面板: 点击任意轨迹后逐行展示其动作序列与单步审查结论, 标记为通过或具体违规标签 (unsupported 表示无检索依据盲目作答, authorization 表示调用未授权动作, budget 表示超出步数预算, wrong-result 表示最终答案错误, after-final 表示提交答案后仍有多余动作), 同时底部指明当前得分居首的轨迹。
- 可以做什么: 调节奖励函数口径与步数预算上限:
  - 「奖励口径」切换为「只看结果」: 只要最终输出数值命中标准答案即赋予基础分 1 分, 再统一扣除 0.05 乘以总动作步数的执行成本。
  - 切换为「结果与过程」: 要求最终答案正确且轨迹每一步均通过来源、授权与预算审查才赋予 1 分基础分, 同样按步扣除 0.05 的动作成本。
  - 滑杆「动作预算」: 设定单次任务允许的最大动作步数, 将其调大至 4 可观察 over-budget 轨迹是否随之转化为合规可学习样本。
- 应该看到: 两种奖励函数引导出的策略偏好反差与右侧读数:
  - 在「只看结果」口径下: lucky-guess 获得全场最高奖励 (0.95), 因为它侥幸蒙对答案且仅消耗 1 个动作步, 跳过工具检索反而节省了步数惩罚; 违规取数的 forbidden 同样获得 0.90 的高分。
  - 切换至「结果与过程」口径后: 唯有合规检索作答的 verified 保持正奖励, 其余四条问题轨迹因基础分归零而全部落入负分区间, 最高奖励回归 verified。
  - 右侧「结果正确」: 统计最终输出等于标准答案 10 的轨迹总数 (默认 3 条)。
  - 「可学习轨迹」: 统计同时满足结果正确与全过程合规的轨迹数量, 默认预算下仅有 verified 共 1 条, 将动作预算放宽至 4 后 over-budget 亦通过审查, 增至 2 条。
  - 「最高奖励」: 显示当前奖励口径下五条轨迹中的最高得分。"
    module="llm_agent/m25_trajectory_learning"
    run="python -m llm_agent.m25_trajectory_learning.demo"
    :challenge="{
      ask: '若奖励函数仅依据最终答案是否正确给分, 并按步扣除少量工具调用成本, 优化过程会优先偏好哪条轨迹?',
      answer: '会优先偏好 lucky-guess, 即未执行任何检索却侥幸猜中答案的最短轨迹。\n标量奖励定义为 $\\text{奖励} = \\text{验收分} - 0.05 \\times \\text{动作数}$。在「只看结果」口径下, 验收分仅核对最终答案是否等于 10:\n- verified: 答案正确, 共 2 个动作, 得分为 $1 - 0.10 = 0.90$。\n- lucky-guess: 答案正确, 仅 1 个动作, 得分为 $1 - 0.05 = 0.95$。\n- forbidden: 答案正确 (尽管数据源自越权窃取, 但纯结果口径不予审查), 得分为 0.90。\n- over-budget 与 wrong: 验收分均为 0, 扣除动作成本后仅剩负分 (wrong 输出数值错误; over-budget 在默认 3 步预算内未能完成作答)。\n由此可见, 盲猜命中的最短路径反而获得了全场最高分, 直接用该信号做策略优化会诱导模型习得「放弃检索、直接猜测」的奖励黑客 (reward hacking) 捷径。\n切换至「结果与过程」口径后, 验收分要求轨迹每一步均通过证据来源、操作授权与步数预算三项审查: lucky-guess 的作答步因缺乏检索支撑被标记为 unsupported, forbidden 的 steal 动作因违规被标记为 authorization, 两者验收分均归 0 并倒扣步数成本, 全场唯有合规轨迹 verified 维持 0.90 的正奖励。\n结构化轨迹验收结果在后训练阶段主要有三类用途:\n- SFT 数据过滤: 仅筛选全过程通过验收的高质量轨迹构建 SFT 训练集。\n- 偏好对构造: 将同一任务下的合规轨迹与违规轨迹配对, 用于偏好对齐训练。\n- 过程监督信号: 利用单步审查标签执行过程监督。\n需要注意, 离线训练样本通过验收并不自动意味着训练后的模型已在下游评测中获得提升, 后者仍需通过独立评测集加以验证。',
    }"
  >
    <template #controls>
      <label
        >奖励口径
        <select v-model="aware">
          <option :value="false">只看结果</option>
          <option :value="true">结果与过程</option>
        </select></label
      >
      <LabSlider v-model="budget" label="动作预算" :min="1" :max="5" />
    </template>
    <div class="trace" aria-label="候选轨迹">
      <button
        v-for="(row, i) in rows"
        :key="row.name"
        type="button"
        :class="{ active: selected === i, good: row.report.eligible }"
        @click="selected = i"
      >
        {{ row.name }}<br />奖励 {{ row.reward.toFixed(2) }}
      </button>
    </div>
    <div class="detail">
      <p v-for="(action, i) in trajectories[selected].actions" :key="i">
        {{ i + 1 }} · {{ action }} → {{ rows[selected].report.reasons[i] || '通过' }}
      </p>
      <p>当前奖励最高的轨迹: {{ winner.name }}</p>
      <p>这些结果怎么变成训练数据:</p>
      <ul class="pts">
        <li>通过验收的轨迹进 SFT 数据。</li>
        <li>同一道题的好坏轨迹组成偏好对 (chosen / rejected)。</li>
        <li>每一步的检查结果就是过程监督用的逐步标签。</li>
      </ul>
    </div>
    <template #stats>
      <div class="kv">
        <span>结果正确 (最终答案等于 10)</span><b>{{ rows.filter((r) => r.report.result).length }} / 5</b>
      </div>
      <div class="kv">
        <span>可学习轨迹 (结果和过程都通过)</span><b>{{ rows.filter((r) => r.report.eligible).length }} / 5</b>
      </div>
      <div class="kv">
        <span>最高奖励 (当前口径)</span><b>{{ winner.reward.toFixed(2) }}</b>
      </div>
      <p class="lab-note">奖励计算公式为 $\text{奖励} = \text{验收分} - 0.05 \times \text{动作数}$, 其中验收分通过计 1、不通过计 0。「只看结果」仅核对最终答案, 「结果与过程」则额外要求每一步均通过合规检查。本实验仅演示训练样本筛选与过程标签构造, 后续参数更新直接复用微调阶段的标准方法, 未在本实验台内实际执行模型训练。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { trajectories, trajectoryReport } from '@/utils/agent-reliability.js'
const aware = ref(false),
  budget = ref(3),
  selected = ref(1)
const rows = computed(() =>
  trajectories.map((t) => {
    const report = trajectoryReport(t.actions, budget.value)
    return { ...t, report, reward: Number(aware.value ? report.eligible : report.result) - 0.05 * report.calls }
  }),
)
const winner = computed(() => rows.value.reduce((best, row) => (row.reward > best.reward ? row : best)))
</script>
