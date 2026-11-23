<!--
  prompt 相关的可验证奖励实验台。
  只讲一件事: 如果奖励函数不看 prompt (例如"输出落在词表后半区就给 1 分"), 最优策略就是无视 prompt 的常数输出 ——
  reward 涨到 1.0 也证明不了模型学会了"按题作答"。奖励必须是 (prompt, completion) 的函数, RL 才是在学条件生成。
-->
<template>
  <LabFrame
    title="可验证奖励要「看题」— 否则 RL 只学会输出一个常数"
    sub="- 模拟什么: RLVR (基于可验证奖励的强化学习) 使用确定性程序为模型输出赋予 0/1 奖励, 典型场景包括数学题答案核对与代码单元测试。可验证奖励函数必须满足一项基本前提: 判分逻辑必须同时依赖输入 prompt 与模型输出。若奖励函数仅检查输出本身 (例如只要输出落在词表后半区即给 1 分), 最优策略就会退化为无视 prompt、恒定输出同一 token; 此时即使平均奖励升至 1.0, 也无法说明模型学会了按题作答。
- 图上是什么: 面板共 4 行, 每行代表一道形如「a + b = ?」的算术题, 各视觉元素含义如下:
  - 右侧 10 个方格: 策略可输出的 10 个候选答案 token (0 至 9)。
  - 绿色边框: 标出当前奖励函数判定可得 1 分的目标输出位置。
  - 高亮且带数字的方格: 当前策略在该题上实际输出的 token。
  - 行尾数字: 当前题目获得的即时奖励。
- 可以做什么: 初始状态默认采用「看题奖励」配合「常数策略」, 可通过以下控件对比验证:
  - 切换「区域奖励」与「看题奖励」: 改变判分规则, 前者仅检查输出本身 (token ≥ 5 即给分), 后者联合检查输入与输出 (要求输出等于 (a+b) mod 10)。
  - 在「常数策略」模式下点击任意方格: 将 4 道题的输出同步设为同一 token, 模拟完全忽略 prompt 的退化策略。
  - 在「条件策略」模式下点击各行方格: 为每道题独立指定输出 token, 模拟根据输入条件作答的正常策略。
  - 点击「换一组题」: 随机刷新 4 道算术题。
- 应该看到: 在「看题奖励」下, 各行绿框位置随 prompt 动态变化, 常数策略无论选择哪个 token 最多只能答对 1 题; 切换至「区域奖励」后, 绿框变为跨行对齐的竖向条带, 常数策略固定输出 7 即可拿满全部分数。右侧读数含义如下:
  - 「当前策略平均奖励」: 4 道题的平均得分。
  - 「常数策略的上限」: 对所有题目恒定输出同一 token 时所能达到的最高平均得分; 该值等于 100% 时标红预警, 表明强化学习会沿阻力最小方向直接收敛至该常数输出。
  - 「输出依赖 prompt 吗」: 检测 4 道题的输出是否随输入变化。
  - 「满分 ⇒ 读懂了题?」: 当常数策略亦可取得满分时显示「推不出」。"
    module="llm_finetune/data/tasks.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '按顺序执行以下实验并回答:\n1. 在「看题奖励 (a+b) mod 10」下, 点击方格切换常数策略的输出 token, 平均奖励最高可达多少?\n2. 切换至「区域奖励」后, 常数策略最高可拿多少分?\n3. 面板上绿框的分布形态发生了怎样的变化?',
      answer: '答案:\n1. 在看题奖励下, 常数策略最高只能拿到 25% (4 道题中最多命中 1 题)。\n2. 切换至区域奖励后, 常数策略可拿到 100%。\n3. 绿框分布从「每行仅一格且列位置互不相同」变为「跨行对齐的若干条竖向条带」。\n原因解析:\n- 看题奖励约束: 题目生成器保证 4 道题的真实答案两两不同, 因此每行仅有单个绿框且分布在不同列上。常数策略固定输出同一 token 时最多只能命中其中 1 题, 得分上限为 1/4; 模型唯有建立从输入到输出的条件映射才能取得满分。\n- 区域奖励漏洞 (token ≥ 5 即得分): 合法输出集合与输入无关, 在所有行上完全一致, 恒定输出 7 的常数策略即可稳拿满分。此时训练奖励曲线虽然快速收敛至上限, 但策略完全未读取 prompt, 仅学到了偏向特定 token 的一元分布 (unigram) 偏置而非目标任务。\n- 实验设计启示: 构建强化学习合成任务时必须严格区分「管线能否推高奖励数值」与「策略是否习得条件推理能力」。\n- 真实的 RLVR 场景 (数学竞赛题、代码单元测试) 天然具备输入条件依赖性; 在设计教学或调试用玩具任务时则需显式校验这一性质, 并单独汇报常数策略基线的得分上限。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: reward === 'region' }" @click="reward = 'region'">区域奖励: token ≥ 5 (不看题)</button>
        <button type="button" :class="{ active: reward === 'sum' }" @click="reward = 'sum'">看题奖励: 输出 = (a+b) mod 10</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: constant }" @click="setConstant(true)">常数策略 (无视 prompt)</button>
        <button type="button" :class="{ active: !constant }" @click="setConstant(false)">条件策略 (每题各自输出)</button>
        <button type="button" @click="seed++">换一组题</button>
      </div>
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `92px repeat(10, 30px) 52px` }">
      <span class="rl">prompt</span>
      <span v-for="d in 10" :key="'h' + d" class="hd mono">{{ d - 1 }}</span>
      <span class="hd">奖励</span>
      <template v-for="(p, i) in prompts" :key="i">
        <span class="rl mono">{{ p.a }} + {{ p.b }} = ?</span>
        <button
          v-for="d in 10" :key="d" type="button" class="cell ans"
          :class="{ win: rewardOf(p, d - 1) === 1, on: outs[i] === d - 1 }"
          :aria-label="`题 ${i + 1} 输出 ${d - 1}`" :aria-pressed="outs[i] === d - 1"
          @click="pick(i, d - 1)"
        >{{ outs[i] === d - 1 ? d - 1 : '' }}</button>
        <span class="mono rw" :class="rewardOf(p, outs[i]) ? 'good' : 'bad'">{{ rewardOf(p, outs[i]) }}</span>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span>当前策略平均奖励 (4 题平均)</span><b :class="mean === 1 ? 'good' : mean < 0.5 ? 'bad' : ''">{{ (mean * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>常数策略的上限 (无视题目能拿几分)</span><b :class="constBest === 1 ? 'bad' : 'good'">{{ (constBest * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>输出依赖 prompt 吗 (4 题输出是否不同)</span><b>{{ depends ? '依赖' : '不依赖' }}</b></div>
      <div class="kv"><span>满分 ⇒ 读懂了题?</span><b :class="constBest === 1 ? 'bad' : 'good'">{{ constBest === 1 ? '推不出' : '是' }}</b></div>
      <p class="lab-note">常数策略上限是指对所有输入恒定输出同一 token 时所能获得的最高平均奖励。当奖励函数设计不当使该上限达到 100% 时, 强化学习会沿最短路径直接收敛至常数输出, 导致策略分布发生熵塌缩且输出与输入脱钩; 此时奖励曲线上升并不能证明模型学会了按题作答。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

// 默认是看题奖励: 常数策略拿不到满分。切到区域奖励才会看到 100%, 留给读者自己去点
const reward = ref('sum'), constant = ref(true), seed = ref(1)

// 4 道题, 保证答案两两不同 (否则常数策略能蒙对不止一题, 对比不够干净)
const prompts = computed(() => {
  const rand = mulberry32(seed.value * 389), ps = []
  while (ps.length < 4) {
    const a = 1 + Math.floor(rand() * 9), b = 1 + Math.floor(rand() * 9)
    if (!ps.some((p) => (p.a + p.b) % 10 === (a + b) % 10)) ps.push({ a, b })
  }
  return ps
})
const outs = ref([7, 7, 7, 7])
watch(prompts, () => { outs.value = [7, 7, 7, 7]; constant.value = true })

// ★ 两种奖励函数的唯一区别: 签名里有没有 prompt
const rewardOf = (p, tok) => (reward.value === 'region' ? +(tok >= 5) : +(tok === (p.a + p.b) % 10))

const pick = (i, tok) => { outs.value = constant.value ? outs.value.map(() => tok) : outs.value.map((o, k) => (k === i ? tok : o)) }
const setConstant = (c) => { constant.value = c; if (c) outs.value = outs.value.map(() => outs.value[0]) }

const mean = computed(() => sum(prompts.value.map((p, i) => rewardOf(p, outs.value[i]))) / prompts.value.length)
const constBest = computed(() => Math.max(...range(10).map((t) => sum(prompts.value.map((p) => rewardOf(p, t))) / prompts.value.length)))
const depends = computed(() => new Set(outs.value).size > 1)
</script>

<style scoped>
.grid { align-items: center; gap: 4px; width: max-content; }
.rl { font-size: 12px; color: var(--text-muted); }
.hd { font-size: 10px; color: var(--text-dim); text-align: center; }
.ans { min-width: 0; min-height: 0; width: 30px; height: 30px; padding: 0; font-size: 13px; cursor: pointer; }
.ans.win { border: 2px solid var(--left); }
.ans.on { background: var(--accent); color: var(--bg-card); font-weight: 700; }
.rw { text-align: center; font-size: 14px; }
.rw.good { color: var(--left); }
.rw.bad { color: var(--danger); }
</style>
