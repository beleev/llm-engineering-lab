<!--
  prompt 相关的可验证奖励实验台。
  只讲一件事: 如果奖励函数不看 prompt (例如"输出落在词表后半区就给 1 分"), 最优策略就是无视 prompt 的常数输出 ——
  reward 涨到 1.0 也证明不了模型学会了"按题作答"。奖励必须是 (prompt, completion) 的函数, RL 才是在学条件生成。
-->
<template>
  <LabFrame
    title="可验证奖励要「看题」— 否则 RL 只学会输出一个常数"
    sub="- 模拟什么: RLVR (可验证奖励的强化学习) 用一个程序给模型的输出打 0/1 分, 比如数学题答案对不对、单元测试过不过。这个台子看奖励函数的一个必要条件: 它必须同时看 prompt 和输出。如果奖励只看输出 (比如「输出落在词表后半区就给 1 分」), 最优策略就是无视 prompt、永远输出同一个 token; reward 涨到 1.0 也证明不了模型学会了按题作答。
      - 图上是什么: 4 行, 每行是一道题, prompt 是「a + b = ?」; 右边 10 个格子是策略可能输出的 10 个答案 token (0 到 9)。绿框标出当前奖励函数会给 1 分的输出; 高亮并写着数字的格子是策略现在在这道题上实际输出的 token; 行尾是这道题拿到的奖励。
      - 可以做什么: 「区域奖励」和「看题奖励」切换奖励函数: 前者只看输出 (token ≥ 5 就给分), 后者看 prompt (输出要等于 (a+b) mod 10)。「常数策略」下点任意格子, 4 道题的输出一起变成那个 token (模拟一个无视 prompt 的策略); 「条件策略」下每道题可以各自点 (模拟真的在按题作答)。「换一组题」换 4 道新题。一打开是看题奖励加常数策略, 先试着用它拿满分。
      - 应该看到: 看题奖励下, 每行的绿框位置随 prompt 变, 常数策略怎么点最多只能拿到 1 行的分; 区域奖励下绿框是几条竖带, 每行一样, 常数策略输出 7 就满分。右侧读数: 「当前策略平均奖励」是 4 道题的平均分; 「常数策略的上限」是所有题输出同一个 token 时能拿到的最高平均分, 它等于 100% 时变红, 因为这时 RL 会直接收敛到那个常数; 「输出依赖 prompt 吗」看 4 道题的输出是否相同; 「满分 ⇒ 读懂了题?」在常数策略也能满分时显示「推不出」。"
    module="llm_finetune/data/tasks.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '「看题奖励 (a+b) mod 10」下, 点格子换常数策略的输出, 最高能拿多少分? 先猜再点: 切到「区域奖励」后, 常数策略最高能拿多少? 绿框的形状从什么变成了什么?',
      answer: '答案: 看题奖励下常数策略最多 25% (4 题蒙对 1 题); 切到区域奖励后常数策略能拿 100%; 绿框从「每行一个、位置各不同」变成「几条竖带、每行一样」。\n一步步看:\n- 看题奖励: 4 道题的答案两两不同 (出题时保证的), 每行只有一个绿框且位置不同。常数策略只能让一个 token 同时对一道题, 上限 1/4。要拿满分必须让输出依赖输入。\n- 区域奖励 (token ≥ 5 就给分): 可行的输出每行都一样, 永远输出 7 的常数策略就是满分。训练曲线会很漂亮, 但策略根本没读 prompt, 学到的是一个 unigram 偏好 (对某几个 token 的固定偏爱), 不是任务。\n- 这是自己造教学用 RL 任务最容易踩的坑: 验证「RL 管线能让 reward 上升」和验证「RL 学会了条件行为」是两回事。\n- 真实 RLVR (数学答案、单元测试) 天然是看题的。自己造玩具任务时要主动保证这一点, 并单独报告「常数基线」能拿多少分。',
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
      <p class="lab-note">
        常数策略上限是让所有题输出同一个 token 时能拿到的最高平均分。奖励函数让这个上限等于 100% 时,
        RL 会沿最短路径收敛到那个常数: 分布塌缩成一个尖 (熵塌缩), 输出和输入无关。这时 reward 曲线再漂亮也证明不了什么。
      </p>
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
