<!--
  PPO vs GRPO: 同样每步 256 条回复, 预算怎么花。
  只讲一件事: GRPO 把 256 条分成 32 题 × 8 条, 组内全对/全错时 A ≡ 0, 这组白花; PPO 一题一采, 256 道题都有信号。
  格子里的 0/1 奖励是前端按随机难度抽的示意 (可以点格子改); 下方表格来自
  python -m llm_finetune.run_finetune.ppo.train_ppo 的结果表 (与 run_finetune/ppo/README.md 一致), 不随格子变化。
-->
<template>
  <LabFrame
    title="PPO vs GRPO — 同样 256 条回复, baseline 用组均值还是 critic"
    sub="- 模拟什么: 策略梯度算法需要为每条采样回复计算优势估计 $A$, 等于实际回报减去状态方差基线 baseline (即对该题目期望得分的估计, 减去基线方可区分当前采样是优于还是劣于题目平均水平)。在固定单步 256 条在线采样预算下存在两条技术路线:
  - GRPO: 将每步 256 条采样预算分配给 32 道题目、每题重复采样 8 条回复, 直接以同题 8 条样本的得分均值充当经验 baseline, 无需额外训练价值网络。
  - PPO: 每步覆盖 256 道不同题目、每题仅采样 1 条回复, 由独立的参数化价值网络 critic 预测状态初值 $V(s_0)$ 作为 baseline, 需在训练循环中同步更新该网络。
- 图上是什么: 主视图包含 256 个小方格, 每格代表单步批次内的一条采样回复, 绿色代表回答正确 (奖励为 1), 灰色代表回答错误 (奖励为 0), 正误结果由前端按随机题目难度分布采样示意:
  - GRPO 模式下: 每行 8 格对应同一道题目的 8 次采样, 行尾标注组内均值 μ; 整组全对或全错的行会整体置淡并标记 A≡0 (所有样本奖励减组均值均为 0, 整行 8 条样本无梯度贡献)。
  - PPO 模式下: 256 格分别对应 256 道独立题目, 经全批优势白化后每条样本均具有非零梯度。
  - 下方固定对比表: 展示 train_ppo.py 在真实留出集上的训练评测数据 (不随上方示意网格变动)。
- 可以做什么: 按界面顺序可执行以下操作:
  - 点击「baseline 来源」三个切换按钮: 对比 GRPO、标准 PPO, 以及冻结 critic 的对照 PPO (学习率取 0, 价值网络参数全程不更新)。
  - 点击「换一批」按钮: 重新采样一批 256 条回复的正误分布。
  - 点击任意小方格: 单独翻转该条回复的正误状态 (亦可使用键盘方向键移动焦点并按回车翻转)。
- 应该看到: 在 GRPO 模式下批次内总会出现若干整行置淡的零梯度题目组; 切换至 PPO 模式后全部 256 条样本均保持有效激活。右侧汇总读数与下方表格各列含义如下:
  - 「有梯度的回复」: 当前批次内优势非零的有效样本条数, 占比低于 8 成时显示为红色。
  - 「整组全对或全错的题」: 触发组内零方差退化的题目数量 (仅在 GRPO 组采样模式下有定义)。
  - 「本批平均奖励」: 当前 256 条采样回复的得分算术均值。
  - 表格「pass@1」: 留出集单次采样答对率 (SFT 初始起点为 0.186)。
  - 表格「训练奖励」: 训练前 10 步与最后 10 步的平均批次奖励对比。
  - 表格「baseline 解释的方差」: 基线预测值所能解释的真实回报方差比例 (Explained Variance), 越高说明基线估计越准确。
  - 表格「常驻 / 要训参数」与「60 步耗时」: 不同方案的显存参数开销与墙钟运行时间代价。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '请按顺序对比网格视图与下方实测表:\n1. 观察默认批次网格: 在 GRPO 模式下有多少组题目出现 A≡0 退化?\n2. 切换至 PPO 模式后, 批次内还有多少条回复丧失梯度?\n3. 对比下方实测表: 将 critic 完全冻结后留出集 pass@1 与正常 PPO 基本一致 (0.389 对比 0.381), 那么本实验中 PPO 相对 GRPO 高出的约 0.09 准确率增益真正来源于什么因素?',
      answer: '结论:\n1. 默认批次下 GRPO 共有 9 组题目触发 A≡0, 合计 72 条采样回复丧失梯度。\n2. 切换至 PPO 模式后, 全部 256 条回复均保留非零有效梯度。\n3. 本实验中 PPO 高出的约 0.09 准确率增益来源于「单题单采」带来的题目覆盖度与有效样本利用率提升, 而非 critic 价值网络。\n机制解析:\n- 组内零方差样本损耗: GRPO 在单题 8 次采样结果全对或全错时, 各条回复奖励减去组内均值后恒为 0, 整组 8 次采样预算无法产生任何梯度信号; 默认批次的 32 组题目中有 9 组发生该退化。相比之下, PPO 对全批 256 条样本执行优势白化, 只要整批得分不全相同, 每条回复均能获得非零优势权重。\n- 冻结 critic 消融验证: 当 critic 被冻结且输出 $V \\approx 0$ 时, PPO 的优势计算退化为对原始回报直接做全批中心化与标准差归一化, 等价于带全局批均值基线的 REINFORCE。在二元 0/1 奖励任务上, 该全局基线同样能稳定推高 256 道题目中答对样本的概率、压低答错样本的概率, 因此留出集 pass@1 (0.389) 与正常训练 critic 的 PPO (0.381) 相当。\n- 采样分配效率差异: 在固定每步 256 条采样的算力上限下, GRPO 仅覆盖 32 道题目且约 $1/3$ 的组因全对或全错而浪费采样预算; PPO 每步覆盖 256 道不同题目并充分利用全部样本梯度, 这部分样本效率与题目多样性增益贡献了约 0.09 的 pass@1 领先幅度。\n- 小规模短步数下 critic 的真实收益边界: 在本仓库 60 步训练规模下, 独立价值网络几乎未能拟合出有效估值, $V(s_0)$ 仅能解释 2.5% 的真实回报方差, 远低于 GRPO 零训练开销的同题组均值基线 (可解释 36.5% 的方差)。在该实验规模下, 引入 critic 额外付出了 2 倍可训练参数量与约 2 倍训练耗时, 却未带来额外的策略性能提升。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="lbl">baseline 来源:</span>
        <button v-for="m in METHODS" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
        <button type="button" @click="reseed">换一批</button>
      </div>
    </template>

    <div class="grid" :class="{ grouped: mode === 'grpo' }" role="group" aria-label="256 条回复的对错, 方向键移动, 回车翻转">
      <!-- ★ 整张表只占一个 Tab 位: 当前行可聚焦, 方向键在行内和行间移动 -->
      <div
        v-for="(row, g) in cells" :key="g" :ref="(el) => { rowEls[g] = el }"
        class="grow" :class="{ dead: mode === 'grpo' && row.dead }"
        :tabindex="g === cur.g ? 0 : -1" role="group"
        :aria-label="`第 ${g + 1} 行: 8 条里对 ${row.n} 条; 光标在第 ${cur.i + 1} 条, ${row.r[cur.i] ? '对' : '错'}`"
        @keydown.left.prevent="move(0, -1)" @keydown.right.prevent="move(0, 1)"
        @keydown.up.prevent="move(-1, 0)" @keydown.down.prevent="move(1, 0)"
        @keydown.enter.prevent="flip(cur.g, cur.i)" @keydown.space.prevent="flip(cur.g, cur.i)"
      >
        <button
          v-for="(c, i) in row.r" :key="i" type="button" tabindex="-1" class="cell rc"
          :class="{ ok: c, cur: g === cur.g && i === cur.i }"
          :aria-label="`第 ${g * G + i + 1} 条: ${c ? '对' : '错'}, 点击翻转`" @click="tap(g, i)"
        />
        <span v-if="mode === 'grpo'" class="gl mono">{{ row.dead ? 'A≡0' : 'μ=' + row.mu.toFixed(2) }}</span>
      </div>
    </div>

    <p class="tcap">train_ppo.py 留出集实测 (不随上面的格子变化; SFT 起点 pass@1 0.186)</p>
    <table class="res mono">
      <thead><tr><th /><th>pass@1</th><th>训练奖励 前 10 步 → 后 10 步</th><th>baseline 解释的回报方差</th><th>常驻 / 要训参数</th><th>60 步耗时</th></tr></thead>
      <tbody>
        <tr v-for="m in METHODS" :key="m.id" :class="{ sel: mode === m.id }">
          <td>{{ m.name }}</td><td>{{ m.pass1.toFixed(3) }}</td><td>{{ m.first.toFixed(3) }} → {{ m.last.toFixed(3) }}</td>
          <td>{{ m.ev.toFixed(3) }}</td><td>{{ m.params.toLocaleString() }} / {{ m.train.toLocaleString() }}</td><td>{{ m.sec }}</td>
        </tr>
      </tbody>
    </table>

    <template #stats>
      <div class="kv"><span>有梯度的回复 (本批示意, 没被浪费的)</span><b :class="st.live < 256 * 0.8 ? 'bad' : 'good'">{{ st.live }} / 256</b></div>
      <div class="kv"><span>整组全对或全错的题 (GRPO 下 A≡0)</span><b>{{ st.dead }} / {{ NQ }}</b></div>
      <div class="kv"><span>本批平均奖励 (256 格的均值)</span><b>{{ st.mean.toFixed(3) }}</b></div>
      <p class="lab-note">{{ cur_m.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { clamp, mulberry32, range, sum } from '@/utils/labmath.js'

const G = 8, NQ = 32
// 来自 train_ppo.py 结果表 (每步 256 条, 60 步, 留出集); 耗时两次实测有波动, 取 readme 的数
const METHODS = [
  { id: 'grpo', name: 'GRPO (32 题 × 8)', pass1: 0.287, ev: 0.365, first: 0.205, last: 0.295, params: 99648, train: 99648, sec: '6.1 s',
    note: 'baseline = 同题 8 条的均值, 不用训。组内全对或全错时 A 全为 0, 这 8 条白占预算。' },
  { id: 'ppo', name: 'PPO (256 题 × 1)', pass1: 0.381, ev: 0.025, first: 0.2, last: 0.356, params: 299008, train: 199360, sec: '11.8 s',
    note: 'baseline = critic 的 V(s_0)。常驻 policy + ref + critic 三份, 要训两份。V(s_0) 只解释 2.5% 的回报方差, 几乎是个常数。' },
  { id: 'frozen', name: 'PPO, critic 冻结 (lr=0)', pass1: 0.389, ev: 0.004, first: 0.204, last: 0.357, params: 299008, train: 199360, sec: '12.0 s',
    note: 'critic 一步没学, pass@1 和正常 PPO 在噪声内。高出 GRPO 的那截来自一题一采, 不是 critic。(lr=0 时梯度照算, 参数照记。)' },
]
const mode = ref('grpo'), seed = ref(1)
const cur_m = computed(() => METHODS.find((x) => x.id === mode.value))

// 示意: 每道题的通过率 p 偏低且不均 (训练奖励约 0.2–0.36), 抽 0/1 奖励
const draw = (s) => {
  const rand = mulberry32(s * 4099)
  return range(NQ).map(() => { const p = 0.7 * rand() ** 2; return range(G).map(() => +(rand() < p)) })
}
const R = ref(draw(1))
const reseed = () => { seed.value++; R.value = draw(seed.value) }
const flip = (g, i) => { R.value = R.value.map((row, k) => (k === g ? row.map((c, j) => (j === i ? 1 - c : c)) : row)) }

// 键盘光标: 方向键改 (行, 列), 焦点跟着行走
const cur = ref({ g: 0, i: 0 })
const rowEls = []
const tap = (g, i) => { cur.value = { g, i }; flip(g, i) }
const move = (dg, di) => {
  cur.value = { g: clamp(cur.value.g + dg, 0, NQ - 1), i: clamp(cur.value.i + di, 0, G - 1) }
  nextTick(() => rowEls[cur.value.g]?.focus())
}

const cells = computed(() => R.value.map((r) => {
  const n = sum(r)
  return { r, n, mu: n / G, dead: r.every((c) => c === r[0]) }   // ★ 组内奖励全相同 → r − μ ≡ 0
}))
const st = computed(() => {
  const all = R.value.flat(), dead = cells.value.filter((c) => c.dead).length
  // PPO: 批内白化后, 只要整批不全相同, 每条都有非零 A
  const live = mode.value === 'grpo' ? (NQ - dead) * G : all.every((c) => c === all[0]) ? 0 : all.length
  return { live, dead, mean: sum(all) / all.length }
})
</script>

<style scoped>
.lbl { font-size: 12px; color: var(--text-muted); }
.grid { display: flex; flex-direction: column; gap: 2px; width: max-content; }
.grid.grouped { gap: 4px; }
.grow { display: flex; gap: 2px; align-items: center; }
.grow:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.grow:focus-visible .rc.cur { outline: 2px solid var(--text); outline-offset: 0; }
.grow.dead .rc { opacity: 0.3; }
.rc { min-width: 0; min-height: 0; width: 18px; height: 14px; padding: 0; cursor: pointer; }
.gl { font-size: 10px; color: var(--text-dim); margin-left: 6px; width: 44px; }
.grow.dead .gl { color: var(--danger); }
.tcap { margin: 14px 0 4px; font-size: 12px; color: var(--text); }
.res { border-collapse: collapse; font-size: 11px; min-width: 520px; }
.res th { font-size: 10.5px; color: var(--text-dim); font-weight: 400; text-align: right; padding: 3px 6px; }
.res td { padding: 3px 6px; text-align: right; color: var(--text-muted); border-top: 1px solid var(--border); }
.res td:first-child, .res th:first-child { text-align: left; }
.res tr.sel td { background: var(--bg-elev); color: var(--text); }
</style>
