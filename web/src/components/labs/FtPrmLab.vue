<!--
  PRM vs ORM 实验台。标签规则与 llm_finetune/methods/prm.py:ArithChain.sample 相同:
  第 i 步 "对" = 写下的 v_i 等于 op_i(解里写的 v_{i−1}, k_i); 最终答案对 = 写下的 v_K 等于真值。
  只讲一件事: 同一条解, ORM 拿到 1 个标签, PRM 拿到 K 个; 过程错、答案蒙对时 ORM 会被骗。
  右侧 best-of-8 表格来自 python -m llm_finetune.run_finetune.prm.train_prm 的输出表, 不随左侧变化。
-->
<template>
  <LabFrame
    title="PRM vs ORM — 同一条解, 结果奖励看到 1 个标签, 过程奖励看到 4 个"
    sub="- 模拟什么: 多步推理任务的奖励模型分为两类: 结果奖励模型 (ORM) 仅根据最终答案正误为整条解答赋予单一标签; 过程奖励模型 (PRM) 则为每个推理步骤逐一标注正误。本实验以模 10 的 4 步算术链作为评测任务: 从初始值 a0 出发, 每步执行一次加、减或乘法并取模 10。标注规则与 prm.py 保持一致:
  - 第 i 步判为正确: 该步写出的结果等于基于上一步账面数值 (而非全局真值) 经当前运算所得的结果。
  - 最终答案判为正确: 第 4 步写出的结果等于全局真值。
- 图上是什么: 主体为按步骤排列的对比表, 各列与各行含义如下:
  - 「解里写的」: 候选解在各步实际写出的算式与结果 (计算错误的步骤标红)。
  - 「真值」: 从初始值出发每步均正确计算时应得的中间值与终值。
  - 「PRM」列: 当前步骤的过程监督标签, 记为 ✓ 1 或 ✗ 0。
  - 末行 EOS: 整条解答的最终输出答案, 右侧并列显示 ORM 给出的结果标签。
  - 表格下方提示语: 说明当前样本上 ORM 与 PRM 两类监督信号是否一致。
- 可以做什么: 支持以下交互操作:
  - 点击任意步骤的「写错」: 将该步结果篡改为错误值, 后续步骤将基于该错误值继续向后推导, 再次点击可恢复正确值。
  - 点击「全部写对」: 清除所有人为注入的步骤错误。
  - 点击「换一道题」: 重新随机生成算术链题目。
  - 点击「蒙对的例子」: 载入一条特定构造的样本, 其中第 1 步计算出错, 但第 2 步乘以 2 在模 10 下恰好消除了偏差, 使最终答案巧合命中真值。
- 应该看到: 默认状态下第 2 步出错, PRM 标签序列为 ✓ ✗ ✓ ✓ (第 3、4 步基于前一步账面值的局部推导本身无误), 而 ORM 仅给出单个 ✗; 在「蒙对的例子」中 ORM 给出 ✓, PRM 则给出 ✗ ✓ ✓ ✓。下方表格列出 train_prm.py 的实测 best-of-8 选解结果 (每题采样 8 条候选解, 取打分器评分最高的一条)。右侧读数与表格列说明如下:
  - 「ORM 拿到的标签」与「PRM 拿到的标签」: 两类奖励模型各自接收的训练监督目标。
  - 「PRM 整条得分」: 取各步得分的最小值, 只要任一步骤出错则整条解答得分为 0。
  - 「两种标签说的是一回事吗」: 当 ORM 对过程出错但答案巧合命中的样本给出 ✓ 时标红报警。
  - 表中「答案对」与「每步都对」: 所选最高分候选解的最终答案准确率与全链路步骤准确率。
  - 表中「600 步后 BCE」: 奖励模型训练 600 步后的二元交叉熵损失, 越接近 0.693 (即 $\ln 2$) 说明判别能力越接近随机猜测。"
    module="llm_finetune/methods/prm.py"
    run="python -m llm_finetune.run_finetune.prm.train_prm"
    :challenge="{
      ask: '在「蒙对的例子」中, 第 1 步计算出错, 第 2 步执行 ×2 运算恰好消除了数值偏差:\n1. ORM 给出的监督标签是什么?\n2. PRM 给出的四个步骤标签依次是什么?\n3. 在留出集的候选解中, 这类「过程出错但最终答案碰巧正确」的样本占比是多少?',
      answer: '答案:\n1. ORM 的监督标签为 1。\n2. PRM 的四个步骤标签依次为 ✗ ✓ ✓ ✓。\n3. 这类「过程出错但最终答案碰巧正确」的样本在留出集候选解中占 4.5% (由 train_prm.py 在 test 划分上统计所得)。\n原因解析:\n- ORM 仅检查最终结果: 末行输出答案与真值一致时标签即为 1, 这类错误推理过程在训练中会被当作正样本强化。\n- PRM 按步骤局部校验, 标注结果为 ✗ ✓ ✓ ✓:\n  - 第 1 步: 正确算式为 $3 + 4 = 7$, 候选解写为 2, 故标为 ✗。\n  - 第 2 步: 基于上一步写出的 2 继续计算, $2 \\times 2 = 4$, 与候选解写出值一致, 故标为 ✓。\n  - 后续步骤同理按局部推导一致性判别。\n- PRM 对整条解答取各步分数的最小值聚合, 该样本因第 1 步为 0 而被整体判为 0 分, 从而被有效剔除。\n- 训练收敛差异: ORM 只能从单一的句级 0/1 标签中反推错误位置, 信噪比极低。训练 600 步后 ORM 的 BCE 损失仍高达 0.682, 接近随机猜测水平 ($\\ln 2 \\approx 0.693$); 而 PRM 的 BCE 损失降至 0.359。在 best-of-8 选解评测中, PRM 选中正确答案的比例达 0.648, 显著高于 ORM 的 0.488 与随机抽选基线的 0.473。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="lucky">蒙对的例子</button>
        <button type="button" @click="reseed">换一道题</button>
        <button type="button" @click="err = [0, 0, 0, 0]">全部写对</button>
      </div>
    </template>

    <table class="chain mono">
      <thead><tr><th>步</th><th>解里写的</th><th>真值</th><th>PRM</th><th /></tr></thead>
      <tbody>
        <tr><td>0</td><td>a0 = {{ q.a0 }}</td><td>{{ q.a0 }}</td><td /><td /></tr>
        <tr v-for="(s, i) in steps" :key="i" :class="{ bad: !s.ok }">
          <td>{{ i + 1 }}</td>
          <td>{{ s.prev }} {{ OPS[q.ops[i]] }} {{ q.ks[i] }} = <b>{{ s.v }}</b></td>
          <td>{{ s.truth }}</td>
          <td><span class="lbl" :class="s.ok ? 'good' : 'bad'">{{ s.ok ? '✓ 1' : '✗ 0' }}</span></td>
          <td><button type="button" :class="{ active: err[i] }" @click="toggle(i)">{{ err[i] ? '改回对' : '写错' }}</button></td>
        </tr>
        <tr class="orm"><td>EOS</td><td>答案 {{ steps[3].v }}</td><td>{{ steps[3].truth }}</td><td colspan="2">ORM <span class="lbl" :class="outcome ? 'good' : 'bad'">{{ outcome ? '✓ 1' : '✗ 0' }}</span></td></tr>
      </tbody>
    </table>
    <p class="lab-note" style="margin-top: 8px;">
      {{ luckyNow ? '过程错、答案蒙对: ORM 会把这条当成好解, PRM 取各步最小值会把它拉下来。'
        : firstErr < 0 ? '每步都对: 两种标签一致, 都说这是好解。' : `第一个错步是第 ${firstErr + 1} 步。PRM 在这一步的标签为 0, 直接指出错在哪; ORM 只知道「整条不行」。` }}
    </p>

    <template #stats>
      <div class="kv"><span>ORM 拿到的标签 (整条 1 个: 答案对不对)</span><b :class="outcome ? 'good' : 'bad'">{{ outcome ? '✓' : '✗' }}</b></div>
      <div class="kv"><span>PRM 拿到的标签 (每步 1 个, 共 4 个)</span><b>{{ steps.map((s) => (s.ok ? '✓' : '✗')).join(' ') }}</b></div>
      <div class="kv"><span>PRM 整条得分 (各步最小值)</span><b :class="firstErr < 0 ? 'good' : 'bad'">{{ firstErr < 0 ? 1 : 0 }}</b></div>
      <div class="kv"><span>两种标签说的是一回事吗</span><b :class="luckyNow ? 'bad' : 'good'">{{ luckyNow ? 'ORM 误判' : '一致' }}</b></div>
      <p class="tcap">train_prm.py best-of-8 挑解 (实测, 不随左侧变化)</p>
      <table class="res mono">
        <thead><tr><th>打分器</th><th>答案对</th><th>每步都对</th><th>600 步后 BCE</th></tr></thead>
        <tbody>
          <tr v-for="p in PICK" :key="p.id"><td>{{ p.id }}</td><td>{{ p.ans.toFixed(3) }}</td><td>{{ p.proc.toFixed(3) }}</td><td>{{ p.bce }}</td></tr>
        </tbody>
      </table>
      <div class="lab-note">
        <p>评测方法: 在留出集 512 道题上, 每题生成 8 条带噪候选解 (每步以 20% 概率注入计算错误), 由奖励模型选出得分最高的一条, 分别统计其最终答案准确率与全步骤准确率。「随机」行代表从中随机抽取一条候选解的基线。</p>
        <p>每题 8 条候选解中至少包含一条正确答案的比例 (pass@8 上限) 为 0.992, 因此当前 PRM 取得的 0.648 距理论上限仍有较大空间。</p>
        <p>PRM 同时可用于定位首个出错步骤, 定位准确率为 0.575; 而恒定猜测「第 1 步出错」的基线准确率仅为 0.341。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, range } from '@/utils/labmath.js'

const OPS = ['+', '−', '×']
const apply = (op, a, k) => (((op === 0 ? a + k : op === 1 ? a - k : a * k) % 10) + 10) % 10
// 来自 train_prm.py 输出的 best-of-8 表
const PICK = [
  { id: '随机', ans: 0.473, proc: 0.426, bce: '—' },
  { id: 'ORM', ans: 0.488, proc: 0.422, bce: '0.682' },
  { id: 'PRM', ans: 0.648, proc: 0.627, bce: '0.359' },
]

const make = (s) => {
  const rand = mulberry32(s * 7919)
  return { a0: Math.floor(rand() * 10), ops: range(4).map(() => Math.floor(rand() * 3)), ks: range(4).map(() => 1 + Math.floor(rand() * 9)), off: range(4).map(() => 1 + Math.floor(rand() * 9)) }
}
const seed = ref(1), q = ref(make(1)), err = ref([0, 1, 0, 0])
const reseed = () => { seed.value++; q.value = make(seed.value); err.value = [0, 1, 0, 0] }
const toggle = (i) => { err.value = err.value.map((e, k) => (k === i ? 1 - e : e)) }
// 3 +4 → 7, 写成 2 (差 5); ×2 后 4 = 4, 差距被抹平
const lucky = () => { q.value = { a0: 3, ops: [0, 2, 1, 0], ks: [4, 2, 3, 6], off: [5, 1, 1, 1] }; err.value = [1, 0, 0, 0] }

const steps = computed(() => {
  let truth = q.value.a0, written = q.value.a0
  return range(4).map((i) => {
    const { ops, ks, off } = q.value, prev = written
    truth = apply(ops[i], truth, ks[i])
    const right = apply(ops[i], written, ks[i])       // ★ 照着解里写的上一步算
    written = err.value[i] ? (right + off[i]) % 10 : right
    return { prev, v: written, truth, ok: !err.value[i] }
  })
})
const outcome = computed(() => steps.value[3].v === steps.value[3].truth)
const firstErr = computed(() => steps.value.findIndex((s) => !s.ok))
const luckyNow = computed(() => outcome.value && firstErr.value >= 0)
</script>

<style scoped>
.chain { border-collapse: collapse; font-size: 12px; }
.chain th { font-size: 11px; color: var(--text-dim); font-weight: 400; text-align: left; padding: 4px 5px; }
.chain td { padding: 4px 5px; border-top: 1px solid var(--border); color: var(--text-muted); }
.chain tr.bad td:nth-child(2) b { color: var(--danger); }
.chain tr.orm td { border-top: 1px solid var(--border-strong); }
.chain button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.lbl.good { color: var(--left); }
.lbl.bad { color: var(--danger); }
.tcap { font-size: 11px; color: var(--text); margin-top: 4px; }
.res { width: 100%; border-collapse: collapse; font-size: 11px; }
.res th { font-size: 10.5px; color: var(--text-dim); font-weight: 400; text-align: right; padding: 3px 4px; }
.res td { padding: 3px 4px; text-align: right; color: var(--text-muted); border-top: 1px solid var(--border); }
.res td:first-child, .res th:first-child { text-align: left; }
</style>
