<!--
  PRM vs ORM 实验台。标签规则与 llm_finetune/methods/prm.py:ArithChain.sample 相同:
  第 i 步 "对" = 写下的 v_i 等于 op_i(解里写的 v_{i−1}, k_i); 最终答案对 = 写下的 v_K 等于真值。
  只讲一件事: 同一条解, ORM 拿到 1 个标签, PRM 拿到 K 个; 过程错、答案蒙对时 ORM 会被骗。
  右侧 best-of-8 表格来自 python -m llm_finetune.run_finetune.prm.train_prm 的输出表, 不随左侧变化。
-->
<template>
  <LabFrame
    title="PRM vs ORM — 同一条解, 结果奖励看到 1 个标签, 过程奖励看到 4 个"
    sub="- 模拟什么: 多步推理题的打分器有两种。ORM (结果奖励模型) 只看最终答案对不对, 整条解一个标签; PRM (过程奖励模型) 给每一步各打一个标签。这个台子用模 10 的 4 步算术链当题目: 从 a0 出发, 每步做一次加、减或乘再取模 10。标签规则和 prm.py 一致: 第 i 步「对」是指它写下的数等于拿解里上一步写下的数 (不是真值) 算出来的结果; 最终答案「对」是指第 4 步写下的数等于真值。
      - 图上是什么: 一张表, 每行一步。「解里写的」是这条解实际写下的算式和结果 (写错的结果标红); 「真值」是从头全算对时该是多少; 「PRM」列是这一步的标签, ✓ 1 或 ✗ 0; 最后一行 EOS 是整条解的最终答案, 旁边是 ORM 的标签。表下面一句话说明两种标签现在是否一致。
      - 可以做什么: 点某一步的「写错」把这一步写下的值改错, 后面几步照着错的值继续算 (像真人的草稿), 再点一次改回来; 「全部写对」清掉所有错误; 「换一道题」换随机数; 「蒙对的例子」载入一条特别构造的解: 第 1 步写错, 第 2 步 ×2 刚好把差距抹平, 最终答案碰巧对了。
      - 应该看到: 默认第 2 步写错, PRM 的标签是 ✓ ✗ ✓ ✓ (第 3、4 步照着错值算是对的), ORM 只给一个 ✗。「蒙对的例子」里 ORM 给 ✓, PRM 给 ✗ ✓ ✓ ✓。右侧读数: 「ORM 拿到的标签」和「PRM 拿到的标签」就是两种打分器的训练目标; 「PRM 整条得分」取各步最小值, 有一步错就是 0; 「两种标签说的是一回事吗」在 ORM 被蒙对的解骗到时变红。下面的表是 train_prm.py 的实测 best-of-8 结果 (每题采 8 个候选解, 用打分器挑分最高的那条): 「答案对」和「每步都对」是挑中的解的两种正确率, 「600 步后 BCE」是打分器的训练 loss, 越接近 0.693 (= ln 2) 越像在瞎猜。"
    module="llm_finetune/methods/prm.py"
    run="python -m llm_finetune.run_finetune.prm.train_prm"
    :challenge="{
      ask: '在「蒙对的例子」里, 第 1 步写错, 第 2 步 ×2 把差距抹平了。ORM 的标签是什么? PRM 四个标签是什么? 训练数据里这种解占多少?',
      answer: '答案: ORM 的标签是 1, PRM 的四个标签是 ✗ ✓ ✓ ✓, 这种「过程错、答案蒙对」的解在候选里占 4.5%。\n一步步看:\n- ORM 只看最后一行: 写下的答案等于真值, 标签就是 1, 它被教成「这是好解」。\n- PRM 逐步看: 第 1 步该写 3 + 4 = 7, 解里写的是 2, 标 ✗; 第 2 步照着写下的 2 算, 2 × 2 = 4, 和它写的一致, 标 ✓; 后面同理。所以是 ✗ ✓ ✓ ✓。\n- PRM 给整条解打分取各步最小值, 这条解会被第 1 步拖成 0, 不会被当成好解。\n- 训练难度: ORM 只能从一个 0/1 里自己推出错在哪。600 步后 ORM 的 BCE 是 0.682, 几乎还在猜 (瞎猜是 ln 2 = 0.693); PRM 降到 0.359。best-of-8 挑解时 PRM 挑中答案对的解的比例是 0.648, ORM 0.488, 随机挑 0.473。',
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
      <div class="kv"><span>两种标签说的是一回事吗</span><b :class="luckyNow ? 'bad' : 'good'">{{ luckyNow ? 'ORM 被骗' : '一致' }}</b></div>
      <p class="tcap">train_prm.py best-of-8 挑解 (实测, 不随左侧变化)</p>
      <table class="res mono">
        <thead><tr><th>打分器</th><th>答案对</th><th>每步都对</th><th>600 步后 BCE</th></tr></thead>
        <tbody>
          <tr v-for="p in PICK" :key="p.id"><td>{{ p.id }}</td><td>{{ p.ans.toFixed(3) }}</td><td>{{ p.proc.toFixed(3) }}</td><td>{{ p.bce }}</td></tr>
        </tbody>
      </table>
      <div class="lab-note">
        <p>怎么测的: 留出集 512 题, 每题 8 个带噪候选解 (每步 20% 概率写错), 用打分器挑分最高的那条, 看它答案对不对、每步都对不对。「随机」一行是随便挑一条的基线。</p>
        <p>8 个候选里至少有一个答案对的比例是 0.992, 所以 PRM 的 0.648 离上限还远。</p>
        <p>PRM 还能指出第一个错步在哪, 准确率 0.575; 一律猜「第 1 步」只有 0.341。</p>
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
