<!--
  模型合并实验台。上半: 8 维玩具任务向量, 前端实时跑 llm_finetune/methods/merge.py 的同一套逐坐标规则
  (task_arithmetic / ties_merge / dare)。下半: python -m llm_finetune.run_finetune.merge.train_merge 的留出集结果表。
  只讲一件事: 合并 = 对任务向量做逐坐标处理再加回基座; λ 决定每个任务 "加进去几成"。
-->
<template>
  <LabFrame
    title="合并两个任务向量: 相加、TIES、DARE"
    sub="$\tau_A$ 主要改前 4 维, $\tau_B$ 主要改后 4 维, 各带一点对方那段的小噪声。点格子翻转符号, 制造冲突。
      第三行是合并后加回基座的量 $\theta - \theta_0$。下面的表格是训练脚本的实测结果, 点一行把玩具切到同一种方法。"
    module="llm_finetune/methods/merge.py"
    run="python -m llm_finetune.run_finetune.merge.train_merge"
    :challenge="{
      ask: '选「简单平均」, 看「A 段改到几成」。再选 Task Arithmetic 并把 λ 拉到 1。为什么真实实验里前者两段全对 EM 只有 0.039, 后者 1.000? TIES、DARE 又为什么没赢?',
      answer: '简单平均就是 $\\lambda = 0.5$: $\\theta_0 + 0.5(\\tau_A + \\tau_B) = (\\theta_A + \\theta_B)/2$。每个任务向量只加了一半, 两段都只改到一半。\n本例两个任务各管一段, 对方那段的分量很小, 直接 $\\lambda = 1$ 相加就是两个完整的改动。\nTIES、DARE 没赢的原因:\n- TIES: 修掉一半坐标时, 也会剪掉有用的小分量 (0.734)。\n- DARE: 丢 90% 时, 单次随机掩码的方差太大 (0.066)。\n它们修的是大模型里大量冗余坐标的冲突。约 10 万参数的玩具任务向量里, 没有这种冗余。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lam" label="λ" :min="0" :max="2" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-if="method === 'ties'" v-model="density" label="TIES 保留比例" :min="0.25" :max="1" :step="0.125" :format="(v) => v.toFixed(3)" />
      <LabSlider v-if="method === 'dare'" v-model="p" label="DARE 丢弃率 p" :min="0" :max="0.9" :step="0.1" :format="(v) => v.toFixed(1)" />
      <div class="row">
        <button v-for="m in TOY" :key="m.id" type="button" :class="{ active: method === m.id }" @click="method = m.id">{{ m.name }}</button>
        <button v-if="method === 'dare'" type="button" @click="seed++">换一个 DARE 掩码</button>
      </div>
    </template>

    <div class="vecs">
      <div v-for="row in vecRows" :key="row.id" class="vrow">
        <span class="vl"><Tex :text="row.tex" /></span>
        <template v-for="(x, i) in row.v" :key="i">
          <button v-if="row.edit" type="button" class="cell vc" :style="cellStyle(x)" :aria-label="`翻转 ${row.id} 第 ${i + 1} 维的符号`" @click="flip(row.id, i)">{{ x.toFixed(2) }}</button>
          <span v-else class="cell vc" :style="cellStyle(x)">{{ x.toFixed(2) }}</span>
        </template>
      </div>
    </div>

    <div class="lab-note tcap">
      <p>train_merge.py 留出集 (实测, 不随上面的玩具变化)。</p>
      <p>表里的 λ 是各方法在验证集上挑出来的。玩具的 λ 只取 0.5 或 1: 点 TIES d=0.5 (表里 1.5) 和 DARE p=0.9 (表里 0.7) 时, 滑杆停在 1。</p>
    </div>
    <table class="res mono">
      <thead><tr><th /><th>λ</th><th>A 段</th><th>B 段</th><th>两段全对 EM</th><th>干扰</th></tr></thead>
      <tbody>
        <tr v-for="r in MEASURED" :key="r.id" :class="{ sel: sel === r.id }" tabindex="0" @click="choose(r)" @keydown.enter="choose(r)">
          <td>{{ r.name }}</td><td>{{ r.lam ?? '–' }}</td><td>{{ r.a.toFixed(3) }}</td><td>{{ r.b.toFixed(3) }}</td>
          <td :class="r.em > 0.9 ? 'good' : r.em < 0.1 ? 'bad' : ''">{{ r.em.toFixed(3) }}</td><td>{{ r.drop === null ? '–' : r.drop.toFixed(3) }}</td>
        </tr>
      </tbody>
    </table>

    <template #stats>
      <div class="kv"><span>A 段改到几成 (玩具)</span><b :class="Math.abs(st.a - 1) < 0.1 ? 'good' : 'bad'">{{ st.a.toFixed(2) }}</b></div>
      <div class="kv"><span>B 段改到几成 (玩具)</span><b :class="Math.abs(st.b - 1) < 0.1 ? 'good' : 'bad'">{{ st.b.toFixed(2) }}</b></div>
      <div class="kv"><span>符号冲突 (玩具, 两者都非零)</span><b>{{ st.conflict }} / 8</b></div>
      <div class="kv"><span>cos(τ_A, τ_B) (玩具)</span><b>{{ st.cos.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>真实实验: cos(τ_A, τ_B) = +0.046, 两者都非零的坐标里符号冲突 47.4%。整体几乎正交, 逐坐标却接近一半冲突。</p>
        <p>λ 在验证集上从 {0.5, 0.7, 1, 1.5} 里挑, 表里是留出集。SLERP 插的是完整权重, 夹角 0.333 rad, 结果 ≈ 简单平均。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const TOY = [{ id: 'ta', name: 'Task Arithmetic' }, { id: 'ties', name: 'TIES' }, { id: 'dare', name: 'DARE' }]
// train_merge.py 输出的留出集表 (λ 按验证集挑)。toy = 点这一行时玩具切到的设置; 玩具的 λ 只区分 0.5 / 1, 不照搬真实实验挑出的 λ
const MEASURED = [
  { id: 'A', name: '只微调 A', lam: null, a: 1, b: 0, em: 0.004, drop: null },
  { id: 'B', name: '只微调 B', lam: null, a: 0, b: 1, em: 0.027, drop: null },
  { id: 'avg', name: '简单平均', lam: 0.5, a: 0.423, b: 0.616, em: 0.039, drop: 0.48, toy: ['ta', 0.5] },
  { id: 'ta', name: 'Task Arithmetic', lam: 1, a: 1, b: 1, em: 1, drop: 0, toy: ['ta', 1] },
  { id: 'ties', name: 'TIES d=0.5', lam: 1.5, a: 0.947, b: 0.948, em: 0.734, drop: 0.053, toy: ['ties', 1, 0.5] },
  { id: 'dare5', name: 'DARE p=0.5', lam: 1, a: 1, b: 0.955, em: 0.875, drop: 0.023, toy: ['dare', 1, 0.5] },
  { id: 'dare9', name: 'DARE p=0.9', lam: 0.7, a: 0.584, b: 0.573, em: 0.066, drop: 0.422, toy: ['dare', 1, 0.9] },
  { id: 'slerp', name: 'SLERP t=0.5', lam: null, a: 0.51, b: 0.739, em: 0.09, drop: 0.376, toy: ['ta', 0.5] },
]
const method = ref('ta'), lam = ref(1), density = ref(0.5), p = ref(0.5), seed = ref(1), sel = ref('ta')
const tA = ref([1.2, -0.9, 1.0, 0.12, 0.05, -0.1, 0.05, -0.05])
const tB = ref([0.05, 0.1, -0.05, 0.1, -1.1, 0.9, -1.0, 0.14])
const flip = (id, i) => { const t = id === 'A' ? tA : tB; t.value = t.value.map((x, k) => (k === i ? -x : x)) }
const choose = (r) => {
  sel.value = r.id
  if (!r.toy) return
  ;[method.value, lam.value] = r.toy
  if (r.toy[0] === 'ties') density.value = r.toy[2]
  if (r.toy[0] === 'dare') p.value = r.toy[2]
}

// TIES: 每个任务留 |τ| 的 top-k → 符号按修剪后求和投票 → 只对非零且同号的任务取平均 (同 merge.py:ties_merge)
const ties = (taus, d) => {
  const k = Math.max(1, Math.floor(d * 8))
  const trimmed = taus.map((t) => { const th = [...t].map(Math.abs).sort((a, b) => b - a)[k - 1]; return t.map((x) => (Math.abs(x) >= th ? x : 0)) })
  return range(8).map((i) => {
    const s = Math.sign(sum(trimmed.map((t) => t[i])))
    const agree = trimmed.map((t) => t[i]).filter((x) => x !== 0 && Math.sign(x) === s)
    return agree.length ? sum(agree) / agree.length : 0      // ★ 冲突的一方不参与, 而不是被平均稀释
  })
}
const dare = (taus, pp) => { const rand = mulberry32(seed.value * 131); return taus.map((t) => t.map((x) => (rand() >= pp ? x / (1 - pp) : 0))) }

const delta = computed(() => {
  const taus = [tA.value, tB.value]
  const m = method.value === 'ties' ? ties(taus, density.value)
    : range(8).map((i) => sum((method.value === 'dare' ? dare(taus, p.value) : taus).map((t) => t[i])))
  return m.map((x) => lam.value * x)                          // θ − θ₀ = λ · (处理后的任务向量)
})
const vecRows = computed(() => [
  { id: 'A', tex: '$\\tau_A$', v: tA.value, edit: true },
  { id: 'B', tex: '$\\tau_B$', v: tB.value, edit: true },
  { id: 'M', tex: '$\\theta - \\theta_0$', v: delta.value, edit: false },
])
const st = computed(() => {
  // 改到几成 = 合并量在该任务自己那段上对 τ 的投影系数; 1 = 原样加回, 0.5 = 只加了一半
  const own = (t, idx) => sum(idx.map((i) => delta.value[i] * t[i])) / sum(idx.map((i) => t[i] * t[i]))
  const dot = sum(range(8).map((i) => tA.value[i] * tB.value[i]))
  const nrm = (t) => Math.sqrt(sum(t.map((x) => x * x)))
  return {
    a: own(tA.value, [0, 1, 2, 3]), b: own(tB.value, [4, 5, 6, 7]),
    conflict: range(8).filter((i) => Math.sign(tA.value[i]) !== Math.sign(tB.value[i])).length,
    cos: dot / (nrm(tA.value) * nrm(tB.value)),
  }
})
const cellStyle = (x) => ({ background: `color-mix(in srgb, ${x >= 0 ? 'var(--left)' : 'var(--danger)'} ${Math.min(100, Math.abs(x) * 60)}%, transparent)` })
</script>

<style scoped>
.vecs { display: flex; flex-direction: column; gap: 4px; margin-bottom: 14px; }
.vrow { display: grid; grid-template-columns: 64px repeat(8, 46px); gap: 3px; align-items: center; }
.vl { font-size: 12px; color: var(--text-muted); }
.vc { min-width: 0; min-height: 0; width: 46px; height: 26px; padding: 0; font-size: 10px; color: var(--text); }
button.vc { cursor: pointer; }
.tcap { margin-bottom: 6px; }
.res { border-collapse: collapse; font-size: 12px; min-width: 460px; }
.res th { font-size: 11px; color: var(--text-dim); font-weight: 400; text-align: right; padding: 3px 6px; }
.res td { padding: 3px 6px; text-align: right; color: var(--text-muted); border-top: 1px solid var(--border); }
.res td:first-child, .res th:first-child { text-align: left; }
.res tr { cursor: pointer; }
.res tr.sel td { background: var(--bg-elev); color: var(--text); }
.res td.good { color: var(--left); }
.res td.bad { color: var(--danger); }
</style>
