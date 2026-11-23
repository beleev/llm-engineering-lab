<!--
  模型合并实验台。上半: 8 维玩具任务向量, 前端实时跑 llm_finetune/methods/merge.py 的同一套逐坐标规则
  (task_arithmetic / ties_merge / dare)。下半: python -m llm_finetune.run_finetune.merge.train_merge 的留出集结果表。
  只讲一件事: 合并 = 对任务向量做逐坐标处理再加回基座; λ 决定每个任务 "加进去几成"。
-->
<template>
  <LabFrame
    title="模型合并 — 把两个任务向量相加、TIES 修剪、DARE 随机丢弃"
    sub="- 模拟什么: 在同一预训练基座 $\theta_0$ 上分别独立微调得到专精不同子任务的模型 A 与模型 B 后, 模型合并旨在无需联合重训的前提下直接融合二者能力。定义任务向量 $\tau = \theta_{\text{ft}} - \theta_0$ 表示微调引入的参数空间位移; 合并算法通过对各任务向量执行逐坐标处理后再按缩放系数 $\lambda$ 叠加回基座: $\theta = \theta_0 + \lambda\cdot(\text{处理后的 } \tau_A + \tau_B)$。前端交互面板运行与本仓库 merge.py 完全一致的逐坐标算子, 仅将参数维度浓缩至 8 维以便逐元素观察。三类代表性处理算子如下:
  - Task Arithmetic: 对多个任务向量直接逐坐标线性相加。
  - TIES: 先按幅值排序仅保留各向量绝对值最大的一定比例坐标 (修剪微小噪声), 再在各坐标上按总幅值做正负符号投票, 剔除与获胜符号相反的冲突分量后求均值。
  - DARE: 以概率 $p$ 随机对任务向量坐标置零丢弃, 并将保留下来的非零坐标除以 $1-p$ 重缩放以保持线性输出期望不变。
- 图上是什么: 上方主视图包含三行各 8 个参数网格, 绿色表示正增量、红色表示负增量, 颜色越深绝对值越大:
  - 前两行: 分别对应任务向量 $\tau_A$ 与 $\tau_B$; 其中任务 A 的有效增量集中于前 4 维, 任务 B 集中于后 4 维, 并在对方主维度上带有少量微小扰动噪声。
  - 第三行: 经当前合并规则处理并乘以 $\lambda$ 后实际叠加回基座的合成位移 $\theta - \theta_0$。
  - 下方固定对比表: 展示 train_merge.py 在约 10 万参数真实模型留出集上的评测数据 (不随上方 8 维玩具交互变动)。
- 可以做什么: 支持以下交互操作:
  - 点击前两行中任意参数网格: 翻转该坐标的正负号, 人为构造跨任务符号冲突。
  - 点击顶部方法按钮: 在不同合并算法间切换。
  - 拖动缩放系数 $\lambda$ 滑杆: 调整叠加回基座的任务向量整体幅度比例。
  - 在 TIES 模式下可额外拖动「保留比例」滑杆; 在 DARE 模式下可拖动「丢弃率 p」滑杆并点击「换一个 DARE 掩码」重抽随机伯努利掩码。
  - 点击下方实测表格任意一行: 上方 8 维示意面板自动同步切换至对应算法 (玩具面板的 $\lambda$ 仅在 0.5 或 1 之间切换, 不直接套用表格在验证集上搜索出的最优 $\lambda$)。
- 应该看到: 在 Task Arithmetic 且 $\lambda = 1$ 配置下, 第三行严格等于前两行对应网格的逐元素代数和。右侧汇总读数与下方表格列含义如下:
  - 「A 段改到几成」: 合成向量在前 4 维主子空间上相对原始 $\tau_A$ 的投影保留比例, 1 代表完整保留, 0.5 代表幅度减半。
  - 「B 段改到几成」: 合成向量在后 4 维主子空间上相对原始 $\tau_B$ 的保留比例。
  - 「符号冲突」: 8 个坐标维中 $\tau_A$ 与 $\tau_B$ 非零且正负号相反的维度个数。
  - 「cos(τ_A, τ_B)」: 两个任务向量之间的夹角余弦相似度, 取 0 代表正交。
  - 表格「A 段」与「B 段」: 合并后模型在两个子任务各自片段上的准确率。
  - 表格「两段全对 EM」: 要求单条复合测试样本同时答对 A、B 两段的完全匹配率。
  - 表格「干扰」: 合并模型在两段上的平均准确率相对各自单任务微调模型 (准确率均为 1.0) 的平均回落幅度。"
    module="llm_finetune/methods/merge.py"
    run="python -m llm_finetune.run_finetune.merge.train_merge"
    :challenge="{
      ask: '请按顺序切换配置并对比下方实测表:\n1. 选择「简单平均」, 观察右侧「A 段改到几成」读数。\n2. 切换至 Task Arithmetic 并将缩放系数 $\\lambda$ 拉至 1。\n为什么真实模型实验中「简单平均」的两段全对 EM 仅为 0.039, 而 $\\lambda = 1$ 的 Task Arithmetic 可达 1.000? 旨在消除参数干扰的 TIES 与 DARE 为何在本实验规模下反而未取得优势?',
      answer: '结论:\n1. 权重简单平均等价于取 $\\lambda = 0.5$ 的任务向量相加, 两个子任务各自的参数更新幅度均被折半, 导致两段全对 EM 仅为 0.039。\n2. 取 $\\lambda = 1$ 的 Task Arithmetic 将两项近似正交的任务位移完整叠加回基座, 留出集两段全对 EM 达到 1.000。\n3. TIES 与 DARE 依赖高维过参数化大模型中的海量参数冗余来裁剪与稀疏化, 而约 10 万参数的小模型缺乏此类冗余容量, 裁剪或丢弃会直接损伤有效功能分量。\n机制解析:\n- 简单平均的幅度折半效应: 对完整模型权重求算术平均 $(\\theta_A + \\theta_B)/2 = \\theta_0 + 0.5(\\tau_A + \\tau_B)$, 等效于将每个任务向量缩放至 0.5 倍后加回基座。此时「A 段改到几成」读数约为 0.5, 两个子任务的微调位移均未达到激活正确输出所需的阈值, 实测复合任务 EM 仅为 0.039。\n- 正交子空间的无损叠加: 本实验中两个任务各自集中修改近似正交的参数子空间, 在对方子空间上的干扰分量微弱, 因此直接以 $\\lambda = 1$ 执行线性相加即可同时完整恢复两项任务能力, 实测 EM 达 1.000。\n- TIES 的小分量误伤: 在保留比例 0.5 下, 按幅值硬阈值修剪会一并抹除对精细决策仍有贡献的中小幅值参数更新, 使实测 EM 回落至 0.734。\n- DARE 的高丢弃方差: 在丢弃率 90% 下, 小维度矩阵上单次伯努利随机掩码与 $1/(1-p)$ 放大引入了剧烈输出方差, 实测 EM 跌至 0.066; 将丢弃率收窄至 50% 时 EM 仍可维持在 0.875。\n- 规模与过参数化边界: TIES 与 DARE 旨在治理数十亿参数级大模型微调时大量冗余参数坐标上的跨任务符号干扰; 在约 10 万参数的紧凑模型上几乎每个非零参数均承担实际功能, 朴素的线性任务向量相加反而最为稳健。',
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
      <p>下方表格为 train_merge.py 在约 10 万参数真实模型留出集上的固定评测结果 (不随上方 8 维示意面板变动); 点击任意表格行可将上方示意面板同步切换至同一种合并方法。</p>
      <p>表格中的缩放系数 λ 由各方法在验证集上独立网格搜索选出; 上方示意面板的 λ 仅取 0.5 或 1 两档, 因此点击 TIES (保留比例 d 为 0.5, 表中 λ 取 1.5) 与 DARE (丢弃率 p 为 0.9, 表中 λ 取 0.7) 两行时, 滑杆均停留在 1。</p>
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
      <div class="kv"><span>A 段改到几成 (玩具, 1 = 原样加回)</span><b :class="Math.abs(st.a - 1) < 0.1 ? 'good' : 'bad'">{{ st.a.toFixed(2) }}</b></div>
      <div class="kv"><span>B 段改到几成 (玩具, 1 = 原样加回)</span><b :class="Math.abs(st.b - 1) < 0.1 ? 'good' : 'bad'">{{ st.b.toFixed(2) }}</b></div>
      <div class="kv"><span>符号冲突的坐标数 (玩具)</span><b>{{ st.conflict }} / 8</b></div>
      <div class="kv"><span>cos(τ_A, τ_B) (玩具, 0 = 正交)</span><b>{{ st.cos.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>真实模型实测统计显示, 两个任务向量的夹角余弦 cos(τ_A, τ_B) 为 +0.046, 而双方同时非零的参数坐标中符号冲突比例高达 47.4%: 全局几何方向近似正交, 微观逐坐标正负号却有近半数相反。TIES 正是针对此类微观符号冲突设计, 但在当前小参数规模下强行裁剪冲突分量反而造成精度损失。</p>
        <p>各方法的缩放系数 λ 在验证集上从候选集合 {0.5, 0.7, 1, 1.5} 中择优确定, 表中汇报留出集上的最终测试指标。SLERP 在两份完整模型权重之间沿超球面大圆弧做球面线性插值; 本实验中两份完整权重夹角仅为 0.333 rad, 球面弧线与弦线高度接近, 因此表现与简单算术平均基本一致。</p>
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
