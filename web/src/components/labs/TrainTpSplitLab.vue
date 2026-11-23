<!--
  张量并行切法 (对应 llm_train/m03_tensor_parallel)。
  只讲一件事: Y = GeLU(X·A)·B, A 列切 + B 行切时, 中间不需要任何通信, 只在最后 all-reduce 一次。
  两张卡上的每个小矩阵都是真算出来的, Δ 是和不切分的 dense 结果逐元素比出来的。
-->
<template>
  <LabFrame
    title="张量并行 — 一层矩阵切给两张卡, 通信到底该插在哪?"
    sub="- 模拟什么: 张量并行 (TP) 将单层内部的大尺寸权重矩阵切分至多张卡上协同计算, 适用于单层参数或中间激活宽到单卡显存无法容纳的场景。本实验台用两张卡协同计算一个双层 MLP 模块 $Y = \mathrm{GeLU}(X \cdot A) \cdot B$: 输入激活 X 为 2×4 矩阵, 第一层权重 A 为 4×8 升维矩阵 (将特征维从 4 扩展至 8), 第二层权重 B 为 8×4 降维矩阵 (将特征维还原回 4), 两层之间夹有逐元素非线性激活函数 GeLU。矩阵切分后究竟应在何处插入集合通信、最少需要通信几次, 正是 Megatron 张量并行架构所解决的核心问题。每个权重矩阵均支持两种切分维度:
  - 按列切分: 每张卡持有输出维度的一半列, 独立乘完输入后得到完整输出矩阵的一段列分片。
  - 按行切分: 每张卡持有输入维度的一半行, 独立相乘后输出矩阵形状虽完整但数值仅为部分和, 必须将两卡结果逐元素相加才等于真实输出。
- 图上是什么: 从上至下依次排列前向计算的五个阶段: X、X·A、GeLU、·B 与 Y。
  - 每个阶段并列画出卡 0 与卡 1 当前各自持有的局部矩阵, 单个小色块代表一个标量元素 (蓝色为正值、红色为负值, 颜色越深代表绝对值越大)。
  - 每阶段右侧小字标明当前张量的分布状态: 「完整」代表每张卡均持有完整副本; 「一段列」代表每张卡仅持有部分列分片, 横向拼接后才构成完整矩阵; 「部分和」代表矩阵形状完整但数值仅为局部贡献, 跨卡求和后才等于真实值。
  - 相邻阶段之间的虚线按钮代表 ①②③ 三个候选同步点, 点击后才会在该位置插入集合通信, 按钮文字会自动标明插入的原语是 all-gather (拼接各卡列分片还原完整张量) 还是 all-reduce (累加各卡部分和并使每卡均持有总和)。
  - 若某一步因前后矩阵维度不匹配而无法相乘, 该阶段会标红并提示维度冲突原因, 后续阶段随之暂停渲染。
- 可以做什么: 控件从上到下依次为:
  - 分别选择第一层权重 A 采用按列切分还是按行切分, 以及第二层权重 B 采用按列切分还是按行切分。
  - 点击图中的三个候选同步点按钮以插入或移除集合通信。
  - 点击「换一组数」重新生成随机输入与权重矩阵并重算结果。
  - 点击「清空同步点」一键关闭全部三个候选同步点。
- 应该看到: 设计目标是令输出误差 Δ 严格为 0 且总通信开销最低。当选择 A 按列切分、B 按行切分且仅开启末尾同步点 ③ 时, 计算结果完全正确, 全程仅需 1 次集合通信, 单卡发送量为 1.00×。反之若将 A 设为按行切分, 则必须在进入非线性函数前的同步点 ① 处先行执行一次 all-reduce, 且该次通信作用在维度最宽的中间激活张量上, 通信开销翻倍。右侧读数含义如下:
  - 「与 dense 的 max|Δ|」: 两张卡最终所得输出 Y 与单卡不切分完整计算的基准输出之间的最大绝对偏差, 读数为 0 方代表计算等价。
  - 「结论」: 直接判定当前切分与同步组合处于维度不匹配、缺少同步通信、数值不等价还是完全正确状态。
  - 「通信次数」: 当前配置中实际生效的集合通信同步点数量。
  - 「每卡发送量」: 以输入矩阵 X 的字节体积为单位归一化折算的单卡总发送数据量。"
    module="llm_train/m03"
    run="python -m llm_train.m03_tensor_parallel.demo"
    :challenge="{
      ask: '当第一层权重 A 按行切分时, 为什么不能像按列切分那样让各卡先独立通过 GeLU 激活函数、留到最后再统一求和? 尝试关闭同步点 ① 并观察输出误差 Δ 的变化。',
      answer: '原因在于 GeLU 是非线性函数, 各卡持有的部分和分别经过非线性映射后再相加, 在代数上不等于先求和还原完整值再经过 GeLU。\n- A 按行切分的情形: 卡 0 仅持有权重 A 的前 4 行, 只能与输入 X 的前 4 列作矩阵乘积, 算得的结果是 $X \\cdot A$ 的一份「部分和」, 矩阵形状虽与完整结果一致但每个元素仅包含一半输入维度的内积贡献。由于逐元素激活函数 GeLU 满足非线性不等式 $\\mathrm{GeLU}(a+b) \\ne \\mathrm{GeLU}(a) + \\mathrm{GeLU}(b)$, 必须在进入 GeLU 之前的同步点 ① 处先行插入一次 all-reduce, 将两张卡的部分和累加还原为完整的 $X \\cdot A$ 后才能执行激活映射。若关闭同步点 ①, 各卡直接对局部部分和施加 GeLU, 此后无论如何补充通信都无法挽回非线性失真, 右侧 Δ 读数会呈显著非零值且「结论」判定为「数值错了」。此外, 同步点 ① 处的张量为 2×8 的升维中间激活, 宽度为输入 X 与输出 Y 的两倍 (在标准 Transformer FFN 中则为 4 倍), 在此处通信代价最高。\n- A 按列切分的优势: 卡 0 持有权重 A 的前 4 列, 与本地完整的输入 X 相乘后直接得到 $X \\cdot A$ 前 4 列的精确值 (属于「一段列」分片而非部分和)。由于 GeLU 为逐元素独立运算, 各卡直接对本地持有的 4 列精确值施加激活函数即可获得完全准确的后激活列分片。随后第二层权重 B 按行切分, 卡 0 持有的 B 前 4 行恰好与本地刚算完的后激活前 4 列在收缩维上对齐相乘, 产出输出矩阵 Y 的一份部分和。直到这一步才首次产生跨卡求和需求, 在末尾同步点 ③ 处执行一次 all-reduce 即可令 Δ 归 0。\nMegatron 经典的「前层按列切分、后层按行切分」范式, 正是利用逐元素激活对列分片的独立性将跨卡同步推迟到唯一不可省略的求和点, 同时确保通信发生在隐藏维已降回窄维度的输出端。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="lbl">A [d×4d]:</span>
        <button type="button" :class="{ active: splitA === 'col' }" @click="splitA = 'col'">按列切 (切输出维)</button>
        <button type="button" :class="{ active: splitA === 'row' }" @click="splitA = 'row'">按行切 (切输入维)</button>
      </div>
      <div class="row">
        <span class="lbl">B [4d×d]:</span>
        <button type="button" :class="{ active: splitB === 'col' }" @click="splitB = 'col'">按列切</button>
        <button type="button" :class="{ active: splitB === 'row' }" @click="splitB = 'row'">按行切</button>
        <button type="button" @click="seed++">换一组数</button>
        <button type="button" @click="sync = [false, false, false]">清空同步点</button>
      </div>
    </template>

    <div class="flow">
      <template v-for="(s, i) in run.stages" :key="i">
        <button
          v-if="s.slot !== undefined" type="button" class="sync" :class="{ active: sync[s.slot], waste: s.waste }"
          :aria-pressed="sync[s.slot]" @click="sync[s.slot] = !sync[s.slot]"
        >
          同步点 {{ '①②③'[s.slot] }} · {{ sync[s.slot] ? s.op : '点击插入通信' }}
        </button>
        <div v-else class="stage" :class="{ bad: s.invalid }">
          <span class="name mono">{{ s.name }}</span>
          <div v-if="!s.invalid" class="ranks">
            <div v-for="(m, r) in s.t" :key="r" class="rank">
              <span class="rk mono">卡{{ r }}</span>
              <div class="mat" :style="{ gridTemplateColumns: `repeat(${m[0].length}, 12px)` }">
                <i v-for="(v, j) in m.flat()" :key="j" :style="{ background: heat(Math.abs(v) / s.max, v >= 0 ? 'var(--accent)' : 'var(--right)') }" :title="v.toFixed(3)" />
              </div>
            </div>
          </div>
          <span class="kind">{{ s.invalid || kindText[s.kind] }}</span>
        </div>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span><Tex text="与 dense (不切分, 单卡算) 的 $\max|\Delta|$, 0 才算对" /></span><b :class="run.ok ? 'good' : 'bad'">{{ run.delta === null ? '—' : run.delta < 1e-12 ? '0' : run.delta.toFixed(3) }}</b></div>
      <div class="kv"><span>结论</span><b :class="run.ok ? 'good' : 'bad'" class="verdict">{{ run.verdict }}</b></div>
      <div class="kv"><span>通信次数 (生效的同步点)</span><b :class="{ good: run.ok && run.calls === 1 }">{{ run.calls }}</b></div>
      <div class="kv"><span><Tex text="每卡发送量 (以 $X$ 的大小为 1)" /></span><b :class="{ good: run.ok && run.vol <= 1 }">{{ run.vol.toFixed(2) }}×</b></div>
      <div class="lab-note">
        <p>各计算阶段右侧标注的局部张量分布状态直接决定了该处所需的集合通信原语类型:</p>
        <ul class="pts">
          <li><b>完整:</b> 每卡都有整个张量, 不需要通信; 在这里再同步是多余的, 按钮会标黄。</li>
          <li><b>一段列:</b> 每卡只有几列, 要 all-gather (各卡把自己那段发给大家, 拼成完整张量)。</li>
          <li><b>部分和:</b> 形状完整但数值只是一部分, 要 all-reduce (各卡把自己那份加起来, 每卡拿到总和)。</li>
        </ul>
        <p>
          <Tex text="通信字节量折算公式: 在 $N$ 卡张量并行下, 一次 all-reduce 单卡需发送 $2(N-1)/N \cdot \text{size}$, 一次 all-gather 单卡需发送 $(N-1)/N \cdot \text{size}$, 其中 size 为待同步张量的字节体积。由于升维中间激活 $H$ 在真实模型中的通道宽度通常为输入 $X$ 的 4 倍 (本示意图简化绘制为 2 倍), 在 $H$ 处插入集合通信代价最高, 最优切分方案应避免在中间激活上同步。" />
        </p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import { heat, mulberry32, randn, range } from '@/utils/labmath.js'

const N = 2, T = 2, D = 4, HID = 8
const splitA = ref('col'), splitB = ref('row'), seed = ref(1)
// 默认一个同步点都没开: 通信插在哪, 留给读者自己试
const sync = ref([false, false, false])
const kindText = { full: '完整 (每卡一份)', shard: '一段列 (拼起来才完整)', partial: '部分和 (加起来才对)' }

const matmul = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
const erf = (x) => { // Abramowitz-Stegun 7.1.26, 误差 < 1.5e-7, 足够画图; dense 和 TP 用同一个函数所以 Δ 仍可精确到 0
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
const gelu = (m) => m.map((row) => row.map((v) => 0.5 * v * (1 + erf(v / Math.SQRT2))))
const cols = (m, lo, hi) => m.map((row) => row.slice(lo, hi))
const part = (w, r) => [(w / N) * r, (w / N) * (r + 1)]

// 线性层的两种切法: 每张卡只用自己手里的东西去算, 不偷看别的卡
const linear = (st, Wm, split) => {
  const wIn = Wm.length, wOut = Wm[0].length
  if (split === 'col') {
    if (st.kind === 'shard') return { invalid: '形状对不上: 列切的权重要吃完整输入, 但每卡只有一段列 → 先 all-gather' }
    return { kind: 'shard', t: range(N).map((r) => matmul(st.t[r], cols(Wm, ...part(wOut, r)))) }
  }
  return { // 行切: 卡 r 只有 W 的第 r 段行, 所以只吃输入的第 r 段列
    kind: 'partial',
    t: range(N).map((r) => matmul(st.kind === 'shard' ? st.t[r] : cols(st.t[r], ...part(wIn, r)), Wm.slice(...part(wIn, r)))),
  }
}

const run = computed(() => {
  const rand = mulberry32(seed.value * 97)
  const mk = (a, b, s) => range(a).map(() => range(b).map(() => randn(rand) * s))
  const X = mk(T, D, 1), A = mk(D, HID, 0.7), B = mk(HID, D, 0.7)
  const dense = matmul(gelu(matmul(X, A)), B)

  const stages = []
  let st = { kind: 'full', t: range(N).map(() => X) }, calls = 0, vol = 0
  let dead = false // 形状已经对不上之后, 后面的阶段不再重复报错
  const push = (name) => !dead && (dead = !!st.invalid, stages.push({ name, ...st, max: st.t ? Math.max(1e-9, ...st.t.flat(2).map(Math.abs)) : 1 }))
  const doSync = (slot) => {
    const size = st.t ? st.t[0].length * (st.kind === 'shard' ? st.t[0][0].length * N : st.t[0][0].length) : 0
    const op = st.kind === 'shard' ? 'all-gather (拼)' : st.kind === 'partial' ? 'all-reduce (加)' : '多余: 已经是完整的';
    stages.push({ slot, op, waste: sync.value[slot] && st.kind === 'full' })
    if (!sync.value[slot] || st.invalid) return
    calls++
    if (st.kind === 'shard') { vol += ((N - 1) / N) * size; const full = st.t[0].map((_, i) => st.t.flatMap((m) => m[i])); st = { kind: 'full', t: range(N).map(() => full) } }
    else { // ★ 对已经完整的张量再同步 (按 mean) 数值不变, 但通信照样花钱 → 标成"多余"
      vol += (2 * (N - 1) / N) * size
      const total = st.t[0].map((row, i) => row.map((_, j) => st.t.reduce((s, m) => s + m[i][j], 0) / (st.kind === 'full' ? N : 1)))
      st = { kind: 'full', t: range(N).map(() => total) }
    }
  }
  const step = (name, fn) => { if (!st.invalid) st = fn(st); push(name) }

  push('X')
  step(`X·A (${splitA.value === 'col' ? '列切' : '行切'})`, (s) => linear(s, A, splitA.value))
  doSync(0)
  step('GeLU', (s) => ({ ...s, t: s.t.map(gelu) }))
  doSync(1)
  step(`·B (${splitB.value === 'col' ? '列切' : '行切'})`, (s) => linear(s, B, splitB.value))
  doSync(2)
  push('Y')

  const done = !st.invalid && st.kind === 'full'
  const delta = done ? Math.max(...st.t[0].flatMap((row, i) => row.map((v, j) => Math.abs(v - dense[i][j])))) : null
  const ok = done && delta < 1e-12
  const verdict = st.invalid ? '形状对不上' : !done ? (st.kind === 'shard' ? '还差一次 all-gather' : '还差一次 all-reduce') : ok ? (calls === 1 ? '正确, 且通信最少' : '正确, 但通信偏多') : '数值错了'
  return { stages, calls, vol: vol / (T * D), delta, ok, verdict }
})
</script>

<style scoped>
.lbl { font-size: 12px; color: var(--text-muted); min-width: 64px; }
.flow { display: flex; flex-direction: column; gap: 6px; }
.stage { display: grid; grid-template-columns: 92px 1fr; gap: 4px 12px; align-items: center; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); }
.stage.bad { border-color: var(--danger); }
.stage.bad .kind { color: var(--danger); }
.name { font-size: 12px; color: var(--text); }
.ranks { display: flex; flex-wrap: wrap; gap: 14px; }
.rank { display: flex; align-items: center; gap: 6px; }
.rk { font-size: 10px; color: var(--text-dim); }
.mat { display: grid; gap: 2px; }
.mat i { width: 12px; height: 12px; border-radius: 2px; border: 1px solid var(--border); }
.kind { grid-column: 2; font-size: 11px; color: var(--text-dim); }
.sync { align-self: flex-start; margin-left: 24px; font-size: 12px; border-style: dashed; }
.sync.active { border-style: solid; }
.sync.waste { border-color: var(--warn); color: var(--warn); }
.verdict { font-size: 13px !important; }
</style>
