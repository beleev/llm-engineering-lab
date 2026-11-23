<!--
  batch 等价实验台 (对应 llm_train/m01_gradient_accumulation + m02_data_parallel)。
  只讲一件事: 梯度累积和 DDP 算出来的是同一个梯度, 区别只在拿时间换显存还是拿卡换时间。
  唯一能打破等价的是"各卡样本数不等" —— 那一个开关比其余控件加起来都重要。
-->
<template>
  <LabFrame
    title="一个 batch 的三种切法 — 梯度累积、DDP 算出来的都是同一个梯度, 付的账不同"
    sub="- 模拟什么: 单步训练需要使用整个 batch (本例为 24 个样本) 的平均梯度更新参数。本实验台用一个标量表示单个样本的梯度, 按步演示累积与同步全过程。当单卡显存无法容纳整个 batch 时, 有两种可叠加的切分方案, 数学结果均等价于单卡一次计算完整 batch, 区别仅在于代价落在时间、显存还是通信开销上:
  - 梯度累积: 单张卡将本地样本分为 K 个 micro-batch 串行计算, 每份梯度按样本数加权累加至缓冲区, 最后统一更新一次参数。
  - DDP (分布式数据并行): P 张卡各持有一份完整模型并分别计算 24/P 个样本, 再通过 all-reduce 汇总并取平均, 使每张卡获得相同的全局平均梯度。
- 图上是什么: 从上到下依次为:
  - 每一行代表一张卡 (当 P 为 1 时仅显示一行「单卡」), 行内每个小格代表一个样本, 格内数字表示该样本所属的 micro-batch 序号 (当 K 为 1 时不显示)。
  - 格子颜色编码计算状态: 高亮格表示已计算完成, 带描边格表示正在计算, 暗色格表示尚未执行。
  - 每行最右侧的数值为该卡的梯度缓冲区, 已完成的 micro-batch 梯度按权重累加于此。
  - 多卡模式下底部设有一条 all-reduce 横条, 同步完成后变为绿色, 显示所有卡共同持有的同步后梯度。
  - 最下方文字说明当前步骤的操作内容, 并对比两个关键读数: 「单卡整 batch 基线」(一次算完 24 个样本的平均梯度基准值) 与「当前方案同步后」(当前切分方案最终所得梯度)。
- 可以做什么: 控件从上到下依次为:
  - 点击「单卡整 batch / 梯度累积 / DDP / 累积 + DDP」四个预设按钮切换典型配置。
  - 通过「各卡等分 / 各卡样本数不等」切换各卡分配的样本数是否一致 (不等分时卡 0 分配较多样本, 其余卡分配较少, 总样本数仍为 24)。
  - 拖动「卡数 P」(仅提供能整除 24 的 1、2、3、4、6、8) 与「每卡 micro-batch 数 K」调整切分方案。
  - 使用播放器的「▶ 播放」或「下一步」逐步观察 micro-batch 依次点亮、缓冲区累加及最终 all-reduce 同步的过程。
  - 悬停或点击任意样本格, 查看该样本所属的卡号、micro-batch 序号及其单样本梯度值。
- 应该看到: 各卡等分样本时, 无论如何切分, 「当前方案同步后」均与「单卡整 batch 基线」一致, 右侧第一项差值 |Δ| 仅为 1e-17 量级的浮点舍入误差并显示对勾。右侧其余三项读数及三种调整的效果如下:
  - 「激活峰值」: 单次前向传播同时处理的最大样本数, 决定激活显存占用。
  - 「步时间」: 单张卡串行处理的样本总数, 决定单步耗时。
  - 「每步通信 / 卡」: 每张卡每步执行 all-reduce 发送的字节数 (按 13 亿参数的 fp32 梯度折算)。
  - 增加卡数 P: 激活峰值与步时间同时除以 P, 通信量从 0 增至接近 2 份梯度的体积。
  - 增加 K: 仅激活峰值下降, 步时间保持不变。
  - 开启「各卡样本数不等」: |Δ| 升至 1e-1 量级并显示叉号。这是因为 all-reduce 计算的是各卡局部均值的简单平均, 当各卡样本数不等时不再等价于全局均值。"
    module="llm_train/m01+m02"
    run="python -m llm_train.m02_data_parallel.demo"
    :challenge="{
      ask: '依次执行以下三步操作, 每步操作前先预测结果:\n1. 将卡数从 1 拖至 8, 右侧四项读数分别如何变化?\n2. 再将 micro-batch 数从 1 拖至 4, 四项读数又如何变化?\n3. 最后开启「各卡样本数不等」, 哪一项读数会首先出现显著偏差?',
      answer: '结论:\n- 增加卡数 P: 激活峰值与步时间同步下降, 通信量上升, |Δ| 保持不变。\n- 增加 micro-batch 数 K: 仅激活峰值下降, 其余读数不变。\n- 开启样本数不等分: |Δ| 立即出现显著误差。\n机制解析:\n- 增加卡数 (P 从 1 增至 8): 每张卡仅处理 24/8 即 3 个样本, 因此「激活峰值」从 24 降至 3, 「步时间」也从 24 降至 3, 两者均除以卡数 P。代价是「每步通信 / 卡」从 0 B 增至 $2(P-1)/P$ 份梯度: 8 卡配置下每卡需发送 1.75 份梯度的字节数 (13 亿参数乘 4 字节再乘 1.75, 约为 8.5 GiB), 且随着卡数继续增加, 该系数上限收敛于 2。与基线的差值 |Δ| 始终维持在 1e-17 量级, 仅源于浮点求和顺序不同引入的舍入误差。DDP 的核心权衡就是以通信开销换取步时间缩短。\n- 增加 micro-batch 数 (K 从 1 增至 4): 每张卡将本地样本拆分为 4 份串行计算, 「激活峰值」除以 4, 但「步时间」并未减少 (4 份累计处理的样本总数不变), 通信量也保持不变 (仅在最后一步同步一次)。梯度累积换取的是显存节省, 而非训练加速。\n- 开启样本数不等分: |Δ| 从 1e-17 升至 1e-1 量级并显示叉号。原因在于 all-reduce(mean) 执行的是各卡局部均值的简单平均: 8 卡不等分时卡 0 分得 10 个样本, 其余 7 张卡各分得 2 个样本; 每张卡先对本地样本求均值, 再将 8 个局部均值取平均。此时卡 0 上每个样本的权重仅为 1/80, 其余卡上每个样本的权重则为 1/16, 不再等价于 24 个样本各占 1/24 的全局均值。此外, 步时间也会受样本数最多的卡拖累, 其余卡算完后必须等待该卡完成。\n主流训练框架通常采用以下两种解法:\n- 使用 DistributedSampler 将各卡样本数补齐或丢弃尾部不足整除的样本。\n- 同步前先将各卡的 loss 乘以重加权系数 $n_r \\cdot P/N$ ($n_r$ 为当前卡样本数, $N$ 为全局总样本数), 将简单平均修正回按样本数加权的全局平均。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'single' }" @click="preset(0, 1)">单卡整 batch</button>
        <button type="button" :class="{ active: mode === 'accum' }" @click="preset(0, 4)">梯度累积</button>
        <button type="button" :class="{ active: mode === 'ddp' }" @click="preset(3, 1)">DDP</button>
        <button type="button" :class="{ active: mode === 'both' }" @click="preset(3, 2)">累积 + DDP</button>
        <span class="sep" />
        <button type="button" :class="{ active: !uneven }" :disabled="P === 1" @click="uneven = false">各卡等分</button>
        <button type="button" class="warnbtn" :class="{ active: uneven }" :disabled="P === 1" @click="uneven = true">各卡样本数不等</button>
      </div>
      <LabSlider v-model="pIdx" label="卡数 P" :min="0" :max="5" :format="(i) => PS[i]" />
      <LabSlider v-model="k" label="每卡 micro-batch 数 K" :min="1" :max="kMax" />
      <StepPlayer :stepper="stepper" :label="phase" />
    </template>

    <!-- 整张表只占一个 Tab 停靠点: 悬停临时看, 点击固定, 左右方向键按样本编号移动 -->
    <div class="cells beq" :style="{ gridTemplateColumns: `46px repeat(${maxCols}, minmax(18px, 1fr)) 76px` }"
         tabindex="0" role="group" aria-label="24 个样本的分配表, 左右方向键换一个样本"
         @keydown.left.prevent="move(-1)" @keydown.right.prevent="move(1)">
      <template v-for="(row, r) in rows" :key="r">
        <span class="row-label mono">{{ P === 1 ? '单卡' : '卡 ' + r }}</span>
        <span
          v-for="(c, ci) in row" :key="ci"
          class="cell" :class="[cellClass(c), { sel: c && cur && c.gi === cur.gi }]"
          :title="c ? `样本 #${c.gi} · 卡 ${r} 的第 ${c.m + 1} 个 micro-batch · 这一个样本的梯度 ${g[c.gi].toFixed(3)}` : ''"
          @mouseenter="hoverGi = c ? c.gi : -1" @mouseleave="hoverGi = -1" @click="c && (pinGi = pinGi === c.gi ? -1 : c.gi)"
        >{{ c && K > 1 ? c.m + 1 : '' }}</span>
        <span class="buf mono" :class="{ done: reduced }">{{ bufs[r].toFixed(3) }}</span>
      </template>
    </div>

    <div v-if="P > 1" class="ar" :class="{ live: reduced }">
      <span class="mono">all-reduce(mean) · {{ reduced ? `${P} 张卡现在握着同一个梯度 ${synced.toFixed(3)}` : '等各卡累积完再同步' }}</span>
    </div>

    <p class="lab-note msg" aria-live="polite">
      <template v-if="cur">
        样本 #{{ cur.gi }} 在卡 {{ cur.r }} 的第 {{ cur.m + 1 }} 个 micro-batch 里, 它自己的梯度是 {{ g[cur.gi].toFixed(3) }}。
      </template>
      <template v-else>{{ phaseText }}</template>
      <br />
      <span class="mono dimtext">单卡整 batch 基线 = {{ baseline.toFixed(6) }}　当前方案同步后 = {{ synced.toFixed(6) }}</span>
    </p>

    <template #stats>
      <div class="kv">
        <span>与单卡基线的差 |Δ| (assert |Δ| &lt; 1e-6 才算等价)</span>
        <b :class="equiv ? 'good' : 'bad'">{{ delta.toExponential(2) }} {{ equiv ? '✓' : '✗' }}</b>
      </div>
      <div class="kv"><span>激活峰值 (一次前向同时处理的样本数, 决定显存; 括号内是相对单卡整 batch)</span><b :class="{ good: peak < 24 }">{{ peak }} · {{ (peak / 24).toFixed(2) }}×</b></div>
      <div class="kv"><span>步时间 (一张卡串行要算的样本数, 决定耗时)</span><b :class="{ good: slow < 24 }">{{ slow }} · {{ (slow / 24).toFixed(2) }}×</b></div>
      <div class="kv"><span>每步通信 / 卡 (为 all-reduce 发出去的字节)</span><b :class="commBytes > 0 ? 'bad' : 'good'">{{ commBytes > 0 ? fmtBytes(commBytes) : '0 B' }}</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>等分:</b> <Tex text="当各卡样本数相同时, all-reduce(mean) (对各卡局部均值再取平均) 严格等于 24 个样本的全局均值, $|\Delta|$ 仅剩浮点求和顺序不同产生的舍入误差 (此处采用 float64 时为 1e-17 量级; m01/m02 采用 float32 时实测输出分别为 2.98e-08 与 3.73e-09)。" /></li>
          <li><b>不等分:</b> 各卡局部均值的简单平均不再是全局均值, |Δ| 从 1e-17 跳到 1e-1, 偏了十几个数量级, 而且框架不会报错。</li>
        </ul>
        <p>
          <Tex text="通信量按 $\Psi = 1.3\text{B}$ (13 亿) 参数与 fp32 梯度 (每参数占 4 字节) 折算: 每卡每步发送 $\Psi \times 4\text{B} \times 2(P-1)/P$, 即使卡数无限增加, 上限也收敛于 $2\Psi \times 4\text{B}$ (即 2 份梯度体积)。" />
          这个 2(P−1)/P 系数来自 ring all-reduce 的实现方式, 和「通信与 full_loop」那一章讲的是同一个公式。
        </p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { fmtBytes, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 24                     // 全局 batch: 24 个样本
const PS = [1, 2, 3, 4, 6, 8]    // 只取能整除 24 的卡数
const PSI = 1.3e9                // 参数量, 用来把通信量换算成看得见的字节

const pIdx = ref(3)
const k = ref(1)
const uneven = ref(false)
const hoverGi = ref(-1), pinGi = ref(-1)   // 悬停的 / 点击固定的样本编号; 只记编号, 换切法后仍然指向同一个样本

const P = computed(() => PS[pIdx.value])
const K = computed(() => k.value)
const mode = computed(() => (P.value === 1 ? (K.value === 1 ? 'single' : 'accum') : K.value === 1 ? 'ddp' : 'both'))
const preset = (pi, kk) => { pIdx.value = pi; k.value = kk; if (pi === 0) uneven.value = false }

// 每个样本自己的梯度 (标量代表整棵梯度树): 固定 seed, 拖滑杆时数字不乱跳
const g = computed(() => { const r = mulberry32(7); return range(N).map(() => randn(r)) })
const baseline = computed(() => sum(g.value) / N)   // ★ 单卡整 batch 的答案, 一切都要跟它对

// 各卡拿几个样本。不等分时把 0 号卡撑大, 其余等量缩小, 总数仍是 N
const sizes = computed(() => {
  const p = P.value
  if (p === 1) return [N]
  const base = N / p
  if (!uneven.value) return Array(p).fill(base)
  const d = Math.max(1, Math.floor(base / 3))
  return range(p).map((r) => (r === 0 ? base + d * (p - 1) : base - d))
})
const kMax = computed(() => Math.min(6, Math.min(...sizes.value)))
watch(kMax, (m) => { if (k.value > m) k.value = m }, { immediate: true })

// 每卡: 把自己的样本切成 K 个 micro-batch, 各自算均值, 再按 n_k/n_rank 加权累进缓冲区
const plan = computed(() => {
  const kk = K.value
  let off = 0
  return sizes.value.map((n) => {
    const micros = range(kk).map((i) => {
      const a = off + Math.floor((n * i) / kk)
      const idx = range(off + Math.floor((n * (i + 1)) / kk) - a).map((j) => a + j)
      const mean = idx.length ? sum(idx.map((j) => g.value[j])) / idx.length : 0
      return { idx, n: idx.length, mean, w: idx.length / n }   // ★ 权重是样本数占比, 不是 1/K
    })
    off += n
    return { n, micros, local: sum(micros.map((m) => m.w * m.mean)) }
  })
})

// all-reduce(mean) = 各卡局部均值的简单平均。等分时 == baseline, 不等分时不是
const synced = computed(() => sum(plan.value.map((r) => r.local)) / P.value)
const delta = computed(() => Math.abs(synced.value - baseline.value))
const equiv = computed(() => delta.value < 1e-6)

const frames = computed(() => K.value + (P.value > 1 ? 1 : 0) + 1)
const stepper = useStepper(frames, { interval: 700 })
const f = computed(() => Math.min(stepper.step.value, frames.value - 1))
const doneK = computed(() => Math.min(f.value, K.value))
const active = computed(() => (f.value >= 1 && f.value <= K.value ? f.value - 1 : -1))
const reduced = computed(() => P.value > 1 && f.value > K.value)

const maxCols = computed(() => Math.max(...sizes.value))
const rows = computed(() => plan.value.map((r) => {
  const cells = r.micros.flatMap((m, mi) => m.idx.map((gi) => ({ gi, m: mi })))
  while (cells.length < maxCols.value) cells.push(null)
  return cells
}))
// 样本编号 → 它落在哪张卡 r 的第几个 micro-batch m
const where = computed(() => rows.value.flatMap((row, r) => row.filter(Boolean).map((c) => ({ ...c, r }))))
const cur = computed(() => {
  const gi = hoverGi.value >= 0 ? hoverGi.value : pinGi.value
  return gi < 0 ? null : where.value.find((c) => c.gi === gi)
})
const move = (d) => { pinGi.value = Math.min(N - 1, Math.max(0, (cur.value ? cur.value.gi + d : 0))); hoverGi.value = -1 }
const bufs = computed(() => plan.value.map((r) =>
  (reduced.value ? synced.value : sum(r.micros.slice(0, doneK.value).map((m) => m.w * m.mean)))))

const cellClass = (c) => {
  if (!c) return 'pad'
  if (reduced.value || c.m < active.value) return 'ok'
  if (c.m === active.value) return 'on now'
  return 'dim'
}

const peak = computed(() => Math.max(...plan.value.flatMap((r) => r.micros.map((m) => m.n))))
const slow = computed(() => Math.max(...sizes.value))
const commBytes = computed(() => (PSI * 4 * 2 * (P.value - 1)) / P.value)

const phase = computed(() => (reduced.value ? 'all-reduce' : f.value === 0 ? '未开始' : `micro ${f.value}/${K.value}`))
const phaseText = computed(() => {
  if (f.value === 0) return '每卡的梯度缓冲区都是 0。按「▶ 播放」, 或点「下一步 ›」一步一步走。'
  if (reduced.value) return `all-reduce(mean): 每卡把自己那份梯度发出去再收回平均值。${P.value} 张卡现在逐位相同, 可以各自 optimizer.step()。`
  const n = plan.value.map((r) => r.micros[active.value].n).join(' / ')
  return `第 ${f.value} 个 micro-batch: 各卡分别前向反向 ${n} 个样本, 梯度乘 n_k/n_rank 后累进缓冲区 (缓冲区数字在右侧)。`
})

watch([P, K, uneven], () => { stepper.pause(); stepper.step.value = frames.value - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.beq .cell { height: 24px; font-size: 9px; cursor: pointer; }
.cell.sel { outline: 2px solid var(--text); outline-offset: -2px; }
.cell.pad { border: none; background: none; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.buf {
  align-self: center; text-align: right; font-size: 12px;
  color: var(--text-muted); font-variant-numeric: tabular-nums; padding-left: 6px;
}
.buf.done { color: var(--left); }
.ar {
  margin-top: 8px; padding: 5px 10px; border-radius: var(--radius-sm); font-size: 11px;
  border: 1px dashed var(--border-strong); color: var(--text-dim); overflow: hidden;
}
.ar.live { border: 1px solid var(--left); color: var(--text); background: color-mix(in srgb, var(--left) 14%, transparent); }
.ar.live span { display: inline-block; animation: sweep 0.7s ease-out; }
@keyframes sweep { from { transform: translateX(-24px); opacity: 0.2; } to { transform: none; opacity: 1; } }
.sep { width: 1px; height: 18px; background: var(--border-strong); }
.warnbtn.active { border-color: var(--danger); color: var(--danger); }
.msg { margin-top: 12px; min-height: 3.4em; }
.dimtext { color: var(--text-dim); font-size: 11px; }
</style>
