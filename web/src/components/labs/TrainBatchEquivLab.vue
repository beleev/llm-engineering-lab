<!--
  batch 等价实验台 (对应 llm_train/m01_gradient_accumulation + m02_data_parallel)。
  只讲一件事: 梯度累积和 DDP 算出来的是同一个梯度, 区别只在拿时间换显存还是拿卡换时间。
  唯一能打破等价的是"各卡样本数不等" —— 那一个开关比其余控件加起来都重要。
-->
<template>
  <LabFrame
    title="一个 batch 的三种切法 — 梯度累积、DDP 算出来的都是同一个梯度, 付的账不同"
    sub="- 模拟什么: 训练一步要用一整个 batch (本例 24 个样本) 的平均梯度去更新参数。这个台子用一个标量代表每个样本的梯度, 把整个过程按步播出来。显存放不下整个 batch 时有两种切法, 两种可以叠加, 数学上都等于单卡一次算完整 batch, 区别只在付的是时间、显存还是通信:
        - 梯度累积: 一张卡把自己的样本分成 K 份 (每份叫一个 micro-batch) 串行算, 每份的梯度按样本数加权累加到一个缓冲区, 最后再更新一次。
        - DDP (分布式数据并行): P 张卡各放一份完整模型, 各算 24/P 个样本, 再用 all-reduce (每张卡交出自己的梯度, 每张卡拿回所有卡的平均) 把梯度合成同一个。
      - 图上是什么: 从上到下:
        - 每一行是一张卡 (P=1 时只有一行「单卡」), 行里每个小格是一个样本, 格子里的数字是它属于这张卡的第几个 micro-batch (K=1 时不显示)。
        - 格子颜色: 亮的是已经算过的, 带描边的是正在算的, 暗的是还没轮到的。
        - 每行最右边的数字是这张卡的梯度缓冲区, 已算完的 micro-batch 的梯度加权累加在这里。
        - 多卡时下面有一条 all-reduce 横条, 同步完成后变绿, 显示所有卡握住的同一个数。
        - 再下面一行文字说明当前这一步在干什么, 以及两个关键数: 「单卡整 batch 基线」(一次算完 24 个样本的平均梯度, 正确答案) 和「当前方案同步后」(当前切法最后得到的梯度)。
      - 可以做什么: 控件从上到下:
        - 点「单卡整 batch / 梯度累积 / DDP / 累积 + DDP」四个预设跳到典型配置。
        - 「各卡等分 / 各卡样本数不等」切换每张卡拿的样本数是否相同 (不等分时卡 0 拿得多、其余卡拿得少, 总数仍是 24)。
        - 拖「卡数 P」(只给能整除 24 的 1、2、3、4、6、8) 和「每卡 micro-batch 数 K」换切法。
        - 用播放器的「▶ 播放」或「下一步」一步步看 micro-batch 依次点亮、缓冲区增长、最后 all-reduce。
        - 悬停或点击任一格子, 看这个样本被分到哪张卡的哪个 micro-batch、它自己的梯度是多少。
      - 应该看到: 等分时不管怎么切, 「当前方案同步后」都等于「单卡整 batch 基线」, 右侧第一项差值 |Δ| 是 1e-17 量级的浮点零头, 打勾。右侧另外三项读数, 以及三种改法的效果:
        - 「激活峰值」: 一次前向最多同时处理几个样本, 决定显存。
        - 「步时间」: 一张卡串行要算几个样本, 决定耗时。
        - 「每步通信 / 卡」: 每张卡每步为 all-reduce 要发出去的字节数 (按 13 亿参数的 fp32 梯度折算)。
        - 加卡: 激活峰值和步时间同时除以 P, 通信从 0 涨到约 2 份梯度的大小。
        - 加 K: 只有激活峰值降, 步时间一点不变。
        - 打开「各卡样本数不等」: |Δ| 跳到 1e-1 量级, 打叉。all-reduce 做的是各卡局部均值的简单平均, 各卡样本数不同时它就不再等于全局均值了。"
    module="llm_train/m01+m02"
    run="python -m llm_train.m02_data_parallel.demo"
    :challenge="{
      ask: '依次做三件事, 每步先猜:\n1. 把卡数从 1 拖到 8, 右边四个数字分别怎么变?\n2. 再把 micro-batch 数从 1 拖到 4 呢?\n3. 最后打开「各卡样本数不等」, 哪个数字先崩?',
      answer: '答案:\n- 加卡: 激活峰值和步时间都降, 通信涨, |Δ| 不变。\n- 加 K: 只有激活峰值降。\n- 不等分: |Δ| 立刻崩。\n为什么:\n- 加卡 (P 从 1 到 8): 每张卡只算 24/8 = 3 个样本, 所以「激活峰值」从 24 降到 3, 「步时间」也从 24 降到 3, 两者都除以卡数。代价是「每步通信 / 卡」从 0 B 涨到 $2(P-1)/P$ 份梯度: 8 卡时每卡要发 1.75 份梯度的字节 (13 亿参数 × 4 字节 × 1.75 约 8.5 GiB), 这个系数在卡数再多时也封顶在 2。与基线的差 |Δ| 一直贴在 1e-17, 那是浮点求和顺序不同留下的零头。DDP 的全部买卖, 就是拿通信换时间。\n- 加 micro-batch (K 从 1 到 4): 每卡把自己的样本分 4 份串行算, 「激活峰值」除以 4, 但「步时间」一点没省 (4 份加起来还是那么多样本要算), 通信也没变 (只在最后同步一次)。梯度累积换的是显存, 不是速度。\n- 打开不等分: |Δ| 从 1e-17 跳到 1e-1 量级, 打叉。原因是 all-reduce(mean) 做的是「各卡局部均值的简单平均」: 8 卡不等分时卡 0 拿 10 个样本、其余 7 张卡各拿 2 个, 每张卡先对自己的样本求均值, 8 个均值再平均。这样卡 0 的每个样本只占 1/80 的权重, 别的卡的每个样本占 1/16, 不再是 24 个样本各占 1/24 的全局均值。步时间也顺带被样本最多的那张卡拖长了, 其他卡算完要等它。\n真实框架的解法有两种:\n- 用 DistributedSampler 把每卡样本数补齐或把尾巴丢掉。\n- 把每卡的 loss 先乘 $n_r \\cdot P/N$ ($n_r$ 是这张卡的样本数, $N$ 是总样本数) 再同步, 把简单平均修正回按样本数加权。',
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
          <li><b>等分:</b> <Tex text="每卡样本数相同时, all-reduce(mean) (各卡局部均值再平均) 恰好等于 24 个样本的全局均值, $|\Delta|$ 只剩浮点求和顺序不同留下的零头 (这里用 float64 是 1e-17 量级; m01/m02 用 float32, 打印出来是 2.98e-08 和 3.73e-09)。" /></li>
          <li><b>不等分:</b> 各卡局部均值的简单平均不再是全局均值, |Δ| 从 1e-17 跳到 1e-1, 偏了十几个数量级, 而且框架不会报错。</li>
        </ul>
        <p>
          <Tex text="通信量按 $\Psi = 1.3\text{B}$ (13 亿) 参数、fp32 梯度 (每个 4 字节) 折算: 每卡每步发 $\Psi \times 4\text{B} \times 2(P-1)/P$, 卡数再多也封顶在 $2\Psi \times 4\text{B}$ (两份梯度)。" />
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
