<!--
  VAR next-scale 预测实验台 (对应 llm_models/models/generative/var.py:VARModel.sample_tokens 与 layers/diffusion/vq.py 的多尺度残差量化)。
  只讲一件事: 图像的自回归单位不必是 “下一个像素/patch”, 可以是 “下一个分辨率”: 1×1 → 2×2 → 4×4 → 8×8,
  每一级量化的是 “上一级还没解释掉的残差”, 同一级内所有 token 并行生成。
-->
<template>
  <LabFrame
    title="VAR 逐尺度生成 — 自回归的一步是「下一个分辨率」, 不是「下一个像素」"
    sub="- 模拟什么: 用自回归模型生成一张 8×8 的灰度图。老办法 (光栅自回归) 把 64 个像素按从左到右、从上到下排成一串, 每次前向只生成 1 个像素, 要 64 次。VAR (Visual AutoRegressive) 把「一步」改成「下一个尺度」: 第 1 次前向出一张 1×1 的 token map (整张图的平均明暗), 第 2 次出 2×2, 然后 4×4、8×8, 每一级内的所有 token 一次并行生成。每一级编码的不是原图, 而是「目标图减去前面所有级累计重建」的残差, 量化后放大叠回去。这里的码本是 K 个均匀分布在 −1 到 1 的灰度值, 量化就是找最近的那一个。
      - 图上是什么: 四张小图从左到右: 「目标图」是要生成的 8×8 灰度图; 「本步并行生成的 token map」是这一级的 s×s 个 token, 每格的颜色是量化后的残差值, s ≤ 4 时格子里写着码字编号; 「VAR 累计重建」是前几级上采样后相加的结果, 走完 4 级就是最终输出; 「光栅自回归」是用同样多次前向、一次 1 个像素能画到哪里, 斜纹是还没画的。
      - 可以做什么: 拖「码本大小 K」改量化有多粗 (K = 2 只有黑白两级, K = 16 有 16 级灰度)。「换一张图」换一张随机目标图。播放控件的每一步就是一次前向, 对应 1×1、2×2、4×4、8×8 四级。
      - 应该看到: 第 1 步之后 VAR 的重建是一整片平均灰, 第 2 步已经能看出明暗布局, 第 3、4 步补细节; 同一时刻光栅自回归只画了左上角 1、2、3、4 个像素。token map 里的值随级数越来越接近 0, 因为前面的级已经解释掉了大部分。右侧「VAR 重建 MSE」是当前重建和目标图逐像素差的平方平均, 每走一级就降一截, K 越大降得越低; 「本步并行生成 token 数」是 s²; 「累计 token (VAR / 光栅)」走完是 85 对 64, VAR 的序列反而更长; 「光栅画完 64 个像素还差」是对照组还要多少次前向。"
    module="llm_models/models/generative/var.py"
    run="python -m llm_models.run_models.generative.var.infer_var"
    :challenge="{
      ask: '走到第 2 次前向 (2×2)。对比 VAR 这时的累计重建和光栅自回归这时的画面, 先猜哪个更「像」目标图。再猜: 全部走完各需要多少次前向? VAR 的 token 总数比光栅多还是少?',
      answer: '答案: 2 次前向后 VAR 更像; 走完 VAR 要 4 次前向、光栅要 64 次; VAR 的 token 总数是 85, 比光栅的 64 多。\n为什么:\n- VAR 2 次前向后: 1×1 那一级定了整张图的平均灰度, 2×2 那一级把图分成四块各补一个残差, 已经有了整张图的大致明暗布局。先构图再细化。\n- 光栅自回归 2 次前向后: 只画了左上角 2 个像素, 要到第 64 步才知道右下角长什么样。而且把二维图拉成一维串, 上下相邻的两个像素在序列里隔了一整行。\n- 前向次数: VAR 每次前向吐出一整级 (1、4、16、64 个 token 并行), 4 级共 4 次; 光栅一次 1 个像素, 64 次。\n- token 总数: VAR 是 $1 + 4 + 16 + 64 = 85$, 比 64 多, 序列更长, 这是它付出的代价。\n为什么级内并行不会乱: 每一级只量化「残差」(目标减去之前所有级上采样之和)。粗尺度先定了大局, 细尺度每个 token 要补的只是一点点, 独立采样的不一致也就很小; 粗尺度的错误还可以被细尺度修正。\n训练时用块状因果 mask: 同一尺度内的 token 互相可见, 另外只能看更粗的尺度, 一次前向就能算完所有级的 loss。',
    }"
  >
    <template #controls>
      <LabSlider v-model="K" label="码本大小 K (量化级数)" :min="2" :max="16" />
      <div class="row"><button type="button" @click="seed++">换一张图</button></div>
      <StepPlayer :stepper="stepper" :label="`第 ${step + 1} 次前向 · ${SCALES[step]}×${SCALES[step]}`" />
    </template>

    <div class="panels">
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in sim.target.flat()" :key="i" :style="{ background: shade(v) }" /></div><figcaption>目标图</figcaption></figure>
      <figure><div class="img" :style="gridOf(SCALES[step])"><i v-for="(v, i) in now.tokens.flat()" :key="i" class="tk mono" :style="{ background: shade(v.val) }">{{ SCALES[step] <= 4 ? v.idx : '' }}</i></div><figcaption>本步并行生成的 token map ({{ SCALES[step] }}×{{ SCALES[step] }}, 量化残差)</figcaption></figure>
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in now.recon.flat()" :key="i" :style="{ background: shade(v) }" /></div><figcaption>VAR 累计重建 (前 {{ step + 1 }} 级之和)</figcaption></figure>
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in sim.raster.flat()" :key="i" :style="i <= step ? { background: shade(v) } : {}" :class="{ todo: i > step }" /></div><figcaption>光栅自回归: {{ step + 1 }} 次前向 = {{ step + 1 }} 个像素</figcaption></figure>
    </div>

    <template #stats>
      <div class="kv"><span>VAR 重建 MSE (和目标图逐像素差的平方平均)</span><b :class="now.mse < 0.01 ? 'good' : ''">{{ now.mse.toFixed(4) }}</b></div>
      <div class="kv"><span>本步并行生成 token 数 ({{ SCALES[step] }}×{{ SCALES[step] }})</span><b>{{ SCALES[step] ** 2 }}</b></div>
      <div class="kv"><span>{{ step + 1 }} 次前向后累计 token (VAR / 光栅)</span><b>{{ cumTokens }} / {{ step + 1 }}</b></div>
      <div class="kv"><span>光栅画完 64 个像素还差</span><b :class="{ bad: 64 - (step + 1) > 0 }">{{ 64 - (step + 1) }} 次前向</b></div>
      <div class="lab-note">
        <p><Tex text="每一级编码的不是原图, 而是「目标 − 已有重建」的残差。做法: 把残差按面积平均缩小到 $s \times s$, 每格查最近的码字, 再放大回 8×8 加到累计重建上。" /></p>
        <p>所以粗尺度量化得再糙也没关系, 误差都留在残差里交给下一级修。走完全程: VAR 4 次前向、85 个 token; 光栅 64 次前向、64 个 token。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, clamp, range, heat } from '@/utils/labmath.js'

const H = 8, SCALES = [1, 2, 4, 8]
const K = ref(8), seed = ref(1)

const sim = computed(() => {
  const r = mulberry32(seed.value * 131)
  const blobs = range(3).map(() => ({ x: r() * 7, y: r() * 7, a: (r() - 0.35) * 1.8, w: 2 + r() * 8 }))
  const target = range(H).map((y) => range(H).map((x) => clamp(-0.3 + blobs.reduce((s, b) => s + b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / b.w), 0), -1, 1)))
  const levels = range(K.value).map((i) => -1 + (2 * i) / (K.value - 1))   // 标量码本
  const quant = (v) => { const idx = levels.reduce((b, l, i) => (Math.abs(l - v) < Math.abs(levels[b] - v) ? i : b), 0); return { idx, val: levels[idx] } }

  let recon = range(H).map(() => Array(H).fill(0))
  const stages = SCALES.map((s) => {
    const blk = H / s
    // ★ 残差 → 面积平均下采样到 s×s → 量化 → 最近邻上采样 → 累加
    const tokens = range(s).map((ty) => range(s).map((tx) => {
      let acc = 0
      for (let y = ty * blk; y < (ty + 1) * blk; y++) for (let x = tx * blk; x < (tx + 1) * blk; x++) acc += target[y][x] - recon[y][x]
      return quant(acc / (blk * blk))
    }))
    recon = recon.map((row, y) => row.map((v, x) => v + tokens[Math.floor(y / blk)][Math.floor(x / blk)].val))
    const mse = recon.flat().reduce((a, v, i) => a + (v - target.flat()[i]) ** 2, 0) / (H * H)
    return { tokens, recon, mse }
  })
  return { target, stages, raster: target.map((row) => row.map((v) => quant(v).val)) }
})

const stepper = useStepper(() => SCALES.length, { interval: 1100 })
const step = computed(() => stepper.step.value)
const now = computed(() => sim.value.stages[step.value])
const cumTokens = computed(() => SCALES.slice(0, step.value + 1).reduce((a, s) => a + s * s, 0))   // 1 + 4 + 16 + 64 = 85
const shade = (v) => heat((v + 1) / 2, 'var(--text)')
const gridOf = (n) => ({ gridTemplateColumns: `repeat(${n}, 1fr)` })
</script>

<style scoped>
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 14px; }
figure { margin: 0; display: grid; gap: 6px; justify-items: center; }
figcaption { font-size: 11px; color: var(--text-dim); text-align: center; line-height: 1.4; }
.img { display: grid; width: 128px; height: 128px; gap: 1px; border: 1px solid var(--border-strong); background: var(--bg); }
.img i { display: grid; place-items: center; font-style: normal; font-size: 10px; color: var(--accent); min-width: 0; }
.img i.todo { background: repeating-linear-gradient(45deg, transparent 0 3px, var(--border) 3px 4px); }
</style>
