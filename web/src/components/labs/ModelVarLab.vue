<!--
  VAR next-scale 预测实验台 (对应 llm_models/models/generative/var.py:VARModel.sample_tokens 与 layers/diffusion/vq.py 的多尺度残差量化)。
  只讲一件事: 图像的自回归单位不必是 “下一个像素/patch”, 可以是 “下一个分辨率”: 1×1 → 2×2 → 4×4 → 8×8,
  每一级量化的是 “上一级还没解释掉的残差”, 同一级内所有 token 并行生成。
-->
<template>
  <LabFrame
    title="VAR 逐尺度生成 — 自回归的一步是「下一个分辨率」, 不是「下一个像素」"
    sub="- 模拟什么: 使用自回归模型生成一张 $8 \times 8$ 灰度图像。传统光栅自回归方案将 64 个像素按从左到右、从上到下的光栅扫描顺序展平为一维序列, 每次前向仅预测 1 个像素, 生成整图需要串行执行 64 次前向。VAR (Visual AutoRegressive) 将自回归单元从「下一个像素」改为「下一个空间尺度」: 第 1 次前向预测 $1 \times 1$ 的全局均值 token map, 第 2 次前向预测 $2 \times 2$, 随后依次预测 $4 \times 4$ 与 $8 \times 8$, 同一尺度内的全部 token 在单次前向中并行生成。每一尺度编码的并非原始图像本身, 而是目标图像减去此前所有更粗尺度累计重建结果后的空间残差, 经离散量化并上采样后叠加回累计重建图。实验所用码本包含 K 个在 $[-1, 1]$ 区间均匀分布的灰度级, 量化操作即将连续残差映射至最近的码字。
- 图上是什么: 从左至右并列展示四幅图像视图:
  - 「目标图」: 待还原的 $8 \times 8$ 原始灰度图像。
  - 「本步并行生成的 token map」: 当前尺度并行输出的 $s \times s$ 个离散 token, 每格灰度对应量化后的残差值, 当 $s \le 4$ 时格内直接标出所选码字索引。
  - 「VAR 累计重建」: 已完成各尺度的残差上采样累加结果, 走完 4 个尺度后即得到最终生成图像。
  - 「光栅自回归」: 在相同串行前向次数下, 逐像素光栅生成所能完成的局部画面, 斜纹区域表示尚未生成的像素。
- 可以做什么: 按界面控件顺序操作:
  - 拖动「码本大小 K」: 调节残差量化的精细程度 ($K = 2$ 仅含两级粗量化, $K = 16$ 提供 16 级灰度)。
  - 「换一张图」: 重新随机生成一幅目标灰度图。
  - 播放控件: 每推进 1 步对应执行 1 次模型前向, 依次遍历 $1 \times 1$、$2 \times 2$、$4 \times 4$、$8 \times 8$ 四个尺度级。
- 应该看到: 逐步播放并对照右侧读数:
  - VAR 重建过程: 第 1 步给出整幅图像的全局平均明暗, 第 2 步已呈现出四个象限的主体明暗布局, 第 3、第 4 步逐步补充局部高频细节。
  - 光栅自回归对照: 在相同的 1 至 4 次前向内, 仅完成了左上角第 1 至第 4 个孤立像素的绘制。
  - token map 幅值变化: 随着尺度变细, 残差幅值越来越接近 0, 因为主要低频结构已被前序粗尺度解释完毕。
  - 「VAR 重建 MSE」: 当前累计重建图与目标图之间的逐像素均方误差, 每推进一个尺度便显著下降一级, 且码本 K 越大最终重建误差越低。
  - 「本步并行生成 token 数」: 当前尺度一次生成的 token 数量 $s^2$。
  - 「累计 token (VAR / 光栅)」: 走完全程后两者分别为 85 与 64, 可见多尺度金字塔表示的总序列长度反而高于逐像素序列。
  - 「光栅画完 64 个像素还差」: 光栅自回归对照组距离生成完整图像还需补充的串行前向次数。"
    module="llm_models/models/generative/var.py"
    run="python -m llm_models.run_models.generative.var.infer_var"
    :challenge="{
      ask: '将播放进度推进至第 2 次前向 ($2 \\times 2$ 尺度), 对比此时 VAR 的累计重建结果与光栅自回归的生成画面: 哪一种更接近目标图的整体结构? 进一步思考: 生成完整图像各自需要多少次串行前向? VAR 生成的 token 总数比光栅自回归更多还是更少?',
      answer: '答案: 完成 2 次前向后 VAR 的重建画面明显更接近目标图; 走完全程 VAR 仅需 4 次串行前向, 光栅自回归则需要 64 次; 但 VAR 的累计 token 总数为 85, 多于光栅自回归的 64。\n原因分析:\n- 第 2 次前向后的视觉差异: VAR 在第 1 级 ($1 \\times 1$) 确定了全图平均亮度, 第 2 级 ($2 \\times 2$) 进一步将画面划分为四个象限并分别补偿区域残差, 仅用两步便已建立起全图的粗粒度明暗构图, 遵循先全局构图后局部细化的生成次序。\n- 光栅自回归的局限: 完成 2 次前向后仅填充了左上角的 2 个像素, 必须串行推理至第 64 步才能触及右下角像素; 此外, 将二维图像强行展平为一维序列会破坏空间局部性, 原本上下紧邻的两个像素在一维序列中相隔了一整行宽度。\n- 串行前向次数对比: VAR 每次前向并行输出同一尺度的整张 token map (依次为 1、4、16、64 个 token), 4 个尺度合计仅需 4 次前向; 光栅自回归每步仅输出 1 个像素, 必须串行执行 64 次。\n- 累计 token 总数对比: VAR 金字塔各层之和为 $1 + 4 + 16 + 64 = 85$, 高于单层像素总数 64, 序列总长度的增加正是换取尺度内并行生成所付出的计算代价。\n尺度内并行采样不会导致画面撕裂的原因在于残差量化机制: 每一级仅编码目标图与已有粗尺度上采样重建之差。粗尺度已先行锁定全局低频结构, 细尺度各位置只需补充小幅高频修正量, 同级各位置独立采样的偏差幅度有限, 且前序粗尺度的量化误差还可在后续细尺度中得到补偿。\n训练阶段则配合块状因果注意力 mask: 同一尺度内的 token 彼此双向可见, 同时只能关注更粗尺度的历史 token, 从而在单次前向中并行计算全部尺度的训练损失。',
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
        <p>每一尺度级编码的并非原始图像, 而是「目标图像减去已有累计重建」所得的残差, 具体步骤如下:</p>
        <ol class="pts">
          <li><Tex text="先将当前残差图按区域面积平均下采样至 $s \times s$ 分辨率。" /></li>
          <li>每格查最近的码字。</li>
          <li>再放大回 8×8 加到累计重建上。</li>
        </ol>
        <p>因此即便粗尺度的量化颗粒度较粗, 所余下的量化误差仍会完整保留在残差中交由后续细尺度逐步修正。走完全部尺度后:</p>
        <ul class="pts">
          <li><b>VAR:</b> 4 次前向、85 个 token。</li>
          <li><b>光栅:</b> 64 次前向、64 个 token。</li>
        </ul>
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
