<!--
  温度实验台 (对应 llm_infer/m10_sampling 的 temperature, 以及 llm_finetune/methods/distill.py 的软标签)。
  只讲一件事: softmax(z/T) 里的 T 怎么把同一组 logits 调尖或压平。
  柱高按绝对概率 0–1 画: T 变大时最高的柱子矮下去, 矮的长起来。
-->
<template>
  <LabFrame
    title="温度实验台 — 一根滑杆连接采样与蒸馏"
    sub="$\mathrm{softmax}(z/T)$。$T \lt 1$ 让分布更尖 (采样更确定), $T \gt 1$ 把分布压平 (蒸馏的「暗知识」显形)。
      柱高是绝对概率, 每根柱上的横线是 $T = 1$ 时的位置。点一根柱子, 右边给出它相对 $T = 1$ 变了几倍。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '默认这组 logits, 把 T 从 1 拖到 2。先猜: top-1 的柱子会矮一半吗? 最右边的 tok7 涨了几倍?',
      answer: 'top-1 从 0.435 降到 0.294, 只矮了约三分之一。tok7 从 0.003 涨到 0.025, 约 8 倍。\n- 原因: 除以 T 缩小的是 logit 之间的差距。两个 token 的概率比是 $e^{(z_i - z_j)/T}$, 差距越大的一对被拉近得越多。\n- 采样: 长尾 token 被抽到的机会成倍增加, 有效候选数从 3.9 变成 6.0。\n- 蒸馏: teacher 给错误答案排的序这时才看得见。梯度随之缩小到约 $1/T^2$, 所以 KL 项要乘 $T^2$ 补回来。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="seed++">重新随机 logits</button>
        <button type="button" :class="{ active: T === 1 }" @click="T = 1">回到 T = 1</button>
      </div>
      <LabSlider v-model="T" label="温度 T" :min="0.1" :max="5" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="chart">
      <div class="plot">
        <span class="grid-line" style="bottom: 50%"><i class="mono">0.5</i></span>
        <button
          v-for="(p, i) in probs" :key="i" type="button" class="bar-slot" :class="{ sel: i === pick }"
          :aria-label="`tok${i}: 概率 ${p.toFixed(3)}, T=1 时 ${base[i].toFixed(3)}`" :aria-pressed="i === pick"
          @click="pick = i"
        >
          <span class="bar-val mono" :style="{ bottom: `calc(${p * 100}% + 3px)` }">{{ p.toFixed(3) }}</span>
          <!-- ★ 柱高 = 概率本身, 不除以最大值 -->
          <span class="bar" :class="{ top: i === 0 }" :style="{ height: p * 100 + '%' }"></span>
          <span class="ref" :style="{ bottom: base[i] * 100 + '%' }"></span>
        </button>
      </div>
    </div>
    <div class="x-labels">
      <span v-for="i in N" :key="i" class="mono">tok{{ i - 1 }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>top-1 概率</span><b>{{ probs[0].toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="熵 $H(p)$ (nats)" /></span><b>{{ H.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="有效候选数 $\exp(H)$" /></span><b>{{ Math.exp(H).toFixed(1) }} / {{ N }}</b></div>
      <div class="kv"><span>tok{{ pick }} 相对 T = 1</span><b>{{ ratio }}</b></div>
      <p class="regime" :class="{ now: T < 1 }"><Tex text="$T \to 0$: 退化为 argmax (greedy 解码)" /></p>
      <p class="regime" :class="{ now: T === 1 }"><Tex text="$T = 1$: 模型原始分布" /></p>
      <p class="regime" :class="{ now: T > 1 }">
        <Tex text="$T \gt 1$: 次优 token 的相对排序被放大, 蒸馏时 student 学到的不止正确答案 (KL 项要乘 $T^2$ 补偿梯度)" />
      </p>
      <p class="lab-note">
        同一公式两处复用: 解码采样 llm_infer/m10_sampling, 知识蒸馏 llm_finetune/run_finetune/distill。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { entropy, mulberry32, range, softmax } from '@/utils/labmath.js'

const N = 8
const T = ref(1.0)
const seed = ref(1)
const pick = ref(N - 1)   // 默认选最矮的那根: 它的变化倍数最大

// logits 只随 seed 变, 拖 T 不会重抽。降序排列, tok0 永远是 top-1
const logits = computed(() => {
  const rand = mulberry32(seed.value * 2731)
  return range(N).map(() => rand() * 6 - 1).sort((a, b) => b - a)
})
const probs = computed(() => softmax(logits.value, T.value))
const base = computed(() => softmax(logits.value, 1))
const H = computed(() => entropy(probs.value))
const ratio = computed(() => {
  const r = probs.value[pick.value] / base.value[pick.value]
  return '×' + (r >= 100 ? r.toFixed(0) : r >= 0.01 ? r.toFixed(2) : r.toExponential(1))
})
</script>

<style scoped>
.chart { padding: 18px 6px 0; background: var(--code-bg); border-radius: var(--radius-sm); }
.plot { position: relative; display: flex; gap: 4px; height: 192px; }
.bar-slot { flex: 1; position: relative; min-width: 0; min-height: 0; padding: 0; border: 0; border-radius: 0; background: none; cursor: pointer; }
.bar-slot.sel { box-shadow: inset 0 -2px 0 var(--accent); }
.bar { position: absolute; bottom: 0; left: 14%; width: 72%; border-radius: 2px 2px 0 0; background: var(--text-dim); opacity: 0.45; }
.bar.top { background: var(--accent); opacity: 1; }
.bar-slot.sel .bar { opacity: 1; }
.ref { position: absolute; left: 6%; width: 88%; border-top: 2px solid var(--warn); }
.bar-val { position: absolute; left: 0; right: 0; text-align: center; font-size: 10px; color: var(--text-muted); }
.grid-line { position: absolute; left: 0; right: 0; border-top: 1px dashed var(--border); height: 0; }
.grid-line i { position: absolute; right: 0; bottom: 1px; font-size: 9px; font-style: normal; color: var(--text-dim); }
.x-labels { display: flex; gap: 4px; padding: 3px 6px 0; }
.x-labels span { flex: 1; text-align: center; font-size: 10px; color: var(--text-dim); }
.regime { font-size: 12px; color: var(--text-dim); line-height: 1.6; border-left: 2px solid var(--border); padding-left: 10px; }
.regime.now { border-left-color: var(--accent); color: var(--text); }
</style>
