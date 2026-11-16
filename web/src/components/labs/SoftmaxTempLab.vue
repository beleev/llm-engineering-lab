<!--
  温度实验台 (对应 llm_infer/m10_sampling 的 temperature, 以及 llm_finetune/methods/distill.py 的软标签)。
  只讲一件事: softmax(z/T) 里的 T 怎么把同一组 logits 调尖或压平。
  柱高按绝对概率 0–1 画: T 变大时最高的柱子矮下去, 矮的长起来。
-->
<template>
  <LabFrame
    title="温度实验台 — 一根滑杆把同一组 logits 调尖或压平"
    sub="- 模拟什么: 模型每生成一步, 都对词表里每个候选 token 给一个分数, 叫 logit; softmax 把这些分数变成加起来等于 1 的概率, 分越高概率越大。采样前常先把所有 logit 除以一个温度 $T$ 再做 softmax: $\mathrm{softmax}(z/T)$。$T$ 小于 1 把高分和低分的差距拉大, 分布更尖, 几乎只会抽到最高分; $T$ 大于 1 把差距缩小, 分布更平, 低分 token 也有机会被抽到。推理时的采样 (llm_infer/m10_sampling) 和知识蒸馏里的软标签 (让学生模型学老师模型的整个分布, 而不只是正确答案) 用的都是这一个公式。
      - 图上是什么: 8 根柱子是 8 个候选 token, 按 logit 从高到低排, tok0 永远是分最高的那个。柱高是这个 token 此刻的概率, 刻度是 0 到 1 的绝对值, 柱顶写着数值; 每根柱上的橙色横线是 $T = 1$ 时它的概率, 方便对比。
      - 可以做什么: 拖「温度 T」从 0.1 到 5。「重新随机 logits」换一组分数, 「回到 T = 1」复位。点一根柱子, 右侧显示它相对 $T = 1$ 时涨了或缩了几倍 (默认选的是最矮的 tok7, 它的变化倍数最大)。
      - 应该看到: $T$ 往小拖, tok0 的柱子涨到接近 1, 其他柱子贴地; $T$ 往大拖, 最高的柱子矮下去, 矮的长起来, 最终趋向 8 根一样高。右侧「top-1 概率」是最高分 token 的概率。「熵 $H(p)$」衡量分布有多「散」: 全押一个 token 时是 0, 8 个一样高时最大 ($\ln 8 \approx 2.08$)。「有效候选数 $\exp(H)$」把熵换算成「相当于在几个 token 里等概率抽」, 更直观: 默认这组 logits 在 $T = 1$ 时约 3.9 个, $T = 2$ 时 6.0 个。下面三行标出当前 $T$ 落在哪个区间。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '默认这组 logits, 把 T 从 1 拖到 2。先猜: top-1 的柱子会矮一半吗? 最右边的 tok7 涨了几倍?',
      answer: '不会矮一半: top-1 从 0.435 降到 0.294, 只矮了约三分之一; tok7 从 0.003 涨到 0.025, 约 8 倍。\n- 原因: 除以 T 缩小的是 logit 之间的差距, 不是概率本身。两个 token 的概率之比是 $e^{(z_i - z_j)/T}$, T 从 1 变 2 等于把所有 logit 差距减半; 原来差距越大的一对 (比如 tok0 和 tok7), 比值被拉近得越厉害, 所以最矮的那根涨得最猛。\n- 对采样的意义: 长尾 token 被抽到的机会成倍增加, 右侧有效候选数从 3.9 变成 6.0, 生成的文本更多样, 也更容易出错。\n- 对蒸馏的意义: T = 1 时老师模型给各个错误答案排的先后几乎看不见 (概率都接近 0), 压平之后才显形, 学生模型能学到「哪些错误答案更接近正确」。代价是梯度随之缩小到约 $1/T^2$, 所以蒸馏的 KL 损失项要乘 $T^2$ 补回来。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="seed++">重新随机 logits</button>
        <button type="button" :class="{ active: T === 1 }" @click="T = 1">回到 T = 1</button>
      </div>
      <LabSlider v-model="T" label="温度 T (logit 先除以它再做 softmax)" :min="0.1" :max="5" :step="0.1" :format="(v) => v.toFixed(1)" />
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
      <div class="kv"><span>top-1 概率 (分最高的 tok0)</span><b>{{ probs[0].toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="熵 $H(p)$ (分布有多散, 单位 nats)" /></span><b>{{ H.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="有效候选数 $\exp(H)$ (相当于在几个里等概率抽)" /></span><b>{{ Math.exp(H).toFixed(1) }} / {{ N }}</b></div>
      <div class="kv"><span>选中的 tok{{ pick }} 的概率相对 T = 1 时</span><b>{{ ratio }}</b></div>
      <p class="regime" :class="{ now: T < 1 }"><Tex text="$T$ 趋向 0: 退化为 argmax, 永远选分最高的那个 (greedy 解码)" /></p>
      <p class="regime" :class="{ now: T === 1 }"><Tex text="$T = 1$: 模型原始分布, 不做任何调整" /></p>
      <p class="regime" :class="{ now: T > 1 }">
        <Tex text="$T \gt 1$: 次优 token 之间的相对排序被放大。蒸馏时学生模型学到的不止正确答案 (KL 项要乘 $T^2$ 补偿变小的梯度)" />
      </p>
      <p class="lab-note">
        同一个公式两处复用: 解码采样在 llm_infer/m10_sampling (temperature_sample 就是 softmax(logits / temperature) 再抽样), 知识蒸馏在 llm_finetune/run_finetune/distill (老师和学生的 logits 都除以同一个 T 再比分布)。
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
