<!--
  温度实验台 (对应 llm_infer/m10_sampling 的 temperature, 以及 llm_finetune/methods/distill.py 的软标签)。
  只讲一件事: softmax(z/T) 里的 T 怎么把同一组 logits 调尖或压平。
  柱高按绝对概率 0–1 画: T 变大时最高的柱子矮下去, 矮的长起来。
-->
<template>
  <LabFrame
    title="温度实验台 — 一根滑杆把同一组 logits 调尖或压平"
    sub="- 模拟什么: 语言模型在每一步解码时对词表中的所有候选 token 各输出一个未归一化得分 logit, 再经 softmax 映射为总和等于 1 的概率分布, logit 越高对应采样概率越大。实际应用中通常先将全部 logit 除以温度系数 $T$, 再计算 $\mathrm{softmax}(z / T)$: 推理阶段的随机采样解码 (见 llm_infer/m10_sampling) 与知识蒸馏中的软标签对齐 (令学生模型拟合教师模型的完整输出分布而非单一硬标签) 均依赖这一温度缩放公式。温度 $T$ 向两个方向调节分布形态:
  - $T \lt  1$: 拉大高分候选与低分候选之间的相对差距, 使概率分布更加尖锐, 采样结果高度集中于最高分 token。
  - $T \gt  1$: 缩小各候选得分差距, 使概率分布趋于平坦, 原本低概率的长尾 token 获得更高的被采样机会。
- 图上是什么: 柱状图中的 8 根柱子代表按原始 logit 从高到低排序的 8 个候选 token, 最左侧 tok0 始终为得分最高的候选。柱高表示当前温度下该候选的采样概率 (纵轴刻度为 0 至 1, 柱顶直接标出数值); 每根柱体上的橙色短横线标示 $T = 1$ 基准温度下该候选的原始概率, 便于直观对比缩放前后的变化幅度。
- 可以做什么: 操作滑杆、快捷按钮与候选柱:
  - 拖动「温度 T」: 在 0.1 至 5 之间连续调节温度系数。
  - 点击「重新随机 logits」更换一组候选得分, 点击「回到 T = 1」恢复标准未缩放状态。
  - 点击任意一根候选柱: 右侧面板显示该候选当前概率相对于 $T = 1$ 基准值的放大或缩小倍数 (默认选中末尾最低分的 tok7, 其相对倍率变化最为剧烈)。
- 应该看到: 将 $T$ 向小处拖动时, 首选 tok0 的概率迅速逼近 1, 其余候选柱趋近于 0; 将 $T$ 向大处拖动时, 头部高柱回落而尾部矮柱抬升, 整体逐渐趋近 8 根等高的均匀分布。右侧读数含义如下:
  - 「top-1 概率」: 最高分候选 tok0 当前所占的概率值。
  - 「熵 $H(p)$」: 度量概率分布的不确定度; 概率完全集中于单一 token 时熵为 0, 8 个候选严格等概率时达到理论最大值 ($\ln 8 \approx 2.08$)。
  - 「有效候选数 $\exp(H)$」: 即分布的困惑度 (perplexity), 等价换算为「相当于在多少个等概率候选中均匀抽样」; 默认这组 logits 在 $T = 1$ 时约为 3.9 个, 升温至 $T = 2$ 时扩展为 6.0 个。
  - 底部三行状态指示: 标示当前温度 $T$ 所处的典型工作区间。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '在默认这组 logits 下将温度 T 从 1 拖至 2。请先思考: 最高分候选 top-1 的概率会直接减半吗? 最右侧最低分的 tok7 其概率会放大多少倍?',
      answer: '答案: top-1 概率不会减半, 而是从 0.435 降至 0.294, 降幅约为三分之一; 末尾 tok7 的概率则从 0.003 升至 0.025, 放大约 8 倍。\n- 指数缩放机制: 除以温度 $T$ 线性缩小的是各候选 logit 之间的绝对差值, 而非最终概率本身。任意两个候选的概率比值为 $e^{(z_i - z_j)/T}$, 将 $T$ 从 1 调至 2 等价于对原始概率比值开平方 (将指数上的分差减半); 原本与最高分差距越悬殊的尾部候选 (如 tok7 相对于 tok0), 其相对概率比值被拉近的倍数越大, 因此最末位候选的相对涨幅最为显著。\n- 对解码采样的影响: 升温后长尾候选的采样概率成倍抬升, 右侧有效候选数从 3.9 增至 6.0, 生成文本的词汇多样性随之提高, 但事实偏离与语法出错的概率也同步上升。\n- 对知识蒸馏的影响: 在 $T = 1$ 下, 教师模型对各非目标候选赋予的概率往往极度接近 0, 蕴含类间相似关系的次序信息几乎被指数函数抹平; 适当调高蒸馏温度将分布软化后, 暗知识 (即哪些错误候选在语义或结构上更接近正确答案) 才得以充分显现供学生模型学习。由于对输入 logit 求导时梯度幅值会随温度按约 $1/T^2$ 的比例缩小, 知识蒸馏中的软标签 KL 散度损失项通常需乘上 $T^2$ 以补偿梯度量级。',
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
      <div class="lab-note">
        <p>同一个公式两处复用:</p>
        <ul class="pts">
          <li><b>解码采样:</b> 在 llm_infer/m10_sampling (temperature_sample 就是 softmax(logits / temperature) 再抽样)。</li>
          <li><b>知识蒸馏:</b> 在 llm_finetune/run_finetune/distill (老师和学生的 logits 都除以同一个 T 再比分布)。</li>
        </ul>
      </div>
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
