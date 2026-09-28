<!--
  温度配比 (对应 llm_train/m17_data_pipeline:temperature_weights)。
  只讲一件事: p_i ∝ n_i^(1/T) 把小语料抬上来, 代价是它在预算内被重复很多遍。
  库存与预算取自 m17 demo 第 [3] 段: web/code/books/wiki/math = 800/150/40/8/2 M token, 预算 500M。
  T=1/2/5 时前端算出的数与 demo 表逐格相同 (例: T=5 math 0.102 ×50.89 25.4ep)。
-->
<template>
  <LabFrame
    title="温度采样 — 小语料抬多高才合适?"
    sub="5 个来源的库存差 400 倍。灰条是按原始比例采样的占比, 彩条是温度 $T$ 下的占比 $p_i \propto n_i^{1/T}$。
      拖温度滑杆, 点任一来源看它被上采样几倍、在 500M 预算里被过几遍。"
    module="llm_train/m17"
    run="python -m llm_train.m17_data_pipeline.demo"
    :challenge="{
      ask: 'T 从 1 调到 5, math 的占比从 0.2% 涨到 10%。这 50 倍的上采样是白送的吗?',
      answer: '不是白送的。\n- 没有新数据: 上采样只是把同样的 2M token 多看几遍。T=5 时 math 在 500M 预算里被过 25.4 遍。\n- 重复有上限: 小语料重复 4 遍以上收益递减, 再多开始背诵 (Muennighoff 2023)。\n- 另一头: web 被下采样到 ×0.42, 只过 0.2 遍, 大量数据没被用上。\n所以 T 和各源权重在真实系统里靠小模型消融实验定 (DoReMi、RegMix)。本例只演示公式的效果。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="t in [1, 2, 5]" :key="t" type="button" :class="{ active: T === t }" @click="T = t">T = {{ t }}</button>
      </div>
      <LabSlider v-model="T" label="温度 T" :min="1" :max="10" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="mix">
      <button v-for="(src, i) in rows" :key="src.name" type="button" class="src" :class="{ sel: sel === i }" @click="sel = i">
        <span class="name mono">{{ src.name }}</span>
        <span class="bars">
          <span class="bar raw" :style="{ width: src.q * 100 + '%' }" />
          <span class="bar now" :style="{ width: src.p * 100 + '%' }" />
        </span>
        <span class="num mono">{{ src.p.toFixed(3) }} <small :class="src.ep > 4 ? 'over' : ''">{{ src.ep.toFixed(1) }} 遍</small></span>
      </button>
    </div>

    <template #stats>
      <div class="kv"><span>{{ cur.name }}: 库存</span><b>{{ cur.n }}M</b></div>
      <div class="kv"><span>{{ cur.name }}: 上采样倍数</span><b :class="cur.up > 1 ? 'good' : 'bad'">×{{ cur.up.toFixed(2) }}</b></div>
      <div class="kv"><span>{{ cur.name }}: 预算内过几遍</span><b :class="cur.ep > 4 ? 'bad' : ''">{{ cur.ep.toFixed(1) }}</b></div>
      <div class="kv"><span>超过 4 遍的来源</span><b :class="over ? 'bad' : 'good'">{{ over }} 个</b></div>
      <p class="lab-note">★ 上采样倍数 = 新占比 / 原始占比, 过几遍 = 新占比 × 500M / 库存。T=1 时每个来源都是 0.5 遍。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { sum } from '@/utils/labmath.js'

// m17 demo [3]: 库存 (M token) 与预算
const SRC = [['web', 800], ['code', 150], ['books', 40], ['wiki', 8], ['math', 2]]
const BUDGET = 500
const T = ref(2), sel = ref(4)

const rows = computed(() => {
  const w = SRC.map(([, n]) => n ** (1 / T.value)) // ★ p_i ∝ n_i^(1/T)
  const tot = sum(w), totN = sum(SRC.map(([, n]) => n))
  return SRC.map(([name, n], i) => {
    const p = w[i] / tot, q = n / totN
    return { name, n, p, q, up: p / q, ep: (p * BUDGET) / n }
  })
})
const cur = computed(() => rows.value[sel.value])
const over = computed(() => rows.value.filter((r) => r.ep > 4).length)
</script>

<style scoped>
.mix { display: grid; gap: 6px; min-width: 0; }
.src { display: grid; grid-template-columns: 52px 1fr auto; align-items: center; gap: 8px; text-align: left; padding: 6px 8px; }
.src.sel { box-shadow: 0 0 0 2px var(--accent); }
.name { font-size: 12px; }
.bars { position: relative; height: 18px; background: var(--bg-elev); border-radius: 3px; overflow: hidden; }
.bar { position: absolute; left: 0; height: 100%; }
.bar.raw { background: var(--border-strong); top: 0; height: 40%; }
.bar.now { background: var(--accent); top: 40%; height: 60%; }
.num { font-size: 11px; min-width: 88px; text-align: right; }
.num small { color: var(--text-muted); }
.num small.over { color: var(--danger); }
</style>
