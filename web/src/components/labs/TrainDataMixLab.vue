<!--
  温度配比 (对应 llm_train/m17_data_pipeline:temperature_weights)。
  只讲一件事: p_i ∝ n_i^(1/T) 把小语料抬上来, 代价是它在预算内被重复很多遍。
  库存与预算取自 m17 demo 第 [3] 段: web/code/books/wiki/math = 800/150/40/8/2 M token, 预算 500M。
  T=1/2/5 时前端算出的数与 demo 表逐格相同 (例: T=5 math 0.102 ×50.89 25.4ep)。
-->
<template>
  <LabFrame
    title="数据源温度配比 — 把小语料抬上来, 要付多少重复的代价?"
    sub="- 模拟什么: 预训练语料来自几个来源 (网页、代码、书、维基、数学), 库存差得很远: 本例 web 有 800M token, math 只有 2M, 差 400 倍。训练前要定每个来源在总预算 (本例 500M token) 里占多大比例。按库存原比例采样, math 只占 0.2%, 模型几乎见不到它。温度配比是把小语料抬上来的一个公式: 来源 i 的占比 $p_i \propto n_i^{1/T}$, 其中 $n_i$ 是它的库存, $T$ 是温度。T=1 就是原比例; T 越大, 各来源的占比越接近平均。这里的温度管的是来源占比, 和解码采样时的温度不是一回事。
      - 图上是什么: 每一行是一个来源, 行首是名字。行里上面那条细灰条是按库存原比例采样时它的占比, 下面那条粗彩条是当前温度 T 下的占比, 两条一对比就看出被抬高还是压低。行末的数字是当前占比, 后面的「x 遍」是在 500M 预算内这个来源的全部数据会被从头到尾看几遍, 超过 4 遍标红。
      - 可以做什么: 点「T = 1 / 2 / 5」跳到三个常用温度; 拖「温度 T」滑杆在 1 到 10 之间连续调。点任一行选中那个来源, 右侧读数就换成它的。
      - 应该看到: T 调大, 粗条在小来源 (wiki、math) 上变长, 在 web 上变短, 五条越来越接近。右侧「库存」是这个来源一共有多少 token; 「上采样倍数」是新占比除以原始占比, 大于 1 是被抬高, 小于 1 是被压低; 「预算内过几遍」是新占比 × 500M ÷ 库存, 大于 1 说明同一批数据在被重复训; 「超过 4 遍的来源」数有几个来源重复得过头了。T=5 时 math 被上采样 50.89 倍, 过 25.4 遍。"
    module="llm_train/m17"
    run="python -m llm_train.m17_data_pipeline.demo"
    :challenge="{
      ask: 'T 从 1 调到 5, math 的占比从 0.2% 涨到 10%。这 50 倍的上采样是白送的吗?',
      answer: '不是白送的, 代价是重复。\n- 没有新数据: 上采样不会凭空多出 math 数据, 只是把同样的 2M token 多看几遍。占比 0.102 乘预算 500M 是 51M token, 除以库存 2M, 等于把 math 从头到尾过 25.4 遍。\n- 重复有上限: 小语料重复 4 遍以内收益还可以, 再多就开始递减, 模型会把这几篇背下来而不是学会数学 (Muennighoff 2023 的实验)。这就是读数里「超过 4 遍」标红的来由。\n- 另一头也在亏: web 被压到原来的 0.42 倍, 500M 预算里只过了它的 0.2 遍, 也就是 800M 的 web 数据有八成根本没被用上。\n所以温度 T 和各来源的权重在真实系统里不是拍脑袋定的, 要靠在小模型上做消融实验来选 (DoReMi、RegMix 这类方法)。这个实验台只演示公式本身会把比例推向哪里。',
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
      <div class="kv"><span>{{ cur.name }}: 库存 (这个来源一共有多少 token)</span><b>{{ cur.n }}M</b></div>
      <div class="kv"><span>{{ cur.name }}: 上采样倍数 (新占比 ÷ 原始占比)</span><b>×{{ cur.up.toFixed(2) }}</b></div>
      <div class="kv"><span>{{ cur.name }}: 预算内过几遍 (同一批数据被重复看的次数)</span><b :class="cur.ep > 4 ? 'bad' : ''">{{ cur.ep.toFixed(1) }}</b></div>
      <div class="kv"><span>超过 4 遍的来源 (重复过头的)</span><b :class="over ? 'bad' : 'good'">{{ over }} 个</b></div>
      <div class="lab-note">
        <p>★ 上采样倍数 = 新占比 / 原始占比, 大于 1 是被抬高, 小于 1 是被压低。过几遍 = 新占比 × 500M / 库存: 预算里分给这个来源的 token 数, 除以它实际有多少, 就是它被从头到尾看几遍。</p>
        <p>T=1 时按库存原比例采样, 预算 500M 是总库存 1000M 的一半, 所以每个来源都恰好过 0.5 遍。</p>
        <p>倍数大于 1 (小语料重复) 和小于 1 (大语料没用完) 各有代价, 所以倍数不标好坏。只有「过几遍」超过 4 才标红: 小语料重复 4 遍以上收益递减, 再多就开始背诵。</p>
      </div>
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
