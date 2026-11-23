<!--
  温度配比 (对应 llm_train/m17_data_pipeline:temperature_weights)。
  只讲一件事: p_i ∝ n_i^(1/T) 把小语料抬上来, 代价是它在预算内被重复很多遍。
  库存与预算取自 m17 demo 第 [3] 段: web/code/books/wiki/math = 800/150/40/8/2 M token, 预算 500M。
  T=1/2/5 时前端算出的数与 demo 表逐格相同 (例: T=5 math 0.102 ×50.89 25.4ep)。
-->
<template>
  <LabFrame
    title="数据源温度配比 — 把小语料抬上来, 要付多少重复的代价?"
    sub="- 模拟什么: 预训练语料通常由多个来源组成 (网页、代码、书籍、维基百科、数学), 各来源的原始库存差距悬殊: 本例中 web 拥有 800M token, 而 math 仅有 2M token, 两者相差 400 倍。训练前需确定各来源在总训练预算 (本例为 500M token) 中的采样占比。若按原始库存比例采样, math 仅占 0.2%, 模型在训练中几乎接触不到数学语料。温度配比通过幂次缩放提高小语料占比: 来源 i 的采样概率满足 $p_i \propto n_i^{1/T}$, 其中 $n_i$ 为该来源的库存量, $T$ 为配比温度。当 T 取 1 时退化为原始库存比例; T 越大, 各来源的采样占比越趋于均匀。此处的温度仅控制数据源配比, 与解码阶段的采样温度无关。
- 图上是什么: 每一行代表一个语料来源, 行首标注来源名称。
  - 每行上方的细灰条表示按原始库存比例采样时的基准占比。
  - 下方的粗彩条表示当前温度 T 下的配比占比, 上下对比即可看出该来源被上采样抬高还是下采样压低。
  - 行末显示当前采样占比与「x 遍」读数, 后者表示在 500M token 总预算内该来源的全部库存会被完整遍历的轮数, 超过 4 遍时标红预警。
- 可以做什么: 控件从上到下依次为:
  - 点击「T = 1 / 2 / 5」快捷切换三个常用配比温度。
  - 拖动「温度 T」滑杆在 1 至 10 之间连续调节。
  - 点击任意一行选中对应来源, 右侧面板随即切换为该来源的详细读数。
- 应该看到: 调大温度 T 后, 小规模语料 (wiki、math) 的粗彩条变长, 大规模语料 (web) 的粗彩条缩短, 五个来源的占比逐渐靠拢。当 T 取 5 时, math 被上采样 50.89 倍, 在预算内重复遍历 25.4 遍。右侧读数含义如下:
  - 「库存」: 当前选中来源包含的原始 token 总数。
  - 「上采样倍数」: 当前占比除以原始库存占比, 大于 1 表示上采样抬高, 小于 1 表示下采样压低。
  - 「预算内过几遍」: 等于当前占比乘以 500M 再除以该来源库存, 大于 1 说明同一批语料在训练中被多轮重复采样。
  - 「超过 4 遍的来源」: 统计当前重复轮数超出安全上限的来源个数。"
    module="llm_train/m17"
    run="python -m llm_train.m17_data_pipeline.demo"
    :challenge="{
      ask: '将温度 T 从 1 调至 5 时, math 的采样占比从 0.2% 升至 10%。这 50 倍的上采样是否没有代价?',
      answer: '并非没有代价, 核心代价在于数据重复与大语料利用率下降。\n- 不会凭空增加新数据: 上采样并未扩充真实的 math 语料库, 只是将原有的 2M token 反复训练多轮。按采样占比 0.102 乘以总预算 500M 得 51M token, 再除以库存 2M, 相当于将 math 语料完整重复训练 25.4 遍。\n- 重复采样存在收益上限: 小规模语料在重复 4 遍以内仍能带来有效收益, 超出后边际收益显著递减, 模型容易转向机械记忆这批文本而非习得通用数学推理能力 (参见 Muennighoff 2023 的消融实验)。这也是实验台将「超过 4 遍」标红预警的依据。\n- 大规模语料覆盖率受损: web 的采样权重被压缩至原始比例的 0.42 倍, 在 500M 总预算内仅遍历 0.2 遍, 意味着 800M 的 web 库存中有八成数据完全未被投入训练。\n因此在实际预训练系统中, 配比温度 T 与各来源权重并非凭经验主观设定, 而是通过在小规模代理模型上运行消融实验来搜索最优配比 (如 DoReMi、RegMix 等方法)。本实验台仅演示幂次温度公式本身对采样分布的数学影响。',
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
        <p>★ 两个读数的算法:</p>
        <ul class="pts">
          <li><b>上采样倍数:</b> 新占比 / 原始占比, 大于 1 是被抬高, 小于 1 是被压低。</li>
          <li><b>过几遍:</b> 新占比 × 500M / 库存。预算里分给这个来源的 token 数, 除以它实际有多少, 就是它被从头到尾看几遍。</li>
        </ul>
        <p>当 T 取 1 时按原始库存比例采样, 总训练预算 500M 恰好为总库存 1000M 的一半, 因此每个来源均精确遍历 0.5 遍。</p>
        <p>上采样倍数大于 1 (小语料多轮重复) 与小于 1 (大语料未充分覆盖) 各自存在权衡代价, 因此倍数本身不设优劣颜色标记; 仅当「过几遍」超过 4 时标红提示, 因为小语料重复超过 4 遍后收益明显递减并易引发过拟合记忆。</p>
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
