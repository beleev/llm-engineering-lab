<!--
  混合架构家族实验台 (对应 llm_models/models/language_models/hybrid.py: HybridLM.cache_elements)。
  只讲一件事: 线性层的状态不随长度增长, 全注意力层的缓存随长度线性增长; 排布比例决定长上下文下的缓存。
  数字用 Python 玩具配置 (d_model 64, 4 头, 头维 16, MLA latent 32 + rope 16), 与 infer 脚本逐个对得上。
-->
<template>
  <LabFrame
    title="混合架构的缓存账 — 3:1 排布省下来的是哪一部分"
    sub="- 模拟什么: 2025-2026 的混合架构按层交替两种序列混合器。线性注意力层 (Gated DeltaNet 或 Kimi Delta Attention) 把历史压进一个固定大小的状态矩阵, 大小与读过多少 token 无关; 全注意力层 (GQA、带输出门的 GQA、或不加位置编码的 MLA) 要把每个 token 的 K/V 都留着。缓存元素数的算式是 '全注意力层数 × T × 每 token 的 K/V' 加 '线性层数 × 一个状态'。
- 图上是什么: 每一行是一层, 从第 0 层排到最后一层。蓝色是线性层, 橙色是全注意力层; 横条的长度与这一层在当前长度 T 下的缓存元素数成正比。右上角写着这一层的算式。
- 可以做什么: 三个按钮切换排布预设 (Kimi 式 KDA + NoPE MLA、Qwen 式 Gated DeltaNet + 门控全注意力、全注意力基线)。两个滑杆分别调序列长度 T 和 '每几层线性配一层全注意力'。点任意一行 (或选中后按回车) 把那一层在线性和全注意力之间切换。
- 应该看到: T 调大时只有橙色的条变长, 蓝色的条一动不动; 比例从 1:1 调到 7:1 时总缓存接近按全注意力层数等比缩小。T 很小时线性层的定长状态反而是大头, 所以混合架构的好处要到长上下文才显出来。MLA 的全注意力层每 token 只存 48 个数, GQA 要 128 个, 所以 Kimi 式排布比 Qwen 式更省。"
    module="llm_models/models/language_models/hybrid.py"
    run="python -m llm_models.run_models.language_models.hybrid.infer_hybrid"
    :challenge="{
      ask: 'T 调到最小、比例调成 7:1, 总缓存里定长状态占多少? 再把 T 拉到最大看一次。为什么论文谈混合架构的收益时总要先说上下文有多长?',
      answer: '答案: T = 32 时定长状态是大头 (线性层每层 1024 个元素, 全注意力层每 token 才 48 到 128 个); T 拉到 1024 以上, 定长那部分的占比掉到个位数百分比。\\n原因:\\n- 两种缓存的量纲不同: 线性层是 H × Dh² 的状态矩阵, 一次性开销; 全注意力层是 T × 2 × Hkv × Dh, 随长度线性长。短上下文下常数项占优, 长上下文下线性项占优。\\n- 所以 3:1 的收益是渐近的: 1M 上下文下 Kimi Linear 报告 KV cache 省到 1/4、解码吞吐最高 6 倍, 这些数字在 4K 上下文上是看不出来的。\\n- 省缓存不等于免费: 固定大小的状态是有损压缩, 长程精确检索要靠那 1/4 的全注意力层兜底。Kimi 把这些层做成不加位置编码的 MLA, 位置信息交给 KDA 的衰减门隐式携带。\\n- Python 实测 (infer 脚本, T = 64): Kimi 式 6,144 个元素, Qwen 式 11,264, 全注意力 32,768。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: preset === 'kimi' }" @click="setPreset('kimi')">Kimi 式 (KDA + NoPE MLA)</button>
        <button type="button" :class="{ active: preset === 'qwen' }" @click="setPreset('qwen')">Qwen 式 (GDN + 门控 GQA)</button>
        <button type="button" :class="{ active: preset === 'full' }" @click="setPreset('full')">全注意力基线</button>
      </div>
      <LabSlider v-model="tExp" label="序列长度 T" :min="5" :max="14" :format="() => T.toLocaleString()" />
      <LabSlider v-model="ratio" label="每几层线性配一层全注意力" :min="0" :max="7" :format="(v) => (v === 0 ? '全注意力' : `${v} : 1`)" />
    </template>

    <svg :viewBox="`0 0 400 ${L * 18 + 16}`" class="stack" role="group" aria-label="每层的缓存占用, 可点击切换这一层的类型">
      <g v-for="(k, i) in kinds" :key="i" :transform="`translate(0, ${i * 18})`">
        <text x="0" y="13" class="lbl mono">L{{ i }}</text>
        <rect x="22" y="3" :width="Math.max(2, 230 * elems(k) / maxElem)" height="12"
              :class="['bar', k === 'linear' ? 'lin' : 'att']"
              tabindex="0" role="button" :aria-label="`第 ${i} 层, ${k === 'linear' ? '线性' : '全注意力'}, 点击切换`"
              @click="toggle(i)" @keydown.enter="toggle(i)" />
        <text x="258" y="13" class="lbl">{{ formula(k) }}</text>
      </g>
      <g :transform="`translate(0, ${L * 18 + 2})`">
        <text x="0" y="9" class="lbl">合计</text>
        <rect x="22" y="0" :width="Math.max(1, 230 * fixed / total)" height="10" class="bar lin" />
        <rect :x="22 + 230 * fixed / total" y="0" :width="Math.max(1, 230 * (total - fixed) / total)" height="10" class="bar att" />
        <text x="258" y="9" class="lbl">定长 {{ (100 * fixed / total).toFixed(0) }}% / 随 T {{ (100 * (total - fixed) / total).toFixed(0) }}%</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>缓存元素数 (全注意力基线 {{ baseline.toLocaleString() }})</span><b :class="total < baseline ? 'good' : 'bad'">{{ total.toLocaleString() }}</b></div>
      <div class="kv"><span>相对全注意力基线</span><b :class="total / baseline < 1 ? 'good' : 'bad'">{{ (100 * total / baseline).toFixed(1) }}%</b></div>
      <div class="kv"><span>全注意力层数 (共 {{ L }} 层)</span>
        <b :class="attnLayers < L ? 'good' : 'bad'">{{ attnLayers }}</b></div>
      <div class="kv"><span>其中定长状态 (不随 T 变)</span><b :class="fixed / total < 0.5 ? 'good' : ''">{{ (100 * fixed / total).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p>算式按 Python 玩具配置: d_model 64、4 个头、头维 16。全注意力层每 token 存 2 × 4 × 16 = 128 个数, NoPE 的 MLA 层只存 latent 32 加 rope 段 16 = 48 个, 线性层一个状态是 4 × 16 × 16 = 1,024 个, 与长度无关。</p>
        <p>Python 实测, 不随左侧变化: 4 层模型在两个长度下的缓存元素数, 与算式逐个相等。</p>
        <ul class="pts">
          <li v-for="r in measured" :key="r.name"><b>{{ r.name }}:</b> T=32 {{ r.t32 }} · T=64 {{ r.t64 }} ({{ r.note }})</li>
        </ul>
        <p>四种排布的逐 token 增量生成与整段前向最大差 6.0e-07。注意 iRoPE 那一行没有省: 本库的分块只改可见性, K/V 仍然整条缓存着, 真实实现会像滑动窗口那样滚动裁剪。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const L = 12                                   // 展示 12 层
const ATT_GQA = 2 * 4 * 16                     // 全注意力层每 token 的 K/V 元素数
const ATT_MLA = 32 + 16                        // NoPE 的 MLA 层每 token 只存 latent + rope 段
const STATE = 4 * 16 * 16                      // 线性层一个状态矩阵

// Python infer 脚本的实测表 (4 层), 不随左侧控件变化
const measured = [
  { name: 'Kimi 式 kda ×3 + mla_nope', t32: '4,608', t64: '6,144', note: '定长 3,072 不动' },
  { name: 'Qwen 式 gdn ×3 + gated_gqa', t32: '7,168', t64: '11,264', note: '定长 3,072 不动' },
  { name: 'iRoPE 分块 RoPE ×3 + NoPE', t32: '16,384', t64: '32,768', note: '全部随 T 增长' },
  { name: '全注意力基线 gqa ×4', t32: '16,384', t64: '32,768', note: '全部随 T 增长' },
]
const preset = ref('kimi'), tExp = ref(10), ratio = ref(3)
const overrides = ref({})                      // 读者点过的层: 下标 → 'linear' | 'attn'
const T = computed(() => 1 << tExp.value)
const attPerToken = computed(() => (preset.value === 'kimi' ? ATT_MLA : ATT_GQA))

// 排布: 每 ratio 层线性之后放一层全注意力; ratio = 0 表示全是全注意力
const kinds = computed(() => Array.from({ length: L }, (_, i) => {
  if (overrides.value[i]) return overrides.value[i]
  if (ratio.value === 0) return 'attn'
  return (i + 1) % (ratio.value + 1) === 0 ? 'attn' : 'linear'
}))
const elems = (k) => (k === 'linear' ? STATE : attPerToken.value * T.value)   // ★ 一个随 T 长, 一个不随
const total = computed(() => kinds.value.reduce((s, k) => s + elems(k), 0))
const fixed = computed(() => kinds.value.filter((k) => k === 'linear').length * STATE)
const attnLayers = computed(() => kinds.value.filter((k) => k === 'attn').length)
const baseline = computed(() => L * ATT_GQA * T.value)
const maxElem = computed(() => Math.max(...kinds.value.map(elems), 1))
const formula = (k) => (k === 'linear' ? `状态 ${STATE}` : `${attPerToken.value} × T`)

function setPreset(p) {
  preset.value = p
  overrides.value = {}
  ratio.value = p === 'full' ? 0 : 3
}
function toggle(i) {
  overrides.value = { ...overrides.value, [i]: kinds.value[i] === 'linear' ? 'attn' : 'linear' }
}
</script>

<style scoped>
.stack { width: 100%; min-width: 400px; display: block; }
.bar { cursor: pointer; }
.bar.lin { fill: var(--left); }
.bar.att { fill: var(--warn); }
.bar:focus { outline: none; stroke: var(--text); stroke-width: 1.5; }
.lbl { font-size: 9px; fill: var(--text-dim); }
</style>
