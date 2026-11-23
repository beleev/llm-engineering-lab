<!-- KV 体积实验台 (对应 llm_infer/m18_kv_attention_variants): 两条公式 + 一条可拖的显存线。 -->
<template>
  <LabFrame
    title="KV 体积 — MHA / GQA / MQA / MLA 每个 token 各占多少显存, 一张卡能同时服务几条"
    sub="- 模拟什么: 单张显卡可同时服务的并发请求数上限取决于「分配给 KV cache 的显存预算 ÷ 单条请求的 KV 体积」。四种注意力机制的核心计算形式相近, 本实验仅核算其显存占用; 每个 token 的 KV 字节数取决于具体结构在 cache 中缓存的内容:
  - MHA (multi-head attention): 每个 query 头各自拥有独立的 K/V 头并全部缓存。
  - GQA (grouped-query attention): 多个 query 头共享一组 KV 头, 缓存份数成比例减少。
  - MQA (multi-query attention): 所有 query 头共享 1 组 KV 头。
  - MLA (multi-head latent attention, DeepSeek 采用): 不直接缓存 K 和 V, 仅缓存一个低维潜向量 (每层维度为 d_c, 计算时乘以投影矩阵还原各头的 K/V) 以及一份所有头共享的 64 维 RoPE key (单独承载位置信息)。
- 图上是什么: 上下两幅视图:
  - 上方四根横条: 四种结构下每个 token 的 KV 字节数 (KiB), 条长按 MHA 归一化。红色为 MHA, 黄色为 GQA, 绿色为 MQA, 紫色为 MLA; 标有「实际采用」的一行代表所选预设模型实际使用的结构。
  - 下方视图: 每种结构占一行, 每个小方格代表一条序列在所选上下文长度下的完整 KV 占用 (行首标注单条序列的 GiB 占用与可容纳条数); 横轴为显存容量 (GiB), 可拖动的竖线代表分配给 KV cache 的显存预算, 连 1 条序列都无法容纳时显示红色虚线框。
- 可以做什么: 按界面从上到下依次操作:
  - 点击预设按钮一键填入真实模型参数 (LLaMA-2-7B、LLaMA-3-8B、LLaMA-3-70B、DeepSeek-V3)。
  - 拖动「层数 n_layer」「query 头数 n_head」「GQA 的 KV 头数」「d_head」(单头维度) 与「MLA latent d_c」修改模型维度配置 (手动拖动后即脱离预设模型)。
  - 拖动「上下文长度」修改单条序列包含的 token 数。
  - 点击 fp16 / int8 / int4 切换 KV 存储精度 (每个数值分别占 2 / 1 / 0.5 字节)。
  - 左右拖动图中的竖线 (或聚焦后按左右方向键) 调整显存预算。
- 应该看到: 同一预设下 MHA 横条最长, GQA 按 KV 头数等比缩短, MQA 最短; DeepSeek-V3 预设下 MLA 每 token 仅占 68.6 KiB, 比同配置 MHA 缩小 56.9×。右侧读数展示每种结构在当前显存预算与上下文长度下最多可同时容纳的序列条数, 无法容纳任何序列时标红, 可容纳条数最多的结构标绿。默认 40 GiB 预算、8192 上下文长度下, LLaMA-3-8B 预设的 MHA / GQA / MQA 分别为 10 / 40 / 320 条, DeepSeek-V3 预设的 MLA 为 74 条, 与 Python demo 结果一致。"
    module="llm_infer/m18"
    run="python -m llm_infer.m18_kv_attention_variants.demo"
    :challenge="{
      ask: '选择 DeepSeek-V3 预设时, 模型拥有 128 个注意力头, 但 MLA 每 token 仅占 68.6 KiB, 甚至低于 GQA-8 (LLaMA-3-8B 的 128 KiB)。MLA 是通过什么机制实现这一压缩比的? 对应的工程代价是什么?',
      answer: 'MLA 不按注意力头分别缓存 K 和 V, 而是仅缓存以下两部分向量:\n- 一份所有头共享的 512 维低秩潜向量 $c_{\\text{kv}}$, 计算注意力时再乘以投影矩阵还原出各头的 K 和 V。\n- 一份所有头共享的 64 维 RoPE key, 单独用于承载位置编码信息。\n按每层每 token 缓存 $512 + 64$ 个数值、共 61 层、fp16 精度每个数值占 2 字节计算, 单 token 占用为 $(512 + 64) \\cdot 61 \\cdot 2 = 70272$ 字节, 即 68.6 KiB。同架构若采用标准 MHA, 每 token 需缓存 128 个头、每头 128 维的 K 和 V (乘 2)、共 61 层、每数 2 字节, 合计 3904 KiB, MLA 相比缩小了 56.9×。\n对应的代价在于计算量增加: decode 阶段需要将潜向量还原为各头的 K/V, 或采用矩阵吸收 (absorb) 形式将上投影矩阵 $W_{\\text{UK}}$ 合并至查询向量 $q$ 中直接在潜空间计算注意力, 其浮点运算量高于 GQA。这一设计本质上是用算力换取显存容量与访存带宽; 由于 decode 阶段属于典型的访存带宽受限阶段、算力通常存在富余, 因此这一置换在工程上是划算的。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in PRESETS" :key="p.name" type="button" :class="{ active: presetName === p.name }" @click="apply(p)">{{ p.name }}</button>
      </div>
      <LabSlider v-model="L" label="层数 n_layer" :min="8" :max="96" />
      <LabSlider v-model="hExp" label="query 头数 n_head" :min="3" :max="7" :format="(v) => 2 ** v" />
      <LabSlider v-model="kvExp" label="GQA 的 KV 头数" :min="0" :max="hExp" :format="(v) => 2 ** v" />
      <LabSlider v-model="dhExp" label="d_head" :min="6" :max="8" :format="(v) => 2 ** v" />
      <LabSlider v-model="dc" label="MLA latent d_c (+64 rope)" :min="128" :max="1024" :step="64" />
      <LabSlider v-model="ctxExp" label="上下文长度" :min="10" :max="17" :format="(v) => 2 ** v" unit=" tok" />
      <div class="row">
        <span class="hint">KV 精度:</span>
        <button v-for="d in DTYPES" :key="d.name" type="button" :class="{ active: nbytes === d.b }" @click="nbytes = d.b">{{ d.name }}</button>
      </div>
    </template>

    <div class="bars">
      <div v-for="v in variants" :key="v.name" class="bar-row" :class="{ native: v.name === native }">
        <span class="bar-name mono">{{ v.name }}<small>{{ v.what }}{{ v.name === native ? ' · 实际采用' : '' }}</small></span>
        <span class="bar-track"><span class="bar-fill" :style="{ width: (v.bytes / variants[0].bytes) * 100 + '%', background: v.color }" /></span>
        <span class="bar-val mono">{{ fmtKiB(v.bytes) }}</span>
      </div>
    </div>

    <svg ref="svg" :viewBox="`0 0 ${W} 190`" role="group" aria-label="每种结构在显存预算内能放几条序列">
      <g v-for="(v, i) in variants" :key="v.name" :transform="`translate(0, ${18 + i * 38})`">
        <text x="0" y="-4" class="ax">{{ v.name }} · 每条 {{ v.gib.toFixed(2) }} GiB · 放得下 {{ v.fit }} 条</text>
        <template v-if="v.gib * sx >= 3">
          <rect
            v-for="k in v.fit" :key="k" :x="(k - 1) * v.gib * sx + 0.5" y="0"
            :width="v.gib * sx - 1" height="18" :fill="v.color" opacity="0.75"
          />
        </template>
        <rect v-else x="0" y="0" :width="v.fit * v.gib * sx" height="18" :fill="v.color" opacity="0.75" />
        <rect v-if="!v.fit" x="0" y="0" :width="Math.min(W, v.gib * sx)" height="18" class="over" />
      </g>
      <line v-for="g in [0, 40, 80, 120, 160]" :key="g" :x1="g * sx" :x2="g * sx" y1="166" y2="172" class="tick" />
      <text v-for="g in [0, 40, 80, 120, 160]" :key="'t' + g" :x="Math.min(W - 2, Math.max(2, g * sx))" y="186" class="ax" :text-anchor="g === 0 ? 'start' : g === 160 ? 'end' : 'middle'">{{ g }} GiB</text>
      <g
        class="draggable focus-inset" tabindex="0" role="slider" aria-label="KV 显存预算" :aria-valuenow="budget" aria-valuemin="4" aria-valuemax="160"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (budget = clamp(Math.round(x / sx), 4, 160)) })"
        @keydown.left.prevent="budget = clamp(budget - 1, 4, 160)" @keydown.right.prevent="budget = clamp(budget + 1, 4, 160)"
      >
        <rect :x="budget * sx - 12" y="0" width="24" height="172" fill="transparent" />
        <line :x1="budget * sx" :x2="budget * sx" y1="0" y2="172" class="budget" />
        <circle :cx="budget * sx" cy="166" r="7" class="knob" />
        <text :x="budget * sx + (budget > 130 ? -10 : 10)" y="10" class="budget-t" :text-anchor="budget > 130 ? 'end' : 'start'">⇆ KV 预算 {{ budget }} GiB</text>
      </g>
    </svg>

    <template #stats>
      <div v-for="v in variants" :key="v.name" class="kv">
        <span>{{ v.name }} 在预算内能同时放几条</span>
        <b :class="v.fit === 0 ? 'bad' : v.fit === best ? 'good' : ''">{{ v.fit }} 条</b>
      </div>
      <div class="lab-note">
        <p>每 token 字节, MHA 是 MLA 的 {{ (variants[0].bytes / variants[3].bytes).toFixed(1) }}×。标着「实际采用」的那一行是所选模型实际用的结构; 拖过滑杆后不再对应任何预设。</p>
        <ul class="pts">
          <li><b>每 token 字节 (MHA / GQA / MQA):</b> <Tex text="$2 \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot n_{\text{layer}} \cdot \text{bytes}$" />。2 是 K 和 V 各一份, <Tex text="$n_{\text{kv}}$" /> 是 KV 头数 (MHA 等于 query 头数, MQA 等于 1)。</li>
          <li><b>MLA:</b> <Tex text="$(d_c + d_{\text{rope}}) \cdot n_{\text{layer}} \cdot \text{bytes}$。K 与 V 共享同一份潜向量, 因此公式中既无系数「$2 \cdot$」, 也不乘以注意力头数。" /></li>
          <li><b>并发上限:</b> <Tex text="$\lfloor \text{预算} \div (\text{每 token 字节} \times \text{上下文长度}) \rfloor$" />。</li>
        </ul>
        <p>与 Python demo 的核算结果一致: 在 40 GiB 显存预算、8192 上下文长度下, 7B 规模模型的 MHA / GQA-8 / MQA 可容纳序列数分别为 10 / 40 / 320 条, DeepSeek-V3 的 MLA 为 74 条。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const PRESETS = [
  { name: 'LLaMA-2-7B', L: 32, h: 5, kv: 5, dh: 7, dc: 512, native: 'MHA' },   // 32 个 KV 头: 没有分组, GQA 那一行和 MHA 一样大
  { name: 'LLaMA-3-8B', L: 32, h: 5, kv: 3, dh: 7, dc: 512, native: 'GQA' },
  { name: 'LLaMA-3-70B', L: 80, h: 6, kv: 3, dh: 7, dc: 512, native: 'GQA' },
  { name: 'DeepSeek-V3', L: 61, h: 7, kv: 3, dh: 7, dc: 512, native: 'MLA' },
]
const DTYPES = [{ name: 'fp16', b: 2 }, { name: 'int8', b: 1 }, { name: 'int4', b: 0.5 }]
const D_ROPE = 64, GIB = 2 ** 30, W = 640, sx = W / 160

const L = ref(32), hExp = ref(5), kvExp = ref(3), dhExp = ref(7), dc = ref(512)
const ctxExp = ref(13), nbytes = ref(2), budget = ref(40)
const apply = (p) => { L.value = p.L; hExp.value = p.h; kvExp.value = p.kv; dhExp.value = p.dh; dc.value = p.dc }
// 选中的预设从滑杆的值反推: 滑杆一动, 对不上任何预设, 高亮就消失
const preset = computed(() => PRESETS.find((p) => p.L === L.value && p.h === hExp.value && p.kv === kvExp.value && p.dh === dhExp.value && p.dc === dc.value))
const presetName = computed(() => preset.value?.name)
const native = computed(() => preset.value?.native)

watch(hExp, (h) => { kvExp.value = Math.min(kvExp.value, h) }) // KV 头数不能超过 query 头数

const svg = ref(null)
const { start } = useDrag()

const variants = computed(() => {
  const H = 2 ** hExp.value, G = 2 ** Math.min(kvExp.value, hExp.value), dh = 2 ** dhExp.value, ctx = 2 ** ctxExp.value
  const gqa = (nkv) => 2 * nkv * dh * L.value * nbytes.value          // ★ 2 = K 和 V
  const rows = [
    { name: 'MHA', what: `${H} kv 头`, bytes: gqa(H), color: 'var(--danger)' },
    { name: 'GQA', what: G === H ? `${G} kv 头 = MHA` : `${G} kv 头`, bytes: gqa(G), color: 'var(--warn)' },
    { name: 'MQA', what: '1 kv 头', bytes: gqa(1), color: 'var(--left)' },
    { name: 'MLA', what: `latent ${dc.value}+${D_ROPE}`, bytes: (dc.value + D_ROPE) * L.value * nbytes.value, color: 'var(--accent)' },
  ]
  return rows.map((r) => ({ ...r, gib: (r.bytes * ctx) / GIB, fit: Math.floor((budget.value * GIB) / (r.bytes * ctx)) }))
})
const best = computed(() => Math.max(...variants.value.map((v) => v.fit)))
const fmtKiB = (b) => (b / 1024 >= 100 ? (b / 1024).toFixed(0) : (b / 1024).toFixed(1)) + ' KiB'
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.hint { font-size: 12px; color: var(--text-muted); }
.bars { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.bar-row { display: grid; grid-template-columns: 96px 1fr 76px; gap: 8px; align-items: center; font-size: 12px; }
.bar-name { color: var(--text); }
.bar-name small { display: block; font-size: 10px; color: var(--text-dim); }
.bar-row.native .bar-name { color: var(--accent); }
.bar-track { height: 14px; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill { display: block; height: 100%; min-width: 2px; }
.bar-val { text-align: right; color: var(--text-muted); }
.ax { font-size: 11px; fill: var(--text-muted); }
.tick { stroke: var(--border-strong); }
.over { fill: none; stroke: var(--danger); stroke-dasharray: 4 3; }
.budget { stroke: var(--text); stroke-width: 2; }
.knob { fill: var(--text); }
.budget-t { font-size: 11px; fill: var(--text); }
</style>
