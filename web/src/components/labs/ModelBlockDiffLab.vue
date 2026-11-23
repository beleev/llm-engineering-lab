<!--
  块扩散实验台 (对应 llm_models/models/language_models/block_diffusion.py: block_causal_mask / training_mask / BlockDiffusionLM.sample)。
  只讲一件事: 注意力改成块因果以后, 定稿的块不再变, K/V 能缓存; 训练时用 x_t ⊕ x 的四块掩码一次算完所有块。
  token 计数与 Python 的 sample(return_stats=True) 同一口径: 每块 block_size 步, 每步送当前块, 块定稿后再送一次写 cache。
-->
<template>
  <LabFrame
    title="块扩散 — 块间因果、块内双向, 定稿的块进 KV cache"
    sub="- 模拟什么: BD3-LM 把长度为 16 的序列切成若干块, 块与块之间从左到右生成, 块内部像 LLaDA 一样从全 [MASK] 并行去噪。注意力是块因果的: 一个位置能看到本块和之前所有块, 看不到之后的块, 所以定稿的块不会再变, 它们的 K/V 可以缓存。
- 图上是什么: 方阵的行是 query (谁在看), 列是 key (被看的位置), 亮格表示可见。采样视图里, 正在去噪的草稿块那几行中, 读 cache 的格子和本次新算的格子用两种颜色标出。训练视图把加噪序列 x_t 和干净序列 x 拼成 32 个位置, 三种颜色分别是块对角 M_BD、错一块的块因果 M_OBC 和干净序列的块因果 M_BC。下方一行是生成进度: 已定稿进 cache 的块、当前草稿块和还没轮到的块。
- 可以做什么: 两个按钮切换采样掩码和训练掩码。拖动块大小滑杆在 1、2、4、8、16 之间切换。播放器逐块推进生成。把鼠标移到某一格上 (或用 Tab 选中左侧行号再按回车), 右侧会读出这一行能看到几个 key。
- 应该看到: 块大小为 1 时方阵就是下三角因果掩码, 为 16 时全亮, 也就是 LLaDA 的全双向。右侧读数中, 送进前向的 token 数在块大小为 4 时是 80 个, 不带 cache 时是 160 个, LLaDA 从零生成 16 个 token 要 256 个; 串行前向次数则比 LLaDA 多出 '块数' 次, 那是每块定稿后写 cache 的那次前向。"
    module="llm_models/models/language_models/block_diffusion.py"
    run="python -m llm_models.run_models.language_models.block_diffusion.infer_block_diffusion"
    :challenge="{
      ask: '块大小从 4 拖到 16 时, 带 cache 送进前向的 token 数会怎样变? 它和 LLaDA 的 256 比谁大? 再拖到 1, 方阵变成了什么?',
      answer: '答案: 块大小为 16 时带 cache 的 token 数是 272, 反而比 LLaDA 的 256 多; 块大小为 1 时方阵是下三角因果掩码。\n原因:\n- 块大小 16: 整条序列只有一个块, 去噪时没有任何定稿的前文可以读 cache, 每步都要送全部 16 个 token, 和 LLaDA 一样是 16 × 16 = 256; 本实现在块定稿后还要再送一次写 cache, 所以多出 16 个。cache 只对 \'已经定稿的前文\' 有用。\n- 块大小 4: 每块去噪 4 步, 每步只送当前块的 4 个 token, 前文从 cache 读, 4 块合计 4 × (4 × 4 + 4) = 80, 和 Python 的 infer 脚本打印的 80 一致。\n- 块大小 1: 每块一个 token, 块因果就是因果, 模型退化成自回归。\n块大小是在 \'块内并行\' 和 \'能缓存的前文多少\' 之间的旋钮: 块越大, 一步能同时定的 token 越多, 但它们彼此看不见, 能缓存的部分也越少。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: view === 'sample' }" @click="view = 'sample'">采样掩码 (16 × 16)</button>
        <button type="button" :class="{ active: view === 'train' }" @click="view = 'train'">训练掩码 x_t ⊕ x (32 × 32)</button>
      </div>
      <LabSlider v-model="logBs" label="块大小 block_size" :min="0" :max="4" :format="(v) => String(2 ** v)" />
      <StepPlayer :stepper="stepper" :label="frame < nb ? `生成第 ${frame + 1} / ${nb} 块` : '全部定稿'" />
    </template>

    <div class="mask" :style="{ gridTemplateColumns: `28px repeat(${N}, 11px)` }" @mouseleave="row = null">
      <template v-for="i in N" :key="i">
        <button type="button" class="rh mono" :class="{ sel: row === i - 1 }" :aria-label="`选择第 ${i - 1} 行`" @click="row = i - 1" @keydown.enter="row = i - 1">{{ label(i - 1) }}</button>
        <span v-for="j in N" :key="j" class="c" :class="[kind(i - 1, j - 1), { rowsel: row === i - 1 }]" @mouseenter="row = i - 1" />
      </template>
    </div>
    <p class="legend" v-if="view === 'sample'"><span class="c cache" /> 草稿块读 cache <span class="c draft" /> 草稿块本次新算 <span class="c ok" /> 其余可见</p>
    <p class="legend" v-else><span class="c bd" /> M_BD 块对角 <span class="c obc" /> M_OBC 看之前块的干净 token <span class="c bc" /> M_BC 干净序列块因果 · 前 16 行/列是 x_t, 后 16 是 x</p>
    <div class="prog" :style="{ gridTemplateColumns: `28px repeat(${T}, 11px)` }">
      <span class="rh mono">生成</span>
      <span v-for="i in T" :key="i" class="c" :class="blk(i - 1) < frame ? 'cache' : blk(i - 1) === frame ? 'draft' : 'off'" />
    </div>

    <template #stats>
      <div class="kv"><span>第 {{ row ?? '–' }} 行能看到的 key 数</span><b>{{ row == null ? '–' : seen }}</b></div>
      <div class="kv"><span>送进前向的 token (带 cache / 不带 / LLaDA)</span><b :class="tok.cache < tok.llada ? 'good' : 'bad'">{{ tok.cache }} / {{ tok.none }} / {{ tok.llada }}</b></div>
      <div class="kv"><span>串行前向次数 (带 cache / LLaDA)</span><b :class="T + nb <= T ? 'good' : 'bad'">{{ T + nb }} / {{ T }}</b></div>
      <div class="kv"><span>当前这一步: 从 cache 读 / 新算的 K/V</span><b>{{ Math.min(frame, nb) * bs }} / {{ frame < nb ? bs : 0 }}</b></div>
      <div class="lab-note">
        <p>口径与 Python 相同: 每块去噪 block_size 步, 每步只送当前块; 块定稿后再干净地送一次, 把 K/V 写进 cache。</p>
        <p>写 cache 的那次前向可以并进下一块的第一步一起算, 串行次数就回到 16; 本库为了代码直白分开做。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const T = 16
const view = ref('sample'), logBs = ref(2), row = ref(null)
const bs = computed(() => 2 ** logBs.value)
const nb = computed(() => T / bs.value)
const N = computed(() => (view.value === 'sample' ? T : 2 * T))
const blk = (i) => Math.floor((i % T) / bs.value)                  // 训练视图里 i 和 i + T 是同一位置
const stepper = useStepper(() => nb.value + 1, { interval: 900 })
const frame = computed(() => stepper.step.value)                   // 块 < frame 已定稿, 块 frame 是草稿
const label = (i) => (view.value === 'sample' ? i : i < T ? `t${i}` : `x${i - T}`)

function kind(i, j) {
  if (view.value === 'sample') {
    if (blk(j) > blk(i)) return 'off'                              // ★ 块因果: 看不到之后的块
    if (blk(i) !== frame.value) return 'ok'
    return blk(j) < frame.value ? 'cache' : 'draft'
  }
  const qNoisy = i < T, kNoisy = j < T
  if (qNoisy && kNoisy) return blk(i) === blk(j) ? 'bd' : 'off'    // M_BD
  if (qNoisy) return blk(j) < blk(i) ? 'obc' : 'off'               // M_OBC: 不看本块的干净答案
  if (kNoisy) return 'off'                                         // 干净 query 不看加噪序列
  return blk(j) <= blk(i) ? 'bc' : 'off'                           // M_BC
}
const seen = computed(() => Array.from({ length: N.value }, (_, j) => kind(row.value, j)).filter((k) => k !== 'off').length)
const tok = computed(() => ({
  cache: nb.value * (bs.value * bs.value + bs.value),              // 每块: bs 步 × bs 个 token + 写 cache 的 bs 个
  none: bs.value * bs.value * (nb.value * (nb.value + 1)) / 2,     // 第 b 块每步送 (b + 1)·bs 个
  llada: T * T,                                                    // 16 步, 每步整段 16 个
}))
</script>

<style scoped>
.mask, .prog { display: grid; gap: 1px; width: max-content; }
.prog { margin-top: 10px; }
.c { width: 11px; height: 11px; border-radius: 2px; background: var(--border); }
.c.ok { background: var(--accent-soft); outline: 1px solid var(--accent); outline-offset: -1px; }
.c.cache { background: var(--left); }
.c.draft { background: var(--warn); }
.c.bd { background: var(--warn); }
.c.obc { background: var(--right); }
.c.bc { background: var(--left); }
.c.off { background: var(--bg-elev); }
.c.rowsel { box-shadow: inset 0 0 0 1px var(--text); }
.rh { font-size: 9px; color: var(--text-dim); background: none; border: 0; padding: 0; min-height: 0; height: 11px; line-height: 11px; text-align: right; cursor: pointer; }
.rh.sel { color: var(--accent); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; font-size: 11px; color: var(--text-dim); margin: 8px 0 0; }
</style>
