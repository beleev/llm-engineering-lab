<!--
  MTP 实验台 (对应 llm_models/models/language_models/mtp.py 和 training/loss.py 的 MTPLoss)。
  只讲一件事: 第 k 级 MTP 在位置 i 拼接真实 token t(i+k) 的 embedding, 预测 t(i+1+k);
  序列末尾 k 个位置没有未来 token, 标签填 -100。
-->
<template>
  <LabFrame
    title="MTP 实验台 — 一次前向, 多步预测"
    sub="主干在位置 $i$ 预测 $t_{i+1}$; 每个 MTP 级联模块拼接一个真实 next-token 的 embedding, 再多看一步。
      序列共 8 个 token, 前 7 个是输入。点第一排的 token 换聚焦位置。"
    module="llm_models/mtp"
    run="python -m llm_models.run_models.language_models.mtp.train_mtp"
    :challenge="{
      ask: '把 K 拖到 2, 依次点 t4、t5、t6。哪个位置开始有预测越界? 整条序列一共多出几个监督目标?',
      answer: 't4 三级都有目标。t5 的 MTP-2 要预测 t8, 越界。t6 的 MTP-1 和 MTP-2 都越界。\n- 规则: 第 $k$ 级的标签是 labels 左移 $k$ 位, 末尾 $k$ 个位置填 -100。\n- 数量: 主干 7 个目标, MTP-1 有 6 个, MTP-2 有 5 个。一次前向从 7 个目标变成 18 个。\n- 代价: 每级多一个拼接投影和一个 Block。2 层玩具主干上每级 +50.2% 参数, 61 层的 DeepSeek-V3 上约 1.5%。',
    }"
  >
    <template #controls>
      <LabSlider v-model="K" label="MTP 深度 K" :min="1" :max="3" />
      <LabSlider v-model="lambda" label="λ (MTP 损失权重)" :min="0" :max="1" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="strip-area">
      <p class="strip-title">输入序列 (聚焦位置 i = {{ focus }}, 点击切换)</p>
      <div class="chip-row">
        <button
          v-for="t in N" :key="t" type="button" class="chip mono" :class="{ focus: t - 1 === focus }"
          :disabled="t === N" :aria-pressed="t - 1 === focus" @click="focus = t - 1"
        >t{{ t - 1 }}</button>
      </div>
      <div v-for="row in rows" :key="row.label" class="pred-row">
        <div class="pred-head">
          <span class="row-label mono">{{ row.label }}</span>
          <span class="row-desc mono" :class="{ muted: row.oob }">{{ row.desc }}</span>
        </div>
        <div v-if="!row.oob" class="chip-row">
          <span v-for="t in N" :key="t" class="chip mono" :class="chipClass(row, t - 1)">t{{ t - 1 }}</span>
        </div>
      </div>
      <p class="strip-title">
        <span class="chip mono consume">拼接的 embedding</span>
        <span class="chip mono predict">MTP 预测的目标</span>
        <span class="chip mono predict-main">主 head 预测的目标</span>
      </p>
    </div>

    <template #stats>
      <div class="kv"><span>位置 {{ focus }} 有目标的预测</span><b :class="st.live === K + 1 ? 'good' : 'bad'">{{ st.live }} / {{ K + 1 }}</b></div>
      <div class="kv"><span>全序列监督目标 (主干单独 {{ N - 1 }})</span><b>{{ st.targets }}</b></div>
      <div class="kv"><span><Tex text="初始 loss $(1+\lambda)\ln V$" /></span><b>{{ st.loss0.toFixed(2) }}</b></div>
      <div class="kv"><span>额外参数 (2 层玩具主干)</span><b>+{{ (K * 50.2).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p><Tex :text="`损失公式: $L = L_{\\text{main}} + ${lambda.toFixed(1)} \\cdot \\mathrm{mean}_k(L_{\\text{mtp}_k})$。$V = 1000$ 时 $\\ln V = 6.908$, MTP 项占初始 loss 的 ${st.share}%。`" /></p>
        <p>λ = 0.3 时训练脚本实测初始 total = 9.167 (main 7.059 + 0.3 × mtp 7.028)。</p>
        <p>
          参数开销: 每级 = 1 个拼接投影 + 1 个 Block (embedding 与 lm_head 共享,
          DeepSeek-V3 61 层主干上 ~1.5%)。
        </p>
        <p>
          训练信号更密 + 表征被迫向前规划 + 推理免费拿草稿: 投机解码的草稿长度是 {{ K }}
          (llm_infer/m07_speculative_decoding; DeepSeek-V3: 接受率 85%+, 解码 ~1.8×)。部署时 MTP 模块可整体丢弃。
        </p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { range, sum } from '@/utils/labmath.js'

const N = 8            // 序列 t0..t7; 输入是前 7 个, 标签是后 7 个
const LN_V = 6.908     // ln 1000, 与 train_mtp.py 的 vocab_size 一致
const K = ref(1)
const lambda = ref(0.3)
const focus = ref(2)

// ★ 主 head 预测 t(i+1); MTP-k 拼接真实 t(i+k) 的 embedding 后预测 t(i+1+k)
const rows = computed(() => range(K.value + 1).map((k) => {
  const consume = focus.value + k, predict = focus.value + 1 + k
  const oob = predict > N - 1
  return {
    label: k === 0 ? '主 head' : `MTP-${k}`,
    isMain: k === 0,
    consume: k === 0 || oob ? -1 : consume,
    predict: oob ? -1 : predict,
    oob,
    desc: oob
      ? `t${predict} 越界 → -100 屏蔽`
      : k === 0
        ? `位置 t${focus.value} → 预测 t${predict}`
        : `拼接 Emb(t${consume}) → 预测 t${predict}`,
  }
}))

const st = computed(() => ({
  live: rows.value.filter((r) => !r.oob).length,
  // 第 k 级有 (输入长度 − k) 个位置带标签
  targets: sum(range(K.value + 1).map((k) => N - 1 - k)),
  loss0: (1 + lambda.value) * LN_V,
  share: ((lambda.value / (1 + lambda.value)) * 100).toFixed(0),
}))

const chipClass = (row, idx) => ({
  consume: idx === row.consume,
  predict: idx === row.predict && !row.isMain,
  'predict-main': idx === row.predict && row.isMain,
})
</script>

<style scoped>
.strip-area { display: flex; flex-direction: column; gap: 10px; background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px; }
.strip-title { font-size: 11px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 4px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font-size: 11px; padding: 2px 6px; min-height: 0; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text-muted); }
button.chip { cursor: pointer; }
button.chip:disabled { opacity: 0.4; cursor: default; }
.chip.focus { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.chip.consume { border-color: var(--eye); color: var(--eye); box-shadow: 0 0 0 1px var(--eye) inset; }
.chip.predict { border-color: var(--left); color: var(--left); box-shadow: 0 0 0 1px var(--left) inset; }
.chip.predict-main { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); box-shadow: 0 0 0 1px var(--accent) inset; }
.pred-row { display: flex; flex-direction: column; gap: 5px; border-top: 1px dashed var(--border); padding-top: 8px; }
.pred-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.row-label { font-size: 11px; color: var(--text); }
.row-desc { font-size: 11px; color: var(--text-muted); }
.row-desc.muted { color: var(--text-dim); }
</style>
