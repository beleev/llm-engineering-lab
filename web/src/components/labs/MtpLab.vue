<!--
  MTP 实验台 (对应 llm_models/models/language_models/mtp.py 和 training/loss.py 的 MTPLoss)。
  只讲一件事: 第 k 级 MTP 在位置 i 拼接真实 token t(i+k) 的 embedding, 预测 t(i+1+k);
  序列末尾 k 个位置没有未来 token, 标签填 -100。
-->
<template>
  <LabFrame
    title="MTP 多 token 预测 — 一次前向, 每个位置多预测几步"
    sub="- 模拟什么: 普通语言模型在位置 $i$ 只预测下一个 token $t_{i+1}$。MTP (Multi-Token Prediction, DeepSeek-V3 的做法) 在主干后面挂 K 个级联模块, 每深一级多看一个真 token、多预测一步, 所以不算偷看未来。序列末尾没有未来 token 可预测的位置, 标签填 −100 屏蔽掉。这个台子用一条 8 个 token 的序列 (t0 到 t7, 前 7 个是输入) 看每个位置预测了谁。第 $k$ 级模块做三件事:
        1. 把主干在位置 $i$ 的隐状态和真实 token $t_{i+k}$ 的 embedding 拼在一起。
        2. 过一个 Block。
        3. 去预测 $t_{i+1+k}$。
      - 图上是什么: 从上到下:
        - 第一排: 输入序列, 高亮的是当前聚焦的位置 $i$。
        - 下面每一行: 一个预测头, 「主 head」和 MTP-1 到 MTP-K。每行左边的文字说明它拼接了谁的 embedding、预测谁。
        - 芯片颜色 (图例在最下面): 「拼接的 embedding」是这一级额外喂进去的真实 token, 「MTP 预测的目标」和「主 head 预测的目标」是要预测的 token。
        - 越界的行: 目标越界 (超过 t7) 的行不画芯片, 只写「越界 → -100 屏蔽」。
      - 可以做什么: 按界面顺序:
        - 拖「MTP 深度 K」: 改有几级 MTP (1 到 3)。
        - 拖「λ」: 改 MTP 损失在总损失里的权重 (DeepSeek-V3 用 0.3)。
        - 点第一排任意 token: 换聚焦位置 (t7 是最后一个, 没有下一个 token 可预测, 点不了)。
      - 应该看到: 聚焦位置往右移, 越深的级越早越界: 第 $k$ 级在位置 $i$ 要预测 $t_{i+1+k}$, 它超过 t7 就没有目标。右侧读数:
        - 「位置 i 有目标的预测」: K+1 个头里有几个没越界。
        - 「全序列监督目标」: 把每一级有标签的位置数加起来 (第 $k$ 级有 7 − k 个), 和主干单独的 7 个对比, 多出来的就是 MTP 额外提供的训练信号。
        - 「初始 loss」: 随机初始化时的总损失 (1+λ)·ln V, 词表 V = 1000, 和 train_mtp 实测的 9.167 对得上。
        - 「额外参数」: 每级多出的拼接投影和 Block, 2 层的玩具主干上每级 +50.2%, 61 层的 DeepSeek-V3 上约 1.5%。"
    module="llm_models/mtp"
    run="python -m llm_models.run_models.language_models.mtp.train_mtp"
    :challenge="{
      ask: '把 K 拖到 2, 依次点 t4、t5、t6。先猜: 从哪个位置开始有预测越界? 整条序列一共比主干单独多出几个监督目标?',
      answer: '答案: t5 开始越界; 整条序列多出 11 个监督目标 (7 个变 18 个)。\n- t4: 主 head 预测 t5, MTP-1 拼上 t5 的 embedding 预测 t6, MTP-2 拼上 t6 预测 t7, 三级都有目标。\n- t5: MTP-2 要预测 t8, 序列只到 t7, 越界。\n- t6: MTP-1 要预测 t8、MTP-2 要预测 t9, 都越界, 只剩主 head 预测 t7。\n- 规则: 第 $k$ 级的标签是 labels 整体左移 $k$ 位, 末尾 $k$ 个位置没有未来 token, 填 -100 让 loss 跳过它们。\n- 数量: 主干 7 个位置各有 1 个目标, MTP-1 有 6 个 (末尾 1 个越界), MTP-2 有 5 个 (末尾 2 个越界), 合计 $7 + 6 + 5 = 18$。一次前向的监督从 7 份变成 18 份, 右侧「全序列监督目标」就是这个数。\n- 代价: 每级多一个拼接投影和一个 Block (embedding 和 lm_head 借主干的, 不另算)。2 层的玩具主干上每级 +50.2% 参数, 61 层的 DeepSeek-V3 上同样一级只多约 1.5%。',
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
      <div class="kv"><span>位置 t{{ focus }} 有目标的预测头 (没越界的 / 全部)</span><b :class="st.live === K + 1 ? 'good' : 'bad'">{{ st.live }} / {{ K + 1 }}</b></div>
      <div class="kv"><span>全序列监督目标数 (主干单独只有 {{ N - 1 }} 个)</span><b>{{ st.targets }}</b></div>
      <div class="kv"><span><Tex text="随机初始化时的总 loss $(1+\lambda)\ln V$" /></span><b>{{ st.loss0.toFixed(2) }}</b></div>
      <div class="kv"><span>额外参数 (2 层玩具主干, 每级 +50.2%)</span><b>+{{ (K * 50.2).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p><Tex :text="`损失公式: $L = L_{\\text{main}} + ${lambda.toFixed(1)} \\cdot \\mathrm{mean}_k(L_{\\text{mtp}_k})$, 主干的 loss 加上各级 MTP loss 的平均乘 λ。随机初始化时每个头的 loss 都约等于 $\\ln V$ (均匀瞎猜), $V = 1000$ 时 $\\ln V = 6.908$, 所以 MTP 项占初始 loss 的 ${st.share}%。`" /></p>
        <p>λ = 0.3 时训练脚本实测初始 total = 9.167 (main 7.059 + 0.3 × mtp 7.028), 和上面的理论值 8.98 只差初始化的随机波动。</p>
        <p>
          参数开销: 每级 = 1 个拼接投影 + 1 个 Block。embedding 和 lm_head 都借主干的, 所以丢掉 MTP 模块后主干的输出一个数都不变。每级多出的参数占比:
        </p>
        <ul class="pts">
          <li><b>玩具主干:</b> 只有 2 层, 所以每级显得贵 (+50.2%)。</li>
          <li><b>DeepSeek-V3:</b> 主干有 61 层, 同样一级只多约 1.5%。</li>
        </ul>
        <p>MTP 带来三样东西:</p>
        <ul class="pts">
          <li><b>训练信号更密:</b> 监督目标变多。</li>
          <li><b>主干的表征被迫「向前规划」:</b> 要能支持预测 t+2、t+3。</li>
          <li><b>推理时免费拿到草稿:</b> MTP 头对后面 {{ K }} 个 token 的预测可以直接当投机解码的草稿, 再由主干验证 (见 llm_infer/m07_speculative_decoding; DeepSeek-V3 报告接受率 85% 以上, 解码约 1.8× 快)。</li>
        </ul>
        <p>不用投机解码时 MTP 模块可以整体丢弃。</p>
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
