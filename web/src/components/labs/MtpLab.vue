<!--
  MTP 实验台 (对应 llm_models/models/language_models/mtp.py 和 training/loss.py 的 MTPLoss)。
  只讲一件事: 第 k 级 MTP 在位置 i 拼接真实 token t(i+k) 的 embedding, 预测 t(i+1+k);
  序列末尾 k 个位置没有未来 token, 标签填 -100。
-->
<template>
  <LabFrame
    title="MTP 多 token 预测 — 一次前向, 每个位置多预测几步"
    sub="- 模拟什么: 标准自回归语言模型在序列位置 $i$ 处仅预测紧邻的下一个 token $t_{i+1}$。多 token 预测 (MTP, Multi-Token Prediction, 对应 DeepSeek-V3 的实现方案) 在主干网络之后串联 K 个级联预测模块, 每加深一级便额外读入一个后续真实 token 并向后多预测一步, 因此严格满足因果性而不会泄露未来标签; 对于序列末尾超出边界、无对应未来 token 可供监督的位置, 标签统一填充为 -100 以在损失计算中屏蔽。本实验在一条长度为 8 的序列 (t0 至 t7, 前 7 个位置作为输入) 上展示各位置的多级预测关系。其中第 $k$ 级 MTP 模块依次执行三步计算:
  1. 将前一级在位置 $i$ 处的输出隐状态与真实 token $t_{i+k}$ 的 embedding 拼接并做线性投影。
  2. 送入一个独立的 Transformer Block 做特征交互。
  3. 经共享输出头预测目标 token $t_{i+1+k}$。
- 图上是什么: 自上而下排列序列与各级预测头视图:
  - 第一排: 原始输入序列, 高亮格标示当前选中的观测位置 $i$。
  - 下方各行: 依次对应主干「主 head」以及级联模块 MTP-1 至 MTP-K; 每行左侧说明文字标明该级拼接了哪个位置的真实 embedding、预测哪个位置的目标 token。
  - 标记块颜色 (对应底部图例): 「拼接的 embedding」表示该级额外注入的真实输入 token, 「MTP 预测的目标」与「主 head 预测的目标」分别标示各级与主干的监督目标位置。
  - 越界状态行: 当预测目标索引超出序列末尾 t7 时不绘制目标块, 直接标注「越界 → -100 屏蔽」。
- 可以做什么: 按界面控件顺序操作:
  - 拖动「MTP 深度 K」: 在 1 至 3 之间调节级联 MTP 模块的级数。
  - 拖动「λ」: 调节 MTP 辅助损失在训练总损失中的加权系数 (DeepSeek-V3 设为 0.3)。
  - 点击第一排任意输入 token: 切换当前聚焦的观测位置 (末尾 t7 无后续可预测 token, 不可选)。
- 应该看到: 随着聚焦位置向右移动, 级数越深的 MTP 头越早发生目标越界: 第 $k$ 级在位置 $i$ 需预测 $t_{i+1+k}$, 一旦下标超过 7 即无有效标签。右侧读数含义如下:
  - 「位置 i 有目标的预测」: 当前位置上全部 $K + 1$ 个预测头中未越界的有效监督个数。
  - 「全序列监督目标」: 汇总主干与各级 MTP 在整条序列上的有效标签总数 (第 $k$ 级共有 $7 - k$ 个有效位置); 与主干单独训练时的 7 个监督信号相比, 差值即为 MTP 额外提供的密集训练信号。
  - 「初始 loss」: 随机初始化下的理论总损失 $(1 + \lambda) \cdot \ln V$ (词表大小 $V = 1000$), 与 train_mtp 脚本实测的 9.167 相吻合。
  - 「额外参数」: 每新增一级 MTP 引入的拼接投影矩阵与单层 Block 参数量占比; 在 2 层玩具主干上每级增加 +50.2%, 而在 61 层的 DeepSeek-V3 上每级仅增加约 1.5%。"
    module="llm_models/mtp"
    run="python -m llm_models.run_models.language_models.mtp.train_mtp"
    :challenge="{
      ask: '将 MTP 深度 K 调至 2, 依次点击位置 t4、t5、t6 观察。请先思考: 从哪个输入位置开始出现预测目标越界? 整条序列合计比仅用主干单步预测多出多少个有效监督目标?',
      answer: '答案: 从位置 t5 开始出现预测越界; 整条序列比主干单步预测多出 11 个有效监督目标 (总监督目标从 7 个增至 18 个)。\n- 位置 t4: 主 head 预测 t5, MTP-1 拼接 t5 的 embedding 预测 t6, MTP-2 拼接 t6 的 embedding 预测 t7, 三个预测头均有合法标签。\n- 位置 t5: MTP-2 需要预测 t8, 而序列末尾仅到 t7, 率先发生越界。\n- 位置 t6: MTP-1 需预测 t8、MTP-2 需预测 t9, 两级 MTP 均越界, 仅剩主 head 正常预测 t7。\n- 标签对齐规则: 第 $k$ 级 MTP 的目标序列等价于将主干 labels 整体左移 $k$ 位, 末尾空出的 $k$ 个位置因无未来 token 可对齐而填入 -100, 使交叉熵损失自动跳过这些位置。\n- 监督信号统计: 主干在 7 个输入位置上各有 1 个目标, MTP-1 有 6 个有效目标 (末尾 1 个越界), MTP-2 有 5 个有效目标 (末尾 2 个越界), 全序列合计 $7 + 6 + 5 = 18$ 个有效监督目标, 单次前向提供的监督信号从 7 份增至 18 份, 对应右侧「全序列监督目标」读数。\n- 参数量开销: 每级 MTP 仅新增一个线性拼接投影层与一个 Transformer Block, 词嵌入表 embedding 与输出头 lm_head 均直接复用主干权重而不额外计入参数。因此在 2 层玩具主干上每级相对增加 +50.2% 参数, 而在 61 层的 DeepSeek-V3 主干上单级相对增量仅约 1.5%。',
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
        <p>当 $\lambda = 0.3$ 时, 训练脚本实测首步总损失为 9.167 (主干损失 7.059 加上 $0.3 \times$ MTP 损失 7.028), 与均匀分布理论值 8.98 之间的微小偏差仅源于随机初始化波动。</p>
        <p>参数开销构成: 每级 MTP 包含 1 个线性拼接投影层与 1 个 Transformer Block; 由于 embedding 与 lm_head 完全共享主干权重, 推理阶段即使直接卸载全部 MTP 模块, 主干网络的输出数值也严格保持不变。单级新增参数占比如下:</p>
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
        <p>常规自回归部署若不启用投机解码 (speculative decoding), 可在导出权重时将 MTP 模块整体裁除, 不增加任何推理显存或计算负担。</p>
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
