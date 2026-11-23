<!--
  SFT 标签对齐实验台 (对应 llm_finetune/data/instruction_data.py)。
  只讲一件事: labels 已经错位一格 (labels = x[:, 1:]) 之后, prompt mask 的边界是 P-1 而不是 P。
  这是本仓库真实出现过的 off-by-one: 写成 labels[:, :P] 会把"第一个 response token"也 mask 掉。
-->
<template>
  <LabFrame
    title="SFT 标签对齐 — mask 边界差一格, 第一个回答 token 就丢了"
    sub="- 模拟什么: SFT 使用「问题 + 标准回答」训练模型, 每个序列位置的自回归目标均为根据前文 token 预测下一个 token。由于微调阶段只需模型学会生成回答而无需学习复述问题, 通常会将 prompt 区段的监督标签设为 -100 (即损失掩码 loss mask, PyTorch 的 cross_entropy 会自动忽略标签为 -100 的位置)。然而在构造数据时标签序列已相对输入整体左移一位 (取切片 x[1:], 第 t 格存放的是 x[t+1]), 因此当 prompt 长度为 P 时, 掩码切片上界应设为 P-1 而非 P。本仓库早期版本曾实际引入过该偏移错误。
- 图上是什么: 将示例句「问: 法国的首都? 答: 巴黎。」分词后按列横向展开, 从上至下共 5 行:
  1. 「位置 t」: 序列列索引。
  2. 「输入 idx」: 模型在各步接收的输入 token, 灰色代表 prompt 区段, 绿框代表回答区段。
  3. 竖向箭头: 指示由上方当前输入 token 预测下方对应位置的目标 token。
  4. 「labels」: 各位置的预测目标, 标为 -100 的方格被掩码屏蔽, 不参与损失计算。
  5. 「进 loss?」: 汇总各列的监督状态, ✓ 表示回答 token 正常参与监督; ✗ 丢了 表示回答 token 被误屏蔽; ⚠ 泄漏 表示 prompt token 混入损失; — 表示 prompt 被正确屏蔽。
- 可以做什么: 可通过三个预设按钮直接切换掩码边界, 或直接点击方格、拖动滑杆调整:
  - 错误边界: 点击「错误写法 [:P]」屏蔽前 P 个位置, 复现本仓库曾出现的越界缺陷。
  - 正确边界: 点击「正确写法 [:P-1]」屏蔽前 P-1 个位置。
  - 无掩码基线: 点击「不 mask」保留全部标签, 等价于连同问题一起拟合的预训练模式。
  - 点击 labels 行的任意方格: 手动将掩码边界移至该列。
  - 拖动「prompt 长度 P」与「response 长度」: 调整输入与回答的序列长度。
  - 悬停任意一列: 在下方查看该列应计入或应屏蔽的具体原因。
- 应该看到: 在错误写法下, 位置 P-1 (输入为「答:」) 原本对应的首个回答目标「巴黎」被误置为 -100, 「进 loss?」行随即出现「✗ 丢了」标记; 若将 response 长度拖至 1 再选择错误写法, 全序列标签均变为 -100, 读数区会提示损失退化为 nan。右侧读数含义如下:
  - 「进入 loss 的位置数」: 未被掩码屏蔽的有效监督位置总数。
  - 「回答 token 被监督」: 实际参与损失计算的回答 token 数占回答总长度的比例, 正常情况下应全部覆盖。
  - 「prompt 混进 loss」: 泄漏进损失计算的 prompt 位置数, 正常应为 0。
  - 「首个回答 token」: 指示回答的起始 token 是正常受监督还是被误屏蔽。"
    module="llm_finetune/data"
    run="python -m llm_finetune.run_finetune.sft.train_sft"
    :challenge="{
      ask: '将 response 长度拖至 1 (对应分类或单项选择等单 token 作答任务), 再点击「错误写法 [:P]」。此时还剩几个 token 参与监督训练? 计算出的 loss 会变成什么?',
      answer: '答案: 剩余 0 个 token 参与监督, loss 变为 nan。\n- 标签左移对齐关系: 标签序列已左移一位, 即 labels[t] 对应 x[t+1]。位置 P-1 的输入是最后一个 prompt token「答:」, 其预测目标恰好是首个回答 token「巴黎」, 存放在 labels 的第 P-1 格。\n- 越界屏蔽后果: 错误写法将 labels[:, :P] 设为 -100, 屏蔽了索引 0 至 P-1 的全部位置, 连同首个回答 token「巴黎」一并抹除。\n- 单 token 回答下的数值退化: 当回答长度仅为 1 个 token 时, 该位置是全序列唯一应计入损失的位置。越界屏蔽导致所有标签均为 -100, cross_entropy 在空集上求均值触发 0 除以 0, 得到 nan, 训练在第一步即发生数值异常。\n- 长回答下的隐蔽失效: 当回答包含多个 token 时, 损失曲线看似正常下降, 但模型始终未能学到如何生成回答的首个起始 token, 而首 token 往往直接决定整条生成的走向。本仓库在相同配置下实测: 正确写法在留出集上的 exact-match 准确率为 1.000, 多屏蔽一位则直接跌至 0.000。\n正确的掩码写法应是将 labels[:, :P-1] 设为 -100。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: k === P }" @click="k = P">错误写法 labels[:, :P] = -100</button>
        <button type="button" :class="{ active: k === P - 1 }" @click="k = P - 1">正确写法 labels[:, :P-1] = -100</button>
        <button type="button" :class="{ active: k === 0 }" @click="k = 0">不 mask (预训练式)</button>
      </div>
      <LabSlider v-model="P" label="prompt 长度 P" :min="3" :max="7" />
      <LabSlider v-model="R" label="response 长度" :min="1" :max="4" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `64px repeat(${T}, 52px)` }" @mouseleave="hover = -1">
      <span class="rl">位置 t</span>
      <span v-for="t in T" :key="'p' + t" class="pos mono" :class="{ hl: hover === t - 1 }">{{ t - 1 }}</span>

      <span class="rl">输入 idx</span>
      <span
        v-for="(tok, t) in idx" :key="'i' + t"
        class="cell tok" :class="[t < P ? 'prompt' : 'resp', { hl: hover === t }]"
        @mouseenter="hover = t"
      >{{ tok }}</span>

      <span class="rl">预测 ↓</span>
      <span v-for="t in T" :key="'a' + t" class="arrow" :class="{ hl: hover === t - 1, off: t - 1 < k }">↓</span>

      <span class="rl">labels</span>
      <button
        v-for="(c, t) in cols" :key="'l' + t" type="button"
        class="cell tok lbl" :class="[c.cls, { hl: hover === t }]"
        :aria-label="`位置 ${t} 的 label ${c.tok}, 点击移动 mask 边界`"
        @mouseenter="hover = t" @focus="hover = t" @click="k = c.masked ? t : t + 1"
      >{{ c.masked ? '-100' : c.tok }}</button>

      <span class="rl">进 loss?</span>
      <span v-for="(c, t) in cols" :key="'s' + t" class="mark" :class="c.cls">{{ c.mark }}</span>
    </div>
    <p class="lab-note" style="margin-top: 12px;">
      {{ hover >= 0 ? hoverText : '悬停任意一列: 看这个位置的输入、目标, 以及它为什么该 / 不该进 loss。' }}
    </p>

    <template #stats>
      <div class="kv"><span>当前 mask 写法 labels[:, :k]</span><b>[:{{ kLabel }}]</b></div>
      <div class="kv"><span>进入 loss 的位置数</span><b>{{ stat.inLoss }}</b></div>
      <div class="kv"><span>回答 token 被监督 (教到了几个)</span><b :class="stat.resp === R ? 'good' : 'bad'">{{ stat.resp }} / {{ R }}</b></div>
      <div class="kv"><span>prompt token 混进 loss (应为 0)</span><b :class="stat.leak ? 'bad' : 'good'">{{ stat.leak }}</b></div>
      <div class="kv">
        <span>首个回答 token「{{ resp[0] }}」</span>
        <b :class="stat.first ? 'good' : 'bad'">{{ stat.first ? '学到了' : '被 mask' }}</b>
      </div>
      <p class="lab-note">
        <template v-if="stat.inLoss === 0">全部标签都是 -100, 没有任何位置进 loss: cross_entropy 对空集取平均, 0 除以 0, loss 是 <b>nan</b>, 训练第一步就炸。</template>
        <template v-else>
          正确的边界是 P−1 = {{ P - 1 }}: labels 整体左移了一格 (labels[t] = x[t+1]), 第一个回答 token x[P] 落在 labels 的第 P−1 格, mask 只能盖到它前面为止。
        </template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const PROMPT = ['<s>', '问:', '法国', '的', '首都', '?', '答:']
const RESP = { 1: ['巴黎'], 2: ['巴黎', '</s>'], 3: ['巴黎', '。', '</s>'], 4: ['巴黎', '市', '。', '</s>'] }

const P = ref(6)
const R = ref(3)
const k = ref(6)          // mask 写成 labels[:, :k] = -100; 默认就是仓库里那个 bug
const hover = ref(-1)

// prompt 永远以 <s> 开头、以「答:」结尾, 中间按 P 截取
const prompt = computed(() => ['<s>', ...PROMPT.slice(PROMPT.length - (P.value - 1))])
const resp = computed(() => RESP[R.value])
const x = computed(() => [...prompt.value, ...resp.value])   // 完整序列, 长 P+R
const T = computed(() => x.value.length - 1)
const idx = computed(() => x.value.slice(0, -1))             // 输入: 去掉最后一个
const labels = computed(() => x.value.slice(1))              // ★ 标签: 整体左移一格, labels[t] = x[t+1]

// P 变了以后, 让 mask 跟着保持"同一种写法" (P / P-1 / 0)
watch(P, (p, old) => { k.value = k.value === old ? p : k.value === old - 1 ? p - 1 : Math.min(k.value, T.value) })
watch(T, (t) => { if (k.value > t) k.value = t })

const cols = computed(() => labels.value.map((tok, t) => {
  const isResp = t + 1 >= P.value            // labels[t] = x[t+1] 是不是 response token
  const masked = t < k.value
  const cls = masked ? (isResp ? 'bad' : 'dim') : (isResp ? 'ok' : 'hot')
  const mark = masked ? (isResp ? '✗ 丢了' : '—') : (isResp ? '✓' : '⚠ 泄漏')
  return { tok, isResp, masked, cls, mark }
}))

const stat = computed(() => {
  const live = cols.value.filter((c) => !c.masked)
  return {
    inLoss: live.length,
    resp: live.filter((c) => c.isResp).length,
    leak: live.filter((c) => !c.isResp).length,
    first: !cols.value[P.value - 1].masked,
  }
})
const kLabel = computed(() => (k.value === P.value ? 'P' : k.value === P.value - 1 ? 'P-1' : String(k.value)))

const hoverText = computed(() => {
  const t = hover.value, c = cols.value[t]
  if (!c) return ''
  const head = `位置 ${t}: 输入「${idx.value[t]}」→ 目标「${c.tok}」。`
  if (c.isResp && c.masked) return head + '目标是 response token 却被 mask, 这部分监督信号白白丢掉。'
  if (c.isResp) return head + '目标是 response token, 进入 loss, 是 SFT 想教的东西。'
  if (c.masked) return head + '目标还是 prompt 的一部分, mask 掉是对的: 不教模型复述用户的问题。'
  return head + '目标是 prompt token 却进了 loss: 模型在学"怎么提问", 稀释了回答的梯度。'
})
</script>

<style scoped>
.grid { width: max-content; min-width: 100%; align-items: center; row-gap: 4px; }
.rl { font-size: 11px; color: var(--text-dim); }
.pos { font-size: 10px; color: var(--text-dim); text-align: center; }
.tok { height: 30px; font-size: 12px; min-width: 0; }
.tok.prompt { color: var(--text-muted); }
.tok.resp { border-color: var(--left); color: var(--text); }
.lbl { cursor: pointer; padding: 0; min-height: 0; }
.lbl.dim { opacity: 0.45; }
.arrow { text-align: center; color: var(--accent); font-size: 12px; }
.arrow.off { color: var(--text-dim); opacity: 0.4; }
.mark { text-align: center; font-size: 10px; color: var(--text-dim); white-space: nowrap; }
.mark.ok { color: var(--left); }
.mark.bad { color: var(--danger); }
.mark.hot { color: var(--warn); }
.hl { outline: 2px solid var(--accent); outline-offset: 1px; border-radius: 3px; }
</style>
