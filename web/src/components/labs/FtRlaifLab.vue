<!--
  RLAIF 实验台。critique / revise 的规则与 llm_finetune/methods/rlaif.py:CONSTITUTION 相同:
  judge 只查 15 (脏话) 和 13 (电话), 条文里写了的变体 14 不查; 改写 = 打码成 3 (***)。
  最终脚本里没有「不啰嗦」原则, 这里按 readme 的描述复现 (删掉相邻重复), 右侧那组数据来自 readme 的调参记录。
  只讲一件事: 偏好对里 chosen 和 rejected 一样的地方, DPO 没有信号 —— judge 没查的, 模型就学不到。
-->
<template>
  <LabFrame
    title="RLAIF — judge 没查的, DPO 学不到"
    sub="- 模拟什么: RLAIF (基于 AI 反馈的强化学习) 用 AI 裁判模型 (judge) 替代人工标注, 依据成文「宪法」审查并改写策略模型的回复。本实验的宪法包含「无害」(将脏话替换为 ***) 与「隐私」(将电话号码替换为 ***) 两条规则; 尽管条文中声明了「变体 14 也算脏话」, 但 judge 的实现代码漏检了该 token。初始策略模型仅追求原样响应, 直接复述输入 prompt。完整管线分三步执行:
  1. judge 检测到违规内容时生成批评并输出改写文本。
  2. 将改写后的合规文本作为 chosen、原始回复作为 rejected, 构造偏好样本对。
  3. 在构造出的偏好数据集上运行标准 DPO 训练。
- 图上是什么: 视图各部分含义如下:
  - 第一行: 用户输入的 prompt, 每格对应 1 个 token, 格内标注其语义类别 (数字代表普通 token; 电话、变体、脏话与 *** 为特殊 token, 图例位于方格下方)。
  - 下方两行: 经 judge 处理后构造的偏好对, 其中 chosen 为改写版本, rejected 为原始回复。
  - 红色边框: 标出 chosen 与 rejected 存在差异的序列位置, 这些位置构成了 DPO 的有效梯度来源; 两侧完全相同的 token 在对数概率差 $\log\pi(\text{chosen}) - \log\pi(\text{rejected})$ 中相互抵消。
  - judge 判定栏: 位于图例下方。若 judge 未检出任何违规, 则该回复无法构造出更优版本, 因无法配对而被直接过滤。
- 可以做什么: 支持以下交互操作:
  - 点击 prompt 行的任意方格: 在普通、电话、变体与脏话四类 token 之间循环切换。
  - 切换「第三条原则 不啰嗦」开关: 向宪法中追加「删除相邻重复 token」的改写规则。
  - 点击「脚本里的示例」与「只有变体 14」: 载入两组预设 prompt。
- 应该看到: 在 prompt 中放入「变体」时, 该 token 在 chosen 与 rejected 中均原样保留且不出现红框。右侧读数含义如下:
  - 「有信号的位置」: 序列中带红框的差异位置数量。
  - 「回复里的 14」: 被 judge 漏检的变体 token 数量, 在 chosen 与 rejected 中同时残留。
  - 「chosen / rejected 长度」: 含句末 EOS 的序列总长度; 当 chosen 短于 rejected 时标红预警, 因为长度缩短本身会成为被 DPO 优先拟合的伪特征。
  - 下方实测指标组: train_rlaif.py 在留出集上采用贪心解码测得的 DPO 训练前后对比值, 仅随「不啰嗦」开关切换, 不随上方单条 prompt 变动。"
    module="llm_finetune/methods/rlaif.py"
    run="python -m llm_finetune.run_finetune.rlaif.train_rlaif"
    :challenge="{
      ask: '按顺序执行以下操作并回答:\n1. 在 prompt 中放入一个「变体」(token 14), 观察它在 chosen 与 rejected 中分别呈现为什么内容?\n2. 开启「不啰嗦」开关, 放入两个相邻的「脏话」(token 15), 此时 chosen 比 rejected 短几个 token?\n3. 在该设定下, DPO 最容易拟合的区分特征是什么?',
      answer: '答案:\n1. 变体 14 在 chosen 与 rejected 中完全一致, 未出现红框标记。\n2. 开启「不啰嗦」后, chosen 比 rejected 短 1 个 token。\n3. DPO 最容易学到的区分特征是「序列更短」。\n原因解析:\n- 变体 14 的漏检效应: judge 的检测代码仅匹配 token 15 (脏话) 与 token 13 (电话), 对 token 14 未做替换。由于 chosen 与 rejected 在该位置具有相同 token, 二者的 $\\log\\pi$ 相减后直接抵消, 训练目标未对该位置施加直接抑制梯度。尽管宪法条文写明「变体 14 也算」, 实际决定训练信号上限的是 judge 的执行逻辑而非文本条文。留出集实测显示 token 14 的出现率仅从 0.418 降至 0.320, 小幅下降仅来自全局分布偏移的附带影响。\n- 「不啰嗦」触发长度差: 两个相邻的脏话 (15 15) 先被「无害」规则替换为 *** *** (3 3), 随后被「不啰嗦」规则合并去重为单个 ***, 导致 chosen 比 rejected 缩短 1 个 token。\n- 长度捷径导致模式坍缩: 「chosen 更短」在每组触发去重的偏好对中稳定成立, 比任何局部 token 替换规律都更易拟合, DPO 会迅速沿该捷径收敛。实验记录表明开启该规则后, 留出集违规率虽从 0.605 降至 0.000, 但理想回复率也从 0.395 跌至 0.000, 模型退化为仅输出 1 个 token 即提前终止。因此最终方案仅保留等长替换类原则。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: terse }" :aria-pressed="terse" @click="terse = !terse">第三条原则「不啰嗦」: {{ terse ? '开' : '关' }}</button>
        <button type="button" @click="x = [4, 8, 12, 15, 13, 15]">脚本里的示例</button>
        <button type="button" @click="x = [5, 14, 9, 14, 7, 6]">只有变体 14</button>
      </div>
    </template>

    <div class="seqs">
      <div class="srow">
        <span class="sl">prompt</span>
        <button v-for="(t, i) in x" :key="i" type="button" class="cell tk" :class="cls(t)" :aria-label="`第 ${i + 1} 个 token ${t} ${NAME[t] || '普通'}, 点击切换`" @click="cycle(i)">{{ NAME[t] || t }}</button>
      </div>
      <div v-for="row in pair" :key="row.id" class="srow">
        <span class="sl">{{ row.name }}</span>
        <span v-for="(t, i) in row.v" :key="i" class="cell tk" :class="[cls(t), { diff: row.diff[i] }]">{{ NAME[t] || t }}</span>
      </div>
    </div>
    <p class="legend mono">图例: 电话 = token 13 · 变体 = token 14 · 脏话 = token 15 · *** = 打码后的 token 3 · 数字 = 普通 token</p>
    <p class="lab-note" style="margin-top: 8px;">
      <template v-if="!crit.length">judge 没发现问题: 这条回复没有「更好的版本」, 凑不成偏好对, 直接丢掉。</template>
      <template v-else>judge 的判定: {{ crit.join('; ') }}</template>
    </p>

    <template #stats>
      <div class="kv"><span>有信号的位置 (红框数, 两边不同)</span><b :class="nDiff ? 'good' : 'bad'">{{ nDiff }}</b></div>
      <div class="kv"><span>回复里的变体 14 (judge 漏检)</span><b :class="n14 ? 'bad' : ''">{{ n14 }} 个{{ n14 ? ', 两边都在' : '' }}</b></div>
      <div class="kv"><span>chosen / rejected 长度 (含 EOS)</span><b :class="lens[0] < lens[1] ? 'bad' : ''">{{ lens[0] }} / {{ lens[1] }}</b></div>
      <p class="tcap">train_rlaif.py 留出集贪心解码, 每项是 DPO 前 → DPO 后 (实测; 只跟着「不啰嗦」开关换一组, 不随 prompt 变化)</p>
      <div v-for="k in M.rows" :key="k[0]" class="kv"><span>{{ k[0] }}</span><b :class="k[3]">{{ k[1] }} → {{ k[2] }}</b></div>
      <p class="lab-note">{{ M.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { range } from '@/utils/labmath.js'

const SWEAR = 15, VARIANT = 14, PHONE = 13, MASK = 3
const CYCLE = [7, PHONE, VARIANT, SWEAR]
const NAME = { [PHONE]: '电话', [VARIANT]: '变体', [SWEAR]: '脏话', [MASK]: '***' }   // 格子里直接写含义
// 关: train_rlaif.py 输出 (留出集贪心, DPO 前 → 后); 开: readme 调参记录 (lr=3e-4, 200 步, 1024 条采样)
const MEASURED = {
  off: { rows: [['违反任一原则', '0.648', '0.031', 'good'], ['变体 14', '0.418', '0.320', 'bad'], ['照抄率', '1.000', '0.856', 'bad'], ['= 理想回复', '0.352', '0.488', 'good']],
    note: '4096 条采样 → 2539 个偏好对 → DPO β=0.5, lr=1e-4, 150 步。judge 查的违规几乎清零, 漏检的 14 基本没动。有用性付了代价, 错误集中在第一个回复 token。' },
  on: { rows: [['违反原则', '0.605', '0.000', 'good'], ['= 理想回复', '0.395', '0.000', 'bad']],
    note: '加了「不啰嗦」: chosen 总比 rejected 短, DPO 学成 "早点说 EOS", 只说 1 个 token 就结束。这组数来自调参记录, 设置与「关」时不同 (lr=3e-4, 200 步, 1024 条采样)。' },
}
const terse = ref(false), x = ref([4, 8, 12, 15, 13, 15])
const M = computed(() => MEASURED[terse.value ? 'on' : 'off'])
const cycle = (i) => { x.value = x.value.map((t, k) => (k === i ? CYCLE[(CYCLE.indexOf(t) + 1) % CYCLE.length] : t)) }

// 宪法: [名字, 检查 (返回违规位置), 改写]。★ 只查 15 和 13, 14 不查
const RULES = computed(() => [
  ['无害', (y) => y.flatMap((t, i) => (t === SWEAR ? [i] : [])), (y) => y.map((t) => (t === SWEAR ? MASK : t))],
  ['隐私', (y) => y.flatMap((t, i) => (t === PHONE ? [i] : [])), (y) => y.map((t) => (t === PHONE ? MASK : t))],
  ...(terse.value ? [['不啰嗦', (y) => y.flatMap((t, i) => (i && t === y[i - 1] ? [i] : [])), (y) => y.filter((t, i) => !i || t !== y[i - 1])]] : []),
])
const rejected = computed(() => [...x.value])                        // 只求有用: 照抄
const crit = computed(() => RULES.value.flatMap(([name, bad]) => { const p = bad(rejected.value); return p.length ? [`违反「${name}」: 位置 [${p.join(', ')}]`] : [] }))
const chosen = computed(() => RULES.value.reduce((y, [, , fix]) => fix(y), rejected.value))   // 依次套用每条原则的 fix
const pair = computed(() => {
  const c = chosen.value, r = rejected.value
  const same = crit.value.length === 0
  return [
    { id: 'c', name: same ? '(不成对)' : 'chosen', v: same ? [] : c, diff: c.map((t, i) => t !== r[i]) },
    { id: 'r', name: same ? '回复' : 'rejected', v: r, diff: r.map((t, i) => !same && t !== c[i]) },
  ]
})
const nDiff = computed(() => (crit.value.length ? range(Math.max(chosen.value.length, rejected.value.length)).filter((i) => chosen.value[i] !== rejected.value[i]).length : 0))
const n14 = computed(() => rejected.value.filter((t) => t === VARIANT).length)
const lens = computed(() => [crit.value.length ? chosen.value.length + 1 : 0, rejected.value.length + 1])   // +1 = EOS
const cls = (t) => (t === SWEAR ? 'bad' : t === PHONE ? 'hot' : t === VARIANT ? 'v14' : t === MASK ? 'ok' : '')
</script>

<style scoped>
.seqs { display: flex; flex-direction: column; gap: 6px; }
.srow { display: flex; gap: 4px; align-items: center; }
.sl { width: 64px; font-size: 12px; color: var(--text-muted); flex-shrink: 0; }
.tk { min-width: 0; min-height: 0; width: 34px; height: 30px; padding: 0; font-size: 11px; }
button.tk { cursor: pointer; }
.tk.v14 { border-color: var(--warn); border-style: dashed; color: var(--text); }
.tk.diff { outline: 2px solid var(--danger); outline-offset: 1px; }
.legend { margin-top: 6px; font-size: 10.5px; color: var(--text-dim); }
.tcap { font-size: 11px; color: var(--text); margin-top: 4px; }
</style>
