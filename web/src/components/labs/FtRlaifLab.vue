<!--
  RLAIF 实验台。critique / revise 的规则与 llm_finetune/methods/rlaif.py:CONSTITUTION 相同:
  judge 只查 15 (脏话) 和 13 (电话), 条文里写了的变体 14 不查; 改写 = 打码成 3 (***)。
  最终脚本里没有「不啰嗦」原则, 这里按 readme 的描述复现 (删掉相邻重复), 右侧那组数据来自 readme 的调参记录。
  只讲一件事: 偏好对里 chosen 和 rejected 一样的地方, DPO 没有信号 —— judge 没查的, 模型就学不到。
-->
<template>
  <LabFrame
    title="judge 没查的, DPO 学不到"
    sub="上面一行是用户的 prompt, 点格子循环切换内容 (普通 token / 电话 / 变体 / 脏话)。起点模型只求有用, 回复就是照抄。
      judge 按宪法批评并改写, 改写结果当 chosen, 原回复当 rejected。红框 = 两边不一样的位置。
      两边相同的前缀在 $\log\pi(\text{chosen}) - \log\pi(\text{rejected})$ 里正好抵消, 偏好信号来自红框。"
    module="llm_finetune/methods/rlaif.py"
    run="python -m llm_finetune.run_finetune.rlaif.train_rlaif"
    :challenge="{
      ask: '在 prompt 里放一个「变体」(token 14), 看它在 chosen 和 rejected 里各是什么。再打开「不啰嗦」, 放两个相邻的「脏话」(token 15): chosen 比 rejected 短几个 token? DPO 最容易学到的区分特征是什么?',
      answer: '变体 14 在两边原样保留, 没有红框。chosen 和 rejected 里都有它, 训练对它没有直接压力。\n宪法条文写了「变体 14 也算」, 但生效的是 judge 的规则。实测 14 只从 0.418 降到 0.320, 降的那点是 DPO 的副作用。\n打开「不啰嗦」后, 两个脏话 (15 15) 先被打码成 *** *** (3 3), 再被去重成一个, chosen 总比 rejected 短。\n「更短」是每一对里都成立的区分特征, DPO 就学它。调参时:\n- 违规: 0.605 → 0.000\n- 理想回复: 0.395 → 0.000, 模型只说 1 个 token 就结束。\n所以最终版只保留不改长度的原则。',
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
    <p class="legend mono">电话 = token 13 · 变体 = token 14 · 脏话 = token 15 · *** = 打码后的 token 3 · 数字 = 普通 token</p>
    <p class="lab-note" style="margin-top: 8px;">
      <template v-if="!crit.length">judge 没发现问题: 这条回复没有 "更好的版本", 不成对, 直接丢掉。</template>
      <template v-else>judge: {{ crit.join('; ') }}</template>
    </p>

    <template #stats>
      <div class="kv"><span>有信号的位置 (两边不同)</span><b :class="nDiff ? 'good' : 'bad'">{{ nDiff }}</b></div>
      <div class="kv"><span>回复里的 14 (judge 漏检)</span><b :class="n14 ? 'bad' : ''">{{ n14 }} 个{{ n14 ? ', 两边都在' : '' }}</b></div>
      <div class="kv"><span>chosen / rejected 长度</span><b :class="lens[0] < lens[1] ? 'bad' : ''">{{ lens[0] }} / {{ lens[1] }}</b></div>
      <p class="tcap">留出集贪心, DPO 前 → 后 (实测; 只跟着「不啰嗦」开关换一组, 不随 prompt 变化)</p>
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
