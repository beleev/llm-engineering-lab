<!--
  无 reference 的偏好优化实验台: DPO vs SimPO vs ORPO。
  只讲一件事: DPO 里 reference 顺带抵消了"长回答 sum log p 天然更低"这件事; 拿掉 reference 之后,
  SimPO 用长度归一化 + 目标间隔 γ 顶上, ORPO 用 SFT 项 + odds ratio 顶上。
  设定: policy 对 chosen / rejected 的每 token 平均 log p 由滑杆给, ref 对两者都是 −1.5/token。
-->
<template>
  <LabFrame
    title="拿掉 reference 之后 — SimPO 与 ORPO 靠什么抵消回答长度"
    sub="- 模拟什么: DPO 需要常驻一份参考模型 (ref), 它除了提供策略锚点外, 还隐式抵消了序列长度偏置: 回答越长, 对数概率之和 $\sum\log p$ 天然越低 (每增加 1 个 token 都相当于乘上一个小于 1 的概率)。若直接将策略模型对 chosen 与 rejected 的 $\sum\log p$ 相减, 只要 rejected 比 chosen 更长, 差值就会凭空增大, 模型无需学到真实偏好即可将损失压向 0。SimPO 与 ORPO 均省去了 ref 模型, 各自采用不同机制消除长度偏置:
  - SimPO: 按序列长度对 log 概率取平均, 再减去目标间隔 $\gamma$。
  - ORPO: 基于每 token 平均概率计算 odds 比值, 并叠加一项 SFT 损失作为锚点。
- 图上是什么: 视图由上下两部分组成:
  - 上方两行方格: 同一 prompt 下的 chosen (绿) 与 rejected (红) 回答, 每格代表 1 个 token, 高亮格数表示序列长度。
  - 下方四根横条: 四种目标函数送入 $\log\sigma(\cdot)$ 的判别量 $z$。中间竖线对应 0, 向右延伸 (绿) 表示模型对当前偏好对排序正确, 向左延伸 (红) 表示排序反转。四行依次为裸 sum 差 (无 ref 且不做长度归一化的对照组)、DPO、SimPO 与 ORPO。
  - 每行右侧读数: 显示当前的 $z$ 值及其对应的梯度权重 $\sigma(-z)$ (趋近 0 表示梯度几乎消失, 停止更新)。
  - 行悬停提示: 悬停任意一行时在下方展开对应的数学公式。
- 可以做什么: 参考模型对两条回答的每 token 对数概率固定为 −1.5, 可调节以下交互项:
  - 点击方格行的第 n 格: 将对应回答的长度设为 n。
  - 拖动「chosen / rejected 每 token log p」: 修改策略模型对两条回答每 token 的平均对数概率 (设为相等即代表模型对二者无偏好)。
  - 拖动「SimPO 目标间隔 γ」: 调节 SimPO 要求的最小领先间隔。
- 应该看到: 当两条回答的每 token 平均 log p 相同时, 仅将 rejected 点长, 「裸 sum 差」的横条会持续向右增长, 而其余三行保持不变。右侧读数含义如下:
  - 「rejected 每多写 1 个 token, z 白涨多少」: 度量各目标函数中单纯由长度差引入的虚假间隔, 理想值应为 0 (绿)。
  - 「裸 sum 差还剩的梯度权重」: 低于 0.05 时标红, 表明该对照组仅因序列长度差异就几乎丧失了学习梯度。
  - 「ORPO 总 loss」: chosen 的负对数似然 (NLL) 加上 $\lambda$ 倍的 odds ratio 损失项。"
    module="llm_finetune/methods/simpo.py · orpo.py"
    run="python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo"
    :challenge="{
      ask: '保持两条回答每 token 的平均 log p 均为 −1.5 (即模型对二者无偏好), 固定 chosen 长度为 8 token, 将 rejected 长度从 12 token 调至 24 token:\n1. 「裸 sum 差」的 $z$ 值变为多少?\n2. 对应的梯度权重还剩多少?\n3. DPO 与 SimPO 的 $z$ 值为什么保持不变?',
      answer: '答案:\n1. 裸 sum 差的 $z$ 值变为 +24。\n2. 梯度权重 $\\sigma(-24)$ 约为 0。\n3. DPO 依靠参考模型对消长度项, SimPO 依靠按长度求平均消除长度影响。\n原因解析:\n- 裸 sum 差满足 $z = (-1.5\\times 8) - (-1.5\\times 24) = +24$。尽管模型对两条回答中每个 token 的对数概率估计完全相同、尚未学到任何实质偏好, 仅因长序列的对数概率之和天然更低, 损失便已趋近于 0。当训练集中 rejected 普遍偏长时, 该目标最终只会学到偏爱短回答的捷径。\n- DPO: 参考模型对同一条长回答同样给出较低的 $\\sum\\log p$, 策略模型与参考模型相减后长度项成对抵消, 仅保留策略相对参考模型的偏移量。该行非零仅发生在策略模型已偏离参考模型时, 反映的是学到的偏好而非长度偏差。\n- SimPO: 省去参考模型, 直接计算每 token 平均 log p 之差, 从定义上约去序列长度。代价是差值量纲缩小, 温度系数 $\\beta$ 需取 2 左右; 同时减去目标间隔 $\\gamma$, 要求平均 log p 至少领先 $\\gamma/\\beta$ 才视为排序达标, 否则继续提供梯度。但由于缺乏分布锚点, 本仓库实测其留出集精确匹配率 (EM) 降至 0.023。\n- ORPO: 同样基于每 token 平均概率 $p$ 计算发生比 $\\mathrm{odds} = p/(1-p)$, 取两条回答的 odds 比值对数作为 $z$; 总损失中同时保留 chosen 的 NLL 损失 (即 SFT 损失) 充当锚点。单阶段、单模型即可完成训练, 无需预先单独执行 SFT, 本仓库实测从零训练 300 步后留出集 EM 达 0.973。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lc" label="chosen 每 token log p" :min="-3" :max="-0.1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="lr" label="rejected 每 token log p" :min="-3" :max="-0.1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="gamma" label="SimPO 目标间隔 γ" :min="0" :max="2" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="seqs">
      <div v-for="s in seqs" :key="s.id" class="seq">
        <span class="rl" :style="{ color: s.color }">{{ s.name }} · {{ s.len }} tok</span>
        <div class="cells" :style="{ gridTemplateColumns: `repeat(${LMAX}, 18px)` }">
          <button
            v-for="n in LMAX" :key="n" type="button" class="cell tk" :class="{ on: n <= s.len }"
            :style="n <= s.len ? { background: heat(clamp(-s.lp / 3, 0.15, 1), s.color) } : null"
            :aria-label="`${s.name} 长度设为 ${n}`" @click="s.set(n)"
          />
        </div>
      </div>
    </div>

    <div class="bars" @mouseleave="hov = -1">
      <div v-for="(r, i) in rows" :key="r.id" class="bar-row" :class="{ hl: hov === i }" tabindex="0" @mouseenter="hov = i" @focus="hov = i">
        <span class="name">{{ r.name }}</span>
        <div class="track">
          <i class="zero" />
          <i class="fill" :style="barStyle(r.z)" :class="r.z >= 0 ? 'pos' : 'neg'" />
        </div>
        <span class="mono z"><Tex text="$z$" /> = {{ sg(r.z) }}</span>
        <span class="mono w"><Tex text="权重 $\sigma(-z)$" /> = {{ sigmoid(-r.z).toFixed(3) }}</span>
      </div>
    </div>
    <p class="lab-note" style="margin-top: 10px;"><Tex :text="hov >= 0 ? rows[hov].formula : '悬停某一行看它的公式。'" /></p>

    <template #stats>
      <p class="cap"><Tex text="rejected 每多写 1 个 token, $z$ 白涨多少 ($\partial z/\partial L_r$):" /></p>
      <div v-for="r in rows" :key="r.id" class="kv">
        <span>{{ r.short }}</span><b :class="Math.abs(r.dz) < 1e-9 ? 'good' : 'bad'">{{ sg(r.dz) }}</b>
      </div>
      <div class="kv"><span>裸 sum 差还剩的梯度权重 (接近 0 就不再学)</span><b :class="sigmoid(-rows[0].z) < 0.05 ? 'bad' : ''">{{ sigmoid(-rows[0].z).toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="ORPO 总 loss = chosen 的 NLL + $\lambda\cdot L_{\text{OR}}$ ($\lambda = 0.1$)" /></span><b>{{ orpoLoss.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>代价对比:</p>
        <ul class="pts">
          <li><b>DPO:</b> 每步 2 次前向, policy、ref 各 1 次, 每次 2B 条序列 (chosen 和 rejected 各 B 条), 要常驻两个模型。</li>
          <li><b>SimPO / ORPO:</b> 只有 policy 1 次, 一个模型。</li>
        </ul>
        <p>本仓库实测单步训练耗时为: DPO 每步 5.4 s, SimPO 与 ORPO 每步 4.0 s。</p>
        <p>DPO 行的判别量仅在策略模型偏离参考模型时非零, 反映的是模型学到的真实偏好而非序列长度偏差。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { clamp, heat } from '@/utils/labmath.js'

const LMAX = 24, REF = -1.5, B_DPO = 0.1, B_SIMPO = 2, LAMBDA = 0.1, ZMAX = 8
// 默认 rejected 比 chosen 长 4 个 token: 一打开就看得到「裸 sum 差」被长度骗, 其余三行不动
const lc = ref(-1.5), lr = ref(-1.5), gamma = ref(0.5), Lc = ref(8), Lr = ref(12)
const hov = ref(-1)

const seqs = computed(() => [
  { id: 'c', name: 'chosen', len: Lc.value, lp: lc.value, color: 'var(--left)', set: (n) => (Lc.value = n) },
  { id: 'r', name: 'rejected', len: Lr.value, lp: lr.value, color: 'var(--danger)', set: (n) => (Lr.value = n) },
])
const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))
const logit = (avgLogp) => { const p = Math.exp(avgLogp); return Math.log(p / (1 - p)) }   // log odds, p = 每 token 平均概率

const rows = computed(() => [
  {
    id: 'naive', name: '裸 sum 差 (无 ref, 不归一)', short: '裸 sum 差',
    z: lc.value * Lc.value - lr.value * Lr.value, dz: -lr.value,
    formula: '$z = \\sum\\log\\pi(\\text{chosen}) - \\sum\\log\\pi(\\text{rejected})$。没有任何校正: 序列越长 $\\sum\\log p$ 越负, rejected 长 = 白送 margin。',
  },
  {
    id: 'dpo', name: `DPO (β = ${B_DPO})`, short: 'DPO',
    // ★ ref 对同一条序列也"越长越负", 相减后长度红利消失
    z: B_DPO * ((lc.value - REF) * Lc.value - (lr.value - REF) * Lr.value), dz: -B_DPO * (lr.value - REF),
    formula: '$z = \\beta\\cdot\\big[(\\sum\\log\\pi_c - \\sum\\log\\mathrm{ref}_c) - (\\sum\\log\\pi_r - \\sum\\log\\mathrm{ref}_r)\\big]$。ref 每 token 也是 −1.5, 长度项成对抵消。',
  },
  {
    id: 'simpo', name: `SimPO (β = ${B_SIMPO}, γ = ${gamma.value.toFixed(1)})`, short: 'SimPO',
    z: B_SIMPO * (lc.value - lr.value) - gamma.value, dz: 0,              // ★ 按长度平均, 再减目标间隔 γ
    formula: '$z = \\beta\\cdot\\big(\\sum\\log\\pi_c/|y_c| - \\sum\\log\\pi_r/|y_r|\\big) - \\gamma$。长度归一化顶替 ref; $\\gamma$ 要求每 token 平均 $\\log p$ 至少领先 $\\gamma/\\beta$。',
  },
  {
    id: 'orpo', name: 'ORPO (odds ratio 项)', short: 'ORPO',
    z: logit(lc.value) - logit(lr.value), dz: 0,
    formula: '$z = \\log\\big[\\mathrm{odds}(\\text{chosen})/\\mathrm{odds}(\\text{rejected})\\big]$, $\\mathrm{odds} = p/(1-p)$, $p = \\exp(\\text{每 token 平均 } \\log p)$。总 loss = chosen 的 NLL $+\\ \\lambda\\cdot(-\\log\\sigma(z))$, SFT 项兼任锚点。',
  },
])
const orpoLoss = computed(() => -lc.value + LAMBDA * softplus(-rows.value[3].z))
const barStyle = (z) => { const w = clamp(Math.abs(z) / ZMAX, 0, 1) * 50; return z >= 0 ? { left: '50%', width: w + '%' } : { left: 50 - w + '%', width: w + '%' } }
const sg = (x) => (x >= 0 ? '+' : '') + x.toFixed(2)
</script>

<style scoped>
.seqs { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.rl { font-size: 11px; font-family: "SF Mono", Menlo, monospace; display: block; margin-bottom: 3px; }
.tk { min-width: 0; min-height: 0; width: 18px; height: 22px; padding: 0; cursor: pointer; }
.bars { display: flex; flex-direction: column; gap: 6px; min-width: 520px; }
.bar-row { display: grid; grid-template-columns: 170px 1fr 84px 150px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); padding: 2px 4px; border-radius: 4px; }
.bar-row.hl { background: var(--bg-elev); color: var(--text); }
.track { position: relative; height: 14px; background: var(--code-bg); border-radius: 2px; }
.zero { position: absolute; left: 50%; top: -2px; bottom: -2px; border-left: 1px solid var(--border-strong); }
.fill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
.fill.pos { background: var(--left); }
.fill.neg { background: var(--danger); }
.z { color: var(--text); }
.w { font-size: 11px; }
.cap { font-size: 12px; color: var(--text-muted); }
</style>
