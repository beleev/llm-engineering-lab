<!--
  无 reference 的偏好优化实验台: DPO vs SimPO vs ORPO。
  只讲一件事: DPO 里 reference 顺带抵消了"长回答 sum log p 天然更低"这件事; 拿掉 reference 之后,
  SimPO 用长度归一化 + 目标间隔 γ 顶上, ORPO 用 SFT 项 + odds ratio 顶上。
  设定: policy 对 chosen / rejected 的每 token 平均 log p 由滑杆给, ref 对两者都是 −1.5/token。
-->
<template>
  <LabFrame
    title="拿掉 reference 之后 — SimPO 与 ORPO 靠什么不被回答长度骗"
    sub="- 模拟什么: DPO 要常驻一份 ref 模型, 它除了当锚, 还顺带抵消了一件事: 一条回答越长, 它的 log 概率之和 $\sum\log p$ 天然越低 (每个 token 都在乘一个小于 1 的概率)。如果直接拿 policy 对 chosen 和 rejected 的 $\sum\log p$ 相减, rejected 只要比 chosen 长, 差值就白白变大, 模型什么偏好都没学也能把 loss 压到 0。SimPO 和 ORPO 都不用 ref, 这个台子看它们各靠什么顶上: SimPO 按长度取平均再减一个目标间隔 $\gamma$; ORPO 用每 token 平均概率算 odds 之比, 再加一项 SFT loss 当锚。
      - 图上是什么: 上面两行格子是同一个 prompt 下的 chosen (绿) 和 rejected (红) 回答, 每个格子一个 token, 亮着的格子数就是长度。下面四根横条是四种写法里送进 $\log\sigma(\cdot)$ 的那个量 $z$: 中间竖线是 0, 往右 (绿) 表示模型认为自己已经把这一对排对, 往左 (红) 表示排反; 每行右侧给出 $z$ 和它对应的梯度权重 $\sigma(-z)$ (接近 0 就是不再学)。四行分别是: 裸 sum 差 (没有 ref 也不归一, 对照组)、DPO、SimPO、ORPO。悬停某一行, 下方显示它的公式。
      - 可以做什么: 点格子行的第 n 个格子, 把那条回答的长度设成 n; 拖「chosen / rejected 每 token log p」改模型对两条回答每个 token 的平均 log 概率 (设成一样就是模型毫无偏好); 拖「SimPO 目标间隔 γ」改 SimPO 要求的最小领先量。ref 对两条回答都固定是每 token −1.5。
      - 应该看到: 两条回答每 token log p 相同时, 只把 rejected 点长, 「裸 sum 差」的横条一路往右, 其余三行不动。右侧读数: 「rejected 每多写 1 个 token, z 白涨多少」是每种写法里长度带来的假 margin, 应为 0 (绿); 「裸 sum 差还剩的梯度权重」低于 0.05 变红, 表示这条对照组已经被长度骗到不学了; 「ORPO 总 loss」是 chosen 的 NLL 加上 $\lambda$ 倍的 odds ratio 项。"
    module="llm_finetune/methods/simpo.py · orpo.py"
    run="python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo"
    :challenge="{
      ask: '保持两条回答每 token 的平均 log p 都是 −1.5 (模型对两者毫无偏好), chosen 留在 8 token, 把 rejected 从 12 点到 24 token。「裸 sum 差」的 z 变成多少? 它的梯度权重还剩多少? DPO 和 SimPO 为什么不上当?',
      answer: '答案: 裸 sum 差的 $z$ 变成 +24, 梯度权重 $\\sigma(-24)$ 约等于 0; DPO 靠 ref 抵消长度, SimPO 靠除以长度抵消。\n一步步看:\n- 裸 sum 差 $= (-1.5\\times 8) - (-1.5\\times 24) = +24$。模型对两条回答每个 token 的看法完全一样, 什么偏好都没学, loss 却已经是 0, 只因为长序列的 log 概率之和天然更小。数据里 rejected 普遍更长时, 这个目标学到的只是「短的好」。\n- DPO: ref 对同一条长回答也给出同样低的 $\\sum\\log p$, 两边相减后长度项成对抵消, 只剩 policy 相对 ref 的变化。这一行不为 0 只发生在 policy 已经偏离 ref 时, 那是学到的偏好, 不是长度红利。\n- SimPO: 没有 ref, 改成每 token 平均 log p 之差, 长度直接约掉。代价是数值范围变小, $\\beta$ 要取 2 左右; 再减一个目标间隔 $\\gamma$, 这个式子在说: 平均 log p 至少要领先 $\\gamma/\\beta$ 才算排对, 否则继续学。但它没有任何锚, 本仓库实测留出集 EM 掉到 0.023。\n- ORPO: 同样用每 token 平均概率 $p$ 算 odds $= p/(1-p)$, 再拿两条的 odds 之比取 log 当 $z$; 总 loss 里还有 chosen 的 NLL (就是 SFT loss) 把模型锚住。一个阶段、一个模型, 连单独的 SFT 都省了, 实测从零训 300 步留出集 EM 0.973。',
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
      <div class="kv"><span>裸 sum 差还剩的梯度权重 (被长度骗到不学了?)</span><b :class="sigmoid(-rows[0].z) < 0.05 ? 'bad' : ''">{{ sigmoid(-rows[0].z).toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="ORPO 总 loss = chosen 的 NLL + $\lambda\cdot L_{\text{OR}}$ ($\lambda = 0.1$)" /></span><b>{{ orpoLoss.toFixed(3) }}</b></div>
      <p class="lab-note">
        代价对比: DPO 每步 2 次前向, policy、ref 各 1 次, 每次 2B 条序列 (chosen 和 rejected 各 B 条), 要常驻两个模型; SimPO / ORPO 只有 policy 1 次, 一个模型。本仓库实测 DPO 每步 5.4 s, SimPO 与 ORPO 4.0 s。
        DPO 这一行不为 0 只发生在 policy 已经偏离 ref 时, 那是学到的偏好, 不是长度红利。
      </p>
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
