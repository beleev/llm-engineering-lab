<!-- 旋转 + 4 bit 实验台 (llm_infer/m38): hadamard + int4_sym 的 JS 移植。一件事: 旋转把离群值摊平, 4 bit 的格点才够普通通道用。 -->
<template>
  <LabFrame
    title="Hadamard 旋转 — 先把离群通道摊平, 再量化到 4 bit"
    sub="- 模拟什么: 一层线性层输入的激活, 64 个通道里有几个离群通道比其余通道大很多倍。4 bit 对称量化每组只有 15 个格点 (−7..7), 格距 = 组内最大绝对值 ÷ 7, 离群通道把格距撑大后, 同组普通通道几乎全被舍成 0。旋转是乘一个正交的 Hadamard 矩阵 $H$: $XW = (XH)(H^\top W)$, 结果不变, 但每个通道都变成所有通道的 $\pm 1/8$ 加权和, 离群能量被均分。
- 图上是什么: ① 是第 0 个 token 送进量化器的 64 个值 (柱) 和量化后的值 (短横), 虚线是分组边界, 橙色是离群通道; ② 是换回原坐标系后每个通道的误差 $|x - \hat x|$。
- 可以做什么: 切换「不旋转 / Hadamard 旋转」; 选分组大小 (64 = 整个 token 一组, 即 per-token); 拖「离群倍数」; 点 ① 或 ② 里任意一根柱, 把这个通道设为或取消离群; 「换一组」重新采样。
- 应该看到: 不旋转、64 一组时普通通道的误差接近 1 (全被舍成 0); 打开旋转后 max/RMS 从 10 以上降到 3 左右, 误差降到 0.3 以下。分组越小, 离群值只伤到自己那一组, 旋转的改善倍数越小。右侧读数在 64 个 token 上平均。"
    module="llm_infer/m38"
    run="python -m llm_infer.m38_rotation_fp4_quant.demo"
    :challenge="{
      ask: '离群倍数保持 20, 把分组从 64 换成 8。打开旋转后, 普通通道误差的改善倍数会怎样变化?',
      answer: '改善倍数从约 2.9 倍缩到约 1.6 倍。\n分组越小, 离群通道只撑大自己那一组的格距, 其余组不受影响, 不旋转时误差本来就小一些。旋转则把离群能量摊给所有组, 每组的格距都抬高一点。两头一挤, 旋转的收益就变小了。\nPython demo 里同样的规律更明显: INT4 每 128 个一组时旋转改善 3.09 倍, 32 个一组 2.01 倍; NVFP4 每 16 个一组, 加上 FP4 格点对大值更宽容, 旋转反而让普通通道误差从 0.2363 升到 0.2685。所以 QuaRot 针对的是 per-token 这种粗粒度的 INT4, 用小 block 的 FP4 格式时旋转不是必需的。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !rot }" :aria-pressed="!rot" @click="rot = false">不旋转</button>
        <button type="button" :class="{ active: rot }" :aria-pressed="rot" @click="rot = true">Hadamard 旋转</button>
        <span class="sep">分组</span>
        <button v-for="g in GROUPS" :key="g" type="button" :class="{ active: group === g }" @click="group = g">{{ g === N ? '64 (per-token)' : g }}</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="mult" label="离群倍数" :min="1" :max="40" unit="×" />
    </template>

    <svg viewBox="0 0 560 300" role="group" aria-label="量化前后的通道值与每个通道的误差">
      <text x="4" y="12" class="cap">① token #0 送进量化器的值 ({{ rot ? '旋转后' : '原始' }}); 短横 = 量化后</text>
      <line x1="20" x2="556" :y1="Y0" :y2="Y0" class="axis" />
      <line v-for="b in N / group - 1" :key="`g${b}`" :x1="bx(b * group) - 1" :x2="bx(b * group) - 1" y1="18" y2="138" class="grp" />
      <g v-for="(v, i) in tok.fed" :key="`b${i}`" class="ch" tabindex="0" role="button" :aria-label="`通道 ${i}${outs.has(i) ? ' (离群)' : ''}`"
         @click="toggle(i)" @keydown.enter="toggle(i)">
        <rect :x="bx(i)" :y="Math.min(Y0, vy(v))" width="6" :height="Math.max(1, Math.abs(vy(v) - Y0))" :class="outs.has(i) && !rot ? 'out' : 'val'" />
        <line :x1="bx(i) - 1" :x2="bx(i) + 7" :y1="vy(tok.q[i])" :y2="vy(tok.q[i])" class="qt" />
      </g>
      <text x="4" y="162" class="cap">② 原坐标系里每个通道的误差 |x − x̂| (橙色 = 离群通道)</text>
      <g v-for="(e, i) in tok.err" :key="`e${i}`" class="ch" @click="toggle(i)">
        <rect :x="bx(i)" :y="290 - ey(e)" width="6" :height="Math.max(1, ey(e))" :class="outs.has(i) ? 'out' : 'err'" />
      </g>
      <line x1="20" x2="556" y1="290" y2="290" class="axis" />
    </svg>

    <template #stats>
      <div class="kv"><span>每 token max/RMS (高斯约 3)</span><b :class="res.maxRms < 4 ? 'good' : 'bad'">{{ res.maxRms.toFixed(2) }}</b></div>
      <div class="kv"><span>平均格距 (普通通道标准差 = 1)</span><b :class="res.step < 1.5 ? 'good' : res.step > 3 ? 'bad' : ''">{{ res.step.toFixed(2) }}</b></div>
      <div class="kv"><span>普通通道相对误差 (1 = 全被舍成 0)</span><b :class="res.err < 0.3 ? 'good' : res.err > 0.6 ? 'bad' : ''">{{ res.err.toFixed(3) }}</b></div>
      <div class="kv"><span>同分组下旋转的改善 (不旋转 ÷ 旋转)</span><b :class="gain > 1.5 ? 'good' : ''">{{ gain.toFixed(2) }}×</b></div>
      <div class="lab-note">
        <p>Python demo [D] 的激活误差 (只算普通通道; 256 通道, 4 个离群 × 20; 实测, 不随左侧变化):</p>
        <ul class="pts">
          <li v-for="r in PY" :key="r[0]"><b>{{ r[0] }}:</b> {{ r[1] }} → 旋转后 {{ r[2] }} ({{ r[3] }})</li>
        </ul>
        <p>旋转只改变数值分布, 不改变矩阵乘结果: demo 里 <Tex text="$(XH)(H^\top W)$" /> 与 <Tex text="$XW$" /> 的相对差是 5.9e-07。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { mulberry32, randn, range } from '@/utils/labmath.js'

const N = 64, TOK = 64, GROUPS = [64, 16, 8], Y0 = 78
const PY = [['INT4 g128 (4.125 bit)', '0.7792', '0.2522', '3.09×'], ['INT4 g32 (4.5 bit)', '0.4528', '0.2249', '2.01×'],
  ['MXFP4 (4.25 bit)', '0.3453', '0.2998', '1.15×'], ['NVFP4 (4.5 bit)', '0.2363', '0.2685', '0.88×']]

const rot = ref(false), group = ref(64), mult = ref(20), seed = ref(1)
const outs = ref(new Set([9, 41]))
const toggle = (i) => { const s = new Set(outs.value); s.has(i) ? s.delete(i) : s.add(i); outs.value = s }

// Sylvester 递推 H_2n = [[H, H], [H, −H]], 除以 √N 后正交 (与 rotation_fp4.hadamard 相同)
const H = (() => { let M = [[1]]; while (M.length < N) M = [...M.map((r) => [...r, ...r]), ...M.map((r) => [...r, ...r.map((x) => -x)])]; return M.map((r) => r.map((x) => x / Math.sqrt(N))) })()
const rotate = (v, inv) => range(N).map((j) => v.reduce((a, x, i) => a + x * (inv ? H[j][i] : H[i][j]), 0))   // v·H 或 v·Hᵀ
// 对称 INT4: 每组 scale = 组内 max|x| / 7 (int4_sym)
const quant = (v, g) => v.map((x, i) => {
  const blk = v.slice(i - (i % g), i - (i % g) + g), sc = Math.max(...blk.map(Math.abs), 1e-12) / 7
  return Math.max(-7, Math.min(7, Math.round(x / sc))) * sc
})

const base = computed(() => { const r = mulberry32(seed.value * 7919); return range(TOK).map(() => range(N).map(() => randn(r))) })
const tokens = computed(() => base.value.map((row) => row.map((x, i) => (outs.value.has(i) ? x * mult.value : x))))

function run(x, useRot) {
  const fed = useRot ? rotate(x) : x
  const q = quant(fed, group.value)
  const back = useRot ? rotate(q, true) : q                // ★ 换回原坐标系再比误差
  return { fed, q, err: x.map((v, i) => Math.abs(v - back[i])), back }
}
function stats(useRot) {
  let en = 0, ee = 0, mr = 0, st = 0
  for (const x of tokens.value) {
    const { fed, back } = run(x, useRot)
    const rms = Math.sqrt(fed.reduce((a, v) => a + v * v, 0) / N)
    mr += Math.max(...fed.map(Math.abs)) / rms / TOK
    for (let s = 0; s < N; s += group.value) st += Math.max(...fed.slice(s, s + group.value).map(Math.abs)) / 7 / TOK / (N / group.value)
    x.forEach((v, i) => { if (!outs.value.has(i)) { en += v * v; ee += (v - back[i]) ** 2 } })
  }
  return { maxRms: mr, step: st, err: Math.sqrt(ee / Math.max(en, 1e-12)) }
}
const tok = computed(() => run(tokens.value[0], rot.value))
const res = computed(() => stats(rot.value))
const gain = computed(() => stats(false).err / Math.max(res.value.err, 1e-12))

const scale = computed(() => Math.max(...tok.value.fed.map(Math.abs), ...tok.value.q.map(Math.abs), 1e-6))
const bx = (i) => 22 + i * 8.35
const vy = (v) => Y0 - (v / scale.value) * 58
const ey = (e) => Math.min(110, (e / Math.max(...tok.value.err, 1e-6)) * 110)
</script>

<style scoped>
svg { min-width: 540px; }
.cap { font-size: 10px; fill: var(--text-muted); }
.axis { stroke: var(--text-dim); }
.grp { stroke: var(--text-dim); stroke-dasharray: 3 3; }
.val { fill: var(--eye); opacity: 0.8; }
.out { fill: var(--warn); }
.err { fill: var(--danger); opacity: 0.75; }
.qt { stroke: var(--text); stroke-width: 1.5; }
.ch { cursor: pointer; }
.sep { color: var(--text-muted); margin-left: 8px; }
</style>
