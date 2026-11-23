<!--
  Flash-Decoding 实验台 (对应 llm_infer/m26_flash_decoding/flash_decoding.py:decode_latency_us)。
  一件事: decode 只有 1 个 query, 并行单位只剩 B×H; 沿 KV 长度切 S 段, 把闲着的 SM 用上。
  延迟用与 Python 完全相同的代价模型在前端现算 (A100 参数), 不是实测。
-->
<template>
  <LabFrame
    title="Flash-Decoding — decode 的 Q 只有 1 行切不动, 就把 KV 切成段并行"
    sub="- 模拟什么: 在自回归 decode 阶段, 每条活跃序列每步仅有长度为 1 的单个新 query 需要计算注意力, 但必须与缓存中长达 T 个 token 的全部历史 KV 做点积。GPU 内核调度的基本并行粒度是线程块 (thread block), 标准 FlashAttention 将单个 block 绑定至「一条序列的一个注意力头」并沿 Q 的长度维度进一步切块并行; 然而 decode 阶段的 Q 仅有 1 行无法再切分, 全场可调度的并行 block 总数仅剩 $B \times H$ 个 (批大小乘以注意力头数)。以拥有 108 个流式多处理器 (SM) 的 A100 为例, 当 $B = 1$、$H = 32$ 时全场仅有 32 个活跃 block, 近 70% 的 SM 处于闲置状态。注: 本实验台中的算子延迟使用与 Python 脚本一致的解析代价模型实时估算 (基于 A100 峰值带宽与启动参数), 并非真卡实测。Flash-Decoding 采用沿键值序列切分的 split-K 策略补足并行度:
  1. 将长度为 T 的历史 KV cache 沿序列维度均分为 S 段, 每段派发独立的线程块并行计算局部输出与局部归一化统计量, 总并行粒度扩充至 $B \times H \times S$ 个 block。
  2. 紧接着启动一个轻量级归约内核 (reduce kernel), 依据各段的对数分母 lse 精确加权合并 S 段局部结果, 最终输出与标准单段注意力严格一致。
- 图上是什么: 视图分为上下两部分:
  - 上方 108 格硬件阵列: 对应 A100 GPU 上的 108 个物理 SM。方格着色越深代表该 SM 需串行排队执行的波次 (wave) 越多 (当总 block 数超出 108 时溢出的线程块顺延至后续波次), 未着色空白格代表全程闲置的 SM。
  - 下方延迟折线图: 展示单层 decode attention 估算延迟 (μs) 随切分段数 S 变化的趋势曲线, 各节点标注具体微秒读数; 绿色圆点标出取得最低延迟的最优切分段数 S, 加粗描边圆点指示当前选中的档位。
- 可以做什么: 提供三项负载与切分调节控件:
  - 点击 B 为 1 / 4 / 16 / 64 的预设按钮切换并发解码批大小。
  - 拖动「切分段数 S」滑杆或直接点击折线上的节点修改 KV 切分粒度。
  - 拖动「上下文长度 T」滑杆调节历史 KV cache 的序列长度。
- 应该看到: 在单请求小批次 ($B = 1$) 下将 S 从 1 增至 64 时, 单层估算延迟从 909 μs 大幅降至 275.5 μs, 上方 SM 阵列也由大半空白变为全部满载工作; 而在大批次 ($B = 64$) 下原始并发度本已填满全部 SM, 继续增大 S 反而因引入额外的归约内核开销而使延迟上升。右侧四项读数含义如下:
  - 并行单位 / 波次: 当前生成的 $B \cdot H \cdot S$ 个并行 block 需分几个硬件波次执行完毕 (单波次最多同时驻留 108 个)。
  - SM 利用率: 全部并行 block 在运行波次内对 108 个物理 SM 的平均占用率。
  - 估算延迟: 当前切分段数 S 对应的单层总估算耗时。
  - 相对 S=1 加速: 当前切分档位相对不切分基线的加速倍率, 括号内给出最优 S 档位所能达到的峰值加速比。"
    module="llm_infer/m26"
    run="python -m llm_infer.m26_flash_decoding.demo"
    :challenge="{
      ask: '当 $B = 1$、上下文长度 T 为 32k 时, 将切分段数 S 从 2 增至 4 能带来多少延迟缩减?',
      answer: '延迟没有任何缩减: 当 $S = 2$ 与 $S = 4$ 时单层估算耗时完全相等, 均为 459.0 μs。\n这源于 GPU 硬件波次 (wave) 的阶梯量化效应:\n- 当 $S = 2$ 时: 共产生 $1 \\times 32 \\times 2 = 64$ 个并行 block, 少于 108 个物理 SM, 单波次即可全部并发执行完毕, 每个 block 负责读取长度为 T/2 的 KV 片段。\n- 当 $S = 4$ 时: 总并行度翻倍至 128 个 block, 超出单波次 108 个 SM 的容量上限, 被迫拆分为 2 个波次串行执行; 虽然每个 block 读取的数据量减半为 T/4, 但两波次串行累加的总耗时与 $S = 2$ 单波次读取 T/2 完全抵消 (外加微小的归约启动开销后耗时持平)。\n因此切分段数应当令总 block 数 $B \\cdot H \\cdot S$ 贴近 108 的整数倍才能避免尾部半空波次浪费。当 $S = 64$ 时共 2048 个 block 分 19 波运行, SM 利用率趋近 100%, 实测估算加速比达 3.30×, 逼近将闲置 SM 全部利用起来的理论上限 $108 / (B \\cdot H) = 3.38$×。\n若将批大小切换至 $B = 64$: 不做切分时初始并行度已达 $B \\cdot H = 2048$, 硬件 SM 早已饱和, 最优档位的加速比仅为 1.00×, 继续切分只会白白增加归约内核的读写损耗。\n注: 上述数值均基于简化线性代价模型推导 (假设单个活跃 SM 最多只能分得总显存带宽的 1/108); 在真实 GPU 硬件上, 少量活跃 SM 即可抢占远超等分比例的 HBM 带宽, 因此真机实测的加速倍率通常低于该理论模型上界。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="b in BS" :key="b" type="button" :class="{ active: B === b }" :aria-pressed="B === b" @click="B = b">B = {{ b }}</button>
      </div>
      <LabSlider v-model="si" label="切分段数 S" :min="0" :max="7" :format="(i) => 2 ** i" />
      <LabSlider v-model="ti" label="上下文长度 T" :min="0" :max="3" :format="(i) => TS[i].toLocaleString()" />
    </template>

    <div class="sms" role="img" :aria-label="`108 个 SM, ${cur.waves} 波`">
      <span v-for="i in N_SM" :key="i" class="sm" :style="{ background: smFill(i - 1) }" :title="`SM ${i - 1}: ${smWaves(i - 1)} 波`" />
    </div>

    <svg viewBox="0 0 560 190" role="group" :aria-label="`延迟随 S 的曲线, 当前 S=${cur.S}: ${cur.t.toFixed(1)} μs`">
      <line x1="40" x2="550" y1="160" y2="160" class="axis" />
      <polyline :points="rows.map((r, i) => `${sx(i)},${ly(r.t)}`).join(' ')" class="line" />
      <g v-for="(r, i) in rows" :key="i" class="pt" tabindex="0" role="button" :aria-label="`S=${r.S}: ${r.t.toFixed(1)} μs`" :aria-pressed="i === si" @click="si = i" @keydown.enter="si = i">
        <circle :cx="sx(i)" :cy="ly(r.t)" r="10" class="hit" />
        <circle :cx="sx(i)" :cy="ly(r.t)" :r="i === si ? 5 : 3.5" :class="{ cur: i === si, best: i === bestI }" />
        <text :x="sx(i)" :y="ly(r.t) - 9" class="tick">{{ r.t.toFixed(0) }}</text>
        <text :x="sx(i)" y="174" class="tick">S={{ r.S }}</text>
      </g>
      <text x="4" y="12" class="cap">一层 decode attention 的估算延迟 (μs), 绿点 = 最优 S</text>
    </svg>

    <template #stats>
      <div class="kv"><span>并行单位 B·H·S (block 数) / 要跑几波</span><b>{{ cur.units }} / {{ cur.waves }}</b></div>
      <div class="kv"><span>SM 利用率 (108 个 SM 平均用满几成)</span><b :class="cur.util < 0.5 ? 'bad' : cur.util > 0.9 ? 'good' : ''">{{ (cur.util * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>当前 S 的估算延迟</span><b>{{ cur.t.toFixed(1) }} μs</b></div>
      <div class="kv"><span>相对 S=1 的加速 (括号里是最优 S={{ rows[bestI].S }} 的)</span><b :class="speed > 1.5 ? 'good' : ''">{{ speed.toFixed(2) }}× ({{ (rows[0].t / rows[bestI].t).toFixed(2) }}×)</b></div>
      <div class="lab-note">
        <p>加速上限 <Tex text="$N_{\text{SM}} / (B \cdot H)$" /> = {{ B * H < N_SM ? `${(N_SM / (B * H)).toFixed(2)}×` : '— (不切 SM 就已经满了)' }}: 最多把闲着的 SM 全用上。</p>
        <p><Tex text="★ 解析代价模型: 主注意力阶段耗时为 $t = 3 + \text{waves} \cdot (T/S) \cdot 512\,\text{B} / (2000\,\text{GB/s} / 108)$; 当切分段数 $S \gt 1$ 时额外叠加归约内核耗时 $3 + \text{units} \cdot 129 \cdot 4\,\text{B} / 2000\,\text{GB/s}$。" /></p>
        <p>其中:</p>
        <ul class="pts">
          <li>每个 kernel 3 μs 固定开销。</li>
          <li>每 token 每 head 的 KV 是 512 B (fp16, head_dim 128)。</li>
          <li>显存带宽 2000 GB/s 平分给 108 个 SM。</li>
          <li>每波每个 block 要读 T/S 个 token 的 KV。</li>
        </ul>
        <p>前端估算公式与脚本中的 decode_latency_us 函数完全一致, 可直接对照 demo 的 [2] 与 [3] 段输出: 当 $B = 1$、上下文长度 T 为 32k 时, $S = 1$ 耗时为 909.0 μs, $S = 64$ 降至 275.5 μs (加速比为 3.30×)。</p>
        <p>分段结果按 <Tex text="$\exp(\mathrm{lse}_s - \mathrm{lse})$" /> 加权合并 (每段按它占的 softmax 质量加权), 与普通 attention 数学相等 (13 组配置最坏误差 2.06e-07)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { argmax, range } from '@/utils/labmath.js'

// 与 flash_decoding.py 的常量一致: A100, 108 SM, 2000 GB/s, 每 token 每 head 512 B, 每个 kernel 3 μs
const N_SM = 108, HBM = 2000, KV_B = 512, LAUNCH = 3, H = 32
const BS = [1, 4, 16, 64], TS = [512, 4096, 32768, 131072]

const B = ref(1), si = ref(0), ti = ref(2)

// ★ decode_latency_us 的逐行翻译
const latency = (b, T, S) => {
  const units = b * H * S, waves = Math.ceil(units / N_SM)
  let t = LAUNCH + waves * (T / S) * KV_B / (HBM * 1e3 / N_SM)
  if (S > 1) t += LAUNCH + units * (128 + 1) * 4 / (HBM * 1e3)
  return { S, units, waves, t, util: units / (waves * N_SM) }
}
const rows = computed(() => range(8).map((i) => latency(B.value, TS[ti.value], 2 ** i)))
const cur = computed(() => rows.value[si.value])
const bestI = computed(() => argmax(rows.value.map((r) => -r.t)))
const speed = computed(() => rows.value[0].t / cur.value.t)

// 每个 SM 分到几波: units 按轮转分给 108 个 SM
const smWaves = (i) => Math.floor(cur.value.units / N_SM) + (i < cur.value.units % N_SM ? 1 : 0)
const smFill = (i) => {
  const w = smWaves(i)
  return w === 0 ? 'transparent' : `color-mix(in srgb, var(--accent) ${Math.round(30 + 70 * w / cur.value.waves)}%, transparent)`
}

const sx = (i) => 60 + i * 68
const maxT = computed(() => Math.max(...rows.value.map((r) => r.t)) * 1.1)   // B 大时 S 越大越慢, 最高点不在 S=1
const ly = (t) => 160 - (t / maxT.value) * 130
</script>

<style scoped>
svg { min-width: 540px; }
.sms { display: grid; grid-template-columns: repeat(27, 1fr); gap: 2px; max-width: 540px; margin-bottom: 10px; }
.sm { aspect-ratio: 1; border: 1px solid var(--border); border-radius: 2px; }
.cap { font-size: 10px; fill: var(--text-muted); }
.axis { stroke: var(--border-strong); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.line { fill: none; stroke: var(--accent); stroke-width: 2; }
.pt { cursor: pointer; }
.pt circle:not(.hit) { fill: var(--bg-elev); stroke: var(--accent); stroke-width: 1.5; }
.pt circle.best { fill: var(--left); stroke: var(--left); }
.pt circle.cur { stroke: var(--text); stroke-width: 2.5; }
.hit { fill: transparent; stroke: none; }
</style>
