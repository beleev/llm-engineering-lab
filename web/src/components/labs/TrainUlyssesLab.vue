<!--
  Ulysses 序列并行 (对应 llm_train/m16_ulysses_sequence_parallel)。
  只讲一件事: 一次 all-to-all 把 "按序列切" 换成 "按头切", 每卡看到完整序列但只算 H/P 个头; 通信随 P 下降, 但 P 不能超过头数。
-->
<template>
  <LabFrame
    title="Ulysses 序列并行 — 用 all-to-all 把「切序列」换成「切头」, 注意力核一行不改"
    sub="- 模拟什么: 当训练序列过长导致单卡显存无法容纳完整激活时, 除自注意力外的其余模块 (如 MLP 与归一化层 norm) 均沿序列维逐 token 独立运算, 将序列均分为 P 段分发至 P 张卡即可无通信并行执行; 唯独注意力层需要跨整条序列交互上下文。DeepSpeed-Ulysses 的核心思路是在进入注意力计算前插入一次 all-to-all 全互换通信 (每张卡向其余每张卡分别发送一份专属数据分块), 将张量切分轴由「按序列维切分」重排为「按注意力头维切分」。重排前每张卡持有局部序列段的全部注意力头, 张量形状为 [T/P, H, d] ($T$ 为总 token 数, $H$ 为总注意力头数, $d$ 为单头特征维); 重排后每张卡持有完整全长序列的 $H/P$ 个注意力头, 形状变为 [T, H/P, d]。由于不同注意力头之间的计算完全独立, 各卡可直接对本地全长序列调用标准高效注意力算子, 算完后再经一次逆向 all-to-all 切回按序列维切分以衔接后续模块。其结构约束是并行度 P 必须整除注意力头数 H。本实验台将三维激活张量展开为二维网格, 分步演示两次切分维度重排的全过程。
- 图上是什么: 主体网格的每一行代表序列的一个切分段 (共 P 段), 每一列代表一个注意力头 (共 H 个), 单个格子即对应激活张量的一个子块, 格内数字与底色标明该子块当前驻留的卡号。
  - 网格上方文字实时说明当前执行步骤的操作语义。
  - 处于 all-to-all 重排步骤时, 需跨卡迁移的子块带有黄色描边提示。
  - 网格下方的图例按钮对应各张计算卡编号。
  - 底部折线图对比单卡通信量随序列并行度 P 扩展的变化趋势: 虚线代表 Ring Attention (保持序列切分不变、令 KV 分块沿环逐跳传递的上下文并行方案), 实线代表 Ulysses, 空心点标示头数 H 无法被对应 P 整除从而无法启用 Ulysses 的配置; 纵轴以一份完整激活张量的体积为单位。
- 可以做什么: 控件从上到下依次为:
  - 拖动「序列并行度 P」调整参与序列并行的卡数。
  - 拖动「注意力头数 H」调整总头数; 当 H 无法被 P 整除时网格上方会给出整除约束报错。
  - 使用播放器推进四个执行阶段: ① 按序列切分初始态、② 前向 all-to-all 转置、③ 本地全序列注意力计算、④ 逆向 all-to-all 还原。
  - 悬停或点击任一网格子块, 图例右侧会显示该子块在重排步中的源卡与目标卡路径 (或驻留本卡无需搬移)。
  - 点击图例中的卡号可单独高亮该卡当前持有的全部子块, 再次点击取消筛选。
- 应该看到: 完成第 ② 步重排后, 同一列 (即同一注意力头) 的全部序列段格子统一变为同一卡的颜色, 表明整条序列在该头上的上下文已完整汇聚至单张卡内; 完成第 ④ 步后网格再次恢复为同行同色的序列切分布局。右侧读数含义如下:
  - 「每卡负责的头」: 等于 H/P。
  - 「Ulysses 每卡发送」: 单层注意力合计发送 $4(P-1)/P^2$ 份完整张量, 随并行度 P 增大反而单调递减。
  - 「Ring 每卡发送」: 单层合计发送 $2(P-1)/P$ 份完整张量, 随 P 增大单调趋近于 2。
  - 「Ring / Ulysses」: 两种方案的单卡通信量比值, 当 P 取 8 时 Ring 的通信量为 Ulysses 的 4.0 倍。"
    module="llm_train/m16"
    run="python -m llm_train.m16_ulysses_sequence_parallel.demo"
    :challenge="{
      ask: '固定注意力头数 H 为 8, 将并行度 P 先从 4 拖至 8、再拖至 6 时分别会出现什么现象? 当 P 为 8 时, Ring Attention 的单卡通信量是 Ulysses 的多少倍?',
      answer: '结论:\n- 当 P 取 8 时恰好到达并行度上限。\n- 当 P 取 6 时因头数无法整除而不可用。\n- 当 P 取 8 时 Ring Attention 的单卡通信量是 Ulysses 的 4 倍。\n机制解析:\n- P 取 8 的上限边界: 每张卡恰好分得 8/8 即 1 个注意力头, 已达 Ulysses 的最大可扩展卡数, 继续加卡将无头可分。对于采用 GQA 或 MQA (多个查询头共享单个键值头) 的现代架构, KV 头数远少于 Q 头数, 这一并行度上限会收得更紧 (除非额外复制 KV 头)。\n- P 取 6 的整除限制: 8 个注意力头无法均分为 6 等份, 网格上方弹出报错提示, 右侧读数显示「除不尽」, 底部折线图上 P 为 6 处显示为空心不可用点。Ring Attention 则完全不受头数整除约束, 因为其切分对象始终是序列长度而非注意力头。\n- 通信量差距推导: Ulysses 在单层注意力前后共执行 4 次 all-to-all (进入前 Q、K、V 各一次, 注意力输出还原一次), 每次每张卡仅需将本地持有的 $1/P$ 序列片中归属其余卡的 $(P-1)/P$ 比例发出, 四次合计发送 $4(P-1)/P^2$ 份完整张量; 并行度 P 越大则本地初始分片越小, 单卡绝对发送量反而随之下降。反观 Ring Attention 需将本地 K 与 V 两份大小为 $1/P$ 的序列块沿环接力传递 $P-1$ 轮, 合计发送 $2(P-1)/P$ 份完整张量, 随 P 增大渐近趋于 2。当 P 取 8 时两者系数分别为 1.75 与 0.44, Ring 的通信量恰好是 Ulysses 的 4 倍 (m16 实测单卡传输字节数分别为 28672 与 7168)。\n此外, 两种通信原语对底层互连拓扑的要求截然不同:\n- all-to-all: 要求任意两张卡之间均具备高对分带宽与低延迟交换能力, 契合单节点内部的 NVLink 全互连拓扑。\n- Ring: 每轮仅在相邻节点间建立点对点流式传输且易与注意力计算重叠掩盖, 更契合跨节点网卡互连。\n因此超长上下文训练系统常将两者正交组合为混合上下文并行 (USP): 节点内做 8 路 Ulysses, 节点间再叠若干路 Ring Attention, 总序列并行度为两者乘积。',
    }"
  >
    <template #controls>
      <LabSlider v-model="P" label="序列并行度 P" :min="1" :max="8" />
      <LabSlider v-model="Hn" label="注意力头数 H" :min="2" :max="16" />
      <StepPlayer :stepper="stepper" :label="phases[phase].short" />
    </template>

    <p class="phase" :class="{ bad: !valid }">{{ valid ? phases[phase].text : `✗ ${Hn} 个头没法平均分给 ${P} 张卡: Ulysses 要求 H 能被 P 整除 (Ring Attention 没有这个限制)。` }}</p>
    <!-- 整张表只占一个 Tab 停靠点: 悬停临时看, 点击固定, 方向键移动固定的格子 -->
    <div class="cells grid" :style="{ gridTemplateColumns: `58px repeat(${Hn}, minmax(20px, 1fr))` }"
         tabindex="0" role="group" aria-label="序列段 × 注意力头的归属表, 方向键移动选中的格子"
         @keydown.left.prevent="move(0, -1)" @keydown.right.prevent="move(0, 1)" @keydown.up.prevent="move(-1, 0)" @keydown.down.prevent="move(1, 0)">
      <span />
      <span v-for="h in Hn" :key="'h' + h" class="head mono">h{{ h - 1 }}</span>
      <template v-for="s in P" :key="s">
        <span class="head mono">序列段 {{ s - 1 }}</span>
        <span
          v-for="h in Hn" :key="h" class="cell"
          :class="{ dim: focus >= 0 && owner(s - 1, h - 1) !== focus, moved: valid && moving && s - 1 !== headRank(h - 1), link: sameLink(s - 1, h - 1) }"
          :style="{ background: valid || !headPhase ? heat(0.55, COLORS[owner(s - 1, h - 1)]) : 'transparent' }"
          :title="`序列段 ${s - 1} · 头 ${h - 1}: 此刻在卡 ${owner(s - 1, h - 1)}`"
          @mouseenter="hover = { s: s - 1, h: h - 1 }" @mouseleave="hover = null" @click="pin = { s: s - 1, h: h - 1 }"
        >{{ valid || !headPhase ? owner(s - 1, h - 1) : '?' }}</span>
      </template>
    </div>
    <div class="row legend">
      <button v-for="r in P" :key="r" type="button" :class="{ active: focus === r - 1 }" @click="focus = focus === r - 1 ? -1 : r - 1">
        <i :style="{ background: COLORS[r - 1] }" />卡 {{ r - 1 }}
      </button>
      <span v-if="cur && valid" class="hint mono" aria-live="polite">段{{ cur.s }}·头{{ cur.h }}: 卡{{ cur.s }} ⇄ 卡{{ headRank(cur.h) }}{{ cur.s === headRank(cur.h) ? ' (不用动)' : '' }}</span>
    </div>

    <svg :viewBox="`0 0 ${W} ${CH}`" role="img" aria-label="每卡通信量随并行度的变化" class="chart">
      <line :x1="X0" :x2="W - 8" :y1="cy(0)" :y2="cy(0)" class="axis" />
      <line :x1="X0" :x2="W - 8" :y1="cy(2)" :y2="cy(2)" class="gridl" />
      <text :x="X0 - 4" :y="cy(2) + 3" class="tick" text-anchor="end">2×</text>
      <text :x="X0 - 4" :y="cy(0) + 3" class="tick" text-anchor="end">0</text>
      <text v-for="p in 8" :key="p" :x="cx(p)" :y="CH - 4" class="tick" text-anchor="middle" :class="{ cur: p === P }">P={{ p }}</text>
      <polyline :points="range(8).map((i) => `${cx(i + 1)},${cy(ringVol(i + 1))}`).join(' ')" class="ring" />
      <polyline :points="range(8).map((i) => `${cx(i + 1)},${cy(ulyVol(i + 1))}`).join(' ')" class="uly" />
      <circle v-for="p in 8" :key="'u' + p" :cx="cx(p)" :cy="cy(ulyVol(p))" r="4" :class="['uly-dot', { no: Hn % p !== 0 }]" />
      <circle :cx="cx(P)" :cy="cy(ringVol(P))" r="5" class="ring-dot" />
      <text :x="X0 + 150" :y="12" class="lab ring-t">Ring 2(P−1)/P (虚线)</text>
      <text :x="X0 + 150" :y="25" class="lab uly-t">Ulysses 4(P−1)/P² (实线; 空心 = H 除不尽, 不可用)</text>
    </svg>

    <template #stats>
      <div class="kv"><span>每卡负责的头 (H / P)</span><b :class="valid ? '' : 'bad'">{{ valid ? Hn / P : '除不尽' }}</b></div>
      <div class="kv"><span>Ulysses 每卡发送 (单位: 一份完整张量)</span><b :class="!valid ? 'bad' : ulyVol(P) < ringVol(P) ? 'good' : ''">{{ valid ? ulyVol(P).toFixed(2) + '×' : '不可用' }}</b></div>
      <div class="kv"><span>Ring 每卡发送 (同样单位)</span><b>{{ ringVol(P).toFixed(2) }}×</b></div>
      <div class="kv"><span>Ring / Ulysses (通信量之比)</span><b>{{ valid && P > 1 ? (ringVol(P) / ulyVol(P)).toFixed(1) + ' 倍' : '—' }}</b></div>
      <div class="lab-note">
        <p>通信量折算基准单位为一份完整的 [T, H, d] 激活张量 (涵盖整条全长序列与全部注意力头)。两种方案的单卡通信量公式如下:</p>
        <ul class="pts">
          <li><b>Ulysses:</b> <Tex text="共执行 4 次 all-to-all (Q、K、V 输入各一次及注意力输出还原一次), 每次单卡仅发送本地 $1/P$ 序列分片中归属其余卡的 $(P-1)/P$ 部分, 合计为 $4(P-1)/P^2$ 份完整张量; 并行度 P 越大则本地分片越小, 单卡发送量随之递减。" /></li>
          <li><b>Ring:</b> <Tex text="需将本地 K、V 两份大小为 $1/P$ 的序列块沿环传递 $P-1$ 轮, 合计发送 $2(P-1)/P$ 份完整张量; 随着并行度 P 增大该值渐近趋近于 2, 不会随加卡而下降。" /></li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { heat, range } from '@/utils/labmath.js'

const COLORS = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--code-fn)', 'var(--text-dim)']
const P = ref(4), Hn = ref(8), focus = ref(-1)
const hover = ref(null), pin = ref(null)                 // 悬停的格子 / 点击固定的格子 { s, h }
const cur = computed(() => hover.value || pin.value)
const move = (ds, dh) => {
  const c = cur.value || { s: 0, h: 0 }
  pin.value = { s: Math.min(P.value - 1, Math.max(0, c.s + ds)), h: Math.min(Hn.value - 1, Math.max(0, c.h + dh)) }
  hover.value = null
}
const valid = computed(() => Hn.value % P.value === 0)
watch([P, Hn], () => { hover.value = null; pin.value = null; focus.value = -1 }) // 网格形状变了, 旧的悬停/固定/聚焦下标作废

const phases = [
  { short: '① 按序列切', text: '① 注意力之外: 卡 r 持有第 r 段序列的全部头 [T/P, H, d]。LayerNorm / MLP 都逐 token 算, 不需要通信。' },
  { short: '② all-to-all', text: '② 第一次 all-to-all (Q/K/V 各一次): 卡 s 把自己的第 j 组头发给卡 j。带边框的格子 = 正在换卡。' },
  { short: '③ 本地注意力', text: '③ 现在卡 j 持有完整序列的第 j 组头 [T, H/P, d]: 直接跑普通 FlashAttention, 因果 mask 天然均衡, 不需要 zigzag。' },
  { short: '④ all-to-all 换回', text: '④ 第二次 all-to-all: 把输出按序列段发回去, 恢复 [T/P, H, d], 接着算 MLP。' },
]
const stepper = useStepper(() => phases.length, { interval: 1600 })
const phase = computed(() => stepper.step.value)
const headPhase = computed(() => phase.value === 1 || phase.value === 2)
const moving = computed(() => phase.value === 1 || phase.value === 3)

// ★ seq_to_head: 头 h 属于第 ⌊h / (H/P)⌋ 组, all-to-all 之后这一列整个搬到那张卡
const headRank = (h) => Math.floor(h / (Hn.value / P.value))
const owner = (s, h) => (headPhase.value && valid.value ? headRank(h) : s)
const sameLink = (s, h) => valid.value && cur.value && cur.value.s === s && headRank(cur.value.h) === headRank(h)

// 每卡发送量, 单位 = 一份完整 [T,H,d]; 与 Python 里 assert 过的公式相同
const ulyVol = (p) => (4 * (p - 1)) / (p * p)
const ringVol = (p) => (2 * (p - 1)) / p

const W = 520, CH = 130, X0 = 30
const cx = (p) => X0 + 24 + ((p - 1) / 7) * (W - X0 - 60)
const cy = (v) => 14 + (1 - v / 2.2) * (CH - 34)
</script>

<style scoped>
.phase { font-size: 12px; color: var(--text-muted); line-height: 1.6; min-height: 3.2em; margin-bottom: 8px; }
.phase.bad { color: var(--danger); }
.head { font-size: 10px; color: var(--text-dim); align-self: center; text-align: center; }
.grid .cell { min-width: 20px; height: 26px; color: var(--text); cursor: pointer; }
.cell.moved { outline: 2px solid var(--warn); outline-offset: -2px; }
.cell.link { outline: 2px solid var(--text); outline-offset: -2px; }
.legend { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.legend button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
.hint { font-size: 11px; color: var(--text-muted); }
.chart { margin-top: 12px; }
.axis { stroke: var(--border-strong); }
.gridl { stroke: var(--border); stroke-dasharray: 2 4; }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.tick.cur { fill: var(--accent); font-weight: 700; }
.ring { fill: none; stroke: var(--eye); stroke-width: 2; stroke-dasharray: 6 4; }
.uly { fill: none; stroke: var(--left); stroke-width: 2; }
.uly-dot { fill: var(--left); }
.uly-dot.no { fill: var(--bg-card); stroke: var(--left); stroke-width: 1.5; }
.ring-dot { fill: var(--eye); }
.lab { font-size: 10px; }
.ring-t { fill: var(--eye); }
.uly-t { fill: var(--left); }
</style>
