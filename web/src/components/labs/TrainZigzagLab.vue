<!--
  Zigzag 序列切分 (对应 llm_train/m12_sequence_parallel:shard_positions + ring_attention 的 work 矩阵)。
  只讲一件事: 因果 mask 下连续切分会让最后一张卡每轮都满载、第一张卡几乎闲着; 一头一尾配对 (zigzag) 后每卡每轮工作量相同。
-->
<template>
  <LabFrame
    title="Zigzag 序列切分 — 因果注意力分给多张卡时, 谁在等谁?"
    sub="- 模拟什么: 一条序列长到一张卡放不下时, 可以切成 D 段分给 D 张卡 (上下文并行)。每张卡常驻自己那段的 Q (query), K 和 V 块像击鼓传花一样沿着一个环每轮传给下一张卡, 传完 D 轮每张卡都见过整条序列的 KV, 用 online softmax 逐块合并, 结果和整条序列一次算完全一样, 这叫 Ring Attention。因果 mask (每个 token 只能看自己和前面的 token) 下, 整块落在未来的 KV 可以直接跳过, 总计算量省将近一半。但环是同步的, 每一轮所有卡都要等最慢的那张, 所以序列怎么切决定省下的计算能不能变成省下的时间。连续切分是卡 r 拿第 r 段; zigzag 是把序列切成 2D 段, 卡 r 拿第 r 段和第 2D−1−r 段, 一头一尾配对。
      - 图上是什么: 左边是 32 个 token 的因果 mask, 行是 query、列是 key, 只有下三角有格子 (query 只看自己和前面的 key)。格子颜色表示这一对 $q \cdot k$ 由哪张卡算 (由 q 所在的卡决定), 最上面一条色带标每个位置属于哪张卡。亮的格子是环上当前这一轮正在算的, 半透明的是已经算过的, 几乎透明的是还没轮到的。右边表格每一行是环上的一轮, 每一列是一张卡, 格子里的数字是这张卡在这一轮要算的 $q \cdot k$ 对数, 带描边和 ▲ 的是这一轮最忙的卡; 最后一行「合计」是每张卡的总工作量。
      - 可以做什么: 点「连续切分」或「Zigzag (一头一尾配对)」切换切法; 点「D = 2 / 4 / 8 卡」改卡数; 用播放器一轮一轮走, 看每一轮各卡在算哪些格子。点表头的卡号只看一张卡 (左图其余卡变淡, 表格下面会列出它持有哪些 token), 再点一次取消。
      - 应该看到: 连续切分时 ▲ 每轮都落在最后一张卡上, 卡 0 从第 1 轮起全是 0 (它拿的是序列开头, 后面传来的 KV 全在它的未来, 全部跳过); zigzag 时每一轮各卡的数字相同 (D=4 时第 0 轮 36, 之后每轮 32)。右侧「每卡总工作量 max / min」两数相等时变绿; 「墙钟」是把每一轮最忙那张卡的工作量加起来, 代表实际要等的时间; 「不利用因果性的墙钟」是不跳过任何块时的 D × 块大小, 当参照 (D=4 时 256); 「zigzag 相对连续切分」是两种切法的墙钟之比, D=4 时快 1.73 倍。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: 'D = 4, 连续切分。因果 mask 省掉了将近一半的计算, 墙钟时间也省一半吗? 换成 zigzag 再看。',
      answer: '没有, 连续切分只省了 11%; zigzag 才把省下的计算变成省下的时间。\n- 先算总计算量: 32 个 token 不利用因果性要算 32 × 32 = 1024 对 q·k, 因果 mask 下只算下三角 528 对, 省了将近一半。这个数两种切法都一样。\n- 连续切分: 卡 3 拿序列最后 8 个 token, 它的 query 在所有 key 后面, 所以每一轮收到的 KV 块全在它的过去, 每轮都要算满 8 × 8 = 64 对 (第 0 轮算自己那块的下三角, 36 对)。卡 0 拿开头 8 个 token, 除了第 0 轮的 36 对, 后面传来的 KV 全在它的未来, 全部跳过, 每轮 0。环上每一轮大家都得等最忙的卡 3, 所以墙钟是 $36 + 64 \\times 3 = 228$, 相比不用因果性的 256 只省了 11%, 右表「合计」一行 [36, 100, 164, 228] 也能看出卡越靠后越忙。\n- zigzag: 序列切成 $2D$ = 8 段, 每段 4 个 token, 卡 $r$ 拿第 $r$ 段和第 $2D-1-r$ 段: 卡 0 拿第 0 段和第 7 段, 卡 3 拿第 3 段和第 4 段, 每张卡都有一段靠前 (轻) 和一段靠后 (重)。这样每张卡每一轮都恰好是 32 对 (第 0 轮 36), 合计 [132, 132, 132, 132], 墙钟 132, 比 228 快 1.73 倍。\n总计算量 528 对和通信量 (每轮传一个 KV 块) 完全没变, 变的只是负载均衡: 省下的计算只有摊平到每张卡上, 才会变成省下的时间。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !zigzag }" @click="zigzag = false">连续切分</button>
        <button type="button" :class="{ active: zigzag }" @click="zigzag = true">Zigzag (一头一尾配对)</button>
        <span class="sep" />
        <button v-for="d in [2, 4, 8]" :key="d" type="button" :class="{ active: D === d }" @click="D = d">D = {{ d }} 卡</button>
      </div>
      <StepPlayer :stepper="stepper" :label="`第 ${round} 轮`" />
    </template>

    <div class="split">
      <svg :viewBox="`0 0 ${T * CS + 2} ${T * CS + 14}`" role="img" aria-label="因果 mask 的分工" class="mask">
        <rect v-for="p in T" :key="'o' + p" :x="(p - 1) * CS + 1" y="0" :width="CS - 1" height="6" :fill="COLORS[ownerOf[p - 1]]" :opacity="focus < 0 || focus === ownerOf[p - 1] ? 1 : 0.15" />
        <rect
          v-for="c in cellsList" :key="c.id" :x="c.k * CS + 1" :y="c.q * CS + 12" :width="CS - 1" :height="CS - 1"
          :fill="COLORS[c.rq]" :opacity="(focus >= 0 && focus !== c.rq) ? 0.06 : c.step === round ? 1 : c.step < round ? 0.35 : 0.1"
        />
      </svg>

      <div class="side lab-scroll">
        <div class="cells work" :style="{ gridTemplateColumns: `34px repeat(${D}, minmax(26px, 1fr))` }">
          <span />
          <button v-for="r in D" :key="r" type="button" class="rk mono" :class="{ active: focus === r - 1 }" @click="focus = focus === r - 1 ? -1 : r - 1">
            <i :style="{ background: COLORS[r - 1] }" />卡{{ r - 1 }}
          </button>
          <template v-for="(row, s) in sim.work" :key="s">
            <span class="rl mono">轮 {{ s }}</span>
            <span
              v-for="(w, r) in row" :key="r" class="cell"
              :class="{ slow: w === sim.rowMax[s] && w > 0, dim: s > round, now: s === round }"
              :style="{ background: heat(w / sim.block * 0.7, COLORS[r]) }"
              :title="`第 ${s} 轮, 卡 ${r}: 处理来自卡 ${(r - s + D) % D} 的 KV 块, ${w} 对 q·k`"
            >{{ w }}{{ w === sim.rowMax[s] && w > 0 ? '▲' : '' }}</span>
          </template>
          <span class="rl mono">合计</span>
          <span v-for="(w, r) in sim.perRank" :key="'t' + r" class="cell tot">{{ w }}</span>
        </div>
        <p class="lab-note">表格: 行是环上的一轮, 列是一张卡, 数字是这一轮这张卡要算的 q·k 对数。颜色 = 卡号, 对照表头的色块。▲ = 这一轮最忙的卡, 其他卡都在等它。</p>
        <p class="lab-note">选中的卡持有的 token 位置: <span class="mono">{{ focus >= 0 ? posText(focus) : '点上面的卡号查看' }}</span></p>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>每卡总工作量 max / min (相等才是负载均衡)</span><b :class="sim.imb < 1.01 ? 'good' : 'bad'">{{ Math.max(...sim.perRank) }} / {{ Math.min(...sim.perRank) }}</b></div>
      <div class="kv"><span><Tex text="墙钟 = $\sum$(每轮最忙那张卡的工作量), 代表实际要等的时间" /></span><b :class="sim.wall <= Math.min(walls.cont, walls.zz) ? 'good' : 'bad'">{{ sim.wall }}</b></div>
      <div class="kv"><span>不利用因果性的墙钟 (每轮都算满一整块, D × 块大小)</span><b>{{ D * sim.block }}</b></div>
      <div class="kv"><span>zigzag 相对连续切分 (两种切法的墙钟之比)</span><b :class="{ good: speedup > 1 }">快 {{ speedup.toFixed(2) }}×</b></div>
      <div class="lab-note">
        <p>★ 同步的环上, 每一轮所有卡要等最慢的那张。因果 mask 省下的计算只有摊平到每张卡上, 才会变成省下的时间; 堆在一张卡上, 别的卡省下的只是空等。</p>
        <p>两种切法的总计算量相同, 都是 {{ sim.total }} 对 q·k; 通信量也相同, 每轮每卡传一个 KV 块。变的只是负载均衡。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { heat, range, sum } from '@/utils/labmath.js'

const COLORS = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--code-fn)', 'var(--text-dim)']
const T = 32, CS = 9
const D = ref(4), zigzag = ref(false), focus = ref(-1)

// ★ 与 Python 的 shard_positions 相同: zigzag 把序列切 2D 段, 卡 r 拿第 r 段和第 2D-1-r 段
const positions = (world, zz) => {
  if (!zz) return range(world).map((r) => range(T / world).map((i) => (r * T) / world + i))
  const c = T / (2 * world)
  return range(world).map((r) => [...range(c).map((i) => r * c + i), ...range(c).map((i) => (2 * world - 1 - r) * c + i)])
}
// work[step][rank] = 本轮卡 rank 要算的 q·k 对数 (q 位置 ≥ k 位置); 第 step 轮卡 r 手里是卡 (r-step)%D 的 KV 块
const simulate = (world, zz) => {
  const pos = positions(world, zz)
  const work = range(world).map((s) => range(world).map((r) => {
    const kv = pos[(r - s + world) % world]
    return sum(pos[r].map((qp) => kv.filter((kp) => kp <= qp).length))
  }))
  const rowMax = work.map((row) => Math.max(...row))
  const perRank = range(world).map((r) => sum(work.map((row) => row[r])))
  return { pos, work, rowMax, perRank, wall: sum(rowMax), total: sum(perRank), block: (T / world) ** 2, imb: Math.max(...perRank) / Math.min(...perRank) }
}
const sim = computed(() => simulate(D.value, zigzag.value))
// 两种切法的墙钟都算出来: 读数的颜色看数值谁小, 不看当前按的是哪个按钮
const walls = computed(() => ({ cont: simulate(D.value, false).wall, zz: simulate(D.value, true).wall }))
const speedup = computed(() => walls.value.cont / walls.value.zz)

const ownerOf = computed(() => { const o = []; sim.value.pos.forEach((ps, r) => ps.forEach((p) => (o[p] = r))); return o })
const cellsList = computed(() => {
  const o = ownerOf.value, out = []
  for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) out.push({ id: q * T + k, q, k, rq: o[q], step: (o[q] - o[k] + D.value) % D.value })
  return out
})
const stepper = useStepper(() => D.value, { interval: 1100 })
const round = computed(() => Math.min(stepper.step.value, D.value - 1))
watch(D, () => { stepper.reset(); focus.value = -1 })
const posText = (r) => { const p = sim.value.pos[r]; return zigzag.value ? `${p[0]}–${p[p.length / 2 - 1]} 和 ${p[p.length / 2]}–${p[p.length - 1]}` : `${p[0]}–${p[p.length - 1]}` }
</script>

<style scoped>
.sep { width: 12px; }
.split { display: grid; grid-template-columns: minmax(180px, 300px) minmax(0, 1fr); gap: 16px; align-items: start; }
.mask { width: 100%; }
.side { min-width: 0; overflow-x: auto; }
.work .cell { height: 28px; min-width: 26px; font-size: 11px; color: var(--text); }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; }
.rk { font-size: 9px; min-height: 24px; padding: 2px 0; white-space: nowrap; }
.rk i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 2px; }
.cell.slow { outline: 2px solid var(--warn); outline-offset: -2px; }
.cell.now { border-color: var(--text); }
.cell.tot { background: var(--bg-elev); font-weight: 600; }
@media (max-width: 720px) { .split { grid-template-columns: 1fr; } }
</style>
