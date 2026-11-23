<!--
  Zigzag 序列切分 (对应 llm_train/m12_sequence_parallel:shard_positions + ring_attention 的 work 矩阵)。
  只讲一件事: 因果 mask 下连续切分会让最后一张卡每轮都满载、第一张卡几乎闲着; 一头一尾配对 (zigzag) 后每卡每轮工作量相同。
-->
<template>
  <LabFrame
    title="Zigzag 序列切分 — 因果注意力分给多张卡时, 谁在等谁?"
    sub="- 模拟什么: 当单条序列长度超出单卡显存容量时, 上下文并行将序列切分为 D 段分发至 D 张卡协同计算。Ring Attention 让每张卡常驻持有本地序列段的查询向量 Q (query), 同时令键值块 K 与 V 沿单向环在每轮通信中依次传递给下一张卡; 历经 D 轮环路流转后每张卡均遍历了完整序列的 KV 上下文, 并借助 online softmax 逐块增量合并结果, 最终输出与单卡全序列注意力在数学上严格一致。在自回归因果 mask (每个 token 仅关注自身及前序位置) 约束下, 整块完全落在未来位置的 KV 分块可直接跳过不作矩阵乘, 使全局总有效计算量节省近一半。然而由于环路每轮迭代存在同步屏障, 全体卡必须等待当轮计算最慢的卡完成后才能进入下一轮传输, 因此序列切分方式直接决定了跳过未来块省下的浮点算力能否真正转化为端到端墙钟时间缩短。对比两种切分策略:
  - 连续切分: 卡 r 直接持有按原始顺序排列的第 r 连续子段。
  - zigzag 对称切分: 将整条序列细分为 2D 段, 令卡 r 同时持有头部的第 r 段与尾部的第 2D−1−r 段, 通过首尾配对平衡前后计算负载。
- 图上是什么: 视图分为左右两部分:
  - 左侧绘制长度为 32 个 token 的因果注意力下三角矩阵, 行对应 query 位置、列对应 key 位置, 仅下三角区域存在有效计算格。
  - 单元格颜色标明该对 $q \cdot k$ 内积由哪张卡负责计算 (取决于 query 所在序列段归属的卡号), 矩阵顶部的水平色带同步标示各序列位置所属的卡号。
  - 高亮鲜艳的格子代表当前通信轮正在执行计算的区块, 半透明格代表前序轮次已完成计算的区块, 极淡格代表后续轮次尚未到达的区块。
  - 右侧表格每一行对应环路流转的一轮, 每一列对应一张计算卡, 格内数字记录该卡在当轮需实际计算的 $q \cdot k$ 有效位置对数, 带外框描边及 ▲ 标记的单元格代表当轮负载最重的瓶颈卡。
  - 表格末行「合计」汇总每张卡在全部 D 轮中的累计有效工作量。
- 可以做什么: 控件从上到下依次为:
  - 点击「连续切分」或「Zigzag (一头一尾配对)」切换序列分片映射方案。
  - 点击「D = 2 / 4 / 8 卡」切换上下文并行卡数。
  - 使用播放器逐轮推进环路迭代, 观察每一轮各卡在左图因果矩阵中实际激活计算的子块位置。
  - 点击右侧表头的卡号可单独聚焦查看该卡负责的全部区块 (左图其余卡区块自动淡化, 表格下方同时列出该卡持有的具体 token 索引范围), 再次点击取消聚焦。
- 应该看到: 在连续切分模式下, 瓶颈标记 ▲ 每一轮都固定落在持有序列末尾的最后一张卡上, 而持有序列开头的卡 0 自第 1 轮起工作量全为 0 (后续传来的外部 KV 块全部位于其未来位置而被整块跳过); 切换至 zigzag 切分后, 每一轮内部所有卡的有效计算对数完全相等 (当 D 为 4 时第 0 轮各卡均为 36, 其余每轮各卡均为 32)。右侧读数含义如下:
  - 「每卡总工作量 max / min」: 最忙卡与最闲卡的全程累计计算对数, 两者相等即完全均衡时显示绿色。
  - 「墙钟」: 将每一轮瓶颈卡的最大工作量逐轮相加所得的同步等待总时间代价。
  - 「不利用因果性的墙钟」: 不跳过任何未来块时的基准耗时 (等于 D 乘单块满算规模, 当 D 为 4 时固定为 256)。
  - 「zigzag 相对连续切分」: 两种切分方案的墙钟耗时之比, 当 D 为 4 时 zigzag 实际吞吐快 1.73 倍。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: '固定并行卡数 D 为 4 并采用连续切分: 因果 mask 虽然省去了全局近一半的注意力计算量, 端到端墙钟耗时是否也随之减半? 切换至 zigzag 切分后结果如何变化?',
      answer: '连续切分下墙钟耗时并未减半, 对比两种切分方案如下:\n- 连续切分仅节省了 11% 的墙钟时间。\n- zigzag 切分才将因果稀疏省下的理论算力完整转化为端到端时间缩减。\n机制解析:\n- 全局总有效计算量: 长度为 32 的序列若忽略因果性需计算 32 乘 32 共 1024 对 q·k 内积, 利用因果 mask 仅需计算下三角区域共 528 对有效内积, 全局浮点工作量确实缩减了近一半, 且这一总量在两种切分下完全相同。\n- 连续切分的负载失衡: 卡 3 持有序列最末尾的 8 个 token, 其本地 query 排在全体前序 key 之后, 导致后续每一轮沿环传来的外部 KV 块全部处于其历史可见范围内, 每轮均需满载计算 8 乘 8 共 64 对内积 (仅第 0 轮计算本地对角下三角块为 36 对)。相反, 卡 0 持有最开头的 8 个 token, 除第 0 轮本地下三角的 36 对外, 后续传来的外部 KV 块全部位于未来区间而被直接跳过, 每轮计算量均为 0。由于同步环路每一轮都必须等待最慢的卡 3 完成计算后才能推进通信, 端到端墙钟耗时等于 $36 + 64 \\times 3 = 228$, 相对不跳过未来块的 256 基准仅缩短了 11%; 右表底部「合计」行的 [36, 100, 164, 228] 也清晰展示了越靠序列尾部的卡累计负担越重。\n- zigzag 的首尾互补均衡: 将整条序列细分为 $2D$ 即 8 段、每段含 4 个 token, 令卡 $r$ 配对持有第 $r$ 段与第 $2D-1-r$ 段 (如卡 0 持有第 0 段与第 7 段, 卡 3 持有第 3 段与第 4 段), 使每张卡同时拥有一段计算极轻的序列前部与一段计算极重的序列后部。互补配对后每张卡在每一轮的有效内积对数严格相等 (第 0 轮各卡均为 36, 此后三轮各卡均为 32), 全程累计工作量拉平为 [132, 132, 132, 132], 墙钟耗时随之降至 132, 相比连续切分的 228 提速 1.73 倍。\n两种方案下的全局总计算量 (528 对) 与跨卡通信体积 (每轮传递一个 KV 分块) 分毫无差, 唯一改变的是各卡在每轮同步屏障内的负载均衡度: 稀疏跳过省下的算力唯有均匀分摊到每一张并行卡上, 才能真正转化为端到端墙钟时间的缩短。',
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
        <p class="lab-note">右侧表格说明: 每一行对应环路流转的一轮, 每一列对应一张计算卡, 格内数字为该卡在当轮实际计算的 q·k 有效位置对数; 颜色编码对应表头色块标示的卡号, ▲ 标记当轮负载最重的瓶颈卡, 其余卡在该轮均需同步等待其计算完成。</p>
        <p class="lab-note">选中的卡持有的 token 位置: <span class="mono">{{ focus >= 0 ? posText(focus) : '点上面的卡号查看' }}</span></p>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>每卡总工作量 max / min (相等才是负载均衡)</span><b :class="sim.imb < 1.01 ? 'good' : 'bad'">{{ Math.max(...sim.perRank) }} / {{ Math.min(...sim.perRank) }}</b></div>
      <div class="kv"><span><Tex text="墙钟 = $\sum$(每轮最忙那张卡的工作量), 代表实际要等的时间" /></span><b :class="sim.wall <= Math.min(walls.cont, walls.zz) ? 'good' : 'bad'">{{ sim.wall }}</b></div>
      <div class="kv"><span>不利用因果性的墙钟 (每轮都算满一整块, D × 块大小)</span><b>{{ D * sim.block }}</b></div>
      <div class="kv"><span>zigzag 相对连续切分 (两种切法的墙钟之比)</span><b :class="{ good: speedup > 1 }">快 {{ speedup.toFixed(2) }}×</b></div>
      <div class="lab-note">
        <p>★ 在同步环路迭代中, 每一轮全体计算卡均受制于当轮耗时最长的瓶颈卡。因果 mask 省下的理论计算量唯有均衡分摊至每一张卡上, 才能真正转化为墙钟耗时缩减; 若计算负载集中堆积在单张卡上, 其余卡省下的算力只会耗散为空闲同步等待。</p>
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
