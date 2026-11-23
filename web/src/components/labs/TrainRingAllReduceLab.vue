<!--
  Ring all-reduce 步进器 (对应 llm_train/core/collectives.py:ring_all_reduce_sum)。
  只讲一件事: 张量切 N 块沿环传, 每卡只发 2(N-1)/N·S 字节, 与 N 几乎无关。
-->
<template>
  <LabFrame
    title="Ring all-reduce — 梯度沿环接力相加, 卡数翻倍, 每张卡发的字节数却几乎不变"
    sub="- 模拟什么: 在数据并行训练中, 每张卡基于本地数据分片独立算出一份完整形状的梯度张量; 参数更新前需将全体卡上的梯度逐元素求和, 并使每张卡均持有这份全局总和, 该集合通信原语称为 all-reduce。ring all-reduce 是带宽最优的经典实现: 将 N 张卡首尾相连构成逻辑单向环, 每张卡每步仅向右侧相邻卡发送数据并从左侧相邻卡接收数据。本实验台每一步的数据块路由下标与 llm_train/core/collectives.py 中的 ring_all_reduce_sum 实现严格一致。完整梯度张量预先均分为 N 个数据块 (chunk), 分两个阶段完成同步:
  1. 前 $N-1$ 步为 reduce-scatter 阶段 (边环路归约求和边分散存储): 每一步每张卡将一个数据块发给右邻居, 右邻居接收后累加至本地同编号数据块上; 该阶段结束时, 每张卡恰好各自持有一个已汇集齐全部 N 张卡梯度的完整求和块, 其余块仅为部分和。
  2. 后 $N-1$ 步为 all-gather 阶段 (沿环广播拼回完整张量): 各卡将已归约完成的求和块继续沿环传递一圈, 接收方直接覆盖本地对应分块而不再累加; 全程结束时每张卡上的全部 N 个分块均为全局梯度总和。
- 图上是什么: 主体矩阵的每一行代表一张计算卡 (卡 0 至卡 N−1), 每一列代表梯度张量的一个切分块 (块 0 至块 N−1), 单个单元格即表示对应卡当前持有的对应数据块状态。
  - 格内如「2/4」的标记表示该分块当前已累加了 4 张卡中 2 张卡的局部梯度; 初始时刻每格均为 1/N, 即仅含本卡自身的梯度。
  - 紫色底色越深代表已汇入的卡数越多; 变为绿色 (N/N) 表示该格已包含全体卡的完整梯度总和。
  - 当前步正在向外发送的数据格带黄色虚线描边, 正在接收外部数据的目标格带黄色实线描边。
  - 矩阵下方文字逐一列出当前步的并发传输映射 (例如「卡0→卡1 发块0」), 环上所有链路在同一通信步内并行传输。
- 可以做什么: 控件从上到下依次为:
  - 拖动「卡数 N」修改环路节点数量 (范围为 2 至 8), 调整后自动重置回第 0 步。
  - 拖动「张量大小 S」设定待同步梯度张量的总字节规模 (范围为 64 MB 至 2048 MB), 该参数仅折算右侧通信字节读数而不改变状态矩阵结构。
  - 通过播放控制栏操作时间步: 点击「« 开头」重置回第 0 步, 点击「‹ 上一步」与「下一步 ›」单步进退, 点击「▶ 播放」自动连续演示, 拖动时间轴滑杆直接跳转至任意通信步, 右侧下拉框可调节自动播放速率。
  - 点击列标题「块 c」单独高亮追踪特定分块沿环流转与归约的全程轨迹, 其余列自动变暗; 再次点击取消聚焦。
  - 将鼠标悬停于任一单元格可在底部查看该分块当前已汇入的具体卡号集合; 点击单元格可锁定选中态, 随后可通过键盘方向键在矩阵内移动焦点。
- 应该看到: 在 reduce-scatter 阶段, 每一步各卡并发完成一发一收, 格内分子计数沿环逐跳递增; 第 N−1 步完成时每行恰好出现唯一一个绿色满和格, 即每张卡各自掌握一个分块的全局总和。进入 all-gather 阶段后, 这批绿格沿环依次接力复制覆盖, 最终一步整张状态表全部变绿。右侧读数含义如下:
  - 「阶段」: 指示当前步处于前一半归约分散还是后一半收集广播。
  - 「每卡已发送」: 截至当前步单张卡累计发出的字节量, 每推进一步固定增加 S/N。
  - 「走完全程的总量」: 斜杠左侧为 ring 方案下单卡全程发出的总字节数, 右侧为朴素汇总方案 (所有卡将完整张量发往 0 号主卡求和后再由 0 号卡广播回各卡) 下 0 号瓶颈卡需承担的收发总字节数。固定 S 为 512 MB 并将 N 从 4 拖至 8 可见: ring 方案的单卡总发送量仅从 768 MB 微增至 896 MB, 而朴素方案下 0 号卡的通信负担从 3.00 GB 激增至 7.00 GB。
  - 「已拿到完整和的格子」: 统计当前矩阵中的绿色满和格总数, 全程结束时达到 N×N 个。
  - 实验边界说明: 本实验台仅统计传输字节量, 对应通信耗时模型中的带宽项 (等于传输字节数除以单卡链路带宽), 未计入固定通信延迟项 (每步握手同步固有的启动开销, 与单跳字节量无关)。当卡数从 4 增至 8 时, ring 的串行通信步数从 6 步增至 14 步, 固定延迟项开销会随卡数线性累积。"
    module="llm_train/core"
    run="python -m llm_train.m09_collectives.demo"
    :challenge="{
      ask: '将环路卡数从 4 拖至 8 时, ring 方案下单张卡全程发送的总字节数会翻倍吗? 与「全部发往 0 号卡汇总后再广播」的朴素方案相比差异如何?',
      answer: '不会翻倍。在梯度张量总大小 S 固定为 512 MB 的条件下:\n- ring 方案: 单卡全程发送总量仅从 768 MB 增至 896 MB, 且无论卡数如何扩展上界恒不超过 $2S$。\n- 朴素中心化方案: 0 号主卡仅单向发送量便从 $3S$ 增至 $7S$, 随卡数增加呈线性增长。\n逐项解析:\n- ring 通信量推导: 完整张量均分为 N 块, 每块体积为 $S/N$。reduce-scatter 阶段执行 $N-1$ 步, all-gather 阶段再执行 $N-1$ 步, 每步每卡恰好发出一块, 全程累计发送 $2(N-1)$ 块, 折合总字节量为 $2(N-1)/N \\cdot S$。当 $N=4$ 时为 $1.5S$, 当 $N=8$ 时为 $1.75S$。随着卡数增加, 虽然串行发送步数增多, 但单块体积按 $1/N$ 同步缩小, 两者相互抵消后总发送量单调趋近但永不超越渐近上界 $2S$。将时间轴推至最后一步, 右侧「每卡已发送」读数即精确停在该理论值上。\n- 朴素中心化方案的瓶颈: 其余各卡需先将完整张量发给 0 号卡, 由 0 号卡求和后再将完整结果逐一发回各卡。0 号卡需接收 $N-1$ 份完整张量并发送 $N-1$ 份完整张量, 仅发送量即达 $(N-1) \\cdot S$ ($N=4$ 时为 $3S$, $N=8$ 时为 $7S$), 收发合计高达 $2(N-1) \\cdot S$, 对应右侧读数斜杠右侧的数值。与此同时其余从卡全程仅收发各一份张量, 集群绝大部分网卡带宽处于闲置状态, 0 号卡的单口带宽成为全局扩展瓶颈。\n- ring 的带宽优势: 将聚合通信流量均匀分摊至环上每一条相邻点对点链路, 每张卡收发字节严格对称, 集群全体网卡在每一步均满载并发工作。\n- ring 的步数延迟代价: 集合通信总耗时可分解为带宽项与延迟项两部分; 带宽项为传输字节量除以链路带宽, ring 已将其压缩至理论下界; 延迟项则是每一轮同步通信固有的启动握手开销, 与传输字节量无关。ring 全程需串行推进 $2(N-1)$ 步 ($N=8$ 时达 14 步), 延迟项随卡数 $N$ 线性增长。当待同步张量较小时延迟项将主导总耗时, 因此分布式训练框架通常先将大量小规模层梯度拼装进固定大小的通信桶 (bucket) 后再合并发起一次 all-reduce; 而在小消息、大规模节点场景下, 多卡通信库 NCCL 会自动切换为对数步数的树形 (tree) 归约算法。本实验台仅核算传输字节量, 未计入握手延迟耗时。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="卡数 N" :min="2" :max="8" />
      <LabSlider v-model="sizeMB" label="张量大小 S" :min="64" :max="2048" :step="64" unit=" MB" />
      <StepPlayer :stepper="stepper" :label="phaseLabel" />
    </template>

    <!-- 格子: 悬停临时看, 点击固定, 方向键移动固定的格子 -->
    <div class="cells ring" :style="{ gridTemplateColumns: `52px repeat(${n}, minmax(34px, 1fr))` }"
         tabindex="0" role="group" aria-label="卡 × 块 的累加进度表, 方向键移动选中的格子"
         @keydown.left.prevent="move(0, -1)" @keydown.right.prevent="move(0, 1)" @keydown.up.prevent="move(-1, 0)" @keydown.down.prevent="move(1, 0)">
      <span />
      <button
        v-for="c in n" :key="'h' + c" type="button" class="col-head mono"
        :class="{ active: trace === c - 1 }" title="只看这一块沿环走的路线, 别的列变暗; 再点一次取消" @click="trace = trace === c - 1 ? -1 : c - 1"
      >块 {{ c - 1 }}</button>
      <template v-for="(row, r) in frame.state" :key="r">
        <span class="row-label mono">卡 {{ r }}</span>
        <span
          v-for="(mask, c) in row" :key="c"
          class="cell"
          :class="[cellClass(r, c, mask), { dim: trace >= 0 && trace !== c, sel: cur && cur.r === r && cur.c === c }]"
          :style="{ background: heat(bits(mask) / n * 0.8, bits(mask) === n ? 'var(--left)' : 'var(--accent)') }"
          :title="`卡 ${r} 的块 ${c}: 含卡 {${members(mask).join(',')}} 的梯度`"
          @mouseenter="hover = { r, c }" @mouseleave="hover = null" @click="pin = { r, c }"
        >{{ bits(mask) }}/{{ n }}</span>
      </template>
    </div>
    <p class="lab-note msg" aria-live="polite">
      <template v-if="frame.msgs.length">本步各卡同时发: {{ frame.msgs.map((m) => `卡${m.src}→卡${m.dst} 发块${m.c}`).join(' · ') }}</template>
      <template v-else>初始: 每张卡的每一块都只含自己的梯度 (每格 1/{{ n }})。按「▶ 播放」, 或点「下一步 ›」一步一步走; 悬停或点击格子看它含哪些卡。</template>
      <br />
      <span v-if="cur">卡 {{ cur.r }} 的块 {{ cur.c }} 已含卡 { {{ members(frame.state[cur.r][cur.c]).join(', ') }} } 的梯度。</span>
    </p>

    <template #stats>
      <div class="kv"><span>阶段 (前一半边传边加, 后一半传一圈拷贝)</span><b>{{ frame.k <= n - 1 ? (frame.k === 0 ? '—' : 'reduce-scatter') : 'all-gather' }}</b></div>
      <div class="kv"><span>每卡已发送 (到这一步为止, 每步加 S/N)</span><b>{{ fmtMB(frame.k * sizeMB / n) }}</b></div>
      <div class="kv"><span><Tex text="走完全程的总量: ring 每卡发送 $\frac{2(N-1)}{N} S$ / 朴素做法 0 号卡收发合计 $2(N-1) S$" /></span><b class="pair">{{ fmtMB(2 * (n - 1) / n * sizeMB) }} / {{ fmtMB(2 * (n - 1) * sizeMB) }}</b></div>
      <div class="kv"><span>已拿到完整和的格子 (绿格数, 走完是 N×N)</span><b :class="{ good: doneCells === n * n }">{{ doneCells }} / {{ n * n }}</b></div>
      <div class="lab-note">
        <p><Tex text="★ 分块路由下标规则: 在 reduce-scatter 阶段的第 1 步, 卡 $r$ 发送本地块 $r$; 此后每一步发送的恰好是上一步刚从左邻居接收并完成累加的分块, 对应块编号为 $(r - t) \bmod N$ (其中 $t$ 为当前阶段已完成的步数, $\bmod N$ 表示模 $N$ 取非负余数, 下标为负时循环绕回 $N-1$)。由此每个逻辑分块均沿单向环依次走遍各卡并逐站累加。" /></p>
        <p><Tex text="当 reduce-scatter 阶段结束时, 卡 $r$ 恰好持有分块 $(r+1) \bmod N$ 的完整全局总和, 对应矩阵中每一行唯一的绿色单元格。" />这一半单独拿出来用, 就是 ZeRO-2 的梯度同步: ZeRO-2 让每张卡只负责更新 1/N 的参数, 所以只需要自己那一片梯度的和; 后半段的 all-gather 挪到更新之后, 用来把各卡更新好的参数片拼回整份。</p>
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

const n = ref(4)
const sizeMB = ref(512)
const trace = ref(-1)
const hover = ref(null), pin = ref(null)     // 悬停的 / 点击固定的格子 { r, c }
const cur = computed(() => hover.value || pin.value)
const move = (dr, dc) => {
  const p = cur.value || { r: 0, c: 0 }, hi = n.value - 1
  pin.value = { r: Math.min(hi, Math.max(0, p.r + dr)), c: Math.min(hi, Math.max(0, p.c + dc)) }
  hover.value = null
}

const bits = (m) => { let k = 0; for (; m; m >>= 1) k += m & 1; return k }
const members = (m) => range(8).filter((i) => m & (1 << i))
const fmtMB = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + ' GB' : mb.toFixed(0) + ' MB')

// 逐帧状态: state[r][c] 是一个位掩码, 记录卡 r 的块 c 已经累加了哪些卡的梯度。与 Python 实现同一套下标。
const frames = computed(() => {
  const N = n.value
  let state = range(N).map((r) => range(N).map(() => 1 << r))
  const out = [{ k: 0, state, msgs: [] }]
  for (const phase of ['reduce', 'gather']) {
    for (let t = 0; t < N - 1; t++) {
      const shift = phase === 'reduce' ? 0 : 1
      // ★ 同一步内所有卡同时发: 先取快照再写, 否则后发的卡会读到本步刚收到的数据
      const msgs = range(N).map((r) => ({ src: r, dst: (r + 1) % N, c: (((r + shift - t) % N) + N) % N }))
      const next = state.map((row) => [...row])
      for (const { src, dst, c } of msgs) next[dst][c] = phase === 'reduce' ? state[dst][c] | state[src][c] : state[src][c]
      state = next
      out.push({ k: out.length, state, msgs, phase })
    }
  }
  return out
})

const stepper = useStepper(() => frames.value.length, { interval: 800 })
const frame = computed(() => frames.value[Math.min(stepper.step.value, frames.value.length - 1)])
const doneCells = computed(() => frame.value.state.flat().filter((m) => bits(m) === n.value).length)
const phaseLabel = computed(() => `第 ${frame.value.k} / ${2 * (n.value - 1)} 步`)
watch(n, () => { stepper.reset(); trace.value = -1; hover.value = null; pin.value = null })

const cellClass = (r, c, mask) => {
  const m = frame.value.msgs
  if (m.some((x) => x.dst === r && x.c === c)) return 'recv'
  if (m.some((x) => x.src === r && x.c === c)) return 'send'
  return bits(mask) === n.value ? 'full' : ''
}
</script>

<style scoped>
.ring .cell { height: 30px; font-size: 11px; color: var(--text); cursor: pointer; }
.cell.sel { box-shadow: inset 0 0 0 2px var(--text); }
.pair { font-size: 13px !important; }
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.col-head { font-size: 11px; min-height: 26px; padding: 2px 4px; }
.cell.send { outline: 2px dashed var(--warn); outline-offset: 1px; }
.cell.recv { outline: 2px solid var(--warn); outline-offset: 1px; }
.cell.full { border-color: var(--left); }
.msg { margin-top: 12px; min-height: 3.4em; }
</style>
