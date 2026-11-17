<!--
  Ring all-reduce 步进器 (对应 llm_train/core/collectives.py:ring_all_reduce_sum)。
  只讲一件事: 张量切 N 块沿环传, 每卡只发 2(N-1)/N·S 字节, 与 N 几乎无关。
-->
<template>
  <LabFrame
    title="Ring all-reduce — 梯度沿环接力相加, 卡数翻倍, 每张卡发的字节数却几乎不变"
    sub="- 模拟什么: 数据并行训练时, 每张卡用自己那份数据算出一份完整的梯度; 更新参数前, 要把所有卡的梯度逐元素加起来, 而且每张卡都要拿到这份和。这个操作叫 all-reduce。ring all-reduce 是它最常用的实现: N 张卡首尾连成一个环, 每张卡只给右边的邻居发数据。台子的每一步和 llm_train/core/collectives.py 里 ring_all_reduce_sum 用的是同一套下标。张量先切成 N 块 (chunk), 分两个阶段传:
        1. 前 $N-1$ 步叫 reduce-scatter (先求和, 再把和分散到各卡): 每步每张卡把一块发给右邻居, 右邻居把它加到自己的同号块上; 结束时每张卡各有一块已经加齐了所有卡, 别的块还不全。
        2. 后 $N-1$ 步叫 all-gather (把各卡手里的片收集成整份): 把加齐的块再绕环传一圈, 这次收到的直接覆盖、不再相加; 结束时每张卡的每一块都是完整的和。
      - 图上是什么: 表格的每一行是一张卡 (卡 0 到卡 N−1), 每一列是张量的一块 (块 0 到块 N−1), 一格就是「这张卡手里的这一块」。
        - 格子里的「2/4」表示这一块已经加进了 4 张卡里 2 张的梯度; 一开始每格都是 1/N, 只有卡自己的。
        - 紫色越深, 加进来的卡越多; 变成绿色 (N/N) 表示这一格已经是全部卡的和。
        - 本步正在发出的格子带黄色虚线框, 正在接收的格子带黄色实线框。
        - 表格下面一行文字列出这一步谁发给谁, 例如「卡0→卡1 发块0」, 所有卡是同时发的。
      - 可以做什么: 控件从上到下:
        - 拖「卡数 N」改环上有几张卡 (2 到 8), 改完会回到第 0 步。
        - 拖「张量大小 S」改要同步的张量有多大 (64 MB 到 2048 MB), 它只影响右侧的字节数, 不影响表格。
        - 播放条上, 「« 开头」回到第 0 步, 「‹ 上一步」和「下一步 ›」一步一步走, 「▶ 播放」自动往后播, 时间轴滑杆直接跳到任意一步, 最右边的下拉框选播放速度。
        - 点列头「块 c」只看这一块的旅程, 别的列会变暗, 再点一次取消。
        - 鼠标悬停在格子上, 下方文字显示这一格已经含有哪几张卡的梯度; 点击格子把它固定住, 之后在表格上按方向键可以移动选中的格子。
      - 应该看到: reduce-scatter 阶段, 每一步每张卡发一块、收一块, 格子里的数字一路涨, 每块沿着环走, 走到哪加到哪。第 N−1 步走完, 每一行恰好有一格变绿, 每张卡各握着一块的全局和。all-gather 阶段, 这些绿格沿着环被复制出去, 最后一步整张表全绿。右侧读数逐项看:
        - 「阶段」: 当前在哪一半。
        - 「每卡已发送」: 到这一步为止每张卡发出的字节数, 每步加 S/N。
        - 「走完全程的总量」: 左边是 ring 每张卡最终发出的字节数, 右边是朴素做法 (所有卡把整份张量发给 0 号卡, 0 号卡加完再发回给每张卡) 下 0 号卡要收发的字节数。把 N 从 4 拖到 8 (S 保持 512 MB): ring 每卡的总量只从 768 MB 涨到 896 MB, 朴素做法的 0 号卡从 3.00 GB 涨到 7.00 GB。
        - 「已拿到完整和的格子」: 数绿格, 全部走完是 N×N 个。
        - 台子只数字节, 对应通信时间里的带宽项 (传字节花的时间, 等于字节数除以网卡带宽); 它不算延迟项 (每一步通信都要付的固定启动开销, 和传多少字节无关)。ring 的步数从 6 涨到 14, 延迟项会跟着涨。"
    module="llm_train/core"
    run="python -m llm_train.m09_collectives.demo"
    :challenge="{
      ask: '把卡数从 4 拖到 8, 每张卡要发送的总字节数会翻倍吗? 对比「全发给 0 号卡再广播」呢?',
      answer: '不会翻倍。S 取 512 MB 时:\n- ring: 每张卡发送的总量从 768 MB 只涨到 896 MB, 卡再多也不超过 $2S$。\n- 朴素做法: 0 号卡光发送就从 $3S$ 涨到 $7S$, 跟着卡数线性涨。\n逐条解释:\n- ring 怎么算: 张量切成 N 块, 每块 $S/N$。reduce-scatter 走 $N-1$ 步, all-gather 再走 $N-1$ 步, 每步每张卡发一块, 一共发 $2(N-1)$ 块, 合计 $2(N-1)/N \\cdot S$。$N=4$ 是 $1.5S$, $N=8$ 是 $1.75S$。卡多了, 要发的块数变多, 但每块变小, 两者几乎抵消, 总量越来越接近 $2S$ 但到不了。右侧「每卡已发送」走到最后一步, 正好等于这个数。\n- 朴素做法: 所有卡把整份张量发给 0 号卡, 0 号卡加完再把结果发回每张卡。0 号卡要收 $N-1$ 份、再发 $N-1$ 份, 光发送就是 $(N-1) \\cdot S$ ($N=4$ 时 $3S$, $N=8$ 时 $7S$), 收发合计 $2(N-1) \\cdot S$, 就是右侧读数斜杠右边的数。其他卡只发一份、收一份, 网口大部分时间闲着, 0 号卡的网口成了瓶颈。\n- ring 的好处: 流量均摊到环上的每一条链路, 每张卡收发一样多, 所有网口同时在干活。\n- ring 的代价是步数。通信时间可以拆成两项: 带宽项是传字节花的时间 (字节数除以网卡带宽), ring 把它压到了最低; 延迟项是每一步都要付的固定启动开销 (发起一次通信、等对方准备好), 和传多少字节无关。ring 要走 $2(N-1)$ 步, $N=8$ 时 14 步, 延迟项随 $N$ 线性涨。张量很小时延迟项占大头, 所以训练框架会把很多小梯度先攒成一个大桶 (bucket) 再一起 all-reduce; 消息小、卡又多时, NCCL (英伟达的多卡通信库) 会改用 tree 等别的算法。这个台子只数字节, 不算延迟。',
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
        <p><Tex text="★ 谁发哪块: reduce-scatter 的第 1 步, 卡 $r$ 发自己的块 $r$; 之后每一步, 发的正是上一步刚收到、刚加好的那块, 也就是块 $(r - t) \bmod N$ ($t$ 是本阶段已经走完的步数; $\bmod N$ 是除以 $N$ 取余数, 块号减到负数就绕回 $N-1$)。所以每块都沿着环走一圈, 走到哪加到哪。" /></p>
        <p><Tex text="reduce-scatter 结束时, 卡 $r$ 恰好握着块 $(r+1) \bmod N$ 的全局和, 就是每一行唯一的那个绿格。" />这一半单独拿出来用, 就是 ZeRO-2 的梯度同步: ZeRO-2 让每张卡只负责更新 1/N 的参数, 所以只需要自己那一片梯度的和; 后半段的 all-gather 挪到更新之后, 用来把各卡更新好的参数片拼回整份。</p>
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
