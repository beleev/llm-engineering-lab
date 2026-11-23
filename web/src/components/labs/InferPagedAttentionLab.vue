<!-- PagedAttention 实验台 (对应 llm_infer/m02_paged_attention): 事件列表 → computed 重放, 拖滑杆时结果确定。 -->
<template>
  <LabFrame
    title="PagedAttention — 把 KV 显存切成 block 按需分配, 看浪费少了多少"
    sub="- 模拟什么: 单张显卡上用于承载 KV cache 的显存池。推理引擎将整块显存池切分为固定大小的物理 block, 多条并发请求在池中交替写入各自的 KV 数据 (每生成一个新 token 即追加写入一个槽位)。
- 图上是什么: 界面分为上下两部分:
  - 上方一排方格: 显存池中的各个物理 block, 总容量固定为 384 个 token 的 KV。方格颜色标识其归属的请求, 无填充色代表空闲块; 方格内部的填充高度表示该 block 当前已写入的 token 数量。
  - 下方各行: 每条活跃请求及其对应的页表 block_table。页表中第 i 项记录的数字, 即该请求第 i 个逻辑页实际映射到的物理 block 编号。
- 可以做什么: 通过三个按钮管理请求生命周期, 通过三个滑杆调整内存管理参数:
  - 点击「新请求」接入一条新 prompt 并分配初始物理块。
  - 点击对应请求的「decode +1 / +8」推进生成并继续写入多个 token。
  - 点击「结束」释放该请求占用的全部物理 block 并归还显存池。
  - 拖动「block 大小」修改单个物理 block 可容纳的 token 数。
  - 拖动「连续预留的 max_len」调整传统连续分配基线为每条请求预先保留的最大长度。
  - 拖动「查地址」选定某一逻辑 token 位置, 右侧会显示其经页表转换后的物理块号与块内槽位偏移。
- 应该看到: 多条请求交替执行 decode 时, 各请求新申请到的物理 block 在显存池中呈非连续交错分布, 但依靠页表映射即可正常寻址而无需连续空间。右侧读数含义如下:
  - 「浪费」一栏同步对比两种管理方式的显存碎片开销: 分页机制仅在每条请求末尾未填满的最后一个 block 内产生少量内部碎片; 而「按 max_len 连续预留」必须为每条活跃请求预先锁定整段 max_len 空间, 尚未写入的槽位全部构成浪费。
  - 「连续预留装得下 N 条」展示同等总容量的显存池在传统整段连续预留方式下最多可容纳的并发请求数。"
    module="llm_infer/m02"
    run="python -m llm_infer.m02_paged_attention.demo"
    :challenge="{
      ask: '引入 PagedAttention 分页管理后, 单次注意力算子本身的计算速度会变快吗?',
      answer: '单次算子本身并不会更快, 反而存在轻微的寻址开销。\n读取任一历史 token 的 KV 时, GPU kernel 需先查询页表 block_table 定位其所在的物理 block, 再按偏移读取对应数据, 比访问连续张量多出一次间接寻址与非连续访存开销。\n分页机制真正换取的是显存利用率的大幅提升:\n- 分页分配: 每条请求仅最后一个物理 block 可能未填满, 最多浪费 $\\text{bs} - 1$ 个 token 槽位, 期望内部碎片约为 $\\text{bs}/2$ 个 token。\n- 连续预留: 由于无法预知请求最终的生成长度, 只能按 max_len 为每条请求预先分配整段连续空间, 未使用的预留槽位与外部碎片造成大量浪费; vLLM 论文实测表明传统连续预留下仅有 20~40% 的 KV 显存用于存储有效数据。\n显存利用率提升后带来以下连锁收益:\n1. 同一张显卡可同时容纳更多并发请求。\n2. 每步 decode 前向计算的实际 batch 更大, 更充分地摊薄权重读取带宽。\n3. 系统整体吞吐率 (每秒生成的 token 总数) 随之显著提升。\n将「block 大小」从 4 拖至 32 时, 右侧「浪费: 分页」读数会随之上升: block 越大, 每条请求末尾未填满块内的平均空闲槽位越多; 反之 block 过小虽能进一步压缩内部碎片, 但会导致页表变长、GPU kernel 访存粒度过碎。vLLM 默认将 block 大小设为 16, 正是在碎片率与访存效率之间取得的折中。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" class="active" @click="push({ type: 'new', len: nextLen })">新请求 (prompt {{ nextLen }} token)</button>
        <button type="button" @click="reset(seed)">重置</button>
        <button type="button" @click="reset(seed + 1)">换一组</button>
        <span class="msg" :class="{ bad: st.bad }">{{ st.msg }}</span>
      </div>
      <LabSlider v-model="bsExp" label="block 大小" :min="2" :max="5" :format="(v) => 2 ** v" unit=" tok" />
      <LabSlider v-model="maxLen" label="连续预留的 max_len" :min="64" :max="256" :step="16" unit=" tok" />
      <LabSlider v-model="pos" label="查地址: 逻辑位置 pos" :min="0" :max="Math.max(0, (sel?.len || 1) - 1)" />
    </template>

    <div class="pool">
      <span
        v-for="(b, i) in st.blocks" :key="i" class="cell blk"
        :class="{ dim: focus !== null && b.owner !== null && b.owner !== focus, hit: i === addr.block }"
        :style="blockStyle(b)" :title="b.owner === null ? `物理块 ${i}: 空闲, 没有请求在用` : `物理块 ${i}: 属于请求 ${b.owner}, 已写入 ${b.fill}/${bs} 个 token 的 KV`"
      >{{ i }}</span>
    </div>

    <div class="reqs">
      <div
        v-for="r in st.live" :key="r.id" class="req" :class="{ sel: r.id === selId }"
        @mouseenter="hover = r.id" @mouseleave="hover = null"
      >
        <button type="button" class="name" :style="{ borderColor: color(r.id) }" @click="selId = r.id">
          请求 {{ r.id }} · {{ r.len }} tok
        </button>
        <span class="table mono">
          [<button
            v-for="(b, i) in r.table" :key="i" type="button" class="tb"
            :class="{ on: r.id === selId && i === addr.page }" :title="`这条请求的第 ${i} 页 KV 放在物理块 ${b}`"
            @click="selId = r.id; pos = i * bs"
          >{{ b }}</button>]
        </span>
        <span class="ops">
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 1 })">decode +1</button>
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 8 })">+8</button>
          <button type="button" @click="push({ type: 'fin', id: r.id })">结束</button>
        </span>
      </div>
      <p v-if="!st.live.length" class="lab-note">当前显存池中暂无活跃请求, 点击「新请求」即可接入一条新序列。</p>
    </div>

    <template #stats>
      <div class="kv"><span>浪费的 KV 位置: 分页 / 按 max_len 连续预留</span><b :class="wasteCls(st.pagedWaste, st.contWaste)">{{ pct(st.pagedWaste) }} / {{ pct(st.contWaste) }}</b></div>
      <div class="kv">
        <span>同一个池子按 max_len 连续预留, 这 {{ st.live.length }} 条请求装得下</span>
        <b :class="st.contFit < st.live.length ? 'bad' : ''">{{ st.contFit }} 条</b>
      </div>
      <div class="kv"><span>空闲物理块 / 总数</span><b>{{ st.free }} / {{ st.blocks.length }}</b></div>
      <div v-if="sel" class="kv">
        <span>请求 {{ sel.id }} 的逻辑位置 pos {{ addr.pos }} 实际在</span>
        <b>物理块 {{ addr.block }} · 槽 {{ addr.slot }}</b>
      </div>
      <div class="lab-note">
        <p>浪费率的算法:</p>
        <ul class="pts">
          <li><b>分页:</b> (已分配的 block 容量 − 实际写入的 token 数) ÷ 已分配容量。</li>
          <li><b>连续预留:</b> (每条请求各占 max_len − 实际写入) ÷ 总预留量。</li>
        </ul>
        <p>分页更省就标绿, 更费就标红。</p>
        <p>地址翻译只有一行: <code class="inline">block_table[pos // bs], pos % bs</code>。</p>
        <ul class="pts">
          <li>pos 除以 block 大小的整数部分是第几页, 查页表得到物理块。</li>
          <li>余数是块内第几个槽。</li>
        </ul>
        <p>例如当块大小设为 8、逻辑位置 pos 为 37、页表为 [9, 4, 6, 1, 3] 时:</p>
        <ol class="pts">
          <li>37 // 8 = 4, 页表第 4 项是 3, 所以在物理块 3。</li>
          <li>37 % 8 = 5, 槽 5。</li>
        </ol>
        <ul class="pts">
          <li>几条请求交替 decode, 各自新要的块就交错排开, 一条请求的 KV 散落在池子各处。</li>
          <li>结束的请求把块还回空闲链表尾部 (与 m02 的 free_list 一致), 池子转过一圈后, 新请求拿到的全是零散块。</li>
        </ul>
        <p>借助页表间接映射, 逻辑上连续的序列 KV 无需占用连续的物理显存, 从而消除了显存池中的外部碎片空洞。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum, clamp } from '@/utils/labmath.js'

const POOL = 384, CAP = 64 // 池容量 (token) 与单请求长度上限
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--eye)']
const color = (id) => PALETTE[id % PALETTE.length]

const bsExp = ref(3)
const bs = computed(() => 2 ** bsExp.value)
const maxLen = ref(128)
const seed = ref(1)
const events = ref([])
const selId = ref(0)
const hover = ref(null)
const pos = ref(0)

// 第 i 个新请求的 prompt 长度只由 (seed, i) 决定, 重放时不会变
const lenOf = (i) => 5 + Math.floor(mulberry32(seed.value * 97 + i)() * 28)
const nextLen = computed(() => lenOf(events.value.filter((e) => e.type === 'new').length))
const push = (e) => { events.value = [...events.value, e]; if (e.type === 'new') selId.value = st.value.reqs.length - 1 }
const reset = (s) => {
  seed.value = s
  // 预置一段剧情: 接 4 条 → 交替 decode (新块交错) → 结束 1 号 → 再接 1 条
  const ev = range(4).map((i) => ({ type: 'new', len: lenOf(i) }))
  ev.push({ type: 'dec', id: 0, n: 8 }, { type: 'dec', id: 2, n: 8 }, { type: 'dec', id: 3, n: 8 }, { type: 'dec', id: 0, n: 8 })
  ev.push({ type: 'fin', id: 1 }, { type: 'new', len: lenOf(4) })
  events.value = ev; selId.value = 4; pos.value = 0
}
reset(1)

const st = computed(() => {
  const B = bs.value, nb = POOL / B, need = (len) => Math.ceil(len / B)
  const free = range(nb) // 空闲链表: 从头取, 还回来的接在尾部
  const reqs = []
  let msg = '', bad = false
  for (const e of events.value) {
    if (e.type === 'new') {
      const r = { id: reqs.length, len: e.len, table: [], live: false }
      reqs.push(r)
      bad = need(e.len) > free.length
      if (bad) { msg = `请求 ${r.id} 被拒: 需要 ${need(e.len)} 个 block, 只剩 ${free.length} 个`; continue }
      r.table = free.splice(0, need(e.len)); r.live = true
      msg = `请求 ${r.id} 拿到物理块 [${r.table.join(', ')}]`
      continue
    }
    const r = reqs[e.id]
    if (!r?.live) continue
    if (e.type === 'fin') { free.push(...r.table); msg = `请求 ${r.id} 结束, 归还 ${r.table.length} 个 block`; bad = false; r.table = []; r.live = false; continue }
    const len = Math.min(CAP, r.len + e.n), extra = need(len) - r.table.length
    bad = extra > free.length || len === r.len
    if (bad) { msg = len === r.len ? `请求 ${r.id} 已到长度上限 ${CAP}` : `请求 ${r.id} decode 失败: 没有空闲 block (真实系统此时会抢占)`; continue }
    // ★ 按需分配: 只有跨过 block 边界时才多要一个物理块
    r.table.push(...free.splice(0, extra)); r.len = len
    msg = extra ? `请求 ${r.id} 跨过 block 边界, 新增物理块 ${r.table.at(-1)}` : `请求 ${r.id} 写进最后一个 block 的空槽, 不需要新块`
  }
  const live = reqs.filter((r) => r.live)
  const blocks = range(nb).map(() => ({ owner: null, fill: 0 }))
  live.forEach((r) => r.table.forEach((b, i) => { blocks[b] = { owner: r.id, fill: Math.min(B, r.len - i * B) } }))
  const used = sum(live.map((r) => r.len)), alloc = sum(live.map((r) => r.table.length)) * B
  const reserved = live.length * maxLen.value
  return {
    reqs, live, blocks, msg, bad, free: free.length,
    pagedWaste: alloc ? (alloc - used) / alloc : 0,
    contWaste: reserved ? (reserved - sum(live.map((r) => Math.min(r.len, maxLen.value)))) / reserved : 0,
    contFit: Math.min(live.length, Math.floor(POOL / maxLen.value)), // 最好情况: 不算外部碎片
  }
})

// 两种分配方式互相比: 分页浪费得少就绿, 多就红, 一样就不上色
const wasteCls = (paged, cont) => (paged < cont ? 'good' : paged > cont ? 'bad' : '')
const sel = computed(() => st.value.live.find((r) => r.id === selId.value) || st.value.live[0] || null)
const focus = computed(() => hover.value ?? sel.value?.id ?? null)
watch(sel, (r) => { if (r) pos.value = clamp(pos.value, 0, r.len - 1) })
const addr = computed(() => {
  const r = sel.value
  if (!r) return { pos: 0, page: -1, block: -1, slot: 0 }
  const p = clamp(pos.value, 0, r.len - 1), page = Math.floor(p / bs.value)
  return { pos: p, page, block: r.table[page], slot: p % bs.value }
})

const pct = (x) => (x * 100).toFixed(1) + '%'
const blockStyle = (b) => {
  if (b.owner === null) return {}
  const c = color(b.owner), f = Math.round((b.fill / bs.value) * 100)
  return {
    borderColor: c,
    background: `linear-gradient(to top, color-mix(in srgb, ${c} 55%, transparent) ${f}%, color-mix(in srgb, ${c} 12%, transparent) ${f}%)`,
  }
}
</script>

<style scoped>
.pool { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 14px; }
.blk { width: 26px; height: 26px; color: var(--text); }
.blk.hit { outline: 2px solid var(--text); outline-offset: 1px; }
.reqs { display: flex; flex-direction: column; gap: 6px; }
.req { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 4px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); }
.req.sel { border-color: var(--border-strong); background: var(--bg-elev); }
.req button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.name { border-left-width: 4px; min-width: 116px; text-align: left; }
.table { font-size: 12px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 2px; align-items: center; }
.tb { min-width: 26px; padding: 2px 4px !important; font-family: inherit; }
.tb.on { border-color: var(--accent); color: var(--accent); }
.ops { display: flex; gap: 4px; margin-left: auto; }
.msg { font-size: 12px; color: var(--text-muted); }
.msg.bad { color: var(--danger); }
</style>
