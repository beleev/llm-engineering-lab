<!--
  全站统一的 DAG 视图。所有数据流 / 架构图 / 流程图都用它, 不要再手写 SVG 坐标。
  坐标由 dagre 算, 所以改数据、换方向、折叠展开都不会把版式挤乱。
  可以拖节点、拖画布, 用右上角的按钮或触控板捏合缩放; 标了 group 的节点可以折叠成一个, 展开时画一个框圈住它们。
  滚轮不缩放: 图嵌在长页面里, 滚轮滚到图上时应该继续滚页面。

  什么会触发重新排版 (节点回到 dagre 算的位置, 读者拖过的位置作废):
    节点集合、边集合、框、方向变了; 节点的 label / sub / note / badge 变了; 节点渲染出来的尺寸变了。
  什么不会: active / dim / color / kind / aria 这类只改外观的字段, 以及 #node 插槽里用到的其他字段。
    它们只更新样式, 所以可以放心把 hover、选中状态写进节点的 active。
  视图 (缩放和平移) 只在三种时候自动对齐到全图: 首次布局; 点「重排」;
    读者还没动过视图时发生的重新排版。读者自己缩放或拖过画布之后, 重新排版时视图保持他调好的样子。

  数据格式:
    nodes: [{ id, label, sub?, note?, kind?, color?, badge?, group?, box?, active?, dim? }]
    edges: [{ from, to, label?, kind?: 'main' | 'side' | 'weak', active?, dim? }]
    groups: [{ id, label, collapsed? }]
    boxes: [{ id, label, parent? }]   只画框、不带折叠。node.box 指向框; parent 指向外层框, 可以嵌套

  想在节点里放别的东西 (比例条、多行数值) 就用 #node 插槽, 外框和连线照旧:
    <DagView v-bind="g" @node-hover="hover = $event">
      <template #node="{ node }">...</template>
    </DagView>
-->
<template>
  <div ref="box" class="dag" :style="{ height: boxH + 'px' }">
    <VueFlow
      v-model:nodes="fNodes" v-model:edges="fEdges"
      :nodes-draggable="true" :nodes-connectable="false" :elements-selectable="false"
      :min-zoom="0.25" :max-zoom="2" :zoom-on-double-click="false" :zoom-on-scroll="false"
      :nodes-focusable="false" :edges-focusable="false"
      fit-view-on-init @nodes-initialized="measure" @node-drag="syncFrames" @move-start="touched = true"
    >
      <!-- Vue Flow 外层的节点和边不进 Tab 顺序 (focusable=false), 每个节点只留里面这一个停靠点。
           节点内容读的是 look(data): 排版时的那份数据 + 调用方当前传进来的外观字段 -->
      <template #node-box="{ data }">
        <Handle type="target" :position="side[0]" />
        <Handle type="source" :position="side[1]" />
        <div :class="['dag-node', look(data).kind, { grouped: data.groupId, on: look(data).active, dim: look(data).dim }]"
             :style="tint(look(data).color)" :title="look(data).note || ''"
             role="button" tabindex="0" :aria-label="look(data).aria || look(data).label"
             @mouseenter="emit('node-hover', data.id)" @mouseleave="emit('node-hover', null)"
             @focus="emit('node-hover', data.id)" @blur="emit('node-hover', null)"
             @click="hit(data)" @keydown.enter="hit(data)" @keydown.space.prevent="hit(data)">
          <slot name="node" :node="look(data)">
            <div class="n-head">
              <span class="n-label">{{ look(data).label }}</span>
              <span v-if="look(data).badge" class="n-badge mono">{{ look(data).badge }}</span>
            </div>
            <div v-if="look(data).sub" class="n-sub mono">{{ look(data).sub }}</div>
            <div v-if="look(data).note" class="n-note">{{ look(data).note }}</div>
          </slot>
          <span v-if="data.groupId" class="n-fold mono">{{ data.folded ? '展开 ▸' : '折叠 ▾' }}</span>
        </div>
      </template>
      <template #node-frame="{ data }">
        <span class="dag-frame-label">{{ data.label }}</span>
      </template>
      <!-- 边走 dagre 算出来的折线, 长边会绕开中间的节点; 端点用 Vue Flow 的实时坐标, 拖节点时线跟着走 -->
      <!-- 线的粗细、颜色、标签也读调用方当前的数据, 所以点亮一条线不用重新排版 -->
      <template #edge-poly="e">
        <BaseEdge :id="e.id" :path="pathOf(e)" :label="edgeNow(e).label"
                  :class="[edgeNow(e).kind || 'side', { on: edgeNow(e).active, dim: edgeNow(e).dim }]"
                  :style="edgeNow(e).color ? { stroke: edgeNow(e).color } : undefined"
                  :label-x="labelAt(e).x" :label-y="labelAt(e).y" label-show-bg />
      </template>
      <Controls position="top-right" :show-interactive="false"
                @zoom-in="touched = true" @zoom-out="touched = true" @fit-view="touched = false" />
    </VueFlow>

    <div class="dag-bar">
      <button v-for="g in foldable" :key="g.id" type="button" :aria-expanded="!folded.has(g.id)"
              :class="{ active: !folded.has(g.id) }" @click="toggle(g.id)">
        {{ folded.has(g.id) ? '▸' : '▾' }} {{ g.label }}
      </button>
      <button type="button" title="节点回到自动排版的位置, 视图对齐到全图" @click="resetView">重排</button>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { BaseEdge, Handle, Position, VueFlow, useVueFlow } from '@vue-flow/core'
import { Controls } from '@vue-flow/controls'
import dagre from '@dagrejs/dagre'
import '@vue-flow/core/dist/style.css'
import '@vue-flow/controls/dist/style.css'

const props = defineProps({
  nodes: { type: Array, required: true },
  edges: { type: Array, default: () => [] },
  groups: { type: Array, default: () => [] },
  boxes: { type: Array, default: () => [] },
  dir: { type: String, default: 'TB' },      // TB 竖排 · LR 横排
  height: { type: Number, default: 0 },      // 0 = 按图的实际高度自适应
})

const emit = defineEmits(['node-hover', 'node-click'])

const { fitView } = useVueFlow()
const folded = ref(new Set(props.groups.filter((g) => g.collapsed).map((g) => g.id)))
const foldable = computed(() => props.groups)
const toggle = (id) => {
  const s = new Set(folded.value)
  s.has(id) ? s.delete(id) : s.add(id)
  folded.value = s
}
// 点组的把手是折叠, 点普通节点是抛给调用方
const hit = (d) => (d.groupId ? toggle(d.groupId) : emit('node-click', d.id))

// 节点尺寸。dagre 要知道多大才能排版, 但排版前节点还没渲染, 所以分两步:
// 先按字数估一版, 渲染出来后 measure() 量到真实尺寸再排一次。中日文字宽、note 会折行,
// 估算必然不准, 这一步不能省, 否则行距会被挤掉。
// 键名必须是 width / height —— dagre 只认这两个, 写成 w/h 它会静默当 undefined, 坐标全变 NaN。
const measured = ref(new Map())
const sizeOf = (n) => {
  const real = measured.value.get(n.id)
  if (real) return real
  const chars = Math.max(String(n.label).length, String(n.sub || '').length * 0.72)
  return {
    width: Math.min(240, Math.max(104, chars * 8.6 + 26)),
    height: 34 + (n.sub ? 16 : 0) + (n.note ? 18 : 0),
  }
}

// 视图什么时候自动对齐到全图, 见文件头的说明
let wantFit = true                 // 下一次排版稳定后要不要 fitView
const touched = ref(false)         // 读者自己缩放或拖过画布

// 量真实尺寸; 和排版用的尺寸有出入就重排一次。尺寸不变时直接返回, 不会自己转起来。
const box = ref(null)
const measure = () => {
  const els = box.value?.querySelectorAll('.vue-flow__node') || []
  let changed = false
  const next = new Map(measured.value)
  for (const el of els) {
    if (el.classList.contains('vue-flow__node-frame')) continue   // 框的尺寸是算出来的, 不参与排版
    const id = el.getAttribute('data-id')
    const size = { width: el.offsetWidth, height: el.offsetHeight }
    if (!size.width || !size.height) continue
    const old = next.get(id)
    if (old && Math.abs(old.width - size.width) < 1 && Math.abs(old.height - size.height) < 1) continue
    next.set(id, size)
    changed = true
  }
  if (changed) measured.value = next
  else if (wantFit) { wantFit = false; fit() }
}

// 折叠: 整组换成一个节点, 进出这一组的边改接到它身上
const resolved = computed(() => {
  const hidden = new Map()                       // 原 id → 它被折进了哪个组
  for (const n of props.nodes) if (n.group && folded.value.has(n.group)) hidden.set(n.id, n.group)

  // 拷一份再改 —— 下面要往节点上挂折叠把手, 直接改 props.nodes 会把标记
  // 永久写进调用方的数据 (静态图是模块级对象, 会跨实例、跨页面残留)
  const nodes = props.nodes.filter((n) => !hidden.has(n.id)).map((n) => ({ ...n }))
  for (const g of props.groups) {
    if (!folded.value.has(g.id)) continue
    const members = props.nodes.filter((n) => n.group === g.id)
    if (!members.length) continue
    nodes.push({ id: g.id, label: g.label, sub: `${members.length} 步`, kind: g.kind || 'group', groupId: g.id, folded: true })
  }
  // 展开状态下, 组内第一个节点挂上折叠把手
  for (const g of props.groups) {
    if (folded.value.has(g.id)) continue
    const first = nodes.find((n) => n.group === g.id)
    if (first) Object.assign(first, { groupId: g.id, folded: false })
  }

  // 去重只针对折叠造出来的重复: 组里三个节点各有一条边连到外面同一个节点, 折起来就成了三条一样的线。
  // 调用方自己写的同向两条边 (端点都没被折叠) 是两条不同的线, 都要画。
  const seen = new Set()
  const edges = []
  for (const e of props.edges) {
    const from = hidden.get(e.from) || e.from
    const to = hidden.get(e.to) || e.to
    if (from === to) continue                     // 组内部的边折叠后不画
    if (hidden.has(e.from) || hidden.has(e.to)) {
      const key = `${from}->${to}`
      if (seen.has(key)) continue
      seen.add(key)
    }
    edges.push({ ...e, from, to })
  }
  // 子图框 = 调用方给的 boxes + 展开着的 group。只留圈得住东西的框, 空框 dagre 会排出一个孤零零的点
  for (const n of nodes) if (!n.box && n.group && !folded.value.has(n.group)) n.box = n.group
  const all = [...props.boxes, ...props.groups.filter((g) => !folded.value.has(g.id))]
  const used = new Set()
  const mark = (id) => { for (; id && !used.has(id); id = all.find((b) => b.id === id)?.parent) used.add(id) }
  for (const n of nodes) mark(n.box)
  return { nodes, edges, boxes: all.filter((b) => used.has(b.id)) }
})

// maxZoom 1: 图小的时候别放大, 不然字会糊
const fit = () => fitView({ padding: 0.12, duration: 0, maxZoom: 1 })

// 容器高度默认跟着图走 —— 固定高度会把竖排的长图压到 0.5 倍, 字就看不清了
const boxH = computed(() => {
  const ns = [...laid.value.nodes, ...frameRects(rectsOf(laid.value.nodes)).values()]
  if (props.height) return props.height
  if (!ns.length) return 240
  const span = Math.max(...ns.map((n) => n.y + n.h)) - Math.min(...ns.map((n) => n.y))
  return Math.round(Math.min(680, Math.max(200, span + 52)))
})

// ── 折线工具 ──────────────────────────────────────────────────────
// dagre 每跨一个 rank 就吐一个点, 一条直线上会有好几个; 只留拐点
const trimStraight = (pts) =>
  pts.filter((p, i) => {
    if (i === 0 || i === pts.length - 1) return true
    const a = pts[i - 1], b = pts[i + 1]
    return Math.abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) > 0.5
  })

// 从 p 朝 q 挪 r 像素, 用来把直角削成圆角
const toward = (p, q, r) => {
  const dx = q.x - p.x, dy = q.y - p.y
  const len = Math.hypot(dx, dy) || 1
  const k = Math.min(r, len / 2) / len
  return { x: p.x + dx * k, y: p.y + dy * k }
}

const polyPath = (pts, r = 9) => {
  if (pts.length < 2) return ''
  let d = `M${pts[0].x},${pts[0].y}`
  for (let i = 1; i < pts.length - 1; i++) {
    const a = toward(pts[i], pts[i - 1], r), b = toward(pts[i], pts[i + 1], r)
    d += ` L${a.x},${a.y} Q${pts[i].x},${pts[i].y} ${b.x},${b.y}`
  }
  const last = pts[pts.length - 1]
  return `${d} L${last.x},${last.y}`
}

const wayPoints = (e) => [
  { x: e.sourceX, y: e.sourceY },
  ...(e.data?.via || []),
  { x: e.targetX, y: e.targetY },
]
const pathOf = (e) => polyPath(wayPoints(e))
const labelAt = (e) => {
  const pts = wayPoints(e)
  const i = (pts.length - 1) / 2
  const a = pts[Math.floor(i)], b = pts[Math.ceil(i)]
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}

// dagre 算坐标 —— 全站唯一一处排版逻辑
const laid = ref({ nodes: [], edges: [] })
const relayout = () => {
  if (!touched.value) wantFit = true
  // compound: 同一个框里的节点排在一起, 框外的节点不会插进来
  // multigraph: 同一对节点之间允许有多条边, 每条边用序号当名字, dagre 会给它们各算一条路
  const g = new dagre.graphlib.Graph({ compound: true, multigraph: true })
  g.setGraph({ rankdir: props.dir, nodesep: 26, ranksep: 46, marginx: 12, marginy: 12 })
  g.setDefaultEdgeLabel(() => ({}))
  const { nodes, edges, boxes } = resolved.value
  // 框在 dagre 里加前缀, 免得和节点重名 (架构图里 blocks 既是节点, 也是它内部组件那个框的 id)
  // 注意: dagre 不支持边直接连到框上, 边只能连节点
  const cid = (id) => `box:${id}`
  for (const b of boxes) g.setNode(cid(b.id), {})
  for (const b of boxes) if (b.parent && g.hasNode(cid(b.parent))) g.setParent(cid(b.id), cid(b.parent))
  for (const n of nodes) {
    g.setNode(n.id, sizeOf(n))
    if (n.box && g.hasNode(cid(n.box))) g.setParent(n.id, cid(n.box))
  }
  edges.forEach((e, i) => { if (g.hasNode(e.from) && g.hasNode(e.to)) g.setEdge(e.from, e.to, {}, `e${i}`) })
  dagre.layout(g)
  laid.value = {
    nodes: nodes.map((n) => {
      const p = g.node(n.id)
      return { ...n, x: p.x - p.width / 2, y: p.y - p.height / 2, w: p.width, h: p.height }
    }),
    // 掐掉首尾两点 —— 它们落在节点边框上, 端点改用 Vue Flow 的实时 handle 坐标
    edges: edges.map((e, i) => ({ ...e, via: trimStraight((g.edge(e.from, e.to, `e${i}`)?.points || []).slice(1, -1)) })),
    boxes,
  }
}
// 「重排」按钮: 节点回到排版位置, 视图也对齐回全图
const resetView = () => { touched.value = false; relayout() }

// ── 子图框 ────────────────────────────────────────────────────────
// 框的位置不用 dagre 给的, 按成员的当前位置现算: 拖节点时框跟着伸缩, 节点不会被拖出框。
// dagre 的 compound 只负责让成员排在一起, 并在框四周留出空隙。
const PAD = { top: 22, side: 10 }                // top 要放得下框的标题
const rectsOf = (ns) => new Map(ns.map((n) => [n.id, { x: n.x, y: n.y, w: n.w, h: n.h }]))
const frameRects = (rects) => {
  const boxes = laid.value.boxes || []
  const out = new Map()
  const rectOf = (b) => {
    if (out.has(b.id)) return out.get(b.id)
    const inner = [                               // 成员节点 + 内层框
      ...laid.value.nodes.filter((n) => n.box === b.id).map((n) => rects.get(n.id)),
      ...boxes.filter((c) => c.parent === b.id).map(rectOf),
    ].filter(Boolean)
    const x0 = Math.min(...inner.map((r) => r.x)) - PAD.side
    const y0 = Math.min(...inner.map((r) => r.y)) - PAD.top
    const x1 = Math.max(...inner.map((r) => r.x + r.w)) + PAD.side
    const y1 = Math.max(...inner.map((r) => r.y + r.h)) + PAD.side
    const r = inner.length ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0, label: b.label } : null
    out.set(b.id, r)
    return r
  }
  boxes.forEach(rectOf)
  for (const [k, v] of out) if (!v) out.delete(k)
  return out
}

const frameNodes = (rects) =>
  [...frameRects(rects)].map(([id, r]) => ({
    id: `box:${id}`, type: 'frame', position: { x: r.x, y: r.y },
    draggable: false, selectable: false, focusable: false, zIndex: -1,
    style: { width: `${r.w}px`, height: `${r.h}px` },
    data: { label: r.label },
  }))

const syncFrames = () => {
  const size = new Map(laid.value.nodes.map((n) => [n.id, n]))
  const rects = new Map()
  for (const n of fNodes.value) {
    const s = size.get(n.id)
    if (s && n.type !== 'frame') rects.set(n.id, { x: n.position.x, y: n.position.y, w: s.w, h: s.h })
  }
  const frames = new Map(frameNodes(rects).map((f) => [f.id, f]))
  fNodes.value = fNodes.value.map((n) => (n.type === 'frame' ? frames.get(n.id) || n : n))
}
// ★ 只有影响版式的字段变了才重新排版。把它们拼成一个字符串来比: 字符串没变, watch 就不触发。
// 调用方每次悬停都会传一个新的 nodes 数组 (active 变了), 直接 watch 数组会次次重排、把读者拖过的位置冲掉。
// 简化: 每次数据变化都把全图序列化一遍, O(节点数 + 边数); 图到几百个节点还嫌慢, 再换成逐字段比较。
const layoutKey = computed(() => JSON.stringify([
  resolved.value.nodes.map((n) => [n.id, n.label, n.sub, n.note, n.badge, n.box, n.groupId, n.folded]),
  resolved.value.edges.map((e) => [e.from, e.to]),
  resolved.value.boxes.map((b) => [b.id, b.label, b.parent]),
]))
watch(layoutKey, relayout, { immediate: true })
watch(() => props.dir, relayout)
watch(measured, relayout)
// 外观字段变了: 模板通过 look() / edgeNow() 自己会更新, 这里只补量一次尺寸 (插槽内容可能变宽变高)
watch(resolved, () => nextTick(measure))

// 调用方当前的数据, 按 id 查。节点和边渲染时都来这里取外观
const nowNodes = computed(() => new Map(resolved.value.nodes.map((n) => [n.id, n])))
const look = (data) => ({ ...data, ...nowNodes.value.get(data.id) })
const edgeNow = (e) => resolved.value.edges[e.data?.i] || {}

// node.color: 给这个节点单独指定颜色, 覆盖 kind 的配色。
// 配置驱动的图会用到 —— 颜色本身就是信息 (选了哪种零件), 不能一律套 kind。
const tint = (c) => (c ? { borderColor: c, background: `color-mix(in srgb, ${c} 16%, var(--bg-elev))` } : null)

const side = computed(() =>
  props.dir === 'LR' ? [Position.Left, Position.Right] : [Position.Top, Position.Bottom])

const fNodes = ref([])
const fEdges = ref([])

const buildNodes = () => [
  ...frameNodes(rectsOf(laid.value.nodes)),      // zIndex -1, 垫在节点和连线底下
  ...laid.value.nodes.map((n) => ({
    id: n.id, type: 'box', position: { x: n.x, y: n.y },
    targetPosition: side.value[0], sourcePosition: side.value[1],
    data: n,
  })),
]

const buildEdges = () =>
  laid.value.edges.map((e, i) => ({
    id: `e${i}-${e.from}-${e.to}`, source: e.from, target: e.to, label: e.label,
    type: 'poly', animated: e.kind === 'main',
    class: 'dag-edge',
    // kind / active / dim / color 不写在这里: 它们在 #edge-poly 模板里按 i 去取调用方当前的值。
    // edges[].color 和 node.color 一样是个逃生口 —— 某些图里线的颜色本身是信息
    data: { via: e.via, i },
  }))

watch(laid, () => {
  fNodes.value = buildNodes()
  fEdges.value = []                      // 先清空, 等节点有了坐标再接线
  nextTick(() => { fEdges.value = buildEdges(); nextTick(measure) })
}, { immediate: true })
</script>

<style>
/* Vue Flow 的内部类不走 scoped, 统一套上本站的颜色变量 */
.dag { position: relative; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--code-bg); overflow: hidden; }
.dag .vue-flow__node { cursor: grab; }
.dag .vue-flow__node:active { cursor: grabbing; }
.dag .vue-flow__handle { opacity: 0; }
.dag .vue-flow__edge-path { stroke: var(--border-strong); stroke-width: 1.6; }
.dag .vue-flow__edge-path.main { stroke: var(--accent); stroke-width: 2.4; }
.dag .vue-flow__edge-path.weak { stroke: var(--text-dim); stroke-dasharray: 4 3; }
.dag .vue-flow__edge-path.on { stroke: var(--warn); stroke-width: 2.6; }
.dag .vue-flow__edge-path.dim { opacity: 0.25; }
.dag .vue-flow__edge-text { fill: var(--text-dim); font-size: 10px; }
.dag .vue-flow__edge-textbg { fill: var(--code-bg); }
.dag .vue-flow__controls { box-shadow: none; }
.dag .vue-flow__controls-button { background: var(--bg-elev); border-bottom: 1px solid var(--border); fill: var(--text-muted); }
.dag .vue-flow__controls-button:hover { background: var(--bg-card); }

/* 宽度交给内容, 只卡上下限 —— measure() 量到的就是 dagre 下一轮排版用的尺寸 */
.dag-node {
  box-sizing: border-box; min-width: 104px; max-width: 240px; padding: 7px 11px;
  border: 1.5px solid var(--border-strong); border-radius: 7px;
  background: var(--bg-elev); color: var(--text); text-align: left;
}
.dag-node.grouped { cursor: pointer; }
/* 选中态边框加粗 1.5px, padding 同步减 1.5px: 节点外沿尺寸不变, 点亮一个节点不会引发重新排版 */
.dag-node.on { border-width: 3px; padding: 5.5px 9.5px; background: var(--bg-card); }
.dag-node.dim { opacity: 0.4; }
.dag-node:focus-visible { outline: 2px solid var(--warn); outline-offset: 2px; }
.dag-node .n-head { display: flex; align-items: baseline; gap: 6px; }
.dag-node .n-label { font-size: 12.5px; font-family: "SF Mono", Menlo, monospace; }
.dag-node .n-badge { font-size: 9.5px; color: var(--accent); margin-left: auto; }
.dag-node .n-sub { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
.dag-node .n-note { font-size: 10px; color: var(--text-dim); margin-top: 2px; line-height: 1.4; }
.dag-node .n-fold { display: block; font-size: 9px; color: var(--accent); margin-top: 3px; }
/* kind → 颜色: 与全站图例一致 */
.dag-node.trunk { border-color: var(--accent); background: var(--accent-soft); }
.dag-node.norm { border-color: var(--warn); }
.dag-node.attn { border-color: var(--accent); }
.dag-node.ffn, .dag-node.ok { border-color: var(--left); }
.dag-node.io { border-color: var(--border-strong); background: var(--bg-card); }
.dag-node.group { border-style: dashed; border-color: var(--text-dim); }
.dag-node.bad { border-color: var(--danger); }

/* 子图框: 纯背景, 不接事件, 不挡拖画布 */
.dag .vue-flow__node-frame { pointer-events: none; box-sizing: border-box;
  border: 1.5px dashed var(--border-strong); border-radius: 10px;
  background: color-mix(in srgb, var(--text-dim) 6%, transparent); }
.dag-frame-label { display: block; padding: 3px 9px; font-size: 10.5px; color: var(--text-muted); white-space: nowrap; }

.dag-bar { position: absolute; left: 8px; bottom: 8px; display: flex; flex-wrap: wrap; gap: 5px; }
/* 全站 button.active 是 accent 底 + 白字; 这里自己定两态的配色, 不然浅色主题下白字看不见 */
.dag-bar button { min-height: 24px; padding: 2px 8px; font-size: 10.5px;
  background: var(--bg-elev); border-color: var(--border); color: var(--text-muted); }
.dag-bar button.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
</style>
