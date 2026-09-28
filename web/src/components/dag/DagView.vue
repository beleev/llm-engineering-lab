<!--
  全站统一的 DAG 视图。所有数据流 / 架构图 / 流程图都用它, 不要再手写 SVG 坐标。
  坐标由 dagre 算, 所以改数据、换方向、折叠展开都不会把版式挤乱。
  可以拖节点、拖画布、滚轮缩放; 标了 group 的节点可以折叠成一个, 展开时画一个框圈住它们。

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
      :min-zoom="0.25" :max-zoom="2" :zoom-on-double-click="false"
      fit-view-on-init @nodes-initialized="measure" @node-drag="syncFrames"
    >
      <template #node-box="{ data }">
        <Handle type="target" :position="side[0]" />
        <Handle type="source" :position="side[1]" />
        <div :class="['dag-node', data.kind, { grouped: data.groupId, on: data.active, dim: data.dim }]"
             :style="tint(data.color)" :title="data.note || ''" tabindex="0" :aria-label="data.aria || data.label"
             @mouseenter="emit('node-hover', data.id)" @mouseleave="emit('node-hover', null)"
             @focus="emit('node-hover', data.id)" @blur="emit('node-hover', null)"
             @click="hit(data)" @keydown.enter="hit(data)">
          <slot name="node" :node="data">
            <div class="n-head">
              <span class="n-label">{{ data.label }}</span>
              <span v-if="data.badge" class="n-badge mono">{{ data.badge }}</span>
            </div>
            <div v-if="data.sub" class="n-sub mono">{{ data.sub }}</div>
            <div v-if="data.note" class="n-note">{{ data.note }}</div>
          </slot>
          <span v-if="data.groupId" class="n-fold mono">{{ data.folded ? '展开 ▸' : '折叠 ▾' }}</span>
        </div>
      </template>
      <template #node-frame="{ data }">
        <span class="dag-frame-label">{{ data.label }}</span>
      </template>
      <!-- 边走 dagre 算出来的折线, 长边会绕开中间的节点; 端点用 Vue Flow 的实时坐标, 拖节点时线跟着走 -->
      <template #edge-poly="e">
        <BaseEdge :id="e.id" :path="pathOf(e)" :label="e.label" :style="e.style"
                  :label-x="labelAt(e).x" :label-y="labelAt(e).y" label-show-bg />
      </template>
      <Controls position="top-right" :show-interactive="false" />
    </VueFlow>

    <div v-if="foldable.length" class="dag-bar">
      <button v-for="g in foldable" :key="g.id" type="button"
              :class="{ active: !folded.has(g.id) }" @click="toggle(g.id)">
        {{ folded.has(g.id) ? '▸' : '▾' }} {{ g.label }}
      </button>
      <button type="button" @click="relayout">重排</button>
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
  else fit()
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

  const seen = new Set()
  const edges = []
  for (const e of props.edges) {
    const from = hidden.get(e.from) || e.from
    const to = hidden.get(e.to) || e.to
    if (from === to) continue                     // 组内部的边折叠后不画
    const key = `${from}->${to}`
    if (seen.has(key)) continue
    seen.add(key)
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
  // compound: 同一个框里的节点排在一起, 框外的节点不会插进来
  const g = new dagre.graphlib.Graph({ compound: true })
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
  for (const e of edges) if (g.hasNode(e.from) && g.hasNode(e.to)) g.setEdge(e.from, e.to)
  dagre.layout(g)
  laid.value = {
    nodes: nodes.map((n) => {
      const p = g.node(n.id)
      return { ...n, x: p.x - p.width / 2, y: p.y - p.height / 2, w: p.width, h: p.height }
    }),
    // 掐掉首尾两点 —— 它们落在节点边框上, 端点改用 Vue Flow 的实时 handle 坐标
    edges: edges.map((e) => ({ ...e, via: trimStraight((g.edge(e.from, e.to)?.points || []).slice(1, -1)) })),
    boxes,
  }
}

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
watch(resolved, relayout, { immediate: true, deep: true })
watch(() => props.dir, relayout)
watch(measured, relayout)

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
    class: ['dag-edge', e.kind || 'side', e.active && 'on', e.dim && 'dim'].filter(Boolean).join(' '),
    // edges[].color: 和 node.color 一样是个逃生口 —— 某些图里线的颜色本身是信息
    style: e.color ? { stroke: e.color } : undefined,
    data: { via: e.via },
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
.dag .dag-edge.main .vue-flow__edge-path { stroke: var(--accent); stroke-width: 2.4; }
.dag .dag-edge.weak .vue-flow__edge-path { stroke: var(--text-dim); stroke-dasharray: 4 3; }
.dag .dag-edge.on .vue-flow__edge-path { stroke: var(--warn); stroke-width: 2.6; }
.dag .dag-edge.dim .vue-flow__edge-path { opacity: 0.25; }
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
.dag-node.on { border-width: 3px; background: var(--bg-card); }
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
