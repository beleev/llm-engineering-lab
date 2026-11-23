<!--
  模型结构浏览器, 挂载于阶段首页。支持切换代表性架构, 查看组件拓扑连接及四种运行态下各节点的行为。
  随模型变化的「规模读数」全部由 modelArchitectures.js 中的 config 动态计算生成。
-->
<template>
  <section class="architecture-lab" aria-labelledby="architecture-lab-title">
    <header class="lab-header">
      <div>
        <span class="lab-kicker">模型结构浏览器</span>
        <h2 id="architecture-lab-title">模型结构与运行态实验台 — 组件拓扑连接与四种运行态下的状态流转</h2>
        <!-- 列表项里再包一层 <p>: 沿用 .lab-header p 的字号和颜色, 和上下段落一致 -->
        <p>模拟什么: 选择一套代表性模型架构, 观察其核心组件的拓扑连接方式, 以及同一计算图在四种运行态下各节点的计算职责与状态流转。四种运行态分别为:</p>
        <ul class="pts">
          <li><p><b>「结构」:</b> 仅展示组件间的静态拓扑连接与层级嵌套关系。</p></li>
          <li><p><b>「训练态」:</b> 整段 token 序列并行执行前向计算, 随后沿反向路径计算参数梯度。</p></li>
          <li><p><b>「并行前向」(prefill):</b> 推理首阶段将整段 prompt 一次性并行送入模型, 完成前向计算并写入初始 KV cache 等持久状态。</p></li>
          <li><p><b>「迭代推理」(decode):</b> 自回归生成阶段每步仅输入 1 个新 token, 直接复用并更新前序步骤保留的持久状态。</p></li>
        </ul>
        <p>图上是什么: 画布中每个矩形节点代表一个计算组件, 有向箭头表示张量流向, 自上而下对应前向执行顺序; 右下角带 + 号的复合节点支持展开内部子图, 展开后的子组件收纳于同一外框内并以虚线表示内部连接。节点右上角标签指示该组件在当前运行态下的角色 (如「只读 W」「写缓存」)。四种颜色图例含义如下:</p>
        <ul class="pts">
          <li><p><b>权重:</b> 训练阶段需要优化更新、推理阶段保持只读的模型参数。</p></li>
          <li><p><b>激活:</b> 单次前向传播过程中动态生成的中间张量。</p></li>
          <li><p><b>梯度:</b> 反向传播阶段计算出的每个权重的更新方向。</p></li>
          <li><p><b>持久状态:</b> 推理阶段跨时间步保留的状态张量, 在 Transformer 中为 KV cache, 在 Mamba 中为 SSM state。</p></li>
        </ul>
        <p>可以做什么:</p>
        <ul class="pts">
          <li><p>点击上排模型标签切换不同架构。</p></li>
          <li><p>通过「结构 / 训练态 / 并行前向 / 迭代推理」按钮切换运行态。</p></li>
          <li><p>拖拽单个节点调整局部位置, 拖拽画布空白区域平移视图, 使用滚轮缩放画布。</p></li>
          <li><p>点击任意组件节点, 在右侧详情面板查看其权重形状、激活生命周期、核心公式与源码入口。</p></li>
          <li><p>点击「展开全部 / 收起」一键展开或折叠所有复合组件。</p></li>
        </ul>
        <p>应该看到: 顶部「规模读数」一行随所选模型动态变化:</p>
        <ul class="pts">
          <li><p>层数 N 与隐藏维 D 取自该模型构造函数的默认配置。</p></li>
          <li><p>「每层跨步保留的状态」给出推理阶段每新生成 1 个 token 时单层需额外缓存的元素数量 (LLaMA 为 K 与 V 各 32 头 × 128 维, 合计 8192 个元素; Mamba 的状态大小恒定, 不随 token 数量增长; DiT 无跨步保留状态)。</p></li>
          <li><p>最后一格按序列长度 1024、fp16 精度 (每元素占 2 字节) 折算出全模型持久状态的总显存占用。</p></li>
        </ul>
        <p>切换至「迭代推理」模式观察 LLaMA 可见: attention 节点标记为「读 / 追加」, 其余节点均为「临时激活」或「只读 W」, 说明自回归解码阶段唯一需要跨步驻留显存的状态仅有 attention 读写的 KV cache。</p>
      </div>
      <RepoLink :path="currentModel.source" label="当前模型源码" tiny />
    </header>

    <div class="model-tabs" role="tablist" aria-label="选择模型结构">
      <button
        v-for="model in modelArchitectures" :key="model.id" type="button" role="tab"
        :aria-selected="model.id === selectedModelId" :class="{ active: model.id === selectedModelId }"
        @click="selectedModelId = model.id"
      >
        <span>{{ model.name }}</span>
        <small>{{ model.badge }}</small>
      </button>
    </div>

    <div class="model-intro" aria-live="polite" aria-atomic="true">
      <div>
        <span class="model-badge">{{ currentModel.badge }}</span>
        <strong>{{ currentModel.name }}</strong>
      </div>
      <p>{{ currentModel.description }}</p>
    </div>

    <!-- 规模读数: 切换模型时只有这一行的数值随之更新 -->
    <dl class="runtime-stats config-stats" aria-live="polite" aria-label="规模读数, 按构造函数默认配置计算">
      <div v-for="r in readings" :key="r.label" :data-tone="r.tone || 'neutral'">
        <dt>{{ r.label }}</dt>
        <dd>{{ r.value }}</dd>
      </div>
    </dl>

    <div class="runtime-toolbar">
      <div class="runtime-modes" role="group" aria-label="选择模型运行状态">
        <button
          v-for="item in runtimeModes" :key="item.id" type="button"
          :class="{ active: runtimeMode === item.id }" :aria-pressed="runtimeMode === item.id"
          @click="runtimeMode = item.id"
        >
          <span aria-hidden="true">{{ item.icon }}</span>
          {{ item.label }}
        </button>
      </div>
      <div class="runtime-legend" aria-label="运行态颜色图例">
        <span><i class="weight"></i>权重 (可训练参数)</span>
        <span><i class="activation"></i>激活 (前向中间张量)</span>
        <span><i class="gradient"></i>梯度 (反向参数更新方向)</span>
        <span><i class="cache"></i>持久状态 (KV cache / SSM state)</span>
      </div>
    </div>

    <section class="runtime-readout" :data-mode="runtimeMode" aria-live="polite" aria-atomic="true">
      <div class="runtime-copy">
        <span>{{ currentModeMeta.label }} · 当前路径</span>
        <strong>{{ currentRuntime.headline }}</strong>
        <p>{{ currentRuntime.note }}</p>
      </div>
      <dl class="runtime-stats">
        <div v-for="stat in currentRuntime.stats" :key="stat.label" :data-tone="stat.tone || 'neutral'">
          <dt>{{ stat.label }}</dt>
          <dd>{{ stat.value }}</dd>
        </div>
      </dl>
    </section>

    <div class="diagram-shell">
      <div class="canvas-column">
        <div class="canvas-toolbar">
          <div class="canvas-hint">
            <span aria-hidden="true">↔</span>
            拖拽节点调整位置 · 拖拽空白处平移 · 滚轮缩放 · 点击节点在右侧查看详情 · 点击 + 展开子组件
          </div>
          <div class="canvas-actions">
            <button type="button" title="展开全部组件" aria-label="展开全部组件" @click="expandAll">展开全部</button>
            <button type="button" title="收起全部组件" aria-label="收起全部组件" @click="collapseAll">收起</button>
          </div>
        </div>

        <!-- 版式交由 DagView 内的 dagre 自动计算: 展开或收起组件后自动重排, 无需手动指定坐标 -->
        <DagView v-bind="graph" @node-click="selectNode">
          <template #node="{ node }">
            <span class="node-topline">
              <span class="category-label"><i aria-hidden="true" :style="{ background: node.color }"></i>{{ categoryLabel(node.category) }}</span>
              <span class="runtime-chip">{{ node.chip }}</span>
            </span>
            <strong>{{ node.label }}</strong>
            <span class="node-shape mono">{{ node.shape || '结构节点' }}</span>
            <button
              v-if="node.expandable" type="button" class="expand-button" :style="{ background: node.color }"
              :title="node.open ? '收起内部组件' : '展开内部组件'"
              :aria-label="`${node.open ? '收起' : '展开'} ${node.label} 内部组件`"
              @click.stop="toggleExpanded(node.id)"
            >
              <span aria-hidden="true">{{ node.open ? '−' : '+' }}</span>
            </button>
          </template>
        </DagView>

        <div class="viewport-status" aria-hidden="true">
          <span>{{ visibleNodes.length }} 个组件</span>
          <span>{{ currentModeMeta.short }}</span>
        </div>
      </div>

      <aside class="node-inspector" aria-labelledby="node-inspector-title">
        <div class="inspector-heading">
          <span>{{ currentModeMeta.label }} · 组件详情</span>
          <h3 id="node-inspector-title">{{ selectedNode?.label || '选择一个组件' }}</h3>
        </div>

        <template v-if="selectedNode">
          <p class="node-summary">{{ selectedNode.summary }}</p>

          <div class="runtime-callout" :data-mode="runtimeMode">
            <span>当前运行态行为</span>
            <p>{{ nodeRuntimeDescription(selectedNode) }}</p>
          </div>

          <dl class="node-facts">
            <div><dt>组件类型</dt><dd>{{ categoryLabel(selectedNode.category) }}</dd></div>
            <div><dt>张量形状</dt><dd class="mono">{{ selectedNode.shape || '随上游保持不变' }}</dd></div>
          </dl>

          <section v-if="selectedNode.weights?.length" class="inspector-section">
            <h4><i class="legend-dot weight"></i>权重参数</h4>
            <div v-for="weight in selectedNode.weights" :key="weight.name" class="tensor-row">
              <div>
                <strong class="mono">{{ weight.name }}</strong>
                <span class="mono">{{ weight.shape }}</span>
              </div>
              <p>{{ weight.note }}</p>
            </div>
          </section>

          <section v-if="selectedNode.activations" class="inspector-section">
            <h4><i class="legend-dot activation"></i>激活生命周期</h4>
            <p>{{ selectedNode.activations }}</p>
          </section>

          <section v-if="selectedNode.formula || selectedNode.detail" class="inspector-section">
            <h4>内部机制</h4>
            <p class="formula"><Tex :text="selectedNode.formula || selectedNode.detail" /></p>
          </section>

          <div v-if="selectedNode.source" class="source-row">
            <span>对应源码</span>
            <RepoLink :path="selectedNode.source" tiny />
          </div>
        </template>

        <p v-else class="inspector-empty">点击画布中的任意组件节点, 此处将展示其权重形状、激活生命周期、当前运行态下的具体行为及对应源码入口。</p>
      </aside>
    </div>
  </section>
</template>

<script setup>
import Tex from '@/components/Tex.vue'
import { computed, ref, watch } from 'vue'
import RepoLink from '@/components/RepoLink.vue'
import DagView from '@/components/dag/DagView.vue'
import { architectureById, categories, modelArchitectures } from '@/data/modelArchitectures.js'
import { fmtBytes } from '@/utils/labmath.js'

const runtimeModes = [
  { id: 'structure', label: '结构', short: '静态拓扑', icon: '◇' },
  { id: 'training', label: '训练态', short: '前向 + 反向', icon: '↔' },
  { id: 'prefill', label: '并行前向', short: '全序列并行', icon: '▦' },
  { id: 'decode', label: '迭代推理', short: '逐 token 解码', icon: '▷' },
]

const selectedModelId = ref('llama')
const runtimeMode = ref('structure')
const selectedNodeId = ref('blocks')
const expanded = ref(new Set())

// 规模读数。★ 跨步保留的状态 = 层数 × (每 token 的元素数 × T + 与 T 无关的固定元素数), fp16 每元素占 2 字节
const T_REF = 1024
const readings = computed(() => {
  const c = currentModel.value.config
  const kept = c.perToken ? `每 token ${c.perToken} 个元素` : c.fixed ? `固定 ${c.fixed} 个元素` : '不保留'
  const bytes = c.layers * (c.perToken * T_REF + c.fixed) * 2
  return [
    { label: '层数 N (默认配置)', value: c.layers },
    { label: '隐藏维 D', value: c.d, tone: 'activation' },
    { label: '每层跨步保留的状态', value: kept, tone: 'cache' },
    { label: `T = ${T_REF} 时全模型状态 (fp16)`, value: bytes ? fmtBytes(bytes) : '0', tone: 'cache' },
  ]
})

const currentModel = computed(() => architectureById[selectedModelId.value] || modelArchitectures[0])
const currentModeMeta = computed(() => runtimeModes.find((item) => item.id === runtimeMode.value))
const currentRuntime = computed(() => currentModel.value.runtime[runtimeMode.value])
const nodeMap = computed(() => new Map(currentModel.value.nodes.map((item) => [item.id, item])))

const hasChildren = (id) => currentModel.value.nodes.some((item) => item.parent === id)

const isNodeVisible = (item) => {
  let parentId = item.parent
  const seen = new Set()
  while (parentId) {
    if (seen.has(parentId) || !expanded.value.has(parentId)) return false
    seen.add(parentId)
    parentId = nodeMap.value.get(parentId)?.parent
  }
  return true
}

const visibleNodes = computed(() => currentModel.value.nodes.filter(isNodeVisible))
const visibleNodeIds = computed(() => new Set(visibleNodes.value.map((item) => item.id)))
const visibleEdges = computed(() => currentModel.value.edges.filter(
  (item) => visibleNodeIds.value.has(item.from) && visibleNodeIds.value.has(item.to),
))
const selectedNode = computed(() => nodeMap.value.get(selectedNodeId.value) || visibleNodes.value[0] || null)

// 传给 DagView 的子图声明: 仅声明当前可见节点与连线, 节点坐标由布局算法自动计算。
// 展开复合组件新增子节点后版式会自动重排, 无需在静态数据中预留坐标空间。
const graph = computed(() => ({
  dir: 'TB',
  nodes: visibleNodes.value.map((n) => ({
    id: n.id,
    label: n.label,
    color: categories[n.category]?.color,
    shape: n.shape,
    category: n.category,
    chip: nodeRuntimeChip(n),
    expandable: hasChildren(n.id),
    open: expanded.value.has(n.id),
    active: selectedNodeId.value === n.id,
    aria: `${categoryLabel(n.category)} ${n.label}`,
    box: n.parent,
  })),
  // 展开的复合组件将其内部子组件收纳于同一外框中以区分主干, 支持多层嵌套
  boxes: visibleNodes.value
    .filter((n) => expanded.value.has(n.id) && hasChildren(n.id))
    .map((n) => ({ id: n.id, label: `${n.label} · 内部`, parent: n.parent })),
  edges: visibleEdges.value.map((e) => ({
    from: e.from,
    to: e.to,
    label: e.label,
    // detail 边代表展开后可见的内部连接, 用虚线绘制; 其余为常规连接。
    kind: e.detail ? 'weak' : 'side',
  })),
}))

const categoryLabel = (category) => categories[category]?.label || '组件'

const nodeRuntimeChip = (item) => {
  if (runtimeMode.value === 'structure') return hasChildren(item.id) ? '可展开' : '组件'
  if (runtimeMode.value === 'training') {
    if (item.category === 'result') return 'loss 起点'
    if (item.weights?.length) return 'W + dW'
    if (item.category === 'state') return '保存状态'
    return '保存激活'
  }
  if (runtimeMode.value === 'prefill') {
    if (item.category === 'attention') return '写缓存'
    if (item.category === 'state') return '建立状态'
    if (item.weights?.length) return '只读 W'
    return '全序列'
  }
  if (item.category === 'attention') return '读 / 追加'
  if (item.category === 'state') return '跨步保留'
  if (item.category === 'input') return '单步输入'
  if (item.weights?.length) return '只读 W'
  return '临时激活'
}

const genericRuntimeDescriptions = {
  structure: (item) => item.detail || `该节点在模型主干中属于「${categoryLabel(item.category)}」组件。`,
  training: (item) => item.weights?.length
    ? '前向传播读取参数并计算输出激活; 反向传播结合上游梯度计算参数梯度 dW, 交由优化器更新权重。'
    : '前向计算结果需保留至 backward 阶段 (或由 activation checkpoint 重算), 梯度沿反向路径流经该节点。',
  prefill: (item) => item.category === 'attention' || item.category === 'state'
    ? '对整段输入序列执行并行前向计算, 并将生成的持久状态写入缓存供后续解码步骤读取。'
    : '权重保持只读, 整段输入序列并行通过该节点; 临时激活在下游节点使用完毕后即可释放。',
  decode: (item) => item.category === 'attention' || item.category === 'state'
    ? '读取前序步骤保留的持久状态, 结合当前单步输入完成计算, 并将新增状态追加或覆盖写入缓存。'
    : '仅对当前单步的激活张量执行前向计算; 模型权重在每一步重复读取, 中间张量不保留至下一步。',
}

const nodeRuntimeDescription = (item) => (
  item.runtime?.[runtimeMode.value]
  || genericRuntimeDescriptions[runtimeMode.value](item)
)

const selectNode = (id) => { selectedNodeId.value = id }

const toggleExpanded = (id) => {
  const next = new Set(expanded.value)
  if (next.has(id)) {
    next.delete(id)
    let parentId = selectedNode.value?.parent
    while (parentId) {
      if (parentId === id) {
        selectedNodeId.value = id
        break
      }
      parentId = nodeMap.value.get(parentId)?.parent
    }
  } else next.add(id)
  expanded.value = next
}

const expandAll = () => {
  expanded.value = new Set(currentModel.value.nodes.filter((item) => hasChildren(item.id)).map((item) => item.id))
}
const collapseAll = () => {
  expanded.value = new Set()
  if (selectedNode.value?.parent) selectedNodeId.value = currentModel.value.nodes.find((item) => !item.parent)?.id || ''
}

watch(selectedModelId, () => {
  expanded.value = new Set(currentModel.value.defaultExpanded || [])
  selectedNodeId.value = currentModel.value.defaultExpanded?.[0]
    || currentModel.value.nodes.find((item) => !item.parent)?.id
    || ''
}, { immediate: true })
</script>

<style scoped>
.architecture-lab { --runtime-weight: var(--warn); --runtime-activation: var(--code-fn); --runtime-gradient: var(--right); --runtime-cache: var(--left); margin-top: 24px; border: 1px solid var(--border-strong); border-radius: calc(var(--radius) + 4px); background: var(--bg-card); overflow: hidden; box-shadow: 0 18px 52px color-mix(in srgb, var(--text) 8%, transparent); }
.lab-header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; padding: 22px 24px 18px; border-bottom: 1px solid var(--border); background: linear-gradient(110deg, color-mix(in srgb, var(--accent) 8%, var(--bg-elev)), var(--bg-card) 62%); }
.lab-kicker, .inspector-heading > span, .runtime-copy > span { color: var(--accent); font-family: "SF Mono", Menlo, monospace; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
.lab-header h2 { margin-top: 3px; font-size: 20px; text-wrap: balance; }
.lab-header p { margin-top: 5px; max-width: 780px; color: var(--text-muted); font-size: 12.5px; line-height: 1.7; text-wrap: pretty; }
.model-tabs { display: flex; gap: 1px; padding: 8px; overflow-x: auto; border-bottom: 1px solid var(--border); background: var(--bg-elev); }
.model-tabs button { flex: 1 0 128px; display: flex; flex-direction: column; align-items: flex-start; min-height: 52px; padding: 8px 11px; border-color: transparent; background: transparent; text-align: left; }
.model-tabs button:hover { background: var(--bg-card); }
.model-tabs button.active { border-color: var(--border-strong); background: var(--bg-card); color: var(--text); box-shadow: 0 3px 10px color-mix(in srgb, var(--text) 6%, transparent); }
.model-tabs button span { font-size: 12px; font-weight: 650; }
.model-tabs button small { margin-top: 1px; color: var(--text-muted); font-size: 10px; }
.model-tabs button.active small { color: var(--accent); }
.model-intro { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 18px; align-items: center; min-height: 66px; padding: 12px 24px; border-bottom: 1px solid var(--border); }
.model-intro > div { display: flex; flex-direction: column; align-items: flex-start; }
.model-intro strong { margin-top: 2px; font-size: 16px; }
.model-badge { color: var(--accent); font-size: 10px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; }
.model-intro p { color: var(--text-muted); font-size: 12.5px; line-height: 1.65; text-wrap: pretty; }
.runtime-toolbar { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--code-bg); }
.runtime-modes { display: flex; flex-wrap: wrap; gap: 6px; }
.runtime-modes button { min-height: 40px; padding: 6px 12px; font-size: 11.5px; }
.runtime-modes button span { margin-right: 4px; font-family: "SF Mono", Menlo, monospace; }
.runtime-legend { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px 12px; color: var(--text-muted); font-size: 10px; }
.runtime-legend span { display: inline-flex; align-items: center; gap: 5px; }
.runtime-legend i,
.legend-dot { width: 8px; height: 8px; border-radius: 50%; }
.weight { background: var(--runtime-weight); }
.activation { background: var(--runtime-activation); }
.gradient { background: var(--runtime-gradient); }
.cache { background: var(--runtime-cache); }
.runtime-readout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 0.9fr); gap: 20px; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--accent) 4%, var(--bg-card)); }
.runtime-readout[data-mode="training"] { background: color-mix(in srgb, var(--runtime-gradient) 6%, var(--bg-card)); }
.runtime-readout[data-mode="prefill"] { background: color-mix(in srgb, var(--runtime-activation) 6%, var(--bg-card)); }
.runtime-readout[data-mode="decode"] { background: color-mix(in srgb, var(--runtime-cache) 6%, var(--bg-card)); }
.runtime-copy strong { display: block; margin-top: 3px; font-size: 13.5px; text-wrap: balance; }
.runtime-copy p { margin-top: 3px; color: var(--text-muted); font-size: 11.5px; line-height: 1.65; text-wrap: pretty; }
.runtime-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }
/* 规模读数: 四格一行, 窄屏折成两格 */
.config-stats { grid-template-columns: repeat(4, minmax(0, 1fr)); padding: 10px 20px; border-bottom: 1px solid var(--border); }
.runtime-stats > div { min-width: 0; padding: 8px 10px; border-left: 2px solid var(--border-strong); background: var(--bg-elev); }
.runtime-stats > div[data-tone="weight"] { border-left-color: var(--runtime-weight); }
.runtime-stats > div[data-tone="activation"] { border-left-color: var(--runtime-activation); }
.runtime-stats > div[data-tone="gradient"] { border-left-color: var(--runtime-gradient); }
.runtime-stats > div[data-tone="cache"] { border-left-color: var(--runtime-cache); }
.runtime-stats dt { color: var(--text-dim); font-size: 10px; letter-spacing: 0.5px; text-transform: uppercase; }
.runtime-stats dd { margin-top: 2px; overflow-wrap: anywhere; color: var(--text); font-family: "SF Mono", Menlo, monospace; font-size: 10.5px; font-variant-numeric: tabular-nums; }
.diagram-shell { display: grid; grid-template-columns: minmax(0, 1fr) 330px; min-height: 650px; }
.canvas-column { min-width: 0; border-right: 1px solid var(--border); }
.canvas-toolbar { display: flex; justify-content: space-between; gap: 12px; align-items: center; min-height: 50px; padding: 6px 10px 6px 14px; border-bottom: 1px solid var(--border); }
.canvas-hint { color: var(--text-muted); font-size: 10.5px; }
.canvas-hint span { margin-right: 6px; color: var(--accent); }
.canvas-actions { display: flex; align-items: center; gap: 5px; }
.canvas-actions button { min-height: 38px; padding: 5px 9px; font-size: 10.5px; }

/* 节点内容渲染在 DagView 的 #node 插槽里 —— 外框、定位、连线都归 DagView 管,
   这里只管插槽内部那几行的排版和分类配色。 */
.node-topline { display: flex; justify-content: space-between; gap: 5px; align-items: center; }
.category-label { display: inline-flex; align-items: center; min-width: 0; color: var(--text-muted); font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
.category-label i { flex: 0 0 auto; width: 6px; height: 6px; margin-right: 4px; border-radius: 50%; background: currentColor; }
.runtime-chip { max-width: 84px; overflow: hidden; color: var(--text-dim); font-family: "SF Mono", Menlo, monospace; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
strong { display: block; margin-top: 6px; color: var(--text); font-size: 12.5px; font-weight: 680; line-height: 1.25; }
.node-shape { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; line-height: 1.3; }
.expand-button { position: absolute; right: -9px; bottom: -9px; width: 26px; min-height: 26px; padding: 0; border: 2px solid var(--bg); border-radius: 50%; background: var(--accent); color: var(--bg); font-family: "SF Mono", Menlo, monospace; font-size: 14px; line-height: 1; box-shadow: 0 4px 10px color-mix(in srgb, var(--text) 15%, transparent); }
.expand-button:hover { filter: brightness(1.08); }
/* 排在图下面: DagView 左下角有自己的折叠条, 浮在画布上会和它重叠 */
.viewport-status { display: flex; gap: 7px; padding: 8px 12px 10px; pointer-events: none; }
.viewport-status span { padding: 3px 7px; border: 1px solid var(--border); border-radius: 4px; background: color-mix(in srgb, var(--bg-elev) 88%, transparent); color: var(--text-muted); font-family: "SF Mono", Menlo, monospace; font-size: 10px; }
.node-inspector { min-width: 0; padding: 18px; background: var(--bg-elev); }
.inspector-heading { padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.inspector-heading h3 { margin-top: 4px; font-size: 17px; line-height: 1.35; text-wrap: balance; }
.node-summary { margin-top: 13px; color: var(--text-muted); font-size: 12px; line-height: 1.7; text-wrap: pretty; }
.runtime-callout { margin-top: 14px; padding: 10px 12px; border-left: 3px solid var(--accent); background: var(--bg-card); }
.runtime-callout[data-mode="training"] { border-left-color: var(--runtime-gradient); }
.runtime-callout[data-mode="prefill"] { border-left-color: var(--runtime-activation); }
.runtime-callout[data-mode="decode"] { border-left-color: var(--runtime-cache); }
.runtime-callout span { color: var(--text-dim); font-size: 10px; font-weight: 700; letter-spacing: 0.7px; }
.runtime-callout p { margin-top: 3px; color: var(--text); font-size: 11px; line-height: 1.65; text-wrap: pretty; }
.node-facts { display: grid; grid-template-columns: 1fr; gap: 1px; margin-top: 14px; border: 1px solid var(--border); background: var(--border); }
.node-facts > div { padding: 8px 10px; background: var(--bg-card); }
.node-facts dt { color: var(--text-dim); font-size: 10px; letter-spacing: 0.5px; text-transform: uppercase; }
.node-facts dd { margin-top: 2px; overflow-wrap: anywhere; color: var(--text); font-size: 10.5px; }
.inspector-section { margin-top: 16px; }
.inspector-section h4 { display: flex; align-items: center; gap: 6px; margin-bottom: 7px; color: var(--text); font-size: 11px; }
.tensor-row { padding: 8px 0; border-top: 1px solid var(--border); }
.tensor-row > div { display: flex; justify-content: space-between; gap: 10px; }
.tensor-row strong { min-width: 0; overflow-wrap: anywhere; color: var(--text); font-size: 10px; }
.tensor-row span { flex: 0 0 auto; color: var(--runtime-weight); font-size: 10px; }
.tensor-row p,
.inspector-section > p { margin-top: 3px; color: var(--text-muted); font-size: 10.5px; line-height: 1.6; text-wrap: pretty; }
.inspector-section .formula { margin: 0; overflow-x: auto; padding: 9px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--code-bg); font-size: 12px; line-height: 1.6; }
.source-row { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; align-items: center; margin-top: 18px; padding-top: 13px; border-top: 1px solid var(--border); }
.source-row > span { color: var(--text-dim); font-size: 10px; letter-spacing: 0.5px; }
.inspector-empty { margin-top: 16px; color: var(--text-muted); font-size: 12px; line-height: 1.7; }

@media (max-width: 1080px) {
  .runtime-readout { grid-template-columns: 1fr; }
  .diagram-shell { grid-template-columns: 1fr; }
  .canvas-column { border-right: 0; border-bottom: 1px solid var(--border); }
  .node-inspector { display: grid; grid-template-columns: minmax(180px, 0.7fr) minmax(260px, 1fr); gap: 0 24px; }
  .inspector-heading,
  .node-summary,
  .runtime-callout { grid-column: 1; }
  .node-facts,
  .inspector-section,
  .source-row { grid-column: 2; }
  .node-facts { grid-row: 1 / span 2; margin-top: 0; }
}

@media (max-width: 720px) {
  .lab-header { flex-direction: column; padding: 18px; }
  .model-intro { grid-template-columns: 1fr; gap: 5px; padding: 12px 18px; }
  .runtime-toolbar { align-items: flex-start; flex-direction: column; }
  .runtime-legend { justify-content: flex-start; }
  .runtime-readout { padding: 14px 16px; }
  .runtime-stats { grid-template-columns: 1fr; }
  .config-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 10px 16px; }
  .canvas-toolbar { align-items: flex-start; flex-direction: column; padding: 9px 10px; }
  .canvas-actions { width: 100%; overflow-x: auto; }
  .canvas-hint { padding-left: 3px; }
  .node-inspector { display: block; padding: 16px; }
  .node-facts { margin-top: 14px; }
}
</style>
