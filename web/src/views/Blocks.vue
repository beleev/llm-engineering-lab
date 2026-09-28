<template>
  <div>
    <h1 class="page-title">Block 组装器</h1>
    <div class="page-subtitle lead-group">
      <p>这个库最省事的一点: <strong>模型之间的差别基本都在 "构造参数" 上, 很少需要新写一个类</strong>。</p>
      <p>
        下面的零件菜单每栏挑一个, 看看拼出来的是哪个模型。骨架就一份:
        <RepoLink path="llm_models/layers/core/blocks.py:PreLNBlock" label="llm_models/layers/core/blocks.py::PreLNBlock(attn, ffn, norm_cls)" tiny />。
      </p>
    </div>

    <ChapterIntro
      tldr="LLaMA、Mixtral、DeepSeek-V3 不是三个新模型, 是同一份 PreLNBlock 换了 attn / ffn / norm / pos 四个参数实例化出来的。"
      question="模型之间就差 4 个构造参数的话, 「读 N 个模型的源码」是不是变成了「读 N 张零件配置表」?"
      :goals="[
        '把任意一个模型翻译成 PreLNBlock 的四个零件配置',
        '说清 Pre-LN 为什么让深层模型不靠精细 warmup 也能开训',
        '在 LLaMA / Mixtral / DeepSeek 的代码里一眼找到零件被换掉的那几行',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/blocks.py', label: 'blocks.py · PreLNBlock' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <div class="grid assembler">
      <!-- 零件选择 -->
      <div class="card">
        <h3>零件库 <span class="tag">4 个槽位</span></h3>
        <div v-for="slot in slots" :key="slot.key" class="slot-section">
          <div class="slot-title">{{ slot.label }}</div>
          <div class="slot-options">
            <button v-for="opt in slot.options" :key="opt.id" type="button"
                    :class="{ active: config[slot.key] === opt.id }"
                    :aria-pressed="config[slot.key] === opt.id"
                    @click="config[slot.key] = opt.id">
              {{ opt.name }}
            </button>
          </div>
        </div>
      </div>

      <!-- 中: Block 数据流可视化 -->
      <div class="card">
        <h3>数据流 · {{ isPostLn ? 'Post-LN' : 'Pre-LN' }} Block <span class="tag">{{ modelMatch.name }}</span></h3>
        <p class="desc">高亮那条是残差流。</p>
        <ul class="desc pts">
          <li>
            <b>Pre-LN:</b> 残差流从块的入口一直贯到出口, 上面只有加法。反传时梯度有一条恒等通路直接到底 (Xiong et al., 2020)。
          </li>
          <li>
            <b>分支:</b> attention 和 FFN 都不在主干上, 从旁边接出去算完再加回来; norm 只归一化送进分支的副本。
          </li>
          <li>
            <b>Post-LN:</b> 把 norm 槽位换过去, 接线跟着变: norm 压回主干, 梯度每层被缩放一次。
            几十层堆起来, 就得靠精细的 warmup 才敢开训。
          </li>
        </ul>
        <p class="desc" style="margin: 6px 0 16px;">图可以拖, 子层可以折叠。</p>

        <DagView v-bind="blockDag" />
      </div>

      <!-- 右: 匹配哪个模型? -->
      <div>
        <div class="card match-card" :style="{ borderColor: modelMatch.color }">
          <h3>
            <span :class="['pill', modelMatch.track ? tracks[modelMatch.track].cls : '']">{{ modelMatch.year || '?' }}</span>
            <span>匹配模型</span>
          </h3>
          <div class="match-name" :style="{ color: modelMatch.color }">{{ modelMatch.name }}</div>
          <p class="desc"><Tex :text="modelMatch.blurb" /></p>
          <p v-if="modelMatch.repoNote" class="desc repo-note"><b>本仓库实现:</b> {{ modelMatch.repoNote }}</p>
          <pre class="code" style="margin-top: 12px;">{{ generatedCode }}</pre>
        </div>

        <div class="card" style="margin-top: 16px;">
          <h3>换零件之后, 参数量和 cache 变成多少</h3>
          <div class="stat" style="margin-top: 4px;">
            <div class="k">FFN 参数</div>
            <div class="v">{{ ffnParams }}</div>
            <div class="hint"><Tex :text="ffnNote" /></div>
          </div>
          <div class="stat" style="margin-top: 8px;">
            <div class="k">KV 每层 / token</div>
            <div class="v">{{ kvNote }}</div>
            <div class="hint">d_model=4096, n_heads=32 下</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速切换到真实模型 -->
    <section class="section">
      <h2>一键载入预设</h2>
      <div class="lead-group">
        <p>
          点任意一个模型, 它的零件配置会自动填进上面的菜单。
          按年份点一遍 Transformer → GPT-3 → LLaMA → Mixtral → DeepSeek-V3, 你会看到四个槽位是一个一个被换掉的:
        </p>
        <ul class="pts">
          <li><b>norm:</b> 从 Post-LN 挪到 Pre-LN, 再瘦成 RMSNorm。</li>
          <li><b>ffn:</b> 从 ReLU 换 GELU, 再换门控的 SwiGLU。</li>
          <li><b>attn:</b> 从 MHA 砍到 GQA, 再压成 MLA。</li>
          <li><b>pos:</b> 从正余弦换成 RoPE。</li>
        </ul>
        <p>
          预设填的是本仓库里这个模型的搭法。BERT 和 GPT-3 各有一处和原版不同, 按钮上标了「本仓库实现」, 载入后「匹配模型」卡片写明差在哪。
        </p>
      </div>
      <div class="btn-group">
        <button v-for="preset in presets" :key="preset.name" type="button"
                @click="loadPreset(preset)">
          {{ preset.name }} <span style="color: var(--text-muted); font-size: 10px; margin-left: 4px;">{{ preset.year }}<template v-if="preset.repoNote"> · 本仓库实现</template></span>
        </button>
      </div>
    </section>

    <!-- ↓↓↓ 矩阵变换详情 ↓↓↓ -->
    <section class="section">
      <h2>内部矩阵变换</h2>
      <div class="lead-group">
        <p>
          把选中的零件拆开看计算流和权重矩阵。每一步都标了张量的符号形状, 以及按当前参数算出来的实际数字。
          拖形状滑条, 所有数字同步刷新。
        </p>
        <p>
          源码:
          <RepoLink path="llm_models/layers/core/{attention,feedforward,normalization,position_encoding}.py" label="llm_models/layers/core/{attention,feedforward,normalization,position_encoding}.py" tiny />。
        </p>
      </div>
      <InspectorPanel :config="config" :tab="inspectorTab" @update:tab="inspectorTab = $event" />
    </section>

    <!-- 零件都拼对了, 模型也不一定训得起来: 初始化是另一道坎 -->
    <section class="section">
      <h2>零件拼对了, 还有一道坎: 初始化</h2>
      <div class="lead-group">
        <p>
          一个刚建好、什么都没学的模型应该在词表上均匀瞎猜, 第一步的交叉熵就该是 <Tex text="$\ln V$" />。
        </p>
        <p>
          典型的坑: LLaMA 把 lm_head 和输入嵌入绑权重, 嵌入又用了 <span class="mono">nn.Embedding</span> 默认的
          <Tex text="$\mathcal{N}(0, 1)$" />。
        </p>
        <p>绑权重之后, 每个 token 都会给「它自己」打一个约 <Tex text="$\sigma\cdot d$" /> 的高分, 而正确答案几乎从来不是它自己。</p>
        <ul class="pts">
          <li><b><Tex text="$\sigma = 1$" />:</b> 首步 CE 是 <strong class="mono">258</strong>, 而 <Tex text="$\ln 1000 = 6.91$" />。</li>
          <li><b><Tex text="$\sigma = 0.02$" />:</b> 改法只有这一行, CE 回到 <strong class="mono">7.09</strong>。</li>
        </ul>
        <p>
          这类 bug 骗人的地方在于 loss 照样在降。它只是在纠正初始化, 不代表模型学会了任务。下面的实验台把 <Tex text="$\sigma$" /> 交到你手上。
        </p>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '本章 pos 槽位的所有候选项的来历' }"
      :next="{ ...nextChapter, hint: '把 ffn 槽位拆成多专家 — Mixtral / DeepSeek 的两条路' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, reactive, computed } from 'vue'
import { tracks, learningPath } from '@/data/models.js'
import InspectorPanel from '@/components/InspectorPanel.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import DagView from '@/components/dag/DagView.vue'
import Tex from '@/components/Tex.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'blocks')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

const inspectorTab = ref('attn')

const slots = [
  { key: 'attn', label: '注意力 (attn)', options: [
    { id: 'mha', name: 'MHA',  note: '原始, cache 最大' },
    { id: 'gqa', name: 'GQA',  note: 'KV 头 ÷ 4' },
    { id: 'mla', name: 'MLA',  note: 'latent 低秩' },
    { id: 'dsa', name: 'DSA',  note: 'MLA + 稀疏 top-k' },
    { id: 'ssm', name: 'SSM',  note: '线性 O(T), Mamba' },
  ]},
  { key: 'ffn', label: '前馈 (ffn)', options: [
    { id: 'relu',   name: 'ReLU',    note: '原始 Transformer' },
    { id: 'gelu',   name: 'GELU',    note: 'BERT / GPT 风' },
    { id: 'swiglu', name: 'SwiGLU',  note: '门控, LLaMA/Qwen' },
    { id: 'moe_mx', name: 'Mixtral-MoE', note: 'softmax top-2' },
    { id: 'moe_ds', name: 'DeepSeek-MoE', note: 'sigmoid + 共享' },
  ]},
  { key: 'norm', label: '归一化 (norm)', options: [
    { id: 'post_ln', name: 'Post-LN', note: '原始, 需 warmup' },
    { id: 'pre_ln',  name: 'Pre-LN LayerNorm', note: 'GPT-3 风' },
    { id: 'pre_rms', name: 'Pre-LN RMSNorm',   note: 'LLaMA/DeepSeek 风' },
    { id: 'ada_ln',  name: 'adaLN-Zero',       note: 'DiT 扩散' },
  ]},
  { key: 'pos', label: '位置编码 (pos)', options: [
    { id: 'sin',   name: 'Sinusoidal' },
    { id: 'learn', name: 'Learnable' },
    { id: 'rope',  name: 'RoPE' },
    { id: 'mrope', name: 'M-RoPE (三轴)' },
  ]},
]

const config = reactive({
  attn: 'gqa',
  ffn: 'swiglu',
  norm: 'pre_rms',
  pos: 'rope',
})

// 颜色只用主题变量: 这些值既当边框也当文字色, 写死色值在浅色主题下看不清
const C = {
  old: 'var(--text-muted)', a: 'var(--accent)', b: 'var(--left)',
  c: 'var(--eye)', d: 'var(--right)', e: 'var(--warn)',
}
const partColorMap = {
  // attn
  mha: C.old, gqa: C.a, mla: C.b, dsa: C.d, ssm: C.e,
  // ffn
  relu: C.old, gelu: C.a, swiglu: C.b,
  moe_mx: C.c, moe_ds: C.d,
  // norm
  post_ln: C.old, pre_ln: C.a, pre_rms: C.b, ada_ln: C.d,
  // pos
  sin: C.old, learn: C.old, rope: C.b, mrope: C.c,
}
const partColor = (k) => partColorMap[config[k]] || C.old

const findOpt = (slot, id) => slots.find(s => s.key === slot).options.find(o => o.id === id)
const partName = (slot) => findOpt(slot, config[slot])?.name
const partNote = (slot) => findOpt(slot, config[slot])?.note || ''

// --- 模型匹配 ---
const presets = [
  { name: 'Transformer', year: 2017, config: { attn: 'mha', ffn: 'relu',   norm: 'post_ln', pos: 'sin'   } },
  // 这两个预设跟着本仓库的实现走, 和原版各差一处, repoNote 会显示在界面上
  { name: 'BERT',        year: 2018, config: { attn: 'mha', ffn: 'gelu',   norm: 'pre_ln',  pos: 'learn' },
    repoNote: 'norm 填的是 Pre-LN, 因为本仓库的 BERT 用 PreLNBlock 搭。原版 BERT 是 Post-LN。' },
  { name: 'GPT-3',       year: 2020, config: { attn: 'mha', ffn: 'gelu',   norm: 'pre_ln',  pos: 'sin'   },
    repoNote: 'pos 填的是 Sinusoidal, 这是本仓库 GPT3 的默认值。原版 GPT-3 用可学位置编码。' },
  { name: 'LLaMA',       year: 2023, config: { attn: 'gqa', ffn: 'swiglu', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'Mixtral',     year: 2024, config: { attn: 'gqa', ffn: 'moe_mx', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'DeepSeek-V3', year: 2024, config: { attn: 'mla', ffn: 'moe_ds', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'DeepSeek-V3.2', year: 2025, config: { attn: 'dsa', ffn: 'moe_ds', norm: 'pre_rms', pos: 'rope' } },
  { name: 'Mamba',       year: 2023, config: { attn: 'ssm', ffn: 'swiglu', norm: 'pre_rms', pos: 'sin'   } },
  { name: 'DiT',         year: 2023, config: { attn: 'mha', ffn: 'gelu',   norm: 'ada_ln',  pos: 'learn' } },
  { name: 'Qwen2-VL',    year: 2024, config: { attn: 'gqa', ffn: 'swiglu', norm: 'pre_rms', pos: 'mrope' } },
]
function loadPreset(p) { Object.assign(config, p.config) }

// 两个子层的纵向布局: 从主干哪里出去、norm 和零件摆在哪、加号在哪
// Block 接线图。坐标交给 DagView 里的 dagre 算, 这里只声明"谁接谁"。
// 换零件只改 label/color, 换 Post-LN 才改接线。
const isPostLn = computed(() => config.norm === 'post_ln')

const blockDag = computed(() => {
  const io = (id, label) => ({ id, label, sub: '[B, T, D]', kind: 'io' })
  const norm = (id, note) => ({ id, label: partName('norm'), color: partColor('norm'), note })
  const ropeNote = config.pos === 'rope' || config.pos === 'mrope'
    ? `${partNote('attn')} · ${partName('pos')} 注入 Q/K`
    : partNote('attn')
  const attn = { id: 'attn', label: partName('attn'), color: partColor('attn'), note: ropeNote }
  const ffn = { id: 'ffn', label: partName('ffn'), color: partColor('ffn'), note: partNote('ffn') }

  if (isPostLn.value) {
    // norm 压在主干上: 梯度每层过一次缩放
    return {
      dir: 'TB',
      nodes: [
        io('x', 'x (in)'), attn, { id: 'add1', label: '+', kind: 'trunk' },
        norm('n1', '★ 在主干上: 梯度过这里要被缩放一次'),
        ffn, { id: 'add2', label: '+', kind: 'trunk' },
        norm('n2', '★ 又缩放一次'), io('y', 'y (out)'),
      ],
      edges: [
        { from: 'x', to: 'add1', kind: 'main', label: '残差' }, { from: 'x', to: 'attn' }, { from: 'attn', to: 'add1' },
        { from: 'add1', to: 'n1', kind: 'main' },
        { from: 'n1', to: 'add2', kind: 'main', label: '残差' }, { from: 'n1', to: 'ffn' }, { from: 'ffn', to: 'add2' },
        { from: 'add2', to: 'n2', kind: 'main' }, { from: 'n2', to: 'y', kind: 'main' },
      ],
    }
  }
  // Pre-LN: 主干是一条纯加法通路, norm 在分支里
  return {
    dir: 'TB',
    nodes: [
      io('x', 'x (in)'),
      { ...norm('n1', '只归一化送进分支的副本'), group: 'attn' }, { ...attn, group: 'attn' },
      { id: 'add1', label: '+', kind: 'trunk', note: '残差相加; 主干上没有 norm' },
      { ...norm('n2'), group: 'ffn' }, { ...ffn, group: 'ffn' },
      { id: 'add2', label: '+', kind: 'trunk' },
      io('y', 'y (out)'),
    ],
    edges: [
      { from: 'x', to: 'add1', kind: 'main', label: '残差流 x' },
      { from: 'x', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
      { from: 'add1', to: 'add2', kind: 'main', label: '残差流 x' },
      { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' }, { from: 'ffn', to: 'add2' },
      { from: 'add2', to: 'y', kind: 'main' },
    ],
    groups: [
      { id: 'attn', label: 'attention 子层' },
      { id: 'ffn', label: 'FFN 子层' },
    ],
  }
})

const modelMatch = computed(() => {
  const match = presets.find(p =>
    p.config.attn === config.attn &&
    p.config.ffn === config.ffn &&
    p.config.norm === config.norm &&
    p.config.pos === config.pos
  )
  if (match) {
    const meta = {
      'Transformer': { track: 'left', color: C.old,
        blurb: '原始 encoder-decoder + Post-LN, 全部零件最朴素' },
      'BERT': { track: 'left', color: C.a,
        blurb: '双向注意力 + MLM, 理解任务里程碑' },
      'GPT-3': { track: 'left', color: C.a,
        blurb: '把 Transformer 纯 decoder 堆到 175B, 生成式范式起点' },
      'LLaMA': { track: 'left', color: C.b,
        blurb: '现代开源 LLM 的事实模板: 四个零件全部换成最新版' },
      'Mixtral': { track: 'left', color: C.c,
        blurb: 'LLaMA 骨架, FFN 换成 softmax top-k MoE' },
      'DeepSeek-V3': { track: 'left', color: C.d,
        blurb: '把 KV cache 压到 latent + MoE 切更细, 671B/37B 激活' },
      'DeepSeek-V3.2': { track: 'left', color: C.d,
        blurb: 'V3 + DSA, 把算力从 $O(T^2)$ 降到 $O(T\\cdot k)$' },
      'Mamba': { track: 'left', color: C.e,
        blurb: '非注意力分支: SSM 线性 $O(T)$, 另一条主线' },
      'DiT': { track: 'right', color: C.d,
        blurb: '扩散 Transformer 骨架, adaLN-Zero 注入 timestep' },
      'Qwen2-VL': { track: 'eye', color: C.c,
        blurb: 'LLaMA + M-RoPE, 视觉/文本共用 decoder' },
    }
    return { ...match, ...meta[match.name], color: meta[match.name]?.color || 'var(--accent)' }
  }
  return {
    name: '自定义组合',
    year: '—',
    track: null,
    color: 'var(--text-muted)',
    blurb: '这组零件没有对应的主流模型。',
  }
})

const generatedCode = computed(() => {
  const cls = {
    mha: 'MultiHeadAttention',
    gqa: 'GroupedQueryAttention',
    mla: 'MultiHeadLatentAttention',
    dsa: 'MultiHeadLatentSparseAttention',
    ssm: 'SelectiveSSM',
  }[config.attn]
  const ffn = {
    relu: 'FeedForward', gelu: 'GeLUFeedForward', swiglu: 'SwiGLUFeedForward',
    moe_mx: 'MixtralMoE', moe_ds: 'DeepSeekMoE',
  }[config.ffn]
  const norm = {
    post_ln: 'nn.LayerNorm', pre_ln: 'nn.LayerNorm',
    pre_rms: 'RMSNorm', ada_ln: 'AdaLNZeroBlock',
  }[config.norm]
  return `PreLNBlock(
    d_model=4096,
    attn=${cls}(d_model, n_heads),
    ffn=${ffn}(d_model, d_ff),
    norm_cls=${norm},
)`
})

// FFN 参数量粗估 (d_model=4096, d_ff=4*d_model; MoE 粗算每层总参数)
const ffnParams = computed(() => {
  const d = 4096, d_ff = 4 * d
  switch (config.ffn) {
    case 'relu':
    case 'gelu':   return (2 * d * d_ff).toLocaleString() + ' ≈ 134M'
    case 'swiglu': return (3 * d * Math.floor(d_ff * 2 / 3)).toLocaleString() + ' ≈ 100M'
    case 'moe_mx': return '8 × 100M ≈ 800M (激活 2×)'
    case 'moe_ds': return '64 × 12.5M + 2 × 12.5M ≈ 825M (激活 8×)'
  }
  return '—'
})
const ffnNote = computed(() => {
  if (config.ffn === 'swiglu') return '三个线性层, 但中间维缩到 $8d/3$, 参数量和两层 GELU 持平: 一路当开关, 一路当内容'
  if (config.ffn.startsWith('moe')) return '总参数很大, 但每个 token 只过 top-k 个专家, 用显存换算力'
  return '标准 FFN (d_ff = 4·d_model), 两个线性层'
})
const kvNote = computed(() => {
  const d = 4096, h = 32
  const kb = (x) => (x * 2 / 1024).toFixed(1) + ' KiB'
  switch (config.attn) {
    case 'mha': return kb(2 * d)  // K+V 全维
    case 'gqa': return kb(2 * d / 4) + ' (kv_heads=8)'
    case 'mla': return kb(512 + 64) + ' (latent + rope)'
    case 'dsa': return kb(512 + 64) + ' + 稀疏'
    case 'ssm': return 'O(d_state) ≈ 0.1 KiB'
  }
  return '—'
})
</script>

<style scoped>
/* 三栏: 零件库 | 数据流 | 匹配模型。窄屏叠成单列, 否则右栏会被 .main 的 overflow-x: hidden 裁掉 */
.assembler {
  grid-template-columns: 260px minmax(0, 1fr) 300px;
  gap: 20px;
}
@media (max-width: 960px) {
  .assembler { grid-template-columns: minmax(0, 1fr); }
}
.repo-note { margin-top: 8px; font-size: 12px; }

.slot-section { margin-top: 14px; }
.slot-section:first-child { margin-top: 0; }
.slot-title {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 6px;
}
.slot-options {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.slot-options button {
  text-align: left;
  padding: 8px 10px;
  font-size: 12px;
}

.match-card { transition: border-color 0.3s; }
.match-name {
  font-size: 22px;
  font-weight: 700;
  margin: 8px 0 8px;
  font-family: "SF Mono", Menlo, monospace;
}
</style>
