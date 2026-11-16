<template>
  <div>
    <h1 class="page-title">Block 组装器</h1>
    <div class="page-subtitle lead-group">
      <p>
        一个 Transformer 模型是几十个长得一模一样的 block 叠起来的。每个 block 里有两个子层: 注意力 (attn) 和前馈网络 (ffn), 各配一个归一化 (norm); 位置编码 (pos) 决定注意力怎么知道 token 的先后。
        这个库最省事的一点: <strong>模型之间的差别基本都在「构造参数」上 (也就是建 block 时传进去的那几个零件), 很少需要新写一个类</strong>。
      </p>
      <p>
        下面的零件菜单每栏挑一个, 看看拼出来的是哪个模型。骨架就一份:
        <RepoLink path="llm_models/layers/core/blocks.py:PreLNBlock" label="llm_models/layers/core/blocks.py::PreLNBlock(attn, ffn, norm_cls)" tiny />。
      </p>
    </div>

    <ChapterIntro
      tldr="- 一个 LLM 是几十个相同的 block 叠起来的。每个 block 里, 向量先过归一化 (norm) 再进注意力 (attn), 结果加回原向量; 再过一次 norm 进前馈网络 (ffn), 再加回来。位置编码 (pos) 告诉注意力 token 的先后。
- 本仓库把这副骨架只写了一份 PreLNBlock, attn / ffn / norm 当构造参数传进去。GPT-3 = MHA + GELU-FFN + LayerNorm, LLaMA = GQA + SwiGLU + RMSNorm, DeepSeek-V3 = MLA + MoE + RMSNorm: 不是三个新模型, 是同一份骨架换了零件。
- 骨架用 Pre-LN: norm 放在分支里, 主干上只有加法, 几十层堆起来, 训练信号也能原样传回底部。"
      question="模型之间就差 4 个构造参数的话, 「读 N 个模型的源码」是不是变成了「读 N 张零件配置表」?"
      :goals="[
        '把任意一个模型翻译成 PreLNBlock 的四个零件配置',
        '说清 Pre-LN 为什么让深层模型不靠精细 warmup (学习率先小后大) 也能开训',
        '在 LLaMA / Mixtral / DeepSeek 的代码里一眼找到零件被换掉的那几行',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/blocks.py', label: 'blocks.py · PreLNBlock' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 骨架先稳住, 再只换零件"
      subtitle="根问题: 几十层堆起来, 训练信号 (梯度) 得能从最后一层传回第一层; 骨架稳住之后, 各家的差别只落在零件上"
      :steps="evoSteps"
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
        <p class="desc">
          图上是什么: 从上到下是一个 token 的向量走过一个 block 的路, 入口 x (in), 出口 y (out); 形状 [B, T, D] 是「B 条序列, 每条 T 个 token, 每个 D 维」。
          高亮那条竖线是残差流 (主干), 它身上只有「+」; 两个子层从旁边接出去, 算完加回主干。
        </p>
        <ul class="desc pts">
          <li>
            <b>Pre-LN:</b> norm (归一化, 把向量拉回固定尺度) 只作用于送进分支的那份副本, 残差流从块的入口一直贯到出口, 上面只有加法。
            训练时梯度 (告诉参数往哪改的信号) 要从最后一层传回第一层; 加法把梯度原样往回传, 所以有一条不被缩放的恒等通路直接到底 (Xiong et al., 2020)。
          </li>
          <li>
            <b>分支:</b> attention 和 FFN 都不在主干上, 从旁边接出去算完再加回来; norm 只归一化送进分支的副本, 主干上的 x 本身不被改。
          </li>
          <li>
            <b>Post-LN:</b> 把 norm 槽位换成 Post-LN, 接线跟着变: 子层算完先加回主干, 再在主干上做 norm。梯度每过一个 norm 就被它缩放一次, 几十层叠起来就可能放大或衰减到训练发散,
            所以得靠精细的 warmup (学习率先很小再慢慢升) 才敢开训。
          </li>
        </ul>
        <p class="desc" style="margin: 6px 0 16px;">可以做什么: 左边换零件, 图里对应节点的名字跟着换; 切到 Post-LN, 看主干上多出两个 norm 节点。图可以拖, 子层可以折叠。</p>

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
            <div class="hint">按 d_model=4096, n_heads=32, fp16 算: 每生成一个 token, 这一层要新存的字节</div>
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
          按年份依次点 Transformer、GPT-3、LLaMA、Mixtral、DeepSeek-V3, 你会看到四个槽位是一个一个被换掉的:
        </p>
        <ul class="pts">
          <li><b>norm:</b> 先从 Post-LN 挪到 Pre-LN (norm 从主干挪进分支), 再瘦成 RMSNorm (只按均方根缩放, 不减均值)。</li>
          <li><b>ffn:</b> 激活函数从 ReLU 换成 GELU (硬开关换成软开关), 再换成带门控的 SwiGLU (一路决定每个通道开多大, 一路装内容)。</li>
          <li><b>attn:</b> 从 MHA 砍到 GQA (几个头共用一对 K/V), 再压成 MLA (K/V 存成压缩件)。</li>
          <li><b>pos:</b> 从加在 embedding 上的正余弦换成旋转 Q/K 的 RoPE。</li>
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
          把选中的零件拆开看计算流和权重矩阵。每一步都标了张量的符号形状 (例如 [B, T, D] 是 B 条序列、每条 T 个 token、每个 D 维), 以及按当前参数算出来的实际数字。
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
          一个刚建好、什么都没学的模型, 应该在词表上均匀瞎猜: 每个词的概率都是 1/V (V 是词表大小)。
          交叉熵 CE 是「正确答案的概率取负对数」, 均匀瞎猜时它就等于 <Tex text="$\ln V$" />; V=1000 时是 6.91。第一步的 loss 该落在这条线附近。
        </p>
        <p>
          典型的坑: LLaMA 把 lm_head (最后把隐藏向量变成每个词一个分数的那个矩阵) 和输入嵌入绑权重 (两处共用同一个矩阵, 省参数), 嵌入又用了 <span class="mono">nn.Embedding</span> 默认的
          <Tex text="$\mathcal{N}(0, 1)$" /> 初始化 (每个数的标准差 <Tex text="$\sigma = 1$" />)。
        </p>
        <p>
          为什么会炸: 残差流一路做加法, 最后一层输出的向量几乎还是 token 自己的嵌入。lm_head 给每个词打分, 是拿这个向量和那个词的嵌入做内积; 和「自己」的嵌入做内积得到约 <Tex text="$\sigma\cdot d$" /> 的高分 (d 是模型宽度), 其它词的分数只有 <Tex text="$\sigma\sqrt{d}$" /> 量级。
          于是模型极度自信地猜「下一个词就是我自己」, 而正确答案几乎从来不是它自己。
        </p>
        <ul class="pts">
          <li><b><Tex text="$\sigma = 1$" />:</b> d=256、V=1000 时首步 CE 是 <strong class="mono">258</strong>, 而 <Tex text="$\ln 1000 = 6.91$" />, 差了 37 倍。</li>
          <li><b><Tex text="$\sigma = 0.02$" />:</b> 改法只有这一行 (本仓库 utils/init.py 把所有 Linear 和 Embedding 初始化成 <Tex text="$\mathcal{N}(0, 0.02^2)$" />), LLaMA 的首步 CE 回到 <strong class="mono">7.07</strong> (llama 的 README 实测 7.073, 和 ln V = 6.908 只差 0.17)。</li>
        </ul>
        <p>
          这类 bug 骗人的地方在于 loss 照样在降。它只是在把「给自己打高分」纠正回来, 不代表模型学会了任务。下面的实验台把 <Tex text="$\sigma$" /> 交到你手上。
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
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import DagView from '@/components/dag/DagView.vue'
import Tex from '@/components/Tex.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'blocks')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 是上一步 fix 留下的窟窿
const evoSteps = [
  { name: 'Post-LN', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) 几十层堆起来, 训练信号 (梯度) 要从最后一层一路传回第一层, 中间每层都可能把它放大或缩小',
    fix: '每个子层算完先和输入相加 (残差), 再做 LayerNorm (把向量拉回固定尺度)。norm 压在主干上, 梯度每层都被它缩放一次, 得靠精细的学习率 warmup 才敢开训' },
  { name: 'Pre-LN', color: 'var(--accent)',
    pain: 'Post-LN 离了精细 warmup 就容易发散 (loss 变成 NaN 或飞走), 层越多越难调',
    fix: '把 norm 挪进分支: 只归一化送进 attn / FFN 的那份副本, 主干上只剩加法。反传时梯度沿着加法原样往回走, 有一条不被缩放的通路直达底部' },
  { name: 'LLaMA 换零件', year: 2023, color: 'var(--left)',
    pain: '骨架定型后, 再想变强只能往零件里找',
    fix: 'LayerNorm 换成 RMSNorm (只按均方根缩放, 不减均值, 少一组参数); FFN 换成 SwiGLU (多一条门控分支决定哪些通道通过); 骨架一行不改' },
  { name: 'PreLNBlock', color: 'var(--right)',
    pain: '每换一个零件就复制一整套模型代码, 看不出哪几行真变了',
    fix: '骨架只写一份, attn / ffn / norm / pos 当构造参数传进去: 新模型就是一张零件配置表' },
]

const inspectorTab = ref('attn')

const slots = [
  { key: 'attn', label: '注意力 (attn)', options: [
    { id: 'mha', name: 'MHA',  note: '原始, 每个头各存 K/V, cache 最大' },
    { id: 'gqa', name: 'GQA',  note: 'KV 头只留 Q 头的 1/4' },
    { id: 'mla', name: 'MLA',  note: 'K/V 压成低维 latent 再存' },
    { id: 'dsa', name: 'DSA',  note: 'MLA + 只算 top-k 个位置' },
    { id: 'ssm', name: 'SSM',  note: '不用注意力, 状态定长 (Mamba)' },
  ]},
  { key: 'ffn', label: '前馈 (ffn)', options: [
    { id: 'relu',   name: 'ReLU',    note: '原始 Transformer' },
    { id: 'gelu',   name: 'GELU',    note: 'BERT / GPT 用的软开关' },
    { id: 'swiglu', name: 'SwiGLU',  note: '带门控, LLaMA / Qwen' },
    { id: 'moe_mx', name: 'Mixtral-MoE', note: '8 个专家, softmax 选 2 个' },
    { id: 'moe_ds', name: 'DeepSeek-MoE', note: 'sigmoid 路由 + 共享专家' },
  ]},
  { key: 'norm', label: '归一化 (norm)', options: [
    { id: 'post_ln', name: 'Post-LN', note: '原始; norm 在主干上, 需 warmup' },
    { id: 'pre_ln',  name: 'Pre-LN LayerNorm', note: 'norm 在分支里, GPT-3 风' },
    { id: 'pre_rms', name: 'Pre-LN RMSNorm',   note: '不减均值, LLaMA / DeepSeek 风' },
    { id: 'ada_ln',  name: 'adaLN-Zero',       note: '用条件向量调 norm 的缩放, DiT 扩散' },
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
        blurb: '2017 年的原始 Transformer: encoder-decoder 结构, Post-LN, 四个零件都是最朴素的版本' },
      'BERT': { track: 'left', color: C.a,
        blurb: '只留 encoder, 注意力双向 (每个 token 能看前后文); 训练目标是 MLM (掩码语言建模: 遮住几个词让模型填), 理解类任务的里程碑' },
      'GPT-3': { track: 'left', color: C.a,
        blurb: '只留 decoder (因果 mask, 只看前文), 堆到 175B 参数, 任务是预测下一个 token; 生成式范式的起点' },
      'LLaMA': { track: 'left', color: C.b,
        blurb: '现代开源 LLM 的事实模板: 在 GPT-3 骨架上把四个零件全换成新版 (GQA、SwiGLU、RMSNorm、RoPE)' },
      'Mixtral': { track: 'left', color: C.c,
        blurb: 'LLaMA 骨架, 只把 FFN 换成 MoE: 8 个专家 FFN, 每个 token 由 softmax 路由选 2 个来算' },
      'DeepSeek-V3': { track: 'left', color: C.d,
        blurb: '注意力换成 MLA (KV cache 压成 latent), MoE 的专家切得更细 (sigmoid 路由 + 共享专家); 671B 总参数, 每个 token 只激活 37B' },
      'DeepSeek-V3.2': { track: 'left', color: C.d,
        blurb: 'V3 再加 DSA: 一个小打分器先挑出 top-k 个位置, 注意力只算这 k 个, 算力从 $O(T^2)$ 降到 $O(T\\cdot k)$' },
      'Mamba': { track: 'left', color: C.e,
        blurb: '不用注意力的另一条主线: SSM (状态空间模型) 用一个定长状态一路往后传, 算力随序列长度线性 $O(T)$ 增长' },
      'DiT': { track: 'right', color: C.d,
        blurb: '扩散模型用的 Transformer: 去噪步数 timestep 等条件经 adaLN-Zero (用条件向量算出 norm 的缩放和平移, 初始为零) 注入每个 block' },
      'Qwen2-VL': { track: 'eye', color: C.c,
        blurb: 'LLaMA 骨架 + M-RoPE (位置按时间 / 行 / 列三轴旋转), 图像 patch 和文本共用一个 decoder' },
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
  if (config.ffn === 'swiglu') return '三个矩阵 (gate、up、down), 但中间维从 4d 缩到 $8d/3$, 总参数量和两个矩阵的 GELU-FFN 持平: gate 这一路决定每个通道开多大, up 这一路装内容'
  if (config.ffn.startsWith('moe')) return '专家就是多份 FFN, 总参数很大, 但每个 token 只经过被路由选中的少数几个专家, 用显存换算力'
  return '标准 FFN: 先升到 d_ff = 4·d_model 维, 过激活函数, 再降回 d_model, 两个矩阵'
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
