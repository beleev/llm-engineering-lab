<template>
  <div>
    <h1 class="page-title">Block 组装器</h1>
    <div class="page-subtitle lead-group">
      <p>
        现代 Transformer 模型由几十个结构相同的编码或解码 block 堆叠而成。每个 block 包含注意力 (attn) 与前馈网络 (ffn) 两个子层, 各自搭配归一化层 (norm), 并由位置编码 (pos) 为注意力机制注入 token 的序列位置信息。
        本仓库在架构设计上的核心抽象在于: <strong>不同主流模型之间的架构差异几乎全部收敛在构造参数 (即实例化 block 时传入的子模块组件) 上, 极少需要重写底层骨架类</strong>。
      </p>
      <p>
        在下方零件库的四个槽位中各选一项, 即可查看对应的主流模型架构。底层通用骨架仅有一份实现:
        <RepoLink path="llm_models/layers/core/blocks.py:PreLNBlock" label="llm_models/layers/core/blocks.py::PreLNBlock(attn, ffn, norm_cls)" tiny />。
      </p>
    </div>

    <ChapterIntro
      tldr="现代大语言模型由几十层结构相同的 Pre-LN Block 堆叠而成: 残差主干一路只做加法, 让梯度无损传回底层; 注意力与前馈网络作为旁路分支接在归一化之后。本仓库只写了一份通用的 PreLNBlock 骨架, 把注意力、前馈网络和归一化做成可插拔的构造参数——从 GPT-3、LLaMA 到 DeepSeek-V3, 本质上都是同一副骨架换上了不同代的零件组合。"
      question="既然主流模型之间的核心差异仅在于 4 个构造参数, 阅读 N 个模型的架构源码是否就可以简化为对照 N 张零件配置表?"
      :goals="[
        '将主流 Transformer 模型拆解并映射为 PreLNBlock 的四个槽位配置',
        '阐明 Pre-LN 为何能让深层网络无需依赖精细的学习率 warmup 即可稳定训练',
        '在 LLaMA、Mixtral 与 DeepSeek 的源码中快速定位组件替换对应的构造逻辑',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/blocks.py', label: 'blocks.py · PreLNBlock' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 稳定残差主干与组件模块化"
      subtitle="根问题: 数十层网络堆叠时, 反向传播梯度必须稳定传回底层; 当残差主干定型后, 模型演进便收敛为内部子模块的迭代"
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
          图上是什么: 自上而下展示隐状态张量流经单个 block 的计算路径, 入口为 x (in), 出口为 y (out); 张量形状 [B, T, D] 对应 batch 大小 B、序列长度 T 与隐藏维 D。
          高亮竖线表示残差主干, 主干上仅包含逐元素加法节点「+」; 两个子层作为旁路分支从主干引出, 计算完成后加回主干。
        </p>
        <ul class="desc pts">
          <li>
            <b>Pre-LN:</b> 归一化层仅作用于送入旁路分支的输入副本, 残差流从入口直通出口且仅经过加法运算。
            反向传播时加法算子将梯度原样回传, 从而形成一条不受归一化重缩放影响的恒等梯度通路直达模型底层 (Xiong et al., 2020)。
          </li>
          <li>
            <b>分支:</b> attention 与 FFN 均位于旁路分支, 计算结果以残差增量形式累加回主干; 归一化不会原地覆盖主干上的隐状态 x。
          </li>
          <li>
            <b>Post-LN:</b> 将 norm 槽位切换为 Post-LN 后, 拓扑连线随之改变: 子层输出先与残差相加, 再在主干上执行归一化。反向梯度每经过一个主干 norm 节点都会被重缩放一次, 数十层累积后容易引发梯度失衡甚至训练发散,
            因此必须配合精细的学习率 warmup 才能稳定启动训练。
          </li>
        </ul>
        <p class="desc" style="margin-top: 6px;">可以做什么:</p>
        <ul class="desc pts" style="margin-bottom: 16px;">
          <li>在左侧切换不同槽位的组件, 观察图中对应子层节点的名称与注释同步更新。</li>
          <li>将归一化切换为 Post-LN, 对比残差主干上新增的两个归一化节点及其拓扑变化。</li>
          <li>拖拽画布平移视图, 或点击子层分组框折叠与展开内部节点。</li>
        </ul>

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
          <h3>组件替换对参数量与 KV cache 的影响</h3>
          <div class="stat" style="margin-top: 4px;">
            <div class="k">FFN 参数</div>
            <div class="v">{{ ffnParams }}</div>
            <div class="hint"><Tex :text="ffnNote" /></div>
          </div>
          <div class="stat" style="margin-top: 8px;">
            <div class="k">KV 每层 / token</div>
            <div class="v">{{ kvNote }}</div>
            <div class="hint">基于 d_model 为 4096、n_heads 为 32、fp16 精度估算: 每生成一个 token 时单层新增的 KV cache 字节数</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速切换到真实模型 -->
    <section class="section">
      <h2>一键载入预设</h2>
      <div class="lead-group">
        <p>
          点击下方任意模型预设, 即可将其组件配置载入上方组装器。
          按时间顺序依次切换 Transformer、GPT-3、LLaMA、Mixtral 与 DeepSeek-V3, 可以清晰观察四个槽位的演进脉络:
        </p>
        <ul class="pts">
          <li><b>norm:</b> 先由 Post-LN 迁移至 Pre-LN (将归一化从残差主干移入旁路分支), 随后精简为省去均值中心化的 RMSNorm。</li>
          <li><b>ffn:</b> 激活函数由分段线性的 ReLU 演进为平滑的 GELU, 再升级为引入逐通道门控分支的 SwiGLU, 乃至稀疏激活的 MoE。</li>
          <li><b>attn:</b> 先由标准多头注意力 MHA 演进为多查询头共享键值头的 GQA, 再进一步发展为将 KV cache 低秩压缩至潜在向量的 MLA。</li>
          <li><b>pos:</b> 由直接在输入 embedding 上相加的正余弦绝对位置编码, 演进为在注意力层内对 Q/K 施加相对位置旋转变换的 RoPE。</li>
        </ul>
        <p>
          各预设均对应本仓库中的实际代码实现。其中 BERT 与 GPT-3 各有一处配置与原始论文存在差异, 按钮上已标注「本仓库实现」, 载入后可在右侧「匹配模型」卡片中查看具体差异说明。
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
          展开当前所选组件的内部计算图与权重矩阵维度。每个计算步骤均同时标注符号张量形状 (如 [B, T, D] 分别代表 batch 大小 B、序列长度 T 与隐藏维 D) 以及当前超参数下的具体维度数值。
          拖动面板中的维度滑杆, 所有张量尺寸与参数统计将实时同步更新。
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
      <h2>组件之外的关键细节: 权重初始化</h2>
      <div class="lead-group">
        <p>
          未经训练的初始语言模型应当在词表上呈近似均匀分布预测, 即每个词的预测概率约为 1/V (V 为词表大小)。
          此时交叉熵损失 CE 的理论基准等于 <Tex text="$\ln V$" />; 当词表大小 V 为 1000 时约为 6.91, 正常初始化模型的首步 loss 应当落在该数值附近。
        </p>
        <p>
          一个典型的工程隐患出现在权重绑定场景中: LLaMA 将输出投影层 lm_head 与输入词嵌入共享同一权重矩阵以节省参数, 若输入嵌入直接沿用 <span class="mono">nn.Embedding</span> 默认的
          <Tex text="$\mathcal{N}(0, 1)$" /> 初始化 (<Tex text="$\sigma = 1$" />), 首步损失便会严重异常。
        </p>
        <p>具体成因如下:</p>
        <ol class="pts">
          <li>残差主干全程执行加法累加, 初始阶段顶层输出的隐状态方向仍由输入 token 自身的嵌入向量主导。</li>
          <li>lm_head 通过计算顶层归一化隐状态与词表各词嵌入的内积生成 logits: 与输入 token 自身嵌入的内积高达约 <Tex text="$\sigma\cdot d$" /> (d 为隐藏维宽度), 而与其他词嵌入的内积仅为 <Tex text="$\sigma\sqrt{d}$" /> 量级。</li>
          <li>这导致模型在初始步便以极高置信度预测「下一个词等于当前输入词本身」, 而真实目标标签几乎不会是当前词自身。</li>
        </ol>
        <ul class="pts">
          <li><b><Tex text="$\sigma = 1$" />:</b> 当 d 为 256、V 为 1000 时, 首步 CE 高达 <strong class="mono">258</strong>, 约为理论值 <Tex text="$\ln 1000 = 6.91$" /> 的 37 倍。</li>
          <li><b><Tex text="$\sigma = 0.02$" />:</b> 本仓库在 utils/init.py 中统一将所有 Linear 与 Embedding 初始化为 <Tex text="$\mathcal{N}(0, 0.02^2)$" />, 修正后 LLaMA 首步 CE 回归至 <strong class="mono">7.07</strong> (llama 目录 README 实测值为 7.073, 与理论值 <Tex text="$\ln V = 6.908$" /> 仅相差 0.17)。</li>
        </ul>
        <p>
          此类初始化缺陷在训练初期极具隐蔽性, 因为损失曲线依然会快速下降; 但前期的优化步主要消耗在纠正「对自身打高分」的初始偏置上, 而非学习真实的语言规律。可在下方实验台中调节 <Tex text="$\sigma$" /> 验证这一现象。
        </p>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />
    <TopicSupplement v-for="section in topicPages.blocks?.sections || []" :key="section.id" :section="section" />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '回顾本章 pos 槽位各位置编码方案的原理与演进' }"
      :next="{ ...nextChapter, hint: '将 ffn 槽位扩展为混合专家架构: Mixtral 与 DeepSeek 的两条技术路线' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, reactive, computed } from 'vue'
import { tracks, learningPath, topicPages } from '@/data/models.js'
import TopicSupplement from '@/components/TopicSupplement.vue'
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
    pain: '(原点) 数十层网络堆叠时, 反向传播梯度需从顶层传回底层, 逐层连乘极易引发梯度爆炸或消失',
    fix: '子层输出与残差相加后再执行 LayerNorm。由于归一化层位于残差主干上, 梯度每经一层都会被重缩放一次, 必须依赖精细的学习率 warmup 才能避免训练初期发散' },
  { name: 'Pre-LN', color: 'var(--accent)',
    pain: 'Post-LN 一旦缺乏精细 warmup 极易训练发散, 网络越深超参数越难调优',
    fix: '将归一化移入旁路分支, 仅对送入 attn 与 FFN 的副本做归一化, 残差主干只保留加法运算, 使反向梯度沿恒等通路无损直达底层' },
  { name: 'LLaMA 换零件', year: 2023, color: 'var(--left)',
    pain: 'Pre-LN 残差骨架定型后, 进一步提升参数效率与训练稳定性的空间转向内部子层',
    fix: '用 RMSNorm 替代 LayerNorm (省去均值中心化与偏置参数, 仅按均方根缩放), 用带门控分支的 SwiGLU 替代传统 FFN, 底层残差骨架保持不变' },
  { name: 'PreLNBlock', color: 'var(--right)',
    pain: '每替换一个子层组件就复制整套模型代码, 导致大量冗余且掩盖了真实的架构差异',
    fix: '将 Pre-LN 残差骨架固定为通用类, 把 attn、ffn、norm 与 pos 抽象为可插拔构造参数, 定义新模型即声明一张组件配置表' },
]

const inspectorTab = ref('attn')

const slots = [
  { key: 'attn', label: '注意力 (attn)', options: [
    { id: 'mha', name: 'MHA',  note: '标准多头, 每头独立缓存 K/V' },
    { id: 'gqa', name: 'GQA',  note: '分组共享, KV 头数为 Q 头的 1/4' },
    { id: 'mla', name: 'MLA',  note: '将 K/V 低秩压缩为 latent 缓存' },
    { id: 'dsa', name: 'DSA',  note: 'MLA 结合 top-k 位置稀疏计算' },
    { id: 'ssm', name: 'SSM',  note: '无注意力机制, 维护定长状态 (Mamba)' },
  ]},
  { key: 'ffn', label: '前馈 (ffn)', options: [
    { id: 'relu',   name: 'ReLU',    note: '原始 Transformer 分段线性激活' },
    { id: 'gelu',   name: 'GELU',    note: '平滑非线性激活, BERT / GPT 采用' },
    { id: 'swiglu', name: 'SwiGLU',  note: '逐通道门控前馈层, LLaMA / Qwen 采用' },
    { id: 'moe_mx', name: 'Mixtral-MoE', note: '8 个专家, softmax 路由选 top-2' },
    { id: 'moe_ds', name: 'DeepSeek-MoE', note: 'sigmoid 路由 + 常驻共享专家' },
  ]},
  { key: 'norm', label: '归一化 (norm)', options: [
    { id: 'post_ln', name: 'Post-LN', note: '归一化位于残差主干, 依赖 warmup' },
    { id: 'pre_ln',  name: 'Pre-LN LayerNorm', note: '归一化移入旁路分支, GPT-3 采用' },
    { id: 'pre_rms', name: 'Pre-LN RMSNorm',   note: '省去均值中心化, LLaMA / DeepSeek 采用' },
    { id: 'ada_ln',  name: 'adaLN-Zero',       note: '由条件向量调制归一化参数, DiT 采用' },
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
    repoNote: '本仓库 BERT 统一基于 PreLNBlock 构建, 因此 norm 槽位配置为 Pre-LN; 原始论文 BERT 采用 Post-LN。' },
  { name: 'GPT-3',       year: 2020, config: { attn: 'mha', ffn: 'gelu',   norm: 'pre_ln',  pos: 'sin'   },
    repoNote: '本仓库 GPT3 默认使用 Sinusoidal 位置编码; 原始论文 GPT-3 采用可学习绝对位置编码 (Learnable)。' },
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
        norm('n1', '★ 位于主干: 反向梯度经此重缩放一次'),
        ffn, { id: 'add2', label: '+', kind: 'trunk' },
        norm('n2', '★ 位于主干: 反向梯度再次重缩放'), io('y', 'y (out)'),
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
      { ...norm('n1', '仅归一化送入分支的输入副本'), group: 'attn' }, { ...attn, group: 'attn' },
      { id: 'add1', label: '+', kind: 'trunk', note: '残差相加; 主干无归一化层' },
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
        blurb: '2017 年提出的原始 Transformer 架构: 采用 encoder-decoder 结构与 Post-LN 主干, 四个槽位均为初代基础实现。' },
      'BERT': { track: 'left', color: C.a,
        blurb: '仅保留双向注意力 encoder 结构, 允许每个 token 同时关注上下文; 基于掩码语言建模 (MLM) 预训练, 奠定了自然语言理解任务的预训练范式。' },
      'GPT-3': { track: 'left', color: C.a,
        blurb: '采用带因果掩码的纯 decoder 结构并扩展至 175B 参数规模, 以自回归预测下一个 token 为统一训练目标, 确立了生成式大模型范式。' },
      'LLaMA': { track: 'left', color: C.b,
        blurb: '现代开源大语言模型的标准架构基线: 沿用 GPT-3 的 Pre-LN 残差骨架, 将四个核心组件全面升级为 GQA、SwiGLU、RMSNorm 与 RoPE。' },
      'Mixtral': { track: 'left', color: C.c,
        blurb: '在 LLaMA 架构基础上仅将稠密 FFN 替换为稀疏 MoE 层: 每层包含 8 个专家网络, 每个 token 经 softmax 路由选出 top-2 个专家参与计算。' },
      'DeepSeek-V3': { track: 'left', color: C.d,
        blurb: '注意力层升级为压缩 KV cache 的 MLA, 前馈层采用细粒度专家搭配 sigmoid 路由与常驻共享专家; 总参数量达 671B, 单 token 仅激活 37B 参数。' },
      'DeepSeek-V3.2': { track: 'left', color: C.d,
        blurb: '在 V3 基础上引入动态稀疏注意力 DSA: 先由轻量级打分器筛选出 top-k 个关键位置再执行注意力计算, 将序列计算复杂度从 $O(T^2)$ 降至 $O(T\\cdot k)$。' },
      'Mamba': { track: 'left', color: C.e,
        blurb: '脱离注意力机制的状态空间模型 (SSM) 路线: 通过选择性状态转移维护定长递归状态, 使计算复杂度随序列长度呈 $O(T)$ 线性增长。' },
      'DiT': { track: 'right', color: C.d,
        blurb: '面向扩散生成模型的 Transformer 架构: 将去噪时间步 timestep 等条件向量经零初始化的自适应归一化层 adaLN-Zero 调制后注入每个 block。' },
      'Qwen2-VL': { track: 'eye', color: C.c,
        blurb: '在 LLaMA 骨架上引入多维旋转位置编码 M-RoPE (沿时间、行、列三个坐标轴分别施加旋转变换), 使视觉图像 patch 与文本 token 共享同一自回归 decoder。' },
    }
    return { ...match, ...meta[match.name], color: meta[match.name]?.color || 'var(--accent)' }
  }
  return {
    name: '自定义组合',
    year: '—',
    track: null,
    color: 'var(--text-muted)',
    blurb: '当前组件组合未对应已收录的主流模型架构。',
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
    case 'swiglu': return (3 * d * Math.floor(d_ff * 2 / 3)).toLocaleString() + ' ≈ 134M'
    case 'moe_mx': return '8 × 134M ≈ 1.07B (激活 2×)'
    case 'moe_ds': return '64 × 12.5M + 2 × 12.5M ≈ 825M (激活 8×)'
  }
  return '—'
})
const ffnNote = computed(() => {
  if (config.ffn === 'swiglu') return '包含三个投影矩阵 (gate、up、down), 中间隐藏维由 $4d$ 缩减为 $8d/3$, 使总参数量与双矩阵 GELU-FFN 持平: gate 分支计算逐通道门控权重, up 分支提取特征表示'
  if (config.ffn.startsWith('moe')) return '由多组独立 FFN 专家组成, 总参数容量大幅扩展, 但每个 token 仅激活路由器选中的少数几个专家, 实现总参数量与单步计算量的解耦'
  return '标准双矩阵 FFN: 先将维度升至 $4d$, 经非线性激活后再投影回 $d$ 维'
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
