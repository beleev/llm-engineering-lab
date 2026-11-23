<template>
  <div>
    <h1 class="page-title">注意力的四代演进</h1>
    <div class="page-subtitle lead-group">
      <p>
        语言模型的自回归生成逐 token 推进。每生成一个新 token, 注意力层都需要访问此前所有历史 token:
        每个历史 token 在每一层都会产生一个 K 向量 (用于匹配相关性) 与一个 V 向量 (用于聚合输出内容)。
        由于历史位置的 K 与 V 在后续解码步中不再改变, 推理引擎会将其计算一次后缓存下来, 即 KV cache。
      </p>
      <p>
        核心瓶颈在于显存容量与访存带宽开销过大: 当上下文扩展至 128K 个 token 时, KV cache 占用的显存甚至超过模型权重本身, 且解码阶段每生成一个新 token 都必须将其完整读取一遍。
        从 MHA 到 MQA、GQA、MLA, 再到 DSA, 四代演进始终围绕同一目标展开:
        <strong>在不损失模型质量的前提下, 持续压缩每层每个 token 需要缓存的状态体积。</strong>
      </p>
      <p>在代表性开源模型中, 单个 token 跨全部层的 KV cache 总量为:</p>
      <ul class="pts">
        <li><b>LLaMA-2-7B:</b> 512 KiB (单层 K 与 V 各含 4096 个元素, fp16 下每元素占 2 字节, 单层共 16 KiB, 合计 32 层)。</li>
        <li><b>LLaMA-3-8B:</b> 128 KiB (K/V 仅保留 8 个头, 每头 128 维)。</li>
        <li><b>DeepSeek-V3:</b> 68.6 KiB (单层仅缓存 576 个元素, 合计 61 层)。</li>
      </ul>
    </div>

    <ChapterIntro
      tldr="- 核心公式不变: 八年来注意力的主干计算始终是 $\mathrm{softmax}(QK^\top / \sqrt{d_k})V$ —— 用当前位置的 Q 与历史位置的 K 做点积打分, 归一化成权重后再对 V 加权求和。
- 演进焦点在于怎么存 K/V、怎么省计算:
  - MHA: 每个查询头独立保留一套 K/V, 表达力最足但缓存最大。
  - GQA: 多个查询头共享一组 K/V, 直接按组数成倍压缩缓存。
  - MLA: 不存完整 K/V, 改存低维压缩向量 $c_{kv}$ 与共享旋转位置向量, 用时再投影还原。
  - DSA: 在 MLA 缓存之上先用轻量打分器筛出 top-k 个候选位置, 把超长上下文的注意力计算从 $O(T^2)$ 降到 $O(T\cdot k)$。
- 直观量级: 同样宽度 $d_{\text{model}}=4096$ 的单层里, 每新增 1 个 token 的缓存可从 MHA 的 16 KiB 一路压到 MLA 的约 1.1 KiB。"
      question="自回归解码每生成一个 token 都必须读取此前所有 token 的 K 与 V。在不损害模型质量的前提下, 每层每个 token 缓存的 K 与 V 还能如何进一步压缩?"
      :goals="[
        '阐明 MHA、GQA、MLA 与 DSA 各自压缩的显存或计算开销来源',
        '定量计算查询头数、KV 头数、低秩压缩维与稀疏 top-k 变化时的 KV cache 规模',
        '对照源码定位四代注意力机制在 forward 实现中的核心差异',
      ]"
      :codes="[{ path: 'llm_models/layers/core/attention.py' }]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 四代注意力的瓶颈与解法"
      subtitle="根问题: 自回归推理需缓存全部历史 token 的 K/V 并逐位计算注意力, 长上下文下显存带宽与计算量双双受限。"
      :steps="evoSteps"
    />

    <!-- 先建立实验任务, 再让读者替换内部机制。 -->
    <section class="concept-workbench card" aria-labelledby="attention-workbench-title">
      <div class="workbench-heading">
        <span class="eyebrow">动手建立直觉</span>
        <div>
          <h2 id="attention-workbench-title">固定上下文与模型宽度, 对比「KV 如何存储」</h2>
          <p>
            本对照实验模拟单层注意力在推理阶段为整段上下文维护的 KV cache 容量。
            实验固定上下文长度 T 与模型宽度 d_model, 仅切换注意力方案, 使下方柱状图的差异完全归因于「K/V 如何存储」这一结构变量。
          </p>
        </div>
      </div>

      <ol class="learning-steps" aria-label="交互阅读顺序">
        <li><span>1</span><strong>选择上下文规模</strong><small>设定序列长度 T</small></li>
        <li><span>2</span><strong>切换注意力方案</strong><small>改变 K/V 的缓存结构</small></li>
        <li><span>3</span><strong>对比量化指标</strong><small>观察显存占用与计算量差异</small></li>
      </ol>

      <fieldset class="context-presets">
        <legend>第一步 · 选择典型上下文场景</legend>
        <div class="preset-grid">
          <button
            v-for="preset in contextPresets"
            :key="preset.id"
            type="button"
            :class="{ active: activePresetId === preset.id }"
            :aria-pressed="activePresetId === preset.id"
            @click="applyContextPreset(preset)"
          >
            <span>{{ preset.label }}</span>
            <small>{{ preset.hint }}</small>
          </button>
        </div>
      </fieldset>

      <div class="preset-readout" aria-live="polite" aria-atomic="true">
        <span class="readout-kicker">当前实验</span>
        <strong>{{ activePresetLabel }}</strong>
        <span>
          上下文长度 {{ formatT(p.T) }} 个 token, 模型宽度 d_model 为 {{ p.d_model }}。
          下一步仅切换注意力方案, 观察不同缓存结构下的显存与计算量变化。
        </span>
      </div>
    </section>

    <!-- 顶部 variant 切换 + paper cite -->
    <div class="variant-tabs" role="group" aria-label="第二步 · 选择 Attention 方案">
      <button
        v-for="(v, index) in variants"
        :key="v.id"
        type="button"
        :class="{ active: current.id === v.id }"
        :aria-pressed="current.id === v.id"
        @click="current = v"
      >
        <span class="variant-index" aria-hidden="true">{{ index + 1 }}</span>
        <span class="variant-name">{{ v.name }}</span>
        <span class="yr">{{ v.year }}</span>
      </button>
    </div>

    <section
      class="concept-bridge"
      :style="{ '--variant-color': current.color }"
      aria-labelledby="current-concept-title"
    >
      <div class="bridge-heading">
        <span class="eyebrow">第二步 · 当前注意力方案</span>
        <h2 id="current-concept-title">{{ current.fullName }}</h2>
        <Prose :text="current.description" />
      </div>
      <div class="causal-flow" aria-live="polite" aria-atomic="true">
        <article>
          <span class="causal-label">输入</span>
          <strong>{{ conceptStory.input }}</strong>
          <p>{{ conceptStory.inputNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="mechanism">
          <span class="causal-label">内部机制</span>
          <strong>{{ conceptStory.mechanism }}</strong>
          <p>{{ conceptStory.mechanismNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="outcome">
          <span class="causal-label">可观察结果</span>
          <strong>{{ conceptStory.outcome }}</strong>
          <p>{{ conceptStory.outcomeNote }}</p>
        </article>
      </div>
      <div class="concept-takeaway">
        <span>核心结论</span>
        <strong>{{ conceptStory.takeaway }}</strong>
      </div>
    </section>

    <!-- 主布局: 左参数, 中公式/数据, 右可视化 -->
    <div class="grid grid-2" style="gap: 20px;">
      <!-- 左: 参数 + 公式 -->
      <div class="card">
        <h3>参数调节 <span class="tag">可拖动</span></h3>
        <div class="form-row">
          <label for="attention-seq-length">序列长度 T</label>
          <input id="attention-seq-length" type="range" min="128" max="131072" step="128" v-model.number="p.T" />
          <span class="val">{{ formatT(p.T) }}</span>
        </div>
        <div class="form-row">
          <label for="attention-model-width">模型宽度 d_model</label>
          <input id="attention-model-width" type="range" min="512" max="8192" step="128" v-model.number="p.d_model" />
          <span class="val">{{ p.d_model }}</span>
        </div>
        <div class="form-row">
          <label for="attention-query-heads">查询头数 n_heads</label>
          <input id="attention-query-heads" type="range" min="4" max="64" step="2" v-model.number="p.n_heads" />
          <span class="val">{{ p.n_heads }}</span>
        </div>
        <div v-if="current.id === 'gqa'" class="form-row">
          <label for="attention-kv-heads">KV 头数 num_kv_heads</label>
          <input id="attention-kv-heads" type="range" :min="1" :max="p.n_heads" step="1" v-model.number="p.num_kv_heads" />
          <span class="val">{{ p.num_kv_heads }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-kv-rank">KV 压缩维 kv_lora_rank</label>
          <input id="attention-kv-rank" type="range" min="64" max="1024" step="32" v-model.number="p.kv_lora_rank" />
          <span class="val">{{ p.kv_lora_rank }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-rope-dim">位置维 qk_rope_head_dim</label>
          <input id="attention-rope-dim" type="range" min="16" max="128" step="8" v-model.number="p.qk_rope_head_dim" />
          <span class="val">{{ p.qk_rope_head_dim }}</span>
        </div>
        <div v-if="current.id === 'dsa'" class="form-row">
          <label for="attention-sparse-top-k">稀疏候选 sparse_top_k</label>
          <input id="attention-sparse-top-k" type="range" :min="32" :max="Math.min(2048, p.T)" step="32" v-model.number="p.sparse_top_k" />
          <span class="val">{{ p.sparse_top_k }}</span>
        </div>

        <div class="formula-box">
          <Tex :text="current.formula" />
        </div>

        <p class="parameter-feedback" aria-live="polite" aria-atomic="true">
          每新增 1 个 token, 本层向 KV cache 追加写入
          <strong class="mono">{{ formatBytes(cacheBytesPerToken(current)) }}</strong> (fp16 精度下每元素占 2 字节);
          当前整段上下文累计占用
          <strong class="mono">{{ formatBytes(cacheBytes(current)) }}</strong>。
        </p>

        <div class="trade" style="margin-top: 16px;">
          <div class="tr-item">
            <span class="ok">✓</span>
            <span>{{ current.pros }}</span>
          </div>
          <div class="tr-item">
            <span class="no">✗</span>
            <span>{{ current.cons }}</span>
          </div>
        </div>

        <div style="margin-top: 12px; font-size: 11px; color: var(--text-dim);">
          <span class="mono">{{ current.paper }}</span>
          · 使用模型: <span class="mono">{{ current.usedIn.join(', ') }}</span>
        </div>
      </div>

      <!-- 右: KV cache 横向对比柱图 -->
      <div class="card">
        <h3>整段 KV cache 对比 <span class="tag">单层 · 单序列 · fp16</span></h3>
        <ul class="desc pts" style="margin-bottom: 16px;">
          <li><b>图上是什么:</b> 四条柱状带分别对应四种注意力方案, 长度表示单层为<strong>整段上下文</strong>维护的 KV cache 字节数, 右侧百分比为相对于 MHA 基线的比例。</li>
          <li>
            <b>计算方式:</b> 单 token 缓存元素数乘以序列长度 <span class="mono">T</span> (当前为 <span class="mono">{{ formatT(p.T) }}</span>), 再乘以 fp16 单元素字节数 (2 字节)。
            当前模型宽度 <span class="mono">d_model</span> 为 <span class="mono">{{ p.d_model }}</span>, 查询头数 <span class="mono">n_heads</span> 为 <span class="mono">{{ p.n_heads }}</span>; 仅统计单层、单条序列 (batch 大小 B 为 1)。
          </li>
          <li><b>读图要点:</b> 柱体越短, 解码阶段每步需要从显存重读的字节数越少。拖动左侧滑杆可实时观察各方案的缓存变化。</li>
        </ul>
        <div class="bars" aria-label="四种 Attention 的单层 KV cache 对比">
          <div v-for="v in variants" :key="v.id" class="bar-row" :class="{ current: v.id === current.id }">
            <div class="bar-label">
              <span class="bar-name">{{ v.name }}</span>
              <span class="bar-sub mono">{{ formatBytes(cacheBytes(v)) }}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill"
                   :style="{ width: barPct(v) + '%', background: v.color }"></div>
            </div>
            <div class="bar-rel mono">{{ relToMHA(v) }}</div>
          </div>
        </div>

        <div class="hero-stat">
          <div class="stat">
            <div class="k">新增 1 token / 层</div>
            <div class="v accent">{{ formatBytes(cacheBytesPerToken(current)) }}</div>
            <div class="hint">每生成一个新 token, 单层需向缓存追加写入的字节数</div>
          </div>
          <div class="stat">
            <div class="k">当前上下文 / 层</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * 1) }}</div>
            <div class="hint">相对 MHA 基线: {{ relToMHA(current) }}</div>
          </div>
          <div class="stat">
            <div class="k">{{ LAYERS }} 层合计</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * LAYERS) }}</div>
            <div class="hint">按 LLaMA-2-7B 的 {{ LAYERS }} 层折算; 仅统计单序列 KV cache, 不含多 batch 与中间激活值</div>
          </div>
          <div class="stat">
            <div class="k">整段 (q, k) 配对计算数</div>
            <div class="v">{{ formatFlops(flopsPerQuery) }}</div>
            <div class="hint"><Tex text="全部 $T$ 个 query 需参与打分的 key 总对数; 增大 $T$ 可对比 $T^2$ 与 $T\cdot k$ 的增长斜率" /></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Attention 矩阵可视化 -->
    <section class="section">
      <h2>Attention 矩阵可视化</h2>
      <div class="lead-group">
        <ul class="pts">
          <li>
            <b>模拟什么:</b> 单层自注意力中各位置 token 对历史 token 的注意力权重分布。图中权重按「近邻距离衰减叠加局部热点」规则合成, 仅作机制示意而非真实模型推理输出。
          </li>
          <li>
            <b>图上是什么:</b>
            <ul class="pts">
              <li>每行对应一个 query (当前计算的 token), 每列对应一个 key (被访问的历史 token), 共 24 个 token。</li>
              <li>色块越深代表该 query 分配给对应 key 的注意力权重越高; 每行权重经归一化后总和为 1。</li>
              <li>右上三角被因果 mask (将未来位置的分数置为负无穷以禁止访问后续 token) 屏蔽, 确保自回归生成不泄漏未来信息。</li>
              <li v-if="current.id === 'dsa'">DSA 仅在 indexer 筛选出的 top-k 个候选位置上保留注意力权重, 因此左下三角亦呈大面积稀疏状态。</li>
            </ul>
          </li>
          <li>
            <b>可以做什么:</b> 在上方切换不同注意力方案。
            <ul class="pts">
              <li><b>MHA / GQA / MLA:</b> 三者仅改变 K/V 的缓存结构, 每个 query 仍对全部可见历史 key 计算分数, 因此注意力矩阵形态一致。</li>
              <li><b>DSA:</b> 切换至 DSA 并拖动「稀疏候选 sparse_top_k」滑杆, 每行仅保留得分最高的若干候选位置。</li>
            </ul>
          </li>
        </ul>
      </div>
      <div class="card">
        <div class="matrix-grid">
          <svg
            :viewBox="`0 0 ${matrixSize} ${matrixSize}`"
            width="100%"
            :height="matrixSize"
            style="max-width: 480px; border: 1px solid var(--border);"
            role="img"
            :aria-label="matrixAriaLabel"
          >
            <rect v-for="cell in matrixCells" :key="cell.i + ':' + cell.j"
                  :x="cell.j * cellSize" :y="cell.i * cellSize"
                  :width="cellSize - 0.5" :height="cellSize - 0.5"
                  :style="{ fill: cell.color }" />
          </svg>
          <div class="legend-attn">
            <div class="legend-strip">
              <span v-for="i in 10" :key="i"
                    :style="{ background: colorFor(i / 10) }"></span>
            </div>
            <div class="legend-labels mono">
              <span>0</span><span>权重</span><span>1</span>
            </div>
            <p class="desc" style="margin-top: 16px; max-width: 280px;">
              序列长度 N 为 <span class="mono">{{ N }}</span> 个 token 的因果注意力矩阵。
              <span v-if="current.id === 'dsa'">滑杆上的 sparse_top_k 在此小尺寸矩阵中截断上限为 <span class="mono">{{ Math.min(p.sparse_top_k, N) }}</span>; 绘制时另按 1/32 比例缩放, 每行最多高亮 8 格, 将滑杆调至 256 以下时高亮格数随之减少。</span>
            </p>
            <p class="matrix-reading">{{ matrixReading }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- mask 一族: 投影压缩之外的另一条降本路线 -->
    <section class="section">
      <h2>另外两条正交路线: 稀疏化注意力掩码与线性状态压缩</h2>
      <div class="lead-group">
        <p>本章主线聚焦于「压缩单 token 的 KV 投影表示」。围绕降低推理显存与计算开销, 工程上还有两条与之正交、可叠加使用的技术路线:</p>
        <ul class="pts">
          <li><b>压缩 KV 投影 (本章主线, 从 MHA、GQA 到 MLA):</b> 缩减每个历史 token 缓存的 K/V 向量体积, 但每个 query 仍对全部历史位置计算注意力。</li>
          <li>
            <b>稀疏化注意力掩码 (Mistral / StreamingLLM / DSA):</b> 通过注意力 mask 限定每个 query 可访问的历史 key 集合。
            <ul class="pts">
              <li><b>Mistral:</b> 采用滑动窗口注意力, 每个 token 仅访问最近 W 个历史位置, 滑出窗口的 KV 直接驱逐。</li>
              <li><b>StreamingLLM:</b> 在局部滑动窗口之外永久保留序列开头的若干个初始 token (即 attention sink)。</li>
              <li><b>DSA:</b> 每个 query 仅访问由轻量级 indexer 动态筛选出的 top-k 个历史位置。</li>
            </ul>
            减少参与计算的 key 数量可直接降低注意力计算量; 其中滑动窗口与 StreamingLLM 因驱逐了窗口外的历史状态, 同时还将 KV cache 限制在固定容量内。
          </li>
          <li>
            <b>线性注意力</b> (如 Gated DeltaNet, 见 <RepoLink path="llm_models/layers/sparse/linear_attention.py" label="linear_attention.py" tiny />):
            不再逐 token 保存 K/V 历史, 而是将全部上下文递归压缩进固定维度的状态矩阵, 使状态空间开销与序列长度完全解耦。代价是定长状态矩阵难以无损完成精确键值召回; 因此 Qwen3-Next 将 75% 的层替换为线性注意力, 其余 25% 仍保留标准全注意力以保障精确检索能力。
          </li>
        </ul>
        <p>下方交互实验台展示第二条路线: 调节滑动窗口宽度与 attention sink 数量, 观察各位置可见历史范围的变化。完整机制详见「SWA · MTP · 混合线性」一章。</p>
      </div>
      <AttnMaskLab />
    </section>

    <!-- 源码速览 -->
    <section class="section">
      <h2>核心代码</h2>
      <p class="lead">当前选中的方案对应 <RepoLink path="llm_models/layers/core/attention.py" label="llm_models/layers/core/attention.py" tiny /> 中的 <span class="mono">{{ classFor(current.id) }}</span> 类。四代注意力实现位于同一源码文件内, 便于直接对照结构差异; 下方代码片段已做精简, 重点展示决定「KV cache 存储结构与稀疏筛选方式」的核心逻辑。</p>
      <CodeBlock :code="codeSnippet" />
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount :names="['ResearchCompletionLab']" />
    <TopicSupplement v-for="section in topicPages.attention?.sections || []" :key="section.id" :section="section" />

    <QuizCard />


    <ChapterNav
      :prev="prevChapter"
      :next="{ ...nextChapter, hint: '理解注意力主干与缓存结构后, 进一步考察 RoPE 如何向 Q/K 注入相对位置信息' }"
    />
  </div>
</template>

<script setup>
import TopicSupplement from '@/components/TopicSupplement.vue'
import { topicPages } from '@/data/models.js'
import Tex from '@/components/Tex.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import Prose from '@/components/Prose.vue'
import { ref, reactive, computed, watch } from 'vue'
import { variants } from '@/data/attention.js'
import { learningPath } from '@/data/models.js'
import { clamp } from '@/utils/labmath.js'
import CodeBlock from '@/components/CodeBlock.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import AttnMaskLab from '@/components/labs/AttnMaskLab.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'attention')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 「N 层合计」用的层数: LLaMA-2-7B 是 32 层, 正文里的 512 KiB = 16 KiB × 32
const LAYERS = 32

const evoSteps = [
  { name: 'MHA', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) 多头注意力将特征空间拆分为多个子空间并行关注不同模式, 每个头均需维护独立的 K 与 V 投影。',
    fix: '各查询头独立投影并缓存专属 K 与 V, 表达能力最为充分, 但 KV cache 开销也最大; 早期短序列下尚不显著, 却成为八年后长上下文推理的首要瓶颈。' },
  { name: 'MQA / GQA', year: '2019 / 2023', color: 'var(--accent)',
    pain: '单层缓存为 $2\\cdot T\\cdot d_{\\text{model}}$ 个元素 ($T$ 个 token、K/V 各 $d_{\\text{model}}$ 维), 上下文翻倍则显存翻倍。',
    fix: '令每组查询头共享同一对 K/V, 仅保留 num_kv_heads 组缓存, 按组内头数成倍压缩显存且精度几乎无损; 本仓库教学配置下每 token 每层由 1024 个元素降至 256 个。' },
  { name: 'MLA', year: 2024, color: 'var(--left)',
    pain: 'KV 头数降至 1 即触及跨头共享下限, 缓存的仍是完整 K 与 V 向量, 无法在头数维度继续压缩。',
    fix: '改存可还原 K/V 的低维潜在向量 c_kv 与多头共享的旋转位置向量 k_rope, 计算时再经上投影恢复 K 与 V; 教学配置下每 token 每层由 256 个元素降至 96 个。' },
  { name: 'DSA', year: 2025, color: 'var(--right)',
    pain: 'KV 缓存虽已大幅压缩, 但每个 query 仍需与全部 $T$ 个 key 计算相关性分数, 128K 上下文下计算量按 $T^2$ 增长。',
    fix: '先用轻量打分器 Lightning Indexer 粗评并筛出分数最高的 k 个位置, 仅在这 k 个候选上计算注意力, 将计算复杂度由 $O(T^2)$ 降至 $O(T\\cdot k)$。' },
]

const contextPresets = [
  { id: 'chat', label: '日常对话 · 4K', hint: '约 10–20 页文本', T: 4096 },
  { id: 'document', label: '长文档 · 32K', hint: '整篇论文或代码库片段', T: 32768 },
  { id: 'long-context', label: '超长上下文 · 128K', hint: '显存与计算差异显著放大', T: 131072 },
]

const current = ref(variants[0])

const p = reactive({
  T: 4096,
  d_model: 4096,
  n_heads: 32,
  num_kv_heads: 8,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  sparse_top_k: 512,
})

const applyContextPreset = (preset) => {
  p.T = preset.T
}

const activePresetId = computed(() =>
  contextPresets.find((preset) => preset.T === p.T)?.id || 'custom'
)
const activePresetLabel = computed(() =>
  contextPresets.find((preset) => preset.id === activePresetId.value)?.label || '自定义上下文'
)

// 保证 num_kv_heads ≤ n_heads 且可整除
watch(() => p.n_heads, () => {
  if (p.num_kv_heads > p.n_heads) p.num_kv_heads = p.n_heads
  // 简单兼容: 找到最近的约数
  while (p.n_heads % p.num_kv_heads !== 0 && p.num_kv_heads > 1) p.num_kv_heads--
})

const cacheBytes = (v) => v.cache({
  T: p.T,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2  // fp16

const cacheBytesPerToken = (v) => v.cache({
  T: 1,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2

const mhaBytes = computed(() => variants[0].cache({ T: p.T, d_model: p.d_model, n_heads: p.n_heads }) * 2)
const maxBytes = computed(() => Math.max(...variants.map(cacheBytes)))
const barPct = (v) => (cacheBytes(v) / maxBytes.value) * 100
const relToMHA = (v) => {
  const ratio = cacheBytes(v) / mhaBytes.value
  if (ratio < 0.001) return (ratio * 1000).toFixed(2) + '‰'
  return (ratio * 100).toFixed(1) + '%'
}

const flopsPerQuery = computed(() => {
  if (current.value.id === 'dsa') return p.T * Math.min(p.sparse_top_k, p.T)
  return p.T * p.T
})

const conceptStory = computed(() => {
  const shared = {
    input: `${formatT(p.T)} tokens × ${p.d_model} 维`,
    inputNote: `每个 token 表示为 ${p.d_model} 维向量; 本层设 ${p.n_heads} 个查询头, 各头在对应子空间内独立计算注意力。切换方案时上下文长度与模型宽度保持不变。`,
    outcome: `${formatBytes(cacheBytes(current.value))} / 层`,
    outcomeNote: `每新增 1 个生成 token, 本层需向缓存追加写入 ${formatBytes(cacheBytesPerToken(current.value))}; 整段上下文缓存体积约为 MHA 基线的 ${relToMHA(current.value)}。`,
  }

  const stories = {
    mha: {
      mechanism: `${p.n_heads} 个查询头各自维护独立的 K/V`,
      mechanismNote: '每个查询头均通过独立投影生成专属的 K 与 V, 不做跨头共享或低秩压缩; 后续三种方案的缓存占比均以 MHA 为 100% 基准。',
      takeaway: 'MHA 具备最充分的多头独立表征能力, 但 KV cache 容量随上下文长度呈严格线性增长。',
    },
    gqa: {
      mechanism: `${p.n_heads} 个查询头分组共享 ${p.num_kv_heads} 组 K/V`,
      mechanismNote: `每 ${Math.max(1, p.n_heads / p.num_kv_heads).toFixed(0)} 个查询头划分为一组并共享同一对 K/V, 单层仅需缓存 ${p.num_kv_heads} 组 K/V; 查询头数量仍为 ${p.n_heads}, 注意力主干计算保持不变。`,
      takeaway: 'GQA 仅裁减冗余的多头 K/V 副本, 在保持序列长度与注意力主干公式不变的前提下成倍降低缓存体积。',
    },
    mla: {
      mechanism: `将 K/V 压缩为 ${p.kv_lora_rank} 维潜在向量`,
      mechanismNote: `不再直接缓存完整的 K 与 V, 而是缓存 ${p.kv_lora_rank} 维低秩潜在向量 c_kv 以及所有头共享的 ${p.qk_rope_head_dim} 维旋转位置向量 k_rope; 参与注意力计算时再经上投影矩阵还原各头的 K 与 V。`,
      takeaway: 'MLA 将「缓存完整 K/V 张量」替换为「缓存可还原 K/V 的低秩潜在向量」, 以可控的投影计算换取大幅显存压缩。',
    },
    dsa: {
      mechanism: `保留 MLA 低秩缓存, 每个 query 仅筛选 ${Math.min(p.sparse_top_k, p.T)} 个候选位置`,
      mechanismNote: '缓存结构与 MLA 完全一致, 所有历史位置的压缩状态均需保留以备后续查询检索; 节省的是注意力打分规模, 每个 query 参与计算的历史位置数由全部 T 个缩减至 k 个。',
      takeaway: 'DSA 实现正交的双重开销削减: 底层依赖 MLA 压缩 KV 显存, 上层通过 top-k 稀疏筛选降低超长上下文的计算复杂度, 但不进一步缩减缓存体积。',
    },
  }

  return { ...shared, ...stories[current.value.id] }
})

const formatT = (t) => t >= 1024 ? (t / 1024).toFixed(t % 1024 === 0 ? 0 : 1) + 'K' : String(t)
const formatBytes = (b) => {
  if (b < 1024) return b.toFixed(0) + ' B'
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KiB'
  if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(2) + ' MiB'
  return (b / 1024 / 1024 / 1024).toFixed(2) + ' GiB'
}
const formatFlops = (f) => {
  if (f < 1e6) return f.toFixed(0)
  if (f < 1e9) return (f / 1e6).toFixed(1) + 'M'
  return (f / 1e9).toFixed(2) + 'G'
}

// --- Attention 矩阵可视化 ---
const N = 24  // 显示用的小矩阵
const matrixSize = 320
const cellSize = matrixSize / N

const matrixCells = computed(() => {
  const cells = []
  for (let i = 0; i < N; i++) {
    // 每行的分数: 近距离 + 一些随机热点, 因果 mask 使 j > i 为 0
    const scores = new Array(N).fill(0)
    for (let j = 0; j <= i; j++) {
      const dist = i - j
      // 模拟真实 attention: 相邻衰减 + 几个高分
      scores[j] = Math.exp(-dist * 0.12) + 0.25 * Math.sin(j * 0.7 + i * 0.3) + 0.2
      scores[j] = Math.max(0, scores[j])
    }
    // 对当前 variant 做处理
    if (current.value.id === 'dsa') {
      const k = Math.min(p.sparse_top_k >> 5, Math.min(8, i + 1))  // 缩放到小矩阵
      // 保留 top-k 最大的
      const top = [...scores.map((s, idx) => ({ s, idx }))].sort((a, b) => b.s - a.s).slice(0, k)
      const keep = new Set(top.map(x => x.idx))
      for (let j = 0; j < N; j++) if (!keep.has(j)) scores[j] = 0
    }
    // 归一化 (softmax 简化)
    const sum = scores.reduce((a, b) => a + b, 0) || 1
    for (let j = 0; j < N; j++) {
      const w = scores[j] / sum
      cells.push({ i, j, color: colorFor(w * 2.5) })  // 放大可视度
    }
  }
  return cells
})

const matrixReading = computed(() => current.value.id === 'dsa'
  ? '读图说明: 右上三角因因果掩码屏蔽未来位置而保持空白; 左下三角仅保留 indexer 筛选出的 top-k 个高分候选位置, 其余历史位置在进入 softmax 前已被剔除。'
  : '读图说明: 右上三角因因果掩码屏蔽未来位置而保持空白; 左下三角色块越深代表当前 query 分配给对应历史位置的注意力权重越高, 对角线附近颜色最深源于合成数据赋予相邻位置更高分数。'
)

const matrixAriaLabel = computed(() =>
  `${current.value.name} 的 ${N}×${N} 因果注意力矩阵。${matrixReading.value}`
)

// 权重 0 (含被 mask 的格子) 用卡片底色, 越大越接近强调色。两个端点都是主题变量, 明暗主题都不刺眼
function colorFor(w) {
  return `color-mix(in srgb, var(--accent) ${Math.round(clamp(w, 0, 1) * 100)}%, var(--bg-card))`
}

// --- 代码片段 ---
const classFor = (id) => ({
  mha: 'MultiHeadAttention', gqa: 'GroupedQueryAttention',
  mla: 'MultiHeadLatentAttention', dsa: 'MultiHeadLatentSparseAttention',
}[id])

const codeSnippet = computed(() => {
  const snippets = {
    mha: `class MultiHeadAttention(nn.Module):
    """原始 MHA (Vaswani et al., 2017)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_head = d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, self.d_head)
            for _ in range(num_heads)
        ])
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None):
        # KV cache: num_heads · d_head = d_model (满)
        outs = [h(q, k, v, mask) for h in self.heads]
        return self.w_o(torch.cat(outs, dim=-1))`,
    gqa: `class GroupedQueryAttention(nn.Module):
    """GQA — Ainslie et al., 2023 (LLaMA-2 70B)"""
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_groups = num_heads // num_kv_heads  # 每组 Q 共享 1 对 KV
        self.w_q = nn.Linear(d_model, num_heads    * head_dim)
        self.w_k = nn.Linear(d_model, num_kv_heads * head_dim)  # ← 更小
        self.w_v = nn.Linear(d_model, num_kv_heads * head_dim)

    def forward(self, q, k=None, v=None, mask=None, rope=None):
        Q = self.w_q(q).view(B, T, num_heads,    head_dim)
        K = self.w_k(k).view(B, T, num_kv_heads, head_dim)
        # 复制 num_groups 份, 对齐 Q
        K = K.repeat_interleave(self.num_groups, dim=1)
        # ... 标准 QK^T · V`,
    mla: `class MultiHeadLatentAttention(nn.Module):
    """MLA — DeepSeek-V2/V3, KV 低秩 + 解耦 RoPE"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 qk_nope_head_dim=64, qk_rope_head_dim=32):
        # KV 一次压成低秩 c_kv + 共享 k_rope
        self.kv_down = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim)
        self.k_up = nn.Linear(kv_lora_rank, num_heads * qk_nope_head_dim)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * v_head_dim)
        # Q 也可低秩 (DeepSeek-V3 省训练显存)

    def forward(self, x, rope=None):
        kv_mix = self.kv_down(x)                    # [B, T, lora + rope]
        c_kv, k_rope = torch.split(kv_mix, [lora, rope], dim=-1)
        # nope 段走 latent, rope 段独立旋转
        # 推理: 只缓存 c_kv + k_rope, 约 MHA 的 7%`,
    dsa: `class MultiHeadLatentSparseAttention(nn.Module):
    """DSA — DeepSeek-V3.2 (2025), MLA + Lightning Indexer"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 indexer_heads=4, sparse_top_k=128):
        self.mla = MultiHeadLatentAttention(...)
        self.indexer = LightningIndexer(d_model, indexer_heads)
        self.sparse_top_k = sparse_top_k

    def forward(self, q, mask=None, rope=None):
        # 1) 廉价 indexer 打分
        idx_scores = self.indexer(q, mask=mask)       # [B, T, S]
        # 2) 先按 mask 屏蔽不可见, 再取 top-k (否则可能泄漏未来)
        sparse_mask = top_k_mask(idx_scores, mask, self.sparse_top_k)
        # 3) MLA 只在 top-k 位置算 softmax
        return self.mla(q, mask=sparse_mask, rope=rope)`
  }
  return snippets[current.value.id]
})

</script>

<style scoped>
.concept-workbench {
  margin-bottom: 18px;
  border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--accent) 11%, transparent), transparent 48%),
    var(--bg-card);
  box-shadow: 0 14px 36px color-mix(in srgb, var(--accent) 10%, transparent);
}
.workbench-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.eyebrow {
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.1px;
}
.workbench-heading h2,
.bridge-heading h2 {
  margin: 0;
  font-size: 20px;
  line-height: 1.35;
  text-wrap: balance;
}
.workbench-heading p,
.bridge-heading p,
.bridge-heading .prose {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.7;
  text-wrap: pretty;
}
.learning-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
  list-style: none;
}
.learning-steps li {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 0 10px;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--bg-elev) 82%, transparent);
}
.learning-steps li > span {
  grid-row: 1 / 3;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 11px;
  font-weight: 700;
}
.learning-steps strong { font-size: 12.5px; }
.learning-steps small { color: var(--text-muted); font-size: 10.5px; }
.context-presets { border: 0; }
.context-presets legend {
  margin-bottom: 8px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.preset-grid button {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  min-height: 58px;
  padding: 9px 12px;
  text-align: left;
}
.preset-grid button span { font-size: 12.5px; font-weight: 600; }
.preset-grid button small { margin-top: 2px; color: var(--text-muted); font-size: 10.5px; }
.preset-grid button.active small { color: rgba(255, 255, 255, 0.76); }
.preset-readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 10px;
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  background: var(--code-bg);
  color: var(--text-muted);
  font-size: 12px;
}
.preset-readout strong { color: var(--text); }
.readout-kicker {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
}

.variant-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
  padding-bottom: 4px;
}
.variant-tabs button {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 8px;
  align-items: center;
  min-height: 48px;
  padding: 8px 10px;
  text-align: left;
}
.variant-index {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--bg-card);
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
}
.variant-name { font-weight: 650; }
.variant-tabs button .yr {
  font-size: 10px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
}
.variant-tabs button.active .yr { color: rgba(255,255,255,0.75); }
.variant-tabs button.active .variant-index { background: rgba(255, 255, 255, 0.18); color: #fff; }

.concept-bridge {
  --variant-color: var(--accent);
  margin-bottom: 20px;
  padding: 18px 20px;
  border: 1px solid color-mix(in srgb, var(--variant-color) 44%, var(--border));
  border-top: 3px solid var(--variant-color);
  border-radius: var(--radius);
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--variant-color) 8%, var(--bg-card)),
    var(--bg-card) 55%
  );
}
.bridge-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 4px 18px;
}
.bridge-heading .eyebrow { grid-row: 1 / 3; color: var(--variant-color); }
.causal-flow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
  margin-top: 16px;
}
.causal-flow article {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-elev);
}
.causal-flow article.mechanism { border-color: color-mix(in srgb, var(--variant-color) 52%, var(--border)); }
.causal-flow article.outcome { background: color-mix(in srgb, var(--variant-color) 7%, var(--bg-elev)); }
.causal-label {
  display: block;
  margin-bottom: 4px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.8px;
}
.causal-flow strong { display: block; font-size: 13px; text-wrap: balance; }
.causal-flow p { margin-top: 5px; color: var(--text-muted); font-size: 11.5px; line-height: 1.6; text-wrap: pretty; }
.causal-arrow { align-self: center; color: var(--variant-color); font-size: 18px; }
.concept-takeaway {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 12px;
  align-items: baseline;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
}
.concept-takeaway span { color: var(--variant-color); font-size: 10px; font-weight: 700; letter-spacing: 0.7px; }
.concept-takeaway strong { font-size: 12.5px; line-height: 1.7; text-wrap: pretty; }

.formula-box {
  margin-top: 16px;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--code-bg);
  color: var(--code-text);
  font-size: 13.5px;
  line-height: 1.6;
  overflow-x: auto;
}
.parameter-feedback {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}
.parameter-feedback strong { color: var(--text); font-variant-numeric: tabular-nums; }

.trade { font-size: 13px; color: var(--text-muted); }
.tr-item { display: flex; gap: 8px; align-items: flex-start; margin-top: 4px; }
.tr-item .ok { color: var(--left); }
.tr-item .no { color: var(--danger); }

.bars { display: flex; flex-direction: column; gap: 12px; }
.bar-row {
  display: grid;
  grid-template-columns: 110px 1fr 70px;
  gap: 12px;
  align-items: center;
  transition-property: margin, padding, background-color;
  transition-duration: 200ms;
  transition-timing-function: ease-out;
}
.bar-row.current { background: var(--bg-elev); margin: -6px -10px; padding: 6px 10px; border-radius: 6px; }
.bar-label { display: flex; flex-direction: column; gap: 2px; }
.bar-label .bar-name { font-size: 13px; font-weight: 500; }
.bar-label .bar-sub { font-size: 11px; color: var(--text-muted); }
.bar-track {
  height: 22px;
  background: var(--bg-elev);
  border-radius: 4px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 4px;
}
.bar-rel { font-size: 12px; color: var(--text-muted); text-align: right; font-variant-numeric: tabular-nums; }

.hero-stat {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 20px;
}

.matrix-grid {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.legend-attn { max-width: 320px; }
.legend-strip {
  display: flex;
  height: 14px;
  border-radius: 3px;
  overflow: hidden;
}
.legend-strip span { flex: 1; }
.legend-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
}
.matrix-reading {
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}

@media (max-width: 960px) {
  .workbench-heading,
  .bridge-heading { grid-template-columns: 1fr; }
  .bridge-heading .eyebrow { grid-row: auto; }
  .learning-steps,
  .preset-grid { grid-template-columns: 1fr; }
  .causal-flow {
    grid-template-columns: 1fr;
  }
  .causal-arrow { justify-self: center; transform: rotate(90deg); }
}

@media (max-width: 720px) {
  .variant-tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .concept-workbench,
  .concept-bridge { padding: 16px; }
  .concept-takeaway { grid-template-columns: 1fr; gap: 4px; }
  .bar-row { grid-template-columns: 88px minmax(90px, 1fr) 58px; gap: 8px; }
  .hero-stat { grid-template-columns: 1fr; }
}
</style>
