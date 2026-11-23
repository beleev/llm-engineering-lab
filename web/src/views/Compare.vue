<template>
  <div>
    <h1 class="page-title">总览对照表</h1>
    <p class="page-subtitle">
      本页汇总主流模型架构的核心组件配置, 并将全仓库六个子项目放回同一条工程演进主线: 原理闭环、架构演进、规模化训练、任务适配、推理服务与 Agent 应用层。
    </p>

    <ChapterIntro
      tldr="本页对照表提供两个维度的全局检索视角:
        - 纵向对比模型配置表: 梳理注意力、前馈网络、归一化与位置编码在各代模型中的演进脉络。
        - 横向串联六阶段工程表: 定位大模型系统从底层算子到上层智能体闭环的职责边界与代码入口。"
      question="面对一个真实大模型系统的性能或效果瓶颈, 如何快速判断问题出在模型结构、分布式训练、后训练对齐、推理引擎还是 Agent 运行时层?"
      :goals="[
        `横向检索 ${timeline.length} 种代表性模型的组件配置 (attn / ffn / norm / pos)`,
        '梳理各核心组件从早期设计向现代主流方案演进的技术脉络',
        '建立六阶段工程闭环的全局索引, 明确不同层级问题对应的源码目录',
      ]"
      :codes="[
        { path: 'llm_basic/' },
        { path: 'llm_models/' },
        { path: 'llm_train/' },
        { path: 'llm_finetune/' },
        { path: 'llm_infer/' },
        { path: 'llm_agent/' },
      ]"
      :prereq="prevChapter"
    />

    <!-- 过滤器 -->
    <div class="btn-group" style="margin-bottom: 16px;" role="group" aria-label="按主线筛选模型">
      <button type="button" :class="{ active: trackFilter === null }" :aria-pressed="trackFilter === null" @click="trackFilter = null">全部</button>
      <button v-for="(t, k) in tracks" :key="k" type="button"
              :class="{ active: trackFilter === k }"
              :aria-pressed="trackFilter === k"
              @click="trackFilter = k">
        <span class="dot" :style="{ background: t.color }"></span>
        {{ trackName(k) }}
      </button>
    </div>

    <div class="card" style="padding: 0; overflow-x: auto;">
      <table class="compare-table">
        <thead>
          <tr>
            <th>模型</th>
            <th>年份</th>
            <th>类型</th>
            <th>注意力</th>
            <th>FFN</th>
            <th>归一化</th>
            <th>位置编码</th>
            <th>源码</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in filtered" :key="m.id"
              :class="{ highlight: focusId === m.id }">
            <td>
              <span :class="['pill', tracks[m.track].cls]" style="font-size: 10px;">{{ trackName(m.track) }}</span>
              <strong>{{ m.name }}</strong>
            </td>
            <td class="mono muted">{{ m.year }}</td>
            <td class="muted small">{{ m.kind }}</td>
            <td><code class="inline">{{ m.parts.attn }}</code></td>
            <td><code class="inline">{{ m.parts.ffn }}</code></td>
            <td><code class="inline">{{ m.parts.norm }}</code></td>
            <td><code class="inline">{{ m.parts.pos }}</code></td>
            <td>
              <RepoLink :path="modelFilePath(m.file)" tiny />
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <section class="section">
      <h2>核心组件演进对照</h2>
      <p class="lead">下表按模块维度归纳了从早期标准实现到现代主流方案的架构取舍与对应源码位置。</p>
      <div class="card">
        <table class="compare-table tradeoffs">
          <thead>
            <tr>
              <th>维度</th>
              <th>早期 (2017–2020)</th>
              <th>现代 (2023–2025)</th>
              <th>代码位置</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in tradeoffs" :key="row.topic">
              <td class="row-label">{{ row.topic }}</td>
              <td><code class="inline">{{ row.early }}</code></td>
              <td><code class="inline">{{ row.modern }}</code></td>
              <td>
                <span class="files-cell">
                  <RepoLink v-for="p in splitFiles(row.file)" :key="p" :path="prefixModels(p)" tiny />
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>六目录工程闭环</h2>
      <p class="lead">
        每个子目录各自聚焦一类核心工程问题; 按序推进时, 后一阶段均在前一阶段抽象之上引入更贴近生产的系统约束。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="compare-table">
          <thead>
            <tr>
              <th>目录</th>
              <th>核心问题</th>
              <th>关键联系</th>
              <th>入口代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in projectRows" :key="row.code">
              <td class="row-label">
                <RepoLink :path="row.code" tiny variant="plain" />
              </td>
              <td>{{ row.question }}</td>
              <td>{{ row.link }}</td>
              <td>
                <span class="files-cell">
                  <RepoLink v-for="p in splitFiles(row.file)" :key="p" :path="prefixForCode(row.code, p)" tiny />
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>完整学习路径 · 按主线回顾</h2>
      <p class="lead">以下卡片按六阶段主线提供章节入口, 分别对应仓库中的六个核心子目录。</p>
      <div class="grid grid-3">
        <router-link :to="{ name: 'basic' }" class="card chapter-card">
          <div class="chapter-idx">1</div>
          <h3>llm_basic · 最小闭环</h3>
          <p class="desc">纯 NumPy 手写前向、反向传播、Adam 与采样闭环</p>
        </router-link>
        <router-link :to="{ name: 'models' }" class="card chapter-card">
          <div class="chapter-idx">2</div>
          <h3>llm_models · 架构家族</h3>
          <p class="desc">注意力 / 位置编码 / Block / MoE / 扩散等 {{ modelChapters.length }} 章</p>
        </router-link>
        <router-link :to="{ name: 'train' }" class="card chapter-card">
          <div class="chapter-idx">3</div>
          <h3>llm_train · 规模化训练</h3>
          <p class="desc">DDP / TP / PP / ZeRO / AMP / checkpoint 组合闭环</p>
        </router-link>
        <router-link :to="{ name: 'finetune' }" class="card chapter-card">
          <div class="chapter-idx">4</div>
          <h3>llm_finetune · 任务适配</h3>
          <p class="desc">从 SFT、LoRA 家族到 DPO / GRPO / 蒸馏的后训练体系</p>
        </router-link>
        <router-link :to="{ name: 'infer' }" class="card chapter-card">
          <div class="chapter-idx">5</div>
          <h3>llm_infer · 推理优化</h3>
          <p class="desc">KV cache / paged attention / continuous batching / mini-vLLM</p>
        </router-link>
        <router-link :to="{ name: 'agent' }" class="card chapter-card">
          <div class="chapter-idx">6</div>
          <h3>llm_agent · Agent 应用层</h3>
          <p class="desc">Agent loop / tools / permissions / memory / subagents</p>
        </router-link>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '从智能体应用闭环回到全栈全局索引' }"
      :next="{ name: 'home', label: '返回主线总览', hint: '回到首页查看六阶段全景与学习路径' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { learningPath, modelChapters } from '@/data/models.js'
import { tracks, timeline, trainModules, inferModules, agentModules } from '@/data/modules.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import { splitRefs } from '@/utils/repo.js'

// 把 "a.py + b.py" 形式的多文件 / 含通配符的文件描述拆成可链接片段
const splitFiles = (s) => {
  if (!s) return []
  return splitRefs(s, 'auto').map(r => r.label || r.path)
}

// 模型表里的 m.file 是相对 llm_models/ 的, 这里补回前缀
const modelFilePath = (f) => f.startsWith('llm_') ? f : `llm_models/${f}`

// tradeoffs 表的 file 是相对 llm_models/ 的
const prefixModels = (p) => p.startsWith('llm_') ? p : `llm_models/${p}`

// projectRows 的 file 是相对该 stage code 的目录的 (例如 row.code = 'llm_basic/')
const prefixForCode = (code, p) => {
  if (p.startsWith('llm_')) return p
  return `${code}${p}`
}

// 上一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)。本页是最后一章, 没有下一章
const prevItem = learningPath[learningPath.findIndex((x) => x.route === 'compare') - 1]
const prevChapter = { name: prevItem.route, label: `上一章 · ${prevItem.label}` }

// 'm01..m21' 这样的模块范围从数据里数, 不写死 (模块表里除了 mNN 还有一行 full)
const moduleSpan = (mods) => {
  const n = mods.filter((m) => /^m\d+$/.test(m.id)).length
  return `m01..m${String(n).padStart(2, '0')}`
}

const route = useRoute()
const focusId = ref(route.query.focus || null)
const trackFilter = ref(null)

const filtered = computed(() => {
  const sorted = [...timeline].sort((a, b) => a.year - b.year || a.track.localeCompare(b.track))
  if (!trackFilter.value) return sorted
  return sorted.filter(m => m.track === trackFilter.value)
})

// 三条线的叫法: 语言 / 多模态理解 / 图像视频生成
const trackName = (k) => ({ left: '语言', eye: '多模态理解', right: '图像视频生成' }[k])

const tradeoffs = [
  { topic: '归一化',           early: 'Post-LN (需 warmup)', modern: 'Pre-LN + RMSNorm',
    file: 'layers/core/{blocks,normalization}.py' },
  { topic: 'FFN 激活',         early: 'ReLU',               modern: 'GELU → SwiGLU (门控)',
    file: 'layers/core/feedforward.py' },
  { topic: '位置编码',          early: 'Sinusoidal / Learnable', modern: 'RoPE → M-RoPE',
    file: 'layers/core/position_encoding.py' },
  { topic: 'KV cache',         early: 'MHA (最大)',         modern: 'MQA → GQA → MLA → MLA+DSA',
    file: 'layers/core/attention.py' },
  { topic: 'FFN 结构',         early: '单 FFN',             modern: 'Mixtral MoE → DeepSeekMoE (sigmoid + shared + aux-free)',
    file: 'layers/sparse/moe.py + models/moe/deepseekV3.py' },
  { topic: '序列建模',          early: 'Attention O(T²)',   modern: 'Mamba SSM O(T) / DSA O(T·k)',
    file: 'layers/sparse/ssm.py + layers/core/attention.py' },
  { topic: '多模态融合',        early: 'Cross-attention (Flamingo)', modern: 'Prefix-token (VL) / Thinker + Talker (Omni)',
    file: 'models/multimodal/{qwen2_vl,qwen2_5_omni}.py' },
  { topic: '多模态对齐',        early: 'VSE++ triplet',      modern: 'CLIP 对称对比 + 可学温度',
    file: 'models/multimodal/clip.py' },
  { topic: '扩散骨架',         early: 'UNet (SD 1.5)',      modern: 'DiT + adaLN-Zero (SD3/FLUX/Sora)',
    file: 'models/generative/dit.py + layers/diffusion/adaln.py' },
  { topic: '扩散目标',         early: 'ε-prediction (DDPM)', modern: 'Rectified Flow v-pred (SD3)',
    file: 'training/diffusion.py' },
  { topic: '图像生成范式',      early: '扩散连续去噪',       modern: '自回归 next-token (VAR/LlamaGen)',
    file: 'models/generative/var.py' },
]

const projectRows = [
  {
    code: 'llm_basic/',
    question: 'Transformer 的 forward/backward/update 最小闭环是什么?',
    link: '为后续基于 PyTorch 自动微分的模块提供手写数学基线。',
    file: 'model.py · train.py · optim.py',
  },
  {
    code: 'llm_models/',
    question: '主流模型的差别落在哪些零件上?',
    link: '将基础闭环中的单层单头结构扩展为现代注意力、FFN、归一化与位置编码组件。',
    file: 'layers/core/ · models/',
  },
  {
    code: 'llm_train/',
    question: '同一个训练循环如何扩到多卡并保持等价?',
    link: '围绕数据批次、张量维度、模型层、优化器状态、数值精度与集合通信展开并行切分。',
    file: `${moduleSpan(trainModules)}/demo.py · full_loop/demo.py`,
  },
  {
    code: 'llm_finetune/',
    question: '如何在受限样本与算力预算下使基座模型遵循指令并对齐人类偏好?',
    link: '通过 SFT 调整监督区间, 通过 LoRA 压缩可训练参数, 通过 DPO 等算法简化偏好对齐链路。',
    file: 'methods/{sft,lora,qlora,dpo,grpo,distill}.py',
  },
  {
    code: 'llm_infer/',
    question: '如何将训练完毕的模型部署为低延迟、高吞吐且输出受控的在线推理服务?',
    link: '在自回归解码循环中集成 KV cache、分页显存管理、连续批调度、前缀复用与语法约束。',
    file: `${moduleSpan(inferModules)}/demo.py · full_engine/engine.py`,
  },
  {
    code: 'llm_agent/',
    question: '如何将模型推理服务封装为支持工具执行、状态持久化与子任务隔离的智能体系统?',
    link: '围绕模型生成接入工具调度、权限护栏、上下文管理、记忆检索、生命周期 Hook 与子智能体。',
    file: `${moduleSpan(agentModules)}/demo.py · full_loop/demo.py`,
  },
]
</script>

<style scoped>
.files-cell {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

table.compare-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.compare-table thead {
  background: var(--bg-elev);
}
table.compare-table th {
  text-align: left;
  padding: 12px 14px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.compare-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
table.compare-table tbody tr:hover {
  background: var(--bg-elev);
}
table.compare-table tbody tr.highlight {
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}
table.compare-table tbody tr.highlight td:first-child {
  border-left: 2px solid var(--accent);
}
table.compare-table strong {
  margin-left: 8px;
  color: var(--text);
}
table.compare-table .muted { color: var(--text-muted); }
table.compare-table .small { font-size: 12px; }
table.compare-table .row-label { color: var(--text-muted); }

.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}

.chapter-card {
  display: block;
  text-decoration: none;
  color: inherit;
  position: relative;
  transition: border-color 150ms ease-out, transform 150ms ease-out, box-shadow 150ms ease-out;
}
.chapter-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  text-decoration: none;
}
.chapter-idx {
  position: absolute;
  top: 14px; right: 16px;
  width: 26px; height: 26px;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 600;
  font-family: "SF Mono", Menlo, monospace;
}
.chapter-card h3 { padding-right: 40px; margin-bottom: 4px; }
.chapter-card.disabled {
  border-style: dashed;
  cursor: not-allowed;
  opacity: 0.55;
}
.chapter-card.disabled:hover {
  border-color: var(--border);
  transform: none;
}
.chapter-card.disabled .chapter-idx {
  background: var(--bg-elev);
  color: var(--text-dim);
}
</style>
