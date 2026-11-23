<template>
  <div>
    <h1 class="page-title">常见模型结构 · 将主流模型拆解为四类组件</h1>
    <div class="page-subtitle lead-group">
      <p>阶段 1 已在 <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 中手写了全部算子的前向与反向梯度, 完成了底层原理闭环。</p>
      <p>
        阶段 2 在 <RepoLink path="llm_models/" label="llm_models/" tiny /> 中切换至 PyTorch,
        将 attention、FFN、norm 与 position 四类组件组装进统一的 Block, 进而堆叠出 Transformer、MoE、多模态与扩散模型。
      </p>
      <p><strong>替换任意组件只需修改前向计算逻辑, 反向传播交由 PyTorch autograd 自动完成</strong>, 因此同一套代码骨架即可容纳二十多种架构。</p>
    </div>

    <ChapterIntro
      tldr="主流模型架构的差异主要集中在 4 个槽位上, 其余骨架结构基本一致:
        - Attention: 决定生成阶段每个 token 的 KV cache 体积, 各代注意力变体均致力于压缩 K/V 缓存开销
        - FFN: 决定是否引入门控结构 (SwiGLU), 以及是否拆分为多个专家并令每个 token 仅激活其中少数几个 (MoE)
        - Norm: 决定归一化层放置在子层前还是子层后
        - 位置信息: 决定直接叠加在输入 embedding 上, 还是通过旋转变换作用在 Q/K 上 (RoPE)"
      question="面对一个陌生的模型架构, 如何快速定位它替换了哪几个核心槽位, 而不必从头通读整套实现?"
      :goals="[
        '借助统一的四槽位对照表拆解 Transformer / LLaMA / Mixtral / DeepSeek 的结构差异',
        '将多模态与扩散模型纳入「先完成 token 化, 再经由 Transformer block 处理」的统一主线',
        '建立各架构分支与仓库核心源码目录之间的一一对应关系',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/' },
        { path: 'llm_models/models/' },
        { path: 'llm_models/layers/sparse/' },
        { path: 'llm_models/models/generative/' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <StageContents stage-id="models" />

    <EvolutionChain
      title="演进逻辑链 · 每一次架构演进都源于明确的成本瓶颈"
      subtitle="根问题: 在算力、显存与标注数据受限的前提下, 模型需要利用更大规模的语料、处理更长的上下文并提升生成吞吐"
      :steps="evoSteps"
    />

    <ModelArchitectureLab />
    <p class="lead">
      关于图像与音频如何切分为连续或离散 token 并接入语言模型主干, 详见独立章节:
      <router-link :to="{ name: 'models-multimodal' }">图像与音频怎样接入语言模型</router-link>。
      该章配有涵盖 CLIP、Qwen2-VL、Whisper 与 Omni 四套架构的交互阅读实验台。
    </p>

    <section class="section">
      <h2>阶段 2 的阅读顺序</h2>
      <div class="lead-group">
        <ol class="pts">
          <li>
            <b>注意力机制:</b> 建议优先阅读该章, 因为注意力结构直接决定了长上下文场景下的推理显存与带宽成本。自回归生成时需缓存历史 token 的 K 与 V 张量, 单个 token 在全部层上的 KV cache 显存占用在不同架构间差异显著:
            <ul class="pts">
              <li>LLaMA-2-7B 采用标准多头注意力 (MHA), 单 token 占用 512 KiB。</li>
              <li>LLaMA-3-8B 改用分组查询注意力 (GQA) 后, 单 token 降至 128 KiB。</li>
              <li>DeepSeek-V3 引入多头潜在注意力 (MLA) 后, 单 token 进一步压缩至 68.6 KiB。</li>
            </ul>
          </li>
          <li>
            <b>位置编码与 Block 组装:</b>
            <ul class="pts">
              <li><b>位置编码:</b> 为自注意力机制注入每个 token 在序列中的绝对或相对位置信息, 并直接影响长度外推表现。</li>
              <li><b>Block 组装:</b> 定义四种核心组件在单层内部的连接拓扑与残差流向。</li>
            </ul>
          </li>
          <li>
            <b>两条扩展分支:</b>
            <ul class="pts">
              <li><b>MoE:</b> 将稠密 FFN 拆分为多个独立专家网络, 每个 token 仅路由激活其中少数专家, 实现参数容量与单步算力解耦。</li>
              <li><b>扩散模型:</b> 跳出「逐 token 串行自回归生成」的范式, 改为从随机噪声或全掩码状态出发通过多步迭代逐步去噪。</li>
            </ul>
          </li>
        </ol>
        <p>本阶段各章统一遵循相同的演进叙事结构:</p>
        <ol class="pts">
          <li>上一代架构暴露的工程或建模瓶颈。</li>
          <li>当前架构针对该瓶颈提出的结构解法。</li>
          <li>新解法引入的权衡与下一代待解问题。</li>
        </ol>
      </div>

    </section>

    <section class="section">
      <h2>四个源码入口</h2>
      <p class="lead">阅读任意具体模型前, 建议先对照以下四个核心源码文件理清槽位接口; 掌握共享组件后, 单个具体模型的实现文件通常只剩几十行。</p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="models-table">
          <thead>
            <tr>
              <th>槽位</th>
              <th>解决的问题</th>
              <th>源码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in sourceRows" :key="r.slot">
              <td class="slot">{{ r.slot }}</td>
              <td>{{ r.problem }}</td>
              <td><RepoLink :path="r.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>模型族谱</h2>
      <p class="lead">
        时间线上的这 {{ timeline.length }} 个代表性模型共用同一套元数据定义, 右侧源码链接可直接跳转至 GitHub 对应实现。
        顺着年份对比可见, 新一代架构极少从零构建, 通常是在前代主干的特定组件槽位上进行针对性改进。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="models-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>年份</th>
              <th>类型</th>
              <th>源码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in sortedTimeline" :key="m.id">
              <td class="slot">{{ m.name }}</td>
              <td class="mono muted">{{ m.year }}</td>
              <td>{{ m.kind }}</td>
              <td><RepoLink :path="`llm_models/${m.file}`" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 未配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '从手写梯度的最小闭环过渡到现代模块化架构' }"
      :next="{ ...nextChapter, hint: '深入剖析决定长上下文 KV cache 成本的 attention 槽位' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import ModelArchitectureLab from '@/components/labs/ModelArchitectureLab.vue'
import RepoLink from '@/components/RepoLink.vue'
import { learningPath } from '@/data/models.js'
import { timeline } from '@/data/modules.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'models')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 对应上一步 fix 遗留的瓶颈
const evoSteps = [
  { name: 'Transformer', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) RNN 按 token 逐步串行计算, 训练阶段无法并行, 长距离信息在多步传递中容易衰减',
    fix: '完全基于注意力机制实现整段序列并行训练, 任意两个位置一步直达; 但早期 Encoder-Decoder 结构依赖成对平行翻译数据' },
  { name: 'GPT', year: 2018, color: 'var(--accent)',
    pain: '成对平行数据规模有限, 难以直接利用海量无标注文本语料',
    fix: '仅保留 Decoder 与因果 mask 进行下一 token 预测: 任意连续文本均可作为训练数据, 模型能力随参数规模扩展' },
  { name: 'MoE', year: 2024, color: 'var(--left)',
    pain: '稠密模型中每个 token 均需经过全部参数计算, 模型容量翻倍会导致算力开销同步翻倍',
    fix: 'Mixtral 与 DeepSeek-V3 将 FFN 拆分为 E 个专家, 每个 token 仅激活其中 k 个: 单步算力仅按 k 计' },
  { name: 'SSM / 混合线性', year: '2023 / 2025', color: 'var(--right)',
    pain: '标准注意力开销仍随上下文长度 T 增长: KV cache 线性扩张, 注意力计算复杂度为 $O(T^2)$',
    fix: 'Mamba 与 Qwen3-Next 用定长状态替代持续增长的 KV cache, 单步解码复杂度降至 $O(1)$; 因定长状态容量有限易丢失细节, 常混合保留数层全注意力兜底' },
  { name: '非自回归', year: '2024 / 2025', color: 'var(--eye)',
    pain: '即便压缩了 KV cache, 自回归解码每步仍只生成 1 个 token, 生成 N 个 token 需串行执行 N 步',
    fix: 'VAR 单步预测一整级分辨率的视觉 token, LLaDA 从全 [MASK] 序列逐步去噪: 单次前向即可并行确定一批 token' },
]

const sourceRows = [
  { slot: 'Attention', problem: '决定生成阶段每个 token 的 KV cache 规模: 从 MHA 到 GQA、MLA 与 DSA, 持续压缩缓存与带宽开销', file: 'llm_models/layers/core/attention.py' },
  { slot: 'Position', problem: '定义位置信息的注入方式: Sin / Learnable 直接叠加于 embedding, RoPE / M-RoPE 以旋转变换乘在 Q/K 上', file: 'llm_models/layers/core/position_encoding.py' },
  { slot: 'Block', problem: '将 attn、ffn、norm 与 pos 组装进统一模块: 承载各主流架构共享的单层前向控制流', file: 'llm_models/layers/core/blocks.py' },
  { slot: 'Sparse / Generative', problem: '承载稠密主干之外的结构扩展: MoE 实现 FFN 稀疏化, SSM 替代标准注意力, DiT / MM-DiT / VAR 重构生成方式', file: 'llm_models/layers/sparse/' },
]

const sortedTimeline = computed(() =>
  [...timeline].sort((a, b) => a.year - b.year || a.track.localeCompare(b.track))
)
</script>

<style scoped>
table.models-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.models-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.models-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}
table.models-table .slot {
  color: var(--text);
  font-weight: 600;
}
.muted { color: var(--text-muted); }
</style>
