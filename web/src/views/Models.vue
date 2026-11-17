<template>
  <div>
    <h1 class="page-title">常见模型结构 · 把主流模型拆成四类零件</h1>
    <div class="page-subtitle lead-group">
      <p>阶段 1 在 <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 里手写了每一个梯度, 原理那一关已经过了。</p>
      <p>
        阶段 2 进 <RepoLink path="llm_models/" label="llm_models/" tiny /> 换成 PyTorch,
        把 attention、FFN、norm、position 四种零件装进同一个 Block, 堆出 Transformer、MoE、多模态和扩散模型。
      </p>
      <p><strong>换一个零件只需要改前向, 反向交给 autograd (PyTorch 的自动求导)</strong>。所以一个仓库里放得下二十多种架构。</p>
    </div>

    <ChapterIntro
      tldr="主流模型的差别几乎都落在 4 个槽位上, 其它部分大同小异:
        - attention (注意力): 生成时每个 token 要缓存多少 K/V (这份缓存叫 KV cache), 各家都在想办法把这个数砍小
        - FFN (前馈网络): 要不要加门控 (SwiGLU), 要不要拆成多个专家、每个 token 只激活其中几个 (MoE)
        - norm (归一化): 放在子层前还是后
        - 位置信息: 加在 embedding 上, 还是乘在 Q/K 上 (RoPE)"
      question="拿到没读过的模型, 能不能先认出它只换了哪几个零件, 而不是从头读整套架构?"
      :goals="[
        '用同一张零件表读懂 Transformer / LLaMA / Mixtral / DeepSeek',
        '把多模态和扩散模型放回「先 token 化, 再过 Transformer block」这条主线',
        '知道每个小标题该对着哪一组源码看',
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
      title="演进逻辑链 · 每次换架构, 都是被一种成本逼的"
      subtitle="根问题: 算力、显存和数据都有限, 模型却要吃更多文本、读更长上下文、生成得更快"
      :steps="evoSteps"
    />

    <ModelArchitectureLab />
    <p class="lead">
      图像和音频怎样切成 token、从哪一扇门进语言模型, 单独成章:
      <router-link :to="{ name: 'models-multimodal' }">图像与音频怎样接入语言模型</router-link>。
      那一章带着 CLIP / Qwen2-VL / Whisper / Omni 四套模型的阅读实验台。
    </p>

    <section class="section">
      <h2>阶段 2 的阅读顺序</h2>
      <div class="lead-group">
        <ol class="pts">
          <li>
            <b>注意力:</b> 先读它, 因为它直接决定长上下文的推理成本。生成时每个 token 的 K 和 V 都要缓存下来,
            一个 token 在全部层上的 KV 占多少显存, 各代差很多:
            <ul class="pts">
              <li>LLaMA-2-7B (MHA) 要 512 KiB。</li>
              <li>LLaMA-3-8B 换成 GQA 后 128 KiB。</li>
              <li>DeepSeek-V3 换成 MLA 后 68.6 KiB。</li>
            </ul>
          </li>
          <li>
            <b>位置编码和 Block 组装:</b>
            <ul class="pts">
              <li><b>位置编码:</b> 告诉模型每个 token 排在第几位。</li>
              <li><b>Block 组装:</b> 决定四种零件怎么接成一层。</li>
            </ul>
          </li>
          <li>
            <b>两条分支:</b>
            <ul class="pts">
              <li><b>MoE:</b> 把 FFN 拆成多个专家, 每个 token 只用其中几个。</li>
              <li><b>扩散模型:</b> 换掉「一个 token 一个 token 生成」这套方式, 改成从噪声逐步去噪。</li>
            </ul>
          </li>
        </ol>
        <p>每一章都按同一个顺序讲:</p>
        <ol class="pts">
          <li>上一代的痛点。</li>
          <li>这一代的解法。</li>
          <li>解法引出的新问题。</li>
        </ol>
      </div>

    </section>

    <section class="section">
      <h2>四个源码入口</h2>
      <p class="lead">读一个模型时先来这四个文件认槽位。认完再去看它自己的实现文件, 一般只剩几十行。</p>
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
        时间线上这 {{ timeline.length }} 个模型共用同一套元数据, 源码列直接跳到 GitHub。
        按年份读一遍: 新模型几乎从不从零开始, 都是在上一个的某个槽位上改。
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

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '从最小闭环进入现代模型结构' }"
      :next="{ ...nextChapter, hint: '先拆 KV cache 成本最高的 attention 槽位' }"
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
import { timeline, learningPath } from '@/data/models.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'models')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 是上一步 fix 留下的窟窿
const evoSteps = [
  { name: 'Transformer', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) RNN 逐 token 串行, 训练没法并行, 远处信息要一步步传过来',
    fix: '全靠注意力: 整段并行训练, 任意两个位置一步直达; Encoder-Decoder 只吃成对的翻译数据' },
  { name: 'GPT', year: 2018, color: 'var(--accent)',
    pain: '成对数据有限, 海量无标注文本用不上',
    fix: '只留 Decoder + 因果 mask, 预测下一个 token: 任何文本都是训练数据, 能力随参数量涨' },
  { name: 'MoE', year: 2024, color: 'var(--left)',
    pain: '稠密模型每个 token 都过全部参数, 容量翻倍, 算力也翻倍',
    fix: 'Mixtral、DeepSeek-V3 把 FFN 拆成 E 个专家, 每 token 只过 k 个: 算力按 k 算' },
  { name: 'SSM / 混合线性', year: '2023 / 2025', color: 'var(--right)',
    pain: '注意力仍随上下文长度 T 变贵: cache 线性涨, 算力 $O(T^2)$',
    fix: 'Mamba、Qwen3-Next 用一个定长状态代替越来越长的 KV cache, 解码每步 $O(1)$; 状态装不下的信息会丢, 所以留几层全注意力兜底' },
  { name: '非自回归', year: '2024 / 2025', color: 'var(--eye)',
    pain: 'cache 再小, 自回归也一次只出 1 个 token, N 个要串行 N 步',
    fix: 'VAR 一步出一整级分辨率, LLaDA 从全 [MASK] 逐步去噪: 一次前向定一批 token' },
]

const sourceRows = [
  { slot: 'Attention', problem: '生成时每个 token 要缓存多少 K/V: 从 MHA 到 GQA、MLA、DSA, 每一代都在砍这个数', file: 'llm_models/layers/core/attention.py' },
  { slot: 'Position', problem: '位置信息从哪进模型: Sin / Learnable 加在 embedding 上, RoPE / M-RoPE 乘在 Q/K 上', file: 'llm_models/layers/core/position_encoding.py' },
  { slot: 'Block', problem: '把 attn、ffn、norm、pos 装进同一个壳: 所有模型共用的那段控制流', file: 'llm_models/layers/core/blocks.py' },
  { slot: 'Sparse / Generative', problem: '主干之外的扩展: MoE 把 FFN 拆稀疏, SSM 换掉注意力, DiT / MM-DiT / VAR 换掉生成方式', file: 'llm_models/layers/sparse/' },
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
