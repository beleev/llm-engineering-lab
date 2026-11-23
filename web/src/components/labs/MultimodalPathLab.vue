<template>
  <LabFrame title="图像与音频怎样接入语言模型 — 四套模型, 两扇门"
            sub="- 模拟什么: 本视图不运行数值模拟, 而是梳理本仓库多模态源码的统一导读路线。四套代表性模型 (CLIP、Qwen2-VL、Whisper、Qwen2.5-Omni) 围绕同一个核心架构问题展开: 连续的图像像素与音频信号并非离散 token, 应如何编码为 Transformer 可处理的向量序列, 又通过何种接口接入语言模型。多模态接入路径主要归纳为两类:
  - 前缀拼接: 将编码后的视觉特征向量直接视作连续 token 拼接到文本 embedding 前方, 语言模型主干无需修改任何结构。
  - 交叉注意力 (cross-attention): 在解码器每层额外插入交叉注意力子层, 以文本隐状态作为 query 去检索多模态编码器的输出表示。
- 图上是什么: 自上而下依次展示四部分导读内容:
  - 顶部加粗数据流: 呈现从原始模态输入 (图像 / mel 声谱 / 文本) 到模型输出的完整张量变换链路, 每个箭头对应一次模块映射。
  - 核心机制说明: 交代该模型的阅读重点、验证脚本运行方式、实测可支撑的结论以及实验无法外推的边界。
  - 关键源码切片: 直接截取自本仓库对应模型实现文件的核心前向代码片段。
  - 底部文件路径: 对应章节本地讲义 README 的仓库相对路径。
- 可以做什么: 点击顶部编号 1 至 4 切换目标模型 (切换操作仅变更导读视图, 不涉及新增模型实现), 建议按如下演进顺序阅读:
  1. CLIP: 仅做跨模态双塔对比对齐, 不包含自回归文本生成。
  2. Qwen2-VL: 将视觉编码特征压缩后作为前缀拼入因果语言模型。
  3. Whisper: 采用编码器-解码器架构, 通过交叉注意力读取音频特征。
  4. Omni: 在 Thinker 主干生成文本的同时, 由独立的 Talker 读取其隐状态并自回归预测离散语音 token。
- 应该看到: 对照右侧两项摘要读数及底部上下文开销说明:
  - 「训练目标」: 标明该模型所优化的损失函数。对比 CE 旨在拉近配对图文向量的余弦相似度并推远非配对样本, next-token CE 则是标准的自回归下一 token 交叉熵。
  - 「输入/输出边界」: 明确本仓库教学实现的截断边界, 例如 Whisper 直接以提取好的 mel 声谱张量作为输入而不含原始波形前端处理, Omni 中的 Talker 仅输出离散 codec token 的 logits 而不含波形还原声码器。
  - 底部注解: 给出单张图像占用上下文窗口长度的换算方法: $224 \times 224$ 分辨率图像按 $16 \times 16$ 划分为 196 个不重叠 patch, 每个 patch 映射为一个视觉 token, 与文本 token 一样参与自注意力计算并占用 KV cache。"
            :module="model.source.split(':')[0]" :run="'python -m llm_models.run_models.multimodal.' + model.id + '.infer_' + model.id"
            :challenge="{
      ask: '请先思考两个模型能力边界问题: CLIP 完成对比训练后配对图文特征相似度显著升高, 它能否直接执行看图说话生成描述文本? 本仓库的 Omni 实现输出了语音 codec logits, 能否直接播放出音频声音?',
      answer: '答案: 两者都不能。\n- CLIP: 双塔编码器分别将整张图像与整句文本压缩为单一全局特征向量, 仅支持计算图文内积相似度 (使配对样本得分高、非配对样本得分低)。由于模型内部不包含因果语言解码器, 缺乏「给定图像特征、逐 token 自回归生成文本」的结构链路, 因此只能用于跨模态检索与零样本分类打分, 无法生成自然语言句子。若要实现看图说话, 需采用 Qwen2-VL 的架构路线: 将视觉特征序列作为前缀送入因果 decoder。\n- Omni: Talker 模块输出的是离散音频编码 token (codec token) 的分类 logits, 仅完成了音频离散符号的预测; 若要还原为可播放的连续音频波形, 还必须在后端接入预训练神经音频声码器 (codec decoder) 执行波形解码, 而本仓库的教学实现未包含声码器模块。\n此外还需注意实验数据边界: 本仓库四套多模态模型的 train 脚本均在单个固定小型随机合成 batch 上运行过拟合验证, 训练损失下降仅证明跨模态前向与梯度反传链路畅通可学, 不代表模型具备真实的视觉理解或语音识别能力。',
    }">
    <template #controls>
      <div class="row" role="group" aria-label="多模态阅读顺序">
        <button v-for="(item, i) in models" :key="item.id" type="button" :class="{ active: chosen === i }"
                :aria-pressed="chosen === i" @click="chosen = i">{{ i + 1 }} · {{ item.title }}</button>
      </div>
    </template>
    <div class="path-content" aria-live="polite">
      <p class="path-flow">{{ model.flow }}</p>
      <Prose :text="model.explanation" />
      <SourceSnippet :key="model.source" :src="model.source" />
      <p class="lab-note">本地配套讲义 (含运行方法和实测数字): llm_models/run_models/multimodal/{{ model.id }}/README.md</p>
    </div>
    <template #stats>
      <div class="kv"><span>训练目标 (loss 是什么)</span><b>{{ model.loss }}</b></div>
      <div class="kv"><span>输入/输出边界 (本仓库做到哪)</span><b>{{ model.boundary }}</b></div>
      <div class="lab-note">
        <p>单张图像的上下文开销换算: 一幅 $224 \times 224$ 像素的图像按 $16 \times 16$ 网格切分为不重叠图像块 (patch), 共得到 $14 \times 14 = 196$ 个 patch (不含 CLS token); 每个 patch 经线性投影后成为一个视觉 token, 与普通文本 token 一样参与全序列注意力计算并占用 KV cache。</p>
        <p>若图像边长翻倍至 $448 \times 448$, patch 数量将增至原来的 4 倍 (784 个); 将视觉特征拼作前缀后, 总序列长度由 $T$ 增至 $T + 196$, 自注意力计算复杂度按 $(T + 196)^2$ 平方级增长。引入视觉 token 压缩模块 (如本仓库 Qwen2-VL 实现所用的 Perceiver Resampler, 将 16 个 patch 重采样压缩为 8 个 latent token) 可显著降低序列长度开销, 代价是压缩后的 latent 向量不再与原图空间网格一一对应, 容易损失密集小字与精细局部特征。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
import Prose from '@/components/Prose.vue'
const chosen = ref(0)
const models = [
  { id: 'clip', title: 'CLIP · 对齐', source: 'llm_models/models/multimodal/clip.py:CLIPModel',
    flow: '图像 → patch → ViT → 归一化向量 ↔ 文本 → 文本编码器 → 归一化向量',
    loss: '双向对比 CE (图找文 + 文找图)', boundary: '只输出向量相似度, 不生成句子',
    explanation: '先读 CLIP。它有两座塔: 图像塔 (ViT, 把图切成 patch 后过 Transformer) 和文本塔, 各自输出一个归一化向量。训练时一个 batch 里的 B 对图文算出 B×B 的相似度矩阵, 对角线上是配对的图文, 其余 B−1 个是免费的负样本; loss 是「图找文」和「文找图」两个方向交叉熵的平均, 目标是让对角线最大。池化 (图像取 CLS, 文本取 EOS 位置) 和投影决定两边怎么进入同一个向量空间。\n看代码时先跑 infer, 检查输出向量的范数是 1、文本确实在 EOS 位置池化; 再跑 train 看配对 loss 从初始 2.180 降下来 (8 对图文随机配对时的理论值是 ln 8 = 2.079, CLIP 的 README 实测 30 步后到 0.0004)。这是固定的随机 batch, 拟合它不代表模型看懂了图文。' },
  { id: 'qwen2_vl', title: 'Qwen2-VL · 前缀', source: 'llm_models/models/multimodal/qwen2_vl.py:Qwen2VLModel',
    flow: '图像 → patch → ViT → resampler/projector → 视觉前缀 + 文本 → LLM',
    loss: '文本位置的 next-token CE (视觉位置不算)', boundary: '图是随机张量, 不是看图基准',
    explanation: '再读视觉条件生成 (第一扇门: 拼前缀)。图切 patch 过 ViT 得到一串视觉向量, 经 resampler 压缩、projector 对齐到语言模型的维度之后, 直接排在文本 token 前面, 同一个因果 decoder 从头读到尾。loss 只在文本位置算, 视觉位置的标签是 −100: 它们是条件, 不是预测目标。\n看代码时检查两件事: 换一张图, 文本位置的 logits 会变 (文本看得到前缀); 换后面的文本, 视觉位置的输出不变 (因果 mask 下前缀看不到后面)。本仓库用固定分辨率和 Perceiver Resampler (16 个 patch 压成 8 个 latent), 和原版 Qwen2-VL 的动态分辨率、相邻 2×2 patch 合并不一样。' },
  { id: 'whisper', title: 'Whisper · 交叉注意力', source: 'llm_models/models/multimodal/whisper.py:Whisper',
    flow: 'mel → 卷积 → 双向音频编码器 → cross-attention → 因果文本 decoder',
    loss: '文本 next-token CE', boundary: '直接输入 mel 声谱, 没做波形前处理',
    explanation: '然后读语音识别 (第二扇门: 交叉注意力)。真实流程里波形要先分帧、做 STFT、过 mel 滤波再取 log, 得到 mel 声谱 [80, 帧数]; 这些前处理本实验没做, 直接构造随机的 mel。音频编码器先用两层卷积沿时间切帧 (第二层 stride=2, 帧数减半), 再用双向注意力读完整段音频。文本 decoder 是因果的, 每一层多一块 cross-attention: 文本当 query, 音频编码器的输出当 K/V, 这是音频进 decoder 的唯一通道, 音频不占 decoder 的序列长度。\n看代码时先看形状断言和「改未来文本不影响过去位置」的断言, 再看 train: 两条样本文本相同、音频不同、标签不同, 只看文本的 loss 下界是 ln 2 = 0.693, 降到它以下说明文本位置读到了音频。loss 下降证明通路可学, 不代表语音识别准确率。' },
  { id: 'qwen2_5_omni', title: 'Omni · Thinker/Talker', source: 'llm_models/models/multimodal/qwen2_5_omni.py:Qwen2_5_OmniModel',
    flow: '图像/音频/文本 → Thinker → 文本 logits / 隐状态 → Talker → codec logits',
    loss: '文本 CE + 0.5 × 语音 CE', boundary: '输出 codec token 分数, 没有真实 codec 解码和声码器',
    explanation: '最后读双输出。Thinker 是一个拼前缀的多模态 decoder (图像、音频、文本都拼进去), 输出文本 logits; Talker 是另一个因果 decoder, 用 cross-attention 读 Thinker 最后一层的隐状态 (不是读它生成的文本, 语气这类信息留在隐状态里), 输出语音 codec token 的 logits。前向是单向的: 语音分支不倒流回文本分支。\n看代码时检查三条: 换图会不会影响语音 logits (图 → Thinker → Talker 这条通路通不通); 改 codec 输入会不会影响文本 logits (应该不会, 前向不走回头路); 改未来的 codec token 会不会影响过去位置的输出 (应该不会, Talker 也是因果的)。输出离散语音 token 的分数不等于得到可播放的波形; 真实的音频 codec、流式对齐和延迟还要单独实现和验收。' },
]
const model = computed(() => models[chosen.value])
</script>

<style scoped>
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.path-content { min-width: 0; }
.path-flow { font-weight: 600; line-height: 1.9; margin-bottom: 12px; }
.kv { gap: 12px; align-items: baseline; }
.kv b { max-width: 65%; }
.lab-note { overflow-wrap: anywhere; }
</style>
