<template>
  <LabFrame title="图像与音频怎样接入语言模型 — 四套模型, 两扇门"
            sub="- 模拟什么: 这个台子不跑模拟, 它是一条阅读路径。四套已有模型 (CLIP、Qwen2-VL、Whisper、Qwen2.5-Omni) 都在回答同一个问题: 图像和声音不是 token, 怎么变成 Transformer 认的向量, 又从哪一扇门进语言模型。答案只有两扇门: 拼前缀 (把视觉向量当 token 排在文本前面, 语言模型本身一行不改) 和交叉注意力 (cross-attention: 语言模型每层多一块注意力, 用文本当 query 去读编码器的输出)。
      - 图上是什么: 第一行加粗的是数据流: 从原始输入 (图像 / mel 声谱 / 文本) 一路到输出, 每个箭头是一次变换。下面几段文字说这套模型先看什么、再跑什么、哪些结论能下、哪些不能。再下面是这个模型的源码片段 (直接从仓库文件里截的), 最后一行是本地讲义 README 的路径。
      - 可以做什么: 点 1 到 4 四个按钮切换模型, 建议按编号顺序读: 1 CLIP 只对齐、不生成; 2 Qwen2-VL 把图当前缀; 3 Whisper 用交叉注意力读音频; 4 Omni 在 Thinker 出文本之后, 再用一个 Talker 读它的隐状态出语音 token。切换只换阅读入口, 不会新增模型实现。
      - 应该看到: 右侧「训练目标」是这套模型的 loss 是什么: 对比 CE 是让配对的图文向量相似度比不配对的高, next-token CE 是预测下一个文本 token。「输入/输出边界」提醒这个仓库里的实现到哪为止, 比如 Whisper 直接吃 mel 声谱、不做波形前处理, Omni 输出的是 codec token 的分数、没有声码器。下面的注解给了「一张图占多少上下文」的算法: 224×224 的图按 16×16 切成 196 个 patch, 每个 patch 就是一个 token, 要过注意力也要存 KV。"
            :module="model.source.split(':')[0]" :run="'python -m llm_models.run_models.multimodal.' + model.id + '.infer_' + model.id"
            :challenge="{ ask: '先猜: CLIP 训练后图文相似度变高了, 它能不能看图说话? Omni 输出了 codec logits, 能不能直接播放出声音?', answer: '答案: 都不能。\n- CLIP: 它把一张图和一句话各压成一个向量, 只会比相似度 (配对的高、不配对的低)。没有 decoder, 没有「给定图像、逐个生成文本 token」的链路, 所以只能打分、检索, 不能生成句子。要看图说话得走 Qwen2-VL 那条路: 视觉向量当前缀, 后面接一个因果 decoder。\n- Omni: Talker 输出的是离散语音 token (codec token) 的分数, 相当于「说出了哪些音频编码」。要变成能播的声音, 还得有一个真实的 codec 把这些 token 解码回波形 (声码器), 这个仓库里没有。\n另外要记住边界: 四套模型的 train 脚本都在一个固定的小随机 batch 上跑, loss 下降只说明「这条通路能学」, 不说明模型真的会看图或会听。' }">
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
        <p>一张图占多少上下文: 224×224 的图像按 16×16 切成不重叠的小块 (patch), 得到 14×14 = 196 个 patch (不计 CLS), 每个 patch 变成一个向量, 和文本 token 一样要过注意力、要存 KV。</p>
        <p>边长翻倍 (448×448), patch 数变 4 倍 (784 个); 拼成前缀后序列长度从 T 变成 T + 196, 注意力按 (T+196)² 涨。压缩 (比如本库 Qwen2-VL 用的 Perceiver Resampler, 把 16 个 patch 压成 8 个 latent) 能省成本, 代价是 latent 不再对应具体位置, 小字和细节会丢。</p>
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
    explanation: '先读 CLIP。它有两座塔: 图像塔 (ViT, 把图切成 patch 后过 Transformer) 和文本塔, 各自输出一个归一化向量。训练时一个 batch 里的 B 对图文算出 B×B 的相似度矩阵, 对角线是真正配对的, 其余 B−1 个是免费的负样本; loss 是「图找文」和「文找图」两个方向交叉熵的平均, 目标是让对角线最大。池化 (图像取 CLS, 文本取 EOS 位置) 和投影决定两边怎么进入同一个向量空间。\n看代码时先跑 infer, 检查输出向量的范数是 1、文本确实在 EOS 位置池化; 再跑 train 看配对 loss 从约 ln 8 = 2.18 降下来。注意这是固定的随机 batch, 拟合它不等于真的看懂了图文。' },
  { id: 'qwen2_vl', title: 'Qwen2-VL · 前缀', source: 'llm_models/models/multimodal/qwen2_vl.py:Qwen2VLModel',
    flow: '图像 → patch → ViT → resampler/projector → 视觉前缀 + 文本 → LLM',
    loss: '文本位置的 next-token CE (视觉位置不算)', boundary: '图是随机张量, 不是看图基准',
    explanation: '再读视觉条件生成 (第一扇门: 拼前缀)。图切 patch 过 ViT 得到一串视觉向量, 经 resampler 压缩、projector 对齐到语言模型的维度之后, 直接排在文本 token 前面, 同一个因果 decoder 从头读到尾。loss 只在文本位置算, 视觉位置的标签是 −100: 它们是条件, 不是预测目标。\n看代码时检查两件事: 换一张图, 文本位置的 logits 会变 (文本看得到前缀); 换后面的文本, 视觉位置的输出不变 (因果 mask 下前缀看不到后面)。本仓库用固定分辨率和 Perceiver Resampler (16 个 patch 压成 8 个 latent), 和原版 Qwen2-VL 的动态分辨率、相邻 2×2 patch 合并不一样。' },
  { id: 'whisper', title: 'Whisper · 交叉注意力', source: 'llm_models/models/multimodal/whisper.py:Whisper',
    flow: 'mel → 卷积 → 双向音频编码器 → cross-attention → 因果文本 decoder',
    loss: '文本 next-token CE', boundary: '直接输入 mel 声谱, 没做波形前处理',
    explanation: '然后读语音识别 (第二扇门: 交叉注意力)。真实流程里波形要先分帧、做 STFT、过 mel 滤波再取 log, 得到 mel 声谱 [80, 帧数]; 这些前处理本实验没做, 直接构造随机的 mel。音频编码器先用两层卷积沿时间切帧 (第二层 stride=2, 帧数减半), 再用双向注意力读完整段音频。文本 decoder 是因果的, 每一层多一块 cross-attention: 文本当 query, 音频编码器的输出当 K/V, 这是音频进 decoder 的唯一通道, 音频不占 decoder 的序列长度。\n看代码时先看形状断言和「改未来文本不影响过去位置」的断言, 再看 train: 两条样本文本相同、音频不同、标签不同, 只看文本的 loss 下界是 ln 2 = 0.693, 降到它以下说明文本位置真的读到了音频。loss 下降证明通路可学, 不代表语音识别准确率。' },
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
