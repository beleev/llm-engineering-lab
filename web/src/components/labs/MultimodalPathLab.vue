<template>
  <LabFrame title="图像与音频怎样接入语言模型" sub="按输入、连接、训练目标和输出读四套已有模型。切换只改变阅读入口，不新增模型实现。"
            :module="model.source.split(':')[0]" :run="'python -m llm_models.run_models.multimodal.' + model.id + '.infer_' + model.id"
            :challenge="{ ask: 'CLIP 的相似度变高，是否就能看图说话？Omni 输出 codec logits，是否就能播放声音？', answer: '都不能直接推出。前者缺文本条件生成链路；后者还需要真实 codec 解码和波形生成。小随机 batch 的 loss 下降只验证通路可学习。' }">
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
      <p class="lab-note">本地配套讲义：llm_models/run_models/multimodal/{{ model.id }}/README.md</p>
    </div>
    <template #stats>
      <div class="kv"><span>训练目标</span><b>{{ model.loss }}</b></div>
      <div class="kv"><span>输入/输出边界</span><b>{{ model.boundary }}</b></div>
      <p class="lab-note">视觉成本例：224×224 图像、16×16 patch 得到 196 个 patch（未计 CLS）。边长翻倍，patch 数变四倍；前缀长度影响注意力和 KV。压缩省成本，也可能损失细节。</p>
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
    loss: '双向对比 CE', boundary: '向量相似度，不生成句子',
    explanation: '先读 CLIP：batch 内配对在相似度矩阵对角线，其余样本作负样本。池化和投影决定两边如何进入共享空间。\n先跑 infer 检查范数与 EOS 池化，再跑 train 看配对 loss。固定随机 batch 的拟合不能当作真实图文理解能力。' },
  { id: 'qwen2_vl', title: 'Qwen2-VL · 前缀', source: 'llm_models/models/multimodal/qwen2_vl.py:Qwen2VLModel',
    flow: '图像 → patch → ViT → resampler/projector → 视觉前缀 + 文本 → LLM',
    loss: '文本位置 next-token CE', boundary: '随机图像批次，不是看图基准',
    explanation: '再读视觉条件生成：projector 对齐维度，视觉位置不计算文本 loss，后面的文本通过因果注意力看到前缀。\n检查换图会改变文本 logits、换未来文本不会改变视觉前缀。仓库用固定分辨率和 Perceiver Resampler，不等同原版动态分辨率及相邻 patch 合并。' },
  { id: 'whisper', title: 'Whisper · 交叉注意力', source: 'llm_models/models/multimodal/whisper.py:Whisper',
    flow: 'mel → 卷积 → 双向音频编码器 → cross-attention → 因果文本 decoder',
    loss: '文本 next-token CE', boundary: '直接输入 mel，未做波形前处理',
    explanation: '波形分帧、STFT、mel 滤波与 log 属于前处理；本实验直接构造随机 mel。编码器理解帧序列，decoder 用文本作 query，音频表示作 K/V。\n先看形状和未来文本不泄漏的断言，再看同文本、不同音频能否学到不同标签。loss 下降证明通路可学习，不代表语音识别准确率。' },
  { id: 'qwen2_5_omni', title: 'Omni · Thinker/Talker', source: 'llm_models/models/multimodal/qwen2_5_omni.py:Qwen2_5_OmniModel',
    flow: '图像/音频/文本 → Thinker → 文本 logits / 隐状态 → Talker → codec logits',
    loss: '文本 CE + 0.5×语音 CE', boundary: '无真实 codec 解码与声码器',
    explanation: '最后读双输出：Talker 读取 Thinker 隐状态，语音分支并不倒流回文本分支。检查换图是否影响语音 logits、改未来 codec 是否影响过去输出。\n输出离散语音 token 的分数不等于得到可播放波形；真实音频 codec、流式对齐与延迟还需单独实现和验收。' },
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
