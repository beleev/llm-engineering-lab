# 多模态阅读路线：从输入到输出逐段查

沿用现有四套模型，不增加独立目录层级。网页入口在阶段 2 总览的“图像与音频怎样接入语言模型”。

| 顺序 | 输入与连接 | 训练目标 | 先检查什么 |
| --- | --- | --- | --- |
| [CLIP](../llm_models/run_models/multimodal/clip/README.md) | image→patch→ViT；text→文本编码器；两边投影归一化 | 双向对比 CE | 为什么句向量相似不等于能生成描述 |
| [Qwen2-VL](../llm_models/run_models/multimodal/qwen2_vl/README.md) | patch→视觉编码器→resampler/projector→视觉前缀+文本 | 文本位置 next-token CE | 换图是否影响文本，文本能否反向影响前缀 |
| [Whisper](../llm_models/run_models/multimodal/whisper/README.md) | mel→卷积→双向音频编码器→decoder cross-attention | 文本 next-token CE | mel 前处理不等于编码器，未来文本不能影响过去 |
| [Omni](../llm_models/run_models/multimodal/qwen2_5_omni/README.md) | 多模态编码→Thinker→隐藏状态→Talker | 文本 CE+语音 codec CE | 改 codec 不应反向改变 Thinker 文本 |

## 视觉 token 的成本

固定不重叠 patch 时，224×224 图像配 16×16 patch 得到 14×14=196 个 patch（未计 CLS）。边长翻倍，patch 数变为四倍。若和文本拼成前缀，自注意力长度从 T 变为 T+N，完整注意力分数规模随 `(T+N)^2` 增加，KV 规模近似随 T+N 增加。压缩视觉 token 能省成本，但可能丢失小字与细节。仓库 Qwen2-VL 用固定分辨率和 Perceiver Resampler，不等同原版动态分辨率及空间 patch 合并。

## 音频前处理和任务边界

波形要经过采样率对齐、分帧/STFT、mel 滤波和 log 等前处理，才能成为 Whisper 的 mel 张量。这里脚本直接构造随机 mel，不提供真实音频解码/前处理。Omni 输出的是 codec logits，不是可播放波形，仓库未实现真实音频 codec 或声码器。

训练脚本在小固定随机 batch 上降低 loss，验证信息通路确实可学习；不是图文/语音基准或上线质量。先跑 infer 的因果/形状断言，再看 train 能否依赖模态区分相同文本对应的不同标签。网页附四道机制题，避免只记模型名字。
