"""
合成数据生成器 — 零依赖的 "假" 训练数据

是什么: 每种模型形态一个生成器, `generate_batch()` 返回 dict = model.forward 的全部 kwargs + "labels"。
        Trainer 弹出 labels 交给 LossComputer, 其余 **展开喂给模型 → Trainer 对模型类型保持中立。
关键约定: 默认 `fixed = True` —— 第一次采样的 batch 被缓存, 之后每步都返回同一个。
        随机 token 没有可学的规律, 每步换新 batch 时 loss 只会停在 ln V;
        固定后 "loss 下降" = "模型背下了这一个 batch", 检验的是 forward/backward/优化器通路, 不是泛化。
怎么扩展: 库内子类实现 `_sample()`; 库外子类直接覆写 `generate_batch()` 也仍然有效 (不走缓存)。
读代码时盯住: 每个 `_sample()` 返回的 dict 的 key —— 它们必须与对应模型 forward 的形参名一一对应。
"""

from typing import Dict, Optional, Tuple

import torch

from llm_models.utils.masks import get_pad_mask, get_subsequent_mask, combine_masks


class SyntheticDataGenerator:
    """
    合成数据生成器基类。Trainer 每步调用一次 `generate_batch()`。

    fixed=True (默认): 第一次 `_sample()` 的结果被缓存, 之后每步都返回同一个 batch。
        为什么要固定、"loss 下降" 此时意味着什么, 见文件头。
    fixed=False: 每步重新采样 (用于真有规律的任务, 或想看 ln V 平台时)。

    内部子类实现 `_sample()`; 外部子类直接覆写 `generate_batch()` 也仍然有效。
    """

    fixed: bool = True
    _cache: Optional[Dict[str, torch.Tensor]] = None

    def _sample(self) -> Dict[str, torch.Tensor]:
        """采一个新 batch: 含 model.forward 的全部 kwargs + "labels"。"""
        raise NotImplementedError

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        """返回一个 batch (dict)。fixed=True 时每次都是第一次采到的那个。"""
        if not self.fixed:
            return self._sample()
        if self._cache is None:
            self._cache = self._sample()
        # 浅拷贝: Trainer 会 pop("labels"), 不能弄坏缓存
        return dict(self._cache)


class DecoderOnlyDataGenerator(SyntheticDataGenerator):
    """
    因果 LM (GPT-3 / LLaMA / Mistral / DeepSeek …) 的数据: teacher forcing + 标签右移一位。

    先采长度 seq_len+1 的序列 X, 再切: idx = X[:, :-1], labels = X[:, 1:]
    → 位置 t 的目标恰好是输入的第 t+1 个 token。
    token 从 1 开始采: 0 约定为 pad, 避开它让所有位置都参与 loss。
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        tokens = torch.randint(                                  # [B, seq_len + 1]
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        return {
            "idx": tokens[:, :-1],     # [B, seq_len]
            "labels": tokens[:, 1:],   # [B, seq_len] 输入右移一位
        }


class EncoderDecoderDataGenerator(SyntheticDataGenerator):
    """
    Encoder-Decoder (seq2seq) 的数据, 给原版 Transformer (Vaswani et al. 2017) 用。

    返回的键 (= Transformer.forward 的形参 + labels):
        src      [B, src_len]           源语言 token, 由 encoder 自注意力处理
        tgt      [B, tgt_len]           目标语言 teacher forcing 输入
        labels   [B, tgt_len]           目标语言右移一位
        src_mask [B, 1, src_len]        源 padding 掩码 (encoder 自注意力里屏蔽 pad)
        tgt_mask [B, tgt_len, tgt_len]  目标 padding 掩码 与 因果掩码 按位 AND (交集):
                                        一个 key 既不是 pad、也不在未来, 才可见

    为什么要因果掩码:
        训练时整个目标序列并行喂进 decoder。位置 t 能看到 t 之后的 token, 就等于看到了答案。
        下三角 mask 在 softmax 前把未来位置的分数置 -inf。
    为什么要屏蔽 pad:
        pad 是为了对齐 batch 长度填的占位符, 没有语义。
        不屏蔽的话 attention 会把它当成有效 key, loss 也会算在 pad 上。
    本生成器的 token 从 1 开始采, 序列里不会出现 pad_idx=0, 两张 padding 掩码全为 True。

    pad_idx: padding token 的 id (默认 0)。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        batch_size: int,
        src_len: int,
        tgt_len: int,
        pad_idx: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.batch_size = batch_size
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_idx = pad_idx
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 源序列从 1 开始采, 避开 pad_idx=0, 全部位置有效
        src = torch.randint(                                             # [B, src_len]
            1, self.src_vocab_size, (self.batch_size, self.src_len), device=self.device
        )

        # 目标序列: 多采 1 个 token, 再切成 "输入" 和 "右移一位的标签"
        tgt_full = torch.randint(                                        # [B, tgt_len + 1]
            1, self.tgt_vocab_size, (self.batch_size, self.tgt_len + 1), device=self.device
        )
        tgt_input = tgt_full[:, :-1]  # decoder 输入 [B, tgt_len]: 去掉最后一个
        labels = tgt_full[:, 1:]      # decoder 目标 [B, tgt_len]: 去掉第一个, 位置 t 的目标是第 t+1 个 token

        # src_mask: 只有 padding mask, [B, 1, src_len], 可广播到 [B, H, T_q, src_len]
        src_mask = get_pad_mask(src, self.pad_idx)
        # tgt_mask: padding ∩ causal, 让 decoder 既忽略 pad 又看不到未来
        tgt_pad_mask = get_pad_mask(tgt_input, self.pad_idx)             # [B, 1, tgt_len]
        tgt_subsequent_mask = get_subsequent_mask(tgt_input)             # [1, tgt_len, tgt_len]
        tgt_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)      # 广播 AND → [B, tgt_len, tgt_len]

        return {
            "src": src,
            "tgt": tgt_input,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "labels": labels,
        }


class VisionLanguageDataGenerator(SyntheticDataGenerator):
    """
    视觉语言模型 (Qwen2-VL) 的数据。

    返回的键:
        input_ids [B, seq_len]                       文本 token
        images    [B, 3, H, W]                       随机像素图
        labels    [B, num_vision_tokens + seq_len]   视觉位置填 -100, 文本位置是 next-token

    为什么视觉位置填 -100:
        Qwen2-VL 的 forward: 图像 → vision encoder → resampler → num_vision_tokens 个视觉 embedding,
        和文本 embedding 拼成 [vision_tokens, text_tokens] 一起进 LLM。
        logits 的形状是 [B, num_vision_tokens + seq_len, V], labels 要和它等长。
        视觉 token 只给文本生成提供条件, 自己没有 "下一个 token" 可预测。
        填 -100 后 cross_entropy 按 ignore_index 跳过这些位置;
        不填的话, 模型要在视觉位置上拟合一个随机文本 token。

    Args:
        image_size:        图像边长 (H = W)。
        num_vision_tokens: 必须等于模型实际产出的视觉 token 数。
                           vision_num_latents > 0 时就是它 (通常远小于 patch 数);
                           = 0 (关 Resampler) 时是 (image_size / patch_size)²。
                           不一致时 labels 与 logits 长度不同, loss 报形状错。
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        image_size: int = 224,
        num_vision_tokens: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_vision_tokens = num_vision_tokens
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 文本 token: 多采 1 个, 用来切出右移一位的标签
        text_tokens = torch.randint(                                     # [B, seq_len + 1]
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        input_ids = text_tokens[:, :-1]      # 模型输入 [B, seq_len]
        text_labels = text_tokens[:, 1:]     # next-token 标签 [B, seq_len]

        # 图像: 标准高斯随机数, 充当标准化后的像素
        images = torch.randn(                                            # [B, 3, H, W]
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )

        # 视觉 token 位置: 全 -100, cross_entropy 会跳过这些位置
        vision_ignore = torch.full(                                      # [B, num_vision_tokens]
            (self.batch_size, self.num_vision_tokens), -100,
            dtype=torch.long, device=self.device,
        )
        # 拼接顺序必须和模型 forward 中 [vision, text] 的拼接顺序一致
        labels = torch.cat([vision_ignore, text_labels], dim=1)          # [B, num_vision_tokens + seq_len]

        return {
            "input_ids": input_ids,
            "images": images,
            "labels": labels,
        }


class OmniDataGenerator(SyntheticDataGenerator):
    """
    全模态模型 Qwen2.5-Omni (文本 + 视觉 + 音频 + 视频) 的数据。

    Thinker: 主 LLM, 处理多模态输入, 产出文本 logits。
    Talker:  额外的小型自回归头, 基于 Thinker 隐状态生成离散音频 token, 实现端到端 "说话"。
    所以本生成器同时给出文本 labels 和音频 labels。

    返回的键:
        input_ids          [B, seq_len]
        images             [B, 3, H, W]
        audio_spectrograms [B, 1, F, T_a]
        videos             [B, 3, T, H, W]
        audio_input_ids    [B, audio_seq_len]        Talker 的输入
        labels             [B, N_prefix + seq_len]   前缀位置全 -100, 只有文本 token 参与 LM loss
        audio_labels       [B, audio_seq_len]        audio_input_ids 右移一位
      N_prefix = num_vision_tokens + num_video_tokens + num_audio_tokens。

    标签处理同 VLM: 各模态 (vision / audio_spec / video) 经各自的 encoder + resampler
    投影成固定数量的 embedding, 拼在文本前面当前缀。

    Args:
        audio_vocab_size: 音频离散 token 的词表大小 (Talker 输出)。
        audio_seq_len:    音频 token 序列长度 (Talker 自回归长度)。
        audio_spec_size:  声谱图尺寸 (F, T_a): F = mel 频率维, T_a = 帧数。
                          生成 [B, 1, F, T_a], 与 Qwen2_5_OmniModel.forward 一致。
        video_size:       视频尺寸 (T, H, W)。
        num_vision_tokens / num_audio_tokens / num_video_tokens:
                          必须等于模型里该模态实际产出的 token 数:
                          对应的 *_num_latents > 0 时就是它, 关 Resampler 时是 patch 数。
                          不一致时 labels 与 logits 长度不同, loss 报形状错。
    """

    def __init__(
        self,
        vocab_size: int,
        audio_vocab_size: int,
        batch_size: int,
        seq_len: int,
        audio_seq_len: int = 16,
        image_size: int = 224,
        audio_spec_size: Tuple[int, int] = (256, 128),
        video_size: Tuple[int, int, int] = (8, 224, 224),
        num_vision_tokens: int = 64,
        num_audio_tokens: int = 64,
        num_video_tokens: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.audio_seq_len = audio_seq_len
        self.image_size = image_size
        self.audio_spec_size = audio_spec_size
        self.video_size = video_size
        self.num_vision_tokens = num_vision_tokens
        self.num_audio_tokens = num_audio_tokens
        self.num_video_tokens = num_video_tokens
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # ---- 文本分支 ----
        text_tokens = torch.randint(                                     # [B, seq_len + 1]
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        input_ids = text_tokens[:, :-1]                                  # [B, seq_len]
        text_labels = text_tokens[:, 1:]                                 # [B, seq_len] 右移一位

        # ---- 多模态原始信号 (随机噪声充当占位)----
        images = torch.randn(                                            # [B, 3, H, W]
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )
        audio_spectrograms = torch.randn(                                # [B, 1, F, T_a]
            self.batch_size, 1, self.audio_spec_size[0], self.audio_spec_size[1],
            device=self.device,
        )
        t, h, w = self.video_size
        videos = torch.randn(                                            # [B, 3, T, H, W]
            self.batch_size, 3, t, h, w, device=self.device
        )

        # ---- 音频离散 token (Talker 输入 + 自回归标签)----
        audio_tokens = torch.randint(                                    # [B, audio_seq_len + 1]
            1, self.audio_vocab_size,
            (self.batch_size, self.audio_seq_len + 1), device=self.device,
        )
        audio_input_ids = audio_tokens[:, :-1]                           # [B, audio_seq_len]
        audio_labels = audio_tokens[:, 1:]                               # [B, audio_seq_len] 右移一位

        # ---- 文本标签: 模态前缀位置全部 -100 ----
        # 拼接顺序必须与模型 forward 内 [vision, video, audio, text] 的拼接顺序一致,
        # 否则 logits 与 labels 错位, loss 没有意义
        num_modality_tokens = (
            self.num_vision_tokens + self.num_video_tokens + self.num_audio_tokens
        )
        modality_ignore = torch.full(                                    # [B, N_prefix]
            (self.batch_size, num_modality_tokens), -100,
            dtype=torch.long, device=self.device,
        )
        labels = torch.cat([modality_ignore, text_labels], dim=1)        # [B, N_prefix + seq_len]

        return {
            "input_ids": input_ids,
            "images": images,
            "audio_spectrograms": audio_spectrograms,
            "videos": videos,
            "audio_input_ids": audio_input_ids,
            "labels": labels,
            "audio_labels": audio_labels,
        }


# =============================================================================
# 其它形态: BERT / CLIP / Whisper / VAE / Diffusion / VAR
# =============================================================================


class MaskedLMDataGenerator(SyntheticDataGenerator):
    """
    BERT MLM 的数据: input_ids [B, seq_len] (部分位置被改过) + labels [B, seq_len]。

    每个位置以 15% 概率被选中 (80% → [MASK] id, 10% → 随机 token, 10% 保持原样)。
    只有被选中位置的 label 是原 token, 其余位置填 -100。

    Args:
        vocab_size: 词表大小
        mask_token_id: [MASK] 的 id (约定为 vocab_size - 1, 避免与真实 token 冲突)
        mlm_prob:   总 mask 概率 (BERT 论文 0.15)
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        mlm_prob: float = 0.15,
        mask_token_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mlm_prob = mlm_prob
        # 默认约定: 最后一个 token id 当 [MASK], 真实 tokenizer 会自带
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size - 1
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 真实 token 的范围是 [1, vocab_size - 2]: 0 留给 pad, vocab_size - 1 留给 [MASK]
        input_ids = torch.randint(                                       # [B, seq_len]
            1, self.vocab_size - 1, (self.batch_size, self.seq_len), device=self.device,
        )
        labels = torch.full_like(input_ids, -100)                        # [B, seq_len] 先全填 -100

        # 每个位置独立掷一次, 小于 mlm_prob 就选中
        mask = torch.rand_like(input_ids, dtype=torch.float) < self.mlm_prob   # [B, seq_len] bool
        labels[mask] = input_ids[mask]   # 必须在改 input_ids 之前存: label 要的是原 token

        # 在选中的位置里再分三份: 80% [MASK], 10% 随机, 10% 保持
        rand = torch.rand_like(input_ids, dtype=torch.float)             # 另掷一次, 与 mask 独立
        mask_replace = mask & (rand < 0.8)
        mask_random = mask & (rand >= 0.8) & (rand < 0.9)
        # 剩下的 mask 区间 (rand >= 0.9) 保持原样, 不改 input_ids

        input_ids = torch.where(
            mask_replace,
            torch.full_like(input_ids, self.mask_token_id),
            input_ids,
        )
        random_tokens = torch.randint(
            1, self.vocab_size - 1, input_ids.shape, device=self.device,
        )
        input_ids = torch.where(mask_random, random_tokens, input_ids)

        return {"input_ids": input_ids, "labels": labels}


class CLIPDataGenerator(SyntheticDataGenerator):
    """
    CLIP 的数据: images [B, 3, H, W] + input_ids [B, text_len] + eos_token_id。

    生成 B 对 (image, text), 对比 loss 的正样本由对角线隐式给出, 不需要额外 labels。
    "labels" 字段保留是为了符合 Trainer 的接口 (用 dummy 零张量)。

    Args:
        vocab_size:   文本词表
        batch_size:   每步样本对数
        text_len:     文本序列长度
        image_size:   图像边长
        eos_token_id: EOS token id (用于 pooler), 默认 vocab_size - 1
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        text_len: int,
        image_size: int = 224,
        eos_token_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.text_len = text_len
        self.image_size = image_size
        self.eos_token_id = eos_token_id if eos_token_id is not None else vocab_size - 1
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 上界 vocab_size - 1: 默认的 EOS id 是 vocab_size - 1, 正文里不能提前出现
        text = torch.randint(                                            # [B, text_len]
            1, self.vocab_size - 1, (self.batch_size, self.text_len), device=self.device,
        )
        # 在每行末尾填 EOS, 保证 CLIPTextEncoder 的 pooler 能找到位置
        text[:, -1] = self.eos_token_id

        images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device,
        )

        return {
            "images": images,
            "input_ids": text,
            "eos_token_id": self.eos_token_id,
            # labels 占位, ContrastiveLoss 不读它
            "labels": torch.zeros(self.batch_size, dtype=torch.long, device=self.device),
        }


class WhisperDataGenerator(SyntheticDataGenerator):
    """
    Whisper 的数据:

    - mel [B, n_mels, T_mel]: 随机张量充当 mel 声谱图
    - decoder_input_ids [B, tgt_len]: 随机 token 序列; labels [B, tgt_len] 为右移一位 (teacher forcing)

    Args:
        vocab_size:   文本词表
        n_mels:       mel 滤波器数 (Whisper 用 80)
        t_mel:        mel 帧数 (Whisper 30s@100Hz = 3000; 教学用更小)
        batch_size / tgt_len / device 同其他生成器
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        tgt_len: int = 32,
        n_mels: int = 80,
        t_mel: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.tgt_len = tgt_len
        self.n_mels = n_mels
        self.t_mel = t_mel
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        mel = torch.randn(self.batch_size, self.n_mels, self.t_mel, device=self.device)  # [B, n_mels, T_mel]

        tokens = torch.randint(                                          # [B, tgt_len + 1]
            1, self.vocab_size, (self.batch_size, self.tgt_len + 1), device=self.device,
        )
        return {
            "mel": mel,
            "decoder_input_ids": tokens[:, :-1],   # [B, tgt_len]
            "labels": tokens[:, 1:],               # [B, tgt_len] 输入右移一位
        }


class ImageDataGenerator(SyntheticDataGenerator):
    """
    通用图像合成数据生成器 (给 VAE / Tokenizer / VAR 用)。

    生成 [B, C, H, W] 的低频色块 (4×4 随机图双线性放大, tanh 压到 [-1, 1], 与 decoder 的
    tanh 输出同值域); labels = 同一张图 (自监督重建目标)。
    不用白噪声: 白噪声不可压缩, VAE 的瓶颈 / VQ 的 "粗尺度管轮廓" 都无从学起。

    Args:
        batch_size:      batch 大小
        image_size:      边长
        image_channels:  通道 (默认 3)
    """

    def __init__(
        self,
        batch_size: int,
        image_size: int,
        image_channels: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        coarse = torch.randn(self.batch_size, self.image_channels, 4, 4, device=self.device)  # [B, C, 4, 4]
        x = torch.nn.functional.interpolate(
            coarse, size=self.image_size, mode="bilinear", align_corners=False,
        ).tanh()                                                  # [B, C, H, W]
        return {"x": x, "labels": x}


class DiffusionDataGenerator(SyntheticDataGenerator):
    """
    扩散训练的合成数据生成器 (Image DiT / MM-DiT)

    一次 _sample:
        1) 采 x_0 (真实训练来自 VAE.encode; 这里用随机 latent 代替)
        2) 采 t, 用 scheduler.add_noise 得 (x_t, target)
        3) 同时提供 y (类别) 或 text_embeds (MM-DiT)

    约定:
        "x" = 含噪 latent, "t" = AddNoiseResult.t_norm (统一 [0, 1000) 量纲, Flow Matching 已 ×1000),
        "labels" = target (noise / velocity); 可选 "y", "text_embeds", "text_pooled"。

    注意 fixed=True 时 (x_0, t, ε) 整个被缓存: loss 下降 = 背下这一批的 ε, 不是学会去噪。
    真实扩散训练每步都重采 t 和 ε (设 fixed=False, 此时 x_0~N(0,I) 下 DDPM loss 的理论下界是 E_t[ᾱ_t] ≈ 0.5)。
    """

    def __init__(
        self,
        scheduler,
        batch_size: int,
        latent_channels: int,
        latent_size: int,
        num_classes: int = 0,
        text_seq_len: Optional[int] = None,
        text_dim: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.text_seq_len = text_seq_len
        self.text_dim = text_dim
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 1) 模拟一批 "x_0" (真实训练应来自 VAE.encode)
        x0 = torch.randn(                                                # [B, C, h, w]
            self.batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=self.device,
        )
        # 2) 采 t, 加噪
        t = self.scheduler.sample_timesteps(self.batch_size, self.device)   # [B]
        noised = self.scheduler.add_noise(x0, t)

        batch = {
            "x": noised.noisy,         # [B, C, h, w] 含噪 latent
            "t": noised.t_norm,        # [B] 给模型看的时间, [0, 1000) 量纲
            "labels": noised.target,   # [B, C, h, w] 回归目标
        }
        # 可选类别条件
        if self.num_classes > 0:
            batch["y"] = torch.randint(                                  # [B]
                0, self.num_classes, (self.batch_size,), device=self.device,
            )
        # 可选文本条件 (MM-DiT)
        if self.text_seq_len is not None and self.text_dim is not None:
            batch["text_embeds"] = torch.randn(                          # [B, text_seq_len, text_dim]
                self.batch_size, self.text_seq_len, self.text_dim, device=self.device,
            )
            batch["text_pooled"] = torch.randn(                          # [B, text_dim]
                self.batch_size, self.text_dim, device=self.device,
            )
        return batch


class VideoDiffusionDataGenerator(SyntheticDataGenerator):
    """
    Video DiT 的训练数据, 与 DiffusionDataGenerator 同构, 只是输入换成 5D 视频 latent:
    x / labels [B, C, T', H', W'], t [B], 可选 y [B]。
    """

    def __init__(
        self,
        scheduler,
        batch_size: int,
        latent_channels: int,
        latent_size: Tuple[int, int, int],   # (T', H', W')
        num_classes: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        T, H, W = self.latent_size
        x0 = torch.randn(                                                # [B, C, T', H', W']
            self.batch_size, self.latent_channels, T, H, W, device=self.device,
        )
        t = self.scheduler.sample_timesteps(self.batch_size, self.device)
        noised = self.scheduler.add_noise(x0, t)

        batch = {"x": noised.noisy, "t": noised.t_norm, "labels": noised.target}
        if self.num_classes > 0:
            batch["y"] = torch.randint(
                0, self.num_classes, (self.batch_size,), device=self.device,
            )
        return batch


class VARImageDataGenerator(SyntheticDataGenerator):
    """
    VAR 训练数据: 只需原始图像 images [B, 3, H, W], tokenizer 在模型内部离散化。
    """

    def __init__(
        self,
        batch_size: int,
        image_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device

    def _sample(self) -> Dict[str, torch.Tensor]:
        # 与 ImageDataGenerator 同款低频色块, 只是字段名对齐 VARModel.forward(images)
        images = ImageDataGenerator(self.batch_size, self.image_size, device=self.device)._sample()["x"]
        # VARLoss 从 model_output 里读 labels, 这里只是 Trainer 接口占位
        return {"images": images, "labels": images}
