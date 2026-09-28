"""
LLM Models — 大语言模型架构演进教具库

PyTorch 实现, 用于对比各主流架构的设计与训练 / 推理数据流。

共 23 个模型。

架构演进主线 (左脑: 理解 + 文本生成), 9 个
    Transformer (2017)      -> Encoder-Decoder + MHA + FFN + LN + Sin-PE
    BERT        (2018)      -> Encoder-only + MLM 预训练 (双向注意力); arXiv 2018, NAACL 2019
    GPT-3       (2020)      -> Decoder-only + MHA + GELU-FFN + LN
    LLaMA       (2023)      -> GQA + SwiGLU + RMSNorm + RoPE (现代模板)
    Mistral     (2023)      -> LLaMA + 滑动窗口注意力 (SWA)
    Mamba       (2023)      -> 非注意力 Selective SSM, O(T) 线性复杂度
    MTP         (2024)      -> LLaMA + 多 token 预测级联 (DeepSeek-V3)
    Qwen3-Next  (2025)      -> Gated DeltaNet 线性注意力 : 全注意力 = 3 : 1 混合
    LLaDA       (2025)      -> 掩码扩散语言模型 (没有因果 mask 的 LLaMA)

MoE, 4 个
    Mixtral     (2024)      -> LLaMA + sparse MoE (softmax top-k)
    DeepSeek-V3 (2024)      -> MLA + 细粒度 MoE(SwiGLU) + RMSNorm + RoPE
    DeepSeek-V3.2 (2025)    -> DSA (Lightning Indexer + MLA) + MoE
    GPT-OSS     (2025)      -> Mixtral 骨架 + 交替 SWA / 全注意力 + attention sink

多模态 (眼耳), 4 个
    CLIP        (2021)      -> 对比学习双塔, 视觉-语言对齐
    Whisper     (2022)      -> 音频 Encoder-Decoder, ASR 经典
    Qwen2-VL    (2024)      -> ViT + Resampler 多模态 LLM; M-RoPE 可选, 默认关 (use_mrope=False)
    Qwen2.5-Omni(2025)      -> Thinker-Talker 全模态 (文本+图像+音频+视频)

生成模型 (右脑), 6 个
    VAE / 3D VAE            -> Latent Diffusion 的前置压缩器 (图像 / 视频各一个)
    DiT         (2023)      -> adaLN-Zero 取代 UNet, 成为扩散 Transformer 主干
    MM-DiT      (2024)      -> SD3/FLUX 双流同 attention, Rectified Flow 目标
    Video DiT   (2024)      -> Sora 风格 Spacetime Patches + DiT
    VAR         (2024)      -> 自回归图像生成: 冻结的多尺度 VQ tokenizer + block-causal Transformer
                               (PreLNBlock + 教学版 MHA, 没有用 GPT3 类)

使用示例:
    >>> from llm_models import GPT3, LLaMA, DeepSeekV3, DiT
    >>> from llm_models import Trainer, TrainingConfig, DDPMScheduler, DiffusionLoss
"""

__version__ = "0.5.0"
__author__ = "LLM Team"

# --- Layers (底层零件) ---
from llm_models.layers import (
    # Attention 家族
    ScaledDotProductAttention,
    SingleHeadSelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
    # Position
    SinPositionalEncoding,
    RotaryPositionalEncoding,
    MultimodalRotaryEmbedding,
    apply_rotary_pos_emb,
    # FFN
    FeedForward,
    GeLUFeedForward,
    SwiGLUFeedForward,
    # Norm
    RMSNorm,
    # Block
    PreLNBlock,
    PreLNCrossBlock,
    # MoE
    MixtralMoE,
    # SSM
    SelectiveSSM,
    # Diffusion 条件注入
    AdaLNZeroBlock,
    FinalLayer,
    TimestepEmbedding,
    modulate,
    # VQ
    VectorQuantizer,
)

# --- Models ---
from llm_models.models import (
    # Transformer / BERT / GPT
    Transformer,
    EncoderLayer,
    DecoderLayer,
    BERT,
    BERTEmbeddings,
    GPT3,
    GPTBlock,
    # LLaMA / Mistral / MTP / Mixtral / Mamba
    LLaMA,
    LlamaBlock,
    Mistral,
    MistralBlock,
    MTPLLaMA,
    MTPModule,
    MTPLoss,
    Qwen3Next,
    Mixtral,
    MixtralBlock,
    Mamba,
    MambaBlock,
    MambaLayer,
    # DeepSeek
    DeepSeekV3,
    DeepSeekBlock,
    DeepSeekMoE,
    DeepSeekV3_2,
    DeepSeekV32Block,
    # LLaDA / GPT-OSS
    LLaDA,
    LLaDALoss,
    GPTOSSMini,
    GPTOSSBlock,
    # Multimodal primitives
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
    # CLIP / Whisper
    CLIPModel,
    CLIPTextEncoder,
    CLIPVisionEncoder,
    Whisper,
    WhisperAudioEncoder,
    WhisperTextDecoder,
    # Qwen2-VL / Qwen2.5-Omni
    Qwen2VLDecoder,
    Qwen2VLModel,
    OmniThinkerDecoder,
    OmniTalkerDecoder,
    Qwen2_5_OmniModel,
    # Generation: VAE / DiT / MM-DiT / Video DiT / VAR
    ImageVAE,
    ImageVAEEncoder,
    ImageVAEDecoder,
    CausalVideoVAE,
    CausalVAE3DEncoder,
    CausalVAE3DDecoder,
    DiT,
    PatchifyConv,
    VideoDiT,
    Patchify3D,
    MMDiT,
    MMDiTBlock,
    ImageTokenizer,
    VARModel,
)

# --- Utils ---
from llm_models.utils import (
    get_pad_mask,
    get_subsequent_mask,
    build_causal_mask,
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
    combine_masks,
)

# --- Training ---
from llm_models.training import (
    Trainer,
    TrainingConfig,
    # Loss
    LossComputer,
    StandardLMLoss,
    MoELMLoss,
    OmniLoss,
    MaskedLMLoss,
    ContrastiveLoss,
    VAELoss,
    VARLoss,
    DiffusionLoss,
    # Data
    SyntheticDataGenerator,
    DecoderOnlyDataGenerator,
    EncoderDecoderDataGenerator,
    VisionLanguageDataGenerator,
    OmniDataGenerator,
    MaskedLMDataGenerator,
    CLIPDataGenerator,
    WhisperDataGenerator,
    ImageDataGenerator,
    DiffusionDataGenerator,
    VideoDiffusionDataGenerator,
    VARImageDataGenerator,
    # Diffusion
    NoiseScheduler,
    DDPMScheduler,
    FlowMatchingScheduler,
    DDIMSampler,
    EulerFlowSampler,
    classifier_free_guidance,
)

__all__ = [
    # ---- Layers ----
    "ScaledDotProductAttention", "SingleHeadSelfAttention",
    "MultiHeadAttention", "GroupedQueryAttention",
    "MultiHeadLatentAttention", "MultiHeadLatentSparseAttention",
    "SinPositionalEncoding", "RotaryPositionalEncoding",
    "MultimodalRotaryEmbedding", "apply_rotary_pos_emb",
    "FeedForward", "GeLUFeedForward", "SwiGLUFeedForward",
    "RMSNorm", "PreLNBlock", "PreLNCrossBlock",
    "MixtralMoE", "SelectiveSSM",
    "AdaLNZeroBlock", "FinalLayer", "TimestepEmbedding", "modulate",
    "VectorQuantizer",
    # ---- Models: left-brain LLM ----
    "Transformer", "EncoderLayer", "DecoderLayer",
    "BERT", "BERTEmbeddings",
    "GPT3", "GPTBlock",
    "LLaMA", "LlamaBlock",
    "Mistral", "MistralBlock",
    "MTPLLaMA", "MTPModule", "MTPLoss",
    "Qwen3Next",
    "Mixtral", "MixtralBlock",
    "Mamba", "MambaBlock", "MambaLayer",
    "DeepSeekV3", "DeepSeekBlock", "DeepSeekMoE",
    "DeepSeekV3_2", "DeepSeekV32Block",
    # ---- Models: multimodal understanding ----
    "PatchEmbed2D", "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock", "PerceiverResampler",
    "ModalityProjector",
    "CLIPModel", "CLIPTextEncoder", "CLIPVisionEncoder",
    "Whisper", "WhisperAudioEncoder", "WhisperTextDecoder",
    "Qwen2VLDecoder", "Qwen2VLModel",
    "OmniThinkerDecoder", "OmniTalkerDecoder", "Qwen2_5_OmniModel",
    # ---- Models: right-brain generation ----
    "ImageVAE", "ImageVAEEncoder", "ImageVAEDecoder",
    "CausalVideoVAE", "CausalVAE3DEncoder", "CausalVAE3DDecoder",
    "DiT", "PatchifyConv",
    "VideoDiT", "Patchify3D",
    "MMDiT", "MMDiTBlock",
    "ImageTokenizer", "VARModel",
    # ---- Utils ----
    "get_pad_mask", "get_subsequent_mask", "build_causal_mask",
    "build_sliding_window_mask",
    "combine_causal_and_padding_mask", "combine_masks",
    # ---- Training ----
    "Trainer", "TrainingConfig",
    "LossComputer", "StandardLMLoss", "MoELMLoss", "OmniLoss",
    "MaskedLMLoss", "ContrastiveLoss", "VAELoss", "VARLoss", "DiffusionLoss",
    "SyntheticDataGenerator",
    "DecoderOnlyDataGenerator", "EncoderDecoderDataGenerator",
    "VisionLanguageDataGenerator", "OmniDataGenerator",
    "MaskedLMDataGenerator", "CLIPDataGenerator", "WhisperDataGenerator",
    "ImageDataGenerator", "DiffusionDataGenerator", "VideoDiffusionDataGenerator",
    "VARImageDataGenerator",
    "NoiseScheduler", "DDPMScheduler", "FlowMatchingScheduler",
    "DDIMSampler", "EulerFlowSampler", "classifier_free_guidance",
]

# --- 不经子包 __init__ 转发的导出: 按完整模块路径直接引入 ---
from llm_models.layers.core.attention import LightningIndexer
from llm_models.layers.core.position_encoding import sinusoidal_embedding, scaled_inv_freq
from llm_models.layers.sparse.linear_attention import GatedDeltaNet
from llm_models.layers.diffusion.vq import MultiScaleVQ
from llm_models.utils.init import init_weights
from llm_models.utils.generation import KVCache, GenerationMixin, benchmark_kv_cache
from llm_models.models.language_models.llada import forward_process

__all__ += [
    "LightningIndexer", "sinusoidal_embedding", "scaled_inv_freq", "GatedDeltaNet",
    "MultiScaleVQ", "init_weights", "KVCache", "GenerationMixin", "benchmark_kv_cache",
    "LLaDA", "LLaDALoss", "forward_process",
]
__all__ += ["GPTOSSMini", "GPTOSSBlock"]
from llm_models.models.multimodal.qwen2_vl import build_mrope_position_ids

__all__ += ["build_mrope_position_ids"]
