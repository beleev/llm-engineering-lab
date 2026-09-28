"""
training — 一个 Trainer 跑全库所有模型 (策略模式)

是什么: Trainer 只认两个接口, 换模型就是换一对 (数据生成器, 损失)。
    Trainer                       训练循环, 与具体模型 / 损失 / 数据无关
    TrainingConfig                不可变 (frozen) 的训练配置
    LossComputer 子类             每种模型形态一个损失 (loss.py)
    SyntheticDataGenerator 子类   每种模型形态一个合成 batch (data.py)
覆盖: LM (GPT / LLaMA / Mamba)、MoE (Mixtral / DeepSeek)、多模态 (CLIP / Whisper / Omni)、
      生成模型 (VAE / DiT / MM-DiT / Video DiT / VAR)。
扩散专用 (diffusion.py):
    DDPMScheduler / FlowMatchingScheduler   噪声调度: 正向加噪 + 给出回归目标
    DDIMSampler / EulerFlowSampler          采样
    DiffusionLoss                           MSE; 目标是 ε 还是 velocity, 由 scheduler 造数据时定
    classifier_free_guidance                CFG 线性外插
不在本包: LLaDALoss 与 LLaDA 模型放在一起 (models/language_models/llada.py)。

典型用法:
    >>> cfg = TrainingConfig()
    >>> data_gen = DecoderOnlyDataGenerator(...)
    >>> loss_fn = StandardLMLoss()
    >>> trainer = Trainer(model, cfg, data_gen, loss_fn)
    >>> trainer.train()
"""

from llm_models.training.config import TrainingConfig
from llm_models.training.trainer import Trainer
from llm_models.training.loss import (
    LossComputer,
    StandardLMLoss,
    MoELMLoss,
    MTPLoss,
    OmniLoss,
    MaskedLMLoss,
    ContrastiveLoss,
    VAELoss,
    VARLoss,
)
from llm_models.training.data import (
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
)
from llm_models.training.diffusion import (
    NoiseScheduler,
    DDPMScheduler,
    FlowMatchingScheduler,
    DDIMSampler,
    EulerFlowSampler,
    DiffusionLoss,
    classifier_free_guidance,
)

__all__ = [
    # 配置
    "TrainingConfig",
    # 训练循环
    "Trainer",
    # 损失策略
    "LossComputer",
    "StandardLMLoss",
    "MoELMLoss",
    "MTPLoss",
    "OmniLoss",
    "MaskedLMLoss",
    "ContrastiveLoss",
    "VAELoss",
    "VARLoss",
    "DiffusionLoss",
    # 数据生成策略
    "SyntheticDataGenerator",
    "DecoderOnlyDataGenerator",
    "EncoderDecoderDataGenerator",
    "VisionLanguageDataGenerator",
    "OmniDataGenerator",
    "MaskedLMDataGenerator",
    "CLIPDataGenerator",
    "WhisperDataGenerator",
    "ImageDataGenerator",
    "DiffusionDataGenerator",
    "VideoDiffusionDataGenerator",
    "VARImageDataGenerator",
    # 扩散调度 & 采样
    "NoiseScheduler",
    "DDPMScheduler",
    "FlowMatchingScheduler",
    "DDIMSampler",
    "EulerFlowSampler",
    "classifier_free_guidance",
]
