"""
模型模块

按架构/用途分为四大类:

语言模型 (Language Models)
- Transformer: Encoder-Decoder (Vaswani et al., 2017), 代码在 foundation/
- BERT: Encoder-only, MLM (Devlin et al., 2018)
- GPT-3: Decoder-only, 自回归 LM (Brown et al., 2020)
- LLaMA: 现代 Decoder-only (Touvron et al., 2023)
- Mistral: LLaMA + 滑动窗口注意力 SWA (Jiang et al., 2023)
- MTP: LLaMA + 多 token 预测级联 (DeepSeek-V3, 2024)
- Qwen3-Next: Gated DeltaNet 3:1 全注意力混合架构 (2025)
- Mamba: Selective SSM, O(T) (Gu & Dao, 2023)
- LLaDA: 掩码扩散语言模型, 没有因果 mask 的 LLaMA (Nie et al., 2025)

MoE 模型 (Mixture of Experts)
- Mixtral: Sparse MoE (Jiang et al., 2024)
- DeepSeek-V3 / V3.2: MLA + MoE + DSA (2024-2025)
- GPT-OSS (GPTOSSMini): Mixtral 骨架 + 交替 SWA/全注意力 + attention sink (2025)

多模态模型 (Multimodal)
- CLIP: 图文对比学习 (Radford et al., 2021)
- Whisper: 语音识别 (Radford et al., 2022)
- Qwen2-VL / Qwen2.5-Omni: 多模态 LLM (2024-2025)
- 公共构件: PatchEmbed, Perceiver, ModalityProjector

生成模型 (Generative)
- VAE / 3D VAE: Latent 压缩器
- DiT / MM-DiT / Video DiT: 扩散 Transformer
- VAR: 自回归图像生成

MODEL_REGISTRY 按上面四类列出 23 个模型的展示名和类, main.py 的架构清单从它生成。


本库统一的教学约定 (不是原模型的特征)
----------------------------------------
下面四件事全库写法一致, 方便横向对比。各模型与原论文的出入写在各自文件头。

1. Pre-LN: `x = x + f(norm(x))`。原版是 Post-LN 的模型 (2017 Transformer, BERT), 这里也用 Pre-LN。
2. weight tying: lm_head 与 token embedding 是同一个矩阵, 省一个 V×D 的参数。
   官方 LLaMA 不共享; GPT-2/3、Mamba、Whisper 原版就共享。
   例外: Transformer (源/目标词表不同) 和 OmniTalkerDecoder 不共享。
3. `·√D`: 带 `* math.sqrt(d_model)` 的模型把 embedding 放大 √D 倍 (2017 Transformer 的做法)。
   - embedding 用 N(0, 0.02²) 初始化, 每维 std 0.02; 乘完每维 std ≈ 0.02·√D。
   - 用 Sin-PE 时, 这让 embedding 与幅度 ~1 的位置编码同量级。
   - 用 RoPE 的模型没有加性位置编码, 也照样乘。效果是 embedding 相对各层残差分支的输出
     放大 √D 倍。官方 LLaMA / GPT 都不乘。
   - 不乘的: BERT、Whisper、CLIP (与原版一致), Mamba (沿用官方实现)。
4. 初始化: 所有 Linear / Embedding 用 N(0, 0.02²), 初始 loss ≈ ln V (utils/init.py)。
   去掉这一步 (PyTorch 默认 N(0,1) embedding) 会怎样:
   - 有 tying 且乘 √D: 最后一层 norm 的输出 h 几乎指向输入 token 自己的 embedding,
     范数 √D; 它与 lm_head 同一行 (范数也 ≈ √D) 的内积 ≈ D。模型确信 "下一个 token
     还是它自己", 初始 CE ≈ D。run_models 的 mini 配置实测: D=256 时 ~250, D=128 时 ~127。
   - 有 tying 不乘 √D (Whisper): 同样的机制, 幅度小一些。D=128, V=500 时 ≈ 80。
   - BERT 的 MLM head 多一层变换, 方向被打散: logits 的 std ≈ √D。D=128, V=500 时 CE ≈ 40。


forward 的返回类型总表
----------------------------------------
B = batch, T = 序列长度, V = 词表, D = d_model。

返回 Tensor:
    Transformer                 logits [B, T, V_tgt]
    GPT3 / Mistral / Mamba / Qwen3Next / LLaDA      logits [B, T, V]
    LLaMA                       logits [B, T, V]; return_hidden=True 时改为返回 hidden [B, T, D]
    BERT / Qwen2VLDecoder       logits [B, T, V]; return_hidden=True 时返回 (logits, hidden)
    Whisper                     logits [B, T, V]
    Qwen2VLModel                logits [B, N_v + T, V] (前 N_v 个位置是视觉 token)
    OmniTalkerDecoder           logits [B, T_audio, V_audio]
    DiT / MMDiT                 与输入同形 [B, C, H, W]
    VideoDiT                    与输入同形 [B, C, T, H, W]

返回 tuple (logits [B, T, V], all_routing_info):
    Mixtral / DeepSeekV3 / DeepSeekV3_2 / GPTOSSMini
    all_routing_info 是 list, 每层一个 dict, 键:
        router_logits [N, E] / selected_experts [N, K] / routing_weights [N, K] /
        routing_probs [N, E], 其中 N = B·T。
        DeepSeekV3_2 在带梯度的前向里每层多一个 index_loss (标量)。

返回 dict:
    MTPLLaMA            {logits [B, T, V], mtp_logits: list, 第 k 项 [B, T, V]}
    CLIPModel           {image_features [B, E], text_features [B, E], logit_scale 标量}
    ImageVAE            {recon [B, 3, H, W], z, mean, logvar}, 后三者 [B, latent_dim, h, w]
    CausalVideoVAE      {recon [B, 3, T, H, W], z, mean, logvar}
    ImageTokenizer      {recon, indices: list of [B, s, s], vq_loss, perplexity}
    VARModel            {logits [B, L, K], labels [B, L]}
    Qwen2_5_OmniModel   {text_logits, audio_logits (或 None), thinker_hidden_states};
                        return_dict=False 时是同顺序的三元 tuple

同名参数 return_hidden 有两种语义: LLaMA 是 "用 hidden 代替 logits", BERT 和
Qwen2VLDecoder 是 "logits 之外再给 hidden"。
GenerationMixin.generate() 三种返回都认: Tensor 直接用, tuple 取第 0 项, dict 取 "logits"。


padding mask 与 KV cache 支持范围
----------------------------------------
                          padding mask 参数                    KV cache / generate()
    Transformer           src_mask / tgt_mask (调用方自己建)     无
    BERT / LLaDA          attention_mask                        无 (双向注意力, 没有 cache 可用)
    GPT3                  attention_mask (Sin-PE 下只收全 1)      有
    LLaMA / Mistral       attention_mask                        有
    MTPLLaMA              attention_mask                        有 (只跑主干, mtp_logits 为空)
    Qwen3Next             attention_mask (delta 层跳过 pad 的写入)  有 (attn 层存 k/v, delta 层存 state)
    Mamba                 attention_mask (pad 处输入置 0)         有 (存递推状态, 不是 K/V)
    Mixtral / GPTOSSMini  attention_mask                        有
    DeepSeekV3 / V3_2     attention_mask                        有 (MLA 的 latent cache)
    CLIP / Whisper        不接受                                 无
    Qwen2VLModel          text_attention_mask                   无
    Qwen2_5_OmniModel     text_attention_mask / audio_attention_mask    无

带 generate() 的模型 forward 都收 attention_mask。只有 Sin-PE 的 GPT3 做不到左 pad:
绝对位置下 pad 会挪动真 token 的位置, 所以遇到有 0 的 mask 就抛 NotImplementedError。
Mamba 在 pad 位置把卷积输入和 SSM 输入置 0, 状态从 0 出发时一直是 0。
其余模型左 pad 批量生成与逐条生成一致。
"不接受" 的模型 forward 签名里没有这个参数, 传了会 TypeError。
"""

# --- 基础组件 ---
from llm_models.models.foundation.transformer import (
    Transformer,
    EncoderLayer,
    DecoderLayer,
)

# --- 语言模型 ---
from llm_models.models.language_models.bert import BERT, BERTEmbeddings
from llm_models.models.language_models.gpt3 import GPT3, GPTBlock
from llm_models.models.language_models.llama import LLaMA, LlamaBlock
from llm_models.models.language_models.mistral import Mistral, MistralBlock
from llm_models.models.language_models.mtp import MTPLLaMA, MTPModule, MTPLoss
from llm_models.models.language_models.qwen3_next import Qwen3Next
from llm_models.models.language_models.mamba import Mamba, MambaBlock, MambaLayer
from llm_models.models.language_models.llada import LLaDA, LLaDALoss

# --- MoE ---
from llm_models.models.moe.mixtral import Mixtral, MixtralBlock
from llm_models.models.moe.deepseekV3 import (
    DeepSeekV3,
    DeepSeekBlock,
    DeepSeekMoE,
    DeepSeekV3_2,
    DeepSeekV32Block,
)
from llm_models.models.moe.gpt_oss import GPTOSSMini, GPTOSSBlock

# --- 多模态 ---
from llm_models.layers.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)
from llm_models.models.multimodal.clip import (
    CLIPModel,
    CLIPTextEncoder,
    CLIPVisionEncoder,
)
from llm_models.models.multimodal.whisper import (
    Whisper,
    WhisperAudioEncoder,
    WhisperTextDecoder,
)
from llm_models.models.multimodal.qwen2_vl import Qwen2VLDecoder, Qwen2VLModel
from llm_models.models.multimodal.qwen2_5_omni import (
    OmniThinkerDecoder,
    OmniTalkerDecoder,
    Qwen2_5_OmniModel,
)

# --- 生成模型 ---
from llm_models.models.generative.vae import (
    ImageVAE,
    ImageVAEEncoder,
    ImageVAEDecoder,
)
from llm_models.models.generative.vae3d import (
    CausalVideoVAE,
    CausalVAE3DEncoder,
    CausalVAE3DDecoder,
)
from llm_models.models.generative.dit import DiT, PatchifyConv
from llm_models.models.generative.video_dit import VideoDiT, Patchify3D
from llm_models.models.generative.mmdit import MMDiT, MMDiTBlock
from llm_models.models.generative.var import ImageTokenizer, VARModel

__all__ = [
    # Language Models
    "Transformer", "EncoderLayer", "DecoderLayer",
    "BERT", "BERTEmbeddings",
    "GPT3", "GPTBlock",
    "LLaMA", "LlamaBlock",
    "Mistral", "MistralBlock",
    "MTPLLaMA", "MTPModule", "MTPLoss",
    "Qwen3Next",
    "Mamba", "MambaBlock", "MambaLayer",
    "LLaDA", "LLaDALoss",
    # MoE
    "Mixtral", "MixtralBlock",
    "DeepSeekV3", "DeepSeekBlock", "DeepSeekMoE",
    "DeepSeekV3_2", "DeepSeekV32Block",
    "GPTOSSMini", "GPTOSSBlock",
    # Multimodal
    "PatchEmbed2D", "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock", "PerceiverResampler",
    "ModalityProjector",
    "CLIPModel", "CLIPTextEncoder", "CLIPVisionEncoder",
    "Whisper", "WhisperAudioEncoder", "WhisperTextDecoder",
    "Qwen2VLDecoder", "Qwen2VLModel",
    "OmniThinkerDecoder", "OmniTalkerDecoder", "Qwen2_5_OmniModel",
    # Generative
    "ImageVAE", "ImageVAEEncoder", "ImageVAEDecoder",
    "CausalVideoVAE", "CausalVAE3DEncoder", "CausalVAE3DDecoder",
    "DiT", "PatchifyConv",
    "VideoDiT", "Patchify3D",
    "MMDiT", "MMDiTBlock",
    "ImageTokenizer", "VARModel",
    "MODEL_REGISTRY",
]

# 类别 → {展示名: 类}。顺序与 llm_models/README.md 的模型清单一致; main.py 按它打印架构清单
MODEL_REGISTRY = {
    "语言模型": {
        "Transformer": Transformer, "BERT": BERT, "GPT-3": GPT3, "LLaMA": LLaMA,
        "Mistral": Mistral, "MTP": MTPLLaMA, "Qwen3-Next": Qwen3Next, "Mamba": Mamba, "LLaDA": LLaDA,
    },
    "MoE": {
        "Mixtral": Mixtral, "DeepSeek-V3": DeepSeekV3, "DeepSeek-V3.2": DeepSeekV3_2, "GPT-OSS": GPTOSSMini,
    },
    "多模态": {
        "CLIP": CLIPModel, "Whisper": Whisper, "Qwen2-VL": Qwen2VLModel, "Qwen2.5-Omni": Qwen2_5_OmniModel,
    },
    "生成模型": {
        "VAE": ImageVAE, "Causal 3D VAE": CausalVideoVAE, "DiT": DiT, "MM-DiT": MMDiT,
        "Video DiT": VideoDiT, "VAR": VARModel,
    },
}
