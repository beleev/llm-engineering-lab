"""
损失函数 — 每种模型形态配一个 LossComputer (策略模式)

是什么: Trainer 持有一个 LossComputer, 每步调 `compute(model_output, labels)`, 自己不关心模型类型。
    StandardLMLoss   next-token 交叉熵 (GPT-3 / Transformer / LLaMA / Mistral / Qwen3-Next / Mamba /
                     Whisper / Qwen2-VL)
    MTPLoss          主 CE + λ·多 token 预测 CE (MTPLLaMA / DeepSeek-V3 MTP)
    MoELMLoss        交叉熵 + Switch-Transformer 风格的负载均衡 aux loss
                     (DeepSeekV3 / V3.2 / Mixtral / GPT-OSS)
    OmniLoss         文本 (Thinker) + 音频 (Talker) 两个分支加权 (Qwen2.5-Omni)
    MaskedLMLoss     BERT 风格 MLM 交叉熵 (只对被 mask 的位置算 loss)
    ContrastiveLoss  CLIP 对称对比 loss (image↔text 双向 CE)
    VAELoss          重建 (MSE) + KL(q || N(0, I))
    VARLoss          next-scale 交叉熵 (整级 token 并行预测);
                     tokenizer 的 vq_loss 不在这里, 见 MultiScaleVQ.forward
    DiffusionLoss    见 training/diffusion.py
约定:
    - labels 里的 -100 不算 loss (PyTorch cross_entropy 默认的 ignore_index)。
      pad / 多模态前缀 token 的位置填 -100, 这些位置没有梯度。
    - 返回的 dict 必含 "total_loss", Trainer 对它调 `.backward()`; 其余键只进日志, 不反传。
读代码时盯住: 每个 compute 里的 reshape。cross_entropy 要 (N, C) 的 logits 和 (N,) 的标签,
             所以 [B, T, V] 先摊平成 [B*T, V]。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class LossComputer(ABC):
    """损失计算的接口。子类实现 `compute()`: 收模型输出和标签, 返回含 "total_loss" 的 dict。"""

    @abstractmethod
    def compute(
        self,
        model_output: Any,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """model_output: 模型 forward 的返回值, 类型由子类约定 (Tensor / tuple / dict)。

        labels: 目标标签。kwargs: Trainer 额外传的标签, 如 audio_labels。
        返回 dict: 必含 "total_loss", 可附带各分量 loss 供日志用。
        """
        raise NotImplementedError


class StandardLMLoss(LossComputer):
    """
    next-token 交叉熵: 所有 (batch, 位置) 的平均 NLL。未训练模型应 ≈ ln V (均匀瞎猜)。

    labels == -100 的位置被跳过 (PyTorch cross_entropy 的默认 ignore_index):
    数据侧只需把 pad / 多模态前缀位置填 -100。
    """

    def compute(
        self,
        model_output: torch.Tensor,     # logits [B, T, V]
        labels: torch.Tensor,           # [B, T]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]  cross_entropy 要 (N, C)
            labels.reshape(-1),                    # [B*T]
            ignore_index=-100,
        )
        # lm_loss 与 total_loss 是同一个张量; 留两个键, 日志的列名和 MoELMLoss 对得上
        return {"total_loss": loss, "lm_loss": loss}


class MTPLoss(LossComputer):
    """
    Multi-Token Prediction 联合损失 (配 models/language_models/mtp.py::MTPLLaMA):

        L = CE(main) + λ · mean_k CE(mtp_k)          DeepSeek-V3: λ = 0.3 → 0.1

    标签对齐 (labels[i] = t_{i+1} 是标准 next-token 标签):
        MTP-k 在位置 i 预测 t_{i+1+k} = labels[i+k]
        → labels 左移 k 位作第 k 级目标, 末尾 k 个位置没有未来 token, 置 -100
    """

    def __init__(self, mtp_lambda: float = 0.3, ignore_index: int = -100) -> None:
        self.mtp_lambda = mtp_lambda
        self.ignore_index = ignore_index

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]
            labels.reshape(-1),                    # [B*T]
            ignore_index=self.ignore_index,
        )

    def compute(
        self,
        model_output: Dict[str, Any],   # {"logits": [B,T,V], "mtp_logits": List[[B,T,V]]}
        labels: torch.Tensor,           # [B, T]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        main_loss = self._ce(model_output["logits"], labels)

        mtp_losses: List[torch.Tensor] = []
        for k, logits_k in enumerate(model_output["mtp_logits"], start=1):
            labels_k = torch.full_like(labels, self.ignore_index)   # [B, T] 先全填 -100
            # 位置 i 的目标是 labels[i+k]: 前 T-k 个位置有目标, 末尾 k 个保持 -100
            labels_k[:, :-k] = labels[:, k:]       # 目标整体左移 k 位
            mtp_losses.append(self._ce(logits_k, labels_k))

        mtp_loss = torch.stack(mtp_losses).mean()  # 各级 MTP 头的 CE 取平均 (公式里的 mean_k)
        return {
            "total_loss": main_loss + self.mtp_lambda * mtp_loss,
            "main_loss": main_loss.detach(),       # 只进日志: detach 后不再拖着计算图
            "mtp_loss": mtp_loss.detach(),
        }


class MoELMLoss(LossComputer):
    """
    MoE 语言模型损失 (Mixtral / DeepSeekV3 / V3.2 / GPT-OSS):

        total = lm_loss + aux_loss_weight · aux_loss  [+ index_loss_weight · index_loss]

    aux_loss (Switch-Transformer 负载均衡), 每层:
        f_i = 专家 i 被选中的次数 / token 数        # 不可导; top-K 下 Σ_i f_i = K
        P_i = mean_t p_t[i]                         # 可导; p_t 是归一化到行和为 1 的路由概率
        aux = E · Σ_i f_i · P_i
      完全均衡时 f_i = K/E, P_i = 1/E → aux = E·E·(K/E)(1/E) = **K** (不是 1; top-1 时才是 1)。
      路由坍塌到固定 K 个专家时 → aux ≈ E。梯度只经 P_i 回到 router, 所以 router_logits 不能 detach。
      p_t 取自 routing_info["routing_probs"]: Mixtral 是 softmax (本来就归一),
      DeepSeek 是 sigmoid 分数, 这里除以行和 (与 V3 论文的 s'_i = s_i / Σ_j s_j 一致),
      而不是拿 router_logits 另算一遍与模型实际路由无关的 softmax。

    index_loss: DeepSeek-V3.2 的 indexer 对齐 KL, 由模型放在 routing_info["index_loss"];
      没有这个键的模型 (Mixtral / V3 / GPT-OSS) 该项为 0, 不出现在返回 dict 里。

    Args:
        aux_loss_weight:   经验值 ~0.01; 太大牺牲主任务, 太小路由坍塌。
        index_loss_weight: indexer 与主模型参数不相交 (输入/目标都 detach), 取 1.0 即可。
    """

    def __init__(self, aux_loss_weight: float = 0.01, index_loss_weight: float = 1.0):
        self.aux_loss_weight = aux_loss_weight
        self.index_loss_weight = index_loss_weight

    def compute(
        self,
        model_output: tuple,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits, all_routing_info = model_output  # [B, T, V], 每层一个 routing_info dict

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]
            labels.reshape(-1),                    # [B*T]
            ignore_index=-100,
        )

        # 没有 MoE 层时返回与 logits 同设备的 0。
        # 写成 torch.tensor(0.0) 会落在 CPU, 模型在 GPU 上时相加报错
        aux_loss = (
            self._compute_load_balancing_loss(all_routing_info)
            if all_routing_info else logits.new_zeros(())
        )
        out = {"lm_loss": lm_loss, "aux_loss": aux_loss}
        total_loss = lm_loss + self.aux_loss_weight * aux_loss

        # 只有 DeepSeek-V3.2 的层会带 index_loss; 各层取平均
        index_losses = [info["index_loss"] for info in all_routing_info if "index_loss" in info]
        if index_losses:
            out["index_loss"] = torch.stack(index_losses).mean()
            total_loss = total_loss + self.index_loss_weight * out["index_loss"]

        return {"total_loss": total_loss, **out}

    @staticmethod
    def _compute_load_balancing_loss(
        all_routing_info: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """跨层平均的 E·Σ f_i·P_i; 完全均衡 = K, 完全坍塌 = E。all_routing_info 非空。"""
        layer_losses = []
        for info in all_routing_info:
            # N = B*T 个 token, E = 专家数, K = 每个 token 选中的专家数
            probs = info["routing_probs"].float()              # [N, E] float32 防 fp16 溢出
            probs = probs / probs.sum(dim=-1, keepdim=True)    # sigmoid 分数 → 行和为 1
            num_experts = probs.size(-1)

            # f_i: one-hot 计数后按 token 求均值 (统计量, 无梯度)
            # selected_experts [N, K] → one_hot [N, K, E] → 对 K 求和 [N, E] → 对 token 求均值 [E]
            fraction = F.one_hot(info["selected_experts"], num_experts).sum(dim=1).float().mean(dim=0)
            mean_prob = probs.mean(dim=0)                      # P_i: [N, E] → [E], 梯度从这里回 router
            layer_losses.append(num_experts * (fraction * mean_prob).sum())

        return torch.stack(layer_losses).mean()


class OmniLoss(LossComputer):
    """
    Qwen2.5-Omni 的两分支损失:

        total_loss = text_loss + audio_loss_weight · audio_loss

    text_loss:  Thinker (主 LLM) 的 next-token 交叉熵, 监督文本生成。
    audio_loss: Talker (语音头) 对离散音频 token 的自回归交叉熵。
                audio_logits 或 audio_labels 缺一个就跳过, 只算 text_loss
                (训练数据可能只有文本标注, 没有音频 ground truth)。
    audio_loss_weight: 音频 loss 乘的系数。两个 loss 的数值范围可能不同
                (文本词表和音频词表大小不一样), 用它调两者在总 loss 里的比例。

    model_output: {"text_logits": [B, N, V], "audio_logits": [B, T_a, V_audio] 或 None}
    labels [B, N]; kwargs["audio_labels"] [B, T_a]。N = 多模态前缀 + 文本长度。
    """

    def __init__(self, audio_loss_weight: float = 0.5):
        self.audio_loss_weight = audio_loss_weight

    def compute(
        self,
        model_output: Dict[str, Optional[torch.Tensor]],
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        text_logits = model_output["text_logits"]
        audio_logits = model_output.get("audio_logits")  # 可能为 None

        # 1) 文本 loss: 与标准 LM 相同, 多模态前缀位置已通过 -100 屏蔽
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),   # [B*N, V]
            labels.reshape(-1),                              # [B*N]
            ignore_index=-100,
        )

        result: Dict[str, torch.Tensor] = {
            "text_loss": text_loss,
        }

        # 2) 音频 loss (可选): 只有当 Talker 有输出且 batch 提供 audio_labels 时才计算
        audio_labels = kwargs.get("audio_labels")
        if audio_logits is not None and audio_labels is not None:
            audio_loss = F.cross_entropy(
                audio_logits.reshape(-1, audio_logits.size(-1)),   # [B*T_a, V_audio]
                audio_labels.reshape(-1),                          # [B*T_a]
                ignore_index=-100,
            )
            result["audio_loss"] = audio_loss
            result["total_loss"] = text_loss + self.audio_loss_weight * audio_loss
        else:
            # 没有音频分支: 只算文本
            result["total_loss"] = text_loss

        return result


class MaskedLMLoss(LossComputer):
    """
    BERT 的 Masked Language Modeling 损失: 只在被 mask 的位置算交叉熵。

    代码和 StandardLMLoss 一样 (logits [B, T, V], labels [B, T]), 差别全在 labels:
        - 被选中的位置: label = 原 token id
        - 其余位置:     label = -100, 不算 loss
    含义也不同: 这里是 "还原被遮住的 token", 不是 "预测下一个"。

    labels 由 MaskedLMDataGenerator 造: 随机挑 15% 位置, 其中
        80% 换成 [MASK] token id / 10% 换成随机 token / 10% 保持原样。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]
            labels.reshape(-1),                    # [B*T], 约 85% 是 -100
            ignore_index=-100,
        )
        return {"total_loss": loss, "mlm_loss": loss}


class ContrastiveLoss(LossComputer):
    """
    CLIP 对称对比 loss (InfoNCE 形式)

    给定 batch 内 B 对 (image, text):
        logits_per_image = logit_scale · image_feats @ text_feats^T        # [B, B]
        logits_per_text  = logits_per_image.T
        对角线为正样本, 其余为负样本
        loss = (CE(logits_per_image, arange(B)) + CE(logits_per_text, arange(B))) / 2

    为什么对称:
        CE(logits)  是每张图在 B 条文本里选对 (行归一化)。
        CE(logitsᵀ) 是每条文本在 B 张图里选对 (列归一化)。
        只做一个方向时, 另一个方向的负样本从不参与归一化, 对应的检索任务没人监督。

    model_output 必须是 CLIPModel.forward 的返回 dict:
        image_features: [B, D]  (已 L2 normalize)
        text_features:  [B, D]  (已 L2 normalize)
        logit_scale:    scalar
    """

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Any = None,  # 不需要显式 labels (对角线隐式给出)
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        image_feats = model_output["image_features"]
        text_feats = model_output["text_features"]
        logit_scale = model_output["logit_scale"]

        B = image_feats.size(0)
        # 特征已归一化, 内积就是余弦相似度 ∈ [-1, 1]; 乘 logit_scale (温度的倒数) 拉开差距
        logits = logit_scale * image_feats @ text_feats.t()     # [B, D] @ [D, B] → [B, B]
        targets = torch.arange(B, device=logits.device)         # [B] 第 i 张图的正确文本就是第 i 条

        loss_i2t = F.cross_entropy(logits, targets)             # image → text
        loss_t2i = F.cross_entropy(logits.t(), targets)         # text  → image
        loss = (loss_i2t + loss_t2i) / 2

        return {
            "total_loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logit_scale": logit_scale.detach(),                # 只进日志, 看温度学到了多少
        }


class VAELoss(LossComputer):
    """
    VAE 重建 + KL 正则

    loss = recon_weight · MSE(x̂, x) + kl_weight · KL(q || N(0, I))
    KL 闭式: -0.5 · Σ (1 + logσ² - μ² - σ²)

    适用模型: ImageVAE, CausalVideoVAE (只要 forward 返回 {recon, mean, logvar}
    且 batch 中的 "labels" 实为原输入 x)

    两项的量纲不同:
        recon 是逐元素均值, 与图像大小无关。
        KL 是每个样本对 latent 全部元素求和, 随 latent 元素数增长。
        所以 kl_weight 要配合 latent 大小调。

    Args:
        recon_weight: 重建项系数 (默认 1.0)
        kl_weight:    KL 项系数 (默认 1e-4, SD 1.5 用到 1e-6 级, 控制潜空间"紧致度")
    """

    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 1e-4):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        recon = model_output["recon"]
        mean = model_output["mean"]
        logvar = model_output["logvar"]

        recon_loss = F.mse_loss(recon, labels)          # 标量: 全部像素的均方误差
        # 每个 latent 元素的 KL(N(μ, σ²) || N(0, 1)) 闭式; logvar = log σ², 所以 σ² = logvar.exp()
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())   # 与 mean 同形 [B, C, h, w] (视频多一维 T)
        # 每个样本对 latent 全部元素求和 (VAE 的标准定义), 再对 batch 取均值
        kl_loss = kl.flatten(1).sum(dim=1).mean()       # [B, C, h, w] → [B, C·h·w] → [B] → 标量

        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class VARLoss(LossComputer):
    """
    VAR (next-scale prediction) loss: 对 token 金字塔全部 L=Σs² 个位置做交叉熵。

    tokenizer 已冻结, 只训 Transformer。没有 shift-by-one —— 第 k 级位置的输入来自
    更粗的级, label 是本级 token, 对齐由 VARModel.forward 完成:
        logits: [B, L, K]   labels: [B, L]   (K = 码本大小, 初始 CE ≈ ln K)

    Trainer 传入的 labels 参数被忽略 (token 要经 tokenizer 才有, 由 model_output 自带)。
    """

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Any = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output["logits"]                 # [B, L, K]
        target = model_output["labels"]                 # [B, L] 每个位置都有标签, 不需要 ignore_index
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),        # [B*L, K]
            target.reshape(-1),                         # [B*L]
        )
        return {"total_loss": loss, "ce_loss": loss}
