"""
ORPO — Odds Ratio Preference Optimization (Hong et al., 2024)

是什么: SFT loss + 一个 odds-ratio 惩罚项, **一个阶段**同时完成 "学会任务" 和 "偏好对齐", 不需要 ref。
解决什么: SFT 只抬高 chosen, 但 rejected 往往和 chosen 很像, 概率被一起抬高; DPO / SimPO 又都需要先有一个 SFT 好的起点。
核心公式:  odds(y|x) = p / (1 − p),  p = exp( (1/|y|)·log π_θ(y|x) )        (长度归一化后的 "平均每 token 概率")
           L = NLL(y_w)  +  λ · ( − log σ( log odds(y_w) − log odds(y_l) ) )
读代码时盯住: `log1p(-exp(logp))` —— log(1−p)。p → 1 时它 → −∞, 所以 odds 比 "概率之比" 对已经很自信的 chosen 更敏感。
代价: λ 要调。NLL 项是锚 (取代 ref 的角色): 它直接抬高 chosen 的概率, DPO / SimPO 的 loss 里没有这一项。
      它只提供一股向上的梯度。λ 大时 odds-ratio 项仍可能把 chosen 拉低。
      本配置下 chosen 的 log-prob 升没升, 见 run_finetune/simpo_orpo/train_simpo_orpo.py 的实测。
与论文的差异: loss 公式相同。默认 λ=0.5 是本库的取值。
"""

from typing import Dict

import torch
import torch.nn.functional as F

from llm_models.training.loss import LossComputer
from llm_finetune.methods.dpo import _preference_metrics, compute_sequence_logprobs


class ORPOLoss(LossComputer):
    """
    lam: odds-ratio 项的权重 λ。0 = 只剩 chosen 上的 SFT loss; 越大越强调把 rejected 压到 chosen 之下。

    model_output = 不带 ref 的 PairwiseForward 的输出 {"chosen", "rejected"}, 各是 logits [B, T, V]。
    """

    def __init__(self, lam: float = 0.5) -> None:
        self.lam = lam

    def compute(self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        logp_w = compute_sequence_logprobs(model_output["chosen"], labels["chosen"], average=True)      # [B] ≤ 0
        logp_l = compute_sequence_logprobs(model_output["rejected"], labels["rejected"], average=True)  # [B]

        def log_odds(logp: torch.Tensor) -> torch.Tensor:
            # log(p/(1−p)) = log p − log(1 − e^{log p}); clamp 防 p=1 时 log(0)
            return logp - torch.log1p(-torch.exp(logp).clamp(max=1 - 1e-6))

        ratio = log_odds(logp_w) - log_odds(logp_l)                   # [B]
        or_loss = -F.logsigmoid(ratio).mean()
        nll = -logp_w.mean()                                          # 就是 chosen 上的 SFT loss
        return {
            "total_loss": nll + self.lam * or_loss,
            "nll": nll.detach(),
            "logp_chosen": logp_w.detach().mean(),
            **_preference_metrics(logp_w, logp_l),
        }
