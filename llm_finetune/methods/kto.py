"""
KTO — Kahneman-Tversky Optimization (Ethayarajh et al., 2024)

是什么: 每条样本只要一个 "好 / 坏" 标签, 不需要同一个 prompt 下的一对 (y_w, y_l)。
解决什么: DPO 的数据单位是 "同题两条回复 + 谁更好"; 线上日志里更常见的是单条回复的 👍 / 👎,
          而且好坏数量严重不均 (大多数回复没人点 👎)。KTO 直接吃这种数据。
核心公式:  r(x,y) = log π_θ(y|x) − log π_ref(y|x)                                   (与 DPO 同一个隐式奖励, 只是没乘 β)
           z0 = max(0, mean_{错配的 (x, y')} r(x, y'))  且不回传梯度                 (对 KL(π_θ‖π_ref) 的有偏估计, 当参考点)
           v(x,y) = λ_D·σ(β(r − z0))   若 y 是好回复
                    λ_U·σ(β(z0 − r))   若 y 是坏回复
           L = E[ λ_y − v(x,y) ]
           直觉: 前景理论的 "价值函数" —— 好回复的收益要高过参考点, 坏回复的损失要低于参考点; σ 让收益 / 损失都饱和。
读代码时盯住: `z0` —— 它用 **错配的** (x_i, y_{i+1}) 估, 不是本条样本; 且 detach。
              正是这个参考点让单条样本也有 "比什么好" 的基准, 代替了 DPO 里的 "对面那条"。
类别不均: 论文建议 λ_D·n_D / (λ_U·n_U) ∈ [1, 4/3], 即少数类的权重按比例调高。
"""

from typing import Dict

import torch

from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer
from llm_finetune.data.tasks import SeqTask, make_labels
from llm_finetune.methods.dpo import PairwiseForward, compute_sequence_logprobs


class UnpairedDataGenerator(SyntheticDataGenerator):
    """
    把偏好对拆开: 每个 prompt 只给**一条**回复 —— 以 desirable_frac 的概率给正确回复 (好), 否则给损坏回复 (坏)。
    同一个 prompt 永远不会同时出现好和坏, 所以 DPO 在这份数据上一对也凑不出来。
    另外附一份错配的 (x_i, y_{i+1}), 只用来估 z0。
    """

    def __init__(self, task: SeqTask, batch_size: int = 128, desirable_frac: float = 0.5, split: str = "train") -> None:
        self.task, self.batch_size, self.desirable_frac, self.split = task, batch_size, desirable_frac, split
        self.fixed = False

    def _sample(self) -> Dict[str, torch.Tensor]:
        P = self.task.prompt_len
        prompts = self.task.sample_prompts(self.batch_size, self.split)                 # [B, P]
        good = self.task.target(prompts)
        desirable = torch.rand(self.batch_size) < self.desirable_frac                   # [B]
        resp = torch.where(desirable.unsqueeze(1), good, self.task.corrupt(good))      # [B, R]
        ids, labels = make_labels(torch.cat([prompts, resp], dim=1), P)
        kl_ids, kl_labels = make_labels(torch.cat([prompts, resp.roll(1, dims=0)], dim=1), P)   # 回复错配到别的 prompt
        return {"input_ids": ids, "kl_input_ids": kl_ids,
                "labels": {"sample": labels, "kl": kl_labels, "desirable": desirable}}


class KTOForward(PairwiseForward):
    """复用 PairwiseForward 的两路前向 + 冻结 ref (不进 optimizer), 只把两路改名: 本条样本 / 错配的 KL 样本。"""

    def forward(self, input_ids, kl_input_ids) -> Dict[str, torch.Tensor]:
        out = super().forward(input_ids, kl_input_ids)
        return {k.replace("chosen", "sample").replace("rejected", "kl"): v for k, v in out.items()}


class KTOLoss(LossComputer):
    def __init__(self, beta: float = 0.1, lambda_d: float = 1.0, lambda_u: float = 1.0) -> None:
        self.beta, self.lambda_d, self.lambda_u = beta, lambda_d, lambda_u

    def compute(self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        logp = {k: compute_sequence_logprobs(v, labels[k.replace("ref_", "")]) for k, v in model_output.items()}
        r = logp["sample"] - logp["ref_sample"]                                         # [B]
        kl_est = (logp["kl"] - logp["ref_kl"]).mean().detach()                          # 可能为负: y' 不是从 π_θ 采的
        z0 = kl_est.clamp(min=0)                                                        # 标量参考点
        d = labels["desirable"]
        v = torch.where(d, self.lambda_d * torch.sigmoid(self.beta * (r - z0)),
                        self.lambda_u * torch.sigmoid(self.beta * (z0 - r)))
        lam = torch.where(d, torch.tensor(self.lambda_d), torch.tensor(self.lambda_u))
        return {
            "total_loss": (lam - v).mean(),
            "z0": z0,
            "kl_estimate": kl_est,
            "reward_desirable": r[d].detach().mean() if d.any() else torch.tensor(0.0),
            "reward_undesirable": r[~d].detach().mean() if (~d).any() else torch.tensor(0.0),
        }
