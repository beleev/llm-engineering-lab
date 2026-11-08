"""IPO 的平方 log-ratio 目标；共用 DPO 的成对前向与序列概率。"""
import math
import torch
from llm_models.training.loss import LossComputer
from llm_finetune.methods.dpo import compute_sequence_logprobs


class IPOLoss(LossComputer):
    def __init__(self, beta=.5):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta 必须大于零")
        self.beta = beta

    def compute(self, model_output, labels, **kwargs):
        for name in ("chosen", "rejected"):
            if labels[name].ndim != 2 or not labels[name].numel() or (labels[name] != -100).sum(-1).eq(0).any():
                raise ValueError("每个回答至少需要一个有效 token")
        # 原始 IPO 的实现惯例：每个回答按有效 token 平均，避免长度缩放目标。
        lp = {k: compute_sequence_logprobs(v, labels[k.replace("ref_", "")], average=True)
              for k, v in model_output.items()}
        margin = lp["chosen"]-lp["rejected"]-lp["ref_chosen"]+lp["ref_rejected"]
        loss = ((margin-1/(2*self.beta))**2).mean()
        return {"total_loss": loss, "margin": margin.detach().mean()}
