"""同一成对数据与参考分布：DPO 与 IPO 的目标形状。"""
import json
import torch
from llm_finetune.methods.ipo import IPOLoss
from llm_finetune.methods.dpo import DPOLoss
MODES = ("dpo", "ipo")


def optimize(mode, steps=400):
    criterion = IPOLoss(.5) if mode == "ipo" else DPOLoss(.5)
    logits = torch.zeros(4, 1, 2, requires_grad=True)
    ref = torch.zeros_like(logits)
    chosen = torch.tensor([[0], [1], [0], [1]])
    rejected = 1-chosen
    optimizer = torch.optim.SGD([logits], lr=.2)
    for _ in range(steps):
        output = criterion.compute({"chosen": logits, "rejected": logits,
                                    "ref_chosen": ref, "ref_rejected": ref},
                                   {"chosen": chosen, "rejected": rejected})
        optimizer.zero_grad()
        output["total_loss"].backward()
        optimizer.step()
    sign = torch.tensor([1., -1., 1., -1.])
    return float(((logits[:,0,0]-logits[:,0,1])*sign).mean().detach())


def experiment(mode="ipo"):
    if mode not in MODES:
        raise ValueError(mode)
    gap = optimize(mode)
    return {"metrics": {"平均偏好log-ratio差": round(gap, 5), "IPO目标差": 1.0, "数据对": 4},
            "steps": ["两种方法使用同一四对单 token 回答、均匀冻结参考分布与优化预算。",
                      "DPO 对可分数据继续增加差距；IPO 的平方目标在 1/(2β) 达到最小。",
                      "IPO 这里平均有效回复 token；多 token 回答不能随意与求和目标混用。"],
            "note": "四个独立的条件 logits，验证 loss 的梯度与平衡点，不代表真实偏好泛化，也不说明 IPO 总比 DPO 好。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert abs(optimize("ipo")-1) < .001
    assert optimize("dpo") > optimize("ipo")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
