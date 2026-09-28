"""
RLAIF / Constitutional AI (Bai et al., 2022; Lee et al., 2023)

是什么: 偏好标签不找人标, 让 "AI 反馈者" 按一份写成条文的 "宪法" 打分、批评、改写, 自动造出偏好对, 再做 DPO / RL。
解决什么: RLHF 的瓶颈是人: 贵、慢、标注员之间不一致, 而且有害内容要人一条条看。
          把 "什么是好回答" 写成原则后, 标签的成本从 "每条一个人" 降到 "每条一次 judge 调用"。
流程:  policy 采样 y  →  critique(y): 逐条原则检查, 给出违反了哪条、在哪
                       →  revise(y):   按批评改写出 y'
                       →  偏好对 (x, chosen = y', rejected = y)  →  现成的 DPOLoss 训一轮
代价 / 上限: 标签质量 = judge 质量。judge 漏检的违规, 在偏好对里 chosen 和 rejected 同时带着, 训练对它没有直接压力。
            这里的 judge 是规则函数 (不是 LLM), 这个上限一眼可见: 宪法写了 "变体 14 也算脏话", 规则只查了 15。
读代码时盯住: `CONSTITUTION` —— 每条原则 = (名字, 条文, 检查函数, 修改函数); 条文是给人看的, 真正生效的是后两个。
"""

from typing import Callable, Dict, List, NamedTuple

import torch

from llm_models.training.data import SyntheticDataGenerator
from llm_finetune.data.tasks import EOS, PAD, completion_mask, make_labels

SWEAR, SWEAR_VARIANT, PHONE, MASK = 15, 14, 13, 3     # 玩具词表里的 "脏词"、它的变体、"电话号码"、打码符 "***"


class Principle(NamedTuple):
    name: str
    text: str                                       # 宪法条文 (自然语言)
    violated: Callable[[List[int]], List[int]]      # 回复内容 → 违规位置 (空 = 没违反)
    fix: Callable[[List[int]], List[int]]           # 回复内容 → 改写后的内容


CONSTITUTION: List[Principle] = [
    Principle("无害", f"不要说脏话 ({SWEAR}, 以及它的变体 {SWEAR_VARIANT}); 需要复述时用 *** 代替",
              # judge 的实现只查了 SWEAR —— 条文和实现之间的这道缝, 就是 AI 反馈的质量上限 (见 train_rlaif.py)
              lambda y: [i for i, t in enumerate(y) if t == SWEAR],
              lambda y: [MASK if t == SWEAR else t for t in y]),
    Principle("隐私", f"不要复述用户的电话号码 ({PHONE}); 用 *** 代替",
              lambda y: [i for i, t in enumerate(y) if t == PHONE],
              lambda y: [MASK if t == PHONE else t for t in y]),
]


def content(response: torch.Tensor) -> List[int]:
    """[R] → 第一个 EOS 之前的 token (去掉 PAD)。"""
    out = []
    for t in response.tolist():
        if t == EOS:
            break
        if t != PAD:
            out.append(t)
    return out


def critique(response: torch.Tensor) -> List[str]:
    """AI 反馈者: 逐条原则检查, 返回批评 (空列表 = 没发现问题)。真 CAI 里这一步是 LLM 按条文写出的一段话。"""
    y = content(response)
    return [f"违反「{p.name}」: 位置 {pos}" for p in CONSTITUTION if (pos := p.violated(y))]


def revise(response: torch.Tensor) -> torch.Tensor:
    """按批评改写: 依次套用每条原则的 fix。长度补 PAD 回 R。"""
    y = content(response)
    for p in CONSTITUTION:
        y = p.fix(y)
    R = len(response)
    y = y[: R - 1] + [EOS]
    return torch.tensor(y + [PAD] * (R - len(y)))


def violation_rate(responses: torch.Tensor) -> Dict[str, float]:
    """每条原则的违规比例 + "任一违规"。只用 judge 自己的规则 —— judge 看不见的, 这里也看不见。"""
    ys = [content(r) for r in responses]
    rates = {p.name: sum(bool(p.violated(y)) for y in ys) / len(ys) for p in CONSTITUTION}
    rates["任一"] = sum(any(p.violated(y) for p in CONSTITUTION) for y in ys) / len(ys)
    return rates


def build_preference_pairs(prompts: torch.Tensor, responses: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    对每条 policy 回复跑 critique; 有问题的才造一对 (chosen = revise(y), rejected = y), 没问题的丢掉 (它没有 "更好的版本")。
    返回 PairwiseForward / DPOLoss 需要的全部张量: {chosen,rejected}_{input_ids,attention_mask} + labels。
    """
    responses = responses.masked_fill(~completion_mask(responses), PAD)     # EOS 之后的 token 不属于回复
    keep = [i for i in range(len(prompts)) if critique(responses[i])]
    x, bad = prompts[keep], responses[keep]
    good = torch.stack([revise(r) for r in bad])
    P = prompts.shape[1]
    batch, labels = {}, {}
    for name, resp in (("chosen", good), ("rejected", bad)):
        idx, labels[name] = make_labels(torch.cat([x, resp], dim=1), P)
        batch[f"{name}_input_ids"] = idx
        batch[f"{name}_attention_mask"] = (idx != PAD).long()
    batch["labels"] = labels
    return batch


class AIPreferenceData(SyntheticDataGenerator):
    """离线 DPO: 偏好对只造一次 (一轮 RLAIF), 之后每步从这个池子里随机取一个 batch。"""

    fixed = False

    def __init__(self, pairs: Dict[str, torch.Tensor], batch_size: int = 64) -> None:
        self.pairs, self.batch_size = pairs, batch_size
        self.n = len(pairs["chosen_input_ids"])

    def _sample(self) -> Dict[str, torch.Tensor]:
        i = torch.randint(0, self.n, (self.batch_size,))
        batch = {k: v[i] for k, v in self.pairs.items() if k != "labels"}
        batch["labels"] = {k: v[i] for k, v in self.pairs["labels"].items()}
        return batch
