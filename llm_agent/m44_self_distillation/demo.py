"""同一线性策略: 学生在自己的轨迹上行动, 教师分支多看一份已验证的方向提示。

复用现有 reverse KL; 四个参数的控制例子隔离特权信息边界, 不冒充语言模型 OPSD 论文复现。
"""
from __future__ import annotations
from functools import lru_cache
import json
import torch
from torch import nn
from llm_finetune.methods.on_policy_distill import token_kl

TRAIN_STARTS = (-3., -1., 1., 3.)
TEST_STARTS = (-4., -2., 2., 4.)


def rollout(model, start, sample=False):
    position, states = start, []
    for _ in range(7):
        if position == 0:
            break
        states.append(position)
        logits = model(torch.tensor([[position, 0.]]))[0]  # 部署观察不含提示。
        index = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax()
        position += -1 if int(index) == 0 else 1
    return states, position == 0


@torch.no_grad()
def score(model):
    return sum(rollout(model, x)[1] for x in TEST_STARTS) / len(TEST_STARTS)


@lru_cache(maxsize=2)
def train(privileged=True):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(31)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            # 教师已有利用提示的能力; 学生状态权重故意指向反方向。
            model.weight.copy_(torch.tensor([[-.1, 2.], [.1, -2.]]))
        before = score(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.08)
        losses, states_seen = [], 0
        for step in range(35):
            positions = [p for start in TRAIN_STARTS for p in rollout(model, start, sample=True)[0]]
            states_seen += len(positions)
            student_input = torch.tensor([[p, 0.] for p in positions])
            hints = [1. if p > 0 else -1. for p in positions]
            teacher_input = torch.tensor([[p, hint if privileged else 0.] for p, hint in zip(positions, hints)])
            with torch.no_grad():
                teacher_logits = model(teacher_input)  # 同一个模型, 梯度在教师分支截断。
            loss = token_kl(model(student_input), teacher_logits).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in {0, 34}:
                losses.append(round(float(loss.detach()), 6))
        after = score(model)
        return {"before": before, "after": after, "losses": losses, "states": states_seen,
                "state_weight": round(float(model.weight[0, 0].detach()), 6),
                "hint_weight": float(model.weight[0, 1].detach())}


def experiment(mode="privileged"):
    if mode not in {"before", "no-hint", "privileged"}:
        raise ValueError(mode)
    row = dict(train(mode != "no-hint"))
    row["score"] = row["before"] if mode == "before" else row["after"]
    row["trace"] = [{"title": "学生采样", "detail": "按当前策略走向目标, 状态分布由自己的动作产生", "ok": True},
                    {"title": "教师条件", "detail": "同一模型额外看已验证方向提示" if mode != "no-hint" else "没有额外信息, 两分布相同", "ok": mode != "no-hint"},
                    {"title": "无提示测试", "detail": f"隔离起点成功率 {row['score']:.0%}; 运行时提示恒为零", "ok": row["score"] == 1}]
    return row


def main():
    good, same = train(True), train(False)
    assert good["before"] == 0 and good["after"] == 1
    assert same["after"] == 0 and same["losses"] == [0., 0.]
    assert good["hint_weight"] == same["hint_weight"] == 2., "教师分支不能反传更新特权通道"
    assert set(TRAIN_STARTS).isdisjoint(TEST_STARTS)
    print(json.dumps({"privileged": good, "no-hint": same}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
