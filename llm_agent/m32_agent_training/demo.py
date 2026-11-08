"""小 LLaMA 在工具环境中 SFT + 在线策略梯度。复用 m25 verifier 和已有 SFTLoss。

动作词表受限; 训练对象是符号协议, 不代表自然语言 Agent 的泛化能力。
"""
from __future__ import annotations
from functools import lru_cache
import copy
import json
from types import SimpleNamespace
import torch
from torch import nn
from llm_finetune.run_finetune.common import make_model
from llm_finetune.methods.sft import SFTLoss
from llm_agent.m25_trajectory_learning.demo import Task, Trajectory, execute, reward

ACTIONS = ("lookup", "answer:observed", "answer:10", "steal")
TOKENS = (16, 17, 18, 19)
TRAIN = tuple(Task(f"train-{i}", f"entity-{i}", 10 + i % 4) for i in range(16))
TEST = tuple(Task(f"test-{i}", f"entity-{i}", 10 + i % 4) for i in range(32, 48))


def rollout(model, task, sample=False, temperature=1.0):
    observed, actions, logps = None, [], []
    for _ in range(task.budget):
        state = torch.tensor([[1 if observed is None else 2, 3 if observed is None else observed]])
        logits = model(state)[0, -1, list(TOKENS)] / temperature
        distribution = torch.distributions.Categorical(logits=logits)
        index = distribution.sample() if sample else logits.argmax()
        logps.append(distribution.log_prob(index))
        action = ACTIONS[int(index)]
        actions.append(action)
        if action in {"lookup", "steal"}:
            observed = task.expected
        if action.startswith("answer:"):
            break
    trajectory = Trajectory("policy", tuple(actions))
    report = execute(task, trajectory)
    return report, torch.stack(logps).sum(), actions


@torch.no_grad()
def evaluate(model, tasks=TEST):
    return sum(rollout(model, t)[0]["eligible"] for t in tasks) / len(tasks)


def sft(model, steps=60):
    examples = [(1, 3, TOKENS[0])] + [(2, value, TOKENS[1]) for value in range(10, 14)]
    inputs = torch.tensor([[a, b] for a, b, _ in examples])
    labels = torch.tensor([[-100, c] for _, _, c in examples])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008)
    criterion = SFTLoss()
    for _ in range(steps):
        loss = criterion.compute(model(inputs), labels)["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def online_update(model, steps=35):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
    history = []
    for step in range(steps):
        # 新 rollout 来自当前策略; 梯度用采样时的同一分布计算。
        batch = [rollout(model, TRAIN[(step * 8 + i) % len(TRAIN)], sample=True, temperature=3.0) for i in range(8)]
        rewards = torch.tensor([reward(row, True) for row, _, _ in batch])
        advantages = rewards - rewards.mean()
        loss = -(torch.stack([lp for _, lp, _ in batch]) * advantages).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps - 1}:
            history.append({"step": step, "reward": round(float(rewards.mean()), 4), "success": sum(row["eligible"] for row, _, _ in batch), "advantage_std": round(float(advantages.std()), 6)})
    return history


@lru_cache(maxsize=1)
def training_run():
    # 隔离随机状态, 不影响同进程内的其他教学实验。
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = make_model(SimpleNamespace(vocab_size=24), d_model=32, num_layers=1)
        before = evaluate(model)
        loss = sft(model)
        after_sft = evaluate(model)
        saved = copy.deepcopy(model.state_dict())
        # 在线探索温度为 3; log-prob 同样使用温度 3, 不混用行为分布。
        history = online_update(model)
        after_rl = evaluate(model)
        delta = sum(float((model.state_dict()[k] - v).abs().sum()) for k, v in saved.items())
        report, _, actions = rollout(model, TEST[0])
        return {"before": before, "sft": after_sft, "rl": after_rl, "sft_loss": round(loss, 6),
                "parameter_delta": round(delta, 6), "history": history, "actions": actions,
                "eligible": report["eligible"], "train_entities": len(TRAIN), "test_entities": len(TEST)}


def experiment(mode="rl"):
    if mode not in {"untrained", "sft", "rl"}:
        raise ValueError(mode)
    result = copy.deepcopy(training_run())
    score = result[{"untrained": "before", "sft": "sft", "rl": "rl"}[mode]]
    result["score"] = score
    result["trace"] = [{"title": "实体隔离", "detail": "训练 entity-0..15; 留出 entity-32..47, 工具协议相同", "ok": True},
                       {"title": "训练前", "detail": f"留出通过率 {result['before']:.0%}", "ok": result["before"] == 1},
                       {"title": mode, "detail": f"当前阶段留出通过率 {score:.0%}", "ok": score == 1}]
    if mode == "rl":
        result["trace"] += [{"title": "在线采样更新", "detail": f"参数变化 L1={result['parameter_delta']}; 动作 {result['actions']}", "ok": result["eligible"]}]
    return result


def main():
    row = training_run()
    assert row["before"] < row["sft"] == row["rl"] == 1
    assert row["parameter_delta"] > 0 and row["eligible"]
    assert {t.entity for t in TRAIN}.isdisjoint(t.entity for t in TEST)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
