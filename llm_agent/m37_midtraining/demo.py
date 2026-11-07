"""继续 next-token 训练与旧知识保持; 小 LLaMA 的固定符号语料, 不代表真实领域语料。"""
from __future__ import annotations
from functools import lru_cache
import copy
import json
from types import SimpleNamespace
import torch
from llm_finetune.run_finetune.common import make_model
from llm_models.training.loss import StandardLMLoss


@torch.no_grad()
def probe(model, domain):
    keys = torch.arange(4, 12)
    inputs = torch.stack((torch.full_like(keys, domain), keys), dim=1)
    labels = keys + (8 if domain == 2 else 0)
    predictions = model(inputs)[:, -1].argmax(-1)
    return float((predictions == labels).float().mean())


def continue_training(model, domains, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    criterion = StandardLMLoss()
    for step in range(steps):
        domain = domains[step % len(domains)]
        keys = torch.arange(4, 12)
        sequence = torch.stack((torch.full_like(keys, domain), keys, keys + (8 if domain == 2 else 0)), dim=1)
        # 两个 next-token 位置都计损失, 未按 assistant 角色遮罩。
        loss = criterion.compute(model(sequence[:, :-1]), sequence[:, 1:])["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


@lru_cache(maxsize=1)
def training_run():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        base = make_model(SimpleNamespace(vocab_size=24), d_model=32, num_layers=1)
        continue_training(base, [1], 100)
        result = {"base": {"old": probe(base, 1), "domain": probe(base, 2)}}
        for mode, schedule in (("domain-only", [2]), ("replay", [2, 2, 1])):
            model = copy.deepcopy(base)
            loss = continue_training(model, schedule, 120)
            result[mode] = {"old": probe(model, 1), "domain": probe(model, 2), "loss": round(loss, 5)}
        return result


def experiment(mode="replay"):
    rows = training_run()
    if mode not in rows:
        raise ValueError(mode)
    row = dict(rows[mode])
    row["trace"] = [{"title": "原始能力", "detail": f"旧域知识正确率 {rows['base']['old']:.0%}", "ok": rows["base"]["old"] == 1},
                    {"title": "新域训练", "detail": f"新域正确率 {row['domain']:.0%}; 模式 {mode}", "ok": row["domain"] == 1},
                    {"title": "保持检查", "detail": f"旧域正确率 {row['old']:.0%}; 总 loss 不代替保持指标", "ok": row["old"] == 1}]
    return row


def main():
    rows = training_run()
    assert rows["base"]["old"] == 1
    assert rows["replay"]["domain"] == rows["replay"]["old"] == 1
    assert rows["domain-only"]["domain"] == 1
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
