"""Workflow / Agent / 混合选择, 结构化状态栏, 以及 KV 不能任意拼接的反例。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Request:
    name: str
    known_steps: bool
    ambiguous: bool
    high_risk: bool


REQUESTS = (Request("fixed-report", True, False, False), Request("investigate", False, False, False),
            Request("ambiguous-owner", False, True, False), Request("transfer", True, False, True))


def choose(request, mode):
    if request.ambiguous or request.high_risk:
        return "human"
    if mode == "workflow":
        return "workflow"
    if mode == "agent":
        return "agent"
    if mode == "hybrid":
        return "workflow" if request.known_steps else "agent"
    raise ValueError(mode)


def status_bar(goal, completed, remaining, pending):
    if remaining < 0:
        raise ValueError("预算不能为负")
    return {"goal": goal, "completed": tuple(completed), "remaining_calls": remaining, "pending": tuple(pending)}


def contextual_states(tokens):
    # 第一层因果上下文混合: 后缀的状态依赖前缀。
    return [value + sum(tokens[:i+1]) / (i+1) for i, value in enumerate(tokens)]


def attend(states):
    query = states[-1]
    logits = [query * x / 10 for x in states]
    weights = [math.exp(x-max(logits)) for x in logits]
    return sum(w*x for w, x in zip(weights, states)) / sum(weights)


def cache_experiment():
    old_prefix, new_prefix, suffix = [1., 2.], [9., 2.], [3., 4.]
    old_states = contextual_states(old_prefix + suffix)
    truth = attend(contextual_states(new_prefix + suffix))
    naive = attend(contextual_states(new_prefix) + old_states[len(old_prefix):])
    key = lambda prefix: hashlib.sha256(json.dumps({"model": "toy-v1", "prefix": prefix, "positions": list(range(len(prefix)))}).encode()).hexdigest()
    return {"reference": round(truth, 6), "spliced": round(naive, 6), "reusable": key(old_prefix) == key(new_prefix)}


def experiment(mode="hybrid"):
    if mode not in {"workflow", "agent", "hybrid"}:
        raise ValueError(mode)
    trace, solved, cost = [], 0, 0
    for request in REQUESTS:
        route = choose(request, mode)
        passed = route == "human" or route == "agent" or request.known_steps
        # 明确的教学费用模型, 不是实测 API 价格或普遍成本排序。
        cost += {"human": 0, "workflow": 1, "agent": 3}[route]
        solved += passed
        trace.append({"title": request.name, "detail": f"route={route}; 完成或安全转交={passed}", "ok": passed})
    cache = cache_experiment()
    trace.append({"title": "组合上下文缓存", "detail": f"完整重算 {cache['reference']}; 拼旧后缀 {cache['spliced']}", "ok": False})
    return {"handled": solved, "cost_units": cost, "cache": cache,
            "status": status_bar("完成报告", ["检索"], 2, ["核对引用"]), "trace": trace}


def main():
    rows = {m: experiment(m) for m in ("workflow", "agent", "hybrid")}
    assert rows["workflow"]["handled"] == 3 < rows["hybrid"]["handled"] == 4
    assert rows["hybrid"]["cost_units"] < rows["agent"]["cost_units"]
    assert not rows["hybrid"]["cache"]["reusable"]
    assert rows["hybrid"]["cache"]["reference"] != rows["hybrid"]["cache"]["spliced"]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
