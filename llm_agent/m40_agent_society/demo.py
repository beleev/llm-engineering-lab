"""重复博弈、个体收益与集体目标。规则策略不是关于人类社会的模型。"""
from __future__ import annotations
import json

PAYOFF = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}


def action(strategy, opponent_history):
    if strategy == "cooperate":
        return "C"
    if strategy == "defect":
        return "D"
    if strategy == "reciprocal":
        return opponent_history[-1] if opponent_history else "C"
    raise ValueError(strategy)


def play(left, right, rounds=6):
    histories, scores, trace = [[], []], [0, 0], []
    for step in range(rounds):
        choices = (action(left, histories[1]), action(right, histories[0]))
        payoff = PAYOFF[choices]
        for i in (0, 1):
            histories[i].append(choices[i])
            scores[i] += payoff[i]
        trace.append({"title": f"第 {step+1} 轮", "detail": f"动作 {choices}, 收益 {payoff}, 累计 {scores}", "ok": choices == ("C", "C")})
    return {"left": scores[0], "right": scores[1], "social": sum(scores), "trace": trace}


def experiment(mode="reciprocal"):
    strategies = {"exploit": ("defect", "cooperate"), "mutual": ("defect", "defect"), "reciprocal": ("reciprocal", "reciprocal")}
    if mode not in strategies:
        raise ValueError(mode)
    return play(*strategies[mode])


def main():
    rows = {m: experiment(m) for m in ("exploit", "mutual", "reciprocal")}
    assert rows["exploit"]["left"] > rows["reciprocal"]["left"]
    assert rows["mutual"]["social"] < rows["reciprocal"]["social"]
    assert play("reciprocal", "defect")["right"] < rows["exploit"]["left"]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
