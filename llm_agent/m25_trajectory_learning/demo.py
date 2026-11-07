"""M25 — Agent 轨迹怎样变成 SFT / 偏好 / 过程监督数据。

有限动作环境重算标签, 按实体划分留出集。只构造训练信号, 不再实现 DPO/GRPO,
也不把数据筛选说成模型训练提升; 算法见 llm_finetune。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from llm_agent.core.utils import banner, kv


@dataclass(frozen=True)
class Task:
    id: str
    entity: str
    expected: int
    budget: int = 3


@dataclass(frozen=True)
class Trajectory:
    name: str
    actions: tuple[str, ...]


def candidates(task: Task) -> list[Trajectory]:
    return [Trajectory("verified", ("lookup", "answer:observed")),
            Trajectory("lucky-guess", (f"answer:{task.expected}",)),
            Trajectory("forbidden", ("steal", "answer:observed")),
            Trajectory("over-budget", ("lookup", "lookup", "lookup", "answer:observed")),
            Trajectory("wrong", ("lookup", f"answer:{task.expected + 1}"))]


def execute(task: Task, trajectory: Trajectory) -> dict:
    observed, final, authorized = None, None, True
    labels, reasons = [], []
    for index, action in enumerate(trajectory.actions):
        reason = ""
        if final is not None:
            reason = "after-final"
        elif index >= task.budget:
            reason = "budget"
        elif action == "lookup":
            observed = task.expected
        elif action == "steal":
            # 反例环境故意没拦; 评估器仍必须判过程不合格。
            observed, authorized, reason = task.expected, False, "authorization"
        elif action.startswith("answer:"):
            raw = action.split(":", 1)[1]
            try:
                final = observed if raw == "observed" else int(raw)
            except ValueError:
                reason = "invalid-answer"
            if not reason and (observed is None or not authorized):
                reason = "unsupported"
            if not reason and final != task.expected:
                reason = "wrong-result"
        else:
            reason = "unknown-action"
        labels.append(not reason)
        reasons.append(reason)
    correct = final == task.expected
    process = bool(labels) and all(labels) and final is not None
    return {"result": correct, "process": process, "eligible": correct and process,
            "step_labels": labels, "reasons": reasons, "calls": len(trajectory.actions)}


def reward(report: dict, process_aware: bool) -> float:
    accepted = report["eligible"] if process_aware else report["result"]
    return float(accepted) - .05 * report["calls"]


def split_for(entity: str) -> str:
    # 同实体的所有任务/轨迹/前缀都进同一个 split, 不按轨迹逐条随机切。
    return "test" if int(hashlib.sha256(entity.encode()).hexdigest(), 16) % 3 == 0 else "train"


def build_dataset(tasks: list[Task]) -> dict:
    sft, preferences, process = [], [], []
    for task in tasks:
        trajectories = candidates(task)
        reports = {t.name: execute(task, t) for t in trajectories}
        good = [t for t in trajectories if reports[t.name]["eligible"]]
        prompt = f"为 {task.entity} 查询当前配额, 先查证据再回答"
        common = {"task_id": task.id, "entity": task.entity, "split": split_for(task.entity), "prompt": prompt}
        for t in good:
            sft.append({**common, "completion": list(t.actions)})
        for t in trajectories:
            process.append({**common, "actions": list(t.actions), **reports[t.name]})
            if not reports[t.name]["eligible"]:
                for chosen in good:
                    preferences.append({**common, "chosen": list(chosen.actions), "rejected": list(t.actions)})
    return {"sft": sft, "preferences": preferences, "process": process}


def masked_example(prompt_ids: list[int], completion_ids: list[int]) -> dict:
    """对齐现有微调阶段的数据约定: 只监督回复, loss 内部再做 causal shift。"""
    return {"input_ids": prompt_ids + completion_ids, "labels": [-100] * len(prompt_ids) + completion_ids}


def main() -> None:
    banner("M25 - Trajectory to training data")
    tasks = [Task(f"quota-{i}-{j}", f"customer-{i}", 10 + i) for i in range(8) for j in range(2)]
    sample = tasks[0]
    rows = [(t, execute(sample, t)) for t in candidates(sample)]
    for trajectory, report in rows:
        kv(trajectory.name, f"result={report['result']}, process={report['process']}, steps={report['step_labels']}")
    result_winner = max(rows, key=lambda pair: reward(pair[1], False))[0].name
    process_winner = max(rows, key=lambda pair: reward(pair[1], True))[0].name
    assert result_winner == "lucky-guess" and process_winner == "verified"
    kv("只看结果 / 同时看过程", f"{result_winner} / {process_winner}")
    data = build_dataset(tasks)
    assert len(data["sft"]) == 16 and len(data["preferences"]) == 64 and len(data["process"]) == 80
    entities = {split: {r["entity"] for r in data["process"] if r["split"] == split} for split in ("train", "test")}
    assert entities["train"] and entities["test"] and not entities["train"] & entities["test"]
    # 直接重放导出的样本, 不信任写入 JSON 的 eligible 标签。
    lookup = {t.id: t for t in tasks}
    for pair in json.loads(json.dumps(data))["preferences"]:
        task = lookup[pair["task_id"]]
        assert execute(task, Trajectory("chosen", tuple(pair["chosen"])))["eligible"]
        assert not execute(task, Trajectory("rejected", tuple(pair["rejected"])))["eligible"]
        assert str(task.expected) not in pair["prompt"], "提示里不能泄漏该任务的标准答案"
    masked = masked_example([1, 2, 3], [4, 5])
    assert masked["labels"] == [-100, -100, -100, 4, 5]
    assert not execute(sample, Trajectory("post-final", ("lookup", "answer:observed", "lookup")))["eligible"]
    kv("数据条数", {k: len(v) for k, v in data.items()})
    kv("实体隔离", {k: len(v) for k, v in entities.items()})
    print("\n  OK: 猜中不等于可学习; 先验证轨迹再造样本; 留出集按实体切。未训练模型。")


if __name__ == "__main__":
    main()
