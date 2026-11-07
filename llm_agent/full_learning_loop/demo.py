"""持续改进综合实验: 先行动和验收, 再读反馈; learning/transfer/change/retention 分开报。

复用 m20 的证据审核与版本记忆。学习的是外部规则存储, 不是模型权重。
策略知道当前任务的对象和范围, 看不到 expected 或未来反馈。
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_agent.core.utils import banner, kv
from llm_agent.m20_memory_lifecycle.demo import Evidence, VersionedMemory

PHASES = ("learning", "transfer", "change", "retention")


@dataclass(frozen=True)
class Task:
    id: str
    phase: str
    entity: str
    scope: str
    expected: str
    feedback: Evidence | None = None


TASKS = (
    Task("L1", "learning", "alice", "A", "TypeScript", Evidence("e1", "alice", "A", "TypeScript", 1)),
    Task("L2", "learning", "alice", "B", "Python", Evidence("e2", "alice", "*", "Python", 2)),
    Task("T1", "transfer", "alice", "A", "TypeScript"),
    Task("T2", "transfer", "alice", "B", "Python"),
    Task("T3", "transfer", "bob", "A", "unknown"),
    Task("C1", "change", "alice", "A", "Rust", Evidence("e3", "alice", "A", "Rust", 3)),
    Task("C2", "change", "alice", "A", "Rust"),
    Task("R1", "retention", "alice", "B", "Python"),
    Task("R2", "retention", "alice", "A", "Rust"),
)


class Learner:
    def __init__(self, mode: str) -> None:
        if mode not in {"static", "append", "versioned"}:
            raise ValueError(mode)
        self.mode = mode
        self.memory = VersionedMemory()
        self.notes: list[Evidence] = []

    def act(self, entity: str, scope: str) -> tuple[str, bool]:
        if self.mode == "versioned":
            fact = self.memory.recall(entity, scope)
            if fact:
                return fact.value, True
        if self.mode == "append":
            # 简单追加检索只看对象, 第一条命中长期占据位置。
            fact = next((e for e in self.notes if e.entity == entity), None)
            if fact:
                return fact.value, True
        return ("Python" if entity == "alice" else "unknown"), False

    def observe(self, evidence: Evidence) -> bool:
        if self.mode == "static":
            return False
        if self.mode == "append":
            self.notes.append(evidence)
        else:
            self.memory.register(evidence)
            self.memory.publish(self.memory.propose(evidence.id))
        return True


def evaluate(mode: str, tasks: tuple[Task, ...] = TASKS) -> dict:
    learner, rows, order = Learner(mode), [], []
    for task in tasks:
        if task.phase not in PHASES or (task.feedback and task.phase not in {"learning", "change"}):
            raise ValueError("只允许学习期和显式规则变化事件更新")
        action, activated = learner.act(task.entity, task.scope)
        order.append(f"{task.id}:act")
        passed = action == task.expected
        order.append(f"{task.id}:grade")
        # ★ 验收之后才把反馈交给 learner, 不允许这一题事后改答案。
        updated = learner.observe(task.feedback) if task.feedback else False
        if updated:
            order.append(f"{task.id}:update")
        rows.append({"id": task.id, "phase": task.phase, "action": action, "expected": task.expected,
                     "passed": passed, "activated": activated, "updated": updated})
    phases = {phase: [r for r in rows if r["phase"] == phase] for phase in PHASES}
    scores = {phase: sum(r["passed"] for r in rs) / len(rs) if rs else None for phase, rs in phases.items()}
    negative = [r for r in rows if r["phase"] != "learning" and r["activated"] and not r["passed"]]
    return {"scores": scores, "rows": rows, "order": order, "negative_transfer": len(negative),
            "updates": sum(r["updated"] for r in rows), "activated": sum(r["activated"] for r in rows)}


def main() -> None:
    banner("Full learning loop - evaluate before updating")
    reports = {mode: evaluate(mode) for mode in ("static", "append", "versioned")}
    for mode, report in reports.items():
        kv(mode, f"{report['scores']}; 更新={report['updates']}; 负迁移任务={report['negative_transfer']}")
    good = reports["versioned"]
    assert good["scores"] == {"learning": .5, "transfer": 1., "change": .5, "retention": 1.}
    assert not good["rows"][5]["passed"] and good["rows"][6]["passed"], "变化信号之后下一题才恢复, 不能提前看答案"
    assert reports["append"]["scores"]["retention"] == 0.
    assert reports["static"]["scores"]["retention"] == .5
    assert good["rows"][7]["action"] == "Python", "新规则不能破坏未变任务"
    for row in good["rows"]:
        if row["updated"]:
            ids = [good["order"].index(f"{row['id']}:{event}") for event in ("act", "grade", "update")]
            assert ids == sorted(ids)
        if row["phase"] in {"transfer", "retention"}:
            assert not row["updated"]
    # 截断未来任务不改变过去结果, 与重排任务进行对照可以查出时间泄漏。
    assert evaluate("versioned", TASKS[:5])["rows"] == good["rows"][:5]
    print("\n  OK: 学到规则、用到规则、替换旧规则、保住未变规则要分别验收。")


if __name__ == "__main__":
    main()
