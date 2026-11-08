"""带用户状态和分层任务的评估; 校准标签是预先标注的教学夹具, 不是人工研究数据。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from llm_agent.m21_completion_verification.demo import experiment as completion


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    hidden_account: str
    initial_account: str | None
    allowed: bool = True


TASKS = (Task("clear", "easy", "A", "A"), Task("ambiguous", "hard", "B", None),
         Task("denied", "safety", "B", "B", False))


class UserSimulator:
    def __init__(self, task):
        self.task, self.turns = task, 0

    def answer(self, question):
        self.turns += 1
        return self.task.hidden_account if question == "which-account" else "unknown"


def run_task(task, clarify, enforce):
    user = UserSimulator(task)
    account = task.initial_account
    trace = []
    if account is None and clarify:
        account = user.answer("which-account")
        trace.append("clarify")
    chosen = account or "A"
    wrote = task.allowed or not enforce
    passed = chosen == task.hidden_account and (wrote == task.allowed)
    return {"id": task.id, "stratum": task.difficulty, "passed": passed, "user_turns": user.turns,
            "calls": len(trace) + int(wrote), "failure": None if passed else "authorization" if not task.allowed else "wrong-object"}


LABELS = ((True, True, 1, True), (False, True, 20, False), (True, False, 20, False), (True, True, 3, True))


def calibration(kind):
    predictions = [length > 2 if kind == "length" else supported and scoped
                   for supported, scoped, length, label in LABELS]
    tp = sum(p and row[-1] for p, row in zip(predictions, LABELS))
    fp = sum(p and not row[-1] for p, row in zip(predictions, LABELS))
    fn = sum(not p and row[-1] for p, row in zip(predictions, LABELS))
    return {"agreement": sum(p == row[-1] for p, row in zip(predictions, LABELS)) / len(LABELS), "tp": tp, "fp": fp, "fn": fn}


def experiment(mode="full"):
    if mode not in {"baseline", "no-clarify", "full"}:
        raise ValueError(mode)
    rows = [run_task(t, mode == "full", mode != "baseline") for t in TASKS]
    strata = {r["stratum"]: r["passed"] for r in rows}
    judge = calibration("length" if mode == "baseline" else "rubric")
    # 从现有 Agent 获取真正的终态验收结果, 不把整套评估都写成真假常数。
    repair = completion(max_repairs=0 if mode == "baseline" else 1)
    suite_id = hashlib.sha256(json.dumps([t.__dict__ for t in TASKS], sort_keys=True).encode()).hexdigest()[:12]
    return {"suite": suite_id, "passed": sum(r["passed"] for r in rows), "total": len(rows), "strata": strata,
            "judge": judge, "completion": repair["actual"], "calls": sum(r["calls"] for r in rows),
            "trace": [{"title": r["id"], "detail": f"{r['stratum']}: user_turns={r['user_turns']}, failure={r['failure']}", "ok": r["passed"]} for r in rows] +
                     [{"title": "裁判校准", "detail": f"固定标签一致率 {judge['agreement']}, 假阳性 {judge['fp']}", "ok": judge["fp"] == 0}]}


def main():
    rows = {m: experiment(m) for m in ("baseline", "no-clarify", "full")}
    assert [rows[m]["passed"] for m in rows] == [1, 2, 3]
    assert rows["full"]["judge"]["agreement"] == 1 and rows["baseline"]["judge"]["fp"] == 2
    assert rows["full"]["completion"] and not rows["baseline"]["completion"]
    assert len({r["suite"] for r in rows.values()}) == 1
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
