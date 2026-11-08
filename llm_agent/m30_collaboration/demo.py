"""共享状态的跨文件冲突, 以及同质投票与独立证据审查。"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import json


@dataclass(frozen=True)
class Patch:
    worker: str
    revision: int
    key: str
    value: int


class SharedPlan:
    def __init__(self):
        self.state = {"cpu": 4, "gpu": 4}
        self.revision = 0

    def merge(self, patch, policy):
        if policy not in {"file-only", "revision", "invariant"}:
            raise ValueError(policy)
        if patch.key not in self.state or type(patch.value) is not int or patch.value < 0:
            return False
        if policy == "revision" and patch.revision != self.revision:
            return False
        candidate = {**self.state, patch.key: patch.value}
        # 两个不同文件不产生文本冲突, 仍可能合起来超过资源上限。
        if policy == "invariant" and sum(candidate.values()) > 10:
            return False
        self.state = candidate
        self.revision += 1
        return True


def reviewed_answer(votes, evidence):
    proposed = Counter(votes).most_common(1)[0][0]
    return {"vote": proposed, "reviewed": evidence["current"], "corrected": proposed != evidence["current"]}


def experiment(mode="invariant"):
    if mode not in {"file-only", "revision", "invariant"}:
        raise ValueError(mode)
    store = SharedPlan()
    patches = [Patch("planner-cpu", 0, "cpu", 6), Patch("planner-gpu", 0, "gpu", 6)]
    trace, accepted = [], 0
    for patch in patches:
        ok = store.merge(patch, mode)
        accepted += ok
        trace.append({"title": patch.worker, "detail": f"申请 {patch.key}={patch.value}; 接受={ok}; 当前 {store.state}", "ok": sum(store.state.values()) <= 10})
    review = reviewed_answer([10, 10, 10], {"current": 20, "source": "policy-v2"})
    trace.append({"title": "同质投票", "detail": f"三票同为 {review['vote']}; 独立读取 v2 得到 {review['reviewed']}", "ok": False})
    trace.append({"title": "提议—审查—修改", "detail": "审查引入新证据, 不是再问同一个错误模型", "ok": review["corrected"]})
    return {"total": sum(store.state.values()), "accepted": accepted, "valid": sum(store.state.values()) <= 10,
            "vote": review["vote"], "reviewed": review["reviewed"], "trace": trace}


def main():
    rows = {m: experiment(m) for m in ("file-only", "revision", "invariant")}
    assert rows["file-only"]["total"] == 12 and not rows["file-only"]["valid"]
    assert rows["invariant"]["total"] == rows["revision"]["total"] == 10
    assert rows["invariant"]["vote"] != rows["invariant"]["reviewed"]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
