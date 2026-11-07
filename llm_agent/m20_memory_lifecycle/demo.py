"""M20 — 记忆更新: 原始证据不覆盖, 生效知识可修订, 索引可重建。

比较追加、全局覆盖、按对象/作用域审核的更新。全用标准库, 不靠模型判断事实真假。
审核只证明候选与登记证据一致; 证据是否可信由上游负责, 这里不是事实核查器。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_agent.core.utils import banner, kv


@dataclass(frozen=True)
class Evidence:
    id: str
    entity: str
    scope: str
    value: str
    time: int


@dataclass(frozen=True)
class Proposal:
    source_id: str
    entity: str
    scope: str
    value: str
    based_on: int  # 0 = 这个作用域还没有生效版本


class VersionedMemory:
    """只追加证据/版本/发布事件。index 是派生物, 可丢弃后重建。"""

    def __init__(self) -> None:
        self.evidence: dict[str, Evidence] = {}
        self.versions: list[Proposal] = []
        self.events: list[tuple[tuple[str, str], int]] = []
        self.index: dict[tuple[str, str], int] = {}

    def register(self, evidence: Evidence) -> None:
        if evidence.id in self.evidence:
            if self.evidence[evidence.id] != evidence:
                raise ValueError("同一证据 ID 不能改写")
            return
        self.evidence[evidence.id] = evidence

    def propose(self, source_id: str) -> Proposal:
        e = self.evidence[source_id]
        return Proposal(e.id, e.entity, e.scope, e.value, self.index.get((e.entity, e.scope), 0))

    def publish(self, proposal: Proposal) -> int:
        e = self.evidence.get(proposal.source_id)
        if e is None or (proposal.entity, proposal.scope, proposal.value) != (e.entity, e.scope, e.value):
            raise ValueError("候选缺少对应证据, 或改丢了对象/范围/原值")
        key = (e.entity, e.scope)
        current = self.index.get(key, 0)
        if proposal.based_on != current:
            raise ValueError("旧版本上的提案不能覆盖后来发布的知识")
        if current:
            old = self.evidence[self.versions[current - 1].source_id]
            if e.time <= old.time:
                raise ValueError("过时或同时间冲突的证据需另行处理")
        self.versions.append(proposal)
        version = len(self.versions)
        self.events.append((key, version))
        self.index[key] = version
        return version

    def rollback(self, entity: str, scope: str, version: int) -> None:
        """显式回滚, 不删除证据。version=0 撤销该范围, 检索会回退到全局。"""
        if version < 0 or version > len(self.versions):
            raise ValueError("未知版本")
        if version and (self.versions[version - 1].entity, self.versions[version - 1].scope) != (entity, scope):
            raise ValueError("不能回滚到别的对象/范围")
        self.events.append(((entity, scope), version))
        self.rebuild()

    def rebuild(self) -> None:
        self.index.clear()
        for key, version in self.events:
            if version:
                self.index[key] = version
            else:
                self.index.pop(key, None)

    def recall(self, entity: str, scope: str) -> Proposal | None:
        version = self.index.get((entity, scope), self.index.get((entity, "*"), 0))
        return self.versions[version - 1] if version else None

    def save(self, path: Path) -> None:
        # 教学快照, 非并发数据库; 不承担崩溃时的原子提交。
        path.write_text(json.dumps({"evidence": [asdict(e) for e in self.evidence.values()],
                                    "versions": [asdict(p) for p in self.versions],
                                    "events": self.events}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> VersionedMemory:
        data = json.loads(path.read_text(encoding="utf-8"))
        memory = cls()
        for e in data["evidence"]:
            memory.register(Evidence(**e))
        memory.versions = [Proposal(**p) for p in data["versions"]]
        memory.events = [(tuple(key), version) for key, version in data["events"]]
        memory.rebuild()
        return memory


def comparison() -> dict[str, list[str]]:
    """同一批事实, 查询 A/B/另一用户。追加基线取首条, 覆盖基线取末条。"""
    facts = [Evidence("global", "alice", "*", "Python", 1),
             Evidence("project", "alice", "A", "TypeScript", 2)]
    memory = VersionedMemory()
    for e in facts:
        memory.register(e)
        memory.publish(memory.propose(e.id))
    queries = [("alice", "A"), ("alice", "B"), ("bob", "A")]
    return {
        "append-first": [next((e.value for e in facts if e.entity == who), "unknown") for who, _ in queries],
        "overwrite": [next((e.value for e in reversed(facts) if e.entity == who), "unknown") for who, _ in queries],
        "versioned": [p.value if (p := memory.recall(who, scope)) else "unknown" for who, scope in queries],
    }


def main() -> None:
    banner("M20 - Memory lifecycle")
    expected = ["TypeScript", "Python", "unknown"]
    rows = comparison()
    for mode, answers in rows.items():
        kv(mode, f"A / B / bob = {answers}, 正确 {sum(a == b for a, b in zip(answers, expected))}/3")
    assert rows["versioned"] == expected
    assert rows["append-first"][0] != expected[0] and rows["overwrite"][1] != expected[1]

    memory = VersionedMemory()
    for e in [Evidence("g", "alice", "*", "Python", 1), Evidence("a", "alice", "A", "TypeScript", 2)]:
        memory.register(e)
        memory.publish(memory.propose(e.id))
    memory.register(Evidence("new", "alice", "A", "Rust", 3))
    proposal = memory.propose("new")
    for invalid in [replace(proposal, source_id="invented"), replace(proposal, scope="*"), replace(proposal, value="Go")]:
        try:
            memory.publish(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("无来源或丢限定的候选不能发布")
    assert memory.recall("alice", "A").value == "TypeScript"
    memory.publish(proposal)
    try:
        memory.publish(proposal)
    except ValueError:
        pass
    else:
        raise AssertionError("过期提案必须拒绝")
    memory.rollback("alice", "A", 2)
    assert memory.recall("alice", "A").value == "TypeScript"
    memory.rollback("alice", "A", 0)
    assert memory.recall("alice", "A").value == "Python"
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        memory.save(path)
        loaded = VersionedMemory.load(path)
        assert loaded.index == memory.index and len(loaded.evidence) == 3
        assert loaded.recall("bob", "A") is None
    kv("审核 / 回滚 / 重建", "拒绝 3 个无依据候选; 撤销项目例外后回到 Python; 原始证据仍在")
    print("\n  OK: 存下来、查出来、用对范围是三件事; 版本化不会自动证明证据为真。")


if __name__ == "__main__":
    main()
