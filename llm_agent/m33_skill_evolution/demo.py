"""从已验证失败生成局部 Skill 提案, 回归后发布, 再做离线合并与回滚。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class Case:
    project: str
    expected: str
    evidence: str


LEARNING = (Case("A", "Rust", "review-1"),)
GATES = (Case("A", "Rust", "heldout-a"), Case("B", "Python", "retention-b"), Case("untrusted", "Python", "safety-c"))


def act(policy, project):
    return policy["projects"].get(project, policy["default"])


def propose(policy, failures, globalize=False):
    candidate = copy.deepcopy(policy)
    for case in failures:
        if not case.evidence.startswith("review-"):
            raise ValueError("更新需要已验收的来源")
        if globalize:
            candidate["default"] = case.expected
        else:
            candidate["projects"][case.project] = case.expected
    return candidate


class SkillStore:
    def __init__(self, root):
        self.root = Path(root)
        self.versions = [{"default": "Python", "projects": {}}]
        self.active = 0
        self.write()

    def write(self):
        policy = self.versions[self.active]
        # 目录、正文、引用资料三层; 内容跟当前版本一致。
        files = {"index.md": "project-language: 按项目选择语言 → SKILL.md\n",
                 "SKILL.md": "先检查项目例外, 否则用默认语言。\n详细规则见 rules.json。\n",
                 "rules.json": json.dumps(policy, sort_keys=True)}
        for name, content in files.items():
            staging = self.root / (name + ".tmp")
            staging.write_text(content)
            staging.replace(self.root / name)

    def publish(self, candidate):
        verdicts = [act(candidate, c.project) == c.expected for c in GATES]
        if not all(verdicts):
            return False, verdicts
        self.versions.append(copy.deepcopy(candidate))
        self.active = len(self.versions) - 1
        self.write()
        return True, verdicts

    def rollback(self, version):
        if not 0 <= version < len(self.versions):
            raise ValueError(version)
        self.active = version
        self.write()


def curate(records, now):
    # 只合并相同的作用域+事实; 冲突不能按最近访问次数偷偷裁决。
    kept, archived = {}, []
    for row in records:
        if row["expires"] <= now:
            archived.append(row["id"])
            continue
        key = row["scope"], row["value"]
        kept.setdefault(key, set()).add(row["source"])
    return [{"scope": k[0], "value": k[1], "sources": sorted(v)} for k, v in sorted(kept.items())], archived


def experiment(mode="scoped"):
    if mode not in {"global", "scoped", "rollback"}:
        raise ValueError(mode)
    with TemporaryDirectory(prefix="skills-") as root:
        store = SkillStore(root)
        failures = [c for c in LEARNING if act(store.versions[0], c.project) != c.expected]
        candidate = propose(store.versions[0], failures, globalize=mode == "global")
        accepted, checks = store.publish(candidate)
        if mode == "rollback":
            store.rollback(0)
        records = [{"id": "a", "scope": "A", "value": "Rust", "source": "review-1", "expires": 10},
                   {"id": "b", "scope": "A", "value": "Rust", "source": "review-2", "expires": 10},
                   {"id": "c", "scope": "A", "value": "Go", "source": "old", "expires": 1}]
        compacted, archived = curate(records, 5)
        policy = json.loads((Path(root) / "rules.json").read_text())
        return {"accepted": accepted, "checks": checks, "active": store.active, "answer_a": act(policy, "A"),
                "answer_b": act(policy, "B"), "kept": len(compacted), "archived": len(archived),
                "trace": [{"title": "从失败提案", "detail": json.dumps(candidate), "ok": True},
                          {"title": "迁移 / 保持 / 安全门", "detail": str(checks), "ok": all(checks)},
                          {"title": "发布与回滚", "detail": f"active={store.active}, A={act(policy, 'A')}, B={act(policy, 'B')}", "ok": act(policy, "B") == "Python"},
                          {"title": "离线整理", "detail": f"合并后 {len(compacted)} 条, 归档 {archived}; 保留两条来源", "ok": True}]}


def main():
    rows = {m: experiment(m) for m in ("global", "scoped", "rollback")}
    assert not rows["global"]["accepted"] and rows["global"]["active"] == 0
    assert rows["scoped"]["accepted"] and rows["scoped"]["answer_a"] == "Rust"
    assert rows["rollback"]["answer_a"] == "Python" and rows["rollback"]["active"] == 0
    assert all(r["answer_b"] == "Python" for r in rows.values())
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
