"""同一份带来源的事实: 平面检索、树状摘要与关系遍历。不是完整 RAPTOR/GraphRAG。"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
import json


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    relation: str
    value: str
    group: str
    version: int = 1


FACTS = (
    Fact("a", "Atlas", "team", "Search", "products"),
    Fact("b", "Search", "owner", "Lin", "people"),
    Fact("c", "Boreal", "team", "Compute", "products"),
    Fact("d", "Compute", "owner", "Mei", "people"),
    Fact("e", "Atlas", "limit", "10", "limits", 1),
    Fact("f", "Atlas", "limit", "20", "limits", 2),
)


def current_facts(facts):
    latest = {}
    for fact in facts:
        key = fact.subject, fact.relation
        previous = latest.get(key)
        if previous and fact.version == previous.version and fact.value != previous.value:
            raise ValueError("同版本冲突需审核")
        if previous is None or previous.version < fact.version:
            latest[key] = fact
    return list(latest.values())


def hierarchy(facts):
    groups = defaultdict(list)
    for fact in current_facts(facts):
        groups[fact.group].append(fact)
    # 摘取式摘要保留叶子引用, 不是 LLM 聚类或生成式摘要。
    return {key: {"summary": sorted({f.subject for f in fs}), "sources": sorted(f.id for f in fs)}
            for key, fs in sorted(groups.items())}


def graph_path(start, relations, facts=FACTS):
    frontier = [(start, [])]
    for relation in relations:
        frontier = [(f.value, sources + [f.id]) for node, sources in frontier
                    for f in current_facts(facts) if f.subject == node and f.relation == relation]
    return frontier


def contextual_chunks(facts):
    return [{"source": f.id, "raw": f.value, "context": f"{f.subject} {f.relation} v{f.version}: {f.value}"}
            for f in current_facts(facts)]


def dataset_insights():
    """短/长任务分配比例不同: 分组都更好, 总体却更差, 不能直接推因果。"""
    counts = {"A": [(81, 87), (192, 263)], "B": [(234, 270), (55, 80)]}
    result = {}
    for method, groups in counts.items():
        result[method] = {"groups": [passed / total for passed, total in groups],
                          "overall": sum(p for p, _ in groups) / sum(n for _, n in groups)}
    return result


def experiment(mode="graph"):
    if mode not in {"flat", "hierarchy", "graph"}:
        raise ValueError(mode)
    facts = current_facts(FACTS)
    if mode == "flat":
        leaves = [f for f in facts if f.subject == "Atlas"][:1]
        owners, products = [], [f.subject for f in leaves if f.relation == "team"]
        sources = [f.id for f in leaves]
    elif mode == "hierarchy":
        root = hierarchy(facts)["products"]
        owners, products, sources = [], root["summary"], root["sources"]
    else:
        paths = graph_path("Atlas", ["team", "owner"], facts)
        owners, products = [v for v, _ in paths], []
        sources = [s for _, ss in paths for s in ss]
    statistics = dataset_insights()
    checks = {"global_products": products == ["Atlas", "Boreal"], "atlas_owner": owners == ["Lin"]}
    return {"products": products, "owners": owners, "sources": sources, "checks": checks,
            "index": hierarchy(facts), "chunks": contextual_chunks(facts), "statistics": statistics,
            "trace": [{"title": "先处理版本", "detail": "Atlas limit 的 v2 替换 v1; 原始证据保留", "ok": True},
                      {"title": "全局问题", "detail": f"列出产品: {products}", "ok": checks["global_products"]},
                      {"title": "关系问题", "detail": f"Atlas → team → owner: {owners}; 引用 {sources}", "ok": checks["atlas_owner"]}]}


def main():
    rows = {m: experiment(m) for m in ("flat", "hierarchy", "graph")}
    assert rows["hierarchy"]["checks"]["global_products"]
    assert rows["graph"]["owners"] == ["Lin"] and rows["graph"]["sources"] == ["a", "b"]
    assert graph_path("Atlas", ["limit"]) == [("20", ["f"])]
    assert not graph_path("missing", ["owner"])
    assert graph_path("Atlas", ["team", "owner"], tuple(reversed(FACTS))) == [("Lin", ["a", "b"])]
    assert {f.id for f in current_facts(FACTS)} == {"a", "b", "c", "d", "f"}
    stats = dataset_insights()
    assert all(a > b for a, b in zip(stats["A"]["groups"], stats["B"]["groups"]))
    assert stats["A"]["overall"] < stats["B"]["overall"]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
