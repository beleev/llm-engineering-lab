"""M24 — 检索不是拿到一个相似块就结束: 对象、版本和证据链都要对。

沿用 core 的 TF-IDF, 多跳计划是确定性规则。真实语义检索/自由规划不在这里模拟。
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_agent.core.retrieval import TfidfIndex
from llm_agent.core.utils import banner, kv


@dataclass(frozen=True)
class Document:
    id: str
    entity: str
    version: str
    body: str
    alias: str = ""
    limit: int | None = None


DOCS = (
    Document("alias-comet", "atlas", "v2", "comet project maps to atlas v2", "comet"),
    Document("atlas-v1", "atlas", "v1", "upload limit is 10 MB", limit=10),
    Document("atlas-v2", "atlas", "v2", "upload limit is 20 MB", limit=20),
    Document("orion-v2", "orion", "v2", "upload limit is 90 MB", limit=90),
)


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    entity: str = ""
    version: str = ""
    alias: str = ""


QUESTIONS = (
    Question("direct", "atlas v2 upload limit", "atlas", "v2"),
    Question("two-hop", "comet upload limit", alias="comet"),
    Question("missing", "nova upload limit", alias="nova"),
)


def retrieve(query: str, contextual: bool, docs: tuple[Document, ...] = DOCS) -> list[Document]:
    # TfidfIndex 连标题一起索引; 用不含对象/版本的键, 防止 plain 基线偷看到元数据。
    lookup = {f"chunk{i}": d for i, d in enumerate(docs)}
    text = {key: (f"{d.entity} {d.version} " if contextual else "") + d.body for key, d in lookup.items()}
    return [lookup[cid] for score, cid in TfidfIndex(text).search(query, k=len(docs)) if score > 0]


def answer(question: Question, mode: str = "agentic", budget: int = 2,
           docs: tuple[Document, ...] = DOCS) -> dict:
    if mode not in {"plain", "contextual", "agentic"} or budget < 0:
        raise ValueError("未知检索策略或预算")
    entity, version = question.entity, question.version
    sources, queries, value = [], [], None
    if budget:
        queries.append(question.query)
        hits = retrieve(question.query, mode != "plain", docs)
        if question.alias:
            alias = next((d for d in hits if d.alias == question.alias), None)
            if alias:
                entity, version = alias.entity, alias.version
                sources.append(alias.id)
            if mode == "agentic" and alias and len(queries) < budget:
                query = f"{entity} {version} upload limit"
                queries.append(query)
                hits = retrieve(query, True, docs)
            else:
                hits = []  # 当前一跳只用来确定对象, 不从别的产品块猜数字
        # context-free 基线只取 top-1; 有元数据后筛选对象/版本, 不把相似度当真实性。
        candidates = hits[:1] if mode == "plain" else hits
        doc = next((d for d in candidates if d.limit is not None and d.entity == entity and d.version == version), None)
        if doc:
            value = doc.limit
            sources.append(doc.id)
    return {"answer": value, "sources": sources, "queries": queries, "status": "answered" if value is not None else "insufficient"}


def supported(question: Question, row: dict, docs: tuple[Document, ...] = DOCS) -> bool:
    """独立判分: 不靠回答是否含某个数字, 要检查所引证据本身。"""
    lookup = {d.id: d for d in docs}
    if len(set(row["sources"])) != len(row["sources"]) or any(s not in lookup for s in row["sources"]):
        return False
    cited = [lookup[s] for s in row["sources"]]
    entity, version = question.entity, question.version
    if question.alias:
        link = next((d for d in cited if d.alias == question.alias), None)
        if not link:
            return False
        entity, version = link.entity, link.version
    return row["answer"] is not None and any(d.entity == entity and d.version == version and d.limit == row["answer"] for d in cited)


def main() -> None:
    banner("M24 - Evidence chains")
    for mode in ("plain", "contextual", "agentic"):
        for question in QUESTIONS:
            row = answer(question, mode)
            kv(f"{mode}/{question.id}", f"{row['answer']}; sources={row['sources']}; queries={len(row['queries'])}")
            if row["answer"] is not None:
                assert supported(question, row), "输出数字必须有匹配对象与版本的证据"
    direct = answer(QUESTIONS[0])
    assert direct["answer"] == 20 and len(direct["queries"]) == 1
    assert answer(QUESTIONS[1], "contextual")["answer"] is None
    two = answer(QUESTIONS[1])
    assert two["answer"] == 20 and set(two["sources"]) == {"alias-comet", "atlas-v2"}
    assert answer(QUESTIONS[1], budget=1)["answer"] is None
    assert answer(QUESTIONS[2])["status"] == "insufficient"
    assert answer(QUESTIONS[0], budget=0)["queries"] == []
    for bad in ({**two, "answer": 10}, {**two, "sources": ["atlas-v2"]}, {**two, "sources": ["alias-comet", "orion-v2"]}):
        assert not supported(QUESTIONS[1], bad), "错版本、断链、错对象都不能冒充有证据"
    assert answer(QUESTIONS[1], docs=tuple(d for d in DOCS if d.id != "atlas-v2"))["answer"] is None
    print("\n  OK: 简单题一跳即可; 别名题要两跳; 缺证据就保留未知, 不用旧版本凑答案。")


if __name__ == "__main__":
    main()
