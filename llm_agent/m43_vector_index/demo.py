"""精确最近邻、投影树叶子和图搜索预算。结构是简化教学索引, 不是完整 ANNOY/HNSW。"""
from __future__ import annotations
import heapq
import json
import math

POINTS = tuple((math.cos(i * .4) * (1 + i / 20), math.sin(i * .4) * (1 + i / 20)) for i in range(24))
QUERIES = tuple((x + .12, y - .1) for x, y in POINTS[1::2])


def distance(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def exact(points, query):
    return min(range(len(points)), key=lambda i: distance(points[i], query)), len(points)


def build_tree(points, ids=None, depth=0):
    ids = list(range(len(points))) if ids is None else ids
    if len(ids) <= 3:
        return {"leaf": ids}
    axis = depth % 2
    ordered = sorted(ids, key=lambda i: points[i][axis])
    mid = len(ordered) // 2
    split = (points[ordered[mid-1]][axis] + points[ordered[mid]][axis]) / 2
    return {"axis": axis, "split": split, "left": build_tree(points, ordered[:mid], depth+1), "right": build_tree(points, ordered[mid:], depth+1)}


def tree_query(points, tree, query):
    node = tree
    while "leaf" not in node:
        node = node["left" if query[node["axis"]] < node["split"] else "right"]
    candidates = node["leaf"]
    return min(candidates, key=lambda i: distance(points[i], query)), len(candidates)


def build_graph(points):
    graph = {}
    for i, point in enumerate(points):
        near = sorted((j for j in range(len(points)) if i != j), key=lambda j: distance(point, points[j]))[:3]
        # 邻接链保证可达性; HNSW 的随机多层入口另在讲义解释。
        graph[i] = sorted(set(near + [j for j in (i-1, i+1) if 0 <= j < len(points)]))
    return graph


def graph_query(points, graph, query, budget=12):
    if budget < 1:
        raise ValueError("至少允许一次距离计算")
    frontier, seen = [(distance(points[0], query), 0)], {0}
    best = (frontier[0][0], 0)
    while frontier:
        _, node = heapq.heappop(frontier)
        for neighbor in graph[node]:
            if neighbor in seen or len(seen) >= budget:
                continue
            seen.add(neighbor)
            entry = distance(points[neighbor], query), neighbor
            best = min(best, entry)
            heapq.heappush(frontier, entry)
    return best[1], len(seen)


def experiment(mode="graph"):
    if mode not in {"exact", "tree", "graph"}:
        raise ValueError(mode)
    tree, graph, trace, correct, costs = build_tree(POINTS), build_graph(POINTS), [], 0, []
    for i, query in enumerate(QUERIES):
        target, _ = exact(POINTS, query)
        found, cost = exact(POINTS, query) if mode == "exact" else tree_query(POINTS, tree, query) if mode == "tree" else graph_query(POINTS, graph, query)
        correct += found == target
        costs.append(cost)
        trace.append({"title": f"查询 {i+1}", "detail": f"近邻 {found}, 精确基准 {target}, 距离计算 {cost}", "ok": found == target})
    return {"recall": round(correct / len(QUERIES), 4), "distance_evaluations": sum(costs), "queries": len(QUERIES), "trace": trace}


def main():
    rows = {m: experiment(m) for m in ("exact", "tree", "graph")}
    assert rows["exact"]["recall"] == 1
    assert rows["tree"]["distance_evaluations"] < rows["exact"]["distance_evaluations"]
    assert rows["tree"]["recall"] < 1, "只查一个叶子会漏掉边界另一侧的近邻"
    graph = build_graph(POINTS)
    assert all(graph_query(POINTS, graph, q, len(POINTS))[0] == exact(POINTS, q)[0] for q in QUERIES)
    assert any(graph_query(POINTS, graph, q, 1)[0] != exact(POINTS, q)[0] for q in QUERIES)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
