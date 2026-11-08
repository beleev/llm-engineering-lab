"""发现树回放: 搜索策略只能看已展开节点, 离线选策略后在新树上评价。"""
from __future__ import annotations
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Node:
    id: str
    prior: float
    score: float
    children: tuple[str, ...] = ()


TREES = {
    "train": {"root": Node("root", 0, 0, ("a", "b")), "a": Node("a", .9, .2, ("c",)),
              "b": Node("b", .2, .8), "c": Node("c", .8, .9)},
    "test": {"root": Node("root", 0, 0, ("a", "b")), "a": Node("a", .9, .1, ("c",)),
             "b": Node("b", .2, .95), "c": Node("c", .8, .2)},
}


class ReplayWorld:
    def __init__(self, tree):
        self._tree = tree
        self.frontier = {"root": 0.}
        self.visited = []
        self.best = 0.

    def options(self):
        return dict(self.frontier)  # 未展开节点仅有先验, 不泄漏 score 和后继。

    def expand(self, node_id):
        if node_id not in self.frontier:
            raise ValueError("节点尚未被发现或已展开")
        del self.frontier[node_id]
        node = self._tree[node_id]
        self.best = max(self.best, node.score)
        self.visited.append(node_id)
        for child in node.children:
            if child not in self.visited:
                self.frontier[child] = self._tree[child].prior
        return node.score


def search(tree, strategy, budget=3):
    world, trace = ReplayWorld(tree), []
    for _ in range(budget):
        options = world.options()
        if not options:
            break
        if strategy == "breadth":
            chosen = next(iter(options))
        elif strategy == "prior":
            chosen = max(options, key=options.get)
        else:
            raise ValueError(strategy)
        score = world.expand(chosen)
        trace.append({"title": f"展开 {chosen}", "detail": f"揭示 score={score}; 当前最优={world.best}", "ok": score >= .8})
    return {"best": world.best, "visited": world.visited, "cost": len(world.visited), "trace": trace}


def experiment(mode="selected"):
    if mode not in {"breadth", "prior", "selected"}:
        raise ValueError(mode)
    train = {p: search(TREES["train"], p)["best"] for p in ("breadth", "prior")}
    strategy = max(train, key=train.get) if mode == "selected" else mode
    row = search(TREES["test"], strategy)
    row.update(strategy=strategy, train=train)
    row["trace"].insert(0, {"title": "离线选策略", "detail": f"训练回放 {train}; 采用 {strategy}", "ok": True})
    return row


def main():
    rows = {m: experiment(m) for m in ("breadth", "prior", "selected")}
    assert rows["selected"]["strategy"] == "prior"
    assert rows["selected"]["train"]["prior"] > rows["selected"]["train"]["breadth"]
    assert rows["selected"]["best"] < rows["breadth"]["best"], "历史回放提升不保证新任务提升"
    world = ReplayWorld(TREES["test"])
    try:
        world.expand("b")
    except ValueError:
        pass
    else:
        raise AssertionError("未发现节点被提前访问")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
