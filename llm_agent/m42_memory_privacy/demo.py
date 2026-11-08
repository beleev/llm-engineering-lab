"""多会话记忆的租户、同意、过期、删除与主动提醒边界。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class Memory:
    id: str
    tenant: str
    user: str
    topic: str
    content: str
    expires: int
    consent: bool = True
    kind: str = "text"


class MemoryService:
    def __init__(self):
        self.raw, self.index, self.cache, self.audit = {}, {}, {}, []

    def remember(self, memory):
        if not memory.consent:
            raise ValueError("无同意的内容不进入长期记忆")
        if memory.id in self.raw:
            raise ValueError("证据 id 不可覆盖")
        self.raw[memory.id] = memory
        self.index.setdefault((memory.tenant, memory.user, memory.topic), []).append(memory.id)

    def recall(self, tenant, user, topic, now):
        key = tenant, user, topic
        # 授权在召回之前, 不能先全库召回再把私密摘要交给过滤器。
        records = [self.raw[i] for i in self.index.get(key, []) if i in self.raw and self.raw[i].expires > now]
        self.cache[key] = [r.id for r in records]
        self.audit.append({"operation": "recall", "scope_hash": hashlib.sha256(repr(key).encode()).hexdigest()[:12], "count": len(records)})
        return records

    def forget(self, tenant, user, memory_id):
        memory = self.raw.get(memory_id)
        if memory is None:
            return False
        if (memory.tenant, memory.user) != (tenant, user):
            raise PermissionError("不能删除别人的记忆")
        del self.raw[memory_id]
        for store in (self.index, self.cache):
            for key in list(store):
                store[key] = [i for i in store[key] if i != memory_id]
                if not store[key]:
                    del store[key]
        self.audit.append({"operation": "forget", "id_hash": hashlib.sha256(memory_id.encode()).hexdigest()[:12]})
        return True

    def notify(self, tenant, user, topic, now, opted_in):
        return self.recall(tenant, user, topic, now) if opted_in else []


def experiment(mode="scoped"):
    if mode not in {"flat", "scoped", "forget"}:
        raise ValueError(mode)
    memory = MemoryService()
    for row in (Memory("a", "org-a", "Lin", "preference", "Python", 10),
                Memory("b", "org-b", "Lin", "preference", "Rust", 10),
                Memory("c", "org-a", "Lin", "preference", "Go", 1)):
        memory.remember(row)
    first = memory.recall("org-a", "Lin", "preference", 5)
    if mode == "flat":
        recalled = list(memory.raw.values())  # 反例: 没有租户或过期过滤。
    else:
        if mode == "forget":
            memory.forget("org-a", "Lin", "a")
        recalled = memory.recall("org-a", "Lin", "preference", 5)
    leaks = sum(r.tenant != "org-a" for r in recalled)
    expired = sum(r.expires <= 5 for r in recalled)
    proactive = memory.notify("org-a", "Lin", "preference", 5, opted_in=False)
    return {"values": [r.content for r in recalled], "leaks": leaks, "expired": expired, "proactive": len(proactive),
            "trace": [{"title": "首次会话", "detail": f"按作用域召回 {[r.content for r in first]}", "ok": len(first) == 1},
                      {"title": "新会话 / 删除后", "detail": f"召回 {[r.content for r in recalled]}; 越界={leaks}, 过期={expired}", "ok": leaks == expired == 0},
                      {"title": "主动服务", "detail": "未开启主动提醒, 不发送个人记忆", "ok": not proactive}]}


def main():
    rows = {m: experiment(m) for m in ("flat", "scoped", "forget")}
    assert rows["flat"]["leaks"] == rows["flat"]["expired"] == 1
    assert rows["scoped"]["values"] == ["Python"]
    assert rows["forget"]["values"] == []
    assert all(r["proactive"] == 0 for r in rows.values())
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
