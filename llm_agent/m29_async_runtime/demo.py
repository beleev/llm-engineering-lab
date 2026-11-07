"""真实 asyncio 任务 + SQLite 幂等提交; 用 Event 屏障复现取消竞态, 不靠 sleep 猜时序。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS operations (key TEXT PRIMARY KEY, amount INTEGER NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, kind TEXT NOT NULL)")
        self.db.commit()

    def commit(self, key, amount):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO operations VALUES (?, ?)", (key, amount))
            previous = self.db.execute("SELECT amount FROM operations WHERE key=?", (key,)).fetchone()[0]
            if previous != amount:
                raise ValueError("幂等键对应的参数变了")
        return previous

    def event(self, kind):
        with self.db:
            self.db.execute("INSERT INTO events(kind) VALUES (?)", (kind,))

    def total(self):
        return self.db.execute("SELECT coalesce(sum(amount),0) FROM operations").fetchone()[0]

    def close(self):
        self.db.close()


class Runtime:
    def __init__(self, journal, budget=2):
        self.journal, self.budget = journal, budget
        self.used, self.generation = 0, 1
        self.cancelled = asyncio.Event()
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.events = []

    def record(self, title, detail, ok=True):
        self.events.append({"title": title, "detail": detail, "ok": ok})
        self.journal.event(title)

    async def execute(self, commit_before_wait=False):
        generation = self.generation
        self.record("working", "外部请求唤醒任务")
        try:
            if commit_before_wait:
                self.write(generation)
            self.ready.set()
            await self.release.wait()  # 真正挂起, 可被 Task.cancel 打断。
            if not commit_before_wait:
                self.write(generation)
            for token in ("已", "处理"):
                await asyncio.sleep(0)
                if generation != self.generation:
                    return
                self.record("stream", token)
            self.record("completed", "执行器已完成")
        except asyncio.CancelledError:
            self.cancelled.set()
            self.record("cancel-ack", "执行器确认停止; 先前提交不回滚")
            raise
        except Exception:
            self.record("failed", "执行器失败, 调用方必须读取异常", False)
            raise
        finally:
            self.ready.set()  # 在首个检查点前失败也不能让等待者永久挂起。

    def write(self, generation):
        if generation != self.generation or self.used >= self.budget:
            raise RuntimeError("过期任务或预算耗尽")
        self.used += 1  # 在无 await 的临界段预留预算。
        self.journal.commit("invoice-1", 10)
        self.record("committed", "SQLite 提交 10 元")

    async def interrupt(self, task):
        if task.done():
            return self.cancelled.is_set()  # 已终止任务的取消是幂等的, 不伪造新确认。
        self.generation += 1
        self.record("cancel-request", "用户打断, 旧 generation 作废")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert self.cancelled.is_set()
        return True


async def run_case(mode):
    if mode not in {"complete", "cancel-before", "cancel-after"}:
        raise ValueError(mode)
    with TemporaryDirectory(prefix="async-agent-") as root:
        path = Path(root) / "journal.db"
        journal = Journal(path)
        runtime = Runtime(journal)
        task = asyncio.create_task(runtime.execute(mode == "cancel-after"))
        await runtime.ready.wait()
        if mode == "complete":
            runtime.release.set()
            await task
        else:
            await runtime.interrupt(task)
        before = journal.total()
        journal.close()
        # 模拟提交后响应丢失和进程重开: 以同一键恢复, 不重复副作用。
        restored = Journal(path)
        if before:
            restored.commit("invoice-1", 10)
        after = restored.total()
        count = restored.db.execute("SELECT count(*) FROM operations").fetchone()[0]
        restored.close()
        runtime.events.append({"title": "重开账本", "detail": f"恢复前后 {before}/{after} 元, {count} 笔", "ok": before == after})
        return {"total": after, "writes": count, "cancelled": runtime.cancelled.is_set(),
                "streamed": sum(e["title"] == "stream" for e in runtime.events), "trace": runtime.events}


def experiment(mode="complete"):
    return asyncio.run(run_case(mode))


def main():
    rows = {m: experiment(m) for m in ("complete", "cancel-before", "cancel-after")}
    assert rows["complete"]["total"] == 10 and rows["complete"]["streamed"] == 2
    assert rows["cancel-before"]["total"] == 0 and rows["cancel-before"]["cancelled"]
    assert rows["cancel-after"]["total"] == 10 and rows["cancel-after"]["streamed"] == 0
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
