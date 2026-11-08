"""M22 — 重试不等于重做, 取消请求不等于动作已撤销。

内存账本 + 逻辑时钟, 无网络、无 sleep。演示服务端幂等与协作取消的状态语义,
不宣称能强杀任意线程, 也不提供分布式 exactly-once 保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_agent.core.utils import banner, kv


class Ledger:
    """提交成功后故意丢掉第一次响应; 幂等记录在副作用所在的服务端。"""

    def __init__(self, lose_response: bool = True) -> None:
        self.rows: list[tuple[str, int]] = []
        self.receipts: dict[str, tuple[str, int]] = {}
        self.lose_response = lose_response

    def submit(self, operation: str, amount: int, key: str | None, allowed: bool = True) -> int:
        if not allowed:
            raise PermissionError("denied")
        if key in self.receipts:
            if self.receipts[key] != (operation, amount):
                raise ValueError("同一幂等键不能换参数")
            return next(i + 1 for i, row in enumerate(self.rows) if row[0] == key)
        row_id = key or f"request-{len(self.rows) + 1}"
        self.rows.append((row_id, amount))
        if key:
            self.receipts[key] = (operation, amount)
        if self.lose_response:
            self.lose_response = False
            raise TimeoutError("committed, response lost")
        return len(self.rows)


def retry_write(mode: str, attempts: int = 2, allowed: bool = True) -> dict:
    if mode not in {"blind", "idempotent", "query"} or attempts < 1:
        raise ValueError("未知策略或无效预算")
    ledger, requests, queries = Ledger(), 0, 0
    key = None if mode == "blind" else "task-1/payment-1"
    status = "unknown"
    for _ in range(attempts):
        requests += 1
        try:
            ledger.submit("payment", 10, key, allowed)
            status = "confirmed"
            break
        except PermissionError:
            status = "denied"  # 永久拒绝, 不重试
            break
        except TimeoutError:
            if mode == "query":
                queries += 1
                if key in ledger.receipts:
                    status = "confirmed"
                    break
    return {"writes": len(ledger.rows), "amount": sum(n for _, n in ledger.rows),
            "requests": requests, "queries": queries, "status": status}


@dataclass
class Job:
    id: str
    ready_at: int
    valid: bool = True
    state: str = "running"
    effects: int = 0


@dataclass
class Runtime:
    """单线程逻辑事件模拟: 取消确认前可能仍有副作用, 确认后禁止新动作。"""
    jobs: dict[str, Job]
    generation: int = 1
    accepted: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    work: int = 0

    def cancel(self, job: Job) -> None:
        if job.state == "running":
            job.state = "cancel_requested"
            self.log.append(f"{job.id}:cancel_requested")

    def acknowledge(self, job: Job) -> None:
        if job.state == "cancel_requested":
            job.state = "canceled"
            self.log.append(f"{job.id}:cancel_ack")

    def effect(self, job: Job) -> None:
        if job.state in {"running", "cancel_requested"}:
            job.effects += 1
            self.log.append(f"{job.id}:effect")

    def receive(self, job_id: str, generation: int) -> None:
        job = self.jobs[job_id]
        if generation == self.generation and job.state == "completed" and job.valid and job_id not in self.accepted:
            self.accepted.append(job_id)
        else:
            self.log.append(f"{job_id}:ignored_result")


def schedule(mode: str, budget: int = 30) -> dict:
    if mode not in {"all", "first-valid"} or budget < 0:
        raise ValueError("未知策略或无效预算")
    runtime = Runtime({j.id: j for j in [Job("fast-wrong", 1, False), Job("correct", 3), Job("slow", 6)]})
    elapsed = 0
    for tick in range(1, 7):
        for job in runtime.jobs.values():
            if job.state != "running":
                continue
            if runtime.work >= budget:
                runtime.cancel(job)
                runtime.acknowledge(job)
                continue
            runtime.work += 1
            elapsed = tick
            if tick >= job.ready_at:
                job.state = "completed"
                runtime.receive(job.id, runtime.generation)
        if mode == "first-valid" and runtime.accepted:
            for job in runtime.jobs.values():
                runtime.cancel(job)
                runtime.acknowledge(job)
            break
    return {"accepted": runtime.accepted, "work": runtime.work, "elapsed": elapsed,
            "states": {k: j.state for k, j in runtime.jobs.items()}}


def main() -> None:
    banner("M22 - Retry / cancellation")
    for mode in ("blind", "idempotent", "query"):
        row = retry_write(mode)
        kv(mode, row)
        assert row["writes"] == (2 if mode == "blind" else 1)
    assert retry_write("idempotent", allowed=False) == {"writes": 0, "amount": 0, "requests": 1, "queries": 0, "status": "denied"}
    assert retry_write("blind", attempts=1)["status"] == "unknown", "响应丢失不能当成失败未执行"
    ledger = Ledger(False)
    ledger.submit("payment", 10, "key")
    try:
        ledger.submit("payment", 20, "key")
    except ValueError:
        pass
    else:
        raise AssertionError("幂等键参数冲突必须拒绝")
    assert len(ledger.rows) == 1

    all_jobs, first = schedule("all"), schedule("first-valid")
    kv("全部等待", all_jobs)
    kv("首个验收通过", first)
    assert first["accepted"] == ["correct"] and first["states"]["slow"] == "canceled"
    assert first["work"] < all_jobs["work"] and first["elapsed"] < all_jobs["elapsed"]
    assert schedule("first-valid", 2)["work"] == 2 and not schedule("first-valid", 2)["accepted"]
    assert all(s != "running" for s in schedule("all", 0)["states"].values())

    job = Job("write", 5)
    runtime = Runtime({job.id: job})
    runtime.cancel(job)
    runtime.effect(job)  # 请求取消但执行器还没确认: 副作用已经发生
    runtime.acknowledge(job)
    runtime.effect(job)
    assert job.effects == 1 and job.state == "canceled"
    job.state = "completed"
    runtime.generation += 1
    runtime.receive(job.id, 1)
    assert not runtime.accepted, "旧任务迟到的结果不能污染新任务"
    kv("取消竞态", runtime.log)
    print("\n  OK: 重试有幂等键和预算; 首个正确结果才触发取消; 取消不撤销已发生的副作用。")


if __name__ == "__main__":
    main()
