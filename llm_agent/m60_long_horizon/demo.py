"""M60 — 跨上下文窗口接力: 新会话从干净上下文开始, 靠磁盘上的进度文件交接。

任务比一个上下文窗口大时有两条路: 一个长会话靠压缩硬撑, 或者把状态写到磁盘上,
让每个新会话从干净上下文开始, 读进度文件和提交记录, 做完一个功能、端到端测过、提交、更新进度。
这里用一个确定性模拟比较四种做法, 在同一条总上下文预算下看真正通过的功能数。

机制 (全部确定性, 纯标准库, 不用随机数):
  - 项目有 N 个功能。功能 i 改动共享模块 MOD(i) = i % 4, 实现它需要知道同模块已通过功能的约束;
    约束的来源是本会话自己做过 (在上下文里) 或清单里记过 (在磁盘上)。两处都没有, 同模块的已通过功能回归。
  - FLAKY(i) = (i % 3 == 2) 的功能第一次写出来是半成品: 只有端到端跑一遍测试才会发现并补完;
    不测就记成完成, 留下坏状态, 后面同模块的功能建在它上面也跟着失败。
  - 一个会话装得下 CAP 个上下文单位。单个长会话超了就压缩: 占用降到 CAP // 4,
    但上下文里只剩最近 2 条记录, 已通过功能的清单和约束一起丢掉, 于是重做已完成的功能。
  - 四种做法共用 BUDGET 的总预算, 预算用完或全部功能通过就停。

简化: 上下文占用、实现与测试的开销都是常数, 不随功能变难; 回归只在同一个共享模块内扩散;
  压缩只按「只留最近 2 条」建模, 不模拟摘要的质量差别。数字是这套假设下算出来的, 不是实测。
简化: 不模拟真实的 git, 提交记录只是「哪些功能进了磁盘」的一个布尔集合。
读代码时盯住: session 里 known 的两个来源 (self_done 与 notes), 以及 tests 为假时 broken 怎么扩散。
"""
from __future__ import annotations

from llm_agent.core.utils import banner, kv

N = 12  # 功能数
READ, IMPL, TEST, COMPACT = 400, 900, 300, 500  # 读进度文件、实现一个功能、跑一遍端到端测试、压缩一次
CAP, BUDGET = 8000, 40000  # 一个会话的上下文容量、四种做法共用的总预算
MOD = lambda i: i % 4  # 功能 i 改动的共享模块
FLAKY = lambda i: i % 3 == 2  # 第一次写出来是半成品, 端到端测一遍才会发现


class Project:
    """磁盘状态 (notes / commits) 与真实状态 (passing / broken) 分开记, 虚报完成就是两者不一致。"""

    def __init__(self) -> None:
        self.passing: set = set()  # 真的能通过测试的功能
        self.broken: set = set()  # 记成完成但其实不通过 (虚报)
        self.notes: set = set()  # 进度文件里记着的功能 (含约束)
        self.commits: set = set()  # 提交记录里的功能
        self.regress = self.redo = self.sessions = self.used = 0

    def claimed(self) -> set:
        return self.passing | self.broken


def session(p: Project, notes: bool, tests: bool, cap: int = CAP) -> int:
    """跑一个会话: 读进度、挑一个功能、实现、(可选) 端到端测试、提交、更新进度。返回这个会话花掉的上下文。"""
    p.sessions += 1
    used = READ if notes else 0  # 没有进度文件就没什么可读的
    self_done: set = set()  # 本会话自己做过的功能, 约束还在上下文里
    while used + IMPL + (TEST if tests else 0) <= cap:
        # 挑下一个功能: 只认自己知道的进度。没有进度文件时每个新会话都从头挑
        seen = (p.notes if notes else set()) | self_done
        todo = [i for i in range(N) if i not in seen]
        if not todo:
            break
        i = todo[0]
        if i in p.passing:
            p.redo += 1  # 它其实已经通过了, 这次实现是白做的
        used += IMPL
        # ★ 约束的两个来源: 本会话自己做过, 或清单里记过。都没有, 同模块的已通过功能回归
        known = self_done | (p.notes if notes else set())
        hit = {j for j in p.passing if MOD(j) == MOD(i) and j not in known}
        if tests:
            used += TEST  # 端到端跑一遍: 半成品和回归都在这里暴露
            if FLAKY(i):
                used += IMPL // 2  # 补完半成品
            used += (IMPL // 2) * len(hit)  # 修回被改坏的功能
            p.regress += len(hit)
            p.passing.add(i)
            p.broken.discard(i)
        else:
            p.regress += len(hit)
            p.passing -= hit
            p.broken |= hit
            # 不测就记成完成: 半成品留在原地, 后面同模块的功能建在坏状态上也失败
            if FLAKY(i) or any(MOD(j) == MOD(i) for j in p.broken):
                p.broken.add(i)
            else:
                p.passing.add(i)
        self_done.add(i)
        p.commits.add(i)
        if notes:
            p.notes.add(i)  # 更新进度文件, 下一个会话才接得上
    return used


def single_long(p: Project) -> None:
    """一个长会话, 靠压缩硬撑: 压缩后上下文里只剩最近 2 条记录, 已完成清单和约束一起丢。"""
    self_done: list = []
    used = 0
    while p.used < BUDGET and len(p.passing) < N:
        todo = [i for i in range(N) if i not in set(self_done)]
        if not todo:
            self_done = self_done[-2:]  # 清单也在上下文里, 压缩之后同样只剩最近 2 条
            continue
        i = todo[0]
        if used + IMPL + TEST > CAP:  # 装不下了, 压缩一次再继续
            used, self_done = CAP // 4, self_done[-2:]
            p.used += COMPACT
            continue
        if i in p.passing:
            p.redo += 1
        hit = {j for j in p.passing if MOD(j) == MOD(i) and j not in set(self_done)}
        step = IMPL + TEST + (IMPL // 2 if FLAKY(i) else 0) + (IMPL // 2) * len(hit)
        if p.used + step > BUDGET:
            break
        p.regress += len(hit)
        used, p.used = used + step, p.used + step
        p.passing.add(i)
        self_done.append(i)
    p.sessions = 1


def run(mode: str, cap: int = CAP) -> Project:
    p = Project()
    if mode == "single-compaction":
        single_long(p)
        return p
    notes, tests = mode != "fresh-no-state", mode != "handoff-no-tests"
    while len(p.claimed()) < N:
        limit = min(cap, BUDGET - p.used)  # 每个会话的容量不超过剩下的总预算
        if limit < IMPL:
            break  # 一个功能也放不进去, 再开会话只是白读一遍进度文件
        spent = session(p, notes, tests, limit)
        if spent == 0:
            break
        p.used += spent
    return p


def row(p: Project) -> dict:
    return {"passed": len(p.passing), "claimed": len(p.claimed()), "false": len(p.broken),
            "regress": p.regress, "redo": p.redo, "sessions": p.sessions, "used": p.used}


MODES = ("single-compaction", "fresh-no-state", "handoff-no-tests", "handoff")
LABEL = {"single-compaction": "单个长会话 (只靠压缩)", "fresh-no-state": "新会话, 没有交接状态",
         "handoff-no-tests": "新会话 + 进度文件, 不跑测试", "handoff": "新会话 + 进度文件 + 清单 + 测试"}


def main() -> None:
    banner("M60 - Long-horizon handoff")

    print(f"\n[1] {N} 个功能, 每个会话装得下 {CAP} 个上下文单位, 四种做法共用 {BUDGET} 的总预算")
    rows = {m: row(run(m)) for m in MODES}
    for m, r in rows.items():
        kv(LABEL[m], f"真正通过 {r['passed']}/{N}, 声称完成 {r['claimed']}, 虚报 {r['false']}, "
                     f"回归 {r['regress']}, 重做 {r['redo']}, {r['sessions']} 个会话, 用掉 {r['used']}")
    hand, long_, bare, untested = rows["handoff"], rows["single-compaction"], rows["fresh-no-state"], rows["handoff-no-tests"]
    assert hand["passed"] == N, f"接力做法应该把 {N} 个功能全做完, 实际 {hand['passed']}"
    assert hand["false"] == 0, "每个功能都端到端测过才记进度, 不该有虚报"
    assert hand["regress"] == 0, "同模块的约束记在清单里, 不该因为换了会话而回归"
    assert long_["passed"] < hand["passed"], "同样的预算下, 只靠压缩的单个长会话完成的功能更少"
    assert long_["redo"] > 0, "压缩丢掉已完成清单后, 长会话会重做已经通过的功能"
    assert long_["regress"] > 0, "压缩丢掉约束后, 长会话会改坏同模块里已经通过的功能"
    assert bare["passed"] < hand["passed"], "没有交接状态时进度停在一个会话装得下的范围内"
    assert bare["redo"] >= bare["passed"], "没有交接状态时每个新会话都从第一个功能重做一遍"
    assert untested["false"] > 0, "不端到端测就记进度, 半成品会被当成完成"
    assert untested["claimed"] > untested["passed"], "声称完成的数量大于真正通过的数量"

    print("\n[2] 每个会话的上下文容量变化 (接力做法, 容量越小, 一个会话做得越少)")
    small = None
    for cap in (1600, 3000, 8000, 20000):
        p = run("handoff", cap)
        small = small or p
        kv(f"容量 {cap}", f"真正通过 {len(p.passing)}/{N}, {p.sessions} 个会话, 用掉 {p.used}")
    assert len(small.passing) == N, "容量只装得下一个功能时, 接力做法仍然做完全部功能"
    assert small.sessions == N, f"容量 1600 时每个会话只做一个功能, 应该开 {N} 个会话, 实际 {small.sessions}"
    assert small.used > rows["handoff"]["used"], "会话越多, 每次读进度文件的开销重复付出, 总上下文更多"

    print("\n[3] 坏状态怎么扩散 (不跑测试时, 按功能编号列出最终状态)")
    p = run("handoff-no-tests")
    kv("半成品的功能", f"{sorted(i for i in range(N) if FLAKY(i))} (FLAKY: i % 3 == 2)")
    kv("最终不通过的功能", f"{sorted(p.broken)}; 其中 {sorted(i for i in p.broken if not FLAKY(i))} 是建在坏状态上的")
    assert any(not FLAKY(i) for i in p.broken), "坏状态会让同模块里本来写对的功能也不通过"

    print("\n  OK: 新会话从干净上下文开始, 靠进度文件、功能清单和端到端测试交接。")


if __name__ == "__main__":
    main()
