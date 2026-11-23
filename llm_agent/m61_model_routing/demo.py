"""M61 — 模型路由与级联: 按每个完成的任务算钱, 不按每次请求算。

一批 agent 任务难度不一, 可以全交给一个大模型, 也可以先让便宜的小模型试 (级联, FrugalGPT),
或者先用一个分类器按预测难度分配 (路由, RouteLLM)。比较这些做法要用同一把尺子:
    每个完成任务的成本 = 总花费 (含失败尝试) / 完成的任务数, 并且完成率要过同一条线 (FLOOR)。
每次请求的成本会误导: 小模型单次便宜, 失败的尝试和升级后的重做都要算进账。

成本模型 (确定性, 纯标准库):
  - 每次尝试是一段 TURNS 轮的 agent 循环。每轮输入 = 共享前缀 (system + tools) + 本任务已有历史 + 新增内容,
    输出 = 该 effort 下的 thinking + 回复。价格按 $/1M token, 缓存读 0.1 倍, 写 1.25 倍。
  - 缓存按模型隔离: 换模型后, 前一个模型留下的历史在新模型那里全部未缓存, 要按写入价重读。
  - 任务难度 d 在 [0, 1] 上按分位数均匀摆放 (不用随机数); 模型在某档 effort 下能力为 c, d ≤ c 即完成。
  - 级联的验收器是完美的 (失败一定被发现、成功一定被接受), 这是对级联最有利的假设。
简化: 能力阈值 CAP 与每轮 token 数是假设值, 不是任何模型的实测; 价格取 Claude 当前价目表的比例
(大模型按 Opus 5.5 的 4 / 20, 中模型按 Sonnet 5.5 的 2 / 10, 小模型按 Haiku 5.5 的 0.10 / 0.50)。
简化: 不计路由分类器本身的开销, 不计延迟的钱, 不模拟验收器漏判, 也不模拟跨模型时 thinking 块读不读得到。
读代码时盯住: attempt 里 carried 那一项 (换模型接着做时被重新写入的历史), 和 run 里 cost / done 的分母。
"""
from __future__ import annotations

import math

from llm_agent.core.utils import banner, kv

PRICE = {"big": (4.0, 20.0), "mid": (2.0, 10.0), "small": (0.10, 0.50)}  # $/1M token, 输入 / 输出
CACHE_READ, CACHE_WRITE = 0.1, 1.25
CAP = {("big", "high"): 0.88, ("big", "medium"): 0.80, ("big", "low"): 0.70,
       ("mid", "medium"): 0.60, ("small", "medium"): 0.35}  # 假设的能力阈值: 难度不超过它就能完成
OUT = {"low": 400, "medium": 900, "high": 2000}  # 每轮输出 token (thinking + 回复), effort 越高想得越多
PREFIX, STEP, TURNS = 6000, 2000, 6  # 共享前缀、每轮新增输入、每次尝试的轮数
FLOOR = 0.80  # 完成率门槛: 低于它的策略不参与比价
N = 200


def tasks(easy_share: float = 0.3, n: int = N) -> list:
    """简单任务 (d < 0.3) 占 easy_share, 其余落在 [0.3, 1); 各段内按分位数均匀摆放。"""
    k = round(n * easy_share)
    return [0.3 * (i + 0.5) / k for i in range(k)] + [0.3 + 0.7 * (j + 0.5) / (n - k) for j in range(n - k)]


def router_noise(i: int, sigma: float) -> float:
    """路由分类器的预测误差: 确定性的均匀噪声, 标准差为 sigma (黄金分割序列, 不用随机数)。"""
    return sigma * math.sqrt(12) * ((i * 0.6180339887) % 1 - 0.5)


def attempt(model: str, effort: str, warm: bool, carried: int) -> dict:
    """一次 TURNS 轮的尝试。warm: 这个模型的共享前缀已在缓存里; carried: 从别的模型接过来的历史 token。"""
    pin, pout = PRICE[model]
    cost = read = written = 0.0
    history = carried
    for k in range(TURNS):
        if k == 0:  # ★ 第一轮: 接过来的历史在本模型缓存里没有, 和新内容一起按写入价算
            hit, write = (PREFIX if warm else 0), (0 if warm else PREFIX) + carried + STEP
        else:  # 之后每轮: 前缀和本次尝试里已写过的历史都能读缓存, 只写新增
            hit, write = PREFIX + history, STEP
        cost += (hit * CACHE_READ + write * CACHE_WRITE) * pin / 1e6 + OUT[effort] * pout / 1e6
        read, written = read + hit, written + write
        history += STEP + OUT[effort]
    return {"cost": cost, "read": read, "written": written, "history": history}


def run(plan, ts: list, carry: bool = False) -> dict:
    """按 plan(i, d) 给出的 [(model, effort), ...] 依次尝试, 直到某一步完成或全部失败。"""
    cost = read = written = 0.0
    done = calls = 0
    warm = set()  # 缓存按模型隔离: 各模型的共享前缀各自要写一次
    for i, d in enumerate(ts):
        carried = 0
        for model, effort in plan(i, d):
            a = attempt(model, effort, model in warm, carried)
            warm.add(model)
            cost, read, written, calls = cost + a["cost"], read + a["read"], written + a["written"], calls + TURNS
            if d <= CAP[(model, effort)]:
                done += 1
                break
            carried = a["history"] if carry else 0  # carry: 把失败的记录交给下一个模型接着做
    return {"rate": done / len(ts), "cost": cost, "per_call": cost / calls, "per_done": cost / max(done, 1),
            "hit": read / (read + written), "turns": calls / len(ts)}


def policies(small: str = "small", sigma: float = 0.15, cut: float = 0.3) -> dict:
    """策略名 → plan。级联与路由的小模型可以换成中模型价位。"""
    cascade = lambda i, d: [(small, "medium"), ("big", "medium")]
    return {
        "单模型 high": lambda i, d: [("big", "high")],
        "单模型 medium": lambda i, d: [("big", "medium")],
        "单模型 low": lambda i, d: [("big", "low")],
        "只用小模型": lambda i, d: [(small, "medium")],
        "级联 (失败从头换大模型)": cascade,
        "级联 (失败把记录交给大模型)": cascade,
        "路由 (按预测难度分配)": lambda i, d: cascade(i, d) if d + router_noise(i, sigma) < cut else [("big", "medium")],
    }


def table(easy_share: float = 0.3, small: str = "small", sigma: float = 0.15) -> dict:
    ts = tasks(easy_share)
    return {name: run(plan, ts, carry="交给" in name) for name, plan in policies(small, sigma).items()}


def show(rows: dict) -> None:
    for name, r in rows.items():
        flag = "" if r["rate"] >= FLOOR else "  (完成率不到门槛)"
        kv(name, f"完成率 {r['rate']:.1%}, 每次请求 ${r['per_call']:.4f}, 每个完成任务 ${r['per_done']:.3f}, "
                 f"缓存命中 {r['hit']:.1%}, 平均 {r['turns']:.1f} 轮{flag}")


def main() -> None:
    banner("M61 - Model routing & cascades")

    print(f"\n[1] {N} 个任务, 简单任务占 30%, 小模型按 Haiku 5.5 价位")
    rows = table()
    show(rows)
    one, med, high, low = rows["只用小模型"], rows["单模型 medium"], rows["单模型 high"], rows["单模型 low"]
    restart, carry, route = rows["级联 (失败从头换大模型)"], rows["级联 (失败把记录交给大模型)"], rows["路由 (按预测难度分配)"]
    assert one["per_done"] < restart["per_done"], "只用小模型的每任务成本最低, 但它的完成率不到门槛"
    assert one["rate"] < FLOOR, f"只用小模型完成率 {one['rate']:.1%} 应低于门槛, 不能参与比价"
    assert low["rate"] < FLOOR, f"单模型 low 完成率 {low['rate']:.1%} 应低于门槛"
    assert med["per_done"] < high["per_done"], "同一模型把 effort 从 high 降到 medium, 每个完成任务更便宜"
    assert restart["rate"] == med["rate"] == route["rate"], "级联与路由最后都交给大模型 medium, 完成率应相同"
    assert restart["per_done"] < med["per_done"], "小模型便宜到 1/40 时, 从头换大模型的级联比单模型 medium 便宜"
    assert restart["turns"] > med["turns"], "级联的代价之一是轮数: 升级的任务要多跑一整段"
    assert carry["per_call"] < med["per_call"], "按每次请求算, 把记录交给大模型的级联比单模型 medium 便宜"
    assert carry["per_done"] > med["per_done"], "按每个完成任务算, 它反而比单模型 medium 贵"
    assert carry["hit"] < restart["hit"], "接着做时, 上一个模型的历史在大模型缓存里没有, 命中率更低"

    print("\n[2] 小模型换成 Sonnet 5.5 价位 (大模型的 1/2)")
    mid = table(small="mid")
    show({k: mid[k] for k in ("级联 (失败从头换大模型)", "级联 (失败把记录交给大模型)", "路由 (按预测难度分配)")})
    assert mid["级联 (失败从头换大模型)"]["per_done"] > restart["per_done"], "小模型越贵, 级联省得越少"

    print("\n[3] 简单任务占比变化: 级联 (从头换) / 单模型 medium 的每任务成本之比 (两者完成率相同)")
    ratio = {}
    for share in (0.1, 0.3, 0.5, 0.7):
        rows_s = table(share, small="mid")
        a, b = rows_s["级联 (失败从头换大模型)"], rows_s["单模型 medium"]
        assert a["rate"] == b["rate"], "级联最后由大模型 medium 兜底, 完成率应与单模型 medium 相同"
        ratio[share] = a["per_done"] / b["per_done"]
        kv(f"简单任务 {share:.0%}", f"完成率 {b['rate']:.1%}, 中模型价位级联 / 单模型 medium = {ratio[share]:.2f}")
    assert ratio[0.1] > 1 > ratio[0.3], "中模型价位时存在交叉点: 简单任务少, 级联更贵; 多了, 级联更便宜"
    assert all(ratio[a] > ratio[b] for a, b in zip((0.1, 0.3, 0.5), (0.3, 0.5, 0.7))), "简单任务越多, 级联越划算"

    print("\n[4] 路由分类器的误差: 预测难度的噪声越大, 升级越多 (小模型按 Haiku 5.5 价位)")
    for sigma in (0.0, 0.15, 0.4):
        r = table(sigma=sigma)["路由 (按预测难度分配)"]
        kv(f"噪声 σ={sigma}", f"每个完成任务 ${r['per_done']:.3f}, 平均 {r['turns']:.1f} 轮")
    perfect, noisy = table(sigma=0.0)["路由 (按预测难度分配)"], table(sigma=0.4)["路由 (按预测难度分配)"]
    assert perfect["per_done"] < noisy["per_done"], "路由越准, 浪费在小模型上的失败尝试越少"
    assert perfect["turns"] < noisy["turns"] < restart["turns"], "路由越准, 升级越少, 平均轮数越少; 都少于级联"
    # 小模型几乎免费时, 每个任务先试一次的浪费很小, 级联反而比完美路由便宜; 小模型贵到 1/2 时反过来
    assert restart["per_done"] < perfect["per_done"], "Haiku 价位: 级联的每任务成本低于完美路由, 路由省的是轮数"
    assert mid["路由 (按预测难度分配)"]["per_done"] < mid["级联 (失败从头换大模型)"]["per_done"], "Sonnet 价位: 路由比级联便宜"

    print("\n  OK: 先过完成率门槛, 再比每个完成任务的成本; 换模型接着做会丢掉缓存。")


if __name__ == "__main__":
    main()
