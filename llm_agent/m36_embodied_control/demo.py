"""一维控制环境: 动作块、闭环反馈、模型偏差与 Sim2Real 随机化。"""
from __future__ import annotations
import json


def rollout(gain, mode, target=1.0, limit=1.5):
    x, trace, violated = 0., [], False
    for step in range(8):
        if mode == "open-loop":
            action = .2 if step < 5 else 0.
        elif mode in {"feedback", "robust"}:
            action = max(-.25, min(.25, target-x))
            if mode == "robust":
                # 用域随机化范围的最坏增益限定动作, 不能用未知真实增益偷看。
                action = max(-.25, min(action, max(0., (limit-x) / 2.0)))
        else:
            raise ValueError(mode)
        x += gain * action
        violated |= x > limit
        trace.append({"title": f"控制步 {step+1}", "detail": f"action={action:.3f}, position={x:.3f}, gain={gain}", "ok": x <= limit})
    return {"position": round(x, 5), "error": round(abs(x-target), 5), "violated": violated, "trace": trace}


def experiment(mode="robust"):
    rows = [rollout(gain, mode, limit=1.0) for gain in (.6, 1., 1.8)]
    return {"worst_error": max(r["error"] for r in rows), "violations": sum(r["violated"] for r in rows),
            "domains": [{k: v for k, v in r.items() if k != "trace"} for r in rows], "trace": rows[-1]["trace"]}


def main():
    rows = {m: experiment(m) for m in ("open-loop", "feedback", "robust")}
    assert rows["open-loop"]["violations"] == 1
    assert rows["robust"]["violations"] == 0 and rows["robust"]["worst_error"] < rows["open-loop"]["worst_error"]
    assert rollout(1., "open-loop")["error"] == 0, "标称模拟器成功仍可能迁移失败"
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
