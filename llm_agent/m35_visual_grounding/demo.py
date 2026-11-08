"""像素定位、过期截图与动作后核对。真实 Chromium 版本在 browser.py。"""
from __future__ import annotations
import json

TARGET = (20, 180, 80)
OTHER = (180, 40, 60)


def render(shift=False):
    pixels = [[(255, 255, 255) for _ in range(80)] for _ in range(80)]
    positions = {"target": (10, 45 if shift else 10), "other": (10, 10 if shift else 45)}
    for key, (x, y) in positions.items():
        color = TARGET if key == "target" else OTHER
        for yy in range(y, y + 16):
            for xx in range(x, x + 40):
                pixels[yy][xx] = color
    return pixels, positions


def locate(pixels, color=TARGET):
    points = [(x, y) for y, row in enumerate(pixels) for x, pixel in enumerate(row) if tuple(pixel[:3]) == color]
    if not points:
        raise ValueError("截图中没有目标, 不猜坐标")
    return (min(x for x, y in points) + max(x for x, y in points)) // 2, (min(y for x, y in points) + max(y for x, y in points)) // 2


def hit(positions, point):
    x, y = point
    for key, (xx, yy) in positions.items():
        if xx <= x < xx + 40 and yy <= y < yy + 16:
            return key
    return None


def experiment(mode="verify"):
    if mode not in {"stale", "refresh", "verify"}:
        raise ValueError(mode)
    old, _ = render()
    current, positions = render(True)
    point = locate(current if mode == "refresh" else old)
    opened = hit(positions, point)
    trace = [{"title": "观察", "detail": f"从截图定位绿色目标, 坐标 {point}", "ok": True},
             {"title": "布局改变后点击", "detail": f"打开 {opened}", "ok": opened == "target"}]
    if mode == "verify" and opened != "target":
        point = locate(current)
        opened = hit(positions, point)
        trace.append({"title": "核对对象后恢复", "detail": f"发现对象不符, 重新截图并点击 {point}", "ok": opened == "target"})
    # 只读导航与副作用分开: 未核对对象不能确认业务操作。
    committed = opened == "target"
    trace.append({"title": "提交前核对", "detail": f"目标匹配={committed}", "ok": committed})
    return {"opened": opened, "committed": committed, "wrong_commit": False, "observations": 2 if mode == "verify" else 1, "trace": trace}


def main():
    rows = {m: experiment(m) for m in ("stale", "refresh", "verify")}
    assert not rows["stale"]["committed"]
    assert rows["refresh"]["committed"] and rows["verify"]["committed"]
    try:
        locate([[(0, 0, 0)]])
    except ValueError:
        pass
    else:
        raise AssertionError("缺目标时不能猜测")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
