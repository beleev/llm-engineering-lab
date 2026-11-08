"""新增专题的真实浏览器验收与截图。先在 web 启动 Vite。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


def display(value):
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "未产生"
    if isinstance(value, list):
        return " · ".join(map(display, value)) if value else "无"
    if isinstance(value, dict):
        return " / ".join(f"{k}: {display(v)}" for k, v in value.items())
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def run(base, screenshots=False):
    data = json.loads((ROOT / "web/src/data/agent-coverage-evidence.json").read_text())
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1, reduced_motion="reduce")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            for route, row in data.items():
                page.goto(base + "#/agent/" + route.removeprefix("agent-"))
                expect(page.locator(".page-title")).to_have_text(row["title"])
                lab = page.locator(".agent-coverage-lab")
                expect(lab).to_be_visible()
                assert page.locator(".katex-error").count() == 0
                assert page.locator(".codeblock pre").first.inner_text().strip()
                for case in row["cases"]:
                    lab.get_by_label("实验条件", exact=True).select_option(case["mode"])
                    expect(lab.locator(".lab-stats .kv")).to_have_count(len(case["metrics"]))
                    expect(lab.locator(".lab-stats .kv b")).to_have_text([display(m["value"]) for m in case["metrics"]])
                    buttons = lab.locator(".coverage-timeline button")
                    expect(buttons).to_have_count(len(case["trace"]))
                    buttons.last.focus()
                    buttons.last.press("Enter")
                    expect(buttons.last).to_have_attribute("aria-pressed", "true")
                    expect(lab.locator(".coverage-detail h4")).to_have_text(case["trace"][-1]["title"])
                    slider = lab.get_by_role("slider", name="时间轴")
                    slider.focus()
                    slider.press("Home")
                    expect(buttons.first).to_have_attribute("aria-pressed", "true")
                    lab.get_by_role("button", name="下一步", exact=True).click()
                    expect(buttons.nth(1)).to_have_attribute("aria-pressed", "true")
                    assert "undefined" not in lab.inner_text() and "NaN" not in lab.inner_text()
                checkbox = lab.get_by_role("checkbox")
                checkbox.check()
                expect(lab.locator(".comparison-row")).to_have_count(3)
                checkbox.uncheck()
                lab.locator(".lab-challenge summary").focus()
                lab.locator(".lab-challenge summary").press("Enter")
                expect(lab.locator(".lab-challenge")).to_have_attribute("open", "")
                lab.locator(".lab-challenge summary").press("Enter")
                for theme in ("light", "dark"):
                    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), route
                    assert lab.is_visible()
                page.set_viewport_size({"width": 1440, "height": 1100})
                if screenshots:
                    preferred = {"m28_coding_agent": "repair", "m26_tool_discovery": "discovery", "m33_skill_evolution": "scoped", "m39_meta_optimization": "breadth"}.get(row["module"], row["cases"][-1]["mode"])
                    lab.get_by_label("实验条件", exact=True).select_option(preferred)
                    lab.locator(".coverage-timeline button").last.click()
                    page.evaluate("document.activeElement.blur()")
                    lab.scroll_into_view_if_needed()
                    lab.screenshot(path=str(ROOT / "docs/screenshots" / f"{route}-coverage.png"))
                page.reload()
                expect(page.locator(".agent-coverage-lab")).to_be_visible()
                print(f"PASS {route}: 3 configs, keyboard, narrow, themes, refresh")
            assert not errors, errors
            print(f"PASS {len(data)} routes, {sum(len(r['cases']) for r in data.values())} configurations")
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:5173/")
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    run(args.base, args.screenshots)
