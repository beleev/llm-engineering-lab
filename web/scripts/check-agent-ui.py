"""Agent 实验台的浏览器验收 (可选依赖 playwright)。

先在 web/ 启动 npm run dev; 再从仓库根运行:
    python web/scripts/check-agent-ui.py
    python web/scripts/check-agent-ui.py --screenshots  # 更新真实章节截图
截图前后端数值的一致性由 tests/test_agent_extensions.py 独立验证。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ["evals", "memory-lifecycle", "completion-verification", "runtime-reliability",
          "context-cost", "evidence-chain", "trajectory-learning", "learning-loop"]


def stats(lab):
    return lab.locator(".lab-stats .kv b").all_text_contents()


def run(base: str, screenshots: bool = False) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        checked = 0
        for route in ROUTES:
            page.goto(base + "#/agent/" + route)
            expect(page.locator(".page-title")).to_contain_text({
                "evals": "Agent 评测", "memory-lifecycle": "记忆更新", "completion-verification": "完成验收",
                "runtime-reliability": "故障恢复", "context-cost": "上下文成本", "evidence-chain": "检索证据",
                "trajectory-learning": "轨迹学习", "learning-loop": "持续改进",
            }[route])
            page.locator(".agent-experiment").first.wait_for()
            labs = page.locator(".agent-experiment")
            count = labs.count()
            assert count == (2 if route == "runtime-reliability" else 1)
            assert page.locator(".page-title").inner_text()
            assert page.locator(".codeblock pre").first.inner_text().strip(), "骨架代码不能是空框"
            for index in range(count):
                lab = labs.nth(index)
                lab.scroll_into_view_if_needed()
                assert lab.locator(".trace button").count() > 0
                # 每个 select 的每个选项都实际触发 Vue 更新。
                selects = lab.locator("select")
                for i in range(selects.count()):
                    select = selects.nth(i)
                    original = select.input_value()
                    values = select.locator("option").evaluate_all("xs => xs.map(x => x.value)")
                    for value in values:
                        select.select_option(value)
                        page.wait_for_timeout(30)
                        assert "NaN" not in lab.inner_text() and "undefined" not in lab.inner_text()
                    select.select_option(original)
                # 原生按钮的键盘激活, 而不只测鼠标。
                buttons = lab.locator(".trace button")
                buttons.last.focus()
                buttons.last.press("Enter")
                assert "active" in buttons.last.get_attribute("class")
                challenge = lab.locator(".lab-challenge summary")
                challenge.focus()
                challenge.press("Enter")
                assert lab.locator(".lab-challenge").get_attribute("open") is not None
                challenge.press("Enter")
                # 滑杆在键盘上也能改变模型, 恢复默认再做数值验收。
                sliders = lab.locator('input[type="range"]')
                for i in range(sliders.count()):
                    slider = sliders.nth(i)
                    original = slider.input_value()
                    slider.focus()
                    slider.press("Home")
                    slider.press("End")
                    slider.fill(original)
                checked += 1

            first = labs.first
            if route == "evals":
                first.locator("select").nth(0).select_option("2")
                first.locator("select").nth(1).select_option("text")
                assert stats(first)[0] == "通过"
                first.locator("select").nth(1).select_option("trace")
                assert stats(first)[0] == "失败" and stats(first)[2] == "2"
            elif route == "memory-lifecycle":
                first.locator("select").select_option("versioned")
                assert stats(first) == ["3 / 3", "0", "0"]
            elif route == "completion-verification":
                first.locator("select").select_option("verified")
                assert stats(first) == ["是", "是", "4 / 4"]
                first.locator('input[type="checkbox"]').check()
                assert stats(first)[0:2] == ["否", "否"]
                first.locator('input[type="checkbox"]').uncheck()
            elif route == "runtime-reliability":
                first.locator("select").select_option("idempotent")
                assert stats(first) == ["1 次", "10 元", "2 / 0"]
                second = labs.nth(1)
                second.locator("select").select_option("first-valid")
                assert stats(second) == ["1", "7", "3"]
            elif route == "context-cost":
                assert stats(first) == ["3202.5", "1327.5", "3 / 3", "530 tok"]
                first.locator('input[type="range"]').fill("2")
                first.locator("select").nth(1).select_option("2")
                assert stats(first)[0] == "840.0"
                first.locator('input[type="range"]').fill("6")
                first.locator("select").nth(1).select_option("4")
            elif route == "evidence-chain":
                first.locator("select").nth(1).select_option("agentic")
                assert stats(first) == ["20 MB", "2 / 2", "2"]
                first.locator('input[type="range"]').fill("1")
                assert stats(first)[0] == "证据不足"
                first.locator('input[type="range"]').fill("2")
            elif route == "trajectory-learning":
                first.locator("select").select_option("true")
                assert "当前最高奖励：verified" in first.inner_text()
            elif route == "learning-loop":
                first.locator("select").select_option("versioned")
                assert stats(first) == ["50%", "100%", "50%", "100%"]

            for index in range(count):
                lab = labs.nth(index)
                lab.locator(".trace button").first.click()
                if screenshots:
                    filename = f"agent-{route}-extension-{index + 1}.png"
                    lab.screenshot(path=str(ROOT / "docs/screenshots" / filename))
            for theme in ("light", "dark"):
                page.evaluate("theme => document.documentElement.setAttribute('data-theme', theme)", theme)
                page.set_viewport_size({"width": 390, "height": 844})
                for index in range(count):
                    labs.nth(index).scroll_into_view_if_needed()
                    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), route + " 页面横向溢出"
                    assert labs.nth(index).bounding_box()["width"] <= 390
                page.set_viewport_size({"width": 1440, "height": 1100})
            page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
            # 刷新直接路由, 确认按需加载仍能恢复。
            page.reload()
            page.locator(".agent-experiment").first.wait_for()
            print(f"PASS {route}: {count} labs, keyboard, controls, narrow screen, reload")
        assert not errors, errors
        print(f"PASS {checked} interactive labs on {len(ROUTES)} routes; no page errors")
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:5173/llm-engineering-lab/")
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    run(args.base, args.screenshots)
