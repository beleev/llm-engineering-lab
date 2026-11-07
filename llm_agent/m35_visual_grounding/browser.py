"""可选真实浏览器验收: pip install playwright pillow; playwright install chromium。

页面在内存中加载, 无网络。颜色定位只适用于本例, 不等价于通用视觉语言模型。
"""
from __future__ import annotations
from io import BytesIO
import json
from PIL import Image
from playwright.sync_api import sync_playwright
from llm_agent.m35_visual_grounding.demo import locate, TARGET

HTML = '''<!doctype html><meta charset="utf-8"><title>Local grounding fixture</title>
<style>button{position:absolute;left:20px;width:140px;height:40px;border:0;color:white}#target{top:20px;background:rgb(20,180,80)}#other{top:90px;background:rgb(180,40,60)}</style>
<button id="target" onclick="openOrder('target')">Order 1004</button>
<button id="other" onclick="openOrder('other')">Order 1002</button>
<p id="detail" style="position:absolute;top:160px"></p>
<script>window.opened=null;window.commits=[];function openOrder(id){opened=id;document.getElementById('detail').textContent='Opened '+id}</script>'''


def screenshot_point(page):
    image = Image.open(BytesIO(page.screenshot())).convert("RGB")
    width, height = image.size
    data = list(image.get_flattened_data()) if hasattr(image, "get_flattened_data") else list(image.getdata())
    return locate([data[i*width:(i+1)*width] for i in range(height)], TARGET)


def experiment(mode="verify"):
    if mode not in {"stale", "refresh", "verify"}:
        raise ValueError(mode)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": 400, "height": 240}, device_scale_factor=1)
            page.route("**/*", lambda route: route.abort())
            page.set_content(HTML)
            old_point = screenshot_point(page)
            page.evaluate("document.querySelector('#target').style.top='90px';document.querySelector('#other').style.top='20px'")
            point = screenshot_point(page) if mode == "refresh" else old_point
            page.mouse.click(*point)
            first = page.evaluate("opened")
            if mode == "verify" and first != "target":
                page.mouse.click(*screenshot_point(page))
            opened = page.evaluate("opened")
            if opened == "target":
                page.evaluate("commits.push(opened)")
            commits = page.evaluate("commits")
            assert commits in ([], ["target"])
            return {"mode": mode, "first": first, "opened": opened, "commits": commits}
        finally:
            browser.close()


def main():
    rows = [experiment(mode) for mode in ("stale", "refresh", "verify")]
    assert rows[0]["first"] == rows[2]["first"] == "other"
    assert rows[0]["commits"] == [] and rows[1]["commits"] == rows[2]["commits"] == ["target"]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
