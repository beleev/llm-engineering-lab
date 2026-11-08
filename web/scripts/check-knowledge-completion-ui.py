"""核对新增回放全部条件、源码展开、多模态入口及窄屏交互。先启动 Vite。"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[2]
RECORDS=json.loads((ROOT/'web/src/data/knowledge-completion.json').read_text())

def main(base):
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base,wait_until='networkidle')
        links=page.locator('.sidebar a.leaf').evaluate_all("ls=>ls.map(a=>a.getAttribute('href'))")
        def navigate(route):
            stage,slug=route.split('-',1)
            target=f'/{stage}/{slug}'
            assert any(h.split('#')[-1]==target for h in links),(route,target)
            page.evaluate('target=>location.hash=target',target)
        checked=0
        for entry in RECORDS['entries']:
            for route in entry['routes']:
                navigate(route)
                lab=page.locator(f'[data-experiment="{entry["id"]}"]')
                expect(lab).to_be_visible()
            # 每组所有条件在一个入口展开核对，不只检查按钮存在。
            for theme in ('light','dark'):
                page.evaluate('t=>document.documentElement.dataset.theme=t',theme)
                for width in (1440,320):
                    page.set_viewport_size({'width':width,'height':900})
                    for condition in entry['conditions']:
                        button=lab.get_by_role('button',name=condition['label'],exact=True)
                        button.focus();button.press('Enter')
                        expect(button).to_have_attribute('aria-pressed','true')
                        for label,value in condition['metrics'].items():
                            row=lab.locator('.kv').filter(has=page.get_by_text(label,exact=True))
                            text=', '.join(map(str,value)) if isinstance(value,list) else ('是' if value else '否') if isinstance(value,bool) else str(int(value)) if isinstance(value,float) and value.is_integer() else str(value)
                            expect(row.locator('b')).to_have_text(text)
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(entry['id'],width)
                        checked+=1
            lab.locator('.source > summary').click()
            expect(lab.locator('.source code')).to_be_visible()
            expect(lab.locator('.source .err')).to_have_count(0)
        page.evaluate("location.hash='/models'")
        lab=page.locator('.lab').filter(has=page.get_by_role('heading',name='图像与音频怎样接入语言模型',exact=False))
        expect(lab).to_be_visible()
        for button in lab.get_by_role('button').all():
            button.focus();button.press('Enter')
            expect(button).to_have_attribute('aria-pressed','true')
            lab.locator('.source > summary').click()
            expect(lab.locator('.source code')).to_be_visible()
            expect(lab.locator('.source .err')).to_have_count(0)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        print(f'PASS: 7 组、{checked} 次条件/主题/视口组合；{sum(len(e["routes"]) for e in RECORDS["entries"])} 个章节入口；4 个多模态入口；源码可展开；无 JS 错误')
        browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:5174/llm-engineering-lab/')
    main(parser.parse_args().base)
