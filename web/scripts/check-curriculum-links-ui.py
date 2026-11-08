"""关联入口真实导航；跨阶段前置链接、三级专题和返回路径。"""
import json
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='http://127.0.0.1:5174/llm-engineering-lab/'
ROOT=Path(__file__).resolve().parents[2]
TITLES=json.loads(subprocess.check_output(['node','--input-type=module','-e',"const all={};for(const s of ['basic','models','train','finetune','infer','agent'])for(const [r,p] of Object.entries((await import('./web/src/data/topics/'+s+'.js')).default.pages))all[r]=p.title;console.log(JSON.stringify(all));"],cwd=ROOT,text=True))
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox']);page=browser.new_page(viewport={'width':320,'height':900})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE,wait_until='networkidle')
    routes=page.evaluate("Object.fromEntries(document.querySelector('#app').__vue_app__.config.globalProperties.$router.getRoutes().map(r=>[r.name,r.path]))")
    checked=[]
    for route in ['finetune-sft','agent-memory-lifecycle','agent-programmatic-actions','agent-tools-permissions','agent-runtime-reliability','agent-evals','train-collectives-loop','infer-scheduler','infer-decode-control','models-mtp','finetune-knowledge-update','agent-verifiable-reasoning']:
        page.goto(BASE+'#'+routes[route],wait_until='networkidle')
        page.wait_for_function("r=>document.querySelector('#app').__vue_app__.config.globalProperties.$route.name===r",arg=route)
        expect(page.locator('.page-title')).to_have_text(TITLES[route])
        links=page.locator('.related-topics a');expect(links.first).to_be_visible()
        destinations=links.evaluate_all('es=>es.map(e=>e.getAttribute("href"))')
        assert all(h.split('#')[-1] in routes.values() for h in destinations)
        first=destinations[0];links.first.focus();links.first.press('Enter')
        page.wait_for_url('**'+first)
        expect(page.locator('.leaf[aria-current="page"]')).to_have_count(1)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),route
        page.go_back();page.wait_for_url('**#'+routes[route])
        expect(page.locator('.page-title')).to_be_visible()
        checked.append({'route':route,'destinations':destinations})
    assert not errors,errors
    Path('/tmp/engineering-links-ui.json').write_text(json.dumps({'routes':checked,'errors':errors},ensure_ascii=False,indent=2))
    print('PASS: 12 个关联入口，键盘导航、返回与 320px 溢出检查')
    browser.close()

# 长源码路径在窄屏应换行，不能只让整页“不横向滚动”却在卡片内部裁掉。
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox']);page=browser.new_page(viewport={'width':320,'height':1000})
    for route in ['finetune/runs','finetune/knowledge-update','agent/mcp','agent/rag','agent/eval-suite','infer/compute']:
        page.goto(BASE+'?source-width='+route+'#/'+route,wait_until='networkidle')
        expect(page.locator('.source-path').first).to_be_visible()
        failures=page.locator('.source-path').evaluate_all('''es=>es.filter(e=>{
          const range=document.createRange();range.selectNodeContents(e);const p=e.closest('summary').getBoundingClientRect();
          return [...range.getClientRects()].some(r=>r.left<p.left-1||r.right>p.right+1);
        }).map(e=>e.textContent)''')
        assert not failures,(route,failures)
    print('PASS: 6 个页面的 320px 源码路径文字边界')
    browser.close()
