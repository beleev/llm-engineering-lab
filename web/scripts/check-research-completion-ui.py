"""研究补充全条件回归：键盘、指标、原理、源码、窄屏和暗色。"""
from decimal import Decimal
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[2]


def display(v):
    if isinstance(v,list):return ', '.join(display(x) for x in v)
    if isinstance(v,bool):return '是' if v else '否'
    if isinstance(v,float):return format(Decimal(str(v)).normalize(),'f')
    return str(v)


def main(base, counts_only=False, entry_id=None, conditions_only=False, dataset='research'):
    records=json.loads((ROOT/f'web/src/data/{dataset}-completion.json').read_text())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        errors=[];http=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:http.append((r.url,r.status)) if r.status>=400 else None)
        page.goto(base,wait_until='networkidle')
        # 本轮路由包含 attention 等无阶段前缀的页面；全站路径在计数阶段从侧栏读取。
        checked=0
        selected=[] if counts_only else [e for e in records['entries'] if entry_id is None or e['id']==entry_id]
        if entry_id:assert selected,entry_id
        for index,entry in enumerate(selected):
            route=entry['routes'][0]
            path='/'+route.replace('-', '/',1) if '-' in route else '/'+route
            page.goto(base+f'?research-qa={index}#'+path,wait_until='networkidle')
            page.wait_for_function('!document.querySelector(".lab-placeholder")')
            lab=page.locator(f'[data-{dataset}="{entry["id"]}"]')
            expect(lab).to_be_visible()
            reading=lab.locator('.reading > summary');reading.focus();reading.press('Enter')
            expect(lab.locator('.protocol')).to_contain_text(entry['protocol'])
            for theme in ('light','dark'):
                page.evaluate('t=>document.documentElement.dataset.theme=t',theme)
                for width in (1440,768,320):
                    page.set_viewport_size({'width':width,'height':1000})
                    for condition in entry['conditions']:
                        button=lab.get_by_role('button',name=condition['label'],exact=True)
                        button.focus();button.press('Enter')
                        expect(button).to_have_attribute('aria-pressed','true')
                        for label,value in condition['metrics'].items():
                            row=lab.locator('.kv').filter(has=page.get_by_text(label,exact=True))
                            expect(row.locator('b')).to_have_text(display(value))
                        expect(lab.locator('.lab-note')).to_have_text(condition['note'])
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(entry['id'],theme,width)
                        checked+=1
            lab.locator('.source > summary').click()
            expect(lab.locator('.source code')).to_be_visible()
            expect(lab.locator('.source .err')).to_have_count(0)
            lab.locator('.lab-challenge > summary').focus();lab.locator('.lab-challenge > summary').press('Enter')
            expect(lab.locator('.answer')).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),entry['id']
            print(f'checked {index+1}/{len(records["entries"])}: {entry["id"]}',flush=True)
        if entry_id or conditions_only:
            assert not errors and not http,(errors,http)
            Path(f'/tmp/{dataset}-ui-conditions.json').write_text(json.dumps({'scope':entry_id or 'all-conditions','checked':checked,'js_errors':errors,'http_errors':http},ensure_ascii=False,indent=2))
            print(f'PASS {entry_id or "all conditions"}: {checked} 条件/主题/屏宽组合，无 JS/HTTP 错误',flush=True)
            browser.close();return
        # 全站路由、实际 LabFrame 数和目录计数交叉核对，避免仅数组件漏掉 v-for 多组实验。
        page.set_viewport_size({'width':1440,'height':1000})
        page.goto(base+'?research-count',wait_until='networkidle')
        links=page.locator('.sidebar a.leaf, .sidebar a.overview').evaluate_all('es=>es.map(e=>({href:e.getAttribute("href"),name:e.textContent}))')
        counts=[]
        for index,link in enumerate(links):
            path=link['href'].split('#')[-1]
            page.goto(base+f'?research-count={index}#'+path,wait_until='networkidle')
            page.wait_for_function('!document.querySelector(".lab-placeholder")')
            predicted=page.evaluate("async()=>{const {labsOf}=await import('/llm-engineering-lab/src/utils/readtime.js'); const app=document.querySelector('#app').__vue_app__;return labsOf(app.config.globalProperties.$route.name)}")
            # 异步组件可能尚未显示 200 ms 后才出现的占位框；等待实际框数，不把瞬时空 DOM 当渲染结果。
            expect(page.locator('section.lab, section.architecture-lab'),f'{path}: {predicted} labs').to_have_count(predicted,timeout=20000)
            count=page.locator('section.lab, section.architecture-lab').count()
            counts.append({'path':path,'labs':count})
            if index%20==0:print(f'counted {index+1}/{len(links)} routes',flush=True)
        assert len(links)==len({link['href'] for link in links}) and len(links)>=111,len(links)
        assert not errors and not http,(errors,http)
        report={'scope':'counts-only' if counts_only else 'full','groups':len(records['entries']),'conditions':sum(len(e['conditions']) for e in records['entries']),
                'viewport_theme_conditions':checked,'routes':counts,'lab_entries':sum(x['labs'] for x in counts),'js_errors':errors,'http_errors':http}
        Path(f'/tmp/{dataset}-ui-counts.json' if counts_only else f'/tmp/{dataset}-ui-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(f'PASS: {checked} 条件/视口/主题组合，{len(links)} 章节，{report["lab_entries"]} 实验入口，无 JS/HTTP 错误',flush=True)
        browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:5174/llm-engineering-lab/')
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--counts-only',action='store_true',help='仅重跑全站挂载计数，不重做已通过的条件回归')
    scope.add_argument('--entry',help='仅复测指定实验，不重复全站计数')
    scope.add_argument('--conditions-only',action='store_true',help='仅检查全部条件、源码与键盘操作')
    parser.add_argument('--dataset',choices=['research','engineering'],default='research')
    args=parser.parse_args();main(args.base,args.counts_only,args.entry,args.conditions_only,args.dataset)
