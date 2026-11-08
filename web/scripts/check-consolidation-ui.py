"""章节导航：路由别名、段落锚点、进度迁移与实验的真实浏览器回归。"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
BASE='http://127.0.0.1:5174/llm-engineering-lab/'
ALIASES={
 'train-low-precision':('train-precision-stability','block-scale','TrainBlockScaleLab'),
 'finetune-dora':('finetune-lora','dora','FtDoraLab'),
 'agent-context-cost':('agent-prompt-caching','context-cost','AgentContextCostLab'),
 'agent-collaboration':('agent-orchestrator','collaboration','AgentCoverageLab'),
 'agent-memory-privacy':('agent-memory-lifecycle','memory-privacy','AgentCoverageLab'),
 'models-qknorm-yarn':('attention','qk-norm','ModelQkNormLab'),
}
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
    errors=[];http=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:http.append((r.url,r.status)) if r.status>=400 else None)
    page.goto(BASE,wait_until='networkidle')
    expect(page.locator('.sidebar .leaf')).to_have_count(101)
    legacy={'visited':{'finetune-dora':200,'finetune-lora':100},'last':'finetune-dora',
            'quiz':{'finetune-dora':{'correct':3,'total':3},'finetune-lora':{'correct':3,'total':3}},'level':'all'}
    page.evaluate('(s)=>localStorage.setItem("llm-progress",JSON.stringify(s))',legacy)
    page.reload(wait_until='networkidle')
    saved=page.evaluate('JSON.parse(localStorage.getItem("llm-progress"))')
    assert saved['visited']['finetune-lora']==200 and 'finetune-dora' not in saved['visited'],saved
    assert 'finetune-lora' not in saved['quiz'] and len(saved['archivedQuiz'])==2,saved
    page.reload(wait_until='networkidle')
    assert page.evaluate('JSON.parse(localStorage.getItem("llm-progress")).archivedQuiz')==saved['archivedQuiz']
    checked=[]
    for old,(target,anchor,_) in ALIASES.items():
        for theme in ('light','dark'):
            page.evaluate('t=>document.documentElement.dataset.theme=t',theme)
            for width in (1440,768,320):
                page.set_viewport_size({'width':width,'height':1000})
                print(old,theme,width,flush=True)
                page.goto(BASE+'#/'+old.replace('-','/',1),wait_until='networkidle')
                page.wait_for_function('r=>document.querySelector("#app").__vue_app__.config.globalProperties.$route.name===r',arg=target)
                expect(page.locator('#'+anchor)).to_be_visible()
                page.wait_for_function('id=>Math.abs(document.getElementById(id).getBoundingClientRect().top-24)<12',arg=anchor)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(old,width)
                expect(page.locator('#'+anchor+' .lab')).to_have_count(1)
                summary=page.locator('#'+anchor+' .supplement-source > summary')
                if page.locator('#'+anchor+' .supplement-source').get_attribute('open') is None:
                    summary.focus();summary.press('Enter')
                expect(page.locator('#'+anchor+' .supplement-source')).to_have_attribute('open','')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(old,width,'sources')
                checked.append((old,theme,width))
    for route,section in [('position','yarn'),('models/mamba','linear-hybrid'),('attention','swa')]:
        page.goto(BASE+'#/'+route,wait_until='networkidle')
        expect(page.locator('#'+section)).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),route
    # 旧章节名内部导航也应走同一 redirect，浏览器后退恢复原页。
    page.goto(BASE+'#/finetune/sft',wait_until='networkidle')
    page.evaluate('()=>document.querySelector("#app").__vue_app__.config.globalProperties.$router.push({name:"finetune-dora"})')
    page.wait_for_url('**#/finetune/lora#dora')
    page.go_back();page.wait_for_url('**#/finetune/sft')
    assert not errors and not http,(errors,http)
    Path('/tmp/consolidation-ui.json').write_text(json.dumps({'alias_theme_viewport_cases':len(checked),'supplements':3,'errors':errors,'http':http},ensure_ascii=False,indent=2))
    page.goto(BASE+'#/finetune/lora#dora',wait_until='networkidle')
    page.screenshot(path='/tmp/consolidation-mobile.png',full_page=False)
    print(f'PASS: {len(checked)} 旧链接/主题/宽度组合，3 个进阶段落，进度迁移与前进后退')
    browser.close()
