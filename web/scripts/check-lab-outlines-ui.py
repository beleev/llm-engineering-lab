"""实验台外描边不能被滚动容器截掉。覆盖全部目录入口、视口和键盘焦点。

先启动 Vite；额外依赖 playwright。此检查验证几何裁剪和交互，不代替全站像素基线。
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

# 检查元素 border box 已经完整可见、但外描边伸出裁剪边界的情形。
# 故意在横向滚动区之外的内容不算错误；focus 会把边缘控件滚入视野后再检查。
CHECK = r'''focusOnly => {
 const issues=[];
 const nodes=focusOnly ? [document.activeElement, ...document.activeElement.querySelectorAll('*')] : document.querySelectorAll('#main .lab *');
 for(const e of nodes) {
  const s=getComputedStyle(e),r=e.getBoundingClientRect();
  const extent=parseFloat(s.outlineWidth)+parseFloat(s.outlineOffset);
  if(s.outlineStyle==='none'||!(extent>0)||!r.width||!r.height)continue;
  for(let p=e.parentElement;p&&p.id!=='main';p=p.parentElement) {
   const c=getComputedStyle(p),b=p.getBoundingClientRect();
   const x=/auto|scroll|hidden|clip/.test(c.overflowX),y=/auto|scroll|hidden|clip/.test(c.overflowY);
   if(!x&&!y)continue;
   const left=b.left+p.clientLeft,top=b.top+p.clientTop,right=left+p.clientWidth,bottom=top+p.clientHeight;
   if(r.left<left-.6||r.right>right+.6||r.top<top-.6||r.bottom>bottom+.6)continue;
   const sides=[];
   if(x&&r.left-extent<left-.6)sides.push('left');
   if(x&&r.right+extent>right+.6)sides.push('right');
   if(y&&r.top-extent<top-.6)sides.push('top');
   if(y&&r.bottom+extent>bottom+.6)sides.push('bottom');
   if(sides.length)issues.push({element:e.tagName+'.'+e.getAttribute('class'),text:e.textContent.trim().slice(0,45),ancestor:p.tagName+'.'+p.getAttribute('class'),sides,extent,outline:s.outline,offset:s.outlineOffset,focused:e.matches(':focus'),focusVisible:e.matches(':focus-visible')});
  }
 }
 return issues;
}'''


def check(page, context, failures):
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
    failures.extend({'context':context,**item} for item in page.evaluate(CHECK, context[-1] == 'keyboard-edge'))
    if not page.evaluate('document.documentElement.scrollWidth <= innerWidth'):
        failures.append({'context':context,'issue':'页面横向溢出'})


def run(base, output, focused=False, routes=None, layout_only=False):
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        errors=[];http_errors=[];failures=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:http_errors.append([r.status,r.url]) if r.status>=400 else None)
        page.goto(base,wait_until='networkidle')
        links=page.locator('.sidebar .leaf,.sidebar .overview').evaluate_all('es=>es.map(e=>e.getAttribute("href"))')
        if routes:
            wanted={'#/'+r.lstrip('#/') for r in routes}
            missing=wanted-set(links)
            assert not missing, f'未知路由: {missing}'
            links=[h for h in links if h in wanted]
        elif focused:
            links=[h for h in links if any(part in h for part in ('voice-interaction','data-packing','moe-seq-parallel','precision-stability','kv-footprint','decode-control','basic/forward','train/eval','finetune/sft','finetune/dpo','grpo-variants','agent/a2a','computer-use'))]
        checks=0
        for index,href in enumerate(links):
            # 搜索串变化强制加载新文档，避免仅 hash 导航时 networkidle 先于懒加载。
            page.goto(base+'?outline-audit='+str(index)+'#'+href.split('#')[-1],wait_until='networkidle')
            page.wait_for_function('!document.querySelector(".lab-placeholder")')
            for width in (1440,768,375):
                page.set_viewport_size({'width':width,'height':1000})
                for theme in ('light','dark'):
                    page.evaluate('t=>document.documentElement.dataset.theme=t',theme)
                    check(page,[href,width,theme,'selected'],failures)
                    for viz in ([] if layout_only else page.locator('.lab-viz').all()):
                        controls=viz.locator('button, input, select, [tabindex="0"]')
                        count=controls.count()
                        if not count:continue
                        for i in sorted({0,count-1}):
                            e=controls.nth(i)
                            if not e.is_visible() or e.evaluate('e=>e.disabled === true'):continue
                            e.focus();page.keyboard.press('Shift')
                            check(page,[href,width,theme,'keyboard-edge'],failures)
                    checks+=1
            if index%10==0:print(f'checked {index+1}/{len(links)} routes; clipping reports={len(failures)}',flush=True)
        # 原反馈：每种条件的第一、第二和最后一个事件，以及播放器，仍能改变选中步骤。
        page.goto(base+'?outline-voice#/agent/voice-interaction',wait_until='networkidle')
        lab=page.locator('.agent-coverage-lab')
        for mode in lab.locator('select[aria-label="实验条件"] option').evaluate_all('es=>es.map(e=>e.value)'):
            lab.get_by_label('实验条件',exact=True).select_option(mode)
            events=lab.locator('.coverage-timeline button')
            for index in sorted({0,1,events.count()-1}):
                button=events.nth(index);button.click()
                expect(button).to_have_attribute('aria-pressed','true')
                check(page,['voice-interaction',mode,index],failures)
            lab.get_by_role('slider',name='时间轴').focus()
            lab.get_by_role('slider',name='时间轴').press('Home')
            expect(events.first).to_have_attribute('aria-pressed','true')
            lab.get_by_role('button',name='下一步',exact=True).click()
            expect(events.nth(1)).to_have_attribute('aria-pressed','true')
        result={'keyboard_edges':not layout_only,'routes':len(links),'viewport_theme_cases':checks,'failures':failures,'js_errors':errors,'http_errors':http_errors}
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        browser.close()
        assert not failures,result
        assert not errors and not http_errors,result
        print(f'PASS: {len(links)} routes / {checks} viewport-theme cases; outlines, voice playback; keyboard edges={not layout_only}; no JS/HTTP errors')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:5174/llm-engineering-lab/')
    parser.add_argument('--output',type=Path,default=Path('/tmp/lab-outline-qa.json'))
    parser.add_argument('--focused',action='store_true',help='只跑原反馈及特殊裁剪容器')
    parser.add_argument('--route',action='append',help='复测指定路由，可重复，如 finetune/sft')
    parser.add_argument('--layout-only',action='store_true',help='只复查全局布局；用于局部键盘回归通过后的全站复查')
    args=parser.parse_args();run(args.base,args.output,args.focused,args.route,args.layout_only)
