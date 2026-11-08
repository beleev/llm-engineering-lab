"""三级导航回归：位置、对比度、裁切和真实键盘/鼠标交互。需先启动 Vite。

python web/scripts/check-navigation-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
依赖 playwright 和 Chromium；不访问模型 API，不改学习内容。
"""
from __future__ import annotations
import argparse
import json
from playwright.sync_api import sync_playwright, expect

# 检查实际文字所在矩形和浏览器最终颜色，而不只检查 DOM 里是否还留着编号。
GEOMETRY = r"""() => {
 const failures = [], sidebar = document.querySelector('#sidebar');
 const visible = e => e.getClientRects().length > 0;
 const rect = e => e.getBoundingClientRect();
 const rgb = c => c.match(/[\d.]+/g).map(Number);
 const over = (fg,bg) => { const a=fg[3] ?? 1; return fg.slice(0,3).map((v,i)=>v*a+bg[i]*(1-a)); };
 const background = e => {const chain=[];for(let n=e;n;n=n.parentElement)chain.unshift(n);return chain.reduce((bg,n)=>over(rgb(getComputedStyle(n).backgroundColor),bg),[255,255,255]);};
 const luminance = c => c.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((v,c,i)=>v+c*[.2126,.7152,.0722][i],0);
 for (const stage of document.querySelectorAll('.stage-links')) {
  if(!visible(stage))continue;
  const overview=stage.querySelector('.overview');
  for (const group of stage.querySelectorAll('.group-toggle')) {
   for(const [a,b] of [[overview.querySelector('.idx'),group.querySelector('.idx')],[overview.lastElementChild,group.querySelector('.group-title')]])
    if(Math.abs(rect(a).left-rect(b).left)>.5) failures.push(`misaligned ${a.textContent} / ${b.textContent}`);
  }
 }
 for(const el of sidebar.querySelectorAll('.idx, .chapter-label, .group-title')) {
  if(!visible(el))continue;
  const range=document.createRange();range.selectNodeContents(el);
  for(const box of range.getClientRects()) {
   if(box.left < rect(el).left-1 || box.right > rect(el).right+1)failures.push(`clipped text ${el.textContent}`);
   if(box.left < rect(sidebar).left-1 || box.right > rect(sidebar).right+1)failures.push(`sidebar overflow ${el.textContent}`);
  }
  if(el.classList.contains('idx')) {
   const bg=background(el),fg=over(rgb(getComputedStyle(el).color),bg);
   const a=luminance(fg),b=luminance(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
   if(ratio<4.5)failures.push(`low contrast ${el.textContent}: ${ratio.toFixed(2)} fg=${fg} bg=${bg}`);
  }
 }
 if(sidebar.scrollWidth>sidebar.clientWidth)failures.push('horizontal sidebar scroll');
 return failures;
}"""


def check_geometry(page):
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
    assert not (failures := page.evaluate(GEOMETRY)), failures


def run(base):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        try:
            page.goto(base, wait_until='networkidle')
            expect(page.locator('.sidebar .leaf')).to_have_count(101)
            chapters = page.locator('.sidebar .overview, .sidebar .leaf').evaluate_all("""links => links.map(a=>({
              href:a.getAttribute('href'), number:a.querySelector('.idx').textContent,
            }))""")
            # All 105 leaf chapters + 6 overviews, selected/unselected numbering in both themes.
            for theme in ('light', 'dark'):
                page.evaluate('t=>document.documentElement.dataset.theme=t', theme)
                for chapter in chapters:
                    page.evaluate('href=>location.hash=href.split("#")[1]', chapter['href'])
                    expect(page.locator('.sidebar a[aria-current="page"] .idx')).to_have_text(chapter['number'])
                    page.wait_for_function("""() => {
                      const e=document.querySelector('.sidebar a[aria-current="page"]');
                      const b=e.getBoundingClientRect(),s=document.querySelector('#sidebar').getBoundingClientRect();
                      return b.top>=s.top-1 && b.bottom<=s.bottom+1;
                    }""")
                    check_geometry(page)
                print(f'PASS {theme}: {len(chapters)} routes, alignment, contrast, text bounds, active item in viewport', flush=True)
            # Narrowest / default / widest desktop sidebar, with all chapters and progress marks present.
            for theme in ('light', 'dark'):
                page.evaluate('t=>document.documentElement.dataset.theme=t', theme)
                for width in (200, 260, 480):
                    handle = page.get_by_role('separator')
                    handle.focus()
                    for _ in range(30): handle.press('ArrowLeft')
                    for _ in range((width-200+15)//16): handle.press('ArrowRight')
                    # Exact boundary/default values also go through the pointer resize handler.
                    x=handle.bounding_box()['x']+2
                    page.mouse.move(x,300);page.mouse.down();page.mouse.move(width,300);page.mouse.up()
                    expect(handle).to_have_attribute('aria-valuenow',str(width))
                    check_geometry(page)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            # Collapsing a theme removes its children from keyboard traversal; Enter/Space restore it.
            page.evaluate("location.hash='/agent/vector-index'")
            expect(page.locator('.leaf[aria-current="page"] .idx')).to_have_text('6.4.4')
            toggle=page.locator('[aria-controls="group-agent-knowledge-nav"]')
            toggle.focus();toggle.press('Enter')
            expect(page.locator('#group-agent-knowledge-nav')).to_be_hidden()
            toggle.press('Tab')
            assert page.evaluate("!document.activeElement.closest('#group-agent-knowledge-nav')")
            toggle.focus();toggle.press('Space')
            expect(toggle).to_have_attribute('aria-expanded','true')
            # Current extension is retained in core/sprint; numbering stays stable and next chapter agrees.
            for label in ('主干','冲刺','全部'):
                page.locator('.level-bar button').filter(has_text=label).click()
                expect(page.locator('.leaf[aria-current="page"] .idx')).to_have_text('6.4.4')
                check_geometry(page)
            # Collapse stage, then navigate inside it: both ancestors reopen.
            page.locator('[aria-controls="stage-agent-nav"]').click()
            page.locator('#main').focus();page.keyboard.press('ArrowRight')
            expect(page.locator('.leaf[aria-current="page"] .idx')).to_have_text('6.5.1')
            expect(page.locator('[aria-controls="stage-agent-nav"]')).to_have_attribute('aria-expanded','true')
            # Browser back and deep-link reload retain route, progress, and active group.
            page.go_back()
            expect(page.locator('.leaf[aria-current="page"] .idx')).to_have_text('6.4.4')
            page.reload(wait_until='networkidle')
            expect(page.locator('.leaf[aria-current="page"]')).to_be_visible()
            assert page.evaluate("JSON.parse(localStorage.getItem('llm-progress')).visited['agent-vector-index']")
            # Mobile drawer, theme states, keyboard closing, route focus and background click.
            for theme in ('light','dark'):
                page.evaluate('t=>document.documentElement.dataset.theme=t',theme)
                for width in (320,390,768):
                    page.set_viewport_size({'width':width,'height':844})
                    page.locator('.hamburger').click()
                    expect(page.locator('.hamburger')).to_have_attribute('aria-expanded','true')
                    check_geometry(page)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    page.keyboard.press('Escape')
                    expect(page.locator('.hamburger')).to_be_focused()
                    expect(page.locator('#sidebar')).to_have_attribute('inert','')
                    page.locator('.hamburger').click()
                    page.locator('#group-agent-knowledge-nav a').first.click()
                    expect(page.locator('.hamburger')).to_have_attribute('aria-expanded','false')
                    expect(page.locator('#main')).to_be_focused()
                    page.locator('.hamburger').click()
                    page.locator('.backdrop').click(position={'x':width-5,'y':400})
                    expect(page.locator('.hamburger')).to_have_attribute('aria-expanded','false')
                    expect(page.locator('.hamburger')).to_be_focused()
            # Overview theme cards and the fast-track list use the same hierarchy on narrow screens.
            page.set_viewport_size({'width':320,'height':844})
            page.evaluate("location.hash='/agent'")
            expect(page.locator('.stage-contents details')).to_have_count(8)
            summary=page.locator('.stage-contents summary').first
            summary.focus();summary.press('Enter')
            expect(page.locator('.stage-contents details').first).to_have_attribute('open','')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.locator('.stage-contents details').first.locator('a').first.click()
            expect(page.locator('.breadcrumb')).to_contain_text('6.1.1')
            page.evaluate("location.hash='/fast-track'")
            expect(page.locator('.path-topic')).to_have_count(26)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert not errors,errors
            print(json.dumps({'routes':len(chapters),'themes':2,'desktop_sidebar_widths':[200,260,480],'mobile_widths':[320,390,768],'js_errors':errors},ensure_ascii=False))
        finally:
            browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:5174/llm-engineering-lab/')
    run(parser.parse_args().base)
