"""验收工程实验的主讲入口、文档、引用和反例，不用关键字命中充当完成。"""
from pathlib import Path
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'docs/engineering-completion-manifest.json').read_text())['entries']
    records=json.loads((ROOT/'web/src/data/engineering-completion.json').read_text())['entries']
    js="""
const stages=['basic','models','train','finetune','infer','agent'];
const pages={}, quizzes={},labs={};
for(const s of stages){Object.assign(pages,(await import('./web/src/data/topics/'+s+'.js')).default.pages);Object.assign(quizzes,(await import('./web/src/data/quiz/'+s+'.js')).default)}
for(const s of [...stages,'research','engineering'])for(const [r,n] of Object.entries((await import('./web/src/data/labmap/'+s+'.js')).default))labs[r]=[...(labs[r]||[]),...n];
console.log(JSON.stringify({pages,quizzes,labs}));
"""
    web=json.loads(subprocess.check_output(['node','--input-type=module','-e',js],cwd=ROOT,text=True))
    assert {g for e in manifest for g in e['topic_ids']}=={f'G{i}' for i in range(1,7)} | {f'K{i}' for i in range(1,4)}
    assert len(manifest)==len(records)==9
    replay_routes=set()
    for e,r in zip(manifest,records):
        assert len(e['routes'])==1
        assert all(e[k]==r[k] for k in ('id','routes','principles','protocol','quiz'))
        route=e['routes'][0];replay_routes.add(route)
        assert 'EngineeringCompletionLab' in web['labs'][route]
        assert e['quiz'] in web['quizzes'][route]
        path=ROOT/r['module'];assert path.exists()
        doc=path.with_name('README.md').read_text()
        assert e['protocol'] in doc and all(p in doc for p in e['principles'])
        for c in r['conditions']:
            for k,v in c['metrics'].items():assert '| '+k+' | '+json.dumps(v,ensure_ascii=False)+' |' in doc
    actual={r for r,labs in web['labs'].items() if 'EngineeringCompletionLab' in labs}
    assert replay_routes==actual
    knowledge=json.loads((ROOT/'web/src/data/knowledge-completion.json').read_text())['entries']
    assert all(len(e['routes'])==1 for e in knowledge)
    for e in knowledge:
        assert 'KnowledgeCompletionLab' in web['labs'][e['routes'][0]]
    assert {r for r,labs in web['labs'].items() if 'KnowledgeCompletionLab' in labs}=={e['routes'][0] for e in knowledge}
    # 所有上下文跳转都指向真实章节；attention 是手写页。
    for route,page in web['pages'].items():
        for ref in page.get('related',[]):assert ref['route'] in web['pages'] or ref['route']=='attention',(route,ref)
    for component,owner in [('SoftmaxTempLab','basic-optim-sample'),('AttnMaskLab',None)]:
        placements=[r for r,p in web['pages'].items() if component in p.get('widgets',[])]
        placements += [r for r,ls in web['labs'].items() if component in ls]
        assert placements==([owner] if owner else []),(component,placements)
    counts={p['title']:[] for p in web['pages']['train-model-parallel']['points']}
    assert not any(p['title']=='多维配置要一起算' for p in web['pages']['train-collectives-loop']['points'])
    print(f'PASS: G1–G6 + K1–K3 / {len(records)} 组 / {sum(len(e["conditions"]) for e in records)} 条件；主入口、文档、自测与关联链接一致')

if __name__=='__main__':main()
