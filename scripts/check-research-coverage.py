"""验证长图条目无遗漏、证据路径/函数存在，网页条件与清单一致。"""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    audit=json.loads((ROOT/'docs/image-knowledge-audit.json').read_text())['items']
    closure=json.loads((ROOT/'docs/research-completion-coverage.json').read_text())['items']
    entries=json.loads((ROOT/'docs/research-completion-manifest.json').read_text())['entries']
    records=json.loads((ROOT/'web/src/data/research-completion.json').read_text())['entries']
    assert len({e['id'] for e in entries})==len(entries)
    assert len({e['id'] for e in closure})==len(closure)==45
    assert {e['id'] for e in audit}=={e['id'] for e in closure}
    old={e['id']:e for e in audit};by_id={e['id']:e for e in entries};replays={e['id']:e for e in records}
    for row in closure:
        assert row['original_gaps']==old[row['id']]['gaps'],row['id']
        assert row['protocol'] and row['evidence'],row['id']
        for path in row['evidence']:assert (ROOT/path).is_file(),path
        for key in row['new_entries']:assert row['id'] in by_id[key]['audit_ids'],(row['id'],key)
    for e in entries:
        path=ROOT/(e['module'].replace('.','/')+'.py')
        tree=ast.parse(path.read_text())
        assert any(isinstance(n,ast.FunctionDef) and n.name=='experiment' for n in tree.body)
        doc=path.with_name('README.md').read_text()
        assert e['challenge']['ask'] in doc and e['protocol'] in doc,e['id']
        assert e['intuition'] in doc and all(p in doc for p in e['principles']),e['id']
        replay=replays[e['id']]
        assert [m['id'] for m in e['modes']]==[c['id'] for c in replay['conditions']]
        for key in ('routes','intuition','principles','quiz','protocol'):assert e[key]==replay[key],(e['id'],key)
        for condition in replay['conditions']:
            section=doc.split('### '+condition['label']+'\n',1)[1].split('\n##',1)[0]
            for key,value in condition['metrics'].items():
                row='| '+key+' | '+json.dumps(value,ensure_ascii=False)+' |'
                assert row in section,(e['id'],condition['id'],key)
        assert set(e['audit_ids']) <= old.keys()
    print(f'PASS: {len(closure)} 原图主题 / {sum(len(r["original_gaps"]) for r in closure)} 条原差距逐项有去向；{len(entries)} 实验入口和文档一致')

if __name__=='__main__':main()
