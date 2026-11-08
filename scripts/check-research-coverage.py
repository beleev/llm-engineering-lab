"""验证研究主题索引、证据路径、函数与网页实验条件。"""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    topics=json.loads((ROOT/'docs/research-completion-coverage.json').read_text())['items']
    entries=json.loads((ROOT/'docs/research-completion-manifest.json').read_text())['entries']
    records=json.loads((ROOT/'web/src/data/research-completion.json').read_text())['entries']
    assert len({e['id'] for e in entries})==len(entries)
    assert len({e['id'] for e in topics})==len(topics)==45
    topic_ids={e['id'] for e in topics};by_id={e['id']:e for e in entries};replays={e['id']:e for e in records}
    for row in topics:
        assert row['scope'],row['id']
        assert row['protocol'] and row['evidence'],row['id']
        for path in row['evidence']:assert (ROOT/path).is_file(),path
        for key in row['experiments']:assert row['id'] in by_id[key]['topic_ids'],(row['id'],key)
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
        assert set(e['topic_ids']) <= topic_ids
    print(f'PASS: {len(topics)} 研究主题 / {sum(len(r["scope"]) for r in topics)} 项检查维度与边界；{len(entries)} 实验入口和文档一致')

if __name__=='__main__':main()
