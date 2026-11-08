"""执行工程流程实验并导出网页证据；--check 核对源码、教学清单与实际结果。"""
from pathlib import Path
import hashlib
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'docs/engineering-completion-manifest.json'
OUT = ROOT / 'web/src/data/engineering-completion.json'


def digest():
    files = sorted(p for pkg in ('llm_agent', 'llm_finetune', 'llm_models', 'llm_train', 'llm_infer')
                   for p in (ROOT / pkg).rglob('*.py'))
    h = hashlib.sha256()
    for path in files + [MANIFEST, Path(__file__).resolve()]:
        h.update(str(path.relative_to(ROOT)).encode()); h.update(b'\0')
        h.update(path.read_bytes()); h.update(b'\0')
    return h.hexdigest()


def export():
    manifest = json.loads(MANIFEST.read_text())
    entries = []
    for item in manifest['entries']:
        module = importlib.import_module(item['module'])
        assert tuple(m['id'] for m in item['modes']) == module.MODES, item['id']
        conditions = [{**mode, **module.experiment(mode['id'])} for mode in item['modes']]
        entries.append({**item, 'module': item['module'].replace('.', '/') + '.py',
                        'run': 'python -m ' + item['module'], 'conditions': conditions})
    result = {'source_digest': digest(),
              'kind': 'Python 实际执行记录回放。切换条件读取预先计算的结果，浏览器不训练模型或调用 API。',
              'entries': entries}
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if '--check' in sys.argv:
        assert OUT.read_text() == encoded, '源码、清单或执行结果变化，请重新导出'
    else:
        OUT.write_text(encoded)
    print(f'{len(entries)} 组 / {sum(len(e["conditions"]) for e in entries)} 个条件：' + ('一致' if '--check' in sys.argv else '已导出'))


if __name__ == '__main__':
    export()
