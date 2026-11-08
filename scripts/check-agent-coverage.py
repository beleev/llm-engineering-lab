"""检查 Agent 实验的文档、源码符号与网页回放入口。"""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]


def check():
    manifest = json.loads((ROOT / 'docs/agent-coverage-manifest.json').read_text())
    records = json.loads((ROOT / 'web/src/data/agent-coverage-evidence.json').read_text())
    assert len({row['route'] for row in manifest}) == len(manifest)
    assert set(records) == {row['route'] for row in manifest}
    for row in manifest:
        module = ROOT / 'llm_agent' / row['module']
        assert (module / 'README.md').is_file(), row['module']
        names = {node.name for node in ast.parse((module / 'demo.py').read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        assert set(row['symbols']) <= names, row['module']
        assert row['modes'] == [case['mode'] for case in records[row['route']]['cases']]
    print(f'PASS: {len(manifest)} 个 Agent 专题的文档、源码符号与回放入口一致')


if __name__ == '__main__':
    check()
