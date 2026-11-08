"""检查知识覆盖索引和可选书版本, 防止新增内容没有入口或旧映射悄悄漂移。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check(book=None):
    manifest = json.loads((ROOT / 'docs/agent-book-coverage.json').read_text())
    entries = manifest['entries']
    ids = [row['id'] for row in entries]
    assert len(ids) == len(set(ids))
    for row in entries:
        assert row['evidence'] and row['level']
        for ref in row['evidence']:
            path = ROOT / ref
            assert path.is_file(), f"缺少覆盖入口: {row['id']} → {ref}"
    if book:
        found = []
        for source in manifest['sources']:
            chapter = source['chapter']
            path = Path(book) / f'book/chapter{chapter}.md'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], f'第 {chapter} 章变化, 需要重新做知识 diff'
            fence = False
            for line, text in enumerate(path.read_text().splitlines(), 1):
                if text.startswith('```'):
                    fence = not fence
                    continue
                heading = re.match(r'^(#{2,4}) (.+)$', text)
                if not fence and heading and heading.group(2) not in ['本章小结', '思考题']:
                    found.append((f'ch{chapter}-L{line}', heading.group(2)))
        assert found == [(row['id'], row['title']) for row in entries], '存在未映射或误匹配的知识标题'
    print(f'{len(entries)} 个知识标题, 10 章, 本地入口完整' + ('; 源文与标题全集一致' if book else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book', type=Path)
    check(parser.parse_args().book)
