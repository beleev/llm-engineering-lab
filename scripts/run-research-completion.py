"""一次运行长图补充的全部 CPU 实验；不访问模型 API。"""
from pathlib import Path
import importlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    entries = json.loads((ROOT / 'docs/research-completion-manifest.json').read_text())['entries']
    for entry in entries:
        print('运行：' + entry['module'], flush=True)
        importlib.import_module(entry['module']).main()
    print(f'全部 {len(entries)} 组实验通过')

if __name__ == '__main__':
    main()
