"""一个小 LLaMA 的交付包：配置、权重、tokenizer、模板与新进程一致性。"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from llm_models.models.language_models.llama import LLaMA

MODES = ('float', 'int8', 'tokenizer', 'template', 'base')
CONFIG = dict(vocab_size=8, d_model=16, n_heads=2, num_kv_heads=1, num_layers=1, max_len=32)
TOKENIZER = {'tokens': ['<pad>', '<bos>', '<user>', '<assistant>', 'a', 'b', 'c', '<eos>'], 'bos': 1, 'eos': 7}
TEMPLATE = {'prefix': ['<bos>', '<user>'], 'suffix': ['<assistant>']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(text, tokenizer, template):
    vocab = {token: i for i, token in enumerate(tokenizer['tokens'])}
    return [vocab[t] for t in template['prefix'] + list(text) + template['suffix']]


def infer(model, package):
    ids = encode('ab', package['tokenizer'], package['template'])
    x = torch.tensor([ids])
    model.eval()
    with torch.no_grad():
        logits = model(x)
        for _ in range(package['generation']['max_new_tokens']):
            token = model(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, token], dim=1)
            if token.item() == package['tokenizer']['eos']:
                break
    return {'ids': ids, 'logits': logits.tolist(), 'generated': x[0, len(ids):].tolist()}


def export(path, quantized=False):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = LLaMA(**CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    package = {'format': 1, 'config': CONFIG, 'tokenizer': copy.deepcopy(TOKENIZER),
               'template': copy.deepcopy(TEMPLATE), 'generation': {'max_new_tokens': 2, 'do_sample': False},
               'base_id': 'tiny-llama-seed21', 'adapter': None,
               'quantization': 'symmetric-int8-per-tensor' if quantized else 'float32'}
    x = torch.tensor([encode(t, TOKENIZER, TEMPLATE) for t in ('ab', 'bc', 'ca')])
    y = torch.tensor([4, 5, 6])
    for _ in range(25):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x)[:, -1], y).backward()
        optimizer.step()
    before = infer(model, package)
    state = model.state_dict()
    if quantized:
        state = {k: {'q': (v / max(float(v.abs().max()) / 127, 1e-8)).round().clamp(-127,127).to(torch.int8),
                     'scale': max(float(v.abs().max()) / 127, 1e-8)} for k,v in state.items()}
    torch.save(state, path / 'weights.pt')
    (path / 'package.json').write_text(json.dumps(package))
    manifest = {'files': {p: sha(path / p) for p in ('weights.pt','package.json')},
                'contract': copy.deepcopy(package)}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return before


def load(path):
    """manifest 是可信交付约定；哈希校验完整性，不提供签名或来源认证。"""
    manifest = json.loads((path / 'manifest.json').read_text())
    if set(manifest.get('files',{})) != {'weights.pt','package.json'}:
        raise ValueError('交付文件清单不完整')
    for name, digest in manifest['files'].items():
        if name not in ('weights.pt','package.json') or sha(path / name) != digest:
            raise ValueError('文件摘要不一致')
    p = json.loads((path / 'package.json').read_text())
    if p != manifest['contract']:
        raise ValueError('配置与交付约定不一致')
    if p['format'] != 1 or p['base_id'] != 'tiny-llama-seed21':
        raise ValueError('不支持的格式或基座')
    if p['config']['vocab_size'] != len(p['tokenizer']['tokens']):
        raise ValueError('词表大小不一致')
    if p['generation']['do_sample']:
        raise ValueError('本验收只接受 greedy')
    state = torch.load(path / 'weights.pt', map_location='cpu', weights_only=True)
    if p['quantization'] == 'symmetric-int8-per-tensor':
        state = {k: v['q'].float() * v['scale'] for k,v in state.items()}
    elif p['quantization'] != 'float32':
        raise ValueError('未知量化格式')
    model = LLaMA(**p['config'])
    model.load_state_dict(state, strict=True)
    return model, p


def experiment(mode='float'):
    if mode not in MODES: raise ValueError(mode)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        before = export(path, mode == 'int8')
        if mode in ('tokenizer','template','base'):
            p = json.loads((path / 'package.json').read_text())
            if mode == 'tokenizer': p['tokenizer']['tokens'][4:6] = ['b','a']
            if mode == 'template': p['template']['suffix'] = ['<user>']
            if mode == 'base': p['base_id'] = 'another-base'
            (path / 'package.json').write_text(json.dumps(p))
            # 更新文件哈希仍无法绕过语义约定；不是只检测损坏文件。
            m = json.loads((path / 'manifest.json').read_text())
            m['files']['package.json'] = sha(path / 'package.json')
            (path / 'manifest.json').write_text(json.dumps(m))
        child = subprocess.run([sys.executable, '-m', __name__ if __name__ != '__main__' else
                                'llm_finetune.run_finetune.artifact_delivery.demo', '--load', str(path)],
                               capture_output=True, text=True, timeout=30)
        rejected = child.returncode != 0
        if rejected:
            assert mode in ('tokenizer','template','base'), child.stderr
            assert '交付约定' in child.stderr, child.stderr
            error, same_ids, same_output = 0., False, False
        else:
            after = json.loads(child.stdout)
            error = float((torch.tensor(before['logits'])-torch.tensor(after['logits'])).abs().max())
            same_ids, same_output = before['ids']==after['ids'], before['generated']==after['generated']
            assert same_ids and same_output
            assert error < (.15 if mode=='int8' else 1e-6)
        return {'metrics': {'加载拒绝':rejected,'token 一致':same_ids,'greedy 一致':same_output,'logits 最大误差':round(error,6)},
                'steps':['训练 25 步后导出权重和完整配置。','独立 Python 进程校验文件与语义约定，再执行同一输入。',
                         '拒绝：'+mode if rejected else '通过：token、logits 与 greedy 输出对照。'],
                'note':'拒绝时误差显示 0 代表未运行推理；int8 是反量化后的误差实验，没有整数算子加速。manifest 需由可信渠道提供；签名、真实 adapter 基座指纹与跨引擎容差属于部署扩展。'}


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--load':
        torch.set_num_threads(1)
        model, package = load(Path(sys.argv[2])); print(json.dumps(infer(model, package)))
    else: print(json.dumps({m:experiment(m) for m in MODES},ensure_ascii=False,indent=2))
