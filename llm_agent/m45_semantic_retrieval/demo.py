"""可训练的双编码器与联合重排。有限词表/任务，CPU 离线运行，不是通用 embedding。"""
from functools import lru_cache
import json
import torch
from torch import nn
from torch.nn import functional as F

DOCS = ['enable email', 'disable email', 'enable alerts', 'disable alerts',
        'enable backup', 'disable backup', 'ticket ZX91', 'ticket ZX92']
ALIASES = [('enable', '开启', 'activate'), ('disable', '关闭', 'deactivate')]
OBJECTS = [('email', '邮箱'), ('alerts', '通知'), ('backup', '备份')]
VOCAB = sorted(set(' '.join(DOCS[:6]).split()+['ticket','<unk>','please','today','help','不要']+
               [w for group in ALIASES+OBJECTS for w in group]))
WORD = {w:i for i,w in enumerate(VOCAB)}

def bow(text):
    ids = [WORD.get(w, WORD['<unk>']) for w in text.split()]
    return F.one_hot(torch.tensor(ids), len(WORD)).float().mean(0)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(len(WORD), 24, bias=False)
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

class Reranker(nn.Module):
    """查询/文档联合输入，并显式允许跨特征交互；不是词面加权规则。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*len(WORD), 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, q, d):
        return self.net(torch.cat((q,d),dim=-1)).squeeze(-1)

@lru_cache(maxsize=1)
def run():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    train=[]
    for o, names in enumerate(OBJECTS):
        for a, verbs in enumerate(ALIASES):
            for verb in verbs:
                for name in names:
                    train.append((f'please {verb} {name}', 2*o+a))
    # 联合重排还看到否定样本；双编码器不看它们，故可分离训练覆盖与 ANN 误差。
    negative=[(f'不要 {v} {n}', 2*o+(1-a)) for o,names in enumerate(OBJECTS)
              for a,verbs in enumerate(ALIASES) for v in verbs for n in names]
    d=torch.stack([bow(s) for s in DOCS])
    q=torch.stack([bow(s) for s,_ in train]); y=torch.tensor([i for _,i in train])
    encoder=Encoder(); optimizer=torch.optim.Adam(encoder.parameters(),lr=.03)
    initial=encoder.projection.weight.detach().clone()
    for _ in range(160):
        loss=F.cross_entropy(encoder(q)@encoder(d[:6]).T/.15,y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    ranker=Reranker(); opt=torch.optim.Adam(ranker.parameters(),lr=.02)
    pairs=train+negative
    rq=torch.stack([bow(s) for s,_ in pairs]); ry=torch.tensor([i for _,i in pairs])
    for _ in range(240):
        scores=ranker(rq[:,None,:].expand(-1,6,-1),d[None,:6,:].expand(len(pairs),-1,-1))
        loss=F.cross_entropy(scores,ry)
        opt.zero_grad(); loss.backward(); opt.step()
    probes=[('同义改写','today activate email',0),('跨语言','today 关闭 备份',5),
            ('否定','不要 开启 邮箱',1),('罕见编号','ticket ZX92',7)]
    rows=[]
    with torch.no_grad():
        embeddings=encoder(d)
        for category,query,target in probes:
            vector=encoder(bow(query)); scores=embeddings@vector
            exact=sorted(range(len(DOCS)),key=lambda i:(-float(scores[i]),i))
            # 固定两分区、只探一个分区，模拟 ANN 剪枝；不是 HNSW 实现。
            groups=[[0,2,4,6],[1,3,5,7]]
            chosen=max(groups,key=lambda g:float(embeddings[g].mean(0)@vector))
            ann=sorted(chosen,key=lambda i:(-float(scores[i]),i))
            candidates=exact[:4]
            lexical=[i for i,doc in enumerate(DOCS) if any(w.startswith('ZX') and w in doc.split() for w in query.split())]
            union=sorted(set(candidates+lexical))
            rerank=sorted(candidates,key=lambda i:-float(ranker(bow(query),d[i])))
            # 稀有编号的精确约束单列，不伪装成神经重排学会了编号。
            final=lexical[0] if lexical else rerank[0]
            rows.append({'case':category,'query':query,'expected':target,'exact':exact[0],
                         'ann':ann[0],'ann_kept_exact':exact[0] in chosen,
                         'candidate_recall':target in candidates,'rerank':rerank[0],'wide':max(range(len(DOCS)),key=lambda i:float(ranker(bow(query),d[i]))),
                         'hybrid':final,'candidate_count':len(union)})
    return {'rows':rows,'parameter_delta':round(float((encoder.projection.weight.detach()-initial).abs().sum()),4),
            'train_queries':len(train),'rerank_queries':len(pairs),'test_queries':len(probes)}

def experiment(mode='exact'):
    result=run()
    if mode not in ('exact','ann','rerank','wide','hybrid'):raise ValueError(mode)
    return {'metrics':{'正确数':sum(r[mode]==r['expected'] for r in result['rows']), '测试题数':4,
                       '编码器参数变化':result['parameter_delta'],
                       '候选召回数':sum(r['candidate_recall'] for r in result['rows'])},
            'steps':[f"{r['case']}: 期望 {DOCS[r['expected']]}；得到 {DOCS[r[mode]]}；top-4 含答案={r['candidate_recall']}；ANN 保留精确赢家={r['ann_kept_exact']}" for r in result['rows']],
            'note':'固定 seed=23；有限词表的真实参数训练。否定题参与重排训练，是机制探针；改写题只留出前缀组合。ANN 是两分区剪枝，hybrid 对编号使用显式精确约束。'}

def main():
    result=run()
    assert result['parameter_delta']>0
    assert all(r['exact']==r['expected'] for r in result['rows'][:2])
    assert result['rows'][-1]['hybrid']==7
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
