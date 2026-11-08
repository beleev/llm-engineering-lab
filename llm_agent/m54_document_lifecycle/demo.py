"""HTML/表格 → 有出处的 chunk → 近似候选 → 权限与版本失效。"""
from __future__ import annotations
from dataclasses import dataclass, replace
from html.parser import HTMLParser
import hashlib
import json
import re

MODES=('prefilter','postfilter','update','revoke','delete','embedding')

@dataclass(frozen=True)
class Chunk:
    source: str
    version: int
    parent: str
    offset: int
    end: int
    text: str
    tenants: tuple[str,...]
    embedding: str

class Elements(HTMLParser):
    """保留段落和表格行，不把表格的每个数字拆成失去表头的孤立块。"""
    def __init__(self, raw):
        super().__init__(convert_charrefs=True)
        self.raw=raw; self.blocks=[]; self.active=None; self.parts=[]; self.start=0
        self.line_starts=[0]+[m.end() for m in re.finditer('\n',raw)]
    def handle_starttag(self, tag, attrs):
        if tag in ('p','tr','h1','h2'):
            if self.active: raise ValueError('不支持嵌套块')
            self.active=tag;self.parts=[]
            line,col=self.getpos();self.start=self.line_starts[line-1]+col
    def handle_data(self,data):
        if self.active:self.parts.append(data.strip())
    def handle_endtag(self,tag):
        if tag==self.active:
            line,col=self.getpos(); end=self.line_starts[line-1]+col+len(tag)+3
            text=' '.join(p for p in self.parts if p)
            if text:self.blocks.append((self.start,end,text))
            self.active=None

def partition(raw):
    if raw.lstrip().startswith('%PDF'): raise ValueError('PDF 需先用版面解析/OCR 适配器转成带页码元素；不能当 HTML')
    parser=Elements(raw);parser.feed(raw);parser.close()
    if parser.active or not parser.blocks:raise ValueError('没有完整可索引元素')
    return parser.blocks

def vector(text):
    # 可解释的两维固定编码器；分桶近似候选不是完整 HNSW。
    nums=re.findall(r'\d+',text)
    return (float(nums[-1]) if nums else 0., float('退款' in text))

def score(a,b):return sum((x-y)**2 for x,y in zip(a,b))

class Store:
    def __init__(self):
        self.documents={};self.chunks=[];self.cache={};self.epoch=0;self.embedding='numbers-v1'
    def invalidate(self):self.epoch+=1;self.cache.clear()
    def put(self,source,version,raw,tenants):
        blocks=partition(raw) # 解析失败不能破坏旧快照
        if source in self.documents and version<=self.documents[source]['version']:raise ValueError('文档版本必须递增')
        new=[Chunk(source,version,f'{source}:v{version}',a,b,text,tuple(tenants),self.embedding) for a,b,text in blocks]
        self.documents[source]={'version':version,'raw':raw,'tenants':list(tenants)}
        self.chunks=[c for c in self.chunks if c.source!=source]+new
        self.invalidate()
    def revoke(self,source,tenant):
        d=self.documents[source];d['tenants']=[t for t in d['tenants'] if t!=tenant]
        self.chunks=[replace(c,tenants=tuple(d['tenants'])) if c.source==source else c for c in self.chunks]
        self.invalidate()
    def delete(self,source):
        del self.documents[source];self.chunks=[c for c in self.chunks if c.source!=source];self.invalidate()
    def search(self,tenant,query,mode='prefilter',embedding='numbers-v1'):
        if mode not in ('prefilter','postfilter'):raise ValueError(mode)
        if embedding!=self.embedding or any(c.embedding!=embedding for c in self.chunks):raise ValueError('embedding 版本不兼容，需要重建')
        key=(self.epoch,tenant,query,mode,embedding)
        if key in self.cache:return self.cache[key]
        q=vector(query)
        eligible=[c for c in self.chunks if tenant in c.tenants]
        pool=eligible if mode=='prefilter' else self.chunks
        # IVF 风格：只探测与 query 数字落在同一区间的桶，然后保留 top-1。
        candidates=[c for c in pool if int(vector(c.text)[0]//20)==int(q[0]//20)]
        ranked=sorted(candidates,key=lambda c:score(vector(c.text),q))[:1]
        result=[c for c in ranked if tenant in c.tenants] # 后过滤仍不能把无权内容送入生成器
        self.cache[key]=result
        return result
    def cite(self,chunk,tenant):
        d=self.documents.get(chunk.source)
        if not d or d['version']!=chunk.version or tenant not in d['tenants']:raise ValueError('引用已过期或无权')
        raw=d['raw'][chunk.offset:chunk.end]
        if partition(raw)[0][2]!=chunk.text:raise ValueError('引用无法回到原文')
        return {'source':chunk.source,'version':chunk.version,'parent':chunk.parent,'offset':[chunk.offset,chunk.end],'raw':raw}
    def snapshot(self):
        return json.dumps({'embedding':self.embedding,'documents':self.documents},ensure_ascii=False,sort_keys=True)
    @classmethod
    def restore(cls,snapshot):
        data=json.loads(snapshot);s=cls();s.embedding=data['embedding']
        for source,d in data['documents'].items():s.put(source,d['version'],d['raw'],d['tenants'])
        return s

def fixture():
    s=Store()
    s.put('private',1,'<table><tr><td>退款天数</td><td>11</td></tr></table>',['B'])
    s.put('policy',1,'<h1>售后政策</h1><table><tr><td>退款天数</td><td>12</td></tr></table>',['A'])
    return s

def experiment(mode='prefilter'):
    if mode not in MODES:raise ValueError(mode)
    s=fixture();old=s.search('A','退款 11')[0];s.cite(old,'A')
    rejected=False;old_rejected=False
    if mode=='update':s.put('policy',2,'<table><tr><td>退款天数</td><td>15</td></tr></table>',['A'])
    if mode=='revoke':s.revoke('policy','A')
    if mode=='delete':s.delete('policy')
    if mode=='embedding':
        try:s.search('A','退款 11',embedding='numbers-v2')
        except ValueError:rejected=True
    try:s.cite(old,'A')
    except ValueError:old_rejected=True
    found=s.search('A','退款 11','postfilter' if mode=='postfilter' else 'prefilter')
    assert all('A' in c.tenants for c in found)
    if mode in ('update','revoke','delete'):assert old_rejected
    if mode in ('revoke','delete'):assert not found
    if mode=='postfilter':assert not found # 无权的近邻挤掉合法候选，安全但召回下降
    restored=Store.restore(s.snapshot())
    assert restored.search('A','退款 11')==s.search('A','退款 11')
    return {'metrics':{'返回片段':len(found),'越权返回':sum('A' not in c.tenants for c in found),
                       '旧引用失效':old_rejected,'embedding 拒绝':rejected,'当前文档版本':s.documents.get('policy',{}).get('version',0)},
            'steps':[f'先建立 A/B 两租户与表格原文，A 的旧答案为 {old.text}。',
                     f'执行 {mode}；结果为 '+('、'.join(c.text for c in found) or '空，不能使用旧缓存补答'),
                     '快照重新加载后查询一致；引用携带 source/version/parent/offset 并在读取时再次校验权限。'],
            'note':'两维固定编码器与单桶 ANN 用于隔离权限、召回和失效行为，不代表真实语义效果。HTML 保留字符偏移；PDF/扫描件必须由单独解析/OCR 适配器提供页码与坐标，本模块显式拒绝原始 PDF。缓存按租户与索引代际隔离，变更时保守全量失效。'}

def main():
    print(json.dumps({m:experiment(m) for m in MODES},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
