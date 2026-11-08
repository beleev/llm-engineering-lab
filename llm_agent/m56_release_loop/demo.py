"""离线切片门槛 → 小流量回放 → 监控回滚 → 隔离的失败回归集。"""
from dataclasses import dataclass
import hashlib
import json
from llm_agent.m13_evals.diagnostics import paired_interval

MODES=('slice','quality','latency','cost','safety','healthy')

@dataclass(frozen=True)
class Version:
    model: str
    prompt: str
    index: str
    tools: str
    @property
    def id(self):return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()[:12]

class Release:
    def __init__(self, old, candidate):
        self.old=old;self.candidate=candidate;self.active=old;self.events=[];self.approved=False
    def record(self,event,**data):self.events.append({'event':event,'active':self.active.id,**data})
    def offline(self,before,after,critical):
        delta=sum(after)/len(after)-sum(before)/len(before)
        lo,hi=paired_interval(before,after)
        slice_ok=sum(after[i] for i in critical)>=sum(before[i] for i in critical)
        allowed=delta>=0 and lo>=-.05 and slice_ok
        self.approved=allowed
        self.record('offline',delta=delta,interval=[lo,hi],slice_ok=slice_ok,allowed=allowed)
        return allowed
    def deploy(self):
        if not self.approved:raise ValueError('候选尚未通过离线门槛')
        self.active=self.candidate;self.record('canary',traffic_fraction=.1)
    def monitor(self,rows,now):
        if not rows:raise ValueError('空监控窗口不能判为成功')
        available=[r for r in rows if r['feedback_at'] is not None and r['feedback_at']<=now]
        missing=sum(r['feedback_at'] is None for r in rows)
        pending=len(rows)-len(available)-missing
        # 质量分母只包含已到达反馈，另设覆盖率门槛，绝不把缺失当成功。
        quality=sum(r['ok'] for r in available)/len(available) if available else None
        latency=sorted(r['latency'] for r in rows)[max(0, int(.95*len(rows)+.999)-1)]
        cost=sum(r['cost'] for r in rows)/len(rows)
        safety=sum(r['unsafe'] for r in rows)
        bad=(quality is not None and quality<.8) or latency>250 or cost>.02 or safety>0
        status='rollback' if bad else 'hold' if len(available)/len(rows)<.8 else 'promote'
        if bad:self.active=self.old;self.approved=False
        self.record(status,quality=quality,p95=latency,cost=cost,safety=safety,pending=pending,missing=missing)
        return status

def feedback_replay(rows, protected, now=2):
    seen=set(protected);out=[]
    for row in rows:
        if row['feedback_at'] is None or row['feedback_at'] > now:continue
        digest=hashlib.sha256(row['input'].strip().encode()).hexdigest()
        if row['ok'] or digest in seen:continue
        seen.add(digest);out.append({'input':row['input'],'split':'regression','source':'production','hash':digest})
    return out

def experiment(mode='quality'):
    if mode not in MODES:raise ValueError(mode)
    old=Version('m1','p1','i1','t1');candidate=Version('m1','p2','i2','t1')
    release=Release(old,candidate)
    before=[True]*8+[False]*12
    after=([False]*2+[True]*18) if mode=='slice' else [True]*18+[False]*2
    allowed=release.offline(before,after,[0,1])
    rows=[{'input':f'query-{i}','ok':True,'latency':120,'cost':.01,'unsafe':False,'feedback_at':1} for i in range(10)]
    rows[-1]['feedback_at']=None;rows[-2]['feedback_at']=3
    if mode=='quality':
        for row in rows[:3]:row['ok']=False
    if mode=='latency':rows[0]['latency']=500
    if mode=='cost':rows[0]['cost']=.2
    if mode=='safety':rows[0]['unsafe']=True
    status='offline-reject'
    if allowed:
        release.deploy();release.monitor(rows,0);status=release.monitor(rows,2)
    protected={hashlib.sha256(b'query-0').hexdigest()}
    regression=feedback_replay(rows+rows,protected)
    if mode=='slice':assert not allowed
    elif mode=='healthy':assert status=='promote'
    else:assert status=='rollback' and release.active==old
    assert len({r['hash'] for r in regression})==len(regression)
    assert not {r['hash'] for r in regression}&protected
    return {'metrics':{'离线通过':allowed,'线上回滚':status=='rollback','活动版本':release.active.id,
                       '失败回归样本':len(regression),'缺失反馈':1,'待到达反馈':1},
            'steps':[json.dumps(event,ensure_ascii=False,sort_keys=True) for event in release.events],
            'note':'固定请求回放模拟 10% 灰度，没有操作真实流量。复用配对 bootstrap；平均提升不能抵消关键切片退化。回归样本按规范化文本去重并排除受保护集合，仍需实体级去重和隐私审核；延迟反馈与缺失分别计数。'}

def main():
    print(json.dumps({m:experiment(m) for m in MODES},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
