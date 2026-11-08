"""确定性的服务队列代价模型：批次步耗时依赖活跃数，扫描负载并回收取消请求。"""
import json
import math
import random
from collections import deque


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q*len(values))-1)] if values else None


def simulate(rate=24, bounded=True, *,duration=10_000,seed=31,limit=8,deadline=1000):
    if rate<=0 or duration<=0 or limit<0 or deadline<=0:raise ValueError('参数越界')
    rng=random.Random(seed); arrivals=[]; at=0
    while True:
        at+=rng.expovariate(rate/1000)
        if at>=duration:break
        long=rng.random()<.25; i=len(arrivals)
        arrivals.append({'id':i,'arrival':at,'prefill':24 if long else 4,
                         'output':48 if long else 12,'emitted':0,'first':None,'last':None,
                         'itls':[],'cancel_after':100 if i%17==0 else None})
    queue=deque(); active=[]; completed=[]; terminal={}; now=0.; cursor=0
    peak_queue=peak_pages=0; page_ms=0.; token_count=0
    def stop(request,reason):
        if request['id'] in terminal:raise AssertionError('重复结束')
        terminal[request['id']]=reason
    while cursor<len(arrivals) or queue or active:
        if not queue and not active and cursor<len(arrivals):now=max(now,arrivals[cursor]['arrival'])
        while cursor<len(arrivals) and arrivals[cursor]['arrival']<=now:
            request=arrivals[cursor]; cursor+=1
            if bounded and len(queue)+len(active)>=4+limit:stop(request,'rejected')
            else:queue.append(request)
        keep=deque()
        for request in list(queue)+active:
            age=now-request['arrival']
            if request['cancel_after'] is not None and age>=request['cancel_after']:stop(request,'cancelled')
            elif age>=deadline:stop(request,'timeout')
            else:keep.append(request)
        active=[r for r in keep if r in active]
        queue=deque(r for r in keep if r not in active)
        while queue and len(active)<4:active.append(queue.popleft())
        peak_queue=max(peak_queue,len(queue))
        pages=sum(2+math.ceil(r['emitted']/16) for r in active)
        peak_pages=max(peak_pages,pages)
        if not active:continue
        dt=2+.5*len(active)  # 虚拟毫秒，不是某款 GPU 的标定结果。
        page_ms+=pages*dt; now+=dt
        for request in list(active):
            if request['prefill']>0:request['prefill']-=1; continue
            request['emitted']+=1; token_count+=1
            if request['first'] is None:request['first']=now
            else:request['itls'].append(now-request['last'])
            request['last']=now
            if request['emitted']==request['output']:
                stop(request,'completed'); completed.append(request); active.remove(request)
    horizon=max(duration,now)
    ttfts=[r['first']-r['arrival'] for r in completed]
    itls=[v for r in completed for v in r['itls']]
    good=[r for r in completed if r['first']-r['arrival']<=200 and max(r['itls'],default=0)<=6 and r['last']-r['arrival']<=1000]
    counts={key:sum(v==key for v in terminal.values()) for key in ('completed','rejected','timeout','cancelled')}
    assert sum(counts.values())==len(arrivals) and not active and not queue
    return {'arrivals':len(arrivals),**counts,'horizon_ms':round(horizon,2),
            'throughput_rps':len(completed)*1000/horizon,'goodput_rps':len(good)*1000/horizon,
            'output_tokens_per_second':token_count*1000/horizon,
            'ttft_p95_ms':percentile(ttfts,.95),'ttft_p99_ms':percentile(ttfts,.99),
            'itl_p95_ms':percentile(itls,.95),'itl_p99_ms':percentile(itls,.99),
            'peak_queue':peak_queue,'peak_pages':peak_pages,'average_pages':page_ms/horizon,'final_pages':0}


def experiment(mode='24-bounded'):
    rate, policy=mode.split('-')
    if policy not in ('bounded','unbounded'):raise ValueError(mode)
    r=simulate(int(rate),policy=='bounded')
    rounded=lambda x:round(x,2) if x is not None else '无完成样本'
    return {'metrics':{'完成请求/s':rounded(r['throughput_rps']),'满足 SLO 请求/s':rounded(r['goodput_rps']),
                       'TTFT p95 / ms':rounded(r['ttft_p95_ms']),'TTFT p99 / ms':rounded(r['ttft_p99_ms']),
                       'ITL p99 / ms':rounded(r['itl_p99_ms']),'拒绝请求':r['rejected'],
                       '超时请求':r['timeout'],'取消请求':r['cancelled'],'队列峰值':r['peak_queue'],
                       'KV 页峰值':r['peak_pages'],'结束残留页':r['final_pages']},
            'steps':[f"到达 {r['arrivals']} = 完成 {r['completed']} + 拒绝 {r['rejected']} + 超时 {r['timeout']} + 取消 {r['cancelled']}",
                     f"观测含排空期 {r['horizon_ms']} ms；输出 token/s={r['output_tokens_per_second']:.2f}",
                     f"平均占页 {r['average_pages']:.2f}；完成样本 ITL p95={r['itl_p95_ms']} ms"],
            'note':'seed=31，25% 长请求；4 活跃槽；有界队列最多 8；1 秒截止，每 17 个请求有 1 个在 100 ms 取消。分位数只统计完成请求，必须同时看拒绝/超时。goodput 要求 TTFT≤200 ms、ITL≤6 ms、完成≤1秒。全部为代价模型。'}


def main():
    results={f'{r}-{p}':experiment(f'{r}-{p}') for r in (8,24,48,96) for p in ('bounded','unbounded')}
    assert simulate(96)['peak_queue']<=8
    assert simulate(96)['rejected']>0
    assert simulate(96,False)['timeout']>0
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
