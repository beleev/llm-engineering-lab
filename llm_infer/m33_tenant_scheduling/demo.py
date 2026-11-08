"""固定到达轨迹的租户调度实验。tick 是服务量单位，不是假造的 GPU 毫秒。"""
from collections import Counter, deque
import json
import math

MODES = ('fifo', 'priority', 'fair', 'quota')


def simulate(mode):
    if mode not in MODES:
        raise ValueError(mode)
    arrivals = [dict(id=i, tenant='A', arrival=i, work=5, priority=0) for i in range(60)]
    arrivals += [dict(id=60+i, tenant='B', arrival=5+15*i, work=2, priority=1) for i in range(4)]
    arrivals.sort(key=lambda r: (r['arrival'], r['id']))
    queues = {t: deque() for t in ('A', 'B')}
    active = []
    terminals = {}
    cursor = 0
    now = 0
    turn = deque(['A', 'B'])
    served = Counter()
    first_wait = {'A': [], 'B': []}
    latency = {'A': [], 'B': []}
    peak = Counter()
    while cursor < len(arrivals) or any(queues.values()) or active:
        while cursor < len(arrivals) and arrivals[cursor]['arrival'] <= now:
            r = {**arrivals[cursor], 'remaining': arrivals[cursor]['work'], 'first': None}
            cursor += 1
            if mode == 'quota' and len(queues[r['tenant']]) >= 8:
                terminals[r['id']] = ('rejected', r['tenant'])
            else:
                queues[r['tenant']].append(r)
        # 排队和运行中的请求共用 end-to-end deadline。
        for tenant, queue in queues.items():
            keep = deque()
            for r in queue:
                if now-r['arrival'] >= 40:
                    terminals[r['id']] = ('timeout', tenant)
                else:
                    keep.append(r)
            queues[tenant] = keep
            peak[tenant] = max(peak[tenant], len(keep))
        if active and now-active[0]['arrival'] >= 40:
            r = active.pop()
            terminals[r['id']] = ('timeout', r['tenant'])
        if mode in ('fair', 'quota'):
            # 每租户轮流获得一个等成本量子；未完成请求放回本租户队首。
            for _ in range(2):
                tenant = turn[0]
                turn.rotate(-1)
                if queues[tenant]:
                    active = [queues[tenant].popleft()]
                    break
        elif not active:
            heads = [q[0] for q in queues.values() if q]
            if heads:
                key = (lambda r: (r['priority'], r['arrival'])) if mode == 'priority' else (lambda r: r['arrival'])
                r = min(heads, key=key)
                active = [queues[r['tenant']].popleft()]
        if active:
            r = active.pop()
            if r['first'] is None:
                r['first'] = now
                first_wait[r['tenant']].append(now-r['arrival'])
            served[r['tenant']] += 1
            r['remaining'] -= 1
            if r['remaining'] == 0:
                terminals[r['id']] = ('completed', r['tenant'])
                latency[r['tenant']].append(now+1-r['arrival'])
            elif mode in ('fair', 'quota'):
                queues[r['tenant']].appendleft(r)
            else:
                active = [r]
        now += 1
    assert len(terminals) == len(arrivals)
    by_tenant = {}
    for tenant in queues:
        counts = Counter(s for s, t in terminals.values() if t == tenant)
        by_tenant[tenant] = {k: counts[k] for k in ('completed', 'timeout', 'rejected')}
        by_tenant[tenant].update(service_units=served[tenant], peak_queue=peak[tenant],
                                max_first_wait=max(first_wait[tenant], default=None),
                                p95_first_wait=(sorted(first_wait[tenant])[math.ceil(.95*len(first_wait[tenant]))-1] if first_wait[tenant] else None),
                                goodput=sum(t <= 10 for t in latency[tenant])/now,
                                latencies=latency[tenant])
    return {'tenants': by_tenant, 'horizon': now, 'arrivals': len(arrivals), 'terminal_count': len(terminals)}


def experiment(mode='fair'):
    row = simulate(mode)
    metrics = {'观测时长 / tick': row['horizon'], '请求终态数': row['terminal_count']}
    for t, r in row['tenants'].items():
        metrics.update({t+' 完成/超时/拒绝': [r[k] for k in ('completed','timeout','rejected')],
                        t+' 最大首服务等待': r['max_first_wait'], t+' 首服务等待 p95': r['p95_first_wait'], t+' goodput / tick': round(r['goodput'],4),
                        t+' 队列峰值': r['peak_queue']})
    return {'metrics': metrics,
            'steps': ['固定 A 连续长请求与 B 间歇短请求；所有策略使用同一到达轨迹。',
                      '分别记录租户的完成、超时和拒绝；goodput 要求端到端延迟不超过 10 tick。',
                      '公平模式按等成本服务量子轮转，配额模式另限制各租户排队数量。'],
            'note': '离散服务量模拟，不含真实 GPU、KV 或 token 预测。首次获得服务的样本才有等待读数，需连同从未获服务的超时/拒绝一起读；有限轨迹不能证明任意负载下无饥饿。'}


def main():
    for mode in MODES:
        print(json.dumps({'mode': mode, **experiment(mode)}, ensure_ascii=False))
    assert simulate('fair')['tenants']['B']['completed'] == 4
    assert simulate('priority')['tenants']['B']['completed'] < 4
    assert simulate('quota')['tenants']['A']['peak_queue'] <= 8


if __name__ == '__main__':
    main()
