"""有界 rollout 队列与不可变策略快照；有限动作空间精确检验重要性校正。"""
from __future__ import annotations
from dataclasses import dataclass
import math
import queue
import threading

MODES = ('uncorrected','corrected','clipped','stale')

@dataclass(frozen=True)
class Policy:
    version: int
    probabilities: tuple[float, ...]

    def __post_init__(self):
        if self.version < 0 or any(p <= 0 or not math.isfinite(p) for p in self.probabilities) or not math.isclose(sum(self.probabilities), 1.):
            raise ValueError('策略必须有合法版本及归一化的正概率')

@dataclass(frozen=True)
class Rollout:
    task: str
    policy: Policy
    action: int
    behavior_logp: float

class Worker:
    def __init__(self, policy, capacity=4):
        self.policy = policy
        self.queue = queue.Queue(capacity)
        self.cancelled = set()
        self.lock = threading.Lock()
        self.backpressure = 0

    def sync(self, policy):
        with self.lock:
            if policy.version <= self.policy.version: raise ValueError('版本必须递增')
            self.policy = policy

    def submit(self, task, action):
        with self.lock:
            if task in self.cancelled: return False
            if not 0 <= action < len(self.policy.probabilities): raise ValueError('action')
            record = Rollout(task, self.policy, action, math.log(self.policy.probabilities[action]))
            try: self.queue.put_nowait(record)
            except queue.Full:
                self.backpressure += 1; return False
            return True

    def cancel(self, task):
        with self.lock: self.cancelled.add(task)

    def consume(self, current, max_staleness):
        record = self.queue.get_nowait()
        try:
            with self.lock:
                if record.task in self.cancelled: return None, 'cancelled'
            age = current.version-record.policy.version
            if age < 0 or age > max_staleness: return None, 'stale'
            if not math.isclose(math.exp(record.behavior_logp),record.policy.probabilities[record.action]):
                raise ValueError('行为概率与采样快照不符')
            return record, 'accepted'
        finally: self.queue.task_done()


def expectation(behavior, target, mode):
    if len(behavior)!=2 or len(target)!=2 or any(not math.isfinite(p) or p<=0 for p in behavior) or any(not math.isfinite(p) or p<0 for p in target) or not math.isclose(sum(behavior),1) or not math.isclose(sum(target),1):
        raise ValueError('行为策略必须覆盖目标支持集且归一化')
    rewards = (0.,1.)
    weights = [t/b for b,t in zip(behavior,target)]
    if mode=='uncorrected': weights=[1.,1.]
    if mode=='clipped': weights=[max(.8,min(1.2,w)) for w in weights]
    value=sum(b*w*r for b,w,r in zip(behavior,weights,rewards))
    mean=sum(b*w for b,w in zip(behavior,weights))
    ess=mean**2/sum(b*w*w for b,w in zip(behavior,weights))
    return value,ess


def experiment(mode='corrected'):
    if mode not in MODES: raise ValueError(mode)
    old, current=Policy(1,(.8,.2)),Policy(3,(.2,.8))
    worker=Worker(old,2)
    worker.submit('old',1); worker.submit('cancel',0)
    assert not worker.submit('overflow',0)
    worker.sync(current); worker.cancel('cancel')
    r,status=worker.consume(current,0 if mode=='stale' else 2)
    _,cancelled=worker.consume(current,2)
    assert cancelled=='cancelled' and worker.queue.unfinished_tasks==0
    worker.submit('new',1)
    fresh,_=worker.consume(current,0)
    assert fresh.policy.version==3 and fresh.behavior_logp==math.log(.8)
    value,ess=expectation(old.probabilities,current.probabilities,mode)
    if mode=='stale': value,ess=0.,0.
    if mode=='corrected': assert abs(value-.8)<1e-12
    return {'metrics':{'旧样本接纳':r is not None,'奖励期望':round(value,4),'目标期望':.8,
                       '相对 ESS':round(ess,4),'背压次数':worker.backpressure,'未回收任务':worker.queue.unfinished_tasks},
            'steps':[f'rollout v1 进入容量为 2 的队列；同步后新任务使用 v3，旧记录仍保存 v1。',
                     f'旧任务状态 {status}；取消任务不参与更新；新任务行为概率为 0.8。',
                     '枚举两种动作，按行为分布计算期望与重要性权重；不混入采样噪声。'],
            'note':'这是异步执行顺序的确定性重放，不是多 GPU 吞吐实测。ESS 是有效样本比例；裁剪会引入偏差。拒收时期望 0 表示没有更新，不能解释成策略奖励降为零。引擎温度与数值精度差异应保留实际行为 logp，不能用训练器重算值替换。'}

if __name__=='__main__':
    import json
    print(json.dumps({m:experiment(m) for m in MODES},ensure_ascii=False,indent=2))
