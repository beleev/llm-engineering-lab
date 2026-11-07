"""同一起点的示范 SFT / 拒绝采样微调 / 在线 RL；训练协议复用 m32，验收复用 m25。"""
import copy
import json
from functools import lru_cache
from types import SimpleNamespace
import torch
from llm_finetune.run_finetune.common import make_model
from llm_finetune.methods.sft import SFTLoss
from llm_agent.m25_trajectory_learning.demo import Task, Trajectory, execute
from llm_agent.m32_agent_training.demo import ACTIONS, TOKENS, rollout

TRAIN = tuple(Task(f'train-{i}',f'entity-{i}',10+i%4,budget=2) for i in range(16))
TEST = tuple(Task(f'test-{i}',f'entity-{i}',10+i%4,budget=2) for i in range(32,48))


def examples(task, actions):
    observed = None
    rows = []
    for action in actions:
        rows.append(([1,3] if observed is None else [2,observed], TOKENS[ACTIONS.index(action)]))
        if action == 'lookup': observed = task.expected
    return rows


def supervised(model, rows, steps=40):
    if not rows: return 0  # 无成功样本不能伪造示范，也不能对空 batch 算 CE。
    x=torch.tensor([r[0] for r in rows]); y=torch.tensor([[-100,r[1]] for r in rows])
    opt=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=0)
    for _ in range(steps):
        loss=SFTLoss().compute(model(x),y)['total_loss']
        opt.zero_grad();loss.backward();opt.step()
    return steps


@torch.no_grad()
def score(model):
    return sum(rollout(model,t)[0]['eligible'] for t in TEST)/len(TEST)


def collect(model, per_task=16):
    rows, counts = [], {t.id:0 for t in TRAIN}
    with torch.no_grad():
        for task in TRAIN:
            for _ in range(per_task):
                report, _, actions=rollout(model,task,sample=True,temperature=2)
                if report['eligible']:
                    assert execute(task,Trajectory('sampled',tuple(actions)))['eligible']
                    rows.extend(examples(task,actions));counts[task.id]+=1
    return rows,counts


@lru_cache(maxsize=1)
def run():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        base=make_model(SimpleNamespace(vocab_size=24),d_model=32,num_layers=1)
        before=score(base); result={}
        for mode in ['sft','rft','rl']:
            torch.manual_seed(101)
            model=copy.deepcopy(base); counts={}; sampled=0
            if mode=='sft':
                rows=[row for t in TRAIN for row in examples(t,['lookup','answer:observed'])]
                updates=supervised(model,rows)
            elif mode=='rft':
                rows,counts=collect(model);sampled=256
                updates=supervised(model,rows)
            else:
                updates=32;sampled=256
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
                for step in range(updates):
                    batch=[rollout(model,TRAIN[(step*8+i)%16],sample=True,temperature=2) for i in range(8)]
                    rewards=torch.tensor([float(row['eligible']) for row,_,_ in batch])
                    loss=-(torch.stack([lp for _,lp,_ in batch])*(rewards-rewards.mean())).mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            delta=sum(float((v-base.state_dict()[k]).abs().sum()) for k,v in model.state_dict().items())
            result[mode]={'初始留出通过率':before,'训练后留出通过率':score(model),'采样轨迹数':sampled,
                          '更新步数':updates,'参数变化 L1':round(delta,4)}
            if mode=='rft':
                result[mode].update({'保留成功轨迹':sum(counts.values()),'覆盖训练任务':sum(v>0 for v in counts.values()),
                                    '各训练任务保留数':list(counts.values())})
        assert set(t.entity for t in TRAIN).isdisjoint(t.entity for t in TEST)
        return result


def experiment(mode='rft'):
    if mode not in ('sft','rft','rl'):raise ValueError(mode)
    return {'metrics':run()[mode], 'steps':['同一随机初始化，训练/留出实体不重叠，协议和值域相同。',
        'RFT 和 RL 各采样 256 条；SFT 用 16 条人工轨迹（32 个状态/动作监督样本）。更新步数单独报告，不声称计算预算完全相同。',
        'RFT 只对通过独立执行验证的当前策略轨迹做监督；零成功时跳过更新。']}


def main():
    result=run()
    assert result['rft']['保留成功轨迹']>0
    assert result['rft']['参数变化 L1']>0
    assert result['rft']['采样轨迹数']==result['rl']['采样轨迹数']
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
