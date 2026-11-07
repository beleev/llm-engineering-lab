"""实际 Agent + 实际适配器路径的 span 树；网络替换为离线流式夹具。"""
import asyncio
import json
import threading
from llm_agent.core.agent import Agent
from llm_agent.core.claude_llm import ClaudeLLM
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, ToolRegistry
from llm_agent.core.tracing import Tracer
from llm_agent.m15_claude_api.stream_demo import FixtureClient


class Rendezvous(Tool):
    def __init__(self,name,barrier,cancel=False):
        self.name=name;self.barrier=barrier;self.cancel=cancel
    def execute(self,args):
        self.barrier.wait(timeout=5)
        if self.cancel:raise asyncio.CancelledError()
        return ToolResult(self.name,'secret-never-in-trace')


def experiment(mode='parallel', include_trace=False):
    if mode not in ('parallel','retry','cancel'):raise ValueError(mode)
    tracer=Tracer();client=FixtureClient('retry' if mode=='retry' else 'valid')
    llm=ClaudeLLM(client=client,sleep=lambda _:None,tracer=tracer)
    barrier=threading.Barrier(2)
    agent=Agent(llm,ToolRegistry([Rendezvous('first',barrier,mode=='cancel'),Rendezvous('second',barrier)]),
                permissions=PermissionGate(mode='auto'),tracer=tracer,context_budget_chars=20000)
    try:agent.run('secret-user-prompt',verbose=False)
    except asyncio.CancelledError:pass
    rows=tracer.export();summary=tracer.summary()
    tools=[r for r in rows if r['kind']=='tool']
    overlap=max(r['start'] for r in tools)<min(r['end'] for r in tools)
    assert overlap
    assert 'secret' not in json.dumps(rows)
    assert all(r['trace_id']==rows[0]['trace_id'] for r in rows)
    assert summary['summed_span_seconds']>=summary['wall_seconds']
    assert all(r['end'] is not None for r in rows)
    result = {'metrics':{'span 数':len(rows),'模型尝试数':summary['attempts'],'逻辑模型请求数':summary['logical_requests'],
                       '工具执行数':len(tools),'并行工具时间重叠':overlap,'根状态':rows[0]['status'],
                       '成功请求实报输出 token':sum(u.get('output_tokens',0) for u in agent.reported_usage),
                       '成功请求费用单位（假设单价）':sum((u.get('input_tokens',0)+2*u.get('output_tokens',0)+.25*u.get('cache_read_input_tokens',0))/1000 for u in agent.reported_usage)},
            'steps':['根 span 是一次 Agent.run；模型与工具是子 span，API attempt 挂在模型下面。',
                     '工具通过同一个线程屏障，时间区间真实重叠；不能把各 span 时长相加当用户等待时间。',
                     '导出只含结构、状态与计数，不收集提示、参数、输出或异常正文。',
                     '演示单价：输入/输出/缓存读每千 token 为 1/2/0.25 单位，非供应商报价；失败尝试账单未知，不能当作零费用。']}

    if include_trace:
        result['spans'] = rows
        result['timing'] = summary
    return result


def main():
    rows={m:experiment(m, include_trace=True) for m in ('parallel','retry','cancel')}
    assert rows['retry']['metrics']['模型尝试数']==3
    assert rows['retry']['metrics']['逻辑模型请求数']==2
    assert rows['cancel']['metrics']['根状态']=='cancelled'
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
