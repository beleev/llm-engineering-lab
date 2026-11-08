"""真实 ClaudeLLM 适配器的离线协议夹具；绝不构造在线客户端。"""
import json
from types import SimpleNamespace
from llm_agent.core.claude_llm import ClaudeLLM
from llm_agent.core.schema import Message
from llm_agent.core.streaming import StreamError


def events(tool_names=('first','second'), final=False):
    yield {'type':'message_start','message':{'usage':{'input_tokens':20,'output_tokens':1,'cache_read_input_tokens':5}}}
    for i,name in enumerate(('text',) if final else tool_names):
        block={'type':'text','text':''} if final else {'type':'tool_use','id':f'call-{i}','name':name,'input':{}}
        yield {'type':'content_block_start','index':i,'content_block':block}
        for part in (['完成'] if final else ['{','}']):
            delta={'type':'text_delta','text':part} if final else {'type':'input_json_delta','partial_json':part}
            yield {'type':'content_block_delta','index':i,'delta':delta}
        yield {'type':'content_block_stop','index':i}
    yield {'type':'message_delta','delta':{'stop_reason':'end_turn' if final else 'tool_use'},'usage':{'output_tokens':4}}
    yield {'type':'message_delta','delta':{},'usage':{'output_tokens':6}}
    yield {'type':'message_stop'}


class RateLimited(Exception):
    status_code=429
    response=SimpleNamespace(headers={'retry-after':'0'})


class FixtureClient:
    def __init__(self, mode='valid'):
        self.messages=self;self.mode=mode;self.calls=0;self.closed=0

    def create(self, **kwargs):
        self.calls+=1
        if self.mode=='retry' and self.calls==1:raise RateLimited('fixture')
        final=any(b['type']=='tool_result' for m in kwargs['messages'] for b in m['content'])
        rows=list(events(final=final))
        if self.mode=='disconnect':rows=rows[:-1]
        if self.mode=='malformed':
            rows[3]['delta']['partial_json']='broken'
        def stream():
            try:yield from rows
            finally:self.closed+=1
        return stream()


def experiment(mode='valid'):
    if mode not in ('valid','disconnect','malformed'):raise ValueError(mode)
    client=FixtureClient(mode)
    llm=ClaudeLLM(client=client,sleep=lambda _:None)
    try:
        action=llm.next([Message('user','运行夹具')],[{'name':'first'},{'name':'second'}])
        metrics={'可交给权限门的工具数':len(action.tool_calls),'API 尝试数':client.calls,
                 '服务端输出 token':llm.last_usage['output_tokens'],'流已关闭':client.closed==1}
        assert llm.last_usage['output_tokens']==6  # 累积 1→4→6 不是 11
        steps=['逐块累积 JSON；完整 message_stop 后才返回 ModelAction。','使用服务端最终 usage；流中累计计数覆盖旧值。']
    except StreamError:
        metrics={'可交给权限门的工具数':0,'API 尝试数':client.calls,'流已关闭':client.closed==1}
        steps=['流或 JSON 未闭合，拒绝返回可执行工具；不自动重发已开始的流。']
    return {'metrics':metrics,'steps':steps}


def main():
    rows={m:experiment(m) for m in ('valid','disconnect','malformed')}
    assert rows['valid']['metrics']['可交给权限门的工具数']==2
    assert rows['disconnect']['metrics']['可交给权限门的工具数']==0
    assert rows['malformed']['metrics']['可交给权限门的工具数']==0
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
