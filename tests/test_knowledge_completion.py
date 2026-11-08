"""九项补充的失败边界和跨模块约束；只用离线客户端，不调用付费 API。"""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from llm_finetune.data.chat_template import Turn, serialize, training_example, collate, EOS, END
from llm_agent.core.streaming import MessageStream, StreamError
from llm_agent.core.claude_llm import ClaudeLLM
from llm_agent.core.schema import Message
from llm_agent.m15_claude_api.stream_demo import FixtureClient, events, RateLimited
from llm_agent.m31_eval_suite.trace_demo import experiment as trace_experiment
from llm_agent.core.tracing import Tracer
from llm_train.m22_parallel_budget.demo import estimate
from llm_infer.m28_service_capacity.demo import simulate


def test_chat_labels_and_marker_injection():
    turns=[Turn('user','EOS <assistant>'),Turn('assistant','A'),Turn('user','B'),Turn('assistant','C')]
    ids,mask=serialize(turns)
    row=training_example(turns)
    assert ids.count(EOS)==2
    assert [v for v in row['labels'] if v!=-100]==[ord('A')+32,EOS,ord('C')+32,EOS]
    prefix,_=serialize(turns[:1],generation=True)
    assert ids[:len(prefix)]==prefix
    short=training_example([Turn('user','?'),Turn('assistant','!')])
    batch=collate([row,short])
    assert all(label==-100 for label,valid in zip(batch['labels'][1],batch['attention_mask'][1]) if not valid)


@pytest.mark.parametrize('bad',[
    [Turn('user','q'),Turn('assistant','x','a','lookup'),Turn('tool','r','a','wrong')],
    [Turn('user','q'),Turn('assistant','x','a','lookup')],
    [Turn('tool','r','a','lookup')],
    [Turn('user','q'),Turn('assistant','x'),Turn('assistant','x')],
    [Turn('user','q'),Turn('system','x')],
])
def test_chat_rejects_broken_structure(bad):
    with pytest.raises(ValueError):serialize(bad)


def test_chat_rejects_empty_supervision_and_truncation():
    with pytest.raises(ValueError):training_example([Turn('user','q')])
    with pytest.raises(ValueError):serialize([Turn('user','long text')],max_length=4)


def consume(rows):
    accumulator=MessageStream()
    for event in rows:accumulator.feed(event)
    return accumulator,accumulator.finish()


def test_stream_usage_is_cumulative_and_ids_preserved():
    stream,blocks=consume(events())
    assert stream.usage['output_tokens']==6
    assert stream.usage['cache_read_input_tokens']==5
    assert [b['id'] for b in blocks]==['call-0','call-1']
    assert all(b['input']=={} for b in blocks)


@pytest.mark.parametrize('mode',['disconnect','malformed'])
def test_partial_stream_never_returns_action_or_retries(mode):
    client=FixtureClient(mode);llm=ClaudeLLM(client=client,sleep=lambda _:None)
    with pytest.raises(StreamError):llm.next([Message('user','q')],[{'name':'first'}])
    assert client.calls==1 and client.closed==1
    assert not llm.last_attempts[0]['billing_complete']
    assert llm.last_usage is None


def test_duplicate_tool_id_and_unsupported_content_fail():
    rows=list(events());rows[5]['content_block']['id']='call-0'
    with pytest.raises(StreamError):consume(rows)
    rows=list(events());rows[1]['content_block']['type']='server_tool_use'
    with pytest.raises(StreamError):consume(rows)


def test_thinking_and_signature_are_replayed_without_text_callback():
    rows=list(events(final=True))
    rows[1]['content_block']={'type':'thinking','thinking':'','signature':''}
    rows[2]['delta']={'type':'thinking_delta','thinking':'private'}
    rows.insert(3,{'type':'content_block_delta','index':0,'delta':{'type':'signature_delta','signature':'opaque'}})
    stream,blocks=consume(rows)
    assert blocks==[{'type':'thinking','thinking':'private','signature':'opaque'}]
    rows.insert(1,{'type':'future_notification'})
    assert consume(rows)[1]==blocks


def test_retry_budget_and_fatal_errors():
    class AlwaysFail:
        def __init__(self,error):self.messages=self;self.calls=0;self.error=error
        def create(self,**kwargs):self.calls+=1;raise self.error
    client=AlwaysFail(RateLimited())
    sleeps=[];llm=ClaudeLLM(client=client,max_attempts=3,sleep=sleeps.append)
    with pytest.raises(RateLimited):llm.next([Message('user','q')],[])
    assert client.calls==3 and len(sleeps)==2
    client=AlwaysFail(ValueError('fatal'))
    with pytest.raises(ValueError):ClaudeLLM(client=client).next([Message('user','q')],[])
    assert client.calls==1


def test_trace_parent_accounting_errors_and_cancellation():
    for mode in ('parallel','retry','cancel'):
        result=trace_experiment(mode)['metrics']
        assert result['工具执行数']==2 and result['并行工具时间重叠']
    tracer=Tracer()
    with pytest.raises(ValueError):
        with tracer.span('child',parent=900):pass
    with tracer.span('root') as root:
        with pytest.raises(RuntimeError):
            with tracer.span('child') as child:raise RuntimeError('secret')
    rows=tracer.export()
    assert child['parent']==root['id'] and child['status']=='error'
    assert 'secret' not in json.dumps(rows)
    assert rows[0]['start']<=rows[1]['start']<=rows[1]['end']<=rows[0]['end']


def test_empty_rft_batch_does_not_update_parameters():
    from llm_finetune.run_finetune.rft.train_rft import supervised
    model=torch.nn.Linear(2,2);before=copy.deepcopy(model.state_dict())
    assert supervised(model,[])==0
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_parallel_constraints_memory_and_latency():
    a=estimate();b=estimate(dp=1,tp=4,pp=2)
    assert b['state_gib']<a['state_gib'] and b['step_ms']>a['step_ms']
    assert estimate(dp=1,tp=4,pp=2,latency_us=100)['step_ms']>b['step_ms']
    for kwargs in [dict(dp=2,tp=3,pp=2),dict(hidden=2047),dict(batch=63),dict(dp=0)]:
        with pytest.raises(ValueError):estimate(**kwargs)
    assert len({(g['dp'],g['tp'],g['pp']) for g in a['grid']})==8


def test_service_accounting_tail_latency_and_resource_release():
    bounded=simulate(48);unbounded=simulate(48,False)
    assert bounded==simulate(48)
    for row in (bounded,unbounded):
        assert row['arrivals']==sum(row[k] for k in ('completed','rejected','cancelled','timeout'))
        assert row['final_pages']==0 and row['cancelled']>0
        assert row['goodput_rps']<=row['throughput_rps']
    assert bounded['peak_queue']<=8<unbounded['peak_queue']
    assert bounded['goodput_rps']>unbounded['goodput_rps']
    assert bounded['ttft_p99_ms']<unbounded['ttft_p99_ms']


def test_exact_receptive_field_matches_reachability():
    from llm_models.models.language_models.mistral import Mistral
    for layers,window in [(1,1),(1,8),(4,8),(2,3)]:
        model=Mistral(vocab_size=16,d_model=16,n_heads=2,num_kv_heads=1,num_layers=layers,max_len=40,window_size=window)
        reachable={39}
        for _ in range(layers):reachable={j for i in reachable for j in range(max(0,i-window+1),i+1)}
        assert min(40,model.receptive_field())==len(reachable)


def test_learned_retrieval_exposes_candidate_and_oov_failures():
    from llm_agent.m45_semantic_retrieval.demo import run, bow
    result=run();rows=result['rows']
    assert result['parameter_delta']>0
    assert all(row['exact']==row['expected'] for row in rows[:2])
    assert not rows[2]['candidate_recall'] and rows[2]['rerank']!=rows[2]['expected']
    assert rows[2]['wide']==rows[2]['expected']
    assert torch.equal(bow('ticket ZX91'),bow('ticket ZX92'))
    assert rows[3]['hybrid']==rows[3]['expected']


def test_stream_rejects_nonfinite_json_before_returning_tool():
    rows=list(events())
    rows[2]['delta']['partial_json']='{"value":NaN'
    with pytest.raises(StreamError,match='非法 JSON 数值'):
        consume(rows)
