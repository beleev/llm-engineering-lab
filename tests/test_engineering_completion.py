"""跨阶段约定的失效反例：不把同一份成功夹具的输出复制成测试。"""
import copy
import json
import math
import queue
import threading
import pytest
from llm_finetune.run_finetune.artifact_delivery import demo as artifacts
from llm_finetune.run_finetune.async_rollout.demo import Policy, Worker, expectation
from llm_agent.m54_document_lifecycle.demo import Store, fixture, partition
from llm_agent.m55_mcp_versions.demo import MODERN, VERSION_KEY, call, dispatch, resource_server, Execution
from llm_agent.m56_release_loop.demo import Release, Version, feedback_replay

@pytest.fixture(scope='module')
def package(tmp_path_factory):
    path=tmp_path_factory.mktemp('artifact');artifacts.export(path);return path

def test_artifact_file_integrity(package,tmp_path):
    import shutil
    shutil.copytree(package,tmp_path,dirs_exist_ok=True)
    with (tmp_path/'weights.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='摘要'):artifacts.load(tmp_path)

@pytest.mark.parametrize('field,value',[('generation',{'max_new_tokens':1,'do_sample':False}),('quantization','invented')])
def test_artifact_semantic_mismatch_after_rehash(package,tmp_path,field,value):
    import shutil
    shutil.copytree(package,tmp_path,dirs_exist_ok=True)
    p=json.loads((tmp_path/'package.json').read_text());p[field]=value
    (tmp_path/'package.json').write_text(json.dumps(p))
    m=json.loads((tmp_path/'manifest.json').read_text());m['files']['package.json']=artifacts.sha(tmp_path/'package.json')
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='约定'):artifacts.load(tmp_path)

def test_partial_parse_does_not_destroy_old_snapshot():
    store=fixture();before=store.snapshot()
    with pytest.raises(ValueError):store.put('policy',2,'<p>unfinished',['A'])
    assert store.snapshot()==before

@pytest.mark.parametrize('raw',['%PDF-1.7 ...','plain text','<p>'])
def test_parser_rejects_unsupported_or_empty(raw):
    with pytest.raises(ValueError):partition(raw)

def test_acl_cache_and_stale_citations():
    store=fixture();a=store.search('A','退款 11')[0];b=store.search('B','退款 11')[0]
    assert a.source!=b.source
    with pytest.raises(ValueError):store.cite(b,'A')
    store.revoke('policy','A')
    assert not store.cache and store.search('A','退款 11')==[]
    with pytest.raises(ValueError):store.cite(a,'A')
    assert Store.restore(store.snapshot()).search('A','退款 11')==[]

def test_postfilter_is_safe_but_not_recall_equivalent():
    s=fixture()
    assert len(s.search('A','退款 11'))==1
    assert s.search('A','退款 11','postfilter')==[]

def test_embedding_mismatch_even_for_cache_hit():
    s=fixture();s.search('A','退款 11')
    with pytest.raises(ValueError):s.search('A','退款 11',embedding='next')

def test_versions_cannot_go_backwards():
    s=fixture()
    with pytest.raises(ValueError):s.put('policy',1,'<p>replacement</p>',['A'])
    w=Worker(Policy(2,(.5,.5)))
    with pytest.raises(ValueError):w.sync(Policy(1,(.5,.5)))

@pytest.mark.parametrize('probabilities',[(0.,1.),(.8,.8),(math.nan,.2)])
def test_invalid_behavior_distribution(probabilities):
    with pytest.raises(ValueError):Policy(1,probabilities)

def test_concurrent_queue_keeps_immutable_policy_versions():
    w=Worker(Policy(1,(.7,.3)),200)
    thread=threading.Thread(target=lambda:[w.submit(str(i),i%2) for i in range(100)])
    thread.start();w.sync(Policy(2,(.2,.8)));thread.join()
    count=0
    while not w.queue.empty():
        r,status=w.consume(w.policy,1)
        assert status=='accepted'
        assert math.isclose(math.exp(r.behavior_logp),r.policy.probabilities[r.action])
        count+=1
    assert count==100 and w.queue.unfinished_tasks==0

def test_cancelled_and_stale_samples_release_queue_accounting():
    w=Worker(Policy(1,(.5,.5)),2);w.submit('cancel',0);w.submit('old',1);w.cancel('cancel')
    assert w.consume(Policy(3,(.5,.5)),0)==(None,'cancelled')
    assert w.consume(Policy(3,(.5,.5)),0)==(None,'stale')
    assert w.queue.unfinished_tasks==0

def test_importance_weight_exact_expectation_and_clipping_bias():
    for p in (.1,.3,.8):
        target=(1-p,p)
        correct,_=expectation((.8,.2),target,'corrected')
        assert correct==pytest.approx(p)
    clipped,_=expectation((.8,.2),(.2,.8),'clipped')
    assert clipped!=pytest.approx(.8)

@pytest.mark.parametrize('token,status',[('invalid',401),('expired',401),('audience',401),('scope',403),('ok',200)])
def test_http_auth_rejects_before_tool_dispatch(token,status):
    with resource_server() as url:
        actual,body=call(url,token=token)
        assert actual==status
        if status!=200:assert 'result' not in body

def test_modern_discovery_and_version_mismatch():
    request={'jsonrpc':'2.0','id':1,'method':'server/discover','params':{'_meta':{VERSION_KEY:MODERN}}}
    assert dispatch(request,MODERN)['result']['supportedVersions']==[MODERN]
    assert dispatch(request,'2025-06-18')['error']['code']==-32020
    request['params']['_meta'][VERSION_KEY]='1900-01-01'
    assert dispatch(request,'1900-01-01')['error']['data']['supported']==[MODERN]
    request['params']['_meta'][VERSION_KEY]=MODERN;request['method']='initialize'
    assert dispatch(request,MODERN)['error']['code']==-32601

def test_cancel_before_commit_and_deadline_equality():
    e=Execution();e.cancelled.set();assert e.run(20,10)=='cancelled' and e.committed==0
    e=Execution();assert e.run(10,10)=='deadline' and e.committed==0

def test_release_missing_feedback_holds_and_safety_rolls_back():
    a=Version('m1','p1','i1','t1');b=Version('m1','p2','i1','t1');r=Release(a,b);r.offline([True]*10,[True]*10,[0]);r.deploy()
    rows=[dict(input='question',ok=True,latency=10,cost=.001,unsafe=False,feedback_at=None)]
    assert r.monitor(rows,100)=='hold'
    rows[0]['unsafe']=True
    assert r.monitor(rows,100)=='rollback' and r.active==a
    with pytest.raises(ValueError):r.deploy()

def test_regression_does_not_use_pending_labels_or_duplicate_protected_inputs():
    import hashlib
    base=dict(input='a',ok=False,latency=10,cost=.01,unsafe=False,feedback_at=1)
    rows=[base,copy.deepcopy(base),dict(base,input='b',feedback_at=5),dict(base,input='c',feedback_at=None)]
    assert len(feedback_replay(rows,set(),now=2))==1
    assert feedback_replay(rows,{hashlib.sha256(b'a').hexdigest()},now=2)==[]


def test_release_cannot_bypass_offline_gate():
    r=Release(Version('m','p','i','t'),Version('m2','p','i','t'))
    with pytest.raises(ValueError):r.deploy()
    with pytest.raises(ValueError):r.monitor([],0)

def test_legacy_client_rejects_incompatible_version_and_closes_child(tmp_path):
    import sys
    from llm_agent.core.mcp import MCPClient, MCPError
    script=tmp_path/'server.py'
    script.write_text('import sys,json\nreq=json.loads(sys.stdin.readline())\nprint(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{"protocolVersion":"1900-01-01"}}),flush=True)\nsys.stdin.read()\n')
    with pytest.raises(MCPError,match='只支持'):
        MCPClient('mismatch',[sys.executable,str(script)],timeout=2)


@pytest.mark.parametrize('headers',[{'Mcp-Method':'wrong'},{'MCP-Protocol-Version':'1900-01-01'}])
def test_http_mirrored_headers_must_match_body(headers):
    with resource_server() as url:
        status,body=call(url,headers_override=headers)
        assert status==400 and body['error']['code']==-32020

def test_http_method_not_found_and_unicode_header_roundtrip():
    from llm_agent.m55_mcp_versions.demo import encode_header,decode_header
    for name in ['echo','工具',' leading','=?base64?literal?=']:
        assert decode_header(encode_header(name))==name
    with resource_server() as url:
        status,body=call(url,method='initialize')
        assert status==404 and body['error']['code']==-32601


def test_incomplete_artifact_manifest_is_rejected(package,tmp_path):
    import shutil
    shutil.copytree(package,tmp_path,dirs_exist_ok=True)
    m=json.loads((tmp_path/'manifest.json').read_text());m['files'].pop('weights.pt')
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='清单不完整'):artifacts.load(tmp_path)
