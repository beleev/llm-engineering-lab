"""迁移与隔离边界：测试失败路径，避免只重放导出夹具。"""
import json
import pytest
from llm_agent.m57_checkpoint_migration.demo import checkpoint, migrate, resume, ledger, trial
from llm_infer.m34_cache_identity.demo import Cache, Gateway, compute
from llm_infer.m33_tenant_scheduling.demo import simulate


def initial(**changes):
    return dict(dict(schema_version=1,flow_version=1,node='submit',operation='invoice-1',amount=10,tool_version=1),**changes)


@pytest.mark.parametrize('change',[dict(node='removed'),dict(schema_version=99),dict(amount=True),dict(amount=-1),dict(flow_version=0),dict(tool_version=8)])
def test_incompatible_checkpoints_rejected_before_effect(tmp_path,change):
    checkpoint(tmp_path,initial(**change))
    with pytest.raises(ValueError):resume(tmp_path)
    assert ledger(tmp_path)==(0,0)


def test_revoked_permission_even_after_external_success(tmp_path):
    checkpoint(tmp_path,initial())
    ledger(tmp_path,10)
    with pytest.raises(PermissionError):resume(tmp_path,authorized=False)
    assert json.loads((tmp_path/'state.json').read_text())['node']=='submit'
    assert ledger(tmp_path)==(1,10)


def test_receipt_cannot_confirm_different_parameters(tmp_path):
    checkpoint(tmp_path,initial(amount=12))
    ledger(tmp_path,10)
    with pytest.raises(ValueError,match='参数'):resume(tmp_path)
    assert ledger(tmp_path)==(1,10)


def test_new_business_policy_does_not_rewrite_old_flow(tmp_path):
    checkpoint(tmp_path,initial(amount=25,flow_version=2))
    with pytest.raises(ValueError,match='policy'):resume(tmp_path)
    assert ledger(tmp_path)==(0,0)
    checkpoint(tmp_path,initial(amount=25,flow_version=1))
    assert resume(tmp_path)['policy_checked'] is False
    assert ledger(tmp_path)==(1,25)


def test_migration_idempotence_and_crash_recovery():
    state=migrate(initial())
    assert migrate(state)==state and state['summary']==''
    r=trial('crash')
    assert r['child_exit']==17 and r['writes']==1 and r['reconciled']


def test_cache_model_adapter_and_prefix_identity():
    c=Cache();g=Gateway();scope=g.scope('credential-A')
    first,_=c.get(scope,[11,12],'base-1','adapter-1')
    assert c.get(scope,[11,12],'base-1','adapter-1')==(first,True)
    for tokens,model,adapter in [([11,13],'base-1','adapter-1'),([11,12],'base-2','adapter-1'),([11,12],'base-1','adapter-2')]:
        value,hit=c.get(scope,tokens,model,adapter)
        assert not hit and value==compute(tokens,model,adapter) and value!=first
    assert not c.get(g.scope('credential-B'),[11,12],'base-1','adapter-1')[1]


@pytest.mark.parametrize('credential,tenant',[('unknown',None),('credential-B','A'),('credential-A','B')])
def test_gateway_rejects_identity_override(credential,tenant):
    with pytest.raises(PermissionError):Gateway().scope(credential,tenant)


@pytest.mark.parametrize('mode',['fifo','priority','fair','quota'])
def test_terminal_accounting(mode):
    r=simulate(mode)
    assert r['terminal_count']==r['arrivals']==64
    assert sum(t[k] for t in r['tenants'].values() for k in ('completed','timeout','rejected'))==64


def test_fairness_and_admission_are_separate():
    priority,fair,quota=(simulate(m)['tenants'] for m in ('priority','fair','quota'))
    assert priority['B']['timeout']==4 and fair['B']['completed']==4
    assert fair['A']['rejected']==0 and quota['A']['rejected']>0
    assert quota['A']['peak_queue']<=8 and quota['B']['completed']==4
