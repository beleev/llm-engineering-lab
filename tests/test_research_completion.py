"""研究补充的失效边界：掩码、非法证书、拒答、梯度与验收隔离。"""
import math
import numpy as np
import pytest
import torch
from llm_finetune.methods.ipo import IPOLoss
from llm_agent.m46_joint_rag.demo import learn, rewrite
from llm_agent.m47_context_utilization.demo import read
from llm_agent.m51_formal_science.demo import verify_proof
from llm_train.m25_reliability_metrics.demo import metrics
from llm_models.run_models.multimodal.vision_tasks.demo import glyph, recognize


def test_ipo_padding_and_reference_margin():
    c=torch.tensor([[[1.,0.],[8.,-8.]]],requires_grad=True)
    r=torch.tensor([[[0.,1.],[-8.,8.]]],requires_grad=True)
    labels={'chosen':torch.tensor([[0,-100]]),'rejected':torch.tensor([[0,-100]])}
    result=IPOLoss(beta=.5).compute({'chosen':c,'rejected':r,'ref_chosen':torch.zeros_like(c),'ref_rejected':torch.zeros_like(r)},labels)
    assert result['total_loss'].item()==pytest.approx(0,abs=1e-12)
    # 与单 token 的目标相同，padding logit 不参与 loss/梯度。
    result['total_loss'].backward()
    assert torch.count_nonzero(c.grad[:,1])==0
    with pytest.raises(ValueError):
        IPOLoss().compute({},dict(chosen=torch.tensor([[-100]]),rejected=torch.tensor([[0]])))


@pytest.mark.parametrize('beta',[0,-1,float('nan'),float('inf')])
def test_ipo_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):IPOLoss(beta)


def test_rag_gradient_paths_and_rewrite_constraints():
    _,p,g=learn(False)
    _,updated,fixed=learn(True)
    _,_,trained=learn(True,True)
    assert updated[1]>p[1]
    assert torch.equal(g,fixed)
    assert not torch.equal(g,trained)
    assert rewrite('Comet v2 不要 邮件',{'Comet':'Atlas'})==['Atlas','v2','不要','邮件']


def test_context_moving_evidence_removes_only_position_bias():
    for n in (16,64,256):
        assert read(n,0,'position-neutral')[0]==pytest.approx(read(n,n//2,'position-neutral')[0])
    assert read(256,128,'position-neutral')[1] is False
    assert read(256,128,'retrieve',True)[1] is True


@pytest.mark.parametrize('refs',[(0,0),(-1,1),(0,3),(0,),('0',1)])
def test_proof_rejects_invalid_references(refs):
    assumptions=['P',('implies','P','Q')]
    proof=[('P','assumption',()),(('implies','P','Q'),'assumption',()),('Q','mp',refs)]
    assert not verify_proof(assumptions,proof)


def test_visual_ambiguity_is_not_first_label():
    assert recognize(glyph('3'))[0]=='3'
    assert recognize((glyph('2')+glyph('3'))/2)[0] is None


@pytest.mark.parametrize('p,y,bins',[([math.nan],[1],5),([.5],[1],0),([.5],[2],5),([] ,[],5)])
def test_calibration_rejects_invalid_inputs(p,y,bins):
    with pytest.raises(ValueError):metrics(p,y,bins)


def test_unlearning_matches_retraining_and_dp_budget_composes():
    from llm_finetune.run_finetune.privacy_unlearning.demo import ridge, private_fit
    rng=np.random.default_rng(11);x=rng.normal(size=(9,4));y=rng.normal(size=9)
    gram=x.T@x-np.outer(x[3],x[3]);rhs=x.T@y-x[3]*y[3]
    deleted=np.linalg.solve(gram+.01*np.eye(4),rhs)
    assert np.allclose(deleted,ridge(np.delete(x,3,axis=0),np.delete(y,3)),atol=1e-12)
    _,eps40=private_fit(x,y,steps=40);_,eps80=private_fit(x,y,steps=80)
    assert eps80>eps40


def test_candidate_cannot_pass_independent_verifier_by_local_test_cheat():
    from llm_agent.m50_repository_verification.demo import execute
    complete=execute('complete');cheat=execute('test-cheat')
    assert complete['exit']==0 and complete['passed']==complete['total']==4
    assert cheat['exit']!=0 and cheat['passed']<cheat['total']
