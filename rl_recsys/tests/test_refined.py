import numpy as np
import pandas as pd
import torch
from rl_recsys.data import ROOT
from rl_recsys.models import Policy,seed_everything
from rl_recsys.refined_data import enrich,OUT,INPUT,RECEIVED
from rl_recsys.refined_models import (ContextReward,ResearchPolicy,state,transition,regularized_loss,features)
import json


def test_strict_causal_feedback_excludes_equal_and_future_events():
    frame=pd.DataFrame({'uid':[1]*6,'timestamp':[0,5,5,10,10,15],
                        'token':[2,3,4,5,6,7],'fraction':[.2,.4,.8,.1,.9,1.],
                        'track_length_seconds':[100]*6})
    examples={'x':np.array([[2,3,4,0]]),'lengths':np.array([3]),'user':np.array([1]),'timestamp':np.array([10])}
    a=enrich(frame,examples)
    frame.loc[frame.timestamp>=10,'fraction']=0
    frame.loc[frame.timestamp>=10,'track_length_seconds']=999
    b=enrich(frame,examples)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(a['hfraction'],[[.2,.4,.8,0]])
    np.testing.assert_allclose(a['past_sum'],[1.4])
    assert a['request_gap'][0]==5


def test_zero_adapter_preserves_received_checkpoint_exactly():
    seed_everything(4)
    cfg=json.loads((RECEIVED/'architecture.json').read_text())
    weights=torch.load(RECEIVED/'best.pt',weights_only=True,map_location='cpu')
    original=Policy(cfg,4098).eval(); original.load_state_dict(weights)
    adapted=ResearchPolicy(cfg,4098).eval(); adapted.load_state_dict(weights,strict=False)
    with np.load(OUT/'validation.npz') as data: s=state(data,slice(0,4))
    with torch.no_grad():
        assert torch.equal(original(s['x'],s['lengths'])[:,2:],adapted(s))
        assert torch.isfinite(features(s)).all()


def test_reward_full_catalog_equals_single_action_and_initial_prior():
    seed_everything(4)
    with np.load(INPUT/'catalog.npz') as cat: model=ContextReward(dict(cat),32)
    with np.load(OUT/'validation.npz') as data: s=state(data,slice(0,4))
    actions=torch.tensor([2,4,8,16]); pred=model(s)
    torch.testing.assert_close(model(s,actions),pred[torch.arange(4),:,actions-2])
    torch.testing.assert_close(pred,model.prior[None].expand(4,-1,-1))
    # Also check the trained, nonzero candidate/history calibration path.
    with torch.no_grad(): model.coeff.normal_(); model.scale.fill_(.7); model.context[-1].weight.normal_()
    pred=model(s)
    torch.testing.assert_close(model(s,actions),pred[torch.arange(4),:,actions-2])


def test_synthetic_transition_retains_feedback_and_only_penalizes_generated():
    with np.load(OUT/'validation.npz') as data: s=state(data,slice(0,2))
    action=torch.tensor([0,2]); frac=torch.tensor([.25,.75]); duration=torch.tensor([100.,200.])
    t=transition(s,action,frac,duration,4096)
    for j in range(2):
        last=int(t['lengths'][j])-1
        assert t['x'][j,last]==action[j]+2
        assert t['hfraction'][j,last]==frac[j]
        assert t['hgap'][j,last]==s['request_gap'][j]
        assert t['hduration'][j,last]==duration[j]
    torch.testing.assert_close(t['past_sum'],s['past_sum']+frac)
    assert t['generated'].sum()==2
    assert 'generated' not in s
    torch.testing.assert_close(t['request_gap'],torch.tensor([25.,150.]))


def test_multistep_kl_gradient_matches_exact_enumerated_mdp():
    # With zero external reward, upstream gradients still must account for
    # action-dependent FUTURE state-KL. A local-KL-only loss fails this test.
    theta=torch.tensor([.2,1.1,-.3],dtype=torch.float64,requires_grad=True)
    beta=.4; gamma=.95
    p0=torch.stack([theta[0],theta[0]*0]).softmax(0); l0=p0.log()
    k0=(p0*(l0+np.log(2))).sum(); expected_objective=theta.sum()*0; sampled_loss=theta.sum()*0
    for a in (0,1):
        p1=torch.stack([theta[a+1],theta[a+1]*0]).softmax(0); l1=p1.log()
        k1=(p1*(l1+np.log(2))).sum()
        expected_objective+=p0[a]*(-beta*k0-gamma*beta*k1)
        for b in (0,1):
            zero=torch.zeros(1,dtype=torch.float64)
            loss=regularized_loss([l0[a,None],l1[b,None]],[zero,zero],[zero,zero],
                                  [k0[None],k1[None]],beta,gamma)
            sampled_loss+=(p0[a]*p1[b]).detach()*loss
    actual=torch.autograd.grad(sampled_loss,theta,retain_graph=True)[0]
    exact=torch.autograd.grad(-expected_objective/2,theta)[0]
    torch.testing.assert_close(actual,exact,rtol=1e-9,atol=1e-10)
    assert actual[0].abs()>.001
