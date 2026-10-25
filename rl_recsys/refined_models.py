"""Feedback-aware actor and calibrated response model; original classes unchanged."""
import math
import numpy as np
import torch
from torch import nn
from .models import Policy, append_action
from .refined_data import STATE_KEYS

FEATURE_DIM = 16


def state(data, idx):
    return {k:torch.as_tensor(data[k][idx]) for k in STATE_KEYS}


def features(s):
    x, lengths = s['x'],s['lengths']
    mask=x.ne(0).float(); n=lengths.float().clamp_min(1)
    f=s['hfraction']; row=torch.arange(len(x)); last=lengths-1
    mean=(f*mask).sum(1)/n
    pos=torch.arange(x.shape[1])[None,:]
    recent=(pos>= (lengths-3).clamp_min(0)[:,None])*mask
    recent_mean=(f*recent).sum(1)/recent.sum(1).clamp_min(1)
    loggap=torch.log1p(s['hgap'].clamp(0,86400))/math.log1p(86400)
    logdur=torch.log1p(s['hduration'].clamp(0,3600))/math.log1p(3600)
    return torch.stack([f[row,last],recent_mean,mean,
        ((f-mean[:,None]).square()*mask).sum(1)/n,
        ((f>.5)*mask).sum(1)/n,((f<.1)*mask).sum(1)/n,
        n/x.shape[1],x.ge(2).float().sum(1)/n,
        s['past_sum']/s['past_count'].clamp_min(1),
        torch.log1p(s['past_count'].clamp_max(100000))/math.log1p(100000),
        (loggap*mask).sum(1)/n,loggap[row,last],
        torch.log1p(s['request_gap'].clamp(0,86400))/math.log1p(86400),
        (logdur*mask).sum(1)/n,logdur[row,last],
        ((s['hgap']<=1800)*mask).sum(1)/n],1)


class ResearchPolicy(Policy):
    def __init__(self,cfg,vocab_size):
        super().__init__(cfg,vocab_size)
        self.feedback_adapter=nn.Linear(FEATURE_DIM,cfg['hidden_size'])
        nn.init.zeros_(self.feedback_adapter.weight); nn.init.zeros_(self.feedback_adapter.bias)

    def forward(self,s):
        x,lengths=s['x'],s['lengths']
        h=self.decoder(input_ids=x, attention_mask=x.ne(0).long(),use_cache=False).last_hidden_state
        last=h[torch.arange(len(x)), lengths-1]+self.feedback_adapter(features(s))
        return (last @ self.decoder.wte.weight.T+self.bias)[:,2:]


class ContextReward(nn.Module):
    def __init__(self,catalog,hidden=0):
        super().__init__()
        prior=np.stack([catalog['item_fraction'][2:],catalog['item_continuation'][2:]]).clip(.01,.99)
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32))
        self.register_buffer('prior_logits',torch.tensor(np.log(prior/(1-prior)),dtype=torch.float32))
        duration=np.clip(catalog['duration'][2:],0,600)
        self.register_buffer('duration',torch.tensor(duration,dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(FEATURE_DIM,hidden),nn.Tanh(),nn.Linear(hidden,2)) if hidden else nn.Linear(FEATURE_DIM,2)
        final=self.context[-1] if hidden else self.context
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
        self.scale=nn.Parameter(torch.ones(2))
        self.coeff=nn.Parameter(torch.zeros(2,4))

    def forward(self,s,actions=None):
        x=s['x']; frac=s['hfraction']; b=len(x); v=self.prior.shape[1]
        if actions is None:
            count=torch.zeros((b,v+2)); total=torch.zeros_like(count)
            count.scatter_add_(1,x,x.ge(2).float())
            total.scatter_add_(1,x,frac*x.ge(2))
            count,total=count[:,2:],total[:,2:]
            prior=self.prior[None].expand(b,-1,-1)
            logits=self.prior_logits[None].expand(b,-1,-1)
            dur=self.duration[None].expand(b,-1)
        else:
            match=x.eq(actions[:,None]).float()
            count=match.sum(1,keepdim=True); total=(match*frac).sum(1,keepdim=True)
            prior=self.prior[:,actions-2].T[:,:,None]
            logits=self.prior_logits[:,actions-2].T[:,:,None]
            dur=self.duration[actions-2,None]
        repeat=count/(count+1)
        relative=(total-count*prior[:,0])/(count+1)
        duration=torch.log1p(dur)/math.log1p(600)-.8
        historymean=s['past_sum']/s['past_count'].clamp_min(1)
        candidate=torch.stack([repeat,relative,duration,duration*(historymean[:,None]-.5)],1)
        adjustment=torch.einsum('bcv,hc->bhv',candidate,self.coeff)
        result=torch.sigmoid(logits*self.scale[None,:,None]+self.context(features(s))[:,:,None]+adjustment)
        return result if actions is None else result.squeeze(-1)


def rewards(pred,duration,kind,generated=None):
    fraction=pred[:,0]; time=fraction*duration.clamp(0,600)/300
    if kind=='fraction': return fraction
    if kind=='time': return time
    value=.5*fraction+.25*time+.25*pred[:,1]
    return value if generated is None else value-.1*generated.float()


def transition(s,action,pred_fraction,duration,vocab):
    """Accepted recommendation, expected feedback; deliberately synthetic."""
    out={}
    for key,value in [('x',action+2),('hfraction',pred_fraction),('hduration',duration),('hgap',s['request_gap'])]:
        out[key],lengths=append_action(s[key],s['lengths'],value)
    out['lengths']=lengths
    out['past_sum']=s['past_sum']+pred_fraction
    out['past_count']=s['past_count']+1
    out['request_gap']=((pred_fraction*duration).clamp(5,600)/5).round().clamp_min(1)*5
    gen=s.get('generated',torch.zeros((len(action),vocab),dtype=torch.bool)).clone()
    gen.scatter_(1,action[:,None],True); out['generated']=gen
    return out


def regularized_loss(log_actions,rs,expectations,kls,beta,gamma):
    """Score-function for future state costs AND direct state-KL derivative."""
    ret=torch.zeros_like(rs[0]); terms=[]; horizon=len(rs)
    for h in reversed(range(horizon)):
        ret=rs[h]-beta*kls[h].detach()+gamma*ret
        scale=sum(gamma**j for j in range(horizon-h))
        baseline=(expectations[h]-beta*kls[h].detach())*scale
        terms.append((gamma**h)*(-(ret-baseline).detach()*log_actions[h]+beta*kls[h]))
    return torch.stack(terms).mean()
