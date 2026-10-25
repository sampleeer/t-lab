"""Reproducible final experiment. Run fit before evaluate (new-user lock)."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from .data import ROOT,dump,digest
from .models import seed_everything
from .experiment import ranking,macro,bootstrap_delta
from .refined_data import OUT,REPORT,INPUT,RECEIVED,prepare
from .refined_models import ResearchPolicy,ContextReward,state,features,rewards,transition,regularized_loss

CFG=json.loads((ROOT/'refined_config.json').read_text())
ARCH=json.loads((RECEIVED/'architecture.json').read_text())


def load(name):
    with np.load(OUT/f'{name}.npz') as z: return dict(z)


def catalog():
    with np.load(INPUT/'catalog.npz') as z: return dict(z)


def save(path,model):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing'); torch.save(model.state_dict(),temporary); temporary.replace(path)


def restore(model,path):
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); return model.eval()


def base_policy(cat):
    model=ResearchPolicy(ARCH,len(cat['duration']))
    result=model.load_state_dict(torch.load(RECEIVED/'best.pt',map_location='cpu',weights_only=True),strict=False)
    assert set(result.missing_keys)=={'feedback_adapter.weight','feedback_adapter.bias'} and not result.unexpected_keys
    return model.eval()


@torch.no_grad()
def response(model,data,shuffle=False):
    idx=np.flatnonzero(data['y']>=2); other=np.random.default_rng(905).permutation(idx) if shuffle else idx
    pred=[]
    for start in range(0,len(idx),512):
        rows=idx[start:start+512]; contexts=other[start:start+512]
        pred.append(model(state(data,contexts),torch.as_tensor(data['y'][rows])).numpy())
    return idx,np.concatenate(pred)


def rm_metrics(model,data,shuffle=False):
    idx,p=response(model,data,shuffle); y=np.stack([data['fraction'][idx],data['continuation'][idx]],1)
    return {'joint_mse':float(np.mean((p-y)**2)), 'fraction_rmse':float(np.sqrt(np.mean((p[:,0]-y[:,0])**2))),
            'fraction_mae':float(np.abs(p[:,0]-y[:,0]).mean()),'continuation_brier':float(np.mean((p[:,1]-y[:,1])**2)),
            'continuation_auc':float(roc_auc_score(y[:,1],p[:,1])) if len(np.unique(y[:,1]))>1 else None,
            'events':len(idx)}


def response_intervals(model,data):
    idx,pred=response(model,data); _,shuffled=response(model,data,True)
    y=data['fraction'][idx]; item=model.prior[0,data['y'][idx]-2].numpy()
    table=macro({'model':(pred[:,0]-y)**2,'item':(item-y)**2,'shuffle':(shuffled[:,0]-y)**2},data['user'][idx])
    rng=np.random.default_rng(42); output={}
    for name in ('item','shuffle'):
        diff=(table.model-table[name]).to_numpy()
        samples=diff[rng.integers(0,len(diff),(CFG['bootstrap_repetitions'],len(diff)))].mean(1)
        output[f'fraction_mse_model_minus_{name}']={'delta':float(diff.mean()),'lower95':float(np.quantile(samples,.025)),
            'upper95':float(np.quantile(samples,.975)),'users':len(diff),'aggregation':'equal-user MSE'}
    return output


def train_rm(train,val,cat):
    selections=[]
    for hidden in CFG['rm_hidden_candidates']:
        seed_everything(42,CFG['threads']); m=ContextReward(cat,hidden)
        opt=torch.optim.AdamW(m.parameters(),lr=CFG['rm_lr'],weight_decay=.01)
        path=OUT/f'rm_h{hidden}.pt'; best=rm_metrics(m,val)['joint_mse']; save(path,m)
        logs=[{'epoch':0,**rm_metrics(m,val)}]; chosen=0; rng=np.random.default_rng(42)
        for epoch in range(1,CFG['rm_epochs']+1):
            order=rng.permutation(np.arange(0,len(train['y']),2)); total=0.
            for start in range(0,len(order),CFG['rm_batch_size']):
                ix=order[start:start+CFG['rm_batch_size']]
                targets=torch.tensor(np.stack([train['fraction'][ix],train['continuation'][ix]],1))
                loss=F.mse_loss(m(state(train,ix),torch.as_tensor(train['y'][ix])),targets)
                opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
            metrics=rm_metrics(m,val); logs.append(dict(epoch=epoch,train_mse=total/len(order),**metrics))
            print('RM',hidden,logs[-1],flush=True)
            if metrics['joint_mse']<best: best=metrics['joint_mse']; chosen=epoch; save(path,m)
        dump(REPORT/f'rm_h{hidden}_training.json',logs)
        selections.append({'hidden':hidden,'epoch':chosen,'joint_mse':best,'path':str(path.relative_to(ROOT))})
    selected=min(selections,key=lambda r:r['joint_mse']); hidden=selected['hidden']
    m=restore(ContextReward(cat,hidden),ROOT/selected['path']); save(OUT/'reward.pt',m)
    seed_everything(2026,CFG['threads']); a=ContextReward(cat,hidden)
    opt=torch.optim.AdamW(a.parameters(),lr=CFG['rm_lr'],weight_decay=.01)
    best=rm_metrics(a,val)['joint_mse']; save(OUT/'auditor.pt',a)
    alogs=[{'epoch':0,**rm_metrics(a,val)}]; rng=np.random.default_rng(2026); ae=0
    for epoch in range(1,CFG['rm_epochs']+1):
        order=rng.permutation(np.arange(1,len(train['y']),2))
        for start in range(0,len(order),CFG['rm_batch_size']):
            ix=order[start:start+CFG['rm_batch_size']]
            targets=torch.tensor(np.stack([train['fraction'][ix],train['continuation'][ix]],1))
            loss=F.mse_loss(a(state(train,ix),torch.as_tensor(train['y'][ix])),targets)
            opt.zero_grad(); loss.backward(); opt.step()
        metrics=rm_metrics(a,val); alogs.append(dict(epoch=epoch,**metrics)); print('Auditor',epoch,metrics,flush=True)
        if metrics['joint_mse']<best: best=metrics['joint_mse']; ae=epoch; save(OUT/'auditor.pt',a)
    dump(REPORT/'auditor_training.json',alogs)
    meta={'architectures':selections,'selected':selected,'auditor_epoch':ae,
          'validation':{'item_mean':rm_metrics(ContextReward(cat,0),val),'reward':rm_metrics(m,val),
                        'shuffled_history':rm_metrics(m,val,True)}}
    dump(REPORT/'rm_selection.json',meta)
    return m.requires_grad_(False),restore(a,OUT/'auditor.pt').requires_grad_(False)


def lp_for(model,s,reference,rm,cat,variant=None):
    if variant is not None and variant.get('type')=='rerank':
        ref=reference(s)
        if variant['beta'] is None: return ref.log_softmax(-1)
        r=rewards(rm(s),rm.duration,variant['reward'],s.get('generated'))
        return (ref+r/variant['beta']).log_softmax(-1)
    return model(s).log_softmax(-1)


@torch.no_grad()
def evaluate(model,reference,rm,aud,data,cat,variant=None):
    chunks=[]; tops=[]; model.eval()
    for start in range(0,len(data['y']),CFG['eval_batch_size']):
        ix=slice(start,start+CFG['eval_batch_size']); s=state(data,ix)
        lp=lp_for(model,s,reference,rm,cat,variant); p=lp.exp(); ref=reference(s).log_softmax(-1)
        r=rm(s); ar=aud(s); top=p.topk(10).indices; tokens=top+2
        vals=ranking(tokens.numpy(),data['y'][ix])
        seen=(tokens[:,:,None]==s['x'][:,None,:]).any(-1).float()
        vals.update(pred_fraction=(p*r[:,0]).sum(1).numpy(),audit_fraction=(p*ar[:,0]).sum(1).numpy(),
                    pred_seconds=(p*r[:,0]*rm.duration).sum(1).numpy(),pred_continuation=(p*r[:,1]).sum(1).numpy(),
                    kl=(p*(lp-ref)).sum(1).numpy(),entropy=-(p*lp).sum(1).numpy(),
                    duration_seconds=(p*rm.duration).sum(1).numpy(),
                    top10_fraction=r[:,0].gather(1,top).mean(1).numpy(),
                    recent_repeat_top1=seen[:,0].numpy(),recent_repeat_top10=seen.mean(1).numpy(),
                    known_history=s['x'].ge(2).float().sum(1).div(s['lengths']).numpy())
        for kind in ('fraction','time','balanced'):
            rr=rewards(r,rm.duration,kind)
            vals[f'reward_{kind}']=(p*rr).sum(1).numpy()
            vals[f'top10_{kind}']=rr.gather(1,top).mean(1).numpy()
        chunks.append(pd.DataFrame(vals)); tops.append(tokens.numpy())
    events=pd.concat(chunks,ignore_index=True); users=macro(events,data['user'])
    summary={k:float(v) for k,v in users.mean().items()}
    top=np.concatenate(tops); counts=np.bincount(top.ravel(),minlength=len(cat['duration']))[2:]
    first=np.bincount(top[:,0],minlength=len(cat['duration']))[2:]
    summary.update(catalog_coverage_top10=float((counts>0).mean()),top1_max_share=float(first.max()/len(top)),
                   top10_hhi=float(((counts/counts.sum())**2).sum()),users=len(users),events=len(events))
    return summary,users,events,top


@torch.no_grad()
def horizon_value(model,base,rm,data,cat,variant):
    if variant['horizon']==1:
        values=[]; users=[]
        for start in range(0,len(data['y']),CFG['eval_batch_size']):
            ix=slice(start,start+CFG['eval_batch_size']); s=state(data,ix)
            p=lp_for(model,s,base,rm,cat,variant).exp()
            values.extend((p*rewards(rm(s),rm.duration,variant['reward'])).sum(1).numpy()); users.extend(data['user'][ix])
        return float(macro({'r':values},users).r.mean())
    ix=np.sort(np.random.default_rng(681).choice(len(data['y']),min(len(data['y']),CFG['rollout_validation_contexts']),replace=False))
    ix=np.repeat(ix,CFG['rollout_validation_samples']); results=[]
    with torch.random.fork_rng():
        torch.manual_seed(7361)
        for start in range(0,len(ix),CFG['eval_batch_size']):
            s=state(data,ix[start:start+CFG['eval_batch_size']]); value=torch.zeros(len(s['x']))
            for h in range(variant['horizon']):
                lp=lp_for(model,s,base,rm,cat,variant); a=torch.multinomial(lp.exp(),1).squeeze(1)
                pred=rm(s); rr=rewards(pred,rm.duration,variant['reward'],s.get('generated'))
                value+=(CFG['gamma']**h)*rr.gather(1,a[:,None]).squeeze(1)
                s=transition(s,a,pred[:,0].gather(1,a[:,None]).squeeze(1),rm.duration[a],len(rm.duration))
            results.extend(value.numpy())
    return float(macro({'r':results},data['user'][ix]).r.mean())


def train_actor(base,rm,aud,train,val,cat,variant,seed):
    seed_everything(seed,CFG['threads']); rng=np.random.default_rng(seed)
    model=copy.deepcopy(base).requires_grad_(True); opt=torch.optim.AdamW(model.parameters(),lr=CFG['rl_lr'],weight_decay=0.)
    name=f"{variant['name']}_seed{seed}"; folder=OUT/name; folder.mkdir(exist_ok=True)
    supervised=variant.get('type')=='continued_sft'; h=variant['horizon']
    if (folder/'temp.pt').exists():
        cached=torch.load(folder/'temp.pt',weights_only=True,map_location='cpu')
        assert cached['variant']==variant and cached['seed']==seed
        if cached['step']==CFG['rl_steps']:
            print('Reusing completed',name,flush=True)
            return dict(variant, name=name,seed=seed,selected_step=cached['best_step'],selection_value=cached['best_value'],
                validation_ndcg_floor=CFG['validation_ndcg_floor']*evaluate(base,base,rm,aud,val,cat)[0]['ndcg10'],
                path=str((folder/'best.pt').relative_to(ROOT)),endpoint_path=str((folder/'endpoint.pt').relative_to(ROOT)),
                seconds=cached.get('elapsed_seconds'),updates=CFG['rl_steps'],scored_states=CFG['rl_steps']*CFG['rl_batch_size']*h)
    initial=evaluate(base,base,rm,aud,val,cat)[0]; threshold=CFG['validation_ndcg_floor']*initial['ndcg10']
    best=initial['ndcg10'] if supervised else horizon_value(base,base,rm,val,cat,variant)
    selected=0; logs=[dict(step=0,**initial,selection_value=best,eligible=True)]
    save(folder/'best.pt',model); timer=time.monotonic()
    for step in range(1,CFG['rl_steps']+1):
        ix=rng.integers(0,len(train['y']),CFG['rl_batch_size']*(h if supervised else 1)); s=state(train,ix)
        if supervised:
            loss=F.cross_entropy(model(s),torch.as_tensor(train['y'][ix])-2)
        else:
            la=[]; rs=[]; es=[]; ks=[]
            for j in range(h):
                lp=model(s).log_softmax(-1); p=lp.exp()
                with torch.no_grad():
                    ref=base(s).log_softmax(-1); pred=rm(s)
                    r=rewards(pred,rm.duration,variant['reward'],s.get('generated'))
                    a=torch.multinomial(p.detach(),1).squeeze(1)
                k=(p*(lp-ref)).sum(1)
                la.append(lp.gather(1,a[:,None]).squeeze(1)); rs.append(r.gather(1,a[:,None]).squeeze(1))
                es.append((p.detach()*r).sum(1)); ks.append(k)
                s=transition(s,a,pred[:,0].gather(1,a[:,None]).squeeze(1),rm.duration[a],len(rm.duration))
            loss=regularized_loss(la,rs,es,ks,variant['beta'],CFG['gamma'])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if step % CFG['eval_every']==0 or step==CFG['rl_steps']:
            score=evaluate(model,base,rm,aud,val,cat)[0]
            value=score['ndcg10'] if supervised else horizon_value(model,base,rm,val,cat,variant)
            eligible=supervised or score['ndcg10']>=threshold
            logs.append(dict(step=step,**score,selection_value=value,eligible=bool(eligible),loss=float(loss.detach())))
            if eligible and value>best: best=value; selected=step; save(folder/'best.pt',model)
            save(folder/'endpoint.pt',model)
            temp=folder/'temp.writing'; torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
                'best_step':selected,'best_value':best,'variant':variant,'seed':seed,'torch_rng':torch.get_rng_state(),
                'numpy_rng_json':json.dumps(rng.bit_generator.state),'elapsed_seconds':time.monotonic()-timer},temp); temp.replace(folder/'temp.pt')
            dump(folder/'history.json',logs)
            print(name,step,'NDCG',round(score['ndcg10'],6),'return',round(value,5),'eligible',eligible,flush=True)
    return dict(variant,name=name,seed=seed,selected_step=selected,selection_value=best,
                validation_ndcg_floor=threshold,path=str((folder/'best.pt').relative_to(ROOT)),
                endpoint_path=str((folder/'endpoint.pt').relative_to(ROOT)),
                seconds=time.monotonic()-timer,updates=CFG['rl_steps'],scored_states=CFG['rl_steps']*CFG['rl_batch_size']*h)


def fit():
    start=time.monotonic(); seed_everything(42,CFG['threads'])
    train,val=load('train'),load('validation'); cat=catalog()
    if (REPORT/'rm_selection.json').exists():
        hidden=json.loads((REPORT/'rm_selection.json').read_text())['selected']['hidden']
        rm=restore(ContextReward(cat,hidden),OUT/'reward.pt').requires_grad_(False)
        aud=restore(ContextReward(cat,hidden),OUT/'auditor.pt').requires_grad_(False)
    else:
        rm,aud=train_rm(train,val,cat)
    base=base_policy(cat).requires_grad_(False)
    selections=[{'name':'colab_sft','type':'base','path':str((RECEIVED/'best.pt').relative_to(ROOT))}]
    for seed in CFG['rl_seeds']:
        for h in (1,5):
            variant={'name':f'continued_sft_h{h}','type':'continued_sft','horizon':h}
            selections.append(train_actor(base,rm,aud,train,val,cat,variant,seed))
        for variant in CFG['variants']:
            selections.append(train_actor(base,rm,aud,train,val,cat,dict(type='rl',**variant),seed))
    floor=evaluate(base,base,rm,aud,val,cat)[0]['ndcg10']*CFG['validation_ndcg_floor']
    grid=[]
    for kind in ('fraction','time','balanced'):
        candidates=[]
        for beta in [None]+CFG['rerank_betas']:
            v={'name':f'rerank_{kind}_{beta}','type':'rerank','reward':kind,'beta':beta,'horizon':1}
            score=evaluate(base,base,rm,aud,val,cat,v)[0]
            row=dict(**v,**score,eligible=bool(score['ndcg10']>=floor)); grid.append(row)
            if row['eligible']: candidates.append(row)
        chosen=max(candidates,key=lambda r:r[f'reward_{kind}'])
        selections.append({'name':f'rerank_{kind}_selected','type':'rerank','reward':kind,'beta':chosen['beta'],'horizon':1})
        selections.append({'name':f'rerank_{kind}_fixed','type':'rerank','reward':kind,'beta':.2,'horizon':1})
    dump(REPORT/'rerank_validation.json',grid)
    sourcefiles=['refined_data.py','refined_models.py','refined_experiment.py','refined_config.json','reports/refined/PROTOCOL.md']
    lock={'models':selections,'rm_hidden':json.loads((REPORT/'rm_selection.json').read_text())['selected']['hidden'],
          'sha256':{r['path']:digest(ROOT/r['path']) for r in selections if 'path'in r},
          'source_sha256':{s:digest(ROOT/s) for s in sourcefiles},'reward_sha256':digest(OUT/'reward.pt'),
          'auditor_sha256':digest(OUT/'auditor.pt'),'fit_seconds':time.monotonic()-start,
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform()},
          'extra_cohort_not_evaluated':True}
    dump(REPORT/'selection.json',lock)
    print('SELECTION LOCKED',lock['fit_seconds'],flush=True)


@torch.no_grad()
def rollouts(model,base,rm,aud,data,cat,variant,name,split):
    ix=np.sort(np.random.default_rng(791).choice(len(data['y']),min(CFG['rollout_test_contexts'],len(data['y'])),replace=False))
    records=[]
    with torch.random.fork_rng():
        torch.manual_seed(202642)
        for start in range(0,len(ix),128):
            rows=ix[start:start+128]; original=state(data,rows); s=original
            frozen=rm(original); frozenlp=base(original).log_softmax(-1)
            for h in range(5):
                lp=lp_for(model,s,base,rm,cat,variant); a=torch.multinomial(lp.exp(),1).squeeze(1)
                pred=rm(s); ap=aud(s); ref=base(s).log_softmax(-1); col=a[:,None]
                rr=rewards(pred,rm.duration,'balanced',s.get('generated')).gather(1,col).squeeze(1)
                generated=s.get('generated',torch.zeros_like(lp,dtype=torch.bool)).gather(1,col).squeeze(1)
                for j in range(len(rows)):
                    records.append({'event':int(rows[j]),'user':int(data['user'][rows[j]]),'step':h+1,'token':int(a[j])+2,
                        'pred_fraction':float(pred[j,0,a[j]]),'audit_fraction':float(ap[j,0,a[j]]),
                        'frozen_fraction':float(frozen[j,0,a[j]]),'balanced':float(rr[j]),
                        'pred_seconds':float(pred[j,0,a[j]]*rm.duration[a[j]]),
                        'continuation':float(pred[j,1,a[j]]),'duration_seconds':float(rm.duration[a[j]]),
                        'repeat_generated':float(generated[j]),'repeat_original':float((original['x'][j]==a[j]+2).any()),
                        'sft_nll':float(-ref[j,a[j]]),'frozen_sft_nll':float(-frozenlp[j,a[j]])})
                s=transition(s,a,pred[:,0].gather(1,col).squeeze(1),rm.duration[a],len(rm.duration))
    events=pd.DataFrame(records); events.to_csv(REPORT/f'{split}_{name}_rollouts.csv',index=False)
    return events.drop(columns=['event','token']).groupby(['step','user']).mean().groupby('step').mean().reset_index().to_dict('records')


def evaluation():
    seed_everything(42,CFG['threads']); lock=json.loads((REPORT/'selection.json').read_text())
    for path,sha in lock['sha256'].items(): assert digest(ROOT/path)==sha
    assert digest(OUT/'reward.pt')==lock['reward_sha256'] and digest(OUT/'auditor.pt')==lock['auditor_sha256']
    if not (OUT/'extra.npz').exists(): prepare(extra=True)
    cat=catalog(); base=base_policy(cat).requires_grad_(False)
    rm=restore(ContextReward(cat,lock['rm_hidden']),OUT/'reward.pt').requires_grad_(False)
    aud=restore(ContextReward(cat,lock['rm_hidden']),OUT/'auditor.pt').requires_grad_(False)
    allmetrics={}; intervals={}; rms={}; rolls={}
    for split in ('test','extra'):
        data=load(split); summaries={}; user_tables={}; temporal=[]; rollout_summary={}
        item=ContextReward(cat,0)
        rms[split]={'item_mean':rm_metrics(item,data),'reward':rm_metrics(rm,data),
                    'auditor':rm_metrics(aud,data),'shuffled_history':rm_metrics(rm,data,True)}
        known=np.flatnonzero(data['y']>=2)
        p=data['past_sum'][known]/data['past_count'][known].clip(1)
        rms[split]['past_user_fraction_rmse']=float(np.sqrt(np.mean((p-data['fraction'][known])**2)))
        _,pr=response(rm,data); _,pa=response(aud,data)
        errors=pr[:,0]-data['fraction'][known]; ae=pa[:,0]-data['fraction'][known]
        rms[split]['rm_auditor_fraction_error_correlation']=float(np.corrcoef(errors,ae)[0,1])
        rms[split]['paired_intervals']=response_intervals(rm,data)
        for v in lock['models']:
            name=v['name']; model=base if v['type'] in ('base','rerank') else restore(ResearchPolicy(ARCH,len(cat['duration'])),ROOT/v['path'])
            score,users,events,top=evaluate(model,base,rm,aud,data,cat,v)
            summaries[name]=score; user_tables[name]=users
            users.to_csv(REPORT/f'{split}_{name}_users.csv'); np.save(OUT/f'{split}_{name}_top10.npy',top)
            for phase, rows in enumerate(np.array_split(np.argsort(data['timestamp'],kind='stable'),3),1):
                sub=macro(events.iloc[rows].reset_index(drop=True),data['user'][rows])
                temporal.append(dict(name=name,phase=phase,start=int(data['timestamp'][rows].min()),
                   end=int(data['timestamp'][rows].max()),users=len(sub),ndcg10=float(sub.ndcg10.mean()),recall10=float(sub.recall10.mean())))
            rollout_summary[name]=rollouts(model,base,rm,aud,data,cat,v,name,split)
            print('EVAL',split,name,'NDCG',round(score['ndcg10'],6),'reward',round(score['pred_fraction'],4),flush=True)
            # Endpoints expose rejected optimization attempts rather than hiding them.
            if v['type']=='rl':
                endpoint=restore(ResearchPolicy(ARCH,len(cat['duration'])),ROOT/v['endpoint_path'])
                es,eu,_,_=evaluate(endpoint,base,rm,aud,data,cat,v)
                summaries[name+'_endpoint']=es; eu.to_csv(REPORT/f'{split}_{name}_endpoint_users.csv')
        pd.DataFrame(summaries).T.to_csv(REPORT/f'{split}_metrics.csv')
        pd.DataFrame(temporal).to_csv(REPORT/f'{split}_temporal_strata.csv',index=False)
        allmetrics[split]=summaries; rolls[split]=rollout_summary
        intervals[split]={name:bootstrap_delta(user_tables['colab_sft'],u,CFG['bootstrap_repetitions'],42)
                          for name,u in user_tables.items() if name!='colab_sft'}
        # Direct matched H5-vs-H1 comparisons for the same reward/KL/seed.
        intervals[split]['matched_horizon']={str(seed):bootstrap_delta(user_tables[f'balanced_h1_seed{seed}'],
                  user_tables[f'balanced_h5_seed{seed}'],CFG['bootstrap_repetitions'],42) for seed in CFG['rl_seeds']}
    dump(REPORT/'metrics.json',allmetrics); dump(REPORT/'paired_intervals.json',intervals)
    dump(REPORT/'rm_test.json',rms); dump(REPORT/'rollout_summary.json',rolls)
    dump(REPORT/'completion.json',{'selection_lock_sha256':digest(REPORT/'selection.json'),'evaluation_completed':True,
        'extra_used_for_selection':False,'models_per_split':len(lock['models'])})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['fit','evaluate'])
    globals()['fit' if p.parse_args().command=='fit' else 'evaluation']()
