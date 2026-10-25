"""Post-selection descriptive diagnostics; no training/model selection."""
import json
import numpy as np
import pandas as pd
import torch
from .data import ROOT,dump
from .refined_data import REPORT,OUT
from .refined_experiment import load,catalog,base_policy,restore
from .refined_models import ContextReward,state


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    lock=json.loads((REPORT/'selection.json').read_text()); cat=catalog()
    rm=restore(ContextReward(cat,lock['rm_hidden']),OUT/'reward.pt'); base=base_policy(cat)
    result={}
    for split in ('test','extra'):
        d=load(split); means=[]; stds=[]; lp=[]
        for start in range(0,len(d['y']),128):
            s=state(d,slice(start,start+128)); p=base(s).softmax(-1); frac=rm(s)[:,0]
            mu=(p*frac).sum(1); sd=((p*(frac-mu[:,None]).square()).sum(1)).sqrt()
            means.extend(mu.numpy()); stds.extend(sd.numpy())
        peruser=pd.DataFrame({'mean':means,'within_state_action_sd':stds,'user':d['user']}).groupby('user').mean()
        idx=np.flatnonzero(d['y']>=2); shuffled=np.random.default_rng(416).permutation(d['y'][idx]); pred=[]; actualpred=[]
        for start in range(0,len(idx),512):
            rows=idx[start:start+512]; s=state(d,rows)
            pred.extend(rm(s,torch.as_tensor(shuffled[start:start+512]))[:,0].numpy())
            actualpred.extend(rm(s,torch.as_tensor(d['y'][rows]))[:,0].numpy())
        truth=d['fraction'][idx]; pp=np.array(actualpred)
        result[split]={'mean_policy_weighted_action_sd':float(peruser.within_state_action_sd.mean()),
            'between_user_expected_fraction_sd':float(peruser['mean'].std(ddof=0)),
            'fraction_rmse_actions_shuffled':float(np.sqrt(np.mean((np.array(pred)-truth)**2))),
            'fraction_rmse_real_actions':float(np.sqrt(np.mean((pp-truth)**2)))}
        history=d['hfraction'][idx]; lengths=d['lengths'][idx]
        recent=np.arange(history.shape[1])[None,:]>=(lengths-3).clip(0)[:,None]
        recent=recent*(np.arange(history.shape[1])[None,:]<lengths[:,None])
        estimates={'last_fraction':history[np.arange(len(idx)),lengths-1],
            'last3_mean':(history*recent).sum(1)/recent.sum(1),
            'last24_mean':history.sum(1)/lengths,
            'past_all_mean':d['past_sum'][idx]/d['past_count'][idx]}
        result[split]['history_baseline_rmse']={k:float(np.sqrt(np.mean((v-truth)**2))) for k,v in estimates.items()}
        bins=pd.qcut(pp,10,duplicates='drop')
        pd.DataFrame({'bin':bins,'prediction':pp,'observed':truth}).groupby('bin',observed=True).agg(
            count=('prediction','size'),prediction=('prediction','mean'),observed=('observed','mean')).to_csv(REPORT/f'{split}_rm_calibration.csv')
    dump(REPORT/'diagnostics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
