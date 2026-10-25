"""Verify locked choices, source/weights, cohorts, and recomputed final metrics."""
import json
import numpy as np
import pandas as pd
import torch
from .data import ROOT,dump,digest
from .refined_data import REPORT,OUT,RECEIVED
from .refined_experiment import CFG,ARCH,load,catalog,base_policy,restore,evaluate
from .refined_models import ContextReward,ResearchPolicy


def main():
    torch.set_num_threads(CFG['threads'])
    lock=json.loads((REPORT/'selection.json').read_text()); cat=catalog()
    checks={'sources':{p:digest(ROOT/p)==s for p,s in lock['source_sha256'].items()},
            'weights':{p:digest(ROOT/p)==s for p,s in lock['sha256'].items()},'choices':{},'metrics':{}}
    checks['original_colab_unchanged']=digest(RECEIVED/'best.pt')=='69210ed9aa7719895b1c3d81466bf0c29aa4f325cce461557253dea06c3fa062'
    train=load('train'); extra=load('extra')
    data_checks=json.loads((REPORT/'data_checks.json').read_text())
    checks['prepared_inputs']={name:digest(OUT/f'{name}.npz')==data_checks[name]['sha256'] for name in ('train','validation','test')}
    extra_checks=json.loads((REPORT/'extra_data_checks.json').read_text())
    checks['prepared_inputs']['extra']=digest(OUT/'extra.npz')==extra_checks['sha256']
    checks['train_before_cut']=bool(train['timestamp'].max()<19934415-1800)
    source=ROOT/'data/listens.parquet'
    if source.exists():
        checks['source_parquet']=digest(source)==json.loads((ROOT/'reports/source_manifest.json').read_text())['sha256']
    checks['no_extra_train_user_overlap']=bool(not len(np.intersect1d(train['user'],extra['user'])))
    checks['selection_locked_before_extra']=json.loads((REPORT/'extra_data_checks.json').read_text())['selection_lock_sha256']==digest(REPORT/'selection.json')
    base=base_policy(cat).requires_grad_(False)
    rm=restore(ContextReward(cat,lock['rm_hidden']),OUT/'reward.pt').requires_grad_(False)
    aud=restore(ContextReward(cat,lock['rm_hidden']),OUT/'auditor.pt').requires_grad_(False)
    saved=json.loads((REPORT/'metrics.json').read_text())['extra']
    for v in lock['models']:
        name=v['name']
        if v['type'] in ('rl','continued_sft'):
            log=json.loads((OUT/name/'history.json').read_text())
            valid=[r for r in log if r['eligible']]; expected=max(valid,key=lambda r:r['selection_value'])
            flags=all(bool(r['eligible'])==(v['type']=='continued_sft' or r['ndcg10']>=v['validation_ndcg_floor']) for r in log)
            checks['choices'][name]=bool(v['selected_step']==expected['step'] and flags)
        elif v['type']=='rerank' and name.endswith('_selected'):
            grid=json.loads((REPORT/'rerank_validation.json').read_text())
            valid=[r for r in grid if r['reward']==v['reward'] and r['eligible']]
            expected=max(valid,key=lambda r:r['reward_'+v['reward']])
            checks['choices'][name]=bool(v['beta']==expected['beta'])
        model=base if v['type'] in ('base','rerank') else restore(ResearchPolicy(ARCH,len(cat['duration'])),ROOT/v['path'])
        summary,users,_,top=evaluate(model,base,rm,aud,extra,cat,v)
        metric_diff=max(abs(summary[k]-saved[name][k]) for k in summary)
        top_equal=bool(np.array_equal(top,np.load(OUT/f'extra_{name}_top10.npy')))
        user_saved=pd.read_csv(REPORT/f'extra_{name}_users.csv',index_col='user').astype(users.dtypes.to_dict())
        user_diff=float(np.max(np.abs(users.to_numpy()-user_saved.to_numpy())))
        checks['metrics'][name]={'max_difference':metric_diff,'top10_identical':top_equal,
            'max_user_difference':user_diff,'passed':bool(metric_diff<1e-10 and top_equal and user_diff<1e-10)}
        print('Verified',name,metric_diff,flush=True)
    checks['passed']=bool(all(checks['sources'].values()) and all(checks['weights'].values()) and all(checks['choices'].values())
        and all(v['passed'] for v in checks['metrics'].values()) and checks['original_colab_unchanged']
        and checks['selection_locked_before_extra'] and checks['no_extra_train_user_overlap']
        and all(checks['prepared_inputs'].values()) and checks['train_before_cut'] and checks.get('source_parquet',True))
    dump(REPORT/'verification.json',checks)
    if not checks['passed']: raise SystemExit('Final verification failed')


if __name__=='__main__': main()
