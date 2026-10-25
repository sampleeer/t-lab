"""Recompute selected-model metrics and check saved outputs without training."""
import copy
import json

import numpy as np
import pandas as pd

from .data import ROOT, dump, load_split, digest
from .experiment import evaluate, restore
from .models import Policy, RewardModel, seed_everything


def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    manifest=json.loads((ROOT/'reports/fit_manifest.json').read_text())
    choices=json.loads((ROOT/'reports/selection.json').read_text())
    saved=pd.read_csv(ROOT/'reports/test_metrics.csv').set_index(['seed','model'])
    data=load_split('test');catalog=dict(np.load(ROOT/'artifacts/catalog.npz'))
    checks={'config_hash':digest(ROOT/'config.json')==manifest['config_sha256'],
            'training_source_hashes':{name:digest(ROOT/name)==manifest['source_sha256'][name]
                                     for name in ['data.py','models.py','experiment.py']},
            'models':{}}
    for seed in cfg['seeds']:
        seed_everything(seed,cfg['threads'])
        folder=ROOT/f'artifacts/seed_{seed}'
        base=restore(Policy(cfg,len(catalog['duration'])),folder/'sft.pt').eval().requires_grad_(False)
        reward=restore(RewardModel(len(catalog['duration']),catalog['item_fraction'],catalog['item_continuation']),folder/'reward.pt').eval().requires_grad_(False)
        auditor=restore(copy.deepcopy(reward),folder/'auditor.pt').eval().requires_grad_(False)
        for name,kind in [('sft','fraction')]+[(v['name']+'_selected',v['reward']) for v in choices[str(seed)]]:
            model=restore(copy.deepcopy(base),folder/f'{name}.pt').eval()
            scores,users,top=evaluate(model,base,reward,auditor,data,catalog,cfg,kind)
            differences={key:abs(float(saved.loc[(seed,name),key])-float(value))
                         for key,value in scores.items()}
            max_diff=max(differences.values())
            top_equal=bool(np.array_equal(top,np.load(folder/f'{name}_top10.npy')))
            users_saved=pd.read_csv(folder/f'{name}_users.csv',index_col='user')
            # Pandas emits shortest round-trip FLOAT32 decimals; reading CSV
            # as float64 does not reconstruct the original binary float32.
            # Restore each stored column's dtype before comparing values.
            users_saved=users_saved.astype(users.dtypes.to_dict())
            user_delta=float(np.max(np.abs(users.to_numpy()-users_saved.to_numpy())))
            checkpoints=[]
            if name!='sft':
                variant=next(v for v in choices[str(seed)] if v['name']+'_selected'==name)
                history=json.loads((folder/f"{variant['name']}_validation.json").read_text())
                valid=[h for h in history if h['eligible']]
                expected=max(valid,key=lambda h:h['objective'])
                checkpoints.append(expected['step']==variant['selected_step'])
            passed=max_diff<1e-10 and user_delta<1e-10 and top_equal and all(checkpoints)
            checks['models'][f'{seed}/{name}']={'passed':bool(passed),'max_metric_difference':max_diff,
                'max_user_metric_difference':user_delta,'top10_identical':top_equal,
                'validation_selection_matches':all(checkpoints)}
    passed=checks['config_hash'] and all(checks['training_source_hashes'].values()) and all(x['passed'] for x in checks['models'].values())
    checks['passed']=passed
    dump(ROOT/'reports/reproduction_checks.json',checks)
    print(json.dumps(checks,indent=2))
    if not passed: raise SystemExit('Reproduction checks failed')


if __name__=='__main__': main()
