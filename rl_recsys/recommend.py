"""Inspect real item recommendations for one saved held-out context."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ROOT, load_split
from .experiment import restore
from .models import Policy, RewardModel, action_logits, seed_everything


def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    choices=['sft']+[v['name']+'_'+s for v in cfg['variants'] for s in ['selected','endpoint']]
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=choices,default='fraction_kl_selected')
    parser.add_argument('--seed',type=int,choices=cfg['seeds'],default=42)
    parser.add_argument('--example-index',type=int,default=0)
    parser.add_argument('--save',type=Path)
    args=parser.parse_args()
    seed_everything(args.seed,cfg['threads'])
    data=load_split('test');i=args.example_index
    if not 0<=i<len(data['y']): parser.error('example-index outside saved test contexts')
    catalog=dict(np.load(ROOT/'artifacts/catalog.npz'))
    folder=ROOT/f'artifacts/seed_{args.seed}'
    policy=restore(Policy(cfg,len(catalog['duration'])),folder/f'{args.model}.pt').eval()
    reward=restore(RewardModel(len(catalog['duration']),catalog['item_fraction'],catalog['item_continuation']),folder/'reward.pt').eval()
    x=torch.as_tensor(data['x'][i:i+1]);lengths=torch.as_tensor(data['lengths'][i:i+1])
    with torch.no_grad():
        probabilities=action_logits(policy(x,lengths)).softmax(-1)
        score,action=probabilities.topk(10)
        predictions=reward(x,lengths)[0]
    token=action[0].numpy()+2
    table=pd.DataFrame({'rank':np.arange(1,11),'item_id':catalog['item_id'][token],
        'probability':score[0].numpy(),'predicted_fraction':predictions[0,action[0]].numpy(),
        'predicted_continuation':predictions[1,action[0]].numpy(),
        'train_duration_seconds':catalog['duration'][token],
        'in_recent_history':[bool(t in data['x'][i]) for t in token]})
    print(f"user={data['user'][i]}, timestamp={data['timestamp'][i]}, example={i}, model={args.model}")
    print(table.to_string(index=False))
    if args.save:
        args.save.parent.mkdir(parents=True,exist_ok=True)
        table.to_csv(args.save,index=False)


if __name__=='__main__': main()
