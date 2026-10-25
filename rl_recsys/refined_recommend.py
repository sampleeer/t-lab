"""Inspect final recommendations, using the exact causal state and catalog."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from .data import ROOT
from .refined_data import OUT,REPORT
from .refined_models import ContextReward,ResearchPolicy,state
from .refined_experiment import ARCH,catalog,base_policy,restore,load,lp_for


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='colab_sft')
    parser.add_argument('--split',choices=['test','extra','validation'],default='test')
    parser.add_argument('--example-index',type=int,default=0)
    parser.add_argument('--save',type=str)
    args=parser.parse_args(); torch.set_num_threads(4)
    lock=json.loads((REPORT/'selection.json').read_text()); choices={m['name']:m for m in lock['models']}
    if args.model not in choices: parser.error('Available models: '+', '.join(choices))
    variant=choices[args.model]; cat=catalog(); base=base_policy(cat)
    model=base if variant['type'] in ('base','rerank') else restore(ResearchPolicy(ARCH,len(cat['duration'])),ROOT/variant['path'])
    rm=restore(ContextReward(cat,lock['rm_hidden']),OUT/'reward.pt')
    data=load(args.split)
    if not 0<=args.example_index<len(data['y']): parser.error('Example index out of range')
    s=state(data,slice(args.example_index,args.example_index+1))
    with torch.no_grad():
        lp=lp_for(model,s,base,rm,cat,variant); top=lp[0].topk(10).indices
        r=rm(s)[0]; tokens=top.numpy()+2
        table=pd.DataFrame({'rank':np.arange(1,11),'token':tokens,'item_id':cat['item_id'][tokens],
            'policy_probability':lp[0,top].exp().numpy(),'model_fraction':r[0,top].numpy(),
            'model_continuation':r[1,top].numpy(),'train_duration_seconds':cat['duration'][tokens],
            'seen_in_last24':np.isin(tokens,s['x'].numpy()[0]),'matches_logged_next':tokens==data['y'][args.example_index]})
    print(f"Model={args.model}; split={args.split}; user={data['user'][args.example_index]}; timestamp={data['timestamp'][args.example_index]}")
    print(table.to_string(index=False))
    if args.save: table.to_csv(args.save,index=False)


if __name__=='__main__': main()
