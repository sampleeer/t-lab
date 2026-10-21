import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import Snapshot, add_labels, targets
from .auto_features import AutoFeatureEngine, kernels
from .auto_pipeline import pair_hash
from .download import sha256
from .pipeline import read_frame, save_json, log, peak_memory_mb


def fold_dates(test, weeks=3):
    test=pd.Timestamp(test)
    validation=test-pd.Timedelta(days=7)
    train=[validation-pd.Timedelta(days=7*j) for j in range(weeks,0,-1)]
    return {'train':[str(t.date()) for t in train],
            'validation':str(validation.date()),'test':str(test.date())}


def key(role,date):
    return f'{role}_{date}'


def sample_training_pairs(frame, limit, seed):
    rng=np.random.default_rng(seed)
    keep=frame.label.to_numpy()==1
    weights=np.ones(len(frame),dtype=float)
    for indices in frame.groupby('user_id',sort=True).indices.values():
        negative=indices[frame.label.to_numpy()[indices]==0]
        n=min(limit,len(negative))
        if n:
            selected=rng.choice(negative,n,replace=False)
            keep[selected]=True
            weights[selected]=len(negative)/n
    result=frame.loc[keep].copy()
    result['sample_weight']=weights[keep]
    assert result.label.sum()==frame.label.sum()
    assert np.isclose(result.sample_weight.sum(),len(frame))
    return result.reset_index(drop=True)


class RankingContext:
    def __init__(self,frame,truth,k=10):
        self.uids=np.array(sorted(truth))
        self.group=np.searchsorted(self.uids,frame.user_id.to_numpy())
        assert (self.uids[self.group]==frame.user_id.to_numpy()).all()
        self.items=frame.item_id.to_numpy()
        self.labels=frame.label.to_numpy()
        self.ntruth=np.array([len(truth[u]) for u in self.uids])
        self.k=k
        self.discounts=np.r_[0,np.cumsum(1/np.log2(np.arange(k)+2))]
        self.ideal=self.discounts[np.minimum(self.ntruth,k)]
        found=np.bincount(self.group,weights=self.labels,minlength=len(self.uids)).astype(int)
        self.coverage=found/self.ntruth
        self.oracle=self.discounts[np.minimum(found,k)]/self.ideal

    def evaluate(self,scores):
        order=np.lexsort((self.items,-np.asarray(scores),self.group))
        g=self.group[order]
        start=np.r_[True,g[1:]!=g[:-1]]
        rank=np.arange(len(g))-np.maximum.accumulate(np.where(start,np.arange(len(g)),0))
        top=rank<self.k
        hits=self.labels[order][top]
        dcg=np.bincount(g[top],weights=hits/np.log2(rank[top]+2),minlength=len(self.uids))
        n_hits=np.bincount(g[top],weights=hits,minlength=len(self.uids))
        per_user=pd.DataFrame({'user_id':self.uids,'ndcg@10':dcg/self.ideal,
            'recall@10':n_hits/self.ntruth,'candidate_recall':self.coverage,
            'oracle_ndcg@10':self.oracle})
        return {**per_user.drop(columns='user_id').mean().to_dict(),'users':len(per_user)},per_user


def prepare(config,out):
    from time import perf_counter
    start=perf_counter()
    data=out/'data';data.mkdir(parents=True,exist_ok=True)
    baseline=Path('artifacts/baseline')
    original=json.loads((baseline/'manifest.json').read_text())
    auto=json.loads(Path('auto_config.json').read_text())
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        assert manifest['config']==config
    else:
        manifest={'config':config,'cases':{},'event_sha256':sha256(baseline/'sample_events.parquet'),
                  'baseline_manifest_sha256':sha256(baseline/'manifest.json')}
    events=pd.read_parquet(baseline/'sample_events.parquet')
    items=pd.read_csv('data/raw/items.csv');users=pd.read_csv('data/raw/users.csv')
    assert sha256(Path('data/raw/items.csv'))==original['data_sha256']['items.csv']
    assert manifest['event_sha256']==sha256(baseline/'sample_events.parquet')
    cases=set()
    for test in config['test_cutoffs']:
        dates=fold_dates(test,config['train_weeks'])
        cases.update(('train',d) for d in dates['train'])
        cases.update(('eval',dates[d]) for d in ['validation','test'])
    old_train={e['cutoff']:e['name'] for e in original['snapshots'] if e['name'].startswith('train')}
    old_eval={e['cutoff']:e['name'] for e in original['snapshots'] if not e['name'].startswith('train')}
    for role,date in sorted(cases):
        name=key(role,date)
        if name in manifest['cases']:continue
        log(f'Prepare {name}')
        lookup=old_train if role=='train' else old_eval
        if date in lookup:
            frame=read_frame(baseline/f'{lookup[date]}.parquet')
            truth=json.loads((baseline/f'{lookup[date]}_truth.json').read_text())
            audit={'reused_source':lookup[date]}
            original_rows=len(frame)
            if role=='train':
                frame=sample_training_pairs(frame,config['negatives_per_user'],
                                            config['seed']+pd.Timestamp(date).toordinal())
        else:
            snap=Snapshot(events,items,users,date,original['config']['candidates'])
            truth,audit=targets(events,date,snap.uids,snap.iids)
            if role=='train':
                date_seed=config['seed']+(pd.Timestamp(date)-pd.Timestamp('2021-07-19')).days//7
                chosen=np.sort(np.random.default_rng(date_seed).choice(snap.uids,config['train_users'],replace=False))
            else:chosen=np.array(sorted(truth))
            pairs=add_labels(snap.generate(chosen),truth)
            original_rows=len(pairs)
            if role=='train':pairs=sample_training_pairs(pairs,config['negatives_per_user'],config['seed']+pd.Timestamp(date).toordinal())
            frame=snap.features(pairs[['user_id','item_id','generator_rrf','generator_rank','cf_score']])
            frame['label']=pairs.label.to_numpy(dtype=np.int8)
            if role=='train':frame['sample_weight']=pairs.sample_weight.to_numpy()
            del snap,pairs
        for col in frame.select_dtypes(include='object'):
            frame[col]=frame[col].astype('category')
        frame.to_parquet(data/f'{name}.parquet',index=False)
        save_json(data/f'{name}_truth.json',{str(u):sorted(v) for u,v in truth.items()})
        engine=AutoFeatureEngine(events,items,date,frame[['user_id','item_id']])
        for kernel,kind,days in kernels(auto):
            block=engine.block(kernel,kind,days)
            assert np.isfinite(block.to_numpy()).all()
            block.to_parquet(data/f'{name}_{kernel}.parquet',index=False)
        manifest['cases'][name]={'cutoff':date,'role':role,'original_candidate_rows':original_rows,
            'rows':len(frame),'positive_candidates':int(frame.label.sum()),
            'weight_sum':float(frame.sample_weight.sum()) if role=='train' else None,
            'pair_hash':pair_hash(frame),'audit':audit}
        save_json(manifest_path,manifest)
        del frame,engine
        gc.collect()
    manifest['prepare_seconds']=perf_counter()-start
    manifest['prepare_peak_rss_mb']=peak_memory_mb()
    save_json(manifest_path,manifest)
