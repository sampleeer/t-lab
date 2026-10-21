import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from .auto_features import AutoFeatureEngine, kernels
from .download import sha256
from .pipeline import log, peak_memory_mb, read_frame, save_json, truth_for
from .metrics import ranking, bootstrap


def pair_hash(pairs):
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(
        pairs[['user_id','item_id']],index=False).to_numpy().tobytes()).hexdigest()


def build_snapshot(name, config, baseline, out):
    manifest = json.loads((baseline/'manifest.json').read_text())
    entry = next(x for x in manifest['snapshots'] if x['name']==name)
    pairs = pd.read_parquet(baseline/f'{name}.parquet',columns=['user_id','item_id'])
    engine = AutoFeatureEngine(pd.read_parquet(baseline/'sample_events.parquet'),
                               pd.read_csv('data/raw/items.csv'),entry['cutoff'],pairs)
    catalog = {}
    for kernel,kind,days in kernels(config):
        block = engine.block(kernel,kind,days)
        assert np.isfinite(block.to_numpy()).all() and len(block)==len(pairs)
        block.to_parquet(out/f'{name}_{kernel}.parquet',index=False)
        for col in block:
            catalog[col] = {'kernel':kernel,'kind':kind,'days':days,
                            'expression':col.removeprefix(f'auto_{kernel}_')}
    result = {'pair_hash':pair_hash(pairs),'rows':len(pairs),'cutoff':entry['cutoff']}
    return catalog,result


def prepare(config, baseline, out):
    start = time.perf_counter()
    bm = json.loads((baseline/'manifest.json').read_text())
    selected = json.loads((baseline/'selection.json').read_text())
    model_path = baseline/f"model_{selected['selected_model']}.cbm"
    result = {'config':config,'baseline_manifest_sha256':sha256(baseline/'manifest.json'),
              'baseline_model_sha256':sha256(model_path),'snapshots':{}}
    for entry in bm['snapshots']:
        if entry['name']=='test':
            continue
        log(f"Generate 144 features: {entry['name']}")
        catalog,snapshot = build_snapshot(entry['name'],config,baseline,out)
        result['snapshots'][entry['name']] = snapshot
        save_json(out/'catalog.json',catalog)
        save_json(out/'manifest.json',result)
        gc.collect()
    result['prepare_seconds'] = time.perf_counter()-start
    result['prepare_peak_rss_mb'] = peak_memory_mb()
    save_json(out/'manifest.json',result)


def additions(name, columns, out):
    pieces = []
    for kernel in sorted({c.split('_')[1] for c in columns}):
        cols = [c for c in columns if c.split('_')[1]==kernel]
        pieces.append(pd.read_parquet(out/f'{name}_{kernel}.parquet',columns=cols))
    return pd.concat(pieces,axis=1)[columns]


def screen(config,baseline,out):
    catalog = json.loads((out/'catalog.json').read_text())
    feature_names = sorted(catalog)
    bselection = json.loads((baseline/'selection.json').read_text())
    model = CatBoostClassifier()
    model.load_model(str(baseline/f"model_{bselection['selected_model']}.cbm"))
    meta = []
    residuals, groups = [],[]
    offset = 0
    train_names = [x['name'] for x in json.loads((baseline/'manifest.json').read_text())['snapshots']
                   if x['name'].startswith('train')]
    for name in train_names:
        frame = read_frame(baseline/f'{name}.parquet')
        mask = frame.groupby('user_id').label.transform('sum').to_numpy()>0
        indices = np.flatnonzero(mask)
        active = frame.loc[mask]
        scores = model.predict_proba(active[bselection['features']],thread_count=config['threads'])[:,1]
        gid, ids = pd.factorize(active.user_id,sort=True)
        groups.append(gid+offset); offset+=len(ids)
        residuals.append(active.label.to_numpy()-scores)
        meta.append((name,indices))
        del frame,active
        gc.collect()
    group = np.concatenate(groups)
    residual = np.concatenate(residuals)
    counts = np.bincount(group)
    residual -= (np.bincount(group,weights=residual)/counts)[group]
    x = np.empty((len(group),len(feature_names)),dtype=np.float32)
    for kernel,_,_ in kernels(config):
        cols = [c for c in feature_names if catalog[c]['kernel']==kernel]
        block = np.concatenate([pd.read_parquet(out/f'{name}_{kernel}.parquet',columns=cols)
                                .iloc[ix].to_numpy() for name,ix in meta])
        for j,col in enumerate(cols):
            v = block[:,j].astype(float)
            v -= (np.bincount(group,weights=v)/counts)[group]
            x[:,feature_names.index(col)] = v
    norms = np.linalg.norm(x.astype(float),axis=0)
    scores = np.divide(np.abs(x.T@residual),norms*np.linalg.norm(residual),
                       out=np.zeros(len(feature_names)),where=norms>1e-7)
    rng = np.random.default_rng(config['seed'])
    ix = rng.choice(len(x),min(len(x),config['correlation_sample_rows']),replace=False)
    z = x[ix].astype(float)
    z -= z.mean(axis=0)
    zn = np.linalg.norm(z,axis=0)
    correlation = np.divide(z.T@z,np.outer(zn,zn),out=np.zeros((len(zn),len(zn))),
                             where=np.outer(zn,zn)>1e-12)
    order = sorted(range(len(scores)),key=lambda j:(-scores[j],feature_names[j]))
    selected, rows = [],[]
    for j in order:
        redundant = max((abs(correlation[j,k]) for k in selected),default=0.)
        keep = scores[j]>1e-10 and redundant<config['correlation_limit']
        if keep:
            selected.append(j)
        rows.append({'feature':feature_names[j],'score':float(scores[j]),
                     'kept':keep,'max_correlation_with_preceding':float(redundant)})
    assert len(selected)>=max(config['selected_counts']), 'Insufficient distinct nonconstant features'
    result = {'ranked_features':[feature_names[j] for j in selected],
              'screening_rows':len(group),'screening_groups':len(counts),
              'method':'absolute within-group centered residual correlation; greedy redundancy filter'}
    pd.DataFrame(rows).to_csv(out/'screening.csv',index=False)
    save_json(out/'screening.json',result)
    log(f"Screened {len(feature_names)} -> {len(selected)} distinct features on {len(group)} rows")
    return result


def fit_variant(name,extra,base_columns,config,baseline,out):
    bselection = json.loads((baseline/'selection.json').read_text())
    bm = json.loads((baseline/'manifest.json').read_text())
    train_names = [e['name'] for e in bm['snapshots'] if e['name'].startswith('train')]
    start = time.perf_counter()
    frames=[]
    for snapshot in train_names:
        f = read_frame(baseline/f'{snapshot}.parquet')[base_columns+['label']]
        frames.append(pd.concat([f,additions(snapshot,extra,out)],axis=1))
    train = pd.concat(frames,ignore_index=True)
    del frames,f
    val = read_frame(baseline/'validation.parquet')
    ids = val[['user_id','item_id','label']].copy()
    val = pd.concat([val[base_columns],additions('validation',extra,out)],axis=1)
    features = base_columns+extra
    categorical = [c for c in base_columns if c in bselection['cat_features']]
    train_pool=Pool(train[features],train.label,cat_features=categorical)
    val_pool=Pool(val[features],ids.label,cat_features=categorical)
    del train,val
    gc.collect()
    params = dict(bselection['models'][bselection['selected_model']]['params'])
    params['iterations']=bselection['models'][bselection['selected_model']]['trees']
    log(f'Fit {name}: {len(features)} features, {params["iterations"]} trees')
    model = CatBoostClassifier(**params,loss_function='Logloss',random_seed=config['seed'],
        thread_count=config['threads'],counter_calc_method='SkipTest',used_ram_limit='1500mb',
        allow_writing_files=False,verbose=25)
    model.fit(train_pool)
    score = model.predict_proba(val_pool,thread_count=config['threads'])[:,1]
    metrics,_ = ranking(ids,score,truth_for(baseline,'validation'))
    model.save_model(str(out/f'{name}.cbm'))
    pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(out/f'importance_{name}.csv',index=False)
    result={'name':name,'extra_features':extra,'base_features':base_columns,'features':features,
            'params':params,'validation':metrics,'seconds':time.perf_counter()-start}
    save_json(out/f'{name}.json',result)
    return result


def fit(config,baseline,out):
    start=time.perf_counter()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['config']==config
    assert manifest['baseline_manifest_sha256']==sha256(baseline/'manifest.json')
    selected=json.loads((baseline/'selection.json').read_text())
    assert manifest['baseline_model_sha256']==sha256(baseline/f"model_{selected['selected_model']}.cbm")
    for name,audit in manifest['snapshots'].items():
        pairs=pd.read_parquet(baseline/f'{name}.parquet',columns=['user_id','item_id'])
        assert pair_hash(pairs)==audit['pair_hash'], f'Candidate order changed: {name}'
    screened=screen(config,baseline,out)
    main=[]
    for k in config['selected_counts']:
        main.append(fit_variant(f'auto_top{k}',screened['ranked_features'][:k],selected['features'],
                                config,baseline,out))
        gc.collect()
    best=max(main,key=lambda v:(v['validation']['ndcg@10'],-len(v['extra_features'])))
    k=len(best['extra_features'])
    all_names=sorted(json.loads((out/'catalog.json').read_text()))
    random_features=np.random.default_rng(config['seed']).choice(all_names,k,replace=False).tolist()
    random=fit_variant('random_matched',random_features,selected['features'],config,baseline,out)
    core=selected['cat_features']+['release_year','generator_rrf','generator_rank','cf_score']
    replacement=fit_variant('auto_replacement',best['extra_features'],core,config,baseline,out)
    result={'config':config,'selected_main':best['name'],'selected_count':k,
            'main_candidates':main,'random_control':random,'replacement':replacement,
            'fit_seconds':time.perf_counter()-start,'fit_peak_rss_mb':peak_memory_mb(),
            'screening_rows':screened['screening_rows'],'screening_groups':screened['screening_groups']}
    save_json(out/'selection.json',result)
    log(f"Locked {best['name']}: validation NDCG@10={best['validation']['ndcg@10']:.6f}")


def evaluate(config,baseline,out):
    selection=json.loads((out/'selection.json').read_text())
    assert selection['config']==config
    start=time.perf_counter()
    _,audit=build_snapshot('test',config,baseline,out)
    frame=read_frame(baseline/'test.parquet')
    assert pair_hash(frame)==audit['pair_hash']
    old_pairs=pd.read_parquet(baseline/'test_predictions.parquet',columns=['user_id','item_id','label'])
    pd.testing.assert_frame_equal(frame[['user_id','item_id','label']],old_pairs,check_dtype=False)
    truth=truth_for(baseline,'test')
    variants=[next(v for v in selection['main_candidates'] if v['name']==selection['selected_main']),
              selection['random_control'],selection['replacement']]
    results={'test':{},'confidence_intervals':{},'paired_differences':{},'test_pairs':audit}
    outputs=frame[['user_id','item_id','label']].copy()
    per_user={}
    for variant in variants:
        name=variant['name']
        features=pd.concat([frame[variant['base_features']],additions('test',variant['extra_features'],out)],axis=1)
        model=CatBoostClassifier();model.load_model(str(out/f'{name}.cbm'))
        t=time.perf_counter()
        predictions=model.predict_proba(features[variant['features']],thread_count=config['threads'])[:,1]
        predict_seconds=time.perf_counter()-t
        metrics,pu=ranking(outputs[['user_id','item_id','label']],predictions,truth)
        results['test'][name]={'metrics':metrics,'predict_seconds':predict_seconds}
        per_user[name]=pu.set_index('user_id')
        results['confidence_intervals'][name]={key:bootstrap(pu[key],config['seed'],config['bootstrap_repetitions'])
                                               for key in ['ndcg@10','recall@10']}
        pu.to_csv(out/f'test_users_{name}.csv',index=False)
        outputs[name]=predictions
        del features,model
        gc.collect()
    for name in ['catboost','recent_popularity']:
        per_user[name]=pd.read_csv(baseline/f'test_users_{name}.csv').set_index('user_id')
    primary=selection['selected_main']
    for control in ['catboost','recent_popularity','random_matched','auto_replacement']:
        assert per_user[primary].index.equals(per_user[control].index)
        results['paired_differences'][f'{primary}_minus_{control}']={
            key:bootstrap(per_user[primary][key]-per_user[control][key],config['seed'],config['bootstrap_repetitions'])
            for key in ['ndcg@10','recall@10']}
    outputs.to_parquet(out/'test_predictions.parquet',index=False)
    results['evaluate_seconds']=time.perf_counter()-start
    results['evaluate_peak_rss_mb']=peak_memory_mb()
    save_json(out/'results.json',results)
    log(json.dumps(results['test'],indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','fit','evaluate','all'])
    p.add_argument('--config',default='auto_config.json')
    p.add_argument('--baseline',default='artifacts/baseline')
    p.add_argument('--output',default='artifacts/auto_features')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    baseline,out=Path(args.baseline),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if args.stage in ['prepare','all']:prepare(config,baseline,out)
    if args.stage in ['fit','all']:fit(config,baseline,out)
    if args.stage in ['evaluate','all']:evaluate(config,baseline,out)


if __name__=='__main__':main()
