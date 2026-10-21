import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from .auto_pipeline import additions,pair_hash
from .auto_features import kernels
from .pipeline import read_frame,save_json,log,peak_memory_mb
from .temporal_data import prepare,fold_dates,key,RankingContext
from .metrics import bootstrap


def load(path):return json.loads(path.read_text())


def table(data,name,extra=()):
    frame=read_frame(data/f'{name}.parquet')
    if extra:frame=pd.concat([frame,additions(name,list(extra),data)],axis=1)
    return frame


def truth(data,name):
    return {int(u):set(v) for u,v in load(data/f'{name}_truth.json').items()}


def train_model(train_keys,extra,base,params,seed,path,data,cats,threads,
                validation=None,checkpoints=None):
    spec={'train':train_keys,'extra':list(extra),'base':base,'params':params,'seed':seed,
          'validation':validation,'checkpoint_grid':checkpoints}
    json_path=path.with_suffix('.json')
    if path.exists() and json_path.exists():
        old=load(json_path)
        assert old['spec']==spec,f'Incompatible cached model {path}'
        return old
    start=time.perf_counter()
    features=base+list(extra)
    frame=pd.concat([table(data,name,extra) for name in train_keys],ignore_index=True)
    categorical=[c for c in cats if c in features]
    pool=Pool(frame[features],frame.label,weight=frame.sample_weight,cat_features=categorical)
    del frame
    gc.collect()
    if validation:
        val=table(data,validation,extra)
        context=RankingContext(val,truth(data,validation))
        val_pool=Pool(val[features],cat_features=categorical)
        del val
        gc.collect()
    log(f'Fit {path.stem}: {len(features)} columns, {params["iterations"]} trees, seed={seed}')
    model=CatBoostClassifier(**params,loss_function='Logloss',random_seed=seed,
        thread_count=threads,counter_calc_method='SkipTest',used_ram_limit='1500mb',
        allow_writing_files=False,verbose=64)
    model.fit(pool)
    records=[]
    if validation:
        for trees in checkpoints or [params['iterations']]:
            p=model.predict_proba(val_pool,ntree_end=trees,thread_count=threads)[:,1]
            metrics,_=context.evaluate(p)
            records.append({'trees':trees,'metrics':metrics})
    model.save_model(str(path))
    result={'spec':spec,'checkpoints':records,'seconds':time.perf_counter()-start,
            'feature_count':len(features)}
    save_json(json_path,result)
    return result


def temporal_screen(train_keys,base,cats,config,fold,data):
    destination=fold/'screening.json'
    if destination.exists():return load(destination)
    frames,indices,residuals,groups,weights=[],[],[],[],[]
    offset=0
    params={'depth':config['prefix_depth'],'iterations':config['prefix_trees'],
            'learning_rate':config['learning_rate'],'l2_leaf_reg':config['l2_leaf_reg']}
    for j in range(1,len(train_keys)):
        held=train_keys[j]
        assert all(pd.Timestamp(k.removeprefix('train_'))+pd.Timedelta(days=7)
                   <=pd.Timestamp(held.removeprefix('train_')) for k in train_keys[:j])
        model_path=fold/f'prefix_{j}.cbm'
        train_model(train_keys[:j],[],base,params,config['seed'],model_path,data,cats,config['threads'])
        frame=table(data,held)
        keep=frame.groupby('user_id').label.transform('sum').to_numpy()>0
        active=frame.loc[keep]
        model=CatBoostClassifier();model.load_model(str(model_path))
        p=model.predict_proba(active[base],thread_count=config['threads'])[:,1]
        gid,unique=pd.factorize(active.user_id,sort=True)
        groups.append(gid+offset);offset+=len(unique)
        weights.append(active.sample_weight.to_numpy())
        residuals.append(active.label.to_numpy()-p)
        indices.append(np.flatnonzero(keep));frames.append(held)
        del model,frame,active
        gc.collect()
    group=np.concatenate(groups);w=np.concatenate(weights);r=np.concatenate(residuals)
    group_weight=np.bincount(group,weights=w)
    r-=(np.bincount(group,weights=w*r)/group_weight)[group]
    auto=load(Path('auto_config.json'))
    names=sorted(c for kernel,_,_ in kernels(auto)
                 for c in pd.read_parquet(data/f'{train_keys[0]}_{kernel}.parquet').columns)
    x=np.empty((len(group),len(names)),dtype=float)
    for kernel,_,_ in kernels(auto):
        cols=[c for c in names if c.split('_')[1]==kernel]
        block=np.concatenate([pd.read_parquet(data/f'{name}_{kernel}.parquet',columns=cols)
                              .iloc[ix].to_numpy() for name,ix in zip(frames,indices)])
        for j,col in enumerate(cols):
            v=block[:,j].astype(float)
            v-=(np.bincount(group,weights=w*v)/group_weight)[group]
            x[:,names.index(col)]=v
    norm=np.sqrt(np.sum(w[:,None]*x*x,axis=0))
    score=np.divide(np.abs(x.T@(w*r)),norm*np.sqrt(np.sum(w*r*r)),
                     out=np.zeros(len(names)),where=norm>1e-7)
    rng=np.random.default_rng(config['seed'])
    ix=rng.choice(len(x),min(len(x),100000),replace=False)
    z=x[ix].copy();zw=w[ix]
    z-=np.average(z,axis=0,weights=zw)
    z*=np.sqrt(zw)[:,None]
    zn=np.linalg.norm(z,axis=0)
    corr=np.divide(z.T@z,np.outer(zn,zn),out=np.zeros((len(names),len(names))),
                    where=np.outer(zn,zn)>1e-12)
    order=sorted(range(len(names)),key=lambda j:(-score[j],names[j]))
    chosen=[];rows=[]
    for j in order:
        maximum=max((abs(corr[j,k]) for k in chosen),default=0.)
        keep=bool(score[j]>1e-10 and maximum<config['correlation_limit'])
        if keep:chosen.append(j)
        rows.append({'feature':names[j],'score':float(score[j]),'kept':keep,
                     'max_correlation':float(maximum)})
    assert len(chosen)>=max(config['selected_counts'])
    result={'ranked_features':[names[j] for j in chosen],'rows':len(group),'groups':offset,
            'held_out_weeks':frames,'prefix_depth':config['prefix_depth'],
            'prefix_trees':config['prefix_trees'],'residuals':'temporal out-of-fold'}
    pd.DataFrame(rows).to_csv(fold/'screening.csv',index=False)
    save_json(destination,result)
    return result


def fit(config,out):
    start=time.perf_counter();data=out/'data'
    assert load(out/'manifest.json')['config']==config
    bs=load(Path('artifacts/baseline/selection.json'))
    base,cats=bs['features'],bs['cat_features']
    all_selections=[]
    for fold_index,test in enumerate(config['test_cutoffs']):
        fold=out/f'fold_{fold_index}';fold.mkdir(exist_ok=True)
        dates=fold_dates(test,config['train_weeks'])
        train_keys=[key('train',d) for d in dates['train']]
        validation=key('eval',dates['validation'])
        if (fold/'selection.json').exists():
            all_selections.append(load(fold/'selection.json'));continue
        log(f'Fold {fold_index}: validation {dates["validation"]}, test {test} sealed')
        grid=[]
        for depth in config['depths']:
            params={'depth':depth,'iterations':max(config['tree_checkpoints']),
                    'learning_rate':config['learning_rate'],'l2_leaf_reg':config['l2_leaf_reg']}
            path=fold/f'manual_depth{depth}.cbm'
            trained=train_model(train_keys,[],base,params,config['seed'],path,data,cats,
                                config['threads'],validation,config['tree_checkpoints'])
            grid.extend({'depth':depth,'trees':r['trees'],'metrics':r['metrics'],'source':path.name}
                         for r in trained['checkpoints'])
        best=max(grid,key=lambda r:(r['metrics']['ndcg@10'],-r['trees'],-r['depth']))
        fixed={'depth':best['depth'],'iterations':best['trees'],
               'learning_rate':config['learning_rate'],'l2_leaf_reg':config['l2_leaf_reg']}
        manual=CatBoostClassifier();manual.load_model(str(fold/best['source']))
        manual.shrink(best['trees']);manual.save_model(str(fold/f'manual_seed{config["seed"]}.cbm'))
        del manual
        screened=temporal_screen(train_keys,base,cats,config,fold,data)
        candidates=[]
        for k in config['selected_counts']:
            extra=screened['ranked_features'][:k]
            path=fold/f'auto_k{k}.cbm'
            trained=train_model(train_keys,extra,base,fixed,config['seed'],path,data,cats,
                                config['threads'],validation)
            candidates.append({'k':k,'extra':extra,'metrics':trained['checkpoints'][0]['metrics'],
                               'source':path.name})
        chosen=max(candidates,key=lambda r:(r['metrics']['ndcg@10'],-r['k']))
        selected_model=CatBoostClassifier();selected_model.load_model(str(fold/chosen['source']))
        selected_model.save_model(str(fold/f'auto_seed{config["seed"]}.cbm'))
        del selected_model
        core=cats+['release_year','generator_rrf','generator_rank','cf_score']
        auto=load(Path('auto_config.json'))
        feature_names=sorted(c for kernel,_,_ in kernels(auto)
                             for c in pd.read_parquet(data/f'{train_keys[0]}_{kernel}.parquet').columns)
        models={}
        for seed in config['model_seeds']:
            random_extra=np.random.default_rng(seed+1000*fold_index).choice(
                feature_names,chosen['k'],replace=False).tolist()
            methods={'manual':([],base),'auto':(chosen['extra'],base),
                     'random':(random_extra,base),'replacement':(chosen['extra'],core)}
            for name,(extra,columns) in methods.items():
                path=fold/f'{name}_seed{seed}.cbm'
                if not (seed==config['seed'] and name in ['manual','auto']):
                    train_model(train_keys,extra,columns,fixed,seed,path,data,cats,config['threads'],validation)
                models[f'{name}_seed{seed}']={'method':name,'seed':seed,'extra':extra,
                                            'base':columns,'path':path.name,'features':columns+extra}
                gc.collect()
        selection={'fold':fold_index,'dates':dates,'config':config,'manual_grid':grid,
            'manual_best':best,'auto_candidates':candidates,'auto_best':chosen,
            'params':fixed,'models':models,'screening_rows':screened['rows'],
            'screening_groups':screened['groups']}
        save_json(fold/'selection.json',selection)
        all_selections.append(selection)
        log(f'Locked fold {fold_index}: depth={best["depth"]}, trees={best["trees"]}, auto +{chosen["k"]}')
    save_json(out/'selection.json',{'config':config,'folds':all_selections,
        'fit_seconds':time.perf_counter()-start,'fit_peak_rss_mb':peak_memory_mb()})


def evaluate(config,out):
    selection=load(out/'selection.json');assert selection['config']==config
    data=out/'data';rows=[];paired=[];weekly=[]
    start=time.perf_counter()
    for fold_index,selected in enumerate(selection['folds']):
        fold=out/f'fold_{fold_index}';test_key=key('eval',selected['dates']['test'])
        frame=table(data,test_key)
        assert pair_hash(frame)==load(out/'manifest.json')['cases'][test_key]['pair_hash']
        context=RankingContext(frame,truth(data,test_key))
        scores_frame=frame[['user_id','item_id','label']].copy()
        user_scores={}
        for name,spec in selected['models'].items():
            x=pd.concat([frame[spec['base']],additions(test_key,spec['extra'],data)],axis=1) if spec['extra'] else frame[spec['base']]
            model=CatBoostClassifier();model.load_model(str(fold/spec['path']))
            prediction=model.predict_proba(x[spec['features']],thread_count=config['threads'])[:,1]
            metrics,users=context.evaluate(prediction)
            rows.append({'fold':fold_index,'test_cutoff':selected['dates']['test'],
                         'method':spec['method'],'seed':spec['seed'],**metrics})
            user_scores.setdefault(spec['method'],[]).append(users.set_index('user_id'))
            users.to_csv(fold/f'test_users_{name}.csv',index=False)
            scores_frame[name]=prediction
            del model,x
            gc.collect()
        for name,column in [('recent_popularity','i_n_7d'),('popularity','i_n_all')]:
            metrics,users=context.evaluate(frame[column].to_numpy())
            rows.append({'fold':fold_index,'test_cutoff':selected['dates']['test'],
                         'method':name,'seed':None,**metrics})
            user_scores[name]=[users.set_index('user_id')]
        average={name:sum(frames)/len(frames) for name,frames in user_scores.items()}
        for name,users in average.items():
            weekly.append({'fold':fold_index,'test_cutoff':selected['dates']['test'],
                           'method':name,**users.mean().to_dict(),'users':len(users)})
        for a,b in [('auto','manual'),('auto','random'),('auto','recent_popularity'),
                    ('replacement','manual'),('replacement','recent_popularity'),('manual','recent_popularity')]:
            for metric in ['ndcg@10','recall@10']:
                ci=bootstrap(average[a][metric]-average[b][metric],config['seed'],config['bootstrap_repetitions'])
                paired.append({'fold':fold_index,'test_cutoff':selected['dates']['test'],
                               'comparison':f'{a}_minus_{b}','metric':metric,**ci})
        scores_frame.to_parquet(fold/'test_predictions.parquet',index=False)
        log(f'Test fold {fold_index} evaluated')
        del frame,scores_frame,user_scores
        gc.collect()
    pd.DataFrame(rows).to_csv(out/'all_results.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_results.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_intervals.csv',index=False)
    summary=pd.DataFrame(weekly).groupby('method')[['ndcg@10','recall@10']].agg(['mean','std','min','max'])
    summary.to_csv(out/'summary.csv')
    save_json(out/'results.json',{'config':config,'rows':rows,'weekly':weekly,'paired_intervals':paired,
        'evaluate_seconds':time.perf_counter()-start,'evaluate_peak_rss_mb':peak_memory_mb(),
        'inference':'weekly paired bootstrap, metrics averaged over seeds; no cross-week CI'})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','evaluate','all'])
    p.add_argument('--config',default='robust_config.json');p.add_argument('--output',default='artifacts/robustness')
    args=p.parse_args();config=load(Path(args.config));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.stage in ['prepare','all']:prepare(config,out)
    if args.stage in ['fit','all']:fit(config,out)
    if args.stage in ['evaluate','all']:evaluate(config,out)


if __name__=='__main__':main()
