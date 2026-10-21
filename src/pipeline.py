import argparse
import gc
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from .download import sha256
from .features import Snapshot, PERSONAL, add_labels, targets
from .metrics import ranking, bootstrap


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def peak_memory_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if platform.system() == 'Darwin' else 1024)


def prepare(config, raw, out):
    start = time.perf_counter()
    data = pd.read_csv(raw / 'interactions.csv', parse_dates=['last_watch_dt'])
    assert pd.api.types.is_datetime64_any_dtype(data.last_watch_dt), 'Invalid dates'
    assert not data[['user_id', 'item_id', 'last_watch_dt']].isna().any().any()
    assert not data.duplicated(['user_id', 'item_id']).any(), 'Expected aggregated unique pairs'
    all_users = np.sort(data.user_id.unique())
    rng = np.random.default_rng(config['seed'])
    sample = np.sort(rng.choice(all_users, min(config['sample_users'], len(all_users)), replace=False))
    full_shape = {'rows': len(data), 'users': len(all_users), 'items': data.item_id.nunique(),
                  'min_date': str(data.last_watch_dt.min().date()),
                  'max_date': str(data.last_watch_dt.max().date())}
    end = data.last_watch_dt.max().normalize() + pd.Timedelta(days=1)
    data = data[data.user_id.isin(sample)].copy()
    data = data.sort_values(['user_id', 'last_watch_dt', 'item_id']).reset_index(drop=True)
    data.to_parquet(out / 'sample_events.parquet', index=False)
    pd.Series(sample, name='user_id').to_csv(out / 'sample_users.csv', index=False)
    users = pd.read_csv(raw / 'users.csv')
    items = pd.read_csv(raw / 'items.csv')
    assert not users.user_id.duplicated().any() and not items.item_id.duplicated().any()
    assert data.item_id.isin(items.item_id).all(), 'Item metadata missing'
    missing_user_metadata = int(data.loc[~data.user_id.isin(users.user_id), 'user_id'].nunique())
    manifest = {'full_data': full_shape, 'sample': {'rows': len(data), 'users': len(sample),
                'items': data.item_id.nunique(), 'users_without_metadata': missing_user_metadata}, 'config': config,
                'data_sha256': {n: sha256(raw / n) for n in
                                ['interactions.csv', 'users.csv', 'items.csv']},
                'versions': {n: importlib.metadata.version(n) for n in
                             ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'pyarrow']},
                'python': platform.python_version(), 'platform': platform.platform(),
                'snapshots': []}
    weeks = config['train_weeks'] + 2
    for j in range(weeks):
        cutoff = end - pd.Timedelta(days=7 * (weeks-j))
        name = f'train_{j}' if j < config['train_weeks'] else ['validation', 'test'][j-config['train_weeks']]
        log(f'Building {name}, cutoff={cutoff.date()}')
        t = time.perf_counter()
        snap = Snapshot(data, items, users, cutoff, config['candidates'])
        truth, audit = targets(data, cutoff, snap.uids, snap.iids)
        if name.startswith('train'):
            selected = np.sort(np.random.default_rng(config['seed']+j).choice(
                snap.uids, min(config['train_users_per_snapshot'], len(snap.uids)), replace=False))
        else:
            selected = np.array(sorted(truth))
        pairs = snap.generate(selected)
        assert not pairs.duplicated(['user_id', 'item_id']).any()
        seen = pd.MultiIndex.from_frame(snap.history[['user_id', 'item_id']])
        assert not pd.MultiIndex.from_frame(pairs[['user_id', 'item_id']]).isin(seen).any()
        features = add_labels(snap.features(pairs), truth)
        features.to_parquet(out / f'{name}.parquet', index=False)
        save_json(out / f'{name}_truth.json', {str(k): sorted(v) for k, v in truth.items()})
        entry = {'name': name, 'cutoff': str(cutoff.date()),
                 'target_end_exclusive': str((cutoff + pd.Timedelta(days=7)).date()),
                 'history_rows': len(snap.history), 'warm_users': len(snap.uids),
                 'catalog_size': len(snap.iids), 'requested_users': len(selected),
                 'candidate_rows': len(features), 'positive_candidates': int(features.label.sum()),
                 'seconds': time.perf_counter()-t, **audit}
        manifest['snapshots'].append(entry)
        log(str(entry))
        save_json(out / 'manifest.json', manifest)
        del snap, pairs, features
        gc.collect()
    boundaries = manifest['snapshots']
    assert all(a['target_end_exclusive'] <= b['cutoff'] for a, b in zip(boundaries, boundaries[1:]))
    manifest['prepare_seconds'] = time.perf_counter()-start
    manifest['prepare_peak_rss_mb'] = peak_memory_mb()
    save_json(out / 'manifest.json', manifest)


def read_frame(path):
    f = pd.read_parquet(path)
    for col in f:
        if isinstance(f[col].dtype, pd.CategoricalDtype):
            f[col] = f[col].astype(object)
    return f


def truth_for(out, name):
    return {int(u): set(v) for u, v in json.loads((out / f'{name}_truth.json').read_text()).items()}


def columns(frame, ablation=False):
    return [c for c in frame if c not in ['user_id', 'item_id', 'label'] and
            not (ablation and c.startswith(PERSONAL))]


def controls(frame):
    return {'popularity': frame.i_n_all.to_numpy(), 'recent_popularity': frame.i_n_7d.to_numpy(),
            'generator_rrf': frame.generator_rrf.to_numpy(), 'item_cosine': frame.cf_score.to_numpy()}


def fit(config, out):
    start = time.perf_counter()
    train = pd.concat([read_frame(out / f'train_{j}.parquet')
                       for j in range(config['train_weeks'])], ignore_index=True)
    val = read_frame(out / 'validation.parquet')
    truth = truth_for(out, 'validation')
    assert train.label.nunique() == val.label.nunique() == 2
    assert config == json.loads((out / 'manifest.json').read_text())['config']
    results = {'controls': {}, 'models': [], 'training_rows': len(train),
               'training_positives': int(train.label.sum())}
    for name, scores in controls(val).items():
        m, _ = ranking(val.drop(columns='label'), scores, truth)
        results['controls'][name] = m
    best_idx, best_metric = None, -np.inf
    features = columns(train)
    cat = [c for c in features if train[c].dtype == object]
    train_pool = Pool(train[features], train.label, cat_features=cat)
    val_pool = Pool(val[features], val.label, cat_features=cat)
    val = val[['user_id', 'item_id', 'label']].copy()
    del train
    gc.collect()
    for j, params in enumerate(config['models']):
        log(f"Fit model {j}: {params}, rows={results['training_rows']}, features={len(features)}")
        t = time.perf_counter()
        model = CatBoostClassifier(**params, loss_function='Logloss', eval_metric='Logloss',
            random_seed=config['seed'], thread_count=config['threads'],
            used_ram_limit='1500mb', allow_writing_files=False, verbose=100)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60)
        score = model.predict_proba(val_pool, thread_count=config['threads'])[:, 1]
        m, _ = ranking(val, score, truth)
        model.save_model(str(out / f'model_{j}.cbm'))
        pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values(
            'importance', ascending=False).to_csv(out / f'importance_{j}.csv', index=False)
        results['models'].append({'name': f'catboost_{j}', 'params': params,
                                  'trees': model.tree_count_, 'seconds': time.perf_counter()-t, 'validation': m})
        if m['ndcg@10'] > best_metric:
            best_idx, best_metric = j, m['ndcg@10']
        del model
        gc.collect()
    results['selected_model'] = best_idx
    results['selected_control'] = max(results['controls'], key=lambda k: results['controls'][k]['ndcg@10'])
    af = [c for c in features if not c.startswith(PERSONAL)]
    del train_pool, val_pool
    gc.collect()
    train = pd.concat([read_frame(out / f'train_{j}.parquet')
                       for j in range(config['train_weeks'])], ignore_index=True)
    val_features = read_frame(out / 'validation.parquet')
    train_pool = Pool(train[af], train.label, cat_features=[c for c in af if c in cat])
    val_pool = Pool(val_features[af], val_features.label, cat_features=[c for c in af if c in cat])
    del train, val_features
    gc.collect()
    log(f'Fit ablation: {len(af)} features')
    t = time.perf_counter()
    model = CatBoostClassifier(**config['models'][best_idx], loss_function='Logloss',
        random_seed=config['seed'], thread_count=config['threads'], used_ram_limit='1500mb',
        allow_writing_files=False, verbose=100)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60)
    model.save_model(str(out / 'ablation.cbm'))
    m, _ = ranking(val, model.predict_proba(val_pool, thread_count=config['threads'])[:, 1], truth)
    results['ablation'] = {'validation': m, 'trees': model.tree_count_,
                           'seconds': time.perf_counter()-t, 'features': af}
    results['features'] = features
    results['cat_features'] = cat
    results['fit_seconds'] = time.perf_counter()-start
    results['fit_peak_rss_mb'] = peak_memory_mb()
    save_json(out / 'selection.json', results)
    log(f'Validation selection saved. Best model={best_idx}, NDCG@10={best_metric:.5f}')


def evaluate(config, out):
    selected = json.loads((out / 'selection.json').read_text())
    assert config == json.loads((out / 'manifest.json').read_text())['config']
    frame = read_frame(out / 'test.parquet')
    truth = truth_for(out, 'test')
    predictions = controls(frame)
    t = time.perf_counter()
    for name, path, features in [
        ('catboost', f"model_{selected['selected_model']}.cbm", selected['features']),
        ('catboost_no_pair', 'ablation.cbm', selected['ablation']['features'])]:
        model = CatBoostClassifier()
        model.load_model(str(out / path))
        predictions[name] = model.predict_proba(frame[features], thread_count=config['threads'])[:, 1]
    result = {'test': {}, 'confidence_intervals': {}, 'paired_differences': {},
              'inference_two_models_seconds': time.perf_counter()-t}
    per_user = {}
    for name, scores in predictions.items():
        m, pu = ranking(frame if name.startswith('catboost') else frame.drop(columns='label'), scores, truth)
        result['test'][name] = m
        per_user[name] = pu.set_index('user_id')
        pu.to_csv(out / f'test_users_{name}.csv', index=False)
        result['confidence_intervals'][name] = {
            key: bootstrap(pu[key], config['seed'], config['bootstrap_repetitions'])
            for key in ['ndcg@10', 'recall@10']}
    for comparison in [selected['selected_control'], 'catboost_no_pair']:
        diff = per_user['catboost']['ndcg@10'] - per_user[comparison]['ndcg@10']
        result['paired_differences'][f'catboost_minus_{comparison}'] = bootstrap(
            diff, config['seed'], config['bootstrap_repetitions'])
    frame[['user_id', 'item_id', 'label']].assign(**predictions).to_parquet(
        out / 'test_predictions.parquet', index=False)
    result['peak_rss_this_process_mb'] = peak_memory_mb()
    save_json(out / 'results.json', result)
    rows = [{'model': k, **v} for k, v in result['test'].items()]
    pd.DataFrame(rows).to_csv(out / 'test_metrics.csv', index=False)
    log(json.dumps(result['test'], indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'fit', 'evaluate', 'all'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--output', default='artifacts/baseline')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ['prepare', 'all']:
        prepare(config, Path(args.raw), out)
    if args.stage in ['fit', 'all']:
        fit(config, out)
    if args.stage in ['evaluate', 'all']:
        evaluate(config, out)


if __name__ == '__main__':
    main()
