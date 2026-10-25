"""Causal feedback enrichment; immutable original inputs and token mapping."""
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ROOT, dump, select_indices, make_examples, digest

OUT = ROOT/'artifacts/refined'
REPORT = ROOT/'reports/refined'
RECEIVED = ROOT/'artifacts/colab_received/2026-10-24_seed42'
INPUT = RECEIVED/'colab_export/inputs'
STATE_KEYS = ('x','lengths','hfraction','hgap','hduration','past_sum','past_count','request_gap')


def buckets(uid):
    h = uid.astype(np.uint64) + np.uint64(42) + np.uint64(0x9E3779B97F4A7C15)
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return (h ^ (h >> np.uint64(31))) % 10


def frame_for(bucket):
    blocks = []
    for batch in pq.ParquetFile(ROOT/'data/listens.parquet').iter_batches(batch_size=250000):
        p = batch.to_pandas()
        blocks.append(p.loc[buckets(p.uid.to_numpy()) == bucket])
    f = pd.concat(blocks, ignore_index=True).sort_values(['uid','timestamp'], kind='stable').reset_index(drop=True)
    f['fraction'] = np.clip(f.played_ratio_pct.to_numpy(dtype=float)/100, 0, 1)
    catalog = dict(np.load(INPUT/'catalog.npz'))
    mapping = {int(item): i for i, item in enumerate(catalog['item_id']) if i >= 2}
    f['token'] = f.item_id.map(mapping).fillna(1).astype('int64')
    nxt = f.groupby('uid').timestamp.shift(-1)
    f['continuation'] = ((nxt-f.timestamp) <= 1800).fillna(False).astype(float)
    f['prior_count'] = f.groupby('uid').timestamp.transform(lambda t: np.searchsorted(t, t, side='left'))
    return f


def enrich(frame, examples):
    """Only strictly earlier events; reconstruction must match original tokens."""
    n, length = examples['x'].shape
    out = {k: v.copy() for k, v in examples.items()}
    for k in ('hfraction','hgap','hduration'):
        out[k] = np.zeros((n, length), np.float32)
    for k in ('past_sum','past_count','request_gap'):
        out[k] = np.zeros(n, np.float32)
    indices = pd.DataFrame({'uid':examples['user'], 'row':np.arange(n)}).groupby('uid', sort=False).row
    groups = dict(tuple(frame.groupby('uid', sort=False)))
    for uid, rows in indices:
        ix = rows.to_numpy()
        g = groups[uid]
        t = g.timestamp.to_numpy()
        end = np.searchsorted(t, examples['timestamp'][ix], side='left')
        assert np.all(end > 0)
        begin = np.maximum(0, end-length)
        pos = begin[:,None] + np.arange(length)
        valid = pos < end[:,None]
        safe = np.minimum(pos, len(g)-1)
        token = np.where(valid, g.token.to_numpy()[safe], 0)
        assert np.array_equal(token, examples['x'][ix]), f'History mismatch uid={uid}'
        assert np.array_equal(np.minimum(end,length), examples['lengths'][ix])
        assert np.all(t[safe][valid] < np.repeat(examples['timestamp'][ix], length).reshape(-1,length)[valid])
        frac = g.fraction.to_numpy()
        gap = np.r_[0,np.diff(t)]
        for key, values in [('hfraction',frac),('hgap',gap),('hduration',g.track_length_seconds.to_numpy())]:
            out[key][ix] = np.where(valid, values[safe], 0)
        out['past_sum'][ix] = np.r_[0,np.cumsum(frac)][end]
        out['past_count'][ix] = end
        out['request_gap'][ix] = examples['timestamp'][ix]-t[end-1]
    return out


def prepare(extra=False):
    OUT.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    if extra:
        assert (REPORT/'selection.json').exists(), 'Lock model selection before new cohort'
        f = frame_for(1)
        rows = f.loc[(f.timestamp >= 23086205) & (f.timestamp < 25999996-1800) & (f.prior_count >= 3)]
        chosen = rows.iloc[select_indices(rows,4000,np.random.default_rng(42491),per_user=8)]
        d = enrich(f, make_examples(f,chosen,24))
        np.savez_compressed(OUT/'extra.npz', **d)
        oldusers = np.load(INPUT/'train.npz')['user']
        assert len(np.intersect1d(oldusers,d['user'])) == 0
        dump(REPORT/'extra_data_checks.json', {'bucket':1,'events':len(d['y']),'users':int(len(np.unique(d['user']))),
             'target_coverage':float((d['y']>=2).mean()), 'train_user_overlap':0,
             'selection_lock_sha256':digest(REPORT/'selection.json'),'sha256':digest(OUT/'extra.npz')})
    else:
        f = frame_for(0)
        checks = {}
        for split in ('train','validation','test'):
            with np.load(INPUT/f'{split}.npz') as z:
                d = enrich(f,dict(z))
            np.savez_compressed(OUT/f'{split}.npz', **d)
            checks[split] = {'events':len(d['y']),'history_exact_match':True,
                            'only_strict_past':True,'sha256':digest(OUT/f'{split}.npz')}
            print('Prepared', split, len(d['y']), flush=True)
        counts = f.loc[f.timestamp<19934415].item_id.value_counts()
        order = sorted(counts.index,key=lambda i:(-int(counts[i]),int(i)))
        coverage = []
        for size in (4096,8192,16384,32768):
            chosen = set(order[:size])
            coverage.append({'catalog_items':size, **{name:float(part.item_id.isin(chosen).mean()) for name,part in
                 [('train',f.loc[f.timestamp<19934415]),('validation',f.loc[(f.timestamp>=19934415)&(f.timestamp<23086205)]),
                  ('test',f.loc[f.timestamp>=23086205])]}})
        checks['event_catalog_coverage_hypothetical'] = coverage
        dump(REPORT/'data_checks.json', checks)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--extra',action='store_true')
    prepare(parser.parse_args().extra)
