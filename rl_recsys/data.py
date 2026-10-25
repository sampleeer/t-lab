"""Chronological data preparation; no held-out labels enter training features."""
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def download():
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
    elif (ROOT / 'reports/source_manifest.json').exists():
        meta = json.loads((ROOT / 'reports/source_manifest.json').read_text())
    else:
        # Pin once, then reuse the exact revision for all subsequent downloads.
        info = json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/yandex/yambda'))
        revision = info['sha']
        meta = {'dataset': 'yandex/yambda', 'revision': revision,
                'file': 'flat/50m/listens.parquet',
                'url': f'https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/flat/50m/listens.parquet'}
    target = directory / 'listens.parquet'
    if not target.exists():
        temporary = target.with_suffix('.download')
        urllib.request.urlretrieve(meta['url'], temporary)
        if 'bytes' in meta and temporary.stat().st_size != meta['bytes']:
            raise ValueError('Incomplete dataset download; retry download before training')
        pq.read_metadata(temporary)  # reject truncated or invalid Parquet
        temporary.replace(target)
    checksum = digest(target)
    if 'sha256' in meta and meta['sha256'] != checksum:
        raise ValueError('Dataset SHA256 mismatch')
    meta.update(sha256=checksum, bytes=target.stat().st_size)
    dump(manifest, meta)
    return target


def select_indices(frame, limit, rng, per_user=None):
    idx = np.arange(len(frame))
    if per_user is not None:
        # Random priority, at most per_user targets; independent of item/reward.
        priority = rng.random(len(frame))
        temp = pd.DataFrame({'uid': frame.uid.to_numpy(), 'priority': priority})
        idx = temp.sort_values('priority').groupby('uid', sort=False).head(per_user).index.to_numpy()
    if len(idx) > limit:
        idx = rng.choice(idx, limit, replace=False)
    return np.sort(idx)


def make_examples(frame, rows, length):
    """Histories contain only events with timestamp strictly below target time."""
    x = np.zeros((len(rows), length), dtype=np.int64)
    lengths = np.zeros(len(rows), dtype=np.int64)
    groups = {int(uid): g for uid, g in frame.groupby('uid', sort=False)}
    for j, row in enumerate(rows.itertuples()):
        g = groups[int(row.uid)]
        times = g.timestamp.to_numpy()
        end = np.searchsorted(times, row.timestamp, side='left')
        history = g.token.to_numpy()[max(0, end-length):end]
        x[j, :len(history)] = history
        lengths[j] = len(history)
    assert np.all(lengths > 0)
    return dict(x=x, lengths=lengths, y=rows.token.to_numpy(np.int64),
                fraction=rows.fraction.to_numpy(np.float32),
                continuation=rows.continuation.to_numpy(np.float32),
                user=rows.uid.to_numpy(np.int64), timestamp=rows.timestamp.to_numpy(np.int64),
                organic=rows.is_organic.to_numpy(np.int64))


def prepare(cfg):
    source = ROOT / 'data/listens.parquet'
    if not source.exists():
        source = download()
    blocks = []
    raw_rows = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=250000):
        part = batch.to_pandas()
        raw_rows += len(part)
        # SplitMix64 avoids parity/modulo structure in anonymized user IDs.
        hashed = part.uid.to_numpy(dtype=np.uint64) + np.uint64(cfg['data_seed']) + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed = hashed ^ (hashed >> np.uint64(31))
        blocks.append(part.loc[(hashed % cfg['user_modulus']) == cfg['user_remainder']])
    frame = pd.concat(blocks, ignore_index=True).sort_values(['uid', 'timestamp'], kind='stable').reset_index(drop=True)
    frame['fraction'] = np.clip(frame.played_ratio_pct.to_numpy(dtype=float) / 100, 0, 1)
    assert frame[['timestamp', 'fraction', 'track_length_seconds']].notna().all().all()
    cut1, cut2 = [int(np.quantile(frame.timestamp, q)) for q in (cfg['train_quantile'], cfg['validation_quantile'])]
    end_time = int(frame.timestamp.max()) + 1
    next_time = frame.groupby('uid').timestamp.shift(-1)
    # Paper §3.3: floor((T-T_start)/5)*5. Values remain SECONDS, rounded
    # to 5s precision; they are not counts of five-second units.
    continuation_seconds = cfg.get('continuation_seconds', 1800)
    frame['continuation'] = ((next_time - frame.timestamp) <= continuation_seconds).fillna(False).astype(float)
    train = frame[frame.timestamp < cut1]
    counts = train.item_id.value_counts()
    # Explicit tie break makes the token vocabulary deterministic.
    items = sorted(counts.index, key=lambda item: (-int(counts[item]), int(item)))[:cfg['vocab_items']]
    mapping = {int(item): i+2 for i, item in enumerate(items)}
    frame['token'] = frame.item_id.map(mapping).fillna(1).astype('int64')
    frame['prior_count'] = frame.groupby('uid').timestamp.transform(lambda t: np.searchsorted(t.to_numpy(), t.to_numpy(), side='left'))
    # Targets too close to each split end are censored for continuation. Use the
    # same eligible population for both heads and ranking to simplify comparison.
    eligible = frame.prior_count >= 3
    splits = {'train': (frame.timestamp < cut1-continuation_seconds),
              'validation': (frame.timestamp >= cut1) & (frame.timestamp < cut2-continuation_seconds),
              'test': (frame.timestamp >= cut2) & (frame.timestamp < end_time-continuation_seconds)}
    rng = np.random.default_rng(cfg['data_seed'])
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    sizes = {}
    for name, mask in splits.items():
        rows = frame.loc[mask & eligible]
        if name == 'train':
            rows = rows.loc[rows.token >= 2]
            idx = select_indices(rows, cfg['reward_examples'], rng)
        else:
            idx = select_indices(rows, cfg['eval_examples'], rng, per_user=8)
        chosen = rows.iloc[idx]
        examples = make_examples(frame, chosen, cfg['context_length'])
        np.savez_compressed(output / f'{name}.npz', **examples)
        sizes[name] = {'eligible_rows': len(rows), 'examples': len(chosen),
                       'users': int(chosen.uid.nunique()),
                       'target_catalog_coverage': float((chosen.token >= 2).mean()),
                       'fraction_mean': float(chosen.fraction.mean()),
                       'continuation_mean': float(chosen.continuation.mean())}
    train = frame.loc[frame.timestamp < cut1]
    known = train.loc[train.token >= 2]
    n = len(mapping)+2
    duration = np.full(n, float(known.track_length_seconds.median()), dtype=np.float32)
    popularity = np.zeros(n, dtype=np.float32)
    item_fraction = np.full(n, float(known.fraction.mean()), dtype=np.float32)
    item_continuation = np.full(n, float(known.continuation.mean()), dtype=np.float32)
    # Continuation aggregates must also exclude right-censored training labels.
    reward_train = known.loc[known.timestamp < cut1-continuation_seconds]
    for token, g in known.groupby('token'):
        duration[token] = g.track_length_seconds.median()
        popularity[token] = len(g)
    for token, g in reward_train.groupby('token'):
        item_fraction[token] = (g.fraction.sum()+20*reward_train.fraction.mean())/(len(g)+20)
        item_continuation[token] = (g.continuation.sum()+20*reward_train.continuation.mean())/(len(g)+20)
    np.savez(output / 'catalog.npz', item_id=np.array([0, 0]+items), duration=duration,
             popularity=popularity, item_fraction=item_fraction, item_continuation=item_continuation,
             global_fraction=np.array(reward_train.fraction.mean()),
             global_continuation=np.array(reward_train.continuation.mean()))
    checks = {'raw_rows': raw_rows, 'selected_rows': len(frame), 'selected_users': int(frame.uid.nunique()),
              'timestamp_unit_seconds': 1, 'timestamp_precision_seconds': 5,
              'continuation_seconds': continuation_seconds, 'time_min': int(frame.timestamp.min()),
              'train_end_exclusive': cut1, 'validation_end_exclusive': cut2, 'time_end_exclusive': end_time,
              'catalog_items': len(mapping), 'train_catalog_event_coverage': float((train.token >= 2).mean()),
              'played_ratio_gt100': int((frame.played_ratio_pct > 100).sum()),
              'splits': sizes, 'config': cfg, 'source': json.loads((ROOT/'data/manifest.json').read_text())}
    dump(ROOT / 'reports/data_checks.json', checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


def load_split(name):
    with np.load(ROOT / f'artifacts/{name}.npz') as data:
        return {key: data[key] for key in data.files}
