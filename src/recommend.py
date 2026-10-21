"""Inspect frozen model recommendations for any warm user in the sample."""
import argparse
import json
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from .features import Snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--cutoff', help='ISO date; defaults to the frozen test cutoff')
    parser.add_argument('--artifacts', default='artifacts/baseline')
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--save', help='Optional output CSV')
    args = parser.parse_args()
    out, raw = Path(args.artifacts), Path(args.raw)
    manifest = json.loads((out / 'manifest.json').read_text())
    selected = json.loads((out / 'selection.json').read_text())
    cutoff = args.cutoff or manifest['snapshots'][-1]['cutoff']
    events = pd.read_parquet(out / 'sample_events.parquet')
    items, users = pd.read_csv(raw / 'items.csv'), pd.read_csv(raw / 'users.csv')
    snap = Snapshot(events, items, users, cutoff, manifest['config']['candidates'])
    if args.user_id not in snap.umap:
        raise SystemExit('User has no available history at this cutoff')
    pairs = snap.generate([args.user_id])
    if pairs.empty:
        raise SystemExit('No unseen candidates at this cutoff')
    features = snap.features(pairs)
    model = CatBoostClassifier()
    model.load_model(str(out / f"model_{selected['selected_model']}.cbm"))
    score = model.predict_proba(features[selected['features']],
                               thread_count=manifest['config']['threads'])[:, 1]
    top = pairs[['user_id', 'item_id']].assign(score=score).sort_values(
        ['score', 'item_id'], ascending=[False, True]).head(10)
    top = top.merge(items[['item_id', 'title', 'genres', 'release_year']], on='item_id', how='left')
    print(f'Cutoff {cutoff}; recorded past pairs: {(snap.history.user_id == args.user_id).sum()}')
    print(top.to_string(index=False))
    if args.save:
        top.to_csv(args.save, index=False)


if __name__ == '__main__':
    main()
