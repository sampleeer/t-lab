import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def ranking(frame, scores, relevance, k=10):
    f = frame[['user_id', 'item_id']].assign(score=np.asarray(scores))
    f = f.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True])
    lists = f.groupby('user_id', sort=False).item_id.agg(list).to_dict()
    discounts = 1 / np.log2(np.arange(k) + 2)
    rows = []
    for uid, truth in sorted(relevance.items()):
        recs = lists.get(uid, [])
        hits = np.array([i in truth for i in recs[:k]], dtype=float)
        found = len(truth.intersection(recs))
        ideal = discounts[:min(k, len(truth))].sum()
        rows.append((uid, float((hits * discounts[:len(hits)]).sum()/ideal),
                     float(hits.sum()/len(truth)), found/len(truth),
                     float(discounts[:min(k, found)].sum()/ideal)))
    per_user = pd.DataFrame(rows, columns=['user_id', 'ndcg@10', 'recall@10',
                                          'candidate_recall', 'oracle_ndcg@10'])
    metrics = per_user.drop(columns='user_id').mean().to_dict()
    metrics['users'] = len(rows)
    if 'label' in frame and frame.label.nunique() == 2:
        p = np.clip(np.asarray(scores), 1e-7, 1-1e-7)
        # Only probability predictions should request these, see caller.
        if np.min(scores) >= 0 and np.max(scores) <= 1:
            metrics['sampled_pair_logloss'] = float(log_loss(frame.label, p))
            metrics['sampled_pair_auc'] = float(roc_auc_score(frame.label, scores))
    return metrics, per_user


def bootstrap(values, seed=42, repetitions=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    # Batches keep memory bounded as the dataset grows.
    estimates = []
    for start in range(0, repetitions, 100):
        n = min(100, repetitions-start)
        estimates.extend(values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1))
    low, high = np.quantile(estimates, [.025, .975])
    return {'mean': float(values.mean()), 'low_95': float(low), 'high_95': float(high)}
