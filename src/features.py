from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse


RELATIONS = {'genres': None, 'countries': None, 'directors': None, 'actors': 5}
PERSONAL = ('aff_', 'cf_', 'generator_')


def tokens(value, limit=None):
    if pd.isna(value):
        return []
    return sorted(set(t.strip() for t in str(value).split(',') if t.strip()))[:limit]


def top_indices(scores, k):
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))[:k]


class Snapshot:
    def __init__(self, events, items, users, cutoff, candidates=200):
        self.cutoff = pd.Timestamp(cutoff)
        self.history = events.loc[events.last_watch_dt < self.cutoff].copy()
        if self.history.empty:
            raise ValueError('Empty past history')
        self.candidates = candidates
        self.uids = np.sort(self.history.user_id.unique())
        self.iids = np.sort(self.history.item_id.unique())
        self.umap = pd.Index(self.uids)
        self.imap = pd.Index(self.iids)
        self.items = items.set_index('item_id').reindex(self.iids)
        self.users = users.set_index('user_id')
        h = self.history
        self.ur = self.umap.get_indexer(h.user_id)
        self.ir = self.imap.get_indexer(h.item_id)
        self.age = (self.cutoff - h.last_watch_dt).dt.days.to_numpy()
        self.B = self.matrix(np.ones(len(h), dtype=np.float32))
        self.B7 = self.matrix((self.age <= 7).astype(np.float32))
        self.count = np.asarray(self.B.sum(axis=1)).ravel()
        self.pop = np.asarray(self.B.sum(axis=0)).ravel()
        self.pop7 = np.asarray(self.B7.sum(axis=0)).ravel()
        self.global_list = top_indices(self.pop, 150)
        self.recent_list = top_indices(self.pop7, 100)
        self.G = {}
        self.H = {}
        self.H7 = {}
        for col, limit in RELATIONS.items():
            rows = [tokens(v, limit) for v in self.items[col]]
            vocab = {s: j for j, s in enumerate(sorted({s for row in rows for s in row}))}
            rr, cc = [], []
            for j, row in enumerate(rows):
                rr.extend([j] * len(row))
                cc.extend(vocab[s] for s in row)
            g = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                                  shape=(len(self.iids), len(vocab)))
            self.G[col] = g
            self.H[col] = self.B @ g
            self.H7[col] = self.B7 @ g
        co = (self.B.T @ self.B).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        rr = np.repeat(np.arange(len(self.iids)), np.diff(co.indptr))
        co.data /= np.sqrt(self.pop[rr] * self.pop[co.indices])
        for i in range(len(self.iids)):
            a, b = co.indptr[i:i+2]
            if b-a > 50:
                idx = np.lexsort((co.indices[a:b], -co.data[a:b]))[50:] + a
                co.data[idx] = 0
        co.eliminate_zeros()
        weighted = self.matrix(np.exp(-self.age / 30).astype(np.float32))
        self.CF = (weighted @ co).tocsr()
        g = self.G['genres'].tocsc()
        self.genre_lists = []
        for j in range(g.shape[1]):
            ids = g.indices[g.indptr[j]:g.indptr[j+1]]
            ids = ids[np.lexsort((ids, -self.pop[ids]))][:20]
            self.genre_lists.append(ids)
        self._aggregate_features()

    def matrix(self, values):
        return sparse.csr_matrix((values, (self.ur, self.ir)),
                                 shape=(len(self.uids), len(self.iids)))

    def _aggregate_features(self):
        h = self.history
        self.uf = pd.DataFrame(index=self.uids)
        self.itf = pd.DataFrame(index=self.iids)
        for suffix, mask in [('all', np.ones(len(h), bool)), ('7d', self.age <= 7),
                             ('30d', self.age <= 30)]:
            part = h.loc[mask]
            for key, table, prefix in [('user_id', self.uf, 'u'), ('item_id', self.itf, 'i')]:
                a = part.groupby(key).agg(n=('item_id', 'size'),
                    pct=('watched_pct', 'mean'), duration=('total_dur', 'mean'))
                for stat in a:
                    table[f'{prefix}_{stat}_{suffix}'] = a[stat].reindex(table.index).fillna(0)
        for key, table, prefix in [('user_id', self.uf, 'u'), ('item_id', self.itf, 'i')]:
            a = h.groupby(key).last_watch_dt.agg(['min', 'max'])
            table[f'{prefix}_recency'] = (self.cutoff - a['max']).dt.days
            table[f'{prefix}_age'] = (self.cutoff - a['min']).dt.days
            table[f'{prefix}_recent_share'] = table[f'{prefix}_n_7d'] / (1 + table[f'{prefix}_n_all'])

    def generate(self, user_ids):
        records = []
        for uid in sorted(user_ids):
            u = self.umap.get_indexer([uid])[0]
            if u < 0:
                continue
            seen = set(self.B[u].indices)
            cf = self.CF[u]
            order = np.lexsort((cf.indices, -cf.data))[:100]
            cf_list = cf.indices[order]
            genre = self.H['genres'][u]
            order = np.lexsort((genre.indices, -genre.data))[:3]
            genre_list = list(dict.fromkeys(int(i) for j in genre.indices[order]
                                           for i in self.genre_lists[j]))
            rrf = Counter()
            for source in [self.global_list, self.recent_list, cf_list, genre_list]:
                for rank, i in enumerate(source):
                    if i not in seen:
                        rrf[int(i)] += 1 / (60 + rank + 1)
            ranked = sorted(rrf, key=lambda i: (-rrf[i], i))[:self.candidates]
            cf_scores = dict(zip(cf.indices.tolist(), cf.data.tolist()))
            for rank, i in enumerate(ranked):
                records.append((uid, int(self.iids[i]), rrf[i], rank + 1, cf_scores.get(i, 0.)))
        return pd.DataFrame(records, columns=['user_id', 'item_id', 'generator_rrf',
                                              'generator_rank', 'cf_score'])

    def features(self, pairs):
        result = pairs.copy().reset_index(drop=True)
        ur = self.umap.get_indexer(result.user_id)
        ir = self.imap.get_indexer(result.item_id)
        if (ur < 0).any() or (ir < 0).any():
            raise ValueError('Unknown user or item at cutoff')
        for col in self.uf:
            result[col] = self.uf[col].to_numpy()[ur]
        for col in self.itf:
            result[col] = self.itf[col].to_numpy()[ir]
        for col in RELATIONS:
            g = self.G[col]
            lengths = np.maximum(1, np.asarray(g.sum(axis=1)).ravel()[ir])
            for suffix, table in [('all', self.H[col]), ('7d', self.H7[col])]:
                values = np.zeros(len(result), dtype=np.float32)
                for start in range(0, len(result), 25000):
                    sl = slice(start, start + 25000)
                    values[sl] = np.asarray(table[ur[sl]].multiply(g[ir[sl]]).sum(axis=1)).ravel()
                result[f'aff_{col}_{suffix}'] = values / lengths / np.maximum(1, self.count[ur])
        for col in ['age', 'income', 'sex', 'kids_flg']:
            source = self.users[col].astype('float64') if col == 'kids_flg' else self.users[col]
            result['user_' + col] = source.reindex(result.user_id).fillna('unknown').astype(str).to_numpy()
        for col in ['content_type', 'age_rating', 'for_kids']:
            result['item_' + col] = self.items[col].iloc[ir].fillna('unknown').astype(str).to_numpy()
        for col in ['genres', 'countries', 'directors']:
            result['item_' + col] = self.items[col].iloc[ir].fillna('unknown').astype(str).to_numpy()
        result['release_year'] = self.items.release_year.iloc[ir].fillna(0).to_numpy()
        for col in result:
            if col in ['user_id', 'item_id']:
                continue
            if result[col].dtype == object:
                result[col] = result[col].astype('category')
            else:
                result[col] = result[col].astype('float32')
        return result


def targets(events, cutoff, warm_users, catalog):
    cutoff = pd.Timestamp(cutoff)
    future = events[(events.last_watch_dt >= cutoff) &
                    (events.last_watch_dt < cutoff + pd.Timedelta(days=7))]
    past_pairs = pd.MultiIndex.from_frame(events.loc[events.last_watch_dt < cutoff, ['user_id', 'item_id']])
    future = future.loc[~pd.MultiIndex.from_frame(future[['user_id', 'item_id']]).isin(past_pairs)]
    warm = future[future.user_id.isin(warm_users)]
    relevance = warm.groupby('user_id').item_id.agg(set).to_dict()
    audit = {'target_users_all': int(future.user_id.nunique()),
             'cold_target_users': int(future.loc[~future.user_id.isin(warm_users), 'user_id'].nunique()),
             'warm_users_with_targets': len(relevance), 'warm_target_pairs': len(warm),
             'cold_item_target_pairs': int((~warm.item_id.isin(catalog)).sum())}
    return relevance, audit


def add_labels(frame, relevance):
    out = frame.copy()
    positive = {(u, i) for u, items in relevance.items() for i in items}
    out['label'] = np.fromiter(((u, i) in positive for u, i in
                               zip(out.user_id, out.item_id)), dtype=np.int8, count=len(out))
    return out
