"""Enumerate as-of aggregate expressions from a bounded feature grammar."""
import numpy as np
import pandas as pd
from scipy import sparse
from .features import RELATIONS, tokens


def kernels(config):
    return [(f'w{d:02d}', 'window', d) for d in config['windows_days']] + [
        (f'e{d:02d}', 'exponential', d) for d in config['decay_days']]


class AutoFeatureEngine:
    def __init__(self, events, items, cutoff, pairs):
        cutoff = pd.Timestamp(cutoff)
        history = events.loc[events.last_watch_dt < cutoff].copy()
        self.uids = pd.Index(np.sort(history.user_id.unique()))
        self.iids = pd.Index(np.sort(history.item_id.unique()))
        self.ur = self.uids.get_indexer(history.user_id)
        self.ir = self.iids.get_indexer(history.item_id)
        self.pu = self.uids.get_indexer(pairs.user_id)
        self.pi = self.iids.get_indexer(pairs.item_id)
        if (self.pu < 0).any() or (self.pi < 0).any():
            raise ValueError('Candidate without historical user/item')
        self.age = (cutoff - history.last_watch_dt).dt.days.to_numpy()
        self.pct = history.watched_pct.to_numpy(dtype=float)
        self.duration = history.total_dur.to_numpy(dtype=float)
        self.nu, self.ni = len(self.uids), len(self.iids)
        self.ucount = np.bincount(self.ur, minlength=self.nu)
        self.icount = np.bincount(self.ir, minlength=self.ni)
        self.G = {}
        metadata = items.set_index('item_id').reindex(self.iids)
        for col, limit in RELATIONS.items():
            rows = [tokens(v, limit) for v in metadata[col]]
            vocab = {v: i for i, v in enumerate(sorted({v for row in rows for v in row}))}
            rr, cc = [], []
            for i, row in enumerate(rows):
                rr.extend([i]*len(row)); cc.extend(vocab[v] for v in row)
            self.G[col] = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr,cc)),
                                           shape=(self.ni,len(vocab)))

    def block(self, name, kind, days):
        if kind == 'window':
            w = (self.age <= days).astype(float)
        elif kind == 'exponential':
            w = np.exp(-self.age/days)
        else:
            raise ValueError(kind)
        uc = np.bincount(self.ur, weights=w, minlength=self.nu)
        ic = np.bincount(self.ir, weights=w, minlength=self.ni)
        result = {}
        for prefix, ix, pidx, count, allcount, size in [
            ('u',self.ur,self.pu,uc,self.ucount,self.nu),
            ('i',self.ir,self.pi,ic,self.icount,self.ni)]:
            result[f'auto_{name}_{prefix}_count'] = np.log1p(count[pidx])
            result[f'auto_{name}_{prefix}_share'] = count[pidx]/(1+allcount[pidx])
            for label, values in [('pct',self.pct),('duration',self.duration)]:
                valid = np.isfinite(values)
                num = np.bincount(ix,weights=w*np.where(valid,values,0),minlength=size)
                den = np.bincount(ix,weights=w*valid,minlength=size)
                mean = np.divide(num,den,out=np.zeros_like(num),where=den>0)[pidx]
                result[f'auto_{name}_{prefix}_{label}'] = np.log1p(mean) if label=='duration' else mean
        b = sparse.csr_matrix((w.astype(np.float32),(self.ur,self.ir)),shape=(self.nu,self.ni))
        b.eliminate_zeros()
        for col,g in self.G.items():
            profile = b@g
            lengths = np.maximum(1,np.asarray(g.sum(axis=1)).ravel()[self.pi])
            affinity = np.zeros(len(self.pu),dtype=np.float32)
            for start in range(0,len(affinity),25000):
                sl = slice(start,start+25000)
                affinity[sl] = np.asarray(profile[self.pu[sl]].multiply(g[self.pi[sl]]).sum(axis=1)).ravel()
            denom = uc[self.pu]*lengths
            result[f'auto_{name}_aff_{col}'] = np.divide(
                affinity,denom,out=np.zeros(len(affinity)),where=denom>0)
        return pd.DataFrame({k:np.asarray(v,dtype=np.float32) for k,v in result.items()})
