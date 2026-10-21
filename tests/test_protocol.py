import unittest
import numpy as np
import pandas as pd
from src.features import Snapshot, targets
from src.metrics import ranking


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 20, 20, 30, 40],
            'last_watch_dt': pd.to_datetime(['2021-01-01', '2021-01-08', '2021-01-02',
                                             '2021-01-09', '2021-01-09']),
            'total_dur': [100, 200, 100, 300, 50], 'watched_pct': [50., 60., 70., 100., 10.]})
        self.items = pd.DataFrame({'item_id': [10, 20, 30, 40], 'genres': ['a','a, b','b','c'],
            'countries': ['x']*4, 'directors': ['d']*4, 'actors': ['aa, bb']*4,
            'content_type': ['film']*4, 'age_rating': [16]*4, 'for_kids': [0]*4,
            'release_year': [2000]*4})
        self.users = pd.DataFrame({'user_id': [1,2,3], 'age': ['a']*3,
                                   'income': ['i']*3, 'sex': ['m']*3, 'kids_flg': [0]*3})

    def test_future_cannot_change_features_or_candidates(self):
        a = Snapshot(self.events, self.items, self.users, '2021-01-08')
        changed = self.events.copy()
        mask = changed.last_watch_dt >= pd.Timestamp('2021-01-08')
        changed.loc[mask, 'total_dur'] = 9999999
        changed.loc[mask, 'watched_pct'] = 0
        changed.loc[mask, 'user_id'] = 999
        changed.loc[mask, 'item_id'] = 40
        b = Snapshot(changed, self.items, self.users, '2021-01-08')
        pa, pb = a.generate([1,2]), b.generate([1,2])
        pd.testing.assert_frame_equal(pa, pb)
        pd.testing.assert_frame_equal(a.features(pa), b.features(pb))
        self.assertEqual(set(a.iids), {10,20})
        self.assertNotIn((1,10), set(zip(pa.user_id, pa.item_id)))
        self.assertNotIn((2,20), set(zip(pa.user_id, pa.item_id)))

    def test_cutoff_and_cold_items(self):
        snap = Snapshot(self.events, self.items, self.users, '2021-01-08')
        truth, audit = targets(self.events, '2021-01-08', snap.uids, snap.iids)
        self.assertEqual(truth, {1: {20}, 2: {30}})
        self.assertEqual(audit['cold_target_users'], 1)
        self.assertEqual(audit['cold_item_target_pairs'], 1)
        self.assertEqual(len(snap.history), 2)

    def test_missing_demographics_do_not_change_other_users_encoding(self):
        events = self.events.copy()
        events.loc[events.user_id == 3, 'last_watch_dt'] = pd.Timestamp('2021-01-03')
        snap = Snapshot(events, self.items, self.users[self.users.user_id != 3], '2021-01-08')
        batch = snap.features(snap.generate([1, 3]))
        alone = snap.features(snap.generate([1]))
        a = batch.loc[batch.user_id == 1].reset_index(drop=True)
        pd.testing.assert_frame_equal(a.astype(object), alone.astype(object))
        self.assertEqual(set(batch.loc[batch.user_id == 3, 'user_kids_flg']), {'unknown'})

    def test_metrics_include_missing_candidates_and_missing_users(self):
        pairs = pd.DataFrame({'user_id': [1,1], 'item_id': [10,20]})
        m, pu = ranking(pairs, np.array([.9,.1]), {1: {10,30}, 2: {40}})
        self.assertAlmostEqual(m['recall@10'], .25)
        self.assertAlmostEqual(m['candidate_recall'], .25)
        self.assertAlmostEqual(m['ndcg@10'], 1/(1+1/np.log2(3))/2)
        self.assertEqual(len(pu), 2)


if __name__ == '__main__':
    unittest.main()
