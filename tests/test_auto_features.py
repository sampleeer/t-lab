import unittest
import numpy as np
import pandas as pd
from src.auto_features import AutoFeatureEngine, kernels


class AutoFeatureTests(unittest.TestCase):
    def setUp(self):
        self.events=pd.DataFrame({'user_id':[1,1,2,1],'item_id':[10,30,20,20],
            'last_watch_dt':pd.to_datetime(['2021-01-07','2021-01-05','2021-01-06','2021-01-08']),
            'watched_pct':[50.,100.,20.,0.],'total_dur':[100.,300.,50.,9999.]})
        self.items=pd.DataFrame({'item_id':[10,20,30,40],'genres':['a','a','b','b'],
            'countries':['x']*4,'actors':['actor']*4,'directors':['director']*4})
        self.pairs=pd.DataFrame({'user_id':[1,2],'item_id':[20,10]})

    def test_window_aggregation_and_affinity(self):
        e=AutoFeatureEngine(self.events,self.items,'2021-01-08',self.pairs)
        one=e.block('w01','window',1)
        three=e.block('w03','window',3)
        self.assertAlmostEqual(one.auto_w01_u_count.iloc[0],np.log(2),places=6)
        self.assertEqual(one.auto_w01_i_count.iloc[0],0)
        self.assertEqual(one.auto_w01_aff_genres.iloc[0],1)
        self.assertAlmostEqual(three.auto_w03_aff_genres.iloc[0],.5)
        self.assertAlmostEqual(three.auto_w03_u_pct.iloc[0],75)
        self.assertAlmostEqual(three.auto_w03_u_duration.iloc[0],np.log(201),places=6)

    def test_exponential_decay_is_normalized_by_weighted_history(self):
        e=AutoFeatureEngine(self.events,self.items,'2021-01-08',self.pairs)
        f=e.block('e01','exponential',1)
        self.assertAlmostEqual(f.auto_e01_aff_genres.iloc[0],np.exp(-1)/(np.exp(-1)+np.exp(-3)),places=6)
        self.assertAlmostEqual(f.auto_e01_u_count.iloc[0],np.log1p(np.exp(-1)+np.exp(-3)),places=6)

    def test_future_changes_and_batch_size_do_not_change_features(self):
        e=AutoFeatureEngine(self.events,self.items,'2021-01-08',self.pairs)
        changed=self.events.copy()
        changed.loc[3,['user_id','item_id','watched_pct','total_dur']]=[999,40,100.,1e9]
        b=AutoFeatureEngine(changed,self.items,'2021-01-08',self.pairs)
        for name,kind,days in [('w03','window',3),('e03','exponential',3)]:
            pd.testing.assert_frame_equal(e.block(name,kind,days),b.block(name,kind,days))
            single=AutoFeatureEngine(self.events,self.items,'2021-01-08',self.pairs.iloc[:1])
            pd.testing.assert_frame_equal(e.block(name,kind,days).iloc[:1],single.block(name,kind,days))

    def test_grammar_has_144_distinct_columns(self):
        config={'windows_days':[1,3,5,10,14,21,45,60],'decay_days':[3,10,21,60]}
        e=AutoFeatureEngine(self.events,self.items,'2021-01-08',self.pairs)
        names=[col for name,kind,days in kernels(config) for col in e.block(name,kind,days)]
        self.assertEqual(len(names),144)
        self.assertEqual(len(set(names)),144)


if __name__=='__main__':unittest.main()
