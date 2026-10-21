import unittest
import numpy as np
import pandas as pd
from src.temporal_data import sample_training_pairs,fold_dates,RankingContext
from src.metrics import ranking


class TemporalTests(unittest.TestCase):
    def test_weights_and_positives_preserved(self):
        frame=pd.DataFrame({'user_id':[1]*10+[2]*5,'item_id':range(15),
                            'label':[1,1]+[0]*8+[0]*5})
        a=sample_training_pairs(frame,3,42)
        self.assertEqual(a.label.sum(),2)
        self.assertEqual(len(a),8)
        self.assertAlmostEqual(a.sample_weight.sum(),15)
        self.assertAlmostEqual(a.loc[a.user_id==1,'sample_weight'].sum(),10)
        pd.testing.assert_frame_equal(a,sample_training_pairs(frame,3,42))

    def test_temporal_boundaries(self):
        dates=fold_dates('2021-08-02')
        self.assertEqual(dates['train'],['2021-07-05','2021-07-12','2021-07-19'])
        self.assertEqual(dates['validation'],'2021-07-26')
        all_dates=dates['train']+[dates['validation'],dates['test']]
        self.assertTrue(all(pd.Timestamp(a)+pd.Timedelta(days=7)<=pd.Timestamp(b)
                            for a,b in zip(all_dates,all_dates[1:])))

    def test_fast_metrics_match_reference_with_ties_and_missing_answers(self):
        frame=pd.DataFrame({'user_id':[2,1,1,2,1],'item_id':[10,30,20,40,10],
                            'label':[0,0,1,1,1]})
        truth={1:{10,20,50},2:{40,60},3:{70}}
        score=np.array([.5,.5,.5,.5,.5])
        expected,pu=ranking(frame.drop(columns='label'),score,truth)
        got,actual=RankingContext(frame,truth).evaluate(score)
        pd.testing.assert_frame_equal(pu,actual)
        self.assertEqual(expected,got)


if __name__=='__main__':unittest.main()
