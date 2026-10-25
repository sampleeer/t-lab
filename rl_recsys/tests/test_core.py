import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from rl_recsys import data
from rl_recsys.models import (Policy, RewardModel, action_logits, append_action,
                              objective_rewards, seed_everything)


class ProtocolTests(unittest.TestCase):
    def test_same_timestamp_and_future_never_enter_context(self):
        frame = pd.DataFrame({'uid':[7]*6, 'timestamp':[1,2,2,3,4,5],
                              'token':[2,3,4,5,6,7], 'fraction':[.5]*6,
                              'continuation':[1]*6, 'is_organic':[0]*6})
        rows = frame.iloc[[2,4]]
        result = data.make_examples(frame, rows, 3)
        np.testing.assert_array_equal(result['x'], [[2,0,0],[3,4,5]])
        np.testing.assert_array_equal(result['lengths'], [1,3])

    def test_shift_handles_padding_and_full_windows(self):
        x = torch.tensor([[2,3,0],[4,5,6]])
        result, length = append_action(x, torch.tensor([2,3]), torch.tensor([7,8]))
        self.assertEqual(result.tolist(), [[2,3,7],[5,6,8]])
        self.assertEqual(length.tolist(), [3,3])
        self.assertEqual(x.tolist(), [[2,3,0],[4,5,6]])

    def test_ranking_oov_is_miss_and_macro_weights_users(self):
        from rl_recsys.experiment import ranking, macro
        ranks = ranking(np.array([[2,3],[2,3],[2,3]]), np.array([1,2,3]))
        self.assertEqual(ranks['recall10'][0], 0)
        average = macro(ranks, np.array([10,20,20]))
        self.assertEqual(average.recall10.mean(), .5)

    def test_data_catalog_and_priors_do_not_use_future(self):
        cfg = {'data_seed':42,'user_modulus':1,'user_remainder':0,'train_quantile':.7,
               'validation_quantile':.85,'vocab_items':3,'reward_examples':100,
               'eval_examples':100,'context_length':4}
        rows = pd.DataFrame({'uid':[1]*100,'timestamp':np.arange(100)*1000,
            'item_id':np.where(np.arange(100)<70, 123, 999),
            'played_ratio_pct':np.where(np.arange(100)<70, 30, 100),
            'track_length_seconds':np.where(np.arange(100)<70, 200, 9999),'is_organic':[0]*100})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'data').mkdir(); (root/'reports').mkdir()
            rows.to_parquet(root/'data/listens.parquet',index=False)
            (root/'data/manifest.json').write_text('{}')
            with patch.object(data,'ROOT',root), patch('builtins.print'):
                data.prepare(cfg)
            catalog = dict(np.load(root/'artifacts/catalog.npz'))
            self.assertEqual(catalog['item_id'].tolist(),[0,0,123])
            self.assertEqual(catalog['duration'][2],200)
            self.assertAlmostEqual(float(catalog['item_fraction'][2]),.3,places=6)
            # A 1000-second gap is within 30 minutes. Timestamp values already
            # represent seconds, despite their five-second granularity.
            self.assertAlmostEqual(float(catalog['item_continuation'][2]),1.,places=6)
            test = dict(np.load(root/'artifacts/test.npz'))
            self.assertTrue(np.all(test['y']==1))
            self.assertTrue(np.all(test['timestamp']>=84150))


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42,1)

    def test_right_padding_cannot_change_next_item_logits(self):
        cfg={'context_length':4,'hidden_size':16,'layers':1,'heads':2}
        model=Policy(cfg,12).eval()
        with torch.no_grad():
            short=model(torch.tensor([[2,3,0,0]]),torch.tensor([2]))
            long=model(torch.tensor([[2,3,9,10]]),torch.tensor([2]))
        torch.testing.assert_close(short,long)
        self.assertEqual(action_logits(short).shape,(1,10))

    def test_response_scores_match_all_actions(self):
        model=RewardModel(12,np.full(12,.5),np.full(12,.7)).eval()
        x=torch.tensor([[2,3,0],[4,5,6]])
        lengths=torch.tensor([2,3]); actions=torch.tensor([7,8])
        all_scores=model(x,lengths)
        one=model(x,lengths,actions)
        torch.testing.assert_close(one,all_scores[torch.arange(2),:,actions-2])

    def test_reinforce_baseline_has_exact_expected_reward_gradient(self):
        logits=torch.tensor([.4,-.3,.9],requires_grad=True)
        rewards=torch.tensor([.1,.8,.4])
        logp=logits.log_softmax(0); p=logp.exp()
        exact=torch.autograd.grad((p*rewards).sum(),logits,retain_graph=True)[0]
        baseline=(p.detach()*rewards).sum()
        expected_loss=-(p.detach()*(rewards-baseline)*logp).sum()
        estimate=-torch.autograd.grad(expected_loss,logits)[0]
        torch.testing.assert_close(exact,estimate)

    def test_reward_time_cap(self):
        pred=torch.tensor([[[.5,.8],[.2,.4]]])
        duration=torch.tensor([300.,1200.])
        torch.testing.assert_close(objective_rewards(pred,duration,'time'),torch.tensor([[.5,1.6]]))


if __name__=='__main__': unittest.main()
