import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl_recsys import colab_runner as runner
from rl_recsys.models import Policy, RewardModel, seed_everything


class ColabTests(unittest.TestCase):
    def setUp(self):seed_everything(42,1)

    def test_device_wrappers_preserve_original_cpu_outputs(self):
        cfg={'context_length':4,'hidden_size':16,'layers':1,'heads':2}
        policy=Policy(cfg,12).eval();wrapped=runner.DevicePolicy(cfg,12).eval()
        wrapped.load_state_dict(policy.state_dict())
        reward=RewardModel(12,np.full(12,.5),np.full(12,.7)).eval()
        adapted=runner.DeviceReward(12,np.full(12,.5),np.full(12,.7)).eval()
        adapted.load_state_dict(reward.state_dict())
        x=torch.tensor([[2,3,0,0],[4,5,6,7]]);lengths=torch.tensor([2,4])
        torch.testing.assert_close(policy(x,lengths),wrapped(x,lengths),rtol=0,atol=0)
        torch.testing.assert_close(reward(x,lengths),adapted(x,lengths),rtol=0,atol=0)
        if torch.cuda.is_available():
            result=wrapped.cuda()(x.cuda(),lengths.cuda()).detach().cpu()
            torch.testing.assert_close(policy(x,lengths),result,rtol=1e-4,atol=1e-5)
            result=adapted.cuda()(x.cuda(),lengths.cuda()).detach().cpu()
            torch.testing.assert_close(reward(x,lengths),result,rtol=1e-4,atol=1e-5)

    @contextlib.contextmanager
    def tiny_training(self):
        cfg={'context_length':4,'hidden_size':16,'layers':1,'heads':2,'threads':1,'train_examples':16}
        rng=np.random.default_rng(51)
        data={'x':rng.integers(2,14,(20,4)),'lengths':np.full(20,4),
              'y':rng.integers(2,14,20),'fraction':rng.random(20).astype('float32'),
              'continuation':rng.integers(0,2,20).astype('float32'),'user':np.arange(20)}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'artifacts/seed_42').mkdir(parents=True)
            (root/'config.json').write_text(json.dumps(cfg))
            for split in ['train','validation']:np.savez(root/f'artifacts/{split}.npz',**data)
            np.savez(root/'artifacts/catalog.npz',duration=np.full(14,200.),
                item_fraction=np.full(14,.5),item_continuation=np.full(14,.7))
            torch.save(Policy(cfg,14).state_dict(),root/'artifacts/seed_42/sft.pt')
            torch.save(RewardModel(14,np.full(14,.5),np.full(14,.7)).state_dict(),root/'artifacts/seed_42/reward.pt')
            def load(name):return dict(np.load(root/f'artifacts/{name}.npz'))
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_split',side_effect=load),contextlib.redirect_stdout(io.StringIO()):
                yield root

    def assert_same_checkpoint(self, first, second):
        self.assertEqual(first['history'],second['history'])
        self.assertEqual(first['global_step'],second['global_step'])
        self.assertEqual(first['best_epoch'],second['best_epoch'])
        for name in first['model']:
            torch.testing.assert_close(first['model'][name],second['model'][name],rtol=0,atol=0)
            torch.testing.assert_close(first['best_model'][name],second['best_model'][name],rtol=0,atol=0)
        for parameter,state in first['optimizer']['state'].items():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(value,second['optimizer']['state'][parameter][key],rtol=0,atol=0)

    def test_epoch_resume_matches_uninterrupted_training(self):
        with self.tiny_training() as root:
            for stage in ['sft','reward']:
                whole=root/f'{stage}_whole';resumed=root/f'{stage}_resumed'
                args=dict(stage=stage,device='cpu',batch_size=8,learning_rate=.001)
                runner.continue_training(**args,epochs=2,output=whole)
                runner.continue_training(**args,epochs=1,output=resumed)
                runner.continue_training(**args,epochs=2,output=resumed)
                first=torch.load(whole/'temp.pt',weights_only=False)
                second=torch.load(resumed/'temp.pt',weights_only=False)
                self.assert_same_checkpoint(first,second)
                with self.assertRaisesRegex(ValueError,'signature mismatch'):
                    runner.continue_training(**{**args,'learning_rate':.002},epochs=3,output=resumed)

    def test_interrupted_batch_and_prevalidation_resume_are_exact(self):
        with self.tiny_training() as root:
            for stage in ['sft','reward']:
                args=dict(stage=stage,device='cpu',batch_size=8,learning_rate=.001,save_every_batches=1)
                whole=root/f'{stage}_whole'
                runner.continue_training(**args,epochs=2,output=whole)
                expected=torch.load(whole/'temp.pt',weights_only=False)
                for location in ['first_batch','before_validation']:
                    with self.subTest(stage=stage,location=location):
                        interrupted=root/f'{stage}_{location}'
                        original_save=runner.atomic_save
                        def interrupt_after_save(value,path):
                            original_save(value,path)
                            if Path(path).name!='temp.pt' or value['epoch']!=1 or value['epoch_complete']:return
                            stop=value['global_step']==1 if location=='first_batch' else value['next_batch_start']==len(value['epoch_order'])
                            if stop:raise InterruptedError('Simulated Colab disconnection')
                        with patch.object(runner,'atomic_save',side_effect=interrupt_after_save):
                            with self.assertRaises(InterruptedError):
                                runner.continue_training(**args,epochs=2,output=interrupted)
                        saved=torch.load(interrupted/'temp.pt',weights_only=False)
                        self.assertFalse(saved['epoch_complete'])
                        self.assertGreater(saved['next_batch_start'],0)
                        runner.continue_training(**args,epochs=2,output=interrupted)
                        actual=torch.load(interrupted/'temp.pt',weights_only=False)
                        self.assert_same_checkpoint(expected,actual)

    def test_best_weights_survive_worse_validation(self):
        with self.tiny_training() as root:
            for stage,scores in [('sft',[.1,.2,.05]),('reward',[.5,.2,.4])]:
                folder=root/stage
                captured=[];original_save=runner.atomic_save
                def capture(value,path):
                    if Path(path).name=='best.pt':captured.append({k:v.clone() for k,v in value.items()})
                    original_save(value,path)
                with patch.object(runner,'validate',side_effect=scores),patch.object(runner,'atomic_save',side_effect=capture):
                    runner.continue_training(stage=stage,device='cpu',batch_size=8,epochs=2,output=folder)
                self.assertEqual(len(captured),2)
                checkpoint=torch.load(folder/'temp.pt',weights_only=False)
                best=torch.load(folder/'best.pt',weights_only=True)
                self.assertEqual(checkpoint['best_epoch'],1)
                for name in best:torch.testing.assert_close(best[name],captured[1][name],rtol=0,atol=0)
                self.assertTrue(any(not torch.equal(best[name],checkpoint['model'][name]) for name in best))

    def test_old_last_checkpoint_migrates_to_temp(self):
        with self.tiny_training() as root:
            for stage in ['sft','reward']:
                args=dict(stage=stage,device='cpu',batch_size=8,learning_rate=.001)
                whole=root/f'{stage}_whole';legacy_folder=root/f'{stage}_legacy'
                runner.continue_training(**args,epochs=2,output=whole)
                runner.continue_training(**args,epochs=1,output=legacy_folder)
                checkpoint=torch.load(legacy_folder/'temp.pt',weights_only=False)
                legacy={key:checkpoint[key] for key in ['signature','epoch','model','optimizer','best_score','history','training_indices','rng']}
                legacy['signature']={**legacy['signature'],'runner_sha256':runner.LEGACY_RUNNER_SHA256}
                torch.save(legacy,legacy_folder/'last.pt')
                (legacy_folder/'temp.pt').unlink()
                runner.continue_training(**args,epochs=2,output=legacy_folder)
                actual=torch.load(legacy_folder/'temp.pt',weights_only=False)
                self.assertEqual(actual['format_version'],2)
                self.assert_same_checkpoint(torch.load(whole/'temp.pt',weights_only=False),actual)

    def test_failed_write_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'temp.pt'
            runner.atomic_save({'value':torch.tensor([1])},path)
            original=path.read_bytes()
            def fail(value,destination):
                Path(destination).write_bytes(b'incomplete')
                raise OSError('Simulated failed Drive write')
            with patch.object(runner.torch,'save',side_effect=fail):
                with self.assertRaises(OSError):runner.atomic_save({'value':torch.tensor([2])},path)
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
