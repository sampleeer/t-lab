"""GPU continuation of existing SFT/RM weights; original experiment is untouched.

Training uses CUDA when available. Validation runs on CPU using the original
metrics, so its interpretation stays identical. A new AdamW optimizer is created
when starting from the original weight-only checkpoint. Subsequent restarts
restore model, optimizer, batch position and RNG from temp.pt. Best weights
are kept separately in best.pt. Old last.pt epoch checkpoints remain readable.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import ROOT, digest, dump, load_split
from .experiment import ranking_score, response_predictions
from .models import Policy, RewardModel, action_logits, seed_everything

LEGACY_RUNNER_SHA256 = 'ca1f23472685763f8db33fdfea754ae72f3b084c58c21c9ad0f8d849015b47ee'


def compatible_signature(checkpoint, signature):
    previous=checkpoint['signature']
    if previous==signature:return True
    # The only supported migration is the previous published epoch-only runner.
    # All data, model initialization and optimization settings must still match.
    if checkpoint.get('format_version',1)==1 and previous.get('runner_sha256')==LEGACY_RUNNER_SHA256:
        return {k:v for k,v in previous.items() if k!='runner_sha256'} == {
            k:v for k,v in signature.items() if k!='runner_sha256'}
    return False


class DevicePolicy(Policy):
    def forward(self, x, lengths):
        hidden=self.decoder(input_ids=x,attention_mask=x.ne(0).long(),use_cache=False).last_hidden_state
        last=hidden[torch.arange(len(x),device=x.device),lengths-1]
        return last @ self.decoder.wte.weight.T + self.bias


class DeviceReward(RewardModel):
    def context(self, x, lengths):
        embedded=self.history(x)
        mean=(embedded*x.ne(0).unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1)
        last=embedded[torch.arange(len(x),device=x.device),lengths-1]
        return self.project(torch.cat([mean,last],dim=-1))


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def atomic_save(value, destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(destination)


def rng_state(generator):
    return {'numpy_generator':copy.deepcopy(generator.bit_generator.state),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate()}


def restore_rng(state, generator):
    generator.bit_generator.state=state['numpy_generator']
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])


def inputs(data, idx, device):
    return (torch.as_tensor(data['x'][idx],device=device),
            torch.as_tensor(data['lengths'][idx],device=device),
            torch.as_tensor(data['y'][idx],device=device))


def validate(model, validation, stage, batch_size, device):
    # Evaluate a CPU copy. Training parameters and optimizer stay on their
    # device; the original evaluator contains NumPy operations.
    evaluation_model=copy.deepcopy(model).to('cpu').eval()
    if stage=='sft':
        score=ranking_score(evaluation_model,validation,batch_size)
    else:
        idx,pred=response_predictions(evaluation_model,validation)
        labels=np.stack([validation['fraction'][idx],validation['continuation'][idx]],axis=1)
        score=float(np.mean((pred-labels)**2))
    return score


def continue_training(stage='sft', seed=42, epochs=20, batch_size=256,
                      learning_rate=0.0002, output=None, device='auto', resume=True,
                      init_weights=None, save_every_batches=100):
    if stage not in ['sft','reward']:raise ValueError(stage)
    if save_every_batches<1:raise ValueError('save_every_batches must be positive')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
    cfg=json.loads((ROOT/'config.json').read_text())
    seed_everything(seed,cfg['threads']);random.seed(seed)
    train=load_split('train');validation=load_split('validation')
    catalog=dict(np.load(ROOT/'artifacts/catalog.npz'))
    output=Path(output) if output else ROOT/f'artifacts/colab/{stage}_{seed}'
    output.mkdir(parents=True,exist_ok=True)
    initial=Path(init_weights) if init_weights else ROOT/f'artifacts/seed_{seed}/{"sft" if stage=="sft" else "reward"}.pt'
    if stage=='sft':model=DevicePolicy(cfg,len(catalog['duration']))
    else:model=DeviceReward(len(catalog['duration']),catalog['item_fraction'],catalog['item_continuation'])
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.01)
    generator=np.random.default_rng(seed+6101)
    signature={'stage':stage,'seed':seed,'batch_size':batch_size,'lr':learning_rate,
               'initial_weights_sha256':digest(initial),
               'runner_sha256':digest(Path(__file__)),
               'config_sha256':digest(ROOT/'config.json'),
               'train_sha256':digest(ROOT/'artifacts/train.npz'),
               'validation_sha256':digest(ROOT/'artifacts/validation.npz'),
               'catalog_sha256':digest(ROOT/'artifacts/catalog.npz')}
    logs=[];completed_epoch=0;best=None;best_model=None;best_epoch=0
    global_step=0;pending=None
    temporary_checkpoint=output/'temp.pt'
    legacy_checkpoint=output/'last.pt'
    resume_path=temporary_checkpoint if temporary_checkpoint.exists() else legacy_checkpoint
    if resume_path.exists():
        if not resume:raise FileExistsError('Use another output folder to start a new run')
        # These full checkpoints are created in the user-owned output folder.
        checkpoint=torch.load(resume_path,map_location='cpu',weights_only=False)
        if not compatible_signature(checkpoint,signature):raise ValueError('Resume signature mismatch; use a new run folder')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        complete=checkpoint.get('epoch_complete',True)
        completed_epoch=checkpoint['epoch'] if complete else checkpoint['epoch']-1
        pending=None if complete else checkpoint
        best=checkpoint['best_score'];logs=checkpoint['history']
        indices=checkpoint['training_indices'].numpy()
        global_step=checkpoint.get('global_step',completed_epoch*((len(indices)+batch_size-1)//batch_size))
        best_epoch=checkpoint.get('best_epoch',max(logs,key=lambda h:h['validation_score'])['epoch']
            if stage=='sft' else min(logs,key=lambda h:h['validation_score'])['epoch'])
        if 'best_model' in checkpoint:best_model=checkpoint['best_model']
        else:
            # Legacy runs stored the best weights separately from last.pt.
            best_model=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        # Repair best.pt if a previous session stopped between its write and
        # committing temp.pt. The checkpoint is the consistent source of truth.
        atomic_save(best_model,output/'best.pt')
        restore_rng(checkpoint['rng'],generator)
    else:
        model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
        indices=np.flatnonzero(train['y']>=2)
        if stage=='sft':
            if len(indices)>cfg['train_examples']:
                indices=generator.choice(indices,cfg['train_examples'],replace=False)
        else:
            # Preserve the optimizer-RM / audit-RM row separation.
            indices=indices[indices%2==0]
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
    maximize=stage=='sft'
    if best is None:
        best=validate(model,validation,stage,batch_size,device)
        logs.append({'epoch':0,'validation_score':best,'train_loss':None})
        best_model=cpu_state(model)
        atomic_save(best_model,output/'best.pt')
        dump(output/'history.json',logs)

    def save_progress(epoch, complete, order=None, next_batch_start=0, loss_sum=0.):
        checkpoint={'format_version':2,'signature':signature,'epoch':epoch,
                    'epoch_complete':complete,'next_batch_start':next_batch_start,
                    'epoch_order':torch.as_tensor(order) if order is not None else None,
                    'train_loss_sum':loss_sum,'global_step':global_step,
                    'model':cpu_state(model),'optimizer':optimizer.state_dict(),
                    'best_model':best_model,'best_score':best,'best_epoch':best_epoch,
                    'history':logs,'training_indices':torch.as_tensor(indices),
                    'rng':rng_state(generator)}
        atomic_save(checkpoint,temporary_checkpoint)

    if pending is None:
        # Also saves epoch 0 and migrates an existing last.pt immediately.
        save_progress(completed_epoch,True)
    run_metadata={**signature,'device':str(device),'epochs_requested':epochs,'initial_weights':str(initial),
                  'optimizer_on_initial_load':'new AdamW; original bundle has weight-only checkpoints',
                  'validation_metric':'NDCG@10' if maximize else 'joint fraction MSE + continuation Brier (mean)',
                  'train_examples_used':len(indices),'torch':torch.__version__,
                  'save_every_batches':save_every_batches,'best_weights':'best.pt',
                  'resume_checkpoint':'temp.pt','checkpoint_format_version':2,
                  'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None}
    dump(output/'run.json',run_metadata)
    labels=np.stack([train['fraction'],train['continuation']],axis=1).astype(np.float32)
    print(json.dumps(run_metadata,ensure_ascii=False,indent=2),flush=True)
    for epoch in range(completed_epoch+1,epochs+1):
        if pending is not None:
            order=pending['epoch_order'].numpy()
            offset=pending['next_batch_start'];total=pending['train_loss_sum']
            pending=None
            print(f'Resuming {stage} epoch={epoch} next_example={offset} step={global_step}',flush=True)
        else:
            order=generator.permutation(indices);offset=0;total=0.
        model.train()
        for start in range(offset,len(order),batch_size):
            idx=order[start:start+batch_size];x,lengths,y=inputs(train,idx,device)
            if stage=='sft':loss=F.cross_entropy(action_logits(model(x,lengths)),y-2)
            else:loss=F.mse_loss(model(x,lengths,y),torch.as_tensor(labels[idx],device=device))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(idx)
            global_step+=1
            next_start=min(start+batch_size,len(order))
            if global_step%save_every_batches==0 or next_start==len(order):
                save_progress(epoch,False,order,next_start,total)
            if (start//batch_size+1)%250==0:
                print(f'{stage} epoch={epoch} batch={start//batch_size+1} loss={float(loss.detach()):.4f}',flush=True)
        score=validate(model,validation,stage,batch_size,device)
        logs.append({'epoch':epoch,'validation_score':score,'train_loss':total/len(order)})
        improved=score>best if maximize else score<best
        if improved:
            best=score;best_epoch=epoch;best_model=cpu_state(model)
            atomic_save(best_model,output/'best.pt')
        save_progress(epoch,True);dump(output/'history.json',logs)
        print(f'{stage} epoch={epoch} val={score:.6f} best={best:.6f}',flush=True)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['sft','reward'],default='sft')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=0.0002)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--init-weights',type=Path)
    parser.add_argument('--save-every-batches',type=int,default=100)
    args=parser.parse_args()
    continue_training(**vars(args))


if __name__=='__main__':main()
