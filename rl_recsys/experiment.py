"""Training, validation selection, evaluation and simulator diagnostics."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch.nn import functional as F

from .data import ROOT, digest, dump, load_split, prepare, download
from .models import (Policy, RewardModel, action_logits, append_action,
                     objective_rewards, seed_everything)


def tensors(data, idx):
    return (torch.as_tensor(data['x'][idx]), torch.as_tensor(data['lengths'][idx]),
            torch.as_tensor(data['y'][idx]))


def save_state(path, model):
    torch.save(model.state_dict(), path)


def restore(model, path):
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    return model


def ranking(top, y):
    hits = top == y[:, None]
    rank = np.arange(1, top.shape[1]+1)
    return {'recall10': hits.any(1).astype(float),
            'ndcg10': (hits/np.log2(rank+1)).sum(1), 'mrr10': (hits/rank).sum(1)}


def macro(values, users):
    return pd.DataFrame(values).assign(user=users).groupby('user', sort=True).mean()


@torch.no_grad()
def ranking_score(model, data, batch_size):
    result = []
    model.eval()
    for start in range(0, len(data['y']), batch_size):
        idx = slice(start, start+batch_size)
        x, lengths, y = tensors(data, idx)
        top = action_logits(model(x, lengths)).topk(10).indices.numpy()+2
        result.append(ranking(top, y.numpy())['ndcg10'])
    return float(macro({'ndcg10': np.concatenate(result)}, data['user']).ndcg10.mean())


def train_policy(cfg, train, val, catalog, folder, seed):
    model = Policy(cfg, len(catalog['duration']))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['sft_lr'], weight_decay=0.01)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train['y']), min(cfg['train_examples'], len(train['y'])), replace=False)
    best = -1
    logs = []
    for epoch in range(cfg['sft_epochs']):
        model.train()
        order = rng.permutation(chosen)
        loss_sum = 0.
        for start in range(0, len(order), cfg['batch_size']):
            idx = order[start:start+cfg['batch_size']]
            x, lengths, y = tensors(train, idx)
            loss = F.cross_entropy(action_logits(model(x, lengths)), y-2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach())*len(y)
        score = ranking_score(model, val, cfg['batch_size'])
        logs.append({'epoch': epoch+1, 'train_ce': loss_sum/len(order), 'validation_ndcg10': score})
        print('SFT', seed, logs[-1], flush=True)
        if score > best:
            best = score
            save_state(folder/'sft.pt', model)
    dump(folder/'sft_training.json', logs)
    return restore(model, folder/'sft.pt').eval()


@torch.no_grad()
def response_predictions(model, data, batch_size=256, shuffle=False):
    idxs = np.flatnonzero(data['y'] >= 2)
    shuffled_contexts = np.random.default_rng(905).permutation(idxs) if shuffle else idxs
    result = []
    model.eval()
    for start in range(0, len(idxs), batch_size):
        idx = idxs[start:start+batch_size]
        x, lengths, y = tensors(data, idx)
        if shuffle:
            other = shuffled_contexts[start:start+batch_size]
            x, lengths, _ = tensors(data, other)
        result.append(model(x, lengths, y).numpy())
    return idxs, np.concatenate(result)


def train_reward(cfg, train, val, catalog, folder, seed, auditor=False):
    seed_everything(seed, cfg['threads'])
    model = RewardModel(len(catalog['duration']), catalog['item_fraction'], catalog['item_continuation'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['reward_lr'], weight_decay=0.01)
    rng = np.random.default_rng(seed)
    # Disjoint sampled rows for optimizer and audit RM. Both share train-only
    # item-mean prior; this is a correlated diagnostic, not an independent truth.
    pool = np.arange(len(train['y']))
    pool = pool[pool % 2 == int(auditor)]
    labels = np.stack([train['fraction'], train['continuation']], axis=1)
    val_idx = np.flatnonzero(val['y'] >= 2)
    val_labels = np.stack([val['fraction'][val_idx], val['continuation'][val_idx]], axis=1)
    name = 'auditor' if auditor else 'reward'
    best = np.inf
    logs = []
    for epoch in range(cfg['reward_epochs']):
        model.train()
        order = rng.permutation(pool)
        total = 0.
        for start in range(0, len(order), 256):
            idx = order[start:start+256]
            x, lengths, y = tensors(train, idx)
            loss = F.mse_loss(model(x, lengths, y), torch.as_tensor(labels[idx]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); total += float(loss.detach())*len(idx)
        _, pred = response_predictions(model, val)
        score = float(np.mean((pred-val_labels)**2))
        logs.append({'epoch': epoch+1, 'train_joint_mse': total/len(order), 'validation_joint_mse': score})
        print(name, seed, logs[-1], flush=True)
        if score < best:
            best = score
            save_state(folder/f'{name}.pt', model)
    dump(folder/f'{name}_training.json', logs)
    return restore(model, folder/f'{name}.pt').eval().requires_grad_(False)


def response_report(model, data, catalog):
    idx, predicted = response_predictions(model, data)
    _, shuffled = response_predictions(model, data, shuffle=True)
    y = np.stack([data['fraction'][idx], data['continuation'][idx]], axis=1)
    actions = data['y'][idx]
    item = np.stack([catalog['item_fraction'][actions], catalog['item_continuation'][actions]], axis=1)
    mean = np.broadcast_to([float(catalog['global_fraction']),
                            float(catalog['global_continuation'])], y.shape)
    result = {}
    for name, p in [('reward_model', predicted), ('shuffled_history', shuffled), ('item_mean', item), ('train_mean', mean)]:
        auc = float(roc_auc_score(y[:, 1], p[:, 1])) if len(np.unique(y[:, 1])) == 2 else None
        result[name] = {'fraction_rmse': float(np.sqrt(mean_squared_error(y[:, 0], p[:, 0]))),
                        'fraction_mae': float(mean_absolute_error(y[:, 0], p[:, 0])),
                        'continuation_brier': float(mean_squared_error(y[:, 1], p[:, 1])),
                        'continuation_auc': auc, 'n': len(idx)}
    bins = []
    for lo in np.arange(0, 1, 0.1):
        mask = (predicted[:, 0] >= lo) & (predicted[:, 0] < lo+0.1)
        if mask.any():
            bins.append({'lower': float(lo), 'n': int(mask.sum()),
                         'predicted': float(predicted[mask, 0].mean()), 'observed': float(y[mask, 0].mean())})
    result['fraction_calibration'] = bins
    return result


@torch.no_grad()
def evaluate(model, reference, reward, auditor, data, catalog, cfg, objective='fraction', fixed=None):
    chunks = []
    tops = []
    vocab = len(catalog['duration'])-2
    duration = torch.as_tensor(catalog['duration'][2:])
    for start in range(0, len(data['y']), cfg['batch_size']):
        idx = slice(start, start+cfg['batch_size'])
        x, lengths, y = tensors(data, idx)
        ref_log = action_logits(reference(x, lengths)).log_softmax(-1)
        if fixed is None:
            logp = action_logits(model(x, lengths)).log_softmax(-1)
        else:
            scores = torch.as_tensor(np.log(catalog['popularity'][2:]+1)).repeat(len(x), 1)
            if fixed == 'last_item':
                last = x[torch.arange(len(x)), lengths-1]
                valid = last >= 2
                scores[torch.arange(len(x))[valid], last[valid]-2] = scores.max()+1
            # Deterministic baselines are evaluated as point distributions for
            # expected reward/KL, ranking uses their complete ordered scores.
            logp = scores.log_softmax(-1)
        top = logp.topk(10).indices
        p = logp.exp()
        if fixed is not None:
            p.zero_(); p.scatter_(1, top[:, :1], 1.)
        r = reward(x, lengths)
        audit = auditor(x, lengths)
        top_tokens = top.numpy()+2
        values = ranking(top_tokens, y.numpy())
        seen = (top[:, :, None]+2 == x[:, None, :]).any(-1)
        values.update({
            'pred_fraction': (p*r[:, 0]).sum(-1).numpy(),
            'audit_fraction': (p*audit[:, 0]).sum(-1).numpy(),
            'pred_seconds': (p*r[:, 0]*duration).sum(-1).numpy(),
            'pred_continuation': (p*r[:, 1]).sum(-1).numpy(),
            'objective': (p*objective_rewards(r, duration, objective)).sum(-1).numpy(),
            'kl': (p*(p.clamp_min(1e-30).log()-ref_log)).sum(-1).numpy(),
            'entropy': -(p*p.clamp_min(1e-30).log()).sum(-1).numpy(),
            'duration_seconds': (p*duration).sum(-1).numpy(),
            'top10_fraction': r[:, 0].gather(1, top).mean(-1).numpy(),
            'recent_repeat_top10': seen.float().mean(-1).numpy(),
            'recent_repeat_top1': seen[:, 0].float().numpy(),
            'log_popularity_top10': np.log1p(catalog['popularity'][top_tokens]).mean(-1)})
        chunks.append(pd.DataFrame(values))
        tops.append(top_tokens)
    per_event = pd.concat(chunks, ignore_index=True)
    per_user = macro(per_event, data['user'])
    summary = {k: float(v) for k, v in per_user.mean().items()}
    top = np.concatenate(tops)
    _, counts = np.unique(top[:, 0], return_counts=True)
    all_counts = np.bincount(top.ravel(), minlength=vocab+2)[2:]
    summary.update(catalog_coverage_top10=float((all_counts > 0).mean()),
                   top1_max_share=float(counts.max()/len(top)),
                   top10_hhi=float(np.sum((all_counts/all_counts.sum())**2)),
                   users=len(per_user), events=len(per_event))
    return summary, per_user, top


def train_rl(cfg, base, reward, auditor, train, val, catalog, folder, variant, seed):
    seed_everything(seed, cfg['threads'])
    model = copy.deepcopy(base).requires_grad_(True)
    base.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['rl_lr'], weight_decay=0.)
    rng = np.random.default_rng(seed)
    duration = torch.as_tensor(catalog['duration'][2:])
    initial, _, _ = evaluate(base, base, reward, auditor, val, catalog, cfg, variant['reward'])
    best = initial['objective']
    threshold = 0.95*initial['ndcg10']
    chosen_step = 0
    save_state(folder/f"{variant['name']}_selected.pt", base)
    logs = [dict(step=0, **initial, eligible=True)]
    training = []
    for step in range(1, cfg['rl_steps']+1):
        idx = rng.integers(0, len(train['y']), cfg['rl_batch_size'])
        x, lengths, _ = tensors(train, idx)
        log_actions, rewards, baselines, kls = [], [], [], []
        model.eval()  # dropout-free identical sampling and optimization policy
        for _ in range(variant['horizon']):
            logp = action_logits(model(x, lengths)).log_softmax(-1)
            p = logp.exp()
            with torch.no_grad():
                ref_log = action_logits(base(x, lengths)).log_softmax(-1)
                r_all = objective_rewards(reward(x, lengths), duration, variant['reward'])
                action = torch.multinomial(p.detach(), num_samples=1)
                r = r_all.gather(1, action).squeeze(1)
                baseline = (p.detach()*r_all).sum(-1)
            kls.append((p*(logp-ref_log)).sum(-1).mean())
            log_actions.append(logp.gather(1, action).squeeze(1))
            rewards.append(r)
            baselines.append(baseline)
            x, lengths = append_action(x, lengths, action.squeeze(1)+2)
        returns = torch.zeros_like(rewards[0])
        losses = []
        for h in reversed(range(variant['horizon'])):
            returns = rewards[h] + 0.95*returns
            # Immediate expectation is a valid state-only control variate even
            # though it doesn't approximate the complete future value.
            losses.append(-(0.95**h)*(returns-baselines[h]).detach()*log_actions[h])
        kl = torch.stack(kls).mean()
        loss = torch.stack(losses).mean() + variant['beta']*kl
        opt.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % 25 == 0:
            training.append({'step': step, 'sample_reward': float(torch.stack(rewards).mean()),
                             'kl': float(kl.detach()), 'gradient_norm': float(norm)})
        if step % cfg['rl_eval_every'] == 0 or step == cfg['rl_steps']:
            scores, _, _ = evaluate(model, base, reward, auditor, val, catalog, cfg, variant['reward'])
            eligible = scores['ndcg10'] >= threshold
            logs.append(dict(step=step, **scores, eligible=bool(eligible)))
            print('RL', seed, variant['name'], step,
                  {k: scores[k] for k in ['ndcg10', 'objective', 'pred_fraction', 'kl']},
                  'eligible', eligible, flush=True)
            if eligible and scores['objective'] > best:
                best = scores['objective']; chosen_step = step
                save_state(folder/f"{variant['name']}_selected.pt", model)
    save_state(folder/f"{variant['name']}_endpoint.pt", model)
    dump(folder/f"{variant['name']}_validation.json", logs)
    dump(folder/f"{variant['name']}_training.json", training)
    return {'selected_step': chosen_step, 'validation_ndcg_floor': threshold,
            'selected_objective': best, **variant}


@torch.no_grad()
def rollout(model, base, reward, auditor, data, catalog, cfg, folder, name, seed):
    seed_everything(seed+9100, cfg['threads'])
    duration = torch.as_tensor(catalog['duration'][2:])
    # Same deterministic random subset across models and seeds, <=8/user.
    n = min(1024, len(data['y']))
    selected_indices = np.sort(np.random.default_rng(cfg['data_seed']+901).choice(len(data['y']),n,replace=False))
    data = {key:value[selected_indices] for key,value in data.items()}
    records = []
    for start in range(0, n, cfg['batch_size']):
        idx = slice(start, min(n, start+cfg['batch_size']))
        original, orig_len, _ = tensors(data, idx)
        x, lengths = original.clone(), orig_len.clone()
        generated = []
        for step in range(5):
            lp = action_logits(model(x, lengths)).log_softmax(-1)
            action = torch.multinomial(lp.exp(), 1)
            a = action.squeeze(1)
            ref = action_logits(base(x, lengths)).log_softmax(-1)
            r = reward(x, lengths)
            frozen = reward(original, orig_len)
            aud = auditor(x, lengths)
            # Compare original-policy propensity on synthetic and frozen state
            # for the SAME action; it diagnoses state drift, not true response.
            frozen_lp = action_logits(base(original, orig_len)).log_softmax(-1)
            previous = torch.stack(generated, 1) if generated else torch.empty((len(x), 0), dtype=torch.long)
            for j in range(len(x)):
                token = int(a[j])+2
                records.append({'user': int(data['user'][start+j]), 'event': int(selected_indices[start+j]),
                    'step': step+1, 'token': token, 'item_id': int(catalog['item_id'][token]),
                    'pred_fraction': float(r[j, 0, a[j]]),
                    'audit_fraction': float(aud[j, 0, a[j]]),
                    'frozen_fraction': float(frozen[j, 0, a[j]]),
                    'pred_continuation': float(r[j, 1, a[j]]),
                    'duration_seconds': float(duration[a[j]]),
                    'repeat_generated': float((previous[j] == token).any()),
                    'repeat_original': float((original[j] == token).any()),
                    'sft_nll': float(-ref[j, a[j]]),
                    'frozen_sft_nll': float(-frozen_lp[j, a[j]]),
                    'kl': float((lp[j].exp()*(lp[j]-ref[j])).sum())})
            generated.append(a+2)
            x, lengths = append_action(x, lengths, a+2)
    table = pd.DataFrame(records)
    table.to_csv(folder/f'{name}_rollouts.csv', index=False)
    summary = table.drop(columns=['event','token','item_id']).groupby(['step','user']).mean().groupby('step').mean()
    summary.to_csv(folder/f'{name}_rollout_summary.csv')
    return summary.reset_index().to_dict('records')


def bootstrap_delta(base, alternative, repetitions, seed):
    joined = base.join(alternative, lsuffix='_base', rsuffix='_new')
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ['ndcg10', 'recall10', 'pred_fraction', 'audit_fraction', 'pred_seconds']:
        diff = (joined[f'{metric}_new']-joined[f'{metric}_base']).to_numpy()
        estimates = np.empty(repetitions)
        for i in range(repetitions):
            estimates[i] = diff[rng.integers(0, len(diff), len(diff))].mean()
        result[metric] = {'delta': float(diff.mean()),
                          'lower95': float(np.quantile(estimates, 0.025)),
                          'upper95': float(np.quantile(estimates, 0.975)), 'users': len(diff)}
    return result


def fit(cfg):
    start_time = time.time()
    train, val = load_split('train'), load_split('validation')
    catalog = dict(np.load(ROOT/'artifacts/catalog.npz'))
    selections = {}
    for seed in cfg['seeds']:
        folder = ROOT/f'artifacts/seed_{seed}'
        folder.mkdir(exist_ok=True)
        seed_everything(seed, cfg['threads'])
        base = train_policy(cfg, train, val, catalog, folder, seed)
        reward = train_reward(cfg, train, val, catalog, folder, seed+100)
        auditor = train_reward(cfg, train, val, catalog, folder, seed+200, auditor=True)
        selections[str(seed)] = []
        for variant in cfg['variants']:
            selections[str(seed)].append(train_rl(cfg, base, reward, auditor, train, val, catalog, folder, variant, seed))
        dump(ROOT/'reports/selection.json', selections)
    dump(ROOT/'reports/fit_manifest.json', {'config': cfg,'config_sha256': digest(ROOT/'config.json'),
         'seconds': time.time()-start_time, 'python': platform.python_version(), 'platform': platform.platform(),
         'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
         'pandas': pd.__version__, 'sklearn': sklearn.__version__,
         'parameters': sum(p.numel() for p in base.parameters()),
         'source_sha256': {p.name: digest(p) for p in ROOT.glob('*.py')}})


def test(cfg):
    seed_everything(42, cfg['threads'])
    data = load_split('test')
    catalog = dict(np.load(ROOT/'artifacts/catalog.npz'))
    selection = json.loads((ROOT/'reports/selection.json').read_text())
    metrics, intervals, responses, rollouts = [], {}, {}, {}
    for seed in cfg['seeds']:
        folder = ROOT/f'artifacts/seed_{seed}'
        base = restore(Policy(cfg, len(catalog['duration'])), folder/'sft.pt').eval().requires_grad_(False)
        reward = restore(RewardModel(len(catalog['duration']), catalog['item_fraction'], catalog['item_continuation']), folder/'reward.pt').eval().requires_grad_(False)
        auditor = restore(copy.deepcopy(reward), folder/'auditor.pt').eval().requires_grad_(False)
        responses[str(seed)] = {name: response_report(m, data, catalog) for name,m in [('optimizer',reward),('auditor',auditor)]}
        base_scores, base_users, base_top = evaluate(base, base, reward, auditor, data, catalog, cfg)
        metrics.append(dict(seed=seed, model='sft', selected_step=0, **base_scores))
        base_users.to_csv(folder/'sft_users.csv')
        np.save(folder/'sft_top10.npy', base_top)
        rollouts[f'{seed}/sft'] = rollout(base, base, reward, auditor, data, catalog, cfg, folder, 'sft', seed)
        for fixed in ['popularity', 'last_item']:
            scores, _, _ = evaluate(base, base, reward, auditor, data, catalog, cfg, fixed=fixed)
            metrics.append(dict(seed=seed, model=fixed, selected_step=0, **scores))
        intervals[str(seed)] = {}
        for variant in selection[str(seed)]:
            for endpoint in [False, True]:
                suffix = 'endpoint' if endpoint else 'selected'
                name = f"{variant['name']}_{suffix}"
                model = restore(copy.deepcopy(base), folder/f'{name}.pt').eval().requires_grad_(False)
                scores, users, top = evaluate(model, base, reward, auditor, data, catalog, cfg, variant['reward'])
                scores['overlap_sft_top10'] = float(np.mean([len(set(a)&set(b))/10 for a,b in zip(top,base_top)]))
                metrics.append(dict(seed=seed, model=name, selected_step=cfg['rl_steps'] if endpoint else variant['selected_step'], **scores))
                users.to_csv(folder/f'{name}_users.csv')
                np.save(folder/f'{name}_top10.npy', top)
                intervals[str(seed)][name] = bootstrap_delta(base_users, users, cfg['bootstrap_repetitions'], seed)
                if endpoint:
                    rollouts[f'{seed}/{name}'] = rollout(model, base, reward, auditor, data, catalog, cfg, folder, name, seed)
                print('TEST', seed, name, scores['ndcg10'], scores['pred_fraction'], flush=True)
        # Example real IDs, observed target/reward and the five recommendation
        # trajectories are retained locally and copied to reports by report.py.
    table = pd.DataFrame(metrics)
    table.to_csv(ROOT/'reports/test_metrics.csv', index=False)
    dump(ROOT/'reports/paired_intervals.json', intervals)
    dump(ROOT/'reports/reward_quality.json', responses)
    dump(ROOT/'reports/rollout_metrics.json', rollouts)
    print(table[['seed','model','ndcg10','recall10','pred_fraction','audit_fraction','kl']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download','prepare','fit','evaluate','all'])
    args = parser.parse_args()
    cfg = json.loads((ROOT/'config.json').read_text())
    if args.command == 'download': download()
    if args.command in ['prepare','all']: prepare(cfg)
    if args.command in ['fit','all']: fit(cfg)
    if args.command in ['evaluate','all']: test(cfg)


if __name__ == '__main__':
    main()
