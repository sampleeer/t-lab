import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


def seed_everything(seed, threads=4):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Policy(nn.Module):
    """Actual GPT-2 decoder, random initialization and item IDs as tokens."""
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.decoder = GPT2Model(GPT2Config(vocab_size=vocab_size,
            n_positions=cfg['context_length'], n_ctx=cfg['context_length'],
            n_embd=cfg['hidden_size'], n_layer=cfg['layers'], n_head=cfg['heads'],
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
            pad_token_id=0, bos_token_id=1, eos_token_id=None, use_cache=False,
            attn_implementation='eager'))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, lengths):
        hidden = self.decoder(input_ids=x, attention_mask=x.ne(0).long(), use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(x)), lengths-1]
        return last @ self.decoder.wte.weight.T + self.bias


def action_logits(logits):
    # Neither PAD nor UNK is a recommendable object. Repeats stay allowed.
    return logits[:, 2:]


def append_action(x, lengths, action):
    """Right padding, then sliding window when full; action is a token ID."""
    out = x.clone()
    full = lengths == x.shape[1]
    out[full, :-1] = x[full, 1:]
    out[full, -1] = action[full]
    out[~full, lengths[~full]] = action[~full]
    return out, torch.clamp(lengths+1, max=x.shape[1])


class RewardModel(nn.Module):
    """History-conditioned factorized response model with two sigmoid heads.

    Predicts clipped fraction and probability of another recorded event within
    30 minutes, conditional on a logged listening event. No exposure model.
    """
    def __init__(self, vocab_size, item_fraction, item_continuation, dim=32):
        super().__init__()
        self.history = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.project = nn.Sequential(nn.Linear(dim*2, dim), nn.Tanh())
        self.action = nn.Parameter(torch.randn(2, vocab_size, dim)*0.02)
        means = np.stack([item_fraction, item_continuation]).clip(0.01, 0.99)
        self.bias = nn.Parameter(torch.tensor(np.log(means/(1-means)), dtype=torch.float32))
        nn.init.normal_(self.history.weight, std=0.05)
        with torch.no_grad():
            self.history.weight[0].zero_()

    def context(self, x, lengths):
        embedded = self.history(x)
        mean = (embedded*x.ne(0).unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1)
        last = embedded[torch.arange(len(x)), lengths-1]
        return self.project(torch.cat([mean, last], dim=-1))

    def forward(self, x, lengths, actions=None):
        context = self.context(x, lengths)
        if actions is None:
            scores = torch.einsum('bd,hvd->bhv', context, self.action[:, 2:]) + self.bias[:, 2:].unsqueeze(0)
        else:
            scores = (context.unsqueeze(1)*self.action[:, actions].permute(1, 0, 2)).sum(-1) + self.bias[:, actions].T
        return torch.sigmoid(scores)


def objective_rewards(predicted, duration, kind):
    fraction = predicted[:, 0]
    if kind == 'fraction':
        return fraction
    # 5 minutes = reward 1. Duration is median from training, capped at 10 min.
    time = fraction*duration.clamp(0, 600)/300
    if kind == 'time':
        return time
    if kind == 'balanced':
        return 0.5*fraction + 0.25*time + 0.25*predicted[:, 1]
    raise ValueError(kind)
