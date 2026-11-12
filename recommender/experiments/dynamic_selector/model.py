"""Small exhaustive scorer over supplied cached vectors; contains no towers."""
import torch
from torch import nn
from .feature_schema import NAMES, STATIC_COLUMNS, DYNAMIC_COLUMNS


class PointwiseSelector(nn.Module):
    def __init__(self, mode='dynamic', dimension=128):
        super().__init__()
        if mode not in ('static', 'dynamic', 'relations'): raise ValueError('Unknown selector')
        self.mode = mode
        columns = STATIC_COLUMNS if mode == 'static' else DYNAMIC_COLUMNS if mode == 'dynamic' else range(len(NAMES))
        mask = torch.zeros(len(NAMES)); mask[list(columns)] = 1
        self.register_buffer('feature_mask', mask)
        self.head = nn.Sequential(nn.Linear(dimension * 4 + len(NAMES) + 1, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, customer_vectors, product_vectors, features, reference_scores):
        u = customer_vectors[:, None].expand(-1, len(product_vectors), -1)
        v = product_vectors[None].expand(len(u), -1, -1)
        x = torch.cat([u, v, u * v, (u - v).abs(), features * self.feature_mask,
                       reference_scores.unsqueeze(-1) / 10], -1)
        return self.head(x).squeeze(-1)


def select_candidates(scores, allowed, unseen, policy='quota50_25'):
    """Deterministic, eligibility-constrained selection; every ID appears once."""
    policies = {'quota50_25': (50, 25), 'quota50_10': (50, 10), 'top50': (50, 0), 'quota100_50': (100, 50)}
    if policy not in policies: raise ValueError('Unknown policy')
    depth, repeats = policies[policy]
    if bool((allowed.sum(-1) < depth).any()): raise ValueError('Insufficient eligible candidates')
    order = scores.masked_fill(~allowed, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    rows = []
    for i, ranked in enumerate(order):
        if not repeats:
            selected = ranked[:depth]
        else:
            repeat_order = ranked[(allowed[i] & ~unseen[i])[ranked]][:repeats]
            unseen_order = ranked[unseen[i, ranked]][:depth - len(repeat_order)]
            selected = torch.cat([repeat_order, unseen_order])
            if len(selected) < depth:
                rest = ranked[allowed[i, ranked] & ~torch.isin(ranked, selected)]
                selected = torch.cat([selected, rest[:depth - len(selected)]])
        rows.append(selected)
    return torch.stack(rows)
