"""Identical CatBoost information fields and importance-weighted binary objective."""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from recommender.experiments.catboost_comparison.features import CATEGORICAL, categorical_frame
from recommender.experiments.catboost_comparison.standalone import FEATURE_NAMES
from recommender.artifacts import json_digest


def categorical_lookup(product_ids, catalogue):
    frame = categorical_frame(product_ids, catalogue, np.arange(len(product_ids)))
    vocab = {name: sorted(frame[name].astype(str).unique()) for name in CATEGORICAL}
    codes = np.stack([frame[name].astype(str).map({v: i for i, v in enumerate(vocab[name])}).to_numpy()
                      for name in CATEGORICAL], axis=-1).astype(np.int64)
    return codes, vocab


def numeric_pairs(split, rows, columns):
    """All 41 rolling scalars plus the same profile8; no latent inputs/labels."""
    return np.concatenate([np.asarray(split['features'][rows, columns], dtype=np.float32),
                           np.asarray(split['profile'][rows], dtype=np.float32)], axis=-1)


def weighted_binary_loss(logits, labels, weights, *, global_mean_weight):
    if logits.shape != labels.shape or weights.shape != labels.shape or global_mean_weight <= 0:
        raise ValueError('Aligned samples and a positive fixed importance normalizer required.')
    # Fixed global normalizer avoids a biased ratio estimate within each minibatch.
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean() / global_mean_weight


class StatisticalMLP(nn.Module):
    """49 scalars + three learned categorical encodings; raw input information is fixed."""
    def __init__(self, category_sizes, *, dropout=(.2, .1), initial_positive_fraction=.003):
        super().__init__()
        if len(category_sizes) != 3 or any(n < 1 for n in category_sizes) or len(dropout) != 2:
            raise ValueError('Three categorical vocabularies and two dropout rates required.')
        self.category_sizes = tuple(map(int, category_sizes))
        self.dropout_rates = tuple(map(float, dropout))
        self.embeddings = nn.ModuleList(nn.Embedding(n, dim) for n, dim in zip(category_sizes, (16, 8, 8)))
        self.head = nn.Sequential(nn.Linear(81, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout[0]),
                                  nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout[1]),
                                  nn.Linear(64, 1))
        with torch.no_grad():
            self.head[-1].bias.fill_(float(np.log(initial_positive_fraction / (1 - initial_positive_fraction))))

    def forward(self, numeric, categories):
        if numeric.shape[:-1] != categories.shape[:-1] or numeric.shape[-1] != 49 or categories.shape[-1] != 3:
            raise ValueError('Expected aligned numeric49 and categorical3 inputs.')
        values = torch.cat([numeric, *[emb(categories[..., i]) for i, emb in enumerate(self.embeddings)]], dim=-1)
        return self.head(values).squeeze(-1)

    def contract(self):
        contract = {'model_class': 'StatisticalMLP', 'raw_feature_names': list(FEATURE_NAMES),
                    'category_sizes': list(self.category_sizes), 'embedding_dims': [16, 8, 8],
                    'numeric_dim': 49, 'head_widths': [81, 128, 64, 1], 'dropout': list(self.dropout_rates),
                    'neural_tower_inputs': [], 'statistics_policy': 'rolling_observed_v2',
                    'objective': 'weighted binary Logloss; no positive multiplier or listwise term'}
        return {**contract, 'digest': json_digest(contract)}
