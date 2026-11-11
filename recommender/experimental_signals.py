"""Independent E1/E2 models and strictly past-only next-basket statistics."""
from collections import Counter
from dataclasses import replace
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from recommender.split import TRAIN_CUTOFF, TEST_START


class SignalRanker(nn.Module):
    """Wrap the unchanged reference; extra feature never enters its attention bias."""
    def __init__(self, base, mode="control"):
        super().__init__()
        if mode not in ("control", "unseen", "transition_residual", "transition_mlp"):
            raise ValueError("Unknown signal mode")
        self.base, self.mode = base, mode
        if mode in ("unseen", "transition_residual"):
            self.residual_weight = nn.Parameter(torch.zeros(()))
        if mode == "transition_mlp":
            self.head = copy.deepcopy(base.mlp)
            first = self.head[0]
            # Zero added column gives identical initial logits and RNG-independent control.
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Linear(first.in_features + 1, first.out_features)
            with torch.no_grad():
                expanded.weight[:, :-1].copy_(first.weight)
                expanded.weight[:, -1].zero_()
                expanded.bias.copy_(first.bias)
            self.head[0] = expanded
            self.base.mlp = nn.Identity()

    def forward(self, *, pairwise_features, **kwargs):
        output = self.base(pairwise_features=pairwise_features[..., :13], **kwargs)
        if self.mode == "control":
            return output
        if self.mode == "unseen":
            logits = output.logits + self.residual_weight * (pairwise_features[..., 0] == 0)
        elif self.mode == "transition_residual":
            logits = output.logits + self.residual_weight * pairwise_features[..., 13]
        else:
            logits = self.head(torch.cat([output.raw_scores, pairwise_features[..., 13:14]], -1)).squeeze(-1)
        return replace(output, scores=logits, raw_scores=logits if self.mode == "transition_mlp" else output.raw_scores)


class TransitionStatistics:
    """Basket set counts. Both ends of a counted transition precede prediction.

    Global counts freeze at TRAIN_CUTOFF on validation. Customer weights are
    purchase-event frequencies over the full strictly prior customer history.
    Jeffreys-smoothed background avoids undefined lift for unseen destinations.
    """
    def __init__(self, purchases, alpha=20.):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not purchases.purchase_date.lt(TEST_START).all():
            raise ValueError("Development purchases only")
        self.alpha = alpha
        self.ids = sorted(purchases.product_id.astype(str).unique())
        self.index = {sku: i for i, sku in enumerate(self.ids)}
        n = len(self.ids)
        self.ab = np.zeros((n, n), dtype=np.float64)
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.total = 0
        self.observations = []
        self.histories = {}
        for cid, group in purchases.groupby("customer_id", sort=False):
            group = group.sort_values(["purchase_date", "product_id"], kind="stable")
            self.histories[str(cid)] = [(pd.Timestamp(date), self.index[str(sku)]) for date, sku in zip(group.purchase_date, group.product_id)]
            baskets = [(pd.Timestamp(date), sorted({self.index[str(s)] for s in rows.product_id}))
                       for date, rows in group.groupby("purchase_date", sort=True)]
            for (source_date, source), (target_date, target) in zip(baskets, baskets[1:]):
                if target_date < TRAIN_CUTOFF:
                    self.observations.append((target_date, source, target))
        self.observations.sort(key=lambda row: row[0])
        self.cursor = 0
        self.last_date = None

    def advance(self, as_of):
        as_of = pd.Timestamp(as_of)
        if self.last_date is not None and as_of < self.last_date:
            raise ValueError("Statistics must be queried chronologically")
        self.last_date = as_of
        cutoff = min(as_of, TRAIN_CUTOFF)
        while self.cursor < len(self.observations) and self.observations[self.cursor][0] < cutoff:
            _, source, target = self.observations[self.cursor]
            self.a[source] += 1
            self.b[target] += 1
            self.ab[np.ix_(source, target)] += 1
            self.total += 1
            self.cursor += 1

    def score(self, customer_id, as_of, candidates):
        self.advance(as_of)
        history = Counter(sku for date, sku in self.histories.get(str(customer_id), []) if date < pd.Timestamp(as_of))
        result = np.zeros(len(candidates), dtype=np.float32)
        if not history or self.total == 0:
            return result
        source = np.array(list(history), dtype=int)
        weights = np.array(list(history.values()), dtype=float)
        weights /= weights.sum()
        known = [(j, self.index[str(sku)]) for j, sku in enumerate(candidates) if str(sku) in self.index]
        if not known:
            return result
        slots, target = zip(*known)
        probability = (self.b[list(target)] + .5) / (self.total + 1.)
        conditional = (self.ab[np.ix_(source, target)] + self.alpha * probability) / (self.a[source, None] + self.alpha)
        result[list(slots)] = weights @ np.log(conditional / probability)
        # Unknown destinations have no observations: smoothed conditional/background
        # ratio is alpha/(N_A+alpha), independent of the arbitrary floor.
        unknown = [j for j, sku in enumerate(candidates) if str(sku) not in self.index]
        result[unknown] = weights @ np.log(self.alpha / (self.a[source] + self.alpha))
        return result


def transition_features(purchases, datasets, alpha=20.):
    stats = TransitionStatistics(purchases, alpha)
    outputs = [torch.zeros(len(data["customer_ids"]), 50) for data in datasets]
    queries = []
    origin = pd.Timestamp("2025-01-01")
    for split, data in enumerate(datasets):
        queries.extend((origin + pd.Timedelta(days=int(day)), split, row)
                       for row, day in enumerate(data["prediction_days"].tolist()))
    for date, split, row in sorted(queries):
        data = datasets[split]
        outputs[split][row] = torch.from_numpy(stats.score(data["customer_ids"][row], date, data["candidate_pids"][row]))
    return outputs
