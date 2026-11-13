"""Shared cheap features for full-pool selection and downstream ablations.

Supports two explicit statistics policies:
- `frozen_train_cutoff_v1`: historical legacy policy where global and relationship
  observations freeze at `TRAIN_CUTOFF` during validation (preserved for historical replay).
- `rolling_observed_v2`: strictly causal rolling statistics incorporating prior observed
  purchases (`< as_of_date`, never `>= as_of_date` and never `>= TEST_START`).
Snapshot metadata is explicit, not effective-dated.
"""
from bisect import bisect_left
from collections import OrderedDict
import math
import numpy as np
import pandas as pd
from recommender.candidates import (
    PointInTimeFeatureStore,
    STATISTICS_POLICY_FROZEN,
    STATISTICS_POLICY_ROLLING,
    VALID_STATISTICS_POLICIES,
)
from recommender.split import TRAIN_CUTOFF, TEST_START
from .feature_schema import NAMES


def log_count(values):
    return np.clip(np.log1p(values) / math.log1p(1000), 0, 1)


class RelationshipTable:
    def __init__(self, mapping, observations, alpha=20.):
        self.mapping = np.asarray(mapping)
        n = int(self.mapping.max()) + 1
        self.pairs = np.zeros((n, n), dtype=np.float64)
        self.source = np.zeros(n)
        self.dest = np.zeros(n)
        self.total = 0
        self.observations = observations
        self.cursor = 0
        self.alpha = alpha

    def reset(self):
        self.pairs.fill(0.0)
        self.source.fill(0.0)
        self.dest.fill(0.0)
        self.total = 0
        self.cursor = 0

    def advance(self, day):
        while self.cursor < len(self.observations) and self.observations[self.cursor][0] < day:
            _, a, b = self.observations[self.cursor]
            a, b = np.unique(self.mapping[a]), np.unique(self.mapping[b])
            # ID zero represents unknown metadata and supplies no relationship.
            a, b = a[a != 0], b[b != 0]
            if len(a) and len(b):
                self.source[a] += 1
                self.dest[b] += 1
                self.pairs[np.ix_(a, b)] += 1
                self.total += 1
            self.cursor += 1

    def score(self, skus):
        source = np.unique(self.mapping[skus]) if len(skus) else np.array([], dtype=int)
        source = source[source != 0]
        if not len(source) or not self.total:
            return np.zeros(len(self.mapping)), np.zeros(len(self.mapping))
        background = (self.dest + .5) / (self.total + 1.)
        conditional = (self.pairs[source] + self.alpha * background) / (self.source[source, None] + self.alpha)
        lift = np.log(conditional / background).mean(0)[self.mapping]
        support = self.pairs[source].sum(0)[self.mapping]
        confidence = support / (support + self.alpha)
        lift[self.mapping == 0] = 0
        confidence[self.mapping == 0] = 0
        return np.clip(lift / 5, -1, 1), confidence


class DynamicFeatureBuilder:
    def __init__(
        self,
        purchases,
        catalogue,
        companies,
        product_ids,
        *,
        relationships=True,
        statistics_policy: str = STATISTICS_POLICY_FROZEN,
        allow_non_monotonic: bool = False,
    ):
        if statistics_policy not in VALID_STATISTICS_POLICIES:
            raise ValueError(
                f'Unknown statistics_policy {statistics_policy!r}; expected one of {VALID_STATISTICS_POLICIES}.'
            )
        if purchases.purchase_date.isna().any() or not purchases.purchase_date.lt(TEST_START).all():
            raise ValueError('Development purchases only')
        self.statistics_policy = statistics_policy
        self.allow_non_monotonic = bool(allow_non_monotonic)
        self.product_ids = list(product_ids)
        self.relationships_enabled = relationships
        self.index = {sku: i for i, sku in enumerate(product_ids)}
        self.store = PointInTimeFeatureStore(
            purchases,
            catalogue,
            companies,
            statistics_policy=statistics_policy,
        )
        # Fixed declared scale: future training-count maxima cannot affect an earlier row.
        self.store._popularity_scale = math.log1p(1000)
        master = catalogue.set_index('product_id').reindex(product_ids)
        self.attrs = {name: master[name].fillna('').astype(str).to_numpy() if name in master else np.full(len(product_ids), '')
                      for name in ('product_category', 'product_family', 'business_line')}
        cat_vals = self.attrs['product_category']
        unique_cats = sorted(set(cat_vals))
        cat_lookup = {cat: idx for idx, cat in enumerate(unique_cats)}
        self._cat_indices = np.array([cat_lookup[c] for c in cat_vals], dtype=np.int64)
        self._num_cats = len(unique_cats)

        maps = {'sku': np.arange(len(product_ids)) + 1}
        for name, values in self.attrs.items():
            vocabulary = {s: i + 1 for i, s in enumerate(sorted(set(values) - {''}))}
            maps[name] = np.array([vocabulary.get(s, 0) for s in values])
        self.histories, self.basket_dates = {}, {}
        observations, complements = [], []
        is_frozen = self.statistics_policy == STATISTICS_POLICY_FROZEN
        for cid, group in purchases.groupby('customer_id', sort=False):
            baskets = []
            for date, rows in group.groupby('purchase_date', sort=True):
                ids = sorted({self.index[str(s)] for s in rows.product_id if str(s) in self.index})
                baskets.append((pd.Timestamp(date), ids))
                if relationships and (not is_frozen or date < TRAIN_CUTOFF):
                    # A basket's source/target co-occurrences are observed strictly after its date.
                    for sku in ids:
                        other = [p for p in ids if p != sku]
                        if other: complements.append((pd.Timestamp(date), [sku], other))
            self.histories[str(cid)] = baskets
            self.basket_dates[str(cid)] = [date for date, _ in baskets]
            if relationships:
                if is_frozen:
                    observations.extend((date, prev, ids) for (_, prev), (date, ids) in zip(baskets, baskets[1:]) if date < TRAIN_CUTOFF)
                else:
                    observations.extend((date, prev, ids) for (_, prev), (date, ids) in zip(baskets, baskets[1:]))
        observations.sort(key=lambda x: x[0]); complements.sort(key=lambda x: x[0])
        self.tables = {name: RelationshipTable(mapping, observations) for name, mapping in maps.items()} if relationships else {}
        self.complements = RelationshipTable(maps['sku'], complements) if relationships else None
        sales_source = (
            purchases.loc[purchases.purchase_date < TRAIN_CUTOFF]
            if is_frozen
            else purchases
        ).sort_values('purchase_date')
        self.sales = {str(sku): rows for sku, rows in sales_source.groupby('product_id', sort=False)}
        self._sales_dates = {
            sku: rows.purchase_date.to_numpy(dtype='datetime64[ns]')
            for sku, rows in self.sales.items()
        }
        self._sales_cids = {
            sku: rows.customer_id.astype(str).to_numpy()
            for sku, rows in self.sales.items()
        }
        self.global_cache = OrderedDict()
        self.last_date = None

    def global_features(self, date):
        date_ts = pd.Timestamp(date)
        if pd.isna(date_ts) or date_ts >= TEST_START:
            raise ValueError('Development predictions only')
        cutoff = min(date_ts, TRAIN_CUTOFF) if self.statistics_policy == STATISTICS_POLICY_FROZEN else date_ts
        if cutoff in self.global_cache:
            self.global_cache.move_to_end(cutoff)
            return self.global_cache[cutoff]
        result = np.zeros((len(self.product_ids), 7), dtype=np.float32)
        result[:, 6] = 1  # Never sold: identical whether or not a future sale exists.
        counts = np.zeros(len(result))
        cutoff_64 = cutoff.to_datetime64()
        start_64s = [(cutoff - pd.Timedelta(days=d)).to_datetime64() for d in (7, 30, 60, 90)]
        for i, sku in enumerate(self.product_ids):
            dates = self._sales_dates.get(sku)
            if dates is None:
                continue
            stop = int(dates.searchsorted(cutoff_64, side='left'))
            s7, s30, s60, s90 = (int(dates.searchsorted(st, side='left')) for st in start_64s)
            n7, n30, n60, n90 = stop - s7, stop - s30, stop - s60, stop - s90
            buyers = len(set(self._sales_cids[sku][s30:stop])) if stop > s30 else 0
            result[i, :4] = log_count([n7, n30, n90, buyers])
            result[i, 4] = np.clip(np.log((n30 + 1) / (n60 - n30 + 1)) / 5, -1, 1)
            counts[i] = n30
            result[i, 6] = min((cutoff - pd.Timestamp(dates[stop - 1])).days / 365, 1) if stop else 1
        cat_totals = np.bincount(self._cat_indices, weights=counts, minlength=self._num_cats)
        result[:, 5] = counts / np.maximum(cat_totals[self._cat_indices], 1.0)
        self.global_cache[cutoff] = result
        while len(self.global_cache) > 512: self.global_cache.popitem(last=False)
        return result

    def build(self, customer_id, as_of_date, *, include_relationships=None, candidate_ids=None):
        date = pd.Timestamp(as_of_date)
        if pd.isna(date) or date >= TEST_START:
            raise ValueError('Development predictions only')
        if include_relationships is None: include_relationships = self.relationships_enabled
        if include_relationships and not self.relationships_enabled: raise ValueError('Relationship tables were omitted from this provider')
        if include_relationships:
            rel_cutoff = min(date, TRAIN_CUTOFF) if self.statistics_policy == STATISTICS_POLICY_FROZEN else date
            if self.last_date is not None and date < self.last_date:
                if not self.allow_non_monotonic:
                    raise ValueError('Relationship features must be built chronologically')
                for table in (*self.tables.values(), self.complements):
                    table.reset()
            self.last_date = date
            for table in (*self.tables.values(), self.complements):
                table.advance(rel_cutoff)
        cid = str(customer_id)
        candidates = self.product_ids if candidate_ids is None else list(candidate_ids)
        if len(candidates) != len(set(candidates)): raise ValueError('Duplicate candidate IDs')
        indices = np.array([self.index[sku] for sku in candidates], dtype=int)
        baskets = self.histories.get(cid, [])[:bisect_left(self.basket_dates.get(cid, []), date)]
        x = np.zeros((len(candidates), len(NAMES)), dtype=np.float32)
        x[:, :13] = self.store.compute_slate_pairwise_features(cid, date, candidates).numpy()
        if not baskets:
            x[:, [13, 17, 19, 20, 22]] = 1
        else:
            dates = [d for d, _ in baskets]
            gaps = np.diff(np.array(dates, dtype='datetime64[ns]')).astype('timedelta64[D]').astype(float)
            median = float(np.median(gaps)) if len(gaps) else 30.
            age = (date - dates[-1]).days
            x[:, 13:17] = [min(age / 365, 1), min(age / max(median, 1) / 4, 1),
                            log_count(sum(d > date - pd.Timedelta(days=30) for d in dates)),
                            log_count(sum(d > date - pd.Timedelta(days=90) for d in dates))]
            for field, offset in (('product_category', 17), ('product_family', 20)):
                values = self.attrs[field]
                observed = {}
                for d, skus in baskets:
                    for value in set(values[skus]) - {''}: observed.setdefault(value, []).append(d)
                candidate_values = values[indices]
                for value in set(candidate_values):
                    positions = candidate_values == value
                    history = observed.get(value, [])
                    if not history:
                        x[positions, offset] = 1; x[positions, offset + 2] = 1
                    else:
                        interval = float(np.median([(b - a).days for a, b in zip(history, history[1:])])) if len(history) > 1 else 30.
                        elapsed = (date - history[-1]).days
                        x[positions, offset] = min(elapsed / 365, 1)
                        x[positions, offset + 1] = min(elapsed / max(interval, 1) / 4, 1)
                        if field == 'product_family': x[positions, 23] = len(history) / len(baskets)
        x[:, 24:31] = self.global_features(date)[indices]
        angle = 2 * math.pi * (date.month - 1) / 12
        x[:, 31:33] = [math.sin(angle), math.cos(angle)]
        if include_relationships and baskets:
            last = baskets[-1][1]
            sku, support = self.tables['sku'].score(last)
            recent = sum(self.tables['sku'].score(skus)[0] * (.5 ** i) for i, (_, skus) in enumerate(reversed(baskets[-3:])))
            recent /= sum(.5 ** i for i in range(min(3, len(baskets))))
            category = self.tables['product_category'].score(last)[0]
            family = self.tables['product_family'].score(last)[0]
            line = self.tables['business_line'].score(last)[0]
            backoff = support * sku + (1 - support) * (family + category) / 2
            same = self.complements.score(last)[0]
            x[:, 33:] = np.stack([sku, recent, category, family, line, support, backoff, same], -1)[indices]
        return np.nan_to_num(x)
