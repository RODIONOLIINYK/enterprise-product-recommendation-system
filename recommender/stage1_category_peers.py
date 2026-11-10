"""A fixed training snapshot for testing category peer scores in retrieval."""

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from recommender.ranker_statistics import FavoriteCategoryAffinity, favorite_category


class CategoryPeerScores:
    def __init__(
        self, affinity: FavoriteCategoryAffinity, product_ids: Sequence[str],
        global_counts: Mapping[str, int], smoothing: float = 20,
    ):
        # Every indexed event is already strictly before the training cutoff.
        self.product_ids = tuple(product_ids)
        product_index = {sku: i for i, sku in enumerate(self.product_ids)}
        counts = np.array([global_counts.get(sku, 0) for sku in self.product_ids], dtype=np.float64)
        self.global_rates = (counts + 1) / (sum(global_counts.values()) + len(product_ids))
        self.smoothing = smoothing
        self.totals = {category: len(dates) for category, dates in affinity.cohort_dates.items()}
        self.buyer_totals = {key: len(dates) for key, dates in affinity.buyer_dates.items()}
        self.sku_counts = {category: np.zeros(len(product_ids)) for category in self.totals}
        for (category, sku), dates in affinity.cohort_sku_dates.items():
            if sku in product_index:
                self.sku_counts[category][product_index[sku]] = len(dates)
        self.buyer_counts = defaultdict(list)
        for (buyer, category, sku), dates in affinity.buyer_sku_dates.items():
            if sku in product_index:
                self.buyer_counts[buyer, category].append((product_index[sku], len(dates)))

    def scores(self, customer: str, category_counts: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return log lifts for the favorite category and the full category mix."""
        favorite = favorite_category(category_counts)
        known = {category: count for category, count in category_counts.items() if category and count > 0}
        total = sum(known.values())
        if total < 3:
            return np.zeros_like(self.global_rates), np.zeros_like(self.global_rates)
        mixed = np.zeros_like(self.global_rates)
        favorite_rates = self.global_rates
        for category, count in known.items():
            key = (customer, category)
            cohort_total = self.totals.get(category, 0) - self.buyer_totals.get(key, 0)
            rates = self.global_rates
            if cohort_total > 0:
                posterior = self.sku_counts[category] + self.smoothing * self.global_rates
                for index, own_count in self.buyer_counts.get(key, ()):
                    posterior[index] -= own_count
                rates = posterior / (cohort_total + self.smoothing)
            mixed += (count / total) * rates
            if category == favorite:
                favorite_rates = rates
        # Keep a single sparse cohort from overwhelming the retrieval logits.
        return tuple(
            2 * np.tanh(np.log(rates / self.global_rates) / 2)
            for rates in (favorite_rates, mixed)
        )
