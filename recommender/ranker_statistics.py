"""Shared point-in-time SKU statistics for Stage 2 queries and history."""

from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import groupby
from typing import Mapping

import pandas as pd
import torch

from recommender.candidates import PointInTimeFeatureStore
from recommender.sequences import ENRICHED_FEATURE_DIM


CATEGORY_PEER_AFFINITY = "favorite_category_peer_sku_affinity"
SHARED_SKU_STATISTICS = (
    "prior_sku_purchase_count_norm",
    "purchase_age_norm",
    "global_sku_popularity_norm",
    "industry_sku_affinity",
    "customer_category_share",
    CATEGORY_PEER_AFFINITY,
)
SHARED_SKU_STATISTICS_VERSION = 2
HISTORY_POPULATION_STATISTICS = ("global_sku_popularity_norm", "industry_sku_affinity")


def ranker_history_options(vocabs, shared: bool, variant: str | None) -> dict:
    if shared:
        return {
            "seq_feature_dim": len(SHARED_SKU_STATISTICS),
            "num_history_brands": max(vocabs.get("brand_vocab", {}).values(), default=0) + 1,
            "precomputed_purchase_age": True,
        }
    if variant == "brand":
        return {"num_history_brands": max(vocabs.get("brand_vocab", {}).values(), default=0) + 1}
    if variant == "population":
        return {"seq_feature_dim": ENRICHED_FEATURE_DIM + len(HISTORY_POPULATION_STATISTICS)}
    return {}


def favorite_category(counts: Mapping[str, int]) -> str:
    """Return a unique favorite when at least three known purchases support the profile."""
    known = Counter({category: count for category, count in counts.items() if category and count})
    top = known.most_common(2)
    if known.total() < 3 or not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return ""
    return top[0][0]


class FavoriteCategoryAffinity:
    """SKU purchase share among other buyers with the same prior favorite."""

    def __init__(self, store: PointInTimeFeatureStore):
        self.cohort_dates = defaultdict(list)
        self.cohort_sku_dates = defaultdict(list)
        self.buyer_dates = defaultdict(list)
        self.buyer_sku_dates = defaultdict(list)
        for customer, events in store.customer_events.items():
            counts = Counter()
            for date, purchases in groupby(events, key=lambda event: event[0]):
                if date >= store.train_cutoff:
                    break
                basket = list(purchases)
                favorite = favorite_category(counts)
                if favorite:
                    for _, sku, _, _, _, _, _ in basket:
                        self.cohort_dates[favorite].append(date)
                        self.cohort_sku_dates[favorite, sku].append(date)
                        self.buyer_dates[customer, favorite].append(date)
                        self.buyer_sku_dates[customer, favorite, sku].append(date)
                counts.update(event[3] for event in basket if event[3])
        for index in (self.cohort_dates, self.cohort_sku_dates):
            for dates in index.values():
                dates.sort()

    def share(self, customer: str, favorite: str, sku: str, as_of: pd.Timestamp) -> float:
        if not favorite:
            return 0.0
        total = bisect_left(self.cohort_dates.get(favorite, ()), as_of)
        total -= bisect_left(self.buyer_dates.get((customer, favorite), ()), as_of)
        count = bisect_left(self.cohort_sku_dates.get((favorite, sku), ()), as_of)
        count -= bisect_left(self.buyer_sku_dates.get((customer, favorite, sku), ()), as_of)
        # Shrink tiny cohorts toward zero so one purchase cannot dominate.
        return count / (total + 20)


def _aligned_slates(slate, transitions):
    for i, (transition, customer, positives, skus) in enumerate(zip(
        transitions, slate["customer_ids"], slate["positive_pids"], slate["candidate_pids"], strict=True,
    )):
        if str(transition["customer_id"]) != customer or set(transition["positive_product_ids"]) != set(positives):
            raise ValueError("SKU statistics require cached slates in chronological transition order.")
        yield i, customer, pd.Timestamp(transition["as_of_date"]), skus


def with_candidate_peer_affinity(
    slate: dict, transitions: list[dict], store: PointInTimeFeatureStore, affinity: FavoriteCategoryAffinity,
) -> dict:
    """Append just the category-peer statistic to the existing candidate inputs."""
    features = slate["pairwise_features"]
    peer = features.new_zeros((*features.shape[:2], 1))
    for i, customer, as_of, skus in _aligned_slates(slate, transitions):
        favorite = favorite_category(store.get_customer_snapshot(customer, as_of).cat_orders)
        if favorite:
            peer[i, :, 0] = torch.tensor([affinity.share(customer, favorite, sku, as_of) for sku in skus])
    return {**slate, "pairwise_features": torch.cat((features, peer), dim=-1)}


def _history_rows(slate, store, i, customer, as_of):
    mask = slate["attention_mask"]
    end = bisect_left(store.customer_event_dates.get(customer, ()), as_of)
    events = store.customer_events.get(customer, ())[max(0, end - mask.size(1)):end]
    positions = mask[i].nonzero().flatten()
    if len(positions) != len(events):
        raise ValueError("Cached ranker history does not match the purchase events before its prediction date.")
    return positions, events


def with_history_variant(slate, transitions, store, brand_vocab, variant):
    """Isolate the brand swap or the addition of population statistics to purchase history."""
    brands = torch.zeros_like(slate["seq_product_ids"]) if variant == "brand" else None
    population = slate["sequence"].new_zeros((*slate["attention_mask"].shape, 2)) if variant == "population" else None
    if brands is None and population is None:
        raise ValueError(f"Unknown history variant: {variant}")
    for i, customer, as_of, _ in _aligned_slates(slate, transitions):
        positions, events = _history_rows(slate, store, i, customer, as_of)
        if not len(positions):
            continue
        if brands is not None:
            brands[i, positions] = torch.tensor([
                brand_vocab.get(store.get_product_brand(event[1]), 0) for event in events
            ])
        else:
            population[i, positions] = torch.tensor([
                store.sku_population_statistics(customer, as_of, event[1]) for event in events
            ])
    if brands is not None:
        return {**slate, "seq_brand_ids": brands}
    return {**slate, "ranker_sequence": torch.cat((slate["sequence"], population), dim=-1)}


def with_shared_sku_statistics(
    slate: dict, transitions: list[dict], store: PointInTimeFeatureStore,
    affinity: FavoriteCategoryAffinity, brand_vocab: Mapping[str, int],
) -> dict:
    """Attach Stage 2 features while retaining the retriever's cached inputs."""
    mask = slate["attention_mask"]
    width = len(SHARED_SKU_STATISTICS)
    history = torch.zeros((*mask.shape, width))
    brands = torch.zeros_like(slate["seq_product_ids"])
    candidates = torch.empty((*slate["candidate_indices"].shape, width))
    for i, customer, as_of, skus in _aligned_slates(slate, transitions):
        snapshot = store.get_customer_snapshot(customer, as_of)
        favorite = favorite_category(snapshot.cat_orders)
        positions, events = _history_rows(slate, store, i, customer, as_of)

        shared = {}
        for sku in set(skus).union(event[1] for event in events):
            prior = snapshot.sku_stats.get(sku)
            count = prior["order_count"] if prior else 0
            popularity, industry_share = store.sku_population_statistics(customer, as_of, sku)
            category = store.get_product_category(sku)
            category_share = snapshot.cat_orders.get(category, 0) / max(snapshot.total_orders, 1)
            peer_share = affinity.share(customer, favorite, sku, as_of)
            shared[sku] = (count / (count + 5), popularity, industry_share, category_share, peer_share)

        def row(sku, date):
            count, *rest = shared[sku]
            days = max(0, (as_of.normalize() - date.normalize()).days) if date is not None else 0
            return (count, days / (days + 90), *rest)

        if len(positions):
            history[i, positions] = torch.tensor([row(event[1], event[0]) for event in events])
            brands[i, positions] = torch.tensor([
                brand_vocab.get(store.get_product_brand(event[1]), 0) for event in events
            ])
        candidates[i] = torch.tensor([
            row(sku, snapshot.sku_stats.get(sku, {}).get("last_date")) for sku in skus
        ])
    return {**slate, "ranker_sequence": history, "seq_brand_ids": brands, "pairwise_features": candidates}
