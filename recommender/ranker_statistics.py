"""Shared point-in-time SKU statistics for Stage 2 queries and history."""

from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import groupby
import math
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from recommender.candidates import PointInTimeFeatureStore
from recommender.purchase_timing import PurchaseTiming, _day
from recommender.sequences import ENRICHED_FEATURE_DIM
from recommender.split import TRAIN_CUTOFF


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


ALIGNED_11D_STATISTICS = (
    "purchase_frequency_norm",
    "item_order_quantity_norm",
    "customer_volume_share",
    "recency_vs_last_order_norm",
    "tenure_vs_purchase_age_norm",
    "reorder_due_score",
    "catalog_sku_popularity_norm",
    "industry_sku_affinity",
    "category_affinity_share",
    "brand_affinity_share",
    "business_line_affinity_share",
)


def _hyperbolic_days(days: float) -> float:
    d = max(0.0, float(days))
    return d / (d + 90.0)


def _customer_order_dates_and_line_counts(
    store: PointInTimeFeatureStore, customer: str, as_of: pd.Timestamp,
) -> tuple[list[pd.Timestamp], Counter]:
    end = bisect_left(store.customer_event_dates.get(customer, ()), as_of)
    prior_events = store.customer_events.get(customer, ())[:end]
    order_dates: list[pd.Timestamp] = []
    line_counts: Counter = Counter()
    last_day = None
    for dt, _, _, _, _, _, line in prior_events:
        day_norm = dt.normalize()
        if last_day is None or day_norm != last_day:
            order_dates.append(day_norm)
            last_day = day_norm
        if line:
            line_counts[line] += 1
    return order_dates, line_counts


def _sku_reorder_gap_days(
    snapshot, timing: PurchaseTiming, sku: str, as_of: pd.Timestamp,
) -> float:
    sku_info = snapshot.sku_stats.get(sku)
    if sku_info is not None and sku_info["order_count"] >= 2 and sku_info.get("median_gap", 0.0) > 0.0:
        return float(sku_info["median_gap"])
    pop_cutoff = min(as_of, TRAIN_CUTOFF)
    return float(timing.population_gap(sku, _day(pop_cutoff)))


def with_due_score_statistics(
    slate: dict,
    transitions: list[dict],
    store: PointInTimeFeatureStore,
    timing: PurchaseTiming,
    *,
    update_candidates: bool = False,
) -> dict:
    """Replace raw history inter-purchase gap and company median gap with (gap_norm - age_norm) using d/(d+90)."""
    seq = slate["sequence"].clone()
    tab = slate["tabular_features"].clone()
    pw = slate["pairwise_features"].clone() if update_candidates else slate["pairwise_features"]

    for i, customer, as_of, skus in _aligned_slates(slate, transitions):
        as_of_day = as_of.normalize()
        snapshot = store.get_customer_snapshot(customer, as_of)
        order_dates, _ = _customer_order_dates_and_line_counts(store, customer, as_of)

        if order_dates:
            last_order_days = max(0, (as_of_day - order_dates[-1]).days)
            last_order_age_norm = _hyperbolic_days(last_order_days)
            if len(order_dates) >= 2:
                gaps = [max(0, (order_dates[j] - order_dates[j - 1]).days) for j in range(1, len(order_dates))]
                med_order_gap = float(np.median(gaps)) if gaps else 0.0
                company_due = _hyperbolic_days(med_order_gap) - last_order_age_norm
            else:
                company_due = 0.0
        else:
            company_due = 0.0
        tab[i, 7] = float(company_due)

        positions, events = _history_rows(slate, store, i, customer, as_of)
        if len(positions):
            # Track per-SKU prior purchase dates within this customer's history
            end = bisect_left(store.customer_event_dates.get(customer, ()), as_of)
            all_prior = store.customer_events.get(customer, ())[:end]
            sku_prev_date: dict[str, pd.Timestamp] = {}
            event_sku_gaps: list[float] = []
            for dt, pid, *_ in all_prior:
                d_norm = dt.normalize()
                prev = sku_prev_date.get(pid)
                if prev is not None and (d_norm - prev).days > 0:
                    event_sku_gaps.append(float((d_norm - prev).days))
                else:
                    event_sku_gaps.append(_sku_reorder_gap_days(snapshot, timing, pid, as_of))
                sku_prev_date[pid] = d_norm
            tail_gaps = event_sku_gaps[-len(positions):]
            due_vals = []
            for (dt, pid, *_), g_days in zip(events, tail_gaps):
                age_days = max(0, (as_of_day - dt.normalize()).days)
                due_vals.append(_hyperbolic_days(g_days) - _hyperbolic_days(age_days))
            seq[i, positions, 2] = torch.tensor(due_vals, dtype=seq.dtype)

        if update_candidates:
            cand_ages = []
            cand_dues = []
            for sku in skus:
                sku_info = snapshot.sku_stats.get(sku)
                if sku_info is not None and sku_info.get("last_date") is not None:
                    age_days = max(0, (as_of_day - sku_info["last_date"].normalize()).days)
                    age_norm = _hyperbolic_days(age_days)
                    gap_days = _sku_reorder_gap_days(snapshot, timing, sku, as_of)
                    due_val = _hyperbolic_days(gap_days) - age_norm if gap_days > 0 else 0.0
                else:
                    age_norm = 1.0
                    due_val = 0.0
                cand_ages.append(age_norm)
                cand_dues.append(due_val)
            pw[i, :, 3] = torch.tensor(cand_ages, dtype=pw.dtype)
            pw[i, :, 4] = torch.tensor(cand_dues, dtype=pw.dtype)

    return {**slate, "ranker_sequence": seq, "tabular_features": tab, "pairwise_features": pw}


def with_aligned_11d_statistics(
    slate: dict,
    transitions: list[dict],
    store: PointInTimeFeatureStore,
    timing: PurchaseTiming,
    *,
    quantity_reference: float | None = None,
) -> dict:
    """Build 11-d aligned correlating statistics for Token 0, Tokens 1..32, and Candidates 1..50."""
    n_rows, seq_len = slate["attention_mask"].shape
    n_cands = slate["candidate_indices"].shape[1]
    width = len(ALIGNED_11D_STATISTICS)

    company_11d = torch.zeros((n_rows, width), dtype=torch.float32)
    history_11d = torch.zeros((n_rows, seq_len, width), dtype=torch.float32)
    candidate_11d = torch.zeros((n_rows, n_cands, width), dtype=torch.float32)

    base_tab = slate["tabular_features"]
    base_seq = slate["sequence"]
    base_pw = slate["pairwise_features"]
    qty_ref_log = math.log1p(quantity_reference) if quantity_reference and quantity_reference > 0 else None

    for i, customer, as_of, skus in _aligned_slates(slate, transitions):
        as_of_day = as_of.normalize()
        snapshot = store.get_customer_snapshot(customer, as_of)
        order_dates, line_counts = _customer_order_dates_and_line_counts(store, customer, as_of)
        total_orders = max(snapshot.total_orders, 1)

        if order_dates:
            last_order_age = _hyperbolic_days(max(0, (as_of_day - order_dates[-1]).days))
            first_order_age = _hyperbolic_days(max(0, (as_of_day - order_dates[0]).days))
            if len(order_dates) >= 2:
                gaps = [max(0, (order_dates[j] - order_dates[j - 1]).days) for j in range(1, len(order_dates))]
                med_gap = float(np.median(gaps)) if gaps else 0.0
                company_due = _hyperbolic_days(med_gap) - last_order_age
            else:
                company_due = 0.0
        else:
            last_order_age = 1.0
            first_order_age = 1.0
            company_due = 0.0

        tot_purch = float(base_tab[i, 0].item())
        tot_qty = float(base_tab[i, 1].item())
        dist_prod = float(base_tab[i, 2].item())
        dist_cat = float(base_tab[i, 4].item())

        company_11d[i] = torch.tensor([
            tot_purch,          # 0: Purchase Frequency
            tot_qty,            # 1: Item / Order Quantity (dim 1 of 2)
            tot_qty,            # 2: Customer Volume Share (dim 2 of 2, duplicated)
            last_order_age,     # 3: Recency vs. Last Order
            first_order_age,    # 4: Tenure vs. Purchase Age
            company_due,        # 5: Reorder Due-ness (gap - age)
            dist_prod,          # 6: Catalog / SKU Popularity (dim 1 of 2)
            dist_prod,          # 7: Industry Affinity (dim 2 of 2, duplicated)
            dist_cat,           # 8: Category Affinity (dim 1 of 3)
            dist_cat,           # 9: Brand Affinity (dim 2 of 3, duplicated)
            dist_cat,           # 10: Business Line Affinity (dim 3 of 3, duplicated)
        ], dtype=torch.float32)

        positions, events = _history_rows(slate, store, i, customer, as_of)
        if len(positions):
            end = bisect_left(store.customer_event_dates.get(customer, ()), as_of)
            all_prior = store.customer_events.get(customer, ())[:end]
            sku_prev_date: dict[str, pd.Timestamp] = {}
            event_sku_gaps: list[float] = []
            for dt, pid, *_ in all_prior:
                d_norm = dt.normalize()
                prev = sku_prev_date.get(pid)
                if prev is not None and (d_norm - prev).days > 0:
                    event_sku_gaps.append(float((d_norm - prev).days))
                else:
                    event_sku_gaps.append(_sku_reorder_gap_days(snapshot, timing, pid, as_of))
                sku_prev_date[pid] = d_norm
            tail_gaps = event_sku_gaps[-len(positions):]

            hist_rows = []
            for pos_idx, (dt, pid, _qty, cat, _subcat, brand, line), g_days in zip(
                positions.tolist(), events, tail_gaps
            ):
                age_days = max(0, (as_of_day - dt.normalize()).days)
                age_norm = _hyperbolic_days(age_days)
                due_val = _hyperbolic_days(g_days) - age_norm if g_days > 0 else 0.0
                pop, ind_aff = store.sku_population_statistics(customer, as_of, pid)
                cat_share = snapshot.cat_orders.get(cat, 0) / total_orders if cat else 0.0
                brand_share = snapshot.brand_orders.get(brand, 0) / total_orders if brand else 0.0
                line_share = line_counts.get(line, 0) / total_orders if line else 0.0
                hist_rows.append([
                    float(base_seq[i, pos_idx, 5].item()),  # 0: user_product_prior_count_norm
                    float(base_seq[i, pos_idx, 0].item()),  # 1: qty_norm
                    float(base_seq[i, pos_idx, 6].item()),  # 2: user_product_quantity_share
                    age_norm,                               # 3: purchase_age_norm (dim 1 of 2)
                    age_norm,                               # 4: purchase_age_norm (dim 2 of 2, duplicated)
                    due_val,                                # 5: sku_gap_norm - purchase_age_norm
                    pop,                                    # 6: global_sku_popularity_norm
                    ind_aff,                                # 7: industry_sku_affinity
                    cat_share,                              # 8: user_cat_order_share
                    brand_share,                            # 9: user_brand_order_share
                    line_share,                             # 10: user_line_order_share
                ])
            history_11d[i, positions] = torch.tensor(hist_rows, dtype=torch.float32)

        cand_rows = []
        for c_idx, sku in enumerate(skus):
            sku_info = snapshot.sku_stats.get(sku)
            meta = store.product_metadata.get(sku, {})
            line = meta.get("business_line", "")
            line_share = line_counts.get(line, 0) / total_orders if line else 0.0
            if sku_info is not None and sku_info["order_count"] > 0:
                avg_qty = max(0.0, sku_info["volume"] / sku_info["order_count"])
                if qty_ref_log is not None:
                    lq = math.log1p(avg_qty)
                    sku_qty_norm = lq / (lq + qty_ref_log)
                else:
                    sku_qty_norm = min(1.0, math.log1p(avg_qty) / math.log1p(100.0))
                age_days = max(0, (as_of_day - sku_info["last_date"].normalize()).days)
                age_norm = _hyperbolic_days(age_days)
                gap_days = _sku_reorder_gap_days(snapshot, timing, sku, as_of)
                due_val = _hyperbolic_days(gap_days) - age_norm if gap_days > 0 else 0.0
            else:
                sku_qty_norm = 0.0
                age_norm = 1.0
                due_val = 0.0

            cand_rows.append([
                float(base_pw[i, c_idx, 1].item()),  # 0: user_sku_order_count_norm
                float(sku_qty_norm),                 # 1: user_sku_qty_norm
                float(base_pw[i, c_idx, 2].item()),  # 2: user_sku_qty_share
                age_norm,                            # 3: days_since_last_sku_buy_norm (dim 1 of 2)
                age_norm,                            # 4: days_since_last_sku_buy_norm (dim 2 of 2)
                due_val,                             # 5: sku_gap_norm - days_since_last_sku_buy_norm
                float(base_pw[i, c_idx, 8].item()),  # 6: global_sku_popularity_norm
                float(base_pw[i, c_idx, 9].item()),  # 7: industry_sku_affinity
                float(base_pw[i, c_idx, 5].item()),  # 8: user_cat_order_share
                float(base_pw[i, c_idx, 7].item()),  # 9: user_brand_order_share
                float(line_share),                   # 10: user_line_order_share
            ])
        candidate_11d[i] = torch.tensor(cand_rows, dtype=torch.float32)

    return {
        **slate,
        "aligned_company_stats": company_11d,
        "aligned_history_stats": history_11d,
        "aligned_candidate_stats": candidate_11d,
    }


def _top1_attribute(counts: Mapping[str, int]) -> str:
    known = Counter({k: v for k, v in counts.items() if k and v > 0})
    if not known:
        return ""
    return known.most_common(1)[0][0]


def with_favorite_attribute_features(
    slate: dict,
    transitions: list[dict],
    store: PointInTimeFeatureStore,
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    append_candidate_features: bool = False,
) -> dict:
    """Attach customer favorite category/brand/line IDs (for Token 0) and candidate affinity shares."""
    n_rows = len(transitions)
    n_cands = slate["candidate_indices"].shape[1]
    fav_cat_ids = torch.zeros(n_rows, dtype=torch.long)
    fav_brand_ids = torch.zeros(n_rows, dtype=torch.long)
    fav_line_ids = torch.zeros(n_rows, dtype=torch.long)
    cat_vocab = vocabs.get("category_vocab", {})
    brand_vocab = vocabs.get("brand_vocab", {})
    line_vocab = vocabs.get("business_line_vocab", {})

    extra_cand = torch.zeros((n_rows, n_cands, 4), dtype=torch.float32) if append_candidate_features else None

    for i, customer, as_of, skus in _aligned_slates(slate, transitions):
        snapshot = store.get_customer_snapshot(customer, as_of)
        _, line_counts = _customer_order_dates_and_line_counts(store, customer, as_of)
        total_orders = max(snapshot.total_orders, 1)

        fav_cat = _top1_attribute(snapshot.cat_orders)
        fav_brand = _top1_attribute(snapshot.brand_orders)
        fav_line = _top1_attribute(line_counts)

        fav_cat_ids[i] = int(cat_vocab.get(fav_cat, 0)) if fav_cat else 0
        fav_brand_ids[i] = int(brand_vocab.get(fav_brand, 0)) if fav_brand else 0
        fav_line_ids[i] = int(line_vocab.get(fav_line, 0)) if fav_line else 0

        if extra_cand is not None:
            for c_idx, sku in enumerate(skus):
                meta = store.product_metadata.get(sku, {})
                cat = meta.get("product_category", "")
                brand = meta.get("brand", "")
                line = meta.get("business_line", "")
                line_share = line_counts.get(line, 0) / total_orders if line else 0.0
                extra_cand[i, c_idx, 0] = float(line_share)
                extra_cand[i, c_idx, 1] = 1.0 if (fav_cat and cat == fav_cat) else 0.0
                extra_cand[i, c_idx, 2] = 1.0 if (fav_brand and brand == fav_brand) else 0.0
                extra_cand[i, c_idx, 3] = 1.0 if (fav_line and line == fav_line) else 0.0

    result = {
        **slate,
        "fav_category_id": fav_cat_ids,
        "fav_brand_id": fav_brand_ids,
        "fav_business_line_id": fav_line_ids,
    }
    if extra_cand is not None:
        result["pairwise_features"] = torch.cat([slate["pairwise_features"], extra_cand], dim=-1)
    return result
