"""Purchase cadence features for the Stage 2 ranker."""

from functools import lru_cache
import math

import numpy as np
import pandas as pd
import torch

from recommender.candidates import PAIRWISE_FEATURE_NAMES
from recommender.split import TRAIN_CUTOFF

TIMING_FEATURE_NAMES = ("typical_reorder_gap_norm",)
TIMING_FEATURE_VERSION = 4
URGENCY_INDEX = PAIRWISE_FEATURE_NAMES.index("sku_urgency")


def _day(date: pd.Timestamp) -> int:
    return int(date.to_datetime64().astype("datetime64[D]").astype(np.int64))


def _pooled_gaps(purchases: pd.DataFrame, column: str) -> dict:
    """Index gaps by the date of the second purchase, using distinct purchase days."""
    events = purchases.drop_duplicates(["customer_id", column, "day"]).sort_values("day")
    events = events.assign(gap=events.groupby(["customer_id", column])["day"].diff())
    return {
        key: (rows["day"].to_numpy(), rows["gap"].to_numpy())
        for key, rows in events.loc[events["gap"] > 0].groupby(column)
    }


class PurchaseTiming:
    def __init__(self, purchases: pd.DataFrame, catalogue: pd.DataFrame, *, category_gap_fallback: bool = True):
        self.category_gap_fallback = category_gap_fallback
        self.category_by_sku = catalogue.set_index("product_id")["product_category"].dropna().to_dict()
        history = purchases[["customer_id", "product_id", "purchase_date"]].copy()
        history["day"] = pd.to_datetime(history["purchase_date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
        history = history.drop_duplicates(["customer_id", "product_id", "day"]).sort_values("day")
        self.personal_days = {
            key: group["day"].to_numpy()
            for key, group in history.groupby(["customer_id", "product_id"])
        }
        training = history.loc[history["day"] < _day(TRAIN_CUTOFF)].copy()
        training["category"] = training["product_id"].map(self.category_by_sku)
        self.sku_gaps = _pooled_gaps(training, "product_id")
        # Category fallback is the equal-weight mean of customer/SKU medians.
        # Keep each pair's gap timeline so a query only uses intervals whose
        # second purchase was strictly before its as-of cutoff.
        category_pair_gaps = training.assign(
            gap=training.groupby(["customer_id", "product_id"])["day"].diff()
        )
        self.category_pair_gaps = {}
        for (category, _customer, _sku), rows in category_pair_gaps.loc[
            category_pair_gaps["gap"].gt(0) & category_pair_gaps["category"].notna()
        ].groupby(["category", "customer_id", "product_id"]):
            self.category_pair_gaps.setdefault(category, []).append(
                (rows["day"].to_numpy(), rows["gap"].to_numpy())
            )
        self.population_gap = lru_cache(maxsize=200_000)(self._population_gap)
        self.category_gap = lru_cache(maxsize=200_000)(self._category_gap)

    @staticmethod
    def _median_before(timeline: tuple[np.ndarray, np.ndarray] | None, cutoff: int) -> float:
        if timeline is None:
            return 0.0
        days, gaps = timeline
        end = np.searchsorted(days, cutoff)
        return float(np.median(gaps[:end])) if end else 0.0

    def _population_gap(self, sku: str, cutoff: int) -> float:
        gap = self._median_before(self.sku_gaps.get(sku), cutoff)
        if gap or not self.category_gap_fallback:
            return gap
        return self.category_gap(self.category_by_sku.get(sku), cutoff)

    def _category_gap(self, category: str | None, cutoff: int) -> float:
        """Mean valid customer/SKU median gaps, with equal weight per pair."""
        if category is None:
            return 0.0
        medians = []
        for days, gaps in self.category_pair_gaps.get(category, ()):
            end = np.searchsorted(days, cutoff)
            if end:
                medians.append(float(np.median(gaps[:end])))
        return float(np.mean(medians)) if medians else 0.0

    def candidate_features(self, customer: str, sku: str, as_of: pd.Timestamp) -> tuple[float, float]:
        day = _day(as_of)
        population_cutoff = min(as_of, TRAIN_CUTOFF)
        history = self.personal_days.get((customer, sku))
        end = np.searchsorted(history, day) if history is not None else 0
        gap = float(np.median(np.diff(history[:end]))) if end >= 2 else 0.0
        gap = gap or self.population_gap(sku, _day(population_cutoff))
        urgency = min(1.0, (day - history[end - 1]) / (gap * 5)) if gap and end else 0.0
        gap_norm = min(1.0, math.log1p(gap) / math.log1p(365))
        return urgency, gap_norm


def add_candidate_timing(slate: dict, transitions: list[dict], timing: PurchaseTiming) -> dict:
    """Add timing to a replay of the same candidate lists and labels."""
    if slate["customer_ids"] != [row["customer_id"] for row in transitions] or any(
        set(pids) != set(row["positive_product_ids"])
        for pids, row in zip(slate["positive_pids"], transitions, strict=True)
    ):
        raise ValueError("Candidate timing rows do not match the cached transitions.")
    features = torch.empty((*slate["pairwise_features"].shape[:2], 1 + len(TIMING_FEATURE_NAMES)))
    for i, (row, pids) in enumerate(zip(transitions, slate["candidate_pids"], strict=True)):
        date = pd.Timestamp(row["as_of_date"])
        features[i] = torch.tensor([timing.candidate_features(row["customer_id"], sku, date) for sku in pids])
    pairwise = torch.cat([slate["pairwise_features"], features[..., 1:]], dim=-1)
    pairwise[..., URGENCY_INDEX] = features[..., 0]
    return {**slate, "pairwise_features": pairwise}
