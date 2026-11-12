"""Candidate Generation and Point-in-Time Pairwise Features for Stage 2 Ranking (Milestone M9).

Implements:
1. PointInTimeFeatureStore:
   - Pre-aggregates customer historical statistics and industry affinities strictly
     point-in-time (< as_of_date) to allow sub-millisecond pairwise feature extraction
     per candidate slate.
   - Looks up global popularity and industry affinity using sales before the
     scoring date, with both counts capped at the training cutoff.
2. 13 Point-in-Time Pairwise Features strictly bounded in [0.0, 1.0] with zero NaNs:
   Exact-SKU history, category/subcategory/brand shares, SKU popularity,
   industry affinity, and category/brand/business-line repeat flags.
3. Candidate Slate Construction:
   - Targets label retrieved candidates without changing the slate.
4. CandidateBatch:
   - Supports dictionary-style access and device transfer.
5. CandidateRetriever:
   - Loads checkpoints and generates candidate batches.
"""

from collections import OrderedDict
from bisect import bisect_left
from heapq import heappush, heappop
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from recommender.catalogue import RollingCandidatePool
from recommender.features import LEGACY_PROFILE_FEATURES, PointInTimeNormalizer
from recommender.familiarity import FamiliarityStore
from recommender.models import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEMPERATURE,
    OUTPUT_DIM,
    TEXT_DIM,
    CustomerTower,
    ProductTower,
    TwoTowerModel,
)
from recommender.scope import actual_purchase_mask
from recommender.split import TRAIN_CUTOFF

PAIRWISE_FEATURE_NAMES: List[str] = [
    "is_repeat_sku",
    "user_sku_order_count_norm",
    "user_sku_qty_share",
    "days_since_last_sku_buy_norm",
    "sku_urgency",
    "user_cat_order_share",
    "user_subcat_order_share",
    "user_brand_order_share",
    "global_sku_popularity_norm",
    "industry_sku_affinity",
    "is_repeat_category",
    "is_repeat_brand",
    "is_repeat_business_line",
]
NUM_PAIRWISE_FEATURES: int = len(PAIRWISE_FEATURE_NAMES)
DEFAULT_TOP_K: int = 100
DEFAULT_TOP_C: int = 50
DEFAULT_HEURISTIC_FRACTION: float = 0.5


def rank_personal_repeat(
    sku_stats: Dict[str, Tuple[int, pd.Timestamp]],
    global_popularity: Sequence[str],
    max_k: int,
) -> List[str]:
    """Rank eligible purchases by frequency and recency, then fill by popularity."""
    eligible = set(global_popularity)
    ranked = sorted(
        (sku for sku in sku_stats if sku in eligible),
        key=lambda sku: (-sku_stats[sku][0], -sku_stats[sku][1].value, sku),
    )
    seen = set(ranked)
    for sku in global_popularity:
        if len(ranked) >= max_k:
            break
        if sku not in seen:
            ranked.append(sku)
            seen.add(sku)
    return ranked[:max_k]


def rank_unseen_candidates(
    scores: torch.Tensor,
    product_ids: Sequence[str],
    eligible_by_query: Sequence[set[str]],
    seen_by_query: Sequence[set[str]],
    top_k: int,
) -> List[List[str]]:
    """Rank available unseen SKUs and return shorter lists for sparse pools."""
    if top_k < 1 or scores.shape != (len(eligible_by_query), len(product_ids)):
        raise ValueError("Discovery scores, catalogue IDs, query pools and top_k must align.")
    if len(seen_by_query) != len(eligible_by_query):
        raise ValueError("Discovery requires one seen-SKU set per query.")
    if not product_ids:
        return [[] for _ in eligible_by_query]
    index = {sku: i for i, sku in enumerate(product_ids)}
    excluded = torch.ones(scores.shape, dtype=torch.bool)
    allowed = [eligible - seen for eligible, seen in zip(eligible_by_query, seen_by_query)]
    for row, products in enumerate(allowed):
        excluded[row, [index[sku] for sku in products if sku in index]] = False
    ranked = scores.masked_fill(excluded.to(scores.device), -torch.inf).topk(min(top_k, len(index)), dim=-1)
    return [
        [product_ids[i] for i in row if product_ids[i] in products]
        for row, products in zip(ranked.indices.tolist(), allowed)
    ]


@dataclass
class CandidateBatch:
    """Batch of candidate items and pairwise features for Stage 2 ranker or evaluation.

    Attributes:
        customer_ids: List of B customer ID strings.
        candidate_pids: List of length B, each containing C candidate product ID strings.
        candidate_indices: LongTensor of shape (B, C) with catalog product indices.
        candidate_scores: FloatTensor of shape (B, C) with Stage 1 scores.
        pairwise_features: FloatTensor of shape (B, C, 13) strictly bounded in [0.0, 1.0].
        labels: Optional FloatTensor of shape (B, C) with binary ground-truth labels {0.0, 1.0}.
        top100_indices: Optional LongTensor of shape (B, K) with Stage 1 top-K catalog indices.
        top100_scores: Optional FloatTensor of shape (B, K) with Stage 1 top-K scores.
        top100_pids: Optional List of length B, each containing K product ID strings.
    """

    customer_ids: List[str]
    candidate_pids: List[List[str]]
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    pairwise_features: torch.Tensor
    labels: Optional[torch.Tensor] = None
    top100_indices: Optional[torch.Tensor] = None
    top100_scores: Optional[torch.Tensor] = None
    top100_pids: Optional[List[List[str]]] = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"CandidateBatch has no attribute '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to(self, device: Union[str, torch.device]) -> "CandidateBatch":
        """Move tensor attributes to target device."""
        target_device = torch.device(device) if isinstance(device, str) else device
        return CandidateBatch(
            customer_ids=self.customer_ids,
            candidate_pids=self.candidate_pids,
            candidate_indices=self.candidate_indices.to(target_device)
            if self.candidate_indices is not None
            else None,
            candidate_scores=self.candidate_scores.to(target_device)
            if self.candidate_scores is not None
            else None,
            pairwise_features=self.pairwise_features.to(target_device)
            if self.pairwise_features is not None
            else None,
            labels=self.labels.to(target_device) if self.labels is not None else None,
            top100_indices=self.top100_indices.to(target_device)
            if self.top100_indices is not None
            else None,
            top100_scores=self.top100_scores.to(target_device)
            if self.top100_scores is not None
            else None,
            top100_pids=self.top100_pids,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateBatch to dictionary."""
        return {
            "customer_ids": self.customer_ids,
            "candidate_pids": self.candidate_pids,
            "candidate_indices": self.candidate_indices,
            "candidate_scores": self.candidate_scores,
            "pairwise_features": self.pairwise_features,
            "labels": self.labels,
            "top100_indices": self.top100_indices,
            "top100_scores": self.top100_scores,
            "top100_pids": self.top100_pids,
        }


@dataclass
class CustomerSnapshot:
    """Pre-aggregated customer historical state strictly prior to as_of_date."""

    total_volume: float
    total_orders: int
    cat_orders: Dict[str, int]
    subcat_orders: Dict[str, int]
    brand_orders: Dict[str, int]
    business_lines: set[str]
    sku_stats: Dict[str, Dict[str, Any]]
    # sku_stats[pid] = {
    #     "order_count": int,
    #     "volume": float,
    #     "last_date": pd.Timestamp,
    #     "median_gap": float,
    # }


class PointInTimeFeatureStore:
    """Pre-aggregates customer historical statistics and industry affinities strictly point-in-time.

    Guarantees:
    - Purchase statistics use strictly prior events; snapshot metadata is an explicit assumption.
    - observed_only_v1 excludes unverifiable master attributes from pairwise features.
    - Sub-millisecond pairwise feature extraction per candidate slate via indexed snapshot lookups.
    - All 13 pairwise features bounded strictly in [0.0, 1.0] with zero NaNs.
    """

    def __init__(
        self,
        purchases_df: pd.DataFrame,
        catalogue_df: Optional[pd.DataFrame] = None,
        companies_df: Optional[pd.DataFrame] = None,
        train_cutoff: Union[str, pd.Timestamp] = TRAIN_CUTOFF,
        *, metadata_policy: str = "snapshot_unverified", max_snapshots: int = 1024,
        max_prefixes: int = 128, max_cached_dates: int = 512,
    ):
        if metadata_policy not in ("snapshot_unverified", "observed_only_v1"):
            raise ValueError("Unknown pairwise metadata policy")
        if min(max_snapshots, max_prefixes, max_cached_dates) < 0:
            raise ValueError("Cache capacities must be nonnegative")
        self.metadata_policy = metadata_policy
        self.max_snapshots, self.max_prefixes, self.max_cached_dates = max_snapshots, max_prefixes, max_cached_dates
        self._prefix_states = OrderedDict()
        self._observed_metadata = {}
        self._observed_metadata_dates = {}
        if metadata_policy == "observed_only_v1":
            # No effective-date proof exists for these snapshots. Unknown is safer
            # than backfilling product attributes or industry from current masters.
            catalogue_df = companies_df = None
        self.train_cutoff = pd.Timestamp(train_cutoff)

        # 1. Product metadata indexing (brand, category, subcategory, business_line)
        self.product_metadata: Dict[str, Dict[str, str]] = {}
        if catalogue_df is not None and not catalogue_df.empty:
            for _, row in catalogue_df.iterrows():
                pid = str(row["product_id"]).strip()
                self.product_metadata[pid] = {
                    "brand": str(row.get("brand", "")).strip() if pd.notna(row.get("brand")) else "",
                    "product_category": str(row.get("product_category", "")).strip()
                    if pd.notna(row.get("product_category"))
                    else "",
                    "product_subcategory": str(row.get("product_subcategory", "")).strip()
                    if pd.notna(row.get("product_subcategory"))
                    else "",
                    "business_line": str(row.get("business_line", "")).strip()
                    if pd.notna(row.get("business_line"))
                    else "",
                }

        # 2. Customer metadata indexing (industry_branch, city, region_code)
        self.customer_metadata: Dict[str, Dict[str, str]] = {}
        if companies_df is not None and not companies_df.empty:
            for _, row in companies_df.iterrows():
                cid = str(row["customer_id"]).strip()
                self.customer_metadata[cid] = {
                    "industry_branch": str(row.get("industry_branch", "")).strip()
                    if pd.notna(row.get("industry_branch"))
                    else "",
                    "city": str(row.get("city", "")).strip() if pd.notna(row.get("city")) else "",
                    "region_code": str(row.get("region_code", "")).strip()
                    if pd.notna(row.get("region_code"))
                    else "",
                }

        # 3. Clean and parse purchases
        df = purchases_df.copy()
        if not df.empty:
            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["product_id"] = df["product_id"].astype(str).str.strip()
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed")
            if "quantity" in df.columns:
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0.0)
            else:
                df["quantity"] = 1.0

            if self.metadata_policy == "observed_only_v1":
                for column in ("brand", "product_category", "product_subcategory", "business_line"):
                    if column not in df:
                        continue
                    for pid, rows in df.sort_values("purchase_date", kind="stable").groupby("product_id"):
                        values = [(date, str(value).strip()) for date, value in zip(rows.purchase_date, rows[column])
                                  if pd.notna(value) and str(value).strip()]
                        self._observed_metadata[(pid, column)] = [value for _, value in values]
                        self._observed_metadata_dates[(pid, column)] = [date for date, _ in values]

            # Legacy snapshots remain available for historical checkpoint replay.
            # Strict mode consults purchase observations by date instead.
            for col in ("product_category", "business_line", "product_name"):
                if col in df.columns and self.metadata_policy != "observed_only_v1":
                    for pid, val in df.groupby("product_id")[col].first().items():
                        pid_str = str(pid).strip()
                        if pid_str not in self.product_metadata:
                            self.product_metadata[pid_str] = {}
                        if col not in self.product_metadata[pid_str] or not self.product_metadata[pid_str][col]:
                            self.product_metadata[pid_str][col] = str(val).strip() if pd.notna(val) else ""

        # 4. Index the global and industry histories used by features 11 and 12.
        self._sku_sale_dates: Dict[str, List[pd.Timestamp]] = {}
        self._industry_sale_dates: Dict[str, List[pd.Timestamp]] = {}
        self._industry_sku_sale_dates: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
        self._popularity_scale = 1.0
        self._index_training_sales(df)

        # 5. Build per-customer chronological event logs for point-in-time snapshot generation
        self.customer_events: Dict[str, List[Tuple[pd.Timestamp, str, float, str, str, str, str]]] = {}
        self.customer_event_dates: Dict[str, List[pd.Timestamp]] = {}
        self._index_customer_events(df)

        # 6. Snapshot cache for sub-millisecond retrieval
        self._snapshots = OrderedDict()
        self._popularity_rankings = OrderedDict()

    def _index_training_sales(self, df: pd.DataFrame) -> None:
        """Fit the popularity scale and index training events for as-of counts."""
        if df.empty:
            return

        train = df.loc[df["purchase_date"] < self.train_cutoff].sort_values("purchase_date")
        self._sku_sale_dates = train.groupby("product_id")["purchase_date"].agg(list).to_dict()
        maximum = max(map(len, self._sku_sale_dates.values()), default=1)
        self._popularity_scale = math.log1p(maximum)

        industries = {cid: meta["industry_branch"] for cid, meta in self.customer_metadata.items()}
        train = train.assign(industry=train["customer_id"].map(industries).fillna(""))
        known = train.loc[train["industry"] != ""]
        self._industry_sale_dates = known.groupby("industry")["purchase_date"].agg(list).to_dict()
        self._industry_sku_sale_dates = (
            known.groupby(["industry", "product_id"])["purchase_date"].agg(list).to_dict()
        )

    def _index_customer_events(self, df: pd.DataFrame) -> None:
        """Index customer events chronologically for fast binary search point-in-time filtering."""
        if df.empty:
            return

        for cid, grp in df.groupby("customer_id"):
            cid_str = str(cid).strip()
            sorted_grp = grp.sort_values(["purchase_date", "product_id"])

            events: List[Tuple[pd.Timestamp, str, float, str, str, str, str]] = []
            dates: List[pd.Timestamp] = []

            for _, row in sorted_grp.iterrows():
                dt = pd.Timestamp(row["purchase_date"])
                pid = str(row["product_id"]).strip()
                qty = float(row.get("quantity", 1.0))

                meta = self.product_metadata.get(pid, {})
                if self.metadata_policy == "observed_only_v1":
                    meta = {key: str(row.get(key)).strip() if pd.notna(row.get(key)) else ""
                            for key in ("brand", "product_category", "product_subcategory", "business_line")}
                cat = meta.get("product_category", "")
                subcat = meta.get("product_subcategory", "")
                brand = meta.get("brand", "")

                line = meta.get("business_line", "")

                events.append((dt, pid, qty, cat, subcat, brand, line))
                dates.append(dt)

            self.customer_events[cid_str] = events
            self.customer_event_dates[cid_str] = dates

    def get_customer_industry(self, customer_id: str) -> str:
        """Get industry branch string for customer."""
        cid = str(customer_id).strip()
        return self.customer_metadata.get(cid, {}).get("industry_branch", "")

    def get_product_attribute(self, product_id, field, as_of_date=None):
        pid = str(product_id).strip()
        if self.metadata_policy == "observed_only_v1":
            if as_of_date is None:
                return ""
            key = (pid, field)
            idx = bisect_left(self._observed_metadata_dates.get(key, ()), pd.Timestamp(as_of_date))
            return self._observed_metadata[key][idx - 1] if idx else ""
        return self.product_metadata.get(pid, {}).get(field, "")

    def get_product_category(self, product_id, as_of_date=None):
        return self.get_product_attribute(product_id, "product_category", as_of_date)

    def get_product_subcategory(self, product_id, as_of_date=None):
        return self.get_product_attribute(product_id, "product_subcategory", as_of_date)

    def get_product_brand(self, product_id, as_of_date=None):
        return self.get_product_attribute(product_id, "brand", as_of_date)

    def sku_population_statistics(self, customer_id: str, as_of: pd.Timestamp, sku: str) -> tuple[float, float]:
        """Training-history popularity and share of this industry's purchases."""
        count = bisect_left(self._sku_sale_dates.get(sku, ()), as_of)
        popularity = math.log1p(count) / self._popularity_scale
        industry = self.get_customer_industry(customer_id)
        industry_count = bisect_left(self._industry_sale_dates.get(industry, ()), as_of)
        if not industry_count:
            return popularity, 0.0
        sku_count = bisect_left(self._industry_sku_sale_dates.get((industry, sku), ()), as_of)
        return popularity, sku_count / industry_count

    @staticmethod
    def _remember(cache, key, value, capacity):
        if capacity:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > capacity:
                cache.popitem(last=False)
        return value

    def get_customer_snapshot(self, customer_id, as_of_date):
        """Extend chronological prefix statistics; backdated requests rebuild safely.

        Cached snapshots own their dictionaries, so later events cannot mutate an
        earlier result. Prefix state and snapshots both have bounded LRU storage.
        Existing event-count and median-gap semantics are unchanged.
        """
        cid, as_of = str(customer_id).strip(), pd.Timestamp(as_of_date)
        key = (cid, as_of)
        if key in self._snapshots:
            self._snapshots.move_to_end(key)
            return self._snapshots[key]
        events = self.customer_events.get(cid, ())
        end = bisect_left(self.customer_event_dates.get(cid, ()), as_of)
        previous = self._prefix_states.get(cid)
        state = previous if previous is not None and previous["end"] <= end else {
            "end": 0, "volume": 0., "cats": {}, "subcats": {}, "brands": {}, "lines": set(), "skus": {}}
        for dt, pid, qty, cat, subcat, brand, line in events[state["end"]:end]:
            state["volume"] += qty
            for field, value in (("cats", cat), ("subcats", subcat), ("brands", brand)):
                if value:
                    state[field][value] = state[field].get(value, 0) + 1
            if line:
                state["lines"].add(line)
            sku = state["skus"].get(pid)
            if sku is None:
                sku = state["skus"][pid] = {"order_count": 0, "volume": 0., "lower_gaps": [], "upper_gaps": [], "median_gap": 0.}
            else:
                gap = max(0., (dt - sku["last_date"]).total_seconds() / 86400.)
                lower, upper = sku["lower_gaps"], sku["upper_gaps"]
                heappush(lower, -gap) if not lower or gap <= -lower[0] else heappush(upper, gap)
                if len(lower) > len(upper) + 1:
                    heappush(upper, -heappop(lower))
                elif len(upper) > len(lower):
                    heappush(lower, -heappop(upper))
                sku["median_gap"] = -lower[0] if len(lower) > len(upper) else (-lower[0] + upper[0]) / 2
            sku["order_count"] += 1
            sku["volume"] += qty
            sku["last_date"] = dt
        state["end"] = end
        if previous is None or end >= previous["end"]:
            self._remember(self._prefix_states, cid, state, self.max_prefixes)
        snap = CustomerSnapshot(
            total_volume=state["volume"], total_orders=end,
            cat_orders=dict(state["cats"]), subcat_orders=dict(state["subcats"]), brand_orders=dict(state["brands"]),
            business_lines=set(state["lines"]),
            sku_stats={pid: {name: value for name, value in sku.items() if name not in ("lower_gaps", "upper_gaps")}
                       for pid, sku in state["skus"].items()})
        return self._remember(self._snapshots, key, snap, self.max_snapshots)

    def precompute_snapshots(self, transitions: Sequence[Dict[str, Any]]) -> None:
        """Precompute and cache snapshots for transitions to accelerate inference/training."""
        for t in transitions:
            cid = str(t["customer_id"]).strip()
            as_of = pd.Timestamp(t["as_of_date"])
            self.get_customer_snapshot(cid, as_of)

    def personal_repeat_candidates(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        eligible_products: set[str],
        limit: int,
    ) -> List[str]:
        """Use prior customer purchases and a point-in-time global popularity fallback."""
        popularity = self.popularity_candidates(as_of_date, eligible_products)
        snapshot = self.get_customer_snapshot(customer_id, as_of_date)
        stats = {sku: (row["order_count"], row["last_date"]) for sku, row in snapshot.sku_stats.items()}
        return rank_personal_repeat(stats, popularity, limit)

    def popularity_candidates(self, as_of_date, eligible_products, *, include_zero_sales=False):
        """Order candidates by paid sales before the query and the training cutoff."""
        cutoff = min(pd.Timestamp(as_of_date), self.train_cutoff)
        if cutoff not in self._popularity_rankings:
            counts = {sku: bisect_left(dates, cutoff) for sku, dates in self._sku_sale_dates.items()}
            self._popularity_rankings[cutoff] = sorted(
                (sku for sku, count in counts.items() if count),
                key=lambda sku: (-counts[sku], sku),
            )
        ranking = self._popularity_rankings[cutoff]
        self._popularity_rankings.move_to_end(cutoff)
        while len(self._popularity_rankings) > self.max_cached_dates:
            self._popularity_rankings.popitem(last=False)
        popularity = [sku for sku in ranking if sku in eligible_products]
        if include_zero_sales:
            popularity.extend(sorted(eligible_products - set(popularity)))
        return popularity

    def compute_pairwise_features(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        product_id: str,
    ) -> np.ndarray:
        """Compute the customer–SKU features in PAIRWISE_FEATURE_NAMES order."""
        cid = str(customer_id).strip()
        pid = str(product_id).strip()
        as_of_ts = pd.Timestamp(as_of_date)
        snap = self.get_customer_snapshot(cid, as_of_ts)

        feats = np.zeros(NUM_PAIRWISE_FEATURES, dtype=np.float32)

        # 1. is_repeat_sku
        is_repeat = pid in snap.sku_stats
        feats[0] = 1.0 if is_repeat else 0.0

        # Customer SKU stats
        if is_repeat:
            sku_info = snap.sku_stats[pid]
            cnt = sku_info["order_count"]
            vol = sku_info["volume"]
            last_dt = sku_info["last_date"]
            med_gap = sku_info["median_gap"]

            # 2. user_sku_order_count_norm: log1p(prior_orders) / log1p(50)
            feats[1] = float(min(1.0, max(0.0, math.log1p(cnt) / math.log1p(50.0))))

            # 3. user_sku_qty_share: sku_vol / total_vol
            if snap.total_volume > 0.0:
                feats[2] = float(min(1.0, max(0.0, vol / snap.total_volume)))
            else:
                feats[2] = 0.0

            # 4. days_since_last_sku_buy_norm: log1p(days) / log1p(365)
            delta_days = max(0.0, (as_of_ts - last_dt).total_seconds() / 86400.0)
            feats[3] = float(min(1.0, max(0.0, math.log1p(delta_days) / math.log1p(365.0))))

            # 5. sku_urgency: min(1.0, (delta_t / median_gap) / 5.0) (0.0 if < 2 orders)
            if cnt >= 2:
                safe_gap = max(1.0, med_gap)
                ratio = delta_days / safe_gap
                feats[4] = float(min(1.0, max(0.0, (ratio / 5.0))))
            else:
                feats[4] = 0.0
        else:
            # Cold SKU for this customer
            feats[1] = 0.0
            feats[2] = 0.0
            feats[3] = 1.0  # 1.0 if never bought
            feats[4] = 0.0  # 0.0 if fewer than two purchases

        # 6. user_cat_order_share
        tot_orders = max(snap.total_orders, 1)
        cat = self.get_product_category(pid, as_of_ts)
        if cat and cat in snap.cat_orders and snap.total_orders > 0:
            feats[5] = float(min(1.0, max(0.0, snap.cat_orders[cat] / tot_orders)))
        else:
            feats[5] = 0.0

        # 7. user_subcat_order_share
        subcat = self.get_product_subcategory(pid, as_of_ts)
        if subcat and subcat in snap.subcat_orders and snap.total_orders > 0:
            feats[6] = float(min(1.0, max(0.0, snap.subcat_orders[subcat] / tot_orders)))
        else:
            feats[6] = 0.0

        # 8. user_brand_order_share
        brand = self.get_product_brand(pid, as_of_ts)
        if brand and brand in snap.brand_orders and snap.total_orders > 0:
            feats[7] = float(min(1.0, max(0.0, snap.brand_orders[brand] / tot_orders)))
        else:
            feats[7] = 0.0

        # 9–10. Global SKU popularity and industry affinity.
        feats[8], feats[9] = self.sku_population_statistics(cid, as_of_ts, pid)

        # Attribute familiarity uses all customer events before the basket.
        line = self.get_product_attribute(pid, "business_line", as_of_ts)
        feats[10] = cat in snap.cat_orders
        feats[11] = brand in snap.brand_orders
        feats[12] = line in snap.business_lines

        # Clean NaNs and enforce [0.0, 1.0] bounds
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(feats, 0.0, 1.0, out=feats)
        return feats

    def compute_slate_pairwise_features(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        product_ids: Sequence[str],
    ) -> torch.Tensor:
        """Fast batch extraction of pairwise features for a candidate slate (shape C x 13).

        Executes in ~0.03 milliseconds for 50 candidates.
        """
        c = len(product_ids)
        mat = np.zeros((c, NUM_PAIRWISE_FEATURES), dtype=np.float32)
        for i in range(c):
            mat[i] = self.compute_pairwise_features(
                customer_id=customer_id,
                as_of_date=as_of_date,
                product_id=product_ids[i],
            )
        return torch.from_numpy(mat)


def construct_candidate_slate(
    stage1_logits: torch.Tensor,
    product_ids: List[str],
    positive_product_ids: Optional[Sequence[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    top_c: int = DEFAULT_TOP_C,
    *,
    heuristic_product_ids: Sequence[str] = (),
    discovery_product_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Reserve heuristic picks, then fill the slate in two-tower score order."""
    p_total = min(len(product_ids), stage1_logits.size(-1))
    k_retrieve = min(top_k, p_total)
    c_slate = min(top_c, k_retrieve)

    rank_depth = p_total if heuristic_product_ids or discovery_product_ids else k_retrieve
    topk_vals, topk_idx = torch.topk(stage1_logits[:p_total], k=rank_depth, dim=-1)
    indices = topk_idx.cpu().tolist()
    scores = topk_vals.cpu().tolist()
    pids = [product_ids[i] for i in indices]
    ranks = list(range(c_slate))
    if discovery_product_ids:
        rank_by_sku = {sku: rank for rank, sku in enumerate(pids)}
        preferred = [rank_by_sku[sku] for sku in (*heuristic_product_ids, *discovery_product_ids) if sku in rank_by_sku]
        ranks = list(dict.fromkeys([*preferred, *range(c_slate)]))[:c_slate]
    elif heuristic_product_ids:
        rank_by_sku = {sku: rank for rank, sku in enumerate(pids)}
        reserved = dict.fromkeys(rank_by_sku[sku] for sku in heuristic_product_ids if sku in rank_by_sku)
        ranks = (list(reserved) + [rank for rank in range(c_slate) if rank not in reserved])[:c_slate]
    candidates = [pids[rank] for rank in ranks]

    labels = None
    if positive_product_ids is not None:
        positives = {str(pid).strip() for pid in positive_product_ids}
        labels = [float(pid in positives) for pid in candidates]

    return {
        "candidate_indices": [indices[rank] for rank in ranks],
        "candidate_pids": candidates,
        "candidate_scores": [scores[rank] for rank in ranks],
        "candidate_ranks": ranks,
        "labels": labels,
        "top100_indices": indices[:k_retrieve],
        "top100_scores": scores[:k_retrieve],
        "top100_pids": pids[:k_retrieve],
    }


def load_retriever_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[TwoTowerModel, Dict[str, Any]]:
    """Load trained Stage 1 TwoTowerModel and associated checkpoint metadata.

    Args:
        checkpoint_path: Path to best_retriever.pt checkpoint.
        device: Target compute device (defaults to CPU).

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)

    state = ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    num_products = len(ckpt.get("product_ids", []))
    if num_products == 0:
        num_products = config.get("num_catalog_products", 927)

    # Determine embedding layer dimensions dynamically from checkpoint state dict
    city_emb_size = state["customer_tower.city_embedding.weight"].shape[0]
    reg_emb_size = state["customer_tower.region_embedding.weight"].shape[0]
    ind_emb_size = state["customer_tower.industry_embedding.weight"].shape[0]
    cat_emb_size = (
        state["customer_tower.cat_embedding.weight"].shape[0]
        if "customer_tower.cat_embedding.weight" in state
        else 50
    )
    product_sizes = config.get("product_embedding_sizes") or {
        "brand": state["product_tower.brand_embedding.weight"].shape[0],
        "category": state["product_tower.cat_embedding.weight"].shape[0],
        "business_line": state["product_tower.line_embedding.weight"].shape[0],
    }
    prod_subcat = state.get("product_tower.subcat_embedding.weight")
    containers = state.get("product_tower.container_embedding.weight")
    sku_embeddings = state.get("customer_tower.product_embedding.weight")
    profile_features = config.get("profile_features", LEGACY_PROFILE_FEATURES)
    history_feature_indices = config.get("history_feature_indices")
    catalog_history_encoder = config.get("catalog_history_encoder")
    tower_hidden_dim = config.get("tower_hidden_dim", 128)
    legacy_projection = state.get("customer_tower.event_cont_proj.weight")
    if legacy_projection is None:
        legacy_projection = state.get("customer_tower.event_proj.0.weight")

    customer_tower = CustomerTower(
        text_dim=config.get("text_dim", TEXT_DIM),
        output_dim=config.get("output_dim", OUTPUT_DIM),
        num_cities=city_emb_size,
        num_regions=reg_emb_size,
        num_industries=ind_emb_size,
        num_categories=cat_emb_size,
        num_business_lines=product_sizes["business_line"],
        seq_feature_dim=legacy_projection.shape[1] if legacy_projection is not None else 2,
        hidden_dim=tower_hidden_dim,
        dim_feedforward=2 * tower_hidden_dim,
        dropout=config.get("dropout", 0.1),
        num_layers=config.get("history_layers", 2),
        extra_mlp_layer=config.get("extra_tower_mlp_layer", False),
        history_feature_indices=tuple(history_feature_indices) if history_feature_indices is not None else None,
        catalog_history_encoder=catalog_history_encoder,
        max_seq_len=config.get("max_seq_len", DEFAULT_SEQUENCE_LENGTH),
        history_gap_reference_days=config.get("history_gap_reference_days"),
        num_products=sku_embeddings.shape[0] if sku_embeddings is not None else None,
        num_tabular_features=len(profile_features),
    )
    product_tower = ProductTower(
        text_dim=config.get("text_dim", TEXT_DIM),
        output_dim=config.get("output_dim", OUTPUT_DIM),
        num_brands=product_sizes["brand"],
        num_categories=product_sizes["category"],
        num_subcategories=prod_subcat.shape[0] if prod_subcat is not None else 0,
        num_business_lines=product_sizes["business_line"],
        num_demand_features=config.get("num_product_features", 6),
        num_containers=containers.size(0) if containers is not None else 0,
        one_hot_sizes=config.get("product_one_hot_sizes"),
        hidden_dim=tower_hidden_dim,
        dropout=config.get("dropout", 0.1),
        extra_mlp_layer=config.get("extra_tower_mlp_layer", False),
    )
    model = TwoTowerModel(
        customer_tower=customer_tower,
        product_tower=product_tower,
        num_catalog_products=num_products,
        temperature=config.get("temperature", DEFAULT_TEMPERATURE),
        num_familiarity_features=len(config.get("familiarity_fields", ())),
    )

    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    return model, ckpt


class CandidateRetriever:
    """Stage 1 Two-Tower candidate retriever and pairwise feature generator.

    Generates top-100 candidates for Recall evaluation and top-50 candidate slates
    with pairwise features for Stage 2 ranker training and evaluation.
    """

    def __init__(
        self,
        model: Optional[TwoTowerModel] = None,
        product_inputs: Optional[Dict[str, torch.Tensor]] = None,
        eligible_product_ids: Optional[List[str]] = None,
        feature_store: Optional[PointInTimeFeatureStore] = None,
        device: Optional[Union[str, torch.device]] = None,
        *,
        profile_features: Sequence[str] = LEGACY_PROFILE_FEATURES,
        profile_normalizer: Optional[PointInTimeNormalizer] = None,
        quantity_reference: Optional[float] = None,
        customer_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        company_cache_name: str = "bge_m3_company.parquet",
        familiarity: Optional[FamiliarityStore] = None,
        rolling_pool: Optional[RollingCandidatePool] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        self.product_inputs = product_inputs
        self.eligible_product_ids = (
            [str(p).strip() for p in eligible_product_ids] if eligible_product_ids else []
        )
        self.feature_store = feature_store
        self.profile_features = tuple(profile_features)
        self.profile_normalizer = profile_normalizer
        self.quantity_reference = quantity_reference
        self.customer_vocabs = customer_vocabs or {}
        self.company_cache_name = company_cache_name
        self.familiarity = familiarity
        self.rolling_pool = rolling_pool

        self._cached_product_vectors: Optional[torch.Tensor] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        purchases_df: pd.DataFrame,
        catalogue_df: Optional[pd.DataFrame] = None,
        companies_df: Optional[pd.DataFrame] = None,
        product_inputs: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[Union[str, torch.device]] = None,
        *,
        candidate_product_ids: Optional[Sequence[str]] = None,
        rolling_pool: Optional[RollingCandidatePool] = None,
        metadata_policy: Optional[str] = None,
    ) -> "CandidateRetriever":
        """Load the checkpoint, optionally appending cold candidates in input order."""
        model, ckpt = load_retriever_from_checkpoint(checkpoint_path, device=device)
        trained_pids = list(ckpt.get("product_ids", []))
        eligible_pids = trained_pids
        if candidate_product_ids is not None:
            eligible_pids = list(candidate_product_ids)
            if (
                len(trained_pids) != model.product_bias.numel()
                or eligible_pids[:len(trained_pids)] != trained_pids
                or len(eligible_pids) != len(set(eligible_pids))
            ):
                raise ValueError("Expanded candidates must start with the checkpoint SKU order and contain unique IDs.")
            if product_inputs is not None and any(tensor.size(0) != len(eligible_pids) for tensor in product_inputs.values()):
                raise ValueError("Product inputs must cover the expanded candidate order.")
            extra = len(eligible_pids) - len(trained_pids)
            if extra:
                bias = model.product_bias.detach()
                model.product_bias = torch.nn.Parameter(torch.cat((bias, bias.new_zeros(extra))), requires_grad=False)
        config = ckpt.get("config", {})
        if product_inputs is not None and config.get("demand_policy") == "disabled_v1":
            product_inputs = dict(product_inputs)
            demand = product_inputs["demand_features"].clone()
            demand[:, :6 + int(config.get("include_sku_median_quantity", False))] = 0
            product_inputs["demand_features"] = demand
        if config.get("discovery_only"):
            purchases_df = purchases_df.loc[actual_purchase_mask(purchases_df)]
            if rolling_pool is None:
                if catalogue_df is None:
                    raise ValueError("Discovery retrieval requires the merchandise catalogue.")
                rolling_pool = RollingCandidatePool(catalogue_df, purchases_df)
        normalizer_state = config.get("profile_normalizer")
        vocabs = ckpt.get("vocabs", {})
        familiarity = None
        if model.familiarity_weights is not None:
            if catalogue_df is None:
                raise ValueError("Familiarity-aware retrieval requires catalogue attributes.")
            familiarity = FamiliarityStore(purchases_df, catalogue_df, eligible_pids, config["familiarity_fields"])

        feature_store = PointInTimeFeatureStore(
            purchases_df=purchases_df,
            catalogue_df=catalogue_df,
            companies_df=companies_df,
            metadata_policy=metadata_policy or config.get("pairwise_metadata_policy", "snapshot_unverified"),
        )

        return cls(
            model=model,
            product_inputs=product_inputs,
            eligible_product_ids=eligible_pids,
            feature_store=feature_store,
            device=device,
            profile_features=config.get("profile_features", LEGACY_PROFILE_FEATURES),
            profile_normalizer=(
                PointInTimeNormalizer.from_state(normalizer_state) if normalizer_state is not None else None
            ),
            quantity_reference=config.get("quantity_reference"),
            company_cache_name=config.get("company_cache_name", "bge_m3_company.parquet"),
            familiarity=familiarity,
            rolling_pool=rolling_pool,
            customer_vocabs={
                key: vocabs[key] for key in ("city_vocab", "region_vocab", "industry_vocab") if key in vocabs
            },
        )

    def get_product_vectors(self) -> torch.Tensor:
        """Compute or retrieve cached L2-normalized product catalog vectors v_p (P, 128)."""
        if self._cached_product_vectors is not None:
            return self._cached_product_vectors

        if self.model is None or self.product_inputs is None:
            raise RuntimeError("Cannot compute product vectors without model and product_inputs")

        with torch.no_grad():
            p_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.product_inputs.items()
            }
            v_p = self.model.product_tower(
                p_inputs["text_embedding"],
                p_inputs["brand_id"],
                p_inputs["category_id"],
                p_inputs.get("subcategory_id"),
                p_inputs["business_line_id"],
                p_inputs["demand_features"],
                container_id=p_inputs.get("container_id"),
            )
            self._cached_product_vectors = v_p
            return v_p

    def compute_logits(
        self,
        customer_vectors: torch.Tensor,
        product_vectors: Optional[torch.Tensor] = None,
        *,
        customer_ids: Optional[Sequence[str]] = None,
        as_of_dates: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
    ) -> torch.Tensor:
        """Score products, using query dates and customer IDs for contextual features."""
        u_c = customer_vectors.to(self.device)
        v_p = product_vectors.to(self.device) if product_vectors is not None else self.get_product_vectors()

        if self.model is not None:
            prod_idx = torch.arange(v_p.size(0), device=self.device)
            familiarity_features = None
            if self.model.familiarity_weights is not None:
                if self.familiarity is None or customer_ids is None or as_of_dates is None:
                    raise ValueError("Familiarity-aware retrieval requires customer IDs and scoring dates.")
                familiarity_features = self.familiarity.batch_features(customer_ids, as_of_dates)
            return self.model.compute_logits(
                u_c, v_p, product_indices=prod_idx,
                familiarity_features=familiarity_features,
            )

        # Standalone vector dot-product fallback
        return torch.matmul(u_c, v_p.t()) / DEFAULT_TEMPERATURE

    def discover_from_vectors(
        self,
        customer_vectors: torch.Tensor,
        customer_ids: Sequence[str],
        as_of_dates: Sequence[Union[str, pd.Timestamp]],
        product_vectors: Optional[torch.Tensor] = None,
        *,
        top_k: int = 25,
    ) -> List[List[str]]:
        """Return up to top_k unseen SKUs per query from the rolling candidate pool."""
        if self.rolling_pool is None or self.feature_store is None:
            raise ValueError("Discovery requires a rolling catalogue and point-in-time purchase history.")
        if top_k < 1 or not len(customer_vectors) == len(customer_ids) == len(as_of_dates):
            raise ValueError("Discovery requires a positive top_k and one customer and date per vector.")
        scores = self.compute_logits(
            customer_vectors, product_vectors=product_vectors, customer_ids=customer_ids, as_of_dates=as_of_dates,
        )
        dated = {}
        eligible, seen = [], []
        for cid, date in zip(customer_ids, as_of_dates):
            date = pd.Timestamp(date).normalize()
            if date not in dated:
                dated[date] = self.rolling_pool.eligible_on(date)
            eligible.append(dated[date])
            seen.append(set(self.feature_store.get_customer_snapshot(cid, date).sku_stats))
        return rank_unseen_candidates(scores, self.eligible_product_ids, eligible, seen, top_k)

    def generate_candidate_batch(
        self,
        logits: torch.Tensor,
        customer_ids: Sequence[str],
        as_of_dates: Sequence[Union[str, pd.Timestamp]],
        positive_product_ids: Optional[Sequence[Optional[Sequence[str]]]] = None,
        top_k: int = DEFAULT_TOP_K,
        top_c: int = DEFAULT_TOP_C,
        *,
        heuristic_fraction: float = 0.0,
    ) -> CandidateBatch:
        """Construct CandidateBatch with 13 pairwise features for a batch of transitions.

        Args:
            logits: FloatTensor of shape (B, P) with Stage 1 retrieval scores.
            customer_ids: List of B customer ID strings.
            as_of_dates: List of B transition timestamps.
            positive_product_ids: Optional list of positive item lists per transition.
            top_k: Retrieval depth for candidate evaluation (default 100).
            top_c: Candidate slate size for Stage 2 ranker (default 50).
            heuristic_fraction: Fraction reserved for the repeat-plus-popularity heuristic.

        Returns:
            CandidateBatch containing tensors with zero NaNs and values in [0.0, 1.0].
        """
        b_size = logits.size(0)
        p_total = logits.size(1)
        k_retrieve = min(top_k, p_total)
        c_slate = min(top_c, k_retrieve)
        if not 0.0 <= heuristic_fraction <= 1.0:
            raise ValueError("heuristic_fraction must be between 0 and 1.")
        heuristic_count = int(c_slate * heuristic_fraction)
        if heuristic_count and self.feature_store is None:
            raise ValueError("Heuristic candidates require a point-in-time feature store.")

        if not self.eligible_product_ids:
            current_pids = [str(i) for i in range(p_total)]
        elif len(self.eligible_product_ids) >= p_total:
            current_pids = self.eligible_product_ids[:p_total]
        else:
            current_pids = list(self.eligible_product_ids) + [
                f"PROD_{i}" for i in range(len(self.eligible_product_ids), p_total)
            ]
        eligible_products = set(current_pids)
        rolling_pools = None
        discovery = None
        row_indices_by_date = {}
        if self.rolling_pool is not None and b_size and c_slate:
            by_date = {date: self.rolling_pool.eligible_on(date) & eligible_products for date in set(as_of_dates)}
            index = {sku: i for i, sku in enumerate(current_pids)}
            row_indices_by_date = {date: sorted(index[sku] for sku in pool) for date, pool in by_date.items()}
            rolling_pools = [by_date[date] for date in as_of_dates]
            seen = [set(self.feature_store.get_customer_snapshot(cid, date).sku_stats)
                    for cid, date in zip(customer_ids, as_of_dates)]
            discovery = rank_unseen_candidates(logits, current_pids, rolling_pools, seen, c_slate)
            k_retrieve = min(k_retrieve, min(map(len, rolling_pools)))
            c_slate = min(c_slate, k_retrieve)
            heuristic_count = int(c_slate * heuristic_fraction)
        if heuristic_count:
            # One transfer for the catalog-wide rankings and scores in this batch.
            logits = logits.cpu()

        all_candidate_indices: List[List[int]] = []
        all_candidate_pids: List[List[str]] = []
        all_candidate_scores: List[List[float]] = []
        all_pairwise_features: List[torch.Tensor] = []
        all_labels: Optional[List[List[float]]] = [] if positive_product_ids is not None else None
        all_top100_indices: List[List[int]] = []
        all_top100_scores: List[List[float]] = []
        all_top100_pids: List[List[str]] = []

        for b in range(b_size):
            cid = str(customer_ids[b]).strip()
            as_of = as_of_dates[b]
            pos_pids = positive_product_ids[b] if positive_product_ids is not None else None
            row_logits = logits[b]

            row_pids = current_pids
            row_indices = None
            if rolling_pools is not None:
                pool = rolling_pools[b]
                row_indices = row_indices_by_date[as_of]
                row_pids = [current_pids[i] for i in row_indices]
                row_logits = row_logits[row_indices]
                snapshot = self.feature_store.get_customer_snapshot(cid, as_of)
                stats = {sku: (data["order_count"], data["last_date"]) for sku, data in snapshot.sku_stats.items()}
                popularity = self.feature_store.popularity_candidates(as_of, pool, include_zero_sales=True)
                heuristic_products = rank_personal_repeat(stats, popularity, heuristic_count)
            else:
                heuristic_products = (
                    self.feature_store.personal_repeat_candidates(cid, as_of, eligible_products, heuristic_count)
                    if heuristic_count else ()
                )
            slate_dict = construct_candidate_slate(
                stage1_logits=row_logits,
                product_ids=row_pids,
                positive_product_ids=pos_pids,
                top_k=k_retrieve,
                top_c=c_slate,
                heuristic_product_ids=heuristic_products,
                discovery_product_ids=discovery[b][:c_slate] if discovery is not None else (),
            )
            if row_indices is not None:
                for key in ("candidate_indices", "top100_indices"):
                    slate_dict[key] = [row_indices[i] for i in slate_dict[key]]

            all_candidate_indices.append(slate_dict["candidate_indices"])
            all_candidate_pids.append(slate_dict["candidate_pids"])
            all_candidate_scores.append(slate_dict["candidate_scores"])
            if all_labels is not None and slate_dict["labels"] is not None:
                all_labels.append(slate_dict["labels"])

            all_top100_indices.append(slate_dict["top100_indices"])
            all_top100_scores.append(slate_dict["top100_scores"])
            all_top100_pids.append(slate_dict["top100_pids"])

            # 2. Pairwise feature extraction
            if self.feature_store is not None:
                feat_matrix = self.feature_store.compute_slate_pairwise_features(
                    customer_id=cid,
                    as_of_date=as_of,
                    product_ids=slate_dict["candidate_pids"],
                )
            else:
                feat_matrix = torch.zeros((len(slate_dict["candidate_pids"]), NUM_PAIRWISE_FEATURES))

            all_pairwise_features.append(feat_matrix)

        # Assemble into CandidateBatch
        candidate_indices_t = torch.tensor(all_candidate_indices, dtype=torch.long)
        candidate_scores_t = torch.tensor(all_candidate_scores, dtype=torch.float32)
        pairwise_features_t = (
            torch.stack(all_pairwise_features, dim=0)
            if all_pairwise_features
            else torch.zeros((0, c_slate, NUM_PAIRWISE_FEATURES), dtype=torch.float32)
        )
        labels_t = (
            torch.tensor(all_labels, dtype=torch.float32)
            if all_labels is not None and len(all_labels) == b_size
            else None
        )
        top100_indices_t = torch.tensor(all_top100_indices, dtype=torch.long)
        top100_scores_t = torch.tensor(all_top100_scores, dtype=torch.float32)

        # Enforce zero NaNs
        candidate_scores_t = torch.nan_to_num(candidate_scores_t, nan=0.0)
        pairwise_features_t = torch.nan_to_num(pairwise_features_t, nan=0.0).clamp(0.0, 1.0)
        top100_scores_t = torch.nan_to_num(top100_scores_t, nan=0.0)

        return CandidateBatch(
            customer_ids=list(customer_ids),
            candidate_pids=all_candidate_pids,
            candidate_indices=candidate_indices_t,
            candidate_scores=candidate_scores_t,
            pairwise_features=pairwise_features_t,
            labels=labels_t,
            top100_indices=top100_indices_t,
            top100_scores=top100_scores_t,
            top100_pids=all_top100_pids,
        )

    def retrieve_from_vectors(
        self,
        customer_vectors: torch.Tensor,
        customer_ids: Sequence[str],
        as_of_dates: Sequence[Union[str, pd.Timestamp]],
        product_vectors: Optional[torch.Tensor] = None,
        positive_product_ids: Optional[Sequence[Optional[Sequence[str]]]] = None,
        top_k: int = DEFAULT_TOP_K,
        top_c: int = DEFAULT_TOP_C,
    ) -> CandidateBatch:
        """Retrieve candidates and construct CandidateBatch given precomputed customer vectors."""
        logits = self.compute_logits(
            customer_vectors, product_vectors=product_vectors, customer_ids=customer_ids, as_of_dates=as_of_dates,
        )
        return self.generate_candidate_batch(
            logits=logits,
            customer_ids=customer_ids,
            as_of_dates=as_of_dates,
            positive_product_ids=positive_product_ids,
            top_k=top_k,
            top_c=top_c,
        )
