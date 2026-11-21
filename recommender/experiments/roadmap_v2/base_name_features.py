"""Strictly causal (< as_of_date) rolling base_name feature builder and sealed cache."""

from bisect import bisect_left
from collections import OrderedDict
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.feature_builder import RelationshipTable, log_count
from recommender.experiments.dynamic_selector.feature_schema import (
    NAMES as SOURCE_41_NAMES,
    quantize_features,
)
from recommender.experiments.rolling_candidate_v2.data import (
    compute_rolling_source_digest,
    prepare_or_load_rolling_feature_bundle,
)
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START

from .config import (
    ARM_BASE_NAME_ONLY,
    ARM_SKU_ONLY,
    ARM_SKU_PLUS_BASE_NAME,
    BASE_NAME_FEATURE_NAMES,
    CACHE_ROOT,
    EXACT_SKU_SOURCE_INDICES,
    NUM_BASE_NAME_FEATURES,
    VALID_BASE_NAME_ARMS,
)

BASE_NAME_CACHE_VERSION = 1
BASE_NAME_CACHE_NAMESPACE = "roadmap_v2_base_name_19d_v1"

_CATEGORY_LIFT_COL = SOURCE_41_NAMES.index("category_last_lift")
_FAMILY_LIFT_COL = SOURCE_41_NAMES.index("family_last_lift")


def base_name_feature_contract() -> Dict[str, Any]:
    """Explicit deterministic schema contract for the 19 strictly-causal base_name features."""
    return {
        "version": BASE_NAME_CACHE_VERSION,
        "namespace": BASE_NAME_CACHE_NAMESPACE,
        "names": list(BASE_NAME_FEATURE_NAMES),
        "dtype_on_disk": "float16",
        "compute_dtype": "float32",
        "statistics_policy": "rolling_observed_v2_strictly_prior",
        "same_day_multipack_aggregation": "sum_quantity_one_order_event_per_customer_date_base_name",
        "same_basket_complement_self_edges": "excluded_distinct_base_names_only",
        "count_scale": "log1p(x)/log1p(1000), clipped [0,1]",
        "user_order_count_scale": "log1p(x)/log1p(50), clipped [0,1]",
        "recency_scale": "log1p(days)/log1p(365), clipped [0,1] (1.0 if never bought)",
        "urgency_scale": "(days/max(1,median_gap))/5, clipped [0,1] (0.0 if <2 orders)",
        "lift_scale": "log_lift/5, clipped [-1,1]",
        "metadata_policy": "snapshot_unverified",
    }


def _clean_base_name(val: Any) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.upper() == "NULL" else s


class BaseNameRollingFeatureBuilder:
    """Computes the 19 strictly-causal (< as_of_date) base_name-level features over a product universe.

    Guarantees:
    - Refuses any purchases or query dates >= TEST_START (zero held-out leakage).
    - Uses strictly prior events (purchase_date < as_of_date) across both train and validation.
    - Same-day multi-pack purchases of the same base_name by a customer are aggregated into a single
      base_name order event (summed quantity, 1 order count) so median inter-purchase gaps are never
      distorted by 0-day multi-pack lines.
    - Same-basket complements only link distinct base_names bought on the same day.
    """

    def __init__(
        self,
        purchases: pd.DataFrame,
        catalogue: pd.DataFrame,
        companies: pd.DataFrame,
        product_ids: Sequence[str],
        *,
        allow_non_monotonic: bool = False,
    ):
        if purchases.empty:
            dates = pd.Series([], dtype="datetime64[ns]")
        else:
            dates = pd.to_datetime(purchases["purchase_date"], format="mixed")
            if bool(dates.isna().any()) or not bool(dates.lt(TEST_START).all()):
                raise ValueError(f"Development purchases only (< {TEST_START.date()}).")

        self.allow_non_monotonic = bool(allow_non_monotonic)
        self.product_ids = [str(p).strip() for p in product_ids]
        self.num_products = len(self.product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # Map all catalogue product_ids -> clean base_name and product_category
        cat_df = catalogue.copy()
        cat_df["product_id"] = cat_df["product_id"].astype(str).str.strip()
        if "base_name" not in cat_df.columns:
            raise ValueError("catalogue DataFrame must contain 'base_name' column.")
        cat_df["base_name_clean"] = cat_df["base_name"].map(_clean_base_name)
        cat_df["cat_clean"] = (
            cat_df["product_category"].fillna("").astype(str).str.strip()
            if "product_category" in cat_df.columns
            else ""
        )
        cat_dedup = cat_df.drop_duplicates("product_id").set_index("product_id")
        self.all_pid_to_bn: Dict[str, str] = cat_dedup["base_name_clean"].to_dict()

        # Canonical base_name vocabulary for the product universe (id 0 = missing/unknown)
        self.prod_base_names = [self.all_pid_to_bn.get(pid, "") for pid in self.product_ids]
        unique_bns = sorted(set(self.prod_base_names) - {""})
        self.bn_to_id: Dict[str, int] = {bn: i + 1 for i, bn in enumerate(unique_bns)}
        self.unique_bns = unique_bns  # length B (e.g., 585)
        self.num_bns = len(unique_bns)
        # prod_bn_id[p] in [0..B], where 0 means empty base_name
        self.prod_bn_id = np.array([self.bn_to_id.get(bn, 0) for bn in self.prod_base_names], dtype=np.int64)

        # Category mapping per unique base_name (for base_name_category_relative_share)
        prod_cats = [
            str(cat_dedup.at[pid, "cat_clean"]) if pid in cat_dedup.index else ""
            for pid in self.product_ids
        ]
        bn_to_cat: Dict[str, str] = {}
        for bn, cat in zip(self.prod_base_names, prod_cats):
            if bn and bn not in bn_to_cat and cat:
                bn_to_cat[bn] = cat
        unique_cats = sorted(set(bn_to_cat.values()) | {""})
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        self._bn_cat_indices = np.array(
            [cat_to_idx[bn_to_cat.get(bn, "")] for bn in self.unique_bns], dtype=np.int64
        )
        self._num_cats = len(unique_cats)

        # Customer -> industry map
        comp_ind: Dict[str, str] = {}
        if companies is not None and not companies.empty:
            for _, row in companies.iterrows():
                cid = str(row["customer_id"]).strip()
                ind = str(row.get("industry_branch", "")).strip() if pd.notna(row.get("industry_branch")) else ""
                comp_ind[cid] = ind
        self.customer_industry = comp_ind

        # Normalize purchases & compute base_name per row
        df = purchases.copy()
        if not df.empty:
            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["product_id"] = df["product_id"].astype(str).str.strip()
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed")
            df["quantity"] = (
                pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0.0)
                if "quantity" in df.columns
                else 1.0
            )
            df["base_name_clean"] = df["product_id"].map(self.all_pid_to_bn).fillna("")
            df = df.sort_values(["purchase_date", "customer_id", "product_id"], kind="stable")

        # 1. Customer total volume timeline (all paid purchases, identical denominator to user_sku_qty_share)
        self._cust_vol_dates: Dict[str, np.ndarray] = {}
        self._cust_vol_cumsum: Dict[str, np.ndarray] = {}
        if not df.empty:
            for cid, grp in df.groupby("customer_id", sort=False):
                by_day = grp.groupby("purchase_date", sort=True)["quantity"].sum()
                self._cust_vol_dates[cid] = by_day.index.to_numpy(dtype="datetime64[ns]")
                self._cust_vol_cumsum[cid] = by_day.to_numpy(dtype=np.float64).cumsum()

        # 2. Aggregated (customer_id, purchase_date, base_name) order events (non-empty base_name only)
        if not df.empty:
            bn_df = df.loc[df["base_name_clean"] != ""].groupby(
                ["customer_id", "purchase_date", "base_name_clean"], as_index=False, sort=False
            )["quantity"].sum()
            bn_df = bn_df.sort_values(["purchase_date", "customer_id", "base_name_clean"], kind="stable")
        else:
            bn_df = pd.DataFrame(columns=["customer_id", "purchase_date", "base_name_clean", "quantity"])

        # Per-customer base_name event stream (chronological) for incremental prefix snapshots
        self._cust_bn_events: Dict[str, List[Tuple[pd.Timestamp, int, float]]] = {}
        self._cust_bn_dates: Dict[str, List[pd.Timestamp]] = {}
        if not bn_df.empty:
            for cid, grp in bn_df.groupby("customer_id", sort=False):
                g = grp.sort_values(["purchase_date", "base_name_clean"], kind="stable")
                evs: List[Tuple[pd.Timestamp, int, float]] = []
                dts: List[pd.Timestamp] = []
                for dt, bn, qty in zip(g["purchase_date"], g["base_name_clean"], g["quantity"]):
                    bid = self.bn_to_id.get(bn, 0)
                    if bid > 0:
                        ts = pd.Timestamp(dt)
                        evs.append((ts, bid, float(qty)))
                        dts.append(ts)
                self._cust_bn_events[cid] = evs
                self._cust_bn_dates[cid] = dts

        self._prefix_states: Dict[str, Dict[str, Any]] = {}

        # 3. Global & industry rolling sale timelines per base_name (order-event granularity)
        self._popularity_scale = math.log1p(1000.0)
        self._bn_sale_dates: Dict[str, np.ndarray] = {}
        self._bn_sale_cids: Dict[str, np.ndarray] = {}
        self._industry_event_dates: Dict[str, np.ndarray] = {}
        self._industry_bn_dates: Dict[Tuple[str, str], np.ndarray] = {}

        if not bn_df.empty:
            for bn, grp in bn_df.groupby("base_name_clean", sort=False):
                if bn in self.bn_to_id:
                    self._bn_sale_dates[bn] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
                    self._bn_sale_cids[bn] = grp["customer_id"].to_numpy()
            bn_ind = bn_df.assign(industry=bn_df["customer_id"].map(self.customer_industry).fillna(""))
            known_ind = bn_ind.loc[bn_ind["industry"] != ""]
            for ind, grp in known_ind.groupby("industry", sort=False):
                self._industry_event_dates[ind] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
            for (ind, bn), grp in known_ind.groupby(["industry", "base_name_clean"], sort=False):
                if bn in self.bn_to_id:
                    self._industry_bn_dates[(ind, bn)] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")

        # 4. Cross-company base_name sequential transition & same-basket complement tables
        # RelationshipTable expects observations containing indices into `mapping` (length = num_products).
        # Here we run RelationshipTable directly in canonical base_name space: mapping = np.arange(B + 1)
        # where index 0 is unused and indices 1..B correspond to base_name ids 1..B!
        bn_identity_map = np.arange(self.num_bns + 1, dtype=np.int64)
        self._cust_baskets: Dict[str, List[Tuple[pd.Timestamp, List[int]]]] = {}
        self._cust_basket_dates: Dict[str, List[pd.Timestamp]] = {}
        transitions: List[Tuple[pd.Timestamp, List[int], List[int]]] = []
        complements: List[Tuple[pd.Timestamp, List[int], List[int]]] = []

        if not df.empty:
            for cid, grp in df.groupby("customer_id", sort=False):
                baskets: List[Tuple[pd.Timestamp, List[int]]] = []
                for dt, rows in grp.groupby("purchase_date", sort=True):
                    bids = sorted({self.bn_to_id[bn] for bn in rows["base_name_clean"] if bn in self.bn_to_id})
                    ts = pd.Timestamp(dt)
                    baskets.append((ts, bids))
                    if len(bids) >= 2:
                        for b in bids:
                            others = [o for o in bids if o != b]
                            complements.append((ts, [b], others))
                self._cust_baskets[cid] = baskets
                self._cust_basket_dates[cid] = [d for d, _ in baskets]
                for (_, prev_bids), (dt_next, next_bids) in zip(baskets, baskets[1:]):
                    if prev_bids and next_bids:
                        transitions.append((dt_next, prev_bids, next_bids))

        transitions.sort(key=lambda x: x[0])
        complements.sort(key=lambda x: x[0])
        self._trans_table = RelationshipTable(bn_identity_map, transitions)
        self._comp_table = RelationshipTable(bn_identity_map, complements)

        self._global_cache: OrderedDict[pd.Timestamp, np.ndarray] = OrderedDict()
        self._industry_cache: OrderedDict[Tuple[str, pd.Timestamp], np.ndarray] = OrderedDict()
        self._last_date: Optional[pd.Timestamp] = None

    def _customer_total_volume(self, cid: str, date_64: np.datetime64) -> float:
        dates = self._cust_vol_dates.get(cid)
        if dates is None:
            return 0.0
        idx = int(dates.searchsorted(date_64, side="left"))
        if idx <= 0:
            return 0.0
        return float(self._cust_vol_cumsum[cid][idx - 1])

    def _customer_bn_snapshot(self, cid: str, as_of: pd.Timestamp) -> Dict[int, Dict[str, Any]]:
        events = self._cust_bn_events.get(cid, ())
        if not events:
            return {}
        end = bisect_left(self._cust_bn_dates[cid], as_of)
        prev = self._prefix_states.get(cid)
        if prev is not None and prev["end"] <= end:
            state = prev
        else:
            state = {"end": 0, "bns": {}}
        for dt, bid, qty in events[state["end"] : end]:
            info = state["bns"].get(bid)
            if info is None:
                info = state["bns"][bid] = {
                    "order_count": 0,
                    "volume": 0.0,
                    "lower_gaps": [],
                    "upper_gaps": [],
                    "median_gap": 0.0,
                }
            else:
                gap = max(0.0, (dt - info["last_date"]).total_seconds() / 86400.0)
                lower, upper = info["lower_gaps"], info["upper_gaps"]
                if not lower or gap <= -lower[0]:
                    heappush(lower, -gap)
                else:
                    heappush(upper, gap)
                if len(lower) > len(upper) + 1:
                    heappush(upper, -heappop(lower))
                elif len(upper) > len(lower):
                    heappush(lower, -heappop(upper))
                info["median_gap"] = -lower[0] if len(lower) > len(upper) else (-lower[0] + upper[0]) / 2.0
            info["order_count"] += 1
            info["volume"] += qty
            info["last_date"] = dt
        state["end"] = end
        if prev is None or end >= prev["end"]:
            self._prefix_states[cid] = state
        return state["bns"]

    def _global_bn_features(self, date_ts: pd.Timestamp) -> np.ndarray:
        """Compute (B + 1, 8) global base_name stats: [global_pop, n7, n30, n90, buyers30, growth30, cat_share, sale_age]."""
        if date_ts in self._global_cache:
            self._global_cache.move_to_end(date_ts)
            return self._global_cache[date_ts]

        out = np.zeros((self.num_bns + 1, 8), dtype=np.float32)
        out[:, 7] = 1.0  # global_sale_age default = 1.0 if never sold < date_ts
        counts_30 = np.zeros(self.num_bns, dtype=np.float64)

        cutoff_64 = date_ts.to_datetime64()
        start_64s = [(date_ts - pd.Timedelta(days=d)).to_datetime64() for d in (7, 30, 60, 90)]

        for idx0, bn in enumerate(self.unique_bns):
            bid = idx0 + 1
            dates = self._bn_sale_dates.get(bn)
            if dates is None:
                continue
            stop = int(dates.searchsorted(cutoff_64, side="left"))
            if stop == 0:
                continue
            s7, s30, s60, s90 = (int(dates.searchsorted(st, side="left")) for st in start_64s)
            n7, n30, n60, n90 = stop - s7, stop - s30, stop - s60, stop - s90
            buyers30 = len(set(self._bn_sale_cids[bn][s30:stop])) if stop > s30 else 0
            out[bid, 0] = min(1.0, math.log1p(stop) / self._popularity_scale)
            out[bid, 1:5] = log_count([n7, n30, n90, buyers30])
            out[bid, 5] = float(np.clip(math.log((n30 + 1.0) / (n60 - n30 + 1.0)) / 5.0, -1.0, 1.0))
            counts_30[idx0] = n30
            out[bid, 7] = min((date_ts - pd.Timestamp(dates[stop - 1])).days / 365.0, 1.0)

        cat_totals = np.bincount(self._bn_cat_indices, weights=counts_30, minlength=self._num_cats)
        out[1:, 6] = (counts_30 / np.maximum(cat_totals[self._bn_cat_indices], 1.0)).astype(np.float32)

        self._global_cache[date_ts] = out
        while len(self._global_cache) > 512:
            self._global_cache.popitem(last=False)
        return out

    def _industry_bn_affinity(self, industry: str, date_ts: pd.Timestamp) -> np.ndarray:
        """Compute (B + 1,) industry share for each base_name strictly < date_ts."""
        if not industry:
            return np.zeros(self.num_bns + 1, dtype=np.float32)
        key = (industry, date_ts)
        if key in self._industry_cache:
            self._industry_cache.move_to_end(key)
            return self._industry_cache[key]

        out = np.zeros(self.num_bns + 1, dtype=np.float32)
        ind_dates = self._industry_event_dates.get(industry)
        if ind_dates is not None:
            cutoff_64 = date_ts.to_datetime64()
            ind_total = int(ind_dates.searchsorted(cutoff_64, side="left"))
            if ind_total > 0:
                inv_total = 1.0 / float(ind_total)
                for idx0, bn in enumerate(self.unique_bns):
                    bn_dates = self._industry_bn_dates.get((industry, bn))
                    if bn_dates is not None:
                        c = int(bn_dates.searchsorted(cutoff_64, side="left"))
                        if c > 0:
                            out[idx0 + 1] = min(1.0, c * inv_total)

        self._industry_cache[key] = out
        while len(self._industry_cache) > 2048:
            self._industry_cache.popitem(last=False)
        return out

    def build(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        *,
        category_last_lift: Optional[np.ndarray] = None,
        family_last_lift: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build (P, 19) float32 strictly-causal base_name features for all canonical products."""
        date_ts = pd.Timestamp(as_of_date)
        if pd.isna(date_ts) or date_ts >= TEST_START:
            raise ValueError("Development predictions only")

        if self._last_date is not None and date_ts < self._last_date:
            if not self.allow_non_monotonic:
                raise ValueError("BaseNameRollingFeatureBuilder must be queried chronologically.")
            self._trans_table.reset()
            self._comp_table.reset()
        self._last_date = date_ts
        self._trans_table.advance(date_ts)
        self._comp_table.advance(date_ts)

        cid = str(customer_id).strip()
        date_64 = date_ts.to_datetime64()

        # Per base_name buffer (B + 1, 19) -> scatter to (P, 19) via self.prod_bn_id
        bn_feats = np.zeros((self.num_bns + 1, NUM_BASE_NAME_FEATURES), dtype=np.float32)
        # Default days_since_last_base_sku_buy_norm = 1.0 if never bought
        bn_feats[:, 3] = 1.0

        # 0..4: Customer x base_name history
        snap = self._customer_bn_snapshot(cid, date_ts)
        if snap:
            total_vol = self._customer_total_volume(cid, date_64)
            log50 = math.log1p(50.0)
            log365 = math.log1p(365.0)
            for bid, info in snap.items():
                cnt = info["order_count"]
                vol = info["volume"]
                delta_days = max(0.0, (date_ts - info["last_date"]).total_seconds() / 86400.0)
                bn_feats[bid, 0] = 1.0
                bn_feats[bid, 1] = min(1.0, max(0.0, math.log1p(cnt) / log50))
                bn_feats[bid, 2] = min(1.0, max(0.0, vol / total_vol)) if total_vol > 0.0 else 0.0
                bn_feats[bid, 3] = min(1.0, max(0.0, math.log1p(delta_days) / log365))
                if cnt >= 2:
                    safe_gap = max(1.0, float(info["median_gap"]))
                    bn_feats[bid, 4] = min(1.0, max(0.0, (delta_days / safe_gap) / 5.0))

        # 5, 7..13: Global popularity & rolling demand
        g = self._global_bn_features(date_ts)  # (B + 1, 8)
        bn_feats[:, 5] = g[:, 0]      # base_name_global_pop_norm
        bn_feats[:, 7:14] = g[:, 1:8]  # orders_7d, 30d, 90d, buyers_30d, growth_30d, cat_rel_share, sale_age

        # 6: Industry affinity
        industry = self.customer_industry.get(cid, "")
        bn_feats[:, 6] = self._industry_bn_affinity(industry, date_ts)

        # 14..18: Cross-company transitions & same-basket complements
        baskets = self._cust_baskets.get(cid, [])
        n_prior = bisect_left(self._cust_basket_dates.get(cid, []), date_ts) if baskets else 0
        prior_baskets = baskets[:n_prior]
        if prior_baskets:
            last_bids = prior_baskets[-1][1]
            bn_last, bn_support = self._trans_table.score(last_bids)
            recent_num = np.zeros(self.num_bns + 1, dtype=np.float64)
            weight_sum = 0.0
            for i, (_, bids) in enumerate(reversed(prior_baskets[-3:])):
                w = 0.5 ** i
                recent_num += self._trans_table.score(bids)[0] * w
                weight_sum += w
            bn_recent3 = recent_num / max(weight_sum, 1e-12)
            bn_same = self._comp_table.score(last_bids)[0]
            bn_feats[:, 14] = bn_last.astype(np.float32)
            bn_feats[:, 15] = bn_recent3.astype(np.float32)
            bn_feats[:, 16] = bn_support.astype(np.float32)
            # Column 17 (hierarchical_backoff) depends on per-SKU category/family lift, computed after scatter
            bn_feats[:, 18] = bn_same.astype(np.float32)

        # Scatter from (B + 1, 19) -> (P, 19)
        prod_feats = bn_feats[self.prod_bn_id].copy()
        if prior_baskets:
            sup = prod_feats[:, 16]
            last_l = prod_feats[:, 14]
            if category_last_lift is not None and family_last_lift is not None:
                backoff = sup * last_l + (1.0 - sup) * 0.5 * (
                    np.asarray(family_last_lift, dtype=np.float32)
                    + np.asarray(category_last_lift, dtype=np.float32)
                )
            else:
                backoff = sup * last_l
            prod_feats[:, 17] = np.clip(backoff, -1.0, 1.0)
        return np.nan_to_num(prod_feats, nan=0.0, posinf=1.0, neginf=-1.0)


def assemble_arm_features_numpy(
    f41: np.ndarray,
    f19: Optional[np.ndarray],
    arm: str,
) -> np.ndarray:
    """Assemble (..., D) float32 feature array for sku_only (41D), sku_plus_base_name (60D), or base_name_only (41D)."""
    base41 = np.asarray(f41, dtype=np.float32)
    if arm == ARM_SKU_ONLY:
        return base41
    if f19 is None:
        raise ValueError(f"Arm {arm!r} requires base_name 19D features.")
    bn19 = np.asarray(f19, dtype=np.float32)
    if arm == ARM_SKU_PLUS_BASE_NAME:
        return np.concatenate([base41, bn19], axis=-1)
    if arm == ARM_BASE_NAME_ONLY:
        out = base41.copy()
        out[..., list(EXACT_SKU_SOURCE_INDICES)] = bn19
        return out
    raise ValueError(f"Unknown arm {arm!r}; expected one of {VALID_BASE_NAME_ARMS}.")


def assemble_arm_features_torch(
    f41: torch.Tensor,
    f19: Optional[torch.Tensor],
    arm: str,
) -> torch.Tensor:
    """Assemble (..., D) float32 torch tensor for one Stage 1 base_name ablation arm."""
    if arm == ARM_SKU_ONLY:
        return f41
    if f19 is None:
        raise ValueError(f"Arm {arm!r} requires base_name 19D features.")
    if arm == ARM_SKU_PLUS_BASE_NAME:
        return torch.cat([f41, f19], dim=-1)
    if arm == ARM_BASE_NAME_ONLY:
        out = f41.clone()
        idx = torch.tensor(EXACT_SKU_SOURCE_INDICES, dtype=torch.long, device=f41.device)
        out.index_copy_(-1, idx, f19)
        return out
    raise ValueError(f"Unknown arm {arm!r}; expected one of {VALID_BASE_NAME_ARMS}.")


def prepare_or_load_base_name_feature_bundle(
    *,
    cache_root: Path = CACHE_ROOT,
    build: bool = True,
    purchases_path: Union[str, Path] = "data/interim/cleaned_purchases.csv",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build or validate sealed (N, 841, 19) float16 rolling base_name features paired with rolling_observed_v2."""
    campaign_data, info, rolling_feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        build=False,
        purchases_path=purchases_path,
    )
    purchases = load_development_purchases(str(purchases_path))
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(get_datasets_locs("companies"), purchases=purchases, minimum_purchases=1)
    source_digest_info = compute_rolling_source_digest(purchases, catalogue, companies)

    # Hash base_name column over canonical product_ids so any catalogue base_name edit invalidates cache
    product_ids = list(info["product_ids"])
    cat_bn = catalogue.drop_duplicates("product_id").set_index("product_id")["base_name"].reindex(product_ids)
    bn_list = [_clean_base_name(v) for v in cat_bn.tolist()]
    bn_mapping_digest = json_digest({"product_ids": product_ids, "base_names": bn_list})

    contract = base_name_feature_contract()
    contract_digest = json_digest(contract)
    identity = json_digest(
        {
            "cache_version": BASE_NAME_CACHE_VERSION,
            "namespace": BASE_NAME_CACHE_NAMESPACE,
            "contract_digest": contract_digest,
            "rolling_features_identity": meta["rolling_features_identity"],
            "observed_source_digest": source_digest_info["observed_source_digest"],
            "base_name_mapping_digest": bn_mapping_digest,
        }
    )

    root = Path(cache_root) / "base_name_19d"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("base_name_19d cache manifest identity mismatch.")
        if manifest.get("heldout_access") is not False:
            raise ValueError("base_name_19d manifest must record heldout_access=false.")
        for split in ("train", "val"):
            p = root / f"{split}_base_name_19d.npy"
            if not p.is_file():
                raise FileNotFoundError(f"Missing base_name_19d array: {p}")
            if file_digest(p) != manifest["feature_sha256"][split]:
                raise ValueError(f"Checksum mismatch for {p}.")
    else:
        if not build:
            raise FileNotFoundError(f"base_name_19d cache missing in {root}; build=True required.")
        builder = BaseNameRollingFeatureBuilder(
            purchases,
            catalogue,
            companies,
            product_ids,
            allow_non_monotonic=False,
        )
        splits = ("train", "val")
        arrays = [
            np.lib.format.open_memmap(
                root / f"{split}_base_name_19d.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(campaign_data[split]["source"]["customer_ids"]), len(product_ids), NUM_BASE_NAME_FEATURES),
            )
            for split in splits
        ]
        queries = sorted(
            (int(day), split_idx, row_idx)
            for split_idx, split in enumerate(splits)
            for row_idx, day in enumerate(campaign_data[split]["source"]["prediction_days"].tolist())
        )
        started = time.perf_counter()
        origin = pd.Timestamp("2025-01-01")
        for i, (day, split_idx, row_idx) in enumerate(queries):
            split = splits[split_idx]
            cid = campaign_data[split]["source"]["customer_ids"][row_idx]
            f41_row = np.asarray(campaign_data[split]["features"][row_idx], dtype=np.float32)
            date_ts = origin + pd.Timedelta(days=day)
            arrays[split_idx][row_idx] = builder.build(
                cid,
                date_ts,
                category_last_lift=f41_row[:, _CATEGORY_LIFT_COL],
                family_last_lift=f41_row[:, _FAMILY_LIFT_COL],
            )
            if i > 0 and i % 5000 == 0:
                print(
                    f"[base_name_19d] {i}/{len(queries)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        for arr in arrays:
            arr.flush()
        prep_seconds = time.perf_counter() - started
        manifest = {
            "version": BASE_NAME_CACHE_VERSION,
            "namespace": BASE_NAME_CACHE_NAMESPACE,
            "identity": identity,
            "contract": contract,
            "contract_digest": contract_digest,
            "rolling_features_identity": meta["rolling_features_identity"],
            "observed_source_digest": source_digest_info["observed_source_digest"],
            "base_name_mapping_digest": bn_mapping_digest,
            "unique_base_names": len(set(bn_list) - {""}),
            "non_empty_base_name_products": sum(1 for b in bn_list if b),
            "feature_sha256": {
                split: file_digest(root / f"{split}_base_name_19d.npy") for split in splits
            },
            "preparation_seconds": float(prep_seconds),
            "heldout_access": False,
        }
        atomic_json(manifest, manifest_path, overwrite=False)

    # Attach read-only memmaps and base_name lookup to campaign_data
    for split in ("train", "val"):
        bn_mmap = np.load(root / f"{split}_base_name_19d.npy", mmap_mode="r")
        expected_shape = (len(campaign_data[split]["u"]), len(product_ids), NUM_BASE_NAME_FEATURES)
        if bn_mmap.shape != expected_shape:
            raise ValueError(f"Unexpected {split}_base_name_19d.npy shape: {bn_mmap.shape} != {expected_shape}")
        campaign_data[split]["base_name_features"] = bn_mmap

    # Also compute per-validation-query pack-switch vs true-new base_name target masks for clean diagnostics!
    cat_all_bn = catalogue.drop_duplicates("product_id").set_index("product_id")["base_name"].map(_clean_base_name).to_dict()
    meta["product_base_names"] = bn_list
    meta["all_pid_to_base_name"] = cat_all_bn
    meta["base_name_manifest"] = manifest
    meta["base_name_identity"] = manifest["identity"]
    return campaign_data, info, manifest, meta
