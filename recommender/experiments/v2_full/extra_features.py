"""Strictly causal (< as_of_date) Wave 2B `extra_cand_5` builder and canonical `cand_42` schemas (`v2_full`).

Implements the canonical pruned `v2_full` feature schema without rebuilding the 12 GB `cand_38.npy` memmap:
  - Pruned candidate schema (`cand_42`): `cand_38` minus dead col `22` (`base_name_growth_30d`) +
    5 strictly point-in-time `< as_of_date` features (`extra_cand_5`; weak `mgr_base_name_growth_30d_vs_90d`
    permanently removed):
      0. `mgr_base_name_share_30d`
      1. `mgr_industry_base_name_share_90d`
      2. `region_base_name_share_90d`
      3. `full_cf_cosine`
      4. `full_cf_recency_weighted`
  - Pruned customer dynamic schema (`CUST_7`): keeps indices `(0, 2, 3, 4, 5, 6, 8)` of `CUST_11`,
    dropping verified dead cols `1, 7, 9, 10`.
  - Pruned history sequence statistics (`SEQ_5`): keeps indices `(0, 1, 5, 6, 8)` of the 9D source
    sequence (`qty_log1p_norm`, `live_age`, `prior_bn_count_norm`, `prior_bn_vol_share`,
    `real_event_indicator`), dropping `gap_days_norm` (col `2`), `month_sin/cos` (`3, 4`), and
    `is_repeat_event` (`7`).

All rolling statistics strictly enforce `event_day < query_day` and reject any `purchase_date >= test_start`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
)
from recommender.sequences import PURCHASE_DATE_ORIGIN

from .prepare_v2_full import load_prepared_metadata
from .config import (
    COMMERCIAL_DEV_PARQUET_PATH,
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    load_v2_development_purchases,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
)

EXTRA_CAND_5_VERSION: int = 3

# 1. Pruned cand_38 -> 37 cols (drops dead col 22: base_name_growth_30d)
DEAD_CAND_38_INDEX: int = 22
assert CANDIDATE_38_FEATURE_NAMES[DEAD_CAND_38_INDEX] == "base_name_growth_30d"
CAND_38_KEEP_INDICES: Tuple[int, ...] = tuple(i for i in range(len(CANDIDATE_38_FEATURE_NAMES)) if i != DEAD_CAND_38_INDEX)
CAND_38_KEEP_INDEX_ARRAY = np.asarray(CAND_38_KEEP_INDICES, dtype=np.int64)

EXTRA_CAND_5_FEATURE_NAMES: Tuple[str, ...] = (
    "mgr_base_name_share_30d",
    "mgr_industry_base_name_share_90d",
    "region_base_name_share_90d",
    "full_cf_cosine",
    "full_cf_recency_weighted",
)
NUM_EXTRA_CAND_FEATURES: int = len(EXTRA_CAND_5_FEATURE_NAMES)
assert NUM_EXTRA_CAND_FEATURES == 5

CAND_42_FEATURE_NAMES: Tuple[str, ...] = (
    tuple(CANDIDATE_38_FEATURE_NAMES[i] for i in CAND_38_KEEP_INDICES) + EXTRA_CAND_5_FEATURE_NAMES
)
NUM_CAND_42_FEATURES: int = len(CAND_42_FEATURE_NAMES)
assert NUM_CAND_42_FEATURES == 42

# Repeat-only SKU columns are indices 0..4; indices 5..41 (37 cols) are non-repeat candidate features
CAND_42_REPEAT_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4)
CAND_42_NON_REPEAT_INDICES: Tuple[int, ...] = tuple(range(5, NUM_CAND_42_FEATURES))
CAND_42_NON_REPEAT_NAMES: Tuple[str, ...] = tuple(CAND_42_FEATURE_NAMES[i] for i in CAND_42_NON_REPEAT_INDICES)
NUM_CAND_42_NON_REPEAT: int = len(CAND_42_NON_REPEAT_INDICES)
assert NUM_CAND_42_NON_REPEAT == 37

# 7 purely global product-level features accessible to all / cold-start customers
# (retained during the 30% cold-start dropout branch of Option A 70/30 feature-group dropout)
CAND_42_GLOBAL_PRODUCT_FEATURE_NAMES: Tuple[str, ...] = (
    "base_name_global_pop_norm",
    "base_name_orders_7d",
    "base_name_orders_30d",
    "base_name_orders_90d",
    "base_name_buyers_30d",
    "base_name_category_relative_share",
    "base_name_global_sale_age",
)
UNSEEN_37_GLOBAL_LOCAL_INDICES: Tuple[int, ...] = tuple(
    CAND_42_NON_REPEAT_NAMES.index(name) for name in CAND_42_GLOBAL_PRODUCT_FEATURE_NAMES
)
assert len(UNSEEN_37_GLOBAL_LOCAL_INDICES) == 7

# 2. Pruned CUST_11 -> CUST_7 (drops verified dead cols 1, 7, 9, 10)
CUST_7_KEEP_INDICES: Tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
CUST_7_FEATURE_NAMES: Tuple[str, ...] = tuple(CUSTOMER_DYNAMIC_11_FEATURE_NAMES[i] for i in CUST_7_KEEP_INDICES)
assert CUST_7_FEATURE_NAMES == (
    "customer_age",
    "customer_30d_activity",
    "customer_90d_activity",
    "cust_prepaid_or_cash_ratio",
    "cust_avg_deferral_days_norm",
    "cust_last_deferral_days_norm",
    "cust_overdue_unpaid_ratio",
)
NUM_CUST_7_FEATURES: int = len(CUST_7_FEATURE_NAMES)

# 3. Pruned history sequence stats -> SEQ_5 (drops col 2 gap_days_norm, cols 3..4 month, col 7 repeat; keeps col 1 live_age!)
SEQ_5_ACTIVE_INDICES: Tuple[int, ...] = (0, 1, 5, 6, 8)
NUM_SEQ_5_STATS: int = len(SEQ_5_ACTIVE_INDICES)

CF_RECENCY_HALF_LIFE_DAYS: float = 180.0


def assemble_cand_42(cand_38: np.ndarray, extra_5: np.ndarray) -> np.ndarray:
    """Concatenate pruned `cand_38` (37 cols, dropping col 22) with `extra_5` (5 cols) -> `cand_42`."""
    if cand_38.shape[:-1] != extra_5.shape[:-1]:
        raise ValueError(f"Shape mismatch: cand_38={cand_38.shape} vs extra_5={extra_5.shape}")
    if cand_38.shape[-1] != 38 or extra_5.shape[-1] != 5:
        raise ValueError(f"Expected trailing dims (38, 5), got ({cand_38.shape[-1]}, {extra_5.shape[-1]})")
    c37 = np.asarray(cand_38[..., CAND_38_KEEP_INDEX_ARRAY], dtype=np.float32)
    e5 = np.asarray(extra_5, dtype=np.float32)
    return np.concatenate([c37, e5], axis=-1)


def prune_cust_7(cust_11: torch.Tensor) -> torch.Tensor:
    """Slice `CUST_11` -> `CUST_7` (keeps indices `(0, 2, 3, 4, 5, 6, 8)`)."""
    if cust_11.shape[-1] != 11:
        raise ValueError(f"Expected 11 customer features, got {cust_11.shape[-1]}")
    return cust_11[..., list(CUST_7_KEEP_INDICES)]


class _RollingWindowCounter:
    """Exact sliding-window per-key formulation counts strictly over `[day - window, day)`."""

    def __init__(self, window_days: int, num_items: int):
        self.window_days = int(window_days)
        self.num_items = int(num_items)
        self._queue: Deque[Tuple[int, Any, int]] = deque()
        self._counts: Dict[Any, np.ndarray] = {}
        self._totals: Dict[Any, int] = defaultdict(int)

    def add(self, day: int, key: Any, item: int) -> None:
        self._queue.append((day, key, item))
        arr = self._counts.get(key)
        if arr is None:
            arr = np.zeros(self.num_items, dtype=np.int32)
            self._counts[key] = arr
        arr[item] += 1
        self._totals[key] += 1

    def expire_before(self, min_day: int) -> None:
        while self._queue and self._queue[0][0] < min_day:
            _, key, item = self._queue.popleft()
            self._counts[key][item] -= 1
            self._totals[key] -= 1
            if self._totals[key] <= 0:
                del self._counts[key]
                del self._totals[key]

    def share_vector(self, key: Any) -> Tuple[Optional[np.ndarray], int]:
        total = self._totals.get(key, 0)
        if total <= 0:
            return None, 0
        return self._counts[key].astype(np.float32) / float(total), int(total)


def _split_identity_digest(split_meta: Mapping[str, Any]) -> str:
    days = split_meta["prediction_days"].cpu().numpy().astype(np.int32).tolist()
    return json_digest(
        {
            "n_rows": len(split_meta["customer_ids"]),
            "customer_ids": [str(c) for c in split_meta["customer_ids"]],
            "prediction_days": days,
            **{k: split_meta[k].tolist() for k in ("region_id", "industry_id") if k in split_meta},
        }
    )


def _extra_cand_5_identity(
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    region_map: Mapping[str, int],
    industry_map: Mapping[str, int],
    commercial_sha256: str,
    train_cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
) -> str:
    return json_digest(
        {
            "version": EXTRA_CAND_5_VERSION,
            "feature_names": list(EXTRA_CAND_5_FEATURE_NAMES),
            "base_names": list(shared_meta["universe"]["base_names"]),
            "sku_to_formulation": shared_meta["universe"]["all_pid_to_bn"],
            "formulation_indices": shared_meta["universe"]["bn_to_idx"],
            "customer_regions": region_map,
            "customer_industries": industry_map,
            "commercial_sha256": str(commercial_sha256),
            "train_cutoff": str(pd.Timestamp(train_cutoff).normalize()),
            "test_start": str(pd.Timestamp(test_start).normalize()),
            "cf_recency_half_life_days": CF_RECENCY_HALF_LIFE_DAYS,
            "train_split": _split_identity_digest(tr_meta),
            "val_split": _split_identity_digest(va_meta),
        }
    )


def compute_extra_cand_5_arrays(
    *,
    commercial_events: pd.DataFrame,
    universe: Mapping[str, Any],
    customer_region_map: Mapping[str, int],
    customer_industry_map: Mapping[str, int],
    splits: Mapping[str, Mapping[str, Any]],
    out_paths: Optional[Mapping[str, Path]] = None,
    test_start: pd.Timestamp = TEST_START_V2,
) -> Dict[str, np.ndarray]:
    """Compute strictly point-in-time (< as_of_date) `(N_split, N_bn, 5)` float16 `extra_cand_5` arrays.

    Every rolling statistic (30d manager share, 90d manager x industry share, 90d regional share,
    exact incremental item-item cosine CF, and 180d recency-weighted CF) uses ONLY events with
    `event_day < query_day`, and raises `ValueError` if any event or query touches `>= test_start`.
    """
    test_start_ts = pd.Timestamp(test_start).normalize()
    test_start_day = int((test_start_ts - PURCHASE_DATE_ORIGIN).days)
    n_bn = int(universe["num_base_names"])
    all_pid_to_bn = universe["all_pid_to_bn"]
    bn_to_idx = universe["bn_to_idx"]

    # Validate query dates strictly < test_start
    split_query_days: Dict[str, np.ndarray] = {}
    split_cids: Dict[str, List[str]] = {}
    split_regions: Dict[str, np.ndarray] = {}
    split_industries: Dict[str, np.ndarray] = {}
    queries_by_day: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for sname, smeta in splits.items():
        days = smeta["prediction_days"].cpu().numpy()
        if not np.isfinite(days).all() or not np.equal(days, np.floor(days)).all():
            raise ValueError("Prediction days must be finite integer calendar days.")
        q_days = days.astype(np.int64)
        if len(q_days) and int(q_days.max()) >= test_start_day:
            raise ValueError(
                f"Held-out test query detected in split={sname!r} (>= {test_start_ts.date()})."
            )
        cids = [str(c).strip() for c in smeta["customer_ids"]]
        split_query_days[sname] = q_days
        split_cids[sname] = cids
        if "region_id" in smeta:
            split_regions[sname] = smeta["region_id"].cpu().numpy().astype(np.int32)
        else:
            split_regions[sname] = np.array([int(customer_region_map.get(c, 0)) for c in cids], dtype=np.int32)
        if "industry_id" in smeta:
            split_industries[sname] = smeta["industry_id"].cpu().numpy().astype(np.int32)
        else:
            split_industries[sname] = np.array([int(customer_industry_map.get(c, 0)) for c in cids], dtype=np.int32)
        for row_idx, d in enumerate(q_days.tolist()):
            queries_by_day[d].append((sname, row_idx))

    # Validate and normalize development commercial events strictly < test_start
    ev = commercial_events.copy()
    if not ev.empty:
        ev["purchase_date"] = pd.to_datetime(ev["purchase_date"], format="mixed").dt.normalize()
        if bool(ev["purchase_date"].isna().any()) or not bool(ev["purchase_date"].lt(test_start_ts).all()):
            raise ValueError(f"Development events only (< {test_start_ts.date()}).")
        ev["customer_id"] = ev["customer_id"].astype(str).str.strip()
        ev["product_id"] = ev["product_id"].astype(str).str.strip()
        ev["bn_idx"] = ev["product_id"].map(all_pid_to_bn).map(bn_to_idx)
        ev = ev.dropna(subset=["bn_idx"]).copy()
        ev["bn_idx"] = ev["bn_idx"].astype(np.int32)
        ev["day"] = (ev["purchase_date"] - PURCHASE_DATE_ORIGIN).dt.days.astype(np.int32)
        if "manager_id" in ev.columns:
            ev["manager_id"] = ev["manager_id"].fillna("").astype(str).str.strip()
        else:
            ev["manager_id"] = ""

        # Collapse to distinct (day, customer_id, bn_idx) keeping last non-empty manager_id for that basket line
        ev = ev.sort_values(["day", "customer_id", "bn_idx", "manager_id"], kind="mergesort")
        ev = ev.drop_duplicates(subset=["day", "customer_id", "bn_idx"], keep="last").reset_index(drop=True)
        ev_days = ev["day"].to_numpy(dtype=np.int32)
        ev_cids = ev["customer_id"].tolist()
        ev_bns = ev["bn_idx"].to_numpy(dtype=np.int32)
        ev_mgrs = ev["manager_id"].tolist()
        ev_regs = np.array([int(customer_region_map.get(c, 0)) for c in ev_cids], dtype=np.int32)
        ev_inds = np.array([int(customer_industry_map.get(c, 0)) for c in ev_cids], dtype=np.int32)
    else:
        ev_days = np.empty(0, dtype=np.int32)
        ev_cids = []
        ev_bns = np.empty(0, dtype=np.int32)
        ev_mgrs = []
        ev_regs = np.empty(0, dtype=np.int32)
        ev_inds = np.empty(0, dtype=np.int32)

    # Allocate output buffers (memmap if out_paths provided, else in-memory float16)
    outputs: Dict[str, np.ndarray] = {}
    for sname in splits:
        n_rows = len(split_cids[sname])
        if out_paths is not None and sname in out_paths:
            p = Path(out_paths[sname])
            p.parent.mkdir(parents=True, exist_ok=True)
            outputs[sname] = np.lib.format.open_memmap(
                p, mode="w+", dtype=np.float16, shape=(n_rows, n_bn, NUM_EXTRA_CAND_FEATURES)
            )
        else:
            outputs[sname] = np.zeros((n_rows, n_bn, NUM_EXTRA_CAND_FEATURES), dtype=np.float16)

    # Strictly causal chronological sweep state
    mgr_30 = _RollingWindowCounter(30, n_bn)
    mgr_ind_90 = _RollingWindowCounter(90, n_bn)
    reg_90 = _RollingWindowCounter(90, n_bn)

    cust_last_mgr: Dict[str, str] = {}
    cust_seen_bns: Dict[str, List[int]] = {}
    cust_seen_sets: Dict[str, set[int]] = {}
    cust_last_bn_day: Dict[str, Dict[int, int]] = {}

    cooc = np.zeros((n_bn, n_bn), dtype=np.float32)
    deg = np.zeros(n_bn, dtype=np.float32)
    cf_dirty = False
    sim_mat = np.zeros((n_bn, n_bn), dtype=np.float32)

    ev_ptr = 0
    n_events = len(ev_days)
    sorted_query_days = sorted(queries_by_day.keys())
    ln2_over_hl = math.log(2.0) / float(CF_RECENCY_HALF_LIFE_DAYS)

    t0 = time.perf_counter()
    done_queries = 0
    total_queries = sum(len(v) for v in queries_by_day.values())

    for day_idx, q_day in enumerate(sorted_query_days):
        # Ingest all events strictly BEFORE q_day (event_day < q_day)
        while ev_ptr < n_events and int(ev_days[ev_ptr]) < q_day:
            d = int(ev_days[ev_ptr])
            cid = ev_cids[ev_ptr]
            bn = int(ev_bns[ev_ptr])
            mgr = ev_mgrs[ev_ptr]
            reg = int(ev_regs[ev_ptr])
            ind = int(ev_inds[ev_ptr])

            if mgr:
                cust_last_mgr[cid] = mgr
                mgr_30.add(d, mgr, bn)
                if ind > 0:
                    mgr_ind_90.add(d, (mgr, ind), bn)
            if reg > 0:
                reg_90.add(d, reg, bn)

            # Update exact incremental customer-formulation co-occurrence matrix
            last_map = cust_last_bn_day.get(cid)
            if last_map is None:
                last_map = {}
                cust_last_bn_day[cid] = last_map
            last_map[bn] = d

            seen_set = cust_seen_sets.get(cid)
            if seen_set is None:
                cust_seen_sets[cid] = {bn}
                cust_seen_bns[cid] = [bn]
                deg[bn] += 1.0
                cf_dirty = True
            elif bn not in seen_set:
                prev_list = cust_seen_bns[cid]
                if prev_list:
                    prev_arr = np.asarray(prev_list, dtype=np.int64)
                    cooc[prev_arr, bn] += 1.0
                    cooc[bn, prev_arr] += 1.0
                seen_set.add(bn)
                prev_list.append(bn)
                deg[bn] += 1.0
                cf_dirty = True

            ev_ptr += 1

        # Expire sliding windows strictly to [q_day - W, q_day)
        mgr_30.expire_before(q_day - 30)
        mgr_ind_90.expire_before(q_day - 90)
        reg_90.expire_before(q_day - 90)

        # Recompute exact cosine similarity matrix S_t only if co-occurrence state changed
        if cf_dirty:
            inv_sqrt = np.zeros(n_bn, dtype=np.float32)
            nz = deg > 0.0
            inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
            sim_mat = cooc * inv_sqrt[:, None] * inv_sqrt[None, :]
            np.fill_diagonal(sim_mat, 0.0)
            cf_dirty = False

        day_queries = queries_by_day[q_day]
        m = len(day_queries)

        # Cache per-day share vectors for repeated managers / regions / (mgr, industry) on the same day
        mgr_30_cache: Dict[str, Optional[np.ndarray]] = {}
        mgr_ind_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
        reg_90_cache: Dict[int, Optional[np.ndarray]] = {}

        # Batch CF queries for this day into two dense (m_cf, n_bn) weight matrices -> single fast BLAS matmul!
        cf_positions: List[int] = []
        cf_unit_rows: List[np.ndarray] = []
        cf_rec_rows: List[np.ndarray] = []

        day_buf = np.zeros((m, n_bn, NUM_EXTRA_CAND_FEATURES), dtype=np.float32)

        for local_i, (sname, row_idx) in enumerate(day_queries):
            cid = split_cids[sname][row_idx]
            reg = int(split_regions[sname][row_idx])
            ind = int(split_industries[sname][row_idx])
            curr_mgr = cust_last_mgr.get(cid, "")

            # Col 0: mgr_base_name_share_30d
            if curr_mgr:
                if curr_mgr not in mgr_30_cache:
                    v30, _ = mgr_30.share_vector(curr_mgr)
                    mgr_30_cache[curr_mgr] = v30
                v30 = mgr_30_cache[curr_mgr]
                if v30 is not None:
                    day_buf[local_i, :, 0] = v30

                # Col 1: mgr_industry_base_name_share_90d
                if ind > 0:
                    mi_key = (curr_mgr, ind)
                    if mi_key not in mgr_ind_cache:
                        vmi, _ = mgr_ind_90.share_vector(mi_key)
                        mgr_ind_cache[mi_key] = vmi
                    vmi = mgr_ind_cache[mi_key]
                    if vmi is not None:
                        day_buf[local_i, :, 1] = vmi

            # Col 2: region_base_name_share_90d (populated for BOTH Returning & First-Observed!)
            if reg > 0:
                if reg not in reg_90_cache:
                    vr, _ = reg_90.share_vector(reg)
                    reg_90_cache[reg] = vr
                vr = reg_90_cache[reg]
                if vr is not None:
                    day_buf[local_i, :, 2] = vr

            # Cols 3 & 4: prepare customer history weights for exact BLAS CF matmul
            seen_list = cust_seen_bns.get(cid)
            if seen_list:
                idx_arr = np.asarray(seen_list, dtype=np.int64)
                w_unit = np.zeros(n_bn, dtype=np.float32)
                w_unit[idx_arr] = 1.0 / math.sqrt(float(len(seen_list)))

                last_map = cust_last_bn_day[cid]
                ages = np.array([max(0, q_day - last_map[b]) for b in seen_list], dtype=np.float32)
                raw_w = np.exp(-ln2_over_hl * ages)
                norm_w = float(np.linalg.norm(raw_w))
                w_rec = np.zeros(n_bn, dtype=np.float32)
                if norm_w > 0.0:
                    w_rec[idx_arr] = raw_w / norm_w

                cf_positions.append(local_i)
                cf_unit_rows.append(w_unit)
                cf_rec_rows.append(w_rec)

        if cf_positions:
            w_u_mat = np.stack(cf_unit_rows, axis=0)  # (m_cf, n_bn)
            w_r_mat = np.stack(cf_rec_rows, axis=0)   # (m_cf, n_bn)
            raw_u = np.maximum(w_u_mat @ sim_mat, 0.0)
            raw_r = np.maximum(w_r_mat @ sim_mat, 0.0)
            cf_u = raw_u / (raw_u + 1.0)
            cf_r = raw_r / (raw_r + 1.0)
            for k, pos in enumerate(cf_positions):
                day_buf[pos, :, 3] = cf_u[k]
                day_buf[pos, :, 4] = cf_r[k]

        day_f16 = np.nan_to_num(day_buf, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float16)
        for local_i, (sname, row_idx) in enumerate(day_queries):
            outputs[sname][row_idx] = day_f16[local_i]

        done_queries += m
        if (day_idx + 1) % 250 == 0 or (day_idx + 1) == len(sorted_query_days):
            print(
                f"[v2-extra5] day {day_idx + 1}/{len(sorted_query_days)} "
                f"queries={done_queries:,}/{total_queries:,} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    for arr in outputs.values():
        if isinstance(arr, np.memmap):
            arr.flush()
    return outputs


def _build_customer_region_industry_maps(
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    *,
    companies: Optional[pd.DataFrame] = None,
    vocabs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build consistent `customer_id -> region_id` and `customer_id -> industry_id` maps."""
    reg_map: Dict[str, int] = {}
    ind_map: Dict[str, int] = {}
    if companies is not None and vocabs is not None:
        from recommender.train_retriever import build_customer_metadata

        profiles = build_customer_metadata(companies, vocabs)
        for cid, prof in profiles.items():
            reg_map[str(cid).strip()] = int(prof.get("region_id", 0))
            ind_map[str(cid).strip()] = int(prof.get("industry_id", 0))
    for smeta in (tr_meta, va_meta):
        if "region_id" in smeta and "industry_id" in smeta:
            regs = smeta["region_id"].cpu().numpy().tolist()
            inds = smeta["industry_id"].cpu().numpy().tolist()
            for cid, r, ind in zip(smeta["customer_ids"], regs, inds, strict=True):
                reg_map[str(cid).strip()] = int(r)
                ind_map[str(cid).strip()] = int(ind)
    return reg_map, ind_map


def build_or_load_extra_cand_5(
    *,
    prepared_root: Path = PREPARED_ROOT,
    commercial_parquet_path: Path = COMMERCIAL_DEV_PARQUET_PATH,
    companies: Optional[pd.DataFrame] = None,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
    test_start: pd.Timestamp = TEST_START_V2,
    force: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Build or validate sealed `train_extra_cand_5.npy` & `val_extra_cand_5.npy` memmaps."""
    prepared_root = Path(prepared_root)
    shared_meta, tr_meta, va_meta = load_prepared_metadata(
        prepared_root, train_cutoff=train_cutoff, test_start=test_start
    )

    if companies is None and RAW_CUSTOMERS_PATH.is_file():
        from recommender.experiments.static_enrichment.metadata import load_enriched_companies

        companies = load_enriched_companies(
            RAW_CUSTOMERS_PATH, purchases=load_v2_development_purchases(), minimum_purchases=0
        )
    reg_map, ind_map = _build_customer_region_industry_maps(
        tr_meta, va_meta, companies=companies, vocabs=shared_meta.get("vocabs")
    )
    comm_sha = file_digest(commercial_parquet_path)
    identity = _extra_cand_5_identity(
        shared_meta=shared_meta,
        tr_meta=tr_meta,
        va_meta=va_meta,
        region_map=reg_map,
        industry_map=ind_map,
        commercial_sha256=comm_sha,
        train_cutoff=train_cutoff,
        test_start=test_start,
    )

    cache_dir = prepared_root / "extra_cand_5" / identity
    cache_dir.mkdir(parents=True, exist_ok=True)
    tr_path = cache_dir / "train_extra_cand_5.npy"
    va_path = cache_dir / "val_extra_cand_5.npy"
    manifest_path = cache_dir / "extra_cand_5_manifest.json"

    if not force and manifest_path.is_file() and tr_path.is_file() and va_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("identity") == identity
            and manifest.get("output_sha256", {}).get("train") == file_digest(tr_path)
            and manifest.get("output_sha256", {}).get("val") == file_digest(va_path)
        ):
            print(f"[v2-extra5] Reusing verified extra_cand_5 cache -> {manifest_path}", flush=True)
            return tr_path, va_path, manifest

    t0 = time.perf_counter()
    test_start_ts = pd.Timestamp(test_start).normalize()
    date = ds.field("purchase_date")
    if ds.dataset(commercial_parquet_path, format="parquet").count_rows(
        filter=date.is_null() | (date >= test_start_ts)
    ):
        raise ValueError(f"Development events only (< {test_start_ts.date()}).")
    commercial_events = pd.read_parquet(commercial_parquet_path, filters=[("purchase_date", "<", test_start_ts)])

    with TemporaryDirectory(dir=cache_dir) as scratch:
        tr_tmp = Path(scratch) / tr_path.name
        va_tmp = Path(scratch) / va_path.name
        outputs = compute_extra_cand_5_arrays(
            commercial_events=commercial_events,
            universe=shared_meta["universe"],
            customer_region_map=reg_map,
            customer_industry_map=ind_map,
            splits={"train": tr_meta, "val": va_meta},
            out_paths={"train": tr_tmp, "val": va_tmp},
            test_start=test_start,
        )
        # Close memmaps cleanly before atomic replace
        del outputs
        tr_tmp.replace(tr_path)
        va_tmp.replace(va_path)

    manifest = {
        "version": EXTRA_CAND_5_VERSION,
        "identity": identity,
        "feature_names": list(EXTRA_CAND_5_FEATURE_NAMES),
        "cand_42_feature_names": list(CAND_42_FEATURE_NAMES),
        "cust_7_feature_names": list(CUST_7_FEATURE_NAMES),
        "seq_5_active_indices": list(SEQ_5_ACTIVE_INDICES),
        "train_cutoff": str(pd.Timestamp(train_cutoff).normalize()),
        "test_start": str(pd.Timestamp(test_start).normalize()),
        "heldout_access": False,
        "shapes": {
            "train": [len(tr_meta["customer_ids"]), int(shared_meta["universe"]["num_base_names"]), NUM_EXTRA_CAND_FEATURES],
            "val": [len(va_meta["customer_ids"]), int(shared_meta["universe"]["num_base_names"]), NUM_EXTRA_CAND_FEATURES],
        },
        "output_sha256": {
            "train": file_digest(tr_path),
            "val": file_digest(va_path),
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    atomic_json(manifest, manifest_path, overwrite=True)
    print(f"[v2-extra5] Built and sealed extra_cand_5 in {manifest['elapsed_seconds']:.1f}s -> {manifest_path}", flush=True)
    return tr_path, va_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_or_load_extra_cand_5(force=args.force)


if __name__ == "__main__":
    main()
