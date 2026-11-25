"""Exact offline 70/30 synthetic unseen dataset builder for Wave 2C (`v2_full`).

Converts `< TRAIN_CUTOFF_V2` training repeat baskets into exact synthetic unseen examples for
`OptionAUnseenHead` with zero leakage and zero un-recalculated feature shortcuts:
  - **70% Synthetic Returning Unseen (Masked-b*):**
      Picks 1 repeat positive formulation `b*` from the basket, removes ALL prior purchases of `b*`
      from that customer's history (`< as_of_date`), and exactly recalculates:
        1. `Tokens 1..32` (`sequence` with cols `3:5 = 0.0`, `attention_mask`, `seq_product_ids`,
           `seq_purchase_dates`), matching `build_base_name_sequences`.
        2. `CUST_7` (customer age, 30d/90d activity, and commercial financial ratios excluding `b*`
           lines), matching `NativeBaseNameFeatureEngine` + `CommercialRollingFeatureBuilder`.
        3. All 27 customer/manager/history-dependent columns of `cand_37` (`CAND_42_CHANGED_27_INDICES`:
           cat/subcat/brand/line/family/viscosity/app shares & ages, transition/complement lifts,
           30d/90d manager shares for the remaining active manager, and exact cosine/recency CF
           excluding `b*`).
      Treats `b*` as unseen (`unseen_bn[b*] = True`, `y_unseen[b*] = 1`) along with any genuine
      unseen positives in that basket, while keeping the customer's other unmasked prior items
      excluded from `candidate_mask_unseen`.
  - **30% Synthetic First-Occurrence (Cold-Start) Unseen:**
      Wipes the entire purchase history (`attention_mask = 0`, `Tokens 1..32 = 0`), sets `CUST_7`
      to exact first-observed defaults (`customer_age = 1.0`, other 6 = `0.0`), preserves the 10
      history-independent `cand_37` columns (`CAND_42_FO_VARIABLE_INDICES`), and sets the remaining
      27 `cand_37` columns to exact zero-history defaults (`category_age = 1.0`, `family_age = 1.0`,
      others `0.0`). Computed deterministically on-the-fly with zero extra disk storage!

All `cand_42` repeat columns `0..4` stay at exact unseen defaults `[0, 0, 0, 1, 0]` for the shared
cross-attention bias and are NEVER passed into `OptionAUnseenHead` (which strictly takes `cand_37`).
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import RollingCandidatePool
from recommender.experiments.dynamic_selector.feature_builder import log_count
from recommender.experiments.roadmap_v2.commercial_features import parse_deferral_days
from recommender.experiments.roadmap_v2.dated_baskets import aligned_commercial_events
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    observed_pack_ids,
)
from recommender.experiments.static_enrichment.metadata import (
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.scope import actual_purchase_mask
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save

from .config import (
    COMMERCIAL_DEV_PARQUET_PATH,
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    RAW_ITEMS_PATH,
    SEED,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
    load_v2_development_purchases,
)
from .extra_features import (
    CAND_42_FEATURE_NAMES,
    CAND_42_NON_REPEAT_INDICES,
    CAND_42_REPEAT_INDICES,
    CF_RECENCY_HALF_LIFE_DAYS,
    CUST_7_FEATURE_NAMES,
    EXTRA_CAND_5_FEATURE_NAMES,
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    _RollingWindowCounter,
    _build_customer_region_industry_maps,
    _split_identity_digest,
)
from .prepare_v2_full import load_prepared_metadata
from .provenance import input_digest

SYNTHETIC_UNSEEN_VERSION: int = 2
SYNTHETIC_RETURNING_FRACTION: float = 0.70

# Exact 10 cand_42 columns that vary even for a zero-history First-Occurrence customer
# (and are unchanged when masking a single customer's prior purchases of b*).
CAND_42_FO_VARIABLE_NAMES: Tuple[str, ...] = (
    "base_name_global_pop_norm",
    "base_name_industry_affinity",
    "base_name_orders_7d",
    "base_name_orders_30d",
    "base_name_orders_90d",
    "base_name_buyers_30d",
    "base_name_category_relative_share",
    "base_name_global_sale_age",
    "app_matches_company_equipment",
    "region_base_name_share_90d",
)
CAND_42_FO_VARIABLE_INDICES: Tuple[int, ...] = tuple(
    CAND_42_FEATURE_NAMES.index(name) for name in CAND_42_FO_VARIABLE_NAMES
)
CAND_42_FO_VARIABLE_ARRAY = np.asarray(CAND_42_FO_VARIABLE_INDICES, dtype=np.int64)
assert len(CAND_42_FO_VARIABLE_INDICES) == 10

# Minimal source selectors to read ONLY the 10 FO_VAR columns from cand_38 (9 cols) + extra_5 (1 col)
CAND_38_FO_9_NAMES: Tuple[str, ...] = tuple(
    name for name in CAND_42_FO_VARIABLE_NAMES if name in CANDIDATE_38_FEATURE_NAMES
)
CAND_38_FO_9_INDICES: Tuple[int, ...] = tuple(CANDIDATE_38_FEATURE_NAMES.index(name) for name in CAND_38_FO_9_NAMES)
CAND_38_FO_9_ARRAY = np.asarray(CAND_38_FO_9_INDICES, dtype=np.int64)
CAND_42_FO_FROM_C38_ARRAY = np.asarray(
    [CAND_42_FEATURE_NAMES.index(name) for name in CAND_38_FO_9_NAMES], dtype=np.int64
)
EXTRA_5_REGION_COL: int = EXTRA_CAND_5_FEATURE_NAMES.index("region_base_name_share_90d")
CAND_42_REGION_COL: int = CAND_42_FEATURE_NAMES.index("region_base_name_share_90d")
assert len(CAND_38_FO_9_INDICES) == 9
assert sorted(CAND_42_FO_FROM_C38_ARRAY.tolist() + [CAND_42_REGION_COL]) == sorted(CAND_42_FO_VARIABLE_INDICES)

# Exact 27 cand_42 columns (strictly inside cand_37 = 5..41) that depend on customer history or active manager.
CAND_42_CHANGED_27_INDICES: Tuple[int, ...] = tuple(
    i for i in CAND_42_NON_REPEAT_INDICES if i not in CAND_42_FO_VARIABLE_INDICES
)
CAND_42_CHANGED_27_NAMES: Tuple[str, ...] = tuple(
    CAND_42_FEATURE_NAMES[i] for i in CAND_42_CHANGED_27_INDICES
)
CAND_42_CHANGED_27_ARRAY = np.asarray(CAND_42_CHANGED_27_INDICES, dtype=np.int64)
NUM_CAND_42_CHANGED_27: int = len(CAND_42_CHANGED_27_INDICES)
assert NUM_CAND_42_CHANGED_27 == 27
assert sorted(CAND_42_FO_VARIABLE_INDICES + CAND_42_CHANGED_27_INDICES) == list(CAND_42_NON_REPEAT_INDICES)
assert not (set(CAND_42_REPEAT_INDICES) & set(CAND_42_CHANGED_27_INDICES))

# Local slot lookup inside the 27-col recalculated returning array (guaranteed aligned with CAND_42_CHANGED_27_NAMES)
_C27_SLOT: Dict[str, int] = {name: i for i, name in enumerate(CAND_42_CHANGED_27_NAMES)}
_DAYS_SINCE_LAST_COL: int = CAND_42_FEATURE_NAMES.index("days_since_last_base_sku_buy_norm")
_CAT_AGE_COL: int = CAND_42_FEATURE_NAMES.index("category_age")
_FAM_AGE_COL: int = CAND_42_FEATURE_NAMES.index("family_age")
assert (_DAYS_SINCE_LAST_COL, _CAT_AGE_COL, _FAM_AGE_COL) == (3, 13, 15)

# Exact zero-history First-Occurrence defaults (verified on train_meta.pt & train_cand_38.npy)
CUST_7_FO_DEFAULT: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
assert len(CUST_7_FO_DEFAULT) == NUM_CUST_7_FEATURES


def build_synthetic_fo_cand_42(base_cand_42: np.ndarray) -> np.ndarray:
    """Transform `cand_42` into an exact zero-history First-Occurrence candidate matrix.

    Preserves the 10 history-independent columns (`CAND_42_FO_VARIABLE_INDICES`), sets the 3 exact
    unseen/never-seen age defaults (`days_since_last=1.0`, `category_age=1.0`, `family_age=1.0`),
    and zeroes all remaining 29 customer-history/repeat/manager/CF columns.
    """
    arr = np.asarray(base_cand_42, dtype=np.float32)
    if arr.shape[-1] != NUM_CAND_42_FEATURES:
        raise ValueError(f"Expected trailing dim {NUM_CAND_42_FEATURES}, got {arr.shape[-1]}")
    out = np.zeros_like(arr, dtype=np.float32)
    out[..., _DAYS_SINCE_LAST_COL] = 1.0
    out[..., _CAT_AGE_COL] = 1.0
    out[..., _FAM_AGE_COL] = 1.0
    out[..., CAND_42_FO_VARIABLE_ARRAY] = arr[..., CAND_42_FO_VARIABLE_ARRAY]
    return out


def assemble_synthetic_returning_cand_42(
    base_cand_42: np.ndarray, ret_cand_27: np.ndarray
) -> np.ndarray:
    """Assemble exact `(..., 42)` candidate matrix for masked-b* synthetic returning rows.

    Repeat columns `0..4` sit at exact unseen defaults `[0, 0, 0, 1, 0]` (never passed to
    `OptionAUnseenHead`, and candidate-wise exact for every unseen item in cross-attention bias).
    The 10 history-independent cols come from `base_cand_42`; the 27 recalculated cols come from
    `ret_cand_27`.
    """
    base = np.asarray(base_cand_42, dtype=np.float32)
    c27 = np.asarray(ret_cand_27, dtype=np.float32)
    if base.shape[:-1] != c27.shape[:-1]:
        raise ValueError(f"Shape mismatch: base_cand_42={base.shape} vs ret_cand_27={c27.shape}")
    if base.shape[-1] != NUM_CAND_42_FEATURES or c27.shape[-1] != NUM_CAND_42_CHANGED_27:
        raise ValueError(
            f"Expected trailing dims ({NUM_CAND_42_FEATURES}, {NUM_CAND_42_CHANGED_27}), "
            f"got ({base.shape[-1]}, {c27.shape[-1]})"
        )
    out = np.zeros_like(base, dtype=np.float32)
    out[..., _DAYS_SINCE_LAST_COL] = 1.0
    out[..., CAND_42_FO_VARIABLE_ARRAY] = base[..., CAND_42_FO_VARIABLE_ARRAY]
    out[..., CAND_42_CHANGED_27_ARRAY] = c27
    return out


def extract_fo_10_from_sources(
    cand_38_slice: np.ndarray, extra_5_slice: np.ndarray, *, dtype: Any = np.float32
) -> np.ndarray:
    """Extract ONLY the 10 history-independent FO_VAR columns directly from `(cand_38, extra_5)`."""
    if cand_38_slice.shape[:-1] != extra_5_slice.shape[:-1]:
        raise ValueError(f"Shape mismatch: cand_38={cand_38_slice.shape} vs extra_5={extra_5_slice.shape}")
    out = np.empty((*cand_38_slice.shape[:-1], len(CAND_42_FO_VARIABLE_INDICES)), dtype=dtype)
    out[..., :9] = cand_38_slice[..., CAND_38_FO_9_ARRAY]
    out[..., 9] = extra_5_slice[..., EXTRA_5_REGION_COL]
    return out


_FO_10_TARGET_CAND42_ARRAY = np.asarray(
    [*CAND_42_FO_FROM_C38_ARRAY.tolist(), CAND_42_REGION_COL], dtype=np.int64
)


def build_synthetic_fo_cand_42_from_fo10(fo_10: np.ndarray, *, dtype: Any = np.float32) -> np.ndarray:
    """Build exact `(..., 42)` FO candidate matrix directly from extracted 10 FO_VAR columns."""
    arr = np.asarray(fo_10, dtype=dtype)
    if arr.shape[-1] != 10:
        raise ValueError(f"Expected trailing dim 10, got {arr.shape[-1]}")
    out = np.zeros((*arr.shape[:-1], NUM_CAND_42_FEATURES), dtype=dtype)
    out[..., _DAYS_SINCE_LAST_COL] = 1.0
    out[..., _CAT_AGE_COL] = 1.0
    out[..., _FAM_AGE_COL] = 1.0
    out[..., _FO_10_TARGET_CAND42_ARRAY] = arr
    return out


def assemble_synthetic_returning_cand_42_from_fo10(
    fo_10: np.ndarray, ret_cand_27: np.ndarray, *, dtype: Any = np.float32
) -> np.ndarray:
    """Assemble exact `(..., 42)` masked-b* returning matrix from `(fo_10, ret_cand_27)` without full `cand_42`."""
    f10 = np.asarray(fo_10, dtype=dtype)
    c27 = np.asarray(ret_cand_27, dtype=dtype)
    if f10.shape[:-1] != c27.shape[:-1]:
        raise ValueError(f"Shape mismatch: fo_10={f10.shape} vs ret_cand_27={c27.shape}")
    if f10.shape[-1] != 10 or c27.shape[-1] != NUM_CAND_42_CHANGED_27:
        raise ValueError(
            f"Expected trailing dims (10, {NUM_CAND_42_CHANGED_27}), got ({f10.shape[-1]}, {c27.shape[-1]})"
        )
    out = np.zeros((*f10.shape[:-1], NUM_CAND_42_FEATURES), dtype=dtype)
    out[..., _DAYS_SINCE_LAST_COL] = 1.0
    out[..., _FO_10_TARGET_CAND42_ARRAY] = f10
    out[..., CAND_42_CHANGED_27_ARRAY] = c27
    return out


def plan_synthetic_unseen_assignments(
    tr_meta: Mapping[str, Any],
    *,
    returning_fraction: float = SYNTHETIC_RETURNING_FRACTION,
    seed: int = SEED,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
) -> Dict[str, np.ndarray]:
    """Deterministically split `< train_cutoff` repeat-positive training baskets into 70% Ret / 30% FO."""
    if not (0.0 <= float(returning_fraction) <= 1.0):
        raise ValueError(f"returning_fraction must be in [0, 1], got {returning_fraction}")

    cutoff_ts = pd.Timestamp(train_cutoff).normalize()
    cutoff_day = int((cutoff_ts - PURCHASE_DATE_ORIGIN).days)
    pred_days = tr_meta["prediction_days"].cpu().numpy()
    if not np.isfinite(pred_days).all() or not np.equal(pred_days, np.floor(pred_days)).all():
        raise ValueError("Training prediction days must be finite integer calendar days.")
    pred_days_i = pred_days.astype(np.int64)
    if len(pred_days_i) and int(pred_days_i.max()) >= cutoff_day:
        raise ValueError(
            f"Synthetic unseen augmentation is strictly train-only (< {cutoff_ts.date()})."
        )

    allowed = tr_meta["allowed_bn"].bool().cpu().numpy()
    unseen = tr_meta["unseen_bn"].bool().cpu().numpy()
    target = tr_meta["target_bn"].bool().cpu().numpy()
    rep_pos = target & (~unseen) & allowed

    elig_rows = np.where(rep_pos.any(axis=1))[0].astype(np.int64)
    if len(elig_rows) == 0:
        return {
            "ret_source_rows": np.empty(0, dtype=np.int64),
            "ret_masked_bn_idx": np.empty(0, dtype=np.int64),
            "fo_source_rows": np.empty(0, dtype=np.int64),
        }

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(elig_rows)
    n_ret = int(round(float(returning_fraction) * len(elig_rows)))
    ret_rows_raw = perm[:n_ret]
    fo_rows_raw = perm[n_ret:]

    # Pick 1 repeat positive b* per returning synthetic row deterministically in sorted source-row order
    ret_order = np.lexsort((ret_rows_raw, pred_days_i[ret_rows_raw]))
    ret_source_rows = ret_rows_raw[ret_order].astype(np.int64)
    ret_masked_bn_idx = np.empty(len(ret_source_rows), dtype=np.int64)
    for idx_i, src_row in enumerate(ret_source_rows.tolist()):
        pos_bns = np.where(rep_pos[src_row])[0]
        pick = int(rng.integers(0, len(pos_bns)))
        ret_masked_bn_idx[idx_i] = int(pos_bns[pick])

    fo_order = np.lexsort((fo_rows_raw, pred_days_i[fo_rows_raw]))
    fo_source_rows = fo_rows_raw[fo_order].astype(np.int64)

    return {
        "ret_source_rows": ret_source_rows,
        "ret_masked_bn_idx": ret_masked_bn_idx,
        "fo_source_rows": fo_source_rows,
    }


def _aggregate_order_slice(
    d_slice: np.ndarray,
    cash_slice: np.ndarray,
    s_slice: np.ndarray,
    m_slice: np.ndarray,
) -> Tuple[float, float, np.datetime64, str]:
    """Aggregate one order-date's kept commercial lines into `(deferral, is_prepaid, settle_date, last_mgr)`."""
    valid_d = d_slice[np.isfinite(d_slice)]
    dd = float(np.median(valid_d)) if len(valid_d) else float("nan")
    is_prep = 1.0 if bool(cash_slice.any() or (not math.isnan(dd) and dd <= 0.0)) else 0.0
    # Strict basket settlement: settled iff ALL kept lines on this order date have non-NaT settlement < cutoff.
    s_max = np.datetime64("NaT") if bool(np.isnat(s_slice).any()) else s_slice.max()
    mgr = ""
    for m_val in m_slice:
        if m_val and m_val != "<NA>":
            mgr = str(m_val)
    return dd, is_prep, s_max, mgr


def _index_customer_commercial_lines(
    commercial_events: pd.DataFrame,
    all_pid_to_bn: Mapping[str, str],
    *,
    train_cutoff: pd.Timestamp,
) -> Dict[str, Dict[str, Any]]:
    """Pre-index per-customer raw commercial lines & per-order-date summaries (strictly `< train_cutoff`)."""
    cutoff_ts = pd.Timestamp(train_cutoff).normalize()
    if commercial_events is None or commercial_events.empty:
        return {}
    df = commercial_events.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
    if bool(df["purchase_date"].isna().any()) or not bool(df["purchase_date"].lt(cutoff_ts).all()):
        raise ValueError(f"Training commercial events only (< {cutoff_ts.date()}).")
    if "full_payment_date" in df.columns:
        df["full_payment_date"] = pd.to_datetime(df["full_payment_date"], errors="coerce").dt.normalize()
        # Strict pre-cutoff clamp on settlement dates during training augmentation
        df.loc[df["full_payment_date"] >= cutoff_ts, "full_payment_date"] = pd.NaT
    else:
        df["full_payment_date"] = pd.NaT

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["bn"] = df["product_id"].map(all_pid_to_bn).fillna("")
    df["deferral_days"] = (
        df["payment_terms"].map(parse_deferral_days) if "payment_terms" in df.columns else np.nan
    )
    df["is_cash"] = (
        df["payment_type"].fillna("").astype(str).str.strip().str.upper().eq("НАЛ")
        if "payment_type" in df.columns
        else False
    )
    df["manager_id"] = (
        df["manager_id"].fillna("").astype(str).str.strip() if "manager_id" in df.columns else ""
    )
    df = df.sort_values(["purchase_date", "customer_id", "product_id"], kind="stable")

    indexed: Dict[str, Dict[str, Any]] = {}
    for cid, grp in df.groupby("customer_id", sort=False):
        dates = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
        bns = grp["bn"].to_numpy(dtype=object)
        defs = grp["deferral_days"].to_numpy(dtype=np.float64)
        cash = grp["is_cash"].to_numpy(dtype=bool)
        settles = grp["full_payment_date"].to_numpy(dtype="datetime64[ns]")
        mgrs = grp["manager_id"].to_numpy(dtype=object)

        n_lines = len(dates)
        change = np.empty(n_lines, dtype=bool)
        change[0] = True
        if n_lines > 1:
            change[1:] = dates[1:] != dates[:-1]
        starts = np.flatnonzero(change)
        ends = np.empty(len(starts), dtype=np.int64)
        ends[:-1] = starts[1:]
        ends[-1] = n_lines

        n_ord = len(starts)
        ord_dates = dates[starts]
        ord_def = np.empty(n_ord, dtype=np.float64)
        ord_prep = np.empty(n_ord, dtype=np.float64)
        ord_settle = np.empty(n_ord, dtype="datetime64[ns]")
        ord_mgr = np.empty(n_ord, dtype=object)
        bn_to_orders: Dict[str, List[int]] = defaultdict(list)

        for k in range(n_ord):
            s0, s1 = int(starts[k]), int(ends[k])
            dd, is_prep, s_max, mgr_k = _aggregate_order_slice(
                defs[s0:s1], cash[s0:s1], settles[s0:s1], mgrs[s0:s1]
            )
            ord_def[k] = dd
            ord_prep[k] = is_prep
            ord_settle[k] = s_max
            ord_mgr[k] = mgr_k
            seen_bns: set[str] = set()
            for bn_v in bns[s0:s1]:
                if bn_v and bn_v not in seen_bns:
                    seen_bns.add(bn_v)
                    bn_to_orders[bn_v].append(k)

        indexed[str(cid)] = {
            "dates": dates,
            "bn": bns,
            "deferral_days": defs,
            "is_cash": cash,
            "settle_dates": settles,
            "manager_id": mgrs,
            "ord_dates": ord_dates,
            "ord_starts": starts,
            "ord_ends": ends,
            "ord_def": ord_def,
            "ord_prep": ord_prep,
            "ord_settle": ord_settle,
            "ord_mgr": ord_mgr,
            "bn_orders": {bn_k: np.asarray(ks, dtype=np.int64) for bn_k, ks in bn_to_orders.items()},
        }
    return indexed


def _recompute_commercial_cust_and_mgr(
    lines: Optional[Mapping[str, Any]],
    date_ts: pd.Timestamp,
    masked_bn_name: str,
) -> Tuple[float, float, float, float, str]:
    """Exactly recompute `(prepaid_ratio, avg_def, last_def, overdue_unpaid, curr_mgr)` excluding `masked_bn_name`.

    Matches `CommercialRollingFeatureBuilder` order-date aggregation line-by-line after removing
    all commercial lines whose `base_name == masked_bn_name`, only re-aggregating order dates that
    actually contained `masked_bn_name`.
    """
    if lines is None:
        return 0.0, 0.0, 0.0, 0.0, ""
    ord_dates_all = lines["ord_dates"]
    cutoff_64 = date_ts.to_datetime64()
    stop_ord = int(ord_dates_all.searchsorted(cutoff_64, side="left"))
    if stop_ord == 0:
        return 0.0, 0.0, 0.0, 0.0, ""

    affected_all = lines["bn_orders"].get(masked_bn_name)
    if affected_all is None:
        affected_stop = 0
    else:
        affected_stop = int(affected_all.searchsorted(stop_ord, side="left"))

    if affected_stop == 0:
        ord_dates = ord_dates_all[:stop_ord]
        def_sl = lines["ord_def"][:stop_ord]
        prep_sl = lines["ord_prep"][:stop_ord]
        settle_sl = lines["ord_settle"][:stop_ord]
        mgr_sl = lines["ord_mgr"][:stop_ord]
    else:
        keep_ord = np.ones(stop_ord, dtype=bool)
        def_buf = lines["ord_def"][:stop_ord].copy()
        prep_buf = lines["ord_prep"][:stop_ord].copy()
        settle_buf = lines["ord_settle"][:stop_ord].copy()
        mgr_buf = lines["ord_mgr"][:stop_ord].copy()

        bns = lines["bn"]
        defs = lines["deferral_days"]
        cash = lines["is_cash"]
        settles = lines["settle_dates"]
        mgrs = lines["manager_id"]
        starts = lines["ord_starts"]
        ends = lines["ord_ends"]

        for k in affected_all[:affected_stop].tolist():
            s0, s1 = int(starts[k]), int(ends[k])
            keep_l = bns[s0:s1] != masked_bn_name
            if not bool(keep_l.any()):
                keep_ord[k] = False
                continue
            dd, is_prep, s_max, mgr_k = _aggregate_order_slice(
                defs[s0:s1][keep_l],
                cash[s0:s1][keep_l],
                settles[s0:s1][keep_l],
                mgrs[s0:s1][keep_l],
            )
            def_buf[k] = dd
            prep_buf[k] = is_prep
            settle_buf[k] = s_max
            mgr_buf[k] = mgr_k

        if not bool(keep_ord.any()):
            return 0.0, 0.0, 0.0, 0.0, ""

        ord_dates = ord_dates_all[:stop_ord][keep_ord]
        def_sl = def_buf[keep_ord]
        prep_sl = prep_buf[keep_ord]
        settle_sl = settle_buf[keep_ord]
        mgr_sl = mgr_buf[keep_ord]

    curr_mgr = ""
    for k_idx in range(len(mgr_sl) - 1, -1, -1):
        m_v = mgr_sl[k_idx]
        if m_v:
            curr_mgr = str(m_v)
            break

    prepaid_ratio = float(prep_sl.mean())
    valid_def = def_sl[np.isfinite(def_sl)]
    avg_def = float(np.clip(valid_def.mean() / 90.0, 0.0, 1.0)) if len(valid_def) else 0.0
    last_def = float(np.clip(valid_def[-1] / 90.0, 0.0, 1.0)) if len(valid_def) else 0.0

    overdue_ratio = 0.0
    known_def = np.isfinite(def_sl)
    if bool(known_def.any()):
        is_settled_before_t = (~np.isnat(settle_sl)) & (settle_sl < cutoff_64)
        due_delta = np.rint(def_sl[known_def]).astype(np.int64) * np.timedelta64(1, "D")
        due_dates = ord_dates[known_def] + due_delta
        unpaid_at_t = ~is_settled_before_t[known_def]
        overdue_ratio = float(((due_dates < cutoff_64) & unpaid_at_t).mean())

    return prepaid_ratio, avg_def, last_def, overdue_ratio, curr_mgr


def _rebuild_sequence_from_events(
    remaining_evs: Sequence[Tuple[Any, ...]],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild `(sequence [S,9], attention_mask [S], seq_product_ids [S], seq_purchase_dates [S])`.

    Exact match to `build_base_name_sequences` (with cols 3:5 zeroed matching `prepare_v2_full.py`)
    using collapsed `(ts, bid, bn_s, qty, ...)` events from `NativeBaseNameFeatureEngine._cust_events`.
    """
    sequence = np.zeros((seq_len, 9), dtype=np.float32)
    attn_mask = np.zeros(seq_len, dtype=np.int64)
    seq_bn_ids = np.zeros(seq_len, dtype=np.int64)
    seq_dates = np.zeros(seq_len, dtype=np.float32)
    stop = len(remaining_evs)
    if stop == 0:
        return sequence, attn_mask, seq_bn_ids, seq_dates

    log60 = math.log1p(60.0)
    log50 = math.log1p(50.0)
    origin = PURCHASE_DATE_ORIGIN
    start_keep = max(0, stop - seq_len)

    bn_counts: Dict[str, int] = {}
    bn_vols: Dict[str, float] = {}
    cum_vol = 0.0
    prev_day = int((remaining_evs[0][0] - origin).days)
    last_day = int((remaining_evs[-1][0] - origin).days)

    for idx_e, ev in enumerate(remaining_evs):
        ts, b_id, bn_s, q_v = ev[0], int(ev[1]), ev[2], float(ev[3])
        d_i = int((ts - origin).days)
        prev_cnt = bn_counts.get(bn_s, 0)
        bn_counts[bn_s] = prev_cnt + 1
        bn_vols[bn_s] = bn_vols.get(bn_s, 0.0) + q_v
        cum_vol += q_v
        gap_days = max(0, d_i - prev_day) if idx_e > 0 else 0
        prev_day = d_i

        if idx_e >= start_keep:
            slot = idx_e - start_keep
            attn_mask[slot] = 1
            seq_bn_ids[slot] = b_id
            seq_dates[slot] = float(d_i) / 365.25
            age_rel_last = max(0, last_day - d_i)
            sequence[slot, 0] = min(1.0, math.log1p(q_v) / log60)
            sequence[slot, 1] = float(age_rel_last) / (float(age_rel_last) + 90.0)
            sequence[slot, 2] = min(1.0, float(gap_days) / 365.0)
            # cols 3:5 (month_sin/cos) stay 0.0 matching prepare_v2_full.py
            sequence[slot, 5] = min(1.0, math.log1p(prev_cnt) / log50)
            sequence[slot, 6] = min(1.0, bn_vols[bn_s] / max(cum_vol, 1e-6))
            sequence[slot, 7] = 1.0 if prev_cnt > 0 else 0.0
            sequence[slot, 8] = 1.0

    return sequence, attn_mask, seq_bn_ids, seq_dates


def compute_synthetic_returning_features(
    *,
    purchases: pd.DataFrame,
    commercial_events: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    reps: Sequence[int],
    allowed_train_pids: set[str],
    tr_meta: Mapping[str, Any],
    ret_source_rows: np.ndarray,
    ret_masked_bn_idx: np.ndarray,
    customer_region_map: Mapping[str, int],
    customer_industry_map: Mapping[str, int],
    seq_len: int = 32,
    out_cand27_path: Optional[Path] = None,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
    test_start: pd.Timestamp = TEST_START_V2,
) -> Dict[str, Any]:
    """Compute exact recalculated `(cand_27, CUST_7, SEQ, masks)` for all 70% masked-b* returning rows.

    Strictly enforces `< train_cutoff` on every purchase, commercial event, and query date.
    """
    cutoff_ts = pd.Timestamp(train_cutoff).normalize()
    test_start_ts = pd.Timestamp(test_start).normalize()
    if cutoff_ts > test_start_ts:
        raise ValueError("train_cutoff must be <= test_start.")
    cutoff_day = int((cutoff_ts - PURCHASE_DATE_ORIGIN).days)

    if len(ret_source_rows) != len(ret_masked_bn_idx):
        raise ValueError("ret_source_rows and ret_masked_bn_idx must have equal length.")

    # Strict < train_cutoff validation on raw inputs
    if not purchases.empty:
        p_dates = pd.to_datetime(purchases["purchase_date"], format="mixed").dt.normalize()
        if bool(p_dates.isna().any()) or not bool(p_dates.lt(cutoff_ts).all()):
            raise ValueError(f"Synthetic returning builder accepts strictly train purchases (< {cutoff_ts.date()}).")
    if commercial_events is not None and not commercial_events.empty:
        c_dates = pd.to_datetime(commercial_events["purchase_date"], format="mixed").dt.normalize()
        if bool(c_dates.isna().any()) or not bool(c_dates.lt(cutoff_ts).all()):
            raise ValueError(
                f"Synthetic returning builder accepts strictly train commercial events (< {cutoff_ts.date()})."
            )

    n_ret = len(ret_source_rows)
    n_bn = int(universe["num_base_names"])
    base_names = list(universe["base_names"])
    all_pid_to_bn = universe["all_pid_to_bn"]
    bn_to_idx = universe["bn_to_idx"]

    cids_all = [str(c).strip() for c in tr_meta["customer_ids"]]
    pred_days_all = tr_meta["prediction_days"].cpu().numpy().astype(np.int64)
    inds_all = (
        tr_meta["industry_id"].cpu().numpy().astype(np.int32)
        if "industry_id" in tr_meta
        else np.array([int(customer_industry_map.get(c, 0)) for c in cids_all], dtype=np.int32)
    )

    if n_ret > 0 and int(pred_days_all[ret_source_rows].max()) >= cutoff_day:
        raise ValueError(f"Synthetic returning query >= train_cutoff ({cutoff_ts.date()}).")

    # Allocate output buffers
    if out_cand27_path is not None:
        out_p = Path(out_cand27_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cand27_out: np.ndarray = np.lib.format.open_memmap(
            out_p, mode="w+", dtype=np.float16, shape=(n_ret, n_bn, NUM_CAND_42_CHANGED_27)
        )
    else:
        cand27_out = np.zeros((n_ret, n_bn, NUM_CAND_42_CHANGED_27), dtype=np.float16)

    cust7_out = np.zeros((n_ret, NUM_CUST_7_FEATURES), dtype=np.float32)
    seq_out = np.zeros((n_ret, seq_len, 9), dtype=np.float32)
    attn_mask_out = np.zeros((n_ret, seq_len), dtype=np.int64)
    seq_pid_out = np.zeros((n_ret, seq_len), dtype=np.int64)
    seq_dates_out = np.zeros((n_ret, seq_len), dtype=np.float32)
    unseen_bn_out = np.zeros((n_ret, n_bn), dtype=bool)
    target_bn_out = np.zeros((n_ret, n_bn), dtype=bool)

    if n_ret == 0:
        return {
            "cand_27": cand27_out,
            "cust_7": torch.from_numpy(cust7_out),
            "sequence": torch.from_numpy(seq_out),
            "attention_mask": torch.from_numpy(attn_mask_out),
            "seq_product_ids": torch.from_numpy(seq_pid_out),
            "seq_purchase_dates": torch.from_numpy(seq_dates_out),
            "unseen_bn": torch.from_numpy(unseen_bn_out),
            "target_bn": torch.from_numpy(target_bn_out),
        }

    # Instantiate engine strictly on < train_cutoff (reuses exact taxonomy, relationship tables & 90d manager shares)
    engine = NativeBaseNameFeatureEngine(
        purchases,
        commercial_events,
        catalogue,
        companies,
        product_ids,
        universe,
        reps,
        allowed_member_pids=allowed_train_pids,
        frozen_v5_replay=False,
        test_start=cutoff_ts,
    )
    cust_comm_lines = _index_customer_commercial_lines(
        commercial_events, all_pid_to_bn, train_cutoff=cutoff_ts
    )

    # Index per-(cid, bn_name, manager_id) deduplicated dates matching engine._mgr_bn_events for exact 90d self-subtraction
    cust_bn_mgr90_dates: Dict[Tuple[str, str, str], np.ndarray] = {}
    if commercial_events is not None and not commercial_events.empty and "manager_id" in commercial_events.columns:
        mc90 = commercial_events[["manager_id", "customer_id", "purchase_date", "product_id"]].copy()
        mc90["manager_id"] = mc90["manager_id"].fillna("").astype(str).str.strip()
        mc90["customer_id"] = mc90["customer_id"].astype(str).str.strip()
        mc90["product_id"] = mc90["product_id"].astype(str).str.strip()
        mc90["purchase_date"] = pd.to_datetime(mc90["purchase_date"], format="mixed").dt.normalize()
        mc90["bn"] = mc90["product_id"].map(all_pid_to_bn).fillna("")
        mc90 = mc90.loc[(mc90["manager_id"] != "") & (mc90["bn"] != "")]
        mc90 = mc90.drop_duplicates(["manager_id", "customer_id", "purchase_date", "bn"])
        mc90 = mc90.sort_values(["customer_id", "bn", "manager_id", "purchase_date"], kind="stable")
        for (c_k, b_k, m_k), grp in mc90.groupby(["customer_id", "bn", "manager_id"], sort=False):
            cust_bn_mgr90_dates[(str(c_k), str(b_k), str(m_k))] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")

    # Prepare chronological extra_cand_5 sweep state (exact parity with extra_features.py)
    ev = commercial_events.copy() if commercial_events is not None else pd.DataFrame()
    if not ev.empty:
        ev["purchase_date"] = pd.to_datetime(ev["purchase_date"], format="mixed").dt.normalize()
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
        ev = ev.sort_values(["day", "customer_id", "bn_idx", "manager_id"], kind="mergesort")
        ev = ev.drop_duplicates(subset=["day", "customer_id", "bn_idx"], keep="last").reset_index(drop=True)
        ev_days = ev["day"].to_numpy(dtype=np.int32)
        ev_cids = ev["customer_id"].tolist()
        ev_bns = ev["bn_idx"].to_numpy(dtype=np.int32)
        ev_mgrs = ev["manager_id"].tolist()
        ev_inds = np.array([int(customer_industry_map.get(c, 0)) for c in ev_cids], dtype=np.int32)
    else:
        ev_days = np.empty(0, dtype=np.int32)
        ev_cids = []
        ev_bns = np.empty(0, dtype=np.int32)
        ev_mgrs = []
        ev_inds = np.empty(0, dtype=np.int32)

    mgr_30 = _RollingWindowCounter(30, n_bn)
    mgr_ind_90 = _RollingWindowCounter(90, n_bn)
    cust_bn_mgr_days: Dict[Tuple[str, int], List[Tuple[int, str, int]]] = defaultdict(list)
    cust_seen_bns: Dict[str, List[int]] = {}
    cust_seen_sets: Dict[str, set[int]] = {}
    cust_last_bn_day: Dict[str, Dict[int, int]] = {}
    cooc = np.zeros((n_bn, n_bn), dtype=np.float32)
    deg = np.zeros(n_bn, dtype=np.float32)
    inv_sqrt = np.zeros(n_bn, dtype=np.float32)
    sim_mat = np.zeros((n_bn, n_bn), dtype=np.float32)
    cf_dirty = False
    ev_ptr = 0
    n_events = len(ev_days)
    ln2_over_hl = math.log(2.0) / float(CF_RECENCY_HALF_LIFE_DAYS)

    # Group synthetic returning rows by query day (strictly monotonic chronological sweep)
    queries_by_day: Dict[int, List[int]] = defaultdict(list)
    for syn_i, src_row in enumerate(ret_source_rows.tolist()):
        queries_by_day[int(pred_days_all[src_row])].append(syn_i)
    sorted_days = sorted(queries_by_day.keys())

    # Local slot indices inside cand_27
    s_cat_sh = _C27_SLOT["user_cat_order_share"]
    s_sub_sh = _C27_SLOT["user_subcat_order_share"]
    s_br_sh = _C27_SLOT["user_brand_order_share"]
    s_rep_cat = _C27_SLOT["is_repeat_category"]
    s_rep_br = _C27_SLOT["is_repeat_brand"]
    s_rep_ln = _C27_SLOT["is_repeat_business_line"]
    s_cat_age = _C27_SLOT["category_age"]
    s_cat_ovd = _C27_SLOT["category_overdue_ratio"]
    s_fam_age = _C27_SLOT["family_age"]
    s_fam_ovd = _C27_SLOT["family_overdue_ratio"]
    s_fam_sh = _C27_SLOT["family_history_share"]
    s_bn_last = _C27_SLOT["base_name_last_lift"]
    s_bn_rec3 = _C27_SLOT["base_name_recent3_lift"]
    s_cat_last = _C27_SLOT["category_last_lift"]
    s_fam_last = _C27_SLOT["family_last_lift"]
    s_ln_last = _C27_SLOT["line_last_lift"]
    s_sup = _C27_SLOT["base_name_support_confidence"]
    s_same = _C27_SLOT["base_name_same_basket_lift"]
    s_mgr90_bn = _C27_SLOT["mgr_base_name_share_90d"]
    s_mgr90_br = _C27_SLOT["mgr_brand_share_90d"]
    s_mgr90_cat = _C27_SLOT["mgr_cat_share_90d"]
    s_visc_sh = _C27_SLOT["user_viscosity_order_share"]
    s_app_sh = _C27_SLOT["user_application_order_share"]
    s_mgr30_bn = _C27_SLOT["mgr_base_name_share_30d"]
    s_mgr_ind90 = _C27_SLOT["mgr_industry_base_name_share_90d"]
    s_cf_u = _C27_SLOT["full_cf_cosine"]
    s_cf_r = _C27_SLOT["full_cf_recency_weighted"]

    cat_pos = engine._attr_positions["product_category"]
    sub_pos = engine._attr_positions["product_subcategory"]
    br_pos = engine._attr_positions["brand"]
    ln_pos = engine._attr_positions["business_line"]
    fam_pos = engine._attr_positions["product_family"]
    app_pos = engine._attr_positions["application"]
    visc_matches = engine._viscosity_matches

    tr_unseen_np = tr_meta["unseen_bn"].bool().cpu().numpy()
    tr_target_np = tr_meta["target_bn"].bool().cpu().numpy()
    tr_allowed_np = tr_meta["allowed_bn"].bool().cpu().numpy()

    t0 = time.perf_counter()
    done = 0

    for day_idx, q_day in enumerate(sorted_days):
        date_ts = (PURCHASE_DATE_ORIGIN + pd.Timedelta(days=int(q_day))).normalize()

        # 1. Advance relationship tables strictly < date_ts
        for tbl in (*engine._tables.values(), engine._complements):
            tbl.advance(date_ts)

        # 2. Advance rolling manager windows & incremental CF matrix strictly < q_day
        while ev_ptr < n_events and int(ev_days[ev_ptr]) < q_day:
            d = int(ev_days[ev_ptr])
            cid_e = ev_cids[ev_ptr]
            bn_e = int(ev_bns[ev_ptr])
            mgr_e = ev_mgrs[ev_ptr]
            ind_e = int(ev_inds[ev_ptr])

            if mgr_e:
                mgr_30.add(d, mgr_e, bn_e)
                cust_bn_mgr_days[(cid_e, bn_e)].append((d, mgr_e, ind_e))
                if ind_e > 0:
                    mgr_ind_90.add(d, (mgr_e, ind_e), bn_e)

            last_map = cust_last_bn_day.get(cid_e)
            if last_map is None:
                last_map = {}
                cust_last_bn_day[cid_e] = last_map
            last_map[bn_e] = d

            seen_set = cust_seen_sets.get(cid_e)
            if seen_set is None:
                cust_seen_sets[cid_e] = {bn_e}
                cust_seen_bns[cid_e] = [bn_e]
                deg[bn_e] += 1.0
                cf_dirty = True
            elif bn_e not in seen_set:
                prev_list = cust_seen_bns[cid_e]
                if prev_list:
                    prev_arr = np.asarray(prev_list, dtype=np.int64)
                    cooc[prev_arr, bn_e] += 1.0
                    cooc[bn_e, prev_arr] += 1.0
                seen_set.add(bn_e)
                prev_list.append(bn_e)
                deg[bn_e] += 1.0
                cf_dirty = True
            ev_ptr += 1

        mgr_30.expire_before(q_day - 30)
        mgr_ind_90.expire_before(q_day - 90)
        if cf_dirty:
            inv_sqrt = np.zeros(n_bn, dtype=np.float32)
            nz = deg > 0.0
            inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
            sim_mat = cooc * inv_sqrt[:, None] * inv_sqrt[None, :]
            np.fill_diagonal(sim_mat, 0.0)
            cf_dirty = False

        day_syns = queries_by_day[q_day]
        m = len(day_syns)
        day_c27 = np.zeros((m, n_bn, NUM_CAND_42_CHANGED_27), dtype=np.float32)
        day_c27[:, :, s_cat_age] = 1.0
        day_c27[:, :, s_fam_age] = 1.0

        mgr_30_cache: Dict[str, Optional[np.ndarray]] = {}
        mgr_ind_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
        cf_positions: List[int] = []
        cf_unit_rows: List[np.ndarray] = []
        cf_rec_rows: List[np.ndarray] = []
        cf_self_fix: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []  # (b_star, idx_arr, w_u_nz, w_r_nz)

        d30_ts = date_ts - pd.Timedelta(days=30)
        d90_ts = date_ts - pd.Timedelta(days=90)
        cutoff_64 = date_ts.to_datetime64()
        d90_64 = d90_ts.to_datetime64()

        for local_i, syn_i in enumerate(day_syns):
            src_row = int(ret_source_rows[syn_i])
            b_star = int(ret_masked_bn_idx[syn_i])
            b_star_name = base_names[b_star]
            cid = cids_all[src_row]
            ind = int(inds_all[src_row])

            # Filter customer's collapsed purchase events strictly < date_ts, erasing ALL prior b_star
            all_evs = engine._cust_events.get(cid)
            if all_evs:
                end_ev = bisect_left(engine._cust_event_dates[cid], date_ts)
                rem_evs = [ev_t for ev_t in all_evs[:end_ev] if ev_t[2] != b_star_name]
            else:
                rem_evs = []

            # Rebuild 32-token history sequence strictly from rem_evs
            seq_i, am_i, spid_i, sdt_i = _rebuild_sequence_from_events(rem_evs, seq_len=seq_len)
            seq_out[syn_i] = seq_i
            attn_mask_out[syn_i] = am_i
            seq_pid_out[syn_i] = spid_i
            seq_dates_out[syn_i] = sdt_i

            # Exact customer attribute shares, cadence & relationship lifts strictly from rem_evs
            n_ev = len(rem_evs)
            if n_ev == 0:
                cust7_out[syn_i, 0] = 1.0  # customer_age = 1.0 when zero remaining history
            else:
                inv_ev = 1.0 / float(n_ev)
                cats: Counter[str] = Counter()
                subcats: Counter[str] = Counter()
                brands: Counter[str] = Counter()
                lines_set: set[str] = set()
                viscs: Counter[Tuple[str, ...]] = Counter()
                apps: Counter[str] = Counter()

                # Also reconstruct remaining dated baskets in a single pass
                prior_b: List[Tuple[pd.Timestamp, List[int], set[str], set[str]]] = []
                cur_dt: Optional[pd.Timestamp] = None
                cur_bids: set[int] = set()
                cur_cats: set[str] = set()
                cur_fams: set[str] = set()

                for ev_t in rem_evs:
                    ts, bid, _bn_s, _qty, cat, subcat, brand, line, fam, v_grades, app = ev_t
                    if cat:
                        cats[cat] += 1
                    if subcat:
                        subcats[subcat] += 1
                    if brand:
                        brands[brand] += 1
                    if line:
                        lines_set.add(line)
                    if v_grades:
                        viscs[v_grades] += 1
                    if app and app != "other":
                        apps[app] += 1

                    if cur_dt is None:
                        cur_dt = ts
                    elif ts != cur_dt:
                        prior_b.append((cur_dt, sorted(cur_bids), cur_cats, cur_fams))
                        cur_dt = ts
                        cur_bids = set()
                        cur_cats = set()
                        cur_fams = set()
                    if bid > 0:
                        cur_bids.add(int(bid))
                    if cat:
                        cur_cats.add(cat)
                    if fam:
                        cur_fams.add(fam)
                if cur_dt is not None:
                    prior_b.append((cur_dt, sorted(cur_bids), cur_cats, cur_fams))

                row_buf = day_c27[local_i]
                for v, cnt in cats.items():
                    pos = cat_pos.get(v)
                    if pos is not None:
                        row_buf[pos, s_cat_sh] = min(1.0, cnt * inv_ev)
                        row_buf[pos, s_rep_cat] = 1.0
                for v, cnt in subcats.items():
                    pos = sub_pos.get(v)
                    if pos is not None:
                        row_buf[pos, s_sub_sh] = min(1.0, cnt * inv_ev)
                for v, cnt in brands.items():
                    pos = br_pos.get(v)
                    if pos is not None:
                        row_buf[pos, s_br_sh] = min(1.0, cnt * inv_ev)
                        row_buf[pos, s_rep_br] = 1.0
                for v in lines_set:
                    pos = ln_pos.get(v)
                    if pos is not None:
                        row_buf[pos, s_rep_ln] = 1.0
                matches = np.zeros(n_bn, dtype=np.int64)
                for grades, cnt in viscs.items():
                    pos = visc_matches.get(grades)
                    if pos is not None:
                        matches[pos] += cnt
                row_buf[:, s_visc_sh] = matches * inv_ev
                for v, cnt in apps.items():
                    pos = app_pos.get(v)
                    if pos is not None:
                        row_buf[pos, s_app_sh] = min(1.0, cnt * inv_ev)

                # Customer cadence -> CUST_7[0..2]
                bdates = [d for d, _, _, _ in prior_b]
                age_d = (date_ts - bdates[-1]).days
                cust7_out[syn_i, 0] = min(age_d / 365.0, 1.0)
                cust7_out[syn_i, 1] = float(log_count(sum(d >= d30_ts for d in bdates)))
                cust7_out[syn_i, 2] = float(log_count(sum(d >= d90_ts for d in bdates)))

                # Category & family recency/overdue/share
                cat_hist: Dict[str, List[pd.Timestamp]] = {}
                fam_hist: Dict[str, List[pd.Timestamp]] = {}
                for d, _, cats_d, fams_d in prior_b:
                    for v in cats_d:
                        cat_hist.setdefault(v, []).append(d)
                    for v in fams_d:
                        fam_hist.setdefault(v, []).append(d)
                for v, h in cat_hist.items():
                    pos = cat_pos.get(v)
                    if pos is not None:
                        iv = float(np.median([(b - a).days for a, b in zip(h, h[1:])])) if len(h) > 1 else 30.0
                        el = (date_ts - h[-1]).days
                        row_buf[pos, s_cat_age] = min(el / 365.0, 1.0)
                        row_buf[pos, s_cat_ovd] = min(el / max(iv, 1.0) / 4.0, 1.0)
                inv_nb = 1.0 / float(len(prior_b))
                for v, h in fam_hist.items():
                    pos = fam_pos.get(v)
                    if pos is not None:
                        iv = float(np.median([(b - a).days for a, b in zip(h, h[1:])])) if len(h) > 1 else 30.0
                        el = (date_ts - h[-1]).days
                        row_buf[pos, s_fam_age] = min(el / 365.0, 1.0)
                        row_buf[pos, s_fam_ovd] = min(el / max(iv, 1.0) / 4.0, 1.0)
                        row_buf[pos, s_fam_sh] = len(h) * inv_nb

                # Cross-company relationship lifts strictly from remaining prior_b
                last_bids = prior_b[-1][1]
                bn_last, bn_sup = engine._tables["bn"].score(last_bids)
                rec_num = np.zeros(n_bn + 1, dtype=np.float64)
                w_sum = 0.0
                for idx_r, (_, bids_r, _, _) in enumerate(reversed(prior_b[-3:])):
                    w = 0.5 ** idx_r
                    rec_num += engine._tables["bn"].score(bids_r)[0] * w
                    w_sum += w
                bn_rec3 = rec_num / max(w_sum, 1e-12)
                row_buf[:, s_bn_last] = bn_last[1:].astype(np.float32)
                row_buf[:, s_bn_rec3] = bn_rec3[1:].astype(np.float32)
                row_buf[:, s_cat_last] = engine._tables["product_category"].score(last_bids)[0][1:].astype(np.float32)
                row_buf[:, s_fam_last] = engine._tables["product_family"].score(last_bids)[0][1:].astype(np.float32)
                row_buf[:, s_ln_last] = engine._tables["business_line"].score(last_bids)[0][1:].astype(np.float32)
                row_buf[:, s_sup] = bn_sup[1:].astype(np.float32)
                row_buf[:, s_same] = engine._complements.score(last_bids)[0][1:].astype(np.float32)

            # Commercial financial CUST_7[3..6] & remaining active manager excluding b_star lines
            prep_r, avg_d, last_d, ovd_r, curr_mgr = _recompute_commercial_cust_and_mgr(
                cust_comm_lines.get(cid), date_ts, b_star_name
            )
            cust7_out[syn_i, 3] = prep_r
            cust7_out[syn_i, 4] = avg_d
            cust7_out[syn_i, 5] = last_d
            cust7_out[syn_i, 6] = ovd_r

            # Manager rolling shares (cols 31..33, 37, 38) for remaining curr_mgr with exact self-subtraction of cid's b_star
            if curr_mgr:
                m90 = engine._manager_bn_shares_90d(curr_mgr, date_ts)
                own90_dts = cust_bn_mgr90_dates.get((cid, b_star_name, curr_mgr))
                own90_m = 0
                if own90_dts is not None:
                    own90_m = int(own90_dts.searchsorted(cutoff_64, side="left")) - int(
                        own90_dts.searchsorted(d90_64, side="left")
                    )
                if own90_m > 0 and curr_mgr in engine._mgr_bn_events:
                    m_dts = engine._mgr_bn_events[curr_mgr]["dates"]
                    tot90_m = int(m_dts.searchsorted(cutoff_64, side="left")) - int(
                        m_dts.searchsorted(d90_64, side="left")
                    )
                    if tot90_m > own90_m:
                        cnt90 = m90.astype(np.float64) * float(tot90_m)
                        cnt90[b_star, 0] = max(0.0, cnt90[b_star, 0] - float(own90_m))
                        b_brand = engine._bn_attr["brand"][b_star]
                        if b_brand:
                            bp = br_pos.get(b_brand)
                            if bp is not None:
                                cnt90[bp, 1] = np.maximum(0.0, cnt90[bp, 1] - float(own90_m))
                        b_cat = engine._bn_attr["product_category"][b_star]
                        if b_cat:
                            cp = cat_pos.get(b_cat)
                            if cp is not None:
                                cnt90[cp, 2] = np.maximum(0.0, cnt90[cp, 2] - float(own90_m))
                        m90_clean = (cnt90 / float(tot90_m - own90_m)).astype(np.float32)
                        day_c27[local_i, :, s_mgr90_bn] = m90_clean[:, 0]
                        day_c27[local_i, :, s_mgr90_br] = m90_clean[:, 1]
                        day_c27[local_i, :, s_mgr90_cat] = m90_clean[:, 2]
                else:
                    day_c27[local_i, :, s_mgr90_bn] = m90[:, 0]
                    day_c27[local_i, :, s_mgr90_br] = m90[:, 1]
                    day_c27[local_i, :, s_mgr90_cat] = m90[:, 2]

                own_ev = cust_bn_mgr_days.get((cid, b_star))
                own30 = 0
                own90_ind = 0
                if own_ev:
                    d30_lim = q_day - 30
                    d90_lim = q_day - 90
                    for d_o, m_o, ind_o in own_ev:
                        if m_o == curr_mgr:
                            if d_o >= d30_lim:
                                own30 += 1
                            if ind > 0 and ind_o == ind and d_o >= d90_lim:
                                own90_ind += 1

                if own30 > 0:
                    tot30 = mgr_30._totals.get(curr_mgr, 0)
                    if tot30 > own30:
                        cnt30 = mgr_30._counts[curr_mgr].astype(np.float32)
                        cnt30[b_star] = max(0.0, cnt30[b_star] - float(own30))
                        day_c27[local_i, :, s_mgr30_bn] = cnt30 / float(tot30 - own30)
                else:
                    if curr_mgr not in mgr_30_cache:
                        v30, _ = mgr_30.share_vector(curr_mgr)
                        mgr_30_cache[curr_mgr] = v30
                    v30 = mgr_30_cache[curr_mgr]
                    if v30 is not None:
                        day_c27[local_i, :, s_mgr30_bn] = v30

                if ind > 0:
                    mi_key = (curr_mgr, ind)
                    if own90_ind > 0:
                        totmi = mgr_ind_90._totals.get(mi_key, 0)
                        if totmi > own90_ind:
                            cntmi = mgr_ind_90._counts[mi_key].astype(np.float32)
                            cntmi[b_star] = max(0.0, cntmi[b_star] - float(own90_ind))
                            day_c27[local_i, :, s_mgr_ind90] = cntmi / float(totmi - own90_ind)
                    else:
                        if mi_key not in mgr_ind_cache:
                            vmi, _ = mgr_ind_90.share_vector(mi_key)
                            mgr_ind_cache[mi_key] = vmi
                        vmi = mgr_ind_cache[mi_key]
                        if vmi is not None:
                            day_c27[local_i, :, s_mgr_ind90] = vmi

            # Exact CF history weights excluding b_star (plus exact self-subtraction of cid's cooc/deg for b_star)
            seen_list = cust_seen_bns.get(cid)
            if seen_list:
                rem_seen = [b for b in seen_list if b != b_star]
                if rem_seen:
                    idx_arr = np.asarray(rem_seen, dtype=np.int64)
                    w_u_nz = np.full(len(rem_seen), 1.0 / math.sqrt(float(len(rem_seen))), dtype=np.float32)
                    w_unit = np.zeros(n_bn, dtype=np.float32)
                    w_unit[idx_arr] = w_u_nz

                    last_map = cust_last_bn_day[cid]
                    ages = np.array([max(0, q_day - last_map[b]) for b in rem_seen], dtype=np.float32)
                    raw_w = np.exp(-ln2_over_hl * ages)
                    norm_w = float(np.linalg.norm(raw_w))
                    w_r_nz = (raw_w / norm_w).astype(np.float32) if norm_w > 0.0 else np.zeros(len(rem_seen), dtype=np.float32)
                    w_rec = np.zeros(n_bn, dtype=np.float32)
                    if norm_w > 0.0:
                        w_rec[idx_arr] = w_r_nz

                    had_b_star = b_star in cust_seen_sets[cid]
                    cf_positions.append(local_i)
                    cf_unit_rows.append(w_unit)
                    cf_rec_rows.append(w_rec)
                    cf_self_fix.append((b_star if had_b_star else -1, idx_arr, w_u_nz, w_r_nz))

            # Synthetic unseen mask & target (b_star becomes unseen positive; other prior buys stay excluded)
            u_mask = tr_unseen_np[src_row].copy()
            u_mask[b_star] = True
            unseen_bn_out[syn_i] = u_mask
            target_bn_out[syn_i] = tr_target_np[src_row] & u_mask & tr_allowed_np[src_row]

        if cf_positions:
            w_u_mat = np.stack(cf_unit_rows, axis=0)
            w_r_mat = np.stack(cf_rec_rows, axis=0)
            raw_u = np.maximum(w_u_mat @ sim_mat, 0.0)
            raw_r = np.maximum(w_r_mat @ sim_mat, 0.0)
            # Remove cid's own (+1) self-contribution to cooc[rem_seen, b_star] and deg[b_star]
            for k, (bs_fix, idx_arr, w_u_nz, w_r_nz) in enumerate(cf_self_fix):
                if bs_fix < 0:
                    continue
                deg_clean = float(deg[bs_fix]) - 1.0
                if deg_clean <= 0.0:
                    raw_u[k, bs_fix] = 0.0
                    raw_r[k, bs_fix] = 0.0
                else:
                    cooc_clean = np.maximum(cooc[idx_arr, bs_fix] - 1.0, 0.0)
                    sim_clean = cooc_clean * inv_sqrt[idx_arr] / math.sqrt(deg_clean)
                    raw_u[k, bs_fix] = max(0.0, float(np.dot(w_u_nz, sim_clean)))
                    raw_r[k, bs_fix] = max(0.0, float(np.dot(w_r_nz, sim_clean)))
            cf_u = raw_u / (raw_u + 1.0)
            cf_r = raw_r / (raw_r + 1.0)
            for k, pos in enumerate(cf_positions):
                day_c27[pos, :, s_cf_u] = cf_u[k]
                day_c27[pos, :, s_cf_r] = cf_r[k]

        day_f16 = np.nan_to_num(day_c27, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float16)
        if m > 0 and day_syns[-1] - day_syns[0] + 1 == m and day_syns == list(range(day_syns[0], day_syns[-1] + 1)):
            cand27_out[day_syns[0] : day_syns[-1] + 1] = day_f16
        else:
            for local_i, syn_i in enumerate(day_syns):
                cand27_out[syn_i] = day_f16[local_i]

        done += m
        if (day_idx + 1) % 200 == 0 or (day_idx + 1) == len(sorted_days):
            print(
                f"[v2-syn] day {day_idx + 1}/{len(sorted_days)} "
                f"ret_masked={done:,}/{n_ret:,} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if isinstance(cand27_out, np.memmap):
        cand27_out.flush()
    cust7_out = np.nan_to_num(cust7_out, nan=0.0, posinf=1.0, neginf=0.0)
    return {
        "cand_27": cand27_out,
        "cust_7": torch.from_numpy(cust7_out),
        "sequence": torch.from_numpy(seq_out),
        "attention_mask": torch.from_numpy(attn_mask_out),
        "seq_product_ids": torch.from_numpy(seq_pid_out),
        "seq_purchase_dates": torch.from_numpy(seq_dates_out),
        "unseen_bn": torch.from_numpy(unseen_bn_out),
        "target_bn": torch.from_numpy(target_bn_out),
    }


def _synthetic_unseen_identity(
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    region_map: Mapping[str, int],
    industry_map: Mapping[str, int],
    commercial_sha256: str,
    purchases_sha256: str,
    catalogue_sha256: str,
    returning_fraction: float,
    seed: int,
    seq_len: int,
    train_cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
) -> str:
    return json_digest(
        {
            "version": SYNTHETIC_UNSEEN_VERSION,
            "returning_fraction": float(returning_fraction),
            "seed": int(seed),
            "seq_len": int(seq_len),
            "changed_27_names": list(CAND_42_CHANGED_27_NAMES),
            "fo_variable_names": list(CAND_42_FO_VARIABLE_NAMES),
            "cust_7_names": list(CUST_7_FEATURE_NAMES),
            "base_names": list(shared_meta["universe"]["base_names"]),
            "sku_to_formulation": shared_meta["universe"]["all_pid_to_bn"],
            "formulation_indices": shared_meta["universe"]["bn_to_idx"],
            "customer_regions": region_map,
            "customer_industries": industry_map,
            "commercial_sha256": str(commercial_sha256),
            "purchases_sha256": str(purchases_sha256),
            "catalogue_sha256": str(catalogue_sha256),
            "train_cutoff": str(pd.Timestamp(train_cutoff).normalize()),
            "test_start": str(pd.Timestamp(test_start).normalize()),
            "train_split": _split_identity_digest(tr_meta),
        }
    )


def build_or_load_synthetic_unseen_cache(
    *,
    prepared_root: Path = PREPARED_ROOT,
    commercial_parquet_path: Path = COMMERCIAL_DEV_PARQUET_PATH,
    companies: Optional[pd.DataFrame] = None,
    catalogue_path: Path = RAW_ITEMS_PATH,
    catalogue: Optional[pd.DataFrame] = None,
    dev_purchases: Optional[pd.DataFrame] = None,
    returning_fraction: float = SYNTHETIC_RETURNING_FRACTION,
    seed: int = SEED,
    seq_len: int = 32,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
    test_start: pd.Timestamp = TEST_START_V2,
    force: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Build or validate sealed Wave 2C 70/30 synthetic unseen cache (strictly `< train_cutoff`)."""
    prepared_root = Path(prepared_root)
    catalogue_path = Path(catalogue_path)
    shared_meta, tr_meta, va_meta = load_prepared_metadata(
        prepared_root, train_cutoff=train_cutoff, test_start=test_start
    )
    if dev_purchases is None:
        dev_purchases = load_v2_development_purchases()
    cutoff_ts = pd.Timestamp(train_cutoff).normalize()
    test_start_ts = pd.Timestamp(test_start).normalize()

    if companies is None and RAW_CUSTOMERS_PATH.is_file():
        companies = load_enriched_companies(RAW_CUSTOMERS_PATH, purchases=dev_purchases, minimum_purchases=0)
    reg_map, ind_map = _build_customer_region_industry_maps(
        tr_meta, va_meta, companies=companies, vocabs=shared_meta.get("vocabs")
    )

    train_purchases = dev_purchases.loc[dev_purchases["purchase_date"] < cutoff_ts].copy()
    purch_digest = input_digest(train_purchases)
    comm_sha = file_digest(commercial_parquet_path)
    if catalogue is not None:
        cat_sha = input_digest(catalogue)
    else:
        cat_sha = file_digest(catalogue_path) if catalogue_path.is_file() else ""
    identity = _synthetic_unseen_identity(
        shared_meta=shared_meta,
        tr_meta=tr_meta,
        region_map=reg_map,
        industry_map=ind_map,
        commercial_sha256=comm_sha,
        purchases_sha256=purch_digest,
        catalogue_sha256=cat_sha,
        returning_fraction=returning_fraction,
        seed=seed,
        seq_len=seq_len,
        train_cutoff=cutoff_ts,
        test_start=test_start_ts,
    )

    cache_dir = prepared_root / "synthetic_unseen_w2c" / identity
    cache_dir.mkdir(parents=True, exist_ok=True)
    cand27_path = cache_dir / "syn_ret_cand27.npy"
    meta_path = cache_dir / "syn_meta.pt"
    manifest_path = cache_dir / "synthetic_unseen_manifest.json"

    if not force and manifest_path.is_file() and cand27_path.is_file() and meta_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("identity") == identity
            and manifest.get("output_sha256", {}).get("cand27") == file_digest(cand27_path)
            and manifest.get("output_sha256", {}).get("meta") == file_digest(meta_path)
        ):
            print(f"[v2-syn] Reusing verified synthetic unseen cache -> {manifest_path}", flush=True)
            return cand27_path, meta_path, manifest

    t0 = time.perf_counter()
    date_col = ds.field("purchase_date")
    if ds.dataset(commercial_parquet_path, format="parquet").count_rows(
        filter=date_col.is_null() | (date_col >= test_start_ts)
    ):
        raise ValueError(f"Development events only (< {test_start_ts.date()}).")
    comm_dev = pd.read_parquet(commercial_parquet_path, filters=[("purchase_date", "<", test_start_ts)])
    # Verify commercial paid lines align strictly with cleaned development purchases,
    # while keeping line-level settlement dates intact prior to masking b* (per-order NaT
    # checks happen strictly AFTER erasing b* inside `_recompute_commercial_cust_and_mgr`).
    aligned_commercial_events(comm_dev, dev_purchases, test_start=test_start_ts)
    paid_dev = comm_dev.loc[actual_purchase_mask(comm_dev)].copy()
    paid_dev["purchase_date"] = pd.to_datetime(paid_dev["purchase_date"], format="mixed").dt.normalize()
    comm_train = paid_dev.loc[paid_dev["purchase_date"] < cutoff_ts].copy()

    if catalogue is None:
        catalogue = load_enriched_catalogue(catalogue_path, descriptions_path=None)
    pool = RollingCandidatePool(catalogue, train_purchases)
    allowed_train_pids = observed_pack_ids(pool.first_sales, cutoff_ts)
    assert allowed_train_pids is not None

    plan = plan_synthetic_unseen_assignments(
        tr_meta, returning_fraction=returning_fraction, seed=seed, train_cutoff=cutoff_ts
    )

    with TemporaryDirectory(dir=cache_dir) as scratch:
        c27_tmp = Path(scratch) / cand27_path.name
        meta_tmp = Path(scratch) / meta_path.name
        built = compute_synthetic_returning_features(
            purchases=train_purchases,
            commercial_events=comm_train,
            catalogue=catalogue,
            companies=companies if companies is not None else pd.DataFrame(),
            product_ids=shared_meta["product_ids"],
            universe=shared_meta["universe"],
            reps=shared_meta["reps"],
            allowed_train_pids=allowed_train_pids,
            tr_meta=tr_meta,
            ret_source_rows=plan["ret_source_rows"],
            ret_masked_bn_idx=plan["ret_masked_bn_idx"],
            customer_region_map=reg_map,
            customer_industry_map=ind_map,
            seq_len=seq_len,
            out_cand27_path=c27_tmp,
            train_cutoff=cutoff_ts,
            test_start=test_start_ts,
        )
        del built["cand_27"]  # flush/close memmap cleanly before atomic move

        syn_payload = {
            "version": SYNTHETIC_UNSEEN_VERSION,
            "identity": identity,
            "returning_fraction": float(returning_fraction),
            "seed": int(seed),
            "seq_len": int(seq_len),
            "ret_source_rows": torch.from_numpy(plan["ret_source_rows"]),
            "ret_masked_bn_idx": torch.from_numpy(plan["ret_masked_bn_idx"]),
            "fo_source_rows": torch.from_numpy(plan["fo_source_rows"]),
            "ret_cust_7": built["cust_7"],
            "ret_sequence": built["sequence"],
            "ret_attention_mask": built["attention_mask"],
            "ret_seq_product_ids": built["seq_product_ids"],
            "ret_seq_purchase_dates": built["seq_purchase_dates"],
            "ret_unseen_bn": built["unseen_bn"],
            "ret_target_bn": built["target_bn"],
        }
        atomic_torch_save(syn_payload, meta_tmp)
        c27_tmp.replace(cand27_path)
        meta_tmp.replace(meta_path)

    n_bn = int(shared_meta["universe"]["num_base_names"])
    manifest = {
        "version": SYNTHETIC_UNSEEN_VERSION,
        "identity": identity,
        "returning_fraction": float(returning_fraction),
        "seed": int(seed),
        "seq_len": int(seq_len),
        "changed_27_names": list(CAND_42_CHANGED_27_NAMES),
        "fo_variable_names": list(CAND_42_FO_VARIABLE_NAMES),
        "n_ret_synthetic": int(len(plan["ret_source_rows"])),
        "n_fo_synthetic": int(len(plan["fo_source_rows"])),
        "n_total_synthetic": int(len(plan["ret_source_rows"]) + len(plan["fo_source_rows"])),
        "ret_cand27_shape": [int(len(plan["ret_source_rows"])), n_bn, NUM_CAND_42_CHANGED_27],
        "train_cutoff": str(cutoff_ts),
        "test_start": str(test_start_ts),
        "heldout_access": False,
        "output_sha256": {
            "cand27": file_digest(cand27_path),
            "meta": file_digest(meta_path),
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    atomic_json(manifest, manifest_path, overwrite=True)
    print(
        f"[v2-syn] Built & sealed synthetic unseen cache "
        f"(ret={manifest['n_ret_synthetic']:,}, fo={manifest['n_fo_synthetic']:,}) "
        f"in {manifest['elapsed_seconds']:.1f}s -> {manifest_path}",
        flush=True,
    )
    return cand27_path, meta_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--returning-fraction", type=float, default=SYNTHETIC_RETURNING_FRACTION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_or_load_synthetic_unseen_cache(
        returning_fraction=args.returning_fraction,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
