"""Strictly causal (< as_of_date) Stage 3A-3D commercial, financial, rolling-manager, and warehouse features."""

from bisect import bisect_left
from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.dynamic_selector.feature_builder import log_count
from recommender.experiments.rolling_candidate_v2.data import prepare_or_load_rolling_feature_bundle
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START

from .base_name_features import _clean_base_name
from .config import CACHE_ROOT

COMMERCIAL_CACHE_VERSION = 1
COMMERCIAL_CACHE_NAMESPACE = "roadmap_v2_commercial_21d_v1"
COMMERCIAL_PARQUET_PATH = CACHE_ROOT / "commercial_dev_events.parquet"

GROUP_3A_NAMES: Tuple[str, ...] = (
    "cust_prepaid_or_cash_ratio",
    "cust_avg_deferral_days_norm",
    "cust_last_deferral_days_norm",
    "cust_actual_vs_contract_pay_ratio",
    "cust_overdue_unpaid_ratio",
    "cust_bonus_rate",
)
GROUP_3B_NAMES: Tuple[str, ...] = (
    "sku_unit_price_norm",
    "sku_unit_cost_usd_norm",
    "sku_margin_ratio",
    "sku_price_change_90d",
    "cust_spend_90d_norm",
    "cust_avg_order_rev_norm",
    "sku_price_vs_cust_avg_price",
)
GROUP_3C_NAMES: Tuple[str, ...] = (
    "mgr_sku_share_90d",
    "mgr_base_name_share_90d",
    "mgr_brand_share_90d",
    "mgr_cat_share_90d",
    "mgr_changed_recently",
)
GROUP_3D_NAMES: Tuple[str, ...] = (
    "wh_sku_shipped_30d",
    "wh_sku_shipped_90d",
    "wh_bn_shipped_30d",
)

COMMERCIAL_FEATURE_NAMES: Tuple[str, ...] = (
    *GROUP_3A_NAMES,
    *GROUP_3B_NAMES,
    *GROUP_3C_NAMES,
    *GROUP_3D_NAMES,
)
NUM_COMMERCIAL_FEATURES = len(COMMERCIAL_FEATURE_NAMES)

COMMERCIAL_GROUPS: Dict[str, Tuple[int, ...]] = {
    "3a_financial": tuple(range(0, 6)),
    "3b_pricing": tuple(range(6, 13)),
    "3c_manager": tuple(range(13, 18)),
    "3d_warehouse": tuple(range(18, 21)),
}

_DN_RE = re.compile(r"ДН\s*(\d+)", re.IGNORECASE)


def parse_deferral_days(term: Any) -> float:
    """Parse Ukrainian ERP payment term into contractual deferral days (0 for prepaid/fact/COD)."""
    if pd.isna(term):
        return float("nan")
    s = str(term).strip().upper()
    if not s or s == "NAN" or s == "<NA>":
        return float("nan")
    if "ПРЕДОПЛАТ" in s or "ФАКТ" in s or "НАЛОЖ" in s or "ГОТІВК" in s or "НАЛ" == s:
        return 0.0
    m = _DN_RE.search(s)
    if m:
        return float(min(int(m.group(1)), 180))
    if "ЧАСТИЧ" in s:
        return 7.0
    return float("nan")


def commercial_feature_contract() -> Dict[str, Any]:
    return {
        "version": COMMERCIAL_CACHE_VERSION,
        "namespace": COMMERCIAL_CACHE_NAMESPACE,
        "names": list(COMMERCIAL_FEATURE_NAMES),
        "groups": {k: list(v) for k, v in COMMERCIAL_GROUPS.items()},
        "dtype_on_disk": "float16",
        "compute_dtype": "float32",
        "causality_rule": "purchase_date < as_of_date AND full_payment_date < as_of_date; never >= TEST_START",
        "manager_policy": "rolling_90d_behavioral_shares_zero_static_manager_ids",
        "cost_sign_convention": "abs(cost) and abs(cost_usd) because NAV exports negative COGS",
    }


class CommercialRollingFeatureBuilder:
    """Builds 21 strictly-causal (< as_of_date) commercial, financial, rolling-manager, and warehouse features."""

    def __init__(
        self,
        events: pd.DataFrame,
        catalogue: pd.DataFrame,
        product_ids: Sequence[str],
        *,
        allow_heldout_test: bool = False,
        historical_recipe: bool = False,
        test_start: pd.Timestamp = TEST_START,
    ):
        self.allow_heldout_test = bool(allow_heldout_test)
        self.historical_recipe = historical_recipe
        self.test_start = pd.Timestamp(test_start).normalize()
        self.product_ids = [str(p).strip() for p in product_ids]
        self.num_products = len(self.product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # Catalogue metadata maps for the canonical 841 universe
        cat_df = catalogue.copy()
        cat_df["product_id"] = cat_df["product_id"].astype(str).str.strip()
        cat_dedup = cat_df.drop_duplicates("product_id").set_index("product_id")
        all_pid_to_bn = cat_dedup["base_name"].map(_clean_base_name).to_dict() if "base_name" in cat_dedup else {}
        all_pid_to_brand = (
            cat_dedup["brand"].fillna("").astype(str).str.strip().to_dict() if "brand" in cat_dedup else {}
        )
        all_pid_to_cat = (
            cat_dedup["product_category"].fillna("").astype(str).str.strip().to_dict()
            if "product_category" in cat_dedup
            else {}
        )

        # Canonical integer IDs (0 = missing) for fast vectorized counting
        self.prod_bn = [all_pid_to_bn.get(p, "") for p in self.product_ids]
        self.prod_brand = [all_pid_to_brand.get(p, "") for p in self.product_ids]
        self.prod_cat = [all_pid_to_cat.get(p, "") for p in self.product_ids]

        uniq_bn = sorted(set(self.prod_bn) - {""})
        uniq_brand = sorted(set(self.prod_brand) - {""})
        uniq_cat = sorted(set(self.prod_cat) - {""})
        self.bn_to_id = {v: i + 1 for i, v in enumerate(uniq_bn)}
        self.brand_to_id = {v: i + 1 for i, v in enumerate(uniq_brand)}
        self.cat_to_id = {v: i + 1 for i, v in enumerate(uniq_cat)}
        self.num_bns = len(uniq_bn)
        self.num_brands = len(uniq_brand)
        self.num_cats = len(uniq_cat)

        self.prod_bn_id = np.array([self.bn_to_id.get(v, 0) for v in self.prod_bn], dtype=np.int64)
        self.prod_brand_id = np.array([self.brand_to_id.get(v, 0) for v in self.prod_brand], dtype=np.int64)
        self.prod_cat_id = np.array([self.cat_to_id.get(v, 0) for v in self.prod_cat], dtype=np.int64)

        if events.empty:
            df = pd.DataFrame()
        else:
            df = events.copy()
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
            if bool(df["purchase_date"].isna().any()) or (
                not self.allow_heldout_test and not bool(df["purchase_date"].lt(self.test_start).all())
            ):
                raise ValueError(f"Development events only (< {self.test_start.date()}).")
            if "full_payment_date" in df.columns:
                df["full_payment_date"] = pd.to_datetime(df["full_payment_date"], errors="coerce").dt.normalize()
                if not self.allow_heldout_test:
                    # Strict pre-test clamp on settlement dates during development
                    df.loc[df["full_payment_date"] >= self.test_start, "full_payment_date"] = pd.NaT
            else:
                df["full_payment_date"] = pd.NaT

            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["product_id"] = df["product_id"].astype(str).str.strip()
            df["quantity"] = pd.to_numeric(df.get("quantity", 1.0), errors="coerce").fillna(1.0).clip(lower=1e-6)
            df["revenue"] = pd.to_numeric(df.get("revenue", np.nan), errors="coerce")
            # NAV accounting exports cost/cost_usd as negative numbers; use abs()
            df["cost_abs"] = pd.to_numeric(df.get("cost", np.nan), errors="coerce").abs()
            df["cost_usd_abs"] = pd.to_numeric(df.get("cost_usd", np.nan), errors="coerce").abs()
            df["doc_bonus"] = pd.to_numeric(df.get("doc_bonus", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
            df["deferral_days"] = (
                df["payment_terms"].map(parse_deferral_days)
                if "payment_terms" in df.columns
                else np.nan
            )
            df["is_cash"] = (
                df["payment_type"].fillna("").astype(str).str.strip().str.upper().eq("НАЛ")
                if "payment_type" in df.columns
                else False
            )
            df["manager_id"] = (
                df["manager_id"].fillna("").astype(str).str.strip() if "manager_id" in df.columns else ""
            )
            df["warehouse"] = (
                df["warehouse"].fillna("").astype(str).str.strip() if "warehouse" in df.columns else ""
            )
            df["p_idx"] = df["product_id"].map(self.pid_to_idx).fillna(-1).astype(np.int64)
            df["bn_id"] = df["product_id"].map(lambda p: self.bn_to_id.get(all_pid_to_bn.get(p, ""), 0)).astype(np.int64)
            df["brand_id"] = df["product_id"].map(lambda p: self.brand_to_id.get(all_pid_to_brand.get(p, ""), 0)).astype(np.int64)
            df["cat_id"] = df["product_id"].map(lambda p: self.cat_to_id.get(all_pid_to_cat.get(p, ""), 0)).astype(np.int64)
            df = df.sort_values(["purchase_date", "customer_id", "product_id"], kind="stable")

        # 1. Per-customer order-date timeline (for 3A financial, 3B spend, 3C current manager, 3D primary warehouse)
        self._cust_orders: Dict[str, Dict[str, np.ndarray]] = {}
        if not df.empty:
            for cid, grp in df.groupby("customer_id", sort=False):
                # Aggregate to order-date level (avoid multi-line baskets overweighting payment terms)
                order_dates = []
                def_days = []
                prepaid_flags = []
                settle_dates = []
                revs = []
                qtys = []
                bonuses = []
                mgrs = []
                whs = []
                for dt, rows in grp.groupby("purchase_date", sort=True):
                    order_dates.append(pd.Timestamp(dt).to_datetime64())
                    d_vals = rows["deferral_days"].dropna()
                    dd = float(d_vals.median()) if len(d_vals) else float("nan")
                    def_days.append(dd)
                    is_prep = bool(rows["is_cash"].any() or (not math.isnan(dd) and dd <= 0.0))
                    prepaid_flags.append(1.0 if is_prep else 0.0)
                    fpd = rows["full_payment_date"].dropna()
                    settle_dates.append(fpd.max().to_datetime64() if len(fpd) else np.datetime64("NaT"))
                    pos_rev = rows["revenue"].clip(lower=0.0).fillna(0.0)
                    revs.append(float(pos_rev.sum()))
                    qtys.append(float(rows["quantity"].sum()))
                    # doc_bonus is repeated per document line or per doc -> use max per day to avoid summing duplicates
                    bonuses.append(float(rows["doc_bonus"].max()))
                    valid_m = [m for m in rows["manager_id"].tolist() if m and m != "<NA>"]
                    mgrs.append(valid_m[-1] if valid_m else "")
                    valid_w = [w for w in rows["warehouse"].tolist() if w and w != "<NA>"]
                    whs.append(valid_w[-1] if valid_w else "")

                self._cust_orders[cid] = {
                    "dates": np.array(order_dates, dtype="datetime64[ns]"),
                    "deferral_days": np.array(def_days, dtype=np.float64),
                    "prepaid": np.array(prepaid_flags, dtype=np.float64),
                    "settle_dates": np.array(settle_dates, dtype="datetime64[ns]"),
                    "revenue": np.array(revs, dtype=np.float64),
                    "quantity": np.array(qtys, dtype=np.float64),
                    "bonus": np.array(bonuses, dtype=np.float64),
                    "manager": np.array(mgrs, dtype=object),
                    "warehouse": np.array(whs, dtype=object),
                }

        # 2. Per-SKU pricing/margin timeline (for 3B)
        self._sku_pricing: Dict[int, Dict[str, np.ndarray]] = {}
        if not df.empty:
            valid_p = df.loc[(df["p_idx"] >= 0) & (df["revenue"] > 0.0)]
            for p_idx, grp in valid_p.groupby("p_idx", sort=False):
                dts = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
                qty = grp["quantity"].to_numpy(dtype=np.float64)
                rev = grp["revenue"].to_numpy(dtype=np.float64)
                c_abs = grp["cost_abs"].to_numpy(dtype=np.float64)
                cu_abs = grp["cost_usd_abs"].to_numpy(dtype=np.float64)
                unit_p = rev / qty
                unit_cu = np.where(np.isfinite(cu_abs) & (cu_abs > 0), cu_abs / qty, np.nan)
                margin = np.where(np.isfinite(c_abs) & (c_abs > 0), np.clip((rev - c_abs) / rev, 0.0, 1.0), np.nan)
                self._sku_pricing[int(p_idx)] = {
                    "dates": dts,
                    "unit_price": unit_p,
                    "unit_cost_usd": unit_cu,
                    "margin": margin,
                }

        # 3. Per-manager chronological event arrays (for 3C 90d rolling shares — zero static manager IDs!)
        self._mgr_events: Dict[str, Dict[str, np.ndarray]] = {}
        if not df.empty:
            valid_m = df.loc[df["manager_id"].ne("") & df["manager_id"].ne("<NA>")]
            for mgr, grp in valid_m.groupby("manager_id", sort=False):
                self._mgr_events[mgr] = {
                    "dates": grp["purchase_date"].to_numpy(dtype="datetime64[ns]"),
                    "p_idx": grp["p_idx"].to_numpy(dtype=np.int64),
                    "bn_id": grp["bn_id"].to_numpy(dtype=np.int64),
                    "brand_id": grp["brand_id"].to_numpy(dtype=np.int64),
                    "cat_id": grp["cat_id"].to_numpy(dtype=np.int64),
                }

        # 4. Per-warehouse chronological event arrays (for 3D 30d/90d regional availability)
        self._wh_events: Dict[str, Dict[str, np.ndarray]] = {}
        if not df.empty:
            valid_w = df.loc[df["warehouse"].ne("") & df["warehouse"].ne("<NA>")]
            for wh, grp in valid_w.groupby("warehouse", sort=False):
                self._wh_events[wh] = {
                    "dates": grp["purchase_date"].to_numpy(dtype="datetime64[ns]"),
                    "p_idx": grp["p_idx"].to_numpy(dtype=np.int64),
                    "bn_id": grp["bn_id"].to_numpy(dtype=np.int64),
                }

        self._sku_price_cache: OrderedDict[pd.Timestamp, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._mgr_cache: OrderedDict[Tuple[str, pd.Timestamp], np.ndarray] = OrderedDict()
        self._wh_cache: OrderedDict[Tuple[str, pd.Timestamp], np.ndarray] = OrderedDict()

    def _sku_pricing_at(self, date_ts: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """Return ((P, 4) normalized [unit_price, unit_cost_usd, margin, price_change_90d], (P,) raw_unit_price)."""
        if date_ts in self._sku_price_cache:
            self._sku_price_cache.move_to_end(date_ts)
            return self._sku_price_cache[date_ts]

        feats = np.zeros((self.num_products, 4), dtype=np.float32)
        raw_p = np.zeros(self.num_products, dtype=np.float32)

        cutoff_64 = date_ts.to_datetime64()
        d30_64 = (date_ts - pd.Timedelta(days=30)).to_datetime64()
        d90_64 = (date_ts - pd.Timedelta(days=90)).to_datetime64()
        log_p_scale = math.log1p(50000.0)
        log_cu_scale = math.log1p(1000.0)

        for p_idx, d in self._sku_pricing.items():
            dts = d["dates"]
            stop = int(dts.searchsorted(cutoff_64, side="left"))
            if stop == 0:
                continue
            s90 = int(dts.searchsorted(d90_64, side="left"))
            s30 = int(dts.searchsorted(d30_64, side="left"))

            # Prefer last 90d window if non-empty, else fallback to all prior < t
            lo = s90 if stop > s90 else 0
            p_slice = d["unit_price"][lo:stop]
            med_p = float(np.median(p_slice))
            raw_p[p_idx] = med_p
            feats[p_idx, 0] = min(1.0, max(0.0, math.log1p(max(med_p, 0.0)) / log_p_scale))

            cu_slice = d["unit_cost_usd"][lo:stop]
            cu_valid = cu_slice[np.isfinite(cu_slice)]
            if len(cu_valid):
                feats[p_idx, 1] = min(1.0, max(0.0, math.log1p(float(np.median(cu_valid))) / log_cu_scale))

            m_slice = d["margin"][lo:stop]
            m_valid = m_slice[np.isfinite(m_slice)]
            if len(m_valid):
                feats[p_idx, 2] = float(np.clip(np.median(m_valid), 0.0, 1.0))

            if stop > s30 and s30 > s90:
                p_recent = float(np.median(d["unit_price"][s30:stop]))
                p_prev = float(np.median(d["unit_price"][s90:s30]))
                feats[p_idx, 3] = float(np.clip(math.log((p_recent + 1.0) / (p_prev + 1.0)) / 2.0, -1.0, 1.0))

        val = (feats, raw_p)
        self._sku_price_cache[date_ts] = val
        while len(self._sku_price_cache) > 512:
            self._sku_price_cache.popitem(last=False)
        return val

    def _manager_shares_at(self, mgr: str, date_ts: pd.Timestamp) -> np.ndarray:
        """Compute (P, 4) rolling 90d behavioral shares for `mgr` strictly in [t-90d, t)."""
        if not mgr or mgr not in self._mgr_events:
            return np.zeros((self.num_products, 4), dtype=np.float32)
        key = (mgr, date_ts)
        if key in self._mgr_cache:
            self._mgr_cache.move_to_end(key)
            return self._mgr_cache[key]

        ev = self._mgr_events[mgr]
        dts = ev["dates"]
        stop = int(dts.searchsorted(date_ts.to_datetime64(), side="left"))
        start = int(dts.searchsorted((date_ts - pd.Timedelta(days=90)).to_datetime64(), side="left"))
        out = np.zeros((self.num_products, 4), dtype=np.float32)
        total = stop - start
        if total > 0:
            inv = 1.0 / float(total)
            p_sl = ev["p_idx"][start:stop]
            valid_p = p_sl[p_sl >= 0]
            if len(valid_p):
                out[:, 0] = np.bincount(valid_p, minlength=self.num_products) * inv
            bn_cnt = np.bincount(ev["bn_id"][start:stop], minlength=self.num_bns + 1)
            bn_cnt[0] = 0
            out[:, 1] = bn_cnt[self.prod_bn_id] * inv
            br_cnt = np.bincount(ev["brand_id"][start:stop], minlength=self.num_brands + 1)
            br_cnt[0] = 0
            out[:, 2] = br_cnt[self.prod_brand_id] * inv
            ct_cnt = np.bincount(ev["cat_id"][start:stop], minlength=self.num_cats + 1)
            ct_cnt[0] = 0
            out[:, 3] = ct_cnt[self.prod_cat_id] * inv

        self._mgr_cache[key] = out
        while len(self._mgr_cache) > 2048:
            self._mgr_cache.popitem(last=False)
        return out

    def _warehouse_avail_at(self, wh: str, date_ts: pd.Timestamp) -> np.ndarray:
        """Compute (P, 3) rolling warehouse availability [sku_30d, sku_90d, bn_30d] strictly < t."""
        if not wh or wh not in self._wh_events:
            return np.zeros((self.num_products, 3), dtype=np.float32)
        key = (wh, date_ts)
        if key in self._wh_cache:
            self._wh_cache.move_to_end(key)
            return self._wh_cache[key]

        ev = self._wh_events[wh]
        dts = ev["dates"]
        stop = int(dts.searchsorted(date_ts.to_datetime64(), side="left"))
        s30 = int(dts.searchsorted((date_ts - pd.Timedelta(days=30)).to_datetime64(), side="left"))
        s90 = int(dts.searchsorted((date_ts - pd.Timedelta(days=90)).to_datetime64(), side="left"))

        out = np.zeros((self.num_products, 3), dtype=np.float32)
        if stop > s30:
            p30 = ev["p_idx"][s30:stop]
            p30 = p30[p30 >= 0]
            if len(p30):
                out[:, 0] = log_count(np.bincount(p30, minlength=self.num_products))
            bn30 = np.bincount(ev["bn_id"][s30:stop], minlength=self.num_bns + 1)
            bn30[0] = 0
            out[:, 2] = log_count(bn30[self.prod_bn_id])
        if stop > s90:
            p90 = ev["p_idx"][s90:stop]
            p90 = p90[p90 >= 0]
            if len(p90):
                out[:, 1] = log_count(np.bincount(p90, minlength=self.num_products))

        self._wh_cache[key] = out
        while len(self._wh_cache) > 1024:
            self._wh_cache.popitem(last=False)
        return out

    def build(self, customer_id: str, as_of_date: Union[str, pd.Timestamp]) -> np.ndarray:
        """Build (P, 21) float32 strictly-causal Stage 3A-3D features for all canonical products."""
        date_ts = pd.Timestamp(as_of_date).normalize()
        if pd.isna(date_ts) or (not self.allow_heldout_test and date_ts >= self.test_start):
            raise ValueError("Development predictions only")

        cid = str(customer_id).strip()
        cutoff_64 = date_ts.to_datetime64()
        out = np.zeros((self.num_products, NUM_COMMERCIAL_FEATURES), dtype=np.float32)
        # Neutral defaults
        out[:, 3] = 0.5  # cust_actual_vs_contract_pay_ratio neutral when no settled invoices yet

        # SKU pricing block (cols 6..9)
        sku_p_feats, raw_sku_p = self._sku_pricing_at(date_ts)
        out[:, 6:10] = sku_p_feats

        ord_data = self._cust_orders.get(cid)
        curr_mgr = ""
        curr_wh = ""
        mgr_changed = 0.0

        if ord_data is not None:
            dts = ord_data["dates"]
            stop = int(dts.searchsorted(cutoff_64, side="left"))
            if stop > 0:
                # 3A: Payment deferrals & financial health (cols 0..5)
                prepaid_sl = ord_data["prepaid"][:stop]
                def_sl = ord_data["deferral_days"][:stop]
                settle_sl = ord_data["settle_dates"][:stop]
                rev_sl = ord_data["revenue"][:stop]
                qty_sl = ord_data["quantity"][:stop]
                bonus_sl = ord_data["bonus"][:stop]

                out[:, 0] = float(prepaid_sl.mean())
                valid_def = def_sl[np.isfinite(def_sl)]
                if len(valid_def):
                    out[:, 1] = float(np.clip(valid_def.mean() / 90.0, 0.0, 1.0))
                    out[:, 2] = float(np.clip(valid_def[-1] / 90.0, 0.0, 1.0))

                # STRICT CAUSALITY: only inspect settlement dates strictly < cutoff_64!
                is_settled_before_t = (~np.isnat(settle_sl)) & (settle_sl < cutoff_64)
                if bool(is_settled_before_t.any()):
                    actual_days = (
                        (settle_sl[is_settled_before_t] - dts[:stop][is_settled_before_t])
                        .astype("timedelta64[D]")
                        .astype(np.float64)
                    )
                    actual_days = np.clip(actual_days, 0.0, 365.0)
                    contract_d = np.where(
                        np.isfinite(def_sl[is_settled_before_t]),
                        np.maximum(def_sl[is_settled_before_t], 7.0),
                        14.0,
                    )
                    ratios = actual_days / contract_d
                    out[:, 3] = float(np.clip(np.median(ratios) / 2.0, 0.0, 1.0))

                # Overdue unpaid as of date t: purchase_date + contract_days < t AND NOT settled strictly < t
                # Only count orders with known contractual deferral_days
                known_def = np.isfinite(def_sl)
                if bool(known_def.any()):
                    due_delta = np.array(
                        [pd.Timedelta(days=int(round(x))).to_timedelta64() for x in def_sl[known_def]],
                        dtype="timedelta64[ns]",
                    )
                    due_dates = dts[:stop][known_def] + due_delta
                    unpaid_at_t = ~is_settled_before_t[known_def]
                    overdue = (due_dates < cutoff_64) & unpaid_at_t
                    out[:, 4] = float(overdue.mean())

                tot_rev = float(rev_sl.sum())
                if tot_rev > 0.0:
                    out[:, 5] = float(np.clip(float(bonus_sl.sum()) / tot_rev, 0.0, 1.0))

                # 3B: Customer spend & price-tier fit (cols 10..12)
                s90 = int(dts.searchsorted((date_ts - pd.Timedelta(days=90)).to_datetime64(), side="left"))
                spend_90 = float(rev_sl[s90:stop].sum()) if stop > s90 else 0.0
                avg_ord_rev = float(rev_sl.mean())
                out[:, 10] = float(np.clip(math.log1p(max(spend_90, 0.0)) / math.log1p(2_000_000.0), 0.0, 1.0))
                out[:, 11] = float(np.clip(math.log1p(max(avg_ord_rev, 0.0)) / math.log1p(500_000.0), 0.0, 1.0))

                tot_qty = float(qty_sl.sum())
                if tot_rev > 0.0 and tot_qty > 0.0:
                    cust_avg_unit_p = tot_rev / tot_qty
                    has_sku_p = raw_sku_p > 0.0
                    ratio_arr = np.zeros(self.num_products, dtype=np.float32)
                    ratio_arr[has_sku_p] = np.clip(
                        np.log((raw_sku_p[has_sku_p] + 1.0) / (cust_avg_unit_p + 1.0)) / 4.0,
                        -1.0,
                        1.0,
                    )
                    out[:, 12] = ratio_arr

                # 3C: Current manager strictly < t (non-empty)
                mgr_sl = [m for m in ord_data["manager"][:stop].tolist() if m]
                if mgr_sl:
                    curr_mgr = mgr_sl[-1]
                    last_two = ord_data["manager"][max(0, stop - 2):stop]
                    if self.historical_recipe:
                        mgr_changed = float(len(mgr_sl) >= 2 and mgr_sl[-1] != mgr_sl[-2])
                    elif len(last_two) == 2 and all(last_two) and last_two[0] != last_two[1]:
                        mgr_changed = float(dts[stop - 1] >= (date_ts - pd.Timedelta(days=90)).to_datetime64())

                # 3D: Dominant recent warehouse strictly < t (last 5 orders, tie-break most recent)
                wh_sl = [w for w in ord_data["warehouse"][:stop].tolist() if w]
                if wh_sl:
                    recent_wh = wh_sl[-5:]
                    counts = Counter(recent_wh)
                    curr_wh = max(recent_wh, key=lambda w: (counts[w], recent_wh[::-1].index(w) * -1))

        # Fill 3C (cols 13..17) & 3D (cols 18..20)
        out[:, 13:17] = self._manager_shares_at(curr_mgr, date_ts)
        out[:, 17] = mgr_changed
        out[:, 18:21] = self._warehouse_avail_at(curr_wh, date_ts)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def prepare_or_load_commercial_feature_bundle(
    *,
    cache_root: Path = CACHE_ROOT,
    parquet_path: Path = COMMERCIAL_PARQUET_PATH,
    build: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build or validate sealed (N, 841, 21) float16 Stage 3A-3D commercial rolling feature arrays."""
    if not Path(parquet_path).is_file():
        raise FileNotFoundError(f"Pre-test commercial parquet missing: {parquet_path}")

    campaign_data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    product_ids = list(info["product_ids"])
    catalogue = load_product_catalogue(get_datasets_locs("items"))

    contract = commercial_feature_contract()
    contract_digest = json_digest(contract)
    parquet_sha = file_digest(parquet_path)
    identity = json_digest(
        {
            "version": COMMERCIAL_CACHE_VERSION,
            "namespace": COMMERCIAL_CACHE_NAMESPACE,
            "contract_digest": contract_digest,
            "parquet_sha256": parquet_sha,
            "rolling_features_identity": meta["rolling_features_identity"],
        }
    )

    root = Path(cache_root) / "commercial_21d"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("commercial_21d cache manifest identity mismatch.")
        for split in ("train", "val"):
            p = root / f"{split}_commercial_21d.npy"
            if file_digest(p) != manifest["feature_sha256"][split]:
                raise ValueError(f"Checksum mismatch for {p}.")
    else:
        if not build:
            raise FileNotFoundError(f"commercial_21d cache missing in {root}; build=True required.")
        events = pd.read_parquet(parquet_path, filters=[("purchase_date", "<", TEST_START)])
        builder = CommercialRollingFeatureBuilder(events, catalogue, product_ids)
        splits = ("train", "val")
        arrays = [
            np.lib.format.open_memmap(
                root / f"{split}_commercial_21d.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(campaign_data[split]["source"]["customer_ids"]), len(product_ids), NUM_COMMERCIAL_FEATURES),
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
            date_ts = origin + pd.Timedelta(days=day)
            arrays[split_idx][row_idx] = builder.build(cid, date_ts)
            if i > 0 and i % 5000 == 0:
                print(
                    f"[commercial_21d] {i}/{len(queries)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        for arr in arrays:
            arr.flush()
        prep_sec = time.perf_counter() - started
        manifest = {
            "version": COMMERCIAL_CACHE_VERSION,
            "namespace": COMMERCIAL_CACHE_NAMESPACE,
            "identity": identity,
            "contract": contract,
            "contract_digest": contract_digest,
            "parquet_sha256": parquet_sha,
            "rolling_features_identity": meta["rolling_features_identity"],
            "feature_sha256": {
                split: file_digest(root / f"{split}_commercial_21d.npy") for split in splits
            },
            "preparation_seconds": float(prep_sec),
            "heldout_access": False,
        }
        atomic_json(manifest, manifest_path, overwrite=False)

    mmaps = {
        split: np.load(root / f"{split}_commercial_21d.npy", mmap_mode="r")
        for split in ("train", "val")
    }
    return mmaps, manifest
