"""Native 585-base_name feature engine, pure base_name BGE-M3 text & multi-hot packaging, and fast inference.

Eliminates all 841-SKU intermediate arrays, removes month_sin/month_cos, collapses all purchase-line
statistics (category/subcat/brand shares, parsed viscosity/application shares, cadence, relationships)
strictly to distinct (customer_id, purchase_date, base_name) events, and computes all 53 features
directly over the 585 canonical base_name entities (<1 ms per live production request).
"""

from bisect import bisect_left
from collections import Counter, OrderedDict
from heapq import heappop, heappush
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import json_digest
from recommender.catalogue import CONTAINER_TYPES
from recommender.experiments.dynamic_selector.feature_builder import RelationshipTable, log_count
from recommender.experiments.static_enrichment.text_cache import _clean_field, _dedupe_tokens
from recommender.split import TEST_START
from recommender.text import load_text_encoder

from .base_name_features import _clean_base_name
from .commercial_features import CommercialRollingFeatureBuilder
from .config import CACHE_ROOT
from .parsed_spec_features import (
    parse_company_equipment_domains,
    parse_pack_size_liters_kg,
    parse_product_application,
    parse_viscosity_grades,
)


CONTAINER_BUCKETS: Tuple[str, ...] = tuple(sorted(CONTAINER_TYPES)) + ("NONE",)
CONTAINER_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CONTAINER_BUCKETS)}

# 8 natural B2B lubricant pack-volume tiers (liters/kg) covering 98.3% of dev SKUs cleanly
VOLUME_BUCKETS: Tuple[str, ...] = (
    "<=0.6",     # 0: tubes, grease cartridges (0.38-0.4kg), aerosols (0.4-0.5L)
    "0.6-1.5",   # 1: 1L bottles / small cans
    "1.5-6",     # 2: 4L-5L canisters
    "6-25",      # 3: 10L-25L pails / canisters (18kg, 20L, 22kg)
    "25-70",     # 4: 30kg-60L kegs
    "70-250",    # 5: 180kg-209L drums
    ">250",      # 6: 850kg-1000L IBC totes / bulk
    "NONE",      # 7: unparsed / unitless
)
VOLUME_TO_IDX: Dict[str, int] = {v: i for i, v in enumerate(VOLUME_BUCKETS)}

# 11 canonical B2B application domains covering 100% of dev formulations
APPLICATION_BUCKETS: Tuple[str, ...] = (
    "compressor",
    "coolant_adblue",
    "food_grade",
    "grease",
    "heavy_diesel_agro",
    "hydraulic",
    "industrial_gear_process",
    "marine_rail_aviation",
    "metalworking",
    "passenger_auto",
    "other",
)
APPLICATION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(APPLICATION_BUCKETS)}

# 34 canonical SAE / ISO VG / NLGI viscosity buckets (+ NONE) for multi-hot ProductTower encoding
VISCOSITY_BUCKETS: Tuple[str, ...] = (
    "0W-20", "0W-30", "5W-20", "5W-30", "5W-40",
    "10W-30", "10W-40", "10W-60", "15W-40", "15W-50", "20W-50",
    "75W-80", "75W-90", "80W-90", "85W-90",
    "ISO15", "ISO22", "ISO32", "ISO46", "ISO68", "ISO100",
    "ISO150", "ISO220", "ISO320", "ISO460", "ISO680",
    "NLGI00", "NLGI0", "NLGI1", "NLGI1.5", "NLGI2", "NLGI2.5", "NLGI3",
    "NONE",
)
VISCOSITY_TO_IDX: Dict[str, int] = {v: i for i, v in enumerate(VISCOSITY_BUCKETS)}


# 11 fresh point-in-time customer-only dynamic features (fed into Stage 1 CustomerTower & Stage 2 Token 0 — NOT broadcast to candidates!)
CUSTOMER_DYNAMIC_11_FEATURE_NAMES: Tuple[str, ...] = (
    "customer_age",
    "customer_overdue_ratio",
    "customer_30d_activity",
    "customer_90d_activity",
    "cust_prepaid_or_cash_ratio",
    "cust_avg_deferral_days_norm",
    "cust_last_deferral_days_norm",
    "cust_actual_vs_contract_pay_ratio",
    "cust_overdue_unpaid_ratio",
    "cust_bonus_rate",
    "mgr_changed_recently",
)
NUM_CUSTOMER_DYNAMIC_FEATURES = len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)
assert NUM_CUSTOMER_DYNAMIC_FEATURES == 11

# 7 legacy profile features recalculated strictly over merged (customer_id, purchase_date, base_name) events
# (days_since_last_order omitted since it duplicates customer_age in CUSTOMER_DYNAMIC_11_FEATURE_NAMES)
BASE_NAME_PROFILE_7_FEATURE_NAMES: Tuple[str, ...] = (
    "total_bn_events",
    "total_quantity",
    "distinct_base_names",
    "distinct_orders",
    "distinct_categories",
    "days_since_first_order",
    "median_order_gap",
)
NUM_BASE_NAME_PROFILE_FEATURES = len(BASE_NAME_PROFILE_7_FEATURE_NAMES)
assert NUM_BASE_NAME_PROFILE_FEATURES == 7

# 38 strictly candidate & customer-candidate features:
# - Zero calendar month features (month_sin / month_cos removed)
# - Zero customer-only broadcast constants (moved into Customer Embedding)
# - Zero exact duplicates (category_never_seen, family_never_seen, same_viscosity_grade_bought, base_name_hierarchical_backoff removed)
# - Keeps base_name_orders_90d and uses multi-value parse_viscosity_grades for user_viscosity_order_share
CANDIDATE_38_FEATURE_NAMES: Tuple[str, ...] = (
    # 0..12: Core 13 pairwise/attention-bias features
    "has_bought_base_sku",
    "user_base_sku_order_count_norm",
    "user_base_sku_qty_share",
    "days_since_last_base_sku_buy_norm",
    "base_sku_urgency",
    "user_cat_order_share",
    "user_subcat_order_share",
    "user_brand_order_share",
    "base_name_global_pop_norm",
    "base_name_industry_affinity",
    "is_repeat_category",
    "is_repeat_brand",
    "is_repeat_business_line",
    # 13..17: Category & family cadence/share (no binary never_seen duplicates)
    "category_age",
    "category_overdue_ratio",
    "family_age",
    "family_overdue_ratio",
    "family_history_share",
    # 18..24: Global base_name velocity & relative share (keeps base_name_orders_90d)
    "base_name_orders_7d",
    "base_name_orders_30d",
    "base_name_orders_90d",
    "base_name_buyers_30d",
    "base_name_growth_30d",
    "base_name_category_relative_share",
    "base_name_global_sale_age",
    # 25..31: Cross-company sequential transitions & complements (no formula duplicate hierarchical_backoff)
    "base_name_last_lift",
    "base_name_recent3_lift",
    "category_last_lift",
    "family_last_lift",
    "line_last_lift",
    "base_name_support_confidence",
    "base_name_same_basket_lift",
    # 32..34: Manager rolling 90d candidate shares
    "mgr_base_name_share_90d",
    "mgr_brand_share_90d",
    "mgr_cat_share_90d",
    # 35..37: Parsed pack-agnostic specs (multi-value viscosity share; no binary duplicate same_viscosity_grade_bought)
    "user_viscosity_order_share",
    "app_matches_company_equipment",
    "user_application_order_share",
)
NUM_CANDIDATE_CLEAN_FEATURES = len(CANDIDATE_38_FEATURE_NAMES)
assert NUM_CANDIDATE_CLEAN_FEATURES == 38


def volume_bucket_index(val: Optional[float]) -> int:
    if val is None or not np.isfinite(val) or val <= 0:
        return VOLUME_TO_IDX["NONE"]
    if val <= 0.6:
        return 0
    if val <= 1.5:
        return 1
    if val <= 6.0:
        return 2
    if val <= 25.0:
        return 3
    if val <= 70.0:
        return 4
    if val <= 250.0:
        return 5
    return 6

# 53 strictly-causal native base_name features (month_sin & month_cos permanently removed!)
NATIVE_53_FEATURE_NAMES: Tuple[str, ...] = (
    # 0..12: Core 13 pairwise/attention-bias features (exact same order as Stage 2 attention bias)
    "has_bought_base_sku",
    "user_base_sku_order_count_norm",
    "user_base_sku_qty_share",
    "days_since_last_base_sku_buy_norm",
    "base_sku_urgency",
    "user_cat_order_share",
    "user_subcat_order_share",
    "user_brand_order_share",
    "base_name_global_pop_norm",
    "base_name_industry_affinity",
    "is_repeat_category",
    "is_repeat_brand",
    "is_repeat_business_line",
    # 13..23: Customer, category & family cadence/recency (no calendar month_sin/month_cos)
    "customer_age",
    "customer_overdue_ratio",
    "customer_30d_activity",
    "customer_90d_activity",
    "category_age",
    "category_overdue_ratio",
    "category_never_seen",
    "family_age",
    "family_overdue_ratio",
    "family_never_seen",
    "family_history_share",
    # 24..30: Global base_name velocity & relative share
    "base_name_orders_7d",
    "base_name_orders_30d",
    "base_name_orders_90d",
    "base_name_buyers_30d",
    "base_name_growth_30d",
    "base_name_category_relative_share",
    "base_name_global_sale_age",
    # 31..38: Cross-company sequential transitions & same-basket complements
    "base_name_last_lift",
    "base_name_recent3_lift",
    "category_last_lift",
    "family_last_lift",
    "line_last_lift",
    "base_name_support_confidence",
    "base_name_hierarchical_backoff",
    "base_name_same_basket_lift",
    # 39..44: 3a_financial (6)
    "cust_prepaid_or_cash_ratio",
    "cust_avg_deferral_days_norm",
    "cust_last_deferral_days_norm",
    "cust_actual_vs_contract_pay_ratio",
    "cust_overdue_unpaid_ratio",
    "cust_bonus_rate",
    # 45..48: 3c_manager (4)
    "mgr_base_name_share_90d",
    "mgr_brand_share_90d",
    "mgr_cat_share_90d",
    "mgr_changed_recently",
    # 49..52: 3e_parsed_specs (4, pack-agnostic, computed on collapsed base_name events)
    "same_viscosity_grade_bought",
    "user_viscosity_order_share",
    "app_matches_company_equipment",
    "user_application_order_share",
)
NUM_NATIVE_FEATURES = len(NATIVE_53_FEATURE_NAMES)
assert NUM_NATIVE_FEATURES == 53


def _format_pack_size(val: float) -> str:
    if not np.isfinite(val) or val <= 0:
        return ""
    if abs(val - round(val)) < 1e-3:
        return f"{int(round(val))}"
    return f"{val:.2f}".rstrip("0").rstrip(".")


def observed_pack_ids(first_sales: pd.Series, observed_before: Optional[pd.Timestamp]) -> Optional[set[str]]:
    """Freeze pack metadata using the rolling pool's first genuine paid sale per SKU."""
    if observed_before is None:
        return None
    return set(first_sales.index[first_sales.lt(pd.Timestamp(observed_before).normalize())])


def representative_attr_ids(enrichment, product_ids, reps, allowed_member_pids):
    values = {key: ids[reps] for key, ids in enrichment["product_attr_ids"].items()}
    if allowed_member_pids is not None:
        missing = torch.tensor([product_ids[i] not in allowed_member_pids for i in reps], dtype=torch.bool)
        values = {key: ids.masked_fill(missing, 0) for key, ids in values.items()}
    return values


def candidate_storage_dtype(name):
    if name not in ("float16", "float32"):
        raise ValueError(f"Unsupported candidate feature storage dtype: {name}")
    return np.dtype(name)


def build_base_name_static_representations(
    universe: Mapping[str, Any],
    product_ids: Sequence[str],
    catalogue: pd.DataFrame,
    reps: Sequence[int],
    *,
    allowed_member_pids: Optional[set[str]] = None,
    frozen_v5_specs: bool = False,
    cache_path: Path = CACHE_ROOT / "pure_bn_text_cache_v1.pt",
) -> Dict[str, Any]:
    """Build & cache BGE-M3 embeddings strictly for the 585 distinct base_names:
      - `text_pack_free`: pure formulation text (brand + base_name + type + cat + subcat + line + family, NO product_name/pack/volume!)
      - `text_all_packs`: pure formulation text + all available container types & pack volumes across member SKUs
      - `container_multihot`: (585, 12) float32 multi-hot vector over all available container_types per base_name
      - `volume_multihot`: (585, 8) float32 multi-hot vector over all available volume tiers per base_name
      - `application_multihot`: (585, 11) float32 multi-hot vector over all parsed application domains per base_name
      - `viscosity_multihot`: (585, 34) float32 multi-hot vector over all parsed SAE/ISO/NLGI grades per base_name
      - `packaging_volume_multihot`: (585, 20) float32 concatenation [container_multihot, volume_multihot]
      - `full_specs_multihot`: (585, 65) float32 concatenation [container(12), volume(8), application(11), viscosity(34)]
    """
    base_names = list(universe["base_names"])
    members = universe["bn_to_member_pid_indices"]
    if allowed_member_pids is not None:
        catalogue = catalogue.loc[catalogue["product_id"].isin(allowed_member_pids)]
    cat_idx = catalogue.drop_duplicates("product_id").set_index("product_id")

    pack_free_texts: List[str] = []
    all_packs_texts: List[str] = []
    multihot = np.zeros((len(base_names), len(CONTAINER_BUCKETS)), dtype=np.float32)
    vol_multihot = np.zeros((len(base_names), len(VOLUME_BUCKETS)), dtype=np.float32)
    app_multihot = np.zeros((len(base_names), len(APPLICATION_BUCKETS)), dtype=np.float32)
    visc_multihot = np.zeros((len(base_names), len(VISCOSITY_BUCKETS)), dtype=np.float32)

    for b_i, (bn_name, mem_idxs) in enumerate(zip(base_names, members)):
        rep_pid = product_ids[int(reps[b_i])]
        rep_row = cat_idx.loc[rep_pid] if rep_pid in cat_idx.index else {}

        # Pure pack-free formulation text (strictly excludes product_name / short_name pack mentions!)
        ordered_parts = []
        brand_v = _clean_field(rep_row.get("brand"))
        if brand_v:
            ordered_parts.append(brand_v)
        ordered_parts.append(bn_name)
        for field in ("product_type", "product_category", "product_subcategory", "business_line", "product_family"):
            v = _clean_field(rep_row.get(field))
            if v:
                ordered_parts.append(v)
        clean_text = _dedupe_tokens(ordered_parts)
        pack_free_texts.append(clean_text)

        # Collect ALL container types, pack volumes, application domains & viscosity grades across member SKUs
        conts: set[str] = set()
        sizes: set[float] = set()
        vol_idxs: set[int] = set()
        apps: set[str] = set()
        viscs: set[str] = set()
        for p_i in mem_idxs:
            pid = product_ids[int(p_i)]
            if pid not in cat_idx.index:
                continue
            r = cat_idx.loc[pid]
            c_val = _clean_field(r.get("container_type")).upper()
            if c_val in CONTAINER_TO_IDX:
                conts.add(c_val)
            p_name = _clean_field(r.get("product_name"))
            txt = " ".join(_clean_field(r.get(k)) for k in ("product_name", "base_name", "short_name"))
            sz = parse_pack_size_liters_kg(txt, c_val)
            if sz is not None and np.isfinite(sz) and sz > 0:
                sizes.add(round(float(sz), 3))
                vol_idxs.add(volume_bucket_index(float(sz)))
            viscs.update(parse_viscosity_grades(p_name, frozen_v5=frozen_v5_specs))
            a = parse_product_application(
                txt,
                category=_clean_field(r.get("product_category")),
                subcategory=_clean_field(r.get("product_subcategory")),
                business_line=_clean_field(r.get("business_line")),
            )
            if a in APPLICATION_TO_IDX:
                apps.add(a)

        if allowed_member_pids is None or any(product_ids[i] in allowed_member_pids for i in mem_idxs):
            viscs.update(parse_viscosity_grades(bn_name, frozen_v5=frozen_v5_specs))
        if conts:
            for c_val in conts:
                multihot[b_i, CONTAINER_TO_IDX[c_val]] = 1.0
        else:
            multihot[b_i, CONTAINER_TO_IDX["NONE"]] = 1.0

        if vol_idxs:
            for v_idx in vol_idxs:
                vol_multihot[b_i, v_idx] = 1.0
        else:
            vol_multihot[b_i, VOLUME_TO_IDX["NONE"]] = 1.0

        if apps:
            for a_val in apps:
                app_multihot[b_i, APPLICATION_TO_IDX[a_val]] = 1.0
        else:
            app_multihot[b_i, APPLICATION_TO_IDX["other"]] = 1.0

        matched_viscs = [v for v in viscs if v in VISCOSITY_TO_IDX]
        if matched_viscs:
            for v_val in matched_viscs:
                visc_multihot[b_i, VISCOSITY_TO_IDX[v_val]] = 1.0
        else:
            visc_multihot[b_i, VISCOSITY_TO_IDX["NONE"]] = 1.0

        pack_suffix_parts = []
        if conts:
            pack_suffix_parts.append("Тара: " + ", ".join(sorted(conts)))
        if sizes:
            size_strs = [_format_pack_size(s) for s in sorted(sizes)]
            pack_suffix_parts.append("Объем: " + ", ".join(s for s in size_strs if s))
        full_text = f"{clean_text} | {' | '.join(pack_suffix_parts)}" if pack_suffix_parts else clean_text
        all_packs_texts.append(full_text)

    pv_multihot = np.concatenate([multihot, vol_multihot], axis=1)
    full_specs = np.concatenate([multihot, vol_multihot, app_multihot, visc_multihot], axis=1)

    cache_path = Path(cache_path)
    # The historical cache remains readable. Write misses to an immutable text-specific path.
    text_key = json_digest([base_names, pack_free_texts, all_packs_texts])
    keyed_cache = cache_path.with_name(f"{cache_path.stem}-{text_key}{cache_path.suffix}")
    for existing in (keyed_cache, cache_path):
        if not existing.is_file():
            continue
        cached = torch.load(existing, map_location="cpu", weights_only=False)
        if (
            cached.get("base_names") == base_names
            and cached.get("pack_free_texts") == pack_free_texts
            and cached.get("all_packs_texts") == all_packs_texts
        ):
            cached["container_multihot"] = torch.from_numpy(multihot)
            cached["volume_multihot"] = torch.from_numpy(vol_multihot)
            cached["application_multihot"] = torch.from_numpy(app_multihot)
            cached["viscosity_multihot"] = torch.from_numpy(visc_multihot)
            cached["packaging_volume_multihot"] = torch.from_numpy(pv_multihot)
            cached["full_specs_multihot"] = torch.from_numpy(full_specs)
            cached["container_buckets"] = CONTAINER_BUCKETS
            cached["volume_buckets"] = VOLUME_BUCKETS
            cached["application_buckets"] = APPLICATION_BUCKETS
            cached["viscosity_buckets"] = VISCOSITY_BUCKETS
            return cached

    print(f"[pure-bn-text] Encoding {len(base_names)} distinct base_names with BAAI/bge-m3 (pack-free & all-packs)...", flush=True)
    encoder = load_text_encoder()
    emb_free = encoder.encode(pack_free_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_packs = encoder.encode(all_packs_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    payload = {
        "base_names": base_names,
        "pack_free_texts": pack_free_texts,
        "all_packs_texts": all_packs_texts,
        "text_pack_free": torch.from_numpy(np.asarray(emb_free, dtype=np.float32)),
        "text_all_packs": torch.from_numpy(np.asarray(emb_packs, dtype=np.float32)),
        "container_multihot": torch.from_numpy(multihot),
        "volume_multihot": torch.from_numpy(vol_multihot),
        "application_multihot": torch.from_numpy(app_multihot),
        "viscosity_multihot": torch.from_numpy(visc_multihot),
        "packaging_volume_multihot": torch.from_numpy(pv_multihot),
        "full_specs_multihot": torch.from_numpy(full_specs),
        "container_buckets": CONTAINER_BUCKETS,
        "volume_buckets": VOLUME_BUCKETS,
        "application_buckets": APPLICATION_BUCKETS,
        "viscosity_buckets": VISCOSITY_BUCKETS,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=keyed_cache.parent) as tmp:
        torch.save(payload, tmp)
        tmp.flush()
        try:
            os.link(tmp.name, keyed_cache)
        except FileExistsError:
            pass  # Another builder published the same text identity.
    return payload


class NativeBaseNameFeatureEngine:
    """Computes all 53 strictly-causal (< as_of_date) features natively over the 585 canonical base_names.

    - Collapses all purchases to distinct (customer_id, purchase_date, base_name) events up front.
    - Zero 841-SKU arrays and zero 842x842 SKU transition tables.
    - Drops month_sin and month_cos completely.
    - Computes customer cat/subcat/brand shares, viscosity shares, and application shares on collapsed
      base_name events so multi-pack same-day lines are never double-counted.
    """

    def __init__(
        self,
        purchases: pd.DataFrame,
        commercial_events: pd.DataFrame,
        catalogue: pd.DataFrame,
        companies: pd.DataFrame,
        product_ids: Sequence[str],
        universe: Mapping[str, Any],
        reps: Sequence[int],
        *,
        allowed_member_pids: Optional[set[str]] = None,
        frozen_v5_replay: bool = False,
        allow_non_monotonic: bool = False,
        allow_heldout_test: bool = False,
    ):
        self.frozen_v5_replay = frozen_v5_replay
        self.allow_non_monotonic = bool(allow_non_monotonic)
        self.allow_heldout_test = bool(allow_heldout_test)
        self.base_names = list(universe["base_names"])
        self.n_bn = len(self.base_names)
        self.bn_to_idx = dict(universe["bn_to_idx"])
        self.all_pid_to_bn = dict(universe["all_pid_to_bn"])
        self.reps = np.asarray(reps, dtype=np.int64)
        self.rep_pids = [product_ids[int(i)] for i in self.reps]

        if not purchases.empty:
            p_dates = pd.to_datetime(purchases["purchase_date"], format="mixed").dt.normalize()
            if bool(p_dates.isna().any()) or (not self.allow_heldout_test and not bool(p_dates.lt(TEST_START).all())):
                raise ValueError(f"Development purchases only (< {TEST_START.date()}).")

        # 1. Canonical per-base_name static attributes (category, subcategory, brand, line, family, viscosity, application)
        cat_df = catalogue.copy()
        cat_df["product_id"] = cat_df["product_id"].astype(str).str.strip()
        if allowed_member_pids is not None:
            cat_df = cat_df.loc[cat_df["product_id"].isin(allowed_member_pids)].copy()
        cat_dedup = cat_df.drop_duplicates("product_id").set_index("product_id")

        # Parse multi-value viscosity grades & application for EVERY base_name in full catalogue (so prior OOV BNs still contribute!)
        cat_df["bn_clean"] = cat_df["base_name"].map(_clean_base_name)
        all_bn_meta: Dict[str, Dict[str, Any]] = {}
        for bn_s, grp in cat_df.loc[cat_df["bn_clean"] != ""].groupby("bn_clean", sort=False):
            viscs_set: set[str] = set(parse_viscosity_grades(bn_s, frozen_v5=frozen_v5_replay))
            apps = []
            for _, row in grp.iterrows():
                p_name = str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else ""
                txt = " ".join(str(row.get(c, "")) for c in ("product_name", "base_name", "short_name") if pd.notna(row.get(c)))
                viscs_set |= parse_viscosity_grades(p_name, frozen_v5=frozen_v5_replay)
                a = parse_product_application(
                    txt,
                    category=str(row.get("product_category", "")) if pd.notna(row.get("product_category")) else "",
                    subcategory=str(row.get("product_subcategory", "")) if pd.notna(row.get("product_subcategory")) else "",
                    business_line=str(row.get("business_line", "")) if pd.notna(row.get("business_line")) else "",
                )
                if a and a != "other":
                    apps.append(a)
            all_bn_meta[bn_s] = {
                "product_category": str(grp["product_category"].dropna().mode().iloc[0]).strip() if grp["product_category"].notna().any() else "",
                "product_subcategory": str(grp["product_subcategory"].dropna().mode().iloc[0]).strip() if grp["product_subcategory"].notna().any() else "",
                "brand": str(grp["brand"].dropna().mode().iloc[0]).strip() if grp["brand"].notna().any() else "",
                "business_line": str(grp["business_line"].dropna().mode().iloc[0]).strip() if grp["business_line"].notna().any() else "",
                "product_family": str(grp["product_family"].dropna().mode().iloc[0]).strip() if grp["product_family"].notna().any() else "",
                "viscosities": tuple(sorted(viscs_set)),
                "application": Counter(apps).most_common(1)[0][0] if apps else "other",
            }
        # Override modeled 585 taxonomy with earliest-pack representative for strict causality, while keeping unified multi-value viscosities/application
        for b_i, bn_s in enumerate(self.base_names):
            rpid = self.rep_pids[b_i]
            if rpid in cat_dedup.index:
                r = cat_dedup.loc[rpid]
                entry = all_bn_meta.setdefault(bn_s, {})
                for col in ("product_category", "product_subcategory", "brand", "business_line", "product_family"):
                    val = str(r.get(col, "")).strip() if pd.notna(r.get(col)) else ""
                    if val:
                        entry[col] = val

        self._bn_attr: Dict[str, List[str]] = {
            k: [all_bn_meta.get(bn_s, {}).get(k, "") for bn_s in self.base_names]
            for k in ("product_category", "product_subcategory", "brand", "business_line", "product_family", "application")
        }
        self._bn_viscosities: List[Tuple[str, ...]] = [
            tuple(all_bn_meta.get(bn_s, {}).get("viscosities", ())) for bn_s in self.base_names
        ]

        # Precompute lookup position arrays for instant vectorized fills over 585 formulations
        self._attr_positions: Dict[str, Dict[str, np.ndarray]] = {}
        for field in ("product_category", "product_subcategory", "brand", "business_line", "product_family", "application"):
            pos_map: Dict[str, List[int]] = {}
            for i, v in enumerate(self._bn_attr[field]):
                if v and not (field == "application" and v == "other"):
                    pos_map.setdefault(v, []).append(i)
            self._attr_positions[field] = {k: np.array(v, dtype=np.int64) for k, v in pos_map.items()}
        visc_pos_map: Dict[str, List[int]] = {}
        for i, v_tuple in enumerate(self._bn_viscosities):
            for v in v_tuple:
                if v:
                    visc_pos_map.setdefault(v, []).append(i)
        # Precompute each observed grade set's matching candidates, counting a multi-grade event once.
        self._viscosity_matches = {}
        for meta in all_bn_meta.values():
            grades = meta.get("viscosities", ())
            if grades and grades not in self._viscosity_matches:
                self._viscosity_matches[grades] = np.asarray(sorted({
                    i for grade in grades for i in visc_pos_map.get(grade, ())
                }), dtype=np.int64)

        if frozen_v5_replay:
            self._viscosity_matches = {v: np.asarray(pos, dtype=np.int64) for v, pos in visc_pos_map.items()}

        # Integer category mapping for category-relative 30d share
        uniq_cats = sorted(set(self._bn_attr["product_category"]) | {""})
        cat_lu = {c: i for i, c in enumerate(uniq_cats)}
        self._bn_cat_idx = np.array([cat_lu[c] for c in self._bn_attr["product_category"]], dtype=np.int64)
        self._num_cats = len(uniq_cats)

        # Customer industry & equipment-domain match vectors (585,)
        self.customer_industry: Dict[str, str] = {}
        self.company_app_match: Dict[str, np.ndarray] = {}
        if companies is not None and not companies.empty:
            apps_585 = self._bn_attr["application"]
            for _, row in companies.iterrows():
                cid = str(row["customer_id"]).strip()
                ind = str(row.get("industry_branch", "")).strip() if pd.notna(row.get("industry_branch")) else ""
                desc = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
                self.customer_industry[cid] = ind
                domains = parse_company_equipment_domains(desc, ind, frozen_v5=frozen_v5_replay)
                if domains:
                    self.company_app_match[cid] = np.array([1.0 if a in domains else 0.0 for a in apps_585], dtype=np.float32)

        # 2. Collapse ALL purchases once to distinct (customer_id, purchase_date, base_name) events!
        if not purchases.empty:
            df = purchases[["customer_id", "purchase_date", "product_id", "quantity"]].copy()
            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["product_id"] = df["product_id"].astype(str).str.strip()
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0.0)
            df["bn"] = df["product_id"].map(self.all_pid_to_bn).fillna("")
            bn_df = (
                df.loc[df["bn"] != ""]
                .groupby(["customer_id", "purchase_date", "bn"], as_index=False, sort=False)["quantity"]
                .sum()
                .sort_values(["purchase_date", "customer_id", "bn"], kind="stable")
            )
        else:
            bn_df = pd.DataFrame(columns=["customer_id", "purchase_date", "bn", "quantity"])

        # Customer total volume timeline (summed across collapsed base_name events)
        self._cust_vol_dates: Dict[str, np.ndarray] = {}
        self._cust_vol_cumsum: Dict[str, np.ndarray] = {}
        # Customer collapsed base_name event stream (chronological) for incremental O(1) prefix stats
        self._cust_events: Dict[str, List[Tuple[pd.Timestamp, int, float, str, str, str, str, str, str, str]]] = {}
        self._cust_event_dates: Dict[str, List[pd.Timestamp]] = {}
        # Customer dated baskets for cadence + category/family recency + cross-company relationship tables
        self._cust_baskets: Dict[str, List[Tuple[pd.Timestamp, List[int], set[str], set[str]]]] = {}
        self._cust_basket_dates: Dict[str, List[pd.Timestamp]] = {}

        # Relationship mappings over indices 0..n_bn (0 = unknown/OOV, 1..n_bn = modeled base_names)
        maps: Dict[str, np.ndarray] = {"bn": np.arange(self.n_bn + 1, dtype=np.int64)}
        for field in ("product_category", "product_family", "business_line"):
            vals = self._bn_attr[field]
            vocab = {v: i + 1 for i, v in enumerate(sorted(set(vals) - {""}))}
            arr = np.zeros(self.n_bn + 1, dtype=np.int64)
            arr[1:] = [vocab.get(v, 0) for v in vals]
            maps[field] = arr

        transitions: List[Tuple[pd.Timestamp, List[int], List[int]]] = []
        complements: List[Tuple[pd.Timestamp, List[int], List[int]]] = []

        if not bn_df.empty:
            for cid, grp in bn_df.groupby("customer_id", sort=False):
                g = grp.sort_values(["purchase_date", "bn"], kind="stable")
                by_day = g.groupby("purchase_date", sort=True)["quantity"].sum()
                self._cust_vol_dates[cid] = by_day.index.to_numpy(dtype="datetime64[ns]")
                self._cust_vol_cumsum[cid] = by_day.to_numpy(dtype=np.float64).cumsum()

                evs = []
                dts = []
                for dt, bn_s, q_v in zip(g["purchase_date"], g["bn"], g["quantity"]):
                    ts = pd.Timestamp(dt)
                    bid = self.bn_to_idx.get(bn_s, -1) + 1  # 1..585, or 0 if OOV
                    m = all_bn_meta.get(bn_s, {})
                    evs.append((
                        ts, bid, bn_s, float(q_v),
                        m.get("product_category", ""),
                        m.get("product_subcategory", ""),
                        m.get("brand", ""),
                        m.get("business_line", ""),
                        m.get("product_family", ""),
                        m.get("viscosities", ()),
                        m.get("application", "other"),
                    ))
                    dts.append(ts)
                self._cust_events[cid] = evs
                self._cust_event_dates[cid] = dts

                baskets = []
                for dt, rows in g.groupby("purchase_date", sort=True):
                    ts = pd.Timestamp(dt)
                    bids = sorted({self.bn_to_idx[b] + 1 for b in rows["bn"] if b in self.bn_to_idx})
                    cats_day = {all_bn_meta.get(b, {}).get("product_category", "") for b in rows["bn"]} - {""}
                    fams_day = {all_bn_meta.get(b, {}).get("product_family", "") for b in rows["bn"]} - {""}
                    baskets.append((ts, bids, cats_day, fams_day))
                    if len(bids) >= 2:
                        for b in bids:
                            complements.append((ts, [b], [o for o in bids if o != b]))
                self._cust_baskets[cid] = baskets
                self._cust_basket_dates[cid] = [d for d, _, _, _ in baskets]
                for (_, prev_b, _, _), (dt_next, next_b, _, _) in zip(baskets, baskets[1:]):
                    if prev_b and next_b:
                        transitions.append((dt_next, prev_b, next_b))

        transitions.sort(key=lambda x: x[0])
        complements.sort(key=lambda x: x[0])
        self._tables = {k: RelationshipTable(m, transitions) for k, m in maps.items()}
        self._complements = RelationshipTable(maps["bn"], complements)

        # 3. Global & industry rolling sale timelines per base_name
        self._pop_scale = math.log1p(1000.0)
        self._bn_sale_dates: Dict[str, np.ndarray] = {}
        self._bn_sale_cids: Dict[str, np.ndarray] = {}
        self._ind_event_dates: Dict[str, np.ndarray] = {}
        self._ind_bn_dates: Dict[Tuple[str, str], np.ndarray] = {}
        if not bn_df.empty:
            for bn_s, grp in bn_df.groupby("bn", sort=False):
                if bn_s in self.bn_to_idx:
                    self._bn_sale_dates[bn_s] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
                    self._bn_sale_cids[bn_s] = grp["customer_id"].to_numpy()
            bn_ind = bn_df.assign(industry=bn_df["customer_id"].map(self.customer_industry).fillna(""))
            known_ind = bn_ind.loc[bn_ind["industry"] != ""]
            for ind, grp in known_ind.groupby("industry", sort=False):
                self._ind_event_dates[ind] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
            for (ind, bn_s), grp in known_ind.groupby(["industry", "bn"], sort=False):
                if bn_s in self.bn_to_idx:
                    self._ind_bn_dates[(ind, bn_s)] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")

        # 4. Commercial rolling builder (order-date 3a_financial & current manager lookup)
        self._comm = CommercialRollingFeatureBuilder(
            commercial_events, catalogue, self.rep_pids, allow_heldout_test=self.allow_heldout_test,
            historical_recipe=frozen_v5_replay,
        )
        # 4b. Manager rolling 90d shares strictly on MERGED (manager_id, customer_id, purchase_date, base_name) events!
        self._mgr_bn_events: Dict[str, Dict[str, np.ndarray]] = {}
        self._mgr_bn_cache: OrderedDict[Tuple[str, pd.Timestamp], np.ndarray] = OrderedDict()
        uniq_br_585 = sorted(set(self._bn_attr["brand"]) - {""})
        uniq_ct_585 = sorted(set(self._bn_attr["product_category"]) - {""})
        self._mgr_br_lu = {v: i + 1 for i, v in enumerate(uniq_br_585)}
        self._mgr_ct_lu = {v: i + 1 for i, v in enumerate(uniq_ct_585)}
        self._prod_mgr_bn_id = np.arange(1, self.n_bn + 1, dtype=np.int64)
        self._prod_mgr_br_id = np.array([self._mgr_br_lu.get(v, 0) for v in self._bn_attr["brand"]], dtype=np.int64)
        self._prod_mgr_ct_id = np.array([self._mgr_ct_lu.get(v, 0) for v in self._bn_attr["product_category"]], dtype=np.int64)
        if commercial_events is not None and not commercial_events.empty and "manager_id" in commercial_events.columns:
            mc = commercial_events[["manager_id", "customer_id", "purchase_date", "product_id"]].copy()
            mc["manager_id"] = mc["manager_id"].fillna("").astype(str).str.strip()
            mc["customer_id"] = mc["customer_id"].astype(str).str.strip()
            mc["product_id"] = mc["product_id"].astype(str).str.strip()
            mc["purchase_date"] = pd.to_datetime(mc["purchase_date"], format="mixed").dt.normalize()
            mc["bn"] = mc["product_id"].map(self.all_pid_to_bn).fillna("")
            mc = mc.loc[(mc["manager_id"] != "") & (mc["bn"] != "")]
            mc = mc.drop_duplicates(["manager_id", "customer_id", "purchase_date", "bn"])
            mc = mc.sort_values(["manager_id", "purchase_date", "customer_id", "bn"], kind="stable")
            for mgr, grp in mc.groupby("manager_id", sort=False):
                bn_ids = np.array([self.bn_to_idx.get(b, -1) + 1 for b in grp["bn"]], dtype=np.int64)
                br_ids = np.array(
                    [self._mgr_br_lu.get(all_bn_meta.get(b, {}).get("brand", ""), 0) for b in grp["bn"]],
                    dtype=np.int64,
                )
                ct_ids = np.array(
                    [self._mgr_ct_lu.get(all_bn_meta.get(b, {}).get("product_category", ""), 0) for b in grp["bn"]],
                    dtype=np.int64,
                )
                self._mgr_bn_events[mgr] = {
                    "dates": grp["purchase_date"].to_numpy(dtype="datetime64[ns]"),
                    "bn_id": bn_ids,
                    "br_id": br_ids,
                    "ct_id": ct_ids,
                }

        self._prefix_states: Dict[str, Dict[str, Any]] = {}
        self._global_cache: OrderedDict[pd.Timestamp, np.ndarray] = OrderedDict()
        self._industry_cache: OrderedDict[Tuple[str, pd.Timestamp], np.ndarray] = OrderedDict()
        self._last_date: Optional[pd.Timestamp] = None

    def _manager_bn_shares_90d(self, mgr: str, date_ts: pd.Timestamp) -> np.ndarray:
        """Compute (585, 3) [mgr_bn_share_90d, mgr_brand_share_90d, mgr_cat_share_90d] on merged base_name events."""
        if not mgr or mgr not in self._mgr_bn_events:
            return np.zeros((self.n_bn, 3), dtype=np.float32)
        key = (mgr, date_ts)
        if key in self._mgr_bn_cache:
            self._mgr_bn_cache.move_to_end(key)
            return self._mgr_bn_cache[key]
        ev = self._mgr_bn_events[mgr]
        dts = ev["dates"]
        stop = int(dts.searchsorted(date_ts.to_datetime64(), side="left"))
        start = int(dts.searchsorted((date_ts - pd.Timedelta(days=90)).to_datetime64(), side="left"))
        out = np.zeros((self.n_bn, 3), dtype=np.float32)
        total = stop - start
        if total > 0:
            inv = 1.0 / float(total)
            bn_cnt = np.bincount(ev["bn_id"][start:stop], minlength=self.n_bn + 1)
            bn_cnt[0] = 0
            out[:, 0] = bn_cnt[self._prod_mgr_bn_id] * inv
            br_cnt = np.bincount(ev["br_id"][start:stop], minlength=len(self._mgr_br_lu) + 1)
            br_cnt[0] = 0
            out[:, 1] = br_cnt[self._prod_mgr_br_id] * inv
            ct_cnt = np.bincount(ev["ct_id"][start:stop], minlength=len(self._mgr_ct_lu) + 1)
            ct_cnt[0] = 0
            out[:, 2] = ct_cnt[self._prod_mgr_ct_id] * inv
        self._mgr_bn_cache[key] = out
        while len(self._mgr_bn_cache) > 2048:
            self._mgr_bn_cache.popitem(last=False)
        return out

    def _customer_prefix(self, cid: str, as_of: pd.Timestamp) -> Dict[str, Any]:
        events = self._cust_events.get(cid)
        if not events:
            return {"end": 0}
        end = bisect_left(self._cust_event_dates[cid], as_of)
        if end == 0:
            return {"end": 0}
        prev = self._prefix_states.get(cid)
        if prev is not None and prev["end"] <= end:
            state = prev
        else:
            state = {
                "end": 0,
                "bns": {},
                "distinct_bn_names": set(),
                "cats": Counter(),
                "subcats": Counter(),
                "brands": Counter(),
                "lines": set(),
                "viscs": Counter(),
                "apps": Counter(),
            }
        for dt, bid, bn_s, qty, cat, subcat, brand, line, _, viscs, app in events[state["end"] : end]:
            state["distinct_bn_names"].add(bn_s)
            if bid > 0:
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
                    lo, hi = info["lower_gaps"], info["upper_gaps"]
                    if not lo or gap <= -lo[0]:
                        heappush(lo, -gap)
                    else:
                        heappush(hi, gap)
                    if len(lo) > len(hi) + 1:
                        heappush(hi, -heappop(lo))
                    elif len(hi) > len(lo):
                        heappush(lo, -heappop(hi))
                    info["median_gap"] = -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2.0
                info["order_count"] += 1
                info["volume"] += qty
                info["last_date"] = dt
            if cat:
                state["cats"][cat] += 1
            if subcat:
                state["subcats"][subcat] += 1
            if brand:
                state["brands"][brand] += 1
            if line:
                state["lines"].add(line)
            if viscs:
                if self.frozen_v5_replay:
                    state["viscs"].update(viscs)
                else:
                    state["viscs"][viscs] += 1
            if app and app != "other":
                state["apps"][app] += 1
        state["end"] = end
        if prev is None or end >= prev["end"]:
            self._prefix_states[cid] = state
        return state

    def _global_bn_features(self, date_ts: pd.Timestamp) -> np.ndarray:
        if date_ts in self._global_cache:
            self._global_cache.move_to_end(date_ts)
            return self._global_cache[date_ts]
        out = np.zeros((self.n_bn, 8), dtype=np.float32)
        out[:, 7] = 1.0
        counts_30 = np.zeros(self.n_bn, dtype=np.float64)
        cutoff_64 = date_ts.to_datetime64()
        start_64s = [(date_ts - pd.Timedelta(days=d)).to_datetime64() for d in (7, 30, 60, 90)]
        for i, bn_s in enumerate(self.base_names):
            dts = self._bn_sale_dates.get(bn_s)
            if dts is None:
                continue
            stop = int(dts.searchsorted(cutoff_64, side="left"))
            if stop == 0:
                continue
            s7, s30, s60, s90 = (int(dts.searchsorted(st, side="left")) for st in start_64s)
            n7, n30, n60, n90 = stop - s7, stop - s30, stop - s60, stop - s90
            buyers30 = len(set(self._bn_sale_cids[bn_s][s30:stop])) if stop > s30 else 0
            out[i, 0] = min(1.0, math.log1p(stop) / self._pop_scale)
            out[i, 1:5] = log_count([n7, n30, n90, buyers30])
            out[i, 5] = float(np.clip(math.log((n30 + 1.0) / (n60 - n30 + 1.0)) / 5.0, -1.0, 1.0))
            counts_30[i] = n30
            out[i, 7] = min((date_ts - pd.Timestamp(dts[stop - 1])).days / 365.0, 1.0)
        cat_tot = np.bincount(self._bn_cat_idx, weights=counts_30, minlength=self._num_cats)
        out[:, 6] = (counts_30 / np.maximum(cat_tot[self._bn_cat_idx], 1.0)).astype(np.float32)
        self._global_cache[date_ts] = out
        while len(self._global_cache) > 512:
            self._global_cache.popitem(last=False)
        return out

    def _industry_bn_affinity(self, industry: str, date_ts: pd.Timestamp) -> np.ndarray:
        if not industry:
            return np.zeros(self.n_bn, dtype=np.float32)
        key = (industry, date_ts)
        if key in self._industry_cache:
            self._industry_cache.move_to_end(key)
            return self._industry_cache[key]
        out = np.zeros(self.n_bn, dtype=np.float32)
        ind_dts = self._ind_event_dates.get(industry)
        if ind_dts is not None:
            cutoff_64 = date_ts.to_datetime64()
            tot = int(ind_dts.searchsorted(cutoff_64, side="left"))
            if tot > 0:
                inv = 1.0 / float(tot)
                for i, bn_s in enumerate(self.base_names):
                    bd = self._ind_bn_dates.get((industry, bn_s))
                    if bd is not None:
                        c = int(bd.searchsorted(cutoff_64, side="left"))
                        if c > 0:
                            out[i] = min(1.0, c * inv)
        self._industry_cache[key] = out
        while len(self._industry_cache) > 2048:
            self._industry_cache.popitem(last=False)
        return out

    def build(self, customer_id: str, as_of_date: Union[str, pd.Timestamp]) -> np.ndarray:
        """Build (585, 53) float32 strictly-causal native base_name features (no month_sin/month_cos)."""
        date_ts = pd.Timestamp(as_of_date).normalize()
        if pd.isna(date_ts) or (not self.allow_heldout_test and date_ts >= TEST_START):
            raise ValueError("Development predictions only")

        if self._last_date is not None and date_ts < self._last_date:
            if not self.allow_non_monotonic:
                raise ValueError("NativeBaseNameFeatureEngine must be queried chronologically.")
            for tbl in (*self._tables.values(), self._complements):
                tbl.reset()
        self._last_date = date_ts
        for tbl in (*self._tables.values(), self._complements):
            tbl.advance(date_ts)

        cid = str(customer_id).strip()
        date_64 = date_ts.to_datetime64()
        out = np.zeros((self.n_bn, NUM_NATIVE_FEATURES), dtype=np.float32)
        out[:, 3] = 1.0  # days_since_last_base_sku_buy_norm default = 1.0

        # Global & industry features (cols 8, 9, 24..30)
        g = self._global_bn_features(date_ts)
        out[:, 8] = g[:, 0]
        out[:, 9] = self._industry_bn_affinity(self.customer_industry.get(cid, ""), date_ts)
        out[:, 24:31] = g[:, 1:8]

        # Equipment domain static match (col 51)
        eq_vec = self.company_app_match.get(cid)
        if eq_vec is not None:
            out[:, 51] = eq_vec

        # Customer prefix on collapsed base_name events (cols 0..7, 10..12, 49, 50, 52)
        st = self._customer_prefix(cid, date_ts)
        n_ev = st["end"]
        if n_ev > 0:
            inv_ev = 1.0 / float(n_ev)
            vdts = self._cust_vol_dates.get(cid)
            tot_vol = 0.0
            if vdts is not None:
                vidx = int(vdts.searchsorted(date_64, side="left"))
                if vidx > 0:
                    tot_vol = float(self._cust_vol_cumsum[cid][vidx - 1])
            log50 = math.log1p(50.0)
            log365 = math.log1p(365.0)
            for bid, info in st["bns"].items():
                i0 = bid - 1
                cnt = info["order_count"]
                delta_days = max(0.0, (date_ts - info["last_date"]).total_seconds() / 86400.0)
                out[i0, 0] = 1.0
                out[i0, 1] = min(1.0, math.log1p(cnt) / log50)
                out[i0, 2] = min(1.0, info["volume"] / tot_vol) if tot_vol > 0 else 0.0
                out[i0, 3] = min(1.0, math.log1p(delta_days) / log365)
                if cnt >= 2:
                    out[i0, 4] = min(1.0, (delta_days / max(1.0, float(info["median_gap"]))) / 5.0)

            cat_pos = self._attr_positions["product_category"]
            for v, cnt in st["cats"].items():
                pos = cat_pos.get(v)
                if pos is not None:
                    out[pos, 5] = min(1.0, cnt * inv_ev)
                    out[pos, 10] = 1.0
            sub_pos = self._attr_positions["product_subcategory"]
            for v, cnt in st["subcats"].items():
                pos = sub_pos.get(v)
                if pos is not None:
                    out[pos, 6] = min(1.0, cnt * inv_ev)
            br_pos = self._attr_positions["brand"]
            for v, cnt in st["brands"].items():
                pos = br_pos.get(v)
                if pos is not None:
                    out[pos, 7] = min(1.0, cnt * inv_ev)
                    out[pos, 11] = 1.0
            ln_pos = self._attr_positions["business_line"]
            for v in st["lines"]:
                pos = ln_pos.get(v)
                if pos is not None:
                    out[pos, 12] = 1.0
            matches = np.zeros(self.n_bn, dtype=np.int64)
            for grades, count in st["viscs"].items():
                pos = self._viscosity_matches.get(grades)
                if pos is not None:
                    if self.frozen_v5_replay:
                        matches[pos] = np.maximum(matches[pos], count)
                    else:
                        matches[pos] += count
            out[:, 49] = matches > 0
            out[:, 50] = matches * inv_ev
            app_pos = self._attr_positions["application"]
            for v, cnt in st["apps"].items():
                pos = app_pos.get(v)
                if pos is not None:
                    out[pos, 52] = min(1.0, cnt * inv_ev)

        # Customer cadence, category/family recency & cross-company relationship lifts (cols 13..23, 31..38)
        baskets = self._cust_baskets.get(cid, [])
        n_prior = bisect_left(self._cust_basket_dates.get(cid, []), date_ts) if baskets else 0
        prior_b = baskets[:n_prior]
        if not prior_b:
            out[:, [13, 17, 19, 20, 22]] = 1.0
        else:
            bdates = [d for d, _, _, _ in prior_b]
            gaps = np.diff(np.array(bdates, dtype="datetime64[ns]")).astype("timedelta64[D]").astype(float)
            med_gap = float(np.median(gaps)) if len(gaps) else 30.0
            age = (date_ts - bdates[-1]).days
            # Archived v5 excluded the first day of the window on its daily timeline.
            start = date_ts + pd.Timedelta(days=int(self.frozen_v5_replay))
            d30 = start - pd.Timedelta(days=30)
            d90 = start - pd.Timedelta(days=90)
            out[:, 13] = min(age / 365.0, 1.0)
            out[:, 14] = min(age / max(med_gap, 1.0) / 4.0, 1.0)
            out[:, 15] = float(log_count(sum(d >= d30 for d in bdates)))
            out[:, 16] = float(log_count(sum(d >= d90 for d in bdates)))

            # Category & family recency/overdue
            out[:, [17, 19, 20, 22]] = 1.0
            cat_hist: Dict[str, List[pd.Timestamp]] = {}
            fam_hist: Dict[str, List[pd.Timestamp]] = {}
            for d, _, cats_d, fams_d in prior_b:
                for v in cats_d:
                    cat_hist.setdefault(v, []).append(d)
                for v in fams_d:
                    fam_hist.setdefault(v, []).append(d)

            cat_pos = self._attr_positions["product_category"]
            for v, h in cat_hist.items():
                pos = cat_pos.get(v)
                if pos is not None:
                    iv = float(np.median([(b - a).days for a, b in zip(h, h[1:])])) if len(h) > 1 else 30.0
                    el = (date_ts - h[-1]).days
                    out[pos, 17] = min(el / 365.0, 1.0)
                    out[pos, 18] = min(el / max(iv, 1.0) / 4.0, 1.0)
                    out[pos, 19] = 0.0

            fam_pos = self._attr_positions["product_family"]
            inv_nb = 1.0 / float(len(prior_b))
            for v, h in fam_hist.items():
                pos = fam_pos.get(v)
                if pos is not None:
                    iv = float(np.median([(b - a).days for a, b in zip(h, h[1:])])) if len(h) > 1 else 30.0
                    el = (date_ts - h[-1]).days
                    out[pos, 20] = min(el / 365.0, 1.0)
                    out[pos, 21] = min(el / max(iv, 1.0) / 4.0, 1.0)
                    out[pos, 22] = 0.0
                    out[pos, 23] = len(h) * inv_nb

            # Relationship lifts (cols 31..38) directly over 1..585
            last_bids = prior_b[-1][1]
            bn_last, bn_sup = self._tables["bn"].score(last_bids)
            rec_num = np.zeros(self.n_bn + 1, dtype=np.float64)
            w_sum = 0.0
            for idx_r, (_, bids_r, _, _) in enumerate(reversed(prior_b[-3:])):
                w = 0.5 ** idx_r
                rec_num += self._tables["bn"].score(bids_r)[0] * w
                w_sum += w
            bn_rec3 = rec_num / max(w_sum, 1e-12)
            cat_l = self._tables["product_category"].score(last_bids)[0][1:]
            fam_l = self._tables["product_family"].score(last_bids)[0][1:]
            ln_l = self._tables["business_line"].score(last_bids)[0][1:]
            same_l = self._complements.score(last_bids)[0][1:]
            bn_l = bn_last[1:]
            sup = bn_sup[1:]
            backoff = np.clip(sup * bn_l + (1.0 - sup) * 0.5 * (fam_l + cat_l), -1.0, 1.0)
            out[:, 31] = bn_l.astype(np.float32)
            out[:, 32] = bn_rec3[1:].astype(np.float32)
            out[:, 33] = cat_l.astype(np.float32)
            out[:, 34] = fam_l.astype(np.float32)
            out[:, 35] = ln_l.astype(np.float32)
            out[:, 36] = sup.astype(np.float32)
            out[:, 37] = backoff.astype(np.float32)
            out[:, 38] = same_l.astype(np.float32)

        # Commercial 3a_financial (cols 39..44) + merged base_name 3c_manager (cols 45..48)
        comm_585 = self._comm.build(cid, date_ts)
        out[:, 39:45] = comm_585[:, 0:6]
        out[:, 48] = comm_585[:, 17]  # mgr_changed_recently
        curr_mgr = ""
        ord_data = self._comm._cust_orders.get(cid)
        if ord_data is not None:
            stop_c = int(ord_data["dates"].searchsorted(date_64, side="left"))
            if stop_c > 0:
                mgr_sl = [m for m in ord_data["manager"][:stop_c].tolist() if m]
                if mgr_sl:
                    curr_mgr = mgr_sl[-1]
        out[:, 45:48] = self._manager_bn_shares_90d(curr_mgr, date_ts)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def build_merged_profile(self, customer_id: str, as_of_date: Union[str, pd.Timestamp]) -> np.ndarray:
        """Compute the 7 legacy profile features strictly over merged (customer_id, purchase_date, base_name) events."""
        date_ts = pd.Timestamp(as_of_date).normalize()
        if pd.isna(date_ts) or (not self.allow_heldout_test and date_ts >= TEST_START):
            raise ValueError("Development predictions only")
        cid = str(customer_id).strip()
        date_64 = date_ts.to_datetime64()
        prof = np.zeros(NUM_BASE_NAME_PROFILE_FEATURES, dtype=np.float32)

        st = self._customer_prefix(cid, date_ts)
        n_ev = int(st["end"])
        if n_ev == 0:
            return prof

        tot_vol = 0.0
        vdts = self._cust_vol_dates.get(cid)
        if vdts is not None:
            vidx = int(vdts.searchsorted(date_64, side="left"))
            if vidx > 0:
                tot_vol = float(self._cust_vol_cumsum[cid][vidx - 1])

        baskets = self._cust_baskets.get(cid, [])
        n_prior = bisect_left(self._cust_basket_dates.get(cid, []), date_ts) if baskets else 0
        prior_b = baskets[:n_prior]
        if prior_b:
            bdates = [d for d, _, _, _ in prior_b]
            tenure = max(0.0, float((date_ts - bdates[0]).days))
            gaps = np.diff(np.array(bdates, dtype="datetime64[ns]")).astype("timedelta64[D]").astype(float)
            med_gap = float(np.median(gaps)) if len(gaps) else 0.0
        else:
            tenure = 0.0
            med_gap = 0.0

        prof[0] = min(1.0, math.log1p(n_ev) / math.log1p(500.0))                          # total_bn_events
        prof[1] = min(1.0, math.log1p(max(tot_vol, 0.0)) / math.log1p(5000.0))            # total_quantity
        prof[2] = min(1.0, math.log1p(len(st["distinct_bn_names"])) / math.log1p(100.0))  # distinct_base_names
        prof[3] = min(1.0, math.log1p(n_prior) / math.log1p(100.0))                       # distinct_orders
        prof[4] = min(1.0, math.log1p(len(st["cats"])) / math.log1p(25.0))                # distinct_categories
        prof[5] = min(1.0, math.log1p(tenure) / math.log1p(1000.0))                       # days_since_first_order
        prof[6] = min(1.0, math.log1p(max(med_gap, 0.0)) / math.log1p(365.0))             # median_order_gap
        return np.nan_to_num(prof, nan=0.0, posinf=1.0, neginf=0.0)

    def build_split(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        *,
        include_merged_profile: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return strictly separated (cand_38 [585, 38], cust [11 or 18]) float32 features.

        - `cand_38`: strictly candidate & customer-candidate features (no month, no customer-only
          broadcasts, no exact duplicates; keeps `base_name_orders_90d` and multi-value viscosity share).
        - `cust`: fresh point-in-time customer state (11D current dynamic features, or 18D when
          `include_merged_profile=True` appends the 7 base_name-merged profile features).
        """
        date_ts = pd.Timestamp(as_of_date).normalize()
        full = self.build(customer_id, date_ts)
        cand_38 = np.ascontiguousarray(full[:, _CLEAN_CAND_38_INDICES], dtype=np.float32)
        cust_11 = np.ascontiguousarray(full[0, _CLEAN_CUST_11_INDICES], dtype=np.float32)
        if include_merged_profile:
            prof_7 = self.build_merged_profile(customer_id, date_ts)
            return cand_38, np.concatenate([cust_11, prof_7], axis=0)
        return cand_38, cust_11


_CLEAN_CAND_38_INDICES: Tuple[int, ...] = tuple(
    NATIVE_53_FEATURE_NAMES.index(n) for n in CANDIDATE_38_FEATURE_NAMES
)
_CLEAN_CUST_11_INDICES: Tuple[int, ...] = tuple(
    NATIVE_53_FEATURE_NAMES.index(n) for n in CUSTOMER_DYNAMIC_11_FEATURE_NAMES
)
