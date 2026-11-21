"""Isolated paths, schemas, and arm definitions for roadmap_v2."""

import os
from pathlib import Path
from typing import Dict, Tuple

from recommender.experiments.dynamic_selector.feature_schema import NAMES as SOURCE_41_NAMES

CACHE_ROOT = Path(os.environ.get("RECSYS_ROADMAP_V2_CACHE", "data/interim/roadmap_v2"))
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_ROADMAP_V2_ROOT", "checkpoints/roadmap-v2"))
EVIDENCE_ROOT = Path(os.environ.get("RECSYS_ROADMAP_V2_EVIDENCE", "experiments/roadmap_v2"))

# Exact 19-for-19 1-to-1 mirror between exact product_id columns in SOURCE_41_NAMES and base_name columns.
EXACT_SKU_TO_BASE_NAME_PAIRS: Tuple[Tuple[str, str], ...] = (
    # Customer x SKU history (5)
    ("is_repeat_sku", "has_bought_base_sku"),
    ("user_sku_order_count_norm", "user_base_sku_order_count_norm"),
    ("user_sku_qty_share", "user_base_sku_qty_share"),
    ("days_since_last_sku_buy_norm", "days_since_last_base_sku_buy_norm"),
    ("sku_urgency", "base_sku_urgency"),
    # Global & industry popularity (2)
    ("global_sku_popularity_norm", "base_name_global_pop_norm"),
    ("industry_sku_affinity", "base_name_industry_affinity"),
    # Global rolling demand (7)
    ("sku_orders_7d", "base_name_orders_7d"),
    ("sku_orders_30d", "base_name_orders_30d"),
    ("sku_orders_90d", "base_name_orders_90d"),
    ("sku_buyers_30d", "base_name_buyers_30d"),
    ("sku_growth_30d", "base_name_growth_30d"),
    ("category_relative_share", "base_name_category_relative_share"),
    ("global_sale_age", "base_name_global_sale_age"),
    # Cross-company transitions & complements (5)
    ("sku_last_lift", "base_name_last_lift"),
    ("sku_recent3_lift", "base_name_recent3_lift"),
    ("sku_support_confidence", "base_name_support_confidence"),
    ("hierarchical_backoff", "base_name_hierarchical_backoff"),
    ("same_basket_lift", "base_name_same_basket_lift"),
)

EXACT_SKU_FEATURE_NAMES: Tuple[str, ...] = tuple(sku for sku, _ in EXACT_SKU_TO_BASE_NAME_PAIRS)
BASE_NAME_FEATURE_NAMES: Tuple[str, ...] = tuple(bn for _, bn in EXACT_SKU_TO_BASE_NAME_PAIRS)
EXACT_SKU_SOURCE_INDICES: Tuple[int, ...] = tuple(SOURCE_41_NAMES.index(name) for name in EXACT_SKU_FEATURE_NAMES)
EXACT_SKU_TO_BASE_NAME_MAP: Dict[str, str] = dict(EXACT_SKU_TO_BASE_NAME_PAIRS)

NUM_BASE_NAME_FEATURES = len(BASE_NAME_FEATURE_NAMES)

ARM_SKU_ONLY = "sku_only"
ARM_SKU_PLUS_BASE_NAME = "sku_plus_base_name"
ARM_BASE_NAME_ONLY = "base_name_only"
VALID_BASE_NAME_ARMS = (ARM_SKU_ONLY, ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY)

STAGE1_OMITTED_CALENDAR_FEATURES: Tuple[str, ...] = ("month_sin", "month_cos")

SEED = 42
STAGE1_EPOCHS = 10
STAGE1_BATCH_SIZE = 64
STAGE1_LR = 0.001
STAGE1_WEIGHT_DECAY = 0.0001
STAGE1_POS_WEIGHT = 20.0

STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 64
STAGE2_LR = 0.001
STAGE2_WEIGHT_DECAY = 0.0001
STAGE2_ETA_MIN = 1e-5
STAGE2_GRAD_CLIP_NORM = 1.0


def feature_names_for_arm(arm: str) -> Tuple[str, ...]:
    """Return ordered feature names for one Stage 1 base_name ablation arm."""
    if arm == ARM_SKU_ONLY:
        return tuple(SOURCE_41_NAMES)
    if arm == ARM_SKU_PLUS_BASE_NAME:
        return tuple(SOURCE_41_NAMES) + BASE_NAME_FEATURE_NAMES
    if arm == ARM_BASE_NAME_ONLY:
        replaced = list(SOURCE_41_NAMES)
        for src_idx, bn_name in zip(EXACT_SKU_SOURCE_INDICES, BASE_NAME_FEATURE_NAMES):
            replaced[src_idx] = bn_name
        return tuple(replaced)
    raise ValueError(f"Unknown arm {arm!r}; expected one of {VALID_BASE_NAME_ARMS}.")
