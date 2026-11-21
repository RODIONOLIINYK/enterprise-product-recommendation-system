"""Strict causality and mathematical parity tests for roadmap_v2 base_name rolling features."""

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.base_name_features import (
    BaseNameRollingFeatureBuilder,
    assemble_arm_features_numpy,
    assemble_arm_features_torch,
)
from recommender.experiments.roadmap_v2.config import (
    ARM_BASE_NAME_ONLY,
    ARM_SKU_ONLY,
    ARM_SKU_PLUS_BASE_NAME,
    BASE_NAME_FEATURE_NAMES,
    EXACT_SKU_SOURCE_INDICES,
    NUM_BASE_NAME_FEATURES,
    feature_names_for_arm,
)


def _synthetic_frames():
    catalogue = pd.DataFrame(
        {
            "product_id": ["P1_20L", "P1_209L", "P2_20L", "P3_1L"],
            "base_name": ["OIL_A", "OIL_A", "OIL_B", "OIL_C"],
            "product_category": ["CAT1", "CAT1", "CAT1", "CAT2"],
            "product_subcategory": ["SUB1", "SUB1", "SUB1", "SUB2"],
            "brand": ["BRAND1", "BRAND1", "BRAND1", "BRAND2"],
            "business_line": ["LINE1", "LINE1", "LINE1", "LINE2"],
            "product_family": ["FAM1", "FAM1", "FAM2", "FAM3"],
        }
    )
    companies = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "industry_branch": ["IND_X", "IND_X"],
            "city": ["Kyiv", "Kyiv"],
            "region_code": ["01", "01"],
        }
    )
    purchases = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2", "C1", "C2"],
            "purchase_date": pd.to_datetime(
                ["2025-01-10", "2025-01-10", "2025-02-09", "2025-02-15", "2025-03-01", "2025-03-05"]
            ),
            "product_id": ["P1_20L", "P1_209L", "P1_20L", "P2_20L", "P2_20L", "P3_1L"],
            "quantity": [2.0, 1.0, 3.0, 5.0, 4.0, 1.0],
        }
    )
    return purchases, catalogue, companies


def test_strict_causality_no_same_day_or_future_leakage():
    purchases, catalogue, companies = _synthetic_frames()
    pids = ["P1_20L", "P1_209L", "P2_20L", "P3_1L"]

    # Truncated purchases strictly < 2025-02-15 vs full purchases queried on 2025-02-15
    # Must be BIT-FOR-BIT identical (same-day 2025-02-15 and future 2025-03-01/05 must have zero effect!)
    b_full = BaseNameRollingFeatureBuilder(purchases, catalogue, companies, pids, allow_non_monotonic=True)
    b_trunc = BaseNameRollingFeatureBuilder(
        purchases.loc[purchases["purchase_date"] < pd.Timestamp("2025-02-15")],
        catalogue,
        companies,
        pids,
        allow_non_monotonic=True,
    )

    cat_lift = np.array([0.1, 0.1, 0.2, -0.1], dtype=np.float32)
    fam_lift = np.array([0.3, 0.3, 0.0, 0.0], dtype=np.float32)

    f_full = b_full.build("C1", "2025-02-15", category_last_lift=cat_lift, family_last_lift=fam_lift)
    f_trunc = b_trunc.build("C1", "2025-02-15", category_last_lift=cat_lift, family_last_lift=fam_lift)
    np.testing.assert_allclose(f_full, f_trunc, rtol=0.0, atol=0.0)

    # On 2025-02-15, C1 bought OIL_A on 2025-01-10 (both 20L + 209L = 1 order event, qty 3) and 2025-02-09 (qty 3).
    # Both P1_20L and P1_209L share identical base_name features!
    np.testing.assert_allclose(f_full[0], f_full[1], rtol=0.0, atol=0.0)
    assert f_full[0, 0] == 1.0  # has_bought_base_sku
    assert f_full[1, 0] == 1.0  # P1_209L inherits base_name history even though 209L was only bought once!
    # Distinct order dates for OIL_A = 2 (Jan 10 + Feb 9), NOT 3!
    expected_cnt_norm = np.log1p(2.0) / np.log1p(50.0)
    assert np.isclose(f_full[0, 1], expected_cnt_norm, atol=1e-6)
    # Median gap = 30 days (Feb 9 - Jan 10), elapsed = 6 days (Feb 15 - Feb 9) -> urgency = (6/30)/5 = 0.04
    assert np.isclose(f_full[0, 4], (6.0 / 30.0) / 5.0, atol=1e-6)
    # P2_20L has NOT been bought yet by C1 or C2 strictly < 2025-02-15!
    assert f_full[2, 0] == 0.0
    assert f_full[2, 3] == 1.0  # never bought recency default
    assert f_full[2, 5] == 0.0  # global pop = 0 before C2's same-day purchase!

    # Same-day multi-pack (P1_20L + P1_209L on Jan 10) must NOT create a self-complement edge!
    assert f_full[0, 18] == 0.0


def test_refuses_heldout_dates():
    purchases, catalogue, companies = _synthetic_frames()
    bad = pd.concat(
        [
            purchases,
            pd.DataFrame(
                {
                    "customer_id": ["C1"],
                    "purchase_date": pd.to_datetime(["2026-02-19"]),
                    "product_id": ["P1_20L"],
                    "quantity": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Development purchases only"):
        BaseNameRollingFeatureBuilder(bad, catalogue, companies, ["P1_20L"])

    b = BaseNameRollingFeatureBuilder(purchases, catalogue, companies, ["P1_20L"])
    with pytest.raises(ValueError, match="Development predictions only"):
        b.build("C1", "2026-02-19")


def test_arm_assembly_shapes_and_parity():
    assert NUM_BASE_NAME_FEATURES == 19
    assert len(feature_names_for_arm(ARM_SKU_ONLY)) == 41
    assert len(feature_names_for_arm(ARM_SKU_PLUS_BASE_NAME)) == 60
    assert len(feature_names_for_arm(ARM_BASE_NAME_ONLY)) == 41

    rng = np.random.default_rng(7)
    f41 = rng.standard_normal((4, 10, 41)).astype(np.float32)
    f19 = rng.standard_normal((4, 10, 19)).astype(np.float32)

    for arm in (ARM_SKU_ONLY, ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY):
        arr = assemble_arm_features_numpy(f41, f19, arm)
        ten = assemble_arm_features_torch(torch.from_numpy(f41), torch.from_numpy(f19), arm).numpy()
        np.testing.assert_allclose(arr, ten, rtol=0.0, atol=0.0)

    c_arr = assemble_arm_features_numpy(f41, f19, ARM_BASE_NAME_ONLY)
    for k, src_idx in enumerate(EXACT_SKU_SOURCE_INDICES):
        np.testing.assert_allclose(c_arr[..., src_idx], f19[..., k], rtol=0.0, atol=0.0)
    # Non-SKU columns untouched
    non_sku = [i for i in range(41) if i not in set(EXACT_SKU_SOURCE_INDICES)]
    np.testing.assert_allclose(c_arr[..., non_sku], f41[..., non_sku], rtol=0.0, atol=0.0)
