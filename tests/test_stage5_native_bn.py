"""Unit tests for Stage 5 v4 native base_name engine, multi-value viscosity, and CleanStage5BaseNameRanker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.clean_ranker import CleanStage5BaseNameRanker
from recommender.experiments.roadmap_v2.native_base_name import (
    APPLICATION_BUCKETS,
    CANDIDATE_38_FEATURE_NAMES,
    CONTAINER_BUCKETS,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    NATIVE_53_FEATURE_NAMES,
    VISCOSITY_BUCKETS,
    VOLUME_BUCKETS,
    NativeBaseNameFeatureEngine,
    volume_bucket_index,
)
from recommender.experiments.roadmap_v2.parsed_spec_features import parse_viscosity_grades


def _synthetic_fixtures():
    catalogue = pd.DataFrame(
        [
            {
                "product_id": "SKU_A_20",
                "product_name": "Shell Rimula R4 X 15W-40 20L",
                "base_name": "Shell Rimula R4 X 15W-40",
                "short_name": "Rimula R4 X 20L",
                "brand": "Shell",
                "product_type": "Масло",
                "product_category": "Моторные масла",
                "product_subcategory": "Дизельные",
                "business_line": "CVL",
                "product_family": "Rimula",
                "container_type": "КАНИСТРА",
            },
            {
                "product_id": "SKU_A_209",
                "product_name": "Shell Rimula R4 X 15W-40 209L",
                "base_name": "Shell Rimula R4 X 15W-40",
                "short_name": "Rimula R4 X 209L",
                "brand": "Shell",
                "product_type": "Масло",
                "product_category": "Моторные масла",
                "product_subcategory": "Дизельные",
                "business_line": "CVL",
                "product_family": "Rimula",
                "container_type": "БОЧКА",
            },
            {
                "product_id": "SKU_B_04",
                "product_name": "SHELL Gadus S2 V220 2 0.4кг",
                "base_name": "SHELL Gadus S2 V220 2",
                "short_name": "Gadus S2 V220 2 0.4кг",
                "brand": "Shell",
                "product_type": "Смазка",
                "product_category": "Смазки",
                "product_subcategory": "Литиевые",
                "business_line": "IND",
                "product_family": "Gadus",
                "container_type": "ТУБА",
            },
        ]
    )
    companies = pd.DataFrame(
        [
            {"customer_id": "C1", "industry_branch": "Агро", "description": "Трактори та підшипники"},
            {"customer_id": "C2", "industry_branch": "Агро", "description": "Вантажівки"},
        ]
    )
    purchases = pd.DataFrame(
        [
            # C1 buys BOTH 20L and 209L of Rimula on 2025-05-01 -> must collapse to 1 base_name event!
            {"customer_id": "C1", "purchase_date": "2025-05-01", "product_id": "SKU_A_20", "quantity": 2.0},
            {"customer_id": "C1", "purchase_date": "2025-05-01", "product_id": "SKU_A_209", "quantity": 1.0},
            # C1 buys multi-grade grease Gadus S2 V220 2 on 2025-06-01
            {"customer_id": "C1", "purchase_date": "2025-06-01", "product_id": "SKU_B_04", "quantity": 3.0},
            # Future events >= 2025-07-01 (must never leak into 2025-07-01!)
            {"customer_id": "C1", "purchase_date": "2025-07-01", "product_id": "SKU_B_04", "quantity": 5.0},
            {"customer_id": "C1", "purchase_date": "2025-08-01", "product_id": "SKU_A_209", "quantity": 4.0},
            {"customer_id": "C2", "purchase_date": "2025-05-10", "product_id": "SKU_A_20", "quantity": 1.0},
            {"customer_id": "C2", "purchase_date": "2025-06-15", "product_id": "SKU_B_04", "quantity": 2.0},
        ]
    )
    commercial = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "purchase_date": "2025-05-01",
                "product_id": "SKU_A_20",
                "quantity": 2.0,
                "manager_id": "M1",
                "payment_terms": "ДН 14",
                "payment_type": "БЕЗНАЛ",
                "full_payment_date": "2025-05-10",
                "revenue": 1000.0,
                "cost": -700.0,
                "cost_usd": -18.0,
                "doc_bonus": 0.0,
                "warehouse": "W1",
            },
            {
                "customer_id": "C1",
                "purchase_date": "2025-08-01",
                "product_id": "SKU_A_209",
                "quantity": 4.0,
                "manager_id": "M2",
                "payment_terms": "ПРЕДОПЛАТА",
                "payment_type": "НАЛ",
                "full_payment_date": "2025-08-01",
                "revenue": 4000.0,
                "cost": -3000.0,
                "cost_usd": -75.0,
                "doc_bonus": 100.0,
                "warehouse": "W1",
            },
        ]
    )
    product_ids = ["SKU_A_20", "SKU_A_209", "SKU_B_04"]
    base_names = ["Shell Rimula R4 X 15W-40", "SHELL Gadus S2 V220 2"]
    universe = {
        "base_names": base_names,
        "num_base_names": 2,
        "bn_to_idx": {b: i for i, b in enumerate(base_names)},
        "bn_to_member_pid_indices": [[0, 1], [2]],
        "all_pid_to_bn": {
            "SKU_A_20": base_names[0],
            "SKU_A_209": base_names[0],
            "SKU_B_04": base_names[1],
        },
    }
    reps = [0, 2]
    return catalogue, companies, purchases, commercial, product_ids, universe, reps


def test_feature_schemas_clean_and_no_duplicates_or_month():
    assert len(NATIVE_53_FEATURE_NAMES) == 53
    assert len(CANDIDATE_38_FEATURE_NAMES) == 38
    assert len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES) == 11
    assert all("month" not in n for n in CANDIDATE_38_FEATURE_NAMES)
    assert "base_name_orders_90d" in CANDIDATE_38_FEATURE_NAMES
    # Ensure exact duplicates & customer-only features are NOT in candidate features
    for forbidden in (
        "category_never_seen",
        "family_never_seen",
        "same_viscosity_grade_bought",
        "base_name_hierarchical_backoff",
        *CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    ):
        assert forbidden not in CANDIDATE_38_FEATURE_NAMES
    assert len(CONTAINER_BUCKETS) == 12
    assert len(VOLUME_BUCKETS) == 8
    assert len(APPLICATION_BUCKETS) == 11
    assert len(VISCOSITY_BUCKETS) == 34


def test_parse_viscosity_grades_fixes_nlgi0_bug_and_keeps_both_iso_and_nlgi():
    # Must NOT match '0' from '0.4кг' as NLGI0, and MUST keep both ISO220 and NLGI2!
    assert parse_viscosity_grades("SHELL Gadus S2 V220 2 0.4кг") == {"ISO220", "NLGI2"}
    assert parse_viscosity_grades("SHELL Gadus S5 T460 1.5 18кг") == {"ISO460", "NLGI1.5"}
    assert parse_viscosity_grades("Shell Rimula R4 X 15W-40 20L") == {"15W-40"}


def test_volume_bucket_boundaries():
    assert volume_bucket_index(None) == 7
    assert volume_bucket_index(0.4) == 0
    assert volume_bucket_index(1.0) == 1
    assert volume_bucket_index(4.0) == 2
    assert volume_bucket_index(20.0) == 3
    assert volume_bucket_index(55.0) == 4
    assert volume_bucket_index(209.0) == 5
    assert volume_bucket_index(1000.0) == 6


def test_native_engine_split_causality_and_multipack_collapse():
    catalogue, companies, purchases, commercial, product_ids, universe, reps = _synthetic_fixtures()
    as_of = pd.Timestamp("2025-07-01")

    full_eng = NativeBaseNameFeatureEngine(
        purchases, commercial, catalogue, companies, product_ids, universe, reps
    )
    trunc_purchases = purchases[pd.to_datetime(purchases["purchase_date"]) < as_of].copy()
    trunc_comm = commercial[pd.to_datetime(commercial["purchase_date"]) < as_of].copy()
    trunc_eng = NativeBaseNameFeatureEngine(
        trunc_purchases, trunc_comm, catalogue, companies, product_ids, universe, reps
    )

    cand_full, cust_full = full_eng.build_split("C1", as_of)
    cand_trunc, cust_trunc = trunc_eng.build_split("C1", as_of)
    assert cand_full.shape == (2, 38)
    assert cust_full.shape == (11,)
    np.testing.assert_allclose(cand_full, cand_trunc, atol=1e-6)
    np.testing.assert_allclose(cust_full, cust_trunc, atol=1e-6)

    # Multi-pack collapse: 1 Rimula event (20L + 209L same day) + 1 Gadus event = 2 events total -> 0.50 each!
    cat_share_idx = CANDIDATE_38_FEATURE_NAMES.index("user_cat_order_share")
    assert np.isclose(cand_full[0, cat_share_idx], 0.5, atol=1e-6)
    assert np.isclose(cand_full[1, cat_share_idx], 0.5, atol=1e-6)

    # Multi-value viscosity share: Gadus (ISO220 & NLGI2) was 1 of 2 events -> 0.50!
    visc_share_idx = CANDIDATE_38_FEATURE_NAMES.index("user_viscosity_order_share")
    assert np.isclose(cand_full[0, visc_share_idx], 0.5, atol=1e-6)
    assert np.isclose(cand_full[1, visc_share_idx], 0.5, atol=1e-6)

    # Merged base_name profile features (7D): 2 merged base_name events (NOT 3 raw SKU lines!), 2 distinct base_names (NOT 3 SKUs!)
    _, cust_18 = full_eng.build_split("C1", as_of, include_merged_profile=True)
    assert cust_18.shape == (18,)
    assert np.isclose(cust_18[11], np.log1p(2) / np.log1p(500.0), atol=1e-6)  # total_bn_events = 2
    assert np.isclose(cust_18[13], np.log1p(2) / np.log1p(100.0), atol=1e-6)  # distinct_base_names = 2

    # Manager rolling 90d shares MUST also collapse same-day multi-pack lines!
    comm_mp = pd.concat(
        [
            commercial,
            pd.DataFrame(
                [
                    {
                        "customer_id": "C1",
                        "purchase_date": "2025-05-01",
                        "product_id": "SKU_A_209",  # 2nd pack of Rimula sold by M1 on same day!
                        "quantity": 1.0,
                        "manager_id": "M1",
                        "payment_terms": "ДН 14",
                        "payment_type": "БЕЗНАЛ",
                        "full_payment_date": "2025-05-10",
                        "revenue": 500.0,
                        "cost": -350.0,
                        "cost_usd": -9.0,
                        "doc_bonus": 0.0,
                        "warehouse": "W1",
                    },
                    {
                        "customer_id": "C1",
                        "purchase_date": "2025-06-01",
                        "product_id": "SKU_B_04",  # 1 Gadus event sold by M1
                        "quantity": 3.0,
                        "manager_id": "M1",
                        "payment_terms": "ДН 14",
                        "payment_type": "БЕЗНАЛ",
                        "full_payment_date": "2025-06-10",
                        "revenue": 600.0,
                        "cost": -400.0,
                        "cost_usd": -10.0,
                        "doc_bonus": 0.0,
                        "warehouse": "W1",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    eng_mp = NativeBaseNameFeatureEngine(
        purchases, comm_mp, catalogue, companies, product_ids, universe, reps
    )
    cand_mp, _ = eng_mp.build_split("C1", as_of)
    mgr_bn_idx = CANDIDATE_38_FEATURE_NAMES.index("mgr_base_name_share_90d")
    # 1 merged Rimula event + 1 Gadus event -> 0.50 each (NOT 2/3 vs 1/3!)
    assert np.isclose(cand_mp[0, mgr_bn_idx], 0.5, atol=1e-6)
    assert np.isclose(cand_mp[1, mgr_bn_idx], 0.5, atol=1e-6)


def test_native_engine_blocks_heldout_dates_by_default():
    catalogue, companies, purchases, commercial, product_ids, universe, reps = _synthetic_fixtures()
    eng = NativeBaseNameFeatureEngine(
        purchases, commercial, catalogue, companies, product_ids, universe, reps
    )
    with pytest.raises(ValueError, match="Development predictions only"):
        eng.build_split("C1", pd.Timestamp("2026-02-19"))


def test_clean_ranker_zero_dead_weights_dynamic_token0_and_cold_start():
    m = CleanStage5BaseNameRanker(
        num_base_names=10,
        num_candidate_features=38,
        num_customer_features=11,
        num_cities=5,
        num_regions=5,
        num_industries=5,
        num_sales_branches=0,
        num_categories=6,
        num_subcategories=6,
        num_business_lines=4,
        aux_60d_enabled=False,
    ).eval()

    # Verify zero dead weights and zero sales_branch embedding exist
    param_names = [n for n, _ in m.named_parameters()]
    assert not any("product_text_proj" in n for n in param_names)
    assert not any("habit_proj" in n for n in param_names)
    assert not any("sales_branch" in n for n in param_names)
    assert not any(n.startswith("base.") for n in param_names)

    # Cold-start basket (empty history, attention_mask all 0) must produce finite logits with zero NaNs
    B, C, L = 2, 10, 32
    base_inp = dict(
        text_embedding=torch.randn(B, 1024),
        city_id=torch.tensor([1, 2]),
        region_id=torch.tensor([1, 1]),
        industry_id=torch.tensor([2, 3]),
        customer_features=torch.zeros(B, 11),
        sequence=torch.zeros(B, L, 9),
        attention_mask=torch.zeros(B, L, dtype=torch.long),
        seq_product_ids=torch.zeros(B, L, dtype=torch.long),
        seq_category_ids=torch.zeros(B, L, dtype=torch.long),
        seq_subcategory_ids=torch.zeros(B, L, dtype=torch.long),
        seq_business_line_ids=torch.zeros(B, L, dtype=torch.long),
        seq_purchase_dates=torch.zeros(B, L),
        prediction_day=torch.tensor([500.0, 500.0]),
        candidate_features=torch.randn(B, C, 38),
        candidate_vectors=torch.randn(B, C, 128),
        candidate_mask=torch.ones(B, C, dtype=torch.bool),
    )
    with torch.no_grad():
        out1 = m(**base_inp, return_diagnostics=True)
        # Mutate ONLY fresh 11D current customer_features -> Token 0 memory & logits MUST change!
        mod_inp = {**base_inp, "customer_features": torch.ones(B, 11)}
        out2 = m(**mod_inp, return_diagnostics=True)

    assert torch.isfinite(out1.logits).all()
    assert torch.isfinite(out2.logits).all()
    assert not torch.allclose(out1.memory[:, 0], out2.memory[:, 0])
    assert not torch.allclose(out1.logits, out2.logits)
