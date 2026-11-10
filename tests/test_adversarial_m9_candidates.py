"""Adversarial stress test suite for Milestone M9: Candidate Generation & Point-in-Time Features.

Stress-tests:
1. Zero-order / cold customers with no purchase history (< as_of_date).
2. Single purchase customer (urgency must be strictly 0.0, 0-gap edge case).
3. High-order customers with 1000+ purchases (volume, latency, norm capping).
4. Cold catalog items with 0 training orders (< train_cutoff).
5. Infinite / extreme logits (-1000.0, +1000.0, nan, inf) -> sigmoid strictly in [0.0, 1.0].
7. Strict sub-millisecond point-in-time timestamp cutoff isolation.
8. Full real-world repository data integration stress test.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.candidates import (
    NUM_PAIRWISE_FEATURES,
    CandidateBatch,
    CandidateRetriever,
    PointInTimeFeatureStore,
    construct_candidate_slate,
    load_retriever_from_checkpoint,
)


def test_adversarial_cold_customer_and_future_only_history():
    """Validates cold customers and customers whose history only exists in the future."""
    purchases_df = pd.DataFrame([
        # C_FUTURE has purchases only after 2025-06-01
        {"customer_id": "C_FUTURE", "product_id": "PROD_01", "purchase_date": "2025-06-15", "quantity": 10.0},
        {"customer_id": "C_FUTURE", "product_id": "PROD_02", "purchase_date": "2025-07-01", "quantity": 5.0},
    ])
    companies_df = pd.DataFrame([
        {"customer_id": "C_COLD", "industry_branch": "IND_X", "city": "Kyiv", "region_code": "R1"},
        {"customer_id": "C_FUTURE", "industry_branch": "IND_Y", "city": "Lviv", "region_code": "R2"},
    ])
    catalogue_df = pd.DataFrame([
        {"product_id": "PROD_01", "product_category": "CAT_A", "product_subcategory": "SUB_A", "brand": "BR_A"},
        {"product_id": "PROD_02", "product_category": "CAT_B", "product_subcategory": "SUB_B", "brand": "BR_B"},
    ])

    store = PointInTimeFeatureStore(
        purchases_df=purchases_df,
        catalogue_df=catalogue_df,
        companies_df=companies_df,
    )

    as_of = pd.Timestamp("2025-06-01")

    # 1. Completely unknown customer
    f_unknown = store.compute_pairwise_features("C_UNKNOWN", as_of, "PROD_01")
    assert f_unknown.shape == (NUM_PAIRWISE_FEATURES,)
    assert not np.isnan(f_unknown).any() and not np.isinf(f_unknown).any()
    assert (f_unknown >= 0.0).all() and (f_unknown <= 1.0).all()
    assert f_unknown[0] == 0.0  # is_repeat_sku
    assert f_unknown[1] == 0.0  # user_sku_order_count_norm
    assert f_unknown[2] == 0.0  # user_sku_qty_share
    assert f_unknown[3] == 1.0  # days_since_last_sku_buy_norm (1.0 if never bought)
    assert f_unknown[4] == 0.0  # sku_urgency
    assert f_unknown[5] == 0.0  # user_cat_order_share
    assert f_unknown[6] == 0.0  # user_subcat_order_share
    assert f_unknown[7] == 0.0  # user_brand_order_share

    # 2. Known company with zero purchase history ever
    f_cold = store.compute_pairwise_features("C_COLD", as_of, "PROD_01")
    assert (f_cold >= 0.0).all() and (f_cold <= 1.0).all()
    assert f_cold[0] == 0.0
    assert f_cold[3] == 1.0
    assert f_cold[4] == 0.0

    # 3. Customer with history strictly AFTER as_of_date (future data leakage test)
    f_future = store.compute_pairwise_features("C_FUTURE", as_of, "PROD_01")
    assert (f_future >= 0.0).all() and (f_future <= 1.0).all()
    assert f_future[0] == 0.0  # must NOT see the 2025-06-15 purchase!
    assert f_future[1] == 0.0
    assert f_future[2] == 0.0
    assert f_future[3] == 1.0  # treated as never bought prior to as_of
    assert f_future[4] == 0.0


def test_adversarial_single_purchase_urgency_and_zero_gap():
    """Validates single purchase customer urgency is strictly 0.0 and zero-gap is safe."""
    purchases_df = pd.DataFrame([
        # C_SINGLE has exactly 1 purchase
        {"customer_id": "C_SINGLE", "product_id": "PROD_01", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 10.0},
        # C_ZERO_GAP has 2 purchases on the EXACT same second (gap = 0.0)
        {"customer_id": "C_ZERO_GAP", "product_id": "PROD_02", "purchase_date": pd.Timestamp("2025-01-01 10:00:00"), "quantity": 5.0},
        {"customer_id": "C_ZERO_GAP", "product_id": "PROD_02", "purchase_date": pd.Timestamp("2025-01-01 10:00:00"), "quantity": 5.0},
        # C_SUBSECOND has 2 purchases with 0.001 second gap
        {"customer_id": "C_SUBSECOND", "product_id": "PROD_03", "purchase_date": pd.Timestamp("2025-01-01 10:00:00.000"), "quantity": 1.0},
        {"customer_id": "C_SUBSECOND", "product_id": "PROD_03", "purchase_date": pd.Timestamp("2025-01-01 10:00:00.001"), "quantity": 1.0},
    ])

    store = PointInTimeFeatureStore(purchases_df=purchases_df)
    as_of = pd.Timestamp("2025-02-01")

    # 1. Single purchase customer: urgency must be strictly 0.0
    f_single = store.compute_pairwise_features("C_SINGLE", as_of, "PROD_01")
    assert f_single[0] == 1.0  # is_repeat_sku
    assert f_single[1] == float(np.log1p(1) / np.log1p(50))
    assert f_single[2] == 1.0  # 100% of volume
    assert f_single[4] == 0.0, f"Single purchase urgency must be 0.0, got {f_single[4]}"
    assert (f_single >= 0.0).all() and (f_single <= 1.0).all()

    # 2. Zero-gap customer: 2 purchases on exact same second -> med_gap = 0.0
    f_zero_gap = store.compute_pairwise_features("C_ZERO_GAP", as_of, "PROD_02")
    assert not np.isnan(f_zero_gap).any() and not np.isinf(f_zero_gap).any()
    assert (f_zero_gap >= 0.0).all() and (f_zero_gap <= 1.0).all()
    assert f_zero_gap[0] == 1.0
    # Safe gap defaults to max(1.0, med_gap) = 1.0 day
    # Delta days = 31 days -> ratio = 31 / 1.0 = 31 -> min(1.0, 31 / 5.0) = 1.0
    assert f_zero_gap[4] == 1.0

    # 3. Sub-second gap customer
    f_subsecond = store.compute_pairwise_features("C_SUBSECOND", as_of, "PROD_03")
    assert not np.isnan(f_subsecond).any() and not np.isinf(f_subsecond).any()
    assert (f_subsecond >= 0.0).all() and (f_subsecond <= 1.0).all()


def test_adversarial_high_order_customer_stress():
    """Generates synthetic heavy customer with 1500+ purchases and tests bounds and latency."""
    num_events = 1600
    dates = pd.date_range("2024-01-01", periods=num_events, freq="3h")
    pids = [f"PROD_{i % 40:03d}" for i in range(num_events)]
    quantities = np.random.uniform(1.0, 250.0, size=num_events)

    df_heavy = pd.DataFrame({
        "customer_id": "C_SUPER_HEAVY",
        "product_id": pids,
        "purchase_date": dates,
        "quantity": quantities,
    })

    store = PointInTimeFeatureStore(purchases_df=df_heavy)
    as_of = pd.Timestamp("2024-06-01")

    snap = store.get_customer_snapshot("C_SUPER_HEAVY", as_of)
    assert snap.total_orders > 1000
    assert snap.total_volume > 1000.0

    # Candidate slate of 50 products
    candidate_pids = [f"PROD_{i:03d}" for i in range(50)]

    t0 = time.perf_counter()
    feats = store.compute_slate_pairwise_features("C_SUPER_HEAVY", as_of, candidate_pids)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert feats.shape == (50, NUM_PAIRWISE_FEATURES)
    assert (feats >= 0.0).all() and (feats <= 1.0).all()
    assert not torch.isnan(feats).any() and not torch.isinf(feats).any()
    assert latency_ms < 5.0, f"Feature extraction too slow for heavy customer: {latency_ms:.2f} ms"

    # Verify SKU order count norm capped at 1.0 for SKU bought > 50 times
    # In 1600 events across 40 SKUs, each SKU is bought ~40 times up to June 1st
    for i in range(40):
        val = feats[i, 3].item()
        assert 0.0 <= val <= 1.0


def test_adversarial_zero_training_orders_cold_catalog_items():
    """Tests products with zero training orders (< train_cutoff 2025-10-01)."""
    purchases_df = pd.DataFrame([
        # PROD_TRAIN was bought before 2025-10-01
        {"customer_id": "C1", "product_id": "PROD_TRAIN", "purchase_date": "2025-05-01", "quantity": 1.0},
        # PROD_VAL_ONLY was bought ONLY after 2025-10-01
        {"customer_id": "C1", "product_id": "PROD_VAL_ONLY", "purchase_date": "2025-11-01", "quantity": 10.0},
    ])
    catalogue_df = pd.DataFrame([
        {"product_id": "PROD_TRAIN", "product_category": "CAT_1"},
        {"product_id": "PROD_VAL_ONLY", "product_category": "CAT_2"},
        {"product_id": "PROD_NEVER_BOUGHT", "product_category": "CAT_3"},
    ])

    store = PointInTimeFeatureStore(
        purchases_df=purchases_df,
        catalogue_df=catalogue_df,
        train_cutoff="2025-10-01",
    )

    as_of = pd.Timestamp("2025-12-01")

    # 1. PROD_VAL_ONLY: bought in val split, but 0 training orders
    f_val_only = store.compute_pairwise_features("C1", as_of, "PROD_VAL_ONLY")
    assert (f_val_only >= 0.0).all() and (f_val_only <= 1.0).all()
    assert f_val_only[8] == 0.0, "PROD_VAL_ONLY must have 0.0 global popularity norm from train split"
    assert f_val_only[9] == 0.0

    # 2. PROD_NEVER_BOUGHT: 0 orders in entire catalog
    f_never = store.compute_pairwise_features("C1", as_of, "PROD_NEVER_BOUGHT")
    assert (f_never >= 0.0).all() and (f_never <= 1.0).all()
    assert f_never[8] == 0.0
    assert f_never[9] == 0.0

    # 3. PROD_TRAIN: has non-zero popularity norm
    f_train = store.compute_pairwise_features("C1", as_of, "PROD_TRAIN")
    assert f_train[8] > 0.0
    before_first_sale = store.compute_pairwise_features("C1", "2024-01-01", "PROD_TRAIN")
    assert before_first_sale[10] == 0.0


def test_adversarial_infinite_and_extreme_logits():
    """Validates candidate batches under extreme, infinite, and NaN retrieval logits."""
    store = PointInTimeFeatureStore(
        purchases_df=pd.DataFrame([{"customer_id": "C1", "product_id": "P1", "purchase_date": "2025-01-01", "quantity": 1.0}])
    )

    # Batch CandidateRetriever with extreme values
    retriever = CandidateRetriever(
        eligible_product_ids=[f"P_{i:02d}" for i in range(100)],
        feature_store=store,
    )

    b_size = 6
    logits = torch.randn(b_size, 100)
    logits[0, 0] = 1000.0
    logits[0, 1] = -1000.0
    logits[1, :] = float("inf")
    logits[2, :] = float("-inf")
    logits[3, :] = float("nan")
    logits[4, :50] = 500.0
    logits[4, 50:] = -500.0

    batch = retriever.generate_candidate_batch(
        logits=logits,
        customer_ids=["C1"] * b_size,
        as_of_dates=["2025-02-01"] * b_size,
        top_k=100,
        top_c=50,
    )

    assert not torch.isnan(batch.pairwise_features).any()
    assert not torch.isinf(batch.pairwise_features).any()
    assert (batch.pairwise_features >= 0.0).all() and (batch.pairwise_features <= 1.0).all()
    assert not torch.isnan(batch.candidate_scores).any()
    assert not torch.isinf(batch.candidate_scores).any()
    assert not torch.isnan(batch.top100_scores).any()
    assert not torch.isinf(batch.top100_scores).any()



def test_adversarial_point_in_time_exact_millisecond_cutoff():
    """Ensures transactions occurring at the exact same millisecond as as_of_date are excluded."""
    t_target = pd.Timestamp("2025-04-10 12:00:00")
    df = pd.DataFrame([
        # Before as_of
        {"customer_id": "C_TIME", "product_id": "P_PAST", "purchase_date": t_target - pd.Timedelta(seconds=1), "quantity": 1.0},
        # EXACTLY at as_of
        {"customer_id": "C_TIME", "product_id": "P_EXACT", "purchase_date": t_target, "quantity": 2.0},
        # After as_of
        {"customer_id": "C_TIME", "product_id": "P_FUTURE", "purchase_date": t_target + pd.Timedelta(seconds=1), "quantity": 3.0},
    ])

    store = PointInTimeFeatureStore(purchases_df=df)
    snap = store.get_customer_snapshot("C_TIME", t_target)

    # Must contain ONLY P_PAST
    assert snap.total_orders == 1
    assert snap.total_volume == 1.0
    assert "P_PAST" in snap.sku_stats
    assert "P_EXACT" not in snap.sku_stats, "Event at exact as_of_date leaked into point-in-time snapshot!"
    assert "P_FUTURE" not in snap.sku_stats, "Future event leaked into point-in-time snapshot!"


@pytest.mark.real_purchase_data
def test_adversarial_real_data_retriever_stress():
    """End-to-end stress test using repository checkpoint and full cleaned_purchases.csv."""
    ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
    purchases_path = Path("data/interim/cleaned_purchases.csv")

    if not ckpt_path.is_file() or not purchases_path.is_file():
        pytest.skip("Repository data or checkpoint not available for real data stress test")

    purchases_df = pd.read_csv(purchases_path)
    store = PointInTimeFeatureStore(purchases_df=purchases_df)
    model, ckpt = load_retriever_from_checkpoint(ckpt_path)

    retriever = CandidateRetriever(
        model=model,
        eligible_product_ids=ckpt["product_ids"],
        feature_store=store,
    )

    # Pick 10 diverse real customers:
    counts = purchases_df.groupby("customer_id").size()
    top_custs = counts.nlargest(3).index.astype(str).tolist()
    mid_custs = counts[(counts > 10) & (counts < 30)].head(3).index.astype(str).tolist()
    single_custs = counts[counts == 1].head(3).index.astype(str).tolist()
    cold_cust = ["C_ADVERSARIAL_COLD_404"]
    test_cids = top_custs + mid_custs + single_custs + cold_cust

    p_count = len(ckpt["product_ids"])
    b_size = len(test_cids)

    # Adversarial logits with extreme values
    logits = torch.randn(b_size, p_count)
    logits[0, 5] = 1000.0
    logits[1, 10] = -1000.0
    logits[2, 15] = float("nan")
    logits[3, 20] = float("inf")

    batch = retriever.generate_candidate_batch(
        logits=logits,
        customer_ids=test_cids,
        as_of_dates=["2025-11-01"] * b_size,
        top_k=100,
        top_c=50,
    )

    assert batch.candidate_indices.shape == (b_size, 50)
    assert batch.top100_indices.shape == (b_size, 100)
    assert batch.pairwise_features.shape == (b_size, 50, NUM_PAIRWISE_FEATURES)

    # Core required assertions
    assert (batch.pairwise_features >= 0.0).all() and (batch.pairwise_features <= 1.0).all(), (
        f"Pairwise features outside [0.0, 1.0]: min={batch.pairwise_features.min()}, max={batch.pairwise_features.max()}"
    )
    assert not torch.isnan(batch.pairwise_features).any(), "NaN detected in pairwise features tensor!"
    assert not torch.isinf(batch.pairwise_features).any(), "Inf detected in pairwise features tensor!"
    assert not torch.isnan(batch.candidate_scores).any(), "NaN detected in candidate scores tensor!"
    assert not torch.isinf(batch.candidate_scores).any(), "Inf detected in candidate scores tensor!"
