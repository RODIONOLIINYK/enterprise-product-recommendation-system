"""Empirical Adversarial Stress Suite for Milestone M9: Candidate Generation & Point-in-Time Pairwise Features.

Author: Challenger M9_2
Evaluates:
1. Candidate slate construction under adversarial conditions:
   - Evaluation mode: confirm NO positive items are ever injected and slate is strictly top-50 from Stage 1.
2. Temporal isolation:
   - Modify future purchases (on or after as_of_date) and verify pairwise features are 100% BIT-FOR-BIT IDENTICAL.
   - Exact timestamp boundary test (purchase_date == as_of_date).
   - Subsecond precision boundary testing.
3. Mixed datetime format parsing bug reproduction.
4. Slate feature extraction throughput across 500 candidate slates to verify sub-millisecond performance.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.candidates import (
    NUM_PAIRWISE_FEATURES,
    PAIRWISE_FEATURE_NAMES,
    CandidateBatch,
    CandidateRetriever,
    PointInTimeFeatureStore,
    construct_candidate_slate,
    load_retriever_from_checkpoint,
)


# ==============================================================================
# Task 1: Adversarial Slate Construction Tests
# ==============================================================================

def test_slate_construction_adversarial_zero_positives():
    num_products = 100
    pids = [f"PROD_{i:03d}" for i in range(num_products)]

    logits = torch.linspace(100.0, 1.0, num_products)

    # 1. Empty list []
    slate_empty = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=[],
        top_k=100,
        top_c=50,
    )
    assert len(slate_empty["candidate_indices"]) == 50
    assert slate_empty["candidate_indices"] == list(range(50))
    assert slate_empty["candidate_ranks"] == list(range(50))
    assert slate_empty["labels"] == [0.0] * 50
    assert sum(slate_empty["labels"]) == 0.0

    # 2. None
    slate_none = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=None,
        top_k=100,
        top_c=50,
    )
    assert len(slate_none["candidate_indices"]) == 50
    assert slate_none["candidate_indices"] == list(range(50))
    assert slate_none["candidate_ranks"] == list(range(50))
    assert slate_none["labels"] is None



def test_slate_construction_adversarial_all_50_items_positive():
    num_products = 100
    pids = [f"PROD_{i:03d}" for i in range(num_products)]

    logits = torch.linspace(100.0, 1.0, num_products)

    # All top-50 items are positive targets
    all_50_pos = pids[:50]
    slate = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=all_50_pos,
        top_k=100,
        top_c=50,
    )

    assert len(slate["candidate_indices"]) == 50
    assert slate["candidate_pids"] == all_50_pos
    assert slate["candidate_ranks"] == list(range(50))
    assert slate["labels"] == [1.0] * 50
    assert sum(slate["labels"]) == 50.0







def test_slate_construction_adversarial_eval_mode_strict_isolation():
    """Adversarial Test 1I: Evaluation mode strictly selects top-50 without injection."""
    num_products = 150
    pids = [f"PROD_{i:03d}" for i in range(num_products)]

    logits = torch.linspace(150.0, 1.0, num_products)

    # 1. Positives completely outside top-50 (ranks 60, 70, 80, 90)
    positives_outside = ["PROD_060", "PROD_070", "PROD_080", "PROD_090"]
    slate_eval = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=positives_outside,
        top_k=100,
        top_c=50,
    )

    assert slate_eval["candidate_pids"] == pids[:50]
    assert slate_eval["candidate_ranks"] == list(range(50))
    for p in positives_outside:
        assert p not in slate_eval["candidate_pids"]
    assert slate_eval["labels"] == [0.0] * 50

    # 2. Positives partially inside top-50 (e.g. ranks 5, 25, 60, 70)
    positives_mixed = ["PROD_005", "PROD_025", "PROD_060", "PROD_070"]
    slate_mixed = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=positives_mixed,
        top_k=100,
        top_c=50,
    )

    assert slate_mixed["candidate_pids"] == pids[:50]
    assert slate_mixed["labels"][5] == 1.0
    assert slate_mixed["labels"][25] == 1.0
    assert sum(slate_mixed["labels"]) == 2.0
    assert "PROD_060" not in slate_mixed["candidate_pids"]
    assert "PROD_070" not in slate_mixed["candidate_pids"]


# ==============================================================================
# Task 2: Temporal Isolation Tests (Bit-for-Bit Identical)
# ==============================================================================

def test_temporal_isolation_future_purchases_modification():
    """Adversarial Test 2A: Modifying future purchases yields 100% BIT-FOR-BIT IDENTICAL features."""
    # Build baseline dataset
    base_rows = []
    # Historical events strictly before as_of_date 2025-10-15
    for i in range(20):
        base_rows.append({
            "customer_id": "CUST_STRESS",
            "product_id": f"P_{i % 5:02d}",
            "purchase_date": f"2025-01-{(i % 28) + 1:02d}",
            "quantity": float(i + 1),
            "product_category": f"CAT_{i % 3}",
            "product_subcategory": f"SUB_{i % 2}",
            "brand": f"BRAND_{i % 4}",
        })
    # Future events on or after as_of_date (2025-10-15)
    for i in range(10):
        base_rows.append({
            "customer_id": "CUST_STRESS",
            "product_id": f"P_{i % 5:02d}",
            "purchase_date": f"2025-11-{(i % 28) + 1:02d}",
            "quantity": 10.0,
            "product_category": f"CAT_{i % 3}",
            "product_subcategory": f"SUB_{i % 2}",
            "brand": f"BRAND_{i % 4}",
        })

    df_base = pd.DataFrame(base_rows)
    as_of = pd.Timestamp("2025-10-15 00:00:00")
    test_pids = [f"P_{i:02d}" for i in range(10)]

    # Compute baseline features
    store_base = PointInTimeFeatureStore(purchases_df=df_base)
    feat_base = store_base.compute_slate_pairwise_features(
        customer_id="CUST_STRESS",
        as_of_date=as_of,
        product_ids=test_pids,
    )
    snap_base = store_base.get_customer_snapshot("CUST_STRESS", as_of)

    # Adversarial Scenario 1: Add 500 massive future orders with 1,000,000 units on 2026-01-01
    df_future_spike = df_base.copy()
    spike_rows = []
    for j in range(500):
        spike_rows.append({
            "customer_id": "CUST_STRESS",
            "product_id": f"P_{j % 10:02d}",
            "purchase_date": "2026-01-01",
            "quantity": 1_000_000.0,
            "product_category": "CAT_999",
            "product_subcategory": "SUB_999",
            "brand": "BRAND_999",
        })
    df_future_spike = pd.concat([df_future_spike, pd.DataFrame(spike_rows)], ignore_index=True)

    store_spike = PointInTimeFeatureStore(purchases_df=df_future_spike)
    feat_spike = store_spike.compute_slate_pairwise_features(
        customer_id="CUST_STRESS",
        as_of_date=as_of,
        product_ids=test_pids,
    )
    snap_spike = store_spike.get_customer_snapshot("CUST_STRESS", as_of)

    # Adversarial Scenario 2: Delete ALL future purchases completely
    df_no_future = df_base[df_base["purchase_date"] < "2025-10-15"].copy()
    store_no_future = PointInTimeFeatureStore(purchases_df=df_no_future)
    feat_no_future = store_no_future.compute_slate_pairwise_features(
        customer_id="CUST_STRESS",
        as_of_date=as_of,
        product_ids=test_pids,
    )
    snap_no_future = store_no_future.get_customer_snapshot("CUST_STRESS", as_of)

    # Adversarial Scenario 3: Corrupt future product categories and quantities
    df_corrupted_future = df_base.copy()
    future_mask = df_corrupted_future["purchase_date"] >= "2025-10-15"
    df_corrupted_future.loc[future_mask, "quantity"] = -999999.0
    df_corrupted_future.loc[future_mask, "product_category"] = "CORRUPTED"
    store_corrupted = PointInTimeFeatureStore(purchases_df=df_corrupted_future)
    feat_corrupted = store_corrupted.compute_slate_pairwise_features(
        customer_id="CUST_STRESS",
        as_of_date=as_of,
        product_ids=test_pids,
    )

    # 1. Check Bit-for-bit identity across tensors
    assert torch.equal(feat_base, feat_spike), "Failed bit-for-bit equality after future volume spike!"
    assert torch.equal(feat_base, feat_no_future), "Failed bit-for-bit equality after removing future events!"
    assert torch.equal(feat_base, feat_corrupted), "Failed bit-for-bit equality after corrupting future events!"

    # 2. Check snapshot exact fields
    assert snap_base.total_volume == snap_spike.total_volume == snap_no_future.total_volume
    assert snap_base.total_orders == snap_spike.total_orders == snap_no_future.total_orders
    assert snap_base.cat_orders == snap_spike.cat_orders == snap_no_future.cat_orders
    assert snap_base.sku_stats == snap_spike.sku_stats == snap_no_future.sku_stats


def test_temporal_isolation_exact_boundary_instant():
    """Adversarial Test 2B: Event at EXACT timestamp purchase_date == as_of_date is excluded."""
    as_of = pd.Timestamp("2025-06-01 12:00:00")
    rows = [
        # 1 millisecond before as_of
        {"customer_id": "C_BOUND", "product_id": "P_PAST", "purchase_date": "2025-06-01 11:59:59.999", "quantity": 1.0},
        # EXACTLY at as_of
        {"customer_id": "C_BOUND", "product_id": "P_EXACT", "purchase_date": "2025-06-01 12:00:00.000", "quantity": 100.0},
        # 1 millisecond after as_of
        {"customer_id": "C_BOUND", "product_id": "P_FUTURE", "purchase_date": "2025-06-01 12:00:00.001", "quantity": 100.0},
    ]
    df = pd.DataFrame(rows)
    store = PointInTimeFeatureStore(purchases_df=df)
    snap = store.get_customer_snapshot("C_BOUND", as_of)

    # Only P_PAST should be in snapshot
    assert snap.total_orders == 1
    assert snap.total_volume == 1.0
    assert "P_PAST" in snap.sku_stats
    assert "P_EXACT" not in snap.sku_stats
    assert "P_FUTURE" not in snap.sku_stats

    # Check pairwise feature for P_EXACT: must have is_repeat_sku == 0.0 and days_since == 1.0
    feats_exact = store.compute_pairwise_features("C_BOUND", as_of, "P_EXACT")
    assert feats_exact[0] == 0.0  # is_repeat_sku
    assert feats_exact[3] == 1.0  # days_since_last_sku_buy_norm (never bought before as_of)


# ==============================================================================
# Task 3: Performance Benchmark Across 500 Candidate Slates
# ==============================================================================

@pytest.mark.real_purchase_data
def test_feature_extraction_throughput_500_slates():
    """Adversarial Test 3: Measure throughput across 500 candidate slates to verify sub-millisecond target."""
    purchases_path = Path("data/interim/cleaned_purchases.csv")
    if not purchases_path.is_file():
        pytest.skip("cleaned_purchases.csv not available")

    # Load 5000 real purchase records to build realistic feature store
    purchases_df = pd.read_csv(purchases_path, nrows=5000)
    store = PointInTimeFeatureStore(purchases_df=purchases_df)

    unique_customers = purchases_df["customer_id"].astype(str).unique().tolist()
    if len(unique_customers) < 10:
        unique_customers = unique_customers * 10

    catalogue_pids = purchases_df["product_id"].astype(str).unique()[:50].tolist()
    if len(catalogue_pids) < 50:
        catalogue_pids = [f"P_{i}" for i in range(50)]

    c_slate = 50
    as_of = pd.Timestamp("2025-11-01")

    # Run 500 candidate slate extractions across varying customers
    n_slates = 500
    latencies_ms = []

    # Warmup
    for i in range(10):
        cid = unique_customers[i % len(unique_customers)]
        store.compute_slate_pairwise_features(cid, as_of, catalogue_pids)

    t_start = time.perf_counter()
    for i in range(n_slates):
        cid = unique_customers[i % len(unique_customers)]
        t0 = time.perf_counter()
        feats = store.compute_slate_pairwise_features(cid, as_of, catalogue_pids)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        assert feats.shape == (c_slate, NUM_PAIRWISE_FEATURES)
        assert not torch.isnan(feats).any()
    t_total = time.perf_counter() - t_start

    avg_ms = np.mean(latencies_ms)
    p50_ms = np.percentile(latencies_ms, 50)
    p95_ms = np.percentile(latencies_ms, 95)
    p99_ms = np.percentile(latencies_ms, 99)
    max_ms = np.max(latencies_ms)
    total_ms = t_total * 1000.0

    print(
        f"\nThroughput Benchmark Across {n_slates} Candidate Slates (50 candidates each):\n"
        f"  Total time: {total_ms:.2f} ms for {n_slates} slates ({n_slates * c_slate} items)\n"
        f"  Mean latency: {avg_ms:.4f} ms/slate\n"
        f"  p50 latency:  {p50_ms:.4f} ms/slate\n"
        f"  p95 latency:  {p95_ms:.4f} ms/slate\n"
        f"  p99 latency:  {p99_ms:.4f} ms/slate\n"
        f"  Max latency:  {max_ms:.4f} ms/slate\n"
    )

    # Sub-millisecond requirement: mean latency must be strictly < 1.0 ms
    assert avg_ms < 1.0, f"Mean latency {avg_ms:.4f} ms exceeded 1.0 ms target!"
    assert p95_ms < 1.0, f"p95 latency {p95_ms:.4f} ms exceeded 1.0 ms target!"


# ==============================================================================
# Task 4: Candidate Pairwise & Slate Challenger (Iteration 2) Stress Suite
# ==============================================================================


def test_slate_dedup_all_50_items_already_in_top50_preserves_ranking():
    """Challenger R2 Test 2: Basket with all 50 items already in top-50 candidates.

    Verifies:
    1. Zero replacements are made.
    2. All 50 candidates preserve retriever ranks 0..49.
    3. Binary labels are all 1.0 (sum == 50.0).
    4. Unique candidate count is strictly 50.
    """
    num_products = 100
    pids = [f"PROD_{i:03d}" for i in range(num_products)]

    logits = torch.linspace(100.0, 1.0, num_products)

    top50_basket = pids[:50]

    slate = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=top50_basket,
        top_k=100,
        top_c=50,
    )

    assert slate["candidate_pids"] == top50_basket
    assert slate["candidate_ranks"] == list(range(50))
    assert slate["labels"] == [1.0] * 50
    assert sum(slate["labels"]) == 50.0
    assert len(set(slate["candidate_pids"])) == 50



def test_slate_eval_mode_pure_retrieval_ranking_preserved():
    """Challenger R2 Test 4: Slate construction preserves the retrieved list under all conditions.

    Verifies pure Stage 1 retrieval ranking is preserved regardless of:
    - 5 duplicate SKUs outside top-50
    - 10 missing positives outside top-50
    - Mix of duplicate inside/outside top-50 items
    """
    num_products = 120
    pids = [f"PROD_{i:03d}" for i in range(num_products)]

    logits = torch.linspace(120.0, 1.0, num_products)

    # Condition A: 5 duplicate items outside top-50
    slate_a = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=["PROD_088"] * 5,
        top_k=100,
        top_c=50,
    )
    assert slate_a["candidate_pids"] == pids[:50]
    assert slate_a["candidate_ranks"] == list(range(50))
    assert "PROD_088" not in slate_a["candidate_pids"]
    assert slate_a["labels"] == [0.0] * 50

    # Condition B: 10 missing items outside top-50
    missing_10 = [f"PROD_{r:03d}" for r in [55, 60, 65, 70, 75, 80, 85, 90, 95, 99]]
    slate_b = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=missing_10,
        top_k=100,
        top_c=50,
    )
    assert slate_b["candidate_pids"] == pids[:50]
    assert slate_b["candidate_ranks"] == list(range(50))
    for pid in missing_10:
        assert pid not in slate_b["candidate_pids"]
    assert slate_b["labels"] == [0.0] * 50

    # Condition C: Mixed inside/outside duplicates
    mixed = ["PROD_005"] * 3 + ["PROD_088"] * 5
    slate_c = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=mixed,
        top_k=100,
        top_c=50,
    )
    assert slate_c["candidate_pids"] == pids[:50]
    assert slate_c["candidate_ranks"] == list(range(50))
    assert slate_c["labels"][5] == 1.0
    assert sum(slate_c["labels"]) == 1.0
    assert "PROD_088" not in slate_c["candidate_pids"]


def test_mixed_datetime_robustness_temporal_ordering():
    """Challenger R2 Test 6: Transaction logs containing mixed date formats.

    Verifies:
    1. Transaction logs with mixed formats (%Y-%m-%d, %Y-%m-%d %H:%M:%S, ISO8601 %Y-%m-%dT%H:%M:%S,
       fractional seconds %Y-%m-%d %H:%M:%S.%f).
    2. PointInTimeFeatureStore ingests all rows without throwing exceptions.
    3. Events are strictly sorted chronologically across all mixed formats.
    4. Subsecond precision is respected during point-in-time snapshot filtering.
    """
    mixed_records = [
        {"customer_id": "C_CHALLENGER", "product_id": "P_SUBSEC_2", "purchase_date": "2025-05-10 14:00:00.654321", "quantity": 1.0},
        {"customer_id": "C_CHALLENGER", "product_id": "P_DATE_ONLY", "purchase_date": "2025-01-01", "quantity": 2.0},
        {"customer_id": "C_CHALLENGER", "product_id": "P_ISO8601", "purchase_date": "2025-04-12T18:45:30", "quantity": 3.0},
        {"customer_id": "C_CHALLENGER", "product_id": "P_SPACE_TIME", "purchase_date": "2025-03-20 08:15:00", "quantity": 4.0},
        {"customer_id": "C_CHALLENGER", "product_id": "P_SUBSEC_1", "purchase_date": "2025-05-10 14:00:00.123456", "quantity": 5.0},
    ]
    # Ingest shuffled mixed datetime dataframe
    df = pd.DataFrame(mixed_records)
    store = PointInTimeFeatureStore(purchases_df=df)

    events = store.customer_events["C_CHALLENGER"]
    dates = store.customer_event_dates["C_CHALLENGER"]

    # Ingestion must succeed with all 5 events
    assert len(events) == 5
    assert len(dates) == 5

    # Strictly monotonic order verified across mixed formats
    expected_order = [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-03-20 08:15:00"),
        pd.Timestamp("2025-04-12 18:45:30"),
        pd.Timestamp("2025-05-10 14:00:00.123456"),
        pd.Timestamp("2025-05-10 14:00:00.654321"),
    ]
    assert dates == expected_order

    # Query snapshot between the two subsecond events: 2025-05-10 14:00:00.500000
    snap_between = store.get_customer_snapshot("C_CHALLENGER", "2025-05-10 14:00:00.500000")
    assert snap_between.total_orders == 4
    assert "P_SUBSEC_1" in snap_between.sku_stats
    assert "P_SUBSEC_2" not in snap_between.sku_stats


def test_temporal_boundary_isolation_exact_date_and_cold_customers():
    """Challenger R2 Test 7: Temporal boundary isolation and cold customer defaults.

    Verifies:
    1. Order placed at date T is NOT visible when querying features as of date T (< as_of_date).
    2. Cold customers (0 prior transactions) return well-defined default features
       (all in [0, 1], zero urgency, days_since_last = 1.0, with NO NaN/Inf values).
    3. Pairwise features for cold customer are valid and properly normalized.
    """
    as_of_t = pd.Timestamp("2025-06-15 10:30:00")
    purchases = pd.DataFrame([
        # Order placed at EXACT timestamp T
        {"customer_id": "C_WARM", "product_id": "P_TARGET", "purchase_date": "2025-06-15 10:30:00", "quantity": 10.0},
        # Prior order before T
        {"customer_id": "C_WARM", "product_id": "P_PRIOR", "purchase_date": "2025-01-01 00:00:00", "quantity": 5.0},
    ])
    companies = pd.DataFrame([
        {"customer_id": "C_COLD", "industry_branch": "IND_ELECTRONICS", "city": "Berlin", "region_code": "BE"},
        {"customer_id": "C_WARM", "industry_branch": "IND_RETAIL", "city": "Munich", "region_code": "BY"},
    ])
    catalogue = pd.DataFrame([
        {"product_id": "P_TARGET", "brand": "BrandA", "product_category": "CatA", "product_subcategory": "SubA", "business_line": "LineA"},
        {"product_id": "P_PRIOR", "brand": "BrandB", "product_category": "CatB", "product_subcategory": "SubB", "business_line": "LineB"},
    ])

    store = PointInTimeFeatureStore(purchases_df=purchases, catalogue_df=catalogue, companies_df=companies)

    # 1. Verification: order at date T is NOT visible as of date T
    snap_t = store.get_customer_snapshot("C_WARM", as_of_t)
    assert snap_t.total_orders == 1
    assert "P_PRIOR" in snap_t.sku_stats
    assert "P_TARGET" not in snap_t.sku_stats, "Order placed at exact date T must NOT be visible as of T!"

    # Verification: order at date T becomes visible strictly after T
    snap_after_t = store.get_customer_snapshot("C_WARM", as_of_t + pd.Timedelta(microseconds=1))
    assert snap_after_t.total_orders == 2
    assert "P_TARGET" in snap_after_t.sku_stats

    # 2. Verification: Cold customer with 0 prior transactions
    snap_cold = store.get_customer_snapshot("C_COLD", as_of_t)
    assert snap_cold.total_orders == 0
    assert snap_cold.total_volume == 0.0
    assert snap_cold.sku_stats == {}

    # Pairwise feature extraction for cold customer
    feats_cold = store.compute_pairwise_features("C_COLD", as_of_t, "P_TARGET")
    assert feats_cold.shape == (NUM_PAIRWISE_FEATURES,)
    assert not np.isnan(feats_cold).any(), "NaN detected in cold customer pairwise features!"
    assert not np.isinf(feats_cold).any(), "Inf detected in cold customer pairwise features!"
    assert (feats_cold >= 0.0).all() and (feats_cold <= 1.0).all(), "Features out of [0, 1]!"

    # Specific default feature contracts
    assert feats_cold[0] == 0.0, "is_repeat_sku must be 0.0 for cold customer"
    assert feats_cold[1] == 0.0, "user_sku_order_count_norm must be 0.0 for cold customer"
    assert feats_cold[2] == 0.0, "user_sku_qty_share must be 0.0 for cold customer"
    assert feats_cold[3] == 1.0, "days_since_last_sku_buy_norm must be 1.0 (never bought)"
    assert feats_cold[4] == 0.0, "sku_urgency must be 0.0 for cold customer (< 2 orders)"
    assert feats_cold[5] == 0.0, "user_cat_order_share must be 0.0 for cold customer"
    assert feats_cold[6] == 0.0, "user_subcat_order_share must be 0.0 for cold customer"
    assert feats_cold[7] == 0.0, "user_brand_order_share must be 0.0 for cold customer"


def test_candidate_slate_exhaustion_small_catalog_and_empty_retriever():
    """Challenger R2 Test 8: Candidate slate exhaustion, small catalogs, and empty retriever output.

    Verifies:
    1. Catalogs with fewer than 50 total products (e.g. 5, 20 items) construct clean candidate slates
       matching available catalog size without indexing or out-of-bounds errors.
    2. CandidateRetriever batch construction handles small catalogs with correct tensor dimensions.
    3. Empty retriever output (top_k=0, top_c=0, or empty catalog) cleanly returns empty slates.
    """
    # 1. Small catalog: 20 products (< 50)
    num_products = 20
    pids = [f"PROD_{i:02d}" for i in range(num_products)]

    logits = torch.linspace(20.0, 1.0, num_products)

    # Eval mode: slate length is capped at min(top_c, num_products) = 20
    slate_eval = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        top_k=100,
        top_c=50,
    )
    assert len(slate_eval["candidate_indices"]) == 20
    assert len(slate_eval["candidate_pids"]) == 20
    assert len(set(slate_eval["candidate_pids"])) == 20
    assert slate_eval["candidate_ranks"] == list(range(20))

    # A target outside a five-item slate leaves all candidate labels negative
    slate_small = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=["PROD_15"],
        top_k=20,
        top_c=5,
    )
    assert len(slate_small["candidate_indices"]) == 5
    assert slate_small["candidate_pids"] == pids[:5]
    assert slate_small["labels"] == [0.0] * 5

    # 2. CandidateRetriever batch with small catalog
    retriever = CandidateRetriever(eligible_product_ids=pids)
    batch_small = retriever.generate_candidate_batch(
        logits=torch.randn(2, 20),
        customer_ids=["C1", "C2"],
        as_of_dates=["2025-01-01", "2025-01-01"],
        positive_product_ids=[["PROD_01"], ["PROD_15"]],
        top_k=100,
        top_c=50,
    )
    assert batch_small.candidate_indices.shape == (2, 20)
    assert batch_small.candidate_scores.shape == (2, 20)
    assert batch_small.pairwise_features.shape == (2, 20, NUM_PAIRWISE_FEATURES)
    assert not torch.isnan(batch_small.pairwise_features).any()

    # 3. Empty retriever output: top_k=0, top_c=0
    slate_zero = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        top_k=0,
        top_c=0,
    )
    assert slate_zero["candidate_indices"] == []
    assert slate_zero["candidate_pids"] == []
    assert slate_zero["candidate_scores"] == []
    assert slate_zero["candidate_ranks"] == []

    # 4. Empty catalog: product_ids=[]
    empty_logits = torch.empty(0)
    slate_empty_catalog = construct_candidate_slate(
        stage1_logits=empty_logits,
        product_ids=[],
        top_k=100,
        top_c=50,
    )
    assert slate_empty_catalog["candidate_indices"] == []
    assert slate_empty_catalog["candidate_pids"] == []

    # 5. Empty batch in CandidateRetriever
    batch_empty = retriever.generate_candidate_batch(
        logits=torch.empty((0, 20)),
        customer_ids=[],
        as_of_dates=[],
    )
    assert batch_empty.candidate_indices.shape[0] == 0
    assert batch_empty.pairwise_features.shape[0] == 0
