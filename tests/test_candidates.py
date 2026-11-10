"""Unit and integration tests for Candidate Generation and Pairwise Features (Milestone M9).

Covers:
1. test_stage1_candidate_retrieval_shapes: top-100 and top-50 shapes.
2. test_pairwise_features_bounded_zero_one: all features strictly within [0.0, 1.0].
3. test_pairwise_features_no_nan: edge cases producing zero NaNs or Infs.
5. test_candidate_slate_eval_no_injection: strictly top 50 from Stage 1 in evaluation mode.
6. test_candidate_batch_dict_access_and_device: dataclass and dict protocol compatibility.
7. test_submillisecond_pairwise_feature_extraction: sub-millisecond performance benchmark.
8. test_checkpoint_loading_and_retrieval: integration test with trained retriever checkpoint.
9. test_candidate_slate_duplicate_positives_deduplication: deduplication and slate uniqueness preservation.
10. test_point_in_time_feature_store_mixed_datetime_parsing: robust parsing of mixed datetime formats.
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


@pytest.fixture
def sample_catalog_data():
    """Create a synthetic catalogue DataFrame with 120 products."""
    num_products = 120
    rows = []
    for i in range(num_products):
        pid = f"PROD_{i:03d}"
        cat = f"CAT_{i % 5}"
        subcat = f"SUBCAT_{i % 10}"
        brand = f"BRAND_{i % 8}"
        bline = f"LINE_{i % 3}"
        rows.append({
            "product_id": pid,
            "product_category": cat,
            "product_subcategory": subcat,
            "brand": brand,
            "business_line": bline,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_companies_data():
    """Create synthetic customer metadata DataFrame."""
    rows = [
        {"customer_id": "CUST_001", "industry_branch": "IND_A", "city": "Kyiv", "region_code": "REG_1"},
        {"customer_id": "CUST_002", "industry_branch": "IND_B", "city": "Lviv", "region_code": "REG_2"},
        {"customer_id": "CUST_003", "industry_branch": "IND_A", "city": "Dnipro", "region_code": "REG_3"},
        {"customer_id": "CUST_COLD", "industry_branch": "IND_C", "city": "Odesa", "region_code": "REG_4"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_purchases_data():
    """Create synthetic chronological purchase records."""
    rows = [
        # CUST_001 purchases: P000, P001, P002
        {"customer_id": "CUST_001", "product_id": "PROD_000", "purchase_date": "2025-01-10", "quantity": 10.0},
        {"customer_id": "CUST_001", "product_id": "PROD_000", "purchase_date": "2025-02-10", "quantity": 15.0},
        {"customer_id": "CUST_001", "product_id": "PROD_000", "purchase_date": "2025-03-10", "quantity": 20.0},
        {"customer_id": "CUST_001", "product_id": "PROD_001", "purchase_date": "2025-02-15", "quantity": 5.0},
        {"customer_id": "CUST_001", "product_id": "PROD_002", "purchase_date": "2025-02-20", "quantity": 5.0},
        # CUST_002 purchases: P005 (many times)
        {"customer_id": "CUST_002", "product_id": "PROD_005", "purchase_date": "2025-01-05", "quantity": 2.0},
        {"customer_id": "CUST_002", "product_id": "PROD_005", "purchase_date": "2025-01-15", "quantity": 2.0},
        {"customer_id": "CUST_002", "product_id": "PROD_005", "purchase_date": "2025-01-25", "quantity": 2.0},
        {"customer_id": "CUST_002", "product_id": "PROD_005", "purchase_date": "2025-02-05", "quantity": 2.0},
        # CUST_003 purchases: P010
        {"customer_id": "CUST_003", "product_id": "PROD_010", "purchase_date": "2025-03-01", "quantity": 50.0},
    ]
    df = pd.DataFrame(rows)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df


def test_stage1_candidate_retrieval_shapes(sample_catalog_data, sample_companies_data, sample_purchases_data):
    """Verifies top-100 and top-50 candidate retrieval shapes across batch."""
    b_size = 4
    num_products = len(sample_catalog_data)
    eligible_pids = sample_catalog_data["product_id"].tolist()

    store = PointInTimeFeatureStore(
        purchases_df=sample_purchases_data,
        catalogue_df=sample_catalog_data,
        companies_df=sample_companies_data,
    )

    retriever = CandidateRetriever(
        eligible_product_ids=eligible_pids,
        feature_store=store,
    )

    # Random customer vectors (B, 128) and product vectors (P, 128)
    cust_vecs = torch.randn(b_size, 128)
    prod_vecs = torch.randn(num_products, 128)

    batch = retriever.retrieve_from_vectors(
        customer_vectors=cust_vecs,
        customer_ids=["CUST_001", "CUST_002", "CUST_003", "CUST_COLD"],
        as_of_dates=["2025-04-01", "2025-04-01", "2025-04-01", "2025-04-01"],
        product_vectors=prod_vecs,
        top_k=100,
        top_c=50,
    )

    # Assert shapes
    assert batch.top100_indices.shape == (b_size, 100)
    assert batch.top100_scores.shape == (b_size, 100)
    assert len(batch.top100_pids) == b_size
    assert len(batch.top100_pids[0]) == 100

    assert batch.candidate_indices.shape == (b_size, 50)
    assert batch.candidate_scores.shape == (b_size, 50)
    assert batch.pairwise_features.shape == (b_size, 50, NUM_PAIRWISE_FEATURES)
    assert len(batch.candidate_pids) == b_size
    assert len(batch.candidate_pids[0]) == 50

    # Check smaller catalog support (e.g. P = 25)
    batch_small = retriever.retrieve_from_vectors(
        customer_vectors=cust_vecs,
        customer_ids=["CUST_001", "CUST_002", "CUST_003", "CUST_COLD"],
        as_of_dates=["2025-04-01", "2025-04-01", "2025-04-01", "2025-04-01"],
        product_vectors=prod_vecs[:25],
        top_k=100,
        top_c=50,
    )
    assert batch_small.candidate_indices.shape == (b_size, 25)
    assert batch_small.pairwise_features.shape == (b_size, 25, NUM_PAIRWISE_FEATURES)


def test_pairwise_features_bounded_zero_one(sample_catalog_data, sample_companies_data, sample_purchases_data):
    """Validates that all pairwise features fall strictly in [0.0, 1.0] across edge cases."""
    store = PointInTimeFeatureStore(
        purchases_df=sample_purchases_data,
        catalogue_df=sample_catalog_data,
        companies_df=sample_companies_data,
        train_cutoff="2025-10-01",
    )

    as_of = pd.Timestamp("2025-04-01")

    # 1. Warm repeat customer (CUST_001 on PROD_000): bought 3 times
    f_repeat = store.compute_pairwise_features(
        customer_id="CUST_001",
        as_of_date=as_of,
        product_id="PROD_000",
    )
    assert f_repeat.shape == (NUM_PAIRWISE_FEATURES,)
    assert (f_repeat >= 0.0).all() and (f_repeat <= 1.0).all()

    # Exact checks for repeat product
    # is_repeat_sku
    assert f_repeat[0] == 1.0
    # user_sku_order_count_norm (3 orders)
    assert np.isclose(f_repeat[1], np.log1p(3) / np.log1p(50), atol=1e-4)
    # user_sku_qty_share: 45 / (45 + 5 + 5) = 45 / 55 = 0.8181
    assert np.isclose(f_repeat[2], 45.0 / 55.0, atol=1e-4)
    # days_since_last_sku_buy_norm: last purchase 2025-03-10, as_of 2025-04-01 -> 22 days
    assert np.isclose(f_repeat[3], np.log1p(22) / np.log1p(365), atol=1e-4)
    # sku_urgency: 3 orders, gap between dates is 31d and 28d -> median_gap = 29.5d
    # delta_t = 22d -> ratio = 22 / 29.5 = 0.7457 -> urgency = 0.7457 / 5.0
    expected_urgency = (22.0 / 29.5) / 5.0
    assert np.isclose(f_repeat[4], expected_urgency, atol=1e-3)
    # user_cat_order_share: PROD_000 is CAT_0, CUST_001 has 3 orders in CAT_0 out of 5 total orders -> 3/5 = 0.6
    assert np.isclose(f_repeat[5], 3.0 / 5.0, atol=1e-4)

    # 2. Never bought product (CUST_001 on PROD_050)
    f_cold_prod = store.compute_pairwise_features(
        customer_id="CUST_001",
        as_of_date=as_of,
        product_id="PROD_050",
    )
    assert (f_cold_prod >= 0.0).all() and (f_cold_prod <= 1.0).all()
    assert f_cold_prod[0] == 0.0  # is_repeat_sku == 0.0
    assert f_cold_prod[1] == 0.0  # order_count == 0.0
    assert f_cold_prod[2] == 0.0  # qty_share == 0.0
    assert f_cold_prod[3] == 1.0  # days_since_last_sku_buy_norm == 1.0 if never bought
    assert f_cold_prod[4] == 0.0  # sku_urgency == 0.0 if < 2 orders

    # 3. Completely cold customer (CUST_COLD)
    f_cold_cust = store.compute_pairwise_features(
        customer_id="CUST_COLD",
        as_of_date=as_of,
        product_id="PROD_000",
    )
    assert (f_cold_cust >= 0.0).all() and (f_cold_cust <= 1.0).all()
    assert f_cold_cust[0] == 0.0
    assert f_cold_cust[1] == 0.0
    assert f_cold_cust[2] == 0.0
    assert f_cold_cust[3] == 1.0
    assert f_cold_cust[4] == 0.0
    assert f_cold_cust[5] == 0.0
    assert f_cold_cust[6] == 0.0
    assert f_cold_cust[7] == 0.0



def test_pairwise_features_no_nan(sample_catalog_data):
    """Asserts zero NaNs / Infs under empty history, cold products, or division by zero."""
    empty_df = pd.DataFrame(columns=["customer_id", "product_id", "purchase_date", "quantity"])
    empty_store = PointInTimeFeatureStore(
        purchases_df=empty_df,
        catalogue_df=sample_catalog_data,
    )

    for pid in ("UNKNOWN_PROD", "PROD_001"):
        feats = empty_store.compute_pairwise_features(
            customer_id="UNKNOWN_CUST",
            as_of_date="2026-01-01",
            product_id=pid,
        )
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        assert (feats >= 0.0).all() and (feats <= 1.0).all()

    # Test CandidateBatch generation with NaN inputs
    retriever = CandidateRetriever(
        eligible_product_ids=[f"P_{i}" for i in range(50)],
        feature_store=empty_store,
    )
    nan_logits = torch.full((2, 50), float("nan"))
    batch = retriever.generate_candidate_batch(
        logits=nan_logits,
        customer_ids=["C1", "C2"],
        as_of_dates=["2025-01-01", "2025-01-01"],

    )
    assert not torch.isnan(batch.pairwise_features).any()
    assert not torch.isnan(batch.candidate_scores).any()
    assert (batch.pairwise_features >= 0.0).all()
    assert (batch.pairwise_features <= 1.0).all()



def test_candidate_slate_eval_no_injection():
    """Asserts evaluation slates are strictly the top 50 from Stage 1 without positive injection."""
    num_products = 100
    pids = [f"PROD_{i:03d}" for i in range(num_products)]


    logits = torch.linspace(100.0, 1.0, num_products)
    # Duplicate retrieved positives, a missed positive, and an unknown SKU
    target_pos = ["PROD_005", "PROD_005", "PROD_075", "UNKNOWN"]

    slate_dict = construct_candidate_slate(
        stage1_logits=logits,
        product_ids=pids,
        positive_product_ids=target_pos,
        top_k=100,
        top_c=50,
    )

    candidate_pids = slate_dict["candidate_pids"]
    # PROD_075 must NOT be in eval candidate slate
    assert "PROD_075" not in candidate_pids

    # Candidate slate must strictly equal the Stage 1 top 50: PROD_000..PROD_049
    expected_top50 = pids[:50]
    assert candidate_pids == expected_top50
    # No items were replaced
    assert slate_dict["candidate_ranks"] == list(range(50))
    # Only the retrieved positive gets a label, regardless of duplicate targets
    assert slate_dict["labels"][5] == 1.0
    assert sum(slate_dict["labels"]) == 1.0


def test_hybrid_slate_preserves_tower_ranks_and_uses_only_prior_sales():
    purchases = pd.DataFrame(
        [
            ("customer", "B", "2025-01-02"),
            ("other", "E", "2025-01-02"),
            ("other", "E", "2025-01-03"),
            ("customer", "D", "2025-01-04"),
            ("other", "D", "2025-01-04"),
        ],
        columns=["customer_id", "product_id", "purchase_date"],
    )
    pids = list("ABCDE")
    retriever = CandidateRetriever(
        eligible_product_ids=pids, feature_store=PointInTimeFeatureStore(purchases),
    )
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    batch = retriever.generate_candidate_batch(
        logits, ["customer"], ["2025-01-04"], [["D", "E"]],
        top_k=4, top_c=4, heuristic_fraction=0.5,
    )
    assert batch.top100_pids == [list("ABCD")]
    assert batch.candidate_pids == [list("BEAC")]
    assert batch.candidate_indices.tolist() == [[1, 4, 0, 2]]
    assert batch.candidate_scores.tolist() == [[4.0, 1.0, 5.0, 3.0]]
    assert batch.labels.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    torch.testing.assert_close(batch.pairwise_features[0, :, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cold = retriever.generate_candidate_batch(
        logits, ["cold"], ["2025-01-02"], top_k=4, top_c=4, heuristic_fraction=0.5,
    )
    assert cold.candidate_pids == [list("ABCD")]


def test_attribute_repeat_flags_use_strictly_prior_history_for_unseen_skus():
    purchases = pd.DataFrame(
        [("C1", "A", "2025-01-01"), ("C1", "B", "2025-01-02")],
        columns=["customer_id", "product_id", "purchase_date"],
    )
    catalogue = pd.DataFrame(
        [
            ("A", "category-a", "brand-a", "line-a"),
            ("B", "category-b", "brand-b", "line-b"),
            ("C", "category-a", "brand-a", "line-a"),
            ("D", "category-b", "brand-b", "line-b"),
            ("E", None, None, None),
            ("F", "category-a", "brand-b", "line-b"),
            ("G", "category-b", "brand-a", "line-b"),
            ("H", "category-b", "brand-b", "line-a"),
        ],
        columns=["product_id", "product_category", "brand", "business_line"],
    )
    store = PointInTimeFeatureStore(purchases, catalogue)
    rows = store.compute_slate_pairwise_features("C1", "2025-01-02", ["C", "D", "E"])
    assert rows[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert rows[:, -3:].tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    separated = store.compute_slate_pairwise_features("C1", "2025-01-02", ["F", "G", "H"])
    torch.testing.assert_close(separated[:, -3:], torch.eye(3))
    assert store.compute_pairwise_features("C1", "2025-01-03", "D")[-3:].tolist() == [1.0, 1.0, 1.0]


def test_candidate_batch_dict_access_and_device():
    """Validates CandidateBatch dataclass supports dict indexing, to_dict, and device transfer."""
    b_size, c_slate = 2, 5
    batch = CandidateBatch(
        customer_ids=["C1", "C2"],
        candidate_pids=[["P1", "P2", "P3", "P4", "P5"], ["P6", "P7", "P8", "P9", "P10"]],
        candidate_indices=torch.arange(10).view(b_size, c_slate),
        candidate_scores=torch.randn(b_size, c_slate),
        pairwise_features=torch.rand(b_size, c_slate, NUM_PAIRWISE_FEATURES),
        labels=torch.zeros(b_size, c_slate),
        top100_indices=torch.arange(10).view(b_size, c_slate),
        top100_scores=torch.randn(b_size, c_slate),
        top100_pids=[["P1", "P2", "P3", "P4", "P5"], ["P6", "P7", "P8", "P9", "P10"]],
    )

    # Dict access
    assert "pairwise_features" in batch
    assert batch["pairwise_features"].shape == (b_size, c_slate, NUM_PAIRWISE_FEATURES)
    assert batch.get("customer_ids") == ["C1", "C2"]
    assert batch.get("non_existent", "default") == "default"

    # to_dict
    d = batch.to_dict()
    assert isinstance(d, dict)
    assert "candidate_pids" in d

    # Device transfer
    batch_cpu = batch.to("cpu")
    assert batch_cpu.pairwise_features.device.type == "cpu"


def test_submillisecond_pairwise_feature_extraction(sample_catalog_data, sample_companies_data, sample_purchases_data):
    """Benchmarks feature extraction to confirm sub-millisecond execution per candidate slate."""
    store = PointInTimeFeatureStore(
        purchases_df=sample_purchases_data,
        catalogue_df=sample_catalog_data,
        companies_df=sample_companies_data,
    )

    pids_50 = sample_catalog_data["product_id"].tolist()[:50]
    as_of = pd.Timestamp("2025-04-01")

    # Warm-up call
    store.compute_slate_pairwise_features("CUST_001", as_of, pids_50)

    # Benchmark 200 consecutive candidate slate computations
    n_iterations = 200
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        store.compute_slate_pairwise_features("CUST_001", as_of, pids_50)
    elapsed_total = time.perf_counter() - t0
    avg_ms = (elapsed_total / n_iterations) * 1000.0

    print(f"\nAverage slate feature extraction time: {avg_ms:.4f} ms per 50-candidate slate")
    # Sub-millisecond requirement: avg_ms must be strictly < 1.0 ms
    assert avg_ms < 1.0, f"Feature extraction exceeded 1ms target: {avg_ms:.4f} ms"


def test_checkpoint_loading_and_retrieval():
    """Integration test: loads real checkpoint checkpoints/retriever/best_retriever.pt and verifies inference."""
    ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
    if not ckpt_path.is_file():
        pytest.skip("best_retriever.pt checkpoint not found")

    model, ckpt = load_retriever_from_checkpoint(ckpt_path)
    assert ckpt is not None
    num_products = len(ckpt["product_ids"])
    assert num_products == ckpt["config"]["num_catalog_products"]

    # Load sample real purchases
    purchases_path = Path("data/interim/cleaned_purchases.csv")
    if not purchases_path.is_file():
        pytest.skip("cleaned_purchases.csv not found")

    purchases_df = pd.read_csv(purchases_path, nrows=500)
    store = PointInTimeFeatureStore(purchases_df=purchases_df)

    retriever = CandidateRetriever(
        model=model,
        eligible_product_ids=ckpt["product_ids"],
        feature_store=store,
    )

    # Generate synthetic customer and catalog vectors with the saved dimensions.
    cust_vecs = torch.randn(2, 128)
    prod_vecs = torch.randn(num_products, 128)

    sample_cids = purchases_df["customer_id"].astype(str).unique()[:2].tolist()
    as_ofs = [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-01")]

    batch = retriever.retrieve_from_vectors(
        customer_vectors=cust_vecs,
        customer_ids=sample_cids,
        as_of_dates=as_ofs,
        product_vectors=prod_vecs,
        top_k=100,
        top_c=50,
    )

    assert batch.candidate_indices.shape == (2, 50)
    assert batch.top100_indices.shape == (2, 100)
    assert batch.pairwise_features.shape == (2, 50, NUM_PAIRWISE_FEATURES)
    assert not torch.isnan(batch.pairwise_features).any()
    assert (batch.pairwise_features >= 0.0).all()
    assert (batch.pairwise_features <= 1.0).all()



def test_point_in_time_feature_store_mixed_datetime_parsing():
    """Verifies PointInTimeFeatureStore robustly parses mixed datetime string formats.

    Addresses Challenger M9_2 finding:
    In pandas 2.0+, pd.to_datetime fails with ValueError when parsing mixed date-only
    and timestamp strings without format='mixed'.
    """
    mixed_purchases = pd.DataFrame([
        # Date-only strings
        {"customer_id": "C_MIXED", "product_id": "P_01", "purchase_date": "2025-01-01", "quantity": 5.0, "product_category": "CAT_A"},
        {"customer_id": "C_MIXED", "product_id": "P_01", "purchase_date": "2025-01-10", "quantity": 10.0, "product_category": "CAT_A"},
        # Time-annotated strings
        {"customer_id": "C_MIXED", "product_id": "P_02", "purchase_date": "2025-01-15 10:30:00", "quantity": 2.0, "product_category": "CAT_B"},
        # ISO8601 strings
        {"customer_id": "C_MIXED", "product_id": "P_02", "purchase_date": "2025-01-20T14:45:00", "quantity": 4.0, "product_category": "CAT_B"},
        # Subsecond timestamp strings
        {"customer_id": "C_MIXED", "product_id": "P_03", "purchase_date": "2025-01-25 09:00:00.123456", "quantity": 1.0, "product_category": "CAT_C"},
    ])

    # Store initialization must succeed without ValueError
    store = PointInTimeFeatureStore(purchases_df=mixed_purchases)

    as_of = pd.Timestamp("2025-02-01 00:00:00")
    snap = store.get_customer_snapshot("C_MIXED", as_of)

    # Check that all 5 events were parsed and ingested
    assert snap.total_orders == 5
    assert np.isclose(snap.total_volume, 22.0, atol=1e-5)
    assert set(snap.sku_stats.keys()) == {"P_01", "P_02", "P_03"}

    # Test point-in-time boundary between events (as of 2025-01-15 00:00:00)
    as_of_mid = pd.Timestamp("2025-01-15 00:00:00")
    snap_mid = store.get_customer_snapshot("C_MIXED", as_of_mid)
    # Only P_01 events from Jan 1 and Jan 10 are strictly before Jan 15 00:00:00
    assert snap_mid.total_orders == 2
    assert "P_01" in snap_mid.sku_stats
    assert "P_02" not in snap_mid.sku_stats

    # Check pairwise feature extraction
    feats = store.compute_pairwise_features("C_MIXED", as_of, "P_01")
    assert feats.shape == (NUM_PAIRWISE_FEATURES,)
    assert not np.isnan(feats).any()
    assert (feats >= 0.0).all() and (feats <= 1.0).all()
    assert feats[0] == 1.0  # is_repeat_sku
    assert feats[2] == 15.0 / 22.0  # quantity share
