import pandas as pd
import pytest
import torch

from recommender.candidates import PointInTimeFeatureStore
from recommender.purchase_timing import PurchaseTiming
from recommender.ranker import UPSCANRRanker
from recommender.ranker_statistics import (
    ALIGNED_11D_STATISTICS,
    FavoriteCategoryAffinity,
    with_aligned_11d_statistics,
    with_due_score_statistics,
    with_favorite_attribute_features,
    with_shared_sku_statistics,
)
from recommender.stage1_category_peers import CategoryPeerScores


def test_peer_affinity_uses_prior_favorites_and_excludes_the_customer():
    rows = [
        (buyer, "cleaner", f"2025-01-0{day}")
        for buyer in ("customer", "peer") for day in (1, 2, 3)
    ]
    rows += [
        ("peer", "glove", "2025-01-03"),  # No favorite at the start of this basket.
        ("customer", "glove", "2025-01-04"),
        ("peer", "glove", "2025-01-05"),
    ]
    purchases = pd.DataFrame(rows, columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({
        "product_id": ["cleaner", "glove"],
        "product_category": ["cleaning", "safety"],
        "brand": ["A", "B"],
    })
    affinity = FavoriteCategoryAffinity(PointInTimeFeatureStore(purchases, catalogue))
    assert affinity.share("customer", "cleaning", "glove", pd.Timestamp("2025-01-05")) == 0
    assert affinity.share("customer", "cleaning", "glove", pd.Timestamp("2025-01-06")) == pytest.approx(1 / 21)
    replay = CategoryPeerScores(affinity, ["cleaner", "glove"], {"cleaner": 6, "glove": 3})
    favorite, mix = replay.scores("customer", {"cleaning": 3, "unknown cohort": 2})
    assert favorite[1] > mix[1] > 0
    assert favorite[0] < mix[0] < 0
    empty, _ = replay.scores("customer", {"cleaning": 2})
    assert empty.tolist() == [0.0, 0.0]


def test_candidate_and_purchase_use_same_statistics_at_matching_purchase_age():
    purchases = pd.DataFrame([
        ("customer", "cleaner", "2025-01-01"),
        ("customer", "cleaner", "2025-01-03"),
        ("customer", "glove", "2025-01-10"),
    ], columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({
        "product_id": ["cleaner", "glove"], "product_category": ["cleaning", "safety"], "brand": ["A", "B"],
    })
    store = PointInTimeFeatureStore(purchases, catalogue)
    slate = {
        "customer_ids": ["customer"],
        "positive_pids": [["glove"]],
        "candidate_pids": [["cleaner", "glove"]],
        "candidate_indices": torch.tensor([[0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "seq_product_ids": torch.tensor([[1, 1, 0]]),
        "sequence": torch.zeros(1, 3, 9),
    }
    transitions = [{
        "customer_id": "customer", "positive_product_ids": ["glove"], "as_of_date": pd.Timestamp("2025-01-10"),
    }]
    result = with_shared_sku_statistics(slate, transitions, store, FavoriteCategoryAffinity(store), {"A": 1, "B": 2})
    history = result["ranker_sequence"]
    candidate = result["pairwise_features"]
    assert history.shape == (1, 3, 6) and candidate.shape == (1, 2, 6)
    torch.testing.assert_close(history[0, 1], candidate[0, 0])
    assert history[0, 0, 1] > history[0, 1, 1]
    assert candidate[0, 1, 0] == 0 and candidate[0, 1, 1] == 0
    assert result["seq_brand_ids"].tolist() == [[1, 1, 0]]
    assert result["sequence"] is slate["sequence"]


def test_due_score_aligned_11d_and_favorite_attribute_modes():
    purchases = pd.DataFrame([
        ("customer", "cleaner", "2025-01-01", 10.0),
        ("customer", "cleaner", "2025-01-11", 10.0),
        ("customer", "glove", "2025-01-21", 5.0),
    ], columns=["customer_id", "product_id", "purchase_date", "quantity"])
    catalogue = pd.DataFrame({
        "product_id": ["cleaner", "glove"],
        "product_category": ["cleaning", "safety"],
        "product_subcategory": ["sub_c", "sub_s"],
        "brand": ["A", "B"],
        "business_line": ["line1", "line2"],
    })
    store = PointInTimeFeatureStore(purchases, catalogue)
    timing = PurchaseTiming(purchases, catalogue)
    mask_32 = torch.zeros(1, 32, dtype=torch.long)
    mask_32[0, :2] = 1
    ids_32 = torch.zeros(1, 32, dtype=torch.long)
    ids_32[0, :2] = 1
    slate = {
        "customer_ids": ["customer"],
        "positive_pids": [["glove"]],
        "candidate_pids": [["cleaner", "glove"]],
        "candidate_indices": torch.tensor([[0, 1]]),
        "attention_mask": mask_32,
        "seq_product_ids": ids_32,
        "seq_category_ids": ids_32,
        "seq_subcategory_ids": ids_32,
        "seq_business_line_ids": ids_32,
        "seq_purchase_dates": torch.zeros(1, 32),
        "prediction_days": torch.zeros(1),
        "tabular_features": torch.full((1, 8), 0.25),
        "sequence": torch.full((1, 32, 9), 0.2),
        "pairwise_features": torch.full((1, 2, 13), 0.3),
    }
    transitions = [{
        "customer_id": "customer",
        "positive_product_ids": ["glove"],
        "as_of_date": pd.Timestamp("2025-01-21"),
    }]

    due = with_due_score_statistics(slate, transitions, store, timing, update_candidates=True)
    # Cleaner was bought on Jan 1 and Jan 11 -> personal gap = 10 days; on Jan 21 age = 10 days -> due_score = 0.0
    expected_gap_norm = 10.0 / (10.0 + 90.0)
    expected_age_norm = 10.0 / (10.0 + 90.0)
    assert due["ranker_sequence"][0, 1, 2].item() == pytest.approx(expected_gap_norm - expected_age_norm)
    assert due["pairwise_features"][0, 0, 4].item() == pytest.approx(expected_gap_norm - expected_age_norm)

    aligned = with_aligned_11d_statistics(slate, transitions, store, timing)
    assert len(ALIGNED_11D_STATISTICS) == 11
    assert aligned["aligned_company_stats"].shape == (1, 11)
    assert aligned["aligned_history_stats"].shape == (1, 32, 11)
    assert aligned["aligned_candidate_stats"].shape == (1, 2, 11)
    # Company quantity duplicated in dims 1-2; distinct_products duplicated in dims 6-7; distinct_categories in dims 8-10
    comp = aligned["aligned_company_stats"][0]
    assert comp[1].item() == pytest.approx(comp[2].item())
    assert comp[6].item() == pytest.approx(comp[7].item())
    assert comp[8].item() == pytest.approx(comp[9].item()) == pytest.approx(comp[10].item())

    vocabs = {
        "category_vocab": {"cleaning": 1, "safety": 2},
        "brand_vocab": {"A": 1, "B": 2},
        "business_line_vocab": {"line1": 1, "line2": 2},
    }
    fav = with_favorite_attribute_features(aligned, transitions, store, vocabs, append_candidate_features=True)
    assert fav["fav_category_id"].tolist() == [1]
    assert fav["fav_brand_id"].tolist() == [1]
    assert fav["fav_business_line_id"].tolist() == [1]
    assert fav["pairwise_features"].shape == (1, 2, 17)

    ranker = UPSCANRRanker(
        text_dim=16,
        memory_dim=16,
        num_cities=4,
        num_regions=4,
        num_industries=4,
        num_products=4,
        num_brands=4,
        num_categories=4,
        num_subcategories=4,
        num_business_lines=4,
        num_pairwise_features=17,
        statistics_attention=True,
        precompiled_history_vectors=True,
        precompiled_customer_vector=True,
        aligned_11d_statistics=True,
        aligned_11d_mode="diagonal",
        favorite_attribute_embeddings=True,
    )
    out = ranker(
        text_embedding=torch.randn(1, 16),
        city_id=torch.tensor([1]),
        region_id=torch.tensor([1]),
        industry_id=torch.tensor([1]),
        tabular_features=fav["tabular_features"],
        sequence=fav["sequence"],
        attention_mask=fav["attention_mask"],
        seq_product_ids=fav["seq_product_ids"],
        seq_category_ids=fav["seq_category_ids"],
        seq_subcategory_ids=fav["seq_subcategory_ids"],
        seq_brand_ids=ids_32,
        seq_business_line_ids=fav["seq_business_line_ids"],
        candidate_indices=fav["candidate_indices"],
        pairwise_features=fav["pairwise_features"],
        catalog_product_vectors=torch.randn(2, 16),
        customer_vector=torch.randn(1, 16),
        aligned_company_stats=fav["aligned_company_stats"],
        aligned_history_stats=fav["aligned_history_stats"],
        aligned_candidate_stats=fav["aligned_candidate_stats"],
        fav_category_id=fav["fav_category_id"],
        fav_brand_id=fav["fav_brand_id"],
        fav_business_line_id=fav["fav_business_line_id"],
    )
    assert out.scores.shape == (1, 2)
    assert torch.isfinite(out.scores).all()
