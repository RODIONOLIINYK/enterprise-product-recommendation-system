import pandas as pd
import pytest
import torch

from recommender.candidates import PointInTimeFeatureStore
from recommender.ranker_statistics import FavoriteCategoryAffinity, with_shared_sku_statistics
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
