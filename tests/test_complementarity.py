from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from recommender.candidates import CandidateRetriever, PointInTimeFeatureStore
from recommender.complementarity import ProductComplementarity, product_snapshot_date
from recommender.train_ranker import precompute_candidate_slates_for_transitions


def test_five_day_share_counts_prior_different_product_baskets_once():
    rows = [
        ("buyer", "2025-01-01", "A"),
        ("buyer", "2025-01-06", "B"),
        ("buyer", "2025-01-06", "B"),
        ("buyer", "2025-01-10", "B"),
        ("buyer", "2025-01-16", "A"),
        ("buyer", "2025-01-22", "B"),
        ("other", "2025-01-07", "B"),
        ("other", "2025-01-07", "C"),
        ("buyer", "2025-09-29", "C"),
        ("buyer", "2025-10-01", "A"),
    ]
    history = pd.DataFrame(rows, columns=["customer_id", "purchase_date", "product_id"])
    feature = ProductComplementarity(history)
    assert feature.shares(["B"], "2025-01-06").item() == 0
    assert feature.shares(["B"], "2025-01-07").item() == 1
    assert feature.shares(["B", "C", "unknown"], "2025-10-01") == pytest.approx([0.25, 0, 0])
    assert feature.shares(["A"], "2026-01-01").item() == 0
    assert product_snapshot_date("2025-01-19") == pd.Timestamp("2025-01-01")
    assert product_snapshot_date("2026-01-19") == pd.Timestamp("2025-10-01")


def test_ranker_keeps_monthly_product_vectors_and_shares_aligned_with_baskets():
    purchases = pd.DataFrame({
        "customer_id": ["buyer", "buyer"],
        "purchase_date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
        "product_id": ["A", "B"],
        "quantity": [1, 1],
    })
    feature = ProductComplementarity(purchases)
    model = Mock()
    model.customer_tower.max_seq_len = 32
    model.customer_tower.side_effect = lambda text_embedding, **_: torch.zeros(len(text_embedding), 2)
    retriever = CandidateRetriever(
        model=model, eligible_product_ids=["A", "B"], device="cpu",
        feature_store=PointInTimeFeatureStore(purchases), product_complementarity=feature,
    )
    retriever.get_product_vectors = Mock(side_effect=lambda date: torch.full((2, 2), float(date.month)))
    retriever.compute_logits = Mock(side_effect=lambda customers, *args, **kw: torch.tensor([[0., 1.]]).repeat(len(customers), 1))
    transitions = [
        {"customer_id": "buyer", "as_of_date": date, "positive_product_ids": ["B"]}
        for date in pd.to_datetime(["2025-01-05", "2025-02-03", "2025-01-02"])
    ]
    result = precompute_candidate_slates_for_transitions(
        transitions, retriever, torch.zeros(2, 2), {"buyer": purchases},
        {}, {}, None, {}, {}, top_c=2, complementarity=feature,
    )
    assert result["candidate_pids"] == [["B", "A"]] * 3
    assert result["candidate_vectors"][:, 0, 0].tolist() == [1.0, 2.0, 1.0]
    assert result["pairwise_features"][:, 0, -1].tolist() == [1.0, 1.0, 0.0]
