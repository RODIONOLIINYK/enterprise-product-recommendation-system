"""Unit tests for Phase 10 evaluation module (recommender/evaluate.py)."""

import math
import pandas as pd
import pytest

from recommender.evaluate import (
    aggregate_transition_metrics,
    build_global_popularity_ranking,
    build_personal_repeat_ranking,
    compute_single_transition_metrics,
)


def test_compute_single_transition_metrics_exact():
    ranked = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
    targets = {"P1", "P4"}

    metrics = compute_single_transition_metrics(ranked, targets)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["hitrate@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)

    expected_dcg = (1.0 / math.log2(2)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert metrics["ndcg@10"] == pytest.approx(expected_dcg / expected_idcg)


def test_compute_single_transition_metrics_empty_targets():
    ranked = ["P1", "P2"]
    metrics = compute_single_transition_metrics(ranked, set())
    assert metrics["recall@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0
    assert metrics["mrr"] == 0.0


def test_aggregate_transition_metrics_stratified():
    per_trans = [
        {"recall@10": 0.5, "ndcg@10": 0.4, "mrr": 0.5},
        {"recall@10": 1.0, "ndcg@10": 0.8, "mrr": 1.0},
        {"recall@10": 0.6, "ndcg@10": 0.5, "mrr": 0.5},
    ]
    prior_counts = [1, 2, 5]  # First two are cold/light (<=2), third is warm (>2)

    agg = aggregate_transition_metrics(per_trans, prior_counts)
    assert agg["overall"]["num_transitions"] == 3
    assert agg["overall"]["recall@10"] == pytest.approx(0.7)
    assert agg["cold_light_lte_2_baskets"]["num_transitions"] == 2
    assert agg["cold_light_lte_2_baskets"]["recall@10"] == pytest.approx(0.75)
    assert agg["warm_gt_2_baskets"]["num_transitions"] == 1
    assert agg["warm_gt_2_baskets"]["recall@10"] == pytest.approx(0.6)


def test_global_and_personal_repeat_baselines_strict_cutoff():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2"],
        "product_id": ["P2", "P2", "P3", "P1", "P1"],
        "purchase_date": pd.to_datetime([
            "2025-01-10",
            "2025-02-15",
            "2025-11-01",  # After train cutoff & after as_of_date
            "2025-03-01",
            "2025-04-01",
        ]),
    })
    eligible = ["P1", "P2", "P3"]
    pop_ranking = build_global_popularity_ranking(df, eligible, cutoff="2025-10-01")
    # P1 and P2 each have 2 purchases before cutoff; P3 has 0 before cutoff
    assert pop_ranking == ["P1", "P2", "P3"]

    c1_df = df.loc[df["customer_id"] == "C1"].copy()
    rep_ranking = build_personal_repeat_ranking(
        c1_df, as_of_date="2025-10-15", global_popularity=pop_ranking, max_k=3
    )
    # C1 bought P2 twice before 2025-10-15; P3 is in the future (2025-11-01) so must not leak
    assert rep_ranking == ["P2", "P1", "P3"]
