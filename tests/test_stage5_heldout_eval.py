"""Synthetic unit tests for held-out Test evaluation guards, causality, and denominator accounting."""

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.roadmap_v2.base_name_features import BaseNameRollingFeatureBuilder
from recommender.experiments.roadmap_v2.commercial_features import CommercialRollingFeatureBuilder
from recommender.experiments.roadmap_v2.dated_baskets import dated_candidate_features
from recommender.experiments.roadmap_v2.evaluate_stage5_test import (
    evaluate_full_catalogue_predictions,
    run_heldout_test_evaluation,
)
from recommender.experiments.roadmap_v2.parsed_spec_features import ParsedSpecRollingFeatureBuilder
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
)


def test_heldout_test_runner_refuses_without_explicit_opt_in():
    with pytest.raises(PermissionError, match="allow-heldout-test"):
        run_heldout_test_evaluation(allow_heldout_test=False)


def test_default_builders_still_reject_post_test_start_dates():
    pids = ["p1", "p2"]
    catalogue = pd.DataFrame({
        "product_id": pids,
        "base_name": ["oil 5w30", "oil 5w40"],
        "product_category": "OIL",
        "brand": "BR",
        "business_line": "L",
        "product_family": "F",
        "product_subcategory": "S",
    })
    companies = pd.DataFrame({"customer_id": ["c1"], "industry_branch": "I"})
    post_test = pd.DataFrame([("c1", "p1", "2026-03-01")], columns=["customer_id", "product_id", "purchase_date"]).assign(
        quantity=1.0, revenue=10.0, cost=-5.0, cost_usd=-0.1, doc_bonus=0.0,
        manager_id="M", payment_terms="ДН14", full_payment_date=pd.Timestamp("2026-03-10"),
    )
    post_test["purchase_date"] = pd.to_datetime(post_test["purchase_date"])

    with pytest.raises(ValueError, match="Development"):
        DynamicFeatureBuilder(post_test, catalogue, companies, pids, statistics_policy=STATISTICS_POLICY_ROLLING)
    with pytest.raises(ValueError, match="Development"):
        BaseNameRollingFeatureBuilder(post_test, catalogue, companies, pids)
    with pytest.raises(ValueError, match="Development"):
        CommercialRollingFeatureBuilder(post_test, catalogue, pids)
    with pytest.raises(ValueError, match="Development"):
        ParsedSpecRollingFeatureBuilder(post_test, catalogue, companies, pids)

    dev_only = post_test.assign(purchase_date=pd.Timestamp("2025-05-01"))
    universe = build_base_name_universe(pids, catalogue, dev_only)
    with pytest.raises(ValueError, match="Development"):
        build_base_name_sequences(post_test, ["c1"], [450], universe, catalogue, {"OIL": 1}, {"S": 1}, {"L": 1})


def test_heldout_opt_in_features_are_strictly_causal_on_test_dates():
    pids = ["a", "b"]
    catalogue = pd.DataFrame({
        "product_id": pids,
        "base_name": ["oil 5w30", "oil 5w40"],
        "product_category": "OIL",
        "brand": "BR",
        "business_line": "L",
        "product_family": "F",
        "product_subcategory": "S",
    })
    companies = pd.DataFrame({"customer_id": ["old", "cold"], "industry_branch": "I"})
    test_date = pd.Timestamp("2026-03-15")
    events = pd.DataFrame(
        [
            ("old", "a", "2025-10-05", "2025-10-20"),
            ("old", "b", "2026-02-20", "2026-03-05"),  # settled strictly < 2026-03-15
            ("old", "a", "2026-03-01", "2026-03-20"),  # settled AFTER 2026-03-15 -> must look unpaid on 03-15
            ("old", "a", "2026-03-15", "2026-03-25"),  # same-day -> must be invisible
            ("cold", "a", "2026-03-15", "2026-03-25"),
            ("old", "b", "2026-03-18", "2026-03-28"),  # future -> must be invisible
        ],
        columns=["customer_id", "product_id", "purchase_date", "full_payment_date"],
    ).assign(quantity=2.0, revenue=10.0, cost=-4.0, cost_usd=-0.1, doc_bonus=0.0, manager_id="M", payment_terms="ДН14")
    events["purchase_date"] = pd.to_datetime(events["purchase_date"])
    events["full_payment_date"] = pd.to_datetime(events["full_payment_date"])

    prefix = events[events["purchase_date"] < test_date].copy()
    prefix.loc[prefix["full_payment_date"] >= test_date, "full_payment_date"] = pd.NaT

    def compute(source):
        dynamic = DynamicFeatureBuilder(
            source, catalogue, companies, pids, statistics_policy=STATISTICS_POLICY_ROLLING, allow_heldout_test=True
        )
        base = BaseNameRollingFeatureBuilder(source, catalogue, companies, pids, allow_heldout_test=True)
        comm = CommercialRollingFeatureBuilder(source, catalogue, pids, allow_heldout_test=True)
        specs = ParsedSpecRollingFeatureBuilder(source, catalogue, companies, pids, allow_heldout_test=True)
        return [dated_candidate_features(cid, test_date, [0, 1], dynamic, base, comm, specs) for cid in ("old", "cold")]

    full_old, full_cold = compute(events)
    pref_old, pref_cold = compute(prefix)
    np.testing.assert_array_equal(full_old, pref_old)
    np.testing.assert_array_equal(full_cold, pref_cold)


def test_full_catalogue_evaluator_penalizes_unmodeled_targets():
    # Basket 0: 1 modeled target (hit at rank 0) + 1 unmodeled target (miss) -> Recall@10 = 0.5, NDCG@10 = 1 / (1 + 1/log2(3))
    # Basket 1: 0 modeled targets + 1 unmodeled target (miss) -> Recall@10 = 0.0, NDCG@10 = 0.0
    ranked = np.array([[0, 1], [0, 1]], dtype=np.int64)
    target = torch.tensor([[True, False], [False, False]])
    new_target = torch.tensor([[True, False], [False, False]])
    extra_tot = np.array([1, 1], dtype=np.int64)
    extra_new = np.array([1, 1], dtype=np.int64)

    res = evaluate_full_catalogue_predictions(ranked, target, new_target, extra_tot, extra_new, ["c1", "c2"])
    assert res["queries"] == 2
    assert res["total_bn_targets"] == 3
    assert res["true_new_bn_targets"] == 3
    assert res["unmodeled_bn_targets"] == 2
    assert res["true_new_bn_hits@10"] == 1
    assert np.isclose(res["bn_recall@10"], 0.25)
    expected_ndcg0 = 1.0 / (1.0 + 1.0 / np.log2(3.0))
    assert np.isclose(res["bn_ndcg@10"], expected_ndcg0 / 2.0)
