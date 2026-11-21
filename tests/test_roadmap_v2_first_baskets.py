import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.first_baskets import append_first_baskets, first_baskets
from recommender.split import TEST_START


def test_first_baskets_include_single_orders_and_all_same_day_products():
    purchases = pd.DataFrame(
        [
            ("a", "2025-03-01", "later"),
            ("a", "2025-01-02", "first"),
            ("b", "2025-01-02", "solo"),
            ("a", "2025-01-02", "other"),
        ],
        columns=["customer_id", "purchase_date", "product_id"],
    )
    assert dict(first_baskets(purchases)) == {
        (pd.Timestamp("2025-01-02"), "a"): {"first", "other"},
        (pd.Timestamp("2025-01-02"), "b"): {"solo"},
    }
    purchases.loc[0, "purchase_date"] = TEST_START.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="development"):
        list(first_baskets(purchases))


def test_first_order_augmentation_keeps_returning_rows_and_feature_schema():
    train = {
        "customer_ids": ["a", "b"],
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "feature_names": ("existing",),
    }
    first = {"customer_ids": ["a"], "attention_mask": torch.zeros(1, 3, dtype=torch.long)}
    mixed = append_first_baskets(train, first)
    assert mixed["customer_ids"] == ["a", "b", "a"]
    assert torch.equal(mixed["attention_mask"][:2], train["attention_mask"])
    assert not mixed["attention_mask"][2].any()
    assert mixed["feature_names"] == train["feature_names"]


def test_complete_novelty_denominator_and_pack_introduction(monkeypatch):
    from recommender.catalogue import RollingCandidatePool
    from recommender.experiments.roadmap_v2 import dated_baskets as dated
    from recommender.experiments.roadmap_v2.stage5_base_entity import build_base_name_universe

    monkeypatch.setattr(dated, "SLATE_SIZE", 1)
    pids = ["later-pack", "original", "new"]
    catalogue = pd.DataFrame({"product_id": pids, "base_name": ["oil", "oil", "filter"],
                              "item_type": "ТОВАР", "product_category": "oil"})
    events = pd.DataFrame([
        ("other", "2025-09-01", "original"), ("customer", "2025-10-01", "original"),
        ("customer", "2025-10-02", "later-pack"), ("customer", "2025-10-02", "new"),
    ], columns=["customer_id", "purchase_date", "product_id"]).assign(
        quantity=1, item_type="ТОВАР", transaction_type="ПРОДАЖА", product_category="oil")
    events["purchase_date"] = pd.to_datetime(events.purchase_date)
    pool = RollingCandidatePool(catalogue, events)
    universe = build_base_name_universe(pids, catalogue, events)
    anchors = dated.first_observed_pack_indices(pids, universe["bn_to_member_pid_indices"], pool.first_sales)
    oil, filt = (universe["bn_to_idx"][key] for key in ("oil", "filter"))
    assert pids[anchors[oil]] == "original"
    cases, _ = dated.basket_cases(events, pids, universe, pool)
    first, returning = cases["val"]
    assert first.first and not returning.first
    assert returning.target[oil] and not returning.new[oil]
    assert returning.new[filt] and not returning.allowed[filt]


def test_commercial_source_must_reconstruct_cleaned_quantities_and_all_paid_lines_settle():
    from recommender.experiments.roadmap_v2.dated_baskets import aligned_commercial_events

    rows = pd.DataFrame({"customer_id": ["c"] * 3, "product_id": ["p"] * 3,
                         "quantity": [1, 2, 0], "purchase_date": pd.Timestamp("2025-01-01"),
                         "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР",
                         "full_payment_date": [pd.Timestamp("2025-01-02"), pd.NaT, pd.NaT]})
    clean = rows.iloc[:1].assign(quantity=3)
    paid = aligned_commercial_events(rows, clean)
    assert len(paid) == 2 and paid.full_payment_date.isna().all()
    with pytest.raises(ValueError, match="snapshots disagree"):
        aligned_commercial_events(rows, clean.assign(quantity=4))


def test_cold_history_has_no_product_gradient_and_finite_profile_gradient():
    from recommender.experiments.static_enrichment.model import StaticEnrichedCustomerTower

    tower = StaticEnrichedCustomerTower(num_cities=2, num_regions=2, num_industries=2,
                                        num_tabular_features=0, dropout=0.0, text_dim=4,
                                        output_dim=8, hidden_dim=8, max_seq_len=2)
    vectors = torch.randn(3, 8, requires_grad=True)
    text = torch.randn(1, 4, requires_grad=True)
    ids = torch.ones(1, dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.long)
    result = tower(text, ids, ids, ids, None, torch.zeros(1, 2, 9), mask, mask,
                   catalog_product_vectors=vectors, seq_purchase_dates=mask.float())
    result[0, 0].backward()
    assert torch.isfinite(result).all() and text.grad.abs().sum() > 0
    assert vectors.grad.eq(0).all()
    assert all(torch.isfinite(p.grad).all() for p in tower.parameters() if p.grad is not None)


def test_dated_reference_rejects_same_day_history_and_ineligible_novelty_errors():
    from recommender.experiments.roadmap_v2.dated_baskets import validate_dated_split
    from recommender.sequences import DAYS_PER_YEAR, PURCHASE_DATE_ORIGIN

    day = (pd.Timestamp("2025-10-04") - PURCHASE_DATE_ORIGIN).days
    data = {
        "prediction_days": torch.tensor([day, day + 1]),
        "customer_ids": ["same customer", "same customer"],
        "attention_mask": torch.tensor([[0], [1]]),
        "seq_purchase_dates": torch.tensor([[0], [day / DAYS_PER_YEAR]]),
        "first_observed": torch.tensor([True, False]),
        "familiarity_bn": torch.zeros(2, 2, 4, dtype=torch.bool),
        "target_bn": torch.tensor([[True, True], [True, False]]),
        "new_target_bn": torch.tensor([[True, True], [False, False]]),
        "allowed_bn": torch.tensor([[True, False], [True, True]]),
        "unseen_bn": torch.tensor([[True, False], [False, False]]),
    }
    validate_dated_split(data, "val")
    with pytest.raises(ValueError, match="partition"):
        validate_dated_split(data, "train")
    with pytest.raises(ValueError, match="same-day"):
        validate_dated_split(dict(data, seq_purchase_dates=data["seq_purchase_dates"] + 1 / DAYS_PER_YEAR), "val")
    with pytest.raises(ValueError, match="First basket repeat"):
        validate_dated_split(dict(data, new_target_bn=data["new_target_bn"] & data["allowed_bn"]), "val")


def test_default_stage5_routes_to_mixed_dated_pipeline(monkeypatch):
    from recommender.experiments.roadmap_v2 import run_stage5 as runner

    monkeypatch.setattr(runner, "run_dated_reference", lambda **kwargs: kwargs)
    assert runner.run_stage5() == {"seed": 42, "device": None}


def test_dated_feature_matrix_is_identical_to_physically_truncated_ledger():
    import numpy as np
    from recommender.experiments.roadmap_v2 import dated_baskets as db

    pids = ["a", "b"]
    catalogue = pd.DataFrame({
        "product_id": pids, "base_name": ["oil 5w30", "oil 5w40"], "product_category": "OIL",
        "brand": "BR", "business_line": "L", "product_family": "F", "product_subcategory": "S",
    })
    companies = pd.DataFrame({"customer_id": ["old", "cold"], "industry_branch": "I"})
    date = pd.Timestamp("2025-12-02")
    events = pd.DataFrame([
        ("old", "a", "2025-10-05"), ("old", "b", "2025-11-25"), ("cold", "a", "2025-12-02"),
        ("old", "a", "2025-12-02"), ("old", "b", "2025-12-03"),
    ], columns=["customer_id", "product_id", "purchase_date"]).assign(
        quantity=2.0, revenue=5.0, cost=-2.0, cost_usd=-0.1, doc_bonus=0.0,
        manager_id="M", payment_terms="ДН14", full_payment_date=pd.Timestamp("2025-12-03"))
    events["purchase_date"] = pd.to_datetime(events.purchase_date)
    prefix = events[events.purchase_date < date].copy()
    prefix["full_payment_date"] = pd.NaT

    def matrices(source):
        dynamic = db.DynamicFeatureBuilder(source, catalogue, companies, pids,
                                           statistics_policy=db.STATISTICS_POLICY_ROLLING)
        base = db.BaseNameRollingFeatureBuilder(source, catalogue, companies, pids)
        commercial = db.CommercialRollingFeatureBuilder(source, catalogue, pids)
        specs = db.ParsedSpecRollingFeatureBuilder(source, catalogue, companies, pids)
        return [db.dated_candidate_features(cid, date, [0, 1], dynamic, base, commercial, specs)
                for cid in companies.customer_id]

    old, cold = matrices(events)
    old_prefix, cold_prefix = matrices(prefix)
    np.testing.assert_array_equal(old, old_prefix)
    np.testing.assert_array_equal(cold, cold_prefix)
    assert old[:, 0].all() and not cold[:, 0].any()
    assert cold[:, db.FEATURE_NAMES.index("base_name_global_pop_norm")].min() > 0
