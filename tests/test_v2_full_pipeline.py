"""Unit tests verifying v2_full split-boundary configurability and zero leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.experiments.roadmap_v2.augmentation import build_strict_60d_aux_labels
from recommender.experiments.roadmap_v2.commercial_features import CommercialRollingFeatureBuilder
from recommender.experiments.roadmap_v2.dated_baskets import aligned_commercial_events
from recommender.experiments.v2_full.config import TEST_START_V2, TRAIN_CUTOFF_V2


def test_v2_full_split_boundaries_enforced():
    assert TRAIN_CUTOFF_V2 == pd.Timestamp("2026-02-01")
    assert TEST_START_V2 == pd.Timestamp("2026-07-01")

    cat = pd.DataFrame(
        [{"product_id": "P1", "base_name": "Oil A", "brand": "B1", "product_category": "C1"}]
    )
    valid_events = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2026-05-15"),
                "product_id": "P1",
                "quantity": 10.0,
                "transaction_type": "ПРОДАЖА",
                "item_type": "ТОВАР",
                "cost": -100.0,
                "cost_usd": -2.5,
                "revenue": 150.0,
                "doc_bonus": 0.0,
                "payment_terms": "14 ДНЕЙ",
                "payment_type": "Б/Н",
                "manager_id": "M1",
                "full_payment_date": pd.Timestamp("2026-08-01"),  # post-TEST_START_V2 -> must be masked!
            }
        ]
    )
    purchases = valid_events[["customer_id", "purchase_date", "product_id", "quantity"]].copy()
    aligned = aligned_commercial_events(valid_events, purchases, test_start=TEST_START_V2)
    assert pd.isna(aligned["full_payment_date"].iloc[0])

    builder = CommercialRollingFeatureBuilder(
        aligned, cat, ["P1"], allow_heldout_test=False, test_start=TEST_START_V2
    )
    out = builder.build("C1", pd.Timestamp("2026-06-30"))
    assert out.shape == (1, 21) and np.isfinite(out).all()

    with pytest.raises(ValueError, match="Development predictions only"):
        builder.build("C1", pd.Timestamp("2026-07-01"))

    post_test = valid_events.assign(purchase_date=pd.Timestamp("2026-07-01"))
    with pytest.raises(ValueError, match="Development events only"):
        CommercialRollingFeatureBuilder(post_test, cat, ["P1"], allow_heldout_test=False, test_start=TEST_START_V2)

    # 60d aux labels must cap strictly at TRAIN_CUTOFF_V2 and reject >= TEST_START_V2
    tr_p = pd.DataFrame(
        [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2026-01-20"), "product_id": "P1"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2026-02-05"), "product_id": "P1"},  # val!
        ]
    )
    origin = pd.Timestamp("2025-01-01")
    d_q = (pd.Timestamp("2026-01-25") - origin).days
    aux = build_strict_60d_aux_labels(
        tr_p, ["C1"], [d_q], ["P1"], train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2
    )
    assert not bool(aux[0, 0].item())  # 2026-02-05 val purchase MUST NOT leak into 60d train aux label!


def test_v2_full_universe_and_sequences_handle_2023_to_2026_dates():
    from recommender.experiments.roadmap_v2.stage5_base_entity import (
        build_base_name_sequences,
        build_base_name_universe,
    )

    cat = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "base_name": "Oil A",
                "product_category": "CAT1",
                "product_subcategory": "SUB1",
                "business_line": "L1",
            },
            {
                "product_id": "P2",
                "base_name": "Oil B",
                "product_category": "CAT1",
                "product_subcategory": "SUB1",
                "business_line": "L1",
            },
        ]
    )
    purchases = pd.DataFrame(
        [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2023-03-10"), "product_id": "P1", "quantity": 5.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2026-01-15"), "product_id": "P1", "quantity": 10.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2026-05-20"), "product_id": "P2", "quantity": 4.0},
        ]
    )
    univ = build_base_name_universe(["P1", "P2"], cat, purchases, train_cutoff=TRAIN_CUTOFF_V2)
    assert univ["trained_base_names"] == 1
    assert univ["base_names"] == ["Oil A", "Oil B"]

    origin = pd.Timestamp("2025-01-01")
    pred_day = (pd.Timestamp("2026-06-15") - origin).days
    seqs = build_base_name_sequences(
        purchases,
        ["C1"],
        [pred_day],
        univ,
        cat,
        {"CAT1": 1},
        {"SUB1": 1},
        {"L1": 1},
        test_start=TEST_START_V2,
    )
    assert int(seqs["attention_mask"][0].sum().item()) == 3
    # 2023-03-10 has negative day offset relative to 2025-01-01; verify exact round-trip days
    d0 = round(float(seqs["seq_purchase_dates"][0, 0].item()) * 365.25)
    assert d0 == (pd.Timestamp("2023-03-10") - origin).days
    with pytest.raises(ValueError, match="Development purchases only"):
        build_base_name_sequences(
            purchases.assign(purchase_date=pd.Timestamp("2026-07-01")),
            ["C1"],
            [pred_day],
            univ,
            cat,
            {"CAT1": 1},
            {"SUB1": 1},
            {"L1": 1},
            test_start=TEST_START_V2,
        )


def test_e2e_stage2_ranker_gradients_causality_and_cached_v_bn_parity():
    import torch
    from recommender.experiments.v2_full.e2e_ranker import EndToEndStage2Ranker

    torch.manual_seed(0)
    b, c, n_tr_c, s = 3, 12, 9, 6
    attr_vocabs = {"category": 4, "subcategory": 5, "business_line": 3, "tier": 3, "unit": 3, "country_of_origin": 3}
    model = EndToEndStage2Ranker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        num_demand_features=65,
        num_candidate_features=38,
        num_customer_features=11,
        aux_60d_enabled=True,
        unseen_attn_aux_enabled=True,
        seed=7,
    )

    prod_inputs = {
        "text_embedding": torch.randn(c, 1024),
        "demand_features": torch.rand(c, 65),
        "attr_ids": {k: torch.randint(0, sz, (c,)) for k, sz in attr_vocabs.items()},
    }
    # Put NaNs in val-only product rows [n_tr_c:] to prove train_bn_limit never touches them!
    prod_inputs_tr = {
        "text_embedding": prod_inputs["text_embedding"].clone(),
        "demand_features": prod_inputs["demand_features"].clone(),
        "attr_ids": {k: v.clone() for k, v in prod_inputs["attr_ids"].items()},
    }
    prod_inputs_tr["text_embedding"][n_tr_c:] = float("nan")
    prod_inputs_tr["demand_features"][n_tr_c:] = float("nan")

    attn_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.long)
    seq_days_raw = torch.tensor([[10.0, 25.0, 40.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6])
    pred_day = torch.tensor([60.0, 60.0, 60.0])
    cand_feats = torch.rand(b, c, 38)
    cand_feats[:, :, 0] = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * b, dtype=torch.float32)
    cand_feats[:, :, 3] = np.log1p(35.0) / np.log1p(365.0)
    cand_mask = torch.zeros(b, c, dtype=torch.bool)
    cand_mask[:, :n_tr_c] = True

    batch = {
        "company_text_embedding": torch.randn(b, 1024),
        "city_id": torch.randint(0, 5, (b,)),
        "region_id": torch.randint(0, 4, (b,)),
        "industry_id": torch.randint(0, 4, (b,)),
        "customer_features": torch.rand(b, 11),
        "sequence": torch.rand(b, s, 9),
        "attention_mask": attn_mask,
        "seq_product_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 0, 0, 0, 0, 0], [0] * 6], dtype=torch.long),
        "seq_purchase_dates": seq_days_raw / 365.25,
        "prediction_day": pred_day,
        "candidate_features": cand_feats,
        "candidate_mask": cand_mask,
    }

    # Forward + backward with train_bn_limit must stay 100% finite and flow gradients into every component
    res = model(**batch, prod_inputs=prod_inputs_tr, train_bn_limit=n_tr_c, return_aux=True)
    logits = res["output"].logits[:, :n_tr_c]
    aux_lg = res["aux_60d_logits"][:, :n_tr_c]
    uattn_lg = res["unseen_attn_logits"][:, :n_tr_c]
    assert torch.isfinite(logits).all() and torch.isfinite(aux_lg).all() and torch.isfinite(uattn_lg).all()
    loss = logits.pow(2).mean() + aux_lg.pow(2).mean() + uattn_lg.pow(2).mean()
    loss.backward()
    for pname in (
        "product_tower.text_proj.0.weight",
        "cust_text_proj.0.weight",
        "statistics_attention_weights",
        "unseen_attn_head.0.weight",
    ):
        param = dict(model.named_parameters())[pname]
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0

    # Verify exact parity between live prod_inputs and precomputed_v_bn in eval mode, plus stripped inference load
    model.eval()
    eval_batch = {**batch, "candidate_mask": torch.ones(b, c, dtype=torch.bool)}
    with torch.no_grad():
        out_live = model(**eval_batch, prod_inputs=prod_inputs, return_aux=False)
        v_cached = model.precompute_product_vectors(prod_inputs)
        out_cached = model(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
        assert torch.allclose(out_live.logits, out_cached.logits, atol=1e-6)

    inf_model = EndToEndStage2Ranker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        aux_60d_enabled=False,
        unseen_attn_aux_enabled=False,
        seed=7,
    ).eval()
    stripped = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("aux_60d_head.") and not k.startswith("unseen_attn_head.")
    }
    inf_model.load_state_dict(stripped, strict=True)
    with torch.no_grad():
        out_inf = inf_model(**eval_batch, prod_inputs=prod_inputs, return_aux=False)
        assert torch.allclose(out_live.logits, out_inf.logits, atol=1e-6)


def test_two_stage_adaptive_ranker_and_block_shuffled_batches():
    import torch
    from recommender.experiments.roadmap_v2.clean_ranker import SharedUVStage5Ranker
    from recommender.experiments.static_enrichment.model import StaticEnrichedProductTower
    from recommender.experiments.v2_full.e2e_ranker import TwoStageAdaptiveRanker
    from recommender.experiments.v2_full.train_v2_full_baseline import iter_block_shuffled_batches

    # 1. Verify block-shuffled batch iterator covers every row exactly once with exact feature alignment
    arr = np.arange(105 * 4 * 2, dtype=np.float16).reshape(105, 4, 2)
    gen = torch.Generator().manual_seed(123)
    seen = []
    for rows, cand_np in iter_block_shuffled_batches(arr, 105, 16, gen, block_rows=20, blocks_per_buffer=3):
        seen.extend(rows.tolist())
        assert np.allclose(cand_np, arr[rows.numpy()].astype(np.float32))
    assert sorted(seen) == list(range(105))

    # 2. Verify TwoStageAdaptiveRanker flows gradients into unfrozen ProductTower + unseen_attn_head
    torch.manual_seed(1)
    b, c, n_tr_c, s = 2, 10, 7, 5
    attr_vocabs = {"category": 4, "subcategory": 4, "business_line": 3, "tier": 3, "unit": 3, "country_of_origin": 3}
    pt = StaticEnrichedProductTower(attribute_vocab_sizes=attr_vocabs, num_demand_features=65)
    base = SharedUVStage5Ranker(aux_60d_enabled=True, seed=1)
    model = TwoStageAdaptiveRanker(ranker=base, product_tower=pt, unseen_attn_aux_enabled=True)

    prod_inputs = {
        "text_embedding": torch.randn(c, 1024),
        "demand_features": torch.rand(c, 65),
        "attr_ids": {k: torch.randint(0, sz, (c,)) for k, sz in attr_vocabs.items()},
    }
    cmask = torch.zeros(b, c, dtype=torch.bool)
    cmask[:, :n_tr_c] = True
    batch = {
        "customer_vector": torch.randn(b, 128),
        "customer_features": torch.rand(b, 11),
        "sequence": torch.rand(b, s, 9),
        "attention_mask": torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long),
        "seq_product_ids": torch.tensor([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long),
        "seq_purchase_dates": torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.0], [0.0] * 5]),
        "prediction_day": torch.tensor([100.0, 100.0]),
        "candidate_features": torch.rand(b, c, 38),
        "candidate_mask": cmask,
    }
    res = model(**batch, prod_inputs=prod_inputs, train_bn_limit=n_tr_c, return_aux=True)
    loss = (
        res["output"].logits[:, :n_tr_c].pow(2).mean()
        + res["aux_60d_logits"][:, :n_tr_c].pow(2).mean()
        + res["unseen_attn_logits"][:, :n_tr_c].pow(2).mean()
    )
    loss.backward()
    assert model.product_tower.text_proj[0].weight.grad is not None
    assert model.product_tower.text_proj[0].weight.grad.abs().sum().item() > 0
    assert model.unseen_attn_head[0].weight.grad.abs().sum().item() > 0


def test_empty_history_still_validates_auxiliary_query_dates():
    origin = pd.Timestamp("2025-01-01")
    cutoff_day = (TRAIN_CUTOFF_V2 - origin).days
    with pytest.raises(ValueError, match="training-only"):
        build_strict_60d_aux_labels(pd.DataFrame(), ["C"], [cutoff_day], ["P"],
                                    train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2)
    with pytest.raises(ValueError, match="prediction day"):
        build_strict_60d_aux_labels(pd.DataFrame(), ["C"], [], ["P"])


def test_sku_ranking_excludes_padding_and_preserves_low_finite_scores():
    import torch
    from recommender.experiments.roadmap_v2.stage5_base_entity import collapse_sku_scores_to_base_name_ranking

    scores = torch.tensor([[4., 3., -1e20, -torch.inf]])
    ids = torch.tensor([[-1, 1, 0, 2]])
    allowed = torch.tensor([[True, False, True, True]])
    ranks = collapse_sku_scores_to_base_name_ranking(scores, ids, np.array([0, 1, 2]), allowed_mask=allowed, max_k=2)
    assert ranks.tolist() == [[0, -1]]
    assert collapse_sku_scores_to_base_name_ranking(scores, ids, np.arange(3), max_k=0).shape == (1, 0)


def test_empty_cohort_metrics_are_finite():
    import torch
    from recommender.experiments.roadmap_v2.stage5_base_entity import evaluate_base_name_predictions

    empty = torch.empty(0, 3, dtype=torch.bool)
    metrics = evaluate_base_name_predictions(np.empty((0, 3), dtype=int), empty, empty, [])
    assert metrics["queries"] == 0
    assert all(value == 0 for value in metrics.values())
