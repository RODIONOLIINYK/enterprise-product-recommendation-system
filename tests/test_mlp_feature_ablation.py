"""Synthetic tests for Stage 1 MLP feature-removal experiment (300D physical omission & overall recall)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recommender.experiments.dynamic_selector.feature_schema import NAMES as SOURCE_CONTEXT_NAMES
from recommender.experiments.mlp_feature_ablation.config import OMITTED_CONTEXT_FEATURES
from recommender.experiments.mlp_feature_ablation.evaluate import (
    evaluate_full_pool_metrics,
    paired_feature_ablation_intervals,
)
from recommender.experiments.mlp_feature_ablation.model import (
    FeatureAblatedStage1Selector,
    load_ablated_selector_checkpoint,
)
from recommender.experiments.mlp_feature_ablation.schema import (
    build_feature_ablation_spec,
    build_input_contract,
    select_retained_context_features,
    validate_input_contract,
)
from recommender.experiments.static_enrichment.model import CompactStage1Selector


def test_exact_name_based_removal_preserves_remaining_order():
    spec = build_feature_ablation_spec(OMITTED_CONTEXT_FEATURES)
    assert spec.source_num_features == 41
    assert spec.omitted_names == ("is_repeat_sku", "month_sin", "month_cos")
    assert spec.omitted_indices == (0, 31, 32)
    assert spec.num_retained_features == 38
    assert spec.scalar_feature_count == 44
    assert spec.input_dim == 300

    # Verify names resolved from ordered schema and remaining 38 preserve exact relative order.
    expected_retained = tuple(
        n for n in SOURCE_CONTEXT_NAMES if n not in {"is_repeat_sku", "month_sin", "month_cos"}
    )
    assert spec.retained_names == expected_retained
    assert spec.retained_indices == tuple(range(1, 31)) + tuple(range(33, 41))
    assert [SOURCE_CONTEXT_NAMES[i] for i in spec.retained_indices] == list(spec.retained_names)

    # Reordered schema should resolve by name, NOT by magic offsets 0, 31, 32.
    reordered = tuple(reversed(SOURCE_CONTEXT_NAMES))
    spec_rev = build_feature_ablation_spec(
        OMITTED_CONTEXT_FEATURES,
        source_names=reordered,
        expected_omitted_indices=None,
    )
    assert spec_rev.omitted_indices == (
        reordered.index("is_repeat_sku"),
        reordered.index("month_sin"),
        reordered.index("month_cos"),
    )
    assert spec_rev.retained_names == tuple(
        n for n in reordered if n not in {"is_repeat_sku", "month_sin", "month_cos"}
    )

    with pytest.raises(ValueError, match="not found in canonical schema"):
        build_feature_ablation_spec(("nonexistent_feature",))
    with pytest.raises(ValueError, match="Duplicate feature names"):
        build_feature_ablation_spec(("month_sin", "month_sin"))


def test_300d_architecture_physical_shape_and_gradients():
    torch.manual_seed(42)
    spec = build_feature_ablation_spec()
    model = FeatureAblatedStage1Selector(spec)

    # Physically 300 input weights, NOT 303.
    assert model.input_dim == 300
    assert model.head[0].in_features == 300
    assert model.head[0].out_features == 128
    assert isinstance(model.head[1], torch.nn.LayerNorm)
    assert isinstance(model.head[2], torch.nn.GELU)
    assert isinstance(model.head[3], torch.nn.Dropout) and model.head[3].p == pytest.approx(0.1)
    assert model.head[4].in_features == 128 and model.head[4].out_features == 64
    assert model.head[8].in_features == 64 and model.head[8].out_features == 1
    assert sum(p.numel() for p in model.parameters()) == 47233

    B, P = 3, 7
    u = torch.randn(B, 128)
    v = torch.randn(P, 128)
    stats = torch.rand(B, 6)
    f41 = torch.rand(B, P, 41, requires_grad=True)
    target = torch.zeros(B, P)
    target[:, 0] = 1.0

    logits = model(u=u, v=v, features_41=f41, customer_stats_6=stats)
    assert logits.shape == (B, P)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    # Gradients MUST be strictly zero on omitted columns (0, 31, 32) and non-zero on retained columns.
    for idx in spec.omitted_indices:
        assert torch.all(f41.grad[..., idx] == 0.0)
    retained_grad_norm = f41.grad[..., list(spec.retained_indices)].abs().sum().item()
    assert retained_grad_norm > 0.0

    # Chunked forward matches unchunked forward identically.
    model.eval()
    with torch.no_grad():
        full = model(u=u, v=v, features_41=f41.detach(), customer_stats_6=stats)
        chunked = model(
            u=u, v=v, features_41=f41.detach(), customer_stats_6=stats, product_chunk_size=3
        )
        torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)

        # Direct 38D context input matches 41D sliced input identically.
        f38 = select_retained_context_features(f41.detach(), spec)
        assert f38.shape == (B, P, 38)
        direct = model(u=u, v=v, context_38=f38, customer_stats_6=stats)
        torch.testing.assert_close(full, direct, atol=0.0, rtol=0.0)


def test_omission_invariance_and_controllable_retained_dependence():
    torch.manual_seed(7)
    spec = build_feature_ablation_spec()
    model = FeatureAblatedStage1Selector(spec)
    model.eval()

    B, P = 4, 6
    u = torch.randn(B, 128)
    v = torch.randn(P, 128)
    stats = torch.rand(B, 6)
    f41 = torch.rand(B, P, 41)

    with torch.no_grad():
        base_logits = model(u=u, v=v, features_41=f41, customer_stats_6=stats)

        # Extreme perturbations on omitted columns (0, 31, 32) produce exact zero logit delta.
        f41_perturbed_omitted = f41.clone()
        f41_perturbed_omitted[..., 0] = 1e6
        f41_perturbed_omitted[..., 31] = -1e6
        f41_perturbed_omitted[..., 32] = 42.0
        omitted_logits = model(
            u=u, v=v, features_41=f41_perturbed_omitted, customer_stats_6=stats
        )
        torch.testing.assert_close(base_logits, omitted_logits, atol=0.0, rtol=0.0)

    # Controlled dependence: make a retained feature directly drive logits after gradient steps.
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    f_ctrl = torch.zeros(2, 2, 41)
    # Positive item uses retained feature 'user_sku_order_count_norm' (index 1) = 1.0, negative = 0.0.
    f_ctrl[:, 0, 1] = 1.0
    y_ctrl = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    for _ in range(10):
        opt.zero_grad()
        out = model(
            u=torch.zeros(2, 128),
            v=torch.zeros(2, 128),
            features_41=f_ctrl,
            customer_stats_6=torch.zeros(2, 6),
        )
        F.binary_cross_entropy_with_logits(out, y_ctrl).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_orig = model(
            u=torch.zeros(2, 128),
            v=torch.zeros(2, 128),
            features_41=f_ctrl,
            customer_stats_6=torch.zeros(2, 6),
        )
        f_flipped = f_ctrl.clone()
        f_flipped[:, 0, 1] = 0.0
        f_flipped[:, 1, 1] = 1.0
        out_flipped = model(
            u=torch.zeros(2, 128),
            v=torch.zeros(2, 128),
            features_41=f_flipped,
            customer_stats_6=torch.zeros(2, 6),
        )
        assert float((out_orig[:, 0] - out_orig[:, 1]).min()) > 0.5
        assert float((out_flipped[:, 1] - out_flipped[:, 0]).min()) > 0.5


def test_strict_persisted_contract_validation_and_checkpoint_loading(tmp_path):
    torch.manual_seed(42)
    spec = build_feature_ablation_spec()
    model = FeatureAblatedStage1Selector(spec)
    contract = model.input_contract()
    ckpt_path = tmp_path / "good_300.pt"
    torch.save(
        {
            "epoch": 1,
            "input_contract": contract,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    loaded_model, loaded_ckpt = load_ablated_selector_checkpoint(ckpt_path, expected_spec=spec)
    assert loaded_ckpt["epoch"] == 1
    assert loaded_model.input_dim == 300

    # Also loads cleanly when expected_spec is omitted (self-validates persisted contract).
    loaded_auto, _ = load_ablated_selector_checkpoint(ckpt_path)
    assert loaded_auto.spec.retained_names == spec.retained_names

    # Reject missing contract.
    missing_contract_path = tmp_path / "missing_contract.pt"
    torch.save({"model_state_dict": model.state_dict()}, missing_contract_path)
    with pytest.raises(ValueError, match="missing required persisted 'input_contract'"):
        load_ablated_selector_checkpoint(missing_contract_path, expected_spec=spec)

    # Reject retained feature name mismatch.
    bad_name_contract = dict(contract)
    bad_names = list(contract["retained_context_names"])
    bad_names[0] = "is_repeat_sku"
    bad_name_contract["retained_context_names"] = bad_names
    with pytest.raises(ValueError, match="digest corrupted|mismatch"):
        validate_input_contract(bad_name_contract, contract)

    # Reject order mismatch even with valid self-digest (reordered retained_names).
    swapped_names = list(spec.retained_names)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
    swapped_contract = dict(contract)
    swapped_contract["retained_context_names"] = swapped_names
    swapped_path = tmp_path / "swapped_order.pt"
    torch.save(
        {"input_contract": swapped_contract, "model_state_dict": model.state_dict()},
        swapped_path,
    )
    with pytest.raises(ValueError, match="digest corrupted|mismatch"):
        load_ablated_selector_checkpoint(swapped_path, expected_spec=spec)

    # Reject a different omitted set (width/contract mismatch) with valid self-digest.
    other_spec = build_feature_ablation_spec(("is_repeat_sku", "month_sin"))
    other_model = FeatureAblatedStage1Selector(other_spec)
    other_path = tmp_path / "other_301.pt"
    torch.save(
        {
            "input_contract": build_input_contract(other_spec),
            "model_state_dict": other_model.state_dict(),
        },
        other_path,
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_ablated_selector_checkpoint(other_path, expected_spec=spec)

    # Reject forged contract claiming 300D while state_dict weights are 303D.
    legacy_303 = CompactStage1Selector(recipe="uv_47")
    forged_303_path = tmp_path / "forged_303.pt"
    torch.save(
        {"input_contract": contract, "model_state_dict": legacy_303.state_dict()},
        forged_303_path,
    )
    with pytest.raises(ValueError, match="weight width mismatch"):
        load_ablated_selector_checkpoint(forged_303_path, expected_spec=spec)


def test_old_303_compact_selector_compatibility(tmp_path):
    torch.manual_seed(42)
    legacy = CompactStage1Selector(recipe="uv_47")
    assert legacy.input_dim == 303
    assert legacy.head[0].in_features == 303
    assert sum(p.numel() for p in legacy.parameters()) == 47617

    ckpt_path = tmp_path / "legacy_303.pt"
    torch.save(
        {
            "input_contract": legacy.input_contract(),
            "model_state_dict": legacy.state_dict(),
        },
        ckpt_path,
    )
    loaded = CompactStage1Selector(recipe="uv_47")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    loaded.eval()

    u = torch.randn(2, 128)
    v = torch.randn(5, 128)
    f41 = torch.rand(2, 5, 41)
    s6 = torch.rand(2, 6)
    with torch.no_grad():
        out = loaded(u=u, v=v, features_41=f41, customer_stats_6=s6)
    assert out.shape == (2, 5)


def test_overall_macro_recall_vs_unseen_only_discovery_semantics(tmp_path):
    # Construct 2 queries x 30 items where repeat items occupy top-10 and a new target sits at rank 12 overall
    # (outside ordinary top-10, inside unseen top-25!).
    # Query 0: targets = {0 (repeat), 15 (new)}; allowed = all 30; unseen = {12..29}
    #   Scores descending: 0..11 (seen), then 12..14 (unseen distractors), then 15 (new target at overall rank 15, unseen rank 3).
    #   Ordinary top-5: {0,1,2,3,4} -> 1/2 = 0.5 overall recall@5, new_hits@10 = 0.
    #   Ordinary top-25: hits both {0, 15} -> 1.0 overall recall@25.
    #   Unseen-only top-25: hits item 15 (rank 3 unseen) -> new_hits@25 = 1, even though new_hits@10 = 0!
    # Query 1: targets = {0 (repeat), 1 (repeat), 16 (new)}; item 29 is ineligible with huge score 999.0.
    #   Item 16 sits at overall rank 8 -> hits in ordinary top-10 (new_hits@10 = 1) and unseen top-25 (new_hits@25 = 1).
    P = 30
    scores = torch.zeros(2, P, dtype=torch.float32)
    scores[0] = torch.arange(P, 0, -1, dtype=torch.float32)
    # Put item 16 at rank 8 (score 25.0) for Query 1, and ineligible item 29 at score 999.0.
    scores[1] = torch.arange(P, 0, -1, dtype=torch.float32)
    scores[1, 16] = 22.5  # between item 7 (score 23) and item 8 (score 22) -> overall rank 8
    scores[1, 29] = 999.0  # ineligible!

    allowed = torch.ones(2, P, dtype=torch.bool)
    allowed[1, 29] = False

    unseen = torch.zeros(2, P, dtype=torch.bool)
    unseen[:, 12:] = True
    unseen[1, 29] = False

    target = torch.zeros(2, P, dtype=torch.bool)
    target[0, [0, 15]] = True
    target[1, [0, 1, 16]] = True

    new_target = torch.zeros(2, P, dtype=torch.bool)
    new_target[0, 15] = True
    new_target[1, 16] = True

    val_data = {
        "allowed": allowed,
        "unseen": unseen,
        "target": target,
        "new_target": new_target,
        "source": {
            "customer_ids": ["C1", "C2"],
            "positive_pids": [["P0", "P15"], ["P0", "P1", "P16"]],
            "new_positive_pids": [["P15"], ["P16"]],
            "attention_mask": torch.ones(2, 32, dtype=torch.long),
        },
    }

    npz_path = tmp_path / "preds.npz"
    m = evaluate_full_pool_metrics(scores, val_data, overall_ks=(5, 10, 25, 50, 100), save_predictions=npz_path)

    # Query 0: recall@5 = 0.5, recall@10 = 0.5, recall@25 = 1.0, new_hits@10 = 0, new_hits@25 = 1.
    # Query 1: recall@5 = 2/3 (ineligible item 29 ignored!), recall@10 = 1.0, new_hits@10 = 1, new_hits@25 = 1.
    assert m["recall@5"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)
    assert m["recall@10"] == pytest.approx((0.5 + 1.0) / 2.0)
    assert m["recall@25"] == pytest.approx(1.0)
    assert m["recall@50"] == pytest.approx(1.0)
    assert m["recall@100"] == pytest.approx(1.0)
    assert m["new_hits@10"] == 1
    assert m["new_hits@25"] == 2
    assert m["new_hits@50"] == 2
    assert m["new_hits@100"] == 2
    assert m["new_targets"] == 2

    # Paired bootstrap check against slightly weaker scores.
    npz_ctrl = tmp_path / "ctrl.npz"
    scores_ctrl = scores.clone()
    scores_ctrl[0, 0] = 0.1  # Query 0 misses P0 in top-5 and top-10
    evaluate_full_pool_metrics(scores_ctrl, val_data, overall_ks=(5, 10, 25, 50, 100), save_predictions=npz_ctrl)
    npz_trt = tmp_path / "trt.npz"
    evaluate_full_pool_metrics(scores, val_data, overall_ks=(5, 10, 25, 50, 100), save_predictions=npz_trt)
    cis = paired_feature_ablation_intervals(npz_ctrl, npz_trt, repeats=200)
    assert "recall@5_delta" in cis and "recall@100_delta" in cis and "new_recall@50_delta" in cis
    assert cis["recall@5_delta"][1] > 0.0


def test_data_date_gates_and_no_tower_fallback(tmp_path, monkeypatch):
    import pandas as pd
    import recommender.experiments.mlp_feature_ablation.data as mfa_data
    from recommender.split import TEST_START, TRAIN_CUTOFF

    origin = pd.Timestamp("2025-01-01")
    train_ok = torch.tensor([(TRAIN_CUTOFF - origin).days - 1])
    val_ok = torch.tensor([(TRAIN_CUTOFF - origin).days, (TEST_START - origin).days - 1])
    heldout_bad = torch.tensor([(TEST_START - origin).days])

    mfa_data._assert_dev_dates_strict("train", train_ok)
    mfa_data._assert_dev_dates_strict("val", val_ok)
    with pytest.raises(ValueError, match="Held-out dates"):
        mfa_data._assert_dev_dates_strict("val", heldout_bad)
    with pytest.raises(ValueError, match="Validation/test dates"):
        mfa_data._assert_dev_dates_strict("train", val_ok)

    # Missing retrieval cache raises FileNotFoundError instead of regenerating raw data.
    monkeypatch.setattr(mfa_data, "E1_E5_RETRIEVAL_CACHE_DIR", tmp_path / "missing_ret")
    monkeypatch.setattr(mfa_data, "verify_no_brand_reference", lambda: {"manifest": {"source_identities": {"enrichment_identity": "x"}}})
    with pytest.raises(FileNotFoundError, match="refusing raw regeneration"):
        mfa_data.load_verified_no_brand_campaign_data()
