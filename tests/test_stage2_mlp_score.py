"""Regression tests for Stage 2 UPS-CANR integration with frozen 301D MLP score."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES
from recommender.experiments.mlp_feature_ablation.config import (
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
)
from recommender.experiments.mlp_feature_ablation.model import FeatureAblatedStage1Selector
from recommender.experiments.mlp_feature_ablation.schema import build_feature_ablation_spec
from recommender.experiments.stage2_mlp_score.config import (
    ARTIFACT_ROOT,
    CACHE_ROOT,
    EVIDENCE_ROOT,
    SCORE_TRANSFORM_VERSION,
)
from recommender.experiments.stage2_mlp_score.evaluate import (
    evaluate_stage2_slate_metrics,
    paired_stage2_intervals,
)
from recommender.experiments.stage2_mlp_score.model import (
    Stage2MLPScoreInferenceAdapter,
    Stage2MLPScoreRanker,
    gather_teacher_scores_by_canonical_indices,
    load_stage2_mlp_score_checkpoint,
    save_stage2_mlp_score_checkpoint,
    transform_teacher_logits_to_mlp_score,
)
from recommender.ranker import UPSCANRRanker


def _make_synthetic_base_and_dyn(seed: int = 42) -> tuple[UPSCANRRanker, CandidateContextRanker]:
    torch.manual_seed(seed)
    base = UPSCANRRanker(
        text_dim=1024,
        memory_dim=128,
        num_cities=20,
        num_regions=10,
        num_industries=12,
        num_products=50,
        num_categories=10,
        num_subcategories=16,
        num_business_lines=8,
        num_tabular_features=8,
        num_pairwise_features=13,
        seq_feature_dim=9,
        hidden_dim=128,
        num_heads=4,
        statistics_attention=True,
        excluded_statistics=("history_repeat",),
        cached_only=True,
    )
    dyn = CandidateContextRanker(copy.deepcopy(base), "dynamic_features")
    return base, dyn


def _make_synthetic_ranker_inputs(batch_size: int = 3, slate_size: int = 6, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    return {
        "text_embedding": torch.randn(batch_size, 1024),
        "city_id": torch.randint(0, 20, (batch_size,)),
        "region_id": torch.randint(0, 10, (batch_size,)),
        "industry_id": torch.randint(0, 12, (batch_size,)),
        "tabular_features": torch.rand(batch_size, 8),
        "sequence": torch.rand(batch_size, 32, 9),
        "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        "seq_product_ids": torch.randint(0, 50, (batch_size, 32)),
        "seq_category_ids": torch.randint(0, 10, (batch_size, 32)),
        "seq_subcategory_ids": torch.randint(0, 16, (batch_size, 32)),
        "seq_business_line_ids": torch.randint(0, 8, (batch_size, 32)),
        "seq_purchase_dates": torch.rand(batch_size, 32) * 0.5,
        "prediction_day": torch.full((batch_size,), 300.0),
        "pairwise_features": torch.rand(batch_size, slate_size, 41),
        "candidate_vectors": torch.randn(batch_size, slate_size, 128),
        "cached_customer_vectors": torch.randn(batch_size, 128),
    }


def test_zero_extension_reproduces_425d_dynamic_ranker_at_init():
    _, dyn = _make_synthetic_base_and_dyn(42)
    dyn.eval()

    expanded = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=True)
    expanded.eval()

    assert expanded.base.num_pairwise_features == 14
    assert tuple(expanded.base.statistics_attention_weights.shape) == (4, 15, 17)
    assert expanded.head[0].in_features == 426
    assert tuple(expanded.base.excluded_statistics) == ("history_repeat",)
    assert all(not p.requires_grad for p in expanded.base.mlp.parameters())

    inputs = _make_synthetic_ranker_inputs(batch_size=4, slate_size=8, seed=11)
    mlp_score = torch.rand(4, 8)

    with torch.no_grad():
        old_out = dyn(**inputs)
        new_actual = expanded(**inputs, mlp_score=mlp_score)
        new_zero = expanded(**inputs, mlp_score=torch.zeros_like(mlp_score))

    torch.testing.assert_close(new_actual.logits, old_out.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(new_zero.logits, old_out.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(new_actual.attn_weights, old_out.attn_weights, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(new_actual.h_attended, old_out.h_attended, atol=1e-6, rtol=1e-6)


def test_cross_attention_and_head_score_observability_and_gradients():
    _, dyn = _make_synthetic_base_and_dyn(42)
    treatment = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=True)
    treatment.eval()

    inputs = _make_synthetic_ranker_inputs(batch_size=3, slate_size=6, seed=19)
    score_hi = torch.linspace(0.05, 0.95, 18).view(3, 6)
    score_zero = torch.zeros_like(score_hi)

    # Perturb ONLY statistics_attention_weights[:, 14, :] while head score column 397 remains 0.
    # This proves cross-attention genuinely uses mlp_score as the 14th candidate statistic.
    with torch.no_grad():
        treatment.base.statistics_attention_weights[:, 14, :].fill_(2.5)
        treatment.head[0].weight[:, 397].zero_()
        out_hi = treatment(**inputs, mlp_score=score_hi)
        out_lo = treatment(**inputs, mlp_score=score_zero)

    attn_diff = float((out_hi.attn_weights - out_lo.attn_weights).abs().max().item())
    h_att_diff = float((out_hi.h_attended - out_lo.h_attended).abs().max().item())
    logit_diff = float((out_hi.logits - out_lo.logits).abs().max().item())
    assert attn_diff > 1e-3
    assert h_att_diff > 1e-3
    assert logit_diff > 1e-3

    # Gradients flow into both W[:, 14, :] and head[0].weight[:, 397].
    treatment.train()
    treatment.zero_grad(set_to_none=True)
    out = treatment(**inputs, mlp_score=score_hi)
    out.logits.sum().backward()
    assert treatment.base.statistics_attention_weights.grad is not None
    assert float(treatment.base.statistics_attention_weights.grad[:, 14, :].abs().sum().item()) > 0.0
    assert treatment.head[0].weight.grad is not None
    assert float(treatment.head[0].weight.grad[:, 397].abs().sum().item()) > 0.0


def test_control_score_channel_disabled_is_strictly_invariant():
    _, dyn = _make_synthetic_base_and_dyn(42)
    control = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=False)
    with torch.no_grad():
        control.base.statistics_attention_weights[:, 14, :].fill_(3.0)
        control.head[0].weight[:, 397].fill_(2.0)

    inputs = _make_synthetic_ranker_inputs(batch_size=3, slate_size=5, seed=23)
    mlp_score = torch.rand(3, 5)
    control.eval()
    with torch.no_grad():
        out_with = control(**inputs, mlp_score=mlp_score)
        out_none = control(**inputs, mlp_score=None)
        out_zero = control(**inputs, mlp_score=torch.zeros_like(mlp_score))

    torch.testing.assert_close(out_with.logits, out_none.logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_with.logits, out_zero.logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_with.attn_weights, out_none.attn_weights, atol=0.0, rtol=0.0)

    control.train()
    control.zero_grad(set_to_none=True)
    control(**inputs, mlp_score=mlp_score).logits.sum().backward()
    assert float(control.base.statistics_attention_weights.grad[:, 14, :].abs().sum().item()) == 0.0
    assert float(control.head[0].weight.grad[:, 397].abs().sum().item()) == 0.0


def test_input_validation_and_mask_isolation_against_nan_and_neginf():
    _, dyn = _make_synthetic_base_and_dyn(42)
    ranker = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=True)
    with torch.no_grad():
        ranker.base.statistics_attention_weights[:, 14, :].fill_(1.0)
        ranker.head[0].weight[:, 397].fill_(1.0)
    ranker.eval()

    # Pre-sigmoid masking prevents -1e9 / -inf / NaN on invalid positions from leaking.
    raw = torch.tensor([[1.5, -1e9, float("nan"), -2.0], [0.0, float("-inf"), 3.0, -1e9]])
    valid = torch.tensor([[True, False, False, True], [True, False, True, False]])
    probs = transform_teacher_logits_to_mlp_score(raw, valid_mask=valid)
    assert probs.dtype == torch.float32
    assert bool(torch.isfinite(probs).all())
    assert bool((probs[~valid] == 0.0).all())
    torch.testing.assert_close(
        probs[valid],
        torch.sigmoid(torch.tensor([1.5, -2.0, 0.0, 3.0])),
        atol=1e-6,
        rtol=1e-6,
    )

    # Reject NaN on valid position.
    with pytest.raises(ValueError, match="NaN or Inf"):
        transform_teacher_logits_to_mlp_score(raw, valid_mask=torch.ones_like(valid))

    # Reject wrong pairwise_features width (no silent _fit_feature_width truncation).
    bad_inputs = _make_synthetic_ranker_inputs(batch_size=2, slate_size=4)
    bad_inputs["pairwise_features"] = torch.rand(2, 4, 40)
    with pytest.raises(ValueError, match="Expected pairwise_features shape"):
        ranker(**bad_inputs, mlp_score=torch.rand(2, 4))

    # Reject out-of-range mlp_score or missing mlp_score when enabled.
    inputs = _make_synthetic_ranker_inputs(batch_size=2, slate_size=4)
    with pytest.raises(ValueError, match="mlp_score is required"):
        ranker(**inputs, mlp_score=None)
    with pytest.raises(ValueError, match="bounded within"):
        ranker(**inputs, mlp_score=torch.full((2, 4), 1.5))
    with pytest.raises(ValueError, match="bounded within"):
        ranker(**inputs, mlp_score=torch.full((2, 4), -0.1))

    # Masking padded candidates with NaN features, NaN candidate_vectors, and -1e9 raw logit -> no leak to valid slots.
    base_inputs = _make_synthetic_ranker_inputs(batch_size=2, slate_size=4, seed=99)
    base_score = torch.tensor([[0.8, 0.4, 0.2, 0.6], [0.3, 0.9, 0.1, 0.5]])
    with torch.no_grad():
        base_out = ranker(**base_inputs, mlp_score=base_score)

    padded_inputs = copy.deepcopy(base_inputs)
    padded_inputs["pairwise_features"] = torch.cat(
        [base_inputs["pairwise_features"], torch.full((2, 2, 41), float("nan"))],
        dim=1,
    )
    padded_inputs["candidate_vectors"] = torch.cat(
        [base_inputs["candidate_vectors"], torch.full((2, 2, 128), float("nan"))],
        dim=1,
    )
    padded_score = torch.cat([base_score, torch.full((2, 2), -1e9)], dim=1)
    cand_mask = torch.cat([torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        padded_out = ranker(**padded_inputs, mlp_score=padded_score, candidate_mask=cand_mask)

    torch.testing.assert_close(padded_out.logits[:, :4], base_out.logits, atol=1e-6, rtol=1e-6)
    assert bool(torch.all(padded_out.logits[:, 4:] == -torch.inf))
    assert bool(torch.isfinite(padded_out.h_attended).all())


def test_canonical_gather_sku_order_remap_and_adapter_slate_invariance():
    canonical_pids = ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]
    reordered_pids = ["SKU_C", "SKU_E", "SKU_A", "SKU_D", "SKU_B"]
    raw_canon = torch.tensor([[1.0, -2.0, 0.5, -1e9, 2.0]])
    remap_idx = [canonical_pids.index(p) for p in reordered_pids]
    raw_reordered = raw_canon[:, remap_idx]
    cand_idx = torch.tensor([[4, 0, 2, -1]])
    mask = torch.tensor([[True, True, True, False]])

    s1, r1 = gather_teacher_scores_by_canonical_indices(
        raw_canon, cand_idx, canonical_product_ids=canonical_pids, valid_mask=mask
    )
    s2, r2 = gather_teacher_scores_by_canonical_indices(
        raw_reordered,
        cand_idx,
        canonical_product_ids=canonical_pids,
        score_product_ids=reordered_pids,
        valid_mask=mask,
    )
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)

    # Adapter parity + slate permutation invariance + frozen teacher enforcement.
    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    torch.manual_seed(42)
    teacher = FeatureAblatedStage1Selector(spec)
    _, dyn = _make_synthetic_base_and_dyn(42)
    ranker = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=True)
    with torch.no_grad():
        ranker.base.statistics_attention_weights[:, 14, :].fill_(0.5)
        ranker.head[0].weight[:, 397].fill_(0.5)

    adapter = Stage2MLPScoreInferenceAdapter(teacher, ranker, expected_teacher_spec=spec)
    assert all(not p.requires_grad for p in adapter.teacher.parameters())
    assert all(not p.requires_grad for p in adapter.ranker.parameters())

    B, P, C = 2, 10, 4
    u = torch.randn(B, 128)
    v = torch.randn(P, 128)
    stats6 = torch.rand(B, 6)
    f41_pool = torch.rand(B, P, 41)
    idx = torch.tensor([[7, 2, 9, 0], [3, 5, 1, 8]])
    row_ar = torch.arange(B)[:, None]
    f41_slate = f41_pool[row_ar, idx]

    ranker_inputs = _make_synthetic_ranker_inputs(batch_size=B, slate_size=C, seed=55)
    ranker_inputs["pairwise_features"] = f41_slate
    ranker_inputs["candidate_vectors"] = v[idx]

    # Full-pool vs pre-gathered slate scores match identically, and flipping slate order permutes identically.
    s_pool = adapter.compute_mlp_score(
        u=u, v=v, features_41=f41_pool, customer_stats_6=stats6, candidate_indices=idx
    )
    s_slate = adapter.compute_mlp_score(
        u=u, v=v, features_41=f41_slate, customer_stats_6=stats6, candidate_indices=idx
    )
    torch.testing.assert_close(s_pool, s_slate, atol=1e-6, rtol=1e-6)

    out_fwd = adapter.score_candidates(ranker_inputs, u=u, v=v, customer_stats_6=stats6, candidate_indices=idx)
    rev_inputs = dict(ranker_inputs)
    rev_inputs["pairwise_features"] = torch.flip(f41_slate, dims=[1])
    rev_inputs["candidate_vectors"] = torch.flip(v[idx], dims=[1])
    out_rev = adapter.score_candidates(
        rev_inputs, u=u, v=v, customer_stats_6=stats6, candidate_indices=torch.flip(idx, dims=[1])
    )
    torch.testing.assert_close(torch.flip(out_rev.logits, dims=[1]), out_fwd.logits, atol=1e-5, rtol=1e-5)


def test_strict_checkpoint_serialization_and_rejection(tmp_path):
    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    teacher = FeatureAblatedStage1Selector(spec)
    teacher_contract = teacher.input_contract()
    _, dyn = _make_synthetic_base_and_dyn(42)
    ranker = Stage2MLPScoreRanker.from_dynamic_context_ranker(dyn, score_channel_enabled=True)
    with torch.no_grad():
        ranker.base.statistics_attention_weights[:, 14, :].fill_(0.25)
        ranker.head[0].weight[:, 397].fill_(-0.75)

    provenance = {
        "teacher_checkpoint_sha256": "t" * 64,
        "upstream_retriever_sha256": "u" * 64,
        "score_feature_transform": SCORE_TRANSFORM_VERSION,
        "score_channel_enabled": True,
    }
    ckpt_path = save_stage2_mlp_score_checkpoint(
        tmp_path / "valid.pt",
        ranker,
        epoch=3,
        metrics={"recall@10": 0.77, "ndcg@10": 0.61},
        selection_criterion="val_ndcg_at_10",
        provenance=provenance,
        teacher_input_contract=teacher_contract,
    )

    loaded, payload = load_stage2_mlp_score_checkpoint(
        ckpt_path,
        expected_provenance=provenance,
        expected_teacher_input_contract=teacher_contract,
        expected_score_channel_enabled=True,
    )
    assert payload["epoch"] == 3
    inputs = _make_synthetic_ranker_inputs(batch_size=2, slate_size=4)
    score = torch.rand(2, 4)
    ranker.eval()
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(**inputs, mlp_score=score).logits,
            ranker(**inputs, mlp_score=score).logits,
            atol=0.0,
            rtol=0.0,
        )

    # Reject teacher hash mismatch.
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_stage2_mlp_score_checkpoint(
            ckpt_path,
            expected_provenance={**provenance, "teacher_checkpoint_sha256": "x" * 64},
        )

    # Reject score_channel_enabled mismatch.
    with pytest.raises(ValueError, match="score_channel_enabled mismatch"):
        load_stage2_mlp_score_checkpoint(ckpt_path, expected_score_channel_enabled=False)

    # Reject wrong score transform or tampered feature list / state width.
    bad_transform = copy.deepcopy(payload)
    bad_transform["model_config"]["score_transform_version"] = "zscore"
    torch.save(bad_transform, tmp_path / "bad_transform.pt")
    with pytest.raises(ValueError, match="Score transform mismatch"):
        load_stage2_mlp_score_checkpoint(tmp_path / "bad_transform.pt")

    bad_order = copy.deepcopy(payload)
    bad_order["model_config"]["ordered_pairwise_features"] = list(reversed(DYNAMIC_FEATURE_NAMES))
    torch.save(bad_order, tmp_path / "bad_order.pt")
    with pytest.raises(ValueError, match="ordered_pairwise_features mismatch"):
        load_stage2_mlp_score_checkpoint(tmp_path / "bad_order.pt")

    bad_width = copy.deepcopy(payload)
    bad_width["model_state_dict"]["head.0.weight"] = torch.zeros(256, 425)
    torch.save(bad_width, tmp_path / "bad_width.pt")
    with pytest.raises(ValueError, match="head.0.weight width mismatch"):
        load_stage2_mlp_score_checkpoint(tmp_path / "bad_width.pt")


def test_slate_evaluation_recall_depths_and_paired_intervals(tmp_path):
    n_rows, n_prods, slate_size = 6, 120, 100
    indices = torch.arange(slate_size).unsqueeze(0).expand(n_rows, -1).clone()
    allowed = torch.ones(n_rows, n_prods, dtype=torch.bool)
    target = torch.zeros(n_rows, n_prods, dtype=torch.bool)
    new_target = torch.zeros(n_rows, n_prods, dtype=torch.bool)
    for i in range(n_rows):
        target[i, 0] = True
        target[i, 30] = True
        new_target[i, 30] = True

    data = {
        "allowed": allowed,
        "target": target,
        "new_target": new_target,
        "source": {
            "customer_ids": [f"C{i % 3}" for i in range(n_rows)],
            "positive_pids": [["P0", "P30"] for _ in range(n_rows)],
            "new_positive_pids": [["P30"] for _ in range(n_rows)],
            "attention_mask": torch.ones(n_rows, 32, dtype=torch.long),
        },
    }
    ctrl_scores = torch.linspace(1.0, 0.0, slate_size).unsqueeze(0).expand(n_rows, -1).clone()
    trt_scores = ctrl_scores.clone()
    trt_scores[:, 30] = 2.0  # Promote new target from rank 30 into rank 0

    m_ctrl = evaluate_stage2_slate_metrics(ctrl_scores, data, indices, tmp_path / "ctrl.npz")
    m_trt = evaluate_stage2_slate_metrics(trt_scores, data, indices, tmp_path / "trt.npz")

    assert m_ctrl["recall@10"] == pytest.approx(0.5)
    assert m_ctrl["recall@25"] == pytest.approx(0.5)
    assert m_ctrl["recall@50"] == pytest.approx(1.0)
    assert m_ctrl["recall@100"] == pytest.approx(1.0)
    assert m_ctrl["new_hits@10"] == 0
    assert m_trt["recall@10"] == pytest.approx(1.0)
    assert m_trt["new_hits@10"] == n_rows
    assert m_trt["new_slate_coverage"] == pytest.approx(1.0)

    ci = paired_stage2_intervals(tmp_path / "ctrl.npz", tmp_path / "trt.npz", repeats=200, seed=42)
    assert ci["recall@10_delta"]["point_delta"] == pytest.approx(0.5)
    assert ci["new_recall@10_delta"]["point_delta"] == pytest.approx(1.0)


@pytest.mark.local_artifact
def test_saved_stage2_mlp_score_artifacts_and_parity():
    manifest_path = CACHE_ROOT / "manifest.json"
    summary_path = EVIDENCE_ROOT / "summary.json"
    report_path = EVIDENCE_ROOT / "REPORT.md"
    if not summary_path.is_file():
        pytest.skip("Stage 2 MLP-score experiment artifacts have not been generated yet.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")

    assert manifest["heldout_access"] is False
    assert summary["heldout_access"] is False
    assert summary["promoted"] is False
    assert summary["preserved_hashes_before"] == summary["preserved_hashes_after"]
    assert summary["slate_coverage_val"]["total_targets"] == 10276
    assert summary["slate_coverage_val"]["new_targets"] == 1857
    assert "Methodological Disclosures" in report_text

    ctrl = summary["arms"]["control"]
    trt = summary["arms"]["treatment"]
    assert ctrl["score_channel_enabled"] is False
    assert trt["score_channel_enabled"] is True
    assert ctrl["parameters"] == trt["parameters"]
    assert ctrl["trainable_parameters"] == trt["trainable_parameters"]
    assert ctrl["discovery"]["full_val_max_abs_logit_diff_vs_zero_score"] == 0.0
    assert trt["discovery"]["score_channel_diagnostics"]["attention_score_weight_l2"] > 0.0
    assert trt["discovery"]["score_channel_diagnostics"]["head_score_weight_l2"] > 0.0
    assert summary["adapter_parity"]["max_abs_mlp_score_diff_request_vs_cache"] < 1e-4

    # Strict load of persisted checkpoints verifies stored hashes and contract.
    ctrl_ckpt = Path(ctrl["discovery"]["checkpoint"])
    trt_ckpt = Path(trt["discovery"]["checkpoint"])
    assert ctrl_ckpt.is_file() and trt_ckpt.is_file()
    ctrl_model, _ = load_stage2_mlp_score_checkpoint(ctrl_ckpt, expected_score_channel_enabled=False)
    trt_model, _ = load_stage2_mlp_score_checkpoint(trt_ckpt, expected_score_channel_enabled=True)
    assert not ctrl_model.score_channel_enabled
    assert trt_model.score_channel_enabled
    assert (ARTIFACT_ROOT / "runs" / "stage2-treatment-mlp_score-seed42" / "best_ndcg.pt").is_file()
