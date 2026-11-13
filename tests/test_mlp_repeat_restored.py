"""Synthetic regression tests for Stage 1 MLP repeat_no_month (uv_45, 301D) experiment."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recommender.experiments.dynamic_selector.feature_schema import NAMES as SOURCE_CONTEXT_NAMES
from recommender.experiments.mlp_feature_ablation.config import (
    OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
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
from recommender.experiments.mlp_feature_ablation.train import measure_repeat_flag_influence
from recommender.experiments.static_enrichment.model import CompactStage1Selector


def _make_301_spec():
    return build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )


def test_repeat_no_month_spec_retains_is_repeat_sku_and_omits_months():
    spec = _make_301_spec()
    assert spec.name == "repeat_no_month"
    assert spec.recipe == "uv_45"
    assert spec.source_num_features == 41
    assert spec.omitted_names == ("month_sin", "month_cos")
    assert spec.omitted_indices == (31, 32)
    assert spec.num_retained_features == 39
    assert spec.num_customer_stats == 6
    assert spec.scalar_feature_count == 45
    assert spec.input_dim == 301
    assert spec.ordered_input_blocks == (
        ("u", 128),
        ("v", 128),
        ("features_39", 39),
        ("customer_stats_6", 6),
    )

    assert spec.retained_names[0] == "is_repeat_sku"
    assert spec.retained_indices[0] == 0
    assert "is_repeat_sku" in spec.retained_names
    assert "month_sin" not in spec.retained_names
    assert "month_cos" not in spec.retained_names
    assert spec.retained_indices == tuple(range(0, 31)) + tuple(range(33, 41))
    assert [SOURCE_CONTEXT_NAMES[i] for i in spec.retained_indices] == list(spec.retained_names)

    # Reordered schema resolves month_sin / month_cos by name, not magic indices 31, 32.
    reordered = tuple(reversed(SOURCE_CONTEXT_NAMES))
    spec_rev = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        source_names=reordered,
        expected_omitted_indices=None,
    )
    assert spec_rev.omitted_indices == (
        reordered.index("month_sin"),
        reordered.index("month_cos"),
    )
    assert "is_repeat_sku" in spec_rev.retained_names


def test_301d_architecture_gradients_and_retained_context_forward():
    torch.manual_seed(42)
    spec = _make_301_spec()
    model = FeatureAblatedStage1Selector(spec)

    assert model.input_dim == 301
    assert model.recipe == "uv_45"
    assert model.head[0].in_features == 301
    assert model.head[0].out_features == 128
    assert isinstance(model.head[1], torch.nn.LayerNorm)
    assert isinstance(model.head[2], torch.nn.GELU)
    assert isinstance(model.head[3], torch.nn.Dropout) and model.head[3].p == pytest.approx(0.1)
    assert model.head[4].in_features == 128 and model.head[4].out_features == 64
    assert model.head[8].in_features == 64 and model.head[8].out_features == 1
    assert sum(p.numel() for p in model.parameters()) == 47361

    B, P = 3, 7
    u = torch.randn(B, 128)
    v = torch.randn(P, 128)
    stats = torch.rand(B, 6)
    f41 = torch.rand(B, P, 41, requires_grad=True)
    target = torch.zeros(B, P)
    target[:, 0] = 1.0

    logits = model(u=u, v=v, features_41=f41, customer_stats_6=stats)
    assert logits.shape == (B, P)
    F.binary_cross_entropy_with_logits(logits, target).backward()

    # Gradients MUST be strictly zero on omitted month cols (31, 32) and non-zero on is_repeat_sku (0).
    for idx in (31, 32):
        assert torch.all(f41.grad[..., idx] == 0.0)
    assert float(f41.grad[..., 0].abs().sum().item()) > 0.0
    assert float(f41.grad[..., list(spec.retained_indices)].abs().sum().item()) > 0.0

    model.eval()
    with torch.no_grad():
        full = model(u=u, v=v, features_41=f41.detach(), customer_stats_6=stats)
        chunked = model(
            u=u, v=v, features_41=f41.detach(), customer_stats_6=stats, product_chunk_size=3
        )
        torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)

        f39 = select_retained_context_features(f41.detach(), spec)
        assert f39.shape == (B, P, 39)
        direct = model(u=u, v=v, retained_context=f39, customer_stats_6=stats)
        torch.testing.assert_close(full, direct, atol=0.0, rtol=0.0)

        with pytest.raises(ValueError, match="at most one of retained_context or context_38"):
            model(u=u, v=v, retained_context=f39, context_38=f39, customer_stats_6=stats)


def test_month_value_invariance_and_controlled_repeat_flag_dependence():
    torch.manual_seed(11)
    spec = _make_301_spec()
    model = FeatureAblatedStage1Selector(spec)
    model.eval()

    B, P = 4, 6
    u = torch.randn(B, 128)
    v = torch.randn(P, 128)
    stats = torch.rand(B, 6)
    f41 = torch.rand(B, P, 41)

    with torch.no_grad():
        base_logits = model(u=u, v=v, features_41=f41, customer_stats_6=stats)
        f41_months_perturbed = f41.clone()
        f41_months_perturbed[..., 31] = 1e5
        f41_months_perturbed[..., 32] = -1e5
        perturbed_logits = model(u=u, v=v, features_41=f41_months_perturbed, customer_stats_6=stats)
        torch.testing.assert_close(base_logits, perturbed_logits, atol=0.0, rtol=0.0)

    # Controlled training where ONLY is_repeat_sku (index 0) separates positive vs negative items.
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    f_ctrl = torch.zeros(2, 2, 41)
    f_ctrl[:, 0, 0] = 1.0  # item 0 has is_repeat_sku=1, item 1 has is_repeat_sku=0
    y_ctrl = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    for _ in range(15):
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
        f_flipped[:, 0, 0] = 0.0
        f_flipped[:, 1, 0] = 1.0
        out_flipped = model(
            u=torch.zeros(2, 128),
            v=torch.zeros(2, 128),
            features_41=f_flipped,
            customer_stats_6=torch.zeros(2, 6),
        )
        assert float((out_orig[:, 0] - out_orig[:, 1]).min()) > 0.5
        assert float((out_flipped[:, 1] - out_flipped[:, 0]).min()) > 0.5

    split_data = {
        "u": torch.zeros(2, 128),
        "v": torch.zeros(2, 128),
        "customer_stats_6": torch.zeros(2, 6),
        "allowed": torch.ones(2, 2, dtype=torch.bool),
        "features": f_ctrl.numpy().astype(np.float16),
    }
    inf = measure_repeat_flag_influence(model, split_data, torch.device("cpu"))
    assert inf["feature_name"] == "is_repeat_sku"
    assert inf["source_context_index"] == 0
    assert inf["retained_context_index"] == 0
    assert inf["max_abs_logit_diff_1_vs_0"] > 0.5
    assert inf["mean_signed_logit_diff_1_minus_0"] > 0.5


def test_strict_301d_contract_reload_and_300_303_preservation(tmp_path):
    torch.manual_seed(42)
    spec_301 = _make_301_spec()
    model_301 = FeatureAblatedStage1Selector(spec_301)
    contract_301 = model_301.input_contract()
    ckpt_301 = tmp_path / "repeat_301.pt"
    torch.save(
        {
            "epoch": 2,
            "input_contract": contract_301,
            "model_state_dict": model_301.state_dict(),
        },
        ckpt_301,
    )

    loaded_301, ckpt = load_ablated_selector_checkpoint(ckpt_301, expected_spec=spec_301)
    assert ckpt["epoch"] == 2
    assert loaded_301.input_dim == 301
    assert loaded_301.recipe == "uv_45"
    assert loaded_301.spec.retained_names[0] == "is_repeat_sku"

    loaded_auto, _ = load_ablated_selector_checkpoint(ckpt_301)
    assert loaded_auto.spec.name == "repeat_no_month"
    assert loaded_auto.spec.retained_names == spec_301.retained_names

    # Reject loading 300D checkpoint against 301D expected_spec and vice versa.
    spec_300 = build_feature_ablation_spec(OMITTED_CONTEXT_FEATURES)
    model_300 = FeatureAblatedStage1Selector(spec_300)
    assert model_300.input_dim == 300
    assert sum(p.numel() for p in model_300.parameters()) == 47233
    ckpt_300 = tmp_path / "ablated_300.pt"
    torch.save(
        {
            "epoch": 5,
            "input_contract": model_300.input_contract(),
            "model_state_dict": model_300.state_dict(),
        },
        ckpt_300,
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_ablated_selector_checkpoint(ckpt_300, expected_spec=spec_301)
    with pytest.raises(ValueError, match="mismatch"):
        load_ablated_selector_checkpoint(ckpt_301, expected_spec=spec_300)

    # Reject forged 303D weights under 301D contract.
    legacy_303 = CompactStage1Selector(recipe="uv_47")
    assert legacy_303.input_dim == 303
    assert sum(p.numel() for p in legacy_303.parameters()) == 47617
    forged_303 = tmp_path / "forged_303.pt"
    torch.save(
        {"input_contract": contract_301, "model_state_dict": legacy_303.state_dict()},
        forged_303,
    )
    with pytest.raises(ValueError, match="weight width mismatch"):
        load_ablated_selector_checkpoint(forged_303, expected_spec=spec_301)

    # Reject variant_name tampering even if contract_digest is unchecked first.
    bad_variant = dict(contract_301)
    bad_variant["variant_name"] = "wrong_name"
    with pytest.raises(ValueError, match="digest corrupted|mismatch"):
        validate_input_contract(bad_variant, contract_301)


def test_synthetic_repeat_restored_runner_end_to_end(tmp_path, monkeypatch):
    import recommender.experiments.mlp_feature_ablation.run_repeat_restored as rr

    torch.manual_seed(42)
    N_tr, N_va, P = 12, 8, 20
    rng = np.random.default_rng(7)

    def _make_split(n):
        pos_pids = [[f"P{i % 5}", f"P{(i + 10) % P}"] for i in range(n)]
        new_pids = [[f"P{(i + 10) % P}"] for i in range(n)]
        target = torch.zeros(n, P, dtype=torch.bool)
        new_target = torch.zeros(n, P, dtype=torch.bool)
        unseen = torch.ones(n, P, dtype=torch.bool)
        for i in range(n):
            target[i, i % 5] = True
            target[i, (i + 10) % P] = True
            new_target[i, (i + 10) % P] = True
            unseen[i, :8] = False
        return {
            "u": F.normalize(torch.randn(n, 128), dim=-1),
            "v": F.normalize(torch.randn(P, 128), dim=-1),
            "features": rng.uniform(0.0, 1.0, size=(n, P, 41)).astype(np.float16),
            "customer_stats_6": torch.rand(n, 6),
            "allowed": torch.ones(n, P, dtype=torch.bool),
            "unseen": unseen,
            "target": target,
            "new_target": new_target,
            "source": {
                "customer_ids": [f"C{i % 4}" for i in range(n)],
                "positive_pids": pos_pids,
                "new_positive_pids": new_pids,
                "attention_mask": torch.ones(n, 32, dtype=torch.long),
            },
        }

    tr = _make_split(N_tr)
    va = _make_split(N_va)
    va["v"] = tr["v"]
    campaign_data = {"train": tr, "val": va}
    info = {"identity": "synthetic-id", "product_ids": [f"P{i}" for i in range(P)]}
    meta = {
        "reference": {
            "reference_pointer": {"active_reference_id": "no-brand-v1"},
            "hashes": {
                "best_retriever_pt": "r" * 64,
                "best_selector_pt": "s" * 64,
            },
        },
        "dynamic_identity": "dyn-id",
        "data_identity": "synthetic-id",
        "enrichment_identity": "enr-id",
        "cache_lineage": {"train": {}, "val": {}},
    }

    fake_scores_303 = torch.randn(N_va, P)
    fake_scores_300 = torch.randn(N_va, P)

    from recommender.experiments.mlp_feature_ablation.evaluate import evaluate_full_pool_metrics

    m303 = evaluate_full_pool_metrics(fake_scores_303, va)
    m300 = evaluate_full_pool_metrics(fake_scores_300, va)

    monkeypatch.setattr(rr, "snapshot_preserved_hashes", lambda **kw: {"x": "ok"})
    monkeypatch.setattr(rr, "load_verified_no_brand_campaign_data", lambda: (campaign_data, info, meta))
    monkeypatch.setattr(
        rr,
        "load_verified_baseline_val_scores",
        lambda _id: (fake_scores_303, {"val_scores_path": "v303", "val_scores_sha256": "h303", "baseline_eval_json": None, "history": [{"epoch": 1, "metrics": m303}]}),
    )
    monkeypatch.setattr(
        rr,
        "load_verified_prior_300_val_scores",
        lambda _id: (
            fake_scores_300,
            {
                "selector_path": "s300",
                "selector_sha256": "h300",
                "val_scores_path": "vp300",
                "val_scores_sha256": "hvp300",
                "result_json": {"selected_epoch": 5, "parameters": 47233, "metrics": m300},
                "history": [{"epoch": 1, "metrics": m300}],
            },
        ),
    )

    art_dir = tmp_path / "artifacts"
    ev_dir = tmp_path / "evidence"
    summary = rr.run_repeat_restored_experiment(
        artifact_root=art_dir,
        evidence_root=ev_dir,
        seed=42,
        epochs=2,
        device="cpu",
    )

    assert summary["variant"] == "repeat_no_month"
    assert summary["feature_spec"]["recipe"] == "uv_45"
    assert summary["feature_spec"]["input_dim"] == 301
    assert summary["variant_301"]["first_layer_in_features"] == 301
    assert summary["variant_301"]["parameters"] == 47361
    assert summary["variant_301"]["zero_ablations"] is None
    assert summary["variant_301"]["repeat_flag_influence"]["max_abs_logit_diff_1_vs_0"] > 0.0
    assert summary["candidate_artifact"]["promoted"] is False
    assert (art_dir / "candidate" / "best_selector.pt").is_file()
    assert (art_dir / "candidate" / "manifest.json").is_file()
    assert (ev_dir / "summary.json").is_file()
    assert (ev_dir / "REPORT.md").is_file()
    report_text = (ev_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "repeat_no_month" in report_text
    assert "uv_45" in report_text
    assert "Descriptive Epoch-1 Comparison" in report_text
    cand_manifest = json.loads((art_dir / "candidate" / "manifest.json").read_text(encoding="utf-8"))
    assert cand_manifest["is_active_reference"] is False
    assert cand_manifest["heldout_access"] is False
