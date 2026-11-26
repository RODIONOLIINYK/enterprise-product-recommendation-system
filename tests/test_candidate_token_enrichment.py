"""Synthetic and artifact-gated regression tests for candidate_token_enrichment."""

import copy
from pathlib import Path

import pytest
import torch

from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.candidate_token_enrichment.config import (
    ARM_A_NAME,
    ARM_B_NAME,
    ARM_C_NAME,
    CACHE_ROOT,
)
from recommender.experiments.candidate_token_enrichment.model import (
    CandidateTokenEnrichmentRanker,
    EnrichableUPSCANRRanker,
    load_enrichment_ranker_checkpoint,
    save_enrichment_ranker_checkpoint,
)
from recommender.experiments.candidate_token_enrichment.normalization import (
    CANDIDATE_STAT_DIM,
    HISTORY_STAT_DIM,
    PROFILE_STAT_DIM,
    SIGNED_CANDIDATE_FEATURE_INDICES,
    SIGNED_HISTORY_FEATURE_INDICES,
    normalize_candidate_stats,
    normalize_history_stats,
    normalize_profile_stats,
)
from recommender.experiments.dynamic_selector.feature_schema import rolling_schema_digest
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss, UPSCANRRanker


def _make_synthetic_arms(seed: int = 42):
    torch.manual_seed(seed)
    base = UPSCANRRanker(
        text_dim=32,
        memory_dim=128,
        num_cities=16,
        num_regions=8,
        num_industries=8,
        num_products=32,
        num_categories=8,
        num_subcategories=8,
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
    dyn = CandidateContextRanker(base, mode="dynamic_features")
    shared = CandidateTokenEnrichmentRanker.from_dynamic_context_ranker(
        dyn, candidate_attention_enabled=False, token_enrichment_enabled=False
    )
    arm_a = shared.clone_arm(candidate_attention_enabled=False, token_enrichment_enabled=False)
    arm_b = shared.clone_arm(candidate_attention_enabled=True, token_enrichment_enabled=False)
    arm_c = shared.clone_arm(
        candidate_attention_enabled=True,
        token_enrichment_enabled=True,
        enrichment_seed=seed + 1000,
    )
    return dyn, arm_a, arm_b, arm_c


def _make_synthetic_inputs(batch_size: int = 4, num_c: int = 6, seq_len: int = 8, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    pf = torch.rand((batch_size, num_c, CANDIDATE_STAT_DIM), generator=g)
    for idx in SIGNED_CANDIDATE_FEATURE_INDICES:
        pf[..., idx] = torch.rand((batch_size, num_c), generator=g) * 2.0 - 1.0

    cand_vecs = torch.randn((batch_size, num_c, 128), generator=g)
    cand_vecs = cand_vecs / cand_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cand_mask = torch.ones((batch_size, num_c), dtype=torch.bool)
    cand_mask[0, -1] = False

    attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attn_mask[0, :4] = 1
    attn_mask[1, :seq_len] = 1
    # Row 2 intentionally has empty history (cold user)
    attn_mask[3, :2] = 1

    seq = torch.rand((batch_size, seq_len, HISTORY_STAT_DIM), generator=g)
    for idx in SIGNED_HISTORY_FEATURE_INDICES:
        seq[..., idx] = torch.rand((batch_size, seq_len), generator=g) * 2.0 - 1.0
    seq[..., 7] = torch.randint(0, 2, (batch_size, seq_len), generator=g).float()
    seq[..., 8] = 1.0
    seq = seq * attn_mask.unsqueeze(-1).float()

    seq_dates = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    seq_dates[attn_mask.bool()] = 0.25
    pred_day = torch.full((batch_size,), 200.0, dtype=torch.float32)

    return {
        "text_embedding": torch.randn((batch_size, 32), generator=g),
        "city_id": torch.randint(1, 16, (batch_size,), generator=g),
        "region_id": torch.randint(1, 8, (batch_size,), generator=g),
        "industry_id": torch.randint(1, 8, (batch_size,), generator=g),
        "tabular_features": torch.rand((batch_size, PROFILE_STAT_DIM), generator=g),
        "sequence": seq,
        "attention_mask": attn_mask,
        "seq_product_ids": torch.randint(1, 32, (batch_size, seq_len), generator=g) * attn_mask,
        "seq_category_ids": torch.randint(1, 8, (batch_size, seq_len), generator=g) * attn_mask,
        "seq_subcategory_ids": torch.randint(1, 8, (batch_size, seq_len), generator=g) * attn_mask,
        "seq_business_line_ids": torch.randint(1, 8, (batch_size, seq_len), generator=g) * attn_mask,
        "seq_purchase_dates": seq_dates,
        "prediction_day": pred_day,
        "pairwise_features": pf,
        "candidate_vectors": cand_vecs,
        "candidate_mask": cand_mask,
    }


def test_scalar_normalization_unit_interval_and_validation():
    inputs = _make_synthetic_inputs()
    pf = inputs["pairwise_features"].clone()
    pf[0, -1, :] = float("nan")  # Masked candidate padding must be sanitized cleanly
    cand_norm = normalize_candidate_stats(pf, inputs["candidate_mask"])
    assert cand_norm.shape == (4, 6, 41)
    assert float(cand_norm.min()) >= 0.0 and float(cand_norm.max()) <= 1.0
    assert torch.all(cand_norm[0, -1] == 0.0)
    # Check signed mapping: -1 -> 0, +1 -> 1
    pf_extreme = torch.zeros(1, 1, 41)
    pf_extreme[0, 0, list(SIGNED_CANDIDATE_FEATURE_INDICES)] = -1.0
    c0 = normalize_candidate_stats(pf_extreme, torch.ones(1, 1, dtype=torch.bool))
    assert torch.allclose(c0[0, 0, list(SIGNED_CANDIDATE_FEATURE_INDICES)], torch.zeros(10))
    pf_extreme[0, 0, list(SIGNED_CANDIDATE_FEATURE_INDICES)] = 1.0
    c1 = normalize_candidate_stats(pf_extreme, torch.ones(1, 1, dtype=torch.bool))
    assert torch.allclose(c1[0, 0, list(SIGNED_CANDIDATE_FEATURE_INDICES)], torch.ones(10))

    # Reject out-of-range valid candidate feature
    bad_pf = inputs["pairwise_features"].clone()
    bad_pf[1, 0, 0] = -0.25  # unsigned col
    with pytest.raises(ValueError, match="out of expected range"):
        normalize_candidate_stats(bad_pf, inputs["candidate_mask"])

    bad_signed = inputs["pairwise_features"].clone()
    bad_signed[1, 0, SIGNED_CANDIDATE_FEATURE_INDICES[0]] = 1.5
    with pytest.raises(ValueError, match="out of expected range"):
        normalize_candidate_stats(bad_signed, inputs["candidate_mask"])

    # Profile normalization & range rejection
    prof = normalize_profile_stats(inputs["tabular_features"])
    assert float(prof.min()) >= 0.0 and float(prof.max()) <= 1.0
    bad_prof = inputs["tabular_features"].clone()
    bad_prof[0, 0] = 1.2
    with pytest.raises(ValueError, match="out of expected range"):
        normalize_profile_stats(bad_prof)

    # History normalization: date-age, attention-derived validity, history_repeat=0, month shift
    seq_nan_pad = inputs["sequence"].clone()
    seq_nan_pad[0, 5:, :] = float("nan")
    hist_norm = normalize_history_stats(
        seq_nan_pad,
        inputs["attention_mask"],
        inputs["seq_purchase_dates"],
        inputs["prediction_day"],
    )
    assert hist_norm.shape == (4, 8, 9)
    assert float(hist_norm.min()) >= 0.0 and float(hist_norm.max()) <= 1.0
    assert torch.all(hist_norm[..., 7] == 0.0)  # history_repeat strictly zeroed
    assert torch.equal(hist_norm[..., 8], (inputs["attention_mask"] != 0).float())
    assert torch.all(hist_norm[0, 5:] == 0.0)

    # Reject valid NaN in history
    bad_seq = inputs["sequence"].clone()
    bad_seq[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_history_stats(
            bad_seq,
            inputs["attention_mask"],
            inputs["seq_purchase_dates"],
            inputs["prediction_day"],
        )


def test_init_parity_shapes_and_mlp_score_rejection():
    dyn, arm_a, arm_b, arm_c = _make_synthetic_arms()
    for m in (dyn, arm_a, arm_b, arm_c):
        m.eval()

    # Exact MLP widths on Arm C
    assert arm_c.base.candidate_enricher.net[0].in_features == 169
    assert arm_c.base.candidate_enricher.net[0].out_features == 128
    assert arm_c.base.candidate_enricher.net[-1].out_features == 128
    assert arm_c.base.profile_enricher.net[0].in_features == 136
    assert arm_c.base.profile_enricher.net[0].out_features == 128
    assert arm_c.base.profile_enricher.net[-1].out_features == 128
    assert arm_c.base.history_enricher.net[0].in_features == 137
    assert arm_c.base.history_enricher.net[0].out_features == 128
    assert arm_c.base.history_enricher.net[-1].out_features == 128

    inputs = _make_synthetic_inputs()
    inputs_dyn = {k: v for k, v in inputs.items() if k != "candidate_mask"}
    with torch.no_grad():
        out_dyn = dyn(**inputs_dyn)
        out_a = arm_a(**inputs)
        out_b = arm_b(**inputs)
        out_c = arm_c(**inputs)
        out_c_no_attn = arm_c(**inputs, disable_candidate_attention=True)
        out_c_no_enrich = arm_c(**inputs, disable_token_enrichment=True)

    valid = inputs["candidate_mask"]
    assert torch.allclose(out_a.logits[valid], out_dyn.logits[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_b.logits[valid], out_dyn.logits[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_c.logits[valid], out_dyn.logits[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_c_no_attn.logits[valid], out_dyn.logits[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_c_no_enrich.logits[valid], out_dyn.logits[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_c.memory, out_b.memory, atol=0.0, rtol=0.0)
    assert torch.allclose(out_c.z_candidate[valid], out_b.z_candidate[valid], atol=0.0, rtol=0.0)
    assert torch.allclose(out_c.attn_weights[valid], out_b.attn_weights[valid], atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="forbids Stage 1 mlp_score"):
        arm_c(**inputs, mlp_score=torch.zeros(4, 6))


def test_nonzero_gradients_branch_attribution_and_trained_invariances():
    _, _, _, arm_c = _make_synthetic_arms()
    inputs = _make_synthetic_inputs()
    labels = torch.zeros((4, 6), dtype=torch.float32)
    labels[:, 0] = 1.0
    labels[:, 2] = 1.0

    arm_c.train()
    opt = torch.optim.AdamW([p for p in arm_c.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = RankerLoss()
    for step in range(4):
        opt.zero_grad()
        out = arm_c(**inputs)
        loss = loss_fn(out.logits, labels, candidate_mask=inputs["candidate_mask"])["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            # Verify nonzero gradients flow into ALL 3 enrichment MLPs + candidate attention
            assert arm_c.base.candidate_enricher.net[-1].weight.grad.abs().sum() > 0
            assert arm_c.base.profile_enricher.net[-1].weight.grad.abs().sum() > 0
            assert arm_c.base.history_enricher.net[-1].weight.grad.abs().sum() > 0
            assert arm_c.residual_head[-1].weight.grad.abs().sum() > 0
        opt.step()

    arm_c.eval()
    with torch.no_grad():
        ref = arm_c(**inputs)
        # 1. Each numeric family affects its target representation & logits
        no_cand = arm_c(**inputs, disable_candidate_enrichment=True)
        assert (ref.z_candidate - no_cand.z_candidate).abs().max() > 1e-4
        assert (ref.memory - no_cand.memory).abs().max() == 0.0
        assert (ref.logits[inputs["candidate_mask"]] - no_cand.logits[inputs["candidate_mask"]]).abs().max() > 1e-4

        no_prof = arm_c(**inputs, disable_profile_enrichment=True)
        assert (ref.memory - no_prof.memory).abs().max() > 1e-4
        assert (ref.z_candidate - no_prof.z_candidate).abs().max() == 0.0
        assert (ref.logits[inputs["candidate_mask"]] - no_prof.logits[inputs["candidate_mask"]]).abs().max() > 1e-4

        no_hist = arm_c(**inputs, disable_history_enrichment=True)
        assert (ref.memory - no_hist.memory).abs().max() > 1e-4
        assert (ref.z_candidate - no_hist.z_candidate).abs().max() == 0.0
        assert (ref.logits[inputs["candidate_mask"]] - no_hist.logits[inputs["candidate_mask"]]).abs().max() > 1e-4

        # 2. Excluded history_repeat invariance (no resurrection)
        rep_flipped = copy.deepcopy(inputs)
        rep_flipped["sequence"][..., 7] = 1.0 - rep_flipped["sequence"][..., 7]
        out_rep = arm_c(**rep_flipped)
        assert torch.allclose(out_rep.memory, ref.memory, atol=0.0, rtol=0.0)
        assert torch.allclose(out_rep.logits[inputs["candidate_mask"]], ref.logits[inputs["candidate_mask"]], atol=0.0, rtol=0.0)

        # 3. Date-age sensitivity when prediction_day shifts
        later = copy.deepcopy(inputs)
        later["prediction_day"] = later["prediction_day"] + 180.0
        out_later = arm_c(**later)
        assert (out_later.memory - ref.memory).abs().max() > 1e-4
        assert (out_later.logits[inputs["candidate_mask"]] - ref.logits[inputs["candidate_mask"]]).abs().max() > 1e-4

        # 4. Candidate permutation equivariance with nonzero trained weights
        rev = {
            k: (torch.flip(v, dims=[1]) if k in ("pairwise_features", "candidate_vectors", "candidate_mask") else v)
            for k, v in inputs.items()
        }
        out_rev = arm_c(**rev)
        rev_back = torch.flip(out_rev.logits, dims=[1])
        assert torch.allclose(rev_back[inputs["candidate_mask"]], ref.logits[inputs["candidate_mask"]], atol=1e-5)

        # 5. NaN-padded candidates + NaN/garbage-padded history isolation
        corrupt = copy.deepcopy(inputs)
        corrupt["pairwise_features"] = torch.cat([inputs["pairwise_features"], torch.full((4, 2, 41), float("nan"))], dim=1)
        corrupt["candidate_vectors"] = torch.cat([inputs["candidate_vectors"], torch.full((4, 2, 128), float("nan"))], dim=1)
        corrupt["candidate_mask"] = torch.cat([inputs["candidate_mask"], torch.zeros((4, 2), dtype=torch.bool)], dim=1)
        hist_pad = inputs["attention_mask"] == 0
        corrupt["sequence"][hist_pad] = float("nan")
        corrupt["seq_purchase_dates"][hist_pad] = float("nan")
        corrupt["seq_product_ids"][hist_pad] = 999999
        out_corrupt = arm_c(**corrupt)
        assert torch.allclose(
            out_corrupt.logits[:, :-2][inputs["candidate_mask"]],
            ref.logits[inputs["candidate_mask"]],
            atol=1e-6,
        )
        assert torch.all(out_corrupt.logits[:, -2:] == -torch.inf)

        # 6. All-masked candidate row safety (all -inf, no NaN grads)
        all_masked = copy.deepcopy(inputs)
        all_masked["candidate_mask"][2] = False
        out_all_masked = arm_c(**all_masked)
        assert torch.all(out_all_masked.logits[2] == -torch.inf)
        assert torch.isfinite(out_all_masked.logits[all_masked["candidate_mask"]]).all()


def test_checkpoint_strict_reload_and_contract_validation(tmp_path):
    _, arm_a, arm_b, arm_c = _make_synthetic_arms()
    inputs = _make_synthetic_inputs()
    # Perturb Arm C weights so reload isn't trivial zero-init
    with torch.no_grad():
        arm_c.base.candidate_enricher.net[-1].weight.fill_(0.02)
        arm_c.base.profile_enricher.net[-1].weight.fill_(0.02)
        arm_c.base.history_enricher.net[-1].weight.fill_(0.02)
    arm_a.eval()
    arm_b.eval()
    arm_c.eval()

    schema_dig = rolling_schema_digest()
    arm_d = arm_c.clone_arm(candidate_attention_enabled=False, token_enrichment_enabled=True)
    arm_d.eval()
    for name, m in ((ARM_A_NAME, arm_a), (ARM_B_NAME, arm_b), (ARM_C_NAME, arm_c), ("arm_d_pointwise_enriched_128d", arm_d)):
        ckpt_path = tmp_path / f"{name}.pt"
        prov = {
            "arm_name": name,
            "statistics_policy": STATISTICS_POLICY_ROLLING,
            "teacher_score_included": False,
            "heldout_access": False,
        }
        save_enrichment_ranker_checkpoint(
            ckpt_path,
            m,
            epoch=3,
            metrics={"recall@10": 0.75, "ndcg@10": 0.59, "new_hits@10": 200},
            selection_criterion="val_ndcg_at_10",
            provenance=prov,
            feature_contract_digest=schema_dig,
        )
        reloaded, _ = load_enrichment_ranker_checkpoint(
            ckpt_path,
            expected_provenance=prov,
            expected_feature_contract_digest=schema_dig,
            expected_candidate_attention_enabled=m.candidate_attention_enabled,
            expected_token_enrichment_enabled=m.token_enrichment_enabled,
        )
        reloaded.eval()
        with torch.no_grad():
            o1 = m(**inputs).logits
            o2 = reloaded(**inputs).logits
            assert torch.allclose(o1[inputs["candidate_mask"]], o2[inputs["candidate_mask"]], atol=0.0, rtol=0.0)

    # Tampering check: dropping enrichment key from Arm C checkpoint must raise ValueError
    c_ckpt_path = tmp_path / f"{ARM_C_NAME}.pt"
    bad_payload = torch.load(c_ckpt_path, map_location="cpu", weights_only=False)
    bad_payload["model_state_dict"].pop("base.profile_enricher.net.0.weight")
    bad_path = tmp_path / "corrupt_c.pt"
    torch.save(bad_payload, bad_path)
    with pytest.raises((ValueError, RuntimeError)):
        load_enrichment_ranker_checkpoint(bad_path)

    # Contract mismatch rejection
    with pytest.raises(ValueError, match="token_enrichment_enabled mismatch"):
        load_enrichment_ranker_checkpoint(c_ckpt_path, expected_token_enrichment_enabled=False)


@pytest.mark.local_artifact
def test_local_v3_cache_lineage_and_real_scalar_validation():
    before = snapshot_preserved_hashes()
    assert len(before) == 17

    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=CACHE_ROOT,
        build=False,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    prepare_or_load_rolling_selector_scores(campaign_data, info, meta, cache_root=CACHE_ROOT)
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(
        campaign_data, info, meta, cache_root=CACHE_ROOT
    )
    assert slate_manifest["teacher_score_included"] is False
    assert slate_manifest["heldout_access"] is False
    assert len(slates["train"]["customer_ids"]) == 26253
    assert len(slates["val"]["customer_ids"]) == 6191

    # Validate [0, 1] normalization on first 64 real validation rows
    val = slates["val"]
    n = 64
    c_norm = normalize_candidate_stats(val["pairwise_features"][:n], val["candidate_mask"][:n])
    p_norm = normalize_profile_stats(val["tabular_features"][:n])
    h_norm = normalize_history_stats(
        val["sequence"][:n],
        val["attention_mask"][:n],
        val["seq_purchase_dates"][:n],
        val["prediction_days"][:n],
    )
    assert float(c_norm.min()) >= 0.0 and float(c_norm.max()) <= 1.0
    assert float(p_norm.min()) >= 0.0 and float(p_norm.max()) <= 1.0
    assert float(h_norm.min()) >= 0.0 and float(h_norm.max()) <= 1.0
    assert torch.all(h_norm[..., 7] == 0.0)


def test_enrichment_only_trains_with_padding_and_remains_candidate_independent():
    from recommender.experiments.candidate_token_enrichment.pointwise_enrichment import ARM_D_NAME

    _, arm_a, _, arm_c = _make_synthetic_arms()
    arm_d = arm_c.clone_arm(candidate_attention_enabled=False, token_enrichment_enabled=True)
    assert ARM_D_NAME == "arm_d_pointwise_enriched_128d"
    arm_a.eval()
    arm_d.eval()
    inputs = _make_synthetic_inputs()
    with torch.no_grad():
        torch.testing.assert_close(arm_a(**inputs).logits, arm_d(**inputs).logits, rtol=0, atol=0)
    # All-masked row and an ordinary padded row must train safely.
    inputs["candidate_mask"][2] = False
    labels = torch.zeros(4, 6)
    labels[[0, 1, 3], 0] = 1.
    opt = torch.optim.AdamW([p for p in arm_d.parameters() if p.requires_grad], lr=1e-3)
    arm_d.train()
    for _ in range(2):
        opt.zero_grad()
        loss = RankerLoss()(arm_d(**inputs).logits, labels, candidate_mask=inputs["candidate_mask"])["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for mod in (arm_d.base.candidate_enricher, arm_d.base.profile_enricher, arm_d.base.history_enricher):
            assert torch.isfinite(mod.net[-1].weight.grad).all()
            assert mod.net[-1].weight.grad.abs().sum() > 0
        for mod in (arm_d.context_project, arm_d.candidate_attention, arm_d.residual_head):
            assert all(p.grad is None and not p.requires_grad for p in mod.parameters())
        opt.step()
    arm_d.eval()
    with torch.no_grad():
        original = arm_d(**inputs).logits[:, :1]
        subset = {key: (v[:, :1] if key in ("pairwise_features", "candidate_vectors", "candidate_mask") else v)
                  for key, v in inputs.items()}
        torch.testing.assert_close(arm_d(**subset).logits, original, rtol=0, atol=1e-6)


def test_enrichment_followup_rejects_changed_source_and_slate_identity():
    from recommender.experiments.candidate_token_enrichment.pointwise_enrichment import validate_matched_inputs

    settings = {"data_identity": "data", "feature_contract_digest": "schema", "observed_source_digest": "source"}
    prior = {"observed_source_digest": "source", "feature_bundle_identity": "features",
             "rolling_selector_scores_identity": "scores", "slate_cache_identities": {"train": "t", "val": "v"},
             "statistics_policy": STATISTICS_POLICY_ROLLING, "teacher_score_included": False,
             "seed": 42, "epochs": 10, "batch_size": 64, "arms": {"a": {"settings": settings}}}
    meta = {"data_identity": "data", "feature_schema_digest": "schema", "observed_source_digest": "source"}
    args = (meta, {"identity": "features"}, {"identity": "scores"}, {"split_cache_identity": {"train": "t", "val": "v"}})
    validate_matched_inputs(prior, *args)
    for key, value in (("observed_source_digest", "new-source"), ("slate_cache_identities", {"train": "t", "val": "changed"})):
        bad = copy.deepcopy(prior)
        bad[key] = value
        with pytest.raises(ValueError, match=key):
            validate_matched_inputs(bad, *args)
