"""Comprehensive synthetic unit tests for Wave 2B (`extra_cand_5`, `cand_42`, Option A 70/30, Pure-Q, Full-42 bias, CatBoost)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
)
from recommender.experiments.v2_full.config import ACTIVE_V2_FULL_BASE_MODEL, TEST_START_V2
from recommender.experiments.v2_full.extra_features import (
    CAND_38_KEEP_INDICES,
    CAND_42_FEATURE_NAMES,
    CAND_42_GLOBAL_PRODUCT_FEATURE_NAMES,
    CAND_42_NON_REPEAT_INDICES,
    CAND_42_NON_REPEAT_NAMES,
    CUST_7_FEATURE_NAMES,
    CUST_7_KEEP_INDICES,
    EXTRA_CAND_5_FEATURE_NAMES,
    SEQ_5_ACTIVE_INDICES,
    UNSEEN_37_GLOBAL_LOCAL_INDICES,
    assemble_cand_42,
    compute_extra_cand_5_arrays,
    prune_cust_7,
)
from recommender.experiments.v2_full.run_v2_full_wave2b import (
    build_wave2b_arm_specs,
    iter_paired_training_batches,
)
from recommender.experiments.v2_full.wave2b_models import (
    OptionAUnseenHead,
    Wave2BE2ERanker,
    _eval_affine_ab_on_subset,
    _prepare_exact_top_candidates,
    evaluate_pure_unseen_predictions,
    fit_oof_affine_unseen_blend,
    strip_wave2b_training_aux,
)
from recommender.sequences import PURCHASE_DATE_ORIGIN


def test_pruned_schemas_and_assemble_cand_42():
    assert len(CAND_38_KEEP_INDICES) == 37
    assert "base_name_growth_30d" not in CAND_42_FEATURE_NAMES
    assert "mgr_base_name_growth_30d_vs_90d" not in CAND_42_FEATURE_NAMES
    assert len(CAND_42_FEATURE_NAMES) == 42
    assert CAND_42_FEATURE_NAMES[-5:] == EXTRA_CAND_5_FEATURE_NAMES
    assert CAND_42_NON_REPEAT_INDICES == tuple(range(5, 42))
    assert len(CAND_42_NON_REPEAT_NAMES) == 37
    assert tuple(CAND_42_NON_REPEAT_NAMES[i] for i in UNSEEN_37_GLOBAL_LOCAL_INDICES) == CAND_42_GLOBAL_PRODUCT_FEATURE_NAMES

    assert CUST_7_KEEP_INDICES == (0, 2, 3, 4, 5, 6, 8)
    for dropped in (1, 7, 9, 10):
        assert CUSTOMER_DYNAMIC_11_FEATURE_NAMES[dropped] not in CUST_7_FEATURE_NAMES
    assert SEQ_5_ACTIVE_INDICES == (0, 1, 5, 6, 8)  # keeps live_age (1), drops gap_days (2), month (3,4), repeat (7)

    c38 = np.arange(2 * 4 * 38, dtype=np.float32).reshape(2, 4, 38)
    e5 = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) + 1000.0
    c42 = assemble_cand_42(c38, e5)
    assert c42.shape == (2, 4, 42)
    # Col 21 matches original 21, col 22 matches original 23 (since original 22 was dropped), cols 37..41 match e5
    assert np.array_equal(c42[..., 21], c38[..., 21])
    assert np.array_equal(c42[..., 22], c38[..., 23])
    assert np.array_equal(c42[..., 37:], e5)

    cu11 = torch.arange(33, dtype=torch.float32).reshape(3, 11)
    cu7 = prune_cust_7(cu11)
    assert cu7.shape == (3, 7)
    assert torch.equal(cu7[0], torch.tensor([0., 2., 3., 4., 5., 6., 8.]))

    # Paired batch iterator visits every row exactly once with exact cand_42 alignment
    gen = torch.Generator().manual_seed(7)
    c38_b = np.arange(35 * 3 * 38, dtype=np.float16).reshape(35, 3, 38)
    e5_b = np.arange(35 * 3 * 5, dtype=np.float16).reshape(35, 3, 5)
    seen = []
    for rows, c42_batch in iter_paired_training_batches(
        c38_b, e5_b, 35, 8, gen, in_memory=False, block_rows=10, blocks_per_buffer=2
    ):
        seen.extend(rows.tolist())
        expected = assemble_cand_42(c38_b[rows.numpy()], e5_b[rows.numpy()])
        assert np.allclose(c42_batch, expected)
    assert sorted(seen) == list(range(35))


def test_extra_cand_5_strict_causality_and_exact_cf_parity():
    universe = {
        "base_names": ["BN0", "BN1", "BN2", "BN3"],
        "num_base_names": 4,
        "trained_base_names": 4,
        "all_pid_to_bn": {"P0": "BN0", "P1": "BN1", "P2": "BN2", "P3": "BN3"},
        "bn_to_idx": {"BN0": 0, "BN1": 1, "BN2": 2, "BN3": 3},
    }
    q_ts = pd.Timestamp("2026-01-15")
    q_day = int((q_ts - PURCHASE_DATE_ORIGIN).days)

    events = pd.DataFrame(
        [
            {"customer_id": "C1", "purchase_date": q_ts - pd.Timedelta(days=20), "product_id": "P0", "manager_id": "M1"},
            {"customer_id": "C1", "purchase_date": q_ts - pd.Timedelta(days=20), "product_id": "P1", "manager_id": "M1"},
            {"customer_id": "C1", "purchase_date": q_ts, "product_id": "P3", "manager_id": "M1"},  # same-day: NO LEAK!
            {"customer_id": "C2", "purchase_date": q_ts - pd.Timedelta(days=50), "product_id": "P0", "manager_id": "M1"},
            {"customer_id": "C2", "purchase_date": q_ts - pd.Timedelta(days=50), "product_id": "P2", "manager_id": "M1"},
            {"customer_id": "C3", "purchase_date": q_ts - pd.Timedelta(days=120), "product_id": "P0", "manager_id": "M2"},
            {"customer_id": "C3", "purchase_date": q_ts - pd.Timedelta(days=120), "product_id": "P2", "manager_id": "M2"},
        ]
    )
    reg_map = {"C1": 1, "C2": 1, "C3": 2, "C_FO": 1}
    ind_map = {"C1": 1, "C2": 1, "C3": 2, "C_FO": 2}
    split_meta = {
        "customer_ids": ["C1", "C_FO"],
        "prediction_days": torch.tensor([float(q_day), float(q_day)]),
        "region_id": torch.tensor([1, 1], dtype=torch.long),
        "industry_id": torch.tensor([1, 2], dtype=torch.long),
    }

    out = compute_extra_cand_5_arrays(
        commercial_events=events,
        universe=universe,
        customer_region_map=reg_map,
        customer_industry_map=ind_map,
        splits={"val": split_meta},
        test_start=TEST_START_V2,
    )["val"].astype(np.float32)

    assert out.shape == (2, 4, 5)
    # Row 0 = C1:
    # 1. Same-day BN3 MUST have 0.0 on all 5 features (no same-day leak!)
    assert np.allclose(out[0, 3, :], 0.0, atol=1e-4)
    # 2. M1 30d window [q-30, q): C1 bought BN0, BN1 -> share_30d = [0.5, 0.5, 0.0, 0.0]
    assert np.allclose(out[0, :, 0], [0.5, 0.5, 0.0, 0.0], atol=1e-3)
    # 3. (M1, ind=1) 90d: [0.5, 0.25, 0.25, 0.0]
    assert np.allclose(out[0, :, 1], [0.5, 0.25, 0.25, 0.0], atol=1e-3)
    # 4. Region 1 90d: [0.5, 0.25, 0.25, 0.0]
    assert np.allclose(out[0, :, 2], [0.5, 0.25, 0.25, 0.0], atol=1e-3)
    # 5. Exact brute-force item-item cosine CF over < q_day:
    raw_bn2 = (2.0 / math.sqrt(6.0)) / math.sqrt(2.0)
    expected_cf_bn2 = raw_bn2 / (raw_bn2 + 1.0)
    assert abs(float(out[0, 2, 3]) - expected_cf_bn2) < 2e-3
    assert abs(float(out[0, 2, 4]) - expected_cf_bn2) < 2e-3  # same age -> exact match

    # Row 1 = C_FO (first-observed): region90 populated, mgr & CF strictly zero!
    assert np.allclose(out[1, :, 2], [0.5, 0.25, 0.25, 0.0], atol=1e-3)
    assert np.allclose(out[1, :, [0, 1, 3, 4]], 0.0, atol=1e-4)

    # Leak guards: events or queries >= TEST_START_V2 must raise ValueError
    leaked_ev = pd.concat(
        [events, pd.DataFrame([{"customer_id": "C1", "purchase_date": TEST_START_V2, "product_id": "P0", "manager_id": "M1"}])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Development events only"):
        compute_extra_cand_5_arrays(
            commercial_events=leaked_ev,
            universe=universe,
            customer_region_map=reg_map,
            customer_industry_map=ind_map,
            splits={"val": split_meta},
            test_start=TEST_START_V2,
        )
    test_day = float((TEST_START_V2 - PURCHASE_DATE_ORIGIN).days)
    with pytest.raises(ValueError, match="Held-out test query"):
        compute_extra_cand_5_arrays(
            commercial_events=events,
            universe=universe,
            customer_region_map=reg_map,
            customer_industry_map=ind_map,
            splits={"val": {**split_meta, "prediction_days": torch.tensor([float(q_day), test_day])}},
            test_start=TEST_START_V2,
        )


def test_option_a_70_30_dropout_unscaled_and_eval_behavior():
    torch.manual_seed(0)
    head = OptionAUnseenHead(d_model=16, cold_dropout_prob=0.30)
    x = torch.ones(2, 5, 37) * 3.5

    # Forced cold mask: row 0 cold (only 7 global cols active, unscaled 3.5), row 1 full (all 37 active, unscaled 3.5)
    head.train()
    masked, cold_b = head.apply_feature_group_dropout(x, force_cold_mask=torch.tensor([True, False]))
    assert cold_b.tolist() == [True, False]
    global_set = set(UNSEEN_37_GLOBAL_LOCAL_INDICES)
    for j in range(37):
        if j in global_set:
            assert torch.allclose(masked[0, :, j], torch.full((5,), 3.5))
        else:
            assert torch.allclose(masked[0, :, j], torch.zeros(5))
    assert torch.allclose(masked[1], x[1])

    # Eval mode: 100% of features active
    head.eval()
    eval_out, eval_cold = head.apply_feature_group_dropout(x)
    assert not bool(eval_cold.any())
    assert torch.equal(eval_out, x)

    # Monte Carlo check of 70/30 split rate
    head.train()
    big_x = torch.ones(5000, 1, 37)
    _, cold_flags = head.apply_feature_group_dropout(big_x)
    rate = float(cold_flags.float().mean().item())
    assert 0.27 <= rate <= 0.33


@pytest.mark.parametrize("arm_idx", [0, 1, 2, 3])
def test_wave2b_models_gradients_bf16_and_inference_parity(arm_idx):
    torch.manual_seed(42)
    spec = build_wave2b_arm_specs(ACTIVE_V2_FULL_BASE_MODEL)[arm_idx]
    b, c, n_tr_c, s = 3, 12, 9, 8
    attr_vocabs = {"category": 4, "subcategory": 4, "business_line": 3, "tier": 3, "unit": 3, "country_of_origin": 3}
    prod_inputs = {
        "text_embedding": torch.randn(c, 1024),
        "demand_features": torch.rand(c, 65),
        "attr_ids": {k: torch.randint(0, sz, (c,)) for k, sz in attr_vocabs.items()},
    }
    common_kw = dict(
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=spec["num_pairwise_bias_features"],
        active_seq_stat_indices=tuple(spec["active_seq_stat_indices"]),
        pure_v_query=spec["pure_v_query"],
        option_a_unseen_head=spec["option_a_unseen_head"],
        unseen_dropout_cold_prob=0.30,
        d_model=128,
        num_heads=8,
        memory_ffn_dim=384,
        aux_60d_enabled=True,
        seed=7,
    )
    model = Wave2BE2ERanker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        unseen_attn_aux_enabled=spec["unseen_attn_aux_enabled"],
        **common_kw,
    )

    assert model.statistics_attention_weights.shape == (8, spec["num_pairwise_bias_features"] + 1, 5 + 7)
    if spec["pure_v_query"]:
        assert model.candidate_mlp is None and model.candidate_norm is None

    cmask = torch.zeros(b, c, dtype=torch.bool)
    cmask[:, :n_tr_c] = True
    batch = {
        "company_text_embedding": torch.randn(b, 1024),
        "city_id": torch.randint(0, 5, (b,)),
        "region_id": torch.randint(0, 4, (b,)),
        "industry_id": torch.randint(0, 4, (b,)),
        "customer_features": torch.rand(b, 7),
        "sequence": torch.rand(b, s, 9),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0] * 8], dtype=torch.long),
        "seq_product_ids": torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0], [0] * 8], dtype=torch.long),
        "seq_purchase_dates": torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0, 0, 0], [0.0] * 8]),
        "prediction_day": torch.tensor([200.0, 200.0, 200.0]),
        "candidate_features": torch.rand(b, c, 42),
        "candidate_mask": cmask,
    }
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        res = model(**batch, prod_inputs=prod_inputs, train_bn_limit=n_tr_c, return_aux=True)

    loss = res["output"].logits[:, :n_tr_c].float().pow(2).mean() + res["aux_60d_logits"][:, :n_tr_c].float().pow(2).mean()
    if spec["option_a_unseen_head"]:
        loss = (
            loss
            + res["unseen_logits"][:, :n_tr_c].float().pow(2).mean()
            + res["unseen_aux_60d_logits"][:, :n_tr_c].float().pow(2).mean()
        )
    elif spec["unseen_attn_aux_enabled"]:
        loss = loss + res["unseen_attn_logits"][:, :n_tr_c].float().pow(2).mean()
    loss.backward()

    assert model.statistics_attention_weights.grad is not None
    assert torch.isfinite(model.statistics_attention_weights.grad).all()
    assert model.statistics_attention_weights.grad.abs().sum().item() > 0
    if spec["option_a_unseen_head"]:
        assert model.option_a_head.main_head.weight.grad.abs().sum().item() > 0
        assert model.option_a_head.aux_60d_head.weight.grad.abs().sum().item() > 0

    # Eval mode: live vs cached parity + stripped-aux strict=True load parity
    model.eval()
    eval_batch = {**batch, "candidate_mask": torch.ones(b, c, dtype=torch.bool)}
    with torch.no_grad():
        out_live = model(**eval_batch, prod_inputs=prod_inputs, return_aux=False)
        v_cached = model.precompute_product_vectors(prod_inputs)
        out_cached = model(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
        assert torch.allclose(out_live["output"].logits, out_cached["output"].logits, atol=1e-5)
        if spec["pure_v_query"]:
            assert torch.allclose(out_cached["output"].z_candidate[0], v_cached, atol=1e-6)

    if spec["option_a_unseen_head"]:
        with torch.no_grad():
            saved_bias = model.statistics_attention_weights.detach().clone()
            model.statistics_attention_weights.normal_()
            force_cold = torch.ones(b, dtype=torch.bool)
            original = model(**eval_batch, precomputed_v_bn=v_cached, force_cold_mask=force_cold)
            changed = eval_batch["candidate_features"].clone()
            changed[..., 5] += 2.0
            alternate = model(**{**eval_batch, "candidate_features": changed},
                              precomputed_v_bn=v_cached, force_cold_mask=force_cold)
            torch.testing.assert_close(original["unseen_logits"], alternate["unseen_logits"], rtol=0, atol=0)
            assert not torch.equal(original["output"].logits, alternate["output"].logits)
            model.statistics_attention_weights.copy_(saved_bias)

    inf_kw = {**common_kw, "aux_60d_enabled": False}
    inf_m = Wave2BE2ERanker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        unseen_attn_aux_enabled=False,
        **inf_kw,
    ).eval()

    inf_m.load_state_dict(strip_wave2b_training_aux(model.state_dict()), strict=True)
    with torch.no_grad():
        out_inf = inf_m(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
        assert torch.allclose(out_cached["output"].logits, out_inf["output"].logits, atol=1e-5)
        if spec["option_a_unseen_head"]:
            assert torch.allclose(out_cached["unseen_logits"], out_inf["unseen_logits"], atol=1e-5)


def test_pure_unseen_eval_and_oof_affine_blend_exactness():
    rng = np.random.default_rng(123)
    n_rows, n_c = 40, 30
    allowed = np.ones((n_rows, n_c), dtype=bool)
    allowed[:, -3:] = False
    unseen = np.ones((n_rows, n_c), dtype=bool)
    unseen[:, :8] = False  # first 8 are repeats

    target = np.zeros((n_rows, n_c), dtype=bool)
    new_target = np.zeros((n_rows, n_c), dtype=bool)
    for i in range(n_rows):
        target[i, i % 6] = True  # 1 repeat target
        if i % 2 == 0:
            u_item = 8 + (i % 15)
            target[i, u_item] = True
            new_target[i, u_item] = True

    s_main = rng.normal(0.0, 1.0, size=(n_rows, n_c)).astype(np.float32)
    # Make s_unseen strongly informative on new_target, but badly offset (+5.0) and scaled (0.5x)
    s_unseen = (rng.normal(0.0, 0.5, size=(n_rows, n_c)) + 3.0 * new_target + 5.0).astype(np.float32)
    cids = [f"C{i // 2}" for i in range(n_rows)]
    fo = torch.tensor([i >= 30 for i in range(n_rows)], dtype=torch.bool)

    # 1. Pure Unseen Eval must ignore high-scoring repeat items completely
    s_with_huge_repeats = s_unseen.copy()
    s_with_huge_repeats[:, :8] = 1e6
    pu = evaluate_pure_unseen_predictions(
        torch.from_numpy(s_with_huge_repeats),
        allowed_bn=torch.from_numpy(allowed),
        unseen_bn=torch.from_numpy(unseen),
        new_target_bn=torch.from_numpy(new_target),
        first_observed=fo,
    )
    assert pu["overall"]["total_new_targets"] == int(new_target.sum())
    assert pu["overall"]["pure_unseen_hits@10"] == int(new_target.sum())
    assert pu["returning"]["total_new_targets"] + pu["first_observed"]["total_new_targets"] == pu["overall"]["total_new_targets"]

    # 2. Exact parity of pre-extracted Top-K affine evaluator vs brute-force full-pool ranking
    pre = _prepare_exact_top_candidates(s_main, s_unseen, allowed, unseen, target, new_target, top_k_each=15)
    mask = np.ones(n_rows, dtype=bool)
    fast_ndcg, fast_new, fast_rec = _eval_affine_ab_on_subset(pre, mask, a=0.75, b=-3.2)
    full_scores = np.where(unseen, 0.75 * s_unseen - 3.2, s_main)
    full_scores = np.where(allowed, full_scores, -np.inf)
    order = np.argsort(-full_scores, axis=1, kind="mergesort")[:, :10]
    rows = np.arange(n_rows)[:, None]
    ty = target[rows, order].astype(np.float64)
    tn = new_target[rows, order].astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    dcg = (ty * disc[None, :]).sum(axis=1)
    idcg = np.concatenate([[0.0], np.cumsum(disc)])[target.sum(axis=1)]
    assert abs(fast_ndcg - float(np.mean(dcg / idcg))) < 1e-10
    assert fast_new == int(tn.sum())
    assert abs(fast_rec - float(np.mean(ty.sum(axis=1) / target.sum(axis=1)))) < 1e-10

    # 3. 2-fold customer-grouped OOF affine blend improves over raw uncalibrated blend and is strictly OOF
    oof_scores, oof_meta = fit_oof_affine_unseen_blend(
        torch.from_numpy(s_main),
        torch.from_numpy(s_unseen),
        allowed_bn=torch.from_numpy(allowed),
        unseen_bn=torch.from_numpy(unseen),
        target_bn=torch.from_numpy(target),
        new_target_bn=torch.from_numpy(new_target),
        customer_ids=cids,
        n_folds=2,
    )
    assert oof_scores.shape == (n_rows, n_c)
    assert sum(f["oof_baskets"] for f in oof_meta["folds"]) == n_rows
    assert all(f["a"] > 0.0 for f in oof_meta["folds"])
    # Check that OOF blend beats uncalibrated s_main on NDCG@10
    assert oof_meta["deployment_full_fit"]["ndcg@10"] > _eval_affine_ab_on_subset(
        _prepare_exact_top_candidates(s_main, s_main, allowed, unseen, target, new_target), mask, 1.0, 0.0
    )[0]


def _make_mini_wave2b_fixture():
    torch.manual_seed(11)
    b, c, n_tr_c, s = 8, 12, 9, 6
    attrs = {"category": 4, "subcategory": 4, "business_line": 3, "tier": 3, "unit": 3, "country_of_origin": 3}
    prod_inputs = {
        "text_embedding": torch.randn(c, 1024),
        "demand_features": torch.rand(c, 65),
        "attr_ids": {k: torch.randint(0, sz, (c,)) for k, sz in attrs.items()},
    }
    shared = {
        "product_ids": [f"P{i}" for i in range(c)],
        "universe": {
            "trained_base_names": n_tr_c,
            "num_base_names": c,
            "base_names": [f"B{i}" for i in range(c)],
            "bn_to_member_pid_indices": [[i] for i in range(c)],
            "all_pid_to_bn": {f"P{i}": f"B{i}" for i in range(c)},
            "bn_to_idx": {f"B{i}": i for i in range(c)},
        },
        "prod_inputs": prod_inputs,
        "vocabs": {k: dict.fromkeys(range(5)) for k in ("city_vocab", "region_vocab", "industry_vocab")},
        "enrichment": {"product_vocabs": {k: dict.fromkeys(range(n)) for k, n in attrs.items()}},
    }
    allowed = torch.zeros(b, c, dtype=torch.bool)
    allowed[:, :n_tr_c] = True
    unseen = allowed.clone()
    unseen[:, :3] = False
    target = torch.zeros(b, c, dtype=torch.bool)
    new_target = torch.zeros(b, c, dtype=torch.bool)
    for i in range(b):
        target[i, 0] = True
        if i % 2 == 0:
            target[i, 4 + (i % 4)] = True
            new_target[i, 4 + (i % 4)] = True

    c38 = np.random.default_rng(1).random((b, c, 38)).astype(np.float16)
    c38[:, :3, 0] = 1.0
    c38[:, 3:, 0] = 0.0
    e5 = np.random.default_rng(2).random((b, c, 5)).astype(np.float16)

    meta = {
        "customer_ids": [f"C{i}" for i in range(b)],
        "prediction_days": torch.linspace(300.0, 320.0, b),
        "first_observed": torch.tensor([i >= b - 2 for i in range(b)], dtype=torch.bool),
        "allowed_bn": allowed,
        "unseen_bn": unseen,
        "target_bn": target,
        "new_target_bn": new_target,
        "aux_60d_bn": target,
        "full_window": torch.ones(b, dtype=torch.bool),
        "text_embedding": torch.randn(b, 1024),
        "city_id": torch.randint(0, 5, (b,)),
        "region_id": torch.randint(0, 4, (b,)),
        "industry_id": torch.randint(0, 4, (b,)),
        "tabular_features": torch.rand(b, 11),
        "sequence": torch.rand(b, s, 9),
        "attention_mask": torch.tensor([[1, 1, 0, 0, 0, 0]] * b, dtype=torch.long),
        "seq_product_ids": torch.tensor([[1, 2, 0, 0, 0, 0]] * b, dtype=torch.long),
        "seq_purchase_dates": torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.0, 0.0]] * b),
    }
    val = {
        **meta,
        "prediction_days": meta["prediction_days"] + 100.0,
        "sku_val": {"target": target, "new_target": new_target, "allowed": allowed},
    }
    dev_purchases = pd.DataFrame(columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({"product_id": shared["product_ids"], "container_type": ["box"] * c})
    return shared, meta, val, c38, e5, dev_purchases, catalogue


def test_wave2b_runner_trains_replays_and_resumes(tmp_path, monkeypatch):
    from recommender.experiments.v2_full import run_v2_full_wave2b as runner

    shared, tr_meta, va_meta, c38, e5, dev_purchases, catalogue = _make_mini_wave2b_fixture()
    spec = build_wave2b_arm_specs(ACTIVE_V2_FULL_BASE_MODEL)[-1]  # optionA + pureQ + full42bias (exercises all paths!)

    kwargs = dict(
        arm_cfg=spec,
        shared_meta=shared,
        tr_meta=tr_meta,
        va_meta=va_meta,
        tr_cand38_mm=c38,
        va_cand38_mm=c38,
        tr_extra5_mm=e5,
        va_extra5_mm=e5,
        extra5_identity="test_id_123",
        dev_purchases=dev_purchases,
        catalogue=catalogue,
        out_dir=tmp_path / "arm_optA",
        seed=42,
        epochs=1,
        batch_size=4,
        dev=torch.device("cpu"),
    )
    res = runner.train_and_eval_wave2b_arm(**kwargs)
    assert res["ndcg"]["epoch"] == 1
    assert "optionA_oof_ndcg" in res and "optionA_oof_unseen" in res
    assert "pure_unseen_head" in res["ndcg"]
    assert (tmp_path / "arm_optA" / "val_logits_ndcg.npy").is_file()
    assert (tmp_path / "arm_optA" / "val_unseen_logits_ndcg.npy").is_file()

    # Instant verified resume without retraining
    monkeypatch.setattr(runner, "iter_paired_training_batches", lambda *a, **kw: pytest.fail("Unexpected retrain"))
    assert runner.train_and_eval_wave2b_arm(**kwargs) == res
    # A single changed candidate value must invalidate resume even with the same row keys.
    c38[0, 0, 1] += 1
    with pytest.raises(pytest.fail.Exception, match="Unexpected retrain"):
        runner.train_and_eval_wave2b_arm(**kwargs)


def test_catboost_v2_full_ipw_sampling_full_pool_eval_and_ensemble(tmp_path):
    pytest.importorskip("catboost")
    from recommender.experiments.v2_full.run_v2_full_catboost import (
        fit_oof_linear_ensemble,
        sample_weighted_catboost_training_pool,
        train_and_eval_catboost_model,
    )

    shared, tr_meta, va_meta, c38, e5, dev_purchases, catalogue = _make_mini_wave2b_fixture()

    # 1. Verify exact Horvitz-Thompson IPW weight sum for Mixed and Unseen-Only pools
    pool_m, stats_m = sample_weighted_catboost_training_pool(
        model_name="cb_mixed_cand42",
        shared_meta=shared,
        tr_meta=tr_meta,
        tr_cand38_mm=c38,
        tr_extra5_mm=e5,
        negatives_per_basket=3,
        seed=42,
    )
    expected_total_neg_m = int(((~tr_meta["target_bn"][:, :9]) & tr_meta["allowed_bn"][:, :9]).sum().item())
    assert stats_m["effective_negative_weight"] == pytest.approx(float(expected_total_neg_m))
    assert stats_m["num_features"] == 42 + 7 + 9

    pool_u, stats_u = sample_weighted_catboost_training_pool(
        model_name="cb_optionA_unseen_cand42",
        shared_meta=shared,
        tr_meta=tr_meta,
        tr_cand38_mm=c38,
        tr_extra5_mm=e5,
        negatives_per_basket=3,
        seed=42,
    )
    assert stats_u["baskets_used"] == 4  # only the 4 baskets with new targets
    assert stats_u["num_features"] == 37 + 7 + 9
    del pool_m, pool_u

    # 2. Train & evaluate cb_mixed_cand42 and cb_optionA_unseen_cand42 end-to-end
    fit_args = dict(
        model_name="cb_mixed_cand42",
        shared_meta=shared,
        tr_meta=tr_meta,
        va_meta=va_meta,
        tr_cand38_mm=c38,
        va_cand38_mm=c38,
        tr_extra5_mm=e5,
        va_extra5_mm=e5,
        extra5_identity="id_cb",
        dev_purchases=dev_purchases,
        catalogue=catalogue,
        out_dir=tmp_path / "cb_mixed_cand42",
        iterations=10,
        depth=4,
        negatives_per_basket=4,
        task_type="CPU",
        seed=42,
    )
    res_m, lg_m = train_and_eval_catboost_model(**fit_args)
    resumed, resumed_logits = train_and_eval_catboost_model(**fit_args)
    assert resumed == res_m
    torch.testing.assert_close(resumed_logits, lg_m, rtol=0, atol=0)
    assert lg_m.shape == (8, 12)
    assert torch.isinf(lg_m[:, 9:]).all() and (lg_m[:, 9:] < 0).all()
    assert torch.isfinite(lg_m[:, :9]).all()
    assert "pure_unseen" in res_m and "mixed_eval" in res_m

    res_u, lg_u = train_and_eval_catboost_model(
        "cb_optionA_unseen_cand42",
        shared_meta=shared,
        tr_meta=tr_meta,
        va_meta=va_meta,
        tr_cand38_mm=c38,
        va_cand38_mm=c38,
        tr_extra5_mm=e5,
        va_extra5_mm=e5,
        extra5_identity="id_cb",
        dev_purchases=dev_purchases,
        catalogue=catalogue,
        out_dir=tmp_path / "cb_optionA_unseen_cand42",
        iterations=10,
        depth=4,
        negatives_per_basket=4,
        task_type="CPU",
        seed=42,
    )
    assert lg_u.shape == (8, 12) and torch.isfinite(lg_u[:, :9]).all()

    # 3. Linear OOF ensemble between synthetic neural logits and CatBoost logits
    neural_lg = torch.randn(8, 12)
    ens_lg, ens_meta = fit_oof_linear_ensemble(
        neural_lg,
        lg_m,
        allowed_bn=va_meta["allowed_bn"],
        target_bn=va_meta["target_bn"],
        new_target_bn=va_meta["new_target_bn"],
        customer_ids=va_meta["customer_ids"],
        n_folds=2,
    )
    assert ens_lg.shape == (8, 12)
    assert sum(f["oof_baskets"] for f in ens_meta["folds"]) == 8
    assert torch.isfinite(ens_lg[:, :9]).all()


def test_calibration_matches_reporting_with_cross_group_ties_and_unavailable_targets():
    from recommender.experiments.roadmap_v2.stage5_base_entity import evaluate_base_name_predictions
    from recommender.experiments.v2_full.train_v2_full_baseline import rank_eligible_formulations

    scores = torch.zeros(3, 24)
    allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed[:, -1] = False
    unseen = allowed.clone()
    unseen[:, 1:15] = False
    target = torch.zeros_like(allowed)
    target[0, [0, 23]] = True
    target[1, 23] = True
    new = target.clone()
    ranked = rank_eligible_formulations(scores, allowed)
    ref = evaluate_base_name_predictions(ranked, target, new, ["A", "B", "C"])
    pre = _prepare_exact_top_candidates(*(t.numpy() for t in (scores, scores, allowed, unseen, target, new)))
    actual = _eval_affine_ab_on_subset(pre, np.ones(3, bool), 1., 0.)
    assert actual == pytest.approx((ref["bn_ndcg@10"], ref["true_new_bn_hits@10"], ref["bn_recall@10"]))
    metrics = evaluate_pure_unseen_predictions(
        scores, allowed_bn=allowed, unseen_bn=unseen, new_target_bn=new,
        first_observed=torch.tensor([False, True, False])
    )
    assert metrics["overall"]["total_new_targets"] == 3
    assert metrics["overall"]["pure_unseen_recall@10"] == pytest.approx(1 / 3)
    assert metrics["first_observed"]["baskets_with_new"] == 1


def test_sealed_ensemble_inputs_distinguish_arms_and_verify_scores(tmp_path):
    import json
    pytest.importorskip("catboost")
    from recommender.artifacts import file_digest
    from recommender.experiments.v2_full.run_v2_full_catboost import _neural_ensemble_inputs

    paths = []
    for arm in ("w2b_cand42_pruned", "w2b_cand42_pureQ_full42bias"):
        path = tmp_path / "wave2b_arm4_e2e_unseen_attn_aux-seed42" / arm / "val_logits_ndcg.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.zeros((2, 12), np.float32))
        manifest = {"run_settings": {"dataset_identity": "shared"}, "output_sha256": {path.name: file_digest(path)}}
        (path.parent / "arm_result.json").write_text(json.dumps(manifest))
        paths.append(path)
    assert set(_neural_ensemble_inputs(paths, "shared").values()) == set(paths)
    with pytest.raises(ValueError, match="Duplicate"):
        _neural_ensemble_inputs([paths[0], paths[0]], "shared")
    with pytest.raises(ValueError, match="different prepared"):
        _neural_ensemble_inputs(paths, "different")
    np.save(paths[0], np.ones((2, 12), np.float32))
    with pytest.raises(ValueError, match="checksum"):
        _neural_ensemble_inputs(paths, "shared")


def test_cache_build_uses_development_reader_and_invalidates_mapping(tmp_path, monkeypatch):
    from recommender.experiments.static_enrichment import metadata
    from recommender.experiments.v2_full import extra_features as extra

    shared, train, val, *_ = _make_mini_wave2b_fixture()
    train = {**train, "prediction_days": torch.arange(8) + 350}
    val = {**val, "prediction_days": torch.arange(8) + 406}
    shared.update(train_cutoff="2026-02-01", test_start="2026-07-01")
    for name, split in (("train", train), ("val", val)):
        split.update(first_observed=torch.zeros(8, dtype=torch.bool), familiarity_bn=torch.zeros(8, 12, 4),
                     feature_names=CANDIDATE_38_FEATURE_NAMES, customer_feature_names=CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
                     cand_38_path=str(tmp_path / f"{name}_cand.npy"))
        np.save(split["cand_38_path"], np.zeros((8, 12, 38), np.float16))
        torch.save(split, tmp_path / f"{name}_meta.pt")
    torch.save(shared, tmp_path / "shared_meta.pt")

    customer_path = tmp_path / "customers.csv"
    customer_path.touch()
    monkeypatch.setattr(extra, "RAW_CUSTOMERS_PATH", customer_path)
    purchases = pd.DataFrame({"customer_id": ["C0"], "product_id": ["P1"],
                               "purchase_date": [pd.Timestamp("2025-12-10")], "manager_id": ["M"]})
    parquet = tmp_path / "events.parquet"
    purchases.to_parquet(parquet)
    monkeypatch.setattr(extra, "load_v2_development_purchases", lambda: purchases)
    def load_companies(path, **kwargs):
        assert path == customer_path and kwargs["purchases"] is purchases and kwargs["minimum_purchases"] == 0
        return pd.DataFrame(columns=["customer_id", "city", "region_code", "industry_branch"])
    monkeypatch.setattr(metadata, "load_enriched_companies", load_companies)
    args = dict(prepared_root=tmp_path, commercial_parquet_path=parquet)
    train_path, _, original = extra.build_or_load_extra_cand_5(**args)
    assert extra.build_or_load_extra_cand_5(**args)[0] == train_path
    shared["universe"]["all_pid_to_bn"]["P1"] = "B2"
    torch.save(shared, tmp_path / "shared_meta.pt")
    updated_path, _, updated = extra.build_or_load_extra_cand_5(**args)
    assert original["identity"] != updated["identity"] and train_path.is_file() and updated_path.is_file()
    assert np.load(train_path)[0, 1, 0] == np.load(updated_path)[0, 2, 0] == 1


def test_pure_ranking_retains_blend_precision_and_rejects_invalid_eligible_scores():
    from recommender.experiments.v2_full.train_v2_full_baseline import rank_eligible_formulations

    scores = torch.ones((1, 11), dtype=torch.float64)
    scores[0, -1] += 1e-10
    eligible = torch.ones_like(scores, dtype=torch.bool)
    target = torch.zeros_like(eligible)
    target[0, -1] = True
    kwargs = dict(allowed_bn=eligible, unseen_bn=eligible, new_target_bn=target,
                  first_observed=torch.tensor([True]))
    assert evaluate_pure_unseen_predictions(scores, **kwargs)["overall"]["pure_unseen_ndcg@10_new_baskets"] == 1
    scores[0, -1] = torch.nan
    with pytest.raises(ValueError, match="finite scores"):
        evaluate_pure_unseen_predictions(scores, **kwargs)
    with pytest.raises(ValueError, match="finite scores"):
        rank_eligible_formulations(scores, eligible)


def test_v2_full_cand42_reference_defaults_and_checksum_verification(tmp_path):
    import json
    from recommender.artifacts import file_digest
    from recommender.experiments.v2_full.config import (
        ACTIVE_V2_FULL_BASE_MODEL,
        ACTIVE_V2_FULL_REFERENCE_ID,
        V2_FULL_REFERENCE_PATH,
        resolve_v2_full_reference_predictions,
    )
    assert ACTIVE_V2_FULL_BASE_MODEL == "arm4_e2e_unseen_attn_aux"
    assert ACTIVE_V2_FULL_REFERENCE_ID == "v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6"
    if pytest.importorskip("catboost"):
        from recommender.experiments.v2_full.run_v2_full_catboost import ALL_CB_MODELS, DEFAULT_CB_MODELS

        assert ALL_CB_MODELS == DEFAULT_CB_MODELS == ("cb_mixed_cand42", "cb_optionA_unseen_cand42")

    if V2_FULL_REFERENCE_PATH.is_file():
        doc = json.loads(V2_FULL_REFERENCE_PATH.read_text(encoding="utf-8"))
        assert doc["reference_id"] == ACTIVE_V2_FULL_REFERENCE_ID
        assert doc["active_development_base"] == "arm4_e2e_unseen_attn_aux"
        assert doc["active_feature_schema"]["num_candidate_features"] == 42
        assert doc["active_feature_schema"]["num_customer_features"] == 7
        assert doc["active_feature_schema"]["active_seq_stat_indices"] == [0, 1, 5, 6, 8]
        assert "mgr_base_name_growth_30d_vs_90d" in doc["active_feature_schema"]["dropped_features"]
        arch = doc["active_architecture"]
        assert arch["arm"] == "w2c_exp1_clean_sep_w10"
        assert arch["pure_v_query"] is True
        assert arch["num_pairwise_bias_features"] == 42
        assert arch["option_a_unseen_head"] is True
        assert arch["unseen_dropout_cold_prob"] == 0.0
        assert arch["separated_main_targets"] is True
        assert arch["unseen_head_weight"] == 1.0
        resolved_real = resolve_v2_full_reference_predictions()
        assert {"v2_full_ref_primary", "v2_full_ref_ndcg", "v2_full_ref_cal_ndcg"}.issubset(resolved_real.keys())
        for path in resolved_real.values():
            assert path.is_file()

    # Synthetic tamper check on resolve_v2_full_reference_predictions
    prim_p = tmp_path / "primary.npz"
    prim_p.write_bytes(b"clean_primary_payload")
    ref_p = tmp_path / "ref.json"
    ref_p.write_text(
        json.dumps(
            {
                "checkpoints": {
                    "predictions_primary": {"path": str(prim_p), "sha256": file_digest(prim_p)},
                }
            }
        ),
        encoding="utf-8",
    )
    assert resolve_v2_full_reference_predictions(reference_path=ref_p) == {"v2_full_ref_primary": prim_p}
    prim_p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_v2_full_reference_predictions(reference_path=ref_p)
