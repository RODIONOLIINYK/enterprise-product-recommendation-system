"""Synthetic unit & integration tests for Wave 2C (Clean Separated Option A Loss Weighting)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.v2_full.config import ACTIVE_V2_FULL_BASE_MODEL, TEST_START_V2
from recommender.experiments.v2_full.extra_features import (
    CAND_42_NON_REPEAT_INDICES,
    CAND_42_REPEAT_INDICES,
)
from recommender.experiments.v2_full.run_v2_full_wave2c import (
    build_wave2c_arm_specs,
    compute_wave2c_losses,
    train_and_eval_wave2c_arm,
)
from recommender.experiments.v2_full.wave2b_models import (
    Wave2BE2ERanker,
    fit_oof_affine_unseen_blend,
)
from recommender.ranker import RankerLoss
from recommender.sequences import PURCHASE_DATE_ORIGIN


def _make_mini_world() -> Tuple[pd.DataFrame, pd.DataFrame, list[str], Dict, list[int]]:
    """Build a minimal 4-item synthetic world strictly < TRAIN_CUTOFF_V2."""
    catalogue = pd.DataFrame(
        [
            {
                "product_id": "P0",
                "product_name": "Shell Tellus S2 M 46 20L",
                "base_name": "Tellus S2 M 46",
                "short_name": "Tellus 46",
                "brand": "Shell",
                "product_type": "Oil",
                "product_category": "Hydraulic",
                "product_subcategory": "HLP",
                "business_line": "Industrial",
                "product_family": "Tellus",
                "container_type": "CANISTER",
                "tier": "Premium",
                "unit": "L",
                "country_of_origin": "DE",
            },
            {
                "product_id": "P1",
                "product_name": "Mobil DTE 25 ISO 46 208L",
                "base_name": "DTE 25 46",
                "short_name": "DTE 46",
                "brand": "Mobil",
                "product_type": "Oil",
                "product_category": "Hydraulic",
                "product_subcategory": "HLP",
                "business_line": "Industrial",
                "product_family": "DTE",
                "container_type": "DRUM",
                "tier": "Premium",
                "unit": "L",
                "country_of_origin": "FR",
            },
            {
                "product_id": "P2",
                "product_name": "Castrol Rimula 15W-40 20L",
                "base_name": "Rimula 15W-40",
                "short_name": "Rimula 15W-40",
                "brand": "Castrol",
                "product_type": "Oil",
                "product_category": "Engine",
                "product_subcategory": "HDDEO",
                "business_line": "Auto",
                "product_family": "Rimula",
                "container_type": "CANISTER",
                "tier": "Mainstream",
                "unit": "L",
                "country_of_origin": "PL",
            },
            {
                "product_id": "P3",
                "product_name": "Shell Gadus S2 V220 NLGI 2 18kg",
                "base_name": "Gadus S2 V220",
                "short_name": "Gadus 2",
                "brand": "Shell",
                "product_type": "Grease",
                "product_category": "Grease",
                "product_subcategory": "Lithium",
                "business_line": "Industrial",
                "product_family": "Gadus",
                "container_type": "PAIL",
                "tier": "Premium",
                "unit": "KG",
                "country_of_origin": "DE",
            },
        ]
    )
    q_ts = pd.Timestamp("2026-01-20")
    purchases = pd.DataFrame(
        [
            {"customer_id": "C1", "purchase_date": q_ts - pd.Timedelta(days=50), "product_id": "P0", "quantity": 10.0},
            {"customer_id": "C1", "purchase_date": q_ts - pd.Timedelta(days=25), "product_id": "P1", "quantity": 8.0},
            {"customer_id": "C1", "purchase_date": q_ts - pd.Timedelta(days=10), "product_id": "P2", "quantity": 4.0},
            {"customer_id": "C2", "purchase_date": q_ts - pd.Timedelta(days=20), "product_id": "P3", "quantity": 2.0},
        ]
    )
    product_ids = ["P0", "P1", "P2", "P3"]
    base_names = ["Tellus S2 M 46", "DTE 25 46", "Rimula 15W-40", "Gadus S2 V220"]
    universe = {
        "base_names": base_names,
        "num_base_names": 4,
        "trained_base_names": 4,
        "bn_to_idx": {b: i for i, b in enumerate(base_names)},
        "all_pid_to_bn": dict(zip(product_ids, base_names)),
        "bn_to_member_pid_indices": [[0], [1], [2], [3]],
        "pid_idx_to_bn_idx": np.arange(4, dtype=np.int64),
    }
    reps = [0, 1, 2, 3]
    return purchases, catalogue, product_ids, universe, reps


def test_wave2c_schemas_and_loss_weight_arm_specs():
    assert len(CAND_42_REPEAT_INDICES) == 5
    assert len(CAND_42_NON_REPEAT_INDICES) == 37
    assert set(CAND_42_REPEAT_INDICES).isdisjoint(CAND_42_NON_REPEAT_INDICES)

    specs = build_wave2c_arm_specs(ACTIVE_V2_FULL_BASE_MODEL)
    assert [s["name"] for s in specs] == [
        "w2c_exp1_clean_sep_w10",
        "w2c_exp1_clean_sep_w20",
    ]
    for s in specs:
        assert s["unseen_dropout_cold_prob"] == 0.0
        assert s["pure_v_query"] is True
        assert s["num_pairwise_bias_features"] == 42
        assert s["option_a_unseen_head"] is True
        assert s["separated_main_targets"] is True
        assert s["unseen_attn_aux_enabled"] is False
        assert "synthetic_unseen_aug" not in s
    assert [s["unseen_head_weight"] for s in specs] == [1.0, 2.0]


def test_wave2c_conflict_free_loss_masks_and_weight_scaling():
    # Row 0: mixed returning basket (BN0=repeat pos, BN1=repeat neg, BN2=unseen pos, BN3=unseen neg)
    # Row 1: 0-repeat returning basket (BN0=repeat neg, BN1=unseen pos, BN2=unseen neg, BN3=unseen neg)
    # Row 2: first-observed basket (all unseen; BN2=unseen pos)
    main_lg = torch.zeros((3, 4), requires_grad=True)
    main_aux_lg = torch.zeros((3, 4), requires_grad=True)
    unseen_lg = torch.zeros((3, 4), requires_grad=True)
    unseen_aux_lg = torch.zeros((3, 4), requires_grad=True)

    allowed = torch.ones((3, 4), dtype=torch.bool)
    unseen = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    target = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
            [False, False, True, False],
        ],
        dtype=torch.bool,
    )
    aux_60d = target.clone()
    full_window = torch.tensor([True, True, True], dtype=torch.bool)
    first_obs = torch.tensor([False, False, True], dtype=torch.bool)

    def _run_weight(w: float):
        main_lg.grad = main_aux_lg.grad = unseen_lg.grad = unseen_aux_lg.grad = None
        out = compute_wave2c_losses(
            main_logits=main_lg,
            main_aux_60d_logits=main_aux_lg,
            gen_unseen_logits=unseen_lg,
            gen_unseen_aux_60d_logits=unseen_aux_lg,
            allowed_bn=allowed,
            unseen_bn=unseen,
            target_bn=target,
            aux_60d_bn=aux_60d,
            full_window=full_window,
            first_observed=first_obs,
            main_loss_fn=RankerLoss(pos_weight=10.0, listwise_weight=1.0),
            main_aux_loss_fn=RankerLoss(pos_weight=5.0, listwise_weight=1.0),
            unseen_next_loss_fn=RankerLoss(pos_weight=10.0, listwise_weight=1.0),
            unseen_60d_loss_fn=RankerLoss(pos_weight=5.0, listwise_weight=1.0),
            unseen_head_weight=w,
        )
        out["loss"].backward()
        return main_lg.grad.clone(), main_aux_lg.grad.clone(), unseen_lg.grad.clone(), unseen_aux_lg.grad.clone()

    g_m05, g_ma05, g_u05, g_ua05 = _run_weight(0.5)

    # 1. Row 0 BN2 (unseen positive in mixed basket): ZERO gradient in Main Head (no conflict!)
    assert float(g_m05[0, 2].abs().item()) == 0.0
    assert float(g_ma05[0, 2].abs().item()) == 0.0
    # 2. Row 0 BN0 (repeat positive): NEGATIVE gradient in Main Head (pushes logit UP)
    assert float(g_m05[0, 0].item()) < 0.0
    # 3. Row 1 (0-repeat returning basket): BN1 (unseen pos) has ZERO grad; BN0 has POSITIVE grad (BCE pushes DOWN)
    assert float(g_m05[1, 1].abs().item()) == 0.0
    assert float(g_m05[1, 0].item()) > 0.0
    # 4. Row 2 (first-observed basket): ALL ZERO grad in Main Head!
    assert torch.all(g_m05[2] == 0.0)
    assert torch.all(g_ma05[2] == 0.0)
    # 5. Option A Head: ZERO grad on ALL repeat items; NEGATIVE grad on unseen positives!
    assert float(g_u05[0, 0].abs().item()) == 0.0
    assert float(g_u05[0, 1].abs().item()) == 0.0
    assert float(g_u05[1, 0].abs().item()) == 0.0
    assert float(g_u05[0, 2].item()) < 0.0
    assert float(g_u05[1, 1].item()) < 0.0
    assert float(g_u05[2, 2].item()) < 0.0

    # 6. Verify unseen_head_weight=1.0 (2x) and 2.0 (4x) scale Option A + aux60d gradients cleanly without altering Main!
    g_m10, _, g_u10, g_ua10 = _run_weight(1.0)
    g_m20, _, g_u20, g_ua20 = _run_weight(2.0)
    assert torch.allclose(g_m10, g_m05, atol=1e-6)
    assert torch.allclose(g_m20, g_m05, atol=1e-6)
    assert torch.allclose(g_u10, 2.0 * g_u05, atol=1e-6)
    assert torch.allclose(g_ua10, 2.0 * g_ua05, atol=1e-6)
    assert torch.allclose(g_u20, 4.0 * g_u05, atol=1e-6)
    assert torch.allclose(g_ua20, 4.0 * g_ua05, atol=1e-6)


def test_wave2c_zero_tabular_dropout_and_skip_main_head_guard():
    torch.manual_seed(0)
    vocab_sizes = {"category": 3, "subcategory": 3, "business_line": 3, "tier": 3, "unit": 3, "country_of_origin": 3}
    model = Wave2BE2ERanker(
        product_attr_vocab_sizes=vocab_sizes,
        num_cities=4,
        num_regions=4,
        num_industries=4,
        num_demand_features=8,
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=42,
        pure_v_query=True,
        option_a_unseen_head=True,
        unseen_dropout_cold_prob=0.0,
        unseen_attn_aux_enabled=False,
        text_dim=16,
        d_model=16,
        num_heads=4,
        memory_ffn_dim=32,
        mlp_widths=(24, 16),
        dropout=0.0,
        aux_60d_enabled=True,
        seed=42,
    )
    model.train()

    # Verify 0% feature-group dropout in train mode (never masks any of the 37 non-repeat features!)
    assert model.option_a_head is not None
    x_tab = torch.randn(8, 5, 37)
    x_out, cold_b = model.option_a_head.apply_feature_group_dropout(x_tab)
    assert not bool(cold_b.any())
    assert torch.equal(x_out, x_tab)

    # Guard: skip_main_head=True without option_a_unseen_head=True must raise ValueError
    m_no_opta = Wave2BE2ERanker(
        product_attr_vocab_sizes=vocab_sizes,
        num_cities=4,
        num_regions=4,
        num_industries=4,
        num_demand_features=8,
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=42,
        pure_v_query=True,
        option_a_unseen_head=False,
        text_dim=16,
        d_model=16,
        num_heads=4,
        memory_ffn_dim=32,
        mlp_widths=(24, 16),
        dropout=0.0,
    )
    prod_inputs = {
        "text_embedding": torch.randn(5, 16),
        "demand_features": torch.randn(5, 8),
        "attr_ids": {k: torch.randint(0, 3, (5,)) for k in vocab_sizes},
    }
    inp = {
        "company_text_embedding": torch.randn(2, 16),
        "city_id": torch.tensor([1, 2]),
        "region_id": torch.tensor([1, 2]),
        "industry_id": torch.tensor([1, 2]),
        "customer_features": torch.rand(2, 7),
        "sequence": torch.rand(2, 4, 9),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
        "seq_product_ids": torch.tensor([[1, 2, 0, 0], [0, 0, 0, 0]]),
        "seq_purchase_dates": torch.zeros(2, 4),
        "prediction_day": torch.tensor([100.0, 100.0]),
        "candidate_features": torch.rand(2, 5, 42),
        "candidate_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="skip_main_head=True requires option_a_unseen_head=True"):
        m_no_opta(**inp, prod_inputs=prod_inputs, skip_main_head=True)


def test_wave2c_end_to_end_train_replay_and_resume(tmp_path: Path):
    """Run both active Wave 2C loss-weighting arms (w=1.0 and w=2.0) end-to-end with replay & resume."""
    torch.manual_seed(42)
    np.random.seed(42)
    n_tr, n_va, n_bn, seq_len = 24, 12, 4, 4

    purchases, catalogue, product_ids, universe, reps = _make_mini_world()
    vocabs = {
        "city_vocab": {"Kyiv": 1},
        "region_vocab": {"R1": 1, "R2": 2},
        "industry_vocab": {"Agro": 1, "Mining": 2},
    }
    enrichment = {
        "product_vocabs": {
            "category": {"Hydraulic": 1, "Engine": 2, "Grease": 3},
            "subcategory": {"HLP": 1, "HDDEO": 2, "Lithium": 3},
            "business_line": {"Industrial": 1, "Auto": 2},
            "tier": {"Premium": 1, "Mainstream": 2},
            "unit": {"L": 1, "KG": 2},
            "country_of_origin": {"DE": 1, "FR": 2, "PL": 3},
        }
    }
    prod_inputs = {
        "text_embedding": torch.randn(n_bn, 1024),
        "demand_features": torch.rand(n_bn, 65),
        "attr_ids": {k: torch.ones(n_bn, dtype=torch.long) for k in enrichment["product_vocabs"]},
    }
    shared_meta = {
        "vocabs": vocabs,
        "enrichment": enrichment,
        "prod_inputs": prod_inputs,
        "product_ids": product_ids,
        "universe": universe,
        "reps": reps,
    }

    q_tr_day = float((pd.Timestamp("2026-01-20") - PURCHASE_DATE_ORIGIN).days)
    q_va_day = float((pd.Timestamp("2026-02-15") - PURCHASE_DATE_ORIGIN).days)

    def _make_split(n_rows: int, day_val: float, is_val: bool) -> Dict:
        cids = [f"C{i % 6}" for i in range(n_rows)]
        fo = torch.tensor([i % 4 == 0 for i in range(n_rows)], dtype=torch.bool)
        allowed = torch.ones((n_rows, n_bn), dtype=torch.bool)
        unseen = torch.ones((n_rows, n_bn), dtype=torch.bool)
        unseen[~fo, 0] = False
        unseen[~fo, 1] = False
        target = torch.zeros((n_rows, n_bn), dtype=torch.bool)
        target[~fo, 0] = True
        target[:, 2] = True
        new_target = target & unseen
        split = {
            "customer_ids": cids,
            "prediction_days": torch.full((n_rows,), day_val, dtype=torch.float32),
            "first_observed": fo,
            "allowed_bn": allowed,
            "unseen_bn": unseen,
            "target_bn": target,
            "new_target_bn": new_target,
            "aux_60d_bn": target.clone(),
            "full_window": torch.ones(n_rows, dtype=torch.bool),
            "text_embedding": torch.randn(n_rows, 1024),
            "city_id": torch.ones(n_rows, dtype=torch.long),
            "region_id": torch.ones(n_rows, dtype=torch.long),
            "industry_id": torch.ones(n_rows, dtype=torch.long),
            "tabular_features": torch.rand(n_rows, 11),
            "sequence": torch.rand(n_rows, seq_len, 9),
            "attention_mask": torch.where(
                fo[:, None],
                torch.zeros((n_rows, seq_len), dtype=torch.long),
                torch.ones((n_rows, seq_len), dtype=torch.long),
            ),
            "seq_product_ids": torch.where(
                fo[:, None],
                torch.zeros((n_rows, seq_len), dtype=torch.long),
                torch.ones((n_rows, seq_len), dtype=torch.long),
            ),
            "seq_purchase_dates": torch.zeros((n_rows, seq_len), dtype=torch.float32),
        }
        if is_val:
            split["sku_val"] = {
                "target": target.clone(),
                "new_target": new_target.clone(),
                "allowed": allowed.clone(),
            }
        return split

    tr_meta = _make_split(n_tr, q_tr_day, is_val=False)
    va_meta = _make_split(n_va, q_va_day, is_val=True)
    tr_c38 = np.random.rand(n_tr, n_bn, 38).astype(np.float16)
    va_c38 = np.random.rand(n_va, n_bn, 38).astype(np.float16)
    tr_e5 = np.random.rand(n_tr, n_bn, 5).astype(np.float16)
    va_e5 = np.random.rand(n_va, n_bn, 5).astype(np.float16)

    specs = {
        s["name"]: {**s, "d_model": 16, "num_heads": 4, "memory_ffn_dim": 32, "mlp_widths": [24, 16]}
        for s in build_wave2c_arm_specs()
    }

    for arm_name in ("w2c_exp1_clean_sep_w10", "w2c_exp1_clean_sep_w20"):
        arm_cfg = specs[arm_name]
        out_dir = tmp_path / arm_name
        res1 = train_and_eval_wave2c_arm(
            arm_cfg,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand38_mm=tr_c38,
            va_cand38_mm=va_c38,
            tr_extra5_mm=tr_e5,
            va_extra5_mm=va_e5,
            extra5_identity="test_e5",
            dataset_identity="test_ds",
            dev_purchases=purchases,
            catalogue=catalogue,
            out_dir=out_dir,
            seed=42,
            epochs=2,
            batch_size=8,
            dev=torch.device("cpu"),
            in_memory=True,
        )
        assert "optionA_oof_unseen" in res1
        assert "optionA_oof_ndcg" in res1
        assert "optionA_oof_discovery" in res1
        assert len(res1["epoch_history"]) == 2
        assert (out_dir / "best_unseen.pt").is_file()
        assert (out_dir / "predictions_optionA_oof_discovery.npz").is_file()
        assert (out_dir / "val_logits_ndcg.npy").is_file()
        assert (out_dir / "val_unseen_logits_ndcg.npy").is_file()

        # Verify checkpoints carry OOF blended selection metrics matching replayed optionA_oof_* results
        for sel in ("ndcg", "discovery", "unseen"):
            ck_sel = torch.load(out_dir / f"best_{sel}.pt", map_location="cpu", weights_only=False)
            assert ck_sel["oof_metrics"] == res1[f"optionA_oof_{sel}"]["bn_metrics"]

        # Resume check: second call must hit verified cache immediately
        res2 = train_and_eval_wave2c_arm(
            arm_cfg,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand38_mm=tr_c38,
            va_cand38_mm=va_c38,
            tr_extra5_mm=tr_e5,
            va_extra5_mm=va_e5,
            extra5_identity="test_e5",
            dataset_identity="test_ds",
            dev_purchases=purchases,
            catalogue=catalogue,
            out_dir=out_dir,
            seed=42,
            epochs=2,
            batch_size=8,
            dev=torch.device("cpu"),
            in_memory=True,
        )
        assert res2["output_sha256"] == res1["output_sha256"]
        assert res2["training_seconds"] == res1["training_seconds"]


def test_affine_calibration_handles_collapsed_unseen_main_logits():
    """Verify OOF affine blend stays invariant when repeat-only Main Head drives s_main[unseen] to -60."""
    rng = np.random.default_rng(7)
    n_rows, n_c = 120, 30
    allowed = np.ones((n_rows, n_c), dtype=bool)
    unseen = np.ones((n_rows, n_c), dtype=bool)
    unseen[:, :8] = False
    target = np.zeros((n_rows, n_c), dtype=bool)
    target[:, 0] = True
    target[:60, 12] = True
    new_target = target & unseen

    base_main = rng.normal(-2.0, 0.8, size=(n_rows, n_c))
    base_main[:, 0] += 4.0
    s_unseen = rng.normal(-5.0, 0.8, size=(n_rows, n_c))
    s_unseen[:60, 12] += 6.5
    cids = [f"CUST_{i % 20}" for i in range(n_rows)]

    s_main_mild = base_main.copy()
    s_main_mild[unseen] = -6.0
    s_main_collapsed = base_main.copy()
    s_main_collapsed[unseen] = -60.0

    oof1, meta1 = fit_oof_affine_unseen_blend(
        torch.from_numpy(s_main_mild),
        torch.from_numpy(s_unseen),
        allowed_bn=torch.from_numpy(allowed),
        unseen_bn=torch.from_numpy(unseen),
        target_bn=torch.from_numpy(target),
        new_target_bn=torch.from_numpy(new_target),
        customer_ids=cids,
        n_folds=2,
    )
    oof2, meta2 = fit_oof_affine_unseen_blend(
        torch.from_numpy(s_main_collapsed),
        torch.from_numpy(s_unseen),
        allowed_bn=torch.from_numpy(allowed),
        unseen_bn=torch.from_numpy(unseen),
        target_bn=torch.from_numpy(target),
        new_target_bn=torch.from_numpy(new_target),
        customer_ids=cids,
        n_folds=2,
    )
    assert meta2["deployment_full_fit"]["new_hits@10"] == 60
    assert meta2["deployment_full_fit"]["ndcg@10"] > 0.98
    assert abs(meta2["deployment_full_fit"]["ndcg@10"] - meta1["deployment_full_fit"]["ndcg@10"]) < 1e-6
    assert torch.allclose(oof1, oof2, atol=1e-5)


def test_aligned_commercial_events_cross_dtype_ms_vs_ns():
    """Guard against pandas 2.2 (Colab) parquet datetime64[ms]+string vs CSV datetime64[ns]+object mismatch."""
    from recommender.experiments.roadmap_v2.dated_baskets import aligned_commercial_events

    purchases = pd.DataFrame(
        {
            "customer_id": pd.Series(["C1", "C2"], dtype=object),
            "purchase_date": pd.to_datetime(["2026-01-10", "2026-01-12"]).astype("datetime64[ns]"),
            "product_id": pd.Series(["P0", "P1"], dtype=object),
            "quantity": [5.0, 3.0],
        }
    )
    events = pd.DataFrame(
        {
            "transaction_type": ["ПРОДАЖА", "ПРОДАЖА", "ПРОДАЖА"],
            "item_type": ["ТОВАР", "ТОВАР", "ТОВАР"],
            "customer_id": pd.Series(["C1", "C1", "C2"], dtype="string"),
            "purchase_date": pd.to_datetime(["2026-01-10", "2026-01-10", "2026-01-12"]).astype("datetime64[ms]"),
            "product_id": pd.Series(["P0", "P0", "P1"], dtype="string"),
            "quantity": [2.0, 3.0, 3.0],
            "full_payment_date": pd.to_datetime(["2026-01-11", "2026-01-11", "2026-01-13"]).astype("datetime64[ms]"),
        }
    )
    out = aligned_commercial_events(events, purchases, test_start=TEST_START_V2)
    assert len(out) == 3
    assert str(out["purchase_date"].dtype) == "datetime64[ns]"
