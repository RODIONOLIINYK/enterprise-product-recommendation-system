"""Unit tests for `v2_full` Wave 2A E2E controlled experiments (`cand_42`, capacity, history length, recency).

Strictly synthetic / leak-free unit tests verifying:
  1. Exact backward compatibility of `SharedUVStage5Ranker` and `EndToEndStage2Ranker` defaults.
  2. Depth (`memory_layers=2`, 4-block MLP), width (`d_model=192, 256`), and longer history
     (`max_seq_len=64, 96`) forward/backward gradient flow, bf16 autocast stability, live-vs-cached
     `v_bn` parity, stripped-aux inference load parity, and `seq_len > max_seq_len` guard.
  3. `WeightedRankerLoss` exact algebraic parity with `RankerLoss` (when unweighted / all-ones),
     weighted directional shift, and NaN-free handling of zero-weight, zero-positive, and masked rows.
  4. `compute_recency_sample_weights` monotonicity, exact half-life decay ratio, floor clipping,
     mean-1.0 normalization, and strict leak guard rejecting `prediction_days >= TRAIN_CUTOFF_V2`.
  5. `ensure_v2_sequences_for_length` strict causality (`< pred_day`), zeroed calendar month slots,
     retention of >32 events up to 64/96, and exact parity with `seq_len=32` on the rightmost 32 events.
  6. Strictly E2E `cand_42 + CUST_7 + SEQ_5` Wave 2A arm specs, training, replay, and resume.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.catalogue import RollingCandidatePool
from recommender.experiments.roadmap_v2.clean_ranker import SharedUVStage5Ranker
from recommender.experiments.roadmap_v2.dated_baskets import first_observed_pack_indices
from recommender.experiments.roadmap_v2.stage5_base_entity import build_base_name_universe
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.experiments.v2_full.config import TEST_START_V2, TRAIN_CUTOFF_V2
from recommender.experiments.v2_full.e2e_ranker import EndToEndStage2Ranker, TwoStageAdaptiveRanker
from recommender.experiments.v2_full.run_v2_full_wave2 import (
    ACTIVE_E2E_BASE_CONFIG,
    FINAL_2MEM_ARM_NAMES,
    SIZE_SCALING_ARM_NAMES,
    build_wave2_arm_specs,
)
from recommender.experiments.v2_full.wave2_utils import (
    WeightedRankerLoss,
    compute_recency_sample_weights,
    ensure_v2_sequences_for_length,
)
from recommender.ranker import RankerLoss


def _make_synthetic_batch(
    *,
    b: int = 3,
    c: int = 12,
    n_tr_c: int = 9,
    s: int = 64,
    d_model: int = 128,
):
    attr_vocabs = {
        "category": 4,
        "subcategory": 5,
        "business_line": 3,
        "tier": 3,
        "unit": 3,
        "country_of_origin": 3,
    }
    prod_inputs = {
        "text_embedding": torch.randn(c, 1024),
        "demand_features": torch.rand(c, 65),
        "attr_ids": {k: torch.randint(0, sz, (c,)) for k, sz in attr_vocabs.items()},
    }
    attn_mask = torch.zeros(b, s, dtype=torch.long)
    attn_mask[0, : min(s, 45)] = 1
    attn_mask[1, : min(s, 5)] = 1
    # row 2 = first-observed customer (zero history)
    raw_days = torch.zeros(b, s, dtype=torch.float32)
    for i in range(b):
        n_real = int(attn_mask[i].sum().item())
        if n_real > 0:
            raw_days[i, :n_real] = torch.linspace(10.0, 290.0, n_real)
    seq_pids = torch.randint(1, n_tr_c + 1, (b, s), dtype=torch.long) * attn_mask
    cmask = torch.zeros(b, c, dtype=torch.bool)
    cmask[:, :n_tr_c] = True
    cand_feats = torch.rand(b, c, 38)
    cand_feats[:, :4, 0] = 1.0
    cand_feats[:, 4:, 0] = 0.0
    cand_feats[:, :, 3] = math.log1p(45.0) / math.log1p(365.0)

    common = {
        "customer_features": torch.rand(b, 11),
        "sequence": torch.rand(b, s, 9),
        "attention_mask": attn_mask,
        "seq_product_ids": seq_pids,
        "seq_purchase_dates": raw_days / 365.25,
        "prediction_day": torch.full((b,), 310.0),
        "candidate_features": cand_feats,
        "candidate_mask": cmask,
    }
    e2e_batch = {
        **common,
        "company_text_embedding": torch.randn(b, 1024),
        "city_id": torch.randint(0, 5, (b,)),
        "region_id": torch.randint(0, 4, (b,)),
        "industry_id": torch.randint(0, 4, (b,)),
    }
    two_stage_batch = {
        **common,
        "customer_vector": torch.randn(b, d_model),
    }
    return attr_vocabs, prod_inputs, e2e_batch, two_stage_batch, cmask


def test_shared_uv_and_e2e_depth_width_and_long_seq_scaling():
    torch.manual_seed(0)

    # 1. Default SharedUVStage5Ranker keeps exact 1-layer memory + 32-slot pos_embedding
    base_default = SharedUVStage5Ranker(aux_60d_enabled=False, seed=42)
    assert len(base_default.memory_encoder.layers) == 1
    assert base_default.pos_embedding.shape == (1, 32, 128)
    assert len(base_default.mlp_blocks) == 3

    # 2. Scaled EndToEndStage2Ranker (d=192, 2-layer memory, 4-block MLP, seq=64)
    attr_vocabs, prod_inputs, e2e_batch, _, cmask = _make_synthetic_batch(b=3, c=12, n_tr_c=9, s=64, d_model=192)
    model = EndToEndStage2Ranker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        d_model=192,
        num_heads=8,
        memory_ffn_dim=576,
        memory_layers=2,
        max_seq_len=64,
        mlp_widths=(512, 384, 192, 96),
        aux_60d_enabled=True,
        unseen_attn_aux_enabled=True,
        seed=7,
    )
    assert len(model.memory_encoder.layers) == 2
    assert len(model.mlp_blocks) == 4
    decay_by_id = {id(p): group["weight_decay"] for group in model.optimizer_param_groups(0.01)
                   for p in group["params"]}
    assert decay_by_id[id(model.statistics_attention_weights)] == 0.01
    assert model.pos_embedding.shape == (1, 64, 192)

    # Forward + backward (including CPU bf16 autocast check matching Colab --amp)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        res = model(**e2e_batch, prod_inputs=prod_inputs, train_bn_limit=9, return_aux=True)
    y = torch.zeros(3, 12)
    y[:, 0] = 1.0
    y[:, 5] = 1.0
    loss_fn = WeightedRankerLoss()
    l_main = loss_fn(res["output"].logits.float(), y, candidate_mask=cmask)["loss"]
    l_aux = loss_fn(res["aux_60d_logits"].float(), y, candidate_mask=cmask)["loss"]
    l_uattn = loss_fn(res["unseen_attn_logits"].float(), y, candidate_mask=cmask)["loss"]
    (l_main + 0.3 * l_aux + 0.25 * l_uattn).backward()

    for pname in (
        "product_tower.text_proj.0.weight",
        "memory_encoder.layers.0.linear1.weight",
        "memory_encoder.layers.1.linear1.weight",
        "mlp_blocks.3.0.weight",
        "statistics_attention_weights",
        "unseen_attn_head.0.weight",
    ):
        p = dict(model.named_parameters())[pname]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0.0

    # Live vs cached v_bn parity + stripped inference load parity
    model.eval()
    eval_batch = {**e2e_batch, "candidate_mask": torch.ones(3, 12, dtype=torch.bool)}
    with torch.no_grad():
        out_live = model(**eval_batch, prod_inputs=prod_inputs, return_aux=False)
        v_cached = model.precompute_product_vectors(prod_inputs)
        assert v_cached.shape == (12, 192)
        out_cached = model(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
        assert torch.allclose(out_live.logits, out_cached.logits, atol=1e-5)

    inf_model = EndToEndStage2Ranker(
        product_attr_vocab_sizes=attr_vocabs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        d_model=192,
        num_heads=8,
        memory_ffn_dim=576,
        memory_layers=2,
        max_seq_len=64,
        mlp_widths=(512, 384, 192, 96),
        aux_60d_enabled=False,
        unseen_attn_aux_enabled=False,
        seed=7,
    ).eval()
    stripped = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("aux_60d_head.") and not k.startswith("unseen_attn_head.")
    }
    inf_model.load_state_dict(stripped, strict=True)
    with torch.no_grad():
        out_inf = inf_model(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
        assert torch.allclose(out_live.logits, out_inf.logits, atol=1e-5)

    # Guard: sequence longer than max_seq_len must raise ValueError
    _, _, long_batch, _, _ = _make_synthetic_batch(b=2, c=12, n_tr_c=9, s=70, d_model=192)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        inf_model(**long_batch, precomputed_v_bn=v_cached, return_aux=False)

    # 3. Scaled 2-stage SharedUVStage5Ranker (d=256, 2-layer memory, 4-block MLP, seq=96)
    _, _, _, ts_batch, ts_mask = _make_synthetic_batch(b=3, c=12, n_tr_c=9, s=96, d_model=256)
    s2 = SharedUVStage5Ranker(
        d_model=256,
        num_heads=8,
        memory_ffn_dim=768,
        memory_layers=2,
        max_seq_len=96,
        mlp_widths=(768, 512, 256, 128),
        aux_60d_enabled=True,
        seed=3,
    )
    wrapper = TwoStageAdaptiveRanker(ranker=s2, product_tower=None, d_model=256, unseen_attn_aux_enabled=True)
    v_256 = torch.randn(12, 256)
    res2 = wrapper(**ts_batch, precomputed_v_bn=v_256, return_aux=True)
    assert res2["output"].logits.shape == (3, 12)
    assert res2["unseen_attn_logits"].shape == (3, 12)
    l2 = loss_fn(res2["output"].logits, y, candidate_mask=ts_mask)["loss"]
    l2.backward()
    assert s2.memory_encoder.layers[1].linear1.weight.grad.abs().sum().item() > 0.0
    assert s2.mlp_blocks[3][0].weight.grad.abs().sum().item() > 0.0


def test_weighted_ranker_loss_parity_and_edge_cases():
    torch.manual_seed(11)
    ref = RankerLoss(pos_weight=10.0, listwise_weight=1.0)
    wloss = WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0)

    # Full-valid & partially-masked parity with RankerLoss when sample_weight is None or all-ones
    for all_valid in (True, False):
        logits = torch.randn(5, 16, requires_grad=True)
        cmask = torch.ones(5, 16, dtype=torch.bool)
        if not all_valid:
            cmask[:, 11:] = False
            cmask[4, :] = False  # completely empty row
        targets = torch.zeros(5, 16)
        targets[0, [1, 4]] = 1.0
        targets[1, [2]] = 1.0
        targets[2, [0, 3, 7]] = 1.0
        # row 3 has valid candidates but zero positives

        r_ref = ref(logits, targets, candidate_mask=cmask)
        r_none = wloss(logits, targets, candidate_mask=cmask, sample_weight=None)
        r_ones = wloss(logits, targets, candidate_mask=cmask, sample_weight=torch.ones(5))
        for k in ("loss", "bce_loss", "list_loss"):
            assert torch.allclose(r_ref[k], r_none[k], atol=1e-6), f"mismatch on {k} (all_valid={all_valid})"
            assert torch.allclose(r_ref[k], r_ones[k], atol=1e-6), f"ones mismatch on {k} (all_valid={all_valid})"

    # Non-uniform weights isolate row 0 when row 0 has weight=1 and others weight=0
    logits = torch.randn(3, 10, requires_grad=True)
    targets = torch.zeros(3, 10)
    targets[0, 1] = 1.0
    targets[1, 2] = 1.0
    cmask = torch.ones(3, 10, dtype=torch.bool)
    r_w = wloss(logits, targets, candidate_mask=cmask, sample_weight=torch.tensor([1.0, 0.0, 0.0]))
    r_row0 = ref(logits[:1], targets[:1], candidate_mask=cmask[:1])
    assert torch.allclose(r_w["loss"], r_row0["loss"], atol=1e-6)
    r_w["loss"].backward()
    assert torch.allclose(logits.grad[1:], torch.zeros_like(logits.grad[1:]), atol=1e-7)
    assert logits.grad[0].abs().sum().item() > 0.0

    # Zero-weight batch stays finite with zero grad
    logits2 = torch.randn(2, 6, requires_grad=True)
    r_zero = wloss(logits2, torch.zeros(2, 6), sample_weight=torch.zeros(2))
    assert float(r_zero["loss"].item()) == 0.0
    r_zero["loss"].backward()
    assert torch.isfinite(logits2.grad).all()

    # Invalid negative weight raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        wloss(torch.randn(2, 6), torch.zeros(2, 6), sample_weight=torch.tensor([1.0, -0.5]))


def test_recency_sample_weights_monotonic_and_leak_guard():
    cutoff_day = float((TRAIN_CUTOFF_V2 - PURCHASE_DATE_ORIGIN).days)
    # 0 days before cutoff (cutoff-1), 365 days before cutoff, 730 days before cutoff, 2000 days before cutoff (floor)
    days = torch.tensor([cutoff_day - 1.0, cutoff_day - 366.0, cutoff_day - 731.0, cutoff_day - 2001.0])
    w = compute_recency_sample_weights(days, train_cutoff=TRAIN_CUTOFF_V2, half_life_days=365.0, min_weight=0.1)

    assert w.shape == (4,)
    assert abs(float(w.mean().item()) - 1.0) < 1e-6
    assert w[0] > w[1] > w[2] > w[3]
    # Exact 2x ratio across 1 half-life (365 days)
    assert abs(float((w[0] / w[1]).item()) - 2.0) < 1e-5
    assert abs(float((w[1] / w[2]).item()) - 2.0) < 1e-5

    # Leak guard: any prediction_day >= train_cutoff must fail loudly
    with pytest.raises(ValueError, match="train_cutoff"):
        compute_recency_sample_weights(torch.tensor([cutoff_day - 5.0, cutoff_day]), train_cutoff=TRAIN_CUTOFF_V2)


def test_longer_history_sequences_prefix_parity_and_causality():
    # Build synthetic 3-customer dataset spanning 2023..2026 (< TEST_START_V2):
    # - C_HEAVY: 45 distinct daily events prior to 2025-12-01 (> 32, <= 64)
    # - C_LIGHT: 5 distinct daily events prior to 2025-12-01 (<= 32)
    # - C_NEW:   0 prior events (only buys on 2026-03-15 in val)
    rows = []
    pids = ["P1", "P2", "P3"]
    base_d = pd.Timestamp("2024-01-01")
    for i in range(45):
        rows.append(
            {
                "customer_id": "C_HEAVY",
                "purchase_date": base_d + pd.Timedelta(days=10 * i),
                "product_id": pids[i % 3],
                "quantity": 300.0 if i == 0 else float(1 + (i % 5)),
            }
        )
    for i in range(5):
        rows.append(
            {
                "customer_id": "C_LIGHT",
                "purchase_date": base_d + pd.Timedelta(days=20 * i),
                "product_id": pids[i % 3],
                "quantity": 2.0,
            }
        )
    # Add val-date purchases (strictly < TEST_START_V2) to test causality at train pred_day
    rows.append({"customer_id": "C_HEAVY", "purchase_date": pd.Timestamp("2026-03-10"), "product_id": "P1", "quantity": 9.0})
    rows.append({"customer_id": "C_NEW", "purchase_date": pd.Timestamp("2026-03-15"), "product_id": "P2", "quantity": 3.0})
    dev_purchases = pd.DataFrame(rows)
    dev_purchases["item_type"] = "ТОВАР"
    dev_purchases["transaction_type"] = "ПРОДАЖА"

    catalogue = pd.DataFrame(
        {
            "product_id": pids,
            "product_name": ["Oil A 4L", "Oil B 4L", "Oil C 4L"],
            "base_name": ["oil a", "oil b", "oil c"],
            "product_category": ["CAT1", "CAT1", "CAT2"],
            "product_subcategory": ["SUB1", "SUB2", "SUB2"],
            "business_line": ["BL1", "BL1", "BL2"],
            "item_type": ["ТОВАР", "ТОВАР", "ТОВАР"],
            "is_blocked": [False, False, False],
        }
    )
    universe = build_base_name_universe(pids, catalogue, dev_purchases, train_cutoff=TRAIN_CUTOFF_V2)
    pool = RollingCandidatePool(catalogue, dev_purchases)
    reps = first_observed_pack_indices(pids, universe["bn_to_member_pid_indices"], pool.first_sales)
    shared_meta = {
        "product_ids": pids,
        "reps": reps,
        "universe": universe,
        "vocabs": {
            "category_vocab": {"CAT1": 1, "CAT2": 2},
            "subcategory_vocab": {"SUB1": 1, "SUB2": 2},
            "business_line_vocab": {"BL1": 1, "BL2": 2},
        },
    }

    tr_day = float((pd.Timestamp("2025-12-01") - PURCHASE_DATE_ORIGIN).days)
    va_day = float((pd.Timestamp("2026-03-15") - PURCHASE_DATE_ORIGIN).days)
    n_bn = int(universe["num_base_names"])
    tr_stub = {
        "customer_ids": ["C_HEAVY", "C_LIGHT"],
        "prediction_days": torch.tensor([tr_day, tr_day], dtype=torch.float32),
        "first_observed": torch.tensor([False, False], dtype=torch.bool),
        "familiarity_bn": torch.zeros((2, n_bn, 4), dtype=torch.float32),
        "target_bn": torch.tensor([[True, False, False], [False, True, False]]),
        "new_target_bn": torch.zeros((2, n_bn), dtype=torch.bool),
        "allowed_bn": torch.ones((2, n_bn), dtype=torch.bool),
        "unseen_bn": torch.zeros((2, n_bn), dtype=torch.bool),
        "attention_mask": torch.zeros((2, 1), dtype=torch.long),
    }
    va_stub = {
        "customer_ids": ["C_HEAVY", "C_NEW"],
        "prediction_days": torch.tensor([va_day, va_day], dtype=torch.float32),
        "first_observed": torch.tensor([False, True], dtype=torch.bool),
        "familiarity_bn": torch.zeros((2, n_bn, 4), dtype=torch.float32),
        "target_bn": torch.tensor([[True, False, False], [False, True, False]]),
        "new_target_bn": torch.tensor([[False, False, False], [False, True, False]]),
        "allowed_bn": torch.ones((2, n_bn), dtype=torch.bool),
        "unseen_bn": torch.tensor([[False, False, False], [True, True, True]]),
        "attention_mask": torch.zeros((2, 1), dtype=torch.long),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tr32, va32 = ensure_v2_sequences_for_length(
            32,
            shared_meta=shared_meta,
            tr_meta=tr_stub,
            va_meta=va_stub,
            dev_purchases=dev_purchases,
            catalogue=catalogue,
            cache_dir=Path(tmp),
        )
        tr64, va64 = ensure_v2_sequences_for_length(
            64,
            shared_meta=shared_meta,
            tr_meta=tr32,
            va_meta=va32,
            dev_purchases=dev_purchases,
            catalogue=catalogue,
            cache_dir=Path(tmp),
        )
        # C_HEAVY has all 45 prior events in seq_len=64 vs truncated 32 in seq_len=32
        assert int(tr32["attention_mask"][0].sum().item()) == 32
        assert int(tr64["attention_mask"][0].sum().item()) == 45
        # Rightmost 32 events of C_HEAVY (slots 13..44) match seq_len=32 identically on IDs & dates!
        assert torch.equal(tr64["seq_product_ids"][0, 13:45], tr32["seq_product_ids"][0, :32])
        assert torch.allclose(tr64["seq_purchase_dates"][0, 13:45], tr32["seq_purchase_dates"][0, :32])
        for key in ("sequence", "seq_rel_qty", "attention_mask", "seq_category_ids",
                    "seq_subcategory_ids", "seq_business_line_ids"):
            assert torch.equal(tr64[key][0, 13:45], tr32[key][0])
            assert torch.equal(tr64[key][1, :32], tr32[key][1])
        # C_LIGHT (5 events <= 32) matches seq_len=32 identically on [:32] and is 0 on [32:]
        assert torch.equal(tr64["seq_product_ids"][1, :32], tr32["seq_product_ids"][1, :32])
        assert (tr64["attention_mask"][1, 32:] == 0).all()
        # Calendar month columns (3:5) are strictly zeroed
        assert (tr64["sequence"][..., 3:5] == 0.0).all()
        assert (va64["sequence"][..., 3:5] == 0.0).all()
        # Val C_HEAVY sees 45 + 1 = 46 events (including 2026-03-10 < 2026-03-15); C_NEW sees 0 (< 2026-03-15)
        assert int(va64["attention_mask"][0].sum().item()) == 46
        assert int(va64["attention_mask"][1].sum().item()) == 0

        # Held-out test event leak must raise ValueError
        leaked = pd.concat(
            [
                dev_purchases,
                pd.DataFrame(
                    [
                        {
                            "customer_id": "C_NEW",
                            "purchase_date": TEST_START_V2,
                            "product_id": "P1",
                            "quantity": 1.0,
                            "item_type": "ТОВАР",
                            "transaction_type": "ПРОДАЖА",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        for length in (32, 64, 96):
            with pytest.raises(ValueError, match="Held-out test event"):
                ensure_v2_sequences_for_length(
                    length,
                    shared_meta=shared_meta,
                    tr_meta=tr32,
                    va_meta=va32,
                    dev_purchases=leaked,
                    catalogue=catalogue,
                    cache_dir=Path(tmp),
                )


def test_wave2_arm_specs_strictly_e2e_cand42():
    e2e_specs = build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG)
    assert [s["name"] for s in e2e_specs] == [
        "w2_2mem_blocks",
        "w2_wider_mlp",
        "w2_2mem_wider_mlp",
        "w2_wider_towers",
        "w2_dim192",
        "w2_dim192_2mem_wider_mlp",
        "w2_dim256",
        "w2_dim256_2mem_4mlp",
        "w2_hist_len64",
        "w2_recency_hl365",
        "w2_2mem_wider_mlp_syn7030",
        "w2_2mem_hist64",
        "w2_2mem_recency_hl365",
        "w2_2mem_hist64_recency_hl365",
    ]
    assert list(SIZE_SCALING_ARM_NAMES) == [
        "w2_2mem_blocks",
        "w2_wider_mlp",
        "w2_2mem_wider_mlp",
        "w2_wider_towers",
        "w2_dim192",
        "w2_dim192_2mem_wider_mlp",
        "w2_dim256",
        "w2_dim256_2mem_4mlp",
    ]
    assert list(FINAL_2MEM_ARM_NAMES) == [
        "w2_2mem_hist64",
        "w2_2mem_recency_hl365",
        "w2_2mem_hist64_recency_hl365",
    ]
    by_name = {s["name"]: s for s in e2e_specs}
    # Model-size scaling ladder
    assert (by_name["w2_2mem_blocks"]["d_model"], by_name["w2_2mem_blocks"]["memory_layers"],
            by_name["w2_2mem_blocks"]["memory_ffn_dim"], by_name["w2_2mem_blocks"]["mlp_widths"],
            by_name["w2_2mem_blocks"]["synthetic_unseen_aug"]) == (128, 2, 384, (384, 192, 96), False)
    assert (by_name["w2_wider_mlp"]["d_model"], by_name["w2_wider_mlp"]["memory_layers"],
            by_name["w2_wider_mlp"]["memory_ffn_dim"], by_name["w2_wider_mlp"]["mlp_widths"],
            by_name["w2_wider_mlp"]["synthetic_unseen_aug"]) == (128, 1, 512, (512, 256, 128), False)
    assert (by_name["w2_2mem_wider_mlp"]["d_model"], by_name["w2_2mem_wider_mlp"]["memory_layers"],
            by_name["w2_2mem_wider_mlp"]["memory_ffn_dim"], by_name["w2_2mem_wider_mlp"]["mlp_widths"],
            by_name["w2_2mem_wider_mlp"]["synthetic_unseen_aug"]) == (128, 2, 512, (512, 256, 128), False)
    assert (
        by_name["w2_wider_towers"]["d_model"],
        by_name["w2_wider_towers"]["memory_layers"],
        by_name["w2_wider_towers"]["product_tower_hidden_dim"],
        by_name["w2_wider_towers"]["cust_tower_hidden_dim"],
        by_name["w2_wider_towers"]["attr_emb_dim"],
        by_name["w2_wider_towers"]["mlp_widths"],
    ) == (128, 1, 256, 256, 32, (384, 192, 96))
    assert (by_name["w2_dim192"]["d_model"], by_name["w2_dim192"]["num_heads"],
            by_name["w2_dim192"]["memory_layers"], by_name["w2_dim192"]["memory_ffn_dim"],
            by_name["w2_dim192"]["mlp_widths"], by_name["w2_dim192"]["synthetic_unseen_aug"]) == (
        192, 8, 1, 576, (384, 192, 96), False
    )
    assert (
        by_name["w2_dim192_2mem_wider_mlp"]["d_model"],
        by_name["w2_dim192_2mem_wider_mlp"]["num_heads"],
        by_name["w2_dim192_2mem_wider_mlp"]["memory_layers"],
        by_name["w2_dim192_2mem_wider_mlp"]["memory_ffn_dim"],
        by_name["w2_dim192_2mem_wider_mlp"]["mlp_widths"],
        by_name["w2_dim192_2mem_wider_mlp"]["attr_emb_dim"],
    ) == (192, 8, 2, 576, (512, 256, 128), 24)
    assert (
        by_name["w2_dim256"]["d_model"],
        by_name["w2_dim256"]["num_heads"],
        by_name["w2_dim256"]["memory_layers"],
        by_name["w2_dim256"]["memory_ffn_dim"],
        by_name["w2_dim256"]["mlp_widths"],
        by_name["w2_dim256"]["attr_emb_dim"],
    ) == (256, 8, 1, 768, (512, 256, 128), 32)
    assert (
        by_name["w2_dim256_2mem_4mlp"]["d_model"],
        by_name["w2_dim256_2mem_4mlp"]["num_heads"],
        by_name["w2_dim256_2mem_4mlp"]["memory_layers"],
        by_name["w2_dim256_2mem_4mlp"]["memory_ffn_dim"],
        by_name["w2_dim256_2mem_4mlp"]["mlp_widths"],
        by_name["w2_dim256_2mem_4mlp"]["product_tower_hidden_dim"],
        by_name["w2_dim256_2mem_4mlp"]["cust_tower_hidden_dim"],
        by_name["w2_dim256_2mem_4mlp"]["attr_emb_dim"],
    ) == (256, 8, 2, 1024, (768, 512, 256, 128), 384, 384, 32)
    assert by_name["w2_hist_len64"]["seq_len"] == 64
    assert by_name["w2_recency_hl365"]["recency_half_life_days"] == 365.0
    # Separate bigger-model synthetic unseen augmentation experiment
    assert (by_name["w2_2mem_wider_mlp_syn7030"]["d_model"], by_name["w2_2mem_wider_mlp_syn7030"]["memory_layers"],
            by_name["w2_2mem_wider_mlp_syn7030"]["memory_ffn_dim"], by_name["w2_2mem_wider_mlp_syn7030"]["mlp_widths"],
            by_name["w2_2mem_wider_mlp_syn7030"]["synthetic_unseen_aug"]) == (128, 2, 512, (512, 256, 128), True)
    # Final 2-memory-block ablation suite (seq_len=64, half_life=365d, combined)
    assert (by_name["w2_2mem_hist64"]["memory_layers"], by_name["w2_2mem_hist64"]["seq_len"],
            by_name["w2_2mem_hist64"]["recency_half_life_days"], by_name["w2_2mem_hist64"]["mlp_widths"]) == (
        2, 64, None, (384, 192, 96)
    )
    assert (by_name["w2_2mem_recency_hl365"]["memory_layers"], by_name["w2_2mem_recency_hl365"]["seq_len"],
            by_name["w2_2mem_recency_hl365"]["recency_half_life_days"], by_name["w2_2mem_recency_hl365"]["mlp_widths"]) == (
        2, 32, 365.0, (384, 192, 96)
    )
    assert (
        by_name["w2_2mem_hist64_recency_hl365"]["memory_layers"],
        by_name["w2_2mem_hist64_recency_hl365"]["seq_len"],
        by_name["w2_2mem_hist64_recency_hl365"]["recency_half_life_days"],
        by_name["w2_2mem_hist64_recency_hl365"]["mlp_widths"],
    ) == (2, 64, 365.0, (384, 192, 96))

    assert all(
        s["num_candidate_features"] == 42
        and s["num_customer_features"] == 7
        and s["active_seq_stat_indices"] == [0, 1, 5, 6, 8]
        and s["pure_v_query"] is True
        and s["num_pairwise_bias_features"] == 42
        and s["option_a_unseen_head"] is True
        and s["unseen_dropout_cold_prob"] == 0.0
        and s["separated_main_targets"] is True
        and s["unseen_head_weight"] == 1.0
        and s["unseen_attn_aux_enabled"] is False
        and s["mode"] == "e2e"
        for s in e2e_specs
    )
    e2e_with_hist96 = build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG, include_hist96=True)
    assert [s["name"] for s in e2e_with_hist96] == [
        "w2_2mem_blocks",
        "w2_wider_mlp",
        "w2_2mem_wider_mlp",
        "w2_wider_towers",
        "w2_dim192",
        "w2_dim192_2mem_wider_mlp",
        "w2_dim256",
        "w2_dim256_2mem_4mlp",
        "w2_hist_len64",
        "w2_recency_hl365",
        "w2_2mem_wider_mlp_syn7030",
        "w2_2mem_hist64",
        "w2_2mem_recency_hl365",
        "w2_2mem_hist64_recency_hl365",
        "w2_hist_len96",
    ]
    with pytest.raises(ValueError, match="exclusively on E2E"):
        build_wave2_arm_specs({"mode": "two_stage"})


def test_short_eligible_rankings():
    from recommender.experiments.v2_full.train_v2_full_baseline import rank_eligible_formulations

    scores = torch.tensor([[4., 2., 2.], [2., 1., 0.]])
    allowed = torch.tensor([[False, True, True], [False, False, False]])
    assert rank_eligible_formulations(scores, allowed).tolist() == [[1, 2, -1], [-1, -1, -1]]


def test_wave2_e2e_arm_trains_replays_and_resumes(tmp_path, monkeypatch):
    from recommender.experiments.v2_full import run_v2_full_wave2 as runner

    torch.manual_seed(17)
    attrs, products, batch, _, allowed = _make_synthetic_batch(s=4)
    shared = {
        "product_ids": [f"P{i}" for i in range(12)],
        "universe": {"trained_base_names": 9, "num_base_names": 12, "bn_to_member_pid_indices": [[i] for i in range(12)],
                     "all_pid_to_bn": {f"P{i}": f"B{i}" for i in range(12)},
                     "bn_to_idx": {f"B{i}": i for i in range(12)}},
        "prod_inputs": products,
        "vocabs": {k: dict.fromkeys(range(5)) for k in ("city_vocab", "region_vocab", "industry_vocab")},
        "enrichment": {"product_vocabs": {k: dict.fromkeys(range(n)) for k, n in attrs.items()}},
    }
    target = torch.zeros_like(allowed)
    target[:, 4] = True
    first_obs = torch.tensor([False, False, True])
    unseen = allowed & (batch["candidate_features"][..., 0] == 0)
    unseen[first_obs] = allowed[first_obs]
    tr_day = float((pd.Timestamp("2025-12-01") - PURCHASE_DATE_ORIGIN).days)
    va_day = float((pd.Timestamp("2026-03-15") - PURCHASE_DATE_ORIGIN).days)
    meta = {
        **batch, "customer_ids": ["A", "B", "C"], "allowed_bn": allowed,
        "target_bn": target, "new_target_bn": target & unseen, "aux_60d_bn": target,
        "unseen_bn": unseen, "familiarity_bn": torch.zeros((3, 12, 4), dtype=torch.float32),
        "first_observed": first_obs, "full_window": torch.tensor([True, False, True]),
        "text_embedding": batch["company_text_embedding"], "tabular_features": batch["customer_features"],
        "prediction_days": torch.tensor([tr_day - 10.0, tr_day - 5.0, tr_day]),
    }
    val = {**meta, "prediction_days": torch.tensor([va_day - 10.0, va_day - 5.0, va_day]),
           "sku_val": {"target": target, "new_target": target & unseen, "allowed": allowed}}
    specs_by_name = {s["name"]: s for s in build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG)}
    spec = specs_by_name["w2_recency_hl365"]
    assert spec["num_candidate_features"] == 42
    assert spec["num_customer_features"] == 7
    assert spec["active_seq_stat_indices"] == [0, 1, 5, 6, 8]
    assert spec["pure_v_query"] is True
    assert spec["num_pairwise_bias_features"] == 42
    assert spec["option_a_unseen_head"] is True
    extra5 = np.random.default_rng(19).random((3, 12, 5)).astype(np.float16)
    kwargs = dict(
        arm_cfg=spec, shared_meta=shared, tr_meta_arm=meta, va_meta_arm=val,
        tr_cand38_mm=batch["candidate_features"].numpy(), va_cand38_mm=batch["candidate_features"].numpy(),
        tr_extra5_mm=extra5, va_extra5_mm=extra5, extra5_identity="w2a_test_extra5",
        dev_purchases=pd.DataFrame(columns=["customer_id", "product_id", "purchase_date"]),
        catalogue=pd.DataFrame({"product_id": shared["product_ids"], "container_type": ["box"] * 12}),
        out_dir=tmp_path / "arm", seed=42, epochs=1, batch_size=3, dev=torch.device("cpu"),
    )
    result = runner.train_and_eval_wave2_arm(**kwargs)
    assert all(1 <= result[selection]["epoch"] <= kwargs["epochs"] for selection in ("ndcg", "discovery"))
    assert "optionA_oof_ndcg" in result and "optionA_oof_discovery" in result and "optionA_oof_unseen" in result
    assert (tmp_path / "arm" / "best_unseen.pt").is_file()
    assert (tmp_path / "arm" / "predictions_optionA_oof_discovery.npz").is_file()
    assert (tmp_path / "arm" / "val_logits_ndcg.npy").is_file()
    assert (tmp_path / "arm" / "val_unseen_logits_ndcg.npy").is_file()
    for sel in ("ndcg", "discovery", "unseen"):
        ck_sel = torch.load(tmp_path / "arm" / f"best_{sel}.pt", map_location="cpu", weights_only=False)
        assert ck_sel["oof_metrics"] == result[f"optionA_oof_{sel}"]["bn_metrics"]
    assert result["recency_weights"]["mean_norm_weight"] == pytest.approx(1.)
    assert result["epoch_history"][0]["option_a_loss"] > 0
    assert result["epoch_history"][0]["unseen_attn_loss"] == 0.0
    assert result["ndcg"]["cohorts"]["first_observed"]["queries"] == 1
    with np.load(result["optionA_oof_discovery"]["predictions_npz"]) as archive:
        assert np.array_equal(archive["prediction_days"], val["prediction_days"])

    # Also exercise the separate bigger-model synthetic unseen arm (`w2_2mem_wider_mlp_syn7030`) end-to-end
    syn_spec = specs_by_name["w2_2mem_wider_mlp_syn7030"]
    syn_meta = {
        "ret_source_rows": torch.tensor([0, 1], dtype=torch.long),
        "fo_source_rows": torch.tensor([2], dtype=torch.long),
        "ret_cust_7": torch.rand((2, 7), dtype=torch.float32),
        "ret_sequence": batch["sequence"][:2].clone(),
        "ret_attention_mask": batch["attention_mask"][:2].clone(),
        "ret_seq_product_ids": batch["seq_product_ids"][:2].clone(),
        "ret_seq_purchase_dates": batch["seq_purchase_dates"][:2].clone(),
        "ret_unseen_bn": unseen[:2].clone(),
        "ret_target_bn": target[:2].clone(),
    }
    syn_ret_cand27 = np.random.default_rng(23).random((2, 12, 27)).astype(np.float16)
    syn_kwargs = {
        **kwargs,
        "arm_cfg": syn_spec,
        "syn_meta": syn_meta,
        "syn_ret_cand27_mm": syn_ret_cand27,
        "syn_identity": "w2a_test_syn7030",
        "out_dir": tmp_path / "arm_syn",
    }
    syn_result = runner.train_and_eval_wave2_arm(**syn_kwargs)
    assert syn_result["epoch_history"][0]["option_a_loss"] > 0
    assert (tmp_path / "arm_syn" / "best_unseen.pt").is_file()

    monkeypatch.setattr(runner, "iter_paired_training_batches", lambda *a, **kw: pytest.fail("Unexpected retrain"))
    assert runner.train_and_eval_wave2_arm(**kwargs) == result
    assert runner.train_and_eval_wave2_arm(**syn_kwargs) == syn_result


def test_wave2_weighted_separated_losses_conflict_free_and_sample_weighted():
    from recommender.experiments.v2_full.run_v2_full_wave2 import compute_wave2_weighted_separated_losses

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
    first_obs = torch.tensor([False, False, True], dtype=torch.bool)
    full_win = torch.tensor([True, True, True], dtype=torch.bool)
    # Give row 0 zero recency weight -> row 0 must receive zero grad on BOTH Main and Option A!
    weights = torch.tensor([0.0, 2.0, 1.0])
    out = compute_wave2_weighted_separated_losses(
        main_logits=main_lg,
        main_aux_60d_logits=main_aux_lg,
        gen_unseen_logits=unseen_lg,
        gen_unseen_aux_60d_logits=unseen_aux_lg,
        allowed_bn=allowed,
        unseen_bn=unseen,
        target_bn=target,
        aux_60d_bn=target.clone(),
        full_window=full_win,
        first_observed=first_obs,
        main_loss_fn=WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0),
        main_aux_loss_fn=WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0),
        unseen_next_loss_fn=WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0),
        unseen_60d_loss_fn=WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0),
        unseen_head_weight=1.0,
        separated_main_targets=True,
        sample_weight=weights,
    )
    out["loss"].backward()
    # Row 0 has zero sample weight -> zero grad everywhere
    assert torch.all(main_lg.grad[0] == 0.0)
    assert torch.all(unseen_lg.grad[0] == 0.0)
    # Row 1 (0-repeat returning): BN1 unseen pos masked out of Main (zero grad), BN0 repeat neg pushed down (>0)
    assert float(main_lg.grad[1, 1].abs().item()) == 0.0
    assert float(main_lg.grad[1, 0].item()) > 0.0
    # Row 2 (FO): zero grad in Main, negative grad on BN2 unseen pos in Option A
    assert torch.all(main_lg.grad[2] == 0.0)
    assert float(unseen_lg.grad[2, 2].item()) < 0.0
    # Row 1 unseen positive receives 2x stronger Option A listwise weight than Row 2
    assert abs(float(unseen_lg.grad[1, 1].item())) > abs(float(unseen_lg.grad[2, 2].item()))


def test_wave2_scaled_optiona_pureq_full42_gradients():
    from recommender.experiments.v2_full.wave2b_models import Wave2BE2ERanker, strip_wave2b_training_aux

    torch.manual_seed(9)
    attrs, products, batch, _, allowed = _make_synthetic_batch(b=3, c=12, n_tr_c=9, s=64, d_model=192)
    model = Wave2BE2ERanker(
        product_attr_vocab_sizes=attrs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=42,
        pure_v_query=True,
        option_a_unseen_head=True,
        d_model=192,
        num_heads=8,
        memory_ffn_dim=576,
        memory_layers=2,
        max_seq_len=64,
        mlp_widths=(512, 384, 192, 96),
        aux_60d_enabled=True,
        seed=5,
    )
    cand42 = torch.rand(3, 12, 42)
    cust7 = torch.rand(3, 7)
    w_batch = {**batch, "candidate_features": cand42, "customer_features": cust7}
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        res = model(**w_batch, prod_inputs=products, train_bn_limit=9, return_aux=True)
    y = torch.zeros(3, 12)
    y[:, 0] = 1.0
    y[:, 5] = 1.0
    wloss = WeightedRankerLoss()
    weights = torch.tensor([1.5, 0.8, 0.7])
    loss = (
        wloss(res["output"].logits.float(), y, candidate_mask=allowed, sample_weight=weights)["loss"]
        + 0.3 * wloss(res["aux_60d_logits"].float(), y, candidate_mask=allowed, sample_weight=weights)["loss"]
        + 0.5 * wloss(res["unseen_logits"].float(), y, candidate_mask=allowed, sample_weight=weights)["loss"]
        + 0.15 * wloss(res["unseen_aux_60d_logits"].float(), y, candidate_mask=allowed, sample_weight=weights)["loss"]
    )
    loss.backward()
    for pname in (
        "memory_encoder.layers.1.linear1.weight",
        "mlp_blocks.3.0.weight",
        "option_a_head.mlp_blocks.3.0.weight",
        "statistics_attention_weights",
    ):
        p = dict(model.named_parameters())[pname]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0.0

    inf = Wave2BE2ERanker(
        product_attr_vocab_sizes=attrs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=42,
        pure_v_query=True,
        option_a_unseen_head=True,
        d_model=192,
        num_heads=8,
        memory_ffn_dim=576,
        memory_layers=2,
        max_seq_len=64,
        mlp_widths=(512, 384, 192, 96),
        aux_60d_enabled=False,
        seed=5,
    ).eval()
    inf.load_state_dict(strip_wave2b_training_aux(model.state_dict()), strict=True)


def test_weighted_loss_denominators_and_invalid_weights():
    logits = torch.tensor([[1., -1., 0.], [0., 0., float("-inf")], [3., 0., 0.]])
    valid = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    y = torch.tensor([[0., 1., 1.], [1., 0., 0.], [0., 0., 0.]])
    weights = torch.tensor([1., 3., 2.])
    loss = WeightedRankerLoss()
    actual = loss(logits, y, valid, weights)
    rows = [RankerLoss()(lg[m][None], t[m][None]) for lg, t, m in zip(logits, y, valid)]
    expected_bce = (3 * rows[0]["bce_loss"] + 6 * rows[1]["bce_loss"] + 2 * rows[2]["bce_loss"]) / 11
    expected_list = (rows[0]["list_loss"] + 3 * rows[1]["list_loss"]) / 4
    torch.testing.assert_close(actual["bce_loss"], expected_bce)
    torch.testing.assert_close(actual["list_loss"], expected_list)
    for bad in (torch.full((3,), torch.nan), weights[:, None]):
        with pytest.raises(ValueError, match="sample_weight"):
            loss(logits, y, valid, bad)
    y[1, -1] = 1.
    with pytest.raises(ValueError, match="masked"):
        loss(logits, y, valid, weights)


def test_v2_legacy_csv_date_gate_preserves_ids(tmp_path, monkeypatch):
    from recommender.experiments.v2_full import config

    csv_path = tmp_path / "cleaned.csv"
    csv_path.write_text("customer_id,product_id,purchase_date,quantity\n"
                        "012,009,2026-06-30,1\n999,888,2026-07-01,unparseable\n")
    monkeypatch.setattr(config, "CLEANED_PURCHASES_PATH", csv_path)
    monkeypatch.setattr(config, "DEV_PURCHASES_PATH", tmp_path / "dev.csv")
    df = config.load_v2_development_purchases()
    assert df[["customer_id", "product_id", "quantity"]].values.tolist() == [["012", "009", 1]]


def test_weighted_ranker_loss_extreme_bf16_logits_stay_finite():
    """Guard against float32 sum overflow and 0*inf=NaN on extreme finite bf16 logits with all-zero targets."""
    logits = torch.full((2, 4), 1e38, dtype=torch.bfloat16).float().requires_grad_(True)
    targets = torch.zeros((2, 4), dtype=torch.float32)
    cmask = torch.ones((2, 4), dtype=torch.bool)
    out = WeightedRankerLoss()(logits, targets, candidate_mask=cmask)
    assert torch.isfinite(out["bce_loss"])
    assert torch.isfinite(out["list_loss"])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_sequence_cache_rejects_empty_truncated_or_tampered_payload(tmp_path):
    """Ensure sequences_len64.pt verifies builder_version, all 9 keys, shapes, month zeros, and payload_sha256."""
    pids = ["P1", "P2"]
    dev_purchases = pd.DataFrame(
        [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-06-01"), "product_id": "P1", "quantity": 2.0,
             "item_type": "ТОВАР", "transaction_type": "ПРОДАЖА"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-07-01"), "product_id": "P2", "quantity": 3.0,
             "item_type": "ТОВАР", "transaction_type": "ПРОДАЖА"},
        ]
    )
    catalogue = pd.DataFrame(
        {
            "product_id": pids,
            "product_name": ["Oil A 4L", "Oil B 4L"],
            "base_name": ["oil a", "oil b"],
            "product_category": ["CAT1", "CAT2"],
            "product_subcategory": ["SUB1", "SUB2"],
            "business_line": ["BL1", "BL2"],
            "item_type": ["ТОВАР", "ТОВАР"],
            "is_blocked": [False, False],
        }
    )
    universe = build_base_name_universe(pids, catalogue, dev_purchases, train_cutoff=TRAIN_CUTOFF_V2)
    pool = RollingCandidatePool(catalogue, dev_purchases)
    reps = first_observed_pack_indices(pids, universe["bn_to_member_pid_indices"], pool.first_sales)
    shared_meta = {
        "product_ids": pids,
        "reps": reps,
        "universe": universe,
        "vocabs": {
            "category_vocab": {"CAT1": 1, "CAT2": 2},
            "subcategory_vocab": {"SUB1": 1, "SUB2": 2},
            "business_line_vocab": {"BL1": 1, "BL2": 2},
        },
    }
    tr_day = float((pd.Timestamp("2025-12-01") - PURCHASE_DATE_ORIGIN).days)
    va_day = float((pd.Timestamp("2026-03-15") - PURCHASE_DATE_ORIGIN).days)
    def _stub(d: float):
        return {
            "customer_ids": ["C1"],
            "prediction_days": torch.tensor([d], dtype=torch.float32),
            "first_observed": torch.tensor([False], dtype=torch.bool),
            "familiarity_bn": torch.zeros((1, 2, 4), dtype=torch.float32),
            "target_bn": torch.tensor([[True, False]]),
            "new_target_bn": torch.zeros((1, 2), dtype=torch.bool),
            "allowed_bn": torch.ones((1, 2), dtype=torch.bool),
            "unseen_bn": torch.zeros((1, 2), dtype=torch.bool),
            "attention_mask": torch.zeros((1, 32), dtype=torch.long),
        }

    tr64, _ = ensure_v2_sequences_for_length(
        64, shared_meta=shared_meta, tr_meta=_stub(tr_day), va_meta=_stub(va_day),
        dev_purchases=dev_purchases, catalogue=catalogue, cache_dir=tmp_path,
    )
    cache_p = tmp_path / "sequences_len64.pt"
    saved = torch.load(cache_p, map_location="cpu", weights_only=False)
    assert "builder_version" in saved["identity"]
    assert isinstance(saved.get("payload_sha256"), str) and len(saved["payload_sha256"]) == 64

    # Corrupt cache with matching identity but empty split dicts -> MUST rebuild cleanly to 64 slots!
    torch.save({"identity": saved["identity"], "payload_sha256": saved["payload_sha256"], "train": {}, "val": {}}, cache_p)
    tr64_rebuilt, _ = ensure_v2_sequences_for_length(
        64, shared_meta=shared_meta, tr_meta=_stub(tr_day), va_meta=_stub(va_day),
        dev_purchases=dev_purchases, catalogue=catalogue, cache_dir=tmp_path,
    )
    assert tr64_rebuilt["attention_mask"].shape == (1, 64)
    assert torch.equal(tr64_rebuilt["seq_product_ids"], tr64["seq_product_ids"])


def test_wave2a_direct_entry_validates_splits_and_requires_calibrated_refs(tmp_path, monkeypatch):
    from recommender.experiments.v2_full import run_v2_full_wave2 as runner

    # 1. Partial reference map containing only raw main-head predictions_ndcg must fail fast
    monkeypatch.setattr(
        runner, "resolve_v2_full_reference_predictions",
        lambda _sync=None: {"v2_full_ref_ndcg": tmp_path / "ndcg.npz"},
    )
    with pytest.raises(ValueError, match="missing keys"):
        runner.run_v2_full_wave2(skip_v1_verify=True, device="cpu")

    # 2. Direct train_and_eval_wave2_arm call with out-of-partition val date in train split must fail fast
    attrs, products, batch, _, allowed = _make_synthetic_batch(s=4)
    va_day = float((pd.Timestamp("2026-03-15") - PURCHASE_DATE_ORIGIN).days)
    bad_tr = {
        **batch, "customer_ids": ["A", "B", "C"], "allowed_bn": allowed,
        "target_bn": torch.zeros_like(allowed), "new_target_bn": torch.zeros_like(allowed),
        "aux_60d_bn": torch.zeros_like(allowed), "unseen_bn": allowed.clone(),
        "familiarity_bn": torch.zeros((3, 12, 4), dtype=torch.float32),
        "first_observed": torch.tensor([False, False, True]), "full_window": torch.ones(3, dtype=torch.bool),
        "text_embedding": batch["company_text_embedding"], "tabular_features": batch["customer_features"],
        "prediction_days": torch.tensor([va_day, va_day + 1.0, va_day + 2.0]),
    }
    spec = build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG)[0]
    with pytest.raises(ValueError, match="outside its development partition"):
        runner.train_and_eval_wave2_arm(
            spec, shared_meta={"prod_inputs": products}, tr_meta_arm=bad_tr, va_meta_arm=bad_tr,
            tr_cand38_mm=batch["candidate_features"].numpy(), va_cand38_mm=batch["candidate_features"].numpy(),
            tr_extra5_mm=np.zeros((3, 12, 5), dtype=np.float16), va_extra5_mm=np.zeros((3, 12, 5), dtype=np.float16),
            extra5_identity="x", dev_purchases=pd.DataFrame(), catalogue=pd.DataFrame(),
            out_dir=tmp_path / "bad_arm", seed=42, epochs=1, batch_size=3, dev=torch.device("cpu"),
        )


def test_wave2_static_shape_tpu_loss_and_tail_padding_exact_parity():
    """Verify static-shape TPU loss + tail-batch padding match sliced GPU loss identically in value & gradients."""
    from recommender.experiments.v2_full.run_v2_full_wave2 import (
        _pad_batch_for_static_tpu,
        compute_wave2_weighted_separated_losses,
    )

    torch.manual_seed(123)
    loss_fns = dict(
        main_loss_fn=WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0),
        main_aux_loss_fn=WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0),
        unseen_next_loss_fn=WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0),
        unseen_60d_loss_fn=WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0),
    )

    scenarios = [
        # (first_observed, target, aux_60d, full_window, weights)
        (
            torch.tensor([False, False, True, False]),
            torch.tensor(
                [[True, False, True, False], [False, True, False, False], [False, False, True, False], [True, True, False, False]]
            ),
            torch.tensor(
                [[True, False, True, True], [False, False, True, False], [False, False, True, False], [True, False, False, False]]
            ),
            torch.tensor([True, True, True, False]),
            torch.tensor([0.4, 1.8, 0.9, 1.3]),
        ),
        # All first-observed (zero returning rows -> Main Head skipped!)
        (
            torch.tensor([True, True, True, True]),
            torch.tensor(
                [[False, False, True, False], [False, False, False, True], [False, False, True, False], [False, False, False, False]]
            ),
            torch.tensor(
                [[False, False, True, False], [False, False, False, True], [False, False, True, False], [False, False, False, False]]
            ),
            torch.tensor([True, False, True, False]),
            None,
        ),
        # Zero unseen positives & zero full_window
        (
            torch.tensor([False, False, False, False]),
            torch.tensor(
                [[True, False, False, False], [False, True, False, False], [False, False, False, False], [True, False, False, False]]
            ),
            torch.zeros((4, 4), dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([1.0, 2.0, 0.5, 1.5]),
        ),
    ]
    allowed = torch.tensor(
        [[True, True, True, True], [True, True, True, False], [True, True, True, True], [True, True, False, True]]
    )
    unseen = torch.tensor(
        [[False, False, True, True], [False, True, True, False], [True, True, True, True], [False, False, False, True]]
    )

    for fo, tgt, aux60, fw, w in scenarios:
        base_m = torch.randn(4, 4)
        base_ma = torch.randn(4, 4)
        base_u = torch.randn(4, 4)
        base_ua = torch.randn(4, 4)
        unseen_eff = torch.where(fo[:, None], allowed, unseen)

        # 1. Dynamic sliced path (CPU/CUDA)
        m1, ma1, u1, ua1 = [t.clone().requires_grad_(True) for t in (base_m, base_ma, base_u, base_ua)]
        r_dyn = compute_wave2_weighted_separated_losses(
            main_logits=m1, main_aux_60d_logits=ma1, gen_unseen_logits=u1, gen_unseen_aux_60d_logits=ua1,
            allowed_bn=allowed, unseen_bn=unseen_eff, target_bn=tgt, aux_60d_bn=aux60,
            full_window=fw, first_observed=fo, unseen_head_weight=1.0, separated_main_targets=True,
            sample_weight=w, static_shapes=False, **loss_fns,
        )
        r_dyn["loss"].backward()

        # 2. Static-shape path (TPU/XLA) WITH 3 padded tail rows (4 -> 7)
        pad_n = 3

        def _pad_cat(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t[:1].expand(pad_n, *t.shape[1:])], dim=0)

        m2, ma2, u2, ua2 = [_pad_cat(t).clone().requires_grad_(True) for t in (base_m, base_ma, base_u, base_ua)]
        w_pad = torch.cat([w, w[:1].expand(pad_n)]) if w is not None else None
        vbm = torch.tensor([True, True, True, True, False, False, False])
        r_stat = compute_wave2_weighted_separated_losses(
            main_logits=m2, main_aux_60d_logits=ma2, gen_unseen_logits=u2, gen_unseen_aux_60d_logits=ua2,
            allowed_bn=_pad_cat(allowed), unseen_bn=_pad_cat(unseen_eff), target_bn=_pad_cat(tgt),
            aux_60d_bn=_pad_cat(aux60), full_window=_pad_cat(fw), first_observed=_pad_cat(fo),
            unseen_head_weight=1.0, separated_main_targets=True, sample_weight=w_pad,
            static_shapes=True, valid_batch_mask=vbm, **loss_fns,
        )
        r_stat["loss"].backward()

        assert torch.allclose(r_stat["loss"], r_dyn["loss"], atol=1e-6)
        for t_dyn, t_stat in ((m1, m2), (ma1, ma2), (u1, u2), (ua1, ua2)):
            g_dyn = torch.zeros_like(t_dyn) if t_dyn.grad is None else t_dyn.grad
            assert t_stat.grad is not None and torch.isfinite(t_stat.grad).all()
            assert torch.allclose(t_stat.grad[:4], g_dyn, atol=1e-6)
            assert torch.all(t_stat.grad[4:] == 0.0)

        # 3. Exact parity WITH synthetic unseen rows (2 real synthetic rows + 2 padded tail rows on TPU)
        syn_base = torch.randn(2, 4)
        syn_alw = torch.tensor([[True, True, True, True], [True, True, False, True]])
        syn_uns = torch.tensor([[False, True, True, True], [True, True, False, True]])
        syn_tgt = torch.tensor([[False, True, False, False], [True, False, False, False]])
        m1s, ma1s, u1s, ua1s, s1s = [t.clone().requires_grad_(True) for t in (base_m, base_ma, base_u, base_ua, syn_base)]
        r_dyn_s = compute_wave2_weighted_separated_losses(
            main_logits=m1s, main_aux_60d_logits=ma1s, gen_unseen_logits=u1s, gen_unseen_aux_60d_logits=ua1s,
            allowed_bn=allowed, unseen_bn=unseen_eff, target_bn=tgt, aux_60d_bn=aux60,
            full_window=fw, first_observed=fo, unseen_head_weight=1.0, separated_main_targets=True,
            sample_weight=w, static_shapes=False,
            syn_unseen_logits=s1s, syn_allowed_bn=syn_alw, syn_unseen_bn=syn_uns, syn_target_bn=syn_tgt,
            **loss_fns,
        )
        r_dyn_s["loss"].backward()

        def _pad2(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t[:1].expand(2, *t.shape[1:])], dim=0)

        m2s, ma2s, u2s, ua2s = [_pad_cat(t).clone().requires_grad_(True) for t in (base_m, base_ma, base_u, base_ua)]
        s2s = _pad2(syn_base).clone().requires_grad_(True)
        syn_vbm = torch.tensor([True, True, False, False])
        r_stat_s = compute_wave2_weighted_separated_losses(
            main_logits=m2s, main_aux_60d_logits=ma2s, gen_unseen_logits=u2s, gen_unseen_aux_60d_logits=ua2s,
            allowed_bn=_pad_cat(allowed), unseen_bn=_pad_cat(unseen_eff), target_bn=_pad_cat(tgt),
            aux_60d_bn=_pad_cat(aux60), full_window=_pad_cat(fw), first_observed=_pad_cat(fo),
            unseen_head_weight=1.0, separated_main_targets=True, sample_weight=w_pad,
            static_shapes=True, valid_batch_mask=vbm,
            syn_unseen_logits=s2s, syn_allowed_bn=_pad2(syn_alw), syn_unseen_bn=_pad2(syn_uns),
            syn_target_bn=_pad2(syn_tgt), syn_valid_batch_mask=syn_vbm,
            **loss_fns,
        )
        r_stat_s["loss"].backward()
        assert torch.allclose(r_stat_s["loss"], r_dyn_s["loss"], atol=1e-6)
        assert s2s.grad is not None and torch.isfinite(s2s.grad).all()
        assert torch.allclose(s2s.grad[:2], s1s.grad, atol=1e-6)
        assert torch.all(s2s.grad[2:] == 0.0)
        for t_dyn, t_stat in ((m1s, m2s), (ma1s, ma2s), (u1s, u2s), (ua1s, ua2s)):
            g_dyn = torch.zeros_like(t_dyn) if t_dyn.grad is None else t_dyn.grad
            assert t_stat.grad is not None and torch.isfinite(t_stat.grad).all()
            assert torch.allclose(t_stat.grad[:4], g_dyn, atol=1e-6)
            assert torch.all(t_stat.grad[4:] == 0.0)

    # Also verify _pad_batch_for_static_tpu preserves real rows cleanly
    rows_p, cand_p, rb = _pad_batch_for_static_tpu(
        torch.tensor([10, 20, 30]), np.ones((3, 4, 42), dtype=np.float32), batch_size=8
    )
    assert rb == 3 and len(rows_p) == 8 and cand_p.shape == (8, 4, 42)
    assert rows_p[:3].tolist() == [10, 20, 30]


def test_wave2_larger_model_size_arms_tower_scaling_bf16_and_presets():
    from recommender.experiments.v2_full import run_v2_full_wave2 as runner
    from recommender.experiments.v2_full.wave2b_models import Wave2BE2ERanker, strip_wave2b_training_aux

    torch.manual_seed(31)
    attrs, products, batch, _, allowed = _make_synthetic_batch(b=3, c=12, n_tr_c=9, s=16, d_model=128)
    cand42 = torch.rand(3, 12, 42)
    cust7 = torch.rand(3, 7)
    w_batch = {**batch, "candidate_features": cand42, "customer_features": cust7}
    y = torch.zeros(3, 12)
    y[:, 0] = 1.0
    y[:, 5] = 1.0

    # Validation guards for preset & tower dimensions
    with pytest.raises(ValueError, match="Unsupported Wave 2 preset"):
        runner.run_v2_full_wave2(preset="invalid", skip_v1_verify=True, device="cpu")
    with pytest.raises(ValueError, match="not both"):
        runner.run_v2_full_wave2(preset="size", arms=["w2_dim256"], skip_v1_verify=True, device="cpu")
    with pytest.raises(ValueError, match="not both"):
        runner.run_v2_full_wave2(preset="final_2mem", arms=["w2_2mem_hist64"], skip_v1_verify=True, device="cpu")
    with pytest.raises(ValueError, match="attr_emb_dim"):
        Wave2BE2ERanker(product_attr_vocab_sizes=attrs, num_cities=5, num_regions=4, num_industries=4, attr_emb_dim=0)
    with pytest.raises(ValueError, match="product_tower_hidden_dim"):
        Wave2BE2ERanker(
            product_attr_vocab_sizes=attrs, num_cities=5, num_regions=4, num_industries=4, product_tower_hidden_dim=0
        )
    with pytest.raises(ValueError, match="cust_tower_hidden_dim"):
        Wave2BE2ERanker(
            product_attr_vocab_sizes=attrs, num_cities=5, num_regions=4, num_industries=4, cust_tower_hidden_dim=0
        )

    # Default Wave2BE2ERanker preserves exact backward-compatible tower dimensions
    base_m = Wave2BE2ERanker(
        product_attr_vocab_sizes=attrs,
        num_cities=5,
        num_regions=4,
        num_industries=4,
        num_candidate_features=42,
        num_customer_features=7,
        num_pairwise_bias_features=42,
        pure_v_query=True,
        option_a_unseen_head=True,
        unseen_dropout_cold_prob=0.0,
        unseen_attn_aux_enabled=False,
        seed=5,
    )
    assert base_m.product_tower_hidden_dim == 128
    assert base_m.cust_tower_hidden_dim == 128
    assert base_m.attr_emb_dim == 16
    base_params = sum(p.numel() for p in base_m.parameters())

    specs_by_name = {s["name"]: s for s in build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG)}
    new_size_arms = ("w2_wider_towers", "w2_dim192_2mem_wider_mlp", "w2_dim256", "w2_dim256_2mem_4mlp")
    param_counts = {}
    wloss = WeightedRankerLoss()

    for arm_name in new_size_arms:
        cfg = specs_by_name[arm_name]
        model = Wave2BE2ERanker(
            product_attr_vocab_sizes=attrs,
            num_cities=5,
            num_regions=4,
            num_industries=4,
            num_candidate_features=42,
            num_customer_features=7,
            num_pairwise_bias_features=42,
            pure_v_query=True,
            option_a_unseen_head=True,
            unseen_dropout_cold_prob=0.0,
            unseen_attn_aux_enabled=False,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            memory_ffn_dim=cfg["memory_ffn_dim"],
            memory_layers=cfg["memory_layers"],
            max_seq_len=cfg["seq_len"],
            mlp_widths=cfg["mlp_widths"],
            product_tower_hidden_dim=cfg["product_tower_hidden_dim"],
            cust_tower_hidden_dim=cfg["cust_tower_hidden_dim"],
            attr_emb_dim=cfg["attr_emb_dim"],
            aux_60d_enabled=True,
            seed=5,
        )
        param_counts[arm_name] = sum(p.numel() for p in model.parameters())
        assert param_counts[arm_name] > base_params

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            res = model(**w_batch, prod_inputs=products, train_bn_limit=9, return_aux=True)
        loss = (
            wloss(res["output"].logits.float(), y, candidate_mask=allowed)["loss"]
            + 0.3 * wloss(res["aux_60d_logits"].float(), y, candidate_mask=allowed)["loss"]
            + wloss(res["unseen_logits"].float(), y, candidate_mask=allowed)["loss"]
            + 0.3 * wloss(res["unseen_aux_60d_logits"].float(), y, candidate_mask=allowed)["loss"]
        )
        loss.backward()

        last_mem = cfg["memory_layers"] - 1
        last_mlp = len(cfg["mlp_widths"]) - 1
        for pname in (
            "product_tower.text_proj.0.weight",
            "cust_text_proj.0.weight",
            "city_embedding.weight",
            f"memory_encoder.layers.{last_mem}.linear1.weight",
            f"mlp_blocks.{last_mlp}.0.weight",
            f"option_a_head.mlp_blocks.{last_mlp}.0.weight",
            "statistics_attention_weights",
        ):
            p = dict(model.named_parameters())[pname]
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0.0

        # Cached v_bn parity + stripped inference load parity
        model.eval()
        eval_batch = {**w_batch, "candidate_mask": torch.ones(3, 12, dtype=torch.bool)}
        with torch.no_grad():
            out_live = model(**eval_batch, prod_inputs=products, return_aux=False)
            v_cached = model.precompute_product_vectors(products)
            assert v_cached.shape == (12, cfg["d_model"])
            out_cached = model(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
            assert torch.allclose(out_live["output"].logits, out_cached["output"].logits, atol=1e-5)
            assert torch.allclose(out_live["unseen_logits"], out_cached["unseen_logits"], atol=1e-5)

        inf = Wave2BE2ERanker(
            product_attr_vocab_sizes=attrs,
            num_cities=5,
            num_regions=4,
            num_industries=4,
            num_candidate_features=42,
            num_customer_features=7,
            num_pairwise_bias_features=42,
            pure_v_query=True,
            option_a_unseen_head=True,
            unseen_dropout_cold_prob=0.0,
            unseen_attn_aux_enabled=False,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            memory_ffn_dim=cfg["memory_ffn_dim"],
            memory_layers=cfg["memory_layers"],
            max_seq_len=cfg["seq_len"],
            mlp_widths=cfg["mlp_widths"],
            product_tower_hidden_dim=cfg["product_tower_hidden_dim"],
            cust_tower_hidden_dim=cfg["cust_tower_hidden_dim"],
            attr_emb_dim=cfg["attr_emb_dim"],
            aux_60d_enabled=False,
            seed=5,
        ).eval()
        inf.load_state_dict(strip_wave2b_training_aux(model.state_dict()), strict=True)
        with torch.no_grad():
            out_inf = inf(**eval_batch, precomputed_v_bn=v_cached, return_aux=False)
            assert torch.allclose(out_live["output"].logits, out_inf["output"].logits, atol=1e-5)
            assert torch.allclose(out_live["unseen_logits"], out_inf["unseen_logits"], atol=1e-5)

    assert (
        base_params
        < param_counts["w2_wider_towers"]
        < param_counts["w2_dim192_2mem_wider_mlp"]
        < param_counts["w2_dim256"]
        < param_counts["w2_dim256_2mem_4mlp"]
    )
