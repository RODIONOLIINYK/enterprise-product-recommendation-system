"""Unit and parity tests for Stage 5 v5c cadence-normalized self-attention bias (linear & MLP)."""

from __future__ import annotations

import math

import pandas as pd
import torch

from recommender.experiments.roadmap_v2.clean_ranker import (
    SELF_ATTENTION_BIAS_NAMES,
    SharedUVStage5Ranker,
)
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
)
from recommender.sequences import DAYS_PER_YEAR


def _synthetic_v5c_inputs(batch_size: int = 3, seq_len: int = 8, num_bn: int = 12):
    torch.manual_seed(7)
    customer_vector = torch.randn(batch_size, 128)
    customer_features = torch.rand(batch_size, 11)
    catalog_product_vectors = torch.randn(num_bn, 128)
    sequence = torch.rand(batch_size, seq_len, 9)
    seq_rel_qty = torch.zeros(batch_size, seq_len)
    cust_median_gap_days = torch.tensor([20.0, 30.0, 4.0])[:batch_size]
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    seq_product_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    seq_purchase_dates = torch.zeros(batch_size, seq_len)
    prediction_day = torch.tensor([120.0, 200.0, 300.0])[:batch_size]

    # Row 0: returning customer (cadence=20d) with 4 purchases (two same-day at day 30, then 90, 110)
    attention_mask[0, :4] = 1
    seq_product_ids[0, :4] = torch.tensor([2, 5, 2, 7])
    seq_purchase_dates[0, :4] = torch.tensor([30.0, 30.0, 90.0, 110.0]) / DAYS_PER_YEAR
    seq_rel_qty[0, :4] = torch.tensor([1.0, 0.5, 0.8, 0.25])

    # Row 1: first-observed basket (0 valid history tokens!)

    # Row 2: weekly buyer (raw gap=4d, clamped to 7d floor) with 1 prior purchase
    if batch_size >= 3:
        attention_mask[2, :1] = 1
        seq_product_ids[2, :1] = torch.tensor([3])
        seq_purchase_dates[2, :1] = torch.tensor([250.0]) / DAYS_PER_YEAR
        seq_rel_qty[2, :1] = torch.tensor([1.0])

    candidate_features = torch.rand(batch_size, num_bn, 38)
    candidate_mask = torch.ones(batch_size, num_bn, dtype=torch.bool)
    candidate_mask[:, -1] = False
    return {
        "customer_vector": customer_vector,
        "customer_features": customer_features,
        "catalog_product_vectors": catalog_product_vectors,
        "sequence": sequence,
        "seq_rel_qty": seq_rel_qty,
        "cust_median_gap_days": cust_median_gap_days,
        "attention_mask": attention_mask,
        "seq_product_ids": seq_product_ids,
        "seq_purchase_dates": seq_purchase_dates,
        "prediction_day": prediction_day,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
    }


def test_seq_rel_qty_and_cust_median_gap_causal():
    catalogue = pd.DataFrame(
        [
            {"product_id": "P1", "base_name": "Oil A", "product_category": "C1", "product_subcategory": "S1", "business_line": "L1"},
            {"product_id": "P2", "base_name": "Grease B", "product_category": "C1", "product_subcategory": "S1", "business_line": "L1"},
            {"product_id": "P3", "base_name": "Sample C", "product_category": "C1", "product_subcategory": "S1", "business_line": "L1"},
        ]
    )
    purchases = pd.DataFrame(
        [
            # BIG: buys on days 1, 1 (same basket!), 11, 41 -> unique gaps [10, 30] -> median 20d; future day 50 excluded
            {"customer_id": "BIG", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P1", "quantity": 2000.0},
            {"customer_id": "BIG", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P2", "quantity": 60.0},
            {"customer_id": "BIG", "purchase_date": pd.Timestamp("2025-01-12"), "product_id": "P1", "quantity": 500.0},
            {"customer_id": "BIG", "purchase_date": pd.Timestamp("2025-02-11"), "product_id": "P2", "quantity": 100.0},
            {"customer_id": "BIG", "purchase_date": pd.Timestamp("2025-02-20"), "product_id": "P3", "quantity": 10000.0},
            # SMALL: 1 prior basket -> fallback median gap 30d; 20L vs 1L on same day
            {"customer_id": "SMALL", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P1", "quantity": 20.0},
            {"customer_id": "SMALL", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P3", "quantity": 1.0},
        ]
    )
    universe = build_base_name_universe(["P1", "P2", "P3"], catalogue, purchases)
    # pred_day = 50 corresponds to 2025-02-20 -> future 10000L order on 2025-02-20 MUST be excluded
    out = build_base_name_sequences(
        purchases,
        ["BIG", "SMALL", "NEW_CUST"],
        [50, 50, 50],
        universe,
        catalogue,
        {"C1": 1},
        {"S1": 1},
        {"L1": 1},
        seq_len=8,
    )
    rel = out["seq_rel_qty"]
    gaps = out["cust_median_gap_days"]
    assert rel.shape == (3, 8) and gaps.shape == (3,)
    assert math.isclose(float(gaps[0]), 20.0, rel_tol=1e-5)
    assert math.isclose(float(gaps[1]), 30.0, rel_tol=1e-5)
    assert math.isclose(float(gaps[2]), 30.0, rel_tol=1e-5)

    # BIG: on 2025-01-02, alphabetical base_name order puts Grease B (60L) at slot 0, Oil A (2000L) at slot 1
    assert math.isclose(float(rel[0, 0]), math.log1p(60.0) / math.log1p(2000.0), rel_tol=1e-5)
    assert math.isclose(float(rel[0, 1]), 1.0, rel_tol=1e-5)
    assert float(rel[0, 4:].abs().sum()) == 0.0
    # SMALL: Oil A (20L) at slot 0, Sample C (1L) at slot 1
    assert math.isclose(float(rel[1, 0]), 1.0, rel_tol=1e-5)
    assert math.isclose(float(rel[1, 1]), math.log1p(1.0) / math.log1p(20.0), rel_tol=1e-5)
    assert float(rel[2].abs().sum()) == 0.0


def test_self_attention_v5c_quadrants_and_cadence_math():
    model = SharedUVStage5Ranker(self_attention_bias_mode="linear", aux_60d_enabled=False, seed=42)
    inp = _synthetic_v5c_inputs(batch_size=3, seq_len=8, num_bn=12)
    purchase_days = (inp["seq_purchase_dates"] * DAYS_PER_YEAR).round()
    feats = model._build_self_attention_pair_features(
        purchase_days=purchase_days,
        prediction_day=inp["prediction_day"],
        attention_mask=inp["attention_mask"],
        rel_qty=inp["seq_rel_qty"],
        cust_median_gap_days=inp["cust_median_gap_days"],
        dtype=torch.float32,
    )
    assert feats.shape == (3, 9, 9, len(SELF_ATTENTION_BIAS_NAMES))

    # Row 0: cadence g=20d, prediction_day=120, Tokens 1..4 at days [30, 30, 90, 110]
    # 1. Token 0 -> Token 0 (0,0): all 5 features strictly 0
    assert float(feats[0, 0, 0].abs().sum()) == 0.0

    # 2. Token 0 -> History Token 1 (day 30): delta=90d -> r=90/20=4.5 -> t0_key_age=4.5/10.5
    #    earlier_hh=0, later_hh=0, key_is_token0=0, key_rel_qty=1.0
    assert float(feats[0, 0, 1, 0]) == 0.0
    assert float(feats[0, 0, 1, 1]) == 0.0
    assert math.isclose(float(feats[0, 0, 1, 2]), 4.5 / 10.5, rel_tol=1e-5)
    assert math.isclose(float(feats[0, 0, 1, 3]), 1.0, rel_tol=1e-5)
    assert float(feats[0, 0, 1, 4]) == 0.0

    # 3. History -> Token 0 (i>=1, j=0): ONLY key_is_token0 == 1.0; all other 4 are 0.0!
    assert float(feats[0, 1:, 0, :4].abs().sum()) == 0.0
    assert torch.equal(feats[0, 1:, 0, 4], torch.ones(8))

    # 4. History -> History same day Token 1 (day 30) & Token 2 (day 30): earlier_hh=0, later_hh=0, t0_key_age=0
    assert float(feats[0, 1, 2, 0]) == 0.0 and float(feats[0, 1, 2, 1]) == 0.0
    assert float(feats[0, 1, 2, 2]) == 0.0 and float(feats[0, 1, 2, 4]) == 0.0
    assert math.isclose(float(feats[0, 1, 2, 3]), 0.5, rel_tol=1e-5)

    # 5. Token 3 (day 90) -> Token 1 (day 30): delta=+60d -> r=60/20=3.0 -> earlier_hh=3/8=0.375, later_hh=0
    assert math.isclose(float(feats[0, 3, 1, 0]), 3.0 / 8.0, rel_tol=1e-5)
    assert float(feats[0, 3, 1, 1]) == 0.0
    # Reverse Token 1 -> Token 3: earlier_hh=0, later_hh=3/8=0.375
    assert float(feats[0, 1, 3, 0]) == 0.0
    assert math.isclose(float(feats[0, 1, 3, 1]), 3.0 / 8.0, rel_tol=1e-5)

    # 6. Row 2: raw cadence 4.0d MUST be floored to 7.0d! prediction_day=300, Token 1 day=250 -> delta=50 -> r=50/7
    expected_t0 = (50.0 / 7.0) / (50.0 / 7.0 + 6.0)
    assert math.isclose(float(feats[2, 0, 1, 2]), expected_t0, rel_tol=1e-5)


def test_shared_uv_backward_compat_zero_init_and_no_weight_decay():
    inp = _synthetic_v5c_inputs(batch_size=3, seq_len=8, num_bn=12)

    # 1. Default mode="none" has zero new state_dict keys and ignores seq_rel_qty / cust_median_gap_days
    m_base = SharedUVStage5Ranker(aux_60d_enabled=False, seed=42).eval()
    assert not any("self_attention_bias" in k for k in m_base.state_dict())
    inp_legacy = {
        k: v for k, v in inp.items() if k not in ("seq_rel_qty", "cust_median_gap_days")
    }
    with torch.no_grad():
        o1 = m_base(**inp_legacy).logits
        o2 = m_base(**inp).logits
    assert torch.equal(o1, o2)

    # 2. Both "linear" and "mlp" start at exact unbiased baseline (zero-init output) and separate wd=0.0
    for mode, expected_bias_params in (("linear", 40), ("mlp", 5 * 16 + 16 + 16 * 8 + 8)):
        m = SharedUVStage5Ranker(self_attention_bias_mode=mode, aux_60d_enabled=False, seed=42).eval()
        missing, unexpected = m.load_state_dict(m_base.state_dict(), strict=False)
        assert all(k.startswith("self_attention_bias_") for k in missing) and unexpected == []
        with torch.no_grad():
            o_zero = m(**inp).logits
        assert torch.allclose(o1[:, :-1], o_zero[:, :-1], atol=1e-5, rtol=1e-5)

        groups = m.optimizer_param_groups(weight_decay=1e-2)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 1e-2
        assert groups[1]["weight_decay"] == 0.0
        assert sum(p.numel() for p in groups[1]["params"]) == expected_bias_params

        # Verify finite outputs and non-zero gradients on both returning and empty-history rows
        m_tr = SharedUVStage5Ranker(self_attention_bias_mode=mode, aux_60d_enabled=True, seed=42).train()
        out, aux = m_tr(**inp, return_aux=True)
        assert torch.isfinite(out.logits[:, :-1]).all() and torch.isfinite(aux[:, :-1]).all()
        loss = out.logits[:, :-1].pow(2).mean() + 0.3 * aux[:, :-1].pow(2).mean()
        loss.backward()
        bias_grads = [
            p.grad for n, p in m_tr.named_parameters() if n.startswith("self_attention_bias_")
        ]
        assert bias_grads and all(g is not None and torch.isfinite(g).all() for g in bias_grads)
        assert sum(float(g.abs().sum()) for g in bias_grads) > 0.0


def test_no_memory_encoder_cross_attention_only_and_padding_isolation():
    inp = _synthetic_v5c_inputs(batch_size=3, seq_len=8, num_bn=12)
    m_full = SharedUVStage5Ranker(use_memory_encoder=True, aux_60d_enabled=False, seed=42)
    m_xattn = SharedUVStage5Ranker(use_memory_encoder=False, aux_60d_enabled=False, seed=42).eval()
    assert not any(k.startswith("memory_encoder.") for k in m_xattn.state_dict())
    n_full = sum(p.numel() for p in m_full.parameters())
    n_xattn = sum(p.numel() for p in m_xattn.parameters())
    assert n_full - n_xattn > 150_000

    # Perturbing padded history slots must produce bit-identical logits
    inp_perturbed = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    inp_perturbed["seq_product_ids"][0, 4:] = 9
    inp_perturbed["sequence"][0, 4:] = 99.0
    inp_perturbed["seq_product_ids"][1, :] = 11
    with torch.no_grad():
        o1 = m_xattn(**inp).logits[:, :-1]
        o2 = m_xattn(**inp_perturbed).logits[:, :-1]
    assert torch.isfinite(o1).all()
    assert torch.equal(o1, o2)

    # Gradients stay finite including empty-history row 1
    m_tr = SharedUVStage5Ranker(use_memory_encoder=False, aux_60d_enabled=True, seed=42).train()
    out, aux = m_tr(**inp, return_aux=True)
    loss = out.logits[:, :-1].pow(2).mean() + 0.3 * aux[:, :-1].pow(2).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m_tr.parameters() if p.requires_grad)

