"""Strict causality and accounting-sign tests for Stage 3A-3D commercial features."""

import numpy as np
import pandas as pd
import pytest

from recommender.experiments.roadmap_v2.commercial_features import (
    COMMERCIAL_FEATURE_NAMES,
    CommercialRollingFeatureBuilder,
    parse_deferral_days,
)


def test_parse_deferral_days():
    assert parse_deferral_days("ПРЕДОПЛАТА") == 0.0
    assert parse_deferral_days("ФАКТ") == 0.0
    assert parse_deferral_days("НАЛОЖКА") == 0.0
    assert parse_deferral_days("ДН14") == 14.0
    assert parse_deferral_days("ДН 7") == 7.0
    assert parse_deferral_days("ДН90") == 90.0


def test_commercial_strict_causality_payment_settlement_and_negative_cogs():
    catalogue = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "base_name": ["OIL_A", "OIL_B"],
            "brand": ["SHELL", "SHELL"],
            "product_category": ["CAT1", "CAT2"],
        }
    )
    events = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C1"],
            "purchase_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-02-10", "2025-03-01"]),
            "product_id": ["P1", "P1", "P2", "P2"],
            "quantity": [10.0, 5.0, 2.0, 1.0],
            "revenue": [10000.0, 5000.0, 4000.0, 2000.0],
            # NAV negative COGS!
            "cost": [-7000.0, -3500.0, -3000.0, -1500.0],
            "cost_usd": [-175.0, -87.5, -75.0, -37.5],
            "doc_bonus": [500.0, 0.0, 0.0, 0.0],
            "payment_terms": ["ДН14", "ДН14", "ПРЕДОПЛАТА", "ПРЕДОПЛАТА"],
            "payment_type": ["БЕЗНАЛ", "БЕЗНАЛ", "НАЛ", "НАЛ"],
            # Order 1 (Jan 1, due Jan 15) paid on Jan 25 (BEFORE Feb 20);
            # Order 2 (Feb 1, due Feb 15) paid on Feb 25 (AFTER Feb 20 -> MUST appear unpaid/overdue on Feb 20!)
            "full_payment_date": pd.to_datetime(["2025-01-25", "2025-02-25", "2025-02-10", "2025-03-01"]),
            "manager_id": ["M1", "M2", "M2", "M2"],
            "warehouse": ["КИЕВ", "КИЕВ", "КИЕВ", "ЛЬВОВ"],
        }
    )

    b_full = CommercialRollingFeatureBuilder(events, catalogue, ["P1", "P2"])
    # Truncated events strictly < 2025-02-20, AND hiding any settlement date >= 2025-02-20!
    ev_trunc = events.loc[events["purchase_date"] < pd.Timestamp("2025-02-20")].copy()
    ev_trunc.loc[ev_trunc["full_payment_date"] >= pd.Timestamp("2025-02-20"), "full_payment_date"] = pd.NaT
    b_trunc = CommercialRollingFeatureBuilder(ev_trunc, catalogue, ["P1", "P2"])

    f_full = b_full.build("C1", "2025-02-20")
    f_trunc = b_trunc.build("C1", "2025-02-20")
    np.testing.assert_allclose(f_full, f_trunc, rtol=0.0, atol=0.0)

    idx = {n: i for i, n in enumerate(COMMERCIAL_FEATURE_NAMES)}
    # On Feb 20:
    # - Order 1 (Jan 1) settled Jan 25 -> actual 24d / contract 14d -> ratio = (24/14)/2 = 0.8571
    assert np.isclose(f_full[0, idx["cust_actual_vs_contract_pay_ratio"]], (24.0 / 14.0) / 2.0, atol=1e-5)
    # - Order 2 (Feb 1, due Feb 15) NOT settled before Feb 20 -> 1 of 2 orders overdue -> 0.5!
    assert np.isclose(f_full[0, idx["cust_overdue_unpaid_ratio"]], 0.5, atol=1e-5)
    # - Margin for P1: (10000 - 7000) / 10000 = 0.30 (properly handling negative NAV COGS!)
    assert np.isclose(f_full[0, idx["sku_margin_ratio"]], 0.30, atol=1e-5)
    # - Manager changed from M1 -> M2 on Feb 1!
    assert f_full[0, idx["mgr_changed_recently"]] == 1.0
    # - Current manager M2 has 2 events in [t-90d, t): P1 (Feb 1) and P2 (Feb 10) -> 0.5 each!
    assert np.isclose(f_full[0, idx["mgr_sku_share_90d"]], 0.5, atol=1e-5)
    assert np.isclose(f_full[1, idx["mgr_sku_share_90d"]], 0.5, atol=1e-5)
    # - Warehouse КИЕВ shipped P1 on Feb 1 (1x in 30d) and P2 on Feb 10 (1x in 30d), NOT Mar 1!
    assert f_full[0, idx["wh_sku_shipped_30d"]] > 0.0
    assert f_full[1, idx["wh_sku_shipped_30d"]] > 0.0

    # Refuses held-out date
    with pytest.raises(ValueError, match="Development predictions only"):
        b_full.build("C1", "2026-02-19")
