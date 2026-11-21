"""Unit tests for Stage 5 base_name canonical entity collapsing, causality, and pack-size resolution."""

import numpy as np
import pandas as pd
import torch

from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    collapse_sku_scores_to_base_name_ranking,
    evaluate_base_name_predictions,
    resolve_base_names_to_skus,
)


def test_stage5_universe_sequences_and_evaluator():
    catalogue = pd.DataFrame(
        {
            "product_id": ["P1_20L", "P1_209L", "P2_20L", "P3_4L"],
            "base_name": ["OIL_A", "OIL_A", "OIL_B", "OIL_C"],
            "container_type": ["ВЕДРО", "БОЧКА", "ВЕДРО", "КАНИСТРА"],
            "product_category": ["CAT1", "CAT1", "CAT1", "CAT2"],
            "product_subcategory": ["SUB1", "SUB1", "SUB1", "SUB2"],
            "business_line": ["L1", "L1", "L1", "L2"],
        }
    )
    purchases = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2"],
            "purchase_date": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-02-01", "2025-02-01"]),
            "product_id": ["P1_20L", "P1_209L", "P2_20L", "P3_4L"],
            "quantity": [2.0, 1.0, 5.0, 3.0],
        }
    )
    pids = ["P1_20L", "P1_209L", "P2_20L", "P3_4L"]
    uni = build_base_name_universe(pids, catalogue, purchases)
    assert uni["num_base_names"] == 3
    assert uni["trained_base_names"] == 3

    # Build sequences strictly < 2025-02-01 for C1: Jan 10 multi-pack (P1_20L + P1_209L) collapses to ONE event!
    origin = pd.Timestamp("2025-01-01")
    day_feb1 = (pd.Timestamp("2025-02-01") - origin).days
    seqs = build_base_name_sequences(
        purchases,
        ["C1"],
        [day_feb1],
        uni,
        catalogue,
        {"CAT1": 1, "CAT2": 2},
        {"SUB1": 1, "SUB2": 2},
        {"L1": 1, "L2": 2},
        seq_len=8,
    )
    # Only 1 valid token (Jan 10 collapsed), left-aligned at slot 0, and Feb 1 same-day purchase is strictly excluded!
    assert int(seqs["attention_mask"][0].sum().item()) == 1
    assert int(seqs["seq_product_ids"][0, 0].item()) == uni["bn_to_idx"]["OIL_A"] + 1

    # Collapse SKU scores: P1_20L (score 10) and P1_209L (score 9) both map to OIL_A -> OIL_A appears ONCE at rank 0!
    sku_scores = torch.tensor([[10.0, 9.0, 5.0, 8.0]])
    ranked_bn = collapse_sku_scores_to_base_name_ranking(sku_scores, None, uni["pid_idx_to_bn_idx"], max_k=3)
    expected_order = [uni["bn_to_idx"]["OIL_A"], uni["bn_to_idx"]["OIL_C"], uni["bn_to_idx"]["OIL_B"]]
    assert ranked_bn[0].tolist() == expected_order

    # Evaluator check: target is OIL_C (rank 1) which is truly new
    t_bn = torch.zeros((1, 3), dtype=torch.bool)
    t_bn[0, uni["bn_to_idx"]["OIL_C"]] = True
    metrics = evaluate_base_name_predictions(ranked_bn, t_bn, t_bn, ["C1"])
    assert metrics["true_new_bn_hits@10"] == 1
    assert np.isclose(metrics["bn_ndcg@10"], 1.0 / np.log2(3.0), atol=1e-6)

    # Pack-size resolution check: for C1 on Feb 15, C1 bought P1_20L and P1_209L once each on Jan 10,
    # and P2_20L (ВЕДРО) on Feb 1 -> for OIL_A both packs bought once; for OIL_B -> P2_20L.
    day_feb15 = (pd.Timestamp("2025-02-15") - origin).days
    allowed = torch.ones((1, 4), dtype=torch.bool)
    resolved = resolve_base_names_to_skus(
        ranked_bn, ["C1"], [day_feb15], allowed, purchases, catalogue, pids, uni, top_k=3
    )
    assert resolved[0, 1] == 3  # OIL_C -> P3_4L
    assert resolved[0, 2] == 2  # OIL_B -> P2_20L
