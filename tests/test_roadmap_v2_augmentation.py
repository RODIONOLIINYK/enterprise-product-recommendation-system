"""Unit tests for Stage 2 60-day auxiliary label causality and History-SKU Dropout."""

import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.augmentation import (
    apply_stage1_history_sku_dropout,
    apply_stage2_history_sku_dropout,
    build_seq_universe_indices,
    build_strict_60d_aux_labels,
)
from recommender.experiments.roadmap_v2.config import ARM_SKU_PLUS_BASE_NAME, feature_names_for_arm
from recommender.split import TRAIN_CUTOFF


def test_60d_aux_labels_strictly_capped_at_train_cutoff():
    purchases = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C1", "C1"],
            "purchase_date": pd.to_datetime(
                ["2025-09-10", "2025-09-25", "2025-10-01", "2025-10-15", "2025-07-01"]
            ),
            "product_id": ["P1", "P2", "P3", "P4", "P5"],
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    origin = pd.Timestamp("2025-01-01")
    day_sep10 = (pd.Timestamp("2025-09-10") - origin).days
    day_jul01 = (pd.Timestamp("2025-07-01") - origin).days

    pids = ["P1", "P2", "P3", "P4", "P5"]
    labels = build_strict_60d_aux_labels(
        purchases,
        customer_ids=["C1", "C1"],
        prediction_days=[day_sep10, day_jul01],
        product_ids=pids,
        horizon_days=60,
        train_cutoff=TRAIN_CUTOFF,
    )
    # Row 0 (2025-09-10): window is [2025-09-10, 2025-10-01). P1 (Sep 10) and P2 (Sep 25) are True;
    # P3 (Oct 1 == TRAIN_CUTOFF) and P4 (Oct 15) MUST BE FALSE (zero val leakage)! P5 (Jul 1) is past -> False.
    assert labels[0].tolist() == [True, True, False, False, False]
    # Row 1 (2025-07-01): window is [2025-07-01, 2025-08-30). Only P5 (Jul 1) is in window!
    assert labels[1].tolist() == [False, False, False, False, True]

    # Refuses validation query date >= TRAIN_CUTOFF
    day_oct01 = (pd.Timestamp("2025-10-01") - origin).days
    with pytest.raises(ValueError, match="training-only"):
        build_strict_60d_aux_labels(purchases, ["C1"], [day_oct01], pids)


def test_stage1_and_stage2_history_sku_dropout_consistency():
    fnames = feature_names_for_arm(ARM_SKU_PLUS_BASE_NAME)
    idx_rep = fnames.index("is_repeat_sku")
    idx_days = fnames.index("days_since_last_sku_buy_norm")
    idx_bn_rep = fnames.index("has_bought_base_sku")
    idx_bn_days = fnames.index("days_since_last_base_sku_buy_norm")

    # 3 products: P0 and P1 share base_name 1; P2 has base_name 2.
    prod_bn_ids = torch.tensor([1, 1, 2], dtype=torch.long)

    # Stage 1 test: P0 is a repeat positive, P1 is a pack-mate (unseen SKU, bought base_name), P2 is repeat neg
    feats = torch.zeros((1, 3, len(fnames)), dtype=torch.float32)
    feats[0, 0, idx_rep] = 1.0
    feats[0, 0, idx_days] = 0.2
    feats[0, 0, idx_bn_rep] = 1.0
    feats[0, 0, idx_bn_days] = 0.2
    feats[0, 1, idx_bn_rep] = 1.0
    feats[0, 1, idx_bn_days] = 0.2
    feats[0, 2, idx_rep] = 1.0
    feats[0, 2, idx_bn_rep] = 1.0

    targets = torch.tensor([[1.0, 0.0, 0.0]])
    gen = torch.Generator().manual_seed(123)
    out_s1, dropped = apply_stage1_history_sku_dropout(
        feats, targets, prod_bn_ids, fnames, p_drop=1.0, generator=gen
    )
    assert dropped == 1
    # P0 exact SKU + base_name personal features reset to unseen defaults!
    assert out_s1[0, 0, idx_rep] == 0.0
    assert out_s1[0, 0, idx_days] == 1.0
    assert out_s1[0, 0, idx_bn_rep] == 0.0
    assert out_s1[0, 0, idx_bn_days] == 1.0
    # Pack-mate P1 also has its base_name repeat features reset so there's no backdoor!
    assert out_s1[0, 1, idx_bn_rep] == 0.0
    assert out_s1[0, 1, idx_bn_days] == 1.0
    # Unrelated P2 is completely untouched!
    assert out_s1[0, 2, idx_rep] == 1.0
    assert out_s1[0, 2, idx_bn_rep] == 1.0

    # Stage 2 test: verify matching 32-event sequence tokens (both P0 and pack-mate P1) are masked!
    seq_u = build_seq_universe_indices([["P0", "P1", "P2", ""]], ["P0", "P1", "P2"])
    inputs = {
        "pairwise_features": feats.clone(),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        "sequence": torch.ones((1, 4, 9), dtype=torch.float32),
        "seq_product_ids": torch.tensor([[10, 11, 12, 0]], dtype=torch.long),
        "seq_category_ids": torch.tensor([[1, 1, 2, 0]], dtype=torch.long),
        "seq_subcategory_ids": torch.tensor([[1, 1, 2, 0]], dtype=torch.long),
        "seq_business_line_ids": torch.tensor([[1, 1, 2, 0]], dtype=torch.long),
    }
    cand_idx = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out_s2, dropped2 = apply_stage2_history_sku_dropout(
        inputs,
        targets,
        cand_idx,
        seq_u,
        prod_bn_ids,
        fnames,
        p_drop=1.0,
        drop_base_name_in_sequence=True,
        generator=torch.Generator().manual_seed(123),
    )
    assert dropped2 == 1
    # Positions 0 (P0) and 1 (P1, same base_name) masked; position 2 (P2) preserved!
    assert out_s2["attention_mask"][0].tolist() == [0, 0, 1, 0]
    assert out_s2["seq_product_ids"][0].tolist() == [0, 0, 12, 0]
    assert float(out_s2["sequence"][0, 0].abs().sum()) == 0.0
    assert float(out_s2["sequence"][0, 2].abs().sum()) == 9.0
