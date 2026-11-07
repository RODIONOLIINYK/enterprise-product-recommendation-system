"""Adversarial and Empirical Verification Test Suite for Milestone M8.

Author: Challenger 2
Evaluates:
1. Stage 1 TwoTowerModel checkpoint restoration and shape invariants (best_retriever.pt).
2. Training history logging completeness and metric sanity across all 10 epochs.
3. Chronological transition partitioning and strict date isolation.
4. Validation history accumulation (including train-period purchases strictly < as_of_date).
5. Zero leakage of test split events (>= 2026-02-19) into train/val transitions.
6. Point-in-time product demand features invariance under future event injection.
7. Cold-start zero-history and boundary condition robustness.
"""

from pathlib import Path
import json
import math
import unittest

import numpy as np
import pandas as pd
import torch

from recommender.candidates import load_retriever_from_checkpoint
from recommender.features import PointInTimeNormalizer, compute_point_in_time_features
from recommender.models import CustomerTower, TwoTowerModel
from recommender.sequences import ENRICHED_FEATURE_DIM, build_customer_sequence
from recommender.split import TEST_START, TRAIN_CUTOFF, VAL_START
from recommender.train_retriever import (
    RetrieverDataset,
    build_product_tower_inputs,
    build_retriever_transitions,
    compute_product_demand_features,
    compute_retrieval_metrics,
)


class TestChallengerM8RetrieverVerification(unittest.TestCase):
    """Empirical verification suite for Milestone M8."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
        cls.history_path = Path("checkpoints/retriever/training_history.json")
        cls.purchases_path = Path("data/interim/cleaned_purchases.csv")

    def test_01_checkpoint_restoration_and_strict_weights(self):
        """Verify best_retriever.pt restores cleanly into TwoTowerModel with strict=True."""
        self.assertTrue(self.ckpt_path.is_file(), f"Checkpoint missing: {self.ckpt_path}")
        self.assertGreater(self.ckpt_path.stat().st_size, 1_000_000, "Checkpoint too small (< 1MB)")

        model, ckpt = load_retriever_from_checkpoint(self.ckpt_path)
        self.assertIsInstance(model, TwoTowerModel)
        required_keys = [
            "epoch", "model_state_dict", "optimizer_state_dict",
            "best_recall_50", "product_ids", "product_to_idx",
            "val_metrics", "vocabs", "config",
        ]
        for k in required_keys:
            self.assertIn(k, ckpt, f"Checkpoint missing key: {k}")

        product_ids = ckpt["product_ids"]
        self.assertEqual(len(product_ids), ckpt["config"]["num_catalog_products"])
        self.assertEqual(ckpt["product_to_idx"], {sku: i for i, sku in enumerate(product_ids)})

        # Confirm that the production loader restored every saved tensor.
        loaded_state = model.state_dict()
        self.assertEqual(loaded_state.keys(), ckpt["model_state_dict"].keys())
        for name, value in loaded_state.items():
            torch.testing.assert_close(value, ckpt["model_state_dict"][name], rtol=0, atol=0)

        # Check weights are finite (no NaNs, no Infs)
        for name, param in model.named_parameters():
            self.assertTrue(torch.isfinite(param).all().item(), f"Non-finite weights in parameter: {name}")

    def test_02_forward_pass_and_shape_invariants(self):
        """Restored model must produce exact output shapes and unit L2 norms."""
        model, ckpt = load_retriever_from_checkpoint(self.ckpt_path)
        config = ckpt["config"]
        P = len(ckpt["product_ids"])
        B = 8
        customer_tower = model.customer_tower
        product_tower = model.product_tower
        seq_len = customer_tower.max_seq_len

        with torch.inference_mode():
            v_p = product_tower(
                torch.randn(P, config["text_dim"]),
                torch.zeros(P, dtype=torch.long),
                torch.zeros(P, dtype=torch.long),
                torch.zeros(P, dtype=torch.long) if product_tower.subcat_embedding is not None else None,
                torch.zeros(P, dtype=torch.long),
                torch.rand(P, product_tower.num_demand_features),
            )
            u_c = customer_tower(
                torch.randn(B, config["text_dim"]),
                torch.zeros(B, dtype=torch.long),
                torch.zeros(B, dtype=torch.long),
                torch.zeros(B, dtype=torch.long),
                torch.rand(B, customer_tower.num_tabular_features),
                torch.rand(B, seq_len, ENRICHED_FEATURE_DIM),
                torch.ones(B, seq_len, dtype=torch.long),
                seq_product_ids=((torch.arange(seq_len) % P) + 1).expand(B, -1),
                seq_purchase_dates=torch.linspace(0, 1, seq_len).expand(B, -1),
                catalog_product_vectors=v_p,
            )

        self.assertEqual(u_c.shape, (B, config["output_dim"]))
        u_c_norms = torch.norm(u_c, dim=-1)
        self.assertTrue(torch.allclose(u_c_norms, torch.ones_like(u_c_norms), atol=1e-4))

        self.assertEqual(v_p.shape, (P, config["output_dim"]))
        v_p_norms = torch.norm(v_p, dim=-1)
        self.assertTrue(torch.allclose(v_p_norms, torch.ones_like(v_p_norms), atol=1e-4))

        logits = model.compute_logits(u_c, v_p, product_indices=torch.arange(P))
        self.assertEqual(logits.shape, (B, P))
        self.assertTrue(torch.isfinite(logits).all().item())

        targets = torch.zeros(B, P)
        targets[0, 5] = 1.0
        targets[1, 10] = 1.0
        loss = model.compute_multi_label_loss(logits, targets, pos_weight=20.0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_03_training_history_metrics_and_monotonicity(self):
        """training_history.json must record all 10 epochs with required metrics and sanity properties."""
        self.assertTrue(self.history_path.is_file(), f"History missing: {self.history_path}")

        with open(self.history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        self.assertEqual(len(history), 10, "training_history.json must log exactly 10 epochs.")

        required_metric_keys = [
            ("Train Loss", "train_loss"),
            ("Val Loss", "val_loss"),
            ("Val Recall@10", "val_recall@10"),
            ("Val Recall@50", "val_recall@50"),
            ("Val Recall@100", "val_recall@100"),
            ("Val NDCG@10", "val_ndcg@10"),
            ("Val MRR", "val_mrr"),
        ]

        for i, ep in enumerate(history, start=1):
            self.assertEqual(ep["epoch"], i, f"Epoch number mismatch at index {i-1}")
            for cap_k, lower_k in required_metric_keys:
                self.assertTrue(cap_k in ep or lower_k in ep, f"Epoch {i} missing metric {cap_k}/{lower_k}")
                val = float(ep.get(cap_k, ep.get(lower_k)))
                self.assertFalse(math.isnan(val), f"Metric {cap_k} in epoch {i} is NaN")

            r10 = float(ep.get("Val Recall@10", ep.get("val_recall@10")))
            r50 = float(ep.get("Val Recall@50", ep.get("val_recall@50")))
            r100 = float(ep.get("Val Recall@100", ep.get("val_recall@100")))
            ndcg10 = float(ep.get("Val NDCG@10", ep.get("val_ndcg@10")))
            mrr = float(ep.get("Val MRR", ep.get("val_mrr")))

            # Metric range bounds
            for name, val in [("Recall@10", r10), ("Recall@50", r50), ("Recall@100", r100), ("NDCG@10", ndcg10), ("MRR", mrr)]:
                self.assertTrue(0.0 <= val <= 1.0, f"Epoch {i} {name}={val} out of [0, 1]")

            # Monotonic hierarchy: Recall@10 <= Recall@50 <= Recall@100
            self.assertLessEqual(r10, r50 + 1e-6, f"Epoch {i}: Recall@10 ({r10}) > Recall@50 ({r50})")
            self.assertLessEqual(r50, r100 + 1e-6, f"Epoch {i}: Recall@50 ({r50}) > Recall@100 ({r100})")

        # Convergence progression
        first_epoch = history[0]
        last_epoch = history[-1]
        self.assertLess(
            float(last_epoch.get("Train Loss", last_epoch.get("train_loss"))),
            float(first_epoch.get("Train Loss", first_epoch.get("train_loss"))),
            "Train loss must decrease from epoch 1 to epoch 10",
        )
        self.assertGreater(
            float(last_epoch.get("Val Recall@50", last_epoch.get("val_recall@50"))),
            float(first_epoch.get("Val Recall@50", first_epoch.get("val_recall@50"))),
            "Val Recall@50 must increase from epoch 1 to epoch 10",
        )

        # Checkpoint best_recall_50 matches history max
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        history_max_r50 = max(float(e.get("Val Recall@50", e.get("val_recall@50"))) for e in history)
        self.assertAlmostEqual(ckpt["best_recall_50"], history_max_r50, places=5)

    def test_04_chronological_partitions_and_split_isolation(self):
        """Validate chronological boundaries: Train < 2025-10-01, Val 2025-10-01..2026-02-18, Test >= 2026-02-19."""
        if not self.purchases_path.is_file():
            self.skipTest(f"Data file missing: {self.purchases_path}")

        purchases_df = pd.read_csv(self.purchases_path, low_memory=False)
        train_t, val_t, test_t = build_retriever_transitions(purchases_df)

        self.assertGreater(len(train_t), 0, "Train transitions empty")
        self.assertGreater(len(val_t), 0, "Val transitions empty")
        self.assertGreater(len(test_t), 0, "Test transitions empty")

        train_cutoff = pd.Timestamp("2025-10-01")
        val_end = pd.Timestamp("2026-02-18 23:59:59.999999")
        test_start = pd.Timestamp("2026-02-19")

        for t in train_t:
            as_of = pd.Timestamp(t["as_of_date"])
            self.assertLess(as_of, train_cutoff, f"Train transition leaked past cutoff: {as_of}")

        for t in val_t:
            as_of = pd.Timestamp(t["as_of_date"])
            self.assertGreaterEqual(as_of, train_cutoff, f"Val transition before 2025-10-01: {as_of}")
            self.assertLessEqual(as_of, val_end, f"Val transition after 2026-02-18: {as_of}")
            self.assertLess(as_of, test_start, f"Val transition in test period: {as_of}")

        for t in test_t:
            as_of = pd.Timestamp(t["as_of_date"])
            self.assertGreaterEqual(as_of, test_start, f"Test transition before 2026-02-19: {as_of}")

    def test_05_validation_transitions_include_prior_train_history(self):
        """Validation transitions must strictly include customer purchases made in the train period."""
        cid = "C_CHALLENGER_VAL_HISTORY"
        records = [
            # Train period events (< 2025-10-01)
            {"customer_id": cid, "purchase_date": pd.Timestamp("2024-06-15"), "product_id": "TRAIN_P1", "quantity": 3.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-03-20"), "product_id": "TRAIN_P2", "quantity": 7.0, "product_category": "CAT_B"},
            # Validation period events (2025-10-01 .. 2026-02-18)
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-10-25"), "product_id": "VAL_P1", "quantity": 2.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-12-10"), "product_id": "VAL_P2", "quantity": 4.0, "product_category": "CAT_B"},
        ]
        df = pd.DataFrame(records)

        train_t, val_t, test_t = build_retriever_transitions(df)
        self.assertEqual(len(train_t), 1)  # Target: 2025-03-20
        self.assertEqual(len(val_t), 2)    # Targets: 2025-10-25 and 2025-12-10
        self.assertEqual(len(test_t), 0)

        # For the first validation transition (target: 2025-10-25), history must include both train purchases
        seq_res1 = build_customer_sequence(df, customer_id=cid, as_of_date=pd.Timestamp("2025-10-25"), max_len=32)
        pids1 = [p for p in seq_res1["product_ids"] if p != "[PAD]"]
        self.assertEqual(len(pids1), 2)
        self.assertIn("TRAIN_P1", pids1)
        self.assertIn("TRAIN_P2", pids1)

        # For the second validation transition (target: 2025-12-10), history must include both train purchases and VAL_P1
        seq_res2 = build_customer_sequence(df, customer_id=cid, as_of_date=pd.Timestamp("2025-12-10"), max_len=32)
        pids2 = [p for p in seq_res2["product_ids"] if p != "[PAD]"]
        self.assertEqual(len(pids2), 3)
        self.assertEqual(pids2, ["TRAIN_P1", "TRAIN_P2", "VAL_P1"])

    def test_06_test_split_future_poisoning_isolation(self):
        """Events from the test period (>= 2026-02-19) must NEVER leak into val transition sequences or features."""
        cid = "C_POISON_ISOLATION_TEST"
        clean_records = [
            {"customer_id": cid, "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "P_PAST_1", "quantity": 5.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-08-01"), "product_id": "P_PAST_2", "quantity": 10.0, "product_category": "CAT_B"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-11-15"), "product_id": "P_VAL_TARGET", "quantity": 2.0, "product_category": "CAT_C"},
        ]
        clean_df = pd.DataFrame(clean_records)

        # Poison with test-split events (>= 2026-02-19) with massive quantities and repeat items
        poison_records = clean_records + [
            {"customer_id": cid, "purchase_date": pd.Timestamp("2026-02-19 00:00:00"), "product_id": "POISON_TEST_1", "quantity": 999999.0, "product_category": "CAT_X"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2026-03-15"), "product_id": "P_PAST_1", "quantity": 500000.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2026-06-01"), "product_id": "POISON_TEST_2", "quantity": 1000000.0, "product_category": "CAT_Y"},
        ]
        poison_df = pd.DataFrame(poison_records)

        val_target_date = pd.Timestamp("2025-11-15")

        # 1. build_retriever_transitions check
        train_t, val_t, test_t = build_retriever_transitions(poison_df)
        self.assertEqual(len(val_t), 1)
        val_entry = val_t[0]
        self.assertNotIn("POISON_TEST_1", val_entry["positive_product_ids"])
        self.assertNotIn("POISON_TEST_2", val_entry["positive_product_ids"])
        self.assertEqual(val_entry["positive_product_ids"], ["P_VAL_TARGET"])

        # 2. Sequence invariance check
        clean_seq = build_customer_sequence(clean_df, customer_id=cid, as_of_date=val_target_date, max_len=32, enriched=True)
        poison_seq = build_customer_sequence(poison_df, customer_id=cid, as_of_date=val_target_date, max_len=32, enriched=True)

        self.assertNotIn("POISON_TEST_1", poison_seq["product_ids"])
        self.assertNotIn("POISON_TEST_2", poison_seq["product_ids"])
        self.assertTrue(torch.equal(clean_seq["sequence"], poison_seq["sequence"]), "Poison test events leaked into val sequence!")
        self.assertTrue(torch.equal(clean_seq["attention_mask"], poison_seq["attention_mask"]), "Poison test events altered val mask!")

        # 3. Tabular features invariance check
        clean_pit = compute_point_in_time_features(clean_df, customer_id=cid, as_of_date=val_target_date)
        poison_pit = compute_point_in_time_features(poison_df, customer_id=cid, as_of_date=val_target_date)
        for k in clean_pit:
            self.assertEqual(clean_pit[k], poison_pit[k], f"Poison test events altered point-in-time feature {k}")

    def test_07_product_demand_features_point_in_time_invariance(self):
        """Product demand features computed as of TRAIN_CUTOFF must be invariant to any purchases after TRAIN_CUTOFF."""
        pids = ["P1", "P2", "P3"]
        base_records = [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "P1", "quantity": 10.0, "product_category": "CAT_A"},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P2", "quantity": 20.0, "product_category": "CAT_B"},
        ]
        base_df = pd.DataFrame(base_records)

        # Baseline demand features as of 2025-10-01
        base_feats = compute_product_demand_features(base_df, as_of_date=TRAIN_CUTOFF, eligible_product_ids=pids)

        # Add future purchases in val and test periods with massive numbers
        future_records = base_records + [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-11-01"), "product_id": "P3", "quantity": 50000.0, "product_category": "CAT_A"},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2026-03-01"), "product_id": "P1", "quantity": 100000.0, "product_category": "CAT_A"},
        ]
        future_df = pd.DataFrame(future_records)
        future_feats = compute_product_demand_features(future_df, as_of_date=TRAIN_CUTOFF, eligible_product_ids=pids)

        self.assertTrue(torch.allclose(base_feats, future_feats, atol=1e-7), "Product demand features leaked future purchases!")

    def test_08_cold_start_and_zero_history_customer_robustness(self):
        """CustomerTower must cleanly handle cold-start customers with 0 past purchases without NaNs or failures."""
        tower = CustomerTower(
            text_dim=1024,
            output_dim=128,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            seq_feature_dim=ENRICHED_FEATURE_DIM,
            num_tabular_features=8,
            hidden_dim=128,
            dropout=0.0,
        )
        tower.eval()

        B = 4
        text_emb = torch.randn(B, 1024)
        city_id = torch.zeros(B, dtype=torch.long)
        region_id = torch.zeros(B, dtype=torch.long)
        industry_id = torch.zeros(B, dtype=torch.long)
        tab_feats = torch.zeros(B, 8)
        # All zeros for sequence and attention mask (complete cold start)
        seq = torch.zeros(B, 32, ENRICHED_FEATURE_DIM)
        mask = torch.zeros(B, 32, dtype=torch.long)

        u_c = tower(text_emb, city_id, region_id, industry_id, tab_feats, seq, mask)
        self.assertEqual(u_c.shape, (B, 128))
        self.assertTrue(torch.isfinite(u_c).all().item(), "Cold start forward pass produced NaNs/Infs")
        norms = torch.norm(u_c, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Cold start output vector not unit norm")

        # Backward pass check under zero history
        tower.train()
        u_c_train = tower(text_emb, city_id, region_id, industry_id, tab_feats, seq, mask)
        loss = u_c_train.sum()
        loss.backward()

        for name, param in tower.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all().item(), f"NaN gradient in parameter: {name}")


if __name__ == "__main__":
    unittest.main()
