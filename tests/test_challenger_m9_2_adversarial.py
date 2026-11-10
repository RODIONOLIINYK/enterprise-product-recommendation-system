"""Adversarial and Empirical Verification Test Suite for Milestone M9.

Author: Reviewer / Challenger M9_2
Evaluates:
1. CandidateRetriever checkpoint loading contracts (best_retriever.pt, weights, product count).
2. CandidateBatch contracts: shapes, device placement, dict-like indexing, and serializability.
3. Strict point-in-time boundary isolation: bisect_left exact timestamp exclusion.
4. Sub-millisecond latency SLA (< 1.0 ms per 50-candidate slate).
5. Edge cases: empty customer history, cold products, single-order customers, extreme time gaps.
6. Zero-gap and multi-order median purchase gap handling without division by zero.
"""

from pathlib import Path
import time
import unittest
import pytest

import numpy as np
import pandas as pd
import torch

from recommender.candidates import (
    NUM_PAIRWISE_FEATURES,
    PAIRWISE_FEATURE_NAMES,
    CandidateBatch,
    CandidateRetriever,
    PointInTimeFeatureStore,
    construct_candidate_slate,
    load_retriever_from_checkpoint,
)
from recommender.models import TwoTowerModel


class TestChallengerM9AdversarialVerification(unittest.TestCase):
    """Adversarial challenge test suite for Milestone M9."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
        cls.purchases_path = Path("data/interim/cleaned_purchases.csv")
        cls.has_ckpt = cls.ckpt_path.is_file()
        cls.has_purchases = cls.purchases_path.is_file()

    def test_01_checkpoint_loading_and_contract(self):
        """Verify best_retriever.pt loads into TwoTowerModel with complete metadata contract."""
        if not self.has_ckpt:
            self.skipTest("best_retriever.pt not present")

        model, ckpt = load_retriever_from_checkpoint(self.ckpt_path)
        self.assertIsInstance(model, TwoTowerModel)
        self.assertIn("product_ids", ckpt)
        self.assertEqual(len(ckpt["product_ids"]), model.product_bias.numel())

        # Check required metadata keys
        required_keys = [
            "epoch",
            "model_state_dict",
            "best_recall_50",
            "product_ids",
            "val_metrics",
            "config",
        ]
        for k in required_keys:
            self.assertIn(k, ckpt, f"Checkpoint missing required key: {k}")

        self.assertGreater(ckpt["best_recall_50"], 0.70)
        self.assertIn("recall@100", ckpt["val_metrics"])

    def test_02_candidate_batch_dict_access_and_device_transfer(self):
        """Verify CandidateBatch dict protocol, attribute access, and device movement."""
        b, c = 3, 50
        pids = [[f"P_{i}" for i in range(c)] for _ in range(b)]
        batch = CandidateBatch(
            customer_ids=["C1", "C2", "C3"],
            candidate_pids=pids,
            candidate_indices=torch.randint(0, 900, (b, c)),
            candidate_scores=torch.randn(b, c),
            pairwise_features=torch.rand(b, c, NUM_PAIRWISE_FEATURES),
            labels=torch.zeros(b, c),
            top100_indices=torch.randint(0, 900, (b, 100)),
            top100_scores=torch.randn(b, 100),
            top100_pids=[[f"P_{i}" for i in range(100)] for _ in range(b)],
        )

        # 1. Attribute access
        self.assertEqual(batch.customer_ids, ["C1", "C2", "C3"])
        self.assertEqual(batch.pairwise_features.shape, (b, c, NUM_PAIRWISE_FEATURES))

        # 2. Dict indexing
        self.assertEqual(batch["pairwise_features"].shape, (b, c, NUM_PAIRWISE_FEATURES))
        self.assertEqual(batch["candidate_indices"].shape, (b, c))
        self.assertIn("candidate_scores", batch)
        self.assertEqual(batch.get("customer_ids"), ["C1", "C2", "C3"])
        self.assertEqual(batch.get("unknown_key", "default_val"), "default_val")

        with self.assertRaises(KeyError):
            _ = batch["invalid_key"]

        # 3. to_dict
        d = batch.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["candidate_indices"].shape, (b, c))

        # 4. Device transfer
        cpu_batch = batch.to("cpu")
        self.assertEqual(cpu_batch.pairwise_features.device.type, "cpu")
        self.assertEqual(cpu_batch.candidate_scores.device.type, "cpu")

        if torch.backends.mps.is_available():
            mps_batch = batch.to("mps")
            self.assertEqual(mps_batch.pairwise_features.device.type, "mps")
            self.assertEqual(mps_batch.candidate_indices.device.type, "mps")

    def test_03_strict_point_in_time_boundary(self):
        """Stress-test binary search prefix: exact timestamp and future events must NOT leak."""
        events = pd.DataFrame([
            {"customer_id": "C_TEST", "product_id": "P_PAST", "purchase_date": "2025-06-01 09:00:00", "quantity": 3.0},
            {"customer_id": "C_TEST", "product_id": "P_BOUNDARY", "purchase_date": "2025-06-01 12:00:00", "quantity": 2.0},
            {"customer_id": "C_TEST", "product_id": "P_FUTURE", "purchase_date": "2025-06-01 12:00:01", "quantity": 1.0},
        ])
        store = PointInTimeFeatureStore(purchases_df=events)

        # Snapshot at exact boundary of P_BOUNDARY: must include P_PAST, but NOT P_BOUNDARY
        snap_boundary = store.get_customer_snapshot("C_TEST", "2025-06-01 12:00:00")
        self.assertEqual(snap_boundary.total_orders, 1)
        self.assertEqual(snap_boundary.total_volume, 3.0)
        self.assertIn("P_PAST", snap_boundary.sku_stats)
        self.assertNotIn("P_BOUNDARY", snap_boundary.sku_stats)
        self.assertNotIn("P_FUTURE", snap_boundary.sku_stats)

        # Snapshot 1 second later: includes P_PAST and P_BOUNDARY, but NOT P_FUTURE
        snap_after = store.get_customer_snapshot("C_TEST", "2025-06-01 12:00:01")
        self.assertEqual(snap_after.total_orders, 2)
        self.assertEqual(snap_after.total_volume, 5.0)
        self.assertIn("P_BOUNDARY", snap_after.sku_stats)
        self.assertNotIn("P_FUTURE", snap_after.sku_stats)

    @pytest.mark.real_purchase_data
    def test_04_submillisecond_pairwise_latency_sla(self):
        """Verify 50-candidate slate feature extraction latency SLA strictly < 1.0 ms."""
        if not self.has_purchases:
            self.skipTest("cleaned_purchases.csv not present")

        df = pd.read_csv(self.purchases_path, nrows=5000)
        store = PointInTimeFeatureStore(purchases_df=df)

        cids = df["customer_id"].astype(str).unique()[:20]
        pids = df["product_id"].astype(str).unique()[:50].tolist()
        as_of = pd.Timestamp("2025-10-15")

        latencies = []
        for cid in cids:
            t0 = time.perf_counter()
            _ = store.compute_slate_pairwise_features(cid, as_of, pids)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        self.assertLess(avg_latency, 1.0, f"Average latency {avg_latency:.4f} ms exceeds 1.0 ms SLA")
        self.assertLess(p95_latency, 1.0, f"P95 latency {p95_latency:.4f} ms exceeds 1.0 ms SLA")

    def test_05_cold_customer_and_cold_products(self):
        """Stress-test completely cold customer and never-purchased products."""
        empty_store = PointInTimeFeatureStore(
            purchases_df=pd.DataFrame(columns=["customer_id", "product_id", "purchase_date", "quantity"])
        )

        feats = empty_store.compute_pairwise_features(
            customer_id="NON_EXISTENT_CUSTOMER",
            as_of_date="2026-01-01",
            product_id="NON_EXISTENT_PRODUCT",
        )

        self.assertEqual(feats.shape, (NUM_PAIRWISE_FEATURES,))
        self.assertFalse(np.isnan(feats).any())
        self.assertTrue((feats >= 0.0).all() and (feats <= 1.0).all())
        # Check specific semantics for cold customer / SKU
        self.assertEqual(feats[0], 0.0)  # is_repeat_sku
        self.assertEqual(feats[1], 0.0)  # user_sku_order_count_norm
        self.assertEqual(feats[2], 0.0)  # user_sku_qty_share
        self.assertEqual(feats[3], 1.0)  # days_since_last_sku_buy_norm (1.0 if never bought)
        self.assertEqual(feats[4], 0.0)  # sku_urgency (0.0 if < 2 orders)

    def test_06_single_order_customer_contract(self):
        """Verify customer with exactly 1 order has urgency=0.0 and correct repeat stats."""
        df = pd.DataFrame([
            {"customer_id": "C_SINGLE", "product_id": "P_BOUGHT", "purchase_date": "2025-03-01", "quantity": 10.0}
        ])
        store = PointInTimeFeatureStore(purchases_df=df)
        as_of = pd.Timestamp("2025-03-21")  # 20 days later

        # Repeat product P_BOUGHT
        feats_bought = store.compute_pairwise_features("C_SINGLE", as_of, "P_BOUGHT")
        self.assertEqual(feats_bought[0], 1.0)  # is_repeat_sku
        self.assertEqual(feats_bought[2], 1.0)  # 100% quantity share
        self.assertEqual(feats_bought[4], 0.0)  # urgency MUST be 0.0 for < 2 orders!

        # Unbought product P_OTHER
        feats_other = store.compute_pairwise_features("C_SINGLE", as_of, "P_OTHER")
        self.assertEqual(feats_other[0], 0.0)
        self.assertEqual(feats_other[3], 1.0)  # 1.0 if never bought
        self.assertEqual(feats_other[4], 0.0)

    def test_07_extreme_gaps_and_zero_median_gap(self):
        """Verify zero-gap repeat purchases and 10-year gaps do not produce NaNs or exceed bounds."""
        df = pd.DataFrame([
            # Two purchases on the exact same timestamp
            {"customer_id": "C_SAME", "product_id": "P_ZERO_GAP", "purchase_date": "2025-01-01 10:00:00", "quantity": 1.0},
            {"customer_id": "C_SAME", "product_id": "P_ZERO_GAP", "purchase_date": "2025-01-01 10:00:00", "quantity": 1.0},
            # 10-year old purchases
            {"customer_id": "C_ANCIENT", "product_id": "P_ANCIENT", "purchase_date": "2015-01-01 00:00:00", "quantity": 5.0},
            {"customer_id": "C_ANCIENT", "product_id": "P_ANCIENT", "purchase_date": "2015-01-10 00:00:00", "quantity": 5.0},
        ])
        store = PointInTimeFeatureStore(purchases_df=df)

        # Zero median gap customer
        feats_zero = store.compute_pairwise_features("C_SAME", "2025-01-10 00:00:00", "P_ZERO_GAP")
        self.assertFalse(np.isnan(feats_zero).any())
        self.assertTrue((feats_zero >= 0.0).all() and (feats_zero <= 1.0).all())

        # 10-year ancient gap customer
        feats_ancient = store.compute_pairwise_features("C_ANCIENT", "2025-01-01 00:00:00", "P_ANCIENT")
        self.assertFalse(np.isnan(feats_ancient).any())
        self.assertEqual(feats_ancient[3], 1.0)  # days_ago capped at 1.0
        self.assertEqual(feats_ancient[4], 1.0)  # urgency capped at 1.0

    def test_09_evaluation_slate_strict_isolation(self):
        """Verify evaluation slates strictly contain top 50 without positive injection."""
        n_p = 100
        pids = [f"P_{i:03d}" for i in range(n_p)]
        p2i = {p: i for i, p in enumerate(pids)}
        logits = torch.linspace(100.0, 1.0, n_p)

        # Positive ranked at position 75
        slate_eval = construct_candidate_slate(
            stage1_logits=logits,
            product_ids=pids,
            positive_product_ids=["P_075"],
            top_c=50,
        )
        self.assertNotIn("P_075", slate_eval["candidate_pids"])
        self.assertEqual(slate_eval["candidate_pids"], pids[:50])
        self.assertEqual(slate_eval["candidate_ranks"], list(range(50)))
        self.assertEqual(sum(slate_eval["labels"]), 0.0)


if __name__ == "__main__":
    unittest.main()
