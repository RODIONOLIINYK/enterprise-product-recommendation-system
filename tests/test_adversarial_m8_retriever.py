"""Empirical Adversarial Stress Harness for Milestone M8: Two-Tower Retriever Architecture & Pipeline.

Authored by Challenger 1 (Milestone M8).
Adversarially stresses:
1. Gradient flow across all 3 CustomerTower modules (text_proj, tabular_proj, transformer,
   history_pooling, late_fusion) under both Mode 1 (continuous) and Mode 2 (token IDs).
2. Numerical stability under extreme inputs: empty sequences, all-masked sequences,
   extreme tabular values, batch sizes 1 and 256.
3. L2-normalization invariant: torch.norm(u_c, dim=-1) == 1.0 and torch.norm(v_p, dim=-1) == 1.0 within 1e-5.
4. Full catalogue retrieval scoring: dot products in [-1, 1], temperature-scaled logits,
   bias additions, and top-K sorting across 927 products.
5. Saved retriever checkpoint and metrics validation on disk.
"""

from pathlib import Path
import json
import math
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from recommender.models import (
    CustomerTower,
    ProductTower,
    TwoTowerModel,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEMPERATURE,
    OUTPUT_DIM,
    TEXT_DIM,
)
from recommender.train_retriever import compute_retrieval_metrics


class TestAdversarialM8Retriever(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        np.random.seed(1337)

    def test_gradient_flow_customer_tower_mode1_continuous(self):
        """Mode 1 (num_products=None): gradients must flow cleanly to all 5 modules."""
        tower = CustomerTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            seq_feature_dim=9,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
            num_products=None,
        )

        b_size = 4
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(1, 50, (b_size,))
        region_id = torch.randint(1, 20, (b_size,))
        industry_id = torch.randint(1, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 9)
        mask = torch.ones(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)

        u_c = tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)
        loss = u_c.sum()
        loss.backward()

        modules_to_verify = {
            "text_proj": tower.text_proj,
            "tabular_proj": tower.tabular_proj,
            "event_proj": tower.event_proj,
            "pos_embedding": [tower.pos_embedding],
            "transformer": tower.transformer,
            "history_pooling": tower.history_pooling,
            "late_fusion": tower.late_fusion,
        }

        for name, mod in modules_to_verify.items():
            params = mod if isinstance(mod, list) else list(mod.parameters())
            for i, p in enumerate(params):
                self.assertIsNotNone(p.grad, f"Module {name} param {i} has None gradient!")
                self.assertFalse(torch.isnan(p.grad).any(), f"Module {name} param {i} has NaN gradient!")
                self.assertFalse(torch.isinf(p.grad).any(), f"Module {name} param {i} has Inf gradient!")
                self.assertGreater(
                    p.grad.norm().item(),
                    0.0,
                    f"Module {name} param {i} received strictly zero gradient!",
                )

    def test_gradient_flow_customer_tower_mode2_token_ids(self):
        """Mode 2 (num_products > 0): gradients must flow to product/cat embeddings and token_proj."""
        tower = CustomerTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=25,
            seq_feature_dim=9,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
        )

        b_size = 4
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(1, 50, (b_size,))
        region_id = torch.randint(1, 20, (b_size,))
        industry_id = torch.randint(1, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 9)
        mask = torch.ones(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)
        seq_p = torch.randint(1, 100, (b_size, DEFAULT_SEQUENCE_LENGTH))
        seq_c = torch.randint(1, 25, (b_size, DEFAULT_SEQUENCE_LENGTH))

        u_c = tower(
            text_c,
            city_id,
            region_id,
            industry_id,
            tabular,
            seq,
            mask,
            seq_product_ids=seq_p,
            seq_category_ids=seq_c,
        )
        loss = u_c.sum()
        loss.backward()

        modules_to_verify = {
            "text_proj": tower.text_proj,
            "tabular_proj": tower.tabular_proj,
            "product_embedding": [tower.product_embedding.weight],
            "cat_embedding": [tower.cat_embedding.weight],
            "event_cont_proj": tower.event_cont_proj,
            "token_proj": tower.token_proj,
            "transformer": tower.transformer,
            "history_pooling": tower.history_pooling,
            "late_fusion": tower.late_fusion,
        }

        for name, mod in modules_to_verify.items():
            params = mod if isinstance(mod, list) else list(mod.parameters())
            for i, p in enumerate(params):
                self.assertIsNotNone(p.grad, f"Module {name} param {i} has None gradient!")
                self.assertFalse(torch.isnan(p.grad).any(), f"Module {name} param {i} has NaN gradient!")
                self.assertFalse(torch.isinf(p.grad).any(), f"Module {name} param {i} has Inf gradient!")
                self.assertGreater(
                    p.grad.norm().item(),
                    0.0,
                    f"Module {name} param {i} received strictly zero gradient!",
                )

    def test_gradient_flow_all_masked_and_mixed_history(self):
        """Mixed batches (active + empty history) and 100% all-masked batches must not produce NaNs."""
        tower = CustomerTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
        )

        b_size = 4
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(1, 50, (b_size,))
        region_id = torch.randint(1, 20, (b_size,))
        industry_id = torch.randint(1, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 4)

        # 1. Mixed batch: Sample 0 has 0 events, Sample 1 has 1 event, Sample 2 has 5, Sample 3 has 32
        mask_mixed = torch.zeros(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)
        mask_mixed[1, :1] = 1
        mask_mixed[2, :5] = 1
        mask_mixed[3, :] = 1

        u_c_mixed = tower(text_c, city_id, region_id, industry_id, tabular, seq, mask_mixed)
        self.assertFalse(torch.isnan(u_c_mixed).any())
        self.assertFalse(torch.isinf(u_c_mixed).any())
        norms = torch.norm(u_c_mixed, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

        loss_mixed = u_c_mixed.sum()
        loss_mixed.backward()
        for name, p in tower.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Grad None in mixed batch for {name}")
                self.assertFalse(torch.isnan(p.grad).any(), f"Grad NaN in mixed batch for {name}")

        # 2. 100% All-masked batch (all samples have zero history)
        tower.zero_grad()
        mask_all_zeros = torch.zeros(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)
        u_c_empty = tower(text_c, city_id, region_id, industry_id, tabular, seq, mask_all_zeros)

        self.assertFalse(torch.isnan(u_c_empty).any(), "NaN found in u_c for all-masked history!")
        self.assertFalse(torch.isinf(u_c_empty).any(), "Inf found in u_c for all-masked history!")
        norms_empty = torch.norm(u_c_empty, p=2, dim=-1)
        torch.testing.assert_close(norms_empty, torch.ones_like(norms_empty), atol=1e-5, rtol=1e-5)

        loss_empty = u_c_empty.sum()
        loss_empty.backward()
        for name, p in tower.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Grad None in all-masked batch for {name}")
                self.assertFalse(torch.isnan(p.grad).any(), f"Grad NaN in all-masked batch for {name}")

    def test_numerical_stability_extreme_tabular_inputs(self):
        """CustomerTower must remain stable under extreme numeric features."""
        tower = CustomerTower(hidden_dim=64, num_heads=2, num_layers=1)

        b_size = 4
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (b_size,))
        region_id = torch.randint(0, 20, (b_size,))
        industry_id = torch.randint(0, 30, (b_size,))
        seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 4)
        mask = torch.ones(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)

        scenarios = {
            "all_zeros": torch.zeros(b_size, 8),
            "extreme_large_positive": torch.full((b_size, 8), 1e6),
            "extreme_large_negative": torch.full((b_size, 8), -1e6),
            "near_zero_epsilon": torch.full((b_size, 8), 1e-8),
            "mixed_outliers": torch.tensor([
                [1e6, -1e6, 0.0, 1e-5, 500.0, -20.0, 1e4, 0.0],
                [0.0, 0.0, 1e7, -1e7, 1e-8, 1e2, -1e2, 1e6],
                [-1e5, 1e5, 1e5, -1e5, 0.0, 0.0, 1e-4, 1e4],
                [1e-9, -1e-9, 1e6, 1e-6, 1e3, -1e3, 0.0, 1e6],
            ]),
        }

        for name, tab in scenarios.items():
            u_c = tower(text_c, city_id, region_id, industry_id, tab, seq, mask)
            self.assertFalse(torch.isnan(u_c).any(), f"NaN in u_c under {name}")
            self.assertFalse(torch.isinf(u_c).any(), f"Inf in u_c under {name}")
            norms = torch.norm(u_c, p=2, dim=-1)
            torch.testing.assert_close(
                norms,
                torch.ones_like(norms),
                atol=1e-5,
                rtol=1e-5,
                msg=f"L2 norm violated under {name}",
            )

    def test_numerical_stability_batch_size_1_and_256(self):
        """Batch sizes 1 and 256 must process cleanly with unit L2 norm."""
        c_tower = CustomerTower(hidden_dim=64, num_heads=2, num_layers=1)
        p_tower = ProductTower(hidden_dim=64)
        model = TwoTowerModel(c_tower, p_tower, num_catalog_products=927)

        for b_size in [1, 256]:
            text_c = torch.randn(b_size, TEXT_DIM)
            city_id = torch.randint(0, 50, (b_size,))
            region_id = torch.randint(0, 20, (b_size,))
            industry_id = torch.randint(0, 30, (b_size,))
            tabular = torch.randn(b_size, 8)
            seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 4)
            mask = torch.randint(0, 2, (b_size, DEFAULT_SEQUENCE_LENGTH))

            u_c = c_tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)
            self.assertEqual(u_c.shape, (b_size, OUTPUT_DIM))
            self.assertFalse(torch.isnan(u_c).any())
            self.assertFalse(torch.isinf(u_c).any())
            torch.testing.assert_close(
                torch.norm(u_c, p=2, dim=-1),
                torch.ones(b_size),
                atol=1e-5,
                rtol=1e-5,
            )

            # Test ProductTower with same batch size
            text_p = torch.randn(b_size, TEXT_DIM)
            brand_id = torch.randint(0, 20, (b_size,))
            cat_id = torch.randint(0, 20, (b_size,))
            subcat_id = torch.randint(0, 40, (b_size,))
            line_id = torch.randint(0, 15, (b_size,))
            demand = torch.rand(b_size, 6)

            v_p = p_tower(text_p, brand_id, cat_id, subcat_id, line_id, demand)
            self.assertEqual(v_p.shape, (b_size, OUTPUT_DIM))
            self.assertFalse(torch.isnan(v_p).any())
            self.assertFalse(torch.isinf(v_p).any())
            torch.testing.assert_close(
                torch.norm(v_p, p=2, dim=-1),
                torch.ones(b_size),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_product_tower_gradient_flow_and_extremes(self):
        """ProductTower must propagate non-zero gradients and survive extreme demand features."""
        p_tower = ProductTower(hidden_dim=64)
        p_count = 16

        text_p = torch.randn(p_count, TEXT_DIM)
        brand_id = torch.randint(1, 20, (p_count,))
        cat_id = torch.randint(1, 20, (p_count,))
        subcat_id = torch.randint(1, 40, (p_count,))
        line_id = torch.randint(1, 15, (p_count,))
        demand = torch.rand(p_count, 6)

        v_p = p_tower(text_p, brand_id, cat_id, subcat_id, line_id, demand)
        loss = v_p.sum()
        loss.backward()

        modules = {
            "brand_emb": [p_tower.brand_embedding.weight],
            "cat_emb": [p_tower.cat_embedding.weight],
            "subcat_emb": [p_tower.subcat_embedding.weight],
            "line_emb": [p_tower.line_embedding.weight],
            "text_proj": p_tower.text_proj,
            "mlp": p_tower.mlp,
        }

        for name, mod in modules.items():
            params = mod if isinstance(mod, list) else list(mod.parameters())
            for i, p in enumerate(params):
                self.assertIsNotNone(p.grad, f"ProductTower {name} param {i} has None gradient!")
                self.assertFalse(torch.isnan(p.grad).any(), f"ProductTower {name} param {i} has NaN gradient!")
                self.assertGreater(p.grad.norm().item(), 0.0, f"ProductTower {name} param {i} zero gradient!")

        # Demand feature extremes
        extremes = [
            torch.zeros(p_count, 6),
            torch.ones(p_count, 6),
            torch.full((p_count, 6), 1e6),
            torch.full((p_count, 6), -1e6),
        ]
        for ext in extremes:
            v_ext = p_tower(text_p, brand_id, cat_id, subcat_id, line_id, ext)
            self.assertFalse(torch.isnan(v_ext).any())
            self.assertFalse(torch.isinf(v_ext).any())
            torch.testing.assert_close(
                torch.norm(v_ext, p=2, dim=-1),
                torch.ones(p_count),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_retrieval_scoring_927_catalogue_products(self):
        """Retrieval scoring across 927 catalogue products must satisfy exact mathematical invariants."""
        c_tower = CustomerTower(hidden_dim=64, num_heads=2, num_layers=1)
        p_tower = ProductTower(hidden_dim=64)
        num_catalog = 927
        temperature = 0.1
        model = TwoTowerModel(c_tower, p_tower, num_catalog_products=num_catalog, temperature=temperature)

        b_size = 16
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (b_size,))
        region_id = torch.randint(0, 20, (b_size,))
        industry_id = torch.randint(0, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, DEFAULT_SEQUENCE_LENGTH, 4)
        mask = torch.ones(b_size, DEFAULT_SEQUENCE_LENGTH, dtype=torch.long)

        u_c = c_tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)

        text_p = torch.randn(num_catalog, TEXT_DIM)
        brand_id = torch.randint(0, 20, (num_catalog,))
        cat_id = torch.randint(0, 20, (num_catalog,))
        subcat_id = torch.randint(0, 40, (num_catalog,))
        line_id = torch.randint(0, 15, (num_catalog,))
        demand = torch.rand(num_catalog, 6)

        v_p = p_tower(text_p, brand_id, cat_id, subcat_id, line_id, demand)

        # 1. Cosine similarity / dot product invariant: u_c . v_p in [-1, 1]
        raw_dot = torch.matmul(u_c, v_p.t())
        self.assertEqual(raw_dot.shape, (b_size, num_catalog))
        self.assertGreaterEqual(raw_dot.min().item(), -1.0 - 1e-5)
        self.assertLessEqual(raw_dot.max().item(), 1.0 + 1e-5)

        # 2. Scaled logits without bias in [-1/tau, 1/tau] = [-10, 10]
        logits_no_bias = model.compute_logits(u_c, v_p)
        self.assertEqual(logits_no_bias.shape, (b_size, num_catalog))
        self.assertGreaterEqual(logits_no_bias.min().item(), -10.0 - 1e-4)
        self.assertLessEqual(logits_no_bias.max().item(), 10.0 + 1e-4)

        # 3. Scaled logits with product bias
        p_indices = torch.arange(num_catalog)
        logits_with_bias = model.compute_logits(u_c, v_p, product_indices=p_indices)
        self.assertEqual(logits_with_bias.shape, (b_size, num_catalog))
        self.assertFalse(torch.isnan(logits_with_bias).any())
        self.assertFalse(torch.isinf(logits_with_bias).any())

        # 4. Top-K sorting monotonicity
        for k in [10, 50, 100]:
            top_vals, top_inds = torch.topk(logits_with_bias, k=k, dim=-1)
            self.assertEqual(top_vals.shape, (b_size, k))
            self.assertEqual(top_inds.shape, (b_size, k))
            # Verify values are non-increasing along dimension 1
            diffs = top_vals[:, :-1] - top_vals[:, 1:]
            self.assertTrue((diffs >= -1e-6).all(), f"Top-{k} logits are not monotonic!")

    def test_retrieval_metrics_adversarial_edge_cases(self):
        """compute_retrieval_metrics must handle empty inputs, perfect rankings, and zero hits."""
        # Edge 1: Empty samples
        empty_res = compute_retrieval_metrics(np.zeros((0, 10)), np.zeros((0, 10)))
        self.assertEqual(empty_res["recall@10"], 0.0)
        self.assertEqual(empty_res["ndcg@10"], 0.0)
        self.assertEqual(empty_res["mrr"], 0.0)

        # Edge 2: Perfect retrieval (top-1 is target)
        perfect_preds = np.array([[10.0, 5.0, 1.0]])
        perfect_tgts = np.array([[1.0, 0.0, 0.0]])
        perf_res = compute_retrieval_metrics(perfect_preds, perfect_tgts, k_list=(1, 2, 3))
        self.assertEqual(perf_res["recall@1"], 1.0)
        self.assertEqual(perf_res["ndcg@10"], 1.0)
        self.assertEqual(perf_res["mrr"], 1.0)

        # Edge 3: Complete miss (target not retrieved in top K)
        miss_preds = np.array([[10.0, 5.0, 1.0, 0.0]])
        miss_tgts = np.array([[0.0, 0.0, 0.0, 1.0]])
        miss_res = compute_retrieval_metrics(miss_preds, miss_tgts, k_list=(1, 2))
        self.assertEqual(miss_res["recall@1"], 0.0)
        self.assertEqual(miss_res["recall@2"], 0.0)
        self.assertEqual(miss_res["mrr"], 1.0 / 4.0)  # item at index 3 is rank 4

    def test_saved_checkpoint_reproducibility_and_inspection(self):
        """Verify saved best_retriever.pt checkpoint and training_history.json on disk."""
        ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
        hist_path = Path("checkpoints/retriever/training_history.json")

        self.assertTrue(ckpt_path.is_file(), "checkpoints/retriever/best_retriever.pt does not exist!")
        self.assertTrue(hist_path.is_file(), "checkpoints/retriever/training_history.json does not exist!")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertEqual(len(ckpt["product_ids"]), 927, "Must contain exactly 927 eligible catalogue products!")
        self.assertIn("val_metrics", ckpt)

        val_metrics = ckpt["val_metrics"]
        rec50 = val_metrics.get("recall@50", 0.0)
        rec100 = val_metrics.get("recall@100", 0.0)
        ndcg10 = val_metrics.get("ndcg@10", 0.0)
        mrr = val_metrics.get("mrr", 0.0)

        # Assert meaningful quality thresholds
        self.assertGreater(rec50, 0.70, f"Recall@50 must be > 0.70 (got {rec50:.4f})")
        self.assertGreater(rec100, 0.80, f"Recall@100 must be > 0.80 (got {rec100:.4f})")
        self.assertGreater(ndcg10, 0.25, f"NDCG@10 must be > 0.25 (got {ndcg10:.4f})")
        self.assertGreater(mrr, 0.25, f"MRR must be > 0.25 (got {mrr:.4f})")

        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        self.assertEqual(len(history), 10, "Training history must have exactly 10 epochs!")
        first_epoch = history[0]
        last_epoch = history[-1]

        # Training loss should decrease from epoch 1 to epoch 10
        self.assertLess(
            last_epoch["train_loss"],
            first_epoch["train_loss"],
            "Train loss did not decrease over training epochs!",
        )
        # Validation Recall@50 should increase from epoch 1 to epoch 10
        self.assertGreater(
            last_epoch["val_recall@50"],
            first_epoch["val_recall@50"],
            "Val Recall@50 did not improve over training epochs!",
        )


if __name__ == "__main__":
    unittest.main()
