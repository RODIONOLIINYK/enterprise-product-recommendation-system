"""Unit tests for Two-Tower Retrieval Model (Customer Tower with Solution A + Product Tower)."""

import unittest
import torch
import torch.nn.functional as F

from recommender.models import CustomerTower, ProductTower, TwoTowerModel, OUTPUT_DIM, TEXT_DIM


class TestTwoTowerModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.num_products = 10
        self.seq_len = 32
        self.seq_feature_dim = 4

        self.customer_tower = CustomerTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            seq_feature_dim=self.seq_feature_dim,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
        )

        self.product_tower = ProductTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_brands=20,
            num_categories=20,
            num_subcategories=40,
            num_business_lines=15,
            num_demand_features=6,
            hidden_dim=64,
        )

        self.model = TwoTowerModel(
            customer_tower=self.customer_tower,
            product_tower=self.product_tower,
            num_catalog_products=self.num_products,
        )

    def test_customer_tower_output_shape_and_unit_norm(self):
        """Customer Tower must output (B, 128) with unit L2 norm."""
        text_emb = torch.randn(self.batch_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (self.batch_size,))
        region_id = torch.randint(0, 20, (self.batch_size,))
        industry_id = torch.randint(0, 30, (self.batch_size,))
        tabular = torch.randn(self.batch_size, 8)
        seq = torch.randn(self.batch_size, self.seq_len, self.seq_feature_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

        u_c = self.customer_tower(
            text_embedding=text_emb,
            city_id=city_id,
            region_id=region_id,
            industry_id=industry_id,
            tabular_features=tabular,
            sequence=seq,
            attention_mask=mask,
        )

        self.assertEqual(u_c.shape, (self.batch_size, OUTPUT_DIM))
        norms = torch.norm(u_c, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.isnan(u_c).any())

    def test_customer_tower_handles_empty_history_without_nan(self):
        """Customers with zero past purchases (empty history) must not produce NaNs."""
        text_emb = torch.randn(self.batch_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (self.batch_size,))
        region_id = torch.randint(0, 20, (self.batch_size,))
        industry_id = torch.randint(0, 30, (self.batch_size,))
        tabular = torch.randn(self.batch_size, 8)
        seq = torch.zeros(self.batch_size, self.seq_len, self.seq_feature_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)  # All padding

        u_c = self.customer_tower(
            text_embedding=text_emb,
            city_id=city_id,
            region_id=region_id,
            industry_id=industry_id,
            tabular_features=tabular,
            sequence=seq,
            attention_mask=mask,
        )

        self.assertEqual(u_c.shape, (self.batch_size, OUTPUT_DIM))
        self.assertFalse(torch.isnan(u_c).any())
        norms = torch.norm(u_c, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_product_tower_output_shape_and_unit_norm(self):
        """Product Tower must output (P, 128) with unit L2 norm."""
        text_emb = torch.randn(self.num_products, TEXT_DIM)
        brand_id = torch.randint(0, 20, (self.num_products,))
        cat_id = torch.randint(0, 20, (self.num_products,))
        subcat_id = torch.randint(0, 40, (self.num_products,))
        line_id = torch.randint(0, 15, (self.num_products,))
        demand = torch.randn(self.num_products, 6)

        v_p = self.product_tower(
            text_embedding=text_emb,
            brand_id=brand_id,
            category_id=cat_id,
            subcategory_id=subcat_id,
            business_line_id=line_id,
            demand_features=demand,
        )

        self.assertEqual(v_p.shape, (self.num_products, OUTPUT_DIM))
        norms = torch.norm(v_p, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.isnan(v_p).any())

    def test_two_tower_scoring_and_topk_retrieval(self):
        """Model must compute logits (B, P) and retrieve top-K candidates."""
        u_c = torch.randn(self.batch_size, OUTPUT_DIM)
        u_c = F.normalize(u_c, p=2, dim=-1)

        v_p = torch.randn(self.num_products, OUTPUT_DIM)
        v_p = F.normalize(v_p, p=2, dim=-1)

        p_indices = torch.arange(self.num_products)
        logits = self.model.compute_logits(u_c, v_p, product_indices=p_indices)

        self.assertEqual(logits.shape, (self.batch_size, self.num_products))
        self.assertFalse(torch.isnan(logits).any())

        # Top-K retrieval
        top_k_scores, top_k_indices = torch.topk(logits, k=5, dim=-1)
        self.assertEqual(top_k_indices.shape, (self.batch_size, 5))
        self.assertEqual(top_k_scores.shape, (self.batch_size, 5))

    def test_multi_label_bce_loss_and_gradient_flow(self):
        """Multi-label BCE loss must be scalar and propagate finite gradients."""
        text_c = torch.randn(self.batch_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (self.batch_size,))
        region_id = torch.randint(0, 20, (self.batch_size,))
        industry_id = torch.randint(0, 30, (self.batch_size,))
        tabular = torch.randn(self.batch_size, 8)
        seq = torch.randn(self.batch_size, self.seq_len, self.seq_feature_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

        u_c = self.customer_tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)

        text_p = torch.randn(self.num_products, TEXT_DIM)
        brand_id = torch.randint(0, 20, (self.num_products,))
        cat_id = torch.randint(0, 20, (self.num_products,))
        subcat_id = torch.randint(0, 40, (self.num_products,))
        line_id = torch.randint(0, 15, (self.num_products,))
        demand = torch.randn(self.num_products, 6)

        v_p = self.product_tower(text_p, brand_id, cat_id, subcat_id, line_id, demand)

        p_indices = torch.arange(self.num_products)
        logits = self.model.compute_logits(u_c, v_p, product_indices=p_indices)

        # Multi-label binary targets: some items 1.0 (in basket), others 0.0
        targets = torch.zeros(self.batch_size, self.num_products)
        targets[0, 1] = 1.0
        targets[0, 3] = 1.0
        targets[1, 2] = 1.0

        loss = self.model.compute_multi_label_loss(logits, targets)
        self.assertTrue(loss.item() > 0.0)
        self.assertFalse(torch.isnan(loss))

        # Backward pass
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_modular_customer_tower_late_fusion_components(self):
        """Verify presence and functionality of Modular Customer Tower late fusion components."""
        # 1. Check all modules exist
        self.assertTrue(hasattr(self.customer_tower, "text_proj"))
        self.assertTrue(hasattr(self.customer_tower, "tabular_proj"))
        self.assertTrue(hasattr(self.customer_tower, "transformer"))
        self.assertTrue(hasattr(self.customer_tower, "history_pooling"))
        self.assertTrue(hasattr(self.customer_tower, "late_fusion"))

        # 2. Forward pass through customer tower
        text_c = torch.randn(self.batch_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (self.batch_size,))
        region_id = torch.randint(0, 20, (self.batch_size,))
        industry_id = torch.randint(0, 30, (self.batch_size,))
        tabular = torch.randn(self.batch_size, 8)
        seq = torch.randn(self.batch_size, self.seq_len, self.seq_feature_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

        u_c = self.customer_tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)
        self.assertEqual(u_c.shape, (self.batch_size, OUTPUT_DIM))
        norms = torch.norm(u_c, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_customer_tower_last_event_pooling_logic(self):
        """Verify masked mean + last-event pooling across varied sequence lengths (1, 5, 32, 0)."""
        b_size = 4
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (b_size,))
        region_id = torch.randint(0, 20, (b_size,))
        industry_id = torch.randint(0, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, 32, self.seq_feature_dim)

        # Varied sequence lengths: sample 0 has 1 event, sample 1 has 5 events, sample 2 has 32, sample 3 has 0
        mask = torch.zeros(b_size, 32, dtype=torch.long)
        mask[0, :1] = 1
        mask[1, :5] = 1
        mask[2, :32] = 1
        mask[3, :] = 0  # Empty history

        u_c = self.customer_tower(text_c, city_id, region_id, industry_id, tabular, seq, mask)
        self.assertEqual(u_c.shape, (b_size, OUTPUT_DIM))
        self.assertFalse(torch.isnan(u_c).any())
        norms = torch.norm(u_c, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_customer_tower_with_enriched_token_ids(self):
        """Customer Tower instantiated with product vocabularies must support token IDs and gradient flow."""
        enriched_tower = CustomerTower(
            text_dim=TEXT_DIM,
            output_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            seq_feature_dim=9,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
        )

        b_size = 3
        text_c = torch.randn(b_size, TEXT_DIM)
        city_id = torch.randint(0, 50, (b_size,))
        region_id = torch.randint(0, 20, (b_size,))
        industry_id = torch.randint(0, 30, (b_size,))
        tabular = torch.randn(b_size, 8)
        seq = torch.randn(b_size, 32, 9)
        mask = torch.ones(b_size, 32, dtype=torch.long)
        seq_pids = torch.randint(1, 100, (b_size, 32))
        seq_cids = torch.randint(1, 20, (b_size, 32))

        u_c = enriched_tower(
            text_c,
            city_id,
            region_id,
            industry_id,
            tabular,
            seq,
            mask,
            seq_product_ids=seq_pids,
            seq_category_ids=seq_cids,
        )
        self.assertEqual(u_c.shape, (b_size, OUTPUT_DIM))
        norms = torch.norm(u_c, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

        loss = u_c.sum()
        loss.backward()

        for name, param in enriched_tower.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing in enriched tower for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_catalogue_history_pooling_uses_shared_vectors(self):
        tower = CustomerTower(catalog_history_encoder="pool", history_gap_reference_days=30.0, dropout=0.0)
        self.assertIsNone(tower.transformer)
        catalog_vectors = torch.randn(3, OUTPUT_DIM, requires_grad=True)
        tokens = torch.zeros(1, 3, 9)
        dates = torch.tensor([[-60.0, -30.0, 0.0]]) / 365.25
        ids = torch.zeros(1, dtype=torch.long)
        args = (torch.zeros(1, TEXT_DIM), ids, ids, ids, torch.zeros(1, 8))
        mask = torch.tensor([[1, 1, 0]])
        kwargs = {
            "seq_product_ids": torch.tensor([[1, 3, 0]]),
            "catalog_product_vectors": catalog_vectors,
            "seq_purchase_dates": dates,
        }

        projected_inputs = []
        hook = tower.token_proj[0].register_forward_pre_hook(lambda _layer, values: projected_inputs.append(values[0]))
        vector = tower(*args, tokens, mask, **kwargs)
        hook.remove()
        torch.testing.assert_close(projected_inputs[0][0, :, -1], torch.tensor([0.5, 0.0, 0.0]))
        vector.sum().backward()
        self.assertGreater(catalog_vectors.grad[0].abs().sum().item(), 0)
        self.assertEqual(catalog_vectors.grad[1].abs().sum().item(), 0)
        self.assertGreater(catalog_vectors.grad[2].abs().sum().item(), 0)

        with torch.no_grad():
            tokens[0, 2, 0] = 1.0
            torch.testing.assert_close(tower(*args, tokens, mask, **kwargs), vector)
            tokens[0, 1, 0] = 1.0
            self.assertFalse(torch.allclose(tower(*args, tokens, mask, **kwargs), vector))

    def test_two_tower_weighted_bce_loss(self):
        """TwoTowerModel must support pos_weight in compute_multi_label_loss."""
        logits = torch.randn(self.batch_size, self.num_products)
        targets = torch.zeros(self.batch_size, self.num_products)
        targets[0, 1] = 1.0
        targets[1, 3] = 1.0

        pos_weight_val = 20.0
        loss_weighted = self.model.compute_multi_label_loss(logits, targets, pos_weight=pos_weight_val)
        expected = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor([pos_weight_val])
        )
        torch.testing.assert_close(loss_weighted, expected)
        self.assertTrue(loss_weighted.item() > 0.0)

        # Also test tensor pos_weight
        pw_tensor = torch.full((self.num_products,), 20.0)
        loss_tensor = self.model.compute_multi_label_loss(logits, targets, pos_weight=pw_tensor)
        expected_tensor = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw_tensor)
        torch.testing.assert_close(loss_tensor, expected_tensor)


if __name__ == "__main__":
    unittest.main()
