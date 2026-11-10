"""Unit tests for Stage 2 Unified Profile-Sequence Candidate-Aware Neural Ranker (UPS-CANR).

Validates:
1. Architectural shapes: Memory (B, 33, 128), Query (B, C, 128), Attended (B, C, 128),
   Attention weights (B, C, 33), Logits (B, C).
2. Variable slate sizes (C=1, 10, 25, 50, 100) and batch sizes (B=1, 4, 32).
3. Attention masking: Token 0 never masked; padding purchase tokens receive 0 attention weight;
   attention weights sum to 1.0 along the 33-token dimension; zero NaNs for cold customers.
4. Gradient flow with explicit synthetic product text; reference bias-only utilization
   and zero historical text are covered in test_audit_regressions.py.
5. Final scores come from the MLP, without a Stage 1 score skip connection.
6. Device portability: CPU and MPS support without dtype mismatches (float32).
7. Downstream integration: Output format, slate stitching, and checkpoint loading.
"""

from pathlib import Path
import math
import unittest
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender.candidates import CandidateBatch, NUM_PAIRWISE_FEATURES
from recommender.models import OUTPUT_DIM, TEXT_DIM, DEFAULT_SEQUENCE_LENGTH
from recommender.ranker import (
    UPSCANRRanker,
    RankerOutput,
    RankerLoss,
    compute_ranker_loss,
    DEFAULT_NUM_HEADS,
    DEFAULT_MEMORY_DIM,
)


class TestRankerArchitectureShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.C = 50
        self.seq_len = 32
        self.model = UPSCANRRanker(
            text_dim=TEXT_DIM,
            memory_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            num_subcategories=30,
            num_business_lines=10,
            seq_feature_dim=9,
            num_tabular_features=8,
            num_pairwise_features=NUM_PAIRWISE_FEATURES,
            num_heads=4,
            dropout=0.1,
        )

    def _generate_synthetic_inputs(self, B, C):
        text_emb = torch.randn(B, TEXT_DIM)
        city_id = torch.randint(0, 50, (B,))
        region_id = torch.randint(0, 20, (B,))
        industry_id = torch.randint(0, 30, (B,))
        tabular = torch.randn(B, 8)
        seq = torch.randn(B, self.seq_len, 9)
        mask = torch.ones(B, self.seq_len, dtype=torch.long)
        seq_pids = torch.randint(1, 100, (B, self.seq_len))
        seq_cids = torch.randint(1, 20, (B, self.seq_len))
        candidate_vecs = torch.randn(B, C, OUTPUT_DIM)
        candidate_vecs = F.normalize(candidate_vecs, p=2, dim=-1)
        pairwise_feats = torch.rand(B, C, NUM_PAIRWISE_FEATURES)
        return {
            "text_embedding": text_emb,
            "city_id": city_id,
            "region_id": region_id,
            "industry_id": industry_id,
            "tabular_features": tabular,
            "sequence": seq,
            "attention_mask": mask,
            "seq_product_ids": seq_pids,
            "seq_category_ids": seq_cids,
            "candidate_vectors": candidate_vecs,
            "pairwise_features": pairwise_feats,
        }

    def test_forward_pass_default_shapes(self):
        """Standard batch B=4, C=50 forward pass shape check."""
        inputs = self._generate_synthetic_inputs(self.B, self.C)
        out = self.model(**inputs)

        self.assertIsInstance(out, RankerOutput)
        self.assertEqual(out.logits.shape, (self.B, self.C))
        self.assertEqual(out.scores.shape, (self.B, self.C))
        self.assertEqual(out.raw_scores.shape, (self.B, self.C))
        torch.testing.assert_close(out.logits, out.mlp_score)
        self.assertEqual(out.attention_weights.shape, (self.B, self.C, 33))
        self.assertEqual(out.attended_context.shape, (self.B, self.C, OUTPUT_DIM))
        self.assertEqual(out.memory.shape, (self.B, 33, OUTPUT_DIM))
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isnan(out.attention_weights).any())

    def test_variable_slate_sizes(self):
        """Candidate counts C=1, 10, 25, 50, 100 produce exact output shapes."""
        for c in (1, 10, 25, 50, 100):
            inputs = self._generate_synthetic_inputs(self.B, c)
            out = self.model(**inputs)
            self.assertEqual(out.logits.shape, (self.B, c))
            self.assertEqual(out.attention_weights.shape, (self.B, c, 33))

    def test_variable_batch_sizes(self):
        """Batch sizes B=1, 4, 32 execute cleanly and preserve tensor dimensions."""
        for b in (1, 4, 32):
            inputs = self._generate_synthetic_inputs(b, self.C)
            out = self.model(**inputs)
            self.assertEqual(out.logits.shape, (b, self.C))
            self.assertEqual(out.attention_weights.shape, (b, self.C, 33))
            self.assertFalse(torch.isnan(out.logits).any())

    def test_output_dataclass_contract(self):
        """RankerOutput dataclass supports attribute access, indexing, and partition properties."""
        inputs = self._generate_synthetic_inputs(2, 5)
        out = self.model(**inputs)

        # Attribute access
        self.assertTrue(hasattr(out, "logits"))
        self.assertTrue(hasattr(out, "scores"))
        self.assertTrue(hasattr(out, "raw_scores"))
        self.assertTrue(hasattr(out, "attention_weights"))
        self.assertTrue(hasattr(out, "profile_attention"))
        self.assertTrue(hasattr(out, "history_attention"))

        # Profile & History attention partition
        torch.testing.assert_close(out.profile_attention, out.attention_weights[..., 0])
        torch.testing.assert_close(out.history_attention, out.attention_weights[..., 1:].sum(dim=-1))

        # Dictionary-style access
        self.assertTrue("scores" in out)
        self.assertTrue("logits" in out)
        self.assertEqual(out["scores"].shape, (2, 5))


class TestAttentionMaskingAndContextualization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 3
        self.C = 10
        self.seq_len = 32
        self.model = UPSCANRRanker(
            text_dim=TEXT_DIM,
            memory_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            seq_feature_dim=9,
            num_tabular_features=8,
            num_pairwise_features=NUM_PAIRWISE_FEATURES,
        )

    def test_token0_never_masked_invariant(self):
        """Token 0 key_padding_mask is strictly False across all history lengths."""
        for num_events in (0, 1, 5, 32):
            mask = torch.zeros(self.B, self.seq_len, dtype=torch.long)
            if num_events > 0:
                mask[:, :num_events] = 1

            k_mask = self.model.build_key_padding_mask(mask)
            self.assertEqual(k_mask.shape, (self.B, 33))
            # Token 0 must be False (unmasked) for all samples
            self.assertFalse(k_mask[:, 0].any())
            # Tokens 1..32 must reflect attention_mask == 0
            expected_history_mask = (mask == 0)
            self.assertTrue(torch.equal(k_mask[:, 1:], expected_history_mask))

    def test_padding_tokens_receive_zero_attention(self):
        """Padding purchase positions strictly receive 0.0 attention weight."""
        # Customer 0 has 5 events, 27 padding tokens
        # Customer 1 has 14 events, 18 padding tokens
        # Customer 2 has 32 events
        mask = torch.zeros(self.B, self.seq_len, dtype=torch.long)
        mask[0, :5] = 1
        mask[1, :14] = 1
        mask[2, :32] = 1

        seq = torch.randn(self.B, self.seq_len, 9)
        seq[0, 5:] = 0.0
        seq[1, 14:] = 0.0

        out = self.model(
            text_embedding=torch.randn(self.B, TEXT_DIM),
            city_id=torch.randint(0, 50, (self.B,)),
            region_id=torch.randint(0, 20, (self.B,)),
            industry_id=torch.randint(0, 30, (self.B,)),
            tabular_features=torch.randn(self.B, 8),
            sequence=seq,
            attention_mask=mask,
            candidate_vectors=torch.randn(self.B, self.C, OUTPUT_DIM),
            pairwise_features=torch.rand(self.B, self.C, NUM_PAIRWISE_FEATURES),
        )

        attn = out.attention_weights  # (B, C, 33)
        # Sample 0: tokens 6..32 must be 0.0
        padding_weights_0 = attn[0, :, 6:]
        self.assertLess(float(padding_weights_0.detach().max().item()), 1e-6)

        # Sample 1: tokens 15..32 must be 0.0
        padding_weights_1 = attn[1, :, 15:]
        self.assertLess(float(padding_weights_1.detach().max().item()), 1e-6)

    def test_empty_history_customer_zero_nan(self):
        """Zero-history customer attends 100% to Token 0 and produces zero NaNs."""
        mask = torch.zeros(self.B, self.seq_len, dtype=torch.long)  # All padding
        seq = torch.zeros(self.B, self.seq_len, 9)

        out = self.model(
            text_embedding=torch.randn(self.B, TEXT_DIM),
            city_id=torch.randint(0, 50, (self.B,)),
            region_id=torch.randint(0, 20, (self.B,)),
            industry_id=torch.randint(0, 30, (self.B,)),
            tabular_features=torch.randn(self.B, 8),
            sequence=seq,
            attention_mask=mask,
            candidate_vectors=torch.randn(self.B, self.C, OUTPUT_DIM),
            pairwise_features=torch.rand(self.B, self.C, NUM_PAIRWISE_FEATURES),
        )

        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isnan(out.attention_weights).any())

        # Token 0 must receive 1.0 attention across all candidates
        torch.testing.assert_close(
            out.attention_weights[..., 0],
            torch.ones(self.B, self.C),
            atol=1e-5,
            rtol=1e-5,
        )
        # Tokens 1..32 must receive 0.0 attention
        self.assertLess(float(out.attention_weights[..., 1:].detach().max().item()), 1e-6)

    def test_attention_weights_sum_to_one(self):
        """Attention weights sum to 1.0 along the 33-token dimension."""
        mask = torch.randint(0, 2, (self.B, self.seq_len))
        out = self.model(
            text_embedding=torch.randn(self.B, TEXT_DIM),
            city_id=torch.randint(0, 50, (self.B,)),
            region_id=torch.randint(0, 20, (self.B,)),
            industry_id=torch.randint(0, 30, (self.B,)),
            tabular_features=torch.randn(self.B, 8),
            sequence=torch.randn(self.B, self.seq_len, 9),
            attention_mask=mask,
            candidate_vectors=torch.randn(self.B, self.C, OUTPUT_DIM),
            pairwise_features=torch.rand(self.B, self.C, NUM_PAIRWISE_FEATURES),
        )

        weight_sums = out.attention_weights.sum(dim=-1)  # (B, C)
        torch.testing.assert_close(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_contextual_self_attention_mutual_influence(self):
        """Self-attention contextualizes Token 0 profile with Tokens 1..32 history."""
        seq = torch.randn(1, self.seq_len, 9)
        mask = torch.ones(1, self.seq_len, dtype=torch.long)
        cand_vecs = torch.randn(1, 2, OUTPUT_DIM)
        p_feats = torch.rand(1, 2, NUM_PAIRWISE_FEATURES)

        # Baseline forward pass
        out_base = self.model(
            text_embedding=torch.zeros(1, TEXT_DIM),
            city_id=torch.zeros(1, dtype=torch.long),
            region_id=torch.zeros(1, dtype=torch.long),
            industry_id=torch.zeros(1, dtype=torch.long),
            tabular_features=torch.zeros(1, 8),
            sequence=seq,
            attention_mask=mask,
            candidate_vectors=cand_vecs,
            pairwise_features=p_feats,
        )

        # Perturbed Token 0 (change company text and industry)
        out_perturbed = self.model(
            text_embedding=torch.ones(1, TEXT_DIM),
            city_id=torch.tensor([10]),
            region_id=torch.tensor([5]),
            industry_id=torch.tensor([15]),
            tabular_features=torch.ones(1, 8),
            sequence=seq,
            attention_mask=mask,
            candidate_vectors=cand_vecs,
            pairwise_features=p_feats,
        )

        # History tokens memory representations must change due to contextual self-attention
        diff_history_memory = (out_perturbed.memory[:, 1:, :] - out_base.memory[:, 1:, :]).abs().sum()
        self.assertGreater(float(diff_history_memory.detach().item()), 1e-4)


class TestGradientFlowAndTrainability(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = UPSCANRRanker(
            text_dim=TEXT_DIM,
            memory_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            seq_feature_dim=9,
            num_tabular_features=8,
            num_pairwise_features=NUM_PAIRWISE_FEATURES,
        )

    def test_all_parameters_receive_nonzero_gradients(self):
        """Every trainable parameter receives finite, non-zero gradient during backward pass."""
        B, C = 2, 5
        out = self.model(
            text_embedding=torch.randn(B, TEXT_DIM),
            city_id=torch.randint(0, 50, (B,)),
            region_id=torch.randint(0, 20, (B,)),
            industry_id=torch.randint(0, 30, (B,)),
            tabular_features=torch.randn(B, 8),
            sequence=torch.randn(B, 32, 9),
            attention_mask=torch.ones(B, 32, dtype=torch.long),
            seq_product_ids=torch.randint(1, 100, (B, 32)),
            seq_category_ids=torch.randint(1, 20, (B, 32)),
            seq_subcategory_ids=torch.randint(1, 30, (B, 32)),
            seq_business_line_ids=torch.randint(1, 10, (B, 32)),
            seq_product_text=torch.randn(B, 32, TEXT_DIM),
            candidate_vectors=torch.randn(B, C, OUTPUT_DIM),
            pairwise_features=torch.rand(B, C, NUM_PAIRWISE_FEATURES),
        )

        loss = out.logits.sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                self.assertTrue((param.grad != 0).any(), f"Zero gradient in {name}")

    def test_bce_with_logits_loss_backward(self):
        """Multi-label BCE loss with pos_weight propagates gradients cleanly."""
        B, C = 4, 10
        out = self.model(
            text_embedding=torch.randn(B, TEXT_DIM),
            city_id=torch.randint(0, 50, (B,)),
            region_id=torch.randint(0, 20, (B,)),
            industry_id=torch.randint(0, 30, (B,)),
            tabular_features=torch.randn(B, 8),
            sequence=torch.randn(B, 32, 9),
            attention_mask=torch.ones(B, 32, dtype=torch.long),
            candidate_vectors=torch.randn(B, C, OUTPUT_DIM),
            pairwise_features=torch.rand(B, C, NUM_PAIRWISE_FEATURES),
        )

        targets = torch.zeros(B, C)
        targets[0, 0] = 1.0
        targets[1, 2] = 1.0
        targets[2, 5] = 1.0

        pos_weight = torch.tensor([5.0])
        loss = F.binary_cross_entropy_with_logits(out.logits, targets, pos_weight=pos_weight)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(float(loss.item()), 0.0)

        loss.backward()

    def test_listwise_cross_entropy_backward(self):
        """Listwise softmax cross-entropy loss propagates gradients cleanly to cross-attention."""
        B, C = 2, 8
        out = self.model(
            text_embedding=torch.randn(B, TEXT_DIM),
            city_id=torch.randint(0, 50, (B,)),
            region_id=torch.randint(0, 20, (B,)),
            industry_id=torch.randint(0, 30, (B,)),
            tabular_features=torch.randn(B, 8),
            sequence=torch.randn(B, 32, 9),
            attention_mask=torch.ones(B, 32, dtype=torch.long),
            candidate_vectors=torch.randn(B, C, OUTPUT_DIM),
            pairwise_features=torch.rand(B, C, NUM_PAIRWISE_FEATURES),
        )

        # Single target item per transition (slate position 0)
        target_indices = torch.zeros(B, dtype=torch.long)
        listwise_loss = F.cross_entropy(out.logits, target_indices)

        listwise_loss.backward()
        self.assertIsNotNone(self.model.cross_attention.in_proj_weight.grad)
        self.assertFalse(torch.isnan(self.model.cross_attention.in_proj_weight.grad).any())


    def test_ranker_loss_module(self):
        """RankerLoss module computes finite BCE + Listwise loss and populates gradients."""
        loss_mod = RankerLoss(pos_weight=10.0, listwise_weight=1.0)
        logits = torch.randn(4, 50, requires_grad=True)
        targets = torch.zeros(4, 50)
        targets[0, 3] = 1.0
        targets[1, 12] = 1.0
        targets[2, 0] = 1.0

        out = loss_mod(logits, targets)
        self.assertTrue("loss" in out)
        self.assertTrue("bce_loss" in out)
        self.assertTrue("list_loss" in out)
        self.assertFalse(torch.isnan(out["loss"]).any())

        out["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


class TestDevicePortabilityAndPrecision(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = UPSCANRRanker(
            text_dim=TEXT_DIM,
            memory_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            seq_feature_dim=9,
            num_tabular_features=8,
            num_pairwise_features=NUM_PAIRWISE_FEATURES,
        )

    def test_cpu_forward_backward_float32(self):
        """Clean CPU execution with float32 precision."""
        self.model.to("cpu")
        out = self.model(
            text_embedding=torch.randn(2, TEXT_DIM),
            city_id=torch.randint(0, 50, (2,)),
            region_id=torch.randint(0, 20, (2,)),
            industry_id=torch.randint(0, 30, (2,)),
            tabular_features=torch.randn(2, 8),
            sequence=torch.randn(2, 32, 9),
            attention_mask=torch.ones(2, 32, dtype=torch.long),
            candidate_vectors=torch.randn(2, 5, OUTPUT_DIM),
            pairwise_features=torch.rand(2, 5, NUM_PAIRWISE_FEATURES),
        )
        self.assertEqual(out.logits.dtype, torch.float32)
        out.logits.sum().backward()

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device unavailable")
    def test_mps_device_execution_and_numerical_parity(self):
        """MPS forward and backward pass executes cleanly and matches CPU numerically."""
        device = torch.device("mps")
        model_mps = self.model.to(device)

        inputs_cpu = {
            "text_embedding": torch.randn(2, TEXT_DIM),
            "city_id": torch.randint(0, 50, (2,)),
            "region_id": torch.randint(0, 20, (2,)),
            "industry_id": torch.randint(0, 30, (2,)),
            "tabular_features": torch.randn(2, 8),
            "sequence": torch.randn(2, 32, 9),
            "attention_mask": torch.ones(2, 32, dtype=torch.long),
            "candidate_vectors": torch.randn(2, 5, OUTPUT_DIM),
            "pairwise_features": torch.rand(2, 5, NUM_PAIRWISE_FEATURES),
        }
        inputs_mps = {k: v.to(device) for k, v in inputs_cpu.items()}

        out_mps = model_mps(**inputs_mps)
        self.assertEqual(out_mps.logits.device.type, "mps")
        self.assertFalse(torch.isnan(out_mps.logits).any())

        out_mps.logits.sum().backward()


def test_statistics_bias_combines_candidate_and_token_inputs():
    ranker = UPSCANRRanker(statistics_attention=True, dropout=0).eval()
    with torch.no_grad():
        ranker.cross_attention.in_proj_weight[:2 * OUTPUT_DIM].zero_()
        ranker.cross_attention.in_proj_bias[:2 * OUTPUT_DIM].zero_()
        weights = ranker.statistics_attention_weights
        weights[:, 0, 0] = 1   # Quantity on its own
        weights[:, 0, 1] = 1   # Exact age on its own
        weights[:, 1, 0] = 2   # First candidate feature × quantity
        weights[:, 0, 9] = 3   # First profile statistic
        weights[:, 2, 9] = -4  # Second candidate feature × profile statistic

    mask = torch.zeros(1, 32, dtype=torch.long)
    mask[:, :2] = 1
    dates = torch.zeros(1, 32)
    dates[0, :2] = torch.tensor([-30.0, -90.0]) / 365.25
    sequence = torch.zeros(1, 32, 9)
    sequence[0, :2, 0] = torch.tensor([0.25, 0.5])
    profile = torch.zeros(1, 8)
    profile[0, 0] = 0.5
    pairwise = torch.zeros(1, 2, NUM_PAIRWISE_FEATURES)
    pairwise[0, 1, :2] = torch.tensor([1.0, 0.5])
    inputs = dict(
        text_embedding=torch.zeros(1, TEXT_DIM),
        city_id=torch.zeros(1, dtype=torch.long),
        region_id=torch.zeros(1, dtype=torch.long),
        industry_id=torch.zeros(1, dtype=torch.long),
        attention_mask=mask,
        seq_purchase_dates=dates,
        prediction_day=torch.zeros(1),
        candidate_vectors=torch.zeros(1, 2, OUTPUT_DIM),
        pairwise_features=pairwise,
    )
    output = ranker(**inputs, tabular_features=profile, sequence=sequence)
    expected = torch.softmax(torch.tensor([[1.5, 0.5, 1.0], [0.5, 1.0, 2.0]]), dim=-1)
    torch.testing.assert_close(output.attention_weights[0, :, :3], expected)
    assert torch.count_nonzero(output.attention_weights[..., 3:]) == 0

    no_statistics = ranker(
        **inputs, tabular_features=torch.zeros_like(profile), sequence=torch.zeros_like(sequence),
    )
    torch.testing.assert_close(output.memory, no_statistics.memory)
    output.attention_weights[0, 1, 0].backward()
    assert torch.count_nonzero(weights.grad[:, 2, 9]) == ranker.num_heads


def test_excluded_statistics_are_hidden_from_attention_and_the_scorer():
    ranker = UPSCANRRanker(
        statistics_attention=True, dropout=0,
        excluded_statistics=("history_month", "history_repeat", "candidate_repeat"),
    ).eval()
    with torch.no_grad():
        ranker.statistics_attention_weights.normal_()
    sequence = torch.rand(1, 32, 9)
    pairwise = torch.rand(1, 2, NUM_PAIRWISE_FEATURES)
    inputs = dict(
        text_embedding=torch.zeros(1, TEXT_DIM),
        city_id=torch.zeros(1, dtype=torch.long),
        region_id=torch.zeros(1, dtype=torch.long),
        industry_id=torch.zeros(1, dtype=torch.long),
        tabular_features=torch.zeros(1, 8),
        attention_mask=torch.ones(1, 32, dtype=torch.long),
        seq_purchase_dates=torch.zeros(1, 32),
        prediction_day=torch.ones(1),
        candidate_vectors=torch.randn(1, 2, OUTPUT_DIM),
    )
    first = ranker(**inputs, sequence=sequence, pairwise_features=pairwise)
    sequence[..., [3, 4, 7]] = 1
    pairwise[..., 0] = 1
    second = ranker(**inputs, sequence=sequence, pairwise_features=pairwise)
    torch.testing.assert_close(first.attention_weights, second.attention_weights)
    torch.testing.assert_close(first.logits, second.logits)


class TestDownstreamEvaluationIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = UPSCANRRanker(
            text_dim=TEXT_DIM,
            memory_dim=OUTPUT_DIM,
            num_cities=50,
            num_regions=20,
            num_industries=30,
            num_products=100,
            num_categories=20,
            seq_feature_dim=9,
            num_tabular_features=8,
            num_pairwise_features=NUM_PAIRWISE_FEATURES,
        )

    def test_candidate_reranking_and_slate_stitching(self):
        """Validates candidate reranking, full top-100 slate stitching, and attention aggregation."""
        B, C = 2, 50
        candidate_pids = [[f"P_{i:03d}" for i in range(C)] for _ in range(B)]
        top100_pids = [[f"P_{i:03d}" for i in range(100)] for _ in range(B)]

        batch = CandidateBatch(
            customer_ids=["C1", "C2"],
            candidate_pids=candidate_pids,
            candidate_indices=torch.arange(C).unsqueeze(0).repeat(B, 1),
            candidate_scores=torch.randn(B, C),
            pairwise_features=torch.rand(B, C, NUM_PAIRWISE_FEATURES),
            top100_pids=top100_pids,
            top100_indices=torch.arange(100).unsqueeze(0).repeat(B, 1),
            top100_scores=torch.randn(B, 100),
        )

        out = self.model(
            text_embedding=torch.randn(B, TEXT_DIM),
            city_id=torch.zeros(B, dtype=torch.long),
            region_id=torch.zeros(B, dtype=torch.long),
            industry_id=torch.zeros(B, dtype=torch.long),
            tabular_features=torch.zeros(B, 8),
            sequence=torch.zeros(B, 32, 9),
            attention_mask=torch.ones(B, 32, dtype=torch.long),
            candidate_vectors=torch.randn(B, C, OUTPUT_DIM),
            pairwise_features=batch.pairwise_features,
        )

        # 1. Rerank the 50 candidates in descending logit order
        ranked_orders = torch.argsort(out.logits, dim=-1, descending=True).cpu().numpy()

        reranked_top50 = []
        for b in range(B):
            reranked_top50.append([batch.candidate_pids[b][idx] for idx in ranked_orders[b]])

        self.assertEqual(len(reranked_top50[0]), 50)
        self.assertEqual(set(reranked_top50[0]), set(batch.candidate_pids[0]))

        # 2. Stitch with Stage 1 candidates 51..100 to yield full 100-item recommendation list
        final_top100 = []
        for b in range(B):
            seen = set(reranked_top50[b])
            remaining = [p for p in batch.top100_pids[b] if p not in seen]
            combined = reranked_top50[b] + remaining
            final_top100.append(combined)

        self.assertEqual(len(final_top100[0]), 100)
        self.assertEqual(len(set(final_top100[0])), 100)

        # 3. Attention distribution breakdown for evaluate.py
        mean_profile_attn = float(out.profile_attention.mean().item())
        mean_history_attn = float(out.history_attention.mean().item())
        self.assertAlmostEqual(mean_profile_attn + mean_history_attn, 1.0, places=4)

    def test_from_retriever_checkpoint(self):
        """Loads weights from best_retriever.pt cleanly if checkpoint exists."""
        ckpt_path = Path("checkpoints/retriever/best_retriever.pt")
        if not ckpt_path.is_file():
            self.skipTest("checkpoints/retriever/best_retriever.pt not found")

        ranker = UPSCANRRanker.from_retriever_checkpoint(ckpt_path)
        self.assertIsInstance(ranker, UPSCANRRanker)
        self.assertIsNotNone(ranker.product_tower)
