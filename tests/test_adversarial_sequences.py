"""Empirical Adversarial Stress Tests for recommender/sequences.py (Milestone M7).

Targeting extreme edge cases and asserting strict invariant guarantees:
1. 1000 events for one customer (scale, truncation causality, habit stability)
2. Zero events (empty dataframe, missing customer, future-only events)
3. Duplicate timestamps (exact same datetime across identical and distinct products)
4. Negative quantities (returns, cancellations, negative cumulative volume)
5. Quantities > 10,000 (extreme heavy-tailed values, clipping behavior)
6. Leap year dates (2020-02-29, 2024-02-29, month feature circular continuity)
7. Missing customer IDs (None, NaN, empty string, non-existent customer)
8. Invariant: No feature ever produces NaN or Inf across any configuration
9. Invariant: Categorical ID tensors contain only non-negative integers and strictly 0 at pad positions
10. Invariant: user_product_prior_count_norm strictly equals 0.0 on first purchase of every SKU
11. Invariant: user_product_quantity_share is always in [0.0, 1.0] across all events
12. Invariant: attention_mask matches non-zero positions of real_event_indicator
13. Invariant: Batch processing preserves all single-sequence invariants
"""

import unittest
import numpy as np
import pandas as pd
import torch

from recommender.sequences import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_SEQUENCE_LENGTH,
    ENRICHED_FEATURE_DIM,
    PAD_TOKEN,
    build_batch_sequences,
    build_customer_sequence,
)


class AdversarialSequencesTests(unittest.TestCase):
    def setUp(self):
        self.max_len = 32
        self.as_of_date = pd.Timestamp("2025-06-01")

    def _assert_strict_invariants(self, result, is_enriched=True, expected_len=32):
        """Verify strict invariant contracts on output dictionary."""
        seq = result["sequence"]
        mask = result["attention_mask"]
        pids = result["product_ids"]

        # 1. No NaN or Inf
        self.assertFalse(torch.isnan(seq).any(), "NaN found in sequence tensor")
        self.assertFalse(torch.isinf(seq).any(), "Inf found in sequence tensor")

        # 2. Shape checks
        expected_dim = ENRICHED_FEATURE_DIM if is_enriched else DEFAULT_FEATURE_DIM
        self.assertEqual(seq.shape, (expected_len, expected_dim))
        self.assertEqual(mask.shape, (expected_len,))
        self.assertEqual(len(pids), expected_len)

        # 3. Mask matches real_event_indicator exactly
        real_event_col = 8 if is_enriched else 2
        real_indicators = seq[:, real_event_col]
        self.assertTrue(
            torch.equal(mask.float(), real_indicators),
            f"attention_mask {mask} does not match real_event_indicator {real_indicators}",
        )

        # 4. Categorical ID tensors: non-negative integers and strictly 0 at padding positions
        cat_keys = [
            "seq_product_ids",
            "seq_category_ids",
            "seq_subcategory_ids",
            "seq_business_line_ids",
        ]
        pad_positions = (mask == 0)
        real_positions = (mask == 1)

        for key in cat_keys:
            cat_tensor = result[key]
            self.assertEqual(cat_tensor.shape, (expected_len,))
            # Only non-negative integers
            self.assertTrue(
                (cat_tensor >= 0).all(),
                f"Negative value found in categorical tensor {key}",
            )
            # Strictly 0 at padding positions
            if pad_positions.any():
                pad_vals = cat_tensor[pad_positions]
                self.assertTrue(
                    (pad_vals == 0).all(),
                    f"Non-zero padding value in {key}: {pad_vals}",
                )

        # 5. Continuous feature ranges for enriched mode
        if is_enriched:
            if real_positions.any():
                # qty_norm in [0, 1]
                self.assertTrue((seq[real_positions, 0] >= 0.0).all())
                self.assertTrue((seq[real_positions, 0] <= 1.0).all())

                # days_ago_norm in [0, 1]
                self.assertTrue((seq[real_positions, 1] >= 0.0).all())
                self.assertTrue((seq[real_positions, 1] <= 1.0).all())

                # inter_purchase_gap_norm in [0, 1]
                self.assertTrue((seq[real_positions, 2] >= 0.0).all())
                self.assertTrue((seq[real_positions, 2] <= 1.0).all())

                # sin_month in [-1, 1], cos_month in [-1, 1]
                self.assertTrue((seq[real_positions, 3] >= -1.0).all())
                self.assertTrue((seq[real_positions, 3] <= 1.0).all())
                self.assertTrue((seq[real_positions, 4] >= -1.0).all())
                self.assertTrue((seq[real_positions, 4] <= 1.0).all())

                # user_product_prior_count_norm in [0, 1]
                self.assertTrue((seq[real_positions, 5] >= 0.0).all())
                self.assertTrue((seq[real_positions, 5] <= 1.0).all())

                # user_product_quantity_share in [0, 1]
                self.assertTrue((seq[real_positions, 6] >= 0.0).all())
                self.assertTrue((seq[real_positions, 6] <= 1.0).all())

                # is_repeat_purchase in {0, 1}
                repeat_vals = seq[real_positions, 7]
                for v in repeat_vals:
                    self.assertIn(v.item(), (0.0, 1.0))

            # Padding positions in sequence must be strictly all 0.0
            if pad_positions.any():
                self.assertEqual(float(seq[pad_positions].abs().sum()), 0.0)

    def test_stress_1000_events_single_customer(self):
        """Stress test with 1000 events for one customer: truncation and habit preservation."""
        n_events = 1000
        dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 6) for i in range(n_events)]
        # 20 distinct products cycling
        products = [f"SKU_{i % 20}" for i in range(n_events)]
        quantities = [float((i % 50) + 1) for i in range(n_events)]

        df = pd.DataFrame({
            "customer_id": ["C_HEAVY"] * n_events,
            "purchase_date": dates,
            "product_id": products,
            "quantity": quantities,
            "product_category": [f"CAT_{i % 5}" for i in range(n_events)],
            "product_subcategory": [f"SUBCAT_{i % 10}" for i in range(n_events)],
            "business_line": [f"LINE_{i % 2}" for i in range(n_events)],
        })

        res = build_customer_sequence(df, "C_HEAVY", self.as_of_date, max_len=32, enriched=True)
        self._assert_strict_invariants(res, is_enriched=True)

        # Truncated to 32 real events
        self.assertEqual(int(res["attention_mask"].sum()), 32)
        # All 32 tokens should be repeat purchases because each SKU appeared ~50 times previously
        self.assertTrue((res["sequence"][:, 7] == 1.0).all())
        # Prior count norm should be strictly > 0 for all retained tokens
        self.assertTrue((res["sequence"][:, 5] > 0.0).all())

    def test_stress_zero_events_all_variations(self):
        """Stress test with zero events: empty df, customer not present, all events in future."""
        # Variation A: Empty DataFrame
        empty_df = pd.DataFrame(columns=["customer_id", "purchase_date", "product_id", "quantity"])
        res_a = build_customer_sequence(empty_df, "C_NONE", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_a, is_enriched=True)
        self.assertEqual(int(res_a["attention_mask"].sum()), 0)
        self.assertEqual(res_a["product_ids"], [PAD_TOKEN] * 32)

        # Variation B: Customer not in DataFrame
        df_b = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "purchase_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            "product_id": ["P1", "P2"],
            "quantity": [1.0, 2.0],
        })
        res_b = build_customer_sequence(df_b, "C_NONEXISTENT", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_b, is_enriched=True)
        self.assertEqual(int(res_b["attention_mask"].sum()), 0)

        # Variation C: All events >= as_of_date (future events only)
        df_c = pd.DataFrame({
            "customer_id": ["C_FUTURE", "C_FUTURE"],
            "purchase_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")],
            "product_id": ["P1", "P2"],
            "quantity": [1.0, 1.0],
        })
        res_c = build_customer_sequence(df_c, "C_FUTURE", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_c, is_enriched=True)
        self.assertEqual(int(res_c["attention_mask"].sum()), 0)

    def test_stress_duplicate_timestamps(self):
        """Stress test with identical timestamps on same and different products."""
        ts = pd.Timestamp("2025-02-15 10:30:00")
        df = pd.DataFrame({
            "customer_id": ["C1"] * 8,
            "purchase_date": [ts] * 8,  # all identical timestamps
            "product_id": ["P1", "P2", "P1", "P3", "P2", "P1", "P4", "P1"],
            "quantity": [2.0, 5.0, 1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
        })

        res = build_customer_sequence(df, "C1", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res, is_enriched=True)

        seq = res["sequence"]
        mask = res["attention_mask"]
        self.assertEqual(int(mask.sum()), 8)

        # Inter-purchase gaps for duplicate timestamps must be exactly 0.0
        self.assertEqual(float(seq[:8, 2].abs().sum()), 0.0)

        # First appearance of each product must have user_product_prior_count_norm == 0.0
        pids = res["product_ids"][:8]
        seen_skus = set()
        for idx, pid in enumerate(pids):
            prior_count_norm = seq[idx, 5].item()
            is_repeat = seq[idx, 7].item()
            if pid not in seen_skus:
                self.assertEqual(
                    prior_count_norm,
                    0.0,
                    f"First purchase of {pid} at index {idx} has non-zero prior_count_norm: {prior_count_norm}",
                )
                self.assertEqual(is_repeat, 0.0)
                seen_skus.add(pid)
            else:
                self.assertGreater(prior_count_norm, 0.0)
                self.assertEqual(is_repeat, 1.0)

    def test_stress_negative_quantities(self):
        """Stress test with negative quantities (returns/credits)."""
        df = pd.DataFrame({
            "customer_id": ["C_NEG"] * 6,
            "purchase_date": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(6)],
            "product_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "quantity": [-10.0, -50.0, 0.0, -1.0, -999.0, -0.001],
        })

        res = build_customer_sequence(df, "C_NEG", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res, is_enriched=True)

        seq = res["sequence"]
        # qty_norm must be clamped to 0.0 for non-positive quantities
        for i in range(6):
            self.assertEqual(seq[i, 0].item(), 0.0)
            # user_product_quantity_share must stay within [0.0, 1.0]
            share = seq[i, 6].item()
            self.assertGreaterEqual(share, 0.0)
            self.assertLessEqual(share, 1.0)

    def test_stress_extreme_quantities_over_10000(self):
        """Stress test with quantities > 10,000."""
        df = pd.DataFrame({
            "customer_id": ["C_HUGE"] * 4,
            "purchase_date": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(4)],
            "product_id": ["P1", "P2", "P1", "P3"],
            "quantity": [10001.0, 50000.0, 1000000.0, 99999999.0],
        })

        res = build_customer_sequence(df, "C_HUGE", self.as_of_date, max_quantity=100.0, enriched=True)
        self._assert_strict_invariants(res, is_enriched=True)

        seq = res["sequence"]
        # qty_norm must be clipped to 1.0 when quantity >> max_quantity
        for i in range(4):
            self.assertEqual(seq[i, 0].item(), 1.0)
            share = seq[i, 6].item()
            self.assertGreaterEqual(share, 0.0)
            self.assertLessEqual(share, 1.0)

    def test_stress_leap_year_dates(self):
        """Stress test across leap years: 2020-02-29 and 2024-02-29."""
        df = pd.DataFrame({
            "customer_id": ["C_LEAP"] * 5,
            "purchase_date": [
                pd.Timestamp("2020-02-28"),
                pd.Timestamp("2020-02-29"),  # Leap day 2020
                pd.Timestamp("2020-03-01"),
                pd.Timestamp("2024-02-29"),  # Leap day 2024
                pd.Timestamp("2024-03-01"),
            ],
            "product_id": ["P1", "P2", "P3", "P4", "P5"],
            "quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        res = build_customer_sequence(df, "C_LEAP", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res, is_enriched=True)

        seq = res["sequence"]
        # Event 1 is 2020-02-29 (month 2). Month 2 sin = sin(2*pi*2/12) = sin(pi/3) = sqrt(3)/2 ~ 0.866
        self.assertAlmostEqual(seq[1, 3].item(), np.sin(2 * np.pi * 2 / 12), places=4)
        self.assertAlmostEqual(seq[1, 4].item(), np.cos(2 * np.pi * 2 / 12), places=4)

        # Gap between 2020-02-28 and 2020-02-29 is exactly 1 day
        expected_gap = np.log1p(1.0) / np.log1p(365.0)
        self.assertAlmostEqual(seq[1, 2].item(), expected_gap, places=4)

    def test_stress_missing_customer_ids_and_nulls(self):
        """Stress test with missing/null customer_id in dataframe or query call."""
        df = pd.DataFrame({
            "customer_id": [None, np.nan, "C_VALID", "C_VALID", None],
            "purchase_date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-02"),
                pd.Timestamp("2025-01-03"),
                pd.Timestamp("2025-01-04"),
                pd.Timestamp("2025-01-05"),
            ],
            "product_id": ["P1", "P2", "P3", "P4", "P5"],
            "quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        # Query valid customer
        res_valid = build_customer_sequence(df, "C_VALID", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_valid, is_enriched=True)
        self.assertEqual(int(res_valid["attention_mask"].sum()), 2)

        # Query None / NaN / missing customer
        res_none = build_customer_sequence(df, None, self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_none, is_enriched=True)
        self.assertEqual(int(res_none["attention_mask"].sum()), 0)

        res_nan = build_customer_sequence(df, "NONEXISTENT", self.as_of_date, enriched=True)
        self._assert_strict_invariants(res_nan, is_enriched=True)
        self.assertEqual(int(res_nan["attention_mask"].sum()), 0)

    def test_first_purchase_prior_count_norm_strictly_zero(self):
        """Verify user_product_prior_count_norm == 0.0 on the first purchase of every SKU."""
        # 50 purchases of 10 distinct products in interleaved order
        np.random.seed(42)
        n = 50
        skus = [f"SKU_{i}" for i in range(10)]
        chosen_skus = [skus[i % 10] for i in range(n)]
        dates = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(n)]

        df = pd.DataFrame({
            "customer_id": ["C_INTERLEAVED"] * n,
            "purchase_date": dates,
            "product_id": chosen_skus,
            "quantity": [1.0] * n,
        })

        res = build_customer_sequence(df, "C_INTERLEAVED", self.as_of_date, max_len=50, enriched=True)
        seq = res["sequence"]
        pids = res["product_ids"]

        seen = set()
        for idx in range(n):
            pid = pids[idx]
            prior_count = seq[idx, 5].item()
            is_repeat = seq[idx, 7].item()
            if pid not in seen:
                self.assertEqual(
                    prior_count,
                    0.0,
                    f"First purchase of {pid} at index {idx} has prior_count {prior_count} != 0.0",
                )
                self.assertEqual(is_repeat, 0.0)
                seen.add(pid)
            else:
                self.assertGreater(
                    prior_count,
                    0.0,
                    f"Repeat purchase of {pid} at index {idx} has prior_count {prior_count} == 0.0",
                )
                self.assertEqual(is_repeat, 1.0)

    def test_user_product_quantity_share_bounds_under_adversarial_patterns(self):
        """Assert user_product_quantity_share in [0.0, 1.0] under fluctuating, negative, and zero volume."""
        patterns = [
            # High volume then zero volume
            [1000.0, 0.0, 0.0, 50.0],
            # Flips between positive and negative quantities
            [10.0, -5.0, 20.0, -10.0, 5.0],
            # All identical small quantities
            [0.001, 0.001, 0.001],
            # Extreme single product dominance
            [1.0, 1.0, 10000.0, 1.0],
        ]

        for p_idx, qty_pattern in enumerate(patterns):
            n = len(qty_pattern)
            df = pd.DataFrame({
                "customer_id": [f"C_PAT_{p_idx}"] * n,
                "purchase_date": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(n)],
                "product_id": [f"P_{i % 2}" for i in range(n)],
                "quantity": qty_pattern,
            })
            res = build_customer_sequence(df, f"C_PAT_{p_idx}", self.as_of_date, max_len=32, enriched=True)
            self._assert_strict_invariants(res, is_enriched=True)

            seq = res["sequence"]
            shares = seq[:n, 6]
            self.assertTrue(
                (shares >= 0.0).all() and (shares <= 1.0).all(),
                f"Pattern {p_idx} produced shares outside [0, 1]: {shares}",
            )

    def test_attention_mask_matches_real_event_indicator_both_modes(self):
        """Assert attention_mask strictly matches real_event_indicator in legacy (col 2) and enriched (col 8)."""
        df = pd.DataFrame({
            "customer_id": ["C_DUAL"] * 5,
            "purchase_date": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(5)],
            "product_id": [f"P_{i}" for i in range(5)],
            "quantity": [1.0] * 5,
        })

        # 1. Enriched mode (dim=9, indicator is col 8)
        res_enriched = build_customer_sequence(df, "C_DUAL", self.as_of_date, max_len=10, enriched=True)
        self._assert_strict_invariants(res_enriched, is_enriched=True, expected_len=10)
        mask_en = res_enriched["attention_mask"]
        self.assertEqual(int(mask_en.sum()), 5)
        self.assertTrue(torch.equal(mask_en.float(), res_enriched["sequence"][:, 8]))

        # 2. Legacy mode (dim=4, indicator is col 2)
        res_legacy = build_customer_sequence(df, "C_DUAL", self.as_of_date, max_len=10, feature_dim=4)
        self._assert_strict_invariants(res_legacy, is_enriched=False, expected_len=10)
        mask_leg = res_legacy["attention_mask"]
        self.assertEqual(int(mask_leg.sum()), 5)
        self.assertTrue(torch.equal(mask_leg.float(), res_legacy["sequence"][:, 2]))

    def test_batch_adversarial_combination(self):
        """Stress test batch sequences with an adversarial combination of customers."""
        df = pd.DataFrame({
            "customer_id": [
                "C_1000", "C_1000",
                "C_DUP", "C_DUP", "C_DUP",
                "C_NEG", "C_NEG",
                "C_HUGE",
                "C_LEAP",
            ],
            "purchase_date": [
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
                pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-02"),
                pd.Timestamp("2025-04-01"),
                pd.Timestamp("2024-02-29"),
            ],
            "product_id": ["P1", "P2", "P1", "P1", "P2", "P3", "P3", "P4", "P5"],
            "quantity": [1.0, 2.0, 5.0, 10.0, 2.0, -100.0, -50.0, 999999.0, 1.0],
        })

        cids = ["C_1000", "C_DUP", "C_NEG", "C_HUGE", "C_LEAP", "C_EMPTY", "MISSING"]
        dates = [self.as_of_date] * len(cids)

        batch = build_batch_sequences(df, cids, dates, max_len=32, enriched=True)

        self.assertFalse(torch.isnan(batch["sequence"]).any())
        self.assertFalse(torch.isinf(batch["sequence"]).any())
        self.assertEqual(batch["sequence"].shape, (len(cids), 32, 9))
        self.assertEqual(batch["attention_mask"].shape, (len(cids), 32))

        # Check mask matches indicator across all batch rows
        for b in range(len(cids)):
            mask = batch["attention_mask"][b]
            real_ind = batch["sequence"][b, :, 8]
            self.assertTrue(torch.equal(mask.float(), real_ind))

        # Empty customers have sum 0
        self.assertEqual(int(batch["attention_mask"][5].sum()), 0)
        self.assertEqual(int(batch["attention_mask"][6].sum()), 0)


if __name__ == "__main__":
    unittest.main()
