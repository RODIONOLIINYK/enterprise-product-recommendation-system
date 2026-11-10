"""Empirical adversarial test suite for Milestone M7: Enriched Sequence Tokens & Causality.

Author: Challenger 2
Evaluates:
1. Strict temporal isolation and non-leakage across as_of_date boundaries.
2. Point-in-time prior count and repeat purchase indicator causality against an independent oracle.
3. Habit statistics invariance across truncation (un-truncated vs truncated histories).
4. Batch vs individual sequence construction fidelity across diverse history lengths (0, 1, 10, 32, 100).
5. Robustness to boundary conditions, same-day events, missing metadata, and empty batches.
"""

import unittest
import pytest
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


class AdversarialMilestoneM7Tests(unittest.TestCase):
    """Adversarial challenge test suite for Milestone M7."""

    def test_01_strict_temporal_isolation_and_future_poisoning(self):
        """Construct customer timeline with events before, on, and after as_of_date.

        Verify that NO event on or after as_of_date leaks into sequence, attention_mask,
        or habit statistics, even when future events contain massive quantities and repeat SKUs.
        """
        as_of_date = pd.Timestamp("2025-10-01 00:00:00")
        cid = "C_POISON_TEST"

        # Baseline clean past events strictly before as_of_date
        past_events = [
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-08-01"), "product_id": "P_PAST_1", "quantity": 5.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-08-15"), "product_id": "P_PAST_2", "quantity": 10.0, "product_category": "CAT_B"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-09-01"), "product_id": "P_PAST_1", "quantity": 5.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-09-30 23:59:59"), "product_id": "P_PAST_3", "quantity": 2.0, "product_category": "CAT_C"},
        ]

        # Poisoning events: exact boundary (00:00:00 on as_of_date), same day later, and future dates
        poison_events = [
            # Exact boundary: should NOT be included (< as_of_ts)
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-10-01 00:00:00"), "product_id": "P_PAST_1", "quantity": 99999.0, "product_category": "CAT_A"},
            # Same day midday: should NOT be included
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-10-01 12:00:00"), "product_id": "P_FUTURE_1", "quantity": 500.0, "product_category": "CAT_F"},
            # Future days with massive repeat purchases of P_PAST_1 and P_PAST_2
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-10-02"), "product_id": "P_PAST_1", "quantity": 100000.0, "product_category": "CAT_A"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-10-05"), "product_id": "P_PAST_2", "quantity": 50000.0, "product_category": "CAT_B"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-11-20"), "product_id": "P_FUTURE_2", "quantity": 10.0, "product_category": "CAT_F"},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2026-03-01"), "product_id": "P_FUTURE_3", "quantity": 1.0, "product_category": "CAT_F"},
        ]

        # Another customer's events (should never leak into cid)
        other_events = [
            {"customer_id": "C_OTHER", "purchase_date": pd.Timestamp("2025-09-01"), "product_id": "P_PAST_1", "quantity": 100.0},
            {"customer_id": "C_OTHER", "purchase_date": pd.Timestamp("2025-10-05"), "product_id": "P_OTHER", "quantity": 200.0},
        ]

        df_clean = pd.DataFrame(past_events)
        df_poisoned = pd.DataFrame(past_events + poison_events + other_events)

        vocabularies = {
            "product_vocab": {"P_PAST_1": 10, "P_PAST_2": 20, "P_PAST_3": 30, "P_FUTURE_1": 90, "P_FUTURE_2": 91, "P_FUTURE_3": 92},
            "category_vocab": {"CAT_A": 1, "CAT_B": 2, "CAT_C": 3, "CAT_F": 9},
        }

        res_clean = build_customer_sequence(
            df_clean, cid, as_of_date, max_len=32, enriched=True, vocabularies=vocabularies
        )
        res_poisoned = build_customer_sequence(
            df_poisoned, cid, as_of_date, max_len=32, enriched=True, vocabularies=vocabularies
        )

        # 1. Attention mask must be identical and have exactly 4 real events
        self.assertTrue(torch.equal(res_clean["attention_mask"], res_poisoned["attention_mask"]))
        self.assertEqual(res_clean["attention_mask"].sum().item(), 4)

        # 2. Sequence continuous features must be 100% bit-for-bit identical
        self.assertTrue(
            torch.equal(res_clean["sequence"], res_poisoned["sequence"]),
            "Poisoned future events altered sequence feature tensors!"
        )

        # 3. Product IDs must match exactly
        self.assertEqual(res_clean["product_ids"], res_poisoned["product_ids"])
        self.assertEqual(res_clean["product_ids"][:4], ["P_PAST_1", "P_PAST_2", "P_PAST_1", "P_PAST_3"])
        for p in res_poisoned["product_ids"][4:]:
            self.assertEqual(p, PAD_TOKEN)

        # 4. Categorical IDs must match exactly
        self.assertTrue(torch.equal(res_clean["seq_product_ids"], res_poisoned["seq_product_ids"]))
        self.assertTrue(torch.equal(res_clean["seq_category_ids"], res_poisoned["seq_category_ids"]))

        # 5. Future products must never appear
        for future_pid in ["P_FUTURE_1", "P_FUTURE_2", "P_FUTURE_3"]:
            self.assertNotIn(future_pid, res_poisoned["product_ids"])
        for future_id in [90, 91, 92]:
            self.assertNotIn(future_id, res_poisoned["seq_product_ids"].tolist())

        # 6. Verify exact boundary rejection: event at 2025-10-01 00:00:00 with qty 99999.0
        # If leaked, cumulative quantity share would be drastically reduced
        # Token 0 is P_PAST_1 with qty 5. Total cum qty at token 0 is 5. Share must be 1.0
        self.assertAlmostEqual(res_poisoned["sequence"][0, 6].item(), 1.0, places=5)
        # Token 2 is second P_PAST_1 with qty 5. Total cum qty = 5 + 10 + 5 = 20. Cum SKU qty = 10. Share = 10/20 = 0.5
        self.assertAlmostEqual(res_poisoned["sequence"][2, 6].item(), 0.5, places=5)

    def test_02_point_in_time_prior_count_independent_oracle(self):
        """Verify that point-in-time prior count for event i equals the exact count

        of purchases of that SKU at strictly earlier times t < t_i, verified against
        an independent ground-truth counting oracle.
        """
        cid = "C_ORACLE_TEST"
        as_of_date = pd.Timestamp("2025-06-01")

        # Complex interleaved purchase history
        skus = ["SKU_ALPHA", "SKU_BETA", "SKU_GAMMA", "SKU_DELTA"]
        event_records = [
            {"date": "2025-01-01", "sku": "SKU_ALPHA", "qty": 2.0},  # i=0: Alpha prior=0
            {"date": "2025-01-03", "sku": "SKU_BETA", "qty": 4.0},   # i=1: Beta prior=0
            {"date": "2025-01-07", "sku": "SKU_ALPHA", "qty": 1.0},  # i=2: Alpha prior=1
            {"date": "2025-01-12", "sku": "SKU_GAMMA", "qty": 5.0},  # i=3: Gamma prior=0
            {"date": "2025-01-15", "sku": "SKU_ALPHA", "qty": 3.0},  # i=4: Alpha prior=2
            {"date": "2025-01-20", "sku": "SKU_BETA", "qty": 2.0},   # i=5: Beta prior=1
            {"date": "2025-02-01", "sku": "SKU_DELTA", "qty": 10.0}, # i=6: Delta prior=0
            {"date": "2025-02-10", "sku": "SKU_ALPHA", "qty": 2.0},  # i=7: Alpha prior=3
            {"date": "2025-02-14", "sku": "SKU_ALPHA", "qty": 1.0},  # i=8: Alpha prior=4
            {"date": "2025-03-01", "sku": "SKU_BETA", "qty": 6.0},   # i=9: Beta prior=2
            {"date": "2025-03-15", "sku": "SKU_ALPHA", "qty": 8.0},  # i=10: Alpha prior=5
            {"date": "2025-04-01", "sku": "SKU_BETA", "qty": 1.0},   # i=11: Beta prior=3
        ]

        df = pd.DataFrame([
            {
                "customer_id": cid,
                "purchase_date": pd.Timestamp(r["date"]),
                "product_id": r["sku"],
                "quantity": r["qty"],
            }
            for r in event_records
        ])

        res = build_customer_sequence(df, cid, as_of_date, max_len=32, enriched=True)
        seq = res["sequence"]

        # Run independent oracle verification for every single event i
        for i, rec in enumerate(event_records):
            cur_date = pd.Timestamp(rec["date"])
            cur_sku = rec["sku"]

            # Independent oracle: strictly prior count
            strictly_prior_events = [
                r for r in event_records
                if pd.Timestamp(r["date"]) < cur_date and r["sku"] == cur_sku
            ]
            oracle_prior_count = len(strictly_prior_events)
            oracle_prior_norm = float(np.clip(np.log1p(oracle_prior_count) / np.log1p(50.0), 0.0, 1.0))
            oracle_is_repeat = 1.0 if oracle_prior_count >= 1 else 0.0

            # Compare against sequence features
            seq_prior_norm = seq[i, 5].item()
            seq_is_repeat = seq[i, 7].item()

            self.assertAlmostEqual(
                seq_prior_norm, oracle_prior_norm, places=5,
                msg=f"Event {i} ({cur_sku}): prior count norm mismatch. Oracle={oracle_prior_norm}, Seq={seq_prior_norm}"
            )
            self.assertEqual(
                seq_is_repeat, oracle_is_repeat,
                msg=f"Event {i} ({cur_sku}): repeat indicator mismatch. Oracle={oracle_is_repeat}, Seq={seq_is_repeat}"
            )

    def test_03_habit_stats_strictly_invariant_across_truncation(self):
        """Verify that habit stats on retained tokens are completely invariant to

        whether earlier events were truncated (compare sequence from un-truncated vs truncated histories).
        """
        cid = "C_TRUNC_TEST"
        as_of_date = pd.Timestamp("2025-08-01")

        # Generate 80 events over 200 days
        np.random.seed(42)
        skus = [f"SKU_{k:02d}" for k in range(10)]
        events = []
        base_date = pd.Timestamp("2025-01-01")
        for i in range(80):
            day_offset = int(i * 2.5)
            dt = base_date + pd.Timedelta(days=day_offset)
            sku = skus[i % len(skus)]
            qty = float((i % 7) + 1)
            events.append({
                "customer_id": cid,
                "purchase_date": dt,
                "product_id": sku,
                "quantity": qty,
                "product_category": f"CAT_{i % 3}",
                "product_subcategory": f"SUB_{i % 5}",
                "business_line": f"LINE_{i % 2}",
            })

        df = pd.DataFrame(events)

        # 1. Un-truncated history: max_len=80 (retains all 80 events)
        res_full = build_customer_sequence(df, cid, as_of_date, max_len=80, enriched=True)
        # 2. Truncated history: max_len=32 (retains the last 32 events: events 48..79)
        res_trunc32 = build_customer_sequence(df, cid, as_of_date, max_len=32, enriched=True)
        # 3. Intermediate truncated history: max_len=50 (retains the last 50 events: events 30..79)
        res_trunc50 = build_customer_sequence(df, cid, as_of_date, max_len=50, enriched=True)

        seq_full_last32 = res_full["sequence"][48:80]
        seq_trunc32 = res_trunc32["sequence"][:32]
        seq_trunc50_last32 = res_trunc50["sequence"][18:50]

        # Assert identical feature tensors across all 9 dimensions
        self.assertTrue(
            torch.allclose(seq_trunc32, seq_full_last32, atol=1e-6),
            "Habit statistics on 32 retained tokens differ from un-truncated sequence!"
        )
        self.assertTrue(
            torch.allclose(seq_trunc32, seq_trunc50_last32, atol=1e-6),
            "Habit statistics differ between max_len=32 and max_len=50 retained slices!"
        )

        # Check each specific habit feature individually on retained tokens
        # Feature 2: inter_purchase_gap_norm
        self.assertTrue(torch.allclose(seq_trunc32[:, 2], seq_full_last32[:, 2], atol=1e-6))
        # Feature 5: user_product_prior_count_norm
        self.assertTrue(torch.allclose(seq_trunc32[:, 5], seq_full_last32[:, 5], atol=1e-6))
        # Feature 6: user_product_quantity_share
        self.assertTrue(torch.allclose(seq_trunc32[:, 6], seq_full_last32[:, 6], atol=1e-6))
        # Feature 7: is_repeat_purchase
        self.assertTrue(torch.allclose(seq_trunc32[:, 7], seq_full_last32[:, 7], atol=1e-6))

        # Check categorical token IDs
        self.assertTrue(torch.equal(res_trunc32["seq_product_ids"][:32], res_full["seq_product_ids"][48:80]))
        self.assertTrue(torch.equal(res_trunc32["seq_category_ids"][:32], res_full["seq_category_ids"][48:80]))
        self.assertTrue(torch.equal(res_trunc32["seq_subcategory_ids"][:32], res_full["seq_subcategory_ids"][48:80]))
        self.assertTrue(torch.equal(res_trunc32["seq_business_line_ids"][:32], res_full["seq_business_line_ids"][48:80]))

        # Product IDs list
        self.assertEqual(res_trunc32["product_ids"][:32], res_full["product_ids"][48:80])

    def test_04_batch_vs_individual_fidelity_diverse_history_lengths(self):
        """Test batch construction with diverse customer history lengths (0, 1, 10, 32, 100 events)

        and assert identical outputs between batch vs individual calls.
        """
        as_of_date = pd.Timestamp("2025-10-01")
        base_date = pd.Timestamp("2025-01-01")

        customers = [
            "C_LEN_0",    # 0 events (empty history)
            "C_LEN_1",    # 1 event
            "C_LEN_10",   # 10 events
            "C_LEN_32",   # 32 events
            "C_LEN_100",  # 100 events (truncated to 32)
        ]
        dates = [as_of_date] * len(customers)

        all_events = []

        # C_LEN_0 has NO events before as_of_date, only one future event
        all_events.append({
            "customer_id": "C_LEN_0",
            "purchase_date": pd.Timestamp("2025-11-01"),
            "product_id": "P_FUTURE",
            "quantity": 1.0,
            "product_category": "CAT_F",
        })

        # C_LEN_1 has 1 event
        all_events.append({
            "customer_id": "C_LEN_1",
            "purchase_date": pd.Timestamp("2025-05-01"),
            "product_id": "P_01",
            "quantity": 3.0,
            "product_category": "CAT_1",
        })

        # C_LEN_10 has 10 events
        for i in range(10):
            all_events.append({
                "customer_id": "C_LEN_10",
                "purchase_date": base_date + pd.Timedelta(days=i * 10),
                "product_id": f"P_{i % 3:02d}",
                "quantity": float(i + 1),
                "product_category": f"CAT_{i % 2}",
            })

        # C_LEN_32 has 32 events
        for i in range(32):
            all_events.append({
                "customer_id": "C_LEN_32",
                "purchase_date": base_date + pd.Timedelta(days=i * 5),
                "product_id": f"P_{i % 5:02d}",
                "quantity": float((i % 4) + 1),
                "product_category": f"CAT_{i % 3}",
            })

        # C_LEN_100 has 100 events
        for i in range(100):
            all_events.append({
                "customer_id": "C_LEN_100",
                "purchase_date": base_date + pd.Timedelta(days=int(i * 2.2)),
                "product_id": f"P_{i % 8:02d}",
                "quantity": float((i % 10) + 1),
                "product_category": f"CAT_{i % 4}",
            })

        events_df = pd.DataFrame(all_events)

        vocabularies = {
            "product_vocab": {f"P_{k:02d}": k + 10 for k in range(10)},
            "category_vocab": {f"CAT_{k}": k + 1 for k in range(5)},
        }

        # Run batch construction
        batch_res = build_batch_sequences(
            events_df=events_df,
            customer_ids=customers,
            as_of_dates=dates,
            max_len=32,
            enriched=True,
            vocabularies=vocabularies,
        )

        # Assert shape
        self.assertEqual(batch_res["sequence"].shape, (5, 32, 9))
        self.assertEqual(batch_res["attention_mask"].shape, (5, 32))
        self.assertEqual(batch_res["seq_product_ids"].shape, (5, 32))
        self.assertEqual(batch_res["seq_category_ids"].shape, (5, 32))
        self.assertEqual(len(batch_res["product_ids"]), 5)

        # Compare each customer in batch against individual call
        for idx, cid in enumerate(customers):
            single_res = build_customer_sequence(
                events_df=events_df,
                customer_id=cid,
                as_of_date=dates[idx],
                max_len=32,
                enriched=True,
                vocabularies=vocabularies,
            )

            # 1. Sequence tensor exact match
            self.assertTrue(
                torch.equal(batch_res["sequence"][idx], single_res["sequence"]),
                f"Mismatch in sequence tensor for customer {cid} (idx {idx})"
            )

            # 2. Attention mask exact match
            self.assertTrue(
                torch.equal(batch_res["attention_mask"][idx], single_res["attention_mask"]),
                f"Mismatch in attention mask for customer {cid} (idx {idx})"
            )

            # 3. Categorical IDs exact match
            self.assertTrue(
                torch.equal(batch_res["seq_product_ids"][idx], single_res["seq_product_ids"]),
                f"Mismatch in seq_product_ids for customer {cid} (idx {idx})"
            )
            self.assertTrue(
                torch.equal(batch_res["seq_category_ids"][idx], single_res["seq_category_ids"]),
                f"Mismatch in seq_category_ids for customer {cid} (idx {idx})"
            )
            self.assertTrue(
                torch.equal(batch_res["seq_subcategory_ids"][idx], single_res["seq_subcategory_ids"]),
                f"Mismatch in seq_subcategory_ids for customer {cid} (idx {idx})"
            )
            self.assertTrue(
                torch.equal(batch_res["seq_business_line_ids"][idx], single_res["seq_business_line_ids"]),
                f"Mismatch in seq_business_line_ids for customer {cid} (idx {idx})"
            )

            # 4. Product IDs list exact match
            self.assertEqual(
                batch_res["product_ids"][idx], single_res["product_ids"],
                f"Mismatch in product_ids for customer {cid} (idx {idx})"
            )

            # 5. Check real event counts
            expected_counts = {"C_LEN_0": 0, "C_LEN_1": 1, "C_LEN_10": 10, "C_LEN_32": 32, "C_LEN_100": 32}
            self.assertEqual(batch_res["attention_mask"][idx].sum().item(), expected_counts[cid])

    def test_05_boundary_conditions_and_edge_cases(self):
        """Stress-test edge cases: empty batch, same-day multiple events,

        out-of-vocabulary tokens, missing metadata columns, zero quantity.
        """
        # 1. Empty batch
        empty_batch = build_batch_sequences(pd.DataFrame(), [], [], max_len=32, enriched=True)
        self.assertEqual(empty_batch["sequence"].shape, (0, 32, 9))
        self.assertEqual(empty_batch["attention_mask"].shape, (0, 32))
        self.assertEqual(empty_batch["seq_product_ids"].shape, (0, 32))
        self.assertEqual(empty_batch["product_ids"], [])

        # 2. Same-day multiple purchases (different products)
        cid = "C_SAME_DAY"
        as_of_date = pd.Timestamp("2025-05-01")
        same_day_events = [
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-04-10 09:00:00"), "product_id": "P_A", "quantity": 1.0},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-04-10 09:00:00"), "product_id": "P_B", "quantity": 2.0},
            {"customer_id": cid, "purchase_date": pd.Timestamp("2025-04-10 14:00:00"), "product_id": "P_C", "quantity": 3.0},
        ]
        res_same_day = build_customer_sequence(pd.DataFrame(same_day_events), cid, as_of_date, max_len=32, enriched=True)
        # Check inter-purchase gap: event 0 has gap=0; events 1 & 2 on same day have gap_days=0, so log1p(0)=0.0
        self.assertEqual(res_same_day["sequence"][0, 2].item(), 0.0)
        self.assertEqual(res_same_day["sequence"][1, 2].item(), 0.0)
        self.assertEqual(res_same_day["sequence"][2, 2].item(), 0.0)

        # 3. Out-of-vocabulary tokens with provided vocabulary
        vocab = {"P_KNOWN": 42}
        res_oov = build_customer_sequence(
            pd.DataFrame([{"customer_id": cid, "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P_UNKNOWN", "quantity": 1.0}]),
            cid, as_of_date, max_len=32, product_vocab=vocab
        )
        # Unknown token must map strictly to 0
        self.assertEqual(res_oov["seq_product_ids"][0].item(), 0)

        # 4. Zero quantity handling (should not produce NaN or inf)
        res_zero_qty = build_customer_sequence(
            pd.DataFrame([{"customer_id": cid, "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P_A", "quantity": 0.0}]),
            cid, as_of_date, max_len=32, enriched=True
        )
        self.assertFalse(torch.isnan(res_zero_qty["sequence"]).any())
        self.assertFalse(torch.isinf(res_zero_qty["sequence"]).any())
        self.assertEqual(res_zero_qty["sequence"][0, 0].item(), 0.0)  # qty_norm = 0.0
        self.assertEqual(res_zero_qty["sequence"][0, 6].item(), 0.0)  # cum_qty_share = 0.0

    def test_06_extreme_feature_bounds_and_saturation(self):
        """Verify that extreme inputs saturate cleanly without exceeding [0, 1] bounds:

        - Prior count > 50 -> clamped to 1.0
        - Days ago > 365 -> clamped to 1.0
        - Inter-purchase gap > 365 -> clamped to 1.0
        - Quantity > max_quantity -> clamped to 1.0
        - Month cyclical features remain in [-1, 1]
        """
        cid = "C_EXTREME"
        as_of_date = pd.Timestamp("2025-10-01")

        # 60 repeat purchases of the same SKU across 600 days
        events = []
        base = pd.Timestamp("2023-01-01")
        for i in range(60):
            events.append({
                "customer_id": cid,
                "purchase_date": base + pd.Timedelta(days=i * 10),
                "product_id": "SKU_REPEATED",
                "quantity": 500.0,  # exceeds max_quantity=100.0
            })

        df = pd.DataFrame(events)
        res = build_customer_sequence(df, cid, as_of_date, max_len=60, max_quantity=100.0, enriched=True)
        seq = res["sequence"]

        # 1. Quantity norm must saturate at 1.0
        self.assertAlmostEqual(seq[0, 0].item(), 1.0, places=5)

        # 2. Early events were > 365 days ago, so days_ago_norm must be clamped to 1.0
        self.assertAlmostEqual(seq[0, 1].item(), 1.0, places=5)

        # 3. For event 55 (55th purchase), prior_counts = 55 > 50 -> must saturate at 1.0
        self.assertAlmostEqual(seq[55, 5].item(), 1.0, places=5)
        self.assertLessEqual(seq[55, 5].item(), 1.0)

        # 4. Repeat purchase indicator is strictly 1.0 for all events > 0
        self.assertEqual(seq[0, 7].item(), 0.0)
        self.assertTrue((seq[1:60, 7] == 1.0).all())

        # 5. All features across the entire sequence must be strictly finite and within bounds
        self.assertTrue(torch.isfinite(seq).all())
        self.assertTrue((seq[:, 0] >= 0.0).all() and (seq[:, 0] <= 1.0).all())
        self.assertTrue((seq[:, 1] >= 0.0).all() and (seq[:, 1] <= 1.0).all())
        self.assertTrue((seq[:, 2] >= 0.0).all() and (seq[:, 2] <= 1.0).all())
        self.assertTrue((seq[:, 3] >= -1.0).all() and (seq[:, 3] <= 1.0).all())
        self.assertTrue((seq[:, 4] >= -1.0).all() and (seq[:, 4] <= 1.0).all())
        self.assertTrue((seq[:, 5] >= 0.0).all() and (seq[:, 5] <= 1.0).all())
        self.assertTrue((seq[:, 6] >= 0.0).all() and (seq[:, 6] <= 1.0).all())
        self.assertTrue((seq[:, 7] >= 0.0).all() and (seq[:, 7] <= 1.0).all())
        self.assertTrue((seq[:, 8] >= 0.0).all() and (seq[:, 8] <= 1.0).all())

    @pytest.mark.real_purchase_data
    def test_07_real_world_data_temporal_isolation_and_truncation(self):
        """Test sequence generation directly on real cleaned_purchases.csv dataset.

        Verifies that customer sequences from real data adhere to temporal cutoffs,
        produce valid non-NaN features, and truncation preserves habit statistics.
        """
        import os
        data_path = "data/interim/cleaned_purchases.csv"
        if not os.path.exists(data_path):
            self.skipTest(f"{data_path} does not exist.")

        df = pd.read_csv(data_path)
        # Select active customer with at least 35 purchases before val split
        as_of_date = pd.Timestamp("2025-10-01")
        df["pdate"] = pd.to_datetime(df["purchase_date"])
        prior_mask = df["pdate"] < as_of_date

        counts = df[prior_mask].groupby("customer_id").size()
        heavy_custs = counts[counts >= 35].index.tolist()
        self.assertTrue(len(heavy_custs) > 0, "No heavy customers found in dataset.")

        test_cid = heavy_custs[0]
        n_prior = counts[test_cid]

        # Call with max_len=32 vs max_len=n_prior
        res_32 = build_customer_sequence(df, test_cid, as_of_date, max_len=32, enriched=True)
        res_full = build_customer_sequence(df, test_cid, as_of_date, max_len=int(n_prior), enriched=True)

        # 32 tokens must match the last 32 tokens of the full history
        self.assertEqual(res_32["attention_mask"].sum().item(), 32)
        self.assertTrue(
            torch.allclose(res_32["sequence"], res_full["sequence"][-32:], atol=1e-5),
            "Habit features on real customer data diverged after truncation!"
        )
        self.assertEqual(res_32["product_ids"], res_full["product_ids"][-32:])
        self.assertTrue(torch.equal(res_32["seq_product_ids"], res_full["seq_product_ids"][-32:]))

    def test_08_integer_and_special_character_customer_ids(self):
        """Verify robust customer ID handling when ID is integer, string with leading zeros,

        or contains special characters / symbols.
        """
        df = pd.DataFrame([
            {"customer_id": 10042, "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P_INT", "quantity": 1.0},
            {"customer_id": "00055", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P_STR", "quantity": 1.0},
            {"customer_id": "C/100-A#1", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P_SPEC", "quantity": 1.0},
        ])
        as_of_date = pd.Timestamp("2025-02-01")

        # Query with int 10042
        res_int = build_customer_sequence(df, 10042, as_of_date, max_len=32)
        self.assertEqual(res_int["attention_mask"].sum().item(), 1)
        self.assertEqual(res_int["product_ids"][0], "P_INT")

        # Query with string "10042"
        res_int_str = build_customer_sequence(df, "10042", as_of_date, max_len=32)
        self.assertTrue(torch.equal(res_int["sequence"], res_int_str["sequence"]))

        # Query with leading zeros string "00055"
        res_zeros = build_customer_sequence(df, "00055", as_of_date, max_len=32)
        self.assertEqual(res_zeros["product_ids"][0], "P_STR")

        # Query with special chars
        res_spec = build_customer_sequence(df, "C/100-A#1", as_of_date, max_len=32)
        self.assertEqual(res_spec["product_ids"][0], "P_SPEC")

    def test_09_vocabularies_alias_unpacking_and_metadata_lookup(self):
        """Verify that vocabularies aliases (product, category, product_category, business_lines)

        are correctly unpacked and used.
        """
        df = pd.DataFrame([
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2025-01-01"),
                "product_id": "P1",
                "quantity": 1.0,
            }
        ])
        item_meta = pd.DataFrame([
            {
                "product_id": "P1",
                "category": "CAT_ALIAS",
                "subcategory": "SUB_ALIAS",
                "line": "LINE_ALIAS",
            }
        ])
        vocabularies = {
            "products": {"P1": 100},
            "categories": {"CAT_ALIAS": 200},
            "subcategories": {"SUB_ALIAS": 300},
            "business_lines": {"LINE_ALIAS": 400},
        }
        res = build_customer_sequence(
            df, "C1", pd.Timestamp("2025-02-01"),
            vocabularies=vocabularies,
            item_metadata=item_meta,
        )
        self.assertEqual(res["seq_product_ids"][0].item(), 100)
        self.assertEqual(res["seq_category_ids"][0].item(), 200)
        self.assertEqual(res["seq_subcategory_ids"][0].item(), 300)
        self.assertEqual(res["seq_business_line_ids"][0].item(), 400)


if __name__ == "__main__":
    unittest.main()
