"""Unit tests for Phase 5: Customer Sequence Generation (R3)."""

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
    fit_quantity_reference,
)


class CustomerSequencesTests(unittest.TestCase):
    def test_fitted_quantity_reference_preserves_bulk_purchase_distinctions(self):
        reference = fit_quantity_reference([-50.0, 0.0, 10.0, 100.0, 1000.0, np.inf, np.nan])
        self.assertEqual(reference, 100.0)
        self.assertEqual(fit_quantity_reference([0.0, np.nan]), 1.0)
        history = pd.DataFrame({
            "customer_id": ["C1"] * 4,
            "purchase_date": pd.date_range("2025-01-01", periods=4),
            "product_id": ["P1"] * 4,
            "quantity": [0.0, reference, 10_000.0, 80_000.0],
        })
        date = pd.Timestamp("2025-02-01")
        single = build_customer_sequence(history, "C1", date, quantity_reference=reference)
        batch = build_batch_sequences(history, ["C1"], [date], quantity_reference=reference)
        torch.testing.assert_close(single["seq_purchase_dates"][:4], torch.arange(4) / 365.25)
        torch.testing.assert_close(single["seq_purchase_dates"], batch["seq_purchase_dates"][0])
        later = build_customer_sequence(history, "C1", date + pd.Timedelta(days=20), quantity_reference=reference)
        torch.testing.assert_close(single["seq_purchase_dates"], later["seq_purchase_dates"])

        scaled = single["sequence"][:4, 0].tolist()
        self.assertEqual(scaled[:2], [0.0, 0.5])
        self.assertTrue(0.5 < scaled[2] < scaled[3] < 1.0)
        self.assertTrue(torch.equal(batch["sequence"][0], single["sequence"]))

    def test_dataframe_metadata_can_contain_duplicate_product_ids(self):
        history = pd.DataFrame([{
            "customer_id": "C1",
            "purchase_date": pd.Timestamp("2025-01-01"),
            "product_id": "7",
            "quantity": 1.0,
        }])
        items = pd.DataFrame({
            "product_id": [7, 7, 8],
            "product_category": ["First", "Duplicate", "Other"],
        })
        result = build_customer_sequence(
            history, "C1", pd.Timestamp("2025-02-01"),
            category_vocab={"First": 1, "Duplicate": 2, "Other": 3},
            item_metadata=items,
        )
        self.assertEqual(result["seq_category_ids"][0].item(), 1)

    def test_sequence_exact_length_32(self):
        history = pd.DataFrame([
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp(f"2025-01-{day:02d}"),
                "product_id": f"P-{day}",
                "quantity": float(day),
            }
            for day in range(1, 15)  # 14 events
        ])
        as_of_date = pd.Timestamp("2025-02-01")
        result = build_customer_sequence(history, "C1", as_of_date, max_len=32)

        sequence = result["sequence"]
        attention_mask = result["attention_mask"]
        product_ids = result["product_ids"]

        self.assertEqual(sequence.shape, (32, 4))
        self.assertEqual(attention_mask.shape, (32,))
        self.assertEqual(len(product_ids), 32)
        # 14 real events, 18 padded
        self.assertEqual(int(attention_mask.sum()), 14)
        self.assertEqual(attention_mask[:14].tolist(), [1] * 14)
        self.assertEqual(attention_mask[14:].tolist(), [0] * 18)
        self.assertEqual(product_ids[14:], [PAD_TOKEN] * 18)

    def test_sequence_truncation_when_longer_than_32(self):
        # 45 events
        history = pd.DataFrame([
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "product_id": f"P-{i:03d}",
                "quantity": 1.0,
            }
            for i in range(45)
        ])
        as_of_date = pd.Timestamp("2025-03-01")
        result = build_customer_sequence(history, "C1", as_of_date, max_len=32)

        sequence = result["sequence"]
        attention_mask = result["attention_mask"]
        product_ids = result["product_ids"]

        self.assertEqual(sequence.shape, (32, 4))
        self.assertEqual(int(attention_mask.sum()), 32)
        # Should keep the most recent 32 events: i=13 to i=44
        self.assertEqual(product_ids[0], "P-013")
        self.assertEqual(product_ids[-1], "P-044")

    def test_sequence_chronological_ordering(self):
        # Shuffle insertion order
        events = [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-03-10"), "product_id": "P3", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-05"), "product_id": "P1", "quantity": 2.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-02-15"), "product_id": "P2", "quantity": 3.0},
        ]
        history = pd.DataFrame(events)
        as_of_date = pd.Timestamp("2025-04-01")
        result = build_customer_sequence(history, "C1", as_of_date, max_len=32)

        product_ids = result["product_ids"]
        self.assertEqual(product_ids[:3], ["P1", "P2", "P3"])

    def test_empty_history_sequence(self):
        history = pd.DataFrame(columns=["customer_id", "purchase_date", "product_id", "quantity"])
        result = build_customer_sequence(history, "UNKNOWN", pd.Timestamp("2025-01-01"), max_len=32)

        sequence = result["sequence"]
        attention_mask = result["attention_mask"]
        product_ids = result["product_ids"]

        self.assertEqual(sequence.shape, (32, 4))
        self.assertEqual(int(attention_mask.sum()), 0)
        self.assertFalse(torch.isnan(sequence).any())
        self.assertEqual(product_ids, [PAD_TOKEN] * 32)

    def test_batch_sequences(self):
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P2", "quantity": 2.0},
        ])
        customers = ["C1", "C2", "EMPTY"]
        dates = [pd.Timestamp("2025-02-01")] * 3

        batch = build_batch_sequences(history, customers, dates, max_len=32)

        self.assertEqual(batch["sequence"].shape, (3, 32, 4))
        self.assertEqual(batch["attention_mask"].shape, (3, 32))
        self.assertEqual(batch["attention_mask"][0, 0].item(), 1)
        self.assertEqual(batch["attention_mask"][0, 1].item(), 0)
        self.assertEqual(batch["attention_mask"][1, 0].item(), 1)
        self.assertEqual(batch["attention_mask"][2, :].sum().item(), 0)

    def test_enriched_sequence_feature_dim_9(self):
        history = pd.DataFrame([
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp(f"2025-01-{day:02d}"),
                "product_id": f"P-{day}",
                "quantity": float(day),
            }
            for day in range(1, 15)  # 14 events
        ])
        as_of_date = pd.Timestamp("2025-02-01")
        result = build_customer_sequence(history, "C1", as_of_date, max_len=32, enriched=True)

        sequence = result["sequence"]
        attention_mask = result["attention_mask"]
        product_ids = result["product_ids"]

        self.assertEqual(sequence.shape, (32, ENRICHED_FEATURE_DIM))
        self.assertEqual(attention_mask.shape, (32,))
        self.assertEqual(len(product_ids), 32)
        self.assertEqual(int(attention_mask.sum()), 14)

        # Check features bounds for real events (0..13)
        for i in range(14):
            # 0: qty_norm in [0, 1]
            self.assertGreaterEqual(sequence[i, 0].item(), 0.0)
            self.assertLessEqual(sequence[i, 0].item(), 1.0)
            # 1: days_ago_norm in [0, 1]
            self.assertGreaterEqual(sequence[i, 1].item(), 0.0)
            self.assertLessEqual(sequence[i, 1].item(), 1.0)
            # 2: inter_purchase_gap_norm in [0, 1]
            self.assertGreaterEqual(sequence[i, 2].item(), 0.0)
            self.assertLessEqual(sequence[i, 2].item(), 1.0)
            # 3: sin_month in [-1, 1]
            self.assertGreaterEqual(sequence[i, 3].item(), -1.0)
            self.assertLessEqual(sequence[i, 3].item(), 1.0)
            # 4: cos_month in [-1, 1]
            self.assertGreaterEqual(sequence[i, 4].item(), -1.0)
            self.assertLessEqual(sequence[i, 4].item(), 1.0)
            # 5: user_product_prior_count_norm in [0, 1]
            self.assertGreaterEqual(sequence[i, 5].item(), 0.0)
            self.assertLessEqual(sequence[i, 5].item(), 1.0)
            # 6: user_product_quantity_share in [0, 1]
            self.assertGreaterEqual(sequence[i, 6].item(), 0.0)
            self.assertLessEqual(sequence[i, 6].item(), 1.0)
            # 7: is_repeat_purchase in {0.0, 1.0}
            self.assertIn(sequence[i, 7].item(), [0.0, 1.0])
            # 8: real_event_indicator == 1.0
            self.assertEqual(sequence[i, 8].item(), 1.0)

        # Padding positions (14..31) must be strictly 0.0
        self.assertEqual(float(sequence[14:].abs().sum()), 0.0)
        self.assertEqual(result["seq_product_ids"][14:].sum().item(), 0)

    def test_point_in_time_habit_statistics_causality(self):
        events = [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 10.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-05"), "product_id": "P2", "quantity": 20.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-10"), "product_id": "P1", "quantity": 10.0},
        ]
        as_of_date = pd.Timestamp("2025-02-01")
        result = build_customer_sequence(pd.DataFrame(events), "C1", as_of_date, max_len=32, enriched=True)
        seq = result["sequence"]

        # Event 0: P1 (first time)
        self.assertEqual(seq[0, 2].item(), 0.0)  # gap = 0 for first customer event
        self.assertEqual(seq[0, 5].item(), 0.0)  # prior count = 0
        self.assertAlmostEqual(seq[0, 6].item(), 1.0, places=4)  # 10 / 10 = 1.0
        self.assertEqual(seq[0, 7].item(), 0.0)  # is_repeat = 0.0

        # Event 1: P2 (first time)
        expected_gap_1 = np.log1p(4.0) / np.log1p(365.0)
        self.assertAlmostEqual(seq[1, 2].item(), expected_gap_1, places=4)
        self.assertEqual(seq[1, 5].item(), 0.0)
        self.assertAlmostEqual(seq[1, 6].item(), 20.0 / 30.0, places=4)
        self.assertEqual(seq[1, 7].item(), 0.0)

        # Event 2: P1 (repeat purchase)
        expected_gap_2 = np.log1p(5.0) / np.log1p(365.0)
        self.assertAlmostEqual(seq[2, 2].item(), expected_gap_2, places=4)
        expected_prior_count = np.log1p(1.0) / np.log1p(50.0)
        self.assertAlmostEqual(seq[2, 5].item(), expected_prior_count, places=4)
        self.assertAlmostEqual(seq[2, 6].item(), 20.0 / 40.0, places=4)
        self.assertEqual(seq[2, 7].item(), 1.0)

    def test_habit_statistics_preserved_across_truncation(self):
        events = []
        for i in range(45):
            dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            if i in (2, 5, 20):
                pid = "P1"
                qty = 10.0
            else:
                pid = f"PX_{i}"
                qty = 1.0
            events.append({"customer_id": "C1", "purchase_date": dt, "product_id": pid, "quantity": qty})

        as_of_date = pd.Timestamp("2025-03-01")
        result = build_customer_sequence(pd.DataFrame(events), "C1", as_of_date, max_len=32, enriched=True)

        seq = result["sequence"]
        pids = result["product_ids"]

        # Truncation keeps events 13 to 44.
        # Event 20 is at index 20 - 13 = 7.
        self.assertEqual(pids[7], "P1")

        # Event 20 must have prior_count == 2 (events 2 and 5), even though events 2 and 5 were truncated
        expected_prior_norm = np.log1p(2.0) / np.log1p(50.0)
        self.assertAlmostEqual(seq[7, 5].item(), expected_prior_norm, places=4)
        self.assertEqual(seq[7, 7].item(), 1.0)

        # Cumulative quantity share at event 20: 30 / 48
        expected_share = 30.0 / 48.0
        self.assertAlmostEqual(seq[7, 6].item(), expected_share, places=4)

        # The first event in the retained sequence (index 0, event 13) has inter_purchase_gap_norm
        # computed relative to event 12 (1 day gap), NOT 0.0
        expected_gap = np.log1p(1.0) / np.log1p(365.0)
        self.assertAlmostEqual(seq[0, 2].item(), expected_gap, places=4)

    def test_categorical_token_ids_with_and_without_vocabulary(self):
        events = [
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2025-01-01"),
                "product_id": "P1",
                "quantity": 1.0,
                "product_category": "CAT_A",
                "product_subcategory": "SUBCAT_A",
                "business_line": "LINE_A",
            },
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2025-01-02"),
                "product_id": "P2",
                "quantity": 1.0,
            },
        ]
        item_metadata = pd.DataFrame([
            {
                "product_id": "P2",
                "product_category": "CAT_B",
                "product_subcategory": "SUBCAT_B",
                "business_line": "LINE_B",
            }
        ])
        as_of_date = pd.Timestamp("2025-02-01")

        # 1. With vocabulary
        product_vocab = {"P1": 101, "P2": 102}
        category_vocab = {"CAT_A": 11, "CAT_B": 12}
        subcategory_vocab = {"SUBCAT_A": 21, "SUBCAT_B": 22}
        business_line_vocab = {"LINE_A": 31, "LINE_B": 32}

        res_vocab = build_customer_sequence(
            pd.DataFrame(events),
            "C1",
            as_of_date,
            max_len=32,
            product_vocab=product_vocab,
            category_vocab=category_vocab,
            subcategory_vocab=subcategory_vocab,
            business_line_vocab=business_line_vocab,
            item_metadata=item_metadata,
        )

        self.assertEqual(res_vocab["seq_product_ids"][0].item(), 101)
        self.assertEqual(res_vocab["seq_product_ids"][1].item(), 102)
        self.assertEqual(res_vocab["seq_category_ids"][0].item(), 11)
        self.assertEqual(res_vocab["seq_category_ids"][1].item(), 12)
        self.assertEqual(res_vocab["seq_subcategory_ids"][0].item(), 21)
        self.assertEqual(res_vocab["seq_subcategory_ids"][1].item(), 22)
        self.assertEqual(res_vocab["seq_business_line_ids"][0].item(), 31)
        self.assertEqual(res_vocab["seq_business_line_ids"][1].item(), 32)

        # Padding positions must be strictly 0
        self.assertEqual(res_vocab["seq_product_ids"][2:].sum().item(), 0)
        self.assertEqual(res_vocab["seq_category_ids"][2:].sum().item(), 0)

        # Aliases
        self.assertTrue(torch.equal(res_vocab["product_indices"], res_vocab["seq_product_ids"]))
        self.assertTrue(torch.equal(res_vocab["category_indices"], res_vocab["seq_category_ids"]))

        # 2. Without vocabulary (deterministic fallback in [1, 10000] for real, 0 for pad)
        res_no_vocab = build_customer_sequence(
            pd.DataFrame(events),
            "C1",
            as_of_date,
            max_len=32,
            item_metadata=item_metadata,
        )
        p1_id = res_no_vocab["seq_product_ids"][0].item()
        p2_id = res_no_vocab["seq_product_ids"][1].item()
        self.assertGreaterEqual(p1_id, 1)
        self.assertGreaterEqual(p2_id, 1)
        self.assertNotEqual(p1_id, p2_id)
        # Padding strictly 0
        self.assertEqual(res_no_vocab["seq_product_ids"][2:].sum().item(), 0)

        # Calling again produces deterministic identical IDs
        res_repeat = build_customer_sequence(
            pd.DataFrame(events),
            "C1",
            as_of_date,
            max_len=32,
            item_metadata=item_metadata,
        )
        self.assertEqual(res_repeat["seq_product_ids"][0].item(), p1_id)
        self.assertEqual(res_repeat["seq_product_ids"][1].item(), p2_id)

    def test_cyclical_month_features(self):
        # Months: March (3), June (6), September (9), December (12)
        events = [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-03-15"), "product_id": "P1", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-06-15"), "product_id": "P2", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-09-15"), "product_id": "P3", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-12-15"), "product_id": "P4", "quantity": 1.0},
        ]
        as_of_date = pd.Timestamp("2026-01-01")
        result = build_customer_sequence(pd.DataFrame(events), "C1", as_of_date, max_len=32, enriched=True)
        seq = result["sequence"]

        # March (m=3): sin(2*pi*3/12) = sin(pi/2) = 1.0, cos(pi/2) = 0.0
        self.assertAlmostEqual(seq[0, 3].item(), 1.0, places=4)
        self.assertAlmostEqual(seq[0, 4].item(), 0.0, places=4)

        # June (m=6): sin(pi) = 0.0, cos(pi) = -1.0
        self.assertAlmostEqual(seq[1, 3].item(), 0.0, places=4)
        self.assertAlmostEqual(seq[1, 4].item(), -1.0, places=4)

        # September (m=9): sin(3*pi/2) = -1.0, cos(3*pi/2) = 0.0
        self.assertAlmostEqual(seq[2, 3].item(), -1.0, places=4)
        self.assertAlmostEqual(seq[2, 4].item(), 0.0, places=4)

        # December (m=12): sin(2*pi) = 0.0, cos(2*pi) = 1.0
        self.assertAlmostEqual(seq[3, 3].item(), 0.0, places=4)
        self.assertAlmostEqual(seq[3, 4].item(), 1.0, places=4)

    def test_backward_compatibility_legacy_signature(self):
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 10.0},
        ])
        as_of_date = pd.Timestamp("2025-02-01")

        # Call with default arguments (no feature_dim, no enriched)
        res_default = build_customer_sequence(history, "C1", as_of_date)
        self.assertEqual(res_default["sequence"].shape, (32, 4))
        self.assertEqual(res_default["sequence"][0, 2].item(), 1.0)  # real event indicator
        self.assertEqual(res_default["sequence"][0, 3].item(), 0.0)  # unassigned 4th dim

        # Call with explicit feature_dim=4
        res_dim4 = build_customer_sequence(history, "C1", as_of_date, feature_dim=DEFAULT_FEATURE_DIM)
        self.assertEqual(res_dim4["sequence"].shape, (32, 4))
        self.assertTrue(torch.equal(res_default["sequence"], res_dim4["sequence"]))

    def test_batch_sequences_enriched(self):
        history = pd.DataFrame([
            {
                "customer_id": "C1",
                "purchase_date": pd.Timestamp("2025-01-01"),
                "product_id": "P1",
                "quantity": 5.0,
                "product_category": "CAT1",
            },
            {
                "customer_id": "C2",
                "purchase_date": pd.Timestamp("2025-01-02"),
                "product_id": "P2",
                "quantity": 10.0,
                "product_category": "CAT2",
            },
        ])
        customers = ["C1", "C2", "EMPTY"]
        dates = [pd.Timestamp("2025-02-01")] * 3

        product_vocab = {"P1": 1, "P2": 2}
        category_vocab = {"CAT1": 10, "CAT2": 20}

        batch = build_batch_sequences(
            history,
            customers,
            dates,
            max_len=32,
            enriched=True,
            product_vocab=product_vocab,
            category_vocab=category_vocab,
        )

        self.assertEqual(batch["sequence"].shape, (3, 32, 9))
        self.assertEqual(batch["attention_mask"].shape, (3, 32))
        self.assertEqual(batch["seq_product_ids"].shape, (3, 32))
        self.assertEqual(batch["seq_category_ids"].shape, (3, 32))
        self.assertEqual(batch["seq_subcategory_ids"].shape, (3, 32))
        self.assertEqual(batch["seq_business_line_ids"].shape, (3, 32))

        # C1
        self.assertEqual(batch["seq_product_ids"][0, 0].item(), 1)
        self.assertEqual(batch["seq_category_ids"][0, 0].item(), 10)
        self.assertEqual(batch["attention_mask"][0, 0].item(), 1)
        self.assertEqual(batch["attention_mask"][0, 1].item(), 0)

        # C2
        self.assertEqual(batch["seq_product_ids"][1, 0].item(), 2)
        self.assertEqual(batch["seq_category_ids"][1, 0].item(), 20)
        self.assertEqual(batch["attention_mask"][1, 0].item(), 1)

        # EMPTY
        self.assertEqual(batch["attention_mask"][2].sum().item(), 0)
        self.assertEqual(batch["seq_product_ids"][2].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
