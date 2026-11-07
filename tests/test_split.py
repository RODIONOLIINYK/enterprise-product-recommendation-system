"""Unit tests for Phase 4: Chronological Data Split (R2)."""

import unittest
import pandas as pd

from recommender.split import (
    TRAIN_CUTOFF,
    VAL_END,
    VAL_START,
    TEST_START,
    chronological_split,
    get_split_proportions,
    validate_no_leakage,
)


class ChronologicalSplitTests(unittest.TestCase):
    def test_chronological_split_boundaries(self):
        records = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-06-15")},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2025-09-30")},
            {"customer_id": "C3", "purchase_date": pd.Timestamp("2025-10-01")},
            {"customer_id": "C4", "purchase_date": pd.Timestamp("2025-12-25")},
            {"customer_id": "C5", "purchase_date": pd.Timestamp("2026-02-18")},
            {"customer_id": "C6", "purchase_date": pd.Timestamp("2026-02-19")},
            {"customer_id": "C7", "purchase_date": pd.Timestamp("2026-05-01")},
        ])

        train, val, test = chronological_split(records)

        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 3)
        self.assertEqual(len(test), 2)
        self.assertEqual(train["customer_id"].tolist(), ["C1", "C2"])
        self.assertEqual(val["customer_id"].tolist(), ["C3", "C4", "C5"])
        self.assertEqual(test["customer_id"].tolist(), ["C6", "C7"])

    def test_boundary_exact_cutoffs(self):
        records = pd.DataFrame([
            {"customer_id": "T1", "purchase_date": pd.Timestamp("2025-09-30 23:59:59")},
            {"customer_id": "V1", "purchase_date": pd.Timestamp("2025-10-01 00:00:00")},
            {"customer_id": "V2", "purchase_date": pd.Timestamp("2026-02-18 23:59:59")},
            {"customer_id": "TE1", "purchase_date": pd.Timestamp("2026-02-19 00:00:00")},
        ])

        train, val, test = chronological_split(records)

        self.assertEqual(train["customer_id"].tolist(), ["T1"])
        self.assertEqual(val["customer_id"].tolist(), ["V1", "V2"])
        self.assertEqual(test["customer_id"].tolist(), ["TE1"])
        self.assertTrue(validate_no_leakage(train, val, test))

    def test_validate_no_leakage_detects_future_event_in_train(self):
        train = pd.DataFrame([
            {"purchase_date": pd.Timestamp("2025-05-01")},
            {"purchase_date": pd.Timestamp("2025-10-02")},  # Leakage!
        ], index=[0, 1])
        val = pd.DataFrame([
            {"purchase_date": pd.Timestamp("2025-11-01")},
        ], index=[2])
        test = pd.DataFrame([
            {"purchase_date": pd.Timestamp("2026-03-01")},
        ], index=[3])

        with self.assertRaisesRegex(ValueError, "Temporal leakage in train split"):
            validate_no_leakage(train, val, test)

    def test_validate_no_leakage_detects_val_out_of_bounds(self):
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-05-01")}], index=[0])
        val = pd.DataFrame([
            {"purchase_date": pd.Timestamp("2025-09-29")},  # Too early for val
        ], index=[1])
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[2])

        with self.assertRaisesRegex(ValueError, "Temporal leakage in val split"):
            validate_no_leakage(train, val, test)

    def test_validate_no_leakage_detects_test_inversion(self):
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-05-01")}], index=[0])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-12-01")}], index=[1])
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-02-15")}], index=[2])  # Too early for test

        with self.assertRaisesRegex(ValueError, "Temporal leakage in test split"):
            validate_no_leakage(train, val, test)

    def test_validate_no_leakage_detects_overlapping_indices(self):
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-05-01")}], index=[0])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-11-01")}], index=[0])  # Duplicate index 0!
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[1])

        with self.assertRaisesRegex(ValueError, "share 1 indices"):
            validate_no_leakage(train, val, test)

    def test_get_split_proportions(self):
        train = pd.DataFrame([1] * 70, columns=["val"])
        val = pd.DataFrame([1] * 15, columns=["val"])
        test = pd.DataFrame([1] * 15, columns=["val"])
        props = get_split_proportions(train, val, test)
        self.assertAlmostEqual(props["train"], 0.70)
        self.assertAlmostEqual(props["val"], 0.15)
        self.assertAlmostEqual(props["test"], 0.15)

    def test_sub_microsecond_timestamp_boundary(self):
        # A timestamp with nanoseconds right before test start (2026-02-18 23:59:59.999999999)
        # must be partitioned into validation without Unassigned events error.
        records = pd.DataFrame([
            {"customer_id": "T1", "purchase_date": pd.Timestamp("2025-09-30 23:59:59.999999")},
            {"customer_id": "V1", "purchase_date": pd.Timestamp("2025-10-01 00:00:00")},
            {"customer_id": "V_NANO", "purchase_date": pd.Timestamp("2026-02-18 23:59:59.999999999")},
            {"customer_id": "TE1", "purchase_date": pd.Timestamp("2026-02-19 00:00:00")},
        ])

        train, val, test = chronological_split(records)
        self.assertEqual(train["customer_id"].tolist(), ["T1"])
        self.assertEqual(val["customer_id"].tolist(), ["V1", "V_NANO"])
        self.assertEqual(test["customer_id"].tolist(), ["TE1"])
        self.assertTrue(validate_no_leakage(train, val, test))


if __name__ == "__main__":
    unittest.main()
