"""Unit tests for Phase 5: Point-in-Time Features and Quantity Normalization (R3)."""

import unittest
import numpy as np
import pandas as pd

from recommender.features import (
    Log1pScaler,
    PointInTimeNormalizer,
    compute_point_in_time_features,
    compute_urgency,
    normalize_feature,
)


class PointInTimeFeaturesTests(unittest.TestCase):
    def test_compute_urgency_formula(self):
        # overdue: 14 days since order, median gap 7 -> +7.0
        self.assertEqual(compute_urgency(14.0, 7.0), 7.0)
        # not yet due: 3 days since order, median gap 14 -> -11.0
        self.assertEqual(compute_urgency(3.0, 14.0), -11.0)
        # on schedule: 10 days since order, median gap 10 -> 0.0
        self.assertEqual(compute_urgency(10.0, 10.0), 0.0)

    def test_urgency_calculation_with_cadence_history(self):
        # 3 distinct orders: 2025-01-01, 2025-01-11 (gap 10), 2025-01-21 (gap 10)
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-11"), "quantity": 2.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-21"), "quantity": 1.0},
        ])
        as_of_date = pd.Timestamp("2025-01-31")  # 10 days since last order
        features = compute_point_in_time_features(history, "C1", as_of_date)

        self.assertEqual(features["days_since_last_order"], 10.0)
        self.assertEqual(features["median_order_gap"], 10.0)
        self.assertEqual(features["urgency"], 0.0)
        self.assertEqual(features["has_cadence_history"], 1.0)
        self.assertEqual(features["distinct_orders"], 3.0)

    def test_urgency_imputation_with_single_prior_order(self):
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 1.0},
        ])
        as_of_date = pd.Timestamp("2025-01-15")  # 14 days since last order
        features = compute_point_in_time_features(
            history, "C1", as_of_date, population_median_gap=20.0,
        )

        self.assertEqual(features["days_since_last_order"], 14.0)
        self.assertEqual(features["median_order_gap"], 20.0)
        self.assertEqual(features["urgency"], -6.0)
        self.assertEqual(features["has_cadence_history"], 0.0)
        self.assertEqual(features["distinct_orders"], 1.0)

    def test_empty_history_graceful_handling(self):
        history = pd.DataFrame(columns=["customer_id", "purchase_date", "quantity"])
        features = compute_point_in_time_features(
            history, "NEW_CUSTOMER", pd.Timestamp("2025-06-01"), population_median_gap=14.0,
        )
        self.assertEqual(features["total_purchases"], 0.0)
        self.assertEqual(features["has_history"], 0.0)
        self.assertEqual(features["has_cadence_history"], 0.0)
        self.assertEqual(features["median_order_gap"], 14.0)
        self.assertEqual(features["urgency"], -14.0)

    def test_point_in_time_strict_isolation(self):
        # 1 order before cutoff, 1 order on cutoff, 1 order after cutoff
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-05-01"), "quantity": 5.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-06-01"), "quantity": 10.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-07-01"), "quantity": 100.0},
        ])
        as_of_date = pd.Timestamp("2025-06-01")  # < 2025-06-01 means only 2025-05-01 is visible!
        features = compute_point_in_time_features(history, "C1", as_of_date)

        self.assertEqual(features["total_purchases"], 1.0)
        self.assertEqual(features["total_quantity"], 5.0)

    def test_log1p_scaler_bounds_and_clipping(self):
        scaler = Log1pScaler()
        train_quantities = np.array([1.0, 5.0, 10.0, 50.0])
        scaler.fit(train_quantities)

        # Transformed train points
        transformed = scaler.transform(train_quantities)
        self.assertTrue((transformed >= 0.0).all())
        self.assertTrue((transformed <= 1.0).all())
        self.assertAlmostEqual(transformed[0], np.log1p(1.0) / np.log1p(50.0))
        self.assertAlmostEqual(transformed[-1], 1.0)

        # Zero input
        self.assertEqual(scaler.transform(0.0), 0.0)

        # Unseen heavy-tailed test outlier (10,000): must be strictly clipped to 1.0
        outlier_scaled = scaler.transform(10000.0)
        self.assertEqual(outlier_scaled, 1.0)

        # Monotonicity check
        inputs = np.linspace(0, 200, 50)
        outputs = scaler.transform(inputs)
        self.assertTrue(np.all(np.diff(outputs) >= 0.0))

    def test_point_in_time_normalizer_pipeline(self):
        df_train = pd.DataFrame({
            "total_purchases": [1.0, 10.0, 50.0],
            "total_quantity": [2.0, 20.0, 100.0],
            "distinct_products": [1.0, 5.0, 20.0],
            "distinct_orders": [1.0, 3.0, 10.0],
            "distinct_categories": [1.0, 2.0, 4.0],
            "days_since_first_order": [10.0, 60.0, 300.0],
            "days_since_last_order": [5.0, 20.0, 90.0],
            "median_order_gap": [7.0, 14.0, 30.0],
            "urgency": [-20.0, 0.0, 60.0],
        })
        normalizer = PointInTimeNormalizer()
        normalizer.fit(df_train)

        raw = {
            "total_purchases": 5.0,
            "total_quantity": 10.0,
            "distinct_products": 3.0,
            "distinct_orders": 2.0,
            "distinct_categories": 2.0,
            "days_since_first_order": 50.0,
            "days_since_last_order": 15.0,
            "median_order_gap": 14.0,
            "urgency": 1.0,
            "has_cadence_history": 1.0,
            "has_history": 1.0,
        }
        normalized = normalizer.transform_dict(raw)

        for key, val in normalized.items():
            if key == "urgency":
                continue  # raw signed urgency
            self.assertGreaterEqual(val, 0.0, f"{key} was < 0.0: {val}")
            self.assertLessEqual(val, 1.0, f"{key} was > 1.0: {val}")

    def test_log1p_scaler_with_python_list_and_tuple(self):
        scaler = Log1pScaler().fit([10.0, 50.0])

        # Test passing Python list
        out_list = scaler.transform([1.0, 2.0, 50.0])
        self.assertIsInstance(out_list, np.ndarray)
        self.assertEqual(len(out_list), 3)
        self.assertTrue((out_list >= 0.0).all() and (out_list <= 1.0).all())
        self.assertEqual(out_list[2], 1.0)

        # Test passing Python tuple
        out_tuple = scaler.transform((5.0, 10.0))
        self.assertIsInstance(out_tuple, np.ndarray)
        self.assertEqual(len(out_tuple), 2)
        self.assertTrue((out_tuple >= 0.0).all() and (out_tuple <= 1.0).all())

        # Test normalize_feature directly on list and tuple
        res_list = normalize_feature([10.0, 20.0], max_val=100.0)
        self.assertIsInstance(res_list, np.ndarray)
        self.assertTrue((res_list >= 0.0).all() and (res_list <= 1.0).all())

        res_tuple = normalize_feature((0.0, 100.0), max_val=100.0)
        self.assertIsInstance(res_tuple, np.ndarray)
        self.assertEqual(res_tuple[0], 0.0)
        self.assertEqual(res_tuple[1], 1.0)


if __name__ == "__main__":
    unittest.main()
