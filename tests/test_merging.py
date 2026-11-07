"""Shared merging preserves quantities, metadata and exact entity identities."""

import json
import unittest

import pandas as pd

from recommender.merging import merge_purchase_events, merge_records


class RecordMergingTests(unittest.TestCase):
    def test_sums_amounts_coalesces_metadata_and_preserves_source_order(self):
        records = pd.DataFrame([
            {"id": "B", "quantity": 2, "name": None, "unit_price": 10},
            {"id": "A", "quantity": 4, "name": "Item A", "unit_price": 20},
            {"id": "B", "quantity": 3, "name": "Item B", "unit_price": 10},
        ])
        original = records.copy(deep=True)
        merged = merge_records(records, keys=["id"], sum_columns=["quantity"])
        self.assertEqual(merged.id.tolist(), ["B", "A"])
        self.assertEqual(merged.quantity.tolist(), [5, 4])
        self.assertEqual(merged.name.tolist(), ["Item B", "Item A"])
        self.assertEqual(merged.unit_price.tolist(), [10, 20])
        self.assertEqual(merged.columns.tolist(), records.columns.tolist())
        pd.testing.assert_frame_equal(records, original)
        pd.testing.assert_frame_equal(merge_records(merged, keys=["id"], sum_columns=["quantity"]), merged)

    def test_conflicts_and_missing_keys_raise_instead_of_losing_records(self):
        with self.assertRaisesRegex(ValueError, "Conflicting metadata.*unit"):
            merge_records(pd.DataFrame({"id": ["A", "A"], "unit": ["kg", "L"]}), keys=["id"])
        for key in (None, " "):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "nonmissing keys"):
                merge_records(pd.DataFrame({"id": [key], "quantity": [2]}), keys=["id"], sum_columns=["quantity"])

    def test_empty_and_identity_only_tables_keep_their_schema(self):
        records = pd.DataFrame({"id": pd.Series(dtype="string"), "quantity": pd.Series(dtype="float64")})
        pd.testing.assert_frame_equal(merge_records(records, keys=["id"], sum_columns=["quantity"]), records)
        merged = merge_records(pd.DataFrame({"id": ["B", "A", "B"]}), keys=["id"])
        self.assertEqual(merged.id.tolist(), ["B", "A"])


class PurchaseMergingTests(unittest.TestCase):
    def test_same_day_lines_sum_and_optional_text_variants_survive(self):
        purchases = pd.DataFrame([
            {"customer_id": " 001 ", "product_id": "SKU-1", "purchase_date": "2026-01-01 09:00",
             "quantity": 2, "business_line": "PCMO", "product_name": "Oil", "unit": "L"},
            {"customer_id": "001", "product_id": "SKU-1", "purchase_date": "2026-01-01 18:00",
             "quantity": 3, "business_line": "HDDO", "product_name": "Oil", "unit": "L"},
            {"customer_id": "001", "product_id": "SKU-4", "purchase_date": "2026-01-01",
             "quantity": 4, "business_line": "PCMO", "product_name": "Oil 4L", "unit": "L"},
            {"customer_id": "001", "product_id": "SKU-1", "purchase_date": "2026-01-02",
             "quantity": 1, "business_line": "PCMO", "product_name": "Oil", "unit": "L"},
        ])
        merged = merge_purchase_events(purchases)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.quantity.tolist(), [5, 4, 1])
        self.assertEqual(merged.quantity.sum(), purchases.quantity.sum())
        self.assertEqual(json.loads(merged.business_line.iloc[0]), ["PCMO", "HDDO"])
        self.assertEqual(merged.unit.tolist(), ["L", "L", "L"])
        pd.testing.assert_frame_equal(merge_purchase_events(merged), merged)
        new_line = purchases.iloc[[0]].copy()
        new_line["quantity"] = 7
        remerged = merge_purchase_events(pd.concat([merged, new_line], ignore_index=True))
        self.assertEqual(remerged.quantity.iloc[0], 12)
        self.assertEqual(json.loads(remerged.business_line.iloc[0]), ["PCMO", "HDDO"])

    def test_classification_or_unit_conflicts_do_not_merge_silently(self):
        for column, values in [("item_type", ["ТОВАР", "ЗАПАСЫ"]), ("unit", ["kg", "L"])]:
            purchases = pd.DataFrame({
                "customer_id": ["001", "001"], "product_id": ["SKU", "SKU"],
                "purchase_date": ["2026-01-01", "2026-01-01"], "quantity": [2, 3], column: values,
            })
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Conflicting metadata"):
                merge_purchase_events(purchases)


if __name__ == "__main__":
    unittest.main()
