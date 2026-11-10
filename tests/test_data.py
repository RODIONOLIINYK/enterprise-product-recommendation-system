"""Purchase preparation checks using synthetic records only."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from openpyxl import Workbook

from recommender.catalogue import ITEM_COLUMN_MAPPING

from recommender.data import (
    COLUMN_MAPPING, aggregate_purchases, load_purchases,
    select_paid_purchases, validate_purchases,
)


def sales_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "customer_id": "company-1", "purchase_date": pd.Timestamp("2026-01-10"),
            "product_id": product_id, "quantity": quantity, "product_name": name,
            "product_category": "АВТО-СМ", "business_line": "PCMO",
            "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР",
        }
        for product_id, quantity, name in [
            ("ТОВ-001", 1.0, "Oil, 1л"), ("ТОВ-001", 2.0, "Oil, 1л"),
            ("ТОВ-004", 4.0, "Oil, 4л"),
        ]
    ])


def write_item_catalogue(path: Path, sales: pd.DataFrame) -> None:
    """Write a minimal headerless item export in the real column positions."""
    rows = []
    for record in sales.drop_duplicates("product_id").to_dict("records"):
        row = [""] * 134
        for source, destination in ITEM_COLUMN_MAPPING.items():
            row[source] = record.get(destination, "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep=";", header=False, index=False)


class PurchasePreparationTests(unittest.TestCase):
    def test_package_variants_and_event_quantities_survive(self):
        paid = sales_frame()
        cleaned = aggregate_purchases(paid)
        validate_purchases(cleaned, paid)
        self.assertEqual(cleaned["product_id"].tolist(), ["ТОВ-001", "ТОВ-004"])
        self.assertEqual(cleaned["product_name"].tolist(), ["Oil, 1л", "Oil, 4л"])
        self.assertEqual(cleaned["quantity"].tolist(), [3.0, 4.0])

    def test_excludes_gifts_returns_invalid_and_nonmerchandise_rows(self):
        valid = sales_frame().iloc[[0]].copy()
        variants = []
        for column, value in [
            ("transaction_type", "ПОДАРОК"), ("transaction_type", "ВОЗВРАТ"),
            ("item_type", "БОНУС КЛН"), ("item_type", "ЗАПАСЫ"),
            ("quantity", 0), ("quantity", -1), ("quantity", np.inf),
            ("quantity", np.nan), ("purchase_date", pd.NaT),
            ("customer_id", "   "), ("product_id", pd.NA),
        ]:
            row = valid.copy()
            row[column] = value
            variants.append(row)
        self.assertEqual(len(select_paid_purchases(pd.concat([valid, *variants]))), 1)

    def test_validation_catches_reassigned_quantities_and_duplicate_events(self):
        paid = sales_frame()
        cleaned = aggregate_purchases(paid)
        changed = cleaned.copy()
        changed["quantity"] = [4.0, 3.0]
        with self.assertRaisesRegex(ValueError, "event quantities"):
            validate_purchases(changed, paid)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_purchases(pd.concat([cleaned, cleaned.iloc[[0]]]), paid)

    def test_blank_first_row_header_detection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sales.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([None] * len(COLUMN_MAPPING))
            sheet.append(list(COLUMN_MAPPING))
            for row in sales_frame().to_dict("records"):
                sheet.append([row[column] for column in COLUMN_MAPPING.values()])
            workbook.save(path)
            write_item_catalogue(Path(directory) / "items.csv", sales_frame())
            cleaned = load_purchases(path)
            self.assertEqual(len(cleaned), 2)
            self.assertEqual(cleaned["quantity"].sum(), 7.0)

    def test_missing_source_header_fails_clearly(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xlsx"
            workbook = Workbook()
            workbook.active.append(["unexpected field"])
            workbook.save(path)
            write_item_catalogue(Path(directory) / "items.csv", sales_frame())
            with self.assertRaisesRegex(ValueError, "headers not found"):
                load_purchases(path)

    def test_gift_only_dates_are_not_purchase_baskets(self):
        paid = sales_frame().iloc[[0]].copy()
        gift = paid.copy()
        gift["purchase_date"] = pd.Timestamp("2026-01-20")
        gift["transaction_type"] = "ПОДАРОК"
        history = aggregate_purchases(select_paid_purchases(pd.concat([paid, gift])))
        self.assertEqual(history["purchase_date"].tolist(), [pd.Timestamp("2026-01-10")])
        self.assertEqual(history["quantity"].sum(), 1.0)

    def test_mixed_day_retains_paid_products_but_never_gifts(self):
        paid = sales_frame().iloc[[0]].copy()
        gift = sales_frame().iloc[[2]].copy()
        gift["transaction_type"] = "ПОДАРОК"
        history = aggregate_purchases(select_paid_purchases(pd.concat([paid, gift])))
        self.assertEqual(history["product_id"].tolist(), ["ТОВ-001"])
        self.assertEqual(history["quantity"].sum(), 1.0)

    def test_classification_not_product_id_prefix_defines_scope(self):
        sale = sales_frame().iloc[[0]].copy()
        sale["product_id"] = "SKU-001"
        self.assertEqual(len(select_paid_purchases(sale)), 1)
        sale["product_id"] = "ТОВ-TOOL"
        sale["item_type"] = "ЗАПАСЫ"
        self.assertTrue(select_paid_purchases(sale).empty)

    def test_training_requires_membership_in_for_sale_catalogue(self):
        sale = sales_frame().iloc[[0]].copy()
        products = sale[["product_id", "item_type", "product_category"]].copy()
        products["item_type"] = "ЗАПАСЫ"
        self.assertTrue(select_paid_purchases(sale, products=products).empty)
        products["product_id"] = "UNMATCHED"
        products["item_type"] = "ТОВАР"
        self.assertTrue(select_paid_purchases(sale, products=products).empty)

    def test_missing_optional_metadata_does_not_remove_a_genuine_sale(self):
        sale = sales_frame().iloc[[0]].copy()
        sale[["product_name", "product_category", "business_line"]] = pd.NA
        selected = select_paid_purchases(sale)
        cleaned = aggregate_purchases(selected)
        validate_purchases(cleaned, selected)
        self.assertEqual(cleaned["product_id"].tolist(), ["ТОВ-001"])


if __name__ == "__main__":
    unittest.main()
