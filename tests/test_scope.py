"""Products must be for sale and have a genuine purchase before the cutoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from recommender.catalogue import load_eligible_products, load_product_catalogue, select_eligible_products
from recommender.data import select_paid_purchases
from recommender.scope import ProductScope, eligible_product_mask, load_product_scope, sellable_product_mask


def product_catalogue():
    return pd.DataFrame([
        {"product_id": identity, "item_type": kind, "product_category": "АВТО-СМ"}
        for identity, kind in [
            ("OIL", "ТОВАР"), ("GIFT_ONLY", "ТОВАР"), ("UNSOLD", "ТОВАР"),
            ("FUTURE", "ТОВАР"), ("AT_CUTOFF", "ТОВАР"), ("INTERNAL", "ЗАПАСЫ"),
        ]
    ])


def purchase_records():
    return pd.DataFrame([
        {
            "customer_id": "company-1", "product_id": identity, "item_type": kind,
            "transaction_type": transaction, "purchase_date": pd.Timestamp(date),
            "quantity": 1.0, "product_name": "Synthetic product",
            "product_category": "АВТО-СМ", "business_line": "PCMO",
        }
        for identity, kind, transaction, date in [
            ("OIL", "ТОВАР", "ПРОДАЖА", "2026-01-10"),
            ("GIFT_ONLY", "ТОВАР", "ПОДАРОК", "2026-01-10"),
            ("FUTURE", "ТОВАР", "ПРОДАЖА", "2026-01-21"),
            ("AT_CUTOFF", "ТОВАР", "ПРОДАЖА", "2026-01-20"),
            ("INTERNAL", "ЗАПАСЫ", "ПРОДАЖА", "2026-01-10"),
        ]
    ])


class ProductScopeTests(unittest.TestCase):
    def test_for_sale_and_prior_paid_purchase_are_both_required(self):
        products = product_catalogue()
        sales = purchase_records()
        eligible = select_eligible_products(products, sales, "2026-01-20")
        self.assertEqual(eligible.product_id.tolist(), ["OIL"])
        self.assertEqual(eligible_product_mask(products, sales, "2026-01-20").tolist(),
                         [True, False, False, False, False, False])
        self.assertEqual(set(eligible.product_id),
                         set(select_paid_purchases(sales[sales.purchase_date.lt(pd.Timestamp("2026-01-20"))],
                                                   products=products).product_id))

    def test_future_sales_do_not_change_previous_eligibility(self):
        products = product_catalogue()
        before = purchase_records().iloc[[0]].copy()
        actual = select_eligible_products(products, before, "2026-01-20").product_id.tolist()
        after = select_eligible_products(products, purchase_records(), "2026-01-20").product_id.tolist()
        self.assertEqual(actual, after)
        self.assertIn("FUTURE", select_eligible_products(products, purchase_records(), "2026-01-22").product_id.tolist())

    def test_category_brand_and_id_prefix_are_not_sale_evidence(self):
        products = product_catalogue().iloc[[0]].copy()
        products["product_id"] = "SKU-001"
        products["product_category"] = "OTHER-CATEGORY"
        products["brand"] = "OTHER-BRAND"
        sale = purchase_records().iloc[[0]].copy()
        sale["product_id"] = "SKU-001"
        self.assertEqual(select_eligible_products(products, sale, "2026-01-20").product_id.tolist(), ["SKU-001"])

    def test_missing_classification_and_missing_evidence_fail_closed(self):
        products = pd.DataFrame({"item_type": [" Товар ", None, "ЗАПАСЫ", "UNKNOWN"]})
        self.assertEqual(sellable_product_mask(products).tolist(), [True, False, False, False])
        with self.assertRaisesRegex(ValueError, "eligibility columns"):
            sellable_product_mask(products.drop(columns="item_type"))
        with self.assertRaisesRegex(ValueError, "actual-purchase columns"):
            eligible_product_mask(product_catalogue(), purchase_records().drop(columns="transaction_type"), "2026-01-20")

    def test_catalogue_positions_and_duplicate_identity_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "items.csv"
            rows = []
            for record in product_catalogue().to_dict("records"):
                row = [""] * 85
                row[1], row[50], row[84] = record["product_id"], record["item_type"], record["product_category"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, sep=";", header=False, index=False)
            self.assertEqual(load_eligible_products(path, purchase_records(), "2026-01-20").product_id.tolist(), ["OIL"])
            pd.DataFrame([rows[0], rows[0]]).to_csv(path, sep=";", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_product_catalogue(path)

    def test_policy_cannot_allow_unsold_products(self):
        with self.assertRaisesRegex(ValueError, "minimum_prior_purchases"):
            ProductScope(version=1, allowed_item_types=frozenset({"ТОВАР"}), minimum_prior_purchases=0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scope.toml"
            path.write_text('version = 1\nallowed_item_types = ["ТОВАР"]\nminimum_prior_purchases = 0\n')
            with self.assertRaisesRegex(ValueError, "minimum_prior_purchases"):
                load_product_scope(path)


if __name__ == "__main__":
    unittest.main()
