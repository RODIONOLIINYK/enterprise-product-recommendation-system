"""Unit tests for Phase 3: Metadata Preparation (R1)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from recommender.catalogue import ITEM_COLUMN_MAPPING, load_product_catalogue
from recommender.company import (
    COMPANY_COLUMN_MAPPING,
    EXCLUDED_COMPANY_COLUMNS,
    load_companies,
)


def make_synthetic_customer_row(
    customer_id="C001",
    company_name="Acme Corp",
    legal_name="Acme Legal LLC",
    region_code="UA_02_02",
    address="м. Київ, вул. Шевченка, 1",
    description="Manufacturer of lubricants",
    industry_branch="ТБД-014",
    sales_branch="ДНЕПРГЛАВ",
    vat_status="К_ПЛАТ",
    city_code="ГОР-000062",
):
    row = [""] * 118
    row[1] = customer_id
    row[2] = company_name
    row[82] = legal_name
    row[85] = region_code
    row[88] = address
    row[90] = description
    row[94] = industry_branch
    # Unwanted columns in raw export
    row[44] = sales_branch
    row[59] = vat_status
    row[98] = city_code
    return row


def make_synthetic_item_row(
    product_id="PROD-001",
    item_type="ТОВАР",
    business_line="PCMO",
    brand="SHELL",
    product_category="АВТО-СМ",
    product_subcategory="МОТОР-МАСЛА",
    product_name="Helix Ultra 5W-40 4л",
):
    row = [""] * 135
    row[1] = product_id
    row[50] = item_type
    row[59] = business_line
    row[83] = brand
    row[84] = product_category
    row[86] = product_subcategory
    row[132] = product_name
    return row


def dummy_purchases():
    return pd.DataFrame([
        {
            "customer_id": "C001",
            "product_id": "PROD-001",
            "purchase_date": pd.Timestamp("2025-05-01"),
            "quantity": 2.0,
            "transaction_type": "ПРОДАЖА",
            "item_type": "ТОВАР",
        }
    ])


class MetadataPreparationTests(unittest.TestCase):
    def test_customer_schema_retains_required_fields(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "customers.csv"
            row = make_synthetic_customer_row()
            pd.DataFrame([row]).to_csv(source, sep=";", header=False, index=False)

            companies = load_companies(source, purchases=dummy_purchases(), minimum_purchases=0)

            expected_columns = [
                "customer_id",
                "company_name",
                "legal_name",
                "address",
                "description",
                "city",
                "region_code",
                "industry_branch",
                "purchase_count",
            ]
            self.assertEqual(companies.columns.tolist(), expected_columns)
            self.assertEqual(companies["region_code"].iloc[0], "UA_02_02")
            self.assertEqual(companies["industry_branch"].iloc[0], "ТБД-014")
            self.assertEqual(companies["city"].iloc[0], "Київ")
            self.assertEqual(companies["purchase_count"].iloc[0], 1)

    def test_customer_schema_excludes_dropped_fields(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "customers.csv"
            row = make_synthetic_customer_row()
            pd.DataFrame([row]).to_csv(source, sep=";", header=False, index=False)

            companies = load_companies(source, purchases=dummy_purchases(), minimum_purchases=0)

            for excluded_col in ["sales_branch", "vat_status", "city_code"]:
                self.assertNotIn(excluded_col, companies.columns)

            for idx in [44, 59, 98]:
                self.assertIn(idx, EXCLUDED_COMPANY_COLUMNS)
                self.assertNotIn(idx, COMPANY_COLUMN_MAPPING)

    def test_customer_metadata_normalization(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "customers.csv"
            row = make_synthetic_customer_row(
                region_code="  UA_16_02  ",
                industry_branch=" NULL ",
            )
            pd.DataFrame([row]).to_csv(source, sep=";", header=False, index=False)

            companies = load_companies(source, purchases=dummy_purchases(), minimum_purchases=0)
            self.assertEqual(companies["region_code"].iloc[0], "UA_16_02")
            self.assertTrue(pd.isna(companies["industry_branch"].iloc[0]))

    def test_product_catalogue_retains_required_fields(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "items.csv"
            row = make_synthetic_item_row()
            pd.DataFrame([row]).to_csv(source, sep=";", header=False, index=False)

            catalogue = load_product_catalogue(source)

            for expected in ["brand", "product_category", "product_subcategory", "business_line"]:
                self.assertIn(expected, catalogue.columns)

            self.assertEqual(catalogue["brand"].iloc[0], "SHELL")
            self.assertEqual(catalogue["product_category"].iloc[0], "АВТО-СМ")
            self.assertEqual(catalogue["product_subcategory"].iloc[0], "МОТОР-МАСЛА")
            self.assertEqual(catalogue["business_line"].iloc[0], "PCMO")

    def test_container_type_note_in_plan(self):
        plan_path = Path(__file__).resolve().parents[1] / "plan.md"
        self.assertTrue(plan_path.is_file(), "plan.md must exist at project root")
        content = plan_path.read_text(encoding="utf-8")
        self.assertIn("container_type", content)
        self.assertIn("discussion with the user", content)


if __name__ == "__main__":
    unittest.main()
