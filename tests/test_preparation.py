"""Company normalization and configuration-driven stage execution checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from recommender.company import count_company_purchases, load_companies, process_address
from recommender.preparation import run_preparation
from recommender.scope import get_datasets_locs


def company_row(customer_id=" 0001 "):
    row = [""] * 91
    row[1], row[2], row[82], row[88], row[90] = (
        customer_id, " Acme ", " NULL ", " Main Street ", "   ",
    )
    return row


def purchase_history():
    return pd.DataFrame([
        {"customer_id": customer, "product_id": product, "purchase_date": date,
         "quantity": quantity, "transaction_type": transaction, "item_type": "ТОВАР"}
        for customer, product, date, quantity, transaction in [
            (" 0001 ", "A", "2026-01-01", 1, "ПРОДАЖА"),
            ("0001", "A", "2026-01-01 12:00:00", 2, "ПРОДАЖА"),
            ("0001", "B", "2026-01-01", 5, "ПРОДАЖА"),
            ("0002", "A", "2026-01-02", 1, "ПРОДАЖА"),
            ("0003", "A", "2026-01-02", 1, "ПОДАРОК"),
            ("0003", "A", "2026-01-03", -1, "ВОЗВРАТ"),
            ("0003", "A", "not-a-date", 1, "ПРОДАЖА"),
            ("0003", "A", "2026-01-04", 0, "ПРОДАЖА"),
            ("9999", "A", "2026-01-02", 1, "ПРОДАЖА"),
        ]
    ])


class CompanyPreparationTests(unittest.TestCase):
    def test_loads_from_config_and_normalizes_selected_columns(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "customers.csv"
            pd.DataFrame([company_row()]).to_csv(source, sep=";", header=False, index=False)
            config = root / "datasets.toml"
            config.write_text('companies = "customers.csv"\n')
            companies = load_companies(config_path=config, purchases=purchase_history())
            self.assertEqual(companies.columns.tolist(), [
                "customer_id", "company_name", "legal_name", "address", "description", "city",
                "purchase_count",
            ])
            self.assertEqual(companies.customer_id.tolist(), ["0001"])
            self.assertEqual(companies.company_name.tolist(), ["Acme"])
            self.assertEqual(companies.address.tolist(), ["Main Street"])
            self.assertTrue(companies.legal_name.isna().all())
            self.assertTrue(companies.description.isna().all())
            self.assertTrue(companies.city.isna().all())
            self.assertIsInstance(companies.city.dtype, pd.StringDtype)
            self.assertEqual(companies.purchase_count.tolist(), [2])

    def test_preserves_address_and_derives_nullable_city(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "customers.csv"
            rows = []
            for customer_id, address in [
                ("0001", " вул. Шевченка, 1, м. Київ, Київська область, Україна "),
                ("0002", "NULL"), ("0003", "Київ, Одеса, Україна"),
            ]:
                row = company_row(customer_id)
                row[88] = address
                rows.append(row)
            pd.DataFrame(rows).to_csv(source, sep=";", header=False, index=False)
            companies = load_companies(source, purchases=purchase_history(), minimum_purchases=0)
            self.assertEqual(companies.address.iloc[0], rows[0][88].strip())
            self.assertEqual(companies.city.iloc[0], "Київ")
            self.assertTrue(companies.city.iloc[1:].isna().all())

    def test_rejects_missing_ids_after_normalization(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "customers.csv"
            for rows in ([company_row("NULL")], [company_row(" ")]):
                with self.subTest(rows=len(rows)):
                    pd.DataFrame(rows).to_csv(source, sep=";", header=False, index=False)
                    with self.assertRaisesRegex(ValueError, "customer IDs"):
                        load_companies(source, purchases=purchase_history())

    def test_merges_duplicate_company_profiles_and_fills_missing_metadata(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "customers.csv"
            first, second = company_row(" 0001 "), company_row("0001")
            first[88], second[90] = "NULL", "Supplies machinery"
            pd.DataFrame([first, second]).to_csv(source, sep=";", header=False, index=False)
            companies = load_companies(source, purchases=purchase_history())
            self.assertEqual(len(companies), 1)
            self.assertEqual(companies.address.iloc[0], "Main Street")
            self.assertEqual(companies.description.iloc[0], "Supplies machinery")
            self.assertEqual(companies.purchase_count.iloc[0], 2)
            second[2] = "Another company"
            pd.DataFrame([first, second]).to_csv(source, sep=";", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting metadata.*company_name"):
                load_companies(source, purchases=purchase_history())

    def test_missing_columns_report_company_export_error(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "customers.csv"
            source.write_text("ignored;0001;Acme\n")
            with self.assertRaisesRegex(ValueError, "Company export.*columns"):
                load_companies(source, purchases=purchase_history())

    def test_keeps_companies_with_one_or_more_purchases_and_matches_exact_ids(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "customers.csv"
            pd.DataFrame([company_row(identity) for identity in ["0001", "0002", "0003", "0004", "1"]]).to_csv(
                source, sep=";", header=False, index=False,
            )
            companies = load_companies(source, purchases=purchase_history())
            self.assertEqual(companies.customer_id.tolist(), ["0001", "0002"])
            self.assertEqual(companies.purchase_count.tolist(), [2, 1])
            all_companies = load_companies(source, purchases=purchase_history(), minimum_purchases=0)
            self.assertEqual(all_companies.purchase_count.tolist(), [2, 1, 0, 0, 0])
            self.assertEqual(companies.index.tolist(), [0, 1])

    def test_automatic_purchase_loading_uses_configured_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "customers.csv"
            pd.DataFrame([company_row("0002")]).to_csv(source, sep=";", header=False, index=False)
            config = root / "datasets.toml"
            config.write_text('companies = "customers.csv"\npurchases = "sales.xlsx"\nitems = "catalogue.csv"\n')
            with patch("recommender.company.load_purchases", return_value=purchase_history()) as loader:
                companies = load_companies(config_path=config)
                loader.assert_called_once_with(root / "sales.xlsx", items_path=root / "catalogue.csv")
            self.assertEqual(companies.customer_id.tolist(), ["0002"])
            self.assertEqual(companies.purchase_count.tolist(), [1])

    def test_invalid_threshold_fails_before_reading_datasets(self):
        for threshold in [-1, True, 1.5, "1"]:
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "minimum_purchases"):
                load_companies("missing.csv", minimum_purchases=threshold)


class CompanyPurchaseCountTests(unittest.TestCase):
    def test_counts_paid_events_without_mutating_input_or_counting_duplicates(self):
        purchases = purchase_history()
        original = purchases.copy(deep=True)
        counts = count_company_purchases(purchases)
        self.assertEqual(counts.to_dict(), {"0001": 2, "0002": 1, "9999": 1})
        pd.testing.assert_frame_equal(purchases, original)

    def test_empty_history_and_missing_schema(self):
        self.assertTrue(count_company_purchases(purchase_history().iloc[:0]).empty)
        with self.assertRaisesRegex(ValueError, "actual-purchase columns"):
            count_company_purchases(pd.DataFrame({"user_id": ["0001"]}))


class AddressParsingTests(unittest.TestCase):
    def test_explicit_locality_markers_and_inline_details(self):
        cases = {
            "Україна, 01001, м.Київ, вул. Хрещатик, 1": "Київ",
            "Г.Харьков, УЛ. Школьная, 5": "Харьков",
            "місто Івано-Франківськ, вул. Шевченка, 12": "Івано-Франківськ",
            "город Белая Церковь, улица Мира, 3": "Белая Церковь",
            "смт.Куликів, вул. Шевченка, 73, Жовква, Львівська область": "Куликів",
            "пгт. Гребенки, Киевская область, 08662, Україна": "Гребенки",
            "с.Новозлатополь ул.Советская 20, Гуляйполе, Запорізька область": "Новозлатополь",
            "Київська обл., селище Коцюбинське, вул. Лісова": "Коцюбинське",
            "село П'ятихатки, вул. Центральна, 1": "П'ятихатки",
            "пос. Васищево,ул. Орешковая, Харків, Харківська область": "Васищево",
            "м. Кривий Ріг Дніпропетровська обл.": "Кривий Ріг",
            "м. Київ вул. Хрещатик 1": "Київ",
            "м. Київ 01001 Україна": "Київ",
            "м. Кривий Ріг 50000 Україна": "Кривий Ріг",
            "с. Федоровка-1, вул. Польова, 30": "Федоровка-1",
            "Вул. Родянська, c. Новоохтирка, Луганська область": "Новоохтирка",
            "м. Київ, м.Київ, Україна": "Київ",
            "м.13, м. Черкаси, Черкаська область": "Черкаси",
            "Відділення №5 м. Чорноморськ": "Чорноморськ",
            "вул. Хрещатик 1 м. Київ": "Київ",
        }
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(process_address(address), expected)

    def test_unmarked_cities_exclude_streets_regions_and_house_numbers(self):
        cases = {
            "Мурманская,41,Кіровоград,Кіровоградська область,25009,Украина": "Кіровоград",
            "Ленина, 130, Синельникове, Дніпропетровська область": "Синельникове",
            "пер. Станционный, 3, Охтирка, Сумська область, 42734": "Охтирка",
            "Южное шоссе, 48, Запорожье, Запорожская область": "Запорожье",
            "ВУЛ.БУДІВЕЛЬНИКІВ, БУД. 16, Комсомольськ, Полтавська область": "Комсомольськ",
            "Київ, Хрещатик, 1, Україна": "Київ",
            "Київ, вул. Хрещатик, 1": "Київ",
            "Київська область, Бровари, вул. Незалежності, 1": "Бровари",
            "07400 Бровари, Україна": "Бровари",
            "Кривий Ріг": "Кривий Ріг",
            "Московская, 11, Феодосія, АР Крим, 98100, Україна": "Феодосія",
            "Алупка, АР Крим": "Алупка",
            "Донецьк; Донецька область; Україна": "Донецьк",
            "вул. М. Троїцького, буд. &nbsp;1, Одеса, Одеська область": "Одеса",
            "\nУкраїна\nм.\u00a0Київ\nвул. Лісова\n1": "Київ",
        }
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(process_address(address), expected)

    def test_missing_nonaddress_and_ambiguous_values_remain_unknown(self):
        cases = [None, pd.NA, float("nan"), 123, "", "   ", " NULL ",
                 "Украина", "Германия", "Київська область", "Броварський р-н",
                 "UA\\shovkun", "вул. Шевченка, 1", "вул. М. Троїцького, 1",
                 "Ленина, 130", "12345", "Київ, Одеса, Україна", "м. Київ, г. Одеса",
                 "с.Попово В.Багачанского р-на, Велика Багачка, Полтавська область",
                 "смт. Степногорск Ленина 1, Василівка, Запорізька область"]
        for address in cases:
            with self.subTest(address=address):
                self.assertIsNone(process_address(address))


class DatasetConfigurationTests(unittest.TestCase):
    def test_relative_absolute_paths_and_customer_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "datasets.toml"
            config.write_text(f'companies = "raw/customers.csv"\nitems = "{root / "items.csv"}"\n')
            self.assertEqual(get_datasets_locs("customers", config), root / "raw/customers.csv")
            self.assertEqual(get_datasets_locs("items", config), root / "items.csv")
            with self.assertRaisesRegex(ValueError, "Unknown dataset"):
                get_datasets_locs("unknown", config)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                get_datasets_locs("purchases", config)
            config.write_text('companies = " "\n')
            with self.assertRaisesRegex(ValueError, "nonempty"):
                get_datasets_locs("companies", config)

    def test_order_override_and_invalid_order_before_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for filename in ("customers.csv", "items.csv", "dataset.xlsx"):
                (root / filename).touch()
            config = root / "datasets.toml"
            config.write_text(
                'companies = "customers.csv"\npurchases = "dataset.xlsx"\n'
                'items = "items.csv"\n[pipeline]\norder = ["companies", "items", "purchases"]\n'
            )
            calls = []

            def record(stage):
                def loader(*args, **kwargs):
                    calls.append(stage)
                    return pd.DataFrame()
                return loader

            with (
                patch("recommender.preparation.load_companies", side_effect=record("companies")) as companies,
                patch("recommender.preparation.load_product_catalogue", side_effect=record("items")),
                patch("recommender.preparation.load_purchases", side_effect=record("purchases")) as purchases,
            ):
                tables = run_preparation(config_path=config)
                self.assertEqual(calls, ["companies", "items", "purchases"])
                self.assertEqual(list(tables), calls)
                purchases.assert_called_once_with(root / "dataset.xlsx", items_path=root / "items.csv")
                calls.clear()
                reordered = run_preparation(config_path=config, order=["purchases", "companies"])
                self.assertEqual(calls, ["purchases", "companies"])
                self.assertIs(companies.call_args.kwargs["purchases"], reordered["purchases"])
                self.assertEqual(companies.call_args.kwargs["config_path"], config)
                calls.clear()
                for invalid in ([], "companies", ["unknown"], ["items", "items"], [None]):
                    with self.subTest(order=invalid), self.assertRaises(ValueError):
                        run_preparation(config_path=config, order=invalid)
                self.assertEqual(calls, [])
                (root / "items.csv").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Dataset file not found"):
                    run_preparation(config_path=config)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
