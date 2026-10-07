"""Adapted purchase preparation that preserves exact sellable product IDs."""

from pathlib import Path

import numpy as np
import pandas as pd

from recommender.catalogue import load_product_catalogue, select_sellable_products
from recommender.scope import ProductScope, actual_purchase_mask


COLUMN_MAPPING = {
    "КлиентДляОплатыКод": "customer_id",
    "ТоварКод": "product_id",
    "Товар": "product_name",
    "Категория": "product_category",
    "БизнесЛиния": "business_line",
    "ДатаПродажи": "purchase_date",
    "Количество": "quantity",
    "Gen_ Bus_ Posting Group": "transaction_type",
    "Gen_ Prod_ Posting Group": "item_type",
}
REQUIRED_TEXT_COLUMNS = [
    "customer_id", "product_id",
]
SOURCE_TEXT_COLUMNS = [*REQUIRED_TEXT_COLUMNS, "product_name", "product_category", "business_line"]
HISTORY_COLUMNS = [
    "customer_id", "purchase_date", "product_id", "quantity",
    "product_category", "business_line", "product_name",
    "item_type", "transaction_type",
]
REQUIRED_HISTORY_COLUMNS = set(HISTORY_COLUMNS)
EVENT_KEY_COLUMNS = ["customer_id", "purchase_date", "product_id"]


def load_source_purchases(input_path: str | Path) -> pd.DataFrame:
    """Preserve the notebook's first-20-rows header detection and normalization."""
    with pd.ExcelFile(input_path) as workbook:
        preview = pd.read_excel(workbook, sheet_name=0, header=None, nrows=20)
        header_row = next(
            (
                index for index, row in preview.iterrows()
                if set(COLUMN_MAPPING).issubset(set(row.dropna()))
            ),
            None,
        )
        if header_row is None:
            raise ValueError("Expected source column headers not found in the first 20 rows.")
        purchases = pd.read_excel(
            workbook, sheet_name=0, header=header_row,
            usecols=lambda column: column in COLUMN_MAPPING,
            dtype={"КлиентДляОплатыКод": "string", "ТоварКод": "string"},
        )
    purchases = purchases.rename(columns=COLUMN_MAPPING)
    for column in [*SOURCE_TEXT_COLUMNS, "transaction_type", "item_type"]:
        purchases[column] = purchases[column].astype("string").str.strip()
    for column in ["transaction_type", "item_type"]:
        purchases[column] = purchases[column].str.upper()
    purchases["purchase_date"] = pd.to_datetime(
        purchases["purchase_date"], errors="coerce"
    ).dt.normalize()
    purchases["quantity"] = pd.to_numeric(purchases["quantity"], errors="coerce")
    return purchases


def complete_history_row_mask(purchases: pd.DataFrame) -> pd.Series:
    """Check the existing required fields, including finite quantities."""
    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(purchases.columns))
    if missing:
        raise ValueError(f"Missing required history columns: {missing}")
    text = purchases[REQUIRED_TEXT_COLUMNS].apply(
        lambda values: values.astype("string").str.strip()
    )
    return (
        text.notna().all(axis=1)
        & text.ne("").all(axis=1)
        & purchases["purchase_date"].notna()
        & np.isfinite(purchases["quantity"]).fillna(False)
    )


def select_paid_purchases(
    purchases: pd.DataFrame, *, products: pd.DataFrame | None = None,
    scope: ProductScope | None = None,
) -> pd.DataFrame:
    """Keep valid sales in the shared business scope, not an ID-prefix heuristic."""
    mask = (
        complete_history_row_mask(purchases)
        & actual_purchase_mask(purchases, scope=scope)
    )
    if products is not None:
        sellable_ids = select_sellable_products(products, scope=scope)["product_id"]
        mask &= purchases["product_id"].isin(sellable_ids)
    return purchases.loc[mask].copy()


def aggregate_purchases(paid_purchases: pd.DataFrame) -> pd.DataFrame:
    """Reuse company/date/product aggregation without remapping any SKU."""
    cleaned = (
        paid_purchases.groupby(EVENT_KEY_COLUMNS, as_index=False, sort=False)
        .agg(
            quantity=("quantity", "sum"),
            product_category=("product_category", "first"),
            business_line=("business_line", "first"),
            product_name=("product_name", "first"),
            item_type=("item_type", "first"),
            transaction_type=("transaction_type", "first"),
        )
        .sort_values(EVENT_KEY_COLUMNS)
        .reset_index(drop=True)
    )
    return cleaned[HISTORY_COLUMNS]


def validate_purchases(
    cleaned: pd.DataFrame, paid: pd.DataFrame, *, products: pd.DataFrame | None = None,
    scope: ProductScope | None = None,
) -> None:
    """Validate event uniqueness, paid-sale scope, quantities and product IDs."""
    if not complete_history_row_mask(cleaned).all() or not actual_purchase_mask(cleaned, scope=scope).all():
        raise ValueError("Cleaned history contains incomplete or non-finite values.")
    if cleaned.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("Cleaned history contains duplicate purchase events.")
    if not cleaned["quantity"].gt(0).all():
        raise ValueError("Cleaned history contains non-positive quantities.")
    if len(select_paid_purchases(paid, products=products, scope=scope)) != len(paid):
        raise ValueError("Source history contains rows outside the paid-sale scope.")
    if set(cleaned["product_id"]) != set(paid["product_id"]):
        raise ValueError("Product IDs changed during aggregation.")
    expected = paid.groupby(EVENT_KEY_COLUMNS)["quantity"].sum().sort_index()
    actual = cleaned.set_index(EVENT_KEY_COLUMNS)["quantity"].sort_index()
    if not expected.index.equals(actual.index):
        raise ValueError("Purchase event keys changed during aggregation.")
    if not np.allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9):
        raise ValueError("Purchase event quantities changed during aggregation.")
    if not np.isclose(cleaned["quantity"].sum(), paid["quantity"].sum(), rtol=1e-9, atol=1e-9):
        raise ValueError("Total quantity changed during aggregation.")


def load_purchases(
    input_path: str | Path, *, items_path: str | Path | None = None,
    scope: ProductScope | None = None,
) -> pd.DataFrame:
    """Prepare sales with the same item catalogue policy used at inference."""
    if items_path is None:
        items_path = Path(input_path).with_name("items.csv")
    products = load_product_catalogue(items_path)
    paid = select_paid_purchases(load_source_purchases(input_path), products=products, scope=scope)
    cleaned = aggregate_purchases(paid)
    validate_purchases(cleaned, paid, products=products, scope=scope)
    return cleaned
