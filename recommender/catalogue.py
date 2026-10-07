"""Read source classification and expose the shared inference product scope."""

from pathlib import Path

import pandas as pd

from recommender.scope import ProductScope, eligible_product_mask, sellable_product_mask


# Zero-based export positions; source CSV has no header.
ITEM_COLUMN_MAPPING = {
    1: "product_id", 3: "short_name", 5: "product_type", 6: "unit",
    50: "item_type", 59: "business_line", 83: "brand",
    84: "product_category", 86: "product_subcategory",
    121: "ukrainian_name", 125: "product_family",
    132: "product_name", 133: "base_name",
}


def load_product_catalogue(input_path: str | Path) -> pd.DataFrame:
    """Load the headerless export and validate exact, unique catalogue identities."""
    raw = pd.read_csv(
        input_path, sep=";", header=None, dtype="string",
        encoding="utf-8-sig", keep_default_na=False,
    )
    if not {1, 50, 84}.issubset(raw.columns):
        raise ValueError("Item export must include ID, type and category columns 2, 51 and 85.")
    available = [column for column in ITEM_COLUMN_MAPPING if column in raw.columns]
    products = raw[available].rename(columns=ITEM_COLUMN_MAPPING).copy()
    for column in products.columns:
        products[column] = products[column].str.strip()
        products[column] = products[column].mask(products[column].isin(["", "NULL"]))
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("Item export contains missing or duplicate product IDs.")
    for column in ["item_type", "product_category", "business_line"]:
        if column in products:
            products[column] = products[column].str.upper()
    return products


def select_sellable_products(
    products: pd.DataFrame, *, scope: ProductScope | None = None
) -> pd.DataFrame:
    """Classify for-sale products; candidate eligibility also requires prior paid sales."""
    if "product_id" not in products:
        raise ValueError("Product catalogue requires product_id.")
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("Product catalogue contains missing or duplicate product IDs.")
    return products.loc[sellable_product_mask(products, scope=scope)].copy()


def select_eligible_products(
    products: pd.DataFrame, purchases: pd.DataFrame, scoring_time: str | pd.Timestamp,
    *, scope: ProductScope | None = None,
) -> pd.DataFrame:
    """Shared training/inference candidates: for sale and already sold at least once."""
    sellable = select_sellable_products(products, scope=scope)
    return sellable.loc[
        eligible_product_mask(sellable, purchases, scoring_time, scope=scope)
    ].copy()


def load_eligible_products(
    input_path: str | Path, purchases: pd.DataFrame, scoring_time: str | pd.Timestamp,
    *, scope: ProductScope | None = None,
) -> pd.DataFrame:
    """Inference entrypoint: never retrieve or score the unfiltered item master."""
    return select_eligible_products(
        load_product_catalogue(input_path), purchases, scoring_time, scope=scope
    )
