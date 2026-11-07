"""Read source classification and expose the shared inference product scope."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from recommender.merging import merge_records
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
    if products["product_id"].isna().any():
        raise ValueError("Item export contains missing product IDs.")
    for column in ["item_type", "product_category", "business_line", "brand", "product_subcategory"]:
        if column in products:
            products[column] = products[column].str.upper()
    return merge_records(products, keys=["product_id"])


def select_sellable_products(
    products: pd.DataFrame, *, scope: ProductScope | None = None
) -> pd.DataFrame:
    """Classify for-sale products; candidate eligibility also requires prior paid sales."""
    if "product_id" not in products:
        raise ValueError("Product catalogue requires product_id.")
    if products["product_id"].isna().any():
        raise ValueError("Product catalogue contains missing product IDs.")
    products = merge_records(products, keys=["product_id"])
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


def order_retriever_catalogue(products: pd.DataFrame, product_ids: Sequence[str]) -> pd.DataFrame:
    """Align every product tensor to the checkpoint's score and label indices."""
    indexed = products.assign(product_id=products["product_id"].astype(str)).set_index("product_id")
    if not indexed.index.is_unique or len(set(product_ids)) != len(product_ids):
        raise ValueError("Retriever catalogue product IDs must be unique.")
    return indexed.loc[list(product_ids)].reset_index()
