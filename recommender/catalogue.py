"""Read source classification and expose the shared inference product scope."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from recommender.merging import merge_records
from recommender.scope import ProductScope, actual_purchase_mask, eligible_product_mask, sellable_product_mask


# Zero-based export positions; source CSV has no header.
ITEM_COLUMN_MAPPING = {
    1: "product_id", 3: "short_name", 5: "product_type", 6: "unit",
    # Stored Dynamics NAV item fields immediately before pricing and posting groups.
    46: "blocked", 47: "last_date_modified",
    50: "item_type", 59: "business_line", 83: "brand",
    84: "product_category", 86: "product_subcategory",
    100: "container_type",
    121: "ukrainian_name", 125: "product_family",
    132: "product_name", 133: "base_name",
}

CONTAINER_TYPES = {
    "БОЧКА", "ВЕДРО", "КАНИСТРА", "ТУБА", "СПРЕЙ", "ДОЙ-ПАК",
    "BULK", "РАСПЫЛИТ.", "FLEX", "МЕШОК", "БАНКА",
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
    if "container_type" in products:
        values = products["container_type"].str.upper()
        products["container_type"] = values.where(values.isin(CONTAINER_TYPES))
    if "blocked" in products:
        products["blocked"] = products["blocked"].map({"0": False, "1": True}).astype("boolean")
    if "last_date_modified" in products:
        modified = pd.to_datetime(products["last_date_modified"], format="mixed", errors="coerce").dt.normalize()
        products["last_date_modified"] = modified.mask(modified.eq(pd.Timestamp("1753-01-01")))
    return merge_records(products, keys=["product_id"])


def select_sellable_products(
    products: pd.DataFrame, *, scope: ProductScope | None = None
) -> pd.DataFrame:
    """Select merchandise by posting group, independently of block status."""
    if "product_id" not in products:
        raise ValueError("Product catalogue requires product_id.")
    if products["product_id"].isna().any():
        raise ValueError("Product catalogue contains missing product IDs.")
    products = merge_records(products, keys=["product_id"])
    return products.loc[sellable_product_mask(products, scope=scope)].copy()


def catalogue_blocked_mask(
    products: pd.DataFrame, as_of_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Use current block status, or only snapshot blocks known before a historical date."""
    blocked = products.get("blocked", pd.Series(False, index=products.index)).eq(True).fillna(False)
    if as_of_date is None:
        return blocked
    cutoff = pd.Timestamp(as_of_date).normalize()
    if pd.isna(cutoff):
        raise ValueError("Catalogue block checks require a valid scoring date.")
    modified = products.get("last_date_modified", pd.Series(pd.NaT, index=products.index))
    return blocked & pd.to_datetime(modified).lt(cutoff)


class RollingCandidatePool:
    """Merchandise with a paid sale before the query and no known block."""

    def __init__(self, catalogue: pd.DataFrame, purchases: pd.DataFrame):
        self.merchandise = select_sellable_products(catalogue)
        self.product_ids = set(self.merchandise["product_id"].astype(str))
        paid = purchases.loc[actual_purchase_mask(purchases), ["product_id", "purchase_date"]].copy()
        paid["product_id"] = paid["product_id"].astype(str).str.strip()
        paid["purchase_date"] = pd.to_datetime(paid["purchase_date"], format="mixed").dt.normalize()
        self.first_sales = paid.groupby("product_id")["purchase_date"].min()

    def blocked_on(self, date: str | pd.Timestamp) -> set[str]:
        return set(self.merchandise.loc[catalogue_blocked_mask(self.merchandise, date), "product_id"].astype(str))

    def eligible_on(self, date: str | pd.Timestamp) -> set[str]:
        cutoff = pd.Timestamp(date).normalize()
        prior = set(self.first_sales.index[self.first_sales.lt(cutoff)])
        return (prior & self.product_ids) - self.blocked_on(cutoff)


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
