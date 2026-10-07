"""One business eligibility policy for training history and inference products."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


DEFAULT_SCOPE_PATH = Path(__file__).resolve().parents[1] / "configs" / "product_scope.toml"


@dataclass(frozen=True)
class ProductScope:
    version: int
    allowed_item_types: frozenset[str]
    minimum_prior_purchases: int

    def __post_init__(self) -> None:
        if type(self.minimum_prior_purchases) is not int or self.minimum_prior_purchases < 1:
            raise ValueError("Product scope requires minimum_prior_purchases >= 1.")
        if not self.allowed_item_types:
            raise ValueError("Product scope requires nonempty allowed_item_types.")


def load_product_scope(path: str | Path = DEFAULT_SCOPE_PATH) -> ProductScope:
    """Read the explicit allowlist; a missing/empty policy never admits all items."""
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    entries = config.get("allowed_item_types")
    if not isinstance(entries, list) or not entries or any(
        not isinstance(entry, str) or not entry.strip() for entry in entries
    ):
        raise ValueError("Product scope requires a nonempty allowed_item_types string list.")
    version = config.get("version")
    if type(version) is not int or version < 1:
        raise ValueError("Product scope requires a positive integer version.")
    minimum = config.get("minimum_prior_purchases")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("Product scope requires minimum_prior_purchases >= 1.")
    return ProductScope(
        version=version,
        allowed_item_types=frozenset(entry.strip().upper() for entry in entries),
        minimum_prior_purchases=minimum,
    )


def sellable_product_mask(
    records: pd.DataFrame, *, scope: ProductScope | None = None
) -> pd.Series:
    """Identify products for sale from source classification, not names or prefixes."""
    missing = sorted({"item_type"} - set(records.columns))
    if missing:
        raise ValueError(f"Missing product eligibility columns: {missing}")
    scope = scope if scope is not None else load_product_scope()
    item_type = records["item_type"].astype("string").str.strip().str.upper()
    return item_type.isin(scope.allowed_item_types)


def actual_purchase_mask(
    records: pd.DataFrame, *, scope: ProductScope | None = None
) -> pd.Series:
    """Positive saleable-product sales only; gifts never count as purchase evidence."""
    required = {
        "customer_id", "product_id", "item_type", "transaction_type", "purchase_date", "quantity",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Missing actual-purchase columns: {missing}")
    quantity = pd.to_numeric(records["quantity"], errors="coerce")
    identity_valid = pd.Series(True, index=records.index)
    for column in ["customer_id", "product_id"]:
        identity = records[column].astype("string").str.strip()
        identity_valid &= identity.notna() & identity.ne("")
    return (
        identity_valid
        & sellable_product_mask(records, scope=scope)
        & records["transaction_type"].astype("string").str.strip().str.upper().eq("ПРОДАЖА")
        & pd.to_datetime(records["purchase_date"], errors="coerce").notna()
        & quantity.gt(0)
        & np.isfinite(quantity).fillna(False)
    ).fillna(False)


def eligible_product_mask(
    products: pd.DataFrame, purchases: pd.DataFrame, scoring_time: str | pd.Timestamp,
    *, scope: ProductScope | None = None,
) -> pd.Series:
    """Require for-sale classification and a genuine prior sale at the same cutoff."""
    if "product_id" not in products:
        raise ValueError("Product catalogue requires product_id.")
    scope = scope if scope is not None else load_product_scope()
    cutoff = pd.Timestamp(scoring_time)
    if pd.isna(cutoff):
        raise ValueError("A valid scoring_time is required for historical product eligibility.")
    paid_mask = actual_purchase_mask(purchases, scope=scope)
    dates = pd.to_datetime(purchases["purchase_date"], errors="coerce")
    prior = purchases.loc[paid_mask & dates.lt(cutoff)]
    counts = prior["product_id"].astype("string").str.strip().value_counts()
    product_ids = products["product_id"].astype("string").str.strip()
    return sellable_product_mask(products, scope=scope) & product_ids.map(counts).fillna(0).ge(
        scope.minimum_prior_purchases
    )
