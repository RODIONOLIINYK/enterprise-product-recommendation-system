from pathlib import Path
import re
import pandas as pd


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

VOLUME_SUFFIX_PATTERN = r",\s*\d+(?:[.,]\d+)?\s*(?:ml|мл|l|л)\s*$"

NULLABLE_CADENCE_FEATURES = {
    "average_days_between_customer_product_purchases",
    "std_days_between_customer_product_purchases",
    "expected_days_before_next_order",
}

REQUIRED_HISTORY_COLUMNS = {
    "customer_id",
    "product_id",
    "product_name",
    "product_category",
    "business_line",
    "purchase_date",
    "quantity",
}

REQUIRED_TEXT_COLUMNS = [
    "customer_id",
    "product_id",
    "product_name",
    "product_category",
    "business_line",
]


def complete_history_row_mask(purchases: pd.DataFrame) -> pd.Series:
    missing_columns = sorted(REQUIRED_HISTORY_COLUMNS - set(purchases.columns))
    if missing_columns:
        raise ValueError(f"Missing required history columns: {missing_columns}")

    required_text = purchases[REQUIRED_TEXT_COLUMNS].apply(
        lambda values: values.astype("string").str.strip()
    )
    return (
        required_text.notna().all(axis=1)
        & required_text.ne("").all(axis=1)
        & purchases["purchase_date"].notna()
        & purchases["quantity"].notna()
    )


def unify_volume_package_variants(purchases: pd.DataFrame) -> pd.DataFrame:
    purchases = purchases.copy()
    volume_variant_rows = purchases["product_name"].str.contains(
        VOLUME_SUFFIX_PATTERN,
        regex=True,
        flags=re.IGNORECASE,
    )
    if not volume_variant_rows.any():
        return purchases

    normalized_product_base = (
        purchases["product_name"]
        .str.replace(
            VOLUME_SUFFIX_PATTERN,
            "",
            regex=True,
            flags=re.IGNORECASE,
        )
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    product_purchasability = purchases["product_id"].value_counts()
    volume_catalogue = (
        purchases.loc[
            volume_variant_rows,
            [
                "product_id",
                "product_name",
            ],
        ]
        .assign(
            normalized_product_base=normalized_product_base.loc[
                volume_variant_rows
            ],
            purchasability=lambda df: df["product_id"]
            .map(product_purchasability)
            .fillna(0),
        )
        .drop_duplicates()
        .sort_values(
            [
                "normalized_product_base",
                "purchasability",
                "product_id",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
    )
    canonical_products = (
        volume_catalogue.drop_duplicates(
            "normalized_product_base",
            keep="first",
        ).set_index("normalized_product_base")
    )

    for column in [
        "product_id",
        "product_name",
    ]:
        purchases.loc[volume_variant_rows, column] = (
            normalized_product_base.loc[volume_variant_rows]
            .map(canonical_products[column])
            .to_numpy()
        )
    return purchases


def load_purchases(input_path: Path) -> pd.DataFrame:
    purchases = pd.read_excel(input_path)
    missing = sorted(set(COLUMN_MAPPING) - set(purchases.columns))
    if missing:
        raise ValueError(f"Missing expected source columns: {missing}")

    purchases = purchases.rename(columns=COLUMN_MAPPING)
    text_columns = [
        "customer_id",
        "product_id",
        "product_name",
        "product_category",
        "business_line",
        "transaction_type",
        "item_type",
    ]
    for column in text_columns:
        purchases[column] = purchases[column].astype("string").str.strip()

    purchases["purchase_date"] = pd.to_datetime(
        purchases["purchase_date"], errors="coerce"
    )
    purchases["quantity"] = pd.to_numeric(purchases["quantity"], errors="coerce")
    purchases = purchases.loc[
        complete_history_row_mask(purchases)
        & purchases["item_type"].eq("ТОВАР")
        & purchases["transaction_type"].eq("ПРОДАЖА")
        & purchases["quantity"].gt(0)
        & purchases["product_id"].str.startswith("ТОВ", na=False)
    ].copy()
    purchases = unify_volume_package_variants(purchases)

    return (
        purchases.groupby(
            ["customer_id", "purchase_date", "product_id"],
            sort=False,
            as_index=False,
        )
        .agg(
            quantity=("quantity", "sum"),
            product_name=("product_name", "first"),
            business_line=("business_line", "first"),
            product_category=("product_category", "first"),
        )
        .sort_values(["customer_id", "purchase_date", "product_id"])
        .reset_index(drop=True)
    )
