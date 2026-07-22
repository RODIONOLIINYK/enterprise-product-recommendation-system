import argparse
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "deployment_pipeline" / "data" / "products.csv"
RECENT_WINDOW_DAYS = 30
TOP_RECOMMENDATIONS = 20

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

MODEL_COLUMNS = [
    "product_id",
    "product_category",
    "business_line",
    "previous_paid_purchase_count",
    "previous_paid_quantity",
    "last_paid_quantity",
    "days_since_last_paid_purchase",
    "average_days_between_customer_product_purchases",
    "std_days_between_customer_product_purchases",
    "expected_days_before_next_order",
    "previous_category_purchase_count",
    "previous_category_purchase_share",
    "previous_business_line_purchase_count",
    "previous_business_line_purchase_share",
    "historical_product_purchase_count",
    "historical_product_unique_customer_count",
    "product_purchase_count_last_30_days",
]
CANDIDATE_OUTPUT_COLUMNS = [
    "product_id",
    "product_name",
    *MODEL_COLUMNS[1:],
]
EXPECTED_DAYS_COLUMN = "expected_days_before_next_order"
PRODUCT_OUTPUT_COLUMNS = [
    column
    for column in CANDIDATE_OUTPUT_COLUMNS
    if column != EXPECTED_DAYS_COLUMN
] + [EXPECTED_DAYS_COLUMN]
MODEL_OUTPUT_COLUMNS = [
    "historical_score",
    "classifier_score",
    "classifier_rank",
]
VOLUME_SUFFIX_PATTERN = (
    r",\s*(?P<package_amount>\d+(?:[.,]\d+)?)\s*"
    r"(?P<package_unit>ml|мл|l|л)\s*$"
)
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
REQUIRED_CATEGORICAL_COLUMNS = [
    "product_id",
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
    volume_parts = purchases["product_name"].str.extract(
        VOLUME_SUFFIX_PATTERN,
        flags=re.IGNORECASE,
    )
    package_volume_litres = pd.to_numeric(
        volume_parts["package_amount"].str.replace(",", ".", regex=False),
        errors="coerce",
    )
    millilitre_rows = volume_parts["package_unit"].str.casefold().isin(
        ["ml", "мл"]
    )
    package_volume_litres = package_volume_litres.mask(
        millilitre_rows,
        package_volume_litres / 1_000,
    )
    volume_variant_rows = package_volume_litres.gt(0)
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
            package_volume_litres=package_volume_litres.loc[
                volume_variant_rows
            ],
        )
        .drop_duplicates()
        .sort_values(
            [
                "normalized_product_base",
                "package_volume_litres",
                "product_id",
            ],
            kind="stable",
        )
    )
    canonical_products = (
        volume_catalogue.drop_duplicates(
            "normalized_product_base",
            keep="first",
        ).set_index("normalized_product_base")
    )

    canonical_package_volume_litres = normalized_product_base.map(
        canonical_products["package_volume_litres"]
    )
    package_conversion_factors = (
        package_volume_litres / canonical_package_volume_litres
    )
    expected_quantity = purchases["quantity"].copy()
    expected_quantity.loc[volume_variant_rows] = (
        expected_quantity.loc[volume_variant_rows]
        * package_conversion_factors.loc[volume_variant_rows]
    )
    purchases.loc[volume_variant_rows, "quantity"] = (
        expected_quantity.loc[volume_variant_rows].to_numpy()
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
    pd.testing.assert_series_equal(
        purchases["quantity"],
        expected_quantity,
        check_names=False,
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


def build_features(
    purchases: pd.DataFrame, customer_id: str, scoring_date: pd.Timestamp
) -> pd.DataFrame:
    purchases = purchases.copy()
    purchases[REQUIRED_TEXT_COLUMNS] = purchases[REQUIRED_TEXT_COLUMNS].apply(
        lambda values: values.astype("string").str.strip()
    )
    purchases = purchases.loc[complete_history_row_mask(purchases)].copy()
    prior_purchases = purchases.loc[purchases["purchase_date"].le(scoring_date)]
    catalogue = (
        prior_purchases.sort_values(["purchase_date", "product_id"])
        .groupby("product_id", sort=False, as_index=False)
        .agg(
            product_name=("product_name", "first"),
            product_category=("product_category", "first"),
            business_line=("business_line", "first"),
        )
    )
    if catalogue.empty:
        raise ValueError(f"No products existed before {scoring_date.date()}.")

    candidates = catalogue.copy()
    customer_history = (
        prior_purchases.loc[prior_purchases["customer_id"].eq(customer_id)]
        .sort_values(["product_id", "purchase_date"])
        .copy()
    )
    customer_history["interval_days"] = (
        customer_history.groupby("product_id")["purchase_date"]
        .diff()
        .dt.days
    )
    product_history = customer_history.groupby("product_id").agg(
        previous_paid_purchase_count=("quantity", "size"),
        previous_paid_quantity=("quantity", "sum"),
        last_paid_quantity=("quantity", "last"),
        last_purchase_date=("purchase_date", "last"),
        average_days_between_customer_product_purchases=("interval_days", "mean"),
        std_days_between_customer_product_purchases=(
            "interval_days",
            lambda values: values.std(ddof=0),
        ),
    )
    candidates = candidates.merge(
        product_history, on="product_id", how="left", validate="one_to_one"
    )
    candidates["days_since_last_paid_purchase"] = (
        scoring_date - candidates["last_purchase_date"]
    ).dt.days.fillna(0)
    average_quantity = (
        candidates["previous_paid_quantity"]
        / candidates["previous_paid_purchase_count"]
    )
    has_cycle = (
        candidates["average_days_between_customer_product_purchases"].gt(0)
        & candidates["last_paid_quantity"].gt(0)
        & average_quantity.gt(0)
    )
    candidates["expected_days_before_next_order"] = np.nan
    candidates.loc[has_cycle, "expected_days_before_next_order"] = (
        candidates.loc[
            has_cycle, "average_days_between_customer_product_purchases"
        ]
        * candidates.loc[has_cycle, "last_paid_quantity"]
        / average_quantity.loc[has_cycle]
        - candidates.loc[has_cycle, "days_since_last_paid_purchase"]
    )

    total_customer_purchases = len(customer_history)
    for source_column, prefix in [
        ("product_category", "category"),
        ("business_line", "business_line"),
    ]:
        counts = customer_history.groupby(source_column).size()
        count_column = f"previous_{prefix}_purchase_count"
        share_column = f"previous_{prefix}_purchase_share"
        candidates[count_column] = candidates[source_column].map(counts).fillna(0)
        candidates[share_column] = (
            candidates[count_column] / total_customer_purchases
            if total_customer_purchases
            else 0.0
        )

    global_history = prior_purchases.groupby("product_id").agg(
        historical_product_purchase_count=("customer_id", "size"),
        historical_product_unique_customer_count=("customer_id", "nunique"),
    )
    recent_start = scoring_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    recent_90days_start = scoring_date - pd.Timedelta(days=90)
    recent_counts = (
        prior_purchases.loc[prior_purchases["purchase_date"].ge(recent_start)]
        .groupby("product_id")
        .size()
        .rename("product_purchase_count_last_30_days")
    )
    recent_90days_counts = (
        prior_purchases.loc[prior_purchases["purchase_date"].ge(recent_90days_start)]
        .groupby("product_id")
        .size()
        .rename("product_purchase_count_last_90_days")
    )

    result = (
        candidates.merge(
            global_history, on="product_id", how="left", validate="one_to_one"
        )
        .merge(recent_counts, on="product_id", how="left", validate="one_to_one")
        .merge(recent_90days_counts, on="product_id", how="left", validate="one_to_one")
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    non_nullable_numeric_columns = [
        column
        for column in MODEL_COLUMNS
        if column not in {"product_id", "product_category", "business_line"}
        and column not in NULLABLE_CADENCE_FEATURES
    ]
    result[non_nullable_numeric_columns] = result[
        non_nullable_numeric_columns
    ].fillna(0)
    required_categorical = result[REQUIRED_CATEGORICAL_COLUMNS]
    complete_candidate_mask = (
        required_categorical.notna().all(axis=1)
        & required_categorical.ne("").all(axis=1)
    )
    result = result.loc[complete_candidate_mask].reset_index(drop=True)
    return result[CANDIDATE_OUTPUT_COLUMNS]

def evaluate(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    count = features["previous_paid_purchase_count"].fillna(0)
    products_purchased = features.loc[
        count.gt(0),
        "product_id",
    ].unique()
    businesslines_purchased = features[features['previous_business_line_purchase_count'] > 0]['business_line'].unique()
    categories_purchased = features[features['previous_category_purchase_count'] > 0]['product_category'].unique()

    features = features[features['business_line'].isin(businesslines_purchased) | features['product_category'].isin(categories_purchased)]

    if len(features) > 100:
        print('only user purchased products as candidates')
        features = features[features['product_id'].isin(products_purchased)]

    count = features["previous_paid_purchase_count"].fillna(0)
    is_repeat = count.gt(0)
    repeat_strength = np.log1p(count)
    maximum_repeat_strength = repeat_strength.max()
    if maximum_repeat_strength > 0:
        repeat_strength = repeat_strength / maximum_repeat_strength

    cadence = (
        features["average_days_between_customer_product_purchases"]
        .fillna(0)
        .where(lambda values: values.gt(0), 30)
        .clip(lower=7)
    )
    recency = (
        np.exp(-features["days_since_last_paid_purchase"].fillna(0) / cadence)
    )
    due_scale = (
        features["std_days_between_customer_product_purchases"]
        .fillna(0)
        .where(lambda values: values.gt(0), 7)
        .clip(lower=7)
    )
    due = 1.0 / (
        1.0
        + np.exp(
            (
                features["expected_days_before_next_order"].fillna(0)
                / due_scale
            ).clip(-50, 50)
        )
    )
    has_cycle = features[
        "average_days_between_customer_product_purchases"
    ].fillna(0).gt(0)
    timing = np.where(has_cycle, due, recency)

    category_affinity = features[
        "previous_category_purchase_share"
    ].fillna(0)
    business_line_affinity = features[
        "previous_business_line_purchase_share"
    ].fillna(0)
    popularity = np.log1p(
        features["product_purchase_count_last_30_days"].fillna(0)
    ).rank(pct=True)

    repeat_score = (
        0.60 * repeat_strength
        + 0.30 * timing
        + 0.07 * category_affinity
        + 0.03 * business_line_affinity
    ).clip(0, 1)
    discovery_score = (
        0.55 * category_affinity
        + 0.25 * business_line_affinity
        + 0.20 * popularity
    ).clip(0, 1)
    features["historical_score"] = np.where(
        is_repeat,
        1.0 + repeat_score,
        0.999999 * discovery_score,
    )

    features = features.sort_values('historical_score').head(min(30, len(features)))

    # here should be sequence model results

    return rank_with_classifier(features)


def prepare_model_input(
    candidates: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    missing_columns = sorted(set(feature_columns) - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Model candidates are missing features: {missing_columns}")

    model_input = candidates[feature_columns].copy()
    categorical_columns = set(REQUIRED_CATEGORICAL_COLUMNS)
    model_input[list(categorical_columns)] = model_input[
        list(categorical_columns)
    ].apply(lambda values: values.astype("string").str.strip())
    invalid_categorical = (
        model_input[list(categorical_columns)].isna()
        | model_input[list(categorical_columns)].eq("")
    )
    if invalid_categorical.any().any():
        raise ValueError(
            "Model candidates contain missing required categorical values."
        )

    for column in feature_columns:
        if column in categorical_columns:
            model_input[column] = model_input[column].astype(object)
        else:
            model_input[column] = pd.to_numeric(
                model_input[column],
                errors="raise",
            )
            if column not in NULLABLE_CADENCE_FEATURES:
                model_input[column] = model_input[column].fillna(0)
    return model_input


def rank_with_classifier(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    if candidates["product_id"].duplicated().any():
        raise ValueError("Classifier candidates must contain one row per product.")

    classifier = CatBoostClassifier()
    classifier.load_model(
        PROJECT_ROOT / "models" / "catboost_classifier.cbm"
    )
    classifier_input = prepare_model_input(
        candidates,
        classifier.feature_names_,
    )
    candidates["classifier_score"] = classifier.predict_proba(
        classifier_input
    )[:, 1]

    ranked_products = candidates.sort_values(
        ["classifier_score", "product_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranked_products["classifier_rank"] = ranked_products.index + 1
    ranked_products = ranked_products.head(
        min(TOP_RECOMMENDATIONS, len(ranked_products))
    )
    return ranked_products[
        PRODUCT_OUTPUT_COLUMNS + MODEL_OUTPUT_COLUMNS
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "customer_id",
        nargs="?",
        help="Customer to score; a random known customer is used when omitted.",
    )
    args = parser.parse_args()

    input_path = next(RAW_DATA_DIR.glob("*.xlsx"), None)
    if input_path is None:
        raise FileNotFoundError(f"No .xlsx source file found in {RAW_DATA_DIR}")

    purchases = load_purchases(input_path)
    known_customers = purchases["customer_id"].dropna().unique().tolist()
    customer_id = args.customer_id or random.choice(known_customers)
    if customer_id not in known_customers:
        raise ValueError(f"Unknown customer_id: {customer_id}")

    scoring_date = pd.Timestamp.today().normalize()
    features = build_features(purchases, customer_id, scoring_date)
    ranked_products = evaluate(features)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked_products.to_csv(OUTPUT_PATH, index=False)
    display_columns = [
        "classifier_rank",
        "product_id",
        "product_name",
        "classifier_score",
    ]
    print(ranked_products[display_columns].to_string(index=False))
    print(
        f"\nSaved {len(ranked_products):,} product rows for customer "
        f"{customer_id} at {scoring_date.date()} to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
