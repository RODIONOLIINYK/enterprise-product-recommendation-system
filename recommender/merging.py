"""Shared record merging that preserves quantities and optional text variants."""

from collections.abc import Sequence
import json

import pandas as pd


EVENT_KEY_COLUMNS = ["customer_id", "purchase_date", "product_id"]
PURCHASE_TEXT_COLUMNS = ["product_name", "product_category", "business_line"]


def _coalesce(values: pd.Series):
    """Fill missing metadata only when all supplied values agree."""
    present = values.dropna().unique()
    if len(present) > 1:
        raise ValueError(f"Conflicting metadata in duplicate records for column {values.name!r}.")
    return present[0] if len(present) else pd.NA


def _combine_text(values: pd.Series):
    """Retain different text values as a JSON array, including prior merges."""
    distinct = {}
    for value in values.dropna():
        if not isinstance(value, str):
            raise ValueError(f"Text merge column {values.name!r} must contain strings.")
        try:
            variants = json.loads(value)
        except (ValueError, TypeError):
            variants = None
        if not isinstance(variants, list) or not variants or not all(isinstance(v, str) for v in variants):
            variants = [value]
        for variant in variants:
            distinct.setdefault(variant, None)
    if not distinct:
        return pd.NA
    if len(distinct) == 1:
        return next(iter(distinct))
    return json.dumps(list(distinct), ensure_ascii=False)


def merge_records(
    records: pd.DataFrame, *, keys: Sequence[str],
    sum_columns: Sequence[str] = (), combine_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Unify matching keys without discarding non-key fields or numeric amounts.

    Sum only explicitly additive columns. Coalesce equal/complementary metadata;
    reject conflicting metadata unless its column opts into text-variant merging.
    Different text values in those columns are stored as a JSON-array string.
    Preserve column order, first-seen group order and caller input.
    """
    keys, sum_columns, combine_columns = list(keys), list(sum_columns), list(combine_columns)
    specified = keys + sum_columns + combine_columns
    if not keys or len(specified) != len(set(specified)):
        raise ValueError("Merge columns must be distinct, with at least one key.")
    if not records.columns.is_unique:
        raise ValueError("Record column names must be unique.")
    missing = sorted(set(specified) - set(records.columns))
    if missing:
        raise ValueError(f"Missing record merge columns: {missing}")
    for key in keys:
        if records[key].isna().any() or records[key].astype("string").str.strip().eq("").any():
            raise ValueError(f"Record merge requires nonmissing keys: {key!r}.")
    result = records.copy()
    for column in sum_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.empty:
        return result.reset_index(drop=True)
    repeated = result.duplicated(keys, keep=False)
    if not repeated.any():
        return result.reset_index(drop=True)
    # Merge only repeated keys. Carry a temporary position to restore source
    # order when combining those groups with records that were already unique.
    position = "_merge_position"
    while position in result.columns:
        position += "_"
    result[position] = range(len(result))
    aggregations = {
        column: (lambda values: values.sum(min_count=1)) if column in sum_columns
        else _combine_text if column in combine_columns else _coalesce
        for column in records.columns if column not in keys
    }
    aggregations[position] = "min"
    grouped = result.loc[repeated].groupby(
        keys, as_index=False, sort=False, observed=True, dropna=False,
    )
    merged = pd.concat([result.loc[~repeated], grouped.agg(aggregations)], ignore_index=True)
    return merged.sort_values(position)[list(records.columns)].reset_index(drop=True)


def merge_purchase_events(purchases: pd.DataFrame) -> pd.DataFrame:
    """Sum same-company, same-calendar-date, exact-product purchase quantities.

    Classification must agree. Retain all optional name/category/business-line
    variants. Other extra fields are preserved and must agree or fill missing
    values. Paid-sale filtering belongs to the calling loader.
    """
    missing = sorted(set([*EVENT_KEY_COLUMNS, "quantity"]) - set(purchases.columns))
    if missing:
        raise ValueError(f"Missing purchase merge columns: {missing}")
    normalized = purchases.copy()
    for column in ("customer_id", "product_id"):
        normalized[column] = normalized[column].astype("string").str.strip()
    normalized["purchase_date"] = pd.to_datetime(
        normalized["purchase_date"], errors="raise", format="mixed",
    ).dt.normalize()
    return merge_records(
        normalized, keys=EVENT_KEY_COLUMNS, sum_columns=["quantity"],
        combine_columns=[column for column in PURCHASE_TEXT_COLUMNS if column in normalized],
    ).sort_values(EVENT_KEY_COLUMNS).reset_index(drop=True)
