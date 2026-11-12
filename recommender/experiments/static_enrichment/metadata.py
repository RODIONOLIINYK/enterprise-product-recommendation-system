"""Load static product/customer metadata, clean HTML descriptions, and fit training-only vocabs."""

from collections import Counter
import html
from pathlib import Path
import re
from typing import Dict, Optional, Sequence

import pandas as pd

from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.merging import merge_records
from recommender.scope import DEFAULT_DATASETS_PATH, get_datasets_locs

from .config import DEFAULT_DESCRIPTION_PATH, MIN_TRAIN_CATEGORY_COUNT

NONE_TOKEN = "<NONE>"
OOV_TOKEN = "<OOV>"
NONE_INDEX = 0
OOV_INDEX = 1

_BLOCK_TAG_RE = re.compile(r"</?(?:br|p|div|li|ol|ul|tr|td|th|table|tbody|h[1-6])\b[^>]*>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Extra static columns explicitly authorized from items.csv:
# - col 52: country_of_origin
# - col 127: tier (chosen over col 124, which duplicates brand/product_family labels)
# Explicitly excluded: col 98 (customs code) and all price columns.
EXTRA_ITEM_COLUMNS = {1: "product_id", 52: "country_of_origin", 127: "tier"}

# Extra static columns explicitly authorized from customers.csv (kept in EXCLUDED_COMPANY_COLUMNS in core):
EXTRA_COMPANY_COLUMNS = {1: "customer_id", 44: "sales_branch", 59: "vat_status"}


def clean_html_description(raw: object) -> str:
    """Strip HTML tags/entities and normalize whitespace safely around Unicode/emojis."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw).strip()
    if not text or text.upper() == "NULL":
        return ""
    text = _BLOCK_TAG_RE.sub(" ", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_cleaned_descriptions(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load and clean snapshot_unverified product descriptions (ignoring time-dependent price col 3)."""
    source = Path(path) if path is not None else DEFAULT_DESCRIPTION_PATH
    raw = pd.read_csv(
        source,
        sep=";",
        header=None,
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    if not {0, 1, 2}.issubset(raw.columns):
        raise ValueError("Description export must have at least 3 columns (name, product_id, html_description).")
    out = pd.DataFrame({
        "product_id": raw[1].astype(str).str.strip(),
        "raw_description": raw[2].astype(str),
    })
    out = out.loc[~out["product_id"].isin(["", "NULL"])].copy()
    out["cleaned_description"] = out["raw_description"].map(clean_html_description)
    out = merge_records(out[["product_id", "cleaned_description"]], keys=["product_id"])
    return out


def load_enriched_catalogue(
    items_path: Optional[str | Path] = None,
    descriptions_path: Optional[str | Path] = None,
    *,
    config_path: str | Path = DEFAULT_DATASETS_PATH,
) -> pd.DataFrame:
    """Load standard product catalogue enriched with static country_of_origin, tier, and cleaned descriptions."""
    resolved_items = Path(items_path) if items_path is not None else get_datasets_locs("items", config_path)
    catalogue = load_product_catalogue(resolved_items)

    raw = pd.read_csv(
        resolved_items,
        sep=";",
        header=None,
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    available = [c for c in EXTRA_ITEM_COLUMNS if c in raw.columns]
    extra = raw[available].rename(columns=EXTRA_ITEM_COLUMNS).copy()
    for col in extra.columns:
        extra[col] = extra[col].str.strip()
        extra[col] = extra[col].mask(extra[col].isin(["", "NULL"]))
        if col != "product_id":
            extra[col] = extra[col].str.upper()
    extra = merge_records(extra, keys=["product_id"])
    enriched = catalogue.merge(extra, on="product_id", how="left", validate="one_to_one")
    for col in ("tier", "country_of_origin"):
        if col not in enriched.columns:
            enriched[col] = pd.Series(pd.NA, index=enriched.index, dtype="string")
    if "product_family" in enriched.columns:
        enriched["product_family"] = enriched["product_family"].astype("string").str.strip().str.upper()
    if "unit" in enriched.columns:
        enriched["unit"] = enriched["unit"].astype("string").str.strip().str.upper()

    if descriptions_path is not None or DEFAULT_DESCRIPTION_PATH.is_file():
        desc_df = load_cleaned_descriptions(descriptions_path)
        enriched = enriched.merge(desc_df, on="product_id", how="left", validate="one_to_one")
        enriched["cleaned_description"] = enriched["cleaned_description"].fillna("")
    else:
        enriched["cleaned_description"] = ""
    return enriched


def load_enriched_companies(
    customers_path: Optional[str | Path] = None,
    *,
    config_path: str | Path = DEFAULT_DATASETS_PATH,
    purchases: Optional[pd.DataFrame] = None,
    minimum_purchases: int = 1,
) -> pd.DataFrame:
    """Load standard companies enriched with static sales_branch and vat_status."""
    resolved_customers = (
        Path(customers_path) if customers_path is not None else get_datasets_locs("companies", config_path)
    )
    companies = load_companies(
        resolved_customers,
        config_path=config_path,
        purchases=purchases,
        minimum_purchases=minimum_purchases,
    )
    raw = pd.read_csv(
        resolved_customers,
        sep=";",
        header=None,
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    available = [c for c in EXTRA_COMPANY_COLUMNS if c in raw.columns]
    extra = raw[available].rename(columns=EXTRA_COMPANY_COLUMNS).copy()
    for col in extra.columns:
        extra[col] = extra[col].str.strip()
        extra[col] = extra[col].mask(extra[col].isin(["", "NULL"]))
        if col != "customer_id":
            extra[col] = extra[col].str.upper()
    extra = merge_records(extra, keys=["customer_id"])
    enriched = companies.merge(extra, on="customer_id", how="left", validate="one_to_one")
    for col in ("sales_branch", "vat_status"):
        if col not in enriched.columns:
            enriched[col] = pd.Series(pd.NA, index=enriched.index, dtype="string")
    return enriched


def normalize_categorical_value(value: object) -> Optional[str]:
    """Normalize a categorical string; return None for missing/NULL."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return None
    return text.upper()


def fit_categorical_vocab(
    values: Sequence[object],
    *,
    min_count: int = MIN_TRAIN_CATEGORY_COUNT,
) -> Dict[str, int]:
    """Fit categorical vocabulary on training items/customers with <NONE>=0 and <OOV>=1."""
    if min_count < 1:
        raise ValueError("min_count must be positive.")
    counts = Counter()
    for raw in values:
        norm = normalize_categorical_value(raw)
        if norm is not None:
            counts[norm] += 1
    vocab: Dict[str, int] = {NONE_TOKEN: NONE_INDEX, OOV_TOKEN: OOV_INDEX}
    frequent = sorted(key for key, cnt in counts.items() if cnt >= min_count)
    for key in frequent:
        vocab[key] = len(vocab)
    return vocab


def encode_categorical_values(
    values: Sequence[object],
    vocab: Dict[str, int],
) -> list[int]:
    """Map values to integer IDs: missing -> 0 (<NONE>), frequent -> vocab[val], rare/unseen -> 1 (<OOV>)."""
    encoded = []
    for raw in values:
        norm = normalize_categorical_value(raw)
        if norm is None:
            encoded.append(vocab.get(NONE_TOKEN, NONE_INDEX))
        else:
            encoded.append(vocab.get(norm, vocab.get(OOV_TOKEN, OOV_INDEX)))
    return encoded
