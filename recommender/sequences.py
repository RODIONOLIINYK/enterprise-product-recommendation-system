"""Customer interaction sequence generation for Transformer-based customer tower.

Constructs sequence tensors of exactly 32 interaction events per customer:
- Events strictly prior to as_of_date are selected
- Sorted chronologically ascending by purchase_date with deterministic tie-breaking on product_id
- Computes causal point-in-time habit statistics across complete customer history
- Truncates older events if history > 32 (preserving pre-computed habit statistics)
- Right-pads if history < 32 with attention_mask (1 for real events, 0 for padding)
- Supports both legacy 4-d continuous features and enriched 9-d feature vectors
- Extracts categorical token IDs (product, category, subcategory, business line)
- Gracefully handles empty history with zero attention mask and missing-history representation
"""

import hashlib
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_FEATURE_DIM = 4
ENRICHED_FEATURE_NAMES = (
    "quantity_norm", "days_since_purchase_norm", "inter_purchase_gap_norm",
    "purchase_month_sin", "purchase_month_cos", "prior_sku_purchase_count_norm",
    "cumulative_sku_quantity_share", "is_repeat_purchase", "real_event_indicator",
)
ENRICHED_FEATURE_DIM = len(ENRICHED_FEATURE_NAMES)
HISTORY_AGE_INDEX = 1
PURCHASE_DATE_ORIGIN = pd.Timestamp("2025-01-01")
DAYS_PER_YEAR = 365.25
PAD_TOKEN = "[PAD]"


def fit_quantity_reference(quantities: Union[List[float], np.ndarray, pd.Series]) -> float:
    """Return the median finite positive training quantity, or 1 for empty history."""
    values = np.asarray(pd.to_numeric(quantities, errors="coerce"), dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.quantile(positive, 0.5)) if positive.size else 1.0


def purchase_history_gaps(purchase_dates: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return each real purchase's calendar-day gap from the latest real purchase."""
    real = attention_mask.bool()
    latest = purchase_dates.masked_fill(~real, -torch.inf).amax(dim=-1, keepdim=True)
    # Dates are stored as calendar-day offsets in years. Round back to whole days.
    days = ((latest - purchase_dates) * DAYS_PER_YEAR).round().clamp_min(0)
    return days.masked_fill(~real, 0)


def fit_history_gap_reference(purchase_dates: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """Fit the median positive gap across training histories, with a one-day fallback."""
    gaps = purchase_history_gaps(purchase_dates, attention_mask)
    positive = gaps[gaps > 0]
    return float(positive.quantile(0.5).item()) if positive.numel() else 1.0


def normalize_purchase_history_gaps(
    purchase_dates: torch.Tensor, attention_mask: torch.Tensor, reference_days: float,
) -> torch.Tensor:
    """Compress the gap from the most recent purchase with a training-fitted scale."""
    log_gaps = torch.log1p(purchase_history_gaps(purchase_dates, attention_mask))
    return log_gaps / (log_gaps + np.log1p(reference_days))


def _map_token_id(
    val: Any,
    vocab: Optional[Dict[str, int]] = None,
) -> int:
    """Map a categorical token value to an integer ID using vocabulary or deterministic fallback.

    Padding / missing tokens map strictly to 0. Real tokens with no vocabulary map to
    a deterministic positive integer in [1, 10000].
    """
    if val is None or pd.isna(val) or val == "" or val == PAD_TOKEN:
        return 0
    val_str = str(val).strip()
    if not val_str or val_str == PAD_TOKEN or val_str.lower() in ("nan", "none", "null"):
        return 0

    if vocab is not None:
        if val_str in vocab:
            return int(vocab[val_str])
        if val in vocab:
            return int(vocab[val])
        return 0

    # Deterministic positive integer fallback in [1, 10000]
    return (int(hashlib.md5(val_str.encode("utf-8")).hexdigest()[:8], 16) % 10000) + 1


def _prepare_item_metadata_lookup(
    item_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]]]]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Convert item metadata DataFrame or dict into a fast dictionary lookup."""
    if item_metadata is None:
        return None
    if isinstance(item_metadata, dict):
        return item_metadata
    if isinstance(item_metadata, pd.DataFrame):
        if "product_id" not in item_metadata.columns:
            return None
        meta_cols = [
            c for c in [
                "product_category", "category",
                "product_subcategory", "subcategory",
                "business_line", "line",
            ] if c in item_metadata.columns
        ]
        if not meta_cols:
            return None
        indexed = item_metadata.assign(product_id=item_metadata["product_id"].astype(str))
        indexed = indexed.drop_duplicates(subset=["product_id"]).set_index("product_id")
        return indexed[meta_cols].to_dict(orient="index")
    return None


def _resolve_metadata(
    p_id: str,
    field_name: str,
    alt_field_names: List[str],
    history_row: pd.Series,
    meta_lookup: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Any]:
    """Resolve category/subcategory/business line from event row or fallback to item metadata."""
    for col in [field_name] + alt_field_names:
        if col in history_row.index and pd.notna(history_row[col]):
            val = str(history_row[col]).strip()
            if val and val.lower() not in ("nan", "none", "null"):
                return history_row[col]

    if meta_lookup and p_id in meta_lookup:
        p_dict = meta_lookup[p_id]
        for col in [field_name] + alt_field_names:
            if col in p_dict and pd.notna(p_dict[col]):
                val = str(p_dict[col]).strip()
                if val and val.lower() not in ("nan", "none", "null"):
                    return p_dict[col]

    return None


def build_customer_sequence(
    events_df: pd.DataFrame,
    customer_id: str,
    as_of_date: pd.Timestamp,
    max_len: int = DEFAULT_SEQUENCE_LENGTH,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    max_quantity: float = 100.0,
    product_vocab: Optional[Dict[str, int]] = None,
    category_vocab: Optional[Dict[str, int]] = None,
    subcategory_vocab: Optional[Dict[str, int]] = None,
    business_line_vocab: Optional[Dict[str, int]] = None,
    vocabularies: Optional[Dict[str, Dict[str, int]]] = None,
    item_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, str]]]] = None,
    enriched: Optional[bool] = None,
    *,
    quantity_reference: Optional[float] = None,
) -> Dict[str, Any]:
    """Construct a sequence tensor, attention mask, and token IDs for a single customer.

    Pass the fitted training quantity median as quantity_reference to map it to 0.5
    and larger finite quantities smoothly toward 1. If omitted, use the legacy
    max_quantity scaling for callers and checkpoints that predate the fitted reference.

    Returns:
        dict containing:
            - 'sequence': FloatTensor of shape (max_len, feature_dim)
            - 'attention_mask': LongTensor of shape (max_len,) where 1=real event, 0=pad
            - 'product_ids': List[str] of length max_len with product IDs or PAD_TOKEN
            - 'seq_product_ids': LongTensor of shape (max_len,) with categorical IDs
            - 'seq_purchase_dates': FloatTensor of shape (max_len,) in years from a fixed origin
            - 'seq_category_ids': LongTensor of shape (max_len,) with category IDs
            - 'seq_subcategory_ids': LongTensor of shape (max_len,) with subcategory IDs
            - 'seq_business_line_ids': LongTensor of shape (max_len,) with business line IDs
            - Aliases: 'product_indices', 'category_indices', 'subcategory_indices', 'business_line_indices'
    """
    if quantity_reference is not None and (not np.isfinite(quantity_reference) or quantity_reference <= 0):
        raise ValueError("quantity_reference must be finite and positive")

    if vocabularies is not None:
        def configured_vocab(*keys: str) -> Optional[Dict[str, int]]:
            # An explicitly empty fitted vocabulary maps every value to the unknown ID.
            return next((vocabularies[key] for key in keys if vocabularies.get(key) is not None), None)

        if product_vocab is None:
            product_vocab = configured_vocab("product_vocab", "product", "product_id", "products")
        if category_vocab is None:
            category_vocab = configured_vocab("category_vocab", "category", "product_category", "categories")
        if subcategory_vocab is None:
            subcategory_vocab = configured_vocab("subcategory_vocab", "subcategory", "product_subcategory", "subcategories")
        if business_line_vocab is None:
            business_line_vocab = configured_vocab("business_line_vocab", "business_line", "line", "business_lines")

    if enriched is not None:
        is_enriched = bool(enriched)
    else:
        is_enriched = (feature_dim == ENRICHED_FEATURE_DIM)

    if is_enriched and feature_dim == DEFAULT_FEATURE_DIM:
        effective_feature_dim = ENRICHED_FEATURE_DIM
    else:
        effective_feature_dim = feature_dim

    as_of_ts = pd.Timestamp(as_of_date)
    cid_str = str(customer_id)

    # Pre-allocate output containers
    sequence_tensor = torch.zeros((max_len, effective_feature_dim), dtype=torch.float32)
    attention_mask = torch.zeros(max_len, dtype=torch.long)
    product_ids: List[str] = [PAD_TOKEN] * max_len
    seq_product_ids = torch.zeros(max_len, dtype=torch.long)
    seq_purchase_dates = torch.zeros(max_len, dtype=torch.float32)
    seq_category_ids = torch.zeros(max_len, dtype=torch.long)
    seq_subcategory_ids = torch.zeros(max_len, dtype=torch.long)
    seq_business_line_ids = torch.zeros(max_len, dtype=torch.long)

    # Filter strictly prior to as_of_date
    if events_df.empty or "customer_id" not in events_df.columns or "purchase_date" not in events_df.columns:
        return {
            "sequence": sequence_tensor,
            "attention_mask": attention_mask,
            "product_ids": product_ids,
            "seq_product_ids": seq_product_ids,
            "seq_purchase_dates": seq_purchase_dates,
            "seq_category_ids": seq_category_ids,
            "seq_subcategory_ids": seq_subcategory_ids,
            "seq_business_line_ids": seq_business_line_ids,
            "product_indices": seq_product_ids,
            "category_indices": seq_category_ids,
            "subcategory_indices": seq_subcategory_ids,
            "business_line_indices": seq_business_line_ids,
        }

    c_mask = events_df["customer_id"].astype(str) == cid_str
    t_mask = pd.to_datetime(events_df["purchase_date"], format="mixed") < as_of_ts
    history = events_df.loc[c_mask & t_mask].copy()

    if history.empty:
        return {
            "sequence": sequence_tensor,
            "attention_mask": attention_mask,
            "product_ids": product_ids,
            "seq_product_ids": seq_product_ids,
            "seq_purchase_dates": seq_purchase_dates,
            "seq_category_ids": seq_category_ids,
            "seq_subcategory_ids": seq_subcategory_ids,
            "seq_business_line_ids": seq_business_line_ids,
            "product_indices": seq_product_ids,
            "category_indices": seq_category_ids,
            "subcategory_indices": seq_subcategory_ids,
            "business_line_indices": seq_business_line_ids,
        }

    # Normalize dates and sort chronologically ascending
    history["purchase_date"] = pd.to_datetime(history["purchase_date"], format="mixed")
    sort_cols = ["purchase_date"]
    if "product_id" in history.columns:
        history["product_id"] = history["product_id"].astype(str)
        sort_cols.append("product_id")

    history = history.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    num_events = len(history)

    # Point-in-time habit statistics computed across complete history BEFORE truncation
    if "quantity" in history.columns:
        raw_qty = history["quantity"].astype(float).values
    else:
        raw_qty = np.ones(num_events, dtype=np.float32)

    quantities = raw_qty.astype(np.float64 if quantity_reference is not None else np.float32)

    if quantity_reference is None:
        log_max_qty = np.log1p(max(max_quantity, 1.0))
        qty_norm = np.clip(np.log1p(np.maximum(quantities, 0.0)) / log_max_qty, 0.0, 1.0)
    else:
        quantities = np.where(np.isfinite(quantities), quantities, 0.0)
        log_qty = np.log1p(np.maximum(quantities, 0.0))
        qty_norm = log_qty / (log_qty + np.log1p(quantity_reference))

    # Days ago norm
    event_dates = history["purchase_date"]
    days_ago = np.maximum(0.0, (as_of_ts - event_dates).dt.days.values.astype(np.float32))
    days_ago_norm = np.clip(np.log1p(days_ago) / np.log1p(365.0), 0.0, 1.0)

    # Inter-purchase gap norm (0.0 for customer's first historical event)
    inter_purchase_gap_norm = np.zeros(num_events, dtype=np.float32)
    if num_events > 1:
        gap_days = (event_dates - event_dates.shift(1)).dt.days.fillna(0.0).clip(lower=0.0).values.astype(np.float32)
        inter_purchase_gap_norm[1:] = np.clip(np.log1p(gap_days[1:]) / np.log1p(365.0), 0.0, 1.0)

    # Cyclical month features
    months = event_dates.dt.month.values.astype(np.float32)
    sin_month = np.sin(2.0 * np.pi * months / 12.0).astype(np.float32)
    cos_month = np.cos(2.0 * np.pi * months / 12.0).astype(np.float32)

    # Prior purchase count and repeat indicator
    prior_counts = history.groupby("product_id").cumcount().values.astype(np.float32)
    user_product_prior_count_norm = np.clip(np.log1p(prior_counts) / np.log1p(50.0), 0.0, 1.0)
    is_repeat_purchase = (prior_counts >= 1.0).astype(np.float32)

    # Cumulative quantity share
    history["_calc_qty"] = quantities
    cum_sku_qty = history.groupby("product_id")["_calc_qty"].cumsum().to_numpy(dtype=quantities.dtype)
    total_cum_qty = history["_calc_qty"].cumsum().to_numpy(dtype=quantities.dtype)
    user_product_quantity_share = np.clip(cum_sku_qty / np.maximum(total_cum_qty, 1e-8), 0.0, 1.0)

    # Truncate to the most recent max_len events AFTER computing full history statistics
    if num_events > max_len:
        history = history.iloc[-max_len:].reset_index(drop=True)
        qty_norm = qty_norm[-max_len:]
        days_ago_norm = days_ago_norm[-max_len:]
        inter_purchase_gap_norm = inter_purchase_gap_norm[-max_len:]
        sin_month = sin_month[-max_len:]
        cos_month = cos_month[-max_len:]
        user_product_prior_count_norm = user_product_prior_count_norm[-max_len:]
        user_product_quantity_share = user_product_quantity_share[-max_len:]
        is_repeat_purchase = is_repeat_purchase[-max_len:]

    dates = history["purchase_date"].dt.tz_localize(None).dt.normalize()
    date_offsets = (dates - PURCHASE_DATE_ORIGIN).dt.days.to_numpy(dtype=np.float32) / DAYS_PER_YEAR
    seq_purchase_dates[:len(history)] = torch.from_numpy(date_offsets)

    meta_lookup = _prepare_item_metadata_lookup(item_metadata)

    for idx, (_, row) in enumerate(history.iterrows()):
        attention_mask[idx] = 1
        p_id = str(row.get("product_id", "UNKNOWN"))
        product_ids[idx] = p_id

        # Continuous feature vector
        if not is_enriched:
            feat_vector = torch.zeros(effective_feature_dim, dtype=torch.float32)
            feat_vector[0] = float(qty_norm[idx])
            feat_vector[HISTORY_AGE_INDEX] = float(days_ago_norm[idx])
            feat_vector[2] = 1.0  # Real event indicator
            # feat_vector[3] is unassigned (0.0)
            sequence_tensor[idx] = feat_vector
        else:
            feat_vector = torch.zeros(effective_feature_dim, dtype=torch.float32)
            feat_vector[0] = float(qty_norm[idx])
            feat_vector[HISTORY_AGE_INDEX] = float(days_ago_norm[idx])
            feat_vector[2] = float(inter_purchase_gap_norm[idx])
            feat_vector[3] = float(sin_month[idx])
            feat_vector[4] = float(cos_month[idx])
            feat_vector[5] = float(user_product_prior_count_norm[idx])
            feat_vector[6] = float(user_product_quantity_share[idx])
            feat_vector[7] = float(is_repeat_purchase[idx])
            feat_vector[8] = 1.0  # real_event_indicator
            sequence_tensor[idx] = feat_vector

        # Categorical IDs
        seq_product_ids[idx] = _map_token_id(p_id, product_vocab)

        cat_val = _resolve_metadata(p_id, "product_category", ["category"], row, meta_lookup)
        seq_category_ids[idx] = _map_token_id(cat_val, category_vocab)

        subcat_val = _resolve_metadata(p_id, "product_subcategory", ["subcategory"], row, meta_lookup)
        seq_subcategory_ids[idx] = _map_token_id(subcat_val, subcategory_vocab)

        bline_val = _resolve_metadata(p_id, "business_line", ["line"], row, meta_lookup)
        seq_business_line_ids[idx] = _map_token_id(bline_val, business_line_vocab)

    return {
        "sequence": sequence_tensor,
        "attention_mask": attention_mask,
        "product_ids": product_ids,
        "seq_product_ids": seq_product_ids,
        "seq_purchase_dates": seq_purchase_dates,
        "seq_category_ids": seq_category_ids,
        "seq_subcategory_ids": seq_subcategory_ids,
        "seq_business_line_ids": seq_business_line_ids,
        "product_indices": seq_product_ids,
        "category_indices": seq_category_ids,
        "subcategory_indices": seq_subcategory_ids,
        "business_line_indices": seq_business_line_ids,
    }


def build_batch_sequences(
    events_df: pd.DataFrame,
    customer_ids: List[str],
    as_of_dates: List[pd.Timestamp],
    max_len: int = DEFAULT_SEQUENCE_LENGTH,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    max_quantity: float = 100.0,
    product_vocab: Optional[Dict[str, int]] = None,
    category_vocab: Optional[Dict[str, int]] = None,
    subcategory_vocab: Optional[Dict[str, int]] = None,
    business_line_vocab: Optional[Dict[str, int]] = None,
    vocabularies: Optional[Dict[str, Dict[str, int]]] = None,
    item_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, str]]]] = None,
    enriched: Optional[bool] = None,
    *,
    quantity_reference: Optional[float] = None,
) -> Dict[str, Any]:
    """Construct batch tensors for multiple customers.

    Returns:
        dict containing:
            - 'sequence': FloatTensor of shape (batch_size, max_len, feature_dim)
            - 'attention_mask': LongTensor of shape (batch_size, max_len)
            - 'product_ids': List[List[str]] of length batch_size
            - 'seq_product_ids': LongTensor of shape (batch_size, max_len)
            - 'seq_category_ids': LongTensor of shape (batch_size, max_len)
            - 'seq_subcategory_ids': LongTensor of shape (batch_size, max_len)
            - 'seq_business_line_ids': LongTensor of shape (batch_size, max_len)
            - Aliases: 'product_indices', 'category_indices', 'subcategory_indices', 'business_line_indices'
    """
    if not customer_ids:
        if enriched is not None:
            is_enriched = bool(enriched)
        else:
            is_enriched = (feature_dim == ENRICHED_FEATURE_DIM)
        eff_dim = ENRICHED_FEATURE_DIM if (is_enriched and feature_dim == DEFAULT_FEATURE_DIM) else feature_dim
        empty_ids = torch.zeros((0, max_len), dtype=torch.long)
        return {
            "sequence": torch.zeros((0, max_len, eff_dim), dtype=torch.float32),
            "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
            "product_ids": [],
            "seq_product_ids": empty_ids,
            "seq_purchase_dates": torch.zeros((0, max_len), dtype=torch.float32),
            "seq_category_ids": empty_ids,
            "seq_subcategory_ids": empty_ids,
            "seq_business_line_ids": empty_ids,
            "product_indices": empty_ids,
            "category_indices": empty_ids,
            "subcategory_indices": empty_ids,
            "business_line_indices": empty_ids,
        }

    meta_lookup = _prepare_item_metadata_lookup(item_metadata)

    batch_seqs = []
    batch_masks = []
    batch_pids = []
    batch_prod_ids = []
    batch_purchase_dates = []
    batch_cat_ids = []
    batch_subcat_ids = []
    batch_bline_ids = []

    for cid, dt in zip(customer_ids, as_of_dates):
        res = build_customer_sequence(
            events_df=events_df,
            customer_id=cid,
            as_of_date=dt,
            max_len=max_len,
            feature_dim=feature_dim,
            max_quantity=max_quantity,
            product_vocab=product_vocab,
            category_vocab=category_vocab,
            subcategory_vocab=subcategory_vocab,
            business_line_vocab=business_line_vocab,
            vocabularies=vocabularies,
            item_metadata=meta_lookup,
            enriched=enriched,
            quantity_reference=quantity_reference,
        )
        batch_seqs.append(res["sequence"])
        batch_masks.append(res["attention_mask"])
        batch_pids.append(res["product_ids"])
        batch_prod_ids.append(res["seq_product_ids"])
        batch_purchase_dates.append(res["seq_purchase_dates"])
        batch_cat_ids.append(res["seq_category_ids"])
        batch_subcat_ids.append(res["seq_subcategory_ids"])
        batch_bline_ids.append(res["seq_business_line_ids"])

    seq_product_ids = torch.stack(batch_prod_ids, dim=0)
    seq_category_ids = torch.stack(batch_cat_ids, dim=0)
    seq_subcategory_ids = torch.stack(batch_subcat_ids, dim=0)
    seq_business_line_ids = torch.stack(batch_bline_ids, dim=0)

    return {
        "sequence": torch.stack(batch_seqs, dim=0),
        "attention_mask": torch.stack(batch_masks, dim=0),
        "product_ids": batch_pids,
        "seq_product_ids": seq_product_ids,
        "seq_purchase_dates": torch.stack(batch_purchase_dates, dim=0),
        "seq_category_ids": seq_category_ids,
        "seq_subcategory_ids": seq_subcategory_ids,
        "seq_business_line_ids": seq_business_line_ids,
        "product_indices": seq_product_ids,
        "category_indices": seq_category_ids,
        "subcategory_indices": seq_subcategory_ids,
        "business_line_indices": seq_business_line_ids,
    }
