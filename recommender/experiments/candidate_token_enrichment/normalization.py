"""Versioned [0, 1] scalar normalization, date-age correction, and padding sanitization.

All new enrichment MLP scalar channels MUST lie strictly in [0.0, 1.0]:
- Candidate pairwise/context features (41D):
  - 31 unsigned features already in [0, 1].
  - 10 signed [-1, 1] context features (`sku_growth_30d`, `month_sin`, `month_cos`,
    `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`,
    `line_last_lift`, `hierarchical_backoff`, `same_basket_lift`) deterministically
    mapped via `(x + 1.0) * 0.5` -> [0, 1].
  - Note: Unlike Stage 1 `uv_45` (which omits `month_sin`/`month_cos` for 39+6=45 scalars),
    Stage 2 retains all 41 pairwise/context columns, so all 10 signed columns are mapped.
- Profile lifetime features (8D, Token 0):
  - `LEGACY_PROFILE_FEATURES` (8 `Log1pScaler` [0, 1] features); respects
    `excluded_statistics` if any company statistic is excluded.
- History event features (9D, Tokens 1..32):
  - Recomputes `days_since_purchase_norm` (index 1) relative to `prediction_day` from
    `seq_purchase_dates` using the exact base statistics-bias formula:
    `age_days = clamp_min(prediction_day[:, None] - round(seq_purchase_dates * DAYS_PER_YEAR), 0)`
    `age = age_days / (age_days + 90.0)`.
  - Derives `real_event_indicator` (index 8) strictly from `attention_mask`.
  - Forces `is_repeat_purchase` (`history_repeat`, index 7) to `0.0` in ALL enriched paths
    when excluded (no resurrection).
  - Maps `purchase_month_sin` (3) and `purchase_month_cos` (4) from [-1, 1] via `(x + 1.0) * 0.5`.
  - Masks and sanitizes padded candidates and padded history positions BEFORE validation/MLP.
  - Preserves signed latent vectors (`u`/`v`), text embeddings, and the original raw scalar
    paths into `statistics_attention_weights` (13D/8D/9D) and the 425D scoring/attention head.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from recommender.artifacts import json_digest
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.sequences import DAYS_PER_YEAR, ENRICHED_FEATURE_NAMES, HISTORY_AGE_INDEX

ENRICHMENT_SCALAR_TRANSFORM_VERSION = "unit_interval_signed_shift_v1"

CANDIDATE_STAT_DIM = 41
PROFILE_STAT_DIM = 8
HISTORY_STAT_DIM = 9

SIGNED_CANDIDATE_FEATURE_NAMES: Tuple[str, ...] = (
    "sku_growth_30d",
    "month_sin",
    "month_cos",
    "sku_last_lift",
    "sku_recent3_lift",
    "category_last_lift",
    "family_last_lift",
    "line_last_lift",
    "hierarchical_backoff",
    "same_basket_lift",
)
SIGNED_CANDIDATE_FEATURE_INDICES: Tuple[int, ...] = tuple(
    DYNAMIC_FEATURE_NAMES.index(name) for name in SIGNED_CANDIDATE_FEATURE_NAMES
)
UNSIGNED_CANDIDATE_FEATURE_INDICES: Tuple[int, ...] = tuple(
    i for i in range(CANDIDATE_STAT_DIM) if i not in SIGNED_CANDIDATE_FEATURE_INDICES
)

SIGNED_HISTORY_FEATURE_NAMES: Tuple[str, ...] = (
    "purchase_month_sin",
    "purchase_month_cos",
)
SIGNED_HISTORY_FEATURE_INDICES: Tuple[int, ...] = tuple(
    ENRICHED_FEATURE_NAMES.index(name) for name in SIGNED_HISTORY_FEATURE_NAMES
)
UNSIGNED_HISTORY_FEATURE_INDICES: Tuple[int, ...] = tuple(
    i for i in range(HISTORY_STAT_DIM) if i not in SIGNED_HISTORY_FEATURE_INDICES
)

HISTORY_GAP_INDEX = ENRICHED_FEATURE_NAMES.index("inter_purchase_gap_norm")
HISTORY_REPEAT_INDEX = ENRICHED_FEATURE_NAMES.index("is_repeat_purchase")
HISTORY_REAL_INDICATOR_INDEX = ENRICHED_FEATURE_NAMES.index("real_event_indicator")

COMPANY_DISTINCT_ORDERS_INDEX = LEGACY_PROFILE_FEATURES.index("distinct_orders")
COMPANY_MEDIAN_GAP_INDEX = LEGACY_PROFILE_FEATURES.index("median_order_gap")

_RANGE_TOL = 1e-5


def _assert_unit_or_signed_range(
    values: torch.Tensor,
    *,
    low: float,
    high: float,
    name: str,
    tol: float = _RANGE_TOL,
) -> None:
    if values.numel() == 0:
        return
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains NaN or Inf on valid entries.")
    vmin = float(values.min().item())
    vmax = float(values.max().item())
    if vmin < low - tol or vmax > high + tol:
        raise ValueError(
            f"{name} out of expected range [{low}, {high}]: observed [{vmin:.6g}, {vmax:.6g}]."
        )


def sanitize_candidate_inputs(
    pairwise_features: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    candidate_vectors: Optional[torch.Tensor] = None,
    candidate_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Mask padded candidate positions to zero BEFORE validation/MLP and verify valid entries finite."""
    if pairwise_features.ndim != 3 or pairwise_features.shape[-1] != CANDIDATE_STAT_DIM:
        raise ValueError(
            f"Expected pairwise_features shape (B, C, {CANDIDATE_STAT_DIM}), "
            f"got {tuple(pairwise_features.shape)}."
        )
    b_size, num_c, _ = pairwise_features.shape
    if candidate_mask is not None:
        if candidate_mask.dtype != torch.bool or candidate_mask.shape != (b_size, num_c):
            raise ValueError(
                f"Expected boolean candidate_mask shape ({b_size}, {num_c}), "
                f"got {tuple(candidate_mask.shape)} ({candidate_mask.dtype})."
            )
        pf = torch.where(candidate_mask.unsqueeze(-1), pairwise_features, pairwise_features.new_zeros(()))
        if candidate_vectors is not None:
            candidate_vectors = torch.where(
                candidate_mask.unsqueeze(-1), candidate_vectors, candidate_vectors.new_zeros(())
            )
        if candidate_indices is not None:
            candidate_indices = torch.where(
                candidate_mask, candidate_indices, candidate_indices.new_zeros(())
            )
    else:
        pf = pairwise_features

    if not bool(torch.isfinite(pf).all()):
        raise ValueError("pairwise_features contains NaN or Inf on valid candidates.")
    if candidate_vectors is not None and not bool(torch.isfinite(candidate_vectors).all()):
        raise ValueError("candidate_vectors contains NaN or Inf on valid candidates.")
    return pf, candidate_vectors, candidate_indices


def normalize_candidate_stats(
    pairwise_features: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Validate and transform 41D candidate context features strictly into [0.0, 1.0].

    Masked padded positions (`~candidate_mask`) remain strictly 0.0.
    """
    pf, _, _ = sanitize_candidate_inputs(pairwise_features, candidate_mask=candidate_mask)
    if candidate_mask is not None:
        valid_pf = pf[candidate_mask]
    else:
        valid_pf = pf.reshape(-1, CANDIDATE_STAT_DIM)

    if valid_pf.numel() > 0:
        _assert_unit_or_signed_range(
            valid_pf[:, UNSIGNED_CANDIDATE_FEATURE_INDICES],
            low=0.0,
            high=1.0,
            name="candidate_unsigned_features_31",
        )
        _assert_unit_or_signed_range(
            valid_pf[:, SIGNED_CANDIDATE_FEATURE_INDICES],
            low=-1.0,
            high=1.0,
            name="candidate_signed_features_10",
        )

    out = pf.clone()
    out[..., SIGNED_CANDIDATE_FEATURE_INDICES] = (
        out[..., SIGNED_CANDIDATE_FEATURE_INDICES] + 1.0
    ) * 0.5
    out = out.clamp(0.0, 1.0)
    if candidate_mask is not None:
        out = torch.where(candidate_mask.unsqueeze(-1), out, out.new_zeros(()))
    return out


def normalize_profile_stats(
    tabular_features: torch.Tensor,
    *,
    excluded_statistics: Iterable[str] = ("history_repeat",),
) -> torch.Tensor:
    """Validate and prepare 8D customer profile statistics in [0.0, 1.0]."""
    if tabular_features.ndim != 2 or tabular_features.shape[-1] != PROFILE_STAT_DIM:
        raise ValueError(
            f"Expected tabular_features shape (B, {PROFILE_STAT_DIM}), "
            f"got {tuple(tabular_features.shape)}."
        )
    _assert_unit_or_signed_range(
        tabular_features,
        low=0.0,
        high=1.0,
        name="profile_tabular_features_8",
    )
    out = tabular_features.clamp(0.0, 1.0).clone()
    excl = set(excluded_statistics)
    if "company_distinct_orders" in excl:
        out[..., COMPANY_DISTINCT_ORDERS_INDEX] = 0.0
    if "company_median_gap" in excl:
        out[..., COMPANY_MEDIAN_GAP_INDEX] = 0.0
    return out


def sanitize_history_inputs(
    *,
    sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_purchase_dates: Optional[torch.Tensor] = None,
    prediction_day: Optional[torch.Tensor] = None,
    seq_product_ids: Optional[torch.Tensor] = None,
    seq_category_ids: Optional[torch.Tensor] = None,
    seq_subcategory_ids: Optional[torch.Tensor] = None,
    seq_business_line_ids: Optional[torch.Tensor] = None,
    seq_product_text: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Mask padded history positions BEFORE embeddings/MLP and validate valid entries."""
    if sequence.ndim != 3 or sequence.shape[-1] != HISTORY_STAT_DIM:
        raise ValueError(
            f"Expected sequence shape (B, L, {HISTORY_STAT_DIM}), got {tuple(sequence.shape)}."
        )
    b_size, seq_len, _ = sequence.shape
    if attention_mask.shape != (b_size, seq_len):
        raise ValueError(
            f"Expected attention_mask shape ({b_size}, {seq_len}), got {tuple(attention_mask.shape)}."
        )
    valid_hist = attention_mask != 0
    seq_clean = torch.where(valid_hist.unsqueeze(-1), sequence, sequence.new_zeros(()))
    if not bool(torch.isfinite(seq_clean).all()):
        raise ValueError("sequence contains NaN or Inf on valid history events.")

    dates_clean: Optional[torch.Tensor] = None
    if seq_purchase_dates is not None:
        if seq_purchase_dates.shape != (b_size, seq_len):
            raise ValueError(
                f"Expected seq_purchase_dates shape ({b_size}, {seq_len}), "
                f"got {tuple(seq_purchase_dates.shape)}."
            )
        dates_clean = torch.where(valid_hist, seq_purchase_dates, seq_purchase_dates.new_zeros(()))
        if not bool(torch.isfinite(dates_clean).all()):
            raise ValueError("seq_purchase_dates contains NaN or Inf on valid history events.")

    if prediction_day is not None:
        if prediction_day.shape != (b_size,):
            raise ValueError(
                f"Expected prediction_day shape ({b_size},), got {tuple(prediction_day.shape)}."
            )
        if not bool(torch.isfinite(prediction_day).all()):
            raise ValueError("prediction_day contains NaN or Inf.")

    def _mask_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return torch.where(valid_hist, t, t.new_zeros(()))

    prod_text_clean = (
        torch.where(valid_hist.unsqueeze(-1), seq_product_text, seq_product_text.new_zeros(()))
        if seq_product_text is not None
        else None
    )
    if prod_text_clean is not None and not bool(torch.isfinite(prod_text_clean).all()):
        raise ValueError("seq_product_text contains NaN or Inf on valid history events.")

    return {
        "valid_hist": valid_hist,
        "sequence": seq_clean,
        "seq_purchase_dates": dates_clean,
        "seq_product_ids": _mask_ids(seq_product_ids),
        "seq_category_ids": _mask_ids(seq_category_ids),
        "seq_subcategory_ids": _mask_ids(seq_subcategory_ids),
        "seq_business_line_ids": _mask_ids(seq_business_line_ids),
        "seq_product_text": prod_text_clean,
    }


def normalize_history_stats(
    sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_purchase_dates: torch.Tensor,
    prediction_day: torch.Tensor,
    *,
    excluded_statistics: Iterable[str] = ("history_repeat",),
) -> torch.Tensor:
    """Validate, date-correct, mask, and transform 9D history event features into [0.0, 1.0].

    - Recomputes `days_since_purchase_norm` (index 1) from `seq_purchase_dates` and
      `prediction_day` exactly as base `UPSCANRRanker` statistics bias does.
    - Derives `real_event_indicator` (index 8) strictly from `attention_mask`.
    - Enforces `history_repeat = 0.0` (index 7) in all enriched paths when excluded.
    - Maps signed `purchase_month_sin`/`purchase_month_cos` (indices 3, 4) via `(x + 1) / 2`.
    - Strictly masks padded positions (`attention_mask == 0`) to `0.0`.
    """
    if seq_purchase_dates is None or prediction_day is None:
        raise ValueError("Date-aware history enrichment requires prediction_day and seq_purchase_dates.")

    cleaned = sanitize_history_inputs(
        sequence=sequence,
        attention_mask=attention_mask,
        seq_purchase_dates=seq_purchase_dates,
        prediction_day=prediction_day,
    )
    valid_hist = cleaned["valid_hist"]
    seq_clean = cleaned["sequence"]
    dates_clean = cleaned["seq_purchase_dates"]

    valid_rows = seq_clean[valid_hist]
    if valid_rows.numel() > 0:
        _assert_unit_or_signed_range(
            valid_rows[:, UNSIGNED_HISTORY_FEATURE_INDICES],
            low=0.0,
            high=1.0,
            name="history_unsigned_features_7",
        )
        _assert_unit_or_signed_range(
            valid_rows[:, SIGNED_HISTORY_FEATURE_INDICES],
            low=-1.0,
            high=1.0,
            name="history_signed_month_features_2",
        )

    # Exact date-age formula matching recommender/ranker.py:encode_candidate_features.
    purchase_days = (dates_clean * DAYS_PER_YEAR).round()
    age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
    age = age_days / (age_days + 90.0)

    out = seq_clean.clone()
    out[..., HISTORY_AGE_INDEX] = age
    out[..., HISTORY_REAL_INDICATOR_INDEX] = valid_hist.to(dtype=out.dtype)

    excl = set(excluded_statistics)
    if "history_gap" in excl:
        out[..., HISTORY_GAP_INDEX] = 0.0
    if "history_month" in excl:
        out[..., SIGNED_HISTORY_FEATURE_INDICES] = 0.0
    else:
        out[..., SIGNED_HISTORY_FEATURE_INDICES] = (
            out[..., SIGNED_HISTORY_FEATURE_INDICES] + 1.0
        ) * 0.5
    if "history_repeat" in excl:
        out[..., HISTORY_REPEAT_INDEX] = 0.0
    if "history_real_indicator" in excl:
        out[..., HISTORY_REAL_INDICATOR_INDEX] = 0.0

    out = out.clamp(0.0, 1.0)
    out = torch.where(valid_hist.unsqueeze(-1), out, out.new_zeros(()))
    return out


def enrichment_normalization_spec(
    *,
    excluded_statistics: Iterable[str] = ("history_repeat",),
) -> Dict[str, Any]:
    """Deterministic contract metadata for [0, 1] token-enrichment scalar transforms."""
    body = {
        "transform_version": ENRICHMENT_SCALAR_TRANSFORM_VERSION,
        "candidate_stat_dim": CANDIDATE_STAT_DIM,
        "profile_stat_dim": PROFILE_STAT_DIM,
        "history_stat_dim": HISTORY_STAT_DIM,
        "total_enriched_scalar_channels": CANDIDATE_STAT_DIM + PROFILE_STAT_DIM + HISTORY_STAT_DIM,
        "candidate_feature_names": list(DYNAMIC_FEATURE_NAMES),
        "signed_candidate_features": list(SIGNED_CANDIDATE_FEATURE_NAMES),
        "signed_candidate_indices": list(SIGNED_CANDIDATE_FEATURE_INDICES),
        "unsigned_candidate_indices": list(UNSIGNED_CANDIDATE_FEATURE_INDICES),
        "profile_feature_names": list(LEGACY_PROFILE_FEATURES),
        "history_feature_names": list(ENRICHED_FEATURE_NAMES),
        "signed_history_features": list(SIGNED_HISTORY_FEATURE_NAMES),
        "signed_history_indices": list(SIGNED_HISTORY_FEATURE_INDICES),
        "unsigned_history_indices": list(UNSIGNED_HISTORY_FEATURE_INDICES),
        "signed_transform": "(x + 1.0) * 0.5 -> [0.0, 1.0]",
        "history_age_source": "recomputed from seq_purchase_dates and prediction_day: age/(age+90)",
        "history_validity_source": "derived from attention_mask",
        "excluded_statistics": list(excluded_statistics),
        "raw_scalar_bias_and_head_preserved": True,
        "latent_uv_signed_preserved": True,
        "stage2_vs_stage1_month_note": (
            "Stage 2 retains all 41 pairwise/context features (including month_sin and month_cos, "
            "so 10 candidate context columns are shifted from [-1,1] to [0,1]), whereas Stage 1 "
            "repeat-no-month-v1 uv_45 omits month_sin and month_cos."
        ),
        "history_text_behavior": (
            "Normal Stage 2 slates pass seq_product_text=None with catalog_text_embeddings=None, "
            "so product_text_proj receives zero vectors and contributes only its learned bias."
        ),
        "metadata_policy": "snapshot_unverified",
    }
    return {**body, "transform_digest": json_digest(body)}
