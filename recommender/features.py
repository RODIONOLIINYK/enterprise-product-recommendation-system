"""Point-in-time lifetime aggregate features, cadence urgency, and log1p normalization.

All features are strictly computed point-in-time at event timestamps:
no events from on or after scoring date are accessed.
All count-based and unbounded numeric features undergo log1p compression
followed by scaling and clipping strictly into [0.0, 1.0].
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


LEGACY_PROFILE_FEATURES = (
    "total_purchases",
    "total_quantity",
    "distinct_products",
    "distinct_orders",
    "distinct_categories",
    "days_since_first_order",
    "days_since_last_order",
    "median_order_gap",
)

RETRIEVER_PROFILE_FEATURES = (
    "distinct_orders",
    "total_quantity",
    "distinct_products",
    "distinct_categories",
    "median_order_gap",
)


def compute_urgency(days_since_last_order: float, median_order_gap: float) -> float:
    """Compute purchase cadence urgency feature: (days_since_last_order - median_order_gap).

    Positive values indicate an order is overdue relative to typical cadence.
    Negative values indicate an order is not yet due.
    """
    return float(days_since_last_order - median_order_gap)


def normalize_feature(
    x: Union[np.ndarray, pd.Series, List[float], Tuple[float, ...], float, int],
    max_val: float,
) -> Union[np.ndarray, float]:
    """Compress non-negative value with log1p and scale strictly into [0.0, 1.0].

    Formula: clip(log1p(max(x, 0)) / log1p(max_val), 0.0, 1.0).
    """
    if max_val <= 0.0:
        denom = 1.0
    else:
        denom = np.log1p(float(max_val))

    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(x, dtype=np.float64)
        non_neg = np.maximum(arr, 0.0)
        scaled = np.clip(np.log1p(non_neg) / denom, 0.0, 1.0)
        scaled[arr >= float(max_val)] = 1.0
        return scaled
    else:
        val = max(float(x), 0.0)
        if val >= float(max_val):
            return 1.0
        return float(np.clip(np.log1p(val) / denom, 0.0, 1.0))


class Log1pScaler:
    """Scaler that fits train bounds and transforms non-negative values into [0.0, 1.0]."""

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.max_val: float = 1.0
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._is_fitted = False

    def fit(self, values: Union[np.ndarray, pd.Series, List[float]]) -> "Log1pScaler":
        arr = np.asarray(values, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0 or np.max(valid) <= 0.0:
            self.max_val = 1.0
        else:
            self.max_val = float(np.max(valid))
        self._is_fitted = True
        return self

    def transform(
        self,
        values: Union[np.ndarray, pd.Series, List[float], float, int],
    ) -> Union[np.ndarray, float]:
        if not self._is_fitted:
            raise RuntimeError("Log1pScaler must be fitted before transforming.")
        return normalize_feature(values, self.max_val)

    def fit_transform(
        self,
        values: Union[np.ndarray, pd.Series, List[float]],
    ) -> np.ndarray:
        return self.fit(values).transform(values)


class PointInTimeNormalizer:
    """Manages individual Log1pScalers per feature, fitted strictly on train data."""

    NUMERIC_FEATURES = list(LEGACY_PROFILE_FEATURES)

    def __init__(self):
        self.scalers: Dict[str, Log1pScaler] = {
            feat: Log1pScaler() for feat in self.NUMERIC_FEATURES
        }
        self.urgency_min: float = -365.0
        self.urgency_max: float = 365.0
        self._is_fitted = False

    def fit(self, features_df: pd.DataFrame) -> "PointInTimeNormalizer":
        for feat in self.NUMERIC_FEATURES:
            if feat in features_df.columns:
                self.scalers[feat].fit(features_df[feat])

        if "urgency" in features_df.columns:
            u = features_df["urgency"].dropna()
            if len(u) > 0:
                self.urgency_min = float(u.min())
                self.urgency_max = float(u.max())
                if self.urgency_max == self.urgency_min:
                    self.urgency_max = self.urgency_min + 1.0

        self._is_fitted = True
        return self

    def to_state(self) -> Dict:
        """Serialize fitted scales as plain checkpoint values."""
        return {
            "maxima": {key: scaler.max_val for key, scaler in self.scalers.items() if scaler._is_fitted},
            "urgency_min": self.urgency_min,
            "urgency_max": self.urgency_max,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "PointInTimeNormalizer":
        normalizer = cls()
        for key, maximum in state["maxima"].items():
            normalizer.scalers[key].fit([maximum])
        normalizer.urgency_min = float(state["urgency_min"])
        normalizer.urgency_max = float(state["urgency_max"])
        normalizer._is_fitted = True
        return normalizer

    def transform_feature(self, feature_name: str, value: float) -> float:
        if feature_name in self.scalers:
            return float(self.scalers[feature_name].transform(value))
        elif feature_name == "urgency":
            denom = self.urgency_max - self.urgency_min
            if denom <= 0:
                denom = 1.0
            return float(np.clip((value - self.urgency_min) / denom, 0.0, 1.0))
        return float(np.clip(value, 0.0, 1.0))

    def transform_dict(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for k, v in raw_features.items():
            if k in self.scalers:
                normalized[k] = self.transform_feature(k, v)
            elif k == "urgency":
                normalized[k] = v  # Keep raw urgency
                normalized["urgency_norm"] = self.transform_feature("urgency", v)
            else:
                normalized[k] = v
        return normalized


def compute_point_in_time_features(
    purchases_df: pd.DataFrame,
    customer_id: str,
    as_of_date: pd.Timestamp,
    population_median_gap: float = 14.0,
    normalizer: Optional[PointInTimeNormalizer] = None,
) -> Dict[str, float]:
    """Compute lifetime aggregate features strictly prior to as_of_date."""
    # Strict point-in-time filter: only events strictly before as_of_date
    customer_mask = purchases_df["customer_id"].astype(str) == str(customer_id)
    time_mask = pd.to_datetime(purchases_df["purchase_date"]) < pd.Timestamp(as_of_date)
    history = purchases_df.loc[customer_mask & time_mask]

    if history.empty:
        raw = {
            "total_purchases": 0.0,
            "total_quantity": 0.0,
            "distinct_products": 0.0,
            "distinct_orders": 0.0,
            "distinct_categories": 0.0,
            "days_since_first_order": 0.0,
            "days_since_last_order": 0.0,
            "median_order_gap": float(population_median_gap),
            "urgency": compute_urgency(0.0, population_median_gap),
            "has_cadence_history": 0.0,
            "has_history": 0.0,
        }
    else:
        dates = pd.to_datetime(history["purchase_date"]).sort_values()
        distinct_dates = dates.drop_duplicates()
        as_of_ts = pd.Timestamp(as_of_date)

        first_order = distinct_dates.iloc[0]
        last_order = distinct_dates.iloc[-1]
        days_since_first = max(0.0, float((as_of_ts - first_order).days))
        days_since_last = max(0.0, float((as_of_ts - last_order).days))

        if len(distinct_dates) >= 2:
            gaps = (distinct_dates.diff().iloc[1:]).dt.days.values
            median_gap = float(np.median(gaps))
            has_cadence = 1.0
        else:
            median_gap = float(population_median_gap)
            has_cadence = 0.0

        urgency = compute_urgency(days_since_last, median_gap)

        total_qty = (
            float(history["quantity"].sum())
            if "quantity" in history.columns
            else float(len(history))
        )
        distinct_prods = (
            float(history["product_id"].nunique())
            if "product_id" in history.columns
            else 0.0
        )
        distinct_cats = (
            float(history["product_category"].nunique())
            if "product_category" in history.columns
            else 0.0
        )

        raw = {
            "total_purchases": float(len(history)),
            "total_quantity": max(0.0, total_qty),
            "distinct_products": distinct_prods,
            "distinct_orders": float(len(distinct_dates)),
            "distinct_categories": distinct_cats,
            "days_since_first_order": days_since_first,
            "days_since_last_order": days_since_last,
            "median_order_gap": median_gap,
            "urgency": urgency,
            "has_cadence_history": has_cadence,
            "has_history": 1.0,
        }

    if normalizer is not None:
        return normalizer.transform_dict(raw)
    return raw
