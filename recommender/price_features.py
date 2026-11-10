"""Point-in-time price preference features for Stage 1 retrieval."""

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd
import torch

from recommender.data import load_source_purchases, priced_purchase_events, select_paid_purchases
from recommender.merging import EVENT_KEY_COLUMNS
from recommender.scope import get_datasets_locs


def _price_timelines(events: pd.DataFrame, keys: str | list[str]) -> dict:
    return {
        key: (group["purchase_date"].to_numpy(dtype="datetime64[D]"),
              group["unit_price_uah"].expanding().median().to_numpy(dtype=np.float32))
        for key, group in events.groupby(keys, sort=False, observed=True)
    }


def _prior_median(timeline: tuple[np.ndarray, np.ndarray], dates: np.ndarray, minimum: int) -> np.ndarray:
    recorded_dates, medians = timeline
    counts = np.searchsorted(recorded_dates, dates, side="left")
    return np.where(counts >= minimum, medians[np.maximum(counts - 1, 0)], np.nan)


class PricePreferenceStore:
    """Compare SKU prices with what the customer paid for comparable products."""

    def __init__(self, priced_events: pd.DataFrame, catalogue: pd.DataFrame, product_ids: Sequence[str]):
        attributes = catalogue.set_index("product_id")[["unit", "product_category"]].copy()
        attributes.index = attributes.index.astype(str)
        attributes = attributes.reindex(product_ids)
        events = priced_events.merge(attributes, on="product_id", how="inner", validate="many_to_one")
        events = events.loc[np.isfinite(events["unit_price_uah"]) & events["unit_price_uah"].gt(0)]
        events = events.dropna(subset=["unit"]).sort_values("purchase_date", kind="stable")
        self.product_ids = tuple(product_ids)
        self.sku_prices = _price_timelines(events, "product_id")
        self.customer_unit_prices = _price_timelines(events, ["customer_id", "unit"])
        self.customer_category_prices = _price_timelines(events, ["customer_id", "unit", "product_category"])
        self.product_groups = {
            key: group["candidate_index"].to_numpy()
            for key, group in attributes.assign(candidate_index=np.arange(len(product_ids)))
            .groupby(["unit", "product_category"], dropna=False, sort=False)
            if pd.notna(key[0])
        }

    def batch_features(self, customer_ids: Sequence[str], as_of_dates: Sequence[object]) -> torch.Tensor:
        """Return signed log-price gap, its magnitude, and a price availability flag."""
        if len(customer_ids) != len(as_of_dates):
            raise ValueError("Price preference requires one customer and date per query.")
        dates = pd.to_datetime(list(as_of_dates), format="mixed").to_numpy(dtype="datetime64[D]")
        if np.isnat(dates).any():
            raise ValueError("Price preference requires a valid scoring date.")
        shape = (len(dates), len(self.product_ids))
        candidate_prices = np.full(shape, np.nan, dtype=np.float32)
        for j, sku in enumerate(self.product_ids):
            if sku in self.sku_prices:
                candidate_prices[:, j] = _prior_median(self.sku_prices[sku], dates, minimum=1)
        typical_prices = np.full(shape, np.nan, dtype=np.float32)
        customer_rows: dict[str, list[int]] = {}
        for i, cid in enumerate(customer_ids):
            customer_rows.setdefault(str(cid), []).append(i)
        for cid, rows in customer_rows.items():
            query_dates = dates[rows]
            unit_fallbacks = {}
            for (unit, category), columns in self.product_groups.items():
                unit_key = (cid, unit)
                if unit_key not in unit_fallbacks:
                    unit_timeline = self.customer_unit_prices.get(unit_key)
                    unit_fallbacks[unit_key] = (
                        _prior_median(unit_timeline, query_dates, minimum=3) if unit_timeline is not None else None
                    )
                typical = unit_fallbacks[unit_key]
                timeline = self.customer_category_prices.get((cid, unit, category))
                if timeline is not None:
                    category_median = _prior_median(timeline, query_dates, minimum=3)
                    typical = category_median if typical is None else np.where(
                        np.isfinite(category_median), category_median, typical,
                    )
                if typical is not None:
                    typical_prices[np.ix_(rows, columns)] = typical[:, None]
        available = np.isfinite(candidate_prices) & np.isfinite(typical_prices)
        gap = np.zeros(shape, dtype=np.float32)
        np.log(candidate_prices, out=gap, where=available)
        np.log(typical_prices, out=typical_prices, where=available)
        np.subtract(gap, typical_prices, out=gap, where=available)
        np.tanh(gap, out=gap)
        return torch.from_numpy(np.stack((gap, np.abs(gap), available), axis=-1))


def load_reconciled_prices(history: pd.DataFrame) -> pd.DataFrame:
    """Use only invoice prices that reconcile with the supplied purchase history."""
    matching = history[[*EVENT_KEY_COLUMNS, "quantity"]].copy()
    matching["purchase_date"] = pd.to_datetime(matching["purchase_date"], format="mixed").dt.normalize()
    matching["customer_id"] = matching["customer_id"].astype("string")
    matching["product_id"] = matching["product_id"].astype("string")
    matching = matching.groupby(EVENT_KEY_COLUMNS, as_index=False, observed=True)["quantity"].sum()
    raw = load_source_purchases(get_datasets_locs("purchases"), include_turnover=True)
    paid = select_paid_purchases(raw.loc[raw["purchase_date"] <= matching["purchase_date"].max()])
    priced = priced_purchase_events(paid)
    priced = priced.merge(matching, on=EVENT_KEY_COLUMNS, suffixes=("", "_history"), validate="one_to_one")
    return priced.loc[np.isclose(priced["quantity"], priced["quantity_history"])]


def load_price_preference_store(
    history: pd.DataFrame, catalogue: pd.DataFrame, product_ids: Sequence[str],
) -> PricePreferenceStore:
    return PricePreferenceStore(load_reconciled_prices(history), catalogue, product_ids)


class CustomerMedianUnitPrice:
    """One median paid price per unit across all prior purchase records."""

    def __init__(self, priced_events: pd.DataFrame, reference: float):
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("The median price reference must be finite and positive.")
        self.log_reference = math.log1p(reference)
        valid = priced_events.loc[np.isfinite(priced_events["unit_price_uah"]) & priced_events["unit_price_uah"].gt(0)]
        self.timelines = _price_timelines(valid.sort_values("purchase_date", kind="stable"), "customer_id")

    def append_to(
        self, profiles: torch.Tensor, customer_ids: Sequence[str], as_of_dates: Sequence[object],
    ) -> torch.Tensor:
        if len(customer_ids) != len(as_of_dates) or len(customer_ids) != len(profiles):
            raise ValueError("The price profile requires one customer and date per row.")
        dates = pd.to_datetime(list(as_of_dates), format="mixed").to_numpy(dtype="datetime64[D]")
        if np.isnat(dates).any():
            raise ValueError("The price profile requires valid scoring dates.")
        medians = np.zeros((len(dates), 1), dtype=np.float32)
        for i, (customer, date) in enumerate(zip(customer_ids, dates)):
            timeline = self.timelines.get(str(customer))
            if timeline is not None:
                past, prices = timeline
                count = past.searchsorted(date, side="left")
                if count:
                    medians[i, 0] = prices[count - 1]
        log_medians = np.log1p(medians)
        scalar = torch.from_numpy(log_medians / (log_medians + self.log_reference)).to(profiles)
        return torch.cat((profiles, scalar), dim=-1)
