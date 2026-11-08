"""Full-history attribute matches for Stage 1 candidates."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch


ATTRIBUTE_COLUMNS = {
    "brand": "brand",
    "category": "product_category",
    "business_line": "business_line",
    "family": "product_family",
}
BASE_FAMILIARITY_FIELDS = ("brand", "category", "business_line")


class FamiliarityStore:
    """Look up whether a customer's first purchase of an attribute predates the query."""

    def __init__(
        self, history: pd.DataFrame, catalogue: pd.DataFrame, product_ids: Sequence[str], fields: Sequence[str],
    ):
        self.fields = tuple(fields)
        if not self.fields or len(set(self.fields)) != len(self.fields) or not set(self.fields) <= ATTRIBUTE_COLUMNS.keys():
            raise ValueError("Select distinct familiarity fields from brand, category, business line and family.")
        columns = [ATTRIBUTE_COLUMNS[field] for field in self.fields]
        attrs = catalogue.set_index("product_id")[columns].astype("string")
        attrs.index = attrs.index.astype(str)
        self.candidate_attributes = attrs.reindex(product_ids)
        events = history[["customer_id", "product_id", "purchase_date"]].copy()
        events[["customer_id", "product_id"]] = events[["customer_id", "product_id"]].astype(str)
        events["purchase_date"] = pd.to_datetime(events["purchase_date"], format="mixed").dt.normalize()
        events = events.join(attrs, on="product_id", validate="many_to_one")
        self.first_purchase = {
            col: events.groupby(["customer_id", col], observed=True)["purchase_date"].min().unstack()
            for col in columns
        }

    def batch_features(self, customer_ids: Sequence[str], as_of_dates: Sequence[object]) -> torch.Tensor:
        if len(customer_ids) != len(as_of_dates):
            raise ValueError("Familiarity requires one customer and date per query.")
        dates = pd.to_datetime(list(as_of_dates), format="mixed").to_numpy(dtype="datetime64[D]")
        if np.isnat(dates).any():
            raise ValueError("Familiarity requires valid scoring dates.")
        customers = [str(customer) for customer in customer_ids]
        columns = []
        for col in self.candidate_attributes:
            first_dates = self.first_purchase[col].reindex(
                index=customers, columns=self.candidate_attributes[col],
            ).to_numpy(dtype="datetime64[D]")
            columns.append(first_dates < dates[:, None])
        return torch.from_numpy(np.stack(columns, axis=-1))
