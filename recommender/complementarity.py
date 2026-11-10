"""Point-in-time SKU shares for purchases following a different product basket."""

from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np
import pandas as pd

from recommender.scope import actual_purchase_mask
from recommender.split import TRAIN_CUTOFF

SKU_COMPLEMENTARITY_FEATURE = "sku_five_day_complementarity_share"


class ProductComplementarity:
    """Count each customer–SKU–day once and use only earlier purchase days."""

    def __init__(self, purchases: pd.DataFrame):
        if {"item_type", "transaction_type", "quantity"} <= set(purchases.columns):
            purchases = purchases.loc[actual_purchase_mask(purchases)]
        events = purchases[["customer_id", "purchase_date", "product_id"]].dropna().copy()
        events["purchase_date"] = pd.to_datetime(events["purchase_date"], format="mixed").dt.normalize()
        events = events.loc[events["purchase_date"] < TRAIN_CUTOFF]
        baskets = events.groupby(["customer_id", "purchase_date"])["product_id"].agg(set)
        timelines = defaultdict(list)
        window = pd.Timedelta(days=5)
        for _, customer_baskets in baskets.groupby(level=0, sort=False):
            recent = deque()
            for (_, day), products in customer_baskets.items():
                while recent and day - recent[0][0] > window:
                    recent.popleft()
                previous_products = set().union(*(skus for _, skus in recent))
                for sku in products:
                    followed_other_product = bool(previous_products - {sku})
                    timelines[str(sku)].append((day, followed_other_product))
                recent.append((day, products))
        self.timelines = {}
        for sku, rows in timelines.items():
            rows.sort(key=lambda row: row[0])
            dates, complementary = zip(*rows)
            self.timelines[sku] = (dates, np.cumsum(complementary))

    def shares(self, product_ids: Sequence[str], as_of: str | pd.Timestamp) -> np.ndarray:
        """Return earlier complementary purchases divided by earlier SKU purchases."""
        cutoff = min(pd.Timestamp(as_of).normalize(), TRAIN_CUTOFF)
        result = np.zeros(len(product_ids), dtype=np.float32)
        for i, sku in enumerate(product_ids):
            timeline = self.timelines.get(sku)
            if timeline is not None:
                dates, complementary = timeline
                count = bisect_left(dates, cutoff)
                if count:
                    result[i] = complementary[count - 1] / count
        return result


def product_snapshot_date(as_of: str | pd.Timestamp) -> pd.Timestamp:
    """Share one train-only product-vector snapshot per calendar month."""
    return min(pd.Timestamp(as_of).normalize().replace(day=1), TRAIN_CUTOFF)


def group_snapshot_rows(snapshots: Sequence[pd.Timestamp]) -> list[list[int]]:
    groups = defaultdict(list)
    for row, snapshot in enumerate(snapshots):
        groups[snapshot].append(row)
    return list(groups.values())
