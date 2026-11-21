import pandas as pd
import pytest
import torch

from recommender.experiments.roadmap_v2.first_baskets import append_first_baskets, first_baskets
from recommender.split import TEST_START


def test_first_baskets_include_single_orders_and_all_same_day_products():
    purchases = pd.DataFrame(
        [
            ("a", "2025-03-01", "later"),
            ("a", "2025-01-02", "first"),
            ("b", "2025-01-02", "solo"),
            ("a", "2025-01-02", "other"),
        ],
        columns=["customer_id", "purchase_date", "product_id"],
    )
    assert dict(first_baskets(purchases)) == {
        (pd.Timestamp("2025-01-02"), "a"): {"first", "other"},
        (pd.Timestamp("2025-01-02"), "b"): {"solo"},
    }
    purchases.loc[0, "purchase_date"] = TEST_START.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="development"):
        list(first_baskets(purchases))


def test_first_order_augmentation_keeps_returning_rows_and_feature_schema():
    train = {
        "customer_ids": ["a", "b"],
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "feature_names": ("existing",),
    }
    first = {"customer_ids": ["a"], "attention_mask": torch.zeros(1, 3, dtype=torch.long)}
    mixed = append_first_baskets(train, first)
    assert mixed["customer_ids"] == ["a", "b", "a"]
    assert torch.equal(mixed["attention_mask"][:2], train["attention_mask"])
    assert not mixed["attention_mask"][2].any()
    assert mixed["feature_names"] == train["feature_names"]
