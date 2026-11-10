import math

import pandas as pd
import pytest

from recommender.purchase_timing import PurchaseTiming


def test_reorder_gap_falls_back_from_customer_to_sku_to_category():
    purchases = pd.DataFrame([
        ("buyer", "sku", "2025-01-01"),
        ("buyer", "sku", "2025-01-11"),
        ("other", "sku", "2025-01-02"),
        ("other", "sku", "2025-01-06"),
        ("once", "sibling", "2025-01-01"),
        ("once", "sibling", "2025-01-21"),
    ], columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({
        "product_id": ["sku", "sibling", "unseen"],
        "product_category": ["category"] * 3,
    })
    timing = PurchaseTiming(purchases, catalogue)
    # Same-day purchases do not supply a gap. SKU population keeps the existing
    # pooled gap median (4 and 10 days).
    urgency, gap = timing.candidate_features("buyer", "sku", pd.Timestamp("2025-01-11"))
    assert urgency == pytest.approx(0.5)
    assert gap == pytest.approx(math.log1p(4) / math.log1p(365))
    urgency, gap = timing.candidate_features("buyer", "sku", pd.Timestamp("2025-01-12"))
    assert urgency == pytest.approx(0.02)
    assert gap == pytest.approx(math.log1p(10) / math.log1p(365))
    urgency, gap = timing.candidate_features("buyer", "unseen", pd.Timestamp("2025-01-22"))
    assert urgency == 0
    # Category fallback is the equal-pair mean of medians: (10 + 4 + 20) / 3.
    assert gap == pytest.approx(math.log1p(34 / 3) / math.log1p(365))
    sku_only = PurchaseTiming(purchases, catalogue, category_gap_fallback=False)
    as_of = pd.Timestamp("2025-01-22")
    assert sku_only.candidate_features("buyer", "unseen", as_of) == (0, 0)
    assert sku_only.candidate_features("buyer", "sku", as_of) == timing.candidate_features("buyer", "sku", as_of)


def test_category_fallback_uses_equal_weight_customer_sku_medians_not_cross_sku_gaps():
    rows = []
    # The first pair has many purchases and a 10-day median interval.
    for date in pd.date_range("2025-01-01", periods=6, freq="10D"):
        rows.append(("frequent", "sku-a", date.strftime("%Y-%m-%d")))
    # The second pair has fewer purchases and a 25-day median interval.
    for date in ("2025-01-01", "2025-01-21", "2025-02-20"):
        rows.append(("infrequent", "sku-b", date))
    # Interleaving different SKUs must not create one-day category intervals.
    for day in (1, 3, 5, 7, 9):
        rows.append(("interleaved", "sku-c", f"2025-03-{day:02d}"))
    for day in (2, 4, 6, 8, 10):
        rows.append(("interleaved", "sku-d", f"2025-03-{day:02d}"))
    purchases = pd.DataFrame(rows, columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({
        "product_id": ["sku-a", "sku-b", "sku-c", "sku-d", "unseen"],
        "product_category": ["cat"] * 5,
    })
    timing = PurchaseTiming(purchases, catalogue)

    # Pair medians are 10, 25, 2, and 2 days: each customer/SKU pair has
    # equal weight regardless of its number of purchases.
    _, gap = timing.candidate_features("new", "unseen", pd.Timestamp("2025-04-01"))
    assert gap == pytest.approx(math.log1p(9.75) / math.log1p(365))


def test_category_pair_medians_obey_strict_as_of_cutoff():
    purchases = pd.DataFrame([
        ("buyer", "sku-a", "2025-01-01"),
        ("buyer", "sku-a", "2025-01-11"),
        ("buyer", "sku-a", "2025-03-12"),
        ("buyer", "sku-a", "2025-03-22"),
        ("other", "sku-b", "2025-01-01"),
        ("other", "sku-b", "2025-01-21"),
    ], columns=["customer_id", "product_id", "purchase_date"])
    catalogue = pd.DataFrame({
        "product_id": ["sku-a", "sku-b", "unseen"],
        "product_category": ["cat"] * 3,
    })
    timing = PurchaseTiming(purchases, catalogue)
    cutoff = pd.Timestamp("2025-02-01").to_datetime64().astype("datetime64[D]").astype(int)
    # Only sku-a's first 10-day interval exists as of the cutoff; its later
    # March interval cannot change that pair's median. sku-b contributes 20.
    assert timing._category_gap("cat", cutoff) == pytest.approx(15.0)
