import numpy as np
import pandas as pd
import pytest
import torch

from recommender.candidates import CandidateRetriever
from recommender.models import TwoTowerModel
from recommender.price_features import CustomerMedianUnitPrice, PricePreferenceStore


def test_customer_profile_has_one_pooled_price_scalar_using_only_prior_sales():
    prices = pd.DataFrame({
        "customer_id": ["buyer", "buyer", "buyer", "other"],
        "product_id": ["A", "B", "C", "A"],
        "purchase_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"],
        "unit_price_uah": [40, 160, 2000, 1000],
    })
    store = CustomerMedianUnitPrice(prices, reference=100)
    retriever = CandidateRetriever(customer_median_price=store)
    profiles = torch.ones(2, 3)
    augmented = retriever.customer_profile_inputs(profiles, ["buyer", "cold"], ["2025-01-03", "2025-01-03"])
    torch.testing.assert_close(augmented, torch.tensor([[1, 1, 1, 0.5], [1, 1, 1, 0]]))


def test_price_preference_is_point_in_time_and_uses_comparable_units():
    catalogue = pd.DataFrame(
        {"product_id": ["A", "B", "C", "D", "E"],
         "unit": ["L", "L", "L", "KG", "L"],
         "product_category": ["oil", "oil", "coolant", "oil", "oil"]},
    )
    events = pd.DataFrame(
        [
            ("other", "B", "2025-01-01", 120),
            ("other", "C", "2025-01-01", 90),
            ("other", "D", "2025-01-01", 100),
            ("buyer", "A", "2025-01-01", 80),
            ("buyer", "A", "2025-01-02", 100),
            ("buyer", "A", "2025-01-03", 110),
            ("buyer", "A", "2025-01-04", 1000),
            ("other", "B", "2025-01-04", 1000),
        ],
        columns=["customer_id", "product_id", "purchase_date", "unit_price_uah"],
    )
    store = PricePreferenceStore(events, catalogue, ["B", "C", "D", "E", "A"])
    features = store.batch_features(["buyer", "buyer", "cold"], ["2025-01-03", "2025-01-04", "2025-01-04"])
    oil = np.tanh(np.log(1.2))
    coolant = np.tanh(np.log(0.9))
    expected = torch.tensor(
        [[oil, abs(oil), 1], [coolant, abs(coolant), 1], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(features[1], expected)
    assert torch.count_nonzero(features[[0, 2]]) == 0

    model = TwoTowerModel(torch.nn.Identity(), torch.nn.Identity(), 5, include_price_preference=True)
    with torch.no_grad():
        model.price_weights.copy_(torch.tensor([0.5, -0.8, 0.1]))
    retriever = CandidateRetriever(model, price_preference=store)
    customers, products = torch.zeros(3, 2), torch.zeros(5, 2)
    with pytest.raises(ValueError, match="scoring dates"):
        retriever.compute_logits(customers, products)
    scores = retriever.compute_logits(
        customers, products, customer_ids=["buyer", "buyer", "cold"],
        as_of_dates=["2025-01-03", "2025-01-04", "2025-01-04"],
    )
    torch.testing.assert_close(scores, features @ model.price_weights)
