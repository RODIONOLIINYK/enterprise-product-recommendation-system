import pandas as pd
import torch

from recommender.candidates import CandidateRetriever
from recommender.familiarity import BASE_FAMILIARITY_FIELDS, FamiliarityStore
from recommender.models import ProductTower, TwoTowerModel


def test_familiarity_uses_prior_attributes_even_for_products_outside_candidate_catalogue():
    catalogue = pd.DataFrame({
        "product_id": ["old", "new", "today", "unknown"],
        "brand": ["B", "B", "C", None],
        "product_category": ["X", "Y", "X", None],
        "business_line": ["L", "L", "M", None],
        "product_family": ["F", "F", "G", None],
    })
    history = pd.DataFrame({
        "customer_id": ["buyer", "buyer"], "product_id": ["old", "today"],
        "purchase_date": ["2025-01-01", "2025-02-01"],
    })
    ids = ["new", "today", "unknown"]
    store = FamiliarityStore(history, catalogue, ids, (*BASE_FAMILIARITY_FIELDS, "family"))
    model = TwoTowerModel(torch.nn.Identity(), torch.nn.Identity(), 3, num_familiarity_features=4)
    with torch.no_grad():
        model.familiarity_weights.copy_(torch.tensor([1., 2., 4., 8.]))
    retriever = CandidateRetriever(model, familiarity=store)
    scores = retriever.compute_logits(
        torch.zeros(2, 3), torch.zeros(3, 3), customer_ids=["buyer", "cold"],
        as_of_dates=["2025-02-01", "2025-02-01"],
    )
    torch.testing.assert_close(scores, torch.tensor([[13., 2., 0.], [0., 0., 0.]]))


def test_product_tower_can_drop_all_three_attribute_embeddings():
    tower = ProductTower(
        text_dim=4, output_dim=4, hidden_dim=4, num_demand_features=1,
        num_brands=0, num_categories=0, num_business_lines=0, num_subcategories=0,
    ).eval()
    assert tower.brand_embedding is tower.cat_embedding is tower.line_embedding is None
    text, demand = torch.ones(2, 4), torch.ones(2, 1)
    a, b = torch.tensor([0, 0]), torch.tensor([1, 2])
    torch.testing.assert_close(tower(text, a, a, None, a, demand), tower(text, b, b, None, b, demand))
