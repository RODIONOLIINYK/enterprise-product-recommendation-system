import csv
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from recommender.candidates import CandidateRetriever
from recommender.catalogue import catalogue_blocked_mask, load_product_catalogue, select_sellable_products
from recommender.evaluate_stage1 import Heuristics, take_unique
from recommender.models import TwoTowerModel
from recommender.split import TRAIN_CUTOFF
from recommender.train_retriever import build_retriever_product_inputs


def test_catalogue_block_status_uses_the_export_fields_and_historical_date(tmp_path):
    source = tmp_path / "items.csv"
    with source.open("w") as handle:
        writer = csv.writer(handle, delimiter=";")
        for sku, blocked, modified in [
            ("past", "1", "2025-10-01 00:00:00.000"),
            ("same_day", "1", "2025-10-02 00:00:00.000"),
            ("unknown_date", "1", "1753-01-01 00:00:00.000"),
            ("unblocked", "0", "2025-10-01 00:00:00.000"),
        ]:
            row = [""] * 85
            row[1], row[46], row[47], row[50] = sku, blocked, modified, "ТОВАР"
            writer.writerow(row)
    products = select_sellable_products(load_product_catalogue(source))
    assert len(products) == 4
    assert catalogue_blocked_mask(products).tolist() == [True, True, True, False]
    assert catalogue_blocked_mask(products, "2025-10-02").tolist() == [True, False, False, False]
    assert catalogue_blocked_mask(products, "2025-10-03").tolist() == [True, True, False, False]


def test_expanded_candidates_keep_checkpoint_order_and_get_zero_bias():
    model = TwoTowerModel(torch.nn.Identity(), torch.nn.Identity(), num_catalog_products=1)
    with torch.no_grad():
        model.product_bias.fill_(2)
    checkpoint = {"product_ids": ["sold"]}
    with patch("recommender.candidates.load_retriever_from_checkpoint", return_value=(model, checkpoint)):
        with pytest.raises(ValueError, match="checkpoint SKU order"):
            CandidateRetriever.from_checkpoint(
                "unused.pt", pd.DataFrame(), candidate_product_ids=["cold", "sold"],
            )
        retriever = CandidateRetriever.from_checkpoint(
            "unused.pt", pd.DataFrame(), candidate_product_ids=["sold", "cold"],
        )
    scores = retriever.compute_logits(torch.zeros(1, 2), torch.zeros(2, 2))
    torch.testing.assert_close(scores, torch.tensor([[2.0, 0.0]]))
    assert retriever.eligible_product_ids == ["sold", "cold"]


def test_discovery_default_admits_paid_skus_after_first_sale_and_excludes_known_blocks():
    ids = ["trained", "late", "blocked", "gift_only", "seen"]
    catalogue = pd.DataFrame({
        "product_id": ids, "item_type": ["ТОВАР"] * len(ids),
        "blocked": [False, False, True, False, False],
        "last_date_modified": pd.to_datetime([None, None, "2025-10-01", None, None]),
    })
    history = pd.DataFrame({
        "customer_id": ["other", "other", "other", "other", "buyer", "buyer"],
        "product_id": ["trained", "late", "blocked", "gift_only", "seen", "trained"],
        "purchase_date": pd.to_datetime(
            ["2025-09-29", "2025-10-02", "2025-09-29", "2025-09-29", "2025-09-29", "2025-09-29"],
        ),
        "transaction_type": ["ПРОДАЖА", "ПРОДАЖА", "ПРОДАЖА", "ПОДАРОК", "ПРОДАЖА", "ПОДАРОК"],
        "item_type": ["ТОВАР"] * 6, "quantity": [1] * 6,
    })
    model = TwoTowerModel(torch.nn.Identity(), torch.nn.Identity(), num_catalog_products=1)
    with torch.no_grad():
        model.product_bias.fill_(2)
    checkpoint = {"product_ids": ["trained"], "config": {"discovery_only": True}}
    with patch("recommender.candidates.load_retriever_from_checkpoint", return_value=(model, checkpoint)):
        retriever = CandidateRetriever.from_checkpoint(
            "unused.pt", history, catalogue, candidate_product_ids=ids,
        )
    ranked = retriever.discover_from_vectors(
        torch.zeros(3, 2), ["buyer"] * 3, ["2025-09-29", "2025-10-02", "2025-10-03"],
        torch.zeros(len(ids), 2),
    )
    assert ranked == [[], ["trained"], ["trained", "late"]]


def test_cold_product_demand_ignores_sales_at_or_after_training_cutoff():
    catalogue = pd.DataFrame({"product_id": ["sold", "cold"], "product_category": ["oil", "oil"]})
    history = pd.DataFrame({
        "customer_id": ["C", "C"],
        "product_id": ["sold", "cold"],
        "purchase_date": [TRAIN_CUTOFF - pd.Timedelta(days=1), TRAIN_CUTOFF],
        "quantity": [1.0, 1000000.0],
    })

    def demand(rows):
        return build_retriever_product_inputs(
            rows, catalogue, {}, ["sold", "cold"], {}, include_subcategory=False,
        )["demand_features"]

    features = demand(history)
    torch.testing.assert_close(features, demand(history.iloc[:1]))
    torch.testing.assert_close(features[1], torch.tensor([0., 0., 0., 0., 0., 1.]))
    assert torch.all((0 <= features) & (features <= 1))


def test_heuristics_respect_the_candidate_pool_and_skip_seen_skus():
    train = pd.DataFrame({
        "customer_id": ["C", "C", "D", "E"],
        "purchase_date": ["2025-01-01"] * 4,
        "product_id": ["seen", "partner", "popular", "popular"],
    })
    heuristics = Heuristics(train, {"C": "oil", "D": "other", "E": "other"})
    popularity = ["popular", "seen", "partner", "cold"]
    sources = heuristics.shortlist_sources(set(popularity), {"seen"}, "oil", popularity,
                                          {sku: i for i, sku in enumerate(popularity)})
    assert take_unique(sources["unseen_popularity"], {"seen"}, 2) == ["popular", "partner"]
    assert take_unique(sources["industry_adoption"], {"seen"}, 2) == ["partner", "popular"]
    assert take_unique(sources["co_purchase"], {"seen"}, 2) == ["partner", "popular"]
