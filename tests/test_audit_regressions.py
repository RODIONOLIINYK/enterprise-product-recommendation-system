"""Production-configuration and correctness regressions from the codebase audit."""

from unittest.mock import patch

import pandas as pd
import pytest
import torch

from recommender.candidates import PAIRWISE_FEATURE_NAMES
from recommender.features import compute_point_in_time_features
from recommender.models import ProductTower
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.scope import actual_purchase_mask
from recommender.sequences import build_customer_sequence
from recommender.split import chronological_split
from recommender.stage1_catalogue import load_stage1_discovery
from recommender.train_ranker import slate_cache_identity
from recommender.train_retriever import (
    build_retriever_product_inputs,
    build_retriever_transitions,
    compute_product_demand_features,
)


def reference_inputs(length=32):
    torch.manual_seed(17)
    mask = torch.zeros(2, length, dtype=torch.long)
    mask[0, :min(3, length)] = 1  # Include an entirely cold second customer.
    return {
        "text_embedding": torch.randn(2, 1024),
        "city_id": torch.ones(2, dtype=torch.long),
        "region_id": torch.ones(2, dtype=torch.long),
        "industry_id": torch.ones(2, dtype=torch.long),
        "tabular_features": torch.rand(2, 8),
        "sequence": torch.rand(2, length, 9),
        "attention_mask": mask,
        "seq_product_ids": mask.clone(),
        "seq_category_ids": mask.clone(),
        "seq_subcategory_ids": mask.clone(),
        "seq_business_line_ids": mask.clone(),
        "seq_purchase_dates": torch.zeros(2, length),
        "prediction_day": torch.full((2,), 100.),
        "candidate_vectors": torch.nn.functional.normalize(torch.randn(2, 50, 128), dim=-1),
        "pairwise_features": torch.rand(2, 50, 13),
    }


def test_reference_configuration_parameter_utilization_and_candidate_precedence():
    torch.manual_seed(23)
    model = UPSCANRRanker(
        statistics_attention=True, excluded_statistics=("history_repeat",),
        product_tower=ProductTower(), dropout=0,
    ).train()
    inputs = reference_inputs()
    text_inputs = []
    hook = model.product_text_proj.register_forward_pre_hook(
        lambda _module, args: text_inputs.append(args[0].detach()),
    )
    # A wrong-width fallback catalogue must be ignored with direct vectors.
    output = model(**inputs, catalog_product_vectors=torch.zeros(3, 7))
    hook.remove()
    labels = torch.zeros(2, 50)
    labels[:, 0] = 1
    RankerLoss()(output.logits, labels)["loss"].backward()

    assert torch.count_nonzero(text_inputs[0]) == 0
    assert all(p.grad is None for p in model.product_tower.parameters())
    assert torch.count_nonzero(model.habit_proj.weight.grad) == 0
    assert torch.count_nonzero(model.product_text_proj.weight.grad) == 0
    assert model.habit_proj.bias.grad.norm() > 0
    assert model.product_text_proj.bias.grad.norm() > 0
    assert torch.count_nonzero(model.tabular_proj[0].weight.grad[:, 48:]) == 0
    assert model.tabular_proj[0].weight.grad[:, :48].norm() > 0
    assert torch.count_nonzero(model.statistics_attention_weights.grad[..., 7]) == 0
    assert model.statistics_attention_weights.grad.norm() > 0
    for projection in model.cross_attention.in_proj_weight.grad.chunk(3):
        assert torch.isfinite(projection).all() and projection.norm() > 0
    padding = torch.cat([torch.zeros(2, 1, dtype=torch.bool), inputs["attention_mask"] == 0], 1)
    assert torch.count_nonzero(output.attn_weights.masked_select(padding[:, None])) == 0


def test_reference_statistics_change_bias_but_not_memory_values():
    model = UPSCANRRanker(
        statistics_attention=True, excluded_statistics=("history_repeat",), dropout=0,
    ).eval()
    with torch.no_grad():
        model.statistics_attention_weights.normal_(std=0.1)
        inputs = reference_inputs()
        first = model(**inputs)
        inputs["sequence"] = torch.zeros_like(inputs["sequence"])
        inputs["tabular_features"] = torch.zeros_like(inputs["tabular_features"])
        second = model(**inputs)
    torch.testing.assert_close(first.memory, second.memory, rtol=0, atol=0)
    assert not torch.allclose(first.attn_weights, second.attn_weights)
    assert not torch.allclose(first.logits, second.logits)


@pytest.mark.parametrize("length", [0, 4, 32, 40])
def test_ranker_masks_follow_actual_history_length(length):
    model = UPSCANRRanker(statistics_attention=True, dropout=0).eval()
    with torch.no_grad():
        output = model(**reference_inputs(length))
    assert output.memory.shape == (2, length + 1, 128)
    assert torch.isfinite(output.logits).all()
    assert output.token_0_weight[1].eq(1).all()


def mixed_sales():
    return pd.DataFrame({
        "customer_id": ["C"] * 3,
        "product_id": ["P"] * 3,
        "purchase_date": ["2025-01-01", "2025-01-02 12:00:00", "2025-01-03T12:00:00"],
        "quantity": [1., 2., 3.],
        "item_type": ["ТОВАР"] * 3,
        "transaction_type": ["ПРОДАЖА"] * 3,
    })


def test_mixed_date_sales_are_not_silently_discarded():
    assert actual_purchase_mask(mixed_sales()).tolist() == [True, True, True]


def test_mixed_dates_match_parsed_dates_across_public_feature_entrypoints():
    mixed = mixed_sales()
    parsed = mixed.assign(purchase_date=pd.to_datetime(mixed["purchase_date"], format="mixed"))
    cutoff = pd.Timestamp("2025-01-03T12:00:00")
    torch.testing.assert_close(
        compute_product_demand_features(mixed, cutoff, ["P"]),
        compute_product_demand_features(parsed, cutoff, ["P"]),
    )
    assert build_retriever_transitions(mixed) == build_retriever_transitions(parsed)
    assert compute_point_in_time_features(mixed, "C", cutoff) == compute_point_in_time_features(parsed, "C", cutoff)
    torch.testing.assert_close(
        build_customer_sequence(mixed, "C", cutoff, enriched=True)["sequence"],
        build_customer_sequence(parsed, "C", cutoff, enriched=True)["sequence"],
    )
    for left, right in zip(chronological_split(mixed), chronological_split(parsed), strict=True):
        assert left.index.equals(right.index)


def test_fixed_product_snapshot_contains_later_training_sales():
    """Document deferred temporal bug without changing the saved model's policy."""
    early = mixed_sales().iloc[:1].copy()
    late = early.assign(purchase_date="2025-09-01", quantity=100.)
    catalogue = pd.DataFrame({"product_id": ["P"], "product_category": ["oil"]})

    def fixed(rows):
        return build_retriever_product_inputs(
            rows, catalogue, {}, ["P"], {}, include_subcategory=False,
        )["demand_features"]

    full = pd.concat([early, late], ignore_index=True)
    assert not torch.equal(fixed(early), fixed(full))
    torch.testing.assert_close(
        compute_product_demand_features(early, "2025-06-01", ["P"]),
        compute_product_demand_features(full, "2025-06-01", ["P"]),
    )


def test_discovery_loader_preserves_optional_sku_median_feature(tmp_path):
    history = mixed_sales()
    catalogue = pd.DataFrame({"product_id": ["P"], "item_type": ["ТОВАР"], "product_category": ["oil"]})
    checkpoint = {"config": {"discovery_only": True, "container_encoding": "one_hot_none",
                              "include_sku_median_quantity": True},
                  "product_ids": ["P"], "vocabs": {}}
    with (
        patch("recommender.stage1_catalogue.torch.load", return_value=checkpoint),
        patch("recommender.stage1_catalogue.load_expanded_catalogue_embeddings", return_value={}),
        patch("recommender.stage1_catalogue.build_retriever_product_inputs", return_value={}) as builder,
        patch("recommender.stage1_catalogue.CandidateRetriever.from_checkpoint"),
    ):
        load_stage1_discovery(tmp_path / "model.pt", history, catalogue, pd.DataFrame(), tmp_path, torch.device("cpu"))
    assert builder.call_args.kwargs["include_sku_median_quantity"] is True


def test_slate_identity_binds_expanded_vectors_schema_and_scope(tmp_path):
    embedding = tmp_path / "base.parquet"
    embedding.write_bytes(b"base vectors")
    scope = tmp_path / "scope.toml"
    scope.write_bytes(b"policy v1")
    frame = pd.DataFrame({"id": ["one"]})
    product_inputs = {"text_embedding": torch.zeros(2, 1024), "demand_features": torch.zeros(2, 6)}

    def identity():
        return slate_cache_identity(
            "checkpoint hash", tmp_path / "model.pt", frame, frame, frame,
            include_subcategory=False, rolling=True, product_inputs=product_inputs,
        )

    with (
        patch("recommender.train_ranker.catalogue_embeddings_path", return_value=embedding),
        patch("recommender.train_ranker.company_embeddings_path", return_value=embedding),
        patch("recommender.train_ranker.DEFAULT_SCOPE_PATH", scope),
    ):
        baseline = identity()
        product_inputs["text_embedding"][1, 0] = 1  # Expanded SKU only.
        assert identity() != baseline
        product_inputs["text_embedding"].zero_()
        with patch("recommender.train_ranker.PAIRWISE_FEATURE_NAMES", list(reversed(PAIRWISE_FEATURE_NAMES))):
            assert identity() != baseline
        scope.write_bytes(b"policy v2")
        assert identity() != baseline
