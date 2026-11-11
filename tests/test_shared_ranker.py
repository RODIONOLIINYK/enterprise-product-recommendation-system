"""Lossless shared inputs, indicator ablations and isolated cache compatibility."""

import copy
from dataclasses import replace

import pandas as pd
import pytest
import torch

from recommender.candidates import CandidateRetriever, PointInTimeFeatureStore
from recommender.features import LEGACY_PROFILE_FEATURES, PointInTimeNormalizer
from recommender.models import CustomerTower, TwoTowerModel
from recommender.ranker import UPSCANRRanker
from recommender.sequences import ENRICHED_FEATURE_NAMES
from recommender.shared_data import (
    SHARED_CACHE_VERSION,
    load_development_purchases,
    validate_shared_cache,
)
from recommender.shared_ranker import (
    HISTORY_INPUT_NAMES,
    REGIONS,
    SharedRankerConfig,
    SharedStage1Ranker,
    SharedStage1TokenBuilder,
    StatisticalAttentionBias,
    feature_schema,
    map_history_skus,
    schema_digest,
)
from recommender.train_ranker import (
    RankerDataset,
    collate_ranker_batch,
    precompute_candidate_slates_for_transitions,
)
from recommender.train_shared_ranker import match_width_control_initialization


def inputs(length=4):
    torch.manual_seed(5)
    vectors = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
    mask = torch.zeros(2, length, dtype=torch.long)
    mask[0, :min(2, length)] = 1
    history = torch.full((2, length), -1, dtype=torch.long)
    history[0, :min(2, length)] = torch.arange(min(2, length))
    sequence = torch.rand(2, length, 9)
    sequence[..., 8] = mask
    return vectors, {
        "stage1_customer_vectors": torch.nn.functional.normalize(torch.randn(2, 128), dim=-1),
        "seq_catalog_indices": history, "candidate_indices": torch.tensor([[0, 1, 2]] * 2),
        "tabular_features": torch.rand(2, 8), "sequence": sequence, "pairwise_features": torch.rand(2, 3, 13),
        "attention_mask": mask, "seq_purchase_dates": torch.zeros(2, length), "prediction_day": torch.full((2,), 100.),
        "city_id": torch.ones(2, dtype=torch.long), "region_id": torch.ones(2, dtype=torch.long),
        "industry_id": torch.ones(2, dtype=torch.long),
    }


def build_tokens(builder, batch):
    return builder(**{key: value for key, value in batch.items() if key not in ("seq_purchase_dates", "prediction_day")})


def test_lossless_regions_source_schema_token_types_and_padding():
    assert HISTORY_INPUT_NAMES == ENRICHED_FEATURE_NAMES[:8]
    assert "real_event_indicator" not in HISTORY_INPUT_NAMES
    assert len(ENRICHED_FEATURE_NAMES) == 9
    vectors, batch = inputs()
    batch["sequence"][0, 0, 3:5] = torch.tensor([-.8, -.3])
    config = SharedRankerConfig(memory_statistics=True, query_statistics=True, exclude_history_repeat=False)
    builder = SharedStage1TokenBuilder(vectors, config)
    memory, candidates = build_tokens(builder, batch)
    assert memory.shape == (2, 5, 176) and candidates.shape == (2, 3, 176)
    torch.testing.assert_close(memory[:, 0, :128], batch["stage1_customer_vectors"], rtol=0, atol=0)
    torch.testing.assert_close(memory[0, 1:3, :128], vectors[:2], rtol=0, atol=0)
    torch.testing.assert_close(candidates[..., :128], vectors.expand(2, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(memory[0, 2, :128], candidates[0, 1, :128], rtol=0, atol=0)
    scales = builder.log_scales.exp()
    torch.testing.assert_close(memory[:, 0, REGIONS["customer_statistics"]] / scales[0], batch["tabular_features"])
    torch.testing.assert_close(memory[0, 1:3, REGIONS["history_statistics"]] / scales[1], batch["sequence"][0, :2, :8])
    torch.testing.assert_close(candidates[..., REGIONS["candidate_statistics"]] / scales[2], batch["pairwise_features"])
    torch.testing.assert_close(memory[:, 0, REGIONS["type"]], torch.tensor([[1., 0., 0.]] * 2))
    torch.testing.assert_close(memory[0, 1:3, REGIONS["type"]], torch.tensor([[0., 1., 0.]] * 2))
    torch.testing.assert_close(candidates[..., REGIONS["type"]], torch.tensor([[[0., 0., 1.]] * 3] * 2))
    assert torch.count_nonzero(memory[:, 1:][~batch["attention_mask"].bool()]) == 0


def test_numerical_groups_have_independent_configurable_scales():
    vectors, batch = inputs()
    config = SharedRankerConfig(memory_statistics=True, query_statistics=True,
                               customer_statistics_scale=.2, history_statistics_scale=.3, candidate_statistics_scale=.4)
    builder = SharedStage1TokenBuilder(vectors, config)
    torch.testing.assert_close(builder.log_scales.exp()[:3], torch.tensor([.2, .3, .4]))
    memory, candidates = build_tokens(builder, batch)
    torch.testing.assert_close(memory[:, 0, 128:136], batch["tabular_features"] * .2)
    torch.testing.assert_close(candidates[..., 144:157], batch["pairwise_features"] * .4)


def test_expanded_sku_mapping_unknown_and_padding_are_explicit():
    mapping = map_history_skus([["expanded", "trained", "unknown", "[PAD]"]], ["trained", "expanded"])
    assert mapping.tolist() == [[1, 0, -1, -1]]
    vectors, batch = inputs()
    batch["seq_catalog_indices"][0, :2] = torch.tensor([1, -1])
    builder = SharedStage1TokenBuilder(vectors, SharedRankerConfig())
    memory, candidates = build_tokens(builder, batch)
    torch.testing.assert_close(memory[0, 1, :128], candidates[0, 1, :128], rtol=0, atol=0)
    assert torch.count_nonzero(memory[0, 2, :128]) == 0  # Unknown real event, not padding.
    assert memory[0, 2, 174] == 1 and memory[0, 3].eq(0).all()


@pytest.mark.parametrize("length", range(33))
@pytest.mark.parametrize("schema", ["legacy17", "reduced16", "none"])
def test_all_history_lengths_cold_outputs_and_attention_padding(length, schema):
    vectors, batch = inputs(length)
    model = SharedStage1Ranker(vectors, SharedRankerConfig(bias_schema=schema, dropout=0)).eval()
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output.logits).all()
    assert output.memory.shape == (2, length + 1, 176)
    mask = torch.nn.functional.pad(~batch["attention_mask"].bool(), (1, 0))
    assert output.attn_weights.masked_select(mask[:, None]).eq(0).all()
    assert output.token_0_weight[1].eq(1).all()


@pytest.mark.parametrize("schema", ["legacy17", "reduced16"])
def test_source_indicator_never_enters_new_tokens_or_changes_bias(schema):
    vectors, batch = inputs()
    model = SharedStage1Ranker(vectors, SharedRankerConfig(memory_statistics=True, query_statistics=True,
                                                        bias_schema=schema, dropout=0)).eval()
    with torch.no_grad():
        model.statistical_bias.weights.normal_()
        first = model(**batch)
        batch["sequence"][..., 8] = 1234  # Ignore source validity; mask is authoritative.
        second = model(**batch)
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    torch.testing.assert_close(first.memory, second.memory, rtol=0, atol=0)
    assert model.statistical_bias.weights.shape == (4, 14, 17 if schema == "legacy17" else 16)


def test_legacy_bias_matches_original_ranker_and_reduced_bias_can_differ():
    _, batch = inputs(32)
    reference = UPSCANRRanker(statistics_attention=True, excluded_statistics=("history_repeat",), dropout=0).eval()
    bias = StatisticalAttentionBias("legacy17")
    with torch.no_grad():
        reference.statistics_attention_weights.normal_()
        bias.weights.copy_(reference.statistics_attention_weights)
    observed = []
    hook = reference.cross_attention.register_forward_pre_hook(
        lambda _module, _args, kwargs: observed.append(kwargs["attn_mask"]), with_kwargs=True,
    )
    old_inputs = {key: batch[key] for key in ("tabular_features", "sequence", "attention_mask", "seq_purchase_dates", "prediction_day", "pairwise_features", "city_id", "region_id", "industry_id")}
    reference(**old_inputs, text_embedding=torch.zeros(2, 1024), candidate_vectors=torch.randn(2, 3, 128))
    hook.remove()
    mask = torch.nn.functional.pad(~batch["attention_mask"].bool(), (1, 0))
    bias_args = [batch[key] for key in ("pairwise_features", "sequence", "tabular_features", "attention_mask", "seq_purchase_dates", "prediction_day")]
    torch.testing.assert_close(bias(*bias_args, mask), observed[0], rtol=0, atol=0)
    reduced = StatisticalAttentionBias("reduced16")
    with torch.no_grad():
        reduced.weights.copy_(torch.cat([bias.weights[..., :8], bias.weights[..., 9:]], -1))
    assert not torch.allclose(bias(*bias_args, mask)[..., :3], reduced(*bias_args, mask)[..., :3])
    keys = bias.key_statistics(*[batch[key] for key in ("sequence", "tabular_features", "attention_mask", "seq_purchase_dates", "prediction_day")])
    assert keys[0, 1, 1] == pytest.approx(100 / 190)
    assert keys[0, 1, 1] != batch["sequence"][0, 0, 1]


def test_indicator_ablation_has_identical_nonbias_initialization_and_initial_scores():
    vectors, batch = inputs()
    torch.manual_seed(42)
    legacy = SharedStage1Ranker(vectors, SharedRankerConfig(memory_statistics=True, query_statistics=True, dropout=0)).eval()
    torch.manual_seed(42)
    reduced = SharedStage1Ranker(vectors, SharedRankerConfig(memory_statistics=True, query_statistics=True,
                                                          bias_schema="reduced16", dropout=0)).eval()
    for name, value in legacy.state_dict().items():
        if name != "statistical_bias.weights":
            torch.testing.assert_close(value, reduced.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(legacy(**batch).logits, reduced(**batch).logits, rtol=0, atol=0)


def test_disabled_numeric_regions_and_repeat_exclusion():
    vectors, batch = inputs()
    builder = SharedStage1TokenBuilder(vectors, SharedRankerConfig())
    first = build_tokens(builder, batch)
    batch["sequence"] *= 10
    batch["tabular_features"] *= 10
    batch["pairwise_features"] *= 10
    second = build_tokens(builder, batch)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    enabled = SharedStage1TokenBuilder(vectors, SharedRankerConfig(memory_statistics=True))
    memory, _ = build_tokens(enabled, batch)
    assert memory[..., 143].eq(0).all()  # Source history-repeat index 7.


@pytest.mark.parametrize("width,adapters", [(176, False), (192, False), (192, True), (224, True)])
def test_qkv_adapters_readout_gradients_and_frozen_stage1(width, adapters):
    vectors, batch = inputs()
    vectors.requires_grad_(True)
    batch["stage1_customer_vectors"].requires_grad_(True)
    config = SharedRankerConfig(width=width, memory_statistics=True, query_statistics=True,
                               learned_relationships=adapters, categorical_sizes=(3,) * 6, dropout=0)
    model = SharedStage1Ranker(vectors, config, torch.ones(3, 3, dtype=torch.long)).eval()
    assert model.context_readout.weight[:, :128].equal(torch.eye(128))
    assert model.context_readout.weight[:, 128:].eq(0).all()
    model(**batch).logits.square().mean().backward()
    assert vectors.grad is None and batch["stage1_customer_vectors"].grad is None
    assert not model.token_builder.catalog_product_vectors.requires_grad
    for attention in (model.cross_attention, model.memory_encoder.layers[0].self_attn):
        for projection in attention.in_proj_weight.grad.chunk(3):
            assert torch.isfinite(projection).all() and projection.norm() > 0
    assert model.statistical_bias.weights.grad.norm() > 0
    assert model.context_readout.weight.grad.norm() > 0
    assert model.mlp[-1].weight.grad.norm() > 0
    assert model.token_builder.log_scales.grad.norm() > 0
    if adapters:
        assert all(layer.weight.grad.norm() > 0 for layer in model.token_builder.adapters.values())
    if width == 192 and not adapters:
        memory, candidates = build_tokens(model.token_builder, batch)
        assert memory[..., 176:].eq(0).all() and candidates[..., 176:].eq(0).all()


def test_192d_width_control_and_zero_adapters_start_with_identical_common_weights():
    vectors, batch = inputs()
    config = SharedRankerConfig(width=192, memory_statistics=True, query_statistics=True, dropout=0)
    torch.manual_seed(42)
    control = SharedStage1Ranker(vectors, config).eval()
    torch.manual_seed(42)
    adapters = SharedStage1Ranker(vectors, replace(config, learned_relationships=True)).eval()
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, adapters.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(control(**batch).logits, adapters(**batch).logits, rtol=0, atol=0)


def test_numeric_memory_changes_outputs_and_profile_ablation_is_finite():
    vectors, batch = inputs()
    config = SharedRankerConfig(memory_statistics=True, bias_schema="none", dropout=0)
    model = SharedStage1Ranker(vectors, config).eval()
    first = model(**batch)
    batch["sequence"][0, 0, 0] += .5
    second = model(**batch)
    assert not torch.allclose(first.memory, second.memory)
    assert not torch.allclose(first.logits, second.logits)
    null_profile = SharedStage1Ranker(vectors, replace(config, profile_token=False)).eval()
    assert torch.isfinite(null_profile(**batch).logits).all()


def test_checkpoint_roundtrip_and_provenance_rejections():
    vectors, batch = inputs()
    model = SharedStage1Ranker(vectors).eval()
    product_ids = ["a", "b", "expanded"]
    checkpoint = {"model_metadata": model.metadata("stage1 sha", product_ids, "cache sha"),
                  "model_state_dict": model.state_dict()}
    restored = SharedStage1Ranker.from_checkpoint(checkpoint, vectors, product_ids, "stage1 sha", "cache sha").eval()
    torch.testing.assert_close(model(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
    broken = copy.deepcopy(checkpoint)
    broken["model_metadata"]["feature_schema"]["history"].reverse()
    with pytest.raises(ValueError, match="incompatible"):
        SharedStage1Ranker.from_checkpoint(broken, vectors, product_ids, "stage1 sha", "cache sha")
    with pytest.raises(ValueError, match="incompatible"):
        SharedStage1Ranker.from_checkpoint(checkpoint, vectors, list(reversed(product_ids)), "stage1 sha", "cache sha")
    broken_state = copy.deepcopy(checkpoint)
    broken_state["model_state_dict"]["token_builder.catalog_product_vectors"][0, 0] += 1
    with pytest.raises(ValueError, match="product-vector"):
        SharedStage1Ranker.from_checkpoint(broken_state, vectors, product_ids, "stage1 sha", "cache sha")
    assert feature_schema()["token_regions"]["type"] == [173, 176]


def test_width_control_uses_old_raw_values_and_only_one_encoder():
    vectors, batch = inputs(32)
    batch.update(text_embedding=torch.randn(2, 1024), **{
        name: torch.ones(2, 32, dtype=torch.long)
        for name in ("seq_product_ids", "seq_category_ids", "seq_subcategory_ids", "seq_business_line_ids")
    })
    old = UPSCANRRanker(statistics_attention=True, excluded_statistics=("history_repeat",), dropout=0).eval()
    seen = []
    hook = old.memory_encoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    old(**{key: value for key, value in batch.items() if key not in ("stage1_customer_vectors", "seq_catalog_indices", "candidate_indices")},
        candidate_vectors=vectors[batch["candidate_indices"]])
    hook.remove()
    model = SharedStage1Ranker(vectors, SharedRankerConfig(representation="legacy_width", dropout=0), legacy_input_model=old).eval()
    raw, _ = build_tokens(model.token_builder, batch)
    valid = torch.nn.functional.pad(batch["attention_mask"].bool(), (1, 0), value=True)
    torch.testing.assert_close(raw[..., :128][valid], seen[0][valid], rtol=0, atol=0)
    assert raw[..., 128:].eq(0).all()
    checkpoint = {"model_metadata": model.metadata("sha", ["a", "b", "c"], "cache"), "model_state_dict": model.state_dict()}
    restored = SharedStage1Ranker.from_checkpoint(checkpoint, vectors, ["a", "b", "c"], "sha", "cache").eval()
    torch.testing.assert_close(model(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
    assert len(model.memory_encoder.layers) == 1


def test_width_control_common_modules_match_the_shared_initialization():
    vectors, _ = inputs()
    control = SharedStage1Ranker(vectors, SharedRankerConfig(representation="legacy_width"),
                                legacy_input_model=UPSCANRRanker())
    match_width_control_initialization(control, vectors, 42)
    torch.manual_seed(42)
    shared = SharedStage1Ranker(vectors)
    for name in ("memory_encoder", "cross_attention", "context_readout", "mlp", "statistical_bias"):
        for key, value in getattr(control, name).state_dict().items():
            torch.testing.assert_close(value, getattr(shared, name).state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["schema", "row_date", "mapping", "heldout_date", "candidate_id"])
def test_shared_cache_rejects_misaligned_rows_and_incompatible_schemas(corruption):
    product_ids = [f"P{i}" for i in range(50)]
    history = [["P0", "P49", *(["[PAD]"] * 30)]]
    cache = {
        "cache_identity": "identity", "shared_cache_version": SHARED_CACHE_VERSION, "feature_schema": feature_schema(),
        "shared_vector_provenance": {"schema_digest": schema_digest(), "product_ids": product_ids, "stage1_digest": "sha"},
        "customer_ids": ["C"], "transition_keys": [("C", 1)], "prediction_days": torch.ones(1),
        "stage1_customer_vectors": torch.ones(1, 128), "sequence": torch.zeros(1, 32, 9),
        "seq_catalog_indices": map_history_skus(history, product_ids), "seq_canonical_skus": history,
        "attention_mask": torch.tensor([[1, 1, *([0] * 30)]]), "candidate_indices": torch.arange(50)[None],
        "candidate_pids": [product_ids.copy()], "pairwise_features": torch.zeros(1, 50, 13), "labels": torch.zeros(1, 50),
        "seq_purchase_dates": torch.zeros(1, 32),
    }
    validate_shared_cache(cache, "identity", product_ids, "sha")
    if corruption == "schema":
        cache["feature_schema"]["history"].reverse()
    elif corruption == "row_date":
        cache["transition_keys"] = [("C", 2)]
    elif corruption == "mapping":
        cache["seq_catalog_indices"][0, 1] = 0
    elif corruption == "candidate_id":
        cache["candidate_pids"][0][0] = "wrong"
    else:
        day = (pd.Timestamp("2026-02-19") - pd.Timestamp("2025-01-01")).days
        cache["prediction_days"] = torch.tensor([day])
        cache["transition_keys"] = [("C", day)]
    with pytest.raises(ValueError):
        validate_shared_cache(cache, "identity", product_ids, "sha")


def test_development_date_gate_does_not_parse_heldout_fields(tmp_path):
    source = tmp_path / "purchases.csv"
    source.write_text('customer_id,product_id,purchase_date,quantity\nC,P,2025-01-01,1\n'
                      'UNREAD,UNREAD,2026-02-19,DO_NOT_CONVERT\nC,P,2026-02-18,2\n')
    frame = load_development_purchases(source)
    assert frame["quantity"].tolist() == [1, 2]
    assert frame["customer_id"].tolist() == ["C", "C"]


def test_precompute_captures_same_retrieval_customer_vectors_and_expanded_history():
    history = pd.DataFrame({"customer_id": ["C"] * 3, "product_id": ["trained", "expanded", "target"],
                            "purchase_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                            "quantity": [1., 2., 3.], "product_category": ["cat"] * 3})
    vocabs = {"product_vocab": {"trained": 1}, "category_vocab": {"cat": 1},
              "subcategory_vocab": {}, "business_line_vocab": {}}
    vectors = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
    model = TwoTowerModel(CustomerTower(num_cities=2, num_regions=2, num_industries=2,
                                       catalog_history_encoder="attention", dropout=0), torch.nn.Identity(), 3)
    retriever = CandidateRetriever(model=model, eligible_product_ids=["trained", "expanded", "target"],
                                   feature_store=PointInTimeFeatureStore(history))
    normalizer = PointInTimeNormalizer().fit(pd.DataFrame({name: [1., 100.] for name in LEGACY_PROFILE_FEATURES}))
    transition = [
        {"customer_id": "C", "as_of_date": pd.Timestamp("2025-01-03"), "positive_product_ids": ["target"]},
        {"customer_id": "C", "as_of_date": pd.Timestamp("2025-01-04"), "positive_product_ids": ["trained"]},
    ]
    seen = []
    hook = model.customer_tower.register_forward_hook(lambda _m, _a, value: seen.append(value.detach()))
    kwargs = {"transitions": transition, "retriever": retriever, "product_vectors": vectors,
                  "cust_groups": {"C": history}, "customer_metadata": {}, "customer_embeddings": {}, "normalizer": normalizer,
                  "vocabs": vocabs, "item_metadata": {}, "top_c": 3, "capture_stage1_vectors": True, "stage1_checkpoint_digest": "sha"}
    cache = precompute_candidate_slates_for_transitions(**kwargs)
    hook.remove()
    torch.testing.assert_close(cache["stage1_customer_vectors"], seen[0], rtol=0, atol=0)
    assert not torch.allclose(cache["stage1_customer_vectors"][0], cache["stage1_customer_vectors"][1])
    assert cache["seq_product_ids"][0, 1] == 0  # Expanded SKU absent from old vocabulary.
    assert cache["seq_catalog_indices"][0, :3].tolist() == [0, 1, -1]
    assert cache["seq_canonical_skus"][0][:3] == ["trained", "expanded", "[PAD]"]
    kwargs["cust_groups"] = {"C": pd.concat([history, history.iloc[-1:].assign(purchase_date=pd.Timestamp("2025-01-04"), quantity=999.)])}
    future = precompute_candidate_slates_for_transitions(**kwargs)
    torch.testing.assert_close(cache["stage1_customer_vectors"], future["stage1_customer_vectors"], rtol=0, atol=0)
    for source in (cache, future):
        batch = collate_ranker_batch([RankerDataset(source)[0]])
        assert batch["stage1_customer_vectors"].shape == (1, 128)
        assert batch["seq_catalog_indices"].shape == (1, 32)
