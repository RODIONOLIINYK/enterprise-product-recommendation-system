"""Synthetic unit and adversarial tests for the static enrichment campaign."""

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.static_enrichment.config import (
    VariantSpec,
    build_clean_v2_variant_specs,
    build_variant_specs,
)
from recommender.experiments.static_enrichment.data import (
    extract_point_in_time_customer_stats,
    load_dynamic_campaign_features_readonly,
    prepare_static_enrichment_bundle,
)
from recommender.experiments.static_enrichment.evaluate import paired_stage1_intervals, stage1_cohort_metrics
from recommender.experiments.static_enrichment.train import (
    make_variant_model,
    run_downstream_for_stage1,
    train_stage1_variant,
)
from recommender.experiments.static_enrichment.metadata import (
    NONE_INDEX,
    OOV_INDEX,
    clean_html_description,
    encode_categorical_values,
    fit_categorical_vocab,
    load_cleaned_descriptions,
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.experiments.static_enrichment.model import (
    PrecomputedEmbeddingMLPRanker,
    StaticEnrichedRetriever,
    TransferredStatsDynamicRanker,
)
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.ranker import UPSCANRRanker
from recommender.experiments.static_enrichment.text_cache import (
    build_product_text,
    compute_text_recipe_hash,
    get_or_build_product_text_embeddings,
)


def test_clean_html_description_strips_tags_entities_null_and_preserves_unicode():
    raw = '<h3><strong>Моторна олива&nbsp;Shell</strong></h3><p>🔧&nbsp;Helix Ultra &amp; Protection</p>'
    cleaned = clean_html_description(raw)
    assert "<" not in cleaned and ">" not in cleaned
    assert "&nbsp;" not in cleaned and "&amp;" not in cleaned
    assert "Моторна олива Shell" in cleaned
    assert "🔧 Helix Ultra & Protection" in cleaned
    assert clean_html_description("NULL") == ""
    assert clean_html_description("   ") == ""
    assert clean_html_description(None) == ""


def test_training_only_vocab_fitting_and_oov_none_buckets():
    train_values = ["SHELL", "SHELL", "SHELL", "PALL", "PALL", None, "NULL", "  ", "rare"]
    vocab = fit_categorical_vocab(train_values, min_count=3)
    assert vocab["<NONE>"] == NONE_INDEX == 0
    assert vocab["<OOV>"] == OOV_INDEX == 1
    assert "SHELL" in vocab and vocab["SHELL"] == 2
    assert "PALL" not in vocab and "rare" not in vocab

    encoded = encode_categorical_values(["shell", None, "NULL", "PALL", "UNSEEN_VAL"], vocab)
    assert encoded == [2, 0, 0, 1, 1]


def test_text_recipe_deduplication_descriptions_and_cache_hash(tmp_path):
    row = {
        "product_id": "P1",
        "brand": "SHELL",
        "product_name": "Масло моторное SHELL Helix Ultra 5W-40, 4л",
        "base_name": "Масло моторное Helix Ultra 5W-40",
        "product_type": "Масло моторное",
        "product_category": "АВТО-СМ",
        "product_subcategory": "АВТ-ЛЕГКОМ",
        "business_line": "PCMO",
        "product_family": "ULTRA",
        "cleaned_description": "Високотехнологічна моторна олива для сучасних двигунів.",
    }
    t1 = build_product_text(row, "t1")
    assert t1 == "SHELL Масло моторное Helix Ultra 5W-40, 4л АВТО-СМ АВТ-ЛЕГКОМ PCMO"
    t2 = build_product_text(row, "t2")
    assert t2 == f"{t1} Високотехнологічна моторна олива для сучасних двигунів."

    class DummyEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
            self.calls += 1
            return np.ones((len(texts), 8), dtype=np.float32)

    df = pd.DataFrame([row])
    enc = DummyEncoder()
    v1, h1, p1 = get_or_build_product_text_embeddings(
        df, ["P1"], "t2", cache_dir=tmp_path, encoder=enc, embedding_dim=8
    )
    assert p1.is_file() and enc.calls == 1 and v1.shape == (1, 8)
    # Second call loads from parquet cache without invoking encoder.
    v2, h2, _ = get_or_build_product_text_embeddings(
        df, ["P1"], "t2", cache_dir=tmp_path, encoder=enc, embedding_dim=8
    )
    assert enc.calls == 1 and h1 == h2
    torch.testing.assert_close(v1, v2)
    # Different recipe produces different hash and separate cache file.
    _, h_t1, p_t1 = get_or_build_product_text_embeddings(
        df, ["P1"], "t1", cache_dir=tmp_path, encoder=enc, embedding_dim=8
    )
    assert h_t1 != h1 and p_t1 != p1 and enc.calls == 2


def test_enriched_synthetic_loaders_ignore_price_and_customs(tmp_path):
    item_row = [""] * 169
    item_row[1] = "SKU-1"
    item_row[5] = "Масло моторное"
    item_row[6] = "Л"
    item_row[46] = "0"
    item_row[47] = "2025-01-01"
    item_row[50] = "ТОВАР"
    item_row[52] = "DEU"
    item_row[59] = "PCMO"
    item_row[83] = "SHELL"
    item_row[84] = "АВТО-СМ"
    item_row[86] = "АВТ-ЛЕГКОМ"
    item_row[98] = "2710198100"
    item_row[125] = "HELIX"
    item_row[127] = "PREMIUM"
    item_row[132] = "Shell Helix 4л"
    item_row[133] = "Shell Helix"
    items_path = tmp_path / "items.csv"
    pd.DataFrame([item_row]).to_csv(items_path, sep=";", header=False, index=False)

    desc_path = tmp_path / "desc.csv"
    pd.DataFrame([["Helix", "SKU-1", "<p>Clean&nbsp;text</p>", "9999.00", "4.0"]]).to_csv(
        desc_path, sep=";", header=False, index=False
    )

    cat = load_enriched_catalogue(items_path, desc_path)
    assert cat["tier"].iloc[0] == "PREMIUM"
    assert cat["country_of_origin"].iloc[0] == "DEU"
    assert cat["cleaned_description"].iloc[0] == "Clean text"
    assert "9999.00" not in cat.to_string()
    assert "2710198100" not in cat.to_string()

    cust_row = [""] * 118
    cust_row[1] = "CUST-1"
    cust_row[2] = "Acme"
    cust_row[44] = "КИЕВ"
    cust_row[59] = "К_ПЛАТ"
    cust_row[82] = "Acme LLC"
    cust_row[85] = "UA_01"
    cust_row[88] = "м. Київ"
    cust_row[90] = "Auto"
    cust_row[94] = "T1"
    cust_path = tmp_path / "customers.csv"
    pd.DataFrame([cust_row]).to_csv(cust_path, sep=";", header=False, index=False)

    purchases = pd.DataFrame([{
        "customer_id": "CUST-1",
        "product_id": "SKU-1",
        "purchase_date": pd.Timestamp("2025-05-01"),
        "quantity": 1.0,
        "transaction_type": "ПРОДАЖА",
        "item_type": "ТОВАР",
    }])
    comp = load_enriched_companies(cust_path, purchases=purchases, minimum_purchases=1)
    assert comp["sales_branch"].iloc[0] == "КИЕВ"
    assert comp["vat_status"].iloc[0] == "К_ПЛАТ"


def test_enriched_model_enforces_zero_demand_and_gradients():
    specs = build_variant_specs()
    u1_spec = specs["U1"]
    p_sizes = {name: 5 for name in u1_spec.product_embeddings}
    c_sizes = {name: 4 for name in u1_spec.customer_embeddings}
    model = StaticEnrichedRetriever(
        u1_spec,
        num_cities=10,
        num_regions=6,
        num_industries=6,
        num_catalog_products=4,
        num_demand_features=18,
        product_attr_vocab_sizes=p_sizes,
        customer_attr_vocab_sizes=c_sizes,
        text_dim=16,
        output_dim=8,
        hidden_dim=16,
    )
    prod_inputs = {
        "text_embedding": torch.randn(4, 16),
        "demand_features": torch.cat([torch.zeros(4, 6), torch.ones(4, 12)], dim=1),
        "attr_ids": {name: torch.tensor([0, 1, 2, 3]) for name in u1_spec.product_embeddings},
    }
    mutated_prod = dict(prod_inputs)
    mutated_prod["demand_features"] = prod_inputs["demand_features"].clone()
    mutated_prod["demand_features"][:, :6] = 123.45

    model.eval()
    v1 = model.product_vectors(prod_inputs)
    v2 = model.product_vectors(mutated_prod)
    torch.testing.assert_close(v1, v2, atol=0.0, rtol=0.0)

    model.train()
    vecs = model.product_vectors(prod_inputs)
    cust_inputs = {
        "text_embedding": torch.randn(2, 16),
        "city_id": torch.zeros(2, dtype=torch.long),
        "region_id": torch.zeros(2, dtype=torch.long),
        "industry_id": torch.zeros(2, dtype=torch.long),
        "tabular_features": torch.rand(2, 6),
        "sequence": torch.rand(2, 4, 9),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
        "seq_product_ids": torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]]),
        "seq_purchase_dates": torch.zeros(2, 4),
        "extra_attr_ids": {name: torch.tensor([0, 1]) for name in u1_spec.customer_embeddings},
    }
    u = model.encode_purchase(cust_inputs, vecs)
    logits = model.score_cached(u, vecs, torch.arange(4), torch.zeros(2, 4, 4))
    logits.sum().backward()
    for name in u1_spec.product_embeddings:
        grad = model.product_tower.attr_embeddings[name].weight.grad
        assert grad is not None and grad[0].abs().sum() > 0 and grad[1].abs().sum() > 0
    for name in u1_spec.customer_embeddings:
        grad = model.customer_tower.extra_attr_embeddings[name].weight.grad
        assert grad is not None and grad[0].abs().sum() > 0 and grad[1].abs().sum() > 0


def test_stage1_cohort_metrics_and_paired_intervals(tmp_path):
    val_data = {
        "unseen": torch.tensor([[True, True, True, False], [True, True, True, True]]),
        "new_target": torch.tensor([[True, False, False, False], [False, True, True, False]]),
    }
    val_source = {
        "customer_ids": ["c1", "c2"],
        "new_positive_pids": [["p0"], ["p1", "p2"]],
    }
    cohort_masks = {
        "has_description": torch.tensor([True, False, True, False]),
        "in_description_file": torch.tensor([True, False, True, False]),
        "brand_missing": torch.tensor([False, True, False, False]),
        "family_missing": torch.tensor([False, False, True, False]),
    }
    logits = torch.tensor([[5.0, 1.0, 0.0, 9.0], [0.0, 4.0, 3.0, 1.0]])
    cohorts = stage1_cohort_metrics(logits, val_data, val_source, cohort_masks, save_predictions=tmp_path / "p.npz")
    assert cohorts["with_description_nonempty"]["targets"] == 2
    assert cohorts["with_description_nonempty"]["hits@25"] == 2
    assert cohorts["brand_missing"]["targets"] == 1
    assert cohorts["family_missing"]["targets"] == 1

    ci = paired_stage1_intervals([tmp_path / "p.npz"], [tmp_path / "p.npz"], repeats=20)
    assert all(vals == [0.0, 0.0, 0.0] for vals in ci.values())


def test_clean_v2_towers_delete_changeable_customer_and_product_inputs():
    clean_specs = build_clean_v2_variant_specs()
    assert set(clean_specs.keys()) == {
        "C0_clean",
        "P_t1_fam_clean",
        "U0_t1_fam_clean",
        "U0_t1_fam_nosubcat",
    }
    assert "subcategory" in clean_specs["U0_t1_fam_clean"].product_embeddings
    assert "subcategory" not in clean_specs["U0_t1_fam_nosubcat"].product_embeddings

    for name in ("C0_clean", "U0_t1_fam_clean"):
        spec = clean_specs[name]
        p_sizes = {k: 5 for k in spec.product_embeddings}
        c_sizes = {k: 4 for k in spec.customer_embeddings}
        model = StaticEnrichedRetriever(
            spec,
            num_cities=10,
            num_regions=6,
            num_industries=6,
            num_catalog_products=4,
            num_demand_features=spec.product_demand_features,
            product_attr_vocab_sizes=p_sizes,
            customer_attr_vocab_sizes=c_sizes,
            text_dim=16,
            output_dim=8,
            hidden_dim=16,
        )
        cust_tower = model.towers.customer_tower if spec.is_control else model.customer_tower
        prod_tower = model.towers.product_tower if spec.is_control else model.product_tower
        assert cust_tower.num_tabular_features == 0
        expected_tab_in = 48 + (16 if "sales_branch" in spec.customer_embeddings else 0)
        assert cust_tower.tabular_proj[0].in_features == expected_tab_in
        assert prod_tower.num_demand_features == 12
        assert prod_tower.mlp[0].in_features == 16 + 16 * len(spec.product_embeddings) + 12

        prod_base = {
            "text_embedding": torch.randn(4, 16),
            "brand_id": torch.zeros(4, dtype=torch.long),
            "category_id": torch.zeros(4, dtype=torch.long),
            "subcategory_id": torch.zeros(4, dtype=torch.long),
            "business_line_id": torch.zeros(4, dtype=torch.long),
            "demand_features": torch.cat([torch.zeros(4, 6), torch.eye(12)[:4]], dim=1),
            "attr_ids": {k: torch.zeros(4, dtype=torch.long) for k in spec.product_embeddings},
        }
        prod_mut = dict(prod_base)
        prod_mut["demand_features"] = prod_base["demand_features"].clone()
        prod_mut["demand_features"][:, :6] = 999.0

        model.eval()
        v1 = model.product_vectors(prod_base)
        v2 = model.product_vectors(prod_mut)
        torch.testing.assert_close(v1, v2, atol=0.0, rtol=0.0)

        cust_base = {
            "text_embedding": torch.randn(2, 16),
            "city_id": torch.zeros(2, dtype=torch.long),
            "region_id": torch.zeros(2, dtype=torch.long),
            "industry_id": torch.zeros(2, dtype=torch.long),
            "tabular_features": torch.zeros(2, 6),
            "sequence": torch.rand(2, 4, 9),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
            "seq_product_ids": torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]]),
            "seq_purchase_dates": torch.zeros(2, 4),
            "extra_attr_ids": {k: torch.zeros(2, dtype=torch.long) for k in spec.customer_embeddings},
        }
        cust_mut = dict(cust_base)
        cust_mut["tabular_features"] = torch.full((2, 6), 999.0)
        u1 = model.encode_purchase(cust_base, v1)
        u2 = model.encode_purchase(cust_mut, v1)
        torch.testing.assert_close(u1, u2, atol=0.0, rtol=0.0)


def test_point_in_time_customer_stats_and_47_feature_rankers():
    purchases = pd.DataFrame([
        {
            "customer_id": "C1",
            "product_id": "P1",
            "purchase_date": pd.Timestamp("2025-02-01"),
            "quantity": 10.0,
            "product_category": "CAT_A",
        },
        {
            "customer_id": "C1",
            "product_id": "P2",
            "purchase_date": pd.Timestamp("2025-04-01"),
            "quantity": 20.0,
            "product_category": "CAT_B",
        },
        {
            "customer_id": "C1",
            "product_id": "P3",
            "purchase_date": pd.Timestamp("2025-11-01"),
            "quantity": 5000.0,
            "product_category": "CAT_C",
        },
    ])
    train_queries = [("C1", pd.Timestamp("2025-03-01"))]
    val_queries = [("C1", pd.Timestamp("2025-11-01")), ("C1", pd.Timestamp("2025-12-01"))]
    tr_stats, val_stats, normalizer = extract_point_in_time_customer_stats(
        purchases, train_queries, val_queries
    )
    assert tr_stats.shape == (1, 6) and val_stats.shape == (2, 6)
    # Normalizer fitted strictly on train_queries (1 prior order, max=1, max_qty=10):
    assert normalizer.scalers["distinct_orders"].max_val == 1.0
    assert normalizer.scalers["total_quantity"].max_val == 10.0
    # 2025-11-01 purchase is strictly excluded on 2025-11-01 (only 2 prior orders), included on 2025-12-01.
    assert val_stats[0, 0] <= val_stats[1, 0] and torch.isfinite(val_stats).all()

    base = UPSCANRRanker(
        text_dim=16,
        memory_dim=16,
        hidden_dim=16,
        num_cities=5,
        num_regions=5,
        num_industries=5,
        num_products=10,
        num_categories=5,
        num_subcategories=5,
        num_business_lines=5,
        statistics_attention=True,
        excluded_statistics=("history_repeat",),
        cached_only=True,
    )
    dyn_41 = CandidateContextRanker(base, mode="dynamic_features")
    dyn_47 = TransferredStatsDynamicRanker(dyn_41, num_customer_stats=6)
    assert dyn_47.ranker.head[0].in_features == 16 * 3 + 41 + 6

    B, C = 2, 5
    common = {
        "text_embedding": torch.randn(B, 16),
        "city_id": torch.zeros(B, dtype=torch.long),
        "region_id": torch.zeros(B, dtype=torch.long),
        "industry_id": torch.zeros(B, dtype=torch.long),
        "tabular_features": torch.rand(B, 8),
        "sequence": torch.rand(B, 4, 9),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
        "seq_product_ids": torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]]),
        "seq_category_ids": torch.zeros(B, 4, dtype=torch.long),
        "seq_subcategory_ids": torch.zeros(B, 4, dtype=torch.long),
        "seq_business_line_ids": torch.zeros(B, 4, dtype=torch.long),
        "seq_purchase_dates": torch.zeros(B, 4),
        "prediction_day": torch.tensor([100.0, 120.0]),
        "candidate_vectors": torch.randn(B, C, 16),
        "pairwise_features": torch.rand(B, C, 41),
        "cached_customer_vectors": torch.randn(B, 16),
        "transferred_customer_stats": torch.rand(B, 6),
    }
    out_47 = dyn_47(**common)
    assert out_47.scores.shape == (B, C) and torch.isfinite(out_47.scores).all()
    out_47.scores.sum().backward()

    mlp_47 = PrecomputedEmbeddingMLPRanker(
        embedding_dim=16, num_pairwise_features=41, num_customer_stats=6
    )
    assert mlp_47.head[0].in_features == 16 * 4 + 1 + 41 + 6
    out_mlp = mlp_47(
        candidate_vectors=common["candidate_vectors"],
        pairwise_features=common["pairwise_features"],
        cached_customer_vectors=common["cached_customer_vectors"],
        transferred_customer_stats=common["transferred_customer_stats"],
    )
    assert out_mlp.scores.shape == (B, C) and out_mlp.attn_weights is None
    out_mlp.scores.sum().backward()
