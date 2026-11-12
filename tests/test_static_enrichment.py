"""Synthetic unit and adversarial tests for the static enrichment campaign."""

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.static_enrichment.config import VariantSpec, build_variant_specs
from recommender.experiments.static_enrichment.data import (
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
from recommender.experiments.static_enrichment.model import StaticEnrichedRetriever
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
