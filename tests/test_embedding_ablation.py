"""Synthetic tests for isolated U0_t1_fam_clean text and product-embedding ablations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.dynamic_selector.evaluate import paired_intervals
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES
from recommender.experiments.embedding_ablation.config import (
    ABLATED_PRODUCT_EMBEDDINGS,
    ABLATED_TEXT_FIELDS,
    BASELINE_PRODUCT_EMBEDDINGS,
    INAPPLICABLE_EMBEDDING_ABLATIONS,
    INAPPLICABLE_TEXT_ABLATIONS,
    build_ablation_specs,
)
from recommender.experiments.embedding_ablation.evaluate import (
    paired_ablation_intervals,
    save_mlp_per_query_predictions,
)
from recommender.experiments.embedding_ablation.text_cache import (
    build_ablated_product_text,
    get_or_build_ablated_text_embeddings,
)
import recommender.experiments.embedding_ablation.train as ea_train
import recommender.experiments.static_enrichment.train as se_train
from recommender.experiments.static_enrichment.config import build_clean_v2_variant_specs
from recommender.experiments.static_enrichment.evaluate import paired_stage1_intervals
from recommender.experiments.static_enrichment.model import (
    CompactStage1Selector,
    StaticEnrichedRetriever,
)
from recommender.experiments.static_enrichment.text_cache import (
    build_product_text,
    compute_text_recipe_hash,
)


def test_exact_field_omission_dedup_and_unchanged_fallback_names():
    row = {
        "product_id": "P1",
        "brand": "SHELL",
        "product_name": "Масло моторное SHELL Helix Ultra 5W-40, 4л",
        "base_name": "Масло моторное Helix Ultra 5W-40",
        "product_type": "Масло моторное",
        "product_category": "АВТО-СМ",
        "product_subcategory": "АВТ-ЛЕГКОМ",
        "business_line": "PCMO",
        "product_family": "ULTRA_FAM",
        "cleaned_description": "Довгий опис товару який не повинен потрапити в t1_fam.",
    }
    expected_base = build_product_text(row, "t1_fam")
    assert build_ablated_product_text(row, omit_fields=()) == expected_base
    assert "Довгий опис" not in expected_base

    # Omitting brand removes the leading structured brand field, NOT "SHELL" inside product_name.
    no_brand = build_ablated_product_text(row, omit_fields=("brand",))
    assert no_brand == "Масло моторное SHELL Helix Ultra 5W-40, 4л АВТО-СМ АВТ-ЛЕГКОМ PCMO ULTRA_FAM"
    assert "SHELL" in no_brand

    # Omitting category/subcategory/business_line/product_family removes only those structured tokens.
    assert "АВТО-СМ" not in build_ablated_product_text(row, omit_fields=("product_category",))
    assert "АВТ-ЛЕГКОМ" not in build_ablated_product_text(row, omit_fields=("product_subcategory",))
    assert "PCMO" not in build_ablated_product_text(row, omit_fields=("business_line",))
    assert "ULTRA_FAM" not in build_ablated_product_text(row, omit_fields=("product_family",))

    # Shared words inside product_name are NEVER stripped when a field sharing words is omitted.
    row_shared = {
        "product_id": "P2",
        "brand": "MOBIL",
        "product_name": "Олива АВТО-СМ Mobil 1 5W-30",
        "base_name": "Mobilgear UniqueBaseToken",
        "product_type": "Олива",
        "product_category": "АВТО-СМ",
        "product_subcategory": "SUB_UNIQ",
        "business_line": "PVL",
        "product_family": "MOBIL1",
    }
    no_cat_shared = build_ablated_product_text(row_shared, omit_fields=("product_category",))
    assert "АВТО-СМ" in no_cat_shared  # preserved because it came from product_name
    no_base = build_ablated_product_text(row_shared, omit_fields=("base_name",))
    assert "UniqueBaseToken" not in no_base and "5W-30" in no_base and "MOBIL" in no_base
    assert "Mobil 1" in build_ablated_product_text(row_shared, omit_fields=("brand",))

    # Fallback product name ordering (product_name -> ukrainian_name -> short_name) stays intact.
    row_uk = {
        "product_id": "P3",
        "brand": "CASTROL",
        "product_name": None,
        "ukrainian_name": "Олива моторна Castrol EDGE",
        "short_name": "EDGE",
        "product_category": "АВТО-СМ",
    }
    assert build_ablated_product_text(row_uk, omit_fields=("brand",)) == "Олива моторна Castrol EDGE АВТО-СМ"
    row_short = {
        "product_id": "P4",
        "brand": "CASTROL",
        "product_name": "NULL",
        "ukrainian_name": "",
        "short_name": "SHORT_EDGE",
        "product_category": "АВТО-СМ",
    }
    assert build_ablated_product_text(row_short, omit_fields=("product_category",)) == "CASTROL SHORT_EDGE"

    with pytest.raises(ValueError, match="Unsupported text field omission"):
        build_ablated_product_text(row, omit_fields=("unit",))


def test_noop_text_detection_and_cache_reuse(tmp_path):
    # Construct a catalogue where base_name is 100% redundant with product_name -> exact no-op.
    df = pd.DataFrame([
        {
            "product_id": "P1",
            "brand": "SHELL",
            "product_name": "Shell Helix Ultra 5W-40",
            "base_name": "Helix Ultra",
            "product_type": "Oil",
            "product_category": "CAT1",
            "product_subcategory": "SUB1",
            "business_line": "PCMO",
            "product_family": "FAM1",
        },
        {
            "product_id": "P2",
            "brand": "MOBIL",
            "product_name": "Mobil Super 3000",
            "base_name": "Mobil Super",
            "product_type": "Oil",
            "product_category": "CAT2",
            "product_subcategory": "SUB2",
            "business_line": "CVL",
            "product_family": "FAM2",
        },
    ])

    class DummyEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
            self.calls += 1
            return np.stack([np.full(8, float(i + 1), dtype=np.float32) for i in range(len(texts))])

    enc = DummyEncoder()
    ref_dir = tmp_path / "ref_text"
    cache_dir = tmp_path / "ablation_text"

    base_vec, base_diag = get_or_build_ablated_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam",
        trained_count=1,
        cache_dir=ref_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=None,
    )
    assert enc.calls == 1
    assert base_diag["changed_text_count"] == 0
    assert base_diag["is_noop"] is False  # baseline itself is not marked no-op

    # Omitting base_name here changes 0 strings -> exact no-op, reuses ref_dir without calling encoder!
    noop_vec, noop_diag = get_or_build_ablated_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam_no_base_name",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
    )
    assert enc.calls == 1  # NOT called again!
    assert noop_diag["changed_text_count"] == 0
    assert noop_diag["changed_trained_text_count"] == 0
    assert noop_diag["is_noop"] is True
    assert noop_diag["content_hash"] == base_diag["content_hash"]
    torch.testing.assert_close(base_vec, noop_vec)

    # Omitting product_family changes both P1 and P2 -> encodes once, then hits cache on repeat.
    fam_vec1, fam_diag = get_or_build_ablated_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam_no_family",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
    )
    assert enc.calls == 2
    assert fam_diag["changed_text_count"] == 2
    assert fam_diag["changed_trained_text_count"] == 1
    assert fam_diag["empty_text_count"] == 0
    assert fam_diag["is_noop"] is False
    assert fam_diag["content_hash"] != base_diag["content_hash"]

    fam_vec2, _ = get_or_build_ablated_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam_no_family",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
    )
    assert enc.calls == 2
    torch.testing.assert_close(fam_vec1, fam_vec2)


def test_attribute_deletion_shapes_and_inapplicable_rules():
    specs = build_ablation_specs()
    assert len(specs) == 1 + len(ABLATED_TEXT_FIELDS) + len(ABLATED_PRODUCT_EMBEDDINGS)
    assert "unit" in INAPPLICABLE_TEXT_ABLATIONS
    assert "product_family" in INAPPLICABLE_EMBEDDING_ABLATIONS
    assert "product_family" not in BASELINE_PRODUCT_EMBEDDINGS

    base_spec = specs["U0_t1_fam_clean"].variant_spec
    assert base_spec.product_embeddings == BASELINE_PRODUCT_EMBEDDINGS
    assert base_spec.customer_embeddings == ("sales_branch",)
    assert base_spec.customer_tabular_features == 0
    assert base_spec.product_demand_features == 12

    for attr in ABLATED_PRODUCT_EMBEDDINGS:
        vname = f"E_no_{attr}"
        sp = specs[vname].variant_spec
        assert sp.text_recipe == "t1_fam"
        assert attr not in sp.product_embeddings
        assert len(sp.product_embeddings) == 6
        model = StaticEnrichedRetriever(
            sp,
            num_cities=8,
            num_regions=4,
            num_industries=4,
            num_catalog_products=5,
            num_demand_features=12,
            product_attr_vocab_sizes={k: 6 for k in BASELINE_PRODUCT_EMBEDDINGS},
            customer_attr_vocab_sizes={"sales_branch": 4},
            text_dim=16,
            output_dim=8,
            hidden_dim=16,
        )
        assert attr not in model.product_tower.attr_embeddings
        assert len(model.product_tower.attr_embeddings) == 6
        assert model.product_tower.mlp[0].in_features == 16 + 16 * 6 + 12
        assert model.customer_tower.num_tabular_features == 0
        assert model.customer_tower.tabular_proj[0].in_features == 48 + 16


def test_original_default_behavior_preserved():
    v2 = build_clean_v2_variant_specs()["U0_t1_fam_clean"]
    ab = build_ablation_specs()["U0_t1_fam_clean"].variant_spec
    assert v2 == ab

    sel = CompactStage1Selector(recipe="uv_47")
    assert sel.input_dim == 303
    assert sel.hidden_dims == (128, 64)
    assert len(DYNAMIC_FEATURE_NAMES) == 41


def test_customer_history_vector_consistency():
    specs = build_ablation_specs()
    sp = specs["E_no_unit"].variant_spec
    model = StaticEnrichedRetriever(
        sp,
        num_cities=6,
        num_regions=4,
        num_industries=4,
        num_catalog_products=4,
        num_demand_features=12,
        product_attr_vocab_sizes={k: 5 for k in BASELINE_PRODUCT_EMBEDDINGS},
        customer_attr_vocab_sizes={"sales_branch": 4},
        text_dim=16,
        output_dim=8,
        hidden_dim=16,
    )
    model.eval()
    prod_a = {
        "text_embedding": torch.randn(4, 16),
        "demand_features": torch.cat([torch.zeros(4, 6), torch.eye(12)[:4]], dim=1),
        "attr_ids": {k: torch.zeros(4, dtype=torch.long) for k in sp.product_embeddings},
    }
    prod_b = dict(prod_a)
    prod_b["text_embedding"] = prod_a["text_embedding"].clone()
    prod_b["text_embedding"][1] += 2.0  # change product index 1 (1-based ID 2 in seq_product_ids)

    vecs_a = model.product_vectors(prod_a)
    vecs_b = model.product_vectors(prod_b)
    assert not torch.allclose(vecs_a[1], vecs_b[1])

    cust_with_p1 = {
        "text_embedding": torch.randn(1, 16),
        "city_id": torch.zeros(1, dtype=torch.long),
        "region_id": torch.zeros(1, dtype=torch.long),
        "industry_id": torch.zeros(1, dtype=torch.long),
        "tabular_features": torch.zeros(1, 6),
        "sequence": torch.rand(1, 3, 9),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "seq_product_ids": torch.tensor([[1, 2, 0]]),  # references product index 1!
        "seq_purchase_dates": torch.tensor([[10.0, 20.0, 0.0]]),
        "extra_attr_ids": {"sales_branch": torch.zeros(1, dtype=torch.long)},
    }
    u_a1 = model.encode_purchase(cust_with_p1, vecs_a)
    u_a2 = model.encode_purchase(cust_with_p1, vecs_a)
    u_b = model.encode_purchase(cust_with_p1, vecs_b)
    torch.testing.assert_close(u_a1, u_a2)
    assert not torch.allclose(u_a1, u_b)


def test_stage1_only_orchestration_and_namespace_provenance(tmp_path, monkeypatch):
    # Synthetic Stage 1 + uv_47 CompactStage1Selector orchestration with zero Stage 2 calls.
    torch.manual_seed(7)
    n_train, n_val, n_prod = 12, 6, 105
    n_trained = 90

    def _make_split(n_rows: int, seed_offset: int):
        g = torch.Generator().manual_seed(100 + seed_offset)
        allowed = torch.ones((n_rows, n_prod), dtype=torch.bool)
        unseen = torch.ones((n_rows, n_prod), dtype=torch.bool)
        unseen[:, :10] = False
        target = torch.zeros((n_rows, n_prod), dtype=torch.bool)
        new_target = torch.zeros((n_rows, n_prod), dtype=torch.bool)
        pos_pids, new_pids = [], []
        for r in range(n_rows):
            t_col = 15 + (r % 20)
            target[r, t_col] = True
            new_target[r, t_col] = True
            pos_pids.append([f"SKU-{t_col}"])
            new_pids.append([f"SKU-{t_col}"])
        ret = {
            "text_embedding": torch.randn(n_rows, 16, generator=g),
            "city_id": torch.zeros(n_rows, dtype=torch.long),
            "region_id": torch.zeros(n_rows, dtype=torch.long),
            "industry_id": torch.zeros(n_rows, dtype=torch.long),
            "tabular_features": torch.rand(n_rows, 6, generator=g),
            "sequence": torch.rand(n_rows, 4, 9, generator=g),
            "attention_mask": torch.tensor([[1, 1, 0, 0]] * n_rows),
            "seq_product_ids": torch.tensor([[1, 2, 0, 0]] * n_rows),
            "seq_purchase_dates": torch.tensor([[10.0, 20.0, 0.0, 0.0]] * n_rows),
            "familiarity": torch.zeros(n_rows, n_prod, 4),
            "target": target,
            "new_target": new_target,
            "allowed": allowed,
            "unseen": unseen,
        }
        src = {
            "customer_ids": [f"C-{r % 3}" for r in range(n_rows)],
            "positive_pids": pos_pids,
            "new_positive_pids": new_pids,
            "retriever_tabular_features": torch.rand(n_rows, 6, generator=g),
            "attention_mask": ret["attention_mask"],
            "prediction_days": torch.full((n_rows,), 300, dtype=torch.long),
            "city_id": ret["city_id"],
            "region_id": ret["region_id"],
            "industry_id": ret["industry_id"],
            "tabular_features": torch.rand(n_rows, 8, generator=g),
            "sequence": ret["sequence"],
            "seq_product_ids": ret["seq_product_ids"],
            "seq_category_ids": torch.zeros(n_rows, 4, dtype=torch.long),
            "seq_subcategory_ids": torch.zeros(n_rows, 4, dtype=torch.long),
            "seq_business_line_ids": torch.zeros(n_rows, 4, dtype=torch.long),
            "seq_purchase_dates": ret["seq_purchase_dates"],
            "text_embedding": ret["text_embedding"],
        }
        return ret, src

    train_ret, train_src = _make_split(n_train, 1)
    val_ret, val_src = _make_split(n_val, 2)
    product_ids = [f"SKU-{i}" for i in range(n_prod)]
    info = {
        "identity": "synthetic-stage1-only-test",
        "product_ids": product_ids,
        "trained_products": n_trained,
        "product_inputs": {
            "text_embedding": torch.randn(n_prod, 16),
            "demand_features": torch.cat([torch.zeros(n_prod, 6), torch.ones(n_prod, 12)], dim=1),
            "brand_id": torch.zeros(n_prod, dtype=torch.long),
            "category_id": torch.zeros(n_prod, dtype=torch.long),
            "subcategory_id": torch.zeros(n_prod, dtype=torch.long),
            "business_line_id": torch.zeros(n_prod, dtype=torch.long),
        },
    }
    enrichment = {
        "identity": "synthetic-enrichment-v1",
        "product_vocabs": {k: {"<NONE>": 0, "<OOV>": 1, "A": 2} for k in BASELINE_PRODUCT_EMBEDDINGS},
        "customer_vocabs": {"sales_branch": {"<NONE>": 0, "<OOV>": 1, "B1": 2}},
        "product_attr_ids": {k: torch.zeros(n_prod, dtype=torch.long) for k in BASELINE_PRODUCT_EMBEDDINGS},
        "train_customer_attrs": {"sales_branch": torch.zeros(n_train, dtype=torch.long)},
        "val_customer_attrs": {"sales_branch": torch.zeros(n_val, dtype=torch.long)},
        "cohort_masks": {
            "has_description": torch.zeros(n_prod, dtype=torch.bool),
            "in_description_file": torch.zeros(n_prod, dtype=torch.bool),
            "brand_missing": torch.zeros(n_prod, dtype=torch.bool),
            "family_missing": torch.zeros(n_prod, dtype=torch.bool),
        },
        "text_embeddings": {
            "t1_fam": torch.randn(n_prod, 16),
            "t1_fam_no_base_name": torch.randn(n_prod, 16),
            "t1_fam_no_brand": torch.randn(n_prod, 16),
        },
        "text_recipe_hashes": {
            "t1_fam": "h_base",
            "t1_fam_no_base_name": "h_noop",
            "t1_fam_no_brand": "h_nobrand",
        },
        "diagnostics": {
            "text_recipes": {
                "t1_fam": {"is_noop": False, "changed_text_count": 0, "empty_text_count": 0},
                "t1_fam_no_base_name": {"is_noop": True, "changed_text_count": 0, "empty_text_count": 0},
                "t1_fam_no_brand": {"is_noop": False, "changed_text_count": 50, "empty_text_count": 0},
            }
        },
    }

    fake_ref = tmp_path / "ref_retriever.pt"
    torch.save({
        "model_state_dict": {
            "customer_tower.city_embedding.weight": torch.zeros(5, 16),
            "customer_tower.region_embedding.weight": torch.zeros(4, 16),
            "customer_tower.industry_embedding.weight": torch.zeros(4, 16),
        }
    }, fake_ref)
    monkeypatch.setattr(se_train, "DEFAULT_RETRIEVER", fake_ref)

    # Patch StaticEnrichedRetriever & CompactStage1Selector dimensions to small synthetic 16D/8D?
    # Better: keep 128D vectors (16D text -> 128D output) via real StaticEnrichedRetriever defaults!
    # Wait: StaticEnrichedRetriever defaults to text_dim=1024, output_dim=128 -> adapt synthetic text to 1024D.
    info["product_inputs"]["text_embedding"] = torch.randn(n_prod, 1024)
    train_ret["text_embedding"] = torch.randn(n_train, 1024)
    val_ret["text_embedding"] = torch.randn(n_val, 1024)
    train_src["text_embedding"] = train_ret["text_embedding"]
    val_src["text_embedding"] = val_ret["text_embedding"]
    enrichment["text_embeddings"] = {k: torch.randn(n_prod, 1024) for k in enrichment["text_embeddings"]}

    f41_tr = np.random.default_rng(1).random((n_train, n_prod, 41), dtype=np.float32).astype(np.float16)
    f41_va = np.random.default_rng(2).random((n_val, n_prod, 41), dtype=np.float32).astype(np.float16)
    dyn_manifest = {"identity": "synthetic-dyn", "source_identity": info["identity"]}
    monkeypatch.setattr(
        se_train,
        "load_dynamic_campaign_features_readonly",
        lambda: (dyn_manifest, f41_tr, f41_va),
    )

    specs = build_ablation_specs()
    out_root = tmp_path / "ablation_ckpts"

    # 1. Exact no-op text ablation skips training completely.
    res_noop = ea_train.run_ablation_variant(
        specs["T_no_base_name"],
        42,
        train_ret,
        val_ret,
        info,
        train_src,
        val_src,
        enrichment,
        out_root,
        stage1_epochs=2,
        selector_epochs=2,
        reuse_clean_v2_baseline=False,
        device="cpu",
    )
    assert res_noop["is_noop"] is True and res_noop["stage1"] is None and res_noop["selector_uv47"] is None

    # 2. Baseline + changed text + embedding deletion execute Stage 1 + uv_47 MLP only.
    res_base = ea_train.run_ablation_variant(
        specs["U0_t1_fam_clean"],
        42,
        train_ret,
        val_ret,
        info,
        train_src,
        val_src,
        enrichment,
        out_root,
        stage1_epochs=2,
        selector_epochs=2,
        reuse_clean_v2_baseline=False,
        device="cpu",
    )
    res_txt = ea_train.run_ablation_variant(
        specs["T_no_brand"],
        42,
        train_ret,
        val_ret,
        info,
        train_src,
        val_src,
        enrichment,
        out_root,
        stage1_epochs=2,
        selector_epochs=2,
        reuse_clean_v2_baseline=False,
        device="cpu",
    )
    res_emb = ea_train.run_ablation_variant(
        specs["E_no_unit"],
        42,
        train_ret,
        val_ret,
        info,
        train_src,
        val_src,
        enrichment,
        out_root,
        stage1_epochs=2,
        selector_epochs=2,
        reuse_clean_v2_baseline=False,
        device="cpu",
    )

    # Verify zero Stage 2 ranker directories were created.
    run_dirs = {p.name for p in (out_root / "runs").iterdir()}
    assert all(not name.startswith("ranker-") for name in run_dirs)
    assert "stage1-U0_t1_fam_clean-seed42" in run_dirs
    assert "selector-U0_t1_fam_clean-uv_47-seed42" in run_dirs
    assert "selector-T_no_brand-uv_47-seed42" in run_dirs
    assert "selector-E_no_unit-uv_47-seed42" in run_dirs
    assert res_emb["stage1"]["parameters"] < res_base["stage1"]["parameters"]

    # Verify paired_ablation_intervals matches paired_intervals & paired_stage1_intervals on overlapping keys.
    base_npz = Path(res_base["mlp_per_query_predictions"])
    txt_npz = Path(res_txt["mlp_per_query_predictions"])
    self_ci = paired_ablation_intervals(base_npz, base_npz, repeats=50)
    assert all(v == [0.0, 0.0, 0.0] for v in self_ci.values())

    ci_ab = paired_ablation_intervals(base_npz, txt_npz, repeats=50)
    ci_dyn = paired_intervals(base_npz, txt_npz, repeats=50)
    for k in ("recall@10_delta", "ndcg@10_delta", "new_recall@10_delta"):
        assert np.allclose(ci_ab[k], ci_dyn[k])

    # Check Stage 1 prediction npz compatibility
    s1_base_npz = Path(res_base["stage1"]["selected_checkpoint"]).parent / "stage1_predictions.npz"
    s1_txt_npz = Path(res_txt["stage1"]["selected_checkpoint"]).parent / "stage1_predictions.npz"
    ci_s1 = paired_stage1_intervals([s1_base_npz], [s1_txt_npz], repeats=50)
    assert set(ci_s1.keys()) == {"new_recall@25_delta", "new_recall@50_delta"}
