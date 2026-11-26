"""Synthetic tests for isolated cleaned-product-description experiment (desc_clean_v1)."""

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.description_cleaning.cleaner import (
    CLEANER_VERSION,
    CLEANING_RULES_HASH,
    clean_description_v1,
    extract_spec_tokens,
    load_descriptions_v1,
)
from recommender.experiments.description_cleaning.config import (
    NO_BRAND_CUSTOMER_EMBEDDINGS,
    NO_BRAND_PRODUCT_EMBEDDINGS,
    build_description_variant_specs,
)
from recommender.experiments.description_cleaning.evaluate import (
    evaluate_description_subgroups,
    paired_subgroup_intervals,
)
from recommender.experiments.description_cleaning.text_cache import (
    apply_token_budget_policy,
    build_description_product_text,
    get_or_build_description_text_embeddings,
    render_description_texts,
)
import recommender.experiments.description_cleaning.train as dc_train
from recommender.experiments.embedding_ablation.text_cache import build_ablated_product_text
from recommender.experiments.static_enrichment.metadata import clean_html_description
from recommender.experiments.static_enrichment.text_cache import TEXT_RECIPES
import recommender.experiments.static_enrichment.train as se_train


def test_html_tag_script_style_comment_hidden_removal():
    raw = (
        "<!-- secret comment -->"
        "<style>.promo{color:red}</style>"
        "<script>alert('xss')</script>"
        "<noscript>Enable JS</noscript>"
        "<template><div>Template hidden</div></template>"
        "<div hidden>Hidden attr text</div>"
        '<span aria-hidden="true">Aria hidden text</span>'
        '<p style="display: none;">Display none text</p>'
        '<div style="visibility: hidden">Visibility hidden text</div>'
        "<h2>Технічні характеристики</h2>"
        "<p>Синтетична олива <strong>5W-40</strong>.</p>"
        "<ul><li>Допуск MB-Approval 229.5</li><li>API SP</li></ul>"
        '<img src="https://example.com/bad_%D0%97.png" alt="ignored" />'
    )
    cleaned, stats = clean_description_v1(raw, return_stats=True)
    for dropped in (
        "secret comment",
        "promo",
        "alert",
        "Enable JS",
        "Template hidden",
        "Hidden attr text",
        "Aria hidden text",
        "Display none text",
        "Visibility hidden text",
        "bad_",
    ):
        assert dropped not in cleaned
    assert "Технічні характеристики" in cleaned
    assert "Синтетична олива 5W-40." in cleaned
    assert "Допуск MB-Approval 229.5" in cleaned
    assert "API SP" in cleaned
    assert stats["tags_stripped"] > 0
    assert stats["hidden_or_script_blocks_dropped"] >= 8


def test_entities_encoded_markup_and_math_comparisons_safety():
    # Genuine encoded and double-encoded HTML tags are decoded and stripped;
    # genuine numeric comparisons (<1%, < 7, >200°C, -38°C, 1:100) remain intact.
    raw = (
        "&amp;lt;p&amp;gt;Вміст сірки &lt;1% та pH &lt; 7.&amp;lt;/p&amp;gt;"
        "&lt;p&gt;Температура спалаху &gt;200&deg;C, застигання -38&#176;C.&lt;/p&gt;"
        "<p>Розведення 1:100&nbsp;або 1:1 &mdash; захист&trade;.</p>"
        "<p>Кількість води &lt;1% у концентраті.</p>"
    )
    cleaned, stats = clean_description_v1(raw, return_stats=True)
    assert stats["encoded_markup_passes"] >= 1
    assert "<p>" not in cleaned and "&lt;" not in cleaned
    assert "<1%" in cleaned
    assert "< 7" in cleaned
    assert ">200°C" in cleaned
    assert "-38°C" in cleaned
    assert "1:100" in cleaned
    assert "1:1" in cleaned
    assert "™" in cleaned


def test_unicode_nfc_controls_emojis_vs_technical_symbols():
    # NFD ї ('і' + U+0308), NBSP, narrow NBSP, ZWSP, C0/C1 controls, emojis, keycaps, bullets
    nfd_word = "Оливноі\u0308"
    raw = (
        f"<p>1️⃣\u200b {nfd_word}\u00a0захист\u202f✅\x07\x8f</p>"
        "<p>· 🔹 Температура: -40°C ± 2°C, в'язкість ≤ 6500 мПа·с, ≥ 98%, 5W-30 × 4л.</p>"
        "<p>Не містить фосфатів, амінів та нітритів; без силікону.</p>"
    )
    cleaned, stats = clean_description_v1(raw, return_stats=True)
    assert "1. Оливної захист" in cleaned
    assert "\u200b" not in cleaned
    assert "\u00a0" not in cleaned
    assert "\u202f" not in cleaned
    assert "\x07" not in cleaned and "\x8f" not in cleaned
    assert "✅" not in cleaned and "🔹" not in cleaned
    assert "-40°C ± 2°C" in cleaned
    assert "≤ 6500" in cleaned
    assert "≥ 98%" in cleaned
    assert "5W-30 × 4л" in cleaned
    assert "Не містить фосфатів" in cleaned
    assert "без силікону" in cleaned
    assert stats["emojis_removed"] >= 2


def test_formal_approvals_vs_meets_requirements_and_conservative_boilerplate():
    raw = (
        "<h3>Чому варто купити Shell Helix HX7 10W-40?</h3>"
        "<ul>"
        "<li>✅ В'язкість 10W-40, температура застигання -36°C.</li>"
        "<li>Офіційні схвалення / Допуски: MB-Approval 229.3; VW 502.00/505.00.</li>"
        "<li>Відповідає вимогам: API SN/CF; ACEA A3/B3, A3/B4.</li>"
        "</ul>"
        "<h3>Як перевірити оригінальність каністри?</h3>"
        "<p>Перевірити оригінальність можна за QR-кодом на сайті https://www.shellcheck.com або за телефоном +380 (44) 123-45-67. "
        "Доставка по всій Україні Новою Поштою, оплата при отриманні, додайте у кошик!</p>"
        "<p>Ідеальний вибір для турбованих двигунів з API SP та ACEA C3, не містить золи! "
        "Це вибір професіоналів та найкращий помічник!</p>"
    )
    cleaned, stats = clean_description_v1(raw, return_stats=True)
    # CTA headings dropped, BUT technical bullets underneath are preserved
    assert "Чому варто купити" not in cleaned
    assert "Як перевірити оригінальність" not in cleaned
    assert "В'язкість 10W-40, температура застигання -36°C." in cleaned
    # Formally approved vs meets-requirements distinction kept verbatim
    assert "Офіційні схвалення / Допуски: MB-Approval 229.3; VW 502.00/505.00." in cleaned
    assert "Відповідає вимогам: API SN/CF; ACEA A3/B3, A3/B4." in cleaned
    # URL/phone/shipping/payment/QR/pure-fluff sentences removed
    assert "shellcheck.com" not in cleaned
    assert "+380" not in cleaned
    assert "Новою Поштою" not in cleaned
    assert "вибір професіоналів" not in cleaned
    # Technical sentence that happens to contain "Ідеальний вибір" + real specs MUST be kept
    assert "Ідеальний вибір для турбованих двигунів з API SP та ACEA C3, не містить золи!" in cleaned
    assert stats["boilerplate_blocks_removed"] == 2
    assert stats["boilerplate_sentences_removed"] >= 3


def test_within_product_exact_dedup_only_and_table_parsing():
    raw_prod1 = (
        "<p>Допуск Ford WSS-M97B44-D. Допуск Ford WSS-M97B44-D.</p>"
        "<p>Допуск Ford WSS-M97B44-D.</p>"
        "<table>"
        "<tr><th>Співвідношення</th><th>Температура замерзання</th></tr>"
        "<tr><td>1:1</td><td>-38°C</td></tr>"
        "<tr><td>1:2</td><td>-20°C</td></tr>"
        "</table>"
    )
    cleaned1, stats1 = clean_description_v1(raw_prod1, return_stats=True)
    assert cleaned1.count("Допуск Ford WSS-M97B44-D.") == 1
    assert "Співвідношення: 1:1 | Температура замерзання: -38°C" in cleaned1
    assert "Співвідношення: 1:2 | Температура замерзання: -20°C" in cleaned1
    assert stats1["tables_parsed"] == 1
    assert stats1["duplicate_blocks_removed"] >= 1 or stats1["duplicate_sentences_removed"] >= 1

    # Zero cross-product deduplication: same exact text in another product stays intact
    cleaned2 = clean_description_v1("<p>Допуск Ford WSS-M97B44-D.</p>")
    assert cleaned2 == "Допуск Ford WSS-M97B44-D."

    # Headerless table falls back cleanly to Cell | Cell2
    cleaned_nohdr = clean_description_v1("<table><tr><td>API</td><td>SN</td></tr></table>")
    assert cleaned_nohdr == "API | SN"


def test_missing_empty_null_fully_removed_fallback_to_exact_t1_fam():
    base_row = {
        "product_id": "P1",
        "brand": "SHELL",
        "product_name": "Олива SHELL Helix Ultra 5W-40",
        "base_name": "Helix Ultra 5W-40",
        "product_type": "Олива моторна",
        "product_category": "АВТО-СМ",
        "product_subcategory": "АВТ-ЛЕГКОМ",
        "business_line": "PCMO",
        "product_family": "ULTRA",
    }
    expected_prefix = build_ablated_product_text(base_row, omit_fields=())
    assert "SHELL" in expected_prefix

    for empty_raw in (None, float("nan"), "", "   ", "NULL", "<div><!-- hidden --></div><p>Додайте у кошик</p>"):
        row = {**base_row, "raw_description": empty_raw}
        assert build_description_product_text(row, "t1_fam") == expected_prefix
        assert build_description_product_text(row, "t1_fam_desc_clean_v1") == expected_prefix

    row_with_desc = {**base_row, "raw_description": "<p>API SP, ACEA A3/B4</p>"}
    txt_html = build_description_product_text(row_with_desc, "t1_fam_desc_html_v1")
    txt_clean = build_description_product_text(row_with_desc, "t1_fam_desc_clean_v1")
    assert txt_html.startswith(expected_prefix + " ")
    assert txt_clean.startswith(expected_prefix + " ")
    assert "API SP, ACEA A3/B4" in txt_clean


def test_load_descriptions_v1_columns_0_1_2_only(tmp_path):
    csv_path = tmp_path / "описание.csv"
    csv_path.write_text(
        "\n".join([
            "Назва 1;SKU-1;<p>Опис <b>5W-40</b></p>;SECRET_PRICE_9999;SECRET_QTY_888",
            "Назва 2;SKU-2;<p>Опис 2</p>;123.45;99",
            "Null row;NULL;<p>Ignored</p>;999;1",
        ]),
        encoding="utf-8",
    )
    df = load_descriptions_v1(csv_path)
    assert list(df["product_id"]) == ["SKU-1", "SKU-2"]
    assert "SECRET_PRICE_9999" not in df.to_string()
    assert "SECRET_QTY_888" not in df.to_string()
    assert df.loc[df["product_id"] == "SKU-1", "cleaned_description_v1"].iloc[0] == "Опис 5W-40"


def test_text_cache_reuse_invalidation_and_token_budget(tmp_path):
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
            "raw_description": "<p>Синтетична олива MB-Approval 229.5</p>",
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
            "raw_description": "",
        },
    ])

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=True, truncation=False):
            return text.split()

    class DummyEncoder:
        def __init__(self, max_seq_length=64):
            self.calls = 0
            self.encoded_batches = []
            self.tokenizer = DummyTokenizer()
            self.max_seq_length = max_seq_length

        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
            self.calls += 1
            self.encoded_batches.append(list(texts))
            return np.stack([np.full(8, float(len(t)), dtype=np.float32) for t in texts])

    enc = DummyEncoder()
    ref_dir = tmp_path / "ref_text"
    cache_dir = tmp_path / "desc_text"

    # Build baseline t1_fam cache
    base_vec, _ = get_or_build_description_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam",
        trained_count=1,
        cache_dir=ref_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=None,
        raw_description_path=None,
    )
    assert enc.calls == 1

    # D_clean reuses P2 from baseline cache and encodes ONLY P1!
    clean_vec, clean_diag = get_or_build_description_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam_desc_clean_v1",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
        raw_description_path=None,
    )
    assert enc.calls == 2
    assert len(enc.encoded_batches[-1]) == 1  # only P1 encoded
    assert clean_diag["changed_text_count"] == 1
    assert clean_diag["reused_unchanged_count"] == 1
    assert clean_diag["newly_encoded_count"] == 1
    assert clean_diag["cleaner_version"] == CLEANER_VERSION
    assert clean_diag["cleaner_rules_hash"] == CLEANING_RULES_HASH
    torch.testing.assert_close(clean_vec[1], base_vec[1])
    assert not torch.allclose(clean_vec[0], base_vec[0])

    # Second call hits cache without calling encoder
    clean_vec2, clean_diag2 = get_or_build_description_text_embeddings(
        df,
        ["P1", "P2"],
        "t1_fam_desc_clean_v1",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
        raw_description_path=None,
    )
    assert enc.calls == 2
    assert clean_diag2["loaded_from_cache"] is True
    torch.testing.assert_close(clean_vec, clean_vec2)

    # Modifying text invalidates cache and re-encodes
    df_mod = df.copy()
    df_mod.loc[df_mod["product_id"] == "P1", "raw_description"] = "<p>Оновлений опис API SP</p>"
    _, mod_diag = get_or_build_description_text_embeddings(
        df_mod,
        ["P1", "P2"],
        "t1_fam_desc_clean_v1",
        trained_count=1,
        cache_dir=cache_dir,
        encoder=enc,
        embedding_dim=8,
        baseline_reference_dir=ref_dir,
        raw_description_path=None,
    )
    assert enc.calls == 3
    assert mod_diag["content_hash"] != clean_diag["content_hash"]

    # Token budget policy preserves prefix + spec/table block when truncating prose
    prefix = "BRAND PRODUCT CAT"
    desc = "Загальний довгий рекламний текст без параметрів один два три чотири п'ять\nСпіввідношення: 1:1 | -38°C"
    kept_txt, tok_n, was_trunc = apply_token_budget_policy(
        prefix, desc, tokenizer=DummyTokenizer(), max_seq_length=8
    )
    assert was_trunc is True
    assert tok_n <= 8
    assert kept_txt.startswith(prefix)
    assert "Співвідношення: 1:1 | -38°C" in kept_txt

    _, r_diag = render_description_texts(df, ["P1", "P2"], "t1_fam_desc_clean_v1", tokenizer=DummyTokenizer(), max_seq_length=6)
    assert r_diag["truncated_count"] == 1
    assert r_diag["truncated_product_ids"] == ["P1"]


def test_full_pool_subgroup_evaluation_and_paired_intervals(tmp_path):
    # 4 queries, 6 products; products 0, 2 have descriptions
    has_desc = torch.tensor([True, False, True, False, False, False])
    allowed = torch.ones((4, 6), dtype=torch.bool)
    allowed[:, 5] = False  # product 5 ineligible
    unseen = torch.ones((4, 6), dtype=torch.bool)
    unseen[:, 0] = False  # product 0 previously seen for all queries

    target = torch.zeros((4, 6), dtype=torch.bool)
    new_target = torch.zeros((4, 6), dtype=torch.bool)
    # Query 0: new target with desc (col 2)
    target[0, 2] = True
    new_target[0, 2] = True
    # Query 1: new target without desc (col 3)
    target[1, 3] = True
    new_target[1, 3] = True
    # Query 2: repeat target with desc (col 0) + new target without desc (col 4)
    target[2, 0] = True
    target[2, 4] = True
    new_target[2, 4] = True
    # Query 3: new target with desc (col 2) and without desc (col 1)
    target[3, 1] = True
    target[3, 2] = True
    new_target[3, 1] = True
    new_target[3, 2] = True

    campaign_data = {
        "allowed": allowed,
        "unseen": unseen,
        "target": target,
        "new_target": new_target,
        "source": {
            "customer_ids": ["C1", "C1", "C2", "C3"],
            "positive_pids": [["P2"], ["P3"], ["P0", "P4"], ["P1", "P2"]],
            "new_positive_pids": [["P2"], ["P3"], ["P4"], ["P1", "P2"]],
        },
    }
    scores_a = torch.tensor([
        [0.1, 0.2, 5.0, 0.3, 0.4, 9.0],
        [0.1, 0.2, 0.3, 5.0, 0.4, 9.0],
        [5.0, 0.2, 0.3, 0.4, 4.0, 9.0],
        [0.1, 5.0, 4.0, 0.2, 0.3, 9.0],
    ])
    npz_a = tmp_path / "a.npz"
    res_a = evaluate_description_subgroups(scores_a, campaign_data, has_desc, save_path=npz_a)
    assert res_a["pool_policy"] == "full_rolling_eligible_pool_fixed"
    assert res_a["overall"]["new_targets"] == 5
    assert res_a["overall"]["new_hits@10"] == 5
    w_sub = res_a["new_target_subgroups"]["with_description"]
    wo_sub = res_a["new_target_subgroups"]["without_description"]
    assert w_sub["new_targets"] == 2
    assert wo_sub["new_targets"] == 3
    assert w_sub["new_targets"] + wo_sub["new_targets"] == res_a["overall"]["new_targets"]

    ci_self = paired_subgroup_intervals(npz_a, npz_a, repeats=50)
    assert all(v == [0.0, 0.0, 0.0] for group in ci_self.values() for v in group.values())


def test_stage1_only_orchestration_and_legacy_isolation(tmp_path, monkeypatch):
    # Verify legacy functions and TEXT_RECIPES remain unmodified
    assert TEXT_RECIPES == ("c0_s0", "t1", "t2", "t1_fam", "t2_fam")
    assert clean_html_description("<p>Test &nbsp; text</p>") == "Test text"

    specs = build_description_variant_specs()
    assert set(specs.keys()) == {"no_brand_v1", "D_html", "D_clean"}
    for sp in specs.values():
        assert sp.variant_spec.product_embeddings == NO_BRAND_PRODUCT_EMBEDDINGS
        assert sp.variant_spec.customer_embeddings == NO_BRAND_CUSTOMER_EMBEDDINGS
        assert sp.variant_spec.customer_tabular_features == 0
        assert sp.variant_spec.product_demand_features == 12

    torch.manual_seed(7)
    n_train, n_val, n_prod, n_trained = 12, 6, 105, 90

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
            "text_embedding": torch.randn(n_rows, 1024, generator=g),
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
        "identity": "synthetic-desc-clean-test",
        "product_ids": product_ids,
        "trained_products": n_trained,
        "product_inputs": {
            "text_embedding": torch.randn(n_prod, 1024),
            "demand_features": torch.cat([torch.zeros(n_prod, 6), torch.ones(n_prod, 12)], dim=1),
            "brand_id": torch.zeros(n_prod, dtype=torch.long),
            "category_id": torch.zeros(n_prod, dtype=torch.long),
            "subcategory_id": torch.zeros(n_prod, dtype=torch.long),
            "business_line_id": torch.zeros(n_prod, dtype=torch.long),
        },
    }
    desc_mask = torch.zeros(n_prod, dtype=torch.bool)
    desc_mask[15:18] = True
    enrichment = {
        "identity": "synthetic-desc-enrichment-v1",
        "product_vocabs": {k: {"<NONE>": 0, "<OOV>": 1, "A": 2} for k in NO_BRAND_PRODUCT_EMBEDDINGS},
        "customer_vocabs": {"sales_branch": {"<NONE>": 0, "<OOV>": 1, "B1": 2}},
        "product_attr_ids": {k: torch.zeros(n_prod, dtype=torch.long) for k in NO_BRAND_PRODUCT_EMBEDDINGS},
        "train_customer_attrs": {"sales_branch": torch.zeros(n_train, dtype=torch.long)},
        "val_customer_attrs": {"sales_branch": torch.zeros(n_val, dtype=torch.long)},
        "cohort_masks": {
            "has_description": desc_mask,
            "has_description_html": desc_mask,
            "has_description_clean": desc_mask,
            "in_description_file": desc_mask,
            "brand_missing": torch.zeros(n_prod, dtype=torch.bool),
            "family_missing": torch.zeros(n_prod, dtype=torch.bool),
        },
        "text_embeddings": {
            "t1_fam": torch.randn(n_prod, 1024),
            "t1_fam_desc_html_v1": torch.randn(n_prod, 1024),
            "t1_fam_desc_clean_v1": torch.randn(n_prod, 1024),
        },
        "text_recipe_hashes": {
            "t1_fam": "h_base",
            "t1_fam_desc_html_v1": "h_html",
            "t1_fam_desc_clean_v1": "h_clean",
        },
        "diagnostics": {
            "text_recipes": {
                "t1_fam_desc_html_v1": {"changed_text_count": 10, "empty_text_count": 0},
                "t1_fam_desc_clean_v1": {"changed_text_count": 10, "empty_text_count": 0},
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

    f41_tr = np.random.default_rng(1).random((n_train, n_prod, 41), dtype=np.float32).astype(np.float16)
    f41_va = np.random.default_rng(2).random((n_val, n_prod, 41), dtype=np.float32).astype(np.float16)
    dyn_manifest = {"identity": "synthetic-dyn", "source_identity": info["identity"]}
    monkeypatch.setattr(
        se_train,
        "load_dynamic_campaign_features_readonly",
        lambda: (dyn_manifest, f41_tr, f41_va),
    )

    out_root = tmp_path / "desc_clean_ckpts"
    with pytest.raises(ValueError, match="no_brand_v1"):
        dc_train.run_description_variant(
            specs["no_brand_v1"],
            42,
            train_ret,
            val_ret,
            info,
            train_src,
            val_src,
            enrichment,
            out_root,
            device="cpu",
        )

    res_html = dc_train.run_description_variant(
        specs["D_html"],
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
        device="cpu",
    )
    res_clean = dc_train.run_description_variant(
        specs["D_clean"],
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
        device="cpu",
    )

    run_dirs = {p.name for p in (out_root / "runs").iterdir()}
    assert all(not name.startswith("ranker-") for name in run_dirs)
    assert "stage1-D_html-seed42" in run_dirs
    assert "selector-D_html-uv_47-seed42" in run_dirs
    assert "stage1-D_clean-seed42" in run_dirs
    assert "selector-D_clean-uv_47-seed42" in run_dirs
    assert res_html["selector_uv47"]["metrics"]["new_targets"] == n_val
    assert res_clean["subgroups_clean_mask"]["new_target_subgroups"]["with_description"]["new_targets"] > 0
    assert res_clean["subgroups_clean_mask"]["new_target_subgroups"]["without_description"]["new_targets"] > 0
    assert extract_spec_tokens("Олива 5W-40 MB 229.5 -38°C")  # sanity check helper
