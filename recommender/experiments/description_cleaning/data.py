"""Isolated data bundle preparation for cleaned-product-description experiments."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.development import load_development_purchases
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.static_enrichment.config import (
    CUSTOMER_ATTR_COLUMNS,
    DEFAULT_DESCRIPTION_PATH,
    MIN_TRAIN_CATEGORY_COUNT,
    PRODUCT_ATTR_COLUMNS,
)
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    fit_categorical_vocab,
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF

from .cleaner import (
    CLEANER_VERSION,
    CLEANING_RULES,
    CLEANING_RULES_HASH,
    extract_spec_tokens,
    load_descriptions_v1,
)
from .config import (
    BASELINE_REFERENCE_DIR,
    BASELINE_TEXT_CACHE_DIR,
    CACHE_DIR,
    DESCRIPTION_TEXT_RECIPES,
)
from .text_cache import get_or_build_description_text_embeddings


def prepare_description_cleaning_bundle(
    recipes: Sequence[str] = DESCRIPTION_TEXT_RECIPES,
    *,
    cache_dir: Optional[Path] = None,
    descriptions_path: Optional[Path] = None,
    device: Optional[str] = None,
    encoder: Any = None,
    baseline_reference_dir: Optional[Path] = BASELINE_TEXT_CACHE_DIR,
    measure_tokens_with_encoder: bool = True,
) -> Dict[str, Any]:
    """Prepare isolated vocabs, snapshot_unverified descriptions (cols 0/1/2), and BGE-M3 embeddings."""
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    desc_source = Path(descriptions_path) if descriptions_path is not None else DEFAULT_DESCRIPTION_PATH

    train, val, info, train_source, val_source = prepare_retrieval_data()
    product_ids: list[str] = list(info["product_ids"])
    trained_count = int(info["trained_products"])

    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    if bool(purchases["purchase_date"].ge(TEST_START).any()):
        raise ValueError("Held-out purchases detected in development frame.")
    train_purchases = purchases.loc[purchases["purchase_date"] < TRAIN_CUTOFF]
    train_customer_ids = set(train_purchases["customer_id"].astype(str))

    catalogue = load_enriched_catalogue(get_datasets_locs("items"), descriptions_path=None)
    if "cleaned_description" in catalogue.columns:
        catalogue = catalogue.drop(columns=["cleaned_description"])

    descriptions_df = load_descriptions_v1(desc_source)
    catalogue = catalogue.merge(descriptions_df, on="product_id", how="left", validate="one_to_one")
    for col in ("raw_name", "raw_description", "legacy_html_description", "cleaned_description_v1", "cleaning_stats_json"):
        catalogue[col] = catalogue[col].fillna("")

    companies = load_enriched_companies(purchases=purchases, minimum_purchases=1)

    ordered_cat = catalogue.set_index("product_id").reindex(product_ids).reset_index()
    train_cat = ordered_cat.iloc[:trained_count]

    product_vocabs: Dict[str, Dict[str, int]] = {}
    product_attr_ids: Dict[str, torch.Tensor] = {}
    for attr_name, col_name in PRODUCT_ATTR_COLUMNS.items():
        vocab = fit_categorical_vocab(
            train_cat[col_name].tolist() if col_name in train_cat else [None] * trained_count,
            min_count=MIN_TRAIN_CATEGORY_COUNT,
        )
        encoded = encode_categorical_values(
            ordered_cat[col_name].tolist() if col_name in ordered_cat else [None] * len(product_ids),
            vocab,
        )
        product_vocabs[attr_name] = vocab
        product_attr_ids[attr_name] = torch.tensor(encoded, dtype=torch.long)

    train_companies = companies.loc[companies["customer_id"].astype(str).isin(train_customer_ids)]
    comp_indexed = companies.drop_duplicates("customer_id").set_index("customer_id")

    customer_vocabs: Dict[str, Dict[str, int]] = {}
    train_customer_attrs: Dict[str, torch.Tensor] = {}
    val_customer_attrs: Dict[str, torch.Tensor] = {}
    for attr_name, col_name in CUSTOMER_ATTR_COLUMNS.items():
        vocab = fit_categorical_vocab(
            train_companies[col_name].tolist() if col_name in train_companies else [],
            min_count=MIN_TRAIN_CATEGORY_COUNT,
        )
        customer_vocabs[attr_name] = vocab
        for _, src, target_dict in (
            ("train", train_source, train_customer_attrs),
            ("val", val_source, val_customer_attrs),
        ):
            vals = [
                comp_indexed.at[cid, col_name] if cid in comp_indexed.index and col_name in comp_indexed.columns else None
                for cid in src["customer_ids"]
            ]
            target_dict[attr_name] = torch.tensor(encode_categorical_values(vals, vocab), dtype=torch.long)

    # Verify vocab compatibility with active no-brand-v1 reference manifest if available.
    ref_manifest_path = BASELINE_REFERENCE_DIR / "manifest.json"
    if ref_manifest_path.is_file():
        ref_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
        static_towers = ref_manifest.get("variant_spec", {}).get("static_towers", {})
        expected_prod = static_towers.get("product_attr_embedding_shapes", {})
        for attr, shape in expected_prod.items():
            if len(product_vocabs[attr]) != int(shape[0]):
                raise ValueError(f"Product vocab size mismatch for {attr}: {len(product_vocabs[attr])} != {shape[0]}")
        expected_cust = static_towers.get("customer_extra_attr_embedding_shapes", {})
        for attr, shape in expected_cust.items():
            if len(customer_vocabs[attr]) != int(shape[0]):
                raise ValueError(f"Customer vocab size mismatch for {attr}: {len(customer_vocabs[attr])} != {shape[0]}")

    desc_pids_all = set(descriptions_df["product_id"].astype(str))
    in_desc_file = ordered_cat["product_id"].astype(str).isin(desc_pids_all)
    has_desc_html = ordered_cat["legacy_html_description"].fillna("").astype(str).str.strip().ne("")
    has_desc_clean = ordered_cat["cleaned_description_v1"].fillna("").astype(str).str.strip().ne("")
    brand_missing = ordered_cat["brand"].isna() | ordered_cat["brand"].astype(str).str.strip().isin(["", "NULL"])
    family_missing = (
        ordered_cat["product_family"].isna()
        | ordered_cat["product_family"].astype(str).str.strip().isin(["", "NULL"])
    )

    cohort_masks = {
        "has_description": torch.tensor(has_desc_html.to_numpy(dtype=bool, copy=True)),
        "has_description_html": torch.tensor(has_desc_html.to_numpy(dtype=bool, copy=True)),
        "has_description_clean": torch.tensor(has_desc_clean.to_numpy(dtype=bool, copy=True)),
        "in_description_file": torch.tensor(in_desc_file.to_numpy(dtype=bool, copy=True)),
        "brand_missing": torch.tensor(brand_missing.to_numpy(dtype=bool, copy=True)),
        "family_missing": torch.tensor(family_missing.to_numpy(dtype=bool, copy=True)),
    }

    text_embeddings: Dict[str, torch.Tensor] = {}
    text_recipe_hashes: Dict[str, str] = {}
    text_content_hashes: Dict[str, str] = {}
    text_cache_paths: Dict[str, str] = {}
    text_diagnostics: Dict[str, Dict[str, Any]] = {}
    for recipe in recipes:
        tensor, diag = get_or_build_description_text_embeddings(
            ordered_cat,
            product_ids,
            recipe,
            trained_count=trained_count,
            cache_dir=root / "text_embeddings",
            device=device,
            encoder=encoder,
            baseline_reference_dir=baseline_reference_dir,
            raw_description_path=desc_source,
            measure_tokens_with_encoder=measure_tokens_with_encoder,
        )
        text_embeddings[recipe] = tensor
        text_recipe_hashes[recipe] = diag["recipe_hash"]
        text_content_hashes[recipe] = diag["content_hash"]
        text_cache_paths[recipe] = diag["cache_path"]
        text_diagnostics[recipe] = diag

    # Summarize cleaning & spec retention diagnostics across all CSV rows and dev 841 SKUs
    cleaning_totals = {
        "csv_rows": int(len(descriptions_df)),
        "csv_nonempty_html": int(descriptions_df["legacy_html_description"].str.strip().ne("").sum()),
        "csv_nonempty_clean_v1": int(descriptions_df["cleaned_description_v1"].str.strip().ne("").sum()),
        "dev_skus_total": len(product_ids),
        "dev_skus_in_csv": int(in_desc_file.sum()),
        "dev_skus_nonempty_html": int(has_desc_html.sum()),
        "dev_skus_nonempty_clean_v1": int(has_desc_clean.sum()),
        "dev_skus_pct_nonempty_html": float(100.0 * has_desc_html.sum() / max(len(product_ids), 1)),
        "dev_skus_pct_nonempty_clean_v1": float(100.0 * has_desc_clean.sum() / max(len(product_ids), 1)),
        "spec_token_losses_vs_legacy": 0,
        "tags_stripped": 0,
        "hidden_or_script_blocks_dropped": 0,
        "tables_parsed": 0,
        "emojis_removed": 0,
        "boilerplate_blocks_removed": 0,
        "boilerplate_sentences_removed": 0,
        "duplicate_blocks_removed": 0,
        "duplicate_sentences_removed": 0,
    }
    for _, row in descriptions_df.iterrows():
        st = json.loads(row["cleaning_stats_json"])
        for k in (
            "tags_stripped",
            "hidden_or_script_blocks_dropped",
            "tables_parsed",
            "emojis_removed",
            "boilerplate_blocks_removed",
            "boilerplate_sentences_removed",
            "duplicate_blocks_removed",
            "duplicate_sentences_removed",
        ):
            cleaning_totals[k] += int(st.get(k, 0))
        lost = extract_spec_tokens(row["legacy_html_description"]) - extract_spec_tokens(row["cleaned_description_v1"])
        if lost:
            cleaning_totals["spec_token_losses_vs_legacy"] += len(lost)

    # Validation new-target description coverage
    _, val_new_cols = val["new_target"].nonzero(as_tuple=True)
    val_new_total = int(len(val_new_cols))
    val_new_in_csv = int(cohort_masks["in_description_file"][val_new_cols].sum().item())
    val_new_html = int(cohort_masks["has_description_html"][val_new_cols].sum().item())
    val_new_clean = int(cohort_masks["has_description_clean"][val_new_cols].sum().item())

    raw_desc_sha = file_digest(desc_source) if desc_source.is_file() else None
    bundle = {
        "identity": json_digest({
            "namespace": "description_cleaning_v1",
            "source_identity": info["identity"],
            "raw_description_sha256": raw_desc_sha,
            "cleaner_version": CLEANER_VERSION,
            "cleaner_rules_hash": CLEANING_RULES_HASH,
            "product_vocabs": product_vocabs,
            "customer_vocabs": customer_vocabs,
            "text_recipe_hashes": text_recipe_hashes,
        }),
        "source_identity": info["identity"],
        "product_ids": product_ids,
        "trained_products": trained_count,
        "product_vocabs": product_vocabs,
        "customer_vocabs": customer_vocabs,
        "product_attr_ids": product_attr_ids,
        "train_customer_attrs": train_customer_attrs,
        "val_customer_attrs": val_customer_attrs,
        "cohort_masks": cohort_masks,
        "text_embeddings": text_embeddings,
        "text_recipe_hashes": text_recipe_hashes,
        "text_content_hashes": text_content_hashes,
        "text_cache_paths": text_cache_paths,
        "diagnostics": {
            "metadata_policy": "snapshot_unverified",
            "raw_description_path": str(desc_source),
            "raw_description_sha256": raw_desc_sha,
            "cleaner_version": CLEANER_VERSION,
            "cleaner_rules_hash": CLEANING_RULES_HASH,
            "cleaning_rules": CLEANING_RULES,
            "cleaning_totals": cleaning_totals,
            "validation_new_target_coverage": {
                "total": val_new_total,
                "in_description_csv": val_new_in_csv,
                "nonempty_html": val_new_html,
                "nonempty_clean_v1": val_new_clean,
                "without_description_html": val_new_total - val_new_html,
                "without_description_clean_v1": val_new_total - val_new_clean,
            },
            "product_vocab_sizes": {k: len(v) for k, v in product_vocabs.items()},
            "customer_vocab_sizes": {k: len(v) for k, v in customer_vocabs.items()},
            "text_recipes": text_diagnostics,
        },
    }
    atomic_torch_save(bundle, root / "description_bundle.pt")
    atomic_json(
        {
            "identity": bundle["identity"],
            "source_identity": bundle["source_identity"],
            "text_recipe_hashes": text_recipe_hashes,
            "text_content_hashes": text_content_hashes,
            "text_cache_paths": text_cache_paths,
            "diagnostics": bundle["diagnostics"],
            "train_cutoff": str(TRAIN_CUTOFF.date()),
            "heldout_excluded": str(TEST_START.date()),
        },
        root / "description_manifest.json",
        overwrite=True,
    )
    return bundle
