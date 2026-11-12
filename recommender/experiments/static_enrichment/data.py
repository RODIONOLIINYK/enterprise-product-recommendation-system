"""Data preparation and training-only categorical encoding for static enrichment."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.development import load_development_purchases
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF

from .config import (
    CACHE_DIR,
    CUSTOMER_ATTR_COLUMNS,
    DYNAMIC_CACHE_DIR,
    MIN_TRAIN_CATEGORY_COUNT,
    PRODUCT_ATTR_COLUMNS,
)
from .metadata import (
    encode_categorical_values,
    fit_categorical_vocab,
    load_cleaned_descriptions,
    load_enriched_catalogue,
    load_enriched_companies,
)
from .text_cache import get_or_build_product_text_embeddings


def prepare_static_enrichment_bundle(
    recipes: Sequence[str] = ("c0_s0", "t1", "t2", "t1_fam", "t2_fam"),
    *,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    encoder: Any = None,
) -> Dict[str, Any]:
    """Prepare and validate versioned static metadata, training-fitted vocabs, and text caches."""
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)

    train, val, info, train_source, val_source = prepare_retrieval_data()
    product_ids: list[str] = list(info["product_ids"])
    trained_count = int(info["trained_products"])
    train_product_ids = product_ids[:trained_count]

    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    if bool(purchases["purchase_date"].ge(TEST_START).any()):
        raise ValueError("Held-out purchases detected in development frame.")
    train_purchases = purchases.loc[purchases["purchase_date"] < TRAIN_CUTOFF]
    train_customer_ids = set(train_purchases["customer_id"].astype(str))

    catalogue = load_enriched_catalogue(get_datasets_locs("items"))
    companies = load_enriched_companies(purchases=purchases, minimum_purchases=1)
    descriptions_df = load_cleaned_descriptions()

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
        for split_name, src, target_dict in (
            ("train", train_source, train_customer_attrs),
            ("val", val_source, val_customer_attrs),
        ):
            vals = [
                comp_indexed.at[cid, col_name] if cid in comp_indexed.index and col_name in comp_indexed.columns else None
                for cid in src["customer_ids"]
            ]
            target_dict[attr_name] = torch.tensor(encode_categorical_values(vals, vocab), dtype=torch.long)

    desc_pids_all = set(descriptions_df["product_id"].astype(str))
    has_nonempty_desc = ordered_cat["cleaned_description"].fillna("").astype(str).str.strip().ne("")
    in_desc_file = ordered_cat["product_id"].astype(str).isin(desc_pids_all)
    brand_missing = ordered_cat["brand"].isna() | ordered_cat["brand"].astype(str).str.strip().isin(["", "NULL"])
    family_missing = (
        ordered_cat["product_family"].isna()
        | ordered_cat["product_family"].astype(str).str.strip().isin(["", "NULL"])
    )

    cohort_masks = {
        "has_description": torch.tensor(has_nonempty_desc.to_numpy(dtype=bool, copy=True)),
        "in_description_file": torch.tensor(in_desc_file.to_numpy(dtype=bool, copy=True)),
        "brand_missing": torch.tensor(brand_missing.to_numpy(dtype=bool, copy=True)),
        "family_missing": torch.tensor(family_missing.to_numpy(dtype=bool, copy=True)),
    }

    text_embeddings: Dict[str, torch.Tensor] = {}
    text_recipe_hashes: Dict[str, str] = {}
    text_cache_paths: Dict[str, str] = {}
    for recipe in recipes:
        if recipe == "c0_s0":
            text_embeddings["c0_s0"] = info["product_inputs"]["text_embedding"].clone()
            text_recipe_hashes["c0_s0"] = "e1_e5_retrieval_v1_reference"
            text_cache_paths["c0_s0"] = "data/interim/e1_e5_retrieval_v1/manifest.pt"
        else:
            tensor, r_hash, p_path = get_or_build_product_text_embeddings(
                ordered_cat,
                product_ids,
                recipe,
                cache_dir=root / "text_embeddings",
                encoder=encoder,
                device=device,
            )
            text_embeddings[recipe] = tensor
            text_recipe_hashes[recipe] = r_hash
            text_cache_paths[recipe] = str(p_path)

    matched_items_total = int(set(descriptions_df["product_id"]).intersection(set(catalogue["product_id"])).__len__())
    matched_tovar_total = int(
        set(descriptions_df["product_id"])
        .intersection(set(catalogue.loc[catalogue["item_type"] == "ТОВАР", "product_id"]))
        .__len__()
    )
    bundle = {
        "identity": json_digest({
            "source_identity": info["identity"],
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
        "text_cache_paths": text_cache_paths,
        "diagnostics": {
            "metadata_policy": "snapshot_unverified",
            "tier_column_chosen": 127,
            "tier_column_rejected": 124,
            "description_rows": len(descriptions_df),
            "description_matched_items": matched_items_total,
            "description_matched_tovar": matched_tovar_total,
            "description_in_dev_841": int(in_desc_file.sum()),
            "description_nonempty_in_dev_841": int(has_nonempty_desc.sum()),
            "product_vocab_sizes": {k: len(v) for k, v in product_vocabs.items()},
            "customer_vocab_sizes": {k: len(v) for k, v in customer_vocabs.items()},
            "train_products_oov_counts": {
                k: int((v[:trained_count] == 1).sum()) for k, v in product_attr_ids.items()
            },
            "train_products_none_counts": {
                k: int((v[:trained_count] == 0).sum()) for k, v in product_attr_ids.items()
            },
        },
    }
    atomic_torch_save(bundle, root / "enrichment_bundle.pt")
    atomic_json(
        {
            "identity": bundle["identity"],
            "source_identity": bundle["source_identity"],
            "text_recipe_hashes": text_recipe_hashes,
            "text_cache_paths": text_cache_paths,
            "diagnostics": bundle["diagnostics"],
            "train_cutoff": str(TRAIN_CUTOFF.date()),
            "heldout_excluded": str(TEST_START.date()),
        },
        root / "enrichment_manifest.json",
        overwrite=True,
    )
    return bundle


def load_dynamic_campaign_features_readonly() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Validate and memory-map the existing dynamic_campaign_v1 float16 feature arrays without modifying them."""
    manifest_path = DYNAMIC_CACHE_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dynamic campaign feature cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        path = DYNAMIC_CACHE_DIR / f"{split}_features.npy"
        if file_digest(path) != manifest["feature_sha256"][split]:
            raise ValueError(f"Dynamic campaign {split} feature checksum mismatch.")
    train_features = np.load(DYNAMIC_CACHE_DIR / "train_features.npy", mmap_mode="r")
    val_features = np.load(DYNAMIC_CACHE_DIR / "val_features.npy", mmap_mode="r")
    return manifest, train_features, val_features
