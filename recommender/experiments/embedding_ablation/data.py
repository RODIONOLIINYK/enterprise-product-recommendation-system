"""Isolated data bundle preparation for product-text and product-embedding ablations."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from recommender.artifacts import atomic_json, json_digest
from recommender.development import load_development_purchases
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.static_enrichment.config import (
    CUSTOMER_ATTR_COLUMNS,
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

from .config import CACHE_DIR, TEXT_ABLATION_RECIPE_NAMES
from .text_cache import RECIPE_TO_OMITTED_FIELDS, get_or_build_ablated_text_embeddings


DEFAULT_RECIPES: Tuple[str, ...] = (
    "t1_fam",
    *TEXT_ABLATION_RECIPE_NAMES.values(),
)


def prepare_ablation_enrichment_bundle(
    recipes: Sequence[str] = DEFAULT_RECIPES,
    *,
    extra_recipe_omissions: Optional[Dict[str, Tuple[str, ...]]] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    encoder: Any = None,
    baseline_reference_dir: Optional[Path] = Path("data/interim/static_enrichment_v1/text_embeddings"),
) -> Dict[str, Any]:
    """Prepare isolated categorical vocabs/IDs and ablated t1_fam text embeddings.

    Strictly preserves all metadata columns used for scalar features and categorical embeddings;
    only text rendering omits the designated field per text-deletion recipe.
    """
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)

    train, val, info, train_source, val_source = prepare_retrieval_data()
    product_ids: list[str] = list(info["product_ids"])
    trained_count = int(info["trained_products"])

    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    if bool(purchases["purchase_date"].ge(TEST_START).any()):
        raise ValueError("Held-out purchases detected in development frame.")
    train_purchases = purchases.loc[purchases["purchase_date"] < TRAIN_CUTOFF]
    train_customer_ids = set(train_purchases["customer_id"].astype(str))

    catalogue = load_enriched_catalogue(get_datasets_locs("items"))
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
        for split_name, src, target_dict in (
            ("train", train_source, train_customer_attrs),
            ("val", val_source, val_customer_attrs),
        ):
            vals = [
                comp_indexed.at[cid, col_name] if cid in comp_indexed.index and col_name in comp_indexed.columns else None
                for cid in src["customer_ids"]
            ]
            target_dict[attr_name] = torch.tensor(encode_categorical_values(vals, vocab), dtype=torch.long)

    brand_missing = ordered_cat["brand"].isna() | ordered_cat["brand"].astype(str).str.strip().isin(["", "NULL"])
    family_missing = (
        ordered_cat["product_family"].isna()
        | ordered_cat["product_family"].astype(str).str.strip().isin(["", "NULL"])
    )
    has_nonempty_desc = ordered_cat["cleaned_description"].fillna("").astype(str).str.strip().ne("")
    cohort_masks = {
        "has_description": torch.tensor(has_nonempty_desc.to_numpy(dtype=bool, copy=True)),
        "in_description_file": torch.tensor(has_nonempty_desc.to_numpy(dtype=bool, copy=True)),
        "brand_missing": torch.tensor(brand_missing.to_numpy(dtype=bool, copy=True)),
        "family_missing": torch.tensor(family_missing.to_numpy(dtype=bool, copy=True)),
    }

    omit_lookup = dict(RECIPE_TO_OMITTED_FIELDS)
    if extra_recipe_omissions:
        omit_lookup.update(extra_recipe_omissions)

    text_embeddings: Dict[str, torch.Tensor] = {}
    text_recipe_hashes: Dict[str, str] = {}
    text_content_hashes: Dict[str, str] = {}
    text_cache_paths: Dict[str, str] = {}
    text_diagnostics: Dict[str, Dict[str, Any]] = {}
    for recipe in recipes:
        if recipe not in omit_lookup:
            raise ValueError(f"Unknown text recipe {recipe!r} in ablation bundle.")
        tensor, diag = get_or_build_ablated_text_embeddings(
            ordered_cat,
            product_ids,
            recipe,
            omit_fields=omit_lookup[recipe],
            trained_count=trained_count,
            cache_dir=root / "text_embeddings",
            device=device,
            encoder=encoder,
            baseline_reference_dir=baseline_reference_dir,
        )
        text_embeddings[recipe] = tensor
        text_recipe_hashes[recipe] = diag["recipe_hash"]
        text_content_hashes[recipe] = diag["content_hash"]
        text_cache_paths[recipe] = diag["cache_path"]
        text_diagnostics[recipe] = diag

    bundle = {
        "identity": json_digest({
            "namespace": "embedding_ablation_v1",
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
        "text_content_hashes": text_content_hashes,
        "text_cache_paths": text_cache_paths,
        "diagnostics": {
            "metadata_policy": "snapshot_unverified",
            "product_vocab_sizes": {k: len(v) for k, v in product_vocabs.items()},
            "customer_vocab_sizes": {k: len(v) for k, v in customer_vocabs.items()},
            "text_recipes": text_diagnostics,
        },
    }
    atomic_torch_save(bundle, root / "ablation_bundle.pt")
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
        root / "ablation_manifest.json",
        overwrite=True,
    )
    return bundle
