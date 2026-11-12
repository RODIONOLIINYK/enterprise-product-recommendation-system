"""Isolated, validated development caches for shared Stage 1 experiments."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import torch

from recommender.catalogue import load_product_catalogue, order_retriever_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs
from recommender.shared_ranker import (
    feature_schema,
    map_history_skus,
    schema_digest,
    tensor_digest,
)
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_ranker import (
    attach_new_positive_metadata,
    checkpoint_digest,
    load_ranker_retriever,
    precompute_candidate_slates_for_transitions,
    prepare_slate_inputs,
    slate_cache_identity,
)
from recommender.train_retriever import build_retriever_transitions

SHARED_CACHE_VERSION = 1
DEFAULT_RETRIEVER = Path("checkpoints/retriever-discovery-reference/best_retriever.pt")
DEFAULT_CACHE = Path("data/interim/shared_stage1_v1")


from recommender.development import load_development_purchases


def atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            torch.save(data, temporary)
            temporary.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def validate_shared_cache(data, identity, product_ids, stage1_digest):
    provenance = data.get("shared_vector_provenance", {})
    if (data.get("cache_identity") != identity or data.get("shared_cache_version") != SHARED_CACHE_VERSION
            or data.get("feature_schema") != feature_schema()
            or provenance.get("schema_digest") != schema_digest()
            or provenance.get("product_ids") != list(product_ids)
            or provenance.get("stage1_digest") != stage1_digest):
        raise ValueError("Shared cache schema, catalogue order or Stage 1 provenance is incompatible.")
    n = len(data["customer_ids"])
    expected = {"stage1_customer_vectors": (n, 128), "seq_catalog_indices": (n, 32),
                "sequence": (n, 32, 9), "attention_mask": (n, 32), "candidate_indices": (n, 50),
                "pairwise_features": (n, 50, 13), "labels": (n, 50), "prediction_days": (n,)}
    for key, shape in expected.items():
        if data[key].shape != shape or not torch.isfinite(data[key]).all():
            raise ValueError(f"Invalid shared cache tensor: {key}.")
    if len(data["seq_canonical_skus"]) != n or any(len(row) != 32 for row in data["seq_canonical_skus"]):
        raise ValueError("Canonical history SKUs must align with every sequence position.")
    if not torch.equal(data["seq_catalog_indices"], map_history_skus(data["seq_canonical_skus"], product_ids)):
        raise ValueError("Canonical historical SKU mapping is inconsistent.")
    indices = data["candidate_indices"]
    if indices.min() < 0 or indices.max() >= len(product_ids):
        raise ValueError("Candidate indices are outside the expanded catalogue.")
    mapped = [[product_ids[i] for i in row] for row in indices.tolist()]
    if mapped != data["candidate_pids"]:
        raise ValueError("Candidate IDs and product-vector indices do not align.")
    if ((data["seq_purchase_dates"] * 365.25).round() >= data["prediction_days"][:, None]).masked_select(
            data["attention_mask"].bool()).any():
        raise ValueError("Shared history includes target/future purchases.")
    if not torch.equal(data["prediction_days"], torch.tensor([key[1] for key in data["transition_keys"]], dtype=torch.float32)):
        raise ValueError("Cache row dates do not match transition keys.")
    if [key[0] for key in data["transition_keys"]] != data["customer_ids"]:
        raise ValueError("Cache row customers do not match transition keys.")
    cutoff_day = (TEST_START - pd.Timestamp("2025-01-01")).days
    if data["prediction_days"].ge(cutoff_day).any():
        raise ValueError("Shared caches must exclude held-out dates.")


def prepare_shared_caches(device, checkpoint_path=DEFAULT_RETRIEVER, cache_dir=DEFAULT_CACHE):
    """Use identical new slates for R0 and every shared-vector comparison."""
    checkpoint_path, cache_dir = Path(checkpoint_path), Path(cache_dir)
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(purchases=purchases, minimum_purchases=1)
    digest = checkpoint_digest(checkpoint_path)
    retriever, vocabs = load_ranker_retriever(checkpoint_path, purchases, catalogue, companies, device)
    retriever.model.requires_grad_(False).eval()
    vectors = retriever.get_product_vectors().detach().cpu()
    product_ids = retriever.eligible_product_ids
    base_identity = slate_cache_identity(
        digest, checkpoint_path, purchases, catalogue, companies, include_subcategory=False,
        heuristic_fraction=.5, company_cache_name=retriever.company_cache_name,
        rolling=True, product_inputs=retriever.product_inputs,
    )
    identity = f"shared{SHARED_CACHE_VERSION}:{schema_digest()}:{base_identity}:{tensor_digest(vectors)}"
    train, val, _ = build_retriever_transitions(purchases)
    usable_dates = {date for date in {row["as_of_date"] for row in train}
                    if len(retriever.rolling_pool.eligible_on(date)) >= 50}
    train = [row for row in train if row["as_of_date"] in usable_dates]
    inputs = None
    outputs = []
    for split, transitions in (("train", train), ("val", val)):
        path = cache_dir / f"{split}.pt"
        data = torch.load(path, map_location="cpu") if path.is_file() else None
        split_identity = f"{identity}:{split}"
        if data is not None and data.get("cache_identity") != split_identity:
            data = None
        if data is None:
            if inputs is None:
                inputs = prepare_slate_inputs(retriever, checkpoint_path, purchases, catalogue, companies, train, vocabs)
            print(f"Preparing isolated {split} cache: {len(transitions)} transitions", flush=True)
            data = precompute_candidate_slates_for_transitions(
                transitions, **inputs, heuristic_fraction=.5, capture_stage1_vectors=True,
                stage1_checkpoint_digest=digest,
            )
            attach_new_positive_metadata(data, transitions, purchases)
            data.update({
                "cache_identity": split_identity, "shared_cache_version": SHARED_CACHE_VERSION,
                "feature_schema": feature_schema(),
                "transition_keys": list(zip(data["customer_ids"], data["prediction_days"].to(torch.int64).tolist(), strict=True)),
            })
            validate_shared_cache(data, split_identity, product_ids, digest)
            atomic_torch_save(data, path)
        else:
            validate_shared_cache(data, split_identity, product_ids, digest)
            # Reject even an otherwise valid cache if the chronological row keys differ.
            expected = [(str(row["customer_id"]), int((row["as_of_date"] - pd.Timestamp("2025-01-01")).days))
                        for row in transitions]
            if data["transition_keys"] != expected:
                raise ValueError("Cached rows do not match the current development transitions.")
        print(f"Validated {split} cache ({len(transitions)} rows)", flush=True)
        outputs.append(data)
    ordered = order_retriever_catalogue(catalogue, product_ids)
    metadata = torch.stack([
        torch.from_numpy(ordered[column].map(vocabs[name]).fillna(0).to_numpy(copy=True)).long()
        for column, name in (("product_category", "category_vocab"), ("product_subcategory", "subcategory_vocab"),
                             ("business_line", "business_line_vocab"))
    ]).T.contiguous()
    manifest = {
        "identity": identity, "stage1_digest": digest, "product_ids": product_ids,
        "product_vector_digest": tensor_digest(vectors), "product_vectors": vectors,
        "product_metadata_ids": metadata, "vocabs": vocabs,
        "feature_schema": feature_schema(), "train_rows": len(train), "val_rows": len(val),
        "training_cutoff": str(TRAIN_CUTOFF.date()), "test_start_excluded": str(TEST_START.date()),
    }
    if checkpoint_digest(checkpoint_path) != digest:
        raise RuntimeError("Stage 1 checkpoint changed during cache generation.")
    atomic_torch_save(manifest, cache_dir / "manifest.pt")
    (cache_dir / "manifest.json").write_text(json.dumps({
        key: value for key, value in manifest.items() if not isinstance(value, torch.Tensor)
    }, indent=2) + "\n")
    return outputs[0], outputs[1], manifest
