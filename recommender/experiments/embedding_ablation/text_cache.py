"""Field-level t1_fam text omission rendering and content-hashed BAAI/bge-m3 caching."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json
from recommender.experiments.static_enrichment.text_cache import (
    _clean_field,
    _dedupe_tokens,
    compute_text_recipe_hash,
    encode_texts_to_embeddings,
)
from recommender.text import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    load_cached_embeddings,
    save_cached_embeddings,
)

from .config import ABLATED_TEXT_FIELDS, CACHE_DIR, TEXT_ABLATION_RECIPE_NAMES


RECIPE_TO_OMITTED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "t1_fam": (),
    **{recipe: (field,) for field, recipe in TEXT_ABLATION_RECIPE_NAMES.items()},
}


def build_ablated_product_text(
    row: Union[pd.Series, Mapping[str, Any]],
    *,
    omit_fields: Sequence[str] = (),
) -> str:
    """Render t1_fam product text with structured field-level omissions (no description, no substring stripping).

    Preserves product_name -> ukrainian_name -> short_name fallback untouched and preserves
    case-insensitive token deduplication in the exact t1_fam field order.
    """
    omit_set = set(omit_fields)
    unknown = omit_set - set(ABLATED_TEXT_FIELDS)
    if unknown:
        raise ValueError(
            f"Unsupported text field omission(s) {sorted(unknown)}; allowed fields are {ABLATED_TEXT_FIELDS}."
        )

    parts: list[str] = []
    if "brand" not in omit_set:
        brand = _clean_field(row.get("brand"))
        if brand:
            parts.append(brand)

    for name_col in ("product_name", "ukrainian_name", "short_name"):
        name_val = _clean_field(row.get(name_col))
        if name_val:
            parts.append(name_val)
            break

    for field in ("base_name", "product_type", "product_category", "product_subcategory", "business_line"):
        if field in omit_set:
            continue
        val = _clean_field(row.get(field))
        if val:
            parts.append(val)

    if "product_family" not in omit_set:
        fam = _clean_field(row.get("product_family"))
        if fam:
            parts.append(fam)

    return _dedupe_tokens(parts)


def compute_text_content_hash(
    product_texts: Sequence[tuple[str, str]],
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Deterministic SHA-256 digest over model_name and sorted (product_id, text) pairs (recipe-name independent)."""
    digest = sha256()
    digest.update(json.dumps({"model_name": model_name}, sort_keys=True).encode("utf-8"))
    for pid, txt in sorted(product_texts, key=lambda item: item[0]):
        digest.update(b"\x00")
        digest.update(pid.encode("utf-8"))
        digest.update(b"\x01")
        digest.update(txt.encode("utf-8"))
    return digest.hexdigest()


def render_recipe_texts(
    catalogue_df: pd.DataFrame,
    product_ids: Sequence[str],
    omit_fields: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Render ordered (product_id, text) pairs for all product_ids."""
    indexed = catalogue_df.drop_duplicates("product_id").set_index("product_id").reindex(list(product_ids))
    return [
        (str(pid), build_ablated_product_text(indexed.loc[pid], omit_fields=omit_fields) if pid in indexed.index else "")
        for pid in product_ids
    ]


def get_or_build_ablated_text_embeddings(
    catalogue_df: pd.DataFrame,
    product_ids: Sequence[str],
    recipe: str,
    *,
    omit_fields: Optional[Sequence[str]] = None,
    trained_count: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    baseline_reference_dir: Optional[Path] = Path("data/interim/static_enrichment_v1/text_embeddings"),
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Build or load cached BAAI/bge-m3 embeddings for an ablated t1_fam text recipe."""
    if omit_fields is None:
        if recipe not in RECIPE_TO_OMITTED_FIELDS:
            raise ValueError(f"Unknown recipe {recipe!r} and no omit_fields provided.")
        resolved_omit = RECIPE_TO_OMITTED_FIELDS[recipe]
    else:
        resolved_omit = tuple(omit_fields)

    root = Path(cache_dir) if cache_dir is not None else (CACHE_DIR / "text_embeddings")
    root.mkdir(parents=True, exist_ok=True)

    base_texts = render_recipe_texts(catalogue_df, product_ids, omit_fields=())
    product_texts = base_texts if not resolved_omit else render_recipe_texts(catalogue_df, product_ids, omit_fields=resolved_omit)

    recipe_hash = compute_text_recipe_hash(product_texts, recipe, model_name=model_name)
    content_hash = compute_text_content_hash(product_texts, model_name=model_name)
    base_content_hash = compute_text_content_hash(base_texts, model_name=model_name)

    changed_indices = [
        idx for idx, ((_, b_txt), (_, a_txt)) in enumerate(zip(base_texts, product_texts)) if b_txt != a_txt
    ]
    changed_text_count = len(changed_indices)
    if trained_count is not None:
        changed_trained_text_count = sum(1 for idx in changed_indices if idx < int(trained_count))
    else:
        changed_trained_text_count = None
    empty_text_count = sum(1 for _, txt in product_texts if not txt)
    non_empty_text_count = len(product_texts) - empty_text_count
    is_noop = (recipe != "t1_fam") and (changed_text_count == 0)

    parquet_path = root / f"{recipe}_{recipe_hash[:16]}.parquet"
    meta_path = root / f"{recipe}_{recipe_hash[:16]}.json"

    reused_from: Optional[str] = None
    if parquet_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("recipe_hash") != recipe_hash
            or meta.get("content_hash") != content_hash
            or meta.get("product_ids") != list(product_ids)
        ):
            raise ValueError(f"Text cache metadata mismatch at {meta_path}")
        emb_map = load_cached_embeddings(parquet_path, id_column="product_id")
    else:
        # Check if baseline t1_fam cache in static_enrichment_v1 matches exact content hash (read-only reuse).
        base_t1_hash = compute_text_recipe_hash(base_texts, "t1_fam", model_name=model_name)
        ref_parquet = (
            Path(baseline_reference_dir) / f"t1_fam_{base_t1_hash[:16]}.parquet"
            if baseline_reference_dir is not None
            else None
        )
        ref_meta = (
            Path(baseline_reference_dir) / f"t1_fam_{base_t1_hash[:16]}.json"
            if baseline_reference_dir is not None
            else None
        )
        if content_hash == base_content_hash and ref_parquet is not None and ref_parquet.is_file() and ref_meta.is_file():
            meta = json.loads(ref_meta.read_text(encoding="utf-8"))
            if meta.get("recipe_hash") == base_t1_hash and meta.get("product_ids") == list(product_ids):
                emb_map = load_cached_embeddings(ref_parquet, id_column="product_id")
                reused_from = str(ref_parquet)
            else:
                emb_map = None
        else:
            emb_map = None

        if emb_map is None:
            emb_map = encode_texts_to_embeddings(
                product_texts,
                model_name=model_name,
                encoder=encoder,
                batch_size=batch_size,
                device=device,
                embedding_dim=embedding_dim,
            )

        with NamedTemporaryFile(dir=root, suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_cached_embeddings(emb_map, tmp_path, id_column="product_id")
            tmp_path.replace(parquet_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        atomic_json(
            {
                "recipe": recipe,
                "omitted_fields": list(resolved_omit),
                "recipe_hash": recipe_hash,
                "content_hash": content_hash,
                "baseline_content_hash": base_content_hash,
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "product_count": len(product_ids),
                "trained_count": trained_count,
                "product_ids": list(product_ids),
                "changed_text_count": changed_text_count,
                "changed_trained_text_count": changed_trained_text_count,
                "empty_text_count": empty_text_count,
                "non_empty_text_count": non_empty_text_count,
                "is_noop": is_noop,
                "reused_from_baseline_cache": reused_from,
                "metadata_policy": "snapshot_unverified",
            },
            meta_path,
            overwrite=True,
        )

    tensor = torch.stack([torch.from_numpy(np.asarray(emb_map[str(pid)], dtype=np.float32)) for pid in product_ids])
    diagnostics = {
        "recipe": recipe,
        "omitted_fields": list(resolved_omit),
        "recipe_hash": recipe_hash,
        "content_hash": content_hash,
        "changed_text_count": changed_text_count,
        "changed_trained_text_count": changed_trained_text_count,
        "empty_text_count": empty_text_count,
        "non_empty_text_count": non_empty_text_count,
        "is_noop": is_noop,
        "cache_path": str(parquet_path),
        "meta_path": str(meta_path),
        "reused_from_baseline_cache": reused_from,
    }
    return tensor, diagnostics
