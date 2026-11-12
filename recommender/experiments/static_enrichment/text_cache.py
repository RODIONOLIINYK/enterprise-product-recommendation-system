"""Versioned product text recipes and BAAI/bge-m3 parquet embedding caches."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json
from recommender.text import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    load_cached_embeddings,
    load_text_encoder,
    save_cached_embeddings,
)

from .config import CACHE_DIR

TEXT_RECIPES = ("c0_s0", "t1", "t2", "t1_fam", "t2_fam")


def _clean_field(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return ""
    return text


def _dedupe_tokens(parts: Sequence[str]) -> str:
    """Case-insensitive token deduplication preserving first occurrence and surface form."""
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        for tok in part.split():
            key = tok.casefold().strip(",.;:!?()[]{}\"'")
            if not key:
                key = tok.casefold()
            if key not in seen:
                seen.add(key)
                kept.append(tok)
    return " ".join(kept)


def build_product_text(row: Union[pd.Series, Mapping[str, Any]], recipe: str) -> str:
    """Format product text according to a named static enrichment recipe."""
    if recipe not in TEXT_RECIPES:
        raise ValueError(f"Unknown text recipe: {recipe!r}")
    if recipe == "c0_s0":
        from recommender.text import get_product_text

        stripped = dict(row)
        stripped.pop("product_subcategory", None)
        return get_product_text(stripped)

    parts: list[str] = []
    brand = _clean_field(row.get("brand"))
    if brand:
        parts.append(brand)

    for name_col in ("product_name", "ukrainian_name", "short_name"):
        name_val = _clean_field(row.get(name_col))
        if name_val:
            parts.append(name_val)
            break

    for field in ("base_name", "product_type", "product_category", "product_subcategory", "business_line"):
        val = _clean_field(row.get(field))
        if val:
            parts.append(val)

    if recipe in ("t1_fam", "t2_fam"):
        fam = _clean_field(row.get("product_family"))
        if fam:
            parts.append(fam)

    prefix = _dedupe_tokens(parts)
    if recipe in ("t2", "t2_fam"):
        desc = _clean_field(row.get("cleaned_description"))
        if desc:
            return f"{prefix} {desc}".strip()
    return prefix


def compute_text_recipe_hash(
    product_texts: Sequence[tuple[str, str]],
    recipe: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Deterministic SHA-256 digest over recipe name, model name, and sorted (product_id, text) pairs."""
    digest = sha256()
    digest.update(json.dumps({"recipe": recipe, "model_name": model_name}, sort_keys=True).encode("utf-8"))
    for pid, txt in sorted(product_texts, key=lambda item: item[0]):
        digest.update(b"\x00")
        digest.update(pid.encode("utf-8"))
        digest.update(b"\x01")
        digest.update(txt.encode("utf-8"))
    return digest.hexdigest()


def encode_texts_to_embeddings(
    product_texts: Sequence[tuple[str, str]],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> Dict[str, np.ndarray]:
    """Encode (product_id, text) pairs into normalized float32 vectors."""
    non_empty = [(pid, txt) for pid, txt in product_texts if txt]
    embeddings_map: Dict[str, np.ndarray] = {
        pid: np.zeros(embedding_dim, dtype=np.float32)
        for pid, txt in product_texts
        if not txt
    }
    if non_empty:
        if encoder is None:
            encoder = load_text_encoder(model_name=model_name, device=device)
        encoded = encoder.encode(
            [txt for _, txt in non_empty],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        encoded = np.asarray(encoded, dtype=np.float32)
        for (pid, _), vec in zip(non_empty, encoded, strict=True):
            embeddings_map[pid] = vec.astype(np.float32)
    return embeddings_map


def get_or_build_product_text_embeddings(
    catalogue_df: pd.DataFrame,
    product_ids: Sequence[str],
    recipe: str,
    *,
    cache_dir: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[torch.Tensor, str, Path]:
    """Build or load cached product text embeddings keyed by the recipe content hash."""
    if recipe not in TEXT_RECIPES:
        raise ValueError(f"Unknown text recipe: {recipe!r}")
    root = Path(cache_dir) if cache_dir is not None else (CACHE_DIR / "text_embeddings")
    root.mkdir(parents=True, exist_ok=True)

    indexed = catalogue_df.drop_duplicates("product_id").set_index("product_id").reindex(list(product_ids))
    product_texts = [
        (str(pid), build_product_text(indexed.loc[pid], recipe) if pid in indexed.index else "")
        for pid in product_ids
    ]
    recipe_hash = compute_text_recipe_hash(product_texts, recipe, model_name=model_name)
    parquet_path = root / f"{recipe}_{recipe_hash[:16]}.parquet"
    meta_path = root / f"{recipe}_{recipe_hash[:16]}.json"

    if parquet_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("recipe_hash") != recipe_hash or meta.get("product_ids") != list(product_ids):
            raise ValueError(f"Text cache metadata mismatch at {meta_path}")
        emb_map = load_cached_embeddings(parquet_path, id_column="product_id")
    else:
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
        non_empty_desc = 0
        if "cleaned_description" in indexed.columns:
            non_empty_desc = int(indexed["cleaned_description"].fillna("").astype(str).str.strip().ne("").sum())
        atomic_json(
            {
                "recipe": recipe,
                "recipe_hash": recipe_hash,
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "product_count": len(product_ids),
                "product_ids": list(product_ids),
                "non_empty_text_count": sum(1 for _, t in product_texts if t),
                "non_empty_description_count": non_empty_desc,
                "metadata_policy": "snapshot_unverified",
            },
            meta_path,
            overwrite=True,
        )

    tensor = torch.stack([torch.from_numpy(np.asarray(emb_map[str(pid)], dtype=np.float32)) for pid in product_ids])
    return tensor, recipe_hash, parquet_path
