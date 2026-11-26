"""Isolated product text rendering, token-budget enforcement, and content-hashed BGE-M3 caching."""

from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.embedding_ablation.text_cache import (
    build_ablated_product_text,
    compute_text_content_hash,
)
from recommender.experiments.static_enrichment.config import DEFAULT_DESCRIPTION_PATH
from recommender.experiments.static_enrichment.metadata import clean_html_description
from recommender.experiments.static_enrichment.text_cache import (
    _clean_field,
    compute_text_recipe_hash,
    encode_texts_to_embeddings,
)
from recommender.text import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    load_cached_embeddings,
    load_text_encoder,
    save_cached_embeddings,
)

from .cleaner import (
    CLEANER_VERSION,
    CLEANING_RULES_HASH,
    _TECHNICAL_SIGNAL_RE,
    clean_description_v1,
    extract_spec_tokens,
)
from .config import BASELINE_TEXT_CACHE_DIR, CACHE_DIR, DESCRIPTION_TEXT_RECIPES

TRUNCATION_POLICY_NAME = "prefix_plus_description_block_priority_v1"
DEFAULT_MAX_SEQ_LENGTH = 8192


def _text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _description_for_recipe(row: Union[pd.Series, Mapping[str, Any]], recipe: str) -> str:
    if recipe == "t1_fam":
        return ""
    if recipe == "t1_fam_desc_html_v1":
        val = row.get("legacy_html_description")
        if val is not None and not (isinstance(val, float) and pd.isna(val)) and not pd.isna(val):
            s = _clean_field(val)
            if s:
                return s
        raw = row.get("raw_description")
        return clean_html_description(raw) if raw is not None and not pd.isna(raw) else ""
    if recipe == "t1_fam_desc_clean_v1":
        val = row.get("cleaned_description_v1")
        if val is not None and not (isinstance(val, float) and pd.isna(val)) and not pd.isna(val):
            s = _clean_field(val)
            if s:
                return s
        raw = row.get("raw_description")
        return clean_description_v1(raw) if raw is not None and not pd.isna(raw) else ""
    raise ValueError(f"Unsupported description recipe {recipe!r}; expected one of {DESCRIPTION_TEXT_RECIPES}.")


def build_description_product_text(
    row: Union[pd.Series, Mapping[str, Any]],
    recipe: str,
) -> str:
    """Render exact t1_fam prefix plus optional HTML/cleaned description, falling back to exact prefix."""
    if recipe not in DESCRIPTION_TEXT_RECIPES:
        raise ValueError(f"Unsupported description recipe {recipe!r}; expected one of {DESCRIPTION_TEXT_RECIPES}.")
    prefix = build_ablated_product_text(row, omit_fields=())
    if recipe == "t1_fam":
        return prefix
    desc = _description_for_recipe(row, recipe)
    if not desc:
        return prefix
    if not prefix:
        return desc
    return f"{prefix} {desc}"


def _count_tokens(tokenizer: Any, text: str) -> Optional[int]:
    if tokenizer is None or not text:
        return 0 if not text else None
    try:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return int(len(ids))
    except TypeError:
        ids = tokenizer.encode(text)
        return int(len(ids))


def apply_token_budget_policy(
    prefix: str,
    desc: str,
    *,
    tokenizer: Any = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> tuple[str, Optional[int], bool]:
    """Enforce deterministic token budget prioritizing exact t1_fam prefix and spec/table blocks."""
    full_text = f"{prefix} {desc}".strip() if (prefix and desc) else (prefix or desc)
    if tokenizer is None or not full_text:
        return full_text, (0 if not full_text else None), False

    tok_count = _count_tokens(tokenizer, full_text)
    if tok_count is not None and tok_count <= int(max_seq_length):
        return full_text, tok_count, False

    if not desc:
        return full_text, tok_count, False

    # Priority 1: spec/table/approval blocks; Priority 2: remaining blocks in original order.
    blocks = [b.strip() for b in re.split(r"\n+", desc) if b.strip()]
    prioritized = [
        (0 if (_TECHNICAL_SIGNAL_RE.search(b) or " | " in b) else 1, idx, b)
        for idx, b in enumerate(blocks)
    ]
    prioritized.sort(key=lambda item: (item[0], item[1]))

    selected_indices: list[int] = []
    current_text = prefix
    for _, idx, _ in prioritized:
        trial_indices = sorted([*selected_indices, idx])
        trial_desc = "\n".join(blocks[i] for i in trial_indices)
        trial_text = f"{prefix} {trial_desc}".strip() if prefix else trial_desc
        trial_tokens = _count_tokens(tokenizer, trial_text)
        if trial_tokens is not None and trial_tokens <= int(max_seq_length):
            selected_indices = trial_indices
            current_text = trial_text

    final_tokens = _count_tokens(tokenizer, current_text)
    return current_text, final_tokens, True


def render_description_texts(
    catalogue_df: pd.DataFrame,
    product_ids: Sequence[str],
    recipe: str,
    *,
    tokenizer: Any = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> tuple[list[tuple[str, str]], Dict[str, Any]]:
    """Render ordered (product_id, text) pairs and record token/truncation/spec diagnostics."""
    if recipe not in DESCRIPTION_TEXT_RECIPES:
        raise ValueError(f"Unsupported description recipe {recipe!r}; expected one of {DESCRIPTION_TEXT_RECIPES}.")
    indexed = catalogue_df.drop_duplicates("product_id").set_index("product_id").reindex(list(product_ids))

    rendered: list[tuple[str, str]] = []
    token_counts: list[int] = []
    truncated_pids: list[str] = []
    char_lengths: list[int] = []
    spec_counts: list[int] = []

    for pid in product_ids:
        pid_str = str(pid)
        if pid not in indexed.index:
            rendered.append((pid_str, ""))
            char_lengths.append(0)
            spec_counts.append(0)
            if tokenizer is not None:
                token_counts.append(0)
            continue
        row = indexed.loc[pid]
        prefix = build_ablated_product_text(row, omit_fields=())
        desc = _description_for_recipe(row, recipe)
        txt, tok_n, was_trunc = apply_token_budget_policy(
            prefix,
            desc,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        rendered.append((pid_str, txt))
        char_lengths.append(len(txt))
        spec_counts.append(len(extract_spec_tokens(desc)))
        if tok_n is not None:
            token_counts.append(tok_n)
        if was_trunc:
            truncated_pids.append(pid_str)

    diag: Dict[str, Any] = {
        "max_chars": max(char_lengths, default=0),
        "mean_chars": float(np.mean(char_lengths)) if char_lengths else 0.0,
        "non_empty_spec_products": sum(1 for c in spec_counts if c > 0),
        "total_spec_tokens_in_descriptions": int(sum(spec_counts)),
        "tokenizer_measured": bool(tokenizer is not None),
        "encoder_max_seq_length": int(max_seq_length),
        "max_tokens": max(token_counts) if token_counts else None,
        "mean_tokens": float(np.mean(token_counts)) if token_counts else None,
        "p95_tokens": float(np.percentile(token_counts, 95)) if token_counts else None,
        "truncated_count": len(truncated_pids),
        "truncated_product_ids": truncated_pids,
        "truncation_policy": TRUNCATION_POLICY_NAME,
    }
    return rendered, diag


def _load_verified_baseline_vectors(
    baseline_reference_dir: Optional[Path],
    base_texts: Sequence[tuple[str, str]],
    product_ids: Sequence[str],
    model_name: str,
) -> tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    if baseline_reference_dir is None:
        return None, None
    base_t1_hash = compute_text_recipe_hash(base_texts, "t1_fam", model_name=model_name)
    base_content_hash = compute_text_content_hash(base_texts, model_name=model_name)
    ref_parquet = Path(baseline_reference_dir) / f"t1_fam_{base_t1_hash[:16]}.parquet"
    ref_meta = Path(baseline_reference_dir) / f"t1_fam_{base_t1_hash[:16]}.json"
    if not (ref_parquet.is_file() and ref_meta.is_file()):
        return None, None
    meta = json.loads(ref_meta.read_text(encoding="utf-8"))
    if (
        meta.get("recipe_hash") != base_t1_hash
        or meta.get("content_hash") != base_content_hash
        or meta.get("model_name") != model_name
        or meta.get("product_ids") != list(product_ids)
    ):
        return None, None
    return load_cached_embeddings(ref_parquet, id_column="product_id"), str(ref_parquet)


def get_or_build_description_text_embeddings(
    catalogue_df: pd.DataFrame,
    product_ids: Sequence[str],
    recipe: str,
    *,
    trained_count: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    baseline_reference_dir: Optional[Path] = BASELINE_TEXT_CACHE_DIR,
    raw_description_path: Optional[Path] = DEFAULT_DESCRIPTION_PATH,
    measure_tokens_with_encoder: bool = True,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Build or load content-hashed BAAI/bge-m3 embeddings, reusing unchanged no-description texts safely."""
    if recipe not in DESCRIPTION_TEXT_RECIPES:
        raise ValueError(f"Unsupported description recipe {recipe!r}; expected one of {DESCRIPTION_TEXT_RECIPES}.")

    root = Path(cache_dir) if cache_dir is not None else (CACHE_DIR / "text_embeddings")
    root.mkdir(parents=True, exist_ok=True)

    # First render without tokenizer to derive base texts and check existing cache hit before loading encoder.
    base_texts, _ = render_description_texts(catalogue_df, product_ids, "t1_fam")
    prelim_texts, prelim_diag = render_description_texts(catalogue_df, product_ids, recipe)
    prelim_recipe_hash = compute_text_recipe_hash(prelim_texts, recipe, model_name=model_name)
    prelim_content_hash = compute_text_content_hash(prelim_texts, model_name=model_name)

    cleaner_ver = CLEANER_VERSION if recipe == "t1_fam_desc_clean_v1" else ("clean_html_description_v1" if recipe == "t1_fam_desc_html_v1" else "none")
    rules_hash = CLEANING_RULES_HASH if recipe == "t1_fam_desc_clean_v1" else ("legacy_html_v1" if recipe == "t1_fam_desc_html_v1" else "none")
    raw_desc_sha = file_digest(raw_description_path) if (raw_description_path is not None and Path(raw_description_path).is_file()) else None

    prelim_parquet = root / f"{recipe}_{prelim_recipe_hash[:16]}.parquet"
    prelim_meta_path = root / f"{recipe}_{prelim_recipe_hash[:16]}.json"

    if prelim_parquet.is_file() and prelim_meta_path.is_file():
        meta = json.loads(prelim_meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("recipe_hash") == prelim_recipe_hash
            and meta.get("content_hash") == prelim_content_hash
            and meta.get("product_ids") == list(product_ids)
            and meta.get("model_name") == model_name
            and meta.get("cleaner_version") == cleaner_ver
            and meta.get("cleaner_rules_hash") == rules_hash
        ):
            emb_map = load_cached_embeddings(prelim_parquet, id_column="product_id")
            tensor = torch.stack([torch.from_numpy(np.asarray(emb_map[str(pid)], dtype=np.float32)) for pid in product_ids])
            diag = {
                "recipe": recipe,
                "cleaner_version": cleaner_ver,
                "cleaner_rules_hash": rules_hash,
                "raw_description_sha256": raw_desc_sha,
                "recipe_hash": meta["recipe_hash"],
                "content_hash": meta["content_hash"],
                "base_content_hash": meta["base_content_hash"],
                "changed_text_count": meta["changed_text_count"],
                "changed_trained_text_count": meta["changed_trained_text_count"],
                "reused_unchanged_count": meta["reused_unchanged_count"],
                "newly_encoded_count": meta["newly_encoded_count"],
                "empty_text_count": meta["empty_text_count"],
                "non_empty_text_count": meta["non_empty_text_count"],
                "reused_from_baseline_cache": meta.get("reused_from_baseline_cache"),
                "token_stats": meta.get("token_stats", prelim_diag),
                "cache_path": str(prelim_parquet),
                "meta_path": str(prelim_meta_path),
                "loaded_from_cache": True,
            }
            return tensor, diag

    # Cache miss: resolve encoder (or mock encoder) and measure actual tokenizer counts if available.
    active_encoder = encoder
    tokenizer = getattr(active_encoder, "tokenizer", None) if active_encoder is not None else None
    max_seq_len = int(getattr(active_encoder, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH) or DEFAULT_MAX_SEQ_LENGTH)
    if active_encoder is None and measure_tokens_with_encoder and recipe != "t1_fam":
        active_encoder = load_text_encoder(model_name=model_name, device=device)
        tokenizer = getattr(active_encoder, "tokenizer", None)
        max_seq_len = int(getattr(active_encoder, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH) or DEFAULT_MAX_SEQ_LENGTH)

    product_texts, token_stats = render_description_texts(
        catalogue_df,
        product_ids,
        recipe,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
    )
    recipe_hash = compute_text_recipe_hash(product_texts, recipe, model_name=model_name)
    content_hash = compute_text_content_hash(product_texts, model_name=model_name)
    base_content_hash = compute_text_content_hash(base_texts, model_name=model_name)

    parquet_path = root / f"{recipe}_{recipe_hash[:16]}.parquet"
    meta_path = root / f"{recipe}_{recipe_hash[:16]}.json"

    changed_indices = [
        idx for idx, ((_, b_txt), (_, a_txt)) in enumerate(zip(base_texts, product_texts)) if b_txt != a_txt
    ]
    changed_text_count = len(changed_indices)
    changed_trained_text_count = (
        sum(1 for idx in changed_indices if idx < int(trained_count))
        if trained_count is not None
        else None
    )
    empty_text_count = sum(1 for _, txt in product_texts if not txt)
    non_empty_text_count = len(product_texts) - empty_text_count

    baseline_map, reused_from = _load_verified_baseline_vectors(
        baseline_reference_dir,
        base_texts,
        product_ids,
        model_name,
    )
    if baseline_map is None and root != baseline_reference_dir:
        baseline_map, reused_from = _load_verified_baseline_vectors(
            root,
            base_texts,
            product_ids,
            model_name,
        )

    emb_map: Dict[str, np.ndarray] = {}
    to_encode: list[tuple[str, str]] = []
    if baseline_map is not None:
        for (pid, b_txt), (_, cur_txt) in zip(base_texts, product_texts):
            if cur_txt == b_txt and pid in baseline_map:
                emb_map[pid] = np.asarray(baseline_map[pid], dtype=np.float32)
            else:
                to_encode.append((pid, cur_txt))
    else:
        to_encode = list(product_texts)

    reused_unchanged_count = len(emb_map)
    newly_encoded_count = len(to_encode)
    if to_encode:
        if active_encoder is None:
            active_encoder = load_text_encoder(model_name=model_name, device=device)
        encoded_subset = encode_texts_to_embeddings(
            to_encode,
            model_name=model_name,
            encoder=active_encoder,
            batch_size=batch_size,
            device=device,
            embedding_dim=embedding_dim,
        )
        emb_map.update(encoded_subset)

    with NamedTemporaryFile(dir=root, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_cached_embeddings(emb_map, tmp_path, id_column="product_id")
        tmp_path.replace(parquet_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    per_text_sha256 = {pid: _text_sha256(txt) for pid, txt in product_texts}
    atomic_json(
        {
            "recipe": recipe,
            "cleaner_version": cleaner_ver,
            "cleaner_rules_hash": rules_hash,
            "raw_description_sha256": raw_desc_sha,
            "recipe_hash": recipe_hash,
            "content_hash": content_hash,
            "base_content_hash": base_content_hash,
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "product_count": len(product_ids),
            "trained_count": trained_count,
            "product_ids": list(product_ids),
            "per_text_sha256": per_text_sha256,
            "changed_text_count": changed_text_count,
            "changed_trained_text_count": changed_trained_text_count,
            "reused_unchanged_count": reused_unchanged_count,
            "newly_encoded_count": newly_encoded_count,
            "empty_text_count": empty_text_count,
            "non_empty_text_count": non_empty_text_count,
            "reused_from_baseline_cache": reused_from,
            "token_stats": token_stats,
            "metadata_policy": "snapshot_unverified",
        },
        meta_path,
        overwrite=True,
    )

    tensor = torch.stack([torch.from_numpy(np.asarray(emb_map[str(pid)], dtype=np.float32)) for pid in product_ids])
    diag = {
        "recipe": recipe,
        "cleaner_version": cleaner_ver,
        "cleaner_rules_hash": rules_hash,
        "raw_description_sha256": raw_desc_sha,
        "recipe_hash": recipe_hash,
        "content_hash": content_hash,
        "base_content_hash": base_content_hash,
        "changed_text_count": changed_text_count,
        "changed_trained_text_count": changed_trained_text_count,
        "reused_unchanged_count": reused_unchanged_count,
        "newly_encoded_count": newly_encoded_count,
        "empty_text_count": empty_text_count,
        "non_empty_text_count": non_empty_text_count,
        "reused_from_baseline_cache": reused_from,
        "token_stats": token_stats,
        "cache_path": str(parquet_path),
        "meta_path": str(meta_path),
        "loaded_from_cache": False,
    }
    return tensor, diag
