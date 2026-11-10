"""Metadata embeddings for Stage 1 candidates with and without prior sales."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

from recommender.candidates import CandidateRetriever
from recommender.catalogue import RollingCandidatePool
from recommender.complementarity import ProductComplementarity
from recommender.scope import actual_purchase_mask
from recommender.text import (
    CATALOGUE_CACHE_WITHOUT_SUBCATEGORY,
    DEFAULT_CACHE_DIR,
    generate_catalogue_embeddings,
    get_product_text,
    load_cached_embeddings,
    load_text_encoder,
    save_cached_embeddings,
)
from recommender.train_retriever import build_retriever_product_inputs


def load_expanded_catalogue_embeddings(
    catalogue: pd.DataFrame, cache_dir: Path, device: str,
) -> dict[str, np.ndarray]:
    """Reuse reference text vectors and cache new vectors in resumable batches."""
    reference_path = DEFAULT_CACHE_DIR / CATALOGUE_CACHE_WITHOUT_SUBCATEGORY
    reference = load_cached_embeddings(reference_path, "product_id") if reference_path.is_file() else {}
    catalogue = catalogue.drop(columns="product_subcategory", errors="ignore")
    catalogue = catalogue.sort_values("product_id")
    ids = set(catalogue["product_id"].astype(str))
    embeddings = {sku: vector for sku, vector in reference.items() if sku in ids}
    missing = catalogue.loc[~catalogue["product_id"].astype(str).isin(embeddings)]
    encoder = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), 512):
        chunk = missing.iloc[start:start + 512]
        fingerprint = sha256()
        for _, row in chunk.iterrows():
            fingerprint.update(repr((row["product_id"], get_product_text(row))).encode())
        path = cache_dir / f"{fingerprint.hexdigest()}.parquet"
        if path.is_file():
            vectors = load_cached_embeddings(path, "product_id")
        else:
            if encoder is None:
                encoder = load_text_encoder(device=device)
            vectors = generate_catalogue_embeddings(chunk, encoder=encoder, batch_size=64)
            with NamedTemporaryFile(dir=cache_dir, suffix=".parquet", delete=False) as temporary:
                tmp_path = Path(temporary.name)
            try:
                save_cached_embeddings(vectors, tmp_path, id_column="product_id")
                tmp_path.replace(path)
            finally:
                tmp_path.unlink(missing_ok=True)
        embeddings.update(vectors)
        print(f"Catalog text embeddings ready: {len(embeddings):,}/{len(ids):,}", flush=True)
    return embeddings


def load_stage1_discovery(
    checkpoint_path: Path,
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    embedding_cache: Path,
    device: torch.device,
) -> tuple[CandidateRetriever, dict]:
    """Load the Stage 1 checkpoint and metadata vectors for its rolling catalogue."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    if not config.get("discovery_only") or config.get("customer_price_reference") is not None:
        raise ValueError("Use the Stage 1 discovery reference with the standard company profile.")
    purchases = purchases.loc[actual_purchase_mask(purchases)]
    pool = RollingCandidatePool(catalogue, purchases)
    trained_ids = ckpt["product_ids"]
    if not set(trained_ids) <= pool.product_ids:
        raise ValueError("The merchandise catalogue must include the reference SKU IDs.")
    candidate_ids = trained_ids + sorted(pool.product_ids - set(trained_ids))
    embeddings = load_expanded_catalogue_embeddings(pool.merchandise, embedding_cache, str(device))
    complementarity = ProductComplementarity(purchases) if config.get("include_complementarity") else None
    inputs = build_retriever_product_inputs(
        purchases, pool.merchandise, embeddings, candidate_ids, ckpt["vocabs"],
        include_subcategory=False, container_encoding=config["container_encoding"],
        complementarity=complementarity,
    )
    retriever = CandidateRetriever.from_checkpoint(
        checkpoint_path, purchases, catalogue, companies, inputs, device,
        candidate_product_ids=candidate_ids, rolling_pool=pool,
        product_complementarity=complementarity,
    )
    return retriever, ckpt
