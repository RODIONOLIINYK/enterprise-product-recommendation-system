"""Text representation generation and offline caching using BAAI/bge-m3.

Extracts text profiles for products and companies, generates 1024-dimensional
dense text embeddings via sentence-transformers, and caches them offline on disk
for zero-redundancy reuse during training and inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_CACHE_DIR = Path("data/interim/embeddings")
CATALOGUE_CACHE_WITHOUT_SUBCATEGORY = "bge_m3_catalogue_without_subcategory.parquet"
COMPANY_CACHE_WITH_REGION = "bge_m3_company_with_region.parquet"


def get_product_text(row: Union[pd.Series, Dict[str, Any]]) -> str:
    """Format rich text description from product catalogue attributes."""
    tokens = []
    # Brand
    brand = row.get("brand")
    if pd.notna(brand) and str(brand).strip() and str(brand).strip().upper() != "NULL":
        tokens.append(str(brand).strip())

    # Names: expanded product_name or ukrainian_name or short_name
    for name_field in ["product_name", "ukrainian_name", "short_name"]:
        val = row.get(name_field)
        if pd.notna(val) and str(val).strip() and str(val).strip().upper() != "NULL":
            tokens.append(str(val).strip())
            break

    # Categories
    cat = row.get("product_category")
    if pd.notna(cat) and str(cat).strip() and str(cat).strip().upper() != "NULL":
        tokens.append(str(cat).strip())

    subcat = row.get("product_subcategory")
    if pd.notna(subcat) and str(subcat).strip() and str(subcat).strip().upper() != "NULL":
        tokens.append(str(subcat).strip())

    bline = row.get("business_line")
    if pd.notna(bline) and str(bline).strip() and str(bline).strip().upper() != "NULL":
        tokens.append(str(bline).strip())

    return " ".join(tokens).strip()


def get_company_text(
    row: Union[pd.Series, Dict[str, Any]], *, include_region: bool = False,
) -> str:
    """Format rich text description from company metadata."""
    tokens = []
    for field in ["company_name", "city", "description", "industry_branch"]:
        val = row.get(field)
        if pd.notna(val) and str(val).strip() and str(val).strip().upper() != "NULL":
            tokens.append(str(val).strip())
    region = row.get("region_code")
    if include_region and pd.notna(region) and str(region).strip().upper() not in ("", "NULL"):
        tokens.append(f"Region {str(region).strip()}")
    return " ".join(tokens).strip()


def load_text_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
):
    """Load BAAI/bge-m3 sentence-transformers encoder with appropriate device."""
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return SentenceTransformer(model_name, device=device)


def save_cached_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    cache_path: Union[str, Path],
    id_column: str = "product_id",
) -> None:
    """Serialize embedding dictionary to disk (supports .parquet and .pt)."""
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix == ".pt":
        # Convert numpy arrays to float32 tensors for pt format
        tensor_dict = {
            k: torch.from_numpy(v.astype(np.float32)) for k, v in embeddings_dict.items()
        }
        torch.save(tensor_dict, target)
    else:
        # Default to parquet format
        records = [
            {id_column: k, "embedding": v.astype(np.float32).tolist()}
            for k, v in embeddings_dict.items()
        ]
        df = pd.DataFrame(records)
        df.to_parquet(target, index=False)


def load_cached_embeddings(
    cache_path: Union[str, Path],
    id_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load offline cached embeddings from disk."""
    target = Path(cache_path)
    if not target.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    if target.suffix == ".pt":
        loaded = torch.load(target, map_location="cpu")
        return {
            str(k): v.numpy().astype(np.float32) if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
            for k, v in loaded.items()
        }
    else:
        df = pd.read_parquet(target)
        if id_column is not None and id_column in df.columns:
            key_col = id_column
        elif "product_id" in df.columns:
            key_col = "product_id"
        elif "customer_id" in df.columns:
            key_col = "customer_id"
        elif "id" in df.columns:
            key_col = "id"
        else:
            key_col = df.columns[0]

        keys = df[key_col].astype(str).tolist()
        embs = [np.asarray(e, dtype=np.float32) for e in df["embedding"]]
        return dict(zip(keys, embs))


def generate_company_embeddings(
    companies_df: pd.DataFrame,
    cache_path: Optional[Union[str, Path]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    force_refresh: bool = False,
    *,
    include_region: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate 1024-d float32 embeddings for company/customer profiles with offline caching."""
    if cache_path is not None and not force_refresh:
        path = Path(cache_path)
        if path.is_file():
            return load_cached_embeddings(path, id_column="customer_id")

    if "customer_id" not in companies_df.columns:
        raise ValueError("Companies DataFrame must contain 'customer_id' column.")

    unique_companies = companies_df.drop_duplicates("customer_id").copy()

    texts: List[str] = []
    customer_ids: List[str] = []
    empty_indices: List[int] = []

    for idx, (_, row) in enumerate(unique_companies.iterrows()):
        cid = str(row["customer_id"])
        txt = get_company_text(row, include_region=include_region)
        customer_ids.append(cid)
        texts.append(txt)
        if not txt:
            empty_indices.append(idx)

    # Encode non-empty texts
    non_empty_texts = [t for i, t in enumerate(texts) if i not in empty_indices]
    embeddings_map: Dict[str, np.ndarray] = {}

    if non_empty_texts:
        if encoder is None:
            encoder = load_text_encoder(model_name=model_name, device=device)

        encoded = encoder.encode(
            non_empty_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        encoded = np.asarray(encoded, dtype=np.float32)

        non_empty_ptr = 0
        for idx, cid in enumerate(customer_ids):
            if idx in empty_indices:
                embeddings_map[cid] = np.zeros(embedding_dim, dtype=np.float32)
            else:
                vec = encoded[non_empty_ptr]
                embeddings_map[cid] = vec.astype(np.float32)
                non_empty_ptr += 1
    else:
        for cid in customer_ids:
            embeddings_map[cid] = np.zeros(embedding_dim, dtype=np.float32)

    if cache_path is not None:
        save_cached_embeddings(embeddings_map, cache_path, id_column="customer_id")

    return embeddings_map


def generate_catalogue_embeddings(
    catalogue_df: pd.DataFrame,
    cache_path: Optional[Union[str, Path]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    force_refresh: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate 1024-d float32 embeddings for catalogue items with offline caching."""
    if cache_path is not None and not force_refresh:
        path = Path(cache_path)
        if path.is_file():
            return load_cached_embeddings(path)

    if "product_id" not in catalogue_df.columns:
        raise ValueError("Catalogue DataFrame must contain 'product_id' column.")

    unique_products = catalogue_df.drop_duplicates("product_id").copy()

    texts: List[str] = []
    product_ids: List[str] = []
    empty_indices: List[int] = []

    for idx, (_, row) in enumerate(unique_products.iterrows()):
        pid = str(row["product_id"])
        txt = get_product_text(row)
        product_ids.append(pid)
        texts.append(txt)
        if not txt:
            empty_indices.append(idx)

    # Encode non-empty texts
    non_empty_texts = [t for i, t in enumerate(texts) if i not in empty_indices]
    embeddings_map: Dict[str, np.ndarray] = {}

    if non_empty_texts:
        if encoder is None:
            encoder = load_text_encoder(model_name=model_name, device=device)

        encoded = encoder.encode(
            non_empty_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        encoded = np.asarray(encoded, dtype=np.float32)

        non_empty_ptr = 0
        for idx, pid in enumerate(product_ids):
            if idx in empty_indices:
                embeddings_map[pid] = np.zeros(embedding_dim, dtype=np.float32)
            else:
                vec = encoded[non_empty_ptr]
                embeddings_map[pid] = vec.astype(np.float32)
                non_empty_ptr += 1
    else:
        for pid in product_ids:
            embeddings_map[pid] = np.zeros(embedding_dim, dtype=np.float32)

    if cache_path is not None:
        save_cached_embeddings(embeddings_map, cache_path)

    return embeddings_map
