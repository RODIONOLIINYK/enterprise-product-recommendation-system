"""Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker (UPS-CANR) Training Pipeline.

Implements:
1. Candidate slate precomputation and caching with the same candidate policy for both splits.
2. RankerDataset and DataLoader for grouped transition batches (B=64, C=50).
3. Hybrid ranking objective: Multi-Label BCEWithLogitsLoss (pos_weight=10.0) +
   Listwise Softmax Cross-Entropy loss.
4. Validation metrics: Recall@{5, 10, 20}, NDCG@10, MRR, plus cross-attention weight tracking
   (Token 0 Profile vs Tokens 1..32 Past Purchases).
5. Checkpointing: writes a fresh checkpoints/ranker/runs/<run-id>/ directory,
   with a manifest identifying its qualifying selection. Legacy aliases are preserved.
"""

import hashlib
import argparse
import json
import math
import pickle
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recommender.candidates import (
    DEFAULT_HEURISTIC_FRACTION, PAIRWISE_FEATURE_NAMES, CandidateRetriever,
)
from recommender.catalogue import load_product_catalogue, order_retriever_catalogue
from recommender.company import load_companies
from recommender.features import (
    LEGACY_PROFILE_FEATURES,
    MEDIAN_BASKET_QUANTITY,
    MEDIAN_PURCHASE_QUANTITY,
    PointInTimeNormalizer,
    compute_point_in_time_features,
)
from recommender.artifacts import (
    CACHE_FORMAT_VERSION, CHECKPOINT_FORMAT_VERSION, atomic_json, create_run_directory, feature_contract,
    json_digest, ranker_provenance, seal_slates, validate_slates,
)
from recommender.development import load_development_purchases
from recommender.models import TEXT_DIM
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.scope import DEFAULT_SCOPE_PATH, get_datasets_locs
from recommender.sequences import (
    DEFAULT_SEQUENCE_LENGTH,
    ENRICHED_FEATURE_DIM,
    ENRICHED_FEATURE_NAMES,
    PURCHASE_DATE_ORIGIN,
    build_customer_sequence,
)
from recommender.split import TEST_START
from recommender.stage1_catalogue import load_stage1_discovery
from recommender.text import (
    CATALOGUE_CACHE_WITHOUT_SUBCATEGORY,
    DEFAULT_CACHE_DIR,
    load_cached_embeddings,
)
from recommender.train_retriever import (
    build_customer_metadata,
    build_retriever_product_inputs,
    build_retriever_transitions,
    get_device,
)


SLATE_SCHEMA_VERSION = 3
DEFAULT_EXCLUDED_STATISTICS: Tuple[str, ...] = ("history_repeat",)


class RankerDataset(Dataset):
    """Dataset holding precomputed candidate slates and customer memory tensors for Stage 2 ranker."""

    def __init__(self, data: Dict[str, Any]):
        self.text_embedding = data["text_embedding"]          # (N, 1024)
        self.city_id = data["city_id"]                        # (N,)
        self.region_id = data["region_id"]                    # (N,)
        self.industry_id = data["industry_id"]                # (N,)
        self.tabular_features = data["tabular_features"]      # (N, 8)
        self.retriever_tabular_features = data.get("retriever_tabular_features", self.tabular_features)
        self.sequence = data["sequence"]                      # (N, 32, 9)
        self.attention_mask = data["attention_mask"]          # (N, 32)
        self.seq_product_ids = data["seq_product_ids"]        # (N, 32)
        self.seq_purchase_dates = data.get("seq_purchase_dates")
        self.prediction_days = data.get("prediction_days")
        self.seq_category_ids = data["seq_category_ids"]      # (N, 32)
        self.seq_subcategory_ids = data.get(
            "seq_subcategory_ids",
            torch.zeros_like(self.seq_product_ids),
        )
        self.seq_business_line_ids = data.get(
            "seq_business_line_ids",
            torch.zeros_like(self.seq_product_ids),
        )
        self.candidate_indices = data["candidate_indices"]    # (N, 50)
        self.candidate_vectors = data.get("candidate_vectors")
        self.pairwise_features = data["pairwise_features"]    # (N, 50, F)
        self.labels = data["labels"]                          # (N, 50)
        self.candidate_pids = data["candidate_pids"]          # List of length N (list of 50 strings)
        self.positive_pids = data["positive_pids"]            # List of length N (list of strings)
        self.new_positive_pids = data.get("new_positive_pids")
        self.customer_ids = data["customer_ids"]              # List of length N (strings)
        self.stage1_customer_vectors = data.get("stage1_customer_vectors")
        self.seq_catalog_indices = data.get("seq_catalog_indices")
        self.seq_canonical_skus = data.get("seq_canonical_skus")

    def __len__(self) -> int:
        return len(self.text_embedding)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text_embedding": self.text_embedding[idx],
            "city_id": self.city_id[idx],
            "region_id": self.region_id[idx],
            "industry_id": self.industry_id[idx],
            "tabular_features": self.tabular_features[idx],
            "retriever_tabular_features": self.retriever_tabular_features[idx],
            "sequence": self.sequence[idx],
            "attention_mask": self.attention_mask[idx],
            "seq_product_ids": self.seq_product_ids[idx],
            "seq_purchase_dates": (
                self.seq_purchase_dates[idx] if self.seq_purchase_dates is not None else None
            ),
            "prediction_day": self.prediction_days[idx] if self.prediction_days is not None else None,
            "seq_category_ids": self.seq_category_ids[idx],
            "seq_subcategory_ids": self.seq_subcategory_ids[idx],
            "seq_business_line_ids": self.seq_business_line_ids[idx],
            "candidate_indices": self.candidate_indices[idx],
            "candidate_vectors": self.candidate_vectors[idx] if self.candidate_vectors is not None else None,
            "pairwise_features": self.pairwise_features[idx],
            "labels": self.labels[idx],
            "candidate_pids": self.candidate_pids[idx],
            "positive_pids": self.positive_pids[idx],
            "new_positive_pids": self.new_positive_pids[idx] if self.new_positive_pids is not None else None,
            "customer_id": self.customer_ids[idx],
            "stage1_customer_vectors": self.stage1_customer_vectors[idx] if self.stage1_customer_vectors is not None else None,
            "seq_catalog_indices": self.seq_catalog_indices[idx] if self.seq_catalog_indices is not None else None,
            "seq_canonical_skus": self.seq_canonical_skus[idx] if self.seq_canonical_skus is not None else None,
        }


def collate_ranker_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate individual samples into batched tensors and metadata lists."""
    return {
        "text_embedding": torch.stack([x["text_embedding"] for x in batch], dim=0),
        "city_id": torch.stack([x["city_id"] for x in batch], dim=0),
        "region_id": torch.stack([x["region_id"] for x in batch], dim=0),
        "industry_id": torch.stack([x["industry_id"] for x in batch], dim=0),
        "tabular_features": torch.stack([x["tabular_features"] for x in batch], dim=0),
        "retriever_tabular_features": torch.stack(
            [x.get("retriever_tabular_features", x["tabular_features"]) for x in batch], dim=0,
        ),
        "sequence": torch.stack([x["sequence"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "seq_product_ids": torch.stack([x["seq_product_ids"] for x in batch], dim=0),
        "seq_purchase_dates": (
            torch.stack([x["seq_purchase_dates"] for x in batch], dim=0)
            if batch[0].get("seq_purchase_dates") is not None else None
        ),
        "prediction_day": (
            torch.stack([x["prediction_day"] for x in batch])
            if batch[0].get("prediction_day") is not None else None
        ),
        "seq_category_ids": torch.stack([x["seq_category_ids"] for x in batch], dim=0),
        "seq_subcategory_ids": torch.stack([x["seq_subcategory_ids"] for x in batch], dim=0),
        "seq_business_line_ids": torch.stack([x["seq_business_line_ids"] for x in batch], dim=0),
        "candidate_indices": torch.stack([x["candidate_indices"] for x in batch], dim=0),
        "candidate_vectors": (
            torch.stack([x["candidate_vectors"] for x in batch]) if batch[0].get("candidate_vectors") is not None else None
        ),
        "pairwise_features": torch.stack([x["pairwise_features"] for x in batch], dim=0),
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
        "candidate_pids": [x["candidate_pids"] for x in batch],
        "positive_pids": [x["positive_pids"] for x in batch],
        "new_positive_pids": (
            [x["new_positive_pids"] for x in batch]
            if batch[0].get("new_positive_pids") is not None else None
        ),
        "customer_ids": [x["customer_id"] for x in batch],
        **{
            name: torch.stack([x[name] for x in batch]) if batch[0].get(name) is not None else None
            for name in ("stage1_customer_vectors", "seq_catalog_indices")
        },
        "seq_canonical_skus": [x.get("seq_canonical_skus") for x in batch],
    }


def ranker_batch_inputs(batch, product_vectors, model, device):
    """Transfer only model inputs; prefer already cached candidate vectors.

    The catalogue is needed only for the indexed fallback. Retriever profiles
    and reporting metadata stay on the CPU, and shared inputs remain opt-in.
    """
    names = ("text_embedding", "city_id", "region_id", "industry_id", "tabular_features",
             "sequence", "attention_mask", "seq_product_ids", "seq_category_ids",
             "seq_subcategory_ids", "seq_business_line_ids", "pairwise_features",
             "seq_purchase_dates", "prediction_day")
    inputs = {name: batch[name].to(device) if batch.get(name) is not None else None for name in names}
    vectors = batch.get("candidate_vectors")
    shared = getattr(model, "requires_shared_stage1_inputs", False)
    indices = batch["candidate_indices"].to(device) if vectors is None or shared else None
    inputs["candidate_vectors"] = product_vectors[indices] if vectors is None else vectors.to(device)
    if shared:
        inputs.update(stage1_customer_vectors=batch["stage1_customer_vectors"].to(device),
                      seq_catalog_indices=batch["seq_catalog_indices"].to(device), candidate_indices=indices)
    return inputs


def precompute_candidate_slates_for_transitions(
    transitions: List[Dict[str, Any]],
    retriever: CandidateRetriever,
    product_vectors: torch.Tensor,
    cust_groups: Dict[str, pd.DataFrame],
    customer_metadata: Dict[str, Dict[str, Any]],
    customer_embeddings: Dict[str, np.ndarray],
    normalizer: PointInTimeNormalizer,
    vocabs: Dict[str, Dict[str, int]],
    item_metadata: Dict[str, Dict[str, Any]],
    top_c: int = 50,
    batch_size: int = 128,
    *,
    heuristic_fraction: float = 0.0,
    capture_stage1_vectors: bool = False,
    stage1_checkpoint_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Precompute full customer memory tensors and CandidateRetriever slates for a transition list."""
    n_transitions = len(transitions)
    device = retriever.device
    if capture_stage1_vectors and not stage1_checkpoint_digest:
        raise ValueError("Shared vector capture requires an exact Stage 1 checkpoint digest.")
    canonical_history = []
    customer_vectors = []

    all_text_embs: List[np.ndarray] = []
    all_city_ids: List[int] = []
    all_region_ids: List[int] = []
    all_industry_ids: List[int] = []
    all_tab_feats: List[List[float]] = []
    all_retriever_tab_feats: List[List[float]] = []
    all_sequences: List[np.ndarray] = []
    all_attention_masks: List[np.ndarray] = []
    all_seq_product_ids: List[np.ndarray] = []
    all_seq_purchase_dates: List[np.ndarray] = []
    all_seq_category_ids: List[np.ndarray] = []
    all_seq_subcat_ids: List[np.ndarray] = []
    all_seq_line_ids: List[np.ndarray] = []
    retriever_length = retriever.model.customer_tower.max_seq_len
    sequence_fields = (
        "sequence", "attention_mask", "seq_product_ids", "seq_category_ids", "seq_purchase_dates",
    )
    separate_retriever_sequence = retriever_length != DEFAULT_SEQUENCE_LENGTH
    retriever_rows = {name: [] for name in sequence_fields}

    # 1. Build customer memory feature representations
    for t in transitions:
        cid = str(t["customer_id"])
        as_of = t["as_of_date"]
        if capture_stage1_vectors and pd.Timestamp(as_of) >= TEST_START:
            raise ValueError("Experimental vector capture excludes held-out prediction dates.")
        grp = cust_groups.get(cid, pd.DataFrame())

        # Company text
        c_text = customer_embeddings.get(cid)
        if c_text is None:
            c_text = np.zeros(TEXT_DIM, dtype=np.float32)
        all_text_embs.append(c_text)

        # Categoricals
        meta = customer_metadata.get(cid, {})
        all_city_ids.append(int(meta.get("city_id", 0)))
        all_region_ids.append(int(meta.get("region_id", 0)))
        all_industry_ids.append(int(meta.get("industry_id", 0)))

        # Tabular lifetime features
        pit = compute_point_in_time_features(
            grp, cid, as_of, normalizer=normalizer,
            include_median_quantity=MEDIAN_BASKET_QUANTITY in retriever.profile_features,
            include_median_purchase_quantity=MEDIAN_PURCHASE_QUANTITY in retriever.profile_features,
        )
        all_tab_feats.append([float(pit[name]) for name in LEGACY_PROFILE_FEATURES])
        all_retriever_tab_feats.append([float(pit[name]) for name in retriever.profile_features])

        # Sequences
        seq_res = build_customer_sequence(
            grp,
            customer_id=cid,
            as_of_date=as_of,
            max_len=max(DEFAULT_SEQUENCE_LENGTH, retriever_length),
            feature_dim=ENRICHED_FEATURE_DIM,
            vocabularies=vocabs,
            item_metadata=item_metadata,
            enriched=True,
            quantity_reference=retriever.quantity_reference,
        )
        if separate_retriever_sequence:
            num_events = int(seq_res["attention_mask"].sum())
            retriever_start = max(0, num_events - retriever_length)
            for name in sequence_fields:
                retriever_rows[name].append(seq_res[name][retriever_start:retriever_start + retriever_length])
            ranker_start = max(0, num_events - DEFAULT_SEQUENCE_LENGTH)
            seq_res = {
                name: values[ranker_start:ranker_start + DEFAULT_SEQUENCE_LENGTH]
                for name, values in seq_res.items()
            }
        all_sequences.append(seq_res["sequence"].numpy())
        all_attention_masks.append(seq_res["attention_mask"].numpy())
        all_seq_product_ids.append(seq_res["seq_product_ids"].numpy())
        all_seq_purchase_dates.append(seq_res["seq_purchase_dates"].numpy())
        all_seq_category_ids.append(seq_res["seq_category_ids"].numpy())
        all_seq_subcat_ids.append(seq_res["seq_subcategory_ids"].numpy())
        all_seq_line_ids.append(seq_res["seq_business_line_ids"].numpy())
        if capture_stage1_vectors:
            canonical_history.append(seq_res["product_ids"])

    t_text = torch.from_numpy(np.array(all_text_embs, dtype=np.float32))
    t_city = torch.tensor(all_city_ids, dtype=torch.long)
    t_region = torch.tensor(all_region_ids, dtype=torch.long)
    t_industry = torch.tensor(all_industry_ids, dtype=torch.long)
    t_tab = torch.tensor(all_tab_feats, dtype=torch.float32)
    t_retriever_tab = torch.tensor(all_retriever_tab_feats, dtype=torch.float32)
    t_seq = torch.from_numpy(np.array(all_sequences, dtype=np.float32))
    t_mask = torch.from_numpy(np.array(all_attention_masks, dtype=np.int64))
    t_pids = torch.from_numpy(np.array(all_seq_product_ids, dtype=np.int64))
    t_dates = torch.from_numpy(np.array(all_seq_purchase_dates, dtype=np.float32))
    t_cids = torch.from_numpy(np.array(all_seq_category_ids, dtype=np.int64))
    t_sub_ids = torch.from_numpy(np.array(all_seq_subcat_ids, dtype=np.int64))
    t_line_ids = torch.from_numpy(np.array(all_seq_line_ids, dtype=np.int64))
    retriever_sequence = (
        {name: torch.stack(rows) for name, rows in retriever_rows.items()}
        if separate_retriever_sequence else {}
    )
    tower_inputs = retriever_sequence or {
        "sequence": t_seq, "attention_mask": t_mask, "seq_product_ids": t_pids,
        "seq_category_ids": t_cids, "seq_purchase_dates": t_dates,
    }

    # 2. Batched Stage 1 retrieval and CandidateBatch generation
    all_cand_indices: List[torch.Tensor] = []
    all_pairwise_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_cand_pids: List[List[str]] = []
    all_pos_pids: List[List[str]] = []
    all_cust_ids: List[str] = []
    batches = [list(range(i, min(i + batch_size, n_transitions))) for i in range(0, n_transitions, batch_size)]

    retriever.model.eval()
    with torch.no_grad():
        for rows in batches:
            batch_slice = [transitions[i] for i in rows]
            b_text = t_text[rows].to(device)
            b_city = t_city[rows].to(device)
            b_reg = t_region[rows].to(device)
            b_ind = t_industry[rows].to(device)
            b_tab = t_retriever_tab[rows].to(device)
            u_c = retriever.model.customer_tower(
                text_embedding=b_text,
                city_id=b_city,
                region_id=b_reg,
                industry_id=b_ind,
                tabular_features=b_tab,
                catalog_product_vectors=product_vectors,
                **{name: tensor[rows].to(device) for name, tensor in tower_inputs.items()},
            )
            if capture_stage1_vectors:
                customer_vectors.append(u_c.detach().cpu())

            b_cids_list = [str(t["customer_id"]) for t in batch_slice]
            b_dates = [t["as_of_date"] for t in batch_slice]
            b_pos_pids = [t["positive_product_ids"] for t in batch_slice]

            logits = retriever.compute_logits(
                u_c, product_vectors, customer_ids=b_cids_list, as_of_dates=b_dates,
            )

            batch_candidates = retriever.generate_candidate_batch(
                logits=logits,
                customer_ids=b_cids_list,
                as_of_dates=b_dates,
                positive_product_ids=b_pos_pids,
                top_k=top_c if retriever.rolling_pool is not None else 100,
                top_c=top_c,
                heuristic_fraction=heuristic_fraction,
            )
            if batch_candidates.candidate_indices.size(1) != top_c and retriever.rolling_pool is not None:
                raise ValueError("Rolling ranker slates require enough eligible candidates for every basket.")

            all_cand_indices.append(batch_candidates.candidate_indices.cpu())
            all_pairwise_feats.append(batch_candidates.pairwise_features.cpu())
            all_labels.append(batch_candidates.labels.cpu())
            all_cand_pids.extend(batch_candidates.candidate_pids)
            all_pos_pids.extend(b_pos_pids)
            all_cust_ids.extend(b_cids_list)

    order = torch.argsort(torch.tensor([i for rows in batches for i in rows]))
    stacked_cand_idx = torch.cat(all_cand_indices, dim=0)[order]
    stacked_pw_feats = torch.cat(all_pairwise_feats, dim=0)[order]
    stacked_labels = torch.cat(all_labels, dim=0)[order]

    result = {
        "text_embedding": t_text,
        "city_id": t_city,
        "region_id": t_region,
        "industry_id": t_industry,
        "tabular_features": t_tab,
        "retriever_tabular_features": t_retriever_tab,
        "sequence": t_seq,
        "attention_mask": t_mask,
        "seq_product_ids": t_pids,
        "seq_purchase_dates": t_dates,
        "seq_category_ids": t_cids,
        "seq_subcategory_ids": t_sub_ids,
        "seq_business_line_ids": t_line_ids,
        "candidate_indices": stacked_cand_idx,
        "pairwise_features": stacked_pw_feats,
        "labels": stacked_labels,
        "candidate_pids": [all_cand_pids[i] for i in order.tolist()],
        "positive_pids": [all_pos_pids[i] for i in order.tolist()],
        "customer_ids": [all_cust_ids[i] for i in order.tolist()],
    }
    result["prediction_days"] = prediction_day_offsets(transitions)
    if retriever_sequence:
        result["retriever_sequence"] = retriever_sequence
    if capture_stage1_vectors:
        from recommender.shared_ranker import map_history_skus, schema_digest

        result.update({
            "stage1_customer_vectors": torch.cat(customer_vectors)[order],
            "seq_canonical_skus": canonical_history,
            "seq_catalog_indices": map_history_skus(canonical_history, retriever.eligible_product_ids),
            "prediction_days": prediction_day_offsets(transitions),
            "shared_vector_provenance": {
                "schema_digest": schema_digest(), "stage1_digest": stage1_checkpoint_digest,
                "product_ids": list(retriever.eligible_product_ids),
                "customer_inputs": "per-transition strictly prior customer events; frozen Stage1",
                "stage1_demand_policy": "fixed_train_cutoff_snapshot; inherited training look-ahead",
            },
        })
    return result


def slate_cache_path(
    split: str, heuristic_fraction: float = 0.0, *, rolling: bool = False, metadata_policy: str = "snapshot_unverified",
) -> Path:
    mix = f"_repeat_{heuristic_fraction}" if heuristic_fraction else ""
    pool = "_rolling_v1" if rolling else ""
    pool += "_observed_only_v1" if metadata_policy == "observed_only_v1" else ""
    return Path("data/interim") / f"ranker_{split}_slates_v{CACHE_FORMAT_VERSION}{mix}{pool}.pt"


def checkpoint_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def company_embeddings_path(checkpoint_path: Path, cache_name: str = "bge_m3_company.parquet") -> Path:
    local_path = checkpoint_path.parent / cache_name
    return local_path if local_path.is_file() else DEFAULT_CACHE_DIR / cache_name


def catalogue_embeddings_path(include_subcategory: bool) -> Path:
    filename = "bge_m3_catalogue.parquet" if include_subcategory else CATALOGUE_CACHE_WITHOUT_SUBCATEGORY
    return DEFAULT_CACHE_DIR / filename


def slate_cache_identity(
    retriever_digest: str,
    checkpoint_path: Path,
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    *,
    include_subcategory: bool,
    heuristic_fraction: float = 0.0,
    company_cache_name: str = "bge_m3_company.parquet",
    rolling: bool = False,
    product_inputs: Optional[Dict[str, torch.Tensor]] = None,
    metadata_policy: str = "snapshot_unverified",
) -> str:
    """Bind slates to source data, ordered schemas and actual catalogue inputs.

    Rolling retrieval includes text vectors from expanded-catalogue shards, which
    are absent from the base embedding files. Hashing the resolved input tensors
    covers those vectors and their order without depending on unused shard files.
    Schema version 3 payloads/checkpoints remain readable; the identity revision
    deliberately rejects older, incompletely fingerprinted cache identities.
    """
    if rolling and product_inputs is None:
        raise ValueError("Rolling slate identity requires resolved product inputs.")
    digest = hashlib.sha256(f"identity_v2:{SLATE_SCHEMA_VERSION}:{retriever_digest}".encode())
    digest.update(json.dumps({
        "pairwise": PAIRWISE_FEATURE_NAMES,
        "profile": LEGACY_PROFILE_FEATURES,
        "history": ENRICHED_FEATURE_NAMES,
    }, sort_keys=True).encode())
    if metadata_policy != "snapshot_unverified":
        digest.update(f":pairwise_metadata:{metadata_policy}".encode())
    digest.update(checkpoint_digest(DEFAULT_SCOPE_PATH).encode())
    for name, tensor in sorted((product_inputs or {}).items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(repr((name, tuple(values.shape), str(values.dtype))).encode())
        digest.update(values.numpy().tobytes())
    if heuristic_fraction:
        digest.update(f":repeat_plus_popularity:{heuristic_fraction}".encode())
    if rolling:
        digest.update(b":rolling_v1")
    for path in (catalogue_embeddings_path(include_subcategory), company_embeddings_path(checkpoint_path, company_cache_name)):
        digest.update(checkpoint_digest(path).encode())
    for frame in (purchases, catalogue, companies):
        digest.update(repr(tuple(frame.dtypes.items())).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def load_ranker_retriever(
    checkpoint_path: Path,
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    device: torch.device,
    *, metadata_policy: Optional[str] = None,
) -> Tuple[CandidateRetriever, Dict[str, Dict[str, int]]]:
    """Use the same ordered catalogue and demand inputs as retriever training."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if ckpt.get("config", {}).get("discovery_only"):
        retriever, ckpt = load_stage1_discovery(
            checkpoint_path, purchases, catalogue, companies,
            Path("checkpoints/stage1-full-catalogue/text_embeddings"), device, metadata_policy=metadata_policy,
        )
        return retriever, ckpt["vocabs"]
    product_ids = ckpt["product_ids"]
    vocabs = ckpt["vocabs"]
    catalogue = order_retriever_catalogue(catalogue, product_ids)
    include_subcategory = "product_tower.subcat_embedding.weight" in ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    cat_embs = load_cached_embeddings(
        catalogue_embeddings_path(include_subcategory), id_column="product_id",
    )
    product_inputs = build_retriever_product_inputs(
        purchases, catalogue, cat_embs, product_ids, vocabs, include_subcategory=include_subcategory,
        include_sku_median_quantity=config.get("include_sku_median_quantity", False),
        demand_policy=config.get("demand_policy", "legacy_train_cutoff_v1"),
        container_encoding=config.get("container_encoding", "one_hot"),
    )
    retriever = CandidateRetriever.from_checkpoint(
        checkpoint_path,
        purchases_df=purchases,
        catalogue_df=catalogue,
        companies_df=companies,
        product_inputs=product_inputs,
        device=device,
        metadata_policy=metadata_policy,
    )
    return retriever, vocabs


def prepare_slate_inputs(
    retriever: CandidateRetriever,
    checkpoint_path: Path,
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    train_transitions: List[Dict[str, Any]],
    vocabs: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Prepare the customer features used identically by ranker training and evaluation."""
    customer_vocabs = retriever.customer_vocabs
    if not {"city_vocab", "region_vocab", "industry_vocab"} <= customer_vocabs.keys():
        customer_vocabs = {
            key: {value: i + 1 for i, value in enumerate(sorted(companies[column].dropna().unique()))}
            for key, column in (
                ("city_vocab", "city"),
                ("region_vocab", "region_code"),
                ("industry_vocab", "industry_branch"),
            )
        }
    groups = {str(cid): group for cid, group in purchases.groupby("customer_id")}
    normalizer = retriever.profile_normalizer
    if normalizer is None:
        # Legacy checkpoints used these training-derived bounds and the full profile.
        rows = [
            compute_point_in_time_features(
                groups[str(t["customer_id"])], str(t["customer_id"]), t["as_of_date"],
            )
            for t in train_transitions[:1000]
        ]
        normalizer = PointInTimeNormalizer()
        normalizer.fit(pd.DataFrame(rows))
    item_metadata = (
        order_retriever_catalogue(catalogue, retriever.eligible_product_ids)
        .set_index("product_id")[["product_category", "product_subcategory", "business_line"]]
        .to_dict(orient="index")
    )
    return {
        "retriever": retriever,
        "product_vectors": retriever.get_product_vectors(),
        "cust_groups": groups,
        "customer_metadata": build_customer_metadata(companies, customer_vocabs),
        "customer_embeddings": load_cached_embeddings(
            company_embeddings_path(checkpoint_path, retriever.company_cache_name), id_column="customer_id",
        ),
        "normalizer": normalizer,
        "vocabs": vocabs,
        "item_metadata": item_metadata,
    }


def versioned_cache_path(path, identity):
    stamp = json_digest({"identity": identity, "contract": feature_contract(), "format": CACHE_FORMAT_VERSION})[:16]
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def load_slate_cache(path: Path, identity: str, *, transitions=None, product_ids=None, split=None, candidates=None) -> Optional[Dict[str, Any]]:
    path = Path(path)
    for source in (path, versioned_cache_path(path, identity)):
        try:
            data = torch.load(source, map_location="cpu")
            if not isinstance(data, dict) or data.get("cache_identity") != identity:
                continue
            validate_slates(data, transitions=transitions, product_ids=product_ids, split=split, candidates=candidates)
            data["cache_path"] = str(source)
            return data
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            continue
    return None


def build_slate_cache(
    path: Path,
    identity: str,
    transitions: List[Dict[str, Any]],
    slate_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    if not transitions:
        raise ValueError(f"No transitions available to build {path.name}.")
    data = precompute_candidate_slates_for_transitions(
        transitions=transitions, **slate_inputs,
    )
    path = Path(path)
    if path.exists():
        path = versioned_cache_path(path, identity)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing cache: {path}")
    data["cache_identity"] = identity
    data["cache_path"] = str(path)
    data["pairwise_metadata_policy"] = getattr(slate_inputs["retriever"].feature_store, "metadata_policy", "snapshot_unverified")
    seal_slates(data, transitions, slate_inputs["retriever"].eligible_product_ids)
    validate_slates(data, transitions=transitions, product_ids=slate_inputs["retriever"].eligible_product_ids,
                    split=identity.rsplit(":", 1)[-1], candidates=slate_inputs.get("top_c", 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            torch.save(data, temporary)
            temporary.flush()
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                existing = load_slate_cache(path, identity, transitions=transitions,
                                            product_ids=slate_inputs["retriever"].eligible_product_ids,
                                            split=identity.rsplit(":", 1)[-1], candidates=slate_inputs.get("top_c", 50))
                if existing is None:
                    raise
                return existing
        finally:
            temporary_path.unlink(missing_ok=True)
    return data


def load_or_precompute_slates(
    purchases_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
    companies_df: pd.DataFrame,
    checkpoint_path: Path,
    train_cache_path: Path,
    val_cache_path: Path,
    device: torch.device,
    *,
    heuristic_fraction: float = 0.0,
    metadata_policy: str = "snapshot_unverified",
) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, List[str], Dict[str, Dict[str, int]]]:
    """Load or generate slates for both splits using the configured mix."""
    retriever, vocabs = load_ranker_retriever(
        checkpoint_path, purchases_df, catalogue_df, companies_df, device, metadata_policy=metadata_policy,
    )
    retriever_digest = checkpoint_digest(checkpoint_path)
    identity = slate_cache_identity(
        retriever_digest, checkpoint_path, purchases_df, catalogue_df, companies_df,
        include_subcategory=retriever.model.product_tower.subcat_embedding is not None,
        heuristic_fraction=heuristic_fraction,
        company_cache_name=retriever.company_cache_name,
        rolling=retriever.rolling_pool is not None,
        product_inputs=retriever.product_inputs,
        metadata_policy=metadata_policy,
    )
    train_transitions, val_transitions, _ = build_retriever_transitions(purchases_df)
    if retriever.rolling_pool is not None:
        dates = {t["as_of_date"] for t in train_transitions}
        full_slate_dates = {date for date in dates if len(retriever.rolling_pool.eligible_on(date)) >= 50}
        before = len(train_transitions)
        train_transitions = [t for t in train_transitions if t["as_of_date"] in full_slate_dates]
        print(f"Excluded {before - len(train_transitions)} training baskets with fewer than 50 eligible SKUs.")
    train_data = load_slate_cache(train_cache_path, f"{identity}:train", transitions=train_transitions,
                                 product_ids=retriever.eligible_product_ids, split="train", candidates=50)
    val_data = load_slate_cache(val_cache_path, f"{identity}:val", transitions=val_transitions,
                               product_ids=retriever.eligible_product_ids, split="val", candidates=50)
    if train_data is None or val_data is None:
        slate_inputs = prepare_slate_inputs(
            retriever, checkpoint_path, purchases_df, catalogue_df, companies_df, train_transitions, vocabs,
        )
        slate_inputs["heuristic_fraction"] = heuristic_fraction
        if train_data is None:
            print(f"Precomputing {len(train_transitions)} training slates from retrieved candidates")
            train_data = build_slate_cache(
                train_cache_path, f"{identity}:train", train_transitions, slate_inputs,
            )
        if val_data is None:
            print(f"Precomputing {len(val_transitions)} validation slates from retrieved candidates")
            val_data = build_slate_cache(
                val_cache_path, f"{identity}:val", val_transitions, slate_inputs,
            )
    # Discovery labels are deliberately attached outside the slate cache so existing
    # expensive candidate caches remain reusable when this metric is introduced.
    attach_new_positive_metadata(train_data, train_transitions, purchases_df)
    attach_new_positive_metadata(val_data, val_transitions, purchases_df)
    product_vectors = retriever.get_product_vectors()
    return train_data, val_data, product_vectors, retriever.eligible_product_ids, vocabs


def prediction_day_offsets(transitions: Sequence[Dict[str, Any]]) -> torch.Tensor:
    """Calendar days from the same origin used by the stored purchase dates."""
    return torch.tensor([
        (pd.Timestamp(row["as_of_date"]).tz_localize(None).normalize() - PURCHASE_DATE_ORIGIN).days
        for row in transitions
    ], dtype=torch.float32)


def attach_new_positive_metadata(
    data: Dict[str, Any], transitions: Sequence[Dict[str, Any]], purchases_df: pd.DataFrame,
) -> None:
    """Attach full-history unseen target IDs without changing slate cache schemas."""
    if len(data["customer_ids"]) != len(transitions):
        raise ValueError("Cached slates and chronological transitions have different lengths.")

    expected_days = prediction_day_offsets(transitions)
    if data.get("prediction_days") is not None and not torch.equal(data["prediction_days"], expected_days):
        raise ValueError("Cached slate prediction dates differ from source transitions.")
    history_by_customer = {
        str(cid): group[["purchase_date", "product_id"]].copy()
        for cid, group in purchases_df.groupby("customer_id", sort=False)
    }
    new_positive_pids: List[List[str]] = []
    for cid, positives, transition in zip(
        data["customer_ids"], data["positive_pids"], transitions, strict=True,
    ):
        expected_positives = sorted(str(pid) for pid in positives)
        transition_positives = sorted(str(pid) for pid in transition["positive_product_ids"])
        if str(cid) != str(transition["customer_id"]) or expected_positives != transition_positives:
            raise ValueError("Cached slate row order does not match chronological transition order.")
        as_of = pd.Timestamp(transition["as_of_date"])
        history = history_by_customer.get(str(cid))
        if history is None:
            prior_products: set[str] = set()
        else:
            prior = history.loc[history["purchase_date"] < as_of, "product_id"]
            prior_products = set(prior.astype(str))
        new_positive_pids.append([str(pid) for pid in positives if str(pid) not in prior_products])
    data["new_positive_pids"] = new_positive_pids
    data["prediction_days"] = prediction_day_offsets(transitions)


def evaluate_ranker(
    model: UPSCANRRanker,
    val_loader: DataLoader,
    catalog_product_vectors: torch.Tensor,
    device: torch.device,
    loss_module: RankerLoss,
    k_list: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    """Evaluate Stage 2 ranker on validation loader, computing Recall@K, NDCG@10, MRR, and attention allocation."""
    model.eval()
    total_val_loss = 0.0
    total_unweighted_val_loss = 0.0
    val_batches = 0

    recalls: Dict[int, List[float]] = {k: [] for k in k_list}
    ndcg_list: List[float] = []
    mrr_list: List[float] = []
    token0_attns: List[float] = []
    history_attns: List[float] = []
    new_target_count = 0
    new_hits: Dict[int, int] = {k: 0 for k in k_list}
    discovery_available = True

    cat_vectors_dev = catalog_product_vectors.to(device)
    unweighted_loss_module = RankerLoss(
        pos_weight=loss_module.pos_weight, listwise_weight=loss_module.listwise_weight,
    ).to(device)

    with torch.no_grad():
        for batch in val_loader:
            b_size = batch["text_embedding"].size(0)

            inputs = ranker_batch_inputs(batch, cat_vectors_dev, model, device)
            labels = batch["labels"].to(device)
            unseen_mask = inputs["pairwise_features"][..., 0] == 0
            out = model(**inputs)

            loss_dict = loss_module(out.logits, labels, unseen_mask=unseen_mask)
            total_val_loss += float(loss_dict["loss"].item())
            total_unweighted_val_loss += float(unweighted_loss_module(out.logits, labels)["loss"].item())
            val_batches += 1

            # Track attention distribution (Token 0 vs Tokens 1..32)
            t0 = float(out.token_0_weight.mean().item())
            th = float(out.history_weights.mean().item())
            token0_attns.append(t0)
            history_attns.append(th)

            # Ranking metrics against true target basket
            logits_np = out.logits.cpu().numpy()
            for b in range(b_size):
                targets = set(batch["positive_pids"][b])
                if not targets:
                    continue
                c_pids = batch["candidate_pids"][b]
                row_new_targets = batch.get("new_positive_pids")
                if row_new_targets is None:
                    discovery_available = False
                    new_targets: set[str] = set()
                else:
                    new_targets = set(row_new_targets[b])
                    new_target_count += len(new_targets)
                ranks = np.argsort(-logits_np[b])
                reranked = [c_pids[i] for i in ranks]

                # Recall@K
                for k in k_list:
                    hits = sum(1 for p in reranked[:k] if p in targets)
                    recalls[k].append(hits / len(targets))
                    new_hits[k] += sum(1 for p in reranked[:k] if p in new_targets)

                # NDCG@10
                dcg = sum(1.0 / math.log2(r + 2) for r, p in enumerate(reranked[:10]) if p in targets)
                idcg = sum(1.0 / math.log2(r + 2) for r in range(min(10, len(targets))))
                ndcg_list.append((dcg / idcg) if idcg > 0 else 0.0)

                # MRR
                rr = 0.0
                for r, p in enumerate(reranked):
                    if p in targets:
                        rr = 1.0 / (r + 1)
                        break
                mrr_list.append(rr)

    res = {
        "val_loss": total_val_loss / max(val_batches, 1),
        "Val Loss": total_val_loss / max(val_batches, 1),
        "val_loss_unweighted": total_unweighted_val_loss / max(val_batches, 1),
        "Val Loss Unweighted": total_unweighted_val_loss / max(val_batches, 1),
        "avg_token0_weight": float(np.mean(token0_attns)) if token0_attns else 0.0,
        "avg_history_weight": float(np.mean(history_attns)) if history_attns else 0.0,
    }
    for k in k_list:
        v = float(np.mean(recalls[k])) if recalls[k] else 0.0
        res[f"val_recall@{k}"] = v
        res[f"Val Recall@{k}"] = v

    v_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    res["val_ndcg@10"] = v_ndcg
    res["Val NDCG@10"] = v_ndcg

    v_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    res["val_mrr"] = v_mrr
    res["Val MRR"] = v_mrr
    if discovery_available:
        res["new_target_count"] = int(new_target_count)
        for k in k_list:
            hits = int(new_hits[k])
            new_recall = float(hits / new_target_count) if new_target_count else 0.0
            res[f"new_hits@{k}"] = hits
            res[f"new_recall@{k}"] = new_recall
            res[f"Val New Recall@{k}"] = new_recall

    return res


def train_ranker(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 10.0,
    listwise_weight: float = 1.0,
    device: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints/ranker",
    retriever_checkpoint_path: Union[str, Path] = "checkpoints/retriever-discovery-reference/best_retriever.pt",
    *,
    heuristic_fraction: float = DEFAULT_HEURISTIC_FRACTION,
    statistics_attention: bool = True,
    excluded_statistics: tuple[str, ...] = DEFAULT_EXCLUDED_STATISTICS,
    new_positive_weight: float = 1.0,
    selection_metric: str = "discovery",
    discovery_min_overall_recall: Optional[float] = None,
    seed: int = 42,
    run_id: Optional[str] = None,
    cached_only_training: bool = True,
    pairwise_metadata_policy: str = "snapshot_unverified",
) -> Dict[str, Any]:
    """Train Stage 2 UPS-CANR Ranker for 10 epochs and save checkpoints and history."""
    if not 0.0 <= heuristic_fraction <= 1.0:
        raise ValueError("heuristic_fraction must be between 0 and 1.")
    if selection_metric not in {"ndcg", "discovery"}:
        raise ValueError("selection_metric must be 'ndcg' or 'discovery'.")
    if discovery_min_overall_recall is not None and not 0.0 <= discovery_min_overall_recall <= 1.0:
        raise ValueError("discovery_min_overall_recall must be between 0 and 1.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = get_device(device)
    print(f"Using compute device: {dev}")

    ckpt_dir = create_run_directory(checkpoint_dir, run_id)
    best_checkpoint_path = ckpt_dir / "best_ranker.pt"
    best_ndcg_checkpoint_path = ckpt_dir / "best_ndcg_ranker.pt"
    best_discovery_checkpoint_path = ckpt_dir / "best_discovery_ranker.pt"
    history_path = ckpt_dir / "training_history.json"

    # 1. Load dataframes
    p_path = Path("data/interim/cleaned_purchases.csv")
    purchases_df = load_development_purchases(p_path)
    catalogue_df = load_product_catalogue(get_datasets_locs("items"))
    companies_df = load_companies(purchases=purchases_df, minimum_purchases=1)
    retriever_checkpoint_path = Path(retriever_checkpoint_path)
    retriever_digest = checkpoint_digest(retriever_checkpoint_path)
    retriever_config = torch.load(retriever_checkpoint_path, map_location="cpu").get("config", {})
    rolling = bool(retriever_config.get("discovery_only"))

    # 2. Load or precompute candidate slates
    train_data, val_data, catalog_product_vectors, eligible_pids, vocabs = load_or_precompute_slates(
        purchases_df=purchases_df,
        catalogue_df=catalogue_df,
        companies_df=companies_df,
        checkpoint_path=retriever_checkpoint_path,
        train_cache_path=slate_cache_path("train", heuristic_fraction, rolling=rolling, metadata_policy=pairwise_metadata_policy),
        val_cache_path=slate_cache_path("val", heuristic_fraction, rolling=rolling, metadata_policy=pairwise_metadata_policy),
        device=dev,
        heuristic_fraction=heuristic_fraction,
        metadata_policy=pairwise_metadata_policy,
    )

    train_dataset = RankerDataset(train_data)
    val_dataset = RankerDataset(val_data)
    print(f"Datasets initialized: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranker_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranker_batch,
        num_workers=0,
    )

    # 3. Model setup
    model = UPSCANRRanker.from_retriever_checkpoint(
        checkpoint_path=retriever_checkpoint_path,
        device=dev,
        num_pairwise_features=train_data["pairwise_features"].shape[-1],
        statistics_attention=statistics_attention,
        excluded_statistics=excluded_statistics,
        cached_only=cached_only_training,
    )
    if checkpoint_digest(retriever_checkpoint_path) != retriever_digest:
        raise RuntimeError("The retriever checkpoint changed during ranker setup. Restart ranker training.")
    model.to(dev)

    loss_fn = RankerLoss(
        pos_weight=pos_weight, listwise_weight=listwise_weight,
        new_positive_weight=new_positive_weight,
    ).to(dev)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    catalog_product_vectors_dev = catalog_product_vectors.to(dev)

    training_history: List[Dict[str, Any]] = []
    best_val_ndcg: float = -1.0
    best_new_recall: float = -1.0
    best_discovery_epoch: Optional[int] = None

    print(f"Beginning Stage 2 UPS-CANR ranker training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            inputs = ranker_batch_inputs(batch, catalog_product_vectors_dev, model, dev)
            labels = batch["labels"].to(dev)
            unseen_mask = inputs["pairwise_features"][..., 0] == 0

            optimizer.zero_grad()
            out = model(**inputs)

            loss_dict = loss_fn(out.logits, labels, unseen_mask=unseen_mask)
            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += float(loss.item())
            train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(train_batches, 1)

        # 4. Evaluate Validation Metrics
        val_metrics = evaluate_ranker(
            model=model,
            val_loader=val_loader,
            catalog_product_vectors=catalog_product_vectors_dev,
            device=dev,
            loss_module=loss_fn,
            k_list=(5, 10, 20),
        )

        rec5 = val_metrics["val_recall@5"]
        rec10 = val_metrics["val_recall@10"]
        rec20 = val_metrics["val_recall@20"]
        ndcg10 = val_metrics["val_ndcg@10"]
        mrr = val_metrics["val_mrr"]
        val_loss = val_metrics["val_loss"]
        new_recall10 = val_metrics.get("new_recall@10")
        t0_attn = val_metrics["avg_token0_weight"]
        hist_attn = val_metrics["avg_history_weight"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_recall@5": rec5,
            "val_recall@10": rec10,
            "val_recall@20": rec20,
            "val_ndcg@10": ndcg10,
            "val_mrr": mrr,
            "val_new_recall@10": new_recall10,
            "val_new_hits@10": val_metrics.get("new_hits@10"),
            "val_new_target_count": val_metrics.get("new_target_count"),
            "avg_token0_weight": t0_attn,
            "avg_history_weight": hist_attn,
            "Train Loss": avg_train_loss,
            "Val Loss": val_loss,
            "Val Recall@5": rec5,
            "Val Recall@10": rec10,
            "Val Recall@20": rec20,
            "Val NDCG@10": ndcg10,
            "Val MRR": mrr,
            "Val New Recall@10": new_recall10,
            "Val New Hits@10": val_metrics.get("new_hits@10"),
            "Val New Target Count": val_metrics.get("new_target_count"),
        }
        training_history.append(epoch_record)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recall@5: {rec5:.4f} | "
            f"Val Recall@10: {rec10:.4f} | "
            f"Val Recall@20: {rec20:.4f} | "
            f"Val NDCG@10: {ndcg10:.4f} | "
            f"Val New Recall@10: {new_recall10 if new_recall10 is not None else float('nan'):.4f} "
            f"({val_metrics.get('new_hits@10', 0)}/{val_metrics.get('new_target_count', 0)}) | "
            f"Val MRR: {mrr:.4f} | "
            f"Token0 Attn: {t0_attn:.4f} | "
            f"Hist Attn: {hist_attn:.4f}"
        )

        # 5. Save independent NDCG and discovery selections, plus selected alias.
        def checkpoint_payload(metric_name: str, metric_value: float) -> Dict[str, Any]:
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ndcg": best_val_ndcg,
                "best_metric_name": metric_name,
                "best_metric_value": metric_value,
                "selection_metric": metric_name,
                "run_selection_metric": selection_metric,
                "new_positive_weight": new_positive_weight,
                "discovery_min_overall_recall": discovery_min_overall_recall,
                "val_metrics": val_metrics,
                "product_ids": eligible_pids,
                "vocabs": vocabs,
                "retriever_digest": retriever_digest,
                "slate_schema_version": SLATE_SCHEMA_VERSION,
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "heuristic_fraction": heuristic_fraction,
                "statistics_attention": statistics_attention,
                "excluded_statistics": excluded_statistics,
                "rolling_candidate_pool": rolling,
                "config": {
                    "hidden_dim": model.hidden_dim,
                    "num_pairwise_features": model.num_pairwise_features,
                    "num_tabular_features": model.num_tabular_features,
                    "seq_feature_dim": model.seq_feature_dim,
                    "text_dim": model.text_dim,
                    "statistics_attention": statistics_attention,
                    "excluded_statistics": list(excluded_statistics),
                    "cached_only": cached_only_training,
                    "metadata_policy": "snapshot_unverified",
                    "pairwise_metadata_policy": pairwise_metadata_policy,
                    "num_heads": model.num_heads,
                    "dropout": 0.1,
                    "pos_weight": pos_weight,
                    "listwise_weight": listwise_weight,
                    "new_positive_weight": new_positive_weight,
                    "seed": seed,
                    "selection_metric": metric_name,
                },
            }
            payload["provenance"] = ranker_provenance(
                retriever_digest=retriever_digest, train_data=train_data, val_data=val_data,
                model_config=payload["config"], selection_metric=metric_name,
                metadata_policy=pairwise_metadata_policy,
            )
            return payload

        if ndcg10 > best_val_ndcg:
            best_val_ndcg = ndcg10
            payload = checkpoint_payload("ndcg@10", ndcg10)
            torch.save(payload, best_ndcg_checkpoint_path)
            if selection_metric == "ndcg":
                torch.save(payload, best_checkpoint_path)

        guardrail_ok = (
            discovery_min_overall_recall is None
            or rec10 >= discovery_min_overall_recall
        )
        if new_recall10 is not None and guardrail_ok and new_recall10 > best_new_recall:
            best_new_recall = new_recall10
            best_discovery_epoch = epoch
            payload = checkpoint_payload("new_recall@10", new_recall10)
            torch.save(payload, best_discovery_checkpoint_path)
            if selection_metric == "discovery":
                torch.save(payload, best_checkpoint_path)

    # Save full training history to JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    print(f"Full training history saved to {history_path}")
    qualified = selection_metric == "ndcg" or best_discovery_epoch is not None
    atomic_json({
        "version": 1, "run_id": ckpt_dir.name,
        "status": "complete" if qualified else "no_qualifying_checkpoint",
        "selected_checkpoint": best_checkpoint_path.name if qualified else None,
        "selected_sha256": checkpoint_digest(best_checkpoint_path) if qualified else None,
        "retriever_digest": retriever_digest, "input_contract_digest": json_digest(feature_contract()),
        "slates": {name: data["cache_identity"] for name, data in (("train", train_data), ("val", val_data))},
        "selection_metric": selection_metric, "discovery_min_overall_recall": discovery_min_overall_recall,
        "seed": seed, "epochs": epochs,
    }, ckpt_dir / "run.json")
    if selection_metric == "discovery" and best_discovery_epoch is None:
        raise RuntimeError(
            "No discovery checkpoint qualified: validation Recall@10 did not meet "
            f"discovery_min_overall_recall={discovery_min_overall_recall}."
        )
    print(f"Best checkpoint saved to {best_checkpoint_path} using {selection_metric} selection.")

    return {
        "run_directory": str(ckpt_dir),
        "run_manifest_path": str(ckpt_dir / "run.json"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "history_path": str(history_path),
        "best_val_ndcg": best_val_ndcg,
        "best_new_recall@10": best_new_recall if best_discovery_epoch is not None else None,
        "best_discovery_epoch": best_discovery_epoch,
        "best_ndcg_checkpoint_path": str(best_ndcg_checkpoint_path),
        "best_discovery_checkpoint_path": (
            str(best_discovery_checkpoint_path) if best_discovery_epoch is not None else None
        ),
        "final_history": training_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an isolated, provenance-bound Stage 2 run.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ranker"))
    parser.add_argument("--retriever-checkpoint", type=Path, default=Path("checkpoints/retriever-discovery-reference/best_retriever.pt"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairwise-metadata-policy", choices=("snapshot_unverified", "observed_only_v1"), default="snapshot_unverified")
    parser.add_argument("--selection-metric", choices=("discovery", "ndcg"), default="discovery")
    args = parser.parse_args()
    train_ranker(checkpoint_dir=args.checkpoint_dir, retriever_checkpoint_path=args.retriever_checkpoint,
                 run_id=args.run_id, epochs=args.epochs, pairwise_metadata_policy=args.pairwise_metadata_policy,
                 selection_metric=args.selection_metric)


if __name__ == "__main__":
    main()
