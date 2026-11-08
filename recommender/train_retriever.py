"""Stage 1 Modular Two-Tower Retriever Training and Evaluation Pipeline.

Implements:
1. Point-in-time product demand features (6 continuous signals normalized strictly to [0, 1]).
2. Next-basket customer lifecycle transition generator (train < 2025-10-01,
   val 2025-10-01..2026-02-18, test >= 2026-02-19) with full prior history.
3. RetrieverDataset and DataLoader for multi-label next-basket prediction.
4. Retrieval evaluation metrics: Recall@K (10, 50, 100), NDCG@10, MRR.
5. End-to-end training loop with BCEWithLogitsLoss (pos_weight=20.0), AdamW,
   CosineAnnealingLR, automatic MPS/CUDA/CPU device selection.
6. Checkpointing: saves best model to checkpoints/retriever/best_retriever.pt
   and full metric history to checkpoints/retriever/training_history.json.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from recommender.catalogue import load_product_catalogue, order_retriever_catalogue, select_eligible_products
from recommender.familiarity import ATTRIBUTE_COLUMNS, BASE_FAMILIARITY_FIELDS, FamiliarityStore
from recommender.merging import EVENT_KEY_COLUMNS, merge_purchase_events
from recommender.features import (
    LEGACY_PROFILE_FEATURES,
    MEDIAN_BASKET_QUANTITY,
    MEDIAN_PURCHASE_QUANTITY,
    RETRIEVER_PROFILE_FEATURES,
    PointInTimeNormalizer,
    compute_point_in_time_features,
    normalize_feature,
)
from recommender.models import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEMPERATURE,
    OUTPUT_DIM,
    TEXT_DIM,
    CustomerTower,
    ProductTower,
    TwoTowerModel,
)
from recommender.sequences import (
    ENRICHED_FEATURE_DIM,
    build_customer_sequence,
    fit_history_gap_reference,
    fit_quantity_reference,
)
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.text import (
    CATALOGUE_CACHE_WITHOUT_SUBCATEGORY,
    COMPANY_CACHE_WITH_REGION,
    DEFAULT_CACHE_DIR,
    generate_catalogue_embeddings,
    generate_company_embeddings,
    load_cached_embeddings,
)


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Select compute device with automatic MPS/CUDA/CPU detection."""
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_product_demand_features(
    purchases_df: pd.DataFrame,
    as_of_date: Union[str, pd.Timestamp],
    eligible_product_ids: List[str],
    product_categories: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Compute 6 point-in-time product demand velocity features strictly prior to as_of_date.

    Features:
    1. lifetime_volume: log1p-normalized cumulative volume sold prior to as_of_date.
    2. orders_30d: log1p-normalized order count in the 30 days prior to as_of_date.
    3. orders_90d: log1p-normalized order count in the 90 days prior to as_of_date.
    4. unique_buyers: log1p-normalized distinct customer count prior to as_of_date.
    5. category_market_share: volume share of product within its category prior to as_of_date in [0, 1].
    6. recency: log1p-normalized days since last purchase prior to as_of_date in [0, 1].

    Returns:
        torch.FloatTensor of shape (len(eligible_product_ids), 6) with all values in [0.0, 1.0].
    """
    as_of_ts = pd.Timestamp(as_of_date)
    num_products = len(eligible_product_ids)

    if (
        purchases_df.empty
        or "purchase_date" not in purchases_df.columns
        or "product_id" not in purchases_df.columns
    ):
        return torch.zeros((num_products, 6), dtype=torch.float32)

    df = purchases_df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["product_id"] = df["product_id"].astype(str)

    history = df.loc[df["purchase_date"] < as_of_ts]
    if history.empty:
        # If no history before as_of_date, recency is 1.0 (longest gap), all else 0.0
        res = torch.zeros((num_products, 6), dtype=torch.float32)
        res[:, 5] = 1.0
        return res

    # 1. Lifetime volume
    if "quantity" in history.columns:
        vol_s = history.groupby("product_id")["quantity"].sum()
    else:
        vol_s = history.groupby("product_id").size().astype(float)
    max_vol = float(vol_s.max()) if len(vol_s) > 0 and vol_s.max() > 0 else 1.0

    # 2. 30d orders
    t_30 = as_of_ts - pd.Timedelta(days=30)
    h30 = history.loc[history["purchase_date"] >= t_30]
    o30_s = h30.groupby("product_id")["purchase_date"].nunique()
    max_o30 = float(o30_s.max()) if len(o30_s) > 0 and o30_s.max() > 0 else 1.0

    # 3. 90d orders
    t_90 = as_of_ts - pd.Timedelta(days=90)
    h90 = history.loc[history["purchase_date"] >= t_90]
    o90_s = h90.groupby("product_id")["purchase_date"].nunique()
    max_o90 = float(o90_s.max()) if len(o90_s) > 0 and o90_s.max() > 0 else 1.0

    # 4. Unique buyers
    cust_col = "customer_id" if "customer_id" in history.columns else None
    if cust_col:
        buyers_s = history.groupby("product_id")[cust_col].nunique()
    else:
        buyers_s = history.groupby("product_id").size()
    max_buyers = float(buyers_s.max()) if len(buyers_s) > 0 and buyers_s.max() > 0 else 1.0

    # 5. Category market share
    cat_volumes: Dict[str, float] = {}
    if "product_category" in history.columns:
        if "quantity" in history.columns:
            cat_volumes = history.groupby("product_category")["quantity"].sum().to_dict()
        else:
            cat_volumes = history.groupby("product_category").size().astype(float).to_dict()

    # 6. Recency (days since last purchase)
    last_dates = history.groupby("product_id")["purchase_date"].max()

    feat_matrix = np.zeros((num_products, 6), dtype=np.float32)

    for i, pid in enumerate(eligible_product_ids):
        pid_str = str(pid)

        # 1. Volume
        v = float(vol_s.get(pid_str, 0.0))
        feat_matrix[i, 0] = float(normalize_feature(v, max_vol))

        # 2. Orders 30d
        o30 = float(o30_s.get(pid_str, 0.0))
        feat_matrix[i, 1] = float(normalize_feature(o30, max_o30))

        # 3. Orders 90d
        o90 = float(o90_s.get(pid_str, 0.0))
        feat_matrix[i, 2] = float(normalize_feature(o90, max_o90))

        # 4. Unique buyers
        b = float(buyers_s.get(pid_str, 0.0))
        feat_matrix[i, 3] = float(normalize_feature(b, max_buyers))

        # 5. Category market share
        if product_categories and pid_str in product_categories:
            cat_name = product_categories[pid_str]
            cat_tot = cat_volumes.get(cat_name, 0.0)
            share = (v / cat_tot) if cat_tot > 0 else 0.0
            feat_matrix[i, 4] = float(np.clip(share, 0.0, 1.0))
        else:
            feat_matrix[i, 4] = 0.0

        # 6. Recency
        if pid_str in last_dates:
            dt_last = last_dates[pid_str]
            days_ago = max(0.0, float((as_of_ts - dt_last).total_seconds() / 86400.0))
            feat_matrix[i, 5] = float(normalize_feature(days_ago, 365.0))
        else:
            feat_matrix[i, 5] = 1.0  # Cold / never purchased

    # Enforce strict [0.0, 1.0] bounds
    feat_matrix = np.clip(feat_matrix, 0.0, 1.0)
    return torch.from_numpy(feat_matrix)


def build_retriever_transitions(
    purchases_df: pd.DataFrame,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
    val_cutoff: pd.Timestamp = TEST_START,
    min_baskets: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample chronological next-basket transitions (B_{<t}, B_t) for customer lifecycles.

    Partitions:
    - Train: target dates < 2025-10-01
    - Validation: target dates 2025-10-01 <= target_date < 2026-02-19
    - Test: target dates >= 2026-02-19

    CRITICAL: For validation and test transitions, a customer's history includes ALL
    events strictly prior to target_date (including purchases made during the train period!).
    """
    df = purchases_df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["customer_id"] = df["customer_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)

    train_c = pd.Timestamp(train_cutoff)
    val_c = pd.Timestamp(val_cutoff)

    train_transitions: List[Dict[str, Any]] = []
    val_transitions: List[Dict[str, Any]] = []
    test_transitions: List[Dict[str, Any]] = []

    for cid, group in df.groupby("customer_id"):
        distinct_dates = group["purchase_date"].sort_values().drop_duplicates().tolist()
        if len(distinct_dates) < min_baskets:
            continue

        for target_date in distinct_dates[1:]:
            target_basket = group.loc[group["purchase_date"] == target_date]
            positive_pids = sorted(target_basket["product_id"].unique().tolist())
            if not positive_pids:
                continue

            record = {
                "customer_id": str(cid),
                "as_of_date": target_date,
                "positive_product_ids": positive_pids,
            }

            if target_date < train_c:
                train_transitions.append(record)
            elif target_date < val_c:
                val_transitions.append(record)
            else:
                test_transitions.append(record)

    return train_transitions, val_transitions, test_transitions


def compute_retrieval_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray, Sequence[Set[int]], Sequence[Sequence[int]]],
    k_list: Sequence[int] = (10, 50, 100),
) -> Dict[str, float]:
    """Compute exact Recall@K, NDCG@10, and MRR metrics for multi-label basket recommendation.

    Args:
        predictions: FloatTensor or ndarray of shape (N, P) logits or (N, max_K) top-K indices.
        targets: FloatTensor or ndarray of shape (N, P) binary indicators, or list of target index sets.
        k_list: Top-K cutoffs to evaluate (e.g. 10, 50, 100).

    Returns:
        Dict mapping metric names to float scores.
    """
    if isinstance(predictions, torch.Tensor):
        preds_np = predictions.detach().cpu().numpy()
    else:
        preds_np = np.asarray(predictions)

    max_k = max(max(k_list), 10)
    n_samples = preds_np.shape[0]

    if n_samples == 0:
        return {
            f"recall@{k}": 0.0 for k in k_list
        } | {"ndcg@10": 0.0, "mrr": 0.0}

    # Extract top-K indices if given logits/scores
    if np.issubdtype(preds_np.dtype, np.floating):
        topk_indices = np.argsort(-preds_np, axis=1)[:, :max_k]
    else:
        topk_indices = preds_np[:, :max_k].astype(int)

    # Convert targets to set of positive indices per sample
    pos_sets: List[Set[int]] = []
    if isinstance(targets, (torch.Tensor, np.ndarray)):
        tgts_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        for i in range(n_samples):
            pos_sets.append(set(np.where(tgts_np[i] > 0.5)[0].tolist()))
    else:
        for t in targets:
            pos_sets.append(set(int(x) for x in t))

    recalls: Dict[int, List[float]] = {k: [] for k in k_list}
    ndcg_list: List[float] = []
    mrr_list: List[float] = []

    for i in range(n_samples):
        target_set = pos_sets[i]
        num_pos = len(target_set)
        if num_pos == 0:
            continue

        retrieved = topk_indices[i]

        # 1. Recall@K
        for k in k_list:
            ret_k = retrieved[:k]
            hits = sum(1 for p in ret_k if p in target_set)
            recalls[k].append(hits / num_pos)

        # 2. NDCG@10
        ret_10 = retrieved[:10]
        dcg = sum(1.0 / math.log2(r + 2) for r, p in enumerate(ret_10) if p in target_set)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(10, num_pos)))
        ndcg_list.append((dcg / idcg) if idcg > 0 else 0.0)

        # 3. MRR
        rr = 0.0
        for r, p in enumerate(retrieved):
            if p in target_set:
                rr = 1.0 / (r + 1)
                break
        mrr_list.append(rr)

    res: Dict[str, float] = {}
    for k in k_list:
        val = float(np.mean(recalls[k])) if recalls[k] else 0.0
        res[f"recall@{k}"] = val
        res[f"Recall@{k}"] = val

    val_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    res["ndcg@10"] = val_ndcg
    res["NDCG@10"] = val_ndcg

    val_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    res["mrr"] = val_mrr
    res["MRR"] = val_mrr

    return res


def build_customer_metadata(
    companies_df: pd.DataFrame, vocabs: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Look up company attributes using the vocabularies fitted for the retriever."""
    return {
        str(row.customer_id): {
            "city_id": vocabs["city_vocab"].get(str(row.city).strip(), 0),
            "region_id": vocabs["region_vocab"].get(str(row.region_code).strip(), 0),
            "industry_id": vocabs["industry_vocab"].get(str(row.industry_branch).strip(), 0),
        }
        for row in companies_df.itertuples(index=False)
    }


def prior_customer_products(history: pd.DataFrame, as_of_date: pd.Timestamp) -> Set[str]:
    prior = history.loc[pd.to_datetime(history["purchase_date"], format="mixed") < as_of_date, "product_id"]
    return set(prior.astype(str))


def new_to_customer_targets(
    history: pd.DataFrame, as_of_date: pd.Timestamp, target_ids: Sequence[str],
) -> List[str]:
    seen = prior_customer_products(history, as_of_date)
    return [product_id for product_id in target_ids if product_id not in seen]


def compute_discovery_loss(
    logits: torch.Tensor, targets: torch.Tensor, unseen: torch.Tensor, pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Learn from unseen candidates in baskets with at least one eligible new purchase."""
    eligible = unseen & (targets.bool() & unseen).any(dim=-1, keepdim=True)
    losses = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    return losses.masked_fill(~eligible, 0).sum() / eligible.sum().clamp_min(1)


class RetrieverDataset(Dataset):
    """Build the same retrieval inputs for training and preloaded or lazy scoring."""

    def __init__(
        self,
        transitions: List[Dict[str, Any]],
        purchases_df: pd.DataFrame,
        product_to_idx: Dict[str, int],
        customer_metadata: Dict[str, Dict[str, Any]],
        customer_embeddings: Dict[str, np.ndarray],
        normalizer: Optional[PointInTimeNormalizer] = None,
        vocabularies: Optional[Dict[str, Dict[str, int]]] = None,
        product_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        seq_feature_dim: int = ENRICHED_FEATURE_DIM,
        preload: bool = True,
        *,
        profile_features: Sequence[str] = LEGACY_PROFILE_FEATURES,
        quantity_reference: Optional[float] = None,
        include_unseen_candidates: bool = False,
    ):
        self.transitions = transitions
        self.product_to_idx = product_to_idx
        self.num_products = len(product_to_idx)
        self.customer_metadata = customer_metadata
        self.customer_embeddings = customer_embeddings
        self.normalizer = normalizer
        self.vocabularies = vocabularies
        self.product_metadata = product_metadata
        self.max_seq_len = max_seq_len
        self.seq_feature_dim = seq_feature_dim
        self.profile_features = tuple(profile_features)
        self.quantity_reference = quantity_reference
        self.include_unseen_candidates = include_unseen_candidates
        self.histories = {str(cid): rows for cid, rows in purchases_df.groupby("customer_id")}
        self.empty_history = purchases_df.iloc[:0]
        self.preloaded = None
        if preload and transitions:
            examples = [self._build_item(transition) for transition in transitions]
            self.preloaded = {
                key: torch.stack([example[key] for example in examples])
                for key in examples[0]
            }

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preloaded is not None:
            return {key: values[idx] for key, values in self.preloaded.items()}
        return self._build_item(self.transitions[idx])

    def _build_item(self, transition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cid = str(transition["customer_id"])
        as_of = transition["as_of_date"]
        history = self.histories.get(cid, self.empty_history)

        text = self.customer_embeddings.get(cid)
        if text is None:
            text = np.zeros(TEXT_DIM, dtype=np.float32)
        metadata = self.customer_metadata.get(cid, {})

        profile = compute_point_in_time_features(
            history, cid, as_of, normalizer=self.normalizer,
            include_median_quantity=MEDIAN_BASKET_QUANTITY in self.profile_features,
            include_median_purchase_quantity=MEDIAN_PURCHASE_QUANTITY in self.profile_features,
        )
        sequence_inputs = {}
        if self.max_seq_len:
            sequence = build_customer_sequence(
                history,
                customer_id=cid,
                as_of_date=as_of,
                max_len=self.max_seq_len,
                feature_dim=self.seq_feature_dim,
                vocabularies=self.vocabularies,
                item_metadata=self.product_metadata,
                enriched=True,
                quantity_reference=self.quantity_reference,
            )
            sequence_inputs = {
                name: sequence[name]
                for name in ("sequence", "attention_mask", "seq_product_ids", "seq_purchase_dates", "seq_category_ids")
            }

        target_ids = transition["positive_product_ids"]
        target = torch.zeros(self.num_products, dtype=torch.float32)
        for product_id in target_ids:
            if product_id in self.product_to_idx:
                target[self.product_to_idx[product_id]] = 1.0

        item = {
            "text_embedding": torch.as_tensor(text, dtype=torch.float32),
            "city_id": torch.tensor(int(metadata.get("city_id", 0)), dtype=torch.long),
            "region_id": torch.tensor(int(metadata.get("region_id", 0)), dtype=torch.long),
            "industry_id": torch.tensor(int(metadata.get("industry_id", 0)), dtype=torch.long),
            "tabular_features": torch.tensor([profile[name] for name in self.profile_features], dtype=torch.float32),
            **sequence_inputs,
            "target": target,
        }
        if self.include_unseen_candidates:
            unseen = torch.ones(self.num_products, dtype=torch.bool)
            for product_id in prior_customer_products(history, as_of):
                if product_id in self.product_to_idx:
                    unseen[self.product_to_idx[product_id]] = False
            item["unseen_candidates"] = unseen
        return item


def build_product_tower_inputs(
    eligible_products: pd.DataFrame,
    catalogue_embeddings: Dict[str, np.ndarray],
    demand_features: torch.Tensor,
    vocabs: Dict[str, Dict[str, int]],
    *,
    include_subcategory: bool = True,
) -> Dict[str, torch.Tensor]:
    """Format and batch all eligible product attributes for ProductTower forward pass."""
    brand_vocab = vocabs.get("brand_vocab", {})
    cat_vocab = vocabs.get("category_vocab", {})
    subcat_vocab = vocabs.get("subcategory_vocab", {})
    line_vocab = vocabs.get("business_line_vocab", {})

    p_ids = eligible_products["product_id"].astype(str).tolist()
    num_p = len(p_ids)

    text_arr = np.zeros((num_p, TEXT_DIM), dtype=np.float32)
    b_ids = np.zeros(num_p, dtype=np.int64)
    c_ids = np.zeros(num_p, dtype=np.int64)
    s_ids = np.zeros(num_p, dtype=np.int64) if include_subcategory else None
    l_ids = np.zeros(num_p, dtype=np.int64)

    for i, (_, row) in enumerate(eligible_products.iterrows()):
        pid = str(row["product_id"])
        if pid in catalogue_embeddings:
            text_arr[i] = catalogue_embeddings[pid]

        b = str(row.get("brand", "")).strip().upper()
        c = str(row.get("product_category", "")).strip().upper()
        l = str(row.get("business_line", "")).strip().upper()

        b_ids[i] = brand_vocab.get(b, 0)
        c_ids[i] = cat_vocab.get(c, 0)
        if s_ids is not None:
            s = str(row.get("product_subcategory", "")).strip().upper()
            s_ids[i] = subcat_vocab.get(s, 0)
        l_ids[i] = line_vocab.get(l, 0)

    inputs = {
        "text_embedding": torch.from_numpy(text_arr),
        "brand_id": torch.from_numpy(b_ids),
        "category_id": torch.from_numpy(c_ids),
        "business_line_id": torch.from_numpy(l_ids),
        "demand_features": demand_features,
    }
    if s_ids is not None:
        inputs["subcategory_id"] = torch.from_numpy(s_ids)
    return inputs


def build_retriever_product_inputs(
    purchases_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
    catalogue_embeddings: Dict[str, np.ndarray],
    product_ids: Sequence[str],
    vocabs: Dict[str, Dict[str, int]],
    *,
    include_subcategory: bool,
    include_sku_median_quantity: bool = False,
    container_encoding: str = "one_hot",
    sku_prices_uah: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Share the training snapshot and product index order with downstream scoring."""
    ordered = order_retriever_catalogue(catalogue_df, product_ids)
    categories = ordered.set_index("product_id")["product_category"].dropna().to_dict()
    demand = compute_product_demand_features(purchases_df, TRAIN_CUTOFF, list(product_ids), categories)
    if include_sku_median_quantity:
        prior = purchases_df.loc[pd.to_datetime(purchases_df["purchase_date"], format="mixed") < TRAIN_CUTOFF]
        prior = merge_purchase_events(prior[[*EVENT_KEY_COLUMNS, "quantity"]])
        medians = prior.groupby("product_id")["quantity"].median().reindex(product_ids, fill_value=0)
        log_values = torch.log1p(torch.as_tensor(medians.to_numpy(dtype=np.float32)))
        reference = math.log1p(fit_quantity_reference(medians))
        demand = torch.cat([demand, (log_values / (log_values + reference)).unsqueeze(1)], dim=1)
    if sku_prices_uah is not None:
        prices = np.array([sku_prices_uah.get(sku, 0.0) for sku in product_ids], dtype=np.float32)
        log_prices = torch.log1p(torch.from_numpy(prices))
        reference = math.log1p(fit_quantity_reference(prices))
        demand = torch.cat([demand, (log_prices / (log_prices + reference)).unsqueeze(1)], dim=1)
    container_ids = None
    if "container_vocab" in vocabs:
        container_vocab = vocabs["container_vocab"]
        container_types = ordered.get("container_type", pd.Series(index=ordered.index, dtype="string"))
        ids = container_types.map(container_vocab).fillna(0).to_numpy(dtype=np.int64)
        container_ids = torch.from_numpy(ids)
        if container_encoding != "embedding_none":
            indicators = torch.nn.functional.one_hot(container_ids, len(container_vocab) + 1)
            if container_encoding == "one_hot":
                indicators = indicators[:, 1:]
            demand = torch.cat([demand, indicators.to(demand.dtype)], dim=1)
    inputs = build_product_tower_inputs(
        ordered, catalogue_embeddings, demand, vocabs, include_subcategory=include_subcategory,
    )
    if container_ids is not None and container_encoding == "embedding_none":
        inputs["container_id"] = container_ids
    return inputs


def train_retriever(
    purchases_df: Optional[pd.DataFrame] = None,
    companies_df: Optional[pd.DataFrame] = None,
    catalogue_df: Optional[pd.DataFrame] = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 20.0,
    device: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints/retriever",
    dry_run: bool = False,
    seed: int = 42,
    *,
    history_encoder: str = "attention",
    max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    include_customer_median_quantity: bool = False,
    include_customer_median_purchase_quantity: bool = True,
    include_sku_median_quantity: bool = False,
    include_container_type: bool = True,
    container_encoding: str = "one_hot_none",
    include_region_in_text: bool = False,
    include_sku_price: bool = False,
    one_hot_product_fields: Sequence[str] = (),
    tower_hidden_dim: int = 128,
    history_layers: int = 2,
    extra_tower_mlp_layer: bool = False,
    output_dim: int = OUTPUT_DIM,
    dropout: float = 0.1,
    train_eval_size: int = 0,
    memorization_baskets: int = 0,
    discovery_only: bool = False,
    familiarity_fields: Sequence[str] = (),
    remove_product_attribute_embeddings: bool = False,
) -> Dict[str, Any]:
    """Train Modular Two-Tower Retriever and save best checkpoint + metrics history.

    Returns:
        Dict containing training summary, best validation Recall@50, and output paths.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if history_encoder not in ("attention", "pool", "none"):
        raise ValueError("history_encoder must be 'attention', 'pool', or 'none'.")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive.")
    if container_encoding not in ("one_hot", "one_hot_none", "embedding_none"):
        raise ValueError("Unknown container encoding.")
    if not set(one_hot_product_fields) <= {"brand", "category", "business_line"}:
        raise ValueError("One-hot product fields must be brand, category, or business_line.")
    if tower_hidden_dim < 4 or tower_hidden_dim % 4:
        raise ValueError("The tower hidden width must be a positive multiple of four attention heads.")
    if history_layers < 1:
        raise ValueError("The history Transformer must have at least one layer.")
    if output_dim < 1 or train_eval_size < 0 or memorization_baskets < 0:
        raise ValueError("The output dimension must be positive and diagnostic sizes must be nonnegative.")
    if memorization_baskets and dry_run:
        raise ValueError("A memorization check and a dry run cannot be combined.")
    if discovery_only and memorization_baskets:
        raise ValueError("Discovery-only training cannot be combined with memorization.")
    if len(set(familiarity_fields)) != len(familiarity_fields) or not set(familiarity_fields) <= ATTRIBUTE_COLUMNS.keys():
        raise ValueError("Invalid or duplicate familiarity field.")
    if remove_product_attribute_embeddings and (
        one_hot_product_fields or not set(BASE_FAMILIARITY_FIELDS) <= set(familiarity_fields)
    ):
        raise ValueError("Replacing product attribute embeddings requires all three familiarity flags and no one-hot fields.")
    profile_features = RETRIEVER_PROFILE_FEATURES
    if include_customer_median_quantity:
        profile_features = (*profile_features, MEDIAN_BASKET_QUANTITY)
    if include_customer_median_purchase_quantity:
        profile_features = (*profile_features, MEDIAN_PURCHASE_QUANTITY)

    dev = get_device(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load DataFrames if not supplied
    if purchases_df is None:
        p_path = Path("data/interim/cleaned_purchases.csv")
        if not p_path.is_file():
            raise FileNotFoundError(f"Cleaned purchases file not found: {p_path}")
        purchases_df = pd.read_csv(
            p_path, dtype={"customer_id": str, "product_id": str}, low_memory=False,
        )

    purchases_df = purchases_df.copy()
    purchases_df["purchase_date"] = pd.to_datetime(purchases_df["purchase_date"], format="mixed")
    if include_customer_median_purchase_quantity:
        purchases_df["purchase_date"] = purchases_df["purchase_date"].dt.normalize()
        if purchases_df.duplicated(EVENT_KEY_COLUMNS).any():
            purchases_df = merge_purchase_events(purchases_df)
    purchases_df = purchases_df.loc[purchases_df["purchase_date"] < TEST_START]
    train_purchases = purchases_df.loc[purchases_df["purchase_date"] < TRAIN_CUTOFF]
    quantity_reference = fit_quantity_reference(train_purchases.get("quantity", []))

    familiarity_catalogue = catalogue_df
    if catalogue_df is None:
        from recommender.scope import get_datasets_locs
        familiarity_catalogue = load_product_catalogue(get_datasets_locs("items"))
        catalogue_df = select_eligible_products(familiarity_catalogue, train_purchases, TRAIN_CUTOFF)

    if companies_df is None:
        from recommender.company import load_companies
        companies_df = load_companies(purchases=purchases_df, minimum_purchases=1)

    # 2. Catalogue and Company Text Embeddings
    cat_cache_path = DEFAULT_CACHE_DIR / CATALOGUE_CACHE_WITHOUT_SUBCATEGORY
    if cat_cache_path.is_file():
        catalogue_embeddings = load_cached_embeddings(cat_cache_path, id_column="product_id")
    else:
        catalogue_embeddings = generate_catalogue_embeddings(
            catalogue_df.drop(columns="product_subcategory", errors="ignore"), cache_path=cat_cache_path,
        )

    company_cache_name = COMPANY_CACHE_WITH_REGION if include_region_in_text else "bge_m3_company.parquet"
    comp_cache_path = ckpt_dir / company_cache_name
    if comp_cache_path.is_file():
        company_embeddings = load_cached_embeddings(comp_cache_path, id_column="customer_id")
    elif (DEFAULT_CACHE_DIR / company_cache_name).is_file():
        company_embeddings = load_cached_embeddings(DEFAULT_CACHE_DIR / company_cache_name, id_column="customer_id")
    else:
        company_embeddings = generate_company_embeddings(
            companies_df, cache_path=comp_cache_path, include_region=include_region_in_text,
        )

    # 3. Product Catalog Mapping and Attribute Vocabularies
    eligible_pids = sorted(catalogue_df["product_id"].astype(str).unique().tolist())
    product_to_idx = {pid: i for i, pid in enumerate(eligible_pids)}
    num_products = len(eligible_pids)
    if num_products == 0:
        raise ValueError("Stage 1 requires at least one eligible catalogue product.")
    catalogue_df = order_retriever_catalogue(catalogue_df, eligible_pids)

    # Build attribute vocabs
    brand_vocab = {b: i + 1 for i, b in enumerate(sorted(catalogue_df["brand"].dropna().unique().tolist()))}
    cat_vocab = {c: i + 1 for i, c in enumerate(sorted(catalogue_df["product_category"].dropna().unique().tolist()))}
    subcat_vocab = {s: i + 1 for i, s in enumerate(sorted(catalogue_df["product_subcategory"].dropna().unique().tolist()))}
    line_vocab = {l: i + 1 for i, l in enumerate(sorted(catalogue_df["business_line"].dropna().unique().tolist()))}

    training_companies = companies_df.loc[
        companies_df["customer_id"].astype(str).isin(train_purchases["customer_id"].astype(str))
    ]
    customer_vocabs = {
        f"{name}_vocab": {
            label: i + 1 for i, label in enumerate(
                sorted(training_companies[column].dropna().astype(str).str.strip().unique())
            )
        }
        for name, column in (("city", "city"), ("region", "region_code"), ("industry", "industry_branch"))
    }
    customer_metadata = build_customer_metadata(companies_df, customer_vocabs)

    # 4. Transitions
    train_transitions, val_transitions, _ = build_retriever_transitions(purchases_df)
    diagnostic_rng = np.random.default_rng(seed)
    if memorization_baskets:
        if len(train_transitions) < memorization_baskets:
            raise ValueError("The training split is too small for the requested memorization sample.")
        sample = diagnostic_rng.choice(len(train_transitions), memorization_baskets, replace=False)
        train_transitions = [train_transitions[i] for i in sorted(sample)]
        val_transitions = []
        train_eval_size = memorization_baskets
    if dry_run:
        train_transitions = train_transitions[:64]
        val_transitions = val_transitions[:32]
        epochs = 1

    # 5. Fit Point-In-Time Normalizer on train events
    cust_groups_train = {
        str(cid): grp
        for cid, grp in train_purchases.groupby("customer_id")
    }
    train_feat_rows = []
    for t in train_transitions:
        cid = str(t["customer_id"])
        as_of = t["as_of_date"]
        c_df = cust_groups_train[cid]
        raw_feat = compute_point_in_time_features(
            c_df, cid, as_of, include_median_quantity=include_customer_median_quantity,
            include_median_purchase_quantity=include_customer_median_purchase_quantity,
        )
        train_feat_rows.append(raw_feat)

    if train_feat_rows:
        train_feat_df = pd.DataFrame(train_feat_rows)
    else:
        train_feat_df = pd.DataFrame({
            feat: [1.0, 10.0] for feat in PointInTimeNormalizer.NUMERIC_FEATURES
        })
    normalizer = PointInTimeNormalizer()
    normalizer.fit(train_feat_df)

    # 6. Datasets & DataLoaders
    vocabs = {
        **customer_vocabs,
        "brand_vocab": brand_vocab,
        "category_vocab": cat_vocab,
        "subcategory_vocab": subcat_vocab,
        "business_line_vocab": line_vocab,
        "product_vocab": {pid: i + 1 for i, pid in enumerate(eligible_pids)},
    }
    if include_container_type:
        container_types = catalogue_df.get("container_type", pd.Series(dtype="string"))
        vocabs["container_vocab"] = {
            value: i + 1 for i, value in enumerate(sorted(container_types.dropna().unique()))
        }
    prod_meta = (
        catalogue_df.drop_duplicates(subset=["product_id"])
        .set_index("product_id")[["product_category", "product_subcategory", "business_line"]]
        .to_dict(orient="index")
    )

    train_ds = RetrieverDataset(
        transitions=train_transitions,
        purchases_df=purchases_df,
        product_to_idx=product_to_idx,
        customer_metadata=customer_metadata,
        customer_embeddings=company_embeddings,
        normalizer=normalizer,
        vocabularies=vocabs,
        product_metadata=prod_meta,
        preload=True,
        profile_features=profile_features,
        quantity_reference=quantity_reference,
        max_seq_len=max_seq_len if history_encoder != "none" else 0,
        include_unseen_candidates=discovery_only,
    )
    val_ds = RetrieverDataset(
        transitions=val_transitions,
        purchases_df=purchases_df,
        product_to_idx=product_to_idx,
        customer_metadata=customer_metadata,
        customer_embeddings=company_embeddings,
        normalizer=normalizer,
        vocabularies=vocabs,
        product_metadata=prod_meta,
        preload=True,
        profile_features=profile_features,
        quantity_reference=quantity_reference,
        max_seq_len=max_seq_len if history_encoder != "none" else 0,
        include_unseen_candidates=discovery_only,
    )

    if familiarity_fields:
        store = FamiliarityStore(purchases_df, familiarity_catalogue, eligible_pids, familiarity_fields)
        for dataset in (train_ds, val_ds):
            if dataset.preloaded is not None:
                dataset.preloaded["familiarity_features"] = store.batch_features(
                    [t["customer_id"] for t in dataset.transitions],
                    [t["as_of_date"] for t in dataset.transitions],
                )

    train_examples = train_ds
    if discovery_only:
        if train_ds.preloaded is None:
            raise ValueError("Discovery-only training requires training baskets.")
        novel_targets = train_ds.preloaded["target"].bool() & train_ds.preloaded["unseen_candidates"]
        discovery_rows = torch.where(novel_targets.any(dim=-1))[0].tolist()
        if not discovery_rows:
            raise ValueError("Discovery-only training requires at least one eligible first-time purchase.")
        train_examples = Subset(train_ds, discovery_rows)
        print(f"Discovery objective uses {len(discovery_rows)} training baskets with "
              f"{int(novel_targets.sum())} eligible new targets.")
    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    train_eval_indices = sorted(diagnostic_rng.choice(
        len(train_ds), min(train_eval_size, len(train_ds)), replace=False,
    )) if train_eval_size else []
    train_eval_loader = DataLoader(
        Subset(train_ds, train_eval_indices), batch_size=batch_size,
        generator=torch.Generator().manual_seed(seed),
    ) if train_eval_indices else None
    history_gap_reference_days = (
        fit_history_gap_reference(train_ds.preloaded["seq_purchase_dates"], train_ds.preloaded["attention_mask"])
        if history_encoder != "none" else None
    )

    # 7. Build the fixed product snapshot shared with downstream scoring.
    sku_prices_uah = None
    if include_sku_price:
        from recommender.data import load_sku_median_prices_uah
        from recommender.scope import get_datasets_locs
        sku_prices_uah = load_sku_median_prices_uah(get_datasets_locs("purchases"), eligible_pids, TRAIN_CUTOFF)
    prod_inputs = build_retriever_product_inputs(
        purchases_df, catalogue_df, catalogue_embeddings, eligible_pids, vocabs, include_subcategory=False,
        include_sku_median_quantity=include_sku_median_quantity,
        container_encoding=container_encoding,
        sku_prices_uah=sku_prices_uah,
    )
    # Move product inputs to device
    for k in prod_inputs:
        prod_inputs[k] = prod_inputs[k].to(dev)

    # 8. Model Initialization
    customer_tower = CustomerTower(
        text_dim=TEXT_DIM,
        output_dim=output_dim,
        num_cities=max(len(customer_vocabs['city_vocab']) + 2, 100),
        num_regions=max(len(customer_vocabs['region_vocab']) + 2, 50),
        num_industries=max(len(customer_vocabs['industry_vocab']) + 2, 100),
        num_categories=max(len(cat_vocab) + 2, 50),
        num_business_lines=max(len(line_vocab) + 2, 30),
        seq_feature_dim=2,
        catalog_history_encoder=history_encoder,
        history_gap_reference_days=history_gap_reference_days,
        max_seq_len=max_seq_len,
        num_tabular_features=len(profile_features),
        hidden_dim=tower_hidden_dim,
        num_heads=4,
        num_layers=history_layers,
        dim_feedforward=2 * tower_hidden_dim,
        dropout=dropout,
        extra_mlp_layer=extra_tower_mlp_layer,
    )
    product_embedding_sizes = {
        "brand": 0 if remove_product_attribute_embeddings else max(len(brand_vocab) + 2, 50),
        "category": 0 if remove_product_attribute_embeddings else max(len(cat_vocab) + 2, 50),
        "business_line": 0 if remove_product_attribute_embeddings else max(len(line_vocab) + 2, 30),
    }
    vocab_sizes = {"brand": len(brand_vocab), "category": len(cat_vocab), "business_line": len(line_vocab)}
    product_one_hot_sizes = {field: vocab_sizes[field] for field in one_hot_product_fields}
    product_tower = ProductTower(
        text_dim=TEXT_DIM,
        output_dim=output_dim,
        num_brands=product_embedding_sizes["brand"],
        num_categories=product_embedding_sizes["category"],
        num_subcategories=0,
        num_business_lines=product_embedding_sizes["business_line"],
        num_demand_features=prod_inputs["demand_features"].size(1),
        num_containers=len(vocabs["container_vocab"]) + 1 if "container_id" in prod_inputs else 0,
        one_hot_sizes=product_one_hot_sizes,
        extra_mlp_layer=extra_tower_mlp_layer,
        hidden_dim=tower_hidden_dim,
        dropout=dropout,
    )
    model = TwoTowerModel(
        customer_tower=customer_tower,
        product_tower=product_tower,
        num_catalog_products=num_products,
        temperature=DEFAULT_TEMPERATURE,
        num_familiarity_features=len(familiarity_fields),
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

    pos_weight_t = torch.tensor([pos_weight], device=dev, dtype=torch.float32)
    product_indices_all = torch.arange(num_products, device=dev)

    best_selection = -1.0
    best_recall_50 = -1.0
    best_checkpoint_path = ckpt_dir / "best_retriever.pt"
    history_path = ckpt_dir / "training_history.json"
    training_history: List[Dict[str, Any]] = []

    print(f"Starting Stage 1 Retriever training on {dev}: {len(train_transitions)} train transitions, "
          f"{len(val_transitions)} val transitions, {num_products} products, {epochs} epochs.")

    def target_indices(product_ids: Sequence[str]) -> Set[int]:
        # Count targets outside the catalogue as misses.
        known = {product_to_idx[pid] for pid in product_ids if pid in product_to_idx}
        return known.union(range(len(known) - len(product_ids), 0))

    val_targets = [target_indices(t["positive_product_ids"]) for t in val_transitions]
    val_new_targets = []
    for transition in val_transitions:
        history = val_ds.histories[str(transition["customer_id"])]
        val_new_targets.append(target_indices(new_to_customer_targets(
            history, transition["as_of_date"], transition["positive_product_ids"],
        )))
    val_new_count = sum(map(len, val_new_targets))
    if discovery_only and not val_new_count:
        raise ValueError("Discovery-only checkpoint selection requires first-time purchases in validation.")
    train_eval_targets = [target_indices(train_transitions[i]["positive_product_ids"]) for i in train_eval_indices]

    def product_vectors() -> torch.Tensor:
        return model.product_tower(
            prod_inputs["text_embedding"], prod_inputs["brand_id"], prod_inputs["category_id"],
            None, prod_inputs["business_line_id"], prod_inputs["demand_features"],
            container_id=prod_inputs.get("container_id"),
        )

    def score_batch(batch: Dict[str, torch.Tensor], vectors: torch.Tensor) -> torch.Tensor:
        sequence_inputs = {
            name: batch[name].to(dev)
            for name in ("sequence", "attention_mask", "seq_product_ids", "seq_purchase_dates")
            if name in batch
        }
        customers = model.customer_tower(
            batch["text_embedding"].to(dev), batch["city_id"].to(dev), batch["region_id"].to(dev),
            batch["industry_id"].to(dev), batch["tabular_features"].to(dev),
            catalog_product_vectors=vectors, **sequence_inputs,
        )
        return model.compute_logits(
            customers, vectors, product_indices=product_indices_all,
            familiarity_features=batch.get("familiarity_features"),
        )

    def evaluate_batches(
        loader: DataLoader, vectors: torch.Tensor, top_k: int,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        total_loss = 0.0
        topk_rows = []
        discovery_loss = 0.0
        discovery_pairs = 0
        discovery_targets = 0
        discovery_hits = {25: 0, 50: 0}
        for batch in loader:
            logits = score_batch(batch, vectors)
            targets = batch["target"].to(dev)
            total_loss += model.compute_multi_label_loss(logits, targets, pos_weight=pos_weight_t).item()
            topk_rows.append(torch.topk(logits, k=min(top_k, num_products), dim=-1).indices.cpu().numpy())
            if discovery_only:
                unseen = batch["unseen_candidates"].to(dev)
                positives = targets.bool() & unseen
                pairs = int((unseen & positives.any(dim=-1, keepdim=True)).sum().item())
                if pairs:
                    discovery_loss += compute_discovery_loss(logits, targets, unseen, pos_weight_t).item() * pairs
                    discovery_pairs += pairs
                discovery_targets += int(positives.sum().item())
                indices = logits.masked_fill(~unseen, -torch.inf).topk(min(50, num_products), dim=-1).indices
                hits = positives.gather(1, indices)
                for k in discovery_hits:
                    discovery_hits[k] += int(hits[:, :k].sum().item())
        topk = np.concatenate(topk_rows, axis=0) if topk_rows else np.zeros((0, top_k), dtype=np.int64)
        discovery_metrics = {}
        if discovery_only:
            discovery_metrics = {
                "discovery_loss": discovery_loss / max(discovery_pairs, 1),
                "discovery_targets": discovery_targets,
                **{f"discovery_hits@{k}": count for k, count in discovery_hits.items()},
            }
        return total_loss / max(len(loader), 1), topk, discovery_metrics

    best_selection_loss = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            logits = score_batch(batch, product_vectors())
            targets = batch["target"].to(dev)
            if discovery_only:
                loss = compute_discovery_loss(logits, targets, batch["unseen_candidates"].to(dev), pos_weight_t)
            else:
                loss = model.compute_multi_label_loss(logits, targets, pos_weight=pos_weight_t)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        val_metrics: Dict[str, float] = {}
        train_eval_metrics: Dict[str, float] = {}
        with torch.no_grad():
            vectors = product_vectors()
            if not memorization_baskets:
                avg_val_loss, val_topk, val_discovery = evaluate_batches(val_loader, vectors, 100)
                val_metrics = compute_retrieval_metrics(val_topk, val_targets, k_list=(10, 50, 100))
                if val_new_count:
                    new_hits = sum(len(targets.intersection(predicted[:50])) for targets, predicted in zip(val_new_targets, val_topk))
                    val_metrics["new_to_customer_recall@50"] = new_hits / val_new_count
                if discovery_only:
                    val_metrics.update(val_discovery)
                    for k in (25, 50):
                        val_metrics[f"discovery_recall@{k}"] = val_discovery[f"discovery_hits@{k}"] / val_new_count
            if train_eval_loader is not None:
                train_eval_loss, train_topk, train_discovery = evaluate_batches(train_eval_loader, vectors, 50)
                train_eval_metrics = compute_retrieval_metrics(train_topk, train_eval_targets, k_list=(10, 50))

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "Train Loss": avg_train_loss,
        }
        summary = f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f}"
        if train_eval_metrics:
            epoch_record.update({
                "train_eval_loss": train_eval_loss,
                "train_eval_recall@10": train_eval_metrics["recall@10"],
                "train_eval_recall@50": train_eval_metrics["recall@50"],
            })
            summary += f" | Train Eval Loss: {train_eval_loss:.4f} | Train Recall@10: {train_eval_metrics['recall@10']:.4f}"
            if discovery_only:
                epoch_record["train_eval_discovery_loss"] = train_discovery["discovery_loss"]
                epoch_record["train_discovery_recall@25"] = (
                    train_discovery["discovery_hits@25"] / max(train_discovery["discovery_targets"], 1)
                )
        if val_metrics:
            rec10, rec50, rec100 = (val_metrics[f"recall@{k}"] for k in (10, 50, 100))
            ndcg10, mrr = val_metrics["ndcg@10"], val_metrics["mrr"]
            epoch_record.update({
                "val_loss": avg_val_loss,
                "val_recall@10": rec10,
                "val_recall@50": rec50,
                "val_recall@100": rec100,
                "val_ndcg@10": ndcg10,
                "val_mrr": mrr,
                "Val Loss": avg_val_loss,
                "Val Recall@10": rec10,
                "Val Recall@50": rec50,
                "Val Recall@100": rec100,
                "Val NDCG@10": ndcg10,
                "Val MRR": mrr,
            })
            summary += (
                f" | Val Loss: {avg_val_loss:.4f} | Val Recall@10: {rec10:.4f} | Val Recall@50: {rec50:.4f}"
                f" | Val Recall@100: {rec100:.4f} | Val NDCG@10: {ndcg10:.4f} | Val MRR: {mrr:.4f}"
            )
            if "new_to_customer_recall@50" in val_metrics:
                novel = val_metrics["new_to_customer_recall@50"]
                epoch_record["val_new_to_customer_recall@50"] = novel
                summary += f" | Val New Recall@50: {novel:.4f}"
            if discovery_only:
                epoch_record.update({f"val_{key}": value for key, value in val_discovery.items()})
                epoch_record.update({f"val_discovery_recall@{k}": val_metrics[f"discovery_recall@{k}"] for k in (25, 50)})
                summary += (f" | Val Discovery Loss: {val_discovery['discovery_loss']:.4f}"
                            f" | Val Discovery Recall@25: {val_metrics['discovery_recall@25']:.4f}")
        training_history.append(epoch_record)
        print(summary)

        # Select by validation coverage. The memorization diagnostic only uses the training sample.
        selected_metrics = train_eval_metrics if memorization_baskets else val_metrics
        selection_metric = "discovery_recall@25" if discovery_only else "recall@50"
        selection = selected_metrics[selection_metric]
        selection_loss = train_eval_loss if memorization_baskets else avg_val_loss
        if selection > best_selection or (
            memorization_baskets and selection == best_recall_50 and selection_loss < best_selection_loss
        ):
            best_selection = selection
            best_recall_50 = selected_metrics["recall@50"]
            best_selection_loss = selection_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_recall_50": best_recall_50,
                    "selection_metric": selection_metric,
                    "product_ids": eligible_pids,
                    "product_to_idx": product_to_idx,
                    "val_metrics": val_metrics,
                    "train_eval_metrics": train_eval_metrics,
                    "vocabs": vocabs,
                    "config": {
                        "output_dim": output_dim,
                        "training_epochs": epochs,
                        "text_dim": TEXT_DIM,
                        "num_catalog_products": num_products,
                        "temperature": DEFAULT_TEMPERATURE,
                        "pos_weight": pos_weight,
                        "discovery_only": discovery_only,
                        "familiarity_fields": list(familiarity_fields),
                        "profile_features": list(profile_features),
                        "company_cache_name": company_cache_name,
                        "include_sku_median_quantity": include_sku_median_quantity,
                        "num_product_features": product_tower.num_demand_features,
                        "product_embedding_sizes": product_embedding_sizes,
                        "product_one_hot_sizes": product_one_hot_sizes,
                        "tower_hidden_dim": tower_hidden_dim,
                        "history_layers": history_layers,
                        "extra_tower_mlp_layer": extra_tower_mlp_layer,
                        "dropout": dropout,
                        "container_encoding": container_encoding,
                        "sku_prices_uah": sku_prices_uah,
                        "catalog_history_encoder": history_encoder,
                        "max_seq_len": max_seq_len,
                        "history_gap_reference_days": history_gap_reference_days,
                        "profile_normalizer": normalizer.to_state(),
                        "quantity_reference": quantity_reference,
                    },
                },
                best_checkpoint_path,
            )

    # Save full training history to JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    return {
        "best_checkpoint_path": str(best_checkpoint_path),
        "history_path": str(history_path),
        "best_recall_50": best_recall_50,
        "final_history": training_history,
        "num_products": num_products,
        "num_train_transitions": len(train_transitions),
        "num_optimizer_transitions": len(train_examples),
        "num_val_transitions": len(val_transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Stage 1 retriever.")
    parser.add_argument(
        "--discovery-reference", action="store_true",
        help="Train the 128D discovery reference with four familiarity flags.",
    )
    args = parser.parse_args()
    if args.discovery_reference:
        train_retriever(
            epochs=30,
            output_dim=128,
            train_eval_size=2048,
            checkpoint_dir="checkpoints/retriever-discovery-reference",
            discovery_only=True,
            familiarity_fields=(*BASE_FAMILIARITY_FIELDS, "family"),
            remove_product_attribute_embeddings=True,
        )
    else:
        train_retriever()


if __name__ == "__main__":
    main()
