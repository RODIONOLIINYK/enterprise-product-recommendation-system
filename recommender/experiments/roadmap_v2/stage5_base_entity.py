"""Stage 5: Full base_name canonical product entity pipeline & apples-to-apples base_name evaluator."""

from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.feature_schema import NAMES as SOURCE_41_NAMES
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.experiments.static_enrichment.config import VariantSpec
from recommender.experiments.static_enrichment.model import StaticEnrichedRetriever
from recommender.models import OUTPUT_DIM
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_retriever import get_device

from .base_name_features import (
    _clean_base_name,
    assemble_arm_features_numpy,
    prepare_or_load_base_name_feature_bundle,
)
from .config import (
    ARM_BASE_NAME_ONLY,
    ARTIFACT_ROOT,
    CACHE_ROOT,
    SEED,
)


def build_base_name_universe(
    product_ids: Sequence[str],
    catalogue: pd.DataFrame,
    purchases: pd.DataFrame,
    *,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
) -> Dict[str, Any]:
    """Build deterministic canonical base_name universe (585 unique bodies) from the 841 active SKUs."""
    cat_dedup = catalogue.copy()
    cat_dedup["product_id"] = cat_dedup["product_id"].astype(str).str.strip()
    cat_dedup = cat_dedup.drop_duplicates("product_id").set_index("product_id")
    all_pid_to_bn = cat_dedup["base_name"].map(_clean_base_name).to_dict()

    prod_bns = [all_pid_to_bn.get(str(p).strip(), "") for p in product_ids]
    if any(not b for b in prod_bns):
        raise ValueError("All canonical product_ids must have non-empty base_name.")

    # Order: base_names seen strictly < train_cutoff first (trained_base_names), then val-only base_names
    train_pids = set(
        purchases.loc[pd.to_datetime(purchases["purchase_date"], format="mixed") < train_cutoff, "product_id"]
        .astype(str)
        .str.strip()
    )
    train_bns_set = {all_pid_to_bn.get(p, "") for p in train_pids} - {""}
    all_unique = sorted(set(prod_bns))
    trained_bns = [b for b in all_unique if b in train_bns_set]
    untrained_bns = [b for b in all_unique if b not in train_bns_set]
    base_names = trained_bns + untrained_bns
    bn_to_idx = {b: i for i, b in enumerate(base_names)}

    pid_idx_to_bn_idx = np.array([bn_to_idx[b] for b in prod_bns], dtype=np.int64)
    bn_to_member_pid_indices: List[List[int]] = [[] for _ in base_names]
    for p_i, b_i in enumerate(pid_idx_to_bn_idx.tolist()):
        bn_to_member_pid_indices[b_i].append(p_i)

    # First representative SKU index per base_name (for scattering/gathering base_name-level features)
    bn_rep_pid_idx = np.array([members[0] for members in bn_to_member_pid_indices], dtype=np.int64)

    return {
        "base_names": base_names,
        "bn_to_idx": bn_to_idx,
        "trained_base_names": len(trained_bns),
        "num_base_names": len(base_names),
        "pid_idx_to_bn_idx": pid_idx_to_bn_idx,
        "bn_to_member_pid_indices": bn_to_member_pid_indices,
        "bn_rep_pid_idx": bn_rep_pid_idx,
        "all_pid_to_bn": all_pid_to_bn,
    }


def evaluate_base_name_predictions(
    ranked_bn_indices: np.ndarray,
    target_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    customer_ids: Sequence[str],
    *,
    save_path: Optional[Path] = None,
    prediction_days: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Apples-to-apples base_name-level evaluator (NDCG@10, Recall@K, truly new base_name discovery@K).

    Args:
        ranked_bn_indices: (N, K_max) int array of unique base_name indices per query (best first; -1 for pad).
        target_bn: (N, B) bool ground-truth base_name targets purchased on query date.
        new_target_bn: (N, B) bool ground-truth TRULY NEW base_name targets (never bought in any pack < date).
        customer_ids: length-N list of customer_id strings (for paired customer-clustered bootstrap).
    """
    n_rows, k_max = ranked_bn_indices.shape
    t_np = target_bn.detach().cpu().numpy().astype(bool)
    nt_np = new_target_bn.detach().cpu().numpy().astype(bool)

    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    per_q_r5 = np.zeros(n_rows, dtype=np.float64)
    per_q_r10 = np.zeros(n_rows, dtype=np.float64)
    per_q_r25 = np.zeros(n_rows, dtype=np.float64)
    per_q_r50 = np.zeros(n_rows, dtype=np.float64)
    per_q_ndcg10 = np.zeros(n_rows, dtype=np.float64)
    per_q_new_hits10 = np.zeros(n_rows, dtype=np.float64)
    per_q_new_targets = nt_np.sum(axis=1).astype(np.float64)
    per_q_targets = t_np.sum(axis=1).astype(np.float64)

    hits_at = {5: 0, 10: 0, 25: 0, 50: 0}
    new_hits_at = {5: 0, 10: 0, 25: 0, 50: 0}

    for i in range(n_rows):
        row_t = t_np[i]
        row_nt = nt_np[i]
        n_pos = int(row_t.sum())
        if n_pos == 0:
            continue
        preds = ranked_bn_indices[i]
        valid = preds >= 0
        rel = np.zeros(preds.shape, dtype=bool)
        new_rel = np.zeros_like(rel)
        rel[valid] = row_t[preds[valid]]
        new_rel[valid] = row_nt[preds[valid]]

        for k in (5, 10, 25, 50):
            hk = int(rel[:k].sum())
            nhk = int(new_rel[:k].sum())
            hits_at[k] += hk
            new_hits_at[k] += nhk

        per_q_r5[i] = float(rel[:5].sum()) / n_pos
        per_q_r10[i] = float(rel[:10].sum()) / n_pos
        per_q_r25[i] = float(rel[:25].sum()) / n_pos
        per_q_r50[i] = float(rel[:50].sum()) / n_pos
        per_q_new_hits10[i] = float(new_rel[:10].sum())

        top10_rel = rel[:10].astype(np.float64)
        dcg = float((top10_rel * discounts[: len(top10_rel)]).sum())
        idcg = float(discounts[: min(n_pos, 10)].sum())
        per_q_ndcg10[i] = dcg / idcg if idcg > 0 else 0.0

    total_targets = int(t_np.sum())
    total_new = int(nt_np.sum())
    metrics = {
        "queries": int(n_rows),
        "total_bn_targets": total_targets,
        "true_new_bn_targets": total_new,
        "bn_recall@5": float(per_q_r5.sum() / max(n_rows, 1)),
        "bn_recall@10": float(per_q_r10.sum() / max(n_rows, 1)),
        "bn_recall@25": float(per_q_r25.sum() / max(n_rows, 1)),
        "bn_recall@50": float(per_q_r50.sum() / max(n_rows, 1)),
        "bn_micro_recall@10": float(hits_at[10] / max(total_targets, 1)),
        "bn_ndcg@10": float(per_q_ndcg10.sum() / max(n_rows, 1)),
        "true_new_bn_hits@5": int(new_hits_at[5]),
        "true_new_bn_hits@10": int(new_hits_at[10]),
        "true_new_bn_hits@25": int(new_hits_at[25]),
        "true_new_bn_hits@50": int(new_hits_at[50]),
        "true_new_bn_recall@10": float(new_hits_at[10] / max(total_new, 1)),
        "true_new_bn_recall@25": float(new_hits_at[25] / max(total_new, 1)),
        "true_new_bn_recall@50": float(new_hits_at[50] / max(total_new, 1)),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save compatible arrays for paired customer-clustered bootstrap
        np.savez(
            save_path,
            customer_ids=np.asarray(list(customer_ids), dtype=object),
            recall_5=per_q_r5,
            recall_10=per_q_r10,
            recall_25=per_q_r25,
            recall_50=per_q_r50,
            ndcg_10=per_q_ndcg10,
            new_hits_10=per_q_new_hits10,
            new_targets=per_q_new_targets,
            targets=per_q_targets,
            **({"prediction_days": np.asarray(prediction_days)} if prediction_days is not None else {}),
        )
    return metrics


def collapse_sku_scores_to_base_name_ranking(
    sku_scores: torch.Tensor,
    candidate_indices: Optional[torch.Tensor],
    pid_idx_to_bn_idx: np.ndarray,
    *,
    allowed_mask: Optional[torch.Tensor] = None,
    max_k: int = 50,
) -> np.ndarray:
    """Collapse ranked SKU scores (full pool or slate) to unique base_name indices by first-occurrence order."""
    if max_k < 0:
        raise ValueError("max_k must be nonnegative.")
    scores_np = sku_scores.detach().cpu().float().numpy() if sku_scores.dtype == torch.bfloat16 else sku_scores.detach().cpu().numpy()
    valid = np.isfinite(scores_np)
    if allowed_mask is not None:
        valid &= allowed_mask.detach().cpu().numpy().astype(bool)
    order = np.argsort(-scores_np, axis=-1, kind="stable")
    cand_np = candidate_indices.detach().cpu().numpy() if candidate_indices is not None else None

    n_rows = scores_np.shape[0]
    out = np.full((n_rows, max_k), -1, dtype=np.int64)
    if max_k == 0:
        return out
    for i in range(n_rows):
        seen: set[int] = set()
        pos = 0
        for col in order[i]:
            if not valid[i, col]:
                continue
            p_idx = int(cand_np[i, col]) if cand_np is not None else int(col)
            if p_idx < 0:
                continue
            b_idx = int(pid_idx_to_bn_idx[p_idx])
            if b_idx not in seen:
                seen.add(b_idx)
                out[i, pos] = b_idx
                pos += 1
                if pos >= max_k:
                    break
    return out


def paired_base_name_bootstrap_ci(
    npz_a: Union[str, Path],
    npz_b: Union[str, Path],
    *,
    n_boot: int = 2000,
    seed: int = 123,
) -> Dict[str, Any]:
    """Customer-clustered paired bootstrap 95% CI comparing two base_name-level prediction archives (B - A)."""
    a = np.load(npz_a, allow_pickle=True)
    b = np.load(npz_b, allow_pickle=True)
    cids = a["customer_ids"].tolist()
    if cids != b["customer_ids"].tolist():
        raise ValueError("Customer ID alignment mismatch for paired bootstrap.")

    for key in ("targets", "new_targets", "prediction_days"):
        if key in a and key in b and not np.array_equal(a[key], b[key]):
            raise ValueError(f"Basket alignment mismatch for paired bootstrap: {key}")

    # Group row indices by customer
    by_c: Dict[str, List[int]] = {}
    for idx, c in enumerate(cids):
        by_c.setdefault(str(c), []).append(idx)
    clusters = list(by_c.values())
    n_c = len(clusters)

    # Pre-aggregate per customer cluster for fast vectorized bootstrap
    def _cluster_sums(arr: np.ndarray) -> np.ndarray:
        return np.array([arr[idxs].sum() for idxs in clusters], dtype=np.float64)

    c_counts = np.array([len(idxs) for idxs in clusters], dtype=np.float64)
    c_new_tot = _cluster_sums(a["new_targets"])
    metric_keys = ("recall_5", "recall_10", "ndcg_10")
    sums_a = {k: _cluster_sums(a[k]) for k in metric_keys}
    sums_b = {k: _cluster_sums(b[k]) for k in metric_keys}
    nh_a = _cluster_sums(a["new_hits_10"])
    nh_b = _cluster_sums(b["new_hits_10"])

    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_c, size=(int(n_boot), n_c))
    # Count matrix via bincount per draw (fast & exact)
    weights = np.zeros((int(n_boot), n_c), dtype=np.float64)
    for r in range(int(n_boot)):
        weights[r] = np.bincount(draws[r], minlength=n_c)

    denom_q = weights @ c_counts
    denom_nt = np.maximum(weights @ c_new_tot, 1.0)

    out: Dict[str, Any] = {}
    for k in metric_keys:
        pt = float((sums_b[k].sum() - sums_a[k].sum()) / c_counts.sum())
        boot_d = ((weights @ sums_b[k]) - (weights @ sums_a[k])) / denom_q
        lo, med, hi = np.quantile(boot_d, [0.025, 0.5, 0.975]).tolist()
        out[f"bn_{k}_delta"] = {"point_delta": pt, "ci_95": [lo, med, hi]}

    pt_nh = float((nh_b.sum() - nh_a.sum()) / max(c_new_tot.sum(), 1.0))
    boot_nh = ((weights @ nh_b) - (weights @ nh_a)) / denom_nt
    lo, med, hi = np.quantile(boot_nh, [0.025, 0.5, 0.975]).tolist()
    out["true_new_bn_recall@10_delta"] = {
        "point_delta": pt_nh,
        "point_hits_delta": int(round(nh_b.sum() - nh_a.sum())),
        "ci_95": [lo, med, hi],
    }
    return out


def build_base_name_sequences(
    purchases: pd.DataFrame,
    customer_ids: Sequence[str],
    prediction_days: Sequence[int],
    universe: Mapping[str, Any],
    catalogue: pd.DataFrame,
    cat_vocab: Mapping[str, int],
    subcat_vocab: Mapping[str, int],
    line_vocab: Mapping[str, int],
    seq_len: int = 32,
    *,
    quantity_reference_length: Optional[int] = None,
    allow_heldout_test: bool = False,
    frozen_v5_replay: bool = False,
    test_start: pd.Timestamp = TEST_START,
) -> Dict[str, torch.Tensor]:
    """Build strictly point-in-time (< as_of_date) base_name-unified 32-event purchase sequences.

    Collapses same-day multi-pack purchases of the same base_name into ONE event (summed quantity),
    producing cleaner, longer-horizon 32-event customer histories for both Stage 1 and Stage 2.
    """
    df = purchases.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
    if not allow_heldout_test and bool(df["purchase_date"].ge(test_start).any()):
        raise ValueError("Development purchases only.")
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0.0)

    all_pid_to_bn = universe["all_pid_to_bn"]
    bn_to_idx = universe["bn_to_idx"]
    df["bn"] = df["product_id"].map(all_pid_to_bn).fillna("")
    df = df.loc[df["bn"] != ""].copy()

    # Aggregate same-day same-base_name multi-pack lines into ONE event
    agg = df.groupby(["customer_id", "purchase_date", "bn"], as_index=False, sort=False)["quantity"].sum()
    agg = agg.sort_values(["customer_id", "purchase_date", "bn"], kind="stable")

    # Map each base_name -> cat/subcat/line vocab IDs
    cat_dedup = catalogue.copy()
    cat_dedup["product_id"] = cat_dedup["product_id"].astype(str).str.strip()
    cat_dedup["bn"] = cat_dedup["base_name"].map(_clean_base_name)
    bn_meta: Dict[str, Tuple[int, int, int]] = {}
    for bn, grp in cat_dedup.loc[cat_dedup["bn"] != ""].groupby("bn", sort=False):
        c_val = str(grp["product_category"].dropna().mode().iloc[0]).strip() if grp["product_category"].notna().any() else ""
        sc_val = str(grp["product_subcategory"].dropna().mode().iloc[0]).strip() if grp["product_subcategory"].notna().any() else ""
        bl_val = str(grp["business_line"].dropna().mode().iloc[0]).strip() if grp["business_line"].notna().any() else ""
        bn_meta[bn] = (
            int(cat_vocab.get(c_val, 0)),
            int(subcat_vocab.get(sc_val, 0)),
            int(line_vocab.get(bl_val, 0)),
        )

    origin = pd.Timestamp("2025-01-01")
    # Pre-index per customer: chronological list of (day_int, bn_id_1based, qty, cat_id, subcat_id, line_id, month)
    cust_ev: Dict[str, Tuple[np.ndarray, List[Tuple[int, str, int, float, int, int, int, int]]]] = {}
    for cid, grp in agg.groupby("customer_id", sort=False):
        dts = ((grp["purchase_date"] - origin).dt.days).to_numpy(dtype=np.int64)
        months = grp["purchase_date"].dt.month.to_numpy(dtype=np.int64)
        evs = []
        for d_i, bn_s, q_v, m_i in zip(dts.tolist(), grp["bn"].tolist(), grp["quantity"].tolist(), months.tolist()):
            b_id = bn_to_idx.get(bn_s, -1) + 1  # 1..B (0 = padding / OOV)
            c_i, sc_i, bl_i = bn_meta.get(bn_s, (0, 0, 0))
            evs.append((d_i, bn_s, b_id, float(q_v), c_i, sc_i, bl_i, int(m_i)))
        cust_ev[cid] = (dts, evs)

    n_rows = len(customer_ids)
    sequence = np.zeros((n_rows, seq_len, 9), dtype=np.float32)
    seq_rel_qty = np.zeros((n_rows, seq_len), dtype=np.float32)
    cust_median_gap_days = np.full((n_rows,), 30.0, dtype=np.float32)
    attn_mask = np.zeros((n_rows, seq_len), dtype=np.int64)
    seq_bn_ids = np.zeros((n_rows, seq_len), dtype=np.int64)
    seq_cat_ids = np.zeros((n_rows, seq_len), dtype=np.int64)
    seq_subcat_ids = np.zeros((n_rows, seq_len), dtype=np.int64)
    seq_line_ids = np.zeros((n_rows, seq_len), dtype=np.int64)
    seq_dates = np.zeros((n_rows, seq_len), dtype=np.float32)

    log60 = math.log1p(60.0)
    log50 = math.log1p(50.0)
    for row, (cid_raw, pred_day) in enumerate(zip(customer_ids, prediction_days)):
        cid = str(cid_raw).strip()
        pday = int(pred_day)
        entry = cust_ev.get(cid)
        if entry is None:
            continue
        dts, evs = entry
        stop = int(dts.searchsorted(pday, side="left"))
        if stop == 0:
            continue
        # Running prior counts & total volume up to each event in the last `seq_len` window
        prior_evs = evs[:stop]
        uniq_days = []
        for ev in prior_evs:
            if not uniq_days or ev[0] != uniq_days[-1]:
                uniq_days.append(ev[0])
        if len(uniq_days) >= 2:
            day_gaps = [b - a for a, b in zip(uniq_days, uniq_days[1:])]
            cust_median_gap_days[row] = float(np.median(day_gaps))
        bn_counts: Dict[str, int] = {}
        bn_vols: Dict[str, float] = {}
        cum_vol = 0.0
        # Compute running stats chronologically across prior_evs, retaining last `seq_len`
        start_keep = max(0, stop - seq_len)
        # Longer-history ablations keep the prepared 32-event quantity scale.
        qty_start = max(0, stop - quantity_reference_length) if quantity_reference_length else start_keep
        max_window_log_q = max(
            (math.log1p(max(0.0, ev[3])) for ev in prior_evs[qty_start:]),
            default=0.0,
        )
        prev_day = prior_evs[0][0]
        last_day = prior_evs[-1][0]
        for idx_e, (d_i, bn_s, b_id, q_v, c_i, sc_i, bl_i, m_i) in enumerate(prior_evs):
            if frozen_v5_replay and b_id == 0:
                bn_s = ""  # Archived validation recipe pooled unknown formulations.
            prev_cnt = bn_counts.get(bn_s, 0)
            bn_counts[bn_s] = prev_cnt + 1
            bn_vols[bn_s] = bn_vols.get(bn_s, 0.0) + q_v
            cum_vol += q_v
            gap_days = max(0, d_i - prev_day) if idx_e > 0 else 0
            prev_day = d_i

            if idx_e >= start_keep:
                slot = idx_e - start_keep  # left-aligned (matching CustomerTower & UPSCANRRanker convention)
                attn_mask[row, slot] = 1
                seq_bn_ids[row, slot] = b_id
                seq_cat_ids[row, slot] = c_i
                seq_subcat_ids[row, slot] = sc_i
                seq_line_ids[row, slot] = bl_i
                seq_dates[row, slot] = float(d_i) / 365.25
                seq_rel_qty[row, slot] = (
                    min(1.0, math.log1p(max(0.0, q_v)) / max_window_log_q)
                    if max_window_log_q > 0.0
                    else 0.0
                )

                age_rel_last = max(0, last_day - d_i)
                angle = 2.0 * math.pi * (m_i - 1) / 12.0
                sequence[row, slot, 0] = min(1.0, math.log1p(q_v) / log60)
                sequence[row, slot, 1] = float(age_rel_last) / (float(age_rel_last) + 90.0)
                sequence[row, slot, 2] = min(1.0, float(gap_days) / 365.0)
                sequence[row, slot, 3] = math.sin(angle)
                sequence[row, slot, 4] = math.cos(angle)
                sequence[row, slot, 5] = min(1.0, math.log1p(prev_cnt) / log50)
                sequence[row, slot, 6] = min(1.0, bn_vols[bn_s] / max(cum_vol, 1e-6))
                sequence[row, slot, 7] = 1.0 if prev_cnt > 0 else 0.0
                sequence[row, slot, 8] = 1.0  # real_event_indicator

    return {
        "sequence": torch.from_numpy(sequence),
        "seq_rel_qty": torch.from_numpy(seq_rel_qty),
        "cust_median_gap_days": torch.from_numpy(cust_median_gap_days),
        "attention_mask": torch.from_numpy(attn_mask),
        "seq_product_ids": torch.from_numpy(seq_bn_ids),
        "seq_category_ids": torch.from_numpy(seq_cat_ids),
        "seq_subcategory_ids": torch.from_numpy(seq_subcat_ids),
        "seq_business_line_ids": torch.from_numpy(seq_line_ids),
        "seq_purchase_dates": torch.from_numpy(seq_dates),
    }


def resolve_base_names_to_skus(
    ranked_bn_indices: np.ndarray,
    val_customer_ids: Sequence[str],
    val_prediction_days: Sequence[int],
    val_allowed_skus: torch.Tensor,
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    *,
    top_k: int = 10,
) -> np.ndarray:
    """Map top-K recommended base_names to concrete eligible product_ids strictly causally (< t).

    Rule per (customer, date, base_name):
    1. Customer's most frequent (tie-break most recent) eligible pack of that base_name strictly < t;
    2. Else eligible pack matching customer's dominant container_type strictly < t;
    3. Else globally most popular eligible pack of that base_name strictly < t.
    """
    df = purchases.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df = df.sort_values("purchase_date", kind="stable")

    pid_to_idx = {str(p).strip(): i for i, p in enumerate(product_ids)}
    cat_dedup = catalogue.copy()
    cat_dedup["product_id"] = cat_dedup["product_id"].astype(str).str.strip()
    cat_dedup = cat_dedup.drop_duplicates("product_id").set_index("product_id")
    pid_cont = [
        str(cat_dedup.at[p, "container_type"]).strip().upper()
        if p in cat_dedup.index and pd.notna(cat_dedup.at[p, "container_type"])
        else ""
        for p in product_ids
    ]

    # Pre-index per-SKU sale dates (global popularity < t) and per-customer history
    sku_dates: Dict[int, np.ndarray] = {}
    for pid, grp in df.groupby("product_id", sort=False):
        if pid in pid_to_idx:
            sku_dates[pid_to_idx[pid]] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")

    cust_hist: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in df.groupby("customer_id", sort=False):
        dts = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
        pidxs = np.array([pid_to_idx.get(p, -1) for p in grp["product_id"]], dtype=np.int64)
        cust_hist[cid] = (dts, pidxs)

    origin = pd.Timestamp("2025-01-01")
    allowed_np = val_allowed_skus.detach().cpu().numpy().astype(bool)
    members_map = universe["bn_to_member_pid_indices"]

    n_rows = ranked_bn_indices.shape[0]
    out_skus = np.full((n_rows, top_k), -1, dtype=np.int64)

    for i in range(n_rows):
        cid = str(val_customer_ids[i]).strip()
        dt_64 = (origin + pd.Timedelta(days=int(val_prediction_days[i]))).to_datetime64()
        row_allowed = allowed_np[i]

        # Customer prior SKU counts, recency, and dominant container strictly < t
        sku_cnt: Counter[int] = Counter()
        sku_last: Dict[int, int] = {}
        cont_cnt: Counter[str] = Counter()
        entry = cust_hist.get(cid)
        if entry is not None:
            dts, pidxs = entry
            stop = int(dts.searchsorted(dt_64, side="left"))
            for pos_e in range(stop):
                p_i = int(pidxs[pos_e])
                if p_i >= 0:
                    sku_cnt[p_i] += 1
                    sku_last[p_i] = pos_e
                    if pid_cont[p_i]:
                        cont_cnt[pid_cont[p_i]] += 1
        dom_cont = cont_cnt.most_common(1)[0][0] if cont_cnt else ""

        for k in range(min(top_k, ranked_bn_indices.shape[1])):
            b_idx = int(ranked_bn_indices[i, k])
            if b_idx < 0:
                continue
            elig_members = [p_i for p_i in members_map[b_idx] if row_allowed[p_i]]
            if not elig_members:
                continue
            if len(elig_members) == 1:
                out_skus[i, k] = elig_members[0]
                continue
            # 1. Customer prior pack for this base_name
            prior_m = [p_i for p_i in elig_members if sku_cnt[p_i] > 0]
            if prior_m:
                out_skus[i, k] = max(prior_m, key=lambda p_i: (sku_cnt[p_i], sku_last[p_i]))
                continue
            # 2. Customer dominant container type
            if dom_cont:
                cont_m = [p_i for p_i in elig_members if pid_cont[p_i] == dom_cont]
                if cont_m:
                    out_skus[i, k] = max(
                        cont_m,
                        key=lambda p_i: int(sku_dates[p_i].searchsorted(dt_64, side="left")) if p_i in sku_dates else 0,
                    )
                    continue
            # 3. Rolling most popular eligible pack strictly < t
            out_skus[i, k] = max(
                elig_members,
                key=lambda p_i: int(sku_dates[p_i].searchsorted(dt_64, side="left")) if p_i in sku_dates else 0,
            )

    return out_skus
