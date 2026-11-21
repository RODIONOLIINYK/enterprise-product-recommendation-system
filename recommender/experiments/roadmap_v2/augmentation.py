"""Stage 2 training-only augmentations: strictly leak-free 60-day auxiliary labels & History-SKU Dropout."""

from bisect import bisect_left
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.split import TEST_START, TRAIN_CUTOFF


def build_strict_60d_aux_labels(
    purchases: pd.DataFrame,
    customer_ids: Sequence[str],
    prediction_days: Sequence[int],
    product_ids: Sequence[str],
    *,
    horizon_days: int = 60,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
) -> torch.Tensor:
    """Build (N_train, P) bool matrix of products purchased in [t, min(t + horizon_days, TRAIN_CUTOFF)).

    Zero leakage guarantees:
    - Refuses any query date >= train_cutoff (strictly training rows only).
    - Refuses any purchase date >= TEST_START.
    - Hard-caps every query's future window strictly at `< train_cutoff`, so even a September 2025
      training transition NEVER reads any October+ validation purchase.
    """
    if purchases.empty:
        return torch.zeros((len(customer_ids), len(product_ids)), dtype=torch.bool)

    dates_all = pd.to_datetime(purchases["purchase_date"], format="mixed")
    if bool(dates_all.isna().any()) or not bool(dates_all.lt(TEST_START).all()):
        raise ValueError(f"Development purchases only (< {TEST_START.date()}).")

    # Filter strictly to training-period purchases (< train_cutoff) BEFORE indexing
    train_mask = dates_all < train_cutoff
    df = purchases.loc[train_mask, ["customer_id", "purchase_date", "product_id"]].copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed")
    df = df.sort_values(["customer_id", "purchase_date"], kind="stable")

    pid_to_idx = {str(p).strip(): i for i, p in enumerate(product_ids)}
    cust_dates: Dict[str, np.ndarray] = {}
    cust_pids: Dict[str, np.ndarray] = {}
    for cid, grp in df.groupby("customer_id", sort=False):
        dts = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
        idxs = np.array([pid_to_idx.get(p, -1) for p in grp["product_id"]], dtype=np.int64)
        valid = idxs >= 0
        if bool(valid.any()):
            cust_dates[cid] = dts[valid]
            cust_pids[cid] = idxs[valid]

    origin = pd.Timestamp("2025-01-01")
    cutoff_64 = pd.Timestamp(train_cutoff).to_datetime64()
    horizon_td = pd.Timedelta(days=int(horizon_days))

    labels = np.zeros((len(customer_ids), len(product_ids)), dtype=bool)
    for row, (cid_raw, day) in enumerate(zip(customer_ids, prediction_days)):
        t = origin + pd.Timedelta(days=int(day))
        if t >= train_cutoff:
            raise ValueError(f"60-day aux labels are training-only (< {train_cutoff.date()}); got {t.date()}.")
        cid = str(cid_raw).strip()
        dts = cust_dates.get(cid)
        if dts is None:
            continue
        t_64 = t.to_datetime64()
        end_ts = min(t + horizon_td, pd.Timestamp(train_cutoff))
        end_64 = min(end_ts.to_datetime64(), cutoff_64)
        start_i = int(dts.searchsorted(t_64, side="left"))
        stop_i = int(dts.searchsorted(end_64, side="left"))
        if stop_i > start_i:
            labels[row, cust_pids[cid][start_i:stop_i]] = True

    return torch.from_numpy(labels)


def resolve_personal_repeat_columns(feature_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Resolve exact-SKU and base_name personal repeat feature column indices by name."""
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    sku_cols: Dict[str, int] = {}
    for k in (
        "is_repeat_sku",
        "user_sku_order_count_norm",
        "user_sku_qty_share",
        "days_since_last_sku_buy_norm",
        "sku_urgency",
    ):
        if k in name_to_idx:
            sku_cols[k] = name_to_idx[k]

    bn_cols: Dict[str, int] = {}
    for k in (
        "has_bought_base_sku",
        "user_base_sku_order_count_norm",
        "user_base_sku_qty_share",
        "days_since_last_base_sku_buy_norm",
        "base_sku_urgency",
    ):
        if k in name_to_idx:
            bn_cols[k] = name_to_idx[k]
    return {"sku": sku_cols, "base_name": bn_cols}


def _zero_personal_block(feats: torch.Tensor, mask_bc: torch.Tensor, cols: Mapping[str, int]) -> None:
    """In-place reset personal repeat features on selected (batch, cand) positions to exact unseen defaults."""
    if not cols or not bool(mask_bc.any()):
        return
    for name, idx in cols.items():
        default_val = 1.0 if "days_since_last_" in name else 0.0
        col = feats[..., idx]
        col[mask_bc] = default_val


def apply_stage1_history_sku_dropout(
    features: torch.Tensor,
    targets: torch.Tensor,
    prod_bn_ids: torch.Tensor,
    feature_names: Sequence[str],
    *,
    p_drop: float = 0.35,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, int]:
    """Training-only History-SKU Dropout for Stage 1 MLP (resets repeat personal features on 1 repeat positive).

    For each query in the batch with at least one repeat positive, with probability `p_drop` picks
    1 repeat positive SKU `p*` and resets:
    - exact-SKU personal repeat features of `p*` (if present in schema), AND
    - base_name personal repeat features of ALL SKUs sharing `p*`'s base_name (if present in schema)
    to exact unseen defaults (`is_repeat=0, count=0, qty_share=0, days_since=1.0, urgency=0`).
    """
    if p_drop <= 0.0:
        return features, 0

    col_map = resolve_personal_repeat_columns(feature_names)
    sku_cols = col_map["sku"]
    bn_cols = col_map["base_name"]
    if not sku_cols and not bn_cols:
        return features, 0

    out = features.clone()
    bsz, n_prod, _ = out.shape
    dev = out.device

    # Identify repeat positives via is_repeat_sku (preferred) or has_bought_base_sku
    if "is_repeat_sku" in sku_cols:
        is_rep = out[..., sku_cols["is_repeat_sku"]] > 0.5
    else:
        is_rep = out[..., bn_cols["has_bought_base_sku"]] > 0.5
    rep_pos = (targets > 0.5) & is_rep

    bn_ids_dev = prod_bn_ids.to(dev)
    sku_mask = torch.zeros((bsz, n_prod), dtype=torch.bool, device=dev)
    bn_mask = torch.zeros((bsz, n_prod), dtype=torch.bool, device=dev)

    # Sample dropout decisions on CPU generator for deterministic reproducibility across MPS/CPU
    gen_cpu = generator
    rand_u = torch.rand(bsz, generator=gen_cpu).tolist()
    rep_pos_cpu = rep_pos.detach().cpu()

    dropped_count = 0
    for b in range(bsz):
        if rand_u[b] >= p_drop:
            continue
        pos_indices = torch.where(rep_pos_cpu[b])[0]
        if len(pos_indices) == 0:
            continue
        pick_i = int(torch.randint(0, len(pos_indices), (1,), generator=gen_cpu).item())
        p_star = int(pos_indices[pick_i].item())
        sku_mask[b, p_star] = True
        if bn_cols:
            b_star = int(prod_bn_ids[p_star].item())
            if b_star > 0:
                bn_mask[b] = bn_ids_dev == b_star
            else:
                bn_mask[b, p_star] = True
        dropped_count += 1

    if dropped_count > 0:
        _zero_personal_block(out, sku_mask, sku_cols)
        _zero_personal_block(out, bn_mask, bn_cols)
    return out, dropped_count


def apply_stage2_history_sku_dropout(
    inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    seq_universe_indices: torch.Tensor,
    prod_bn_ids: torch.Tensor,
    feature_names: Sequence[str],
    *,
    p_drop: float = 0.35,
    drop_base_name_in_sequence: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Training-only History-SKU Dropout for Stage 2 Transformer.

    For each query in the batch with at least one repeat positive in the slate, with probability
    `p_drop` picks 1 repeat positive candidate `p*` and masks it out of BOTH:
    1. Candidate pairwise features (resetting exact-SKU repeat features of `p*` and base_name
       repeat features of any slate candidate sharing `p*`'s base_name to unseen defaults), AND
    2. The customer's 32-event purchase history sequence (`attention_mask`, `sequence`,
       `seq_product_ids`, `seq_category_ids`, `seq_subcategory_ids`, `seq_business_line_ids`)
       for every history event matching `p*` (and optionally any pack-mate sharing `p*`'s base_name).
    """
    if p_drop <= 0.0:
        return inputs, 0

    col_map = resolve_personal_repeat_columns(feature_names)
    sku_cols = col_map["sku"]
    bn_cols = col_map["base_name"]
    if not sku_cols and not bn_cols:
        return inputs, 0

    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    feats = out["pairwise_features"]
    bsz, n_cand, _ = feats.shape
    dev = feats.device

    if "is_repeat_sku" in sku_cols:
        is_rep = feats[..., sku_cols["is_repeat_sku"]] > 0.5
    else:
        is_rep = feats[..., bn_cols["has_bought_base_sku"]] > 0.5
    rep_pos = (labels.to(dev) > 0.5) & is_rep

    bn_ids_cpu = prod_bn_ids.cpu()
    cand_idx_cpu = candidate_indices.detach().cpu()
    cand_bn_cpu = bn_ids_cpu[cand_idx_cpu]  # (B, C)
    seq_u_cpu = seq_universe_indices.detach().cpu()  # (B, 32), -1 for padding/outside universe
    seq_bn_cpu = torch.where(
        seq_u_cpu >= 0,
        bn_ids_cpu[seq_u_cpu.clamp(min=0)],
        torch.zeros_like(seq_u_cpu),
    )

    sku_mask_cpu = torch.zeros((bsz, n_cand), dtype=torch.bool)
    bn_mask_cpu = torch.zeros((bsz, n_cand), dtype=torch.bool)
    seq_drop_mask_cpu = torch.zeros_like(seq_u_cpu, dtype=torch.bool)

    rand_u = torch.rand(bsz, generator=generator).tolist()
    rep_pos_cpu = rep_pos.detach().cpu()

    dropped_count = 0
    for b in range(bsz):
        if rand_u[b] >= p_drop:
            continue
        pos_cands = torch.where(rep_pos_cpu[b])[0]
        if len(pos_cands) == 0:
            continue
        pick_i = int(torch.randint(0, len(pos_cands), (1,), generator=generator).item())
        c_star = int(pos_cands[pick_i].item())
        u_star = int(cand_idx_cpu[b, c_star].item())
        b_star = int(bn_ids_cpu[u_star].item())

        sku_mask_cpu[b, c_star] = True
        if bn_cols:
            if b_star > 0:
                bn_mask_cpu[b] = cand_bn_cpu[b] == b_star
            else:
                bn_mask_cpu[b, c_star] = True

        # Mask matching tokens in 32-event sequence
        if drop_base_name_in_sequence and b_star > 0:
            seq_drop_mask_cpu[b] = seq_bn_cpu[b] == b_star
        else:
            seq_drop_mask_cpu[b] = seq_u_cpu[b] == u_star
        dropped_count += 1

    if dropped_count > 0:
        _zero_personal_block(feats, sku_mask_cpu.to(dev), sku_cols)
        _zero_personal_block(feats, bn_mask_cpu.to(dev), bn_cols)

        seq_drop_dev = seq_drop_mask_cpu.to(dev)
        if bool(seq_drop_dev.any()):
            out["attention_mask"][seq_drop_dev] = 0
            out["sequence"][seq_drop_dev] = 0.0
            for key in ("seq_product_ids", "seq_category_ids", "seq_subcategory_ids", "seq_business_line_ids"):
                if key in out and out[key] is not None:
                    out[key][seq_drop_dev] = 0

    return out, dropped_count


def build_seq_universe_indices(
    seq_canonical_skus: Sequence[Sequence[str]],
    product_ids: Sequence[str],
) -> torch.Tensor:
    """Map (N, 32) canonical sequence SKU strings to 0..P-1 product universe indices (-1 if padding/unmapped)."""
    pid_to_idx = {str(p).strip(): i for i, p in enumerate(product_ids)}
    rows = [
        [pid_to_idx.get(str(sku).strip(), -1) for sku in seq_row]
        for seq_row in seq_canonical_skus
    ]
    return torch.tensor(rows, dtype=torch.long)
