"""Build fast live PyTorch inference caches for the full held-out Test set (13,184 orders)
and a compact portable 1,000-order live bundle (`test_live_bundle_1000.pt.gz`) for the ZIP package.

Strictly inference-only (uses frozen `v2_full` reference metadata and strictly causal `< query_day`
features; zero training, zero tuning, zero calibration fitting).

Usage:
    PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.build_live_test_cache \
        --allow-heldout-test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import file_digest
from recommender.catalogue import RollingCandidatePool
from recommender.experiments.roadmap_v2.evaluate_stage5_test import load_all_cleaned_purchases
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    observed_pack_ids,
)
from recommender.experiments.roadmap_v2.stage5_base_entity import build_base_name_sequences
from recommender.experiments.static_enrichment.metadata import (
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.train_retriever import build_customer_metadata

from .config import (
    CLEANED_PURCHASES_PATH,
    INTERIM_ROOT,
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    RAW_ITEMS_PATH,
    RAW_WORKBOOK_PATH,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
    V2_FULL_REFERENCE_PATH,
)
from .demo_engine import U11_IDX, _verify_v2_checkpoint
from .evaluate_v2_full_test import (
    _encode_test_company_texts,
    _fmt_sku,
    build_v2_test_baskets,
    load_reconciled_full_commercial,
    validate_v2_test_split,
)
from .extra_features import (
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    SEQ_5_ACTIVE_INDICES,
    assemble_cand_42,
    compute_extra_cand_5_arrays,
    prune_cust_7,
)

LIVE_CACHE_DIR = INTERIM_ROOT / "test_live_cache"
PORTABLE_BUNDLE_PATH = INTERIM_ROOT / "test_live_bundle_1000.pt.gz"


def _pack_bool_matrix(mat: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mat, dtype=np.uint8), axis=1)


def _select_portable_indices(
    trained_cp: np.ndarray,
    first_obs: np.ndarray,
    has_new_ret: np.ndarray,
    *,
    n_trained: int = 700,
    n_unlearned: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Select a stratified 1,000-order sample (700 Trained C&P + 300 Unlearned) across all Test dates."""
    rng = np.random.default_rng(seed)

    def _pick_stratified(mask: np.ndarray, quota: int) -> List[int]:
        pool = np.flatnonzero(mask)
        if len(pool) <= quota:
            return pool.tolist()
        # Ensure strong representation of cold-start and returning-with-new-items orders,
        # while spanning the full chronological range evenly.
        fo_idx = pool[first_obs[pool]]
        hn_idx = pool[has_new_ret[pool] & (~first_obs[pool])]
        oth_idx = pool[(~first_obs[pool]) & (~has_new_ret[pool])]

        picked: List[int] = []
        q_fo = min(len(fo_idx), max(25, int(round(quota * 0.12))))
        q_hn = min(len(hn_idx), int(round(quota * 0.35)))
        if q_fo > 0:
            picked.extend(rng.choice(fo_idx, size=q_fo, replace=False).tolist())
        if q_hn > 0:
            picked.extend(rng.choice(hn_idx, size=q_hn, replace=False).tolist())
        rem = quota - len(picked)
        if rem > 0 and len(oth_idx) > 0:
            # Uniform chronological stride + small jitter for full Jul–Nov coverage
            take = min(rem, len(oth_idx))
            picked.extend(rng.choice(oth_idx, size=take, replace=False).tolist())
        if len(picked) < quota:
            picked_set = set(picked)
            leftover = np.array([i for i in pool if i not in picked_set], dtype=np.int64)
            if len(leftover) > 0:
                picked.extend(rng.choice(leftover, size=min(quota - len(picked), len(leftover)), replace=False).tolist())
        return picked[:quota]

    tr_sel = _pick_stratified(trained_cp, n_trained)
    un_sel = _pick_stratified(~trained_cp, n_unlearned)
    combined = np.sort(np.asarray(tr_sel + un_sel, dtype=np.int64))
    return combined


def _subset_payload(
    full_payload: Dict[str, Any],
    cand42_full: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Any]:
    """Slice full live payload down to a compact portable subset with embedded float16 cand_42."""
    idx_list = indices.tolist()
    cids_sub = [full_payload["customer_ids"][i] for i in idx_list]
    cid_set = set(cids_sub)

    # Re-index deduplicated company text embeddings for only customers in subset
    old_text_table = full_payload["cust_text_table"]
    old_text_idx = full_payload["cust_text_idx"]
    new_text_rows: List[torch.Tensor] = []
    old_to_new: Dict[int, int] = {}
    new_text_idx = np.empty(len(idx_list), dtype=np.int32)
    for pos, orig_i in enumerate(idx_list):
        oi = int(old_text_idx[orig_i])
        if oi not in old_to_new:
            old_to_new[oi] = len(new_text_rows)
            new_text_rows.append(old_text_table[oi])
        new_text_idx[pos] = old_to_new[oi]

    meta = dict(full_payload["meta"])
    tr_cp_sub = full_payload["flags"]["trained_cp"][indices]
    meta["num_cases"] = len(idx_list)
    meta["trained_cp_count"] = int(tr_cp_sub.sum())
    meta["unlearned_count"] = int((~tr_cp_sub.astype(bool)).sum())
    meta["is_portable_subset"] = True
    meta["full_test_baskets"] = int(full_payload["meta"]["num_cases"])

    return {
        "meta": meta,
        "arch": full_payload["arch"],
        "prod_inputs": full_payload["prod_inputs"],
        "bn_meta": full_payload["bn_meta"],
        "sku_labels": full_payload["sku_labels"],
        "sku_cont": full_payload["sku_cont"],
        "bn_to_sku_idxs": full_payload["bn_to_sku_idxs"],
        "sku_sale_days": full_payload["sku_sale_days"],
        "cust_text_table": torch.stack(new_text_rows, dim=0),
        "cust_text_idx": torch.from_numpy(new_text_idx),
        "city_id": full_payload["city_id"][indices],
        "region_id": full_payload["region_id"][indices],
        "industry_id": full_payload["industry_id"][indices],
        "cust_7": full_payload["cust_7"][indices],
        "cust_fin": full_payload["cust_fin"][indices],
        "sequence": full_payload["sequence"][indices],
        "attention_mask": full_payload["attention_mask"][indices],
        "seq_product_ids": full_payload["seq_product_ids"][indices],
        "seq_purchase_dates": full_payload["seq_purchase_dates"][indices],
        "prediction_days": full_payload["prediction_days"][indices],
        "allowed_bn_packed": full_payload["allowed_bn_packed"][indices],
        "unseen_bn_packed": full_payload["unseen_bn_packed"][indices],
        "sku_allowed_packed": full_payload["sku_allowed_packed"][indices],
        "target_bn_lists": [full_payload["target_bn_lists"][i] for i in idx_list],
        "new_target_bn_lists": [full_payload["new_target_bn_lists"][i] for i in idx_list],
        "sku_target_lists": [full_payload["sku_target_lists"][i] for i in idx_list],
        "unmodeled_targets": [full_payload["unmodeled_targets"][i] for i in idx_list],
        "flags": {k: v[indices] for k, v in full_payload["flags"].items()},
        "customer_ids": cids_sub,
        "dates_str": [full_payload["dates_str"][i] for i in idx_list],
        "companies": {cid: full_payload["companies"][cid] for cid in cid_set if cid in full_payload["companies"]},
        "cust_bn_hist": {cid: full_payload["cust_bn_hist"][cid] for cid in cid_set if cid in full_payload["cust_bn_hist"]},
        "cust_sku_hist": {cid: full_payload["cust_sku_hist"][cid] for cid in cid_set if cid in full_payload["cust_sku_hist"]},
        "cand_42_embedded": torch.from_numpy(np.ascontiguousarray(cand42_full[indices], dtype=np.float16)),
    }


def build_live_test_caches(
    *,
    allow_heldout_test: bool = False,
    out_dir: Path = LIVE_CACHE_DIR,
    portable_out: Path = PORTABLE_BUNDLE_PATH,
    portable_trained: int = 700,
    portable_unlearned: int = 300,
) -> None:
    if not allow_heldout_test:
        raise RuntimeError("Pass --allow-heldout-test to build held-out Test live inference caches.")

    t0 = time.perf_counter()
    reference = json.loads(V2_FULL_REFERENCE_PATH.read_text(encoding="utf-8"))
    primary_ckpt = _verify_v2_checkpoint(reference["checkpoints"]["ranker"], "primary_ranker")
    shared_meta = torch.load(PREPARED_ROOT / "shared_meta.pt", map_location="cpu", weights_only=False)

    universe = shared_meta["universe"]
    product_ids: List[str] = list(shared_meta["product_ids"])
    reps: np.ndarray = np.asarray(shared_meta["reps"], dtype=np.int64)
    vocabs = shared_meta["vocabs"]
    enrichment = shared_meta["enrichment"]
    n_bn = int(universe["num_base_names"])
    by_bn = universe["bn_to_idx"]
    all_pid_to_bn = universe["all_pid_to_bn"]
    pid_to_idx = {p: i for i, p in enumerate(product_ids)}

    catalogue = load_enriched_catalogue(RAW_ITEMS_PATH, descriptions_path=None)
    cat_idx = catalogue.drop_duplicates("product_id").set_index("product_id")
    all_purchases = load_all_cleaned_purchases(CLEANED_PURCHASES_PATH)
    companies = load_enriched_companies(RAW_CUSTOMERS_PATH, purchases=all_purchases, minimum_purchases=0)
    comp_idx = companies.drop_duplicates("customer_id").set_index("customer_id")
    full_commercial = load_reconciled_full_commercial(RAW_WORKBOOK_PATH, all_purchases, catalogue)

    # Identify strictly trained customers and products (< TRAIN_CUTOFF_V2 = 2026-02-01)
    train_p = all_purchases.loc[all_purchases["purchase_date"] < TRAIN_CUTOFF_V2]
    trained_customers: Set[str] = set(train_p["customer_id"].astype(str).str.strip().unique())
    trained_bns: Set[int] = {
        by_bn[all_pid_to_bn[p]]
        for p in train_p["product_id"].astype(str).str.strip().unique()
        if p in all_pid_to_bn and all_pid_to_bn[p] in by_bn
    }

    pool = RollingCandidatePool(catalogue, all_purchases)
    allowed_train_pids = observed_pack_ids(pool.first_sales, TRAIN_CUTOFF_V2)
    assert allowed_train_pids is not None

    cases, case_counts = build_v2_test_baskets(all_purchases, product_ids, universe, pool, test_start=TEST_START_V2)
    n_test = len(cases)
    cids = [str(c.customer).strip() for c in cases]
    dates = [c.date for c in cases]
    days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]
    print(f"[live-cache] Built {n_test:,} held-out Test baskets ({dates[0].date()}..{dates[-1].date()}).", flush=True)

    observed_catalogue = catalogue.loc[catalogue["product_id"].isin(allowed_train_pids)]
    representatives = catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()
    representatives = representatives.loc[representatives["product_id"].isin(allowed_train_pids)]

    print(f"[live-cache] Building causal sequences for {n_test:,} Test baskets...", flush=True)
    sequences = build_base_name_sequences(
        all_purchases,
        cids,
        days,
        universe,
        representatives,
        vocabs["category_vocab"],
        vocabs["subcategory_vocab"],
        vocabs["business_line_vocab"],
        allow_heldout_test=True,
        frozen_v5_replay=False,
        test_start=TEST_START_V2,
    )
    sequences["sequence"][..., 3:5] = 0.0

    print(f"[live-cache] Building causal cand_38 & CUST_11 for {n_test:,} Test baskets...", flush=True)
    engine = NativeBaseNameFeatureEngine(
        all_purchases,
        full_commercial,
        catalogue,
        companies,
        product_ids,
        universe,
        reps,
        allowed_member_pids=allowed_train_pids,
        frozen_v5_replay=False,
        allow_heldout_test=True,
        test_start=TEST_START_V2,
    )
    cand38 = np.empty((n_test, n_bn, len(CANDIDATE_38_FEATURE_NAMES)), dtype=np.float16)
    cust_11 = np.empty((n_test, len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)), dtype=np.float32)
    t_f = time.perf_counter()
    for i, (cid, dt) in enumerate(zip(cids, dates, strict=True)):
        c38, cu11 = engine.build_split(cid, dt)
        cand38[i] = np.clip(c38, -65500.0, 65500.0).astype(np.float16)
        cust_11[i] = cu11
        if (i + 1) % 3000 == 0 or (i + 1) == n_test:
            print(f"[live-cache] cand_38 {i + 1:,}/{n_test:,} ({time.perf_counter() - t_f:.1f}s)", flush=True)
    cust_11 = np.nan_to_num(cust_11, nan=0.0, posinf=1.0, neginf=-1.0)

    familiarity = FamiliarityStore(
        all_purchases, observed_catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    profiles = build_customer_metadata(companies, vocabs)
    text_emb_full = _encode_test_company_texts(cids, companies, PREPARED_ROOT / "val_meta.pt")

    test_meta_check: Dict[str, Any] = {
        "customer_ids": cids,
        "prediction_days": torch.tensor(days, dtype=torch.float32),
        "first_observed": torch.tensor([c.first for c in cases], dtype=torch.bool),
        "target_bn": torch.from_numpy(np.stack([c.target for c in cases])),
        "new_target_bn": torch.from_numpy(np.stack([c.new for c in cases])),
        "allowed_bn": torch.from_numpy(np.stack([c.allowed for c in cases])),
        "unseen_bn": torch.from_numpy(np.stack([c.unseen for c in cases])),
        "tabular_features": torch.from_numpy(cust_11),
        "familiarity_bn": torch.cat(
            [familiarity.batch_features(cids[i : i + 512], dates[i : i + 512]) for i in range(0, n_test, 512)]
        ),
        **sequences,
    }
    validate_v2_test_split(test_meta_check, test_start=TEST_START_V2)

    print(f"[live-cache] Computing causal extra_cand_5 for {n_test:,} Test baskets...", flush=True)
    reg_map = {str(cid).strip(): int(prof.get("region_id", 0)) for cid, prof in profiles.items()}
    ind_map = {str(cid).strip(): int(prof.get("industry_id", 0)) for cid, prof in profiles.items()}
    extra5 = compute_extra_cand_5_arrays(
        commercial_events=full_commercial,
        universe=universe,
        customer_region_map=reg_map,
        customer_industry_map=ind_map,
        splits={"test": test_meta_check},
        out_paths=None,
        test_start=pd.Timestamp("2099-01-01"),
    )["test"]

    # Assemble float16 cand_42 memmap directly to disk for zero-copy <0.15ms per-case slicing
    out_dir.mkdir(parents=True, exist_ok=True)
    cand42_path = out_dir / "test_cand_42.npy"
    print(f"[live-cache] Writing float16 cand_42 memmap ({n_test:,} x {n_bn} x 42) -> {cand42_path}...", flush=True)
    cand42_mm = np.lib.format.open_memmap(
        cand42_path, mode="w+", dtype=np.float16, shape=(n_test, n_bn, NUM_CAND_42_FEATURES)
    )
    chunk = 512
    for s in range(0, n_test, chunk):
        e = min(n_test, s + chunk)
        c42 = assemble_cand_42(cand38[s:e], extra5[s:e])
        cand42_mm[s:e] = np.clip(c42, -65500.0, 65500.0).astype(np.float16)
    cand42_mm.flush()
    del cand38, extra5

    # Compute exact Trained Customer / Trained Product flags
    trained_c = np.array([cid in trained_customers for cid in cids], dtype=bool)
    trained_p = np.zeros(n_test, dtype=bool)
    target_bn_lists: List[List[int]] = []
    new_target_bn_lists: List[List[int]] = []
    sku_target_lists: List[List[int]] = []
    unmodeled_targets: List[List[Tuple[str, bool]]] = []
    for i, case in enumerate(cases):
        t_idxs = np.flatnonzero(case.target).tolist()
        n_idxs = np.flatnonzero(case.new).tolist()
        target_bn_lists.append(t_idxs)
        new_target_bn_lists.append(n_idxs)
        sku_target_lists.append(np.flatnonzero(case.sku_target).tolist())
        unmod_list = [(str(bn), bool(is_n)) for bn, is_n in case.unmodeled_novelty]
        unmodeled_targets.append(unmod_list)
        all_t_trained = (case.extra_bn_targets == 0) and len(t_idxs) > 0 and all(b in trained_bns for b in t_idxs)
        trained_p[i] = all_t_trained
    trained_cp = trained_c & trained_p
    first_obs = np.array([c.first for c in cases], dtype=bool)
    has_new_ret = np.array([(not c.first) and (bool(c.new.any()) or c.extra_new_bn_targets > 0) for c in cases], dtype=bool)
    multi_item = np.array([(int(c.target.sum()) + int(c.extra_bn_targets)) >= 2 for c in cases], dtype=bool)
    print(
        f"[live-cache] Slice counts: Trained C&P={int(trained_cp.sum()):,} | Unlearned={int((~trained_cp).sum()):,} "
        f"(unlearned_c={int((~trained_c).sum()):,}, unlearned_p={int((~trained_p).sum()):,})",
        flush=True,
    )

    # Deduplicate company text embeddings (float16) by customer_id
    cid_to_tidx: Dict[str, int] = {}
    unique_text_rows: List[torch.Tensor] = []
    cust_text_idx = np.empty(n_test, dtype=np.int32)
    for i, cid in enumerate(cids):
        if cid not in cid_to_tidx:
            cid_to_tidx[cid] = len(unique_text_rows)
            unique_text_rows.append(text_emb_full[i].to(torch.float16))
        cust_text_idx[i] = cid_to_tidx[cid]
    cust_text_table = torch.stack(unique_text_rows, dim=0)
    del text_emb_full

    # Compact per-formulation and per-SKU metadata
    bn_meta: List[Dict[str, str]] = []
    for b_i, bn_name in enumerate(universe["base_names"]):
        rep_pid = product_ids[int(reps[b_i])]
        row = cat_idx.loc[rep_pid] if rep_pid in cat_idx.index else {}
        bn_meta.append(
            {
                "nm": str(bn_name),
                "br": str(row.get("brand", "")) if pd.notna(row.get("brand", "")) else "",
                "cat": str(row.get("product_category", "")) if pd.notna(row.get("product_category", "")) else "",
            }
        )

    sku_labels = [_fmt_sku(pid, cat_idx) for pid in product_ids]
    sku_cont = [
        str(cat_idx.at[p, "container_type"]).strip().upper()
        if p in cat_idx.index and pd.notna(cat_idx.at[p, "container_type"])
        else ""
        for p in product_ids
    ]
    bn_to_sku_idxs = [[int(x) for x in universe["bn_to_member_pid_indices"][b]] for b in range(n_bn)]

    # Compact chronological histories for instant (<0.1ms) causal history display & SKU resolution
    test_cid_set = set(cids)
    df_sorted = all_purchases[["customer_id", "purchase_date", "product_id", "quantity"]].sort_values(
        "purchase_date", kind="stable"
    )
    df_sorted = df_sorted.assign(
        day_i=(df_sorted["purchase_date"] - PURCHASE_DATE_ORIGIN).dt.days.astype(np.int32)
    )
    sku_sale_days: Dict[int, np.ndarray] = {}
    for pid, grp in df_sorted.groupby("product_id", sort=False):
        p_str = str(pid).strip()
        if p_str in pid_to_idx:
            sku_sale_days[pid_to_idx[p_str]] = grp["day_i"].to_numpy(dtype=np.int32)

    cust_bn_hist: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cust_sku_hist: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    df_test_custs = df_sorted.loc[df_sorted["customer_id"].astype(str).str.strip().isin(test_cid_set)]
    for cid_raw, grp in df_test_custs.groupby("customer_id", sort=False):
        cid = str(cid_raw).strip()
        d_arr = grp["day_i"].to_numpy(dtype=np.int32)
        pids_g = grp["product_id"].astype(str).str.strip().tolist()
        s_arr = np.array([pid_to_idx.get(p, -1) for p in pids_g], dtype=np.int16)
        cust_sku_hist[cid] = (d_arr, s_arr)

        # Aggregate by (day, bn_idx) for concise human-readable purchase history
        agg_d: Dict[Tuple[int, int], float] = defaultdict(float)
        for dv, pv, qv in zip(d_arr.tolist(), pids_g, grp["quantity"].to_numpy(dtype=np.float32).tolist(), strict=True):
            bn_s = all_pid_to_bn.get(pv, "")
            b_i = by_bn.get(bn_s, -1)
            if b_i >= 0:
                agg_d[(dv, b_i)] += float(qv)
        if agg_d:
            items = sorted(agg_d.items(), key=lambda kv: (kv[0][0], -kv[1]))
            cust_bn_hist[cid] = (
                np.array([k[0] for k, _ in items], dtype=np.int32),
                np.array([k[1] for k, _ in items], dtype=np.int16),
                np.array([v for _, v in items], dtype=np.float32),
            )

    companies_dict: Dict[str, Dict[str, str]] = {}
    for cid in test_cid_set:
        if cid in comp_idx.index:
            crow = comp_idx.loc[cid]
            nm = str(crow.get("company_name", cid)).strip() if pd.notna(crow.get("company_name", "")) else cid
            ct = str(crow.get("city", "—")).strip() if pd.notna(crow.get("city", "")) else "—"
            rg = str(crow.get("region_code", "—")).strip() if pd.notna(crow.get("region_code", "")) else "—"
            ind = str(crow.get("industry_branch", "—")).strip() if pd.notna(crow.get("industry_branch", "")) else "—"
            desc = str(crow.get("description", "")).strip() if pd.notna(crow.get("description", "")) else ""
            if len(desc) > 220:
                desc = desc[:220] + "…"
        else:
            nm, ct, rg, ind, desc = cid, "—", "—", "—", ""
        companies_dict[cid] = {"nm": nm or cid, "ct": ct or "—", "rg": rg or "—", "ind": ind or "—", "desc": desc or "—"}

    # Customer financial summary (prep%, def_days, ovd%)
    prep_pct = np.rint(100.0 * cust_11[:, U11_IDX["cust_prepaid_or_cash_ratio"]]).astype(np.int16)
    def_days = np.rint(
        np.expm1(np.maximum(0.0, cust_11[:, U11_IDX["cust_avg_deferral_days_norm"]]) * math.log1p(180.0))
    ).astype(np.int16)
    ovd_pct = np.rint(100.0 * cust_11[:, U11_IDX["cust_overdue_unpaid_ratio"]]).astype(np.int16)
    cust_fin = np.stack([prep_pct, def_days, ovd_pct], axis=1)

    cust_7_t = prune_cust_7(torch.from_numpy(cust_11)).to(torch.float16)
    prim_dep = reference["validation_metrics_primary_cal_discovery"]["affine_oof_calibration"]["deployment_full_fit"]
    arch_ref = dict(reference["active_architecture"])
    ck = torch.load(primary_ckpt, map_location="cpu", weights_only=False)

    arch_cfg = {
        "product_attr_vocab_sizes": {k: len(v) for k, v in enrichment["product_vocabs"].items()},
        "num_cities": len(vocabs["city_vocab"]) + 1,
        "num_regions": len(vocabs["region_vocab"]) + 1,
        "num_industries": len(vocabs["industry_vocab"]) + 1,
        "num_demand_features": int(shared_meta["prod_inputs"]["demand_features"].shape[1]),
        "num_candidate_features": int(arch_ref.get("num_candidate_features", NUM_CAND_42_FEATURES)),
        "num_customer_features": int(arch_ref.get("num_customer_features", NUM_CUST_7_FEATURES)),
        "num_pairwise_bias_features": int(arch_ref.get("num_pairwise_bias_features", NUM_CAND_42_FEATURES)),
        "active_seq_stat_indices": list(arch_ref.get("active_seq_stat_indices", SEQ_5_ACTIVE_INDICES)),
        "pure_v_query": bool(arch_ref.get("pure_v_query", True)),
        "option_a_unseen_head": bool(arch_ref.get("option_a_unseen_head", True)),
        "d_model": int(arch_ref["d_model"]),
        "num_heads": int(arch_ref["num_heads"]),
        "memory_ffn_dim": int(arch_ref["memory_ffn_dim"]),
        "memory_layers": int(arch_ref.get("memory_layers", 2)),
        "mlp_widths": [int(w) for w in arch_ref["mlp_widths"]],
        "dropout": float(arch_ref.get("dropout", 0.12)),
    }
    prod_inputs_compact = {
        "text_embedding": shared_meta["prod_inputs"]["text_embedding"].to(torch.float16),
        "demand_features": shared_meta["prod_inputs"]["demand_features"].to(torch.float16),
        "attr_ids": {k: v.to(torch.int64) for k, v in shared_meta["prod_inputs"]["attr_ids"].items()},
    }

    full_payload: Dict[str, Any] = {
        "meta": {
            "reference_id": reference["reference_id"],
            "arm": str(arch_ref.get("arm", "w2_2mem_recency_hl365")),
            "epoch": int(ck.get("epoch", 10)),
            "split": "heldout_test",
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "num_cases": n_test,
            "full_test_baskets": n_test,
            "is_portable_subset": False,
            "trained_cp_count": int(trained_cp.sum()),
            "unlearned_count": int((~trained_cp).sum()),
            "modeled_baskets": int(case_counts["modeled_baskets"]),
            "num_formulations": n_bn,
            "num_skus": len(product_ids),
            "train_params": int(arch_ref.get("trainable_params", 1_369_942)),
            "per_request_params": int(arch_ref.get("per_request_params", 1_125_270)),
            "calibration": {"a": float(prim_dep["a"]), "b": float(prim_dep["b"])},
            "checkpoint_sha256": file_digest(primary_ckpt),
        },
        "arch": arch_cfg,
        "prod_inputs": prod_inputs_compact,
        "bn_meta": bn_meta,
        "sku_labels": sku_labels,
        "sku_cont": sku_cont,
        "bn_to_sku_idxs": bn_to_sku_idxs,
        "sku_sale_days": sku_sale_days,
        "cust_text_table": cust_text_table,
        "cust_text_idx": torch.from_numpy(cust_text_idx),
        "city_id": torch.tensor([profiles.get(cid, {}).get("city_id", 0) for cid in cids], dtype=torch.int32),
        "region_id": torch.tensor([profiles.get(cid, {}).get("region_id", 0) for cid in cids], dtype=torch.int32),
        "industry_id": torch.tensor([profiles.get(cid, {}).get("industry_id", 0) for cid in cids], dtype=torch.int32),
        "cust_7": cust_7_t,
        "cust_fin": torch.from_numpy(cust_fin),
        "sequence": sequences["sequence"].to(torch.float16),
        "attention_mask": sequences["attention_mask"].to(torch.uint8),
        "seq_product_ids": sequences["seq_product_ids"].to(torch.int16),
        "seq_purchase_dates": sequences["seq_purchase_dates"].to(torch.float32),
        "prediction_days": torch.tensor(days, dtype=torch.int32),
        "allowed_bn_packed": _pack_bool_matrix(np.stack([c.allowed for c in cases])),
        "unseen_bn_packed": _pack_bool_matrix(np.stack([c.unseen for c in cases])),
        "sku_allowed_packed": _pack_bool_matrix(np.stack([c.sku_allowed for c in cases])),
        "target_bn_lists": target_bn_lists,
        "new_target_bn_lists": new_target_bn_lists,
        "sku_target_lists": sku_target_lists,
        "unmodeled_targets": unmodeled_targets,
        "flags": {
            "first_observed": first_obs.astype(np.uint8),
            "trained_c": trained_c.astype(np.uint8),
            "trained_p": trained_p.astype(np.uint8),
            "trained_cp": trained_cp.astype(np.uint8),
            "has_new_ret": has_new_ret.astype(np.uint8),
            "multi_item": multi_item.astype(np.uint8),
        },
        "customer_ids": cids,
        "dates_str": [str(dt.date()) for dt in dates],
        "companies": companies_dict,
        "cust_bn_hist": cust_bn_hist,
        "cust_sku_hist": cust_sku_hist,
    }

    meta_path = out_dir / "test_live_meta.pt"
    torch.save(full_payload, meta_path)
    print(f"[live-cache] Saved full 13,184-order metadata -> {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Also build portable 1,000-order compressed bundle for ZIP distribution
    sel_indices = _select_portable_indices(
        trained_cp, first_obs, has_new_ret, n_trained=portable_trained, n_unlearned=portable_unlearned
    )
    portable_payload = _subset_payload(full_payload, cand42_mm, sel_indices)
    portable_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[live-cache] Writing compressed portable {len(sel_indices)}-order bundle -> {portable_out}...", flush=True)
    with gzip.open(portable_out, "wb", compresslevel=6) as gz:
        torch.save(portable_payload, gz)
    print(
        f"[live-cache] Done in {time.perf_counter() - t0:.1f}s!\n"
        f"  Full local cache: {out_dir} ({cand42_path.stat().st_size / 1e6:.1f} MB cand_42 + {meta_path.stat().st_size / 1e6:.1f} MB meta)\n"
        f"  Portable bundle:  {portable_out} ({portable_out.stat().st_size / 1e6:.1f} MB, {len(sel_indices)} real Test orders)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-heldout-test", action="store_true", help="Authorize held-out Test live cache build.")
    parser.add_argument("--out-dir", type=Path, default=LIVE_CACHE_DIR)
    parser.add_argument("--portable-out", type=Path, default=PORTABLE_BUNDLE_PATH)
    parser.add_argument("--portable-trained", type=int, default=700)
    parser.add_argument("--portable-unlearned", type=int, default=300)
    args = parser.parse_args()
    build_live_test_caches(
        allow_heldout_test=args.allow_heldout_test,
        out_dir=args.out_dir,
        portable_out=args.portable_out,
        portable_trained=args.portable_trained,
        portable_unlearned=args.portable_unlearned,
    )


if __name__ == "__main__":
    main()
