"""Standalone CatBoost benchmark (strictly 42-feature `cand_42 + CUST_7` + Option A Unseen-Only) & Neural Blend on `v2_full`.

Strictly chronological, zero held-out Test (>= 2026-07-01) access:
  - Train: `< 2026-02-01` (all eligible positives weight=1.0 + uniformly sampled negatives without
    replacement weighted by exact Horvitz-Thompson inverse probability `N_neg / k`, preserving
    unbiased full-pool log-odds calibration; zero early stopping on validation).
  - Val: `2026-02-01..2026-06-30` — scores ALL `1,141` formulations per basket (zero truncation!).

Models (`ALL_CB_MODELS`, strictly 42-feature `cand_42 + CUST_7` schema):
  1. `cb_mixed_cand42`: Primary mixed CatBoost on `cand_42 + CUST_7 + 9 static categoricals` (58 features).
  2. `cb_optionA_unseen_cand42`: Dedicated Option A Unseen-Only CatBoost on `cand_42[5:]` (37 non-repeat
     features) + `CUST_7 + 9 static categoricals` (53 features), trained strictly on `unseen_bn & allowed_bn`
     baskets with >= 1 new target, evaluated on BOTH Pure Unseen@10/@25 AND 2-fold customer-grouped
     OOF affine blend into `cb_mixed_cand42` Mixed Top-10.
  3. `cb_plus_neural_blend` (optional via `--neural-logits`): 2-fold customer-grouped OOF linear
     logit ensemble combining `cb_mixed_cand42` with saved `arm4` Wave 2B validation logits.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.roadmap_v2.config import EVIDENCE_ROOT as V1_EVIDENCE_ROOT
from recommender.experiments.roadmap_v2.run_stage5 import _verify_reference_files
from recommender.experiments.roadmap_v2.run_stage5_ablations import _evaluate_with_cohorts_and_sku
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    paired_base_name_bootstrap_ci,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes

from .prepare_v2_full import load_prepared_metadata
from .config import (
    ACTIVE_V2_FULL_REFERENCE_ID,
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    PREPARED_ROOT,
    RAW_ITEMS_PATH,
    SEED,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
    load_v2_development_purchases,
    resolve_v2_full_reference_predictions,
)
from .provenance import input_digest, prepared_dataset_digest
from .extra_features import (
    CAND_42_FEATURE_NAMES,
    CAND_42_NON_REPEAT_INDICES,
    CAND_42_NON_REPEAT_NAMES,
    CUST_7_FEATURE_NAMES,
    CUST_7_KEEP_INDICES,
    assemble_cand_42,
    build_or_load_extra_cand_5,
)
from .train_v2_full_baseline import rank_eligible_formulations, sync_outputs_to_dir
from .wave2b_models import (
    _customer_fold_assignment,
    _mixed_top10_metrics,
    evaluate_pure_unseen_predictions,
    fit_oof_affine_unseen_blend,
)

STATIC_CUST_CAT_COLS: Tuple[str, ...] = ("city_id", "region_id", "industry_id")
STATIC_PROD_CAT_COLS: Tuple[str, ...] = (
    "category",
    "subcategory",
    "business_line",
    "tier",
    "unit",
    "country_of_origin",
)
ALL_CAT_COLS: Tuple[str, ...] = STATIC_CUST_CAT_COLS + STATIC_PROD_CAT_COLS
ALL_CB_MODELS: Tuple[str, ...] = ("cb_mixed_cand42", "cb_optionA_unseen_cand42")
DEFAULT_CB_MODELS: Tuple[str, ...] = ALL_CB_MODELS


def _extract_static_arrays(
    shared_meta: Mapping[str, Any], split_meta: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(cust_cats[N, 3], prod_cats[C, 6])` as int32 arrays."""
    cust_cats = np.stack(
        [split_meta[k].cpu().numpy().astype(np.int32) for k in STATIC_CUST_CAT_COLS],
        axis=1,
    )
    attr_ids = shared_meta["prod_inputs"]["attr_ids"]
    prod_cats = np.stack(
        [attr_ids[k].cpu().numpy().astype(np.int32) for k in STATIC_PROD_CAT_COLS],
        axis=1,
    )
    return cust_cats, prod_cats


def _schema_info(model_name: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool]:
    """Return `(cand_names, cust_names, unseen_only)` for a CatBoost model variant."""
    if model_name == "cb_mixed_cand42":
        return CAND_42_FEATURE_NAMES, CUST_7_FEATURE_NAMES, False
    if model_name == "cb_optionA_unseen_cand42":
        return CAND_42_NON_REPEAT_NAMES, CUST_7_FEATURE_NAMES, True
    raise ValueError(f"Unknown CatBoost model_name={model_name!r}; expected one of {ALL_CB_MODELS}")


def _extract_cand_block(
    model_name: str, cand38_slice: np.ndarray, extra5_slice: np.ndarray
) -> np.ndarray:
    c42 = assemble_cand_42(cand38_slice, extra5_slice)
    if model_name == "cb_optionA_unseen_cand42":
        return c42[..., list(CAND_42_NON_REPEAT_INDICES)]
    return c42


def _extract_cust_block(cust11: np.ndarray) -> np.ndarray:
    return np.asarray(cust11[:, list(CUST_7_KEEP_INDICES)], dtype=np.float32)


def _build_dataframe_from_arrays(
    num_mat: np.ndarray,
    cat_mat: np.ndarray,
    num_names: Sequence[str],
    cat_names: Sequence[str],
) -> pd.DataFrame:
    data: Dict[str, np.ndarray] = {}
    for j, name in enumerate(num_names):
        data[name] = num_mat[:, j]
    for j, name in enumerate(cat_names):
        data[name] = cat_mat[:, j]
    return pd.DataFrame(data, copy=False)


def sample_weighted_catboost_training_pool(
    *,
    model_name: str,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    tr_cand38_mm: np.ndarray,
    tr_extra5_mm: np.ndarray,
    negatives_per_basket: int,
    seed: int,
    chunk_rows: int = 2048,
) -> Tuple[Pool, Dict[str, Any]]:
    """Build exact Horvitz-Thompson importance-weighted CatBoost training Pool (zero leakage)."""
    cand_names, cust_names, unseen_only = _schema_info(model_name)
    num_names = tuple(cand_names) + tuple(cust_names)
    if negatives_per_basket <= 0:
        raise ValueError("negatives_per_basket must be positive.")
    rng = np.random.default_rng(int(seed))

    n_rows = len(tr_meta["customer_ids"])
    n_train_bn = int(shared_meta["universe"]["trained_base_names"])
    allowed_np = tr_meta["allowed_bn"].cpu().numpy()[:, :n_train_bn].astype(bool)
    target_np = tr_meta["target_bn"].cpu().numpy()[:, :n_train_bn].astype(bool)
    if unseen_only:
        unseen_np = tr_meta["unseen_bn"].cpu().numpy()[:, :n_train_bn].astype(bool)
        elig_np = allowed_np & unseen_np
    else:
        elig_np = allowed_np

    pos_mask = target_np & elig_np
    neg_mask = (~target_np) & elig_np
    cust_11_all = tr_meta["tabular_features"].cpu().numpy().astype(np.float32)
    cust_all = _extract_cust_block(cust_11_all)
    cust_cats_all, prod_cats_all = _extract_static_arrays(shared_meta, tr_meta)
    prod_cats_tr = prod_cats_all[:n_train_bn]

    num_chunks: List[np.ndarray] = []
    cat_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    w_chunks: List[np.ndarray] = []

    total_baskets_used = 0
    total_positives = 0
    total_negatives_sampled = 0
    total_negative_weight = 0.0

    for s0 in range(0, n_rows, int(chunk_rows)):
        s1 = min(n_rows, s0 + int(chunk_rows))
        sel_rows_list: List[int] = []
        sel_items_list: List[int] = []
        sel_y_list: List[float] = []
        sel_w_list: List[float] = []

        for local_i, global_i in enumerate(range(s0, s1)):
            pos_idx = np.flatnonzero(pos_mask[global_i])
            if unseen_only and len(pos_idx) == 0:
                continue
            neg_idx = np.flatnonzero(neg_mask[global_i])
            if len(pos_idx) == 0 and len(neg_idx) == 0:
                continue
            total_baskets_used += 1

            for p in pos_idx.tolist():
                sel_rows_list.append(local_i)
                sel_items_list.append(p)
                sel_y_list.append(1.0)
                sel_w_list.append(1.0)
            total_positives += len(pos_idx)

            n_neg = len(neg_idx)
            if n_neg > 0:
                k = min(int(negatives_per_basket), n_neg)
                chosen = neg_idx if k == n_neg else rng.choice(neg_idx, size=k, replace=False)
                ipw = float(n_neg) / float(k)
                for n_it in chosen.tolist():
                    sel_rows_list.append(local_i)
                    sel_items_list.append(n_it)
                    sel_y_list.append(0.0)
                    sel_w_list.append(ipw)
                total_negatives_sampled += k
                total_negative_weight += float(n_neg)

        if not sel_rows_list:
            continue

        loc_arr = np.asarray(sel_rows_list, dtype=np.int64)
        glob_arr = loc_arr + s0
        it_arr = np.asarray(sel_items_list, dtype=np.int64)

        sampled_38 = tr_cand38_mm[glob_arr, it_arr]
        sampled_extra = tr_extra5_mm[glob_arr, it_arr]
        cand_feat = _extract_cand_block(model_name, sampled_38, sampled_extra)
        cust_feat = cust_all[glob_arr]
        num_chunks.append(np.concatenate([cand_feat, cust_feat], axis=1).astype(np.float32, copy=False))

        c_cat = cust_cats_all[glob_arr]
        p_cat = prod_cats_tr[it_arr]
        cat_chunks.append(np.concatenate([c_cat, p_cat], axis=1).astype(np.int32, copy=False))
        y_chunks.append(np.asarray(sel_y_list, dtype=np.float32))
        w_chunks.append(np.asarray(sel_w_list, dtype=np.float32))

    if not num_chunks:
        raise RuntimeError(f"Zero training pairs sampled for {model_name}.")

    num_mat = np.concatenate(num_chunks, axis=0)
    cat_mat = np.concatenate(cat_chunks, axis=0)
    y_vec = np.concatenate(y_chunks, axis=0)
    w_vec = np.concatenate(w_chunks, axis=0)

    df = _build_dataframe_from_arrays(num_mat, cat_mat, num_names, ALL_CAT_COLS)
    cat_indices = list(range(len(num_names), len(num_names) + len(ALL_CAT_COLS)))
    pool = Pool(data=df, label=y_vec, weight=w_vec, cat_features=cat_indices)

    stats = {
        "baskets_used": int(total_baskets_used),
        "total_rows": int(len(y_vec)),
        "positives": int(total_positives),
        "negatives_sampled": int(total_negatives_sampled),
        "effective_negative_weight": float(total_negative_weight),
        "num_features": int(len(num_names) + len(ALL_CAT_COLS)),
        "num_numeric_features": int(len(num_names)),
        "num_categorical_features": int(len(ALL_CAT_COLS)),
    }
    return pool, stats


def score_full_pool_catboost(
    model: CatBoostClassifier,
    *,
    model_name: str,
    shared_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    va_cand38_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
    chunk_baskets: int = 256,
) -> torch.Tensor:
    """Score ALL formulations (zero truncation!) on validation using CatBoost raw log-odds."""
    cand_names, cust_names, _ = _schema_info(model_name)
    num_names = tuple(cand_names) + tuple(cust_names)
    cat_indices = list(range(len(num_names), len(num_names) + len(ALL_CAT_COLS)))

    n_rows = len(va_meta["customer_ids"])
    n_bn = int(shared_meta["universe"]["num_base_names"])
    allowed_np = va_meta["allowed_bn"].cpu().numpy().astype(bool)
    cust_all = _extract_cust_block(va_meta["tabular_features"].cpu().numpy().astype(np.float32))
    cust_cats_all, prod_cats_all = _extract_static_arrays(shared_meta, va_meta)

    scores = np.full((n_rows, n_bn), -np.inf, dtype=np.float32)

    for s0 in range(0, n_rows, int(chunk_baskets)):
        s1 = min(n_rows, s0 + int(chunk_baskets))
        c38_sl = va_cand38_mm[s0:s1]
        e5_sl = va_extra5_mm[s0:s1]
        cand_block = _extract_cand_block(model_name, c38_sl, e5_sl)  # (m, n_bn, F_c)

        # Flatten only eligible (basket, formulation) pairs to skip ineligible formulations cleanly
        elig_sl = allowed_np[s0:s1]
        b_idx, p_idx = np.nonzero(elig_sl)
        if len(b_idx) == 0:
            continue
        glob_b = b_idx + s0
        num_mat = np.concatenate([cand_block[b_idx, p_idx], cust_all[glob_b]], axis=1).astype(np.float32, copy=False)
        cat_mat = np.concatenate([cust_cats_all[glob_b], prod_cats_all[p_idx]], axis=1).astype(np.int32, copy=False)

        df = _build_dataframe_from_arrays(num_mat, cat_mat, num_names, ALL_CAT_COLS)
        pool = Pool(data=df, cat_features=cat_indices)
        raw_preds = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float32)
        scores[glob_b, p_idx] = raw_preds

    return torch.from_numpy(scores)


def benchmark_catboost_single_request_ms(
    model: CatBoostClassifier,
    *,
    model_name: str,
    shared_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    va_cand38_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
) -> Dict[str, float]:
    """Benchmark 1-request full-pool CPU inference latency for CatBoost."""
    cand_names, cust_names, _ = _schema_info(model_name)
    num_names = tuple(cand_names) + tuple(cust_names)
    cat_indices = list(range(len(num_names), len(num_names) + len(ALL_CAT_COLS)))

    cust_all = _extract_cust_block(va_meta["tabular_features"][:1].cpu().numpy().astype(np.float32))
    cust_cats, prod_cats = _extract_static_arrays(shared_meta, va_meta)
    c38_0 = va_cand38_mm[0:1]
    e5_0 = va_extra5_mm[0:1]
    cand_0 = _extract_cand_block(model_name, c38_0, e5_0)[0]  # (n_bn, F_c)
    n_bn = cand_0.shape[0]
    num_mat = np.concatenate([cand_0, np.repeat(cust_all, n_bn, axis=0)], axis=1).astype(np.float32)
    cat_mat = np.concatenate([np.repeat(cust_cats[:1], n_bn, axis=0), prod_cats], axis=1).astype(np.int32)
    df = _build_dataframe_from_arrays(num_mat, cat_mat, num_names, ALL_CAT_COLS)
    pool = Pool(data=df, cat_features=cat_indices)

    for _ in range(5):
        model.predict(pool, prediction_type="RawFormulaVal")
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        model.predict(pool, prediction_type="RawFormulaVal")
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": float(np.median(ts)),
        "p95_ms": float(np.percentile(ts, 95)),
        "mean_ms": float(np.mean(ts)),
    }


def fit_oof_linear_ensemble(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    *,
    allowed_bn: torch.Tensor,
    target_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    customer_ids: Sequence[str],
    n_folds: int = 2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """2-fold customer-grouped OOF linear ensemble `scores_a + alpha * scores_b` (alpha >= 0).

    Fits single scalar `alpha` on the complementary customer fold maximizing `(ndcg@10, new@10, rec@10)`.
    """
    a_np = scores_a.detach().cpu().numpy().astype(np.float64)
    b_np = scores_b.detach().cpu().numpy().astype(np.float64)
    allowed_np = allowed_bn.detach().cpu().numpy().astype(bool)
    target_np = target_bn.detach().cpu().numpy().astype(bool)
    new_np = new_target_bn.detach().cpu().numpy().astype(bool)

    folds = _customer_fold_assignment(customer_ids, n_folds=n_folds)
    tot_pos = target_np.sum(axis=1)

    a_safe = np.where(allowed_np, a_np, 0.0)
    b_safe = np.where(allowed_np, b_np, 0.0)

    def _score_alpha(mask: np.ndarray, alpha: float) -> Tuple[float, int, float]:
        comb = np.where(allowed_np[mask], a_safe[mask] + alpha * b_safe[mask], -np.inf)
        order = np.argsort(-comb, axis=1, kind="mergesort")[:, :10]
        r = np.arange(len(comb))[:, None]
        ty = (target_np[mask][r, order] & allowed_np[mask][r, order]).astype(np.float64)
        tn = (new_np[mask][r, order] & allowed_np[mask][r, order]).astype(np.float64)
        return _mixed_top10_metrics(ty, tn, tot_pos[mask])

    alpha_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    oof = np.empty_like(a_np)
    fold_records = []
    for f in range(int(n_folds)):
        val_m = folds == f
        fit_m = ~val_m
        best_alpha = 0.5
        best_key = (-1.0, -1, -1.0)
        for alpha in alpha_grid:
            key = _score_alpha(fit_m, alpha)
            if key > best_key:
                best_key = key
                best_alpha = alpha
        for da in (-0.06, -0.03, 0.03, 0.06):
            cand = max(0.0, best_alpha + da)
            key = _score_alpha(fit_m, cand)
            if key > best_key:
                best_key = key
                best_alpha = cand
        oof[val_m] = np.where(allowed_np[val_m], a_safe[val_m] + best_alpha * b_safe[val_m], -np.inf)
        fold_records.append(
            {
                "fold": f,
                "oof_baskets": int(val_m.sum()),
                "fit_baskets": int(fit_m.sum()),
                "alpha": float(best_alpha),
                "fit_ndcg@10": float(best_key[0]),
                "fit_new_hits@10": int(best_key[1]),
                "fit_recall@10": float(best_key[2]),
            }
        )

    return torch.from_numpy(oof), {"n_folds": int(n_folds), "customer_grouped_oof": True, "folds": fold_records}


def train_and_eval_catboost_model(
    model_name: str,
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    tr_cand38_mm: np.ndarray,
    va_cand38_mm: np.ndarray,
    tr_extra5_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
    extra5_identity: str,
    dataset_identity: Optional[str] = None,
    dev_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    out_dir: Path,
    iterations: int = 1500,
    depth: int = 6,
    learning_rate: float = 0.06,
    l2_leaf_reg: float = 5.0,
    negatives_per_basket: Optional[int] = None,
    task_type: str = "CPU",
    seed: int = SEED,
    sync_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Train, full-pool score, evaluate, and save one CatBoost `v2_full` model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, unseen_only = _schema_info(model_name)
    neg_k = int(negatives_per_basket) if negatives_per_basket is not None else (40 if unseen_only else 25)

    if dataset_identity is None:
        dataset_identity = prepared_dataset_digest(
            shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
        )
        extra5_identity = input_digest((extra5_identity, tr_extra5_mm, va_extra5_mm))
    run_settings = {
        "version": 3,
        "dataset_identity": dataset_identity,
        "model_name": model_name,
        "extra5_identity": str(extra5_identity),
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2_leaf_reg),
        "negatives_per_basket": neg_k,
        "task_type": str(task_type),
        "seed": int(seed),
    }
    expected_files = ("model.cbm", "val_logits.npy", "predictions.npz")
    summary_p = out_dir / "model_result.json"
    candidates = [out_dir]
    if sync_dir is not None and not out_dir.is_absolute():
        candidates.append(sync_dir / out_dir)
    for directory in candidates:
        summary = directory / summary_p.name
        if not summary.is_file():
            continue
        try:
            prev = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if prev.get("run_settings") == run_settings and all(
            (directory / f).is_file() and file_digest(directory / f) == prev.get("output_sha256", {}).get(f)
            for f in expected_files
        ):
            if directory != out_dir:
                shutil.copytree(directory, out_dir, dirs_exist_ok=True)
            logits = torch.from_numpy(np.load(out_dir / "val_logits.npy"))
            sync_outputs_to_dir(sync_dir, (out_dir,))
            return prev, logits

    t0 = time.perf_counter()
    print(
        f"\n[v2-cb] Sampling weighted training Pool for {model_name} "
        f"(neg/basket={neg_k}, iters={iterations}, depth={depth}, task_type={task_type})...",
        flush=True,
    )
    train_pool, pool_stats = sample_weighted_catboost_training_pool(
        model_name=model_name,
        shared_meta=shared_meta,
        tr_meta=tr_meta,
        tr_cand38_mm=tr_cand38_mm,
        tr_extra5_mm=tr_extra5_mm,
        negatives_per_basket=neg_k,
        seed=seed,
    )
    print(
        f"[v2-cb] Sampled {pool_stats['total_rows']:,} rows ({pool_stats['positives']:,} pos, "
        f"{pool_stats['negatives_sampled']:,} neg) in {time.perf_counter() - t0:.1f}s. Fitting CatBoost...",
        flush=True,
    )

    t_fit = time.perf_counter()
    cb_kwargs: Dict[str, Any] = dict(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function="Logloss",
        random_seed=int(seed),
        task_type=str(task_type),
        allow_writing_files=False,
        verbose=250,
    )
    if str(task_type).upper() == "CPU":
        cb_kwargs["thread_count"] = -1
    model = CatBoostClassifier(**cb_kwargs)
    model.fit(train_pool)
    fit_sec = float(time.perf_counter() - t_fit)
    model.save_model(str(out_dir / "model.cbm"))

    del train_pool
    gc.collect()

    t_score = time.perf_counter()
    val_logits = score_full_pool_catboost(
        model,
        model_name=model_name,
        shared_meta=shared_meta,
        va_meta=va_meta,
        va_cand38_mm=va_cand38_mm,
        va_extra5_mm=va_extra5_mm,
    )
    score_sec = float(time.perf_counter() - t_score)
    np.save(out_dir / "val_logits.npy", val_logits.numpy())

    ranked = rank_eligible_formulations(val_logits, va_meta["allowed_bn"])
    ev = _evaluate_with_cohorts_and_sku(
        ranked,
        va_meta,
        va_meta["sku_val"],
        dev_purchases,
        catalogue,
        shared_meta["product_ids"],
        shared_meta["universe"],
        out_dir / "predictions.npz",
    )
    pure_unseen = evaluate_pure_unseen_predictions(
        val_logits,
        allowed_bn=va_meta["allowed_bn"],
        unseen_bn=va_meta["unseen_bn"],
        new_target_bn=va_meta["new_target_bn"],
        first_observed=va_meta["first_observed"],
    )
    cpu_ms = benchmark_catboost_single_request_ms(
        model,
        model_name=model_name,
        shared_meta=shared_meta,
        va_meta=va_meta,
        va_cand38_mm=va_cand38_mm,
        va_extra5_mm=va_extra5_mm,
    )

    cand_names, cust_names, _ = _schema_info(model_name)
    feat_names = list(cand_names) + list(cust_names) + list(ALL_CAT_COLS)
    importances = [float(x) for x in model.get_feature_importance()]
    top_features = sorted(zip(feat_names, importances, strict=True), key=lambda kv: kv[1], reverse=True)[:25]

    res = {
        "model": model_name,
        "pool_stats": pool_stats,
        "fit_seconds": fit_sec,
        "val_score_seconds": score_sec,
        "mixed_eval": ev,
        "pure_unseen": pure_unseen,
        "cpu_single_request_ms": cpu_ms,
        "top_25_feature_importance": [{"feature": k, "importance": v} for k, v in top_features],
        "run_settings": run_settings,
        "output_sha256": {name: file_digest(out_dir / name) for name in expected_files},
    }
    atomic_json(res, summary_p, overwrite=True)
    sync_outputs_to_dir(sync_dir, (out_dir,))
    print(
        f"[v2-cb:{model_name}] R@10={ev['bn_metrics']['bn_recall@10']:.5f} "
        f"NDCG@10={ev['bn_metrics']['bn_ndcg@10']:.5f} New@10={ev['bn_metrics']['true_new_bn_hits@10']} "
        f"PureU@10={pure_unseen['overall']['pure_unseen_hits@10']} "
        f"PureU@25={pure_unseen['overall']['pure_unseen_hits@25']} "
        f"(fit={fit_sec:.1f}s, cpu_1req={cpu_ms['median_ms']:.2f}ms)",
        flush=True,
    )
    return res, val_logits


def _neural_ensemble_inputs(paths, dataset_identity):
    """Require the producing run's ordered inputs and exact score artifact."""
    inputs = {}
    for path in map(Path, paths):
        result = json.loads((path.parent / "arm_result.json").read_text(encoding="utf-8"))
        if result["run_settings"].get("dataset_identity") != dataset_identity:
            raise ValueError(f"Neural logits belong to different prepared data: {path}")
        if result["output_sha256"].get(path.name) != file_digest(path):
            raise ValueError(f"Neural logit checksum mismatch: {path}")
        tag = f"{path.parent.parent.name}_{path.parent.name}_{path.stem}"
        if tag in inputs:
            raise ValueError(f"Duplicate ensemble output name: {tag}")
        inputs[tag] = path
    return inputs


def run_v2_full_catboost(
    *,
    models: Optional[Sequence[str]] = None,
    iterations: int = 1500,
    depth: int = 6,
    learning_rate: float = 0.06,
    l2_leaf_reg: float = 5.0,
    negatives_per_basket: Optional[int] = None,
    task_type: str = "auto",
    neural_logits_paths: Optional[Sequence[Path]] = None,
    seed: int = SEED,
    in_memory: bool = False,
    skip_v1_verify: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    resolved_task = "GPU" if (task_type == "auto" and torch.cuda.is_available()) else ("CPU" if task_type == "auto" else task_type.upper())

    before_hashes = None
    ref_v1 = ref_v1_3s = None
    if not skip_v1_verify:
        before_hashes = snapshot_preserved_hashes()
        ref_v1 = json.loads((V1_EVIDENCE_ROOT / "stage5_reference.json").read_text(encoding="utf-8"))
        ref_v1_3s = json.loads((V1_EVIDENCE_ROOT / "stage5_3stage_reference.json").read_text(encoding="utf-8"))
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    selected_models = list(DEFAULT_CB_MODELS if models is None else models)
    if not selected_models or set(selected_models) - set(ALL_CB_MODELS):
        raise ValueError(f"Unsupported CatBoost models={models!r}. Available: {ALL_CB_MODELS}")

    tr_e5_p, va_e5_p, e5_manifest = build_or_load_extra_cand_5()
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)

    mmap_mode = None if in_memory else "r"
    tr_cand38_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand38_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)
    tr_extra5_mm = np.load(tr_e5_p, mmap_mode=mmap_mode)
    va_extra5_mm = np.load(va_e5_p, mmap_mode=mmap_mode)

    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)

    dataset_identity = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
    )
    neural_inputs = _neural_ensemble_inputs(neural_logits_paths or (), dataset_identity)
    if neural_inputs and "cb_mixed_cand42" not in selected_models:
        raise ValueError("Neural ensembles require cb_mixed_cand42.")
    cb_root = ARTIFACT_ROOT / f"catboost_wave2b-seed{seed}"
    model_results: Dict[str, Any] = {}
    val_logits_map: Dict[str, torch.Tensor] = {}

    for mname in selected_models:
        res, lg = train_and_eval_catboost_model(
            mname,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand38_mm=tr_cand38_mm,
            va_cand38_mm=va_cand38_mm,
            tr_extra5_mm=tr_extra5_mm,
            va_extra5_mm=va_extra5_mm,
            extra5_identity=input_digest((e5_manifest["identity"], e5_manifest["output_sha256"])),
            dataset_identity=dataset_identity,
            dev_purchases=dev_purchases,
            catalogue=catalogue,
            out_dir=cb_root / mname,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            negatives_per_basket=negatives_per_basket,
            task_type=resolved_task,
            seed=seed,
            sync_dir=sync_dir,
        )
        model_results[mname] = res
        val_logits_map[mname] = lg

    # Option A CatBoost 2-param OOF affine blend (cb_mixed_cand42 on repeats + cb_optionA_unseen_cand42 on unseen)
    blends: Dict[str, Any] = {}
    if "cb_mixed_cand42" in val_logits_map and "cb_optionA_unseen_cand42" in val_logits_map:
        oof_cb, oof_meta = fit_oof_affine_unseen_blend(
            val_logits_map["cb_mixed_cand42"],
            val_logits_map["cb_optionA_unseen_cand42"],
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            target_bn=va_meta["target_bn"],
            new_target_bn=va_meta["new_target_bn"],
            customer_ids=va_meta["customer_ids"],
            n_folds=2,
        )
        oof_npz = cb_root / "predictions_cb_optionA_oof_blend.npz"
        ev_oof = _evaluate_with_cohorts_and_sku(
            rank_eligible_formulations(oof_cb, va_meta["allowed_bn"]),
            va_meta,
            va_meta["sku_val"],
            dev_purchases,
            catalogue,
            shared_meta["product_ids"],
            shared_meta["universe"],
            oof_npz,
        )
        blends["cb_optionA_oof_blend"] = {
            "mixed_eval": ev_oof,
            "pure_unseen": evaluate_pure_unseen_predictions(
                oof_cb, allowed_bn=va_meta["allowed_bn"], unseen_bn=va_meta["unseen_bn"],
                new_target_bn=va_meta["new_target_bn"], first_observed=va_meta["first_observed"],
            ),
            "affine_oof_calibration": oof_meta,
            "vs_cb_mixed_cand42_95ci": paired_base_name_bootstrap_ci(
                Path(model_results["cb_mixed_cand42"]["mixed_eval"]["predictions_npz"]),
                oof_npz,
                seed=seed,
            ),
        }

    # Optional CatBoost + Neural 2-fold customer-grouped OOF linear ensembles
    if neural_inputs:
        cb_lg = val_logits_map["cb_mixed_cand42"]
        for tag, n_path in neural_inputs.items():
            n_lg = torch.from_numpy(np.load(n_path))
            if n_lg.shape != cb_lg.shape or not torch.isfinite(n_lg[va_meta["allowed_bn"]]).all():
                raise ValueError(f"Invalid neural scores or eligibility alignment: {n_path}")
            ens_lg, ens_meta = fit_oof_linear_ensemble(
                n_lg,
                cb_lg,
                allowed_bn=va_meta["allowed_bn"],
                target_bn=va_meta["target_bn"],
                new_target_bn=va_meta["new_target_bn"],
                customer_ids=va_meta["customer_ids"],
                n_folds=2,
            )
            ens_npz = cb_root / f"predictions_ens_{tag}.npz"
            ev_ens = _evaluate_with_cohorts_and_sku(
                rank_eligible_formulations(ens_lg, va_meta["allowed_bn"]),
                va_meta,
                va_meta["sku_val"],
                dev_purchases,
                catalogue,
                shared_meta["product_ids"],
                shared_meta["universe"],
                ens_npz,
            )
            pure_ens = evaluate_pure_unseen_predictions(
                ens_lg,
                allowed_bn=va_meta["allowed_bn"],
                unseen_bn=va_meta["unseen_bn"],
                new_target_bn=va_meta["new_target_bn"],
                first_observed=va_meta["first_observed"],
            )
            n_ref_npz = n_path.parent / "predictions_ndcg.npz"
            ci_vs_neural = (
                paired_base_name_bootstrap_ci(n_ref_npz, ens_npz, seed=seed) if n_ref_npz.is_file() else None
            )
            blends[f"ensemble_{tag}_plus_cb42"] = {
                "neural_logits_path": str(n_path),
                "mixed_eval": ev_ens,
                "pure_unseen": pure_ens,
                "linear_oof_calibration": ens_meta,
                "vs_neural_95ci": ci_vs_neural,
                "vs_cb_mixed_cand42_95ci": paired_base_name_bootstrap_ci(
                    Path(model_results["cb_mixed_cand42"]["mixed_eval"]["predictions_npz"]),
                    ens_npz,
                    seed=seed,
                ),
            }

    comparisons: Dict[str, Any] = {}
    ref_preds = resolve_v2_full_reference_predictions(sync_dir)
    primary_ref = ref_preds.get("v2_full_ref_primary")
    if primary_ref is not None:
        if "cb_mixed_cand42" in model_results:
            comparisons["cb_mixed_cand42_vs_v2_full_ref_primary"] = paired_base_name_bootstrap_ci(
                primary_ref,
                Path(model_results["cb_mixed_cand42"]["mixed_eval"]["predictions_npz"]),
                seed=seed,
            )
        for bres in blends.values():
            b_npz = Path(bres["mixed_eval"]["predictions_npz"])
            bres["vs_v2_full_ref_primary_95ci"] = paired_base_name_bootstrap_ci(primary_ref, b_npz, seed=seed)

    if not skip_v1_verify:
        if snapshot_preserved_hashes() != before_hashes:
            raise AssertionError("v1 preserved reference hashes changed!")
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_ROOT / f"v2_full_catboost_wave2b_seed{seed}.json"
    if set(selected_models) != set(DEFAULT_CB_MODELS):
        summary_path = summary_path.with_stem(summary_path.stem + "_" + "_".join(selected_models))
    summary = {
        "experiment": "v2_full_catboost_wave2b_benchmark",
        "dataset": "v2_full",
        "reference_id": ACTIVE_V2_FULL_REFERENCE_ID,
        "feature_schema": "cand_42+CUST_7+9_static_cats",
        "extra_cand_5_identity": str(e5_manifest["identity"]),
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "task_type": resolved_task,
        "models": model_results,
        "blends": blends,
        "paired_bootstrap_95ci": comparisons,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(summary, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (cb_root, summary_path))
    print(f"\nSaved CatBoost Wave 2B summary -> {summary_path} ({summary['elapsed_seconds']:.1f}s)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, choices=ALL_CB_MODELS)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--negatives-per-basket", type=int, default=None)
    parser.add_argument("--task-type", type=str, default="auto", choices=("auto", "CPU", "GPU"))
    parser.add_argument("--neural-logits", nargs="+", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--skip-v1-verify", action="store_true")
    parser.add_argument("--sync-dir", type=Path, default=None)
    args = parser.parse_args()
    run_v2_full_catboost(
        models=args.models,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        negatives_per_basket=args.negatives_per_basket,
        task_type=args.task_type,
        neural_logits_paths=args.neural_logits,
        seed=args.seed,
        in_memory=args.in_memory,
        skip_v1_verify=args.skip_v1_verify,
        sync_dir=args.sync_dir,
    )


if __name__ == "__main__":
    main()
