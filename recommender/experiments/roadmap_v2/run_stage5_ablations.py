"""Validation-only Stage 5 ablations on the canonical dated first/returning basket split:
1. Excluding Stage 1 MLP: running Stage 2 UPS-CANR directly over ALL 585 eligible formulations
   (both zero-shot with pinned weights and trained from scratch full-pool).
2. CatBoost with canonical base_name + Stage 3 (3a_financial, 3c_manager, 3e_parsed_specs) features:
   - Standalone full-pool (zero neural inputs)
   - Full-pool + Two-Tower similarity & familiarity (no Stage 1 MLP, no Stage 2 Transformer)
"""

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.ranker import RankerLoss
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TRAIN_CUTOFF
from recommender.train_retriever import get_device

from .augmentation import build_strict_60d_aux_labels
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_LR,
    STAGE2_WEIGHT_DECAY,
)
from .dated_baskets import FEATURE_NAMES, prepare_dated_baskets
from .run_stage5 import (
    Stage5BaseNameRanker,
    _collapse_bool_matrix_to_bn,
    _stage5_tower,
    _verify_reference_files,
)
from .stage5_base_entity import (
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
    resolve_base_names_to_skus,
)


OUT_ROOT = ARTIFACT_ROOT / "stage5_ablations_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "stage5_nomlp_and_catboost_seed42.json"
REFERENCE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"

PRODUCT_CAT_COLS = ("brand", "product_category", "product_subcategory", "business_line", "product_family")
CUSTOMER_CAT_COLS = ("city_id", "region_id", "industry_id", "sales_branch")
ALL_CAT_COLS = PRODUCT_CAT_COLS + CUSTOMER_CAT_COLS


def sample_training_pairs(
    allowed: np.ndarray, target: np.ndarray, *, negatives: int = 100, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unbiased sampled estimate of full eligible-pool binary Logloss with inverse-probability weights."""
    allowed_b, target_b = np.asarray(allowed, dtype=bool), np.asarray(target, dtype=bool)
    rng = np.random.default_rng(int(seed))
    rows, cols, weights = [], [], []
    for row in range(len(allowed_b)):
        pos = np.flatnonzero(allowed_b[row] & target_b[row])
        neg = np.flatnonzero(allowed_b[row] & ~target_b[row])
        k = min(int(negatives), len(neg))
        chosen = rng.choice(neg, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
        sel = np.concatenate([pos, chosen])
        rows.append(np.full(len(sel), row, dtype=np.int32))
        cols.append(sel.astype(np.int32))
        w_neg = float(len(neg)) / max(k, 1)
        weights.append(np.concatenate([np.ones(len(pos)), np.full(k, w_neg)]).astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)


def _evaluate_with_cohorts_and_sku(
    ranked_bn: np.ndarray,
    val_bn: Mapping[str, Any],
    sku_val: Mapping[str, Any],
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    save_path: Path,
) -> Dict[str, Any]:
    all_m = evaluate_base_name_predictions(
        ranked_bn, val_bn["target_bn"], val_bn["new_target_bn"], val_bn["customer_ids"], save_path=save_path
    )
    cohorts = {}
    for cname, mask in (("first_observed", val_bn["first_observed"]), ("returning", ~val_bn["first_observed"])):
        ids = [cid for cid, keep in zip(val_bn["customer_ids"], mask.tolist()) if keep]
        cohorts[cname] = evaluate_base_name_predictions(
            ranked_bn[mask.numpy()], val_bn["target_bn"][mask], val_bn["new_target_bn"][mask], ids
        )

    resolved_skus = resolve_base_names_to_skus(
        ranked_bn,
        val_bn["customer_ids"],
        val_bn["prediction_days"].tolist(),
        sku_val["allowed"],
        purchases,
        catalogue,
        product_ids,
        universe,
        top_k=10,
    )
    t_sku = sku_val["target"].numpy().astype(bool)
    nt_sku = sku_val["new_target"].numpy().astype(bool)
    disc_arr = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    sku_r10, sku_ndcg10 = [], []
    sku_new10 = 0
    for r_i in range(len(resolved_skus)):
        valid_p = resolved_skus[r_i][resolved_skus[r_i] >= 0]
        rel = t_sku[r_i, valid_p].astype(np.float64)
        n_pos = int(t_sku[r_i].sum())
        if n_pos > 0:
            sku_r10.append(float(rel.sum()) / n_pos)
            dcg = float((rel * disc_arr[: len(rel)]).sum())
            idcg = float(disc_arr[: min(n_pos, 10)].sum())
            sku_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        sku_new10 += int(nt_sku[r_i, valid_p].sum())

    return {
        "bn_metrics": all_m,
        "cohorts": cohorts,
        "resolved_sku_metrics": {
            "sku_recall@10": float(np.mean(sku_r10)),
            "sku_ndcg@10": float(np.mean(sku_ndcg10)),
            "sku_new_hits@10": int(sku_new10),
        },
        "predictions_npz": str(save_path),
    }


def _build_fullpool_batch(
    sp_d: Mapping[str, Any],
    v_dev: torch.Tensor,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    n_bn = v_dev.shape[0]
    bsz = len(rows)
    c_dev = torch.arange(n_bn, device=dev).unsqueeze(0).expand(bsz, n_bn)
    return {
        "text_embedding": sp_d["text_embedding"][rows].to(dev),
        "city_id": sp_d["city_id"][rows].to(dev),
        "region_id": sp_d["region_id"][rows].to(dev),
        "industry_id": sp_d["industry_id"][rows].to(dev),
        "tabular_features": sp_d["tabular_features"][rows].to(dev),
        "sequence": sp_d["sequence"][rows].to(dev),
        "attention_mask": sp_d["attention_mask"][rows].to(dev),
        "seq_product_ids": sp_d["seq_product_ids"][rows].to(dev),
        "seq_category_ids": sp_d["seq_category_ids"][rows].to(dev),
        "seq_subcategory_ids": sp_d["seq_subcategory_ids"][rows].to(dev),
        "seq_business_line_ids": sp_d["seq_business_line_ids"][rows].to(dev),
        "seq_purchase_dates": sp_d["seq_purchase_dates"][rows].to(dev),
        "prediction_day": sp_d["prediction_days"][rows].to(dev),
        "pairwise_features": sp_d["features_bn"][rows].to(dev, dtype=torch.float32),
        "candidate_vectors": v_dev.unsqueeze(0).expand(bsz, n_bn, -1),
        "candidate_mask": sp_d["allowed_bn"][rows].to(dev),
        "candidate_indices": c_dev,
    }


@torch.no_grad()
def _score_fullpool_ranker(
    model: Stage5BaseNameRanker,
    sp_d: Mapping[str, Any],
    v_dev: torch.Tensor,
    dev: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for rows in torch.arange(len(sp_d["customer_ids"])).split(batch_size):
        inp = _build_fullpool_batch(sp_d, v_dev, rows, dev)
        chunks.append(model(**inp).logits.cpu())
    return torch.cat(chunks, dim=0)


def _make_ranker(n_bn: int, vocabs: Mapping[str, Any], orig_s: Mapping[str, Any], *, aux: bool, seed: int, dev: torch.device) -> Stage5BaseNameRanker:
    return Stage5BaseNameRanker(
        num_base_names=n_bn,
        num_features=len(FEATURE_NAMES),
        num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
        num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
        num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
        num_categories=len(vocabs["category_vocab"]) + 1,
        num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
        num_business_lines=len(vocabs["business_line_vocab"]) + 1,
        aux_60d_enabled=aux,
        seed=seed,
    ).to(dev)


def _build_catboost_frame(
    sp_d: Mapping[str, Any],
    rows: np.ndarray,
    cols: np.ndarray,
    bn_cat_meta: Mapping[str, np.ndarray],
    *,
    include_neural: bool,
    dot_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for col_name in PRODUCT_CAT_COLS:
        data[col_name] = bn_cat_meta[col_name][cols]
    for col_name in CUSTOMER_CAT_COLS:
        data[col_name] = sp_d[col_name].numpy()[rows].astype(str)

    feats = sp_d["features_bn"].numpy()[rows, cols].astype(np.float32)
    for i, fname in enumerate(FEATURE_NAMES):
        data[fname] = feats[:, i]

    prof = sp_d["tabular_features"].numpy()[rows].astype(np.float32)
    for i, pname in enumerate(LEGACY_PROFILE_FEATURES):
        data[f"profile_{pname}"] = prof[:, i]

    if include_neural:
        if dot_scores is None:
            raise ValueError("dot_scores required when include_neural=True.")
        data["two_tower_score"] = dot_scores[rows, cols].astype(np.float32)
        u = sp_d["u_bn"].numpy()
        v = sp_d["v_bn"].numpy()
        data["two_tower_cosine"] = np.einsum("nd,nd->n", u[rows], v[cols]).astype(np.float32)
        fam = sp_d["familiarity_bn"].numpy()[rows, cols].astype(np.float32)
        for j, fname in enumerate(("fam_brand", "fam_category", "fam_business_line", "fam_family")):
            data[fname] = fam[:, j]

    return pd.DataFrame(data)


def run_ablations(*, seed: int = SEED, device: Optional[str] = None, skip_train_fullpool: bool = False) -> Dict[str, Any]:
    from catboost import CatBoostClassifier, Pool
    import catboost

    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(reference)
    saved_val = json.loads(Path(reference["checkpoints"]["summary"]["path"]).read_text(encoding="utf-8"))

    print("[ablations] Preparing canonical dated first/returning baskets (strictly < TEST_START)...", flush=True)
    prepared = prepare_dated_baskets()
    (
        bn_data,
        prod_inputs,
        universe,
        enrichment,
        vocabs,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        manifest,
    ) = prepared
    if manifest != saved_val["data_manifest"]:
        raise ValueError("Dated data manifest does not match pinned stage5_reference!")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_bn = universe["num_base_names"]
    bn_to_members = universe["bn_to_member_pid_indices"]
    tr_bn = bn_data["train"]
    va_bn = bn_data["val"]
    sku_val = sku_data["val"]
    orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]

    # Encode frozen Two-Tower vectors & scores for both train and val
    print("[ablations] Loading pinned Two-Tower vectors...", flush=True)
    tt_dir = Path(reference["checkpoints"]["two_tower"]["path"]).parent
    tr_vec_d = torch.load(tt_dir / "train_vectors.pt", map_location="cpu", weights_only=False)
    va_vec_d = torch.load(tt_dir / "val_vectors.pt", map_location="cpu", weights_only=False)
    v_bn = tr_vec_d["v_bn"]
    v_dev = v_bn.to(dev)
    tr_bn["u_bn"] = tr_vec_d["u_bn"]
    tr_bn["v_bn"] = v_bn
    va_bn["u_bn"] = va_vec_d["u_bn"]
    va_bn["v_bn"] = v_bn
    tr_dot = tr_vec_d["dot_scores"].numpy()
    va_dot = va_vec_d["dot_scores"].numpy()

    ref_ndcg_npz = Path(saved_val["stage2_ranker"]["ndcg"]["bn_predictions_npz"])
    results: Dict[str, Any] = {
        "experiment": "stage5_nomlp_and_catboost_ablations",
        "reference_id": reference["reference_id"],
        "seed": int(seed),
        "heldout_access": False,
        "reference_baseline": {
            "ndcg": saved_val["stage2_ranker"]["ndcg"]["bn_metrics"],
            "ndcg_cohorts": saved_val["stage2_ranker"]["ndcg"]["cohorts"],
            "ndcg_resolved_sku": saved_val["stage2_ranker"]["ndcg"]["resolved_sku_metrics"],
            "discovery": saved_val["stage2_ranker"]["discovery"]["bn_metrics"],
            "discovery_cohorts": saved_val["stage2_ranker"]["discovery"]["cohorts"],
            "stage1_mlp": saved_val["stage1_mlp"]["bn_metrics"],
        },
        "arms": {},
    }

    # ------------------------------------------------------------------
    # ARM 1A: Zero-shot full-pool inference of PINNED Stage 2 Rankers (NO Stage 1 MLP!)
    # ------------------------------------------------------------------
    print("[ablations/1A] Zero-shot full-pool (all 585 eligible BNs) inference with pinned Stage 2 Rankers...", flush=True)
    zs_res = {}
    for sel_name, ref_key in (("ndcg", "ranker"), ("discovery", "discovery_ranker")):
        ck = torch.load(reference["checkpoints"][ref_key]["path"], map_location="cpu", weights_only=False)
        m = _make_ranker(n_bn, vocabs, orig_s, aux=False, seed=seed, dev=dev)
        m.load_state_dict(ck["model_state_dict"], strict=True)
        t0 = time.perf_counter()
        logits = _score_fullpool_ranker(m, va_bn, v_dev, dev, batch_size=64)
        inf_sec = time.perf_counter() - t0
        ranked = logits.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        npz_p = OUT_ROOT / f"pinned_fullpool_{sel_name}.npz"
        ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
        ev["inference_seconds_6480"] = float(inf_sec)
        ev["paired_vs_reference_ndcg"] = paired_base_name_bootstrap_ci(ref_ndcg_npz, npz_p)
        zs_res[sel_name] = ev
        print(
            f"[ablations/1A/{sel_name}] R@10={ev['bn_metrics']['bn_recall@10']:.5f} "
            f"NDCG@10={ev['bn_metrics']['bn_ndcg@10']:.5f} "
            f"New@10={ev['bn_metrics']['true_new_bn_hits@10']}/1721 ({inf_sec:.1f}s)",
            flush=True,
        )
    results["arms"]["pinned_ranker_fullpool_no_mlp"] = zs_res

    # ------------------------------------------------------------------
    # ARM 1B: Train Stage 2 UPS-CANR from scratch over ALL 585 eligible BNs (NO Stage 1 MLP!)
    # ------------------------------------------------------------------
    if not skip_train_fullpool:
        print("[ablations/1B] Training Stage 2 UPS-CANR directly over ALL 585 eligible BNs (no Stage 1 MLP)...", flush=True)
        fp_dir = OUT_ROOT / f"direct_fullpool_ranker-seed{seed}"
        fp_dir.mkdir(parents=True, exist_ok=True)

        aux_60d_sku = build_strict_60d_aux_labels(
            purchases, tr_bn["customer_ids"], tr_bn["prediction_days"].tolist(), product_ids
        )
        aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members).float()
        full_window = tr_bn["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        fp_model = _make_ranker(n_bn, vocabs, orig_s, aux=True, seed=seed, dev=dev)
        trainable = [p for p in fp_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STAGE2_EPOCHS, eta_min=STAGE2_ETA_MIN)
        loss_fn = RankerLoss().to(dev)
        aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)

        gen = torch.Generator().manual_seed(int(seed))
        best_ndcg_key = (-1.0, -1, -1.0)
        best_disc_key = (-1, -1.0, -1.0)
        t_train = time.perf_counter()
        fp_bsz = 64  # Fits comfortably in MPS unified memory for 585 candidates

        for ep in range(1, STAGE2_EPOCHS + 1):
            ep_t0 = time.perf_counter()
            fp_model.train()
            losses = []
            perm = torch.randperm(len(tr_bn["customer_ids"]), generator=gen)
            for rows in perm.split(fp_bsz):
                opt.zero_grad(set_to_none=True)
                inp = _build_fullpool_batch(tr_bn, v_dev, rows, dev)
                cmask = inp["candidate_mask"]
                y = tr_bn["target_bn"][rows].to(dev, dtype=torch.float32) * cmask.float()
                out, a_lg = fp_model(**inp, return_aux=True)
                l_main = loss_fn(out.logits, y, candidate_mask=cmask)["loss"]
                aux_rows = full_window[rows].to(dev)
                loss = l_main
                if aux_rows.any():
                    y_aux = aux_60d_bn[rows].to(dev)[aux_rows] * cmask[aux_rows].float()
                    l_aux = aux_loss_fn(
                        a_lg[aux_rows],
                        y_aux,
                        candidate_mask=cmask[aux_rows],
                    )["loss"]
                    loss = loss + 0.3 * l_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, STAGE2_GRAD_CLIP_NORM)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sch.step()

            va_lg = _score_fullpool_ranker(fp_model, va_bn, v_dev, dev, batch_size=64)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            m_ep = evaluate_base_name_predictions(ranked, va_bn["target_bn"], va_bn["new_target_bn"], va_bn["customer_ids"])
            inf_state = {k: v for k, v in fp_model.state_dict().items() if not k.startswith("aux_60d_head.")}

            nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
            dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
            if nk > best_ndcg_key:
                best_ndcg_key = nk
                atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, fp_dir / "best_ndcg.pt")
            if dk > best_disc_key:
                best_disc_key = dk
                atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, fp_dir / "best_discovery.pt")
            print(
                f"[ablations/1B] ep={ep:02d}/{STAGE2_EPOCHS} loss={np.mean(losses):.4f} "
                f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
                f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
                f"({time.perf_counter() - ep_t0:.1f}s)",
                flush=True,
            )

        fp_train_sec = time.perf_counter() - t_train
        fp_sel_res: Dict[str, Any] = {"training_seconds": float(fp_train_sec)}
        for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
            ck = torch.load(fp_dir / fname, map_location="cpu", weights_only=False)
            eval_m = _make_ranker(n_bn, vocabs, orig_s, aux=False, seed=seed, dev=dev)
            eval_m.load_state_dict(ck["model_state_dict"], strict=True)
            va_lg = _score_fullpool_ranker(eval_m, va_bn, v_dev, dev, batch_size=64)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            npz_p = fp_dir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"Full-pool ranker {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            ev["paired_vs_reference_ndcg"] = paired_base_name_bootstrap_ci(ref_ndcg_npz, npz_p)
            fp_sel_res[sel_name] = ev
        results["arms"]["trained_fullpool_ranker_no_mlp"] = fp_sel_res
        del fp_model, aux_60d_sku, aux_60d_bn
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # ARM 2A & 2B: CatBoost with canonical base_name + Stage 3 (3a+3c+3e) features!
    # ------------------------------------------------------------------
    print("[ablations/2] Preparing CatBoost full-pool training & validation tables...", flush=True)
    # Representative static product categoricals for the 585 formulations
    from recommender.catalogue import RollingCandidatePool
    from .dated_baskets import first_observed_pack_indices

    pool = RollingCandidatePool(catalogue, purchases)
    reps = first_observed_pack_indices(product_ids, bn_to_members, pool.first_sales)
    rep_pids = np.asarray(product_ids)[reps]
    cat_idx = catalogue.drop_duplicates("product_id").set_index("product_id").reindex(rep_pids)
    bn_cat_meta = {
        c: cat_idx[c].fillna("MISSING").astype(str).str.strip().replace("", "MISSING").to_numpy()
        for c in PRODUCT_CAT_COLS
    }

    tr_rows, tr_cols, tr_weights = sample_training_pairs(
        tr_bn["allowed_bn"].numpy(), tr_bn["target_bn"].numpy(), negatives=100, seed=seed
    )
    tr_labels = tr_bn["target_bn"].numpy()[tr_rows, tr_cols].astype(np.float32)
    va_rows, va_cols = np.where(va_bn["allowed_bn"].numpy())
    va_labels = va_bn["target_bn"].numpy()[va_rows, va_cols].astype(np.float32)

    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1200,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": int(seed),
        "early_stopping_rounds": 100,
        "thread_count": 6,
        "verbose": 200,
    }

    for arm_name, include_neural in (
        ("catboost_base_name_standalone_fullpool", False),
        ("catboost_base_name_plus_twotower_fullpool", True),
    ):
        print(f"[ablations/2] Building {arm_name} ({len(tr_rows):,} train pairs, {len(va_rows):,} val pairs)...", flush=True)
        cb_dir = OUT_ROOT / f"{arm_name}-seed{seed}"
        cb_dir.mkdir(parents=True, exist_ok=True)

        tr_df = _build_catboost_frame(
            tr_bn, tr_rows, tr_cols, bn_cat_meta, include_neural=include_neural, dot_scores=tr_dot if include_neural else None
        )
        va_df = _build_catboost_frame(
            va_bn, va_rows, va_cols, bn_cat_meta, include_neural=include_neural, dot_scores=va_dot if include_neural else None
        )
        feat_names = list(tr_df.columns)
        tr_pool = Pool(tr_df, label=tr_labels, weight=tr_weights, cat_features=list(ALL_CAT_COLS))
        va_pool = Pool(va_df, label=va_labels, cat_features=list(ALL_CAT_COLS))
        del tr_df, va_df
        gc.collect()

        cb = CatBoostClassifier(**cb_params)
        t_cb = time.perf_counter()
        cb.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        cb_sec = time.perf_counter() - t_cb
        cb.save_model(str(cb_dir / "model.cbm"))

        raw = cb.predict(va_pool, prediction_type="RawFormulaVal")
        score_mat = np.full((len(va_bn["customer_ids"]), n_bn), -1e9, dtype=np.float32)
        score_mat[va_rows, va_cols] = raw.astype(np.float32)
        ranked = torch.from_numpy(score_mat).argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()

        npz_p = cb_dir / "predictions.npz"
        ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
        top_imp = sorted(
            zip(feat_names, cb.get_feature_importance().tolist()), key=lambda kv: kv[1], reverse=True
        )[:15]
        ev.update(
            catboost_version=catboost.__version__,
            include_neural=include_neural,
            num_features=len(feat_names),
            best_iteration=int(cb.get_best_iteration()),
            trees=int(cb.tree_count_),
            training_seconds=float(cb_sec),
            top_15_features=[{"feature": f, "importance": float(imp)} for f, imp in top_imp],
            paired_vs_reference_ndcg=paired_base_name_bootstrap_ci(ref_ndcg_npz, npz_p),
        )
        results["arms"][arm_name] = ev
        print(
            f"[ablations/2/{arm_name}] trees={cb.tree_count_} R@5={ev['bn_metrics']['bn_recall@5']:.5f} "
            f"R@10={ev['bn_metrics']['bn_recall@10']:.5f} NDCG@10={ev['bn_metrics']['bn_ndcg@10']:.5f} "
            f"New@10={ev['bn_metrics']['true_new_bn_hits@10']}/1721 ({cb_sec:.1f}s)",
            flush=True,
        )
        del tr_pool, va_pool, cb
        gc.collect()

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    _verify_reference_files(reference)

    results["preserved_hashes_verified"] = True
    results["elapsed_seconds"] = float(time.perf_counter() - t_all)
    atomic_json(results, SUMMARY_PATH, overwrite=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-train-fullpool", action="store_true")
    args = parser.parse_args()
    res = run_ablations(seed=args.seed, device=args.device, skip_train_fullpool=args.skip_train_fullpool)
    print("\n================ STAGE 5 NO-MLP & CATBOOST ABLATIONS ================", flush=True)
    print(f"Saved summary to {SUMMARY_PATH} ({res['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
