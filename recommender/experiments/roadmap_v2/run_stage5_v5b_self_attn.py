"""Validation-only Stage 5 v5b ablation: per-head self-attention bias & history residual MLP.

Runs three matched arms strictly on Train (29,095 < 2025-10-01) -> Validation (6,480, 2025-10-01..2026-02-18)
using the audited bug-fixed pipeline (observed_before=TRAIN_CUTOFF, float32 candidate features,
customer-relative normalized event quantity `seq_rel_qty`), ZERO held-out Test access:
  1. `control`: SharedUVStage5Ranker (self_attention_bias=False, history_residual_mlp=False)
  2. `self_attn_bias`: + 8-head pairwise self-attention bias (+40 params:
       earlier_days_norm, later_days_norm, key_rel_qty, key_is_token0, same_base_name)
  3. `self_attn_bias_hist_res`: + self-attention bias AND Tokens 1..32 residual MLP over
       [v_bn (128), rel_qty, live_age_before_now, inter_purchase_gap_norm]
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TRAIN_CUTOFF
from recommender.train_retriever import get_device

from .augmentation import build_strict_60d_aux_labels
from .clean_ranker import (
    SELF_ATTENTION_BIAS_NAMES,
    SharedUVStage5Ranker,
)
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_WEIGHT_DECAY,
)
from .native_base_name import CANDIDATE_38_FEATURE_NAMES, CUSTOMER_DYNAMIC_11_FEATURE_NAMES
from .run_stage5 import (
    _collapse_bool_matrix_to_bn,
    _encode_first_baskets,
    _stage5_tower,
    _verify_reference_files,
)
from .run_stage5_ablations import _evaluate_with_cohorts_and_sku
from .run_stage5_clean_v4b import _prepare_clean_v4b_data
from .run_stage5_v5_shared_uv import V4B_TT_DIR
from .stage5_artifacts import stage5_run_paths
from .stage5_base_entity import evaluate_base_name_predictions, paired_base_name_bootstrap_ci


OUT_ROOT = ARTIFACT_ROOT / "stage5_v5b_self_attn"
REF_PATH = EVIDENCE_ROOT / "stage5_reference.json"
REF_3STAGE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"

ARM_SPECS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("control", {"self_attention_bias": False, "history_residual_mlp": False}),
    ("self_attn_bias", {"self_attention_bias": True, "history_residual_mlp": False}),
    ("self_attn_bias_hist_res", {"self_attention_bias": True, "history_residual_mlp": True}),
)


def _build_v5b_batch(
    split_d: Mapping[str, Any],
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    return {
        "customer_vector": u_bn_split[rows].to(dev, dtype=torch.float32),
        "customer_features": split_d["tabular_features"][rows].to(dev, dtype=torch.float32),
        "catalog_product_vectors": v_bn_dev,
        "sequence": split_d["sequence"][rows].to(dev),
        "seq_rel_qty": split_d["seq_rel_qty"][rows].to(dev),
        "attention_mask": split_d["attention_mask"][rows].to(dev),
        "seq_product_ids": split_d["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_d["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_d["prediction_days"][rows].to(dev),
        "candidate_features": split_d["features_bn"][rows].to(dev, dtype=torch.float32),
        "candidate_mask": split_d["allowed_bn"][rows].to(dev),
    }


@torch.no_grad()
def _score_v5b(
    model: SharedUVStage5Ranker,
    split_d: Mapping[str, Any],
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    dev: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for rows in torch.arange(len(split_d["customer_ids"])).split(batch_size):
        out = model(**_build_v5b_batch(split_d, u_bn_split, v_bn_dev, rows, dev), return_aux=False)
        chunks.append(out.logits.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _encode_vectors_from_tower(
    tower_path: Path,
    universe: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    prod_inputs: Mapping[str, Any],
    bn_data: Mapping[str, Dict[str, Any]],
    dev: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tower = _stage5_tower(
        universe,
        enrichment,
        num_demand_features=int(prod_inputs["demand_features"].shape[1]),
        num_customer_tabular_features=0,
        drop_sales_branch=True,
    ).to(dev)
    ck = torch.load(tower_path, map_location="cpu", weights_only=True)
    tower.load_state_dict(ck["model_state_dict"], strict=True)
    tower.eval()

    p_in = {k: v.to(dev) for k, v in prod_inputs.items() if k != "attr_ids"}
    p_in["attr_ids"] = {k: v.to(dev) for k, v in prod_inputs["attr_ids"].items()}
    v_bn = tower.product_vectors(p_in)
    if not torch.isfinite(v_bn).all():
        raise ValueError("Non-finite product vectors from Stage 1 tower.")

    encoded_u: Dict[str, torch.Tensor] = {}
    for split_name in ("train", "val"):
        tmp = dict(bn_data[split_name])
        _encode_first_baskets(tmp, tower, v_bn, dev)
        encoded_u[split_name] = tmp["u_bn"]
    return v_bn.cpu(), encoded_u["train"], encoded_u["val"]


def _format_self_bias_weights(model: SharedUVStage5Ranker) -> Optional[Dict[str, Any]]:
    w = model.self_attention_bias_weights
    if w is None:
        return None
    arr = w.detach().cpu().numpy()
    per_head = []
    for h in range(arr.shape[0]):
        per_head.append(
            {name: round(float(arr[h, j]), 4) for j, name in enumerate(SELF_ATTENTION_BIAS_NAMES)}
        )
    mean_abs = {
        name: round(float(np.abs(arr[:, j]).mean()), 4)
        for j, name in enumerate(SELF_ATTENTION_BIAS_NAMES)
    }
    return {"feature_names": list(SELF_ATTENTION_BIAS_NAMES), "per_head": per_head, "mean_abs": mean_abs}


def run_v5b_ablation(
    *,
    seed: int = SEED,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    two_tower_dir: Path = V4B_TT_DIR,
    arms: Optional[Sequence[str]] = None,
    epochs: int = STAGE2_EPOCHS,
) -> Dict[str, Any]:
    run_root, summary_path = stage5_run_paths(OUT_ROOT, output_dir, summary_path)
    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    ref_active = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ref3 = json.loads(REF_3STAGE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(ref_active)
    _verify_reference_files(ref3)

    selected_specs = [
        (name, kwargs) for name, kwargs in ARM_SPECS if arms is None or name in set(arms)
    ]
    if not selected_specs:
        raise ValueError(f"No valid arms selected from {[n for n, _ in ARM_SPECS]}")

    print(
        "[v5b-self-attn] Preparing audited float32 native features (< TRAIN_CUTOFF pack cutoff) "
        "& encoding fresh Stage 1 u_bn/v_bn...",
        flush=True,
    )
    (
        bn_data,
        bn_static,
        universe,
        enrichment,
        _,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        _,
    ) = _prepare_clean_v4b_data(
        observed_before=TRAIN_CUTOFF,
        candidate_feature_storage_dtype="float32",
        frozen_v5_replay=False,
    )
    prod_inputs = {
        "text_embedding": bn_static["text_pack_free"].float(),
        "demand_features": bn_static["full_specs_multihot"].float(),
        "attr_ids": bn_static["attr_ids"],
    }
    tower_path = two_tower_dir / "best.pt"
    tower_sha = file_digest(tower_path)
    v_bn, u_tr, u_va = _encode_vectors_from_tower(
        tower_path, universe, enrichment, prod_inputs, bn_data, dev
    )
    v_dev = v_bn.to(dev)

    tr_bn = bn_data["train"]
    va_bn = bn_data["val"]
    sku_val = sku_data["val"]
    bn_to_members = universe["bn_to_member_pid_indices"]

    aux_60d_sku = build_strict_60d_aux_labels(
        purchases, tr_bn["customer_ids"], tr_bn["prediction_days"].tolist(), product_ids
    )
    aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members).float()
    full_window = tr_bn["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

    arm_results: Dict[str, Any] = {}
    pred_paths: Dict[str, Dict[str, Path]] = {}

    for arm_name, arm_kwargs in selected_specs:
        arm_dir = run_root / f"{arm_name}-seed{seed}"
        arm_dir.mkdir(parents=True, exist_ok=True)

        def _make_ranker(aux: bool) -> SharedUVStage5Ranker:
            return SharedUVStage5Ranker(
                num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
                num_customer_features=len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
                num_pairwise_bias_features=13,
                d_model=128,
                num_heads=8,
                memory_ffn_dim=384,
                mlp_widths=(384, 192, 96),
                dropout=0.12,
                aux_60d_enabled=aux,
                seed=seed,
                **arm_kwargs,
            ).to(dev)

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = _make_ranker(aux=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(
            f"\n[v5b:{arm_name}] Training SharedUVStage5Ranker ({n_params:,} params, {arm_kwargs})...",
            flush=True,
        )

        opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=STAGE2_WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=STAGE2_ETA_MIN)
        loss_fn = RankerLoss().to(dev)
        aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
        gen = torch.Generator().manual_seed(int(seed))

        best_ndcg_key = (-1.0, -1, -1.0)
        best_disc_key = (-1, -1.0, -1.0)
        epoch_history = []
        t_arm = time.perf_counter()

        for ep in range(1, int(epochs) + 1):
            ep_t0 = time.perf_counter()
            model.train()
            losses = []
            perm = torch.randperm(len(tr_bn["customer_ids"]), generator=gen)
            for rows in perm.split(64):
                opt.zero_grad(set_to_none=True)
                inp = _build_v5b_batch(tr_bn, u_tr, v_dev, rows, dev)
                cmask = inp["candidate_mask"]
                y = tr_bn["target_bn"][rows].to(dev, dtype=torch.float32) * cmask.float()
                out, a_lg = model(**inp, return_aux=True)
                l_main = loss_fn(out.logits, y, candidate_mask=cmask)["loss"]
                aux_rows = full_window[rows].to(dev)
                loss = l_main
                if aux_rows.any():
                    y_aux = aux_60d_bn[rows].to(dev)[aux_rows] * cmask[aux_rows].float()
                    l_aux = aux_loss_fn(a_lg[aux_rows], y_aux, candidate_mask=cmask[aux_rows])["loss"]
                    loss = loss + 0.3 * l_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, STAGE2_GRAD_CLIP_NORM)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sch.step()

            va_lg = _score_v5b(model, va_bn, u_va, v_dev, dev, batch_size=64)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            m_ep = evaluate_base_name_predictions(
                ranked, va_bn["target_bn"], va_bn["new_target_bn"], va_bn["customer_ids"]
            )
            inf_state = {
                k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")
            }

            nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
            dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
            if nk > best_ndcg_key:
                best_ndcg_key = nk
                atomic_torch_save(
                    {"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state, "config": arm_kwargs},
                    arm_dir / "best_ndcg.pt",
                )
            if dk > best_disc_key:
                best_disc_key = dk
                atomic_torch_save(
                    {"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state, "config": arm_kwargs},
                    arm_dir / "best_discovery.pt",
                )
            epoch_history.append(
                {
                    "epoch": ep,
                    "loss": float(np.mean(losses)),
                    "bn_recall@5": float(m_ep["bn_recall@5"]),
                    "bn_recall@10": float(m_ep["bn_recall@10"]),
                    "bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
                    "true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
                }
            )
            print(
                f"[v5b:{arm_name}] ep={ep:02d}/{epochs} loss={np.mean(losses):.4f} "
                f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
                f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
                f"({time.perf_counter() - ep_t0:.1f}s)",
                flush=True,
            )

        s2_res: Dict[str, Any] = {
            "config": dict(arm_kwargs),
            "trainable_params": int(n_params),
            "training_seconds": float(time.perf_counter() - t_arm),
            "epoch_history": epoch_history,
        }
        pred_paths[arm_name] = {}
        cpu_bench_ms = {}
        for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
            ck = torch.load(arm_dir / fname, map_location="cpu", weights_only=False)
            eval_m = _make_ranker(aux=False)
            eval_m.load_state_dict(ck["model_state_dict"], strict=True)
            va_lg = _score_v5b(eval_m, va_bn, u_va, v_dev, dev, batch_size=64)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            npz_p = arm_dir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(
                ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p
            )
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"v5b {arm_name} {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            bias_summary = _format_self_bias_weights(eval_m)
            if bias_summary is not None:
                ev["learned_self_attention_bias"] = bias_summary
            s2_res[sel_name] = ev
            pred_paths[arm_name][sel_name] = npz_p

            if sel_name == "ndcg":
                eval_cpu = eval_m.to("cpu").eval()
                one_inp = _build_v5b_batch(va_bn, u_va, v_bn, torch.tensor([0]), torch.device("cpu"))
                with torch.no_grad():
                    for _ in range(10):
                        eval_cpu(**one_inp)
                    ts = []
                    for _ in range(100):
                        t0 = time.perf_counter()
                        eval_cpu(**one_inp)
                        ts.append((time.perf_counter() - t0) * 1000.0)
                cpu_bench_ms = {
                    "median_ms": float(np.median(ts)),
                    "p95_ms": float(np.percentile(ts, 95)),
                    "mean_ms": float(np.mean(ts)),
                }

        s2_res["cpu_single_request_ranker_ms"] = cpu_bench_ms
        arm_results[arm_name] = s2_res

        del model, opt, sch
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    paired: Dict[str, Any] = {}
    if "control" in pred_paths:
        for arm_name in pred_paths:
            if arm_name == "control":
                continue
            paired[f"{arm_name}_vs_control"] = {
                sel: paired_base_name_bootstrap_ci(pred_paths["control"][sel], pred_paths[arm_name][sel])
                for sel in ("ndcg", "discovery")
            }
    if "self_attn_bias" in pred_paths and "self_attn_bias_hist_res" in pred_paths:
        paired["self_attn_bias_hist_res_vs_self_attn_bias"] = {
            sel: paired_base_name_bootstrap_ci(
                pred_paths["self_attn_bias"][sel], pred_paths["self_attn_bias_hist_res"][sel]
            )
            for sel in ("ndcg", "discovery")
        }

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    _verify_reference_files(ref_active)
    _verify_reference_files(ref3)

    results = {
        "experiment": "stage5_v5b_self_attn_ablation",
        "seed": int(seed),
        "epochs": int(epochs),
        "heldout_access": False,
        "observed_before": str(TRAIN_CUTOFF),
        "candidate_feature_storage_dtype": "float32",
        "frozen_v5_replay": False,
        "two_tower_source": str(tower_path),
        "two_tower_sha256": tower_sha,
        "self_attention_bias_features": list(SELF_ATTENTION_BIAS_NAMES),
        "arms": arm_results,
        "paired_bootstrap": paired,
        "preserved_hashes_verified": True,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(results, summary_path, overwrite=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, help="New exclusive run directory.")
    parser.add_argument("--summary-path", type=Path, help="New summary JSON path.")
    parser.add_argument("--two-tower-dir", type=Path, default=V4B_TT_DIR)
    parser.add_argument("--arms", nargs="+", choices=[n for n, _ in ARM_SPECS], default=None)
    args = parser.parse_args()

    res = run_v5b_ablation(
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        two_tower_dir=args.two_tower_dir,
        arms=args.arms,
    )
    print(f"\nSaved v5b ablation summary to {res['summary_path']} ({res['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
