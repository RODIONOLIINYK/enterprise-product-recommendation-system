"""Stage 2 CLI runner: Multi-Task 60-Day Auxiliary Head + History-SKU Dropout on Stage 1 MLP & Stage 2 Transformer."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from recommender.artifacts import atomic_json
from recommender.development import load_development_purchases
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes

from .augmentation import build_strict_60d_aux_labels
from .base_name_features import prepare_or_load_base_name_feature_bundle
from .config import ARTIFACT_ROOT, EVIDENCE_ROOT, SEED, VALID_BASE_NAME_ARMS
from .stage2_aug_runner import fit_stage1_augmented, fit_stage2_augmented
from .stage2_ranker import attach_arm_features_to_fixed_slate, build_slate_from_stage1_scores


def run_stage2(
    *,
    arm: str,
    use_own_stage1_slates: bool = True,
    seed: int = SEED,
    device: str | None = None,
) -> Dict[str, Any]:
    if arm not in VALID_BASE_NAME_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {VALID_BASE_NAME_ARMS}.")
    before_hashes = snapshot_preserved_hashes()

    campaign_data, info, _, meta = prepare_or_load_base_name_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(campaign_data, info, meta)
    base_slates, _ = prepare_or_load_rolling_slate_cache(campaign_data, info, meta)
    product_ids = list(info["product_ids"])
    product_vectors = campaign_data["train"]["v"]

    # Build strictly leak-free 60-day auxiliary training labels (< TRAIN_CUTOFF)
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    tr_src = campaign_data["train"]["source"]
    aux_60d = build_strict_60d_aux_labels(
        purchases,
        tr_src["customer_ids"],
        tr_src["prediction_days"].tolist(),
        product_ids,
        horizon_days=60,
    )
    next_pos_mean = float(campaign_data["train"]["target"].float().sum(dim=1).mean().item())
    aux_pos_mean = float(aux_60d.float().sum(dim=1).mean().item())
    aux_new_pos_mean = float((aux_60d & campaign_data["train"]["unseen"]).float().sum(dim=1).mean().item())
    next_new_pos_mean = float((campaign_data["train"]["target"] & campaign_data["train"]["unseen"]).float().sum(dim=1).mean().item())
    print(
        f"[stage2] 60d aux labels built (capped < TRAIN_CUTOFF): "
        f"mean_pos/query next={next_pos_mean:.2f} -> 60d={aux_pos_mean:.2f} | "
        f"mean_unseen_pos/query next={next_new_pos_mean:.3f} -> 60d={aux_new_pos_mean:.3f} "
        f"({aux_new_pos_mean / max(next_new_pos_mean, 1e-6):.2f}x discovery positives!)",
        flush=True,
    )

    # Load matched Stage 1 control from Stage 1 ablation
    s1_ctrl_dir = ARTIFACT_ROOT / "stage1_base_name" / "runs" / f"stage1-{arm}-seed{seed}"
    s1_ctrl = json.loads((s1_ctrl_dir / "result.json").read_text(encoding="utf-8"))

    # Select matched Stage 2 slates for Stage 2 augmentation tests (fixed across 2a/2b/2c for clean isolation)
    if use_own_stage1_slates and (s1_ctrl_dir / "train_scores.pt").is_file():
        s1_tr_scores = torch.load(s1_ctrl_dir / "train_scores.pt", map_location="cpu", weights_only=False)["scores"]
        s1_va_scores = torch.load(s1_ctrl_dir / "val_scores.pt", map_location="cpu", weights_only=False)["scores"]
        tr_slate = build_slate_from_stage1_scores(base_slates["train"], campaign_data["train"], s1_tr_scores, product_ids, arm)
        va_slate = build_slate_from_stage1_scores(base_slates["val"], campaign_data["val"], s1_va_scores, product_ids, arm)
        slate_tag = f"own_s1_{arm}"
        s2_ctrl_dir = ARTIFACT_ROOT / "stage2_base_name" / "runs" / f"stage2-{arm}-own_stage1_slates-seed{seed}"
        if not (s2_ctrl_dir / "result.json").is_file():
            s2_ctrl_dir = ARTIFACT_ROOT / "stage2_base_name" / "runs" / f"stage2-{arm}-fixed_rolling100-seed{seed}"
    else:
        tr_slate = attach_arm_features_to_fixed_slate(base_slates["train"], campaign_data["train"], arm)
        va_slate = attach_arm_features_to_fixed_slate(base_slates["val"], campaign_data["val"], arm)
        slate_tag = "fixed_rolling100"
        s2_ctrl_dir = ARTIFACT_ROOT / "stage2_base_name" / "runs" / f"stage2-{arm}-fixed_rolling100-seed{seed}"
    s2_ctrl = json.loads((s2_ctrl_dir / "result.json").read_text(encoding="utf-8"))

    variants = (
        ("2a_aux60d_lam03", 0.3, 0.0),
        ("2b_histdrop_p030", 0.0, 0.30),
        ("2c_aux60d_plus_histdrop", 0.3, 0.30),
    )

    s1_results: Dict[str, Any] = {"control": s1_ctrl}
    s2_results: Dict[str, Any] = {"control": s2_ctrl}

    for vname, lam, pdrop in variants:
        print(f"\n================ STAGE 2 AUGMENTATION (Stage 1 MLP): {vname} ================", flush=True)
        s1_results[vname] = fit_stage1_augmented(
            arm=arm,
            variant_name=vname,
            campaign_data=campaign_data,
            info=info,
            meta=meta,
            aux_60d_labels=aux_60d if lam > 0 else None,
            aux_lambda=lam,
            p_drop=pdrop,
            seed=seed,
            device=device,
        )

    for vname, lam, pdrop in variants:
        print(f"\n================ STAGE 2 AUGMENTATION (Stage 2 Transformer): {vname} ================", flush=True)
        s2_results[vname] = fit_stage2_augmented(
            arm=arm,
            variant_name=vname,
            train_slate=tr_slate,
            val_slate=va_slate,
            train_data=campaign_data["train"],
            val_data=campaign_data["val"],
            product_vectors=product_vectors,
            product_ids=product_ids,
            meta=meta,
            aux_60d_labels=aux_60d if lam > 0 else None,
            aux_lambda=lam,
            p_drop=pdrop,
            slate_tag=slate_tag,
            seed=seed,
            device=device,
        )

    s1_intervals = {
        f"ctrl_to_{vname}": paired_rolling_intervals(
            s1_ctrl["predictions_npz"], s1_results[vname]["predictions_npz"]
        )
        for vname, _, _ in variants
    }
    s2_intervals = {
        sel: {
            f"ctrl_to_{vname}": paired_rolling_intervals(
                s2_ctrl[sel]["predictions_npz"], s2_results[vname][sel]["predictions_npz"]
            )
            for vname, _, _ in variants
        }
        for sel in ("ndcg", "discovery")
    }

    after_hashes = snapshot_preserved_hashes()
    if after_hashes != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")

    summary = {
        "stage": "stage2_augmentation",
        "arm": arm,
        "slate_tag": slate_tag,
        "seed": int(seed),
        "heldout_access": False,
        "aux_60d_stats": {
            "mean_next_pos": next_pos_mean,
            "mean_60d_pos": aux_pos_mean,
            "mean_next_unseen_pos": next_new_pos_mean,
            "mean_60d_unseen_pos": aux_new_pos_mean,
        },
        "stage1_mlp": s1_results,
        "stage1_paired_bootstrap_95ci": s1_intervals,
        "stage2_transformer": s2_results,
        "stage2_paired_bootstrap_95ci": s2_intervals,
        "preserved_hashes_verified": True,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, EVIDENCE_ROOT / "stage2_augmentation_summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=str, required=True, choices=VALID_BASE_NAME_ARMS)
    parser.add_argument("--fixed-slates", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    summary = run_stage2(
        arm=args.arm,
        use_own_stage1_slates=not args.fixed_slates,
        seed=args.seed,
        device=args.device,
    )
    print("\n================ STAGE 2 AUGMENTATION SUMMARY ================", flush=True)
    for k, res in summary["stage1_mlp"].items():
        m = res["metrics"]
        print(
            f"S1 [{k:24s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']}) "
            f"New@25={m['new_hits@25']} New@50={m['new_hits@50']} New@100={m['new_hits@100']}"
        )
    for k, res in summary["stage2_transformer"].items():
        m = res["ndcg"]["metrics"]
        print(
            f"S2 [{k:24s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']})"
        )


if __name__ == "__main__":
    main()
