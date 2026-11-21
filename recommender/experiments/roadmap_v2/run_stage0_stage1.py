"""Stage 0 + Stage 1 runner: native rolling Stage 1 baseline + base_name 3-way ablation on Stage 1 & Stage 2."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from recommender.artifacts import atomic_json
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes

from .base_name_features import prepare_or_load_base_name_feature_bundle
from .config import (
    ARM_BASE_NAME_ONLY,
    ARM_SKU_ONLY,
    ARM_SKU_PLUS_BASE_NAME,
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    VALID_BASE_NAME_ARMS,
    feature_names_for_arm,
)
from .stage1_mlp import fit_stage1_base_name_arm
from .stage2_ranker import (
    attach_arm_features_to_fixed_slate,
    build_slate_from_stage1_scores,
    fit_stage2_roadmap_run,
)


def run_stage0_and_stage1(*, seed: int = SEED, device: str | None = None) -> Dict[str, Any]:
    before_hashes = snapshot_preserved_hashes()

    # 1. Load sealed rolling_observed_v2 41D + base_name 19D feature bundles
    campaign_data, info, bn_manifest, meta = prepare_or_load_base_name_feature_bundle(build=True)
    _, _ = prepare_or_load_rolling_selector_scores(campaign_data, info, meta)
    base_slates, base_slate_manifest = prepare_or_load_rolling_slate_cache(campaign_data, info, meta)
    product_ids = list(info["product_ids"])
    product_vectors = campaign_data["train"]["v"]

    # 2. Stage 1 MLP 3-way ablation under strictly rolling_observed_v2:
    #    - sku_only (also serves as Stage 0 native rolling-selected S1-R0!)
    #    - sku_plus_base_name (41 + 19 = 60 features -> 320D MLP)
    #    - base_name_only (19 SKU cols replaced in-place by base_name -> 301D MLP)
    stage1_results: Dict[str, Any] = {}
    for arm in VALID_BASE_NAME_ARMS:
        print(f"\n================ STAGE 1 MLP: {arm} ================", flush=True)
        stage1_results[arm] = fit_stage1_base_name_arm(
            arm,
            campaign_data,
            info,
            meta,
            artifact_root=ARTIFACT_ROOT / "stage1_base_name",
            seed=seed,
            device=device,
        )

    stage1_intervals = {
        f"sku_only_to_{arm}": paired_rolling_intervals(
            stage1_results[ARM_SKU_ONLY]["predictions_npz"],
            stage1_results[arm]["predictions_npz"],
        )
        for arm in (ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY)
    }

    # 3. Stage 2 Transformer 3-way ablation on IDENTICAL FIXED rolling100 slates
    #    (isolates pure Stage 2 feature effect across sku_only vs sku_plus_base_name vs base_name_only)
    stage2_fixed_results: Dict[str, Any] = {}
    for arm in VALID_BASE_NAME_ARMS:
        print(f"\n================ STAGE 2 (FIXED SLATES): {arm} ================", flush=True)
        tr_slate = attach_arm_features_to_fixed_slate(base_slates["train"], campaign_data["train"], arm)
        va_slate = attach_arm_features_to_fixed_slate(base_slates["val"], campaign_data["val"], arm)
        stage2_fixed_results[arm] = fit_stage2_roadmap_run(
            arm_name=arm,
            train_slate=tr_slate,
            val_slate=va_slate,
            val_data=campaign_data["val"],
            product_vectors=product_vectors,
            product_ids=product_ids,
            meta=meta,
            artifact_root=ARTIFACT_ROOT / "stage2_base_name",
            seed=seed,
            device=device,
            feature_names=feature_names_for_arm(arm),
            slate_tag="fixed_rolling100",
        )

    stage2_fixed_intervals = {
        sel: {
            f"sku_only_to_{arm}": paired_rolling_intervals(
                stage2_fixed_results[ARM_SKU_ONLY][sel]["predictions_npz"],
                stage2_fixed_results[arm][sel]["predictions_npz"],
            )
            for arm in (ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY)
        }
        for sel in ("ndcg", "discovery")
    }

    # 4. End-to-End Stage 1 -> Stage 2 test: rebuild rolling100 slates from each base_name Stage 1 arm
    #    (sku_plus_base_name and base_name_only) and train matched Stage 2 Transformer on its OWN upgraded slates!
    stage2_e2e_results: Dict[str, Any] = {}
    for arm in (ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY):
        print(f"\n================ STAGE 2 (END-TO-END OWN SLATES): {arm} ================", flush=True)
        s1_tr_scores = torch.load(stage1_results[arm]["train_scores_pt"], map_location="cpu", weights_only=False)["scores"]
        s1_va_scores = torch.load(stage1_results[arm]["val_scores_pt"], map_location="cpu", weights_only=False)["scores"]
        tr_slate = build_slate_from_stage1_scores(
            base_slates["train"], campaign_data["train"], s1_tr_scores, product_ids, arm
        )
        va_slate = build_slate_from_stage1_scores(
            base_slates["val"], campaign_data["val"], s1_va_scores, product_ids, arm
        )
        stage2_e2e_results[arm] = fit_stage2_roadmap_run(
            arm_name=arm,
            train_slate=tr_slate,
            val_slate=va_slate,
            val_data=campaign_data["val"],
            product_vectors=product_vectors,
            product_ids=product_ids,
            meta=meta,
            artifact_root=ARTIFACT_ROOT / "stage2_base_name",
            seed=seed,
            device=device,
            feature_names=feature_names_for_arm(arm),
            slate_tag="own_stage1_slates",
        )

    stage2_e2e_intervals = {
        sel: {
            f"sku_only_to_{arm}_e2e": paired_rolling_intervals(
                stage2_fixed_results[ARM_SKU_ONLY][sel]["predictions_npz"],
                stage2_e2e_results[arm][sel]["predictions_npz"],
            )
            for arm in (ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY)
        }
        for sel in ("ndcg", "discovery")
    }

    after_hashes = snapshot_preserved_hashes()
    if after_hashes != before_hashes:
        raise AssertionError("Preserved reference/historical hashes changed!")

    summary = {
        "stage": "stage0_and_stage1_base_name_ablation",
        "heldout_access": False,
        "seed": int(seed),
        "base_name_cache_manifest": bn_manifest,
        "stage1_mlp": stage1_results,
        "stage1_paired_bootstrap_95ci": stage1_intervals,
        "stage2_fixed_slates": stage2_fixed_results,
        "stage2_fixed_paired_bootstrap_95ci": stage2_fixed_intervals,
        "stage2_e2e_own_slates": stage2_e2e_results,
        "stage2_e2e_paired_bootstrap_95ci": stage2_e2e_intervals,
        "preserved_hashes_verified": True,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, EVIDENCE_ROOT / "stage1_base_name_summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    summary = run_stage0_and_stage1(seed=args.seed, device=args.device)
    print("\n================ FINAL SUMMARY (STAGE 0 + STAGE 1) ================", flush=True)
    for arm in VALID_BASE_NAME_ARMS:
        m = summary["stage1_mlp"][arm]["metrics"]
        print(
            f"S1 [{arm:18s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']}) "
            f"New@25={m['new_hits@25']} New@50={m['new_hits@50']} New@100={m['new_hits@100']}"
        )
    for arm in VALID_BASE_NAME_ARMS:
        m = summary["stage2_fixed_slates"][arm]["ndcg"]["metrics"]
        print(
            f"S2-fixed [{arm:18s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']})"
        )
    for arm in (ARM_SKU_PLUS_BASE_NAME, ARM_BASE_NAME_ONLY):
        m = summary["stage2_e2e_own_slates"][arm]["ndcg"]["metrics"]
        print(
            f"S2-e2e   [{arm:18s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']})"
        )


if __name__ == "__main__":
    main()
