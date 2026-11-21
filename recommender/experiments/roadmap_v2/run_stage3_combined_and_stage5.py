"""Combine winning Stage 3 groups end-to-end (SKU-level) + run Stage 5 base_name-canonical head-to-head."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from recommender.artifacts import atomic_json
from recommender.catalogue import load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.rolling_candidate_v2.config import SLATE_POLICY
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.scope import get_datasets_locs
from recommender.train_retriever import get_device

from .augmentation import build_strict_60d_aux_labels
from .base_name_features import prepare_or_load_base_name_feature_bundle
from .commercial_features import prepare_or_load_commercial_feature_bundle
from .config import ARM_SKU_PLUS_BASE_NAME, ARTIFACT_ROOT, EVIDENCE_ROOT, SEED
from .parsed_spec_features import prepare_or_load_parsed_spec_feature_bundle
from .run_stage3 import (
    ALL_STAGE3_GROUPS,
    attach_stage3_groups_to_slate,
    fit_stage1_with_groups,
    fit_stage2_with_groups,
)
from .run_stage5 import _collapse_bool_matrix_to_bn, run_stage5
from .stage1_mlp import compute_pack_switch_masks
from .stage2_ranker import (
    RoadmapPointwiseRanker,
    build_slate_from_stage1_scores,
    predict_roadmap_stage2,
)
from .stage5_base_entity import (
    build_base_name_universe,
    collapse_sku_scores_to_base_name_ranking,
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
)


def _evaluate_sku_stage2_at_base_name_level(
    ckpt_path: Path,
    val_slate: Dict[str, Any],
    product_vectors: torch.Tensor,
    target_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    customer_ids: Sequence[str],
    pid_idx_to_bn_idx: np.ndarray,
    save_npz: Path,
    dev: torch.device,
) -> Dict[str, Any]:
    """Replay a SKU-level Stage 2 checkpoint and evaluate its deduplicated base_name ranking apples-to-apples."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RoadmapPointwiseRanker(num_features=int(ck["num_features"]), seed=SEED, aux_60d_enabled=False).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    pred = predict_roadmap_stage2(model, val_slate, product_vectors.to(dev), dev)
    ranked_bn = collapse_sku_scores_to_base_name_ranking(
        pred["scores"], val_slate["candidate_indices"], pid_idx_to_bn_idx, max_k=50
    )
    return evaluate_base_name_predictions(ranked_bn, target_bn, new_target_bn, customer_ids, save_path=save_npz)


def run_combined_and_stage5(
    *,
    seed: int = SEED,
    device: str | None = None,
    only_top4: bool = False,
) -> Dict[str, Any]:
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()

    campaign_data, info, _, meta = prepare_or_load_base_name_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(campaign_data, info, meta)
    base_slates, _ = prepare_or_load_rolling_slate_cache(campaign_data, info, meta)
    comm_mmaps, _ = prepare_or_load_commercial_feature_bundle(build=False)
    spec_mmaps, _ = prepare_or_load_parsed_spec_feature_bundle(build=False)

    product_ids = list(info["product_ids"])
    product_vectors = campaign_data["train"]["v"]
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    tr_src = campaign_data["train"]["source"]
    aux_60d = build_strict_60d_aux_labels(
        purchases, tr_src["customer_ids"], tr_src["prediction_days"].tolist(), product_ids
    )
    ps_mask, tn_mask = compute_pack_switch_masks(campaign_data["val"], product_ids, meta["all_pid_to_base_name"])

    # Canonical base_name targets for apples-to-apples SKU-collapsed vs Stage 5 evaluation
    universe = build_base_name_universe(product_ids, catalogue, purchases)
    bn_to_members = universe["bn_to_member_pid_indices"]
    bn_rep = universe["bn_rep_pid_idx"]
    val_target_bn = _collapse_bool_matrix_to_bn(campaign_data["val"]["target"], bn_to_members)
    val_allowed_bn = _collapse_bool_matrix_to_bn(campaign_data["val"]["allowed"], bn_to_members)
    val_seen_bn = torch.from_numpy(
        np.array(campaign_data["val"]["base_name_features"][:, bn_rep, 0], dtype=np.float32) > 0.5
    )
    val_new_target_bn = val_target_bn & val_allowed_bn & (~val_seen_bn)

    combos = (
        {"top4_3acde": ("3a_financial", "3c_manager", "3d_warehouse", "3e_parsed_specs")}
        if only_top4
        else {
            "all5_3abcde": ("3a_financial", "3b_pricing", "3c_manager", "3d_warehouse", "3e_parsed_specs"),
            "top4_3acde": ("3a_financial", "3c_manager", "3d_warehouse", "3e_parsed_specs"),
        }
    )

    arm = ARM_SKU_PLUS_BASE_NAME
    s1_combined: Dict[str, Any] = {}
    s2_combined: Dict[str, Any] = {}
    s2_bn_eval: Dict[str, Any] = {}
    slate_cov: Dict[str, Any] = {}

    for cname, groups in combos.items():
        print(f"\n================ COMBINED STAGE 1 MLP: {cname} {groups} ================", flush=True)
        s1_res = fit_stage1_with_groups(
            arm=arm,
            groups=groups,
            tag=cname,
            campaign_data=campaign_data,
            comm_mmaps=comm_mmaps,
            spec_mmaps=spec_mmaps,
            info=info,
            meta=meta,
            aux_60d_labels=aux_60d,
            aux_lambda=0.3,
            p_drop=0.0,
            artifact_root=ARTIFACT_ROOT / "stage3_combined",
            seed=seed,
            device=device,
        )
        s1_combined[cname] = s1_res

        # Rebuild upgraded quota100_50 slates from this combined Stage 1 MLP!
        s1_tr = torch.load(s1_res["train_scores_pt"], map_location="cpu", weights_only=False)["scores"]
        s1_va = torch.load(s1_res["val_scores_pt"], map_location="cpu", weights_only=False)["scores"]
        va_idx = select_candidates(s1_va, campaign_data["val"]["allowed"], campaign_data["val"]["unseen"], SLATE_POLICY)
        slate_cov[cname] = {
            "total_in_slate": int(campaign_data["val"]["target"].gather(1, va_idx).sum().item()),
            "new_in_slate": int(campaign_data["val"]["new_target"].gather(1, va_idx).sum().item()),
            "pack_switch_in_slate": int(ps_mask.gather(1, va_idx).sum().item()),
            "true_new_bn_in_slate": int(tn_mask.gather(1, va_idx).sum().item()),
        }
        print(f"[combined/{cname}] Upgraded Stage 1 slate coverage: {slate_cov[cname]}", flush=True)

        # Build slates with base 41D first, then attach arm + Stage 3 combined groups
        tr_base = build_slate_from_stage1_scores(base_slates["train"], campaign_data["train"], s1_tr, product_ids, arm)
        va_base = build_slate_from_stage1_scores(base_slates["val"], campaign_data["val"], s1_va, product_ids, arm)
        tr_idx_np = tr_base["candidate_indices"].numpy()
        va_idx_np = va_base["candidate_indices"].numpy()
        tr_base["pairwise_features"] = torch.from_numpy(
            np.array(campaign_data["train"]["features"][np.arange(len(tr_idx_np))[:, None], tr_idx_np], dtype=np.float32)
        )
        va_base["pairwise_features"] = torch.from_numpy(
            np.array(campaign_data["val"]["features"][np.arange(len(va_idx_np))[:, None], va_idx_np], dtype=np.float32)
        )
        tr_s = attach_stage3_groups_to_slate(tr_base, campaign_data["train"], comm_mmaps["train"], spec_mmaps["train"], arm, groups)
        va_s = attach_stage3_groups_to_slate(va_base, campaign_data["val"], comm_mmaps["val"], spec_mmaps["val"], arm, groups)

        print(f"\n================ COMBINED STAGE 2 TRANSFORMER (OWN SLATES): {cname} ================", flush=True)
        s2_res = fit_stage2_with_groups(
            arm=arm,
            groups=groups,
            tag=f"{cname}_own_slates",
            train_slate=tr_s,
            val_slate=va_s,
            train_data=campaign_data["train"],
            val_data=campaign_data["val"],
            product_vectors=product_vectors,
            product_ids=product_ids,
            meta=meta,
            aux_60d_labels=aux_60d,
            aux_lambda=0.3,
            p_drop=0.0,
            artifact_root=ARTIFACT_ROOT / "stage3_combined",
            seed=seed,
            device=device,
        )
        s2_combined[cname] = s2_res

        # Apples-to-apples base_name-level evaluation of this SKU-level Stage 2 model (both NDCG & Discovery checkpoints)
        s2_bn_eval[cname] = {}
        for sel in ("ndcg", "discovery"):
            npz_out = Path(s2_res["run_dir"]) / f"bn_collapsed_{sel}.npz"
            m_bn = _evaluate_sku_stage2_at_base_name_level(
                Path(s2_res[sel]["checkpoint"]),
                va_s,
                product_vectors,
                val_target_bn,
                val_new_target_bn,
                campaign_data["val"]["source"]["customer_ids"],
                universe["pid_idx_to_bn_idx"],
                npz_out,
                dev,
            )
            s2_bn_eval[cname][sel] = {"bn_metrics": m_bn, "bn_predictions_npz": str(npz_out)}
            print(
                f"[SKU-collapsed -> base_name] {cname}/{sel}: "
                f"bnR@5={m_bn['bn_recall@5']:.5f} bnR@10={m_bn['bn_recall@10']:.5f} "
                f"bnNDCG@10={m_bn['bn_ndcg@10']:.5f} trueNew@10={m_bn['true_new_bn_hits@10']}/{m_bn['true_new_bn_targets']}",
                flush=True,
            )

    # Pick winning SKU-level combo on North-Star criterion (bn_ndcg@10 + true_new_bn_hits@10 + bn_recall@10)
    def _north_star_key(cname: str) -> tuple:
        m = s2_bn_eval[cname]["ndcg"]["bn_metrics"]
        return (round(m["bn_ndcg@10"], 4), m["true_new_bn_hits@10"], m["bn_recall@10"])

    best_sku_combo = max(combos.keys(), key=_north_star_key)
    best_groups = combos[best_sku_combo]
    print(f"\n>>> Winning SKU-level combination: {best_sku_combo} {best_groups}", flush=True)

    # Run Stage 5 (Full base_name Canonical Entity Pipeline) using the EXACT same winning Stage 3 groups + aux60d!
    print(f"\n================ STAGE 5: FULL BASE_NAME CANONICAL PIPELINE ({best_groups}) ================", flush=True)
    stage5_res = run_stage5(
        stage3_groups=best_groups,
        aux_lambda=0.3,
        seed=seed,
        device=device,
    )

    # Head-to-head paired customer-clustered bootstrap 95% CI: Best SKU-collapsed vs Stage 5 Native base_name!
    h2h_ci = {}
    for sel in ("ndcg", "discovery"):
        h2h_ci[sel] = paired_base_name_bootstrap_ci(
            s2_bn_eval[best_sku_combo][sel]["bn_predictions_npz"],
            stage5_res["stage2_ranker"][sel]["bn_predictions_npz"],
        )

    # Also compare combined SKU-level winner vs Stage 1 sku_only control (when available for this seed)
    ctrl_path = ARTIFACT_ROOT / "stage2_base_name" / "runs" / f"stage2-sku_only-fixed_rolling100-seed{seed}" / "result.json"
    if not ctrl_path.is_file():
        ctrl_path = ARTIFACT_ROOT / "stage2_base_name" / "runs" / "stage2-sku_only-fixed_rolling100-seed42" / "result.json"
    s1_sku_ctrl = json.loads(ctrl_path.read_text()) if ctrl_path.is_file() else None
    sku_vs_baseline_ci = (
        {
            cname: paired_rolling_intervals(
                s1_sku_ctrl["ndcg"]["predictions_npz"],
                s2_combined[cname]["ndcg"]["predictions_npz"],
            )
            for cname in combos
        }
        if s1_sku_ctrl is not None
        else {}
    )

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")

    summary = {
        "stage": "stage3_combined_and_stage5_head_to_head",
        "seed": int(seed),
        "heldout_access": False,
        "combos": {k: list(v) for k, v in combos.items()},
        "best_sku_combo": best_sku_combo,
        "stage1_combined": s1_combined,
        "upgraded_slate_coverage": slate_cov,
        "stage2_combined_sku": s2_combined,
        "stage2_combined_sku_vs_sku_only_ci": sku_vs_baseline_ci,
        "stage2_combined_sku_collapsed_to_bn": s2_bn_eval,
        "stage5_base_name_canonical": stage5_res,
        "head_to_head_sku_collapsed_to_stage5_ci": h2h_ci,
        "preserved_hashes_verified": True,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if not only_top4:
        atomic_json(summary, EVIDENCE_ROOT / "stage3_combined_and_stage5_summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    s = run_combined_and_stage5(seed=args.seed, device=args.device)
    print("\n================ HEAD-TO-HEAD SUMMARY (SKU-COLLAPSED vs STAGE 5 BASE_NAME) ================", flush=True)
    for cname in s["combos"]:
        ms = s["stage2_combined_sku"][cname]["ndcg"]["metrics"]
        mb = s["stage2_combined_sku_collapsed_to_bn"][cname]["ndcg"]["bn_metrics"]
        print(
            f"SKU-level [{cname:12s}] SKU: R@5={ms['recall@5']:.5f} R@10={ms['recall@10']:.5f} "
            f"NDCG@10={ms['ndcg@10']:.5f} New@10={ms['new_hits@10']} (ps={ms['pack_switch_hits@10']}, tn={ms['true_new_bn_hits@10']}) | "
            f"BN-collapsed: bnR@10={mb['bn_recall@10']:.5f} bnNDCG@10={mb['bn_ndcg@10']:.5f} trueNew@10={mb['true_new_bn_hits@10']}"
        )
    m5 = s["stage5_base_name_canonical"]["stage2_ranker"]["ndcg"]["bn_metrics"]
    r5 = s["stage5_base_name_canonical"]["stage2_ranker"]["ndcg"]["resolved_sku_metrics"]
    print(
        f"Stage 5   [base_entity ] BN-native:    bnR@10={m5['bn_recall@10']:.5f} bnNDCG@10={m5['bn_ndcg@10']:.5f} "
        f"trueNew@10={m5['true_new_bn_hits@10']} | resolved SKU: R@10={r5['sku_recall@10']:.5f} NDCG@10={r5['sku_ndcg@10']:.5f} New@10={r5['sku_new_hits@10']}"
    )
    print("\nHead-to-head 95% CI (Stage 5 - Best SKU-collapsed, ndcg):")
    print(json.dumps(s["head_to_head_sku_collapsed_to_stage5_ci"]["ndcg"], indent=2))


if __name__ == "__main__":
    main()
