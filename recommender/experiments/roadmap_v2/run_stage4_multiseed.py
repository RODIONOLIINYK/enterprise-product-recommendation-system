"""Stage 4 Multi-Seed Verification (seeds 42, 43, 44): Stage 5 base_name-canonical vs SKU top4_3acde."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from recommender.artifacts import atomic_json
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes

from .config import EVIDENCE_ROOT
from .run_stage3_combined_and_stage5 import run_combined_and_stage5
from .stage5_base_entity import paired_base_name_bootstrap_ci


def run_multiseed(seeds: List[int] = [42, 43, 44], device: str | None = None) -> Dict[str, Any]:
    before_hashes = snapshot_preserved_hashes()
    per_seed: Dict[str, Any] = {}

    for seed in seeds:
        print(f"\n==================== MULTI-SEED RUN: seed={seed} ====================", flush=True)
        if seed == 42 and (EVIDENCE_ROOT / "stage3_combined_and_stage5_summary.json").is_file():
            # Reuse verified seed=42 summary cleanly without overwriting
            res = json.loads((EVIDENCE_ROOT / "stage3_combined_and_stage5_summary.json").read_text(encoding="utf-8"))
        else:
            res = run_combined_and_stage5(seed=seed, device=device, only_top4=True)
            # Save per-seed copy
            atomic_json(res, EVIDENCE_ROOT / f"seed{seed}_combined_and_stage5_summary.json", overwrite=True)
        per_seed[str(seed)] = res

    # Aggregate 3-seed means & stds for both architectures (base_name-level & SKU-level)
    def _agg(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(x) for x in values],
        }

    summary_stats: Dict[str, Any] = {
        "sku_top4_3acde_sku_metrics": {},
        "sku_top4_3acde_bn_collapsed": {"ndcg": {}, "discovery": {}},
        "stage5_base_name_canonical": {"ndcg": {}, "discovery": {}},
        "per_seed_h2h_ci": {},
    }

    for k in ("recall@5", "recall@10", "ndcg@10", "new_hits@10", "pack_switch_hits@10", "true_new_bn_hits@10"):
        summary_stats["sku_top4_3acde_sku_metrics"][k] = _agg(
            [per_seed[str(s)]["stage2_combined_sku"]["top4_3acde"]["ndcg"]["metrics"][k] for s in seeds]
        )

    for sel in ("ndcg", "discovery"):
        for k in ("bn_recall@5", "bn_recall@10", "bn_ndcg@10", "true_new_bn_hits@10", "true_new_bn_recall@10"):
            summary_stats["sku_top4_3acde_bn_collapsed"][sel][k] = _agg(
                [per_seed[str(s)]["stage2_combined_sku_collapsed_to_bn"]["top4_3acde"][sel]["bn_metrics"][k] for s in seeds]
            )
            summary_stats["stage5_base_name_canonical"][sel][k] = _agg(
                [per_seed[str(s)]["stage5_base_name_canonical"]["stage2_ranker"][sel]["bn_metrics"][k] for s in seeds]
            )

    baseline_bn_npz = EVIDENCE_ROOT / "baseline_pointwise425_bn_collapsed.npz"
    for s in seeds:
        s_str = str(s)
        summary_stats["per_seed_h2h_ci"][s_str] = per_seed[s_str]["head_to_head_sku_collapsed_to_stage5_ci"]
        if baseline_bn_npz.is_file():
            summary_stats["per_seed_h2h_ci"][s_str]["vs_original_baseline_ndcg"] = paired_base_name_bootstrap_ci(
                baseline_bn_npz,
                per_seed[s_str]["stage5_base_name_canonical"]["stage2_ranker"]["ndcg"]["bn_predictions_npz"],
            )
            summary_stats["per_seed_h2h_ci"][s_str]["vs_original_baseline_discovery"] = paired_base_name_bootstrap_ci(
                baseline_bn_npz,
                per_seed[s_str]["stage5_base_name_canonical"]["stage2_ranker"]["discovery"]["bn_predictions_npz"],
            )

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")

    out = {
        "stage": "stage4_multiseed_verification",
        "seeds": seeds,
        "heldout_access": False,
        "aggregates": summary_stats,
        "preserved_hashes_verified": True,
    }
    atomic_json(out, EVIDENCE_ROOT / "stage4_multiseed_summary.json", overwrite=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out = run_multiseed(seeds=args.seeds, device=args.device)
    agg = out["aggregates"]
    print("\n================ 3-SEED FINAL COMPARISON (MEAN ± STD) ================", flush=True)
    for sel in ("ndcg", "discovery"):
        print(f"\n--- Selection: {sel.upper()} ---")
        for label, d in [
            ("SKU top4_3acde (collapsed)", agg["sku_top4_3acde_bn_collapsed"][sel]),
            ("Stage 5 base_name native  ", agg["stage5_base_name_canonical"][sel]),
        ]:
            print(
                f"{label}: "
                f"bnR@5={d['bn_recall@5']['mean']:.5f}±{d['bn_recall@5']['std']:.5f} | "
                f"bnR@10={d['bn_recall@10']['mean']:.5f}±{d['bn_recall@10']['std']:.5f} | "
                f"bnNDCG@10={d['bn_ndcg@10']['mean']:.5f}±{d['bn_ndcg@10']['std']:.5f} | "
                f"trueNew@10={d['true_new_bn_hits@10']['mean']:.1f}±{d['true_new_bn_hits@10']['std']:.1f} ({d['true_new_bn_hits@10']['values']})"
            )


if __name__ == "__main__":
    main()
