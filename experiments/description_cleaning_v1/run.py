"""Isolated resumable runner for cleaned-product-description experiment (Stage 1 + uv_47 MLP only).

Usage:
  PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --prepare-only
  PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --verify-baseline-only
  PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_html
  PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_clean
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from recommender.artifacts import atomic_json
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.description_cleaning.config import (
    ARTIFACT_ROOT,
    CACHE_DIR,
    EVIDENCE_ROOT,
    build_description_variant_specs,
)
from recommender.experiments.description_cleaning.data import prepare_description_cleaning_bundle
from recommender.experiments.description_cleaning.evaluate import verify_and_replay_baseline_reference
from recommender.experiments.description_cleaning.train import run_description_variant
from recommender.train_retriever import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=None, help="Subset of variants to run (D_html, D_clean).")
    parser.add_argument("--prepare-only", action="store_true", help="Only build/verify description bundle and BGE-M3 caches.")
    parser.add_argument("--verify-baseline-only", action="store_true", help="Only verify and replay no-brand-v1 baseline.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--selector-epochs", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = get_device(args.device)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_description_variant_specs()
    results_path = EVIDENCE_ROOT / "results.json"
    if results_path.is_file():
        all_results: Dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        all_results = {
            "campaign": "description_cleaning_v1",
            "baseline_target": "no-brand-v1",
            "selector_recipe": "uv_47",
            "selector_input_dim": 303,
            "seed": args.seed,
            "stage1_epochs": args.stage1_epochs,
            "selector_epochs": args.selector_epochs,
            "validation_only": True,
            "heldout_access": False,
            "stage2_executed": False,
            "promoted": False,
            "metadata_policy": "snapshot_unverified",
            "baseline": None,
            "variants": {},
        }

    print("=== Preparing description_cleaning_v1 bundle & content-hashed BGE-M3 caches ===", flush=True)
    enrichment = prepare_description_cleaning_bundle(cache_dir=CACHE_DIR, device=str(device))
    if args.prepare_only:
        atomic_json(all_results, results_path, overwrite=True)
        print("Description bundle prepared cleanly.", flush=True)
        return

    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()

    if all_results.get("baseline") is None or args.verify_baseline_only:
        print("=== Verifying and replaying no-brand-v1 baseline reference (no retraining) ===", flush=True)
        baseline_res = verify_and_replay_baseline_reference(
            train_ret,
            val_ret,
            info,
            train_src,
            val_src,
            enrichment,
            ARTIFACT_ROOT,
            device=str(device),
        )
        all_results["baseline"] = baseline_res
        atomic_json(all_results, results_path, overwrite=True)
        print("Baseline no-brand-v1 reference verified and replayed.", flush=True)
        if args.verify_baseline_only:
            return

    selected_names = ["D_html", "D_clean"] if not args.variants else list(args.variants)
    for name in selected_names:
        if name not in specs or name == "no_brand_v1":
            raise ValueError(f"Unsupported variant {name!r}; expected D_html and/or D_clean.")

    for name in selected_names:
        print(f"=== Running description variant: {name} (seed={args.seed}, device={device}) ===", flush=True)
        res = run_description_variant(
            specs[name],
            args.seed,
            train_ret,
            val_ret,
            info,
            train_src,
            val_src,
            enrichment,
            ARTIFACT_ROOT,
            stage1_epochs=args.stage1_epochs,
            selector_epochs=args.selector_epochs,
            device=str(device),
        )
        all_results["variants"][name] = res
        atomic_json(all_results, results_path, overwrite=True)
        print(f"=== Completed variant: {name} ===", flush=True)


if __name__ == "__main__":
    main()
