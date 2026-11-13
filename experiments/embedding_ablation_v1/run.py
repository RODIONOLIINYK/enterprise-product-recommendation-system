"""Isolated runner for U0_t1_fam_clean product-text and product-embedding ablations (Stage 1 + uv_47 MLP only).

Usage:
  PYTHONPATH=. ./venv/bin/python experiments/embedding_ablation_v1/run.py
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from recommender.artifacts import atomic_json
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.embedding_ablation.config import (
    ARTIFACT_ROOT,
    CACHE_DIR,
    EVIDENCE_ROOT,
    INAPPLICABLE_EMBEDDING_ABLATIONS,
    INAPPLICABLE_TEXT_ABLATIONS,
    build_ablation_specs,
    make_exploratory_combined_spec,
)
from recommender.experiments.embedding_ablation.data import (
    DEFAULT_RECIPES,
    prepare_ablation_enrichment_bundle,
)
from recommender.experiments.embedding_ablation.train import run_ablation_variant
from recommender.train_retriever import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of individual ablation variants to run.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--selector-epochs", type=int, default=10)
    parser.add_argument("--no-reuse-baseline", action="store_true")
    parser.add_argument("--exploratory-name", default=None, help="Optional exploratory cumulative combination variant name.")
    parser.add_argument("--exploratory-omit-text", nargs="*", default=())
    parser.add_argument("--exploratory-omit-emb", nargs="*", default=())
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = get_device(args.device)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_ablation_specs()
    results_path = EVIDENCE_ROOT / "results.json"
    if results_path.is_file():
        all_results: Dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        all_results = {
            "campaign": "embedding_ablation_v1",
            "baseline_target": "U0_t1_fam_clean",
            "selector_recipe": "uv_47",
            "seed": args.seed,
            "stage1_epochs": args.stage1_epochs,
            "selector_epochs": args.selector_epochs,
            "heldout_access": False,
            "stage2_executed": False,
            "inapplicable": {
                "text_ablations": INAPPLICABLE_TEXT_ABLATIONS,
                "embedding_ablations": INAPPLICABLE_EMBEDDING_ABLATIONS,
            },
            "individual": {},
            "exploratory_combined": {},
        }

    recipes = list(DEFAULT_RECIPES)
    extra_omissions = {}
    combo_spec = None
    if args.exploratory_name:
        combo_recipe = f"t1_fam_combo_{args.exploratory_name}"
        combo_spec = make_exploratory_combined_spec(
            args.exploratory_name,
            omitted_text_fields=tuple(args.exploratory_omit_text),
            omitted_product_embeddings=tuple(args.exploratory_omit_emb),
            recipe_name=combo_recipe,
        )
        if combo_recipe not in recipes:
            recipes.append(combo_recipe)
            extra_omissions[combo_recipe] = tuple(args.exploratory_omit_text)

    enrichment = prepare_ablation_enrichment_bundle(
        recipes=tuple(recipes),
        extra_recipe_omissions=extra_omissions or None,
        cache_dir=CACHE_DIR,
        device=str(device),
    )
    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()

    if args.exploratory_name and combo_spec is not None:
        res = run_ablation_variant(
            combo_spec,
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
            reuse_clean_v2_baseline=not args.no_reuse_baseline,
            device=str(device),
        )
        all_results["exploratory_combined"][args.exploratory_name] = res
        atomic_json(all_results, results_path)
        print(f"Exploratory combination {args.exploratory_name} complete.", flush=True)
        return

    selected_names = list(specs.keys()) if not args.variants else list(args.variants)
    for name in selected_names:
        if name not in specs:
            raise ValueError(f"Unknown variant {name!r}; expected one of {list(specs.keys())}")

    for name in selected_names:
        print(f"=== Running ablation variant: {name} ===", flush=True)
        res = run_ablation_variant(
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
            reuse_clean_v2_baseline=not args.no_reuse_baseline,
            device=str(device),
        )
        all_results["individual"][name] = res
        atomic_json(all_results, results_path)

    print("Embedding & text ablation individual runs complete.", flush=True)


if __name__ == "__main__":
    main()
