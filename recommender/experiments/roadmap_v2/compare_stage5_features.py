"""Run or summarize the seed-42 Stage 5 comparisons without warehouse inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from recommender.artifacts import atomic_json, file_digest

from .config import ARTIFACT_ROOT, EVIDENCE_ROOT
from .stage5_base_entity import paired_base_name_bootstrap_ci


VARIANTS = {
    "aux_only": (),
    "financial_manager": ("3a_financial", "3c_manager"),
    "financial_manager_specs": ("3a_financial", "3c_manager", "3e_parsed_specs"),
}


def _validate_run(summary: dict, groups: tuple[str, ...]) -> None:
    if (
        summary["seed"] != 42
        or summary["aux_lambda"] != 0.3
        or tuple(summary["stage3_groups"]) != groups
        or summary["heldout_access"] is not False
        or summary["preserved_hashes_verified"] is not True
    ):
        raise ValueError(f"Unexpected run configuration for {groups}.")


def _paired(reference: Path, treatment: Path, *, queries=6191, targets=9809, new_targets=1340) -> dict:
    with np.load(reference, allow_pickle=True) as a, np.load(treatment, allow_pickle=True) as b:
        if (
            len(a["targets"]) != queries
            or a["targets"].sum() != targets
            or a["new_targets"].sum() != new_targets
            or not np.array_equal(a["targets"], b["targets"])
            or not np.array_equal(a["new_targets"], b["new_targets"])
        ):
            raise ValueError("Validation target counts differ between comparison arms.")
    return paired_base_name_bootstrap_ci(reference, treatment)


def compare(*, train_missing: bool = False) -> dict:
    tower = ARTIFACT_ROOT / "stage5_base_entity" / "two_tower-seed42"
    tower_files = ("best.pt", "train_vectors.pt", "val_vectors.pt")
    tower_hashes = {name: file_digest(tower / name) for name in tower_files}
    results = {}
    for name, groups in VARIANTS.items():
        tag = "plus_" + "_".join(groups) if groups else "base_only"
        path = EVIDENCE_ROOT / f"stage5_{tag}_summary.json"
        if train_missing and not path.is_file():
            from .run_stage5 import run_stage5

            print(f"[stage5] Running {name} (seed 42)", flush=True)
            run_stage5(stage3_groups=groups, aux_lambda=0.3, seed=42)
        results[name] = json.loads(path.read_text())
        _validate_run(results[name], groups)

    # This combined report preserves seed 42. The stand-alone combined summary
    # was overwritten by the later multi-seed campaign.
    combined = EVIDENCE_ROOT / "stage3_combined_and_stage5_summary.json"
    results["historical_warehouse"] = json.loads(combined.read_text())["stage5_base_name_canonical"]
    _validate_run(
        results["historical_warehouse"],
        ("3a_financial", "3c_manager", "3d_warehouse", "3e_parsed_specs"),
    )
    if any(run["two_tower"] != results["aux_only"]["two_tower"] for run in results.values()):
        raise ValueError("The comparison requires the same cached Two-Tower run.")
    if tower_hashes != {name: file_digest(tower / name) for name in tower_files}:
        raise ValueError("The seed-42 Two-Tower artifacts changed during the comparison.")

    comparisons = {
        "add_financial_and_manager": ("aux_only", "financial_manager"),
        "add_specs": ("financial_manager", "financial_manager_specs"),
        "add_all_three": ("aux_only", "financial_manager_specs"),
        "remove_warehouse": ("historical_warehouse", "financial_manager_specs"),
    }
    paired = {}
    for selection in ("ndcg", "discovery"):
        paths = {
            name: Path(run["stage2_ranker"][selection]["bn_predictions_npz"])
            for name, run in results.items()
        }
        paths["pre_roadmap"] = EVIDENCE_ROOT / "baseline_pointwise425_bn_collapsed.npz"
        paired[selection] = {
            name: _paired(paths[reference], paths[treatment])
            for name, (reference, treatment) in comparisons.items()
        }
        paired[selection]["vs_pre_roadmap"] = {
            name: _paired(paths["pre_roadmap"], paths[name]) for name in VARIANTS
        }
    return {
        "seed": 42,
        "heldout_access": False,
        "two_tower_sha256": tower_hashes,
        "bootstrap": {"draws": 2000, "seed": 123, "cluster": "customer"},
        "variants": results,
        "paired_deltas": paired,
    }


def compare_first_order() -> dict:
    runs = {
        mode: json.loads((EVIDENCE_ROOT / f"stage5_first_order_{mode}_seed42.json").read_text())
        for mode in ("control", "mixed")
    }
    control, mixed = runs["control"], runs["mixed"]
    cohort = control["first_order"]["cohorts"]
    if (
        control["reference_id"] != mixed["reference_id"]
        or cohort != mixed["first_order"]["cohorts"]
        or control["first_order"]["total_train_rows"] + cohort["train"]["included"]
        != mixed["first_order"]["total_train_rows"]
    ):
        raise ValueError("First-order arms must share the reference and validation cases.")
    deltas = {}
    for selection in ("ndcg", "discovery"):
        returning = [Path(run["stage2_ranker"][selection]["bn_predictions_npz"]) for run in runs.values()]
        first = [
            Path(run["first_order"]["stage2_ranker"][selection]["predictions_npz"])
            for run in runs.values()
        ]
        deltas[selection] = {
            "returning": _paired(*returning),
            "first": _paired(
                *first, queries=cohort["val"]["included"],
                targets=cohort["val"]["targets"], new_targets=cohort["val"]["targets"],
            ),
        }
    return {"reference_id": control["reference_id"], "seed": 42, "heldout_access": False, "paired_deltas": deltas}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--first-order", action="store_true")
    args = parser.parse_args()
    if args.first_order:
        destination = EVIDENCE_ROOT / "stage5_first_order_comparison_seed42.json"
        result = compare_first_order()
    else:
        destination = EVIDENCE_ROOT / "stage5_warehouse_exclusion_seed42.json"
        result = compare(train_missing=args.train_missing)
    atomic_json(result, destination, overwrite=True)
    print(destination)
