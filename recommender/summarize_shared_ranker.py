"""Rebuild aggregate evidence from completed experimental checkpoints only."""

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from recommender.shared_data import DEFAULT_CACHE
from recommender.shared_ranker import SCALE_GROUP_NAMES, feature_schema
from recommender.train_retriever import get_device
from recommender.train_shared_ranker import (
    DEFAULT_OUTPUT,
    diagnostics,
    evaluate_checkpoint,
    paired_customer_bootstrap,
)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = get_device(args.device)
    manifest = torch.load(DEFAULT_CACHE / "manifest.pt", map_location="cpu")
    val = torch.load(DEFAULT_CACHE / "val.pt", map_location="cpu")
    vectors = manifest["product_vectors"].to(device)
    destination = Path("experiments/shared_stage1")
    destination.mkdir(parents=True, exist_ok=True)
    rows, curves, grouped = [], {}, defaultdict(list)
    metric_names = ("val_loss", "val_recall@10", "val_ndcg@10", "new_hits@10", "new_target_count", "new_recall@10")
    for path in sorted(DEFAULT_OUTPUT.glob("*/result.json")):
        result = json.loads(path.read_text())
        run_dir = path.parent
        model, metrics = evaluate_checkpoint(result["selected_checkpoint"], val, manifest, device)
        for name in ("val_recall@10", "val_ndcg@10", "new_hits@10"):
            if abs(metrics[name] - result["metrics"][name]) > 1e-12:
                raise ValueError(f"Selected-checkpoint replay changed {run_dir.name} {name}.")
        diag = diagnostics(model, val, vectors, device, run_dir / "predictions.npz")
        gradients = result["parameter_utilization"]
        row = {key: result[key] for key in (
            "settings", "selected_checkpoint", "selected_epoch", "qualified_recall_guardrail", "parameter_count",
            "training_seconds", "peak_mps_driver_bytes_sampled", "process_peak_rss_bytes_cumulative",
            "peak_ndcg_other_checkpoint", "inference_perturbations",
        )}
        row.update({"metrics": {name: metrics[name] for name in metric_names}, "diagnostics": diag,
                    "checkpoint_sha256": digest(result["selected_checkpoint"]),
                    "utilization": {
                        "parameters_without_gradient": sum(item["count"] for item in gradients if item["grad_none"]),
                        "zero_gradient_entries_in_probe_batch": sum(item["zero_gradient_entries"] or 0 for item in gradients),
                    }})
        if hasattr(model, "token_builder") and hasattr(model.token_builder, "log_scales"):
            row["learned_input_group_scales"] = dict(zip(SCALE_GROUP_NAMES, model.token_builder.log_scales.exp().detach().cpu().tolist(), strict=True))
            row["context_readout_auxiliary_weight_norm"] = float(model.context_readout.weight[:, 128:].detach().norm())
        rows.append(row)
        grouped[result["settings"]["run"]].append(row)
        history = json.loads((run_dir / "history.json").read_text())
        curves[run_dir.name] = [{"epoch": item["epoch"], "train_loss": item["train_loss"], "seconds": item["seconds"],
                                **{name: item["validation"][name] for name in metric_names}} for item in history]
        print(f"Verified selected checkpoint {run_dir.name}", flush=True)
    statistics = {}
    for run, results in grouped.items():
        statistics[run] = {"seeds": [result["settings"]["seed"] for result in results], "n": len(results)}
        for name in ("new_hits@10", "val_recall@10", "val_ndcg@10"):
            values = [result["metrics"][name] for result in results]
            statistics[run][name] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    comparisons = {}
    pairs = [(("R0", 42), (run, 42)) for run in ("R0W", "R1", "R2", "R3", "R4")]
    pairs += [(("R0W", 42), ("R1", 42)), (("R1", 42), ("R2", 42)), (("R2", 42), ("R3", 42))]
    pairs += [(("R3", 42), (run, 42)) for run in ("R3P0", "R3H0")]
    for seed in (42, 17, 2026):
        pairs += [(("R3", seed), ("R4", seed))]
        pairs += [(("R0", seed), ("R3P0", seed))]
        if seed != 42:
            pairs += [(("R0", seed), (run, seed)) for run in ("R3", "R4")]
            pairs += [(("R3", seed), ("R3P0", seed))]
    for left, right in pairs:
        left_name, right_name = [f"{run}-seed{seed}" for run, seed in (left, right)]
        left_path, right_path = [DEFAULT_OUTPUT / name / "predictions.npz" for name in (left_name, right_name)]
        if left_path.is_file() and right_path.is_file():
            comparisons[f"{right_name}_minus_{left_name}"] = paired_customer_bootstrap(left_path, right_path)
    baselines = json.loads((DEFAULT_OUTPUT / "baseline_replay.json").read_text())
    for row in rows:
        run, seed = row["settings"]["run"], row["settings"]["seed"]
        name = f"{run}-seed{seed}"
        comparisons[f"{name}_minus_preserved_discovery_reference"] = paired_customer_bootstrap(
            DEFAULT_OUTPUT / "reference-best_discovery_ranker.npz", DEFAULT_OUTPUT / name / "predictions.npz",
        )
    integrity = {}
    for path, expected in json.loads(Path("audit/artifacts.json").read_text()).items():
        actual = digest(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Original checkpoint/cache changed: {path}.")
        integrity[path] = {"sha256": actual, "unchanged": True}
    payload = {"date": "2026-11-11", "starting_commit": "99bc1c5c17138c45ed7fe2ec128cf14c46dbe400",
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "cpu_threads": torch.get_num_threads(), "evaluation_device": str(device)},
               "stage1_digest": manifest["stage1_digest"], "cache_identity": manifest["identity"],
               "train_rows": manifest["train_rows"], "validation_rows": manifest["val_rows"],
               "expanded_catalogue_size": len(manifest["product_ids"]), "feature_schema": feature_schema(),
               "stage1_temporal_caveat": "Frozen reference still contains training-period demand look-ahead. No Stage1 retraining performed.",
               "interval_method": "2000 paired customer-cluster bootstrap samples; fixed selected models, not training/selection uncertainty",
               "memory_caveat": "Sampled MPS driver memory includes caching; process peak RSS is cumulative. Timings share hardware with verification work.",
               "reference_replays": baselines, "runs": rows, "seed_statistics": statistics,
               "deferred_training": ["R5", "R6", "R7", "R8", "R9"],
               "archived_control": "Earlier unmatched-initialization R0W archived and excluded; current R0W matches R1 common module initialization."}
    for name, data in (("summary.json", payload), ("curves.json", curves), ("paired_comparisons.json", comparisons),
                       ("reference_integrity.json", integrity)):
        (destination / name).write_text(json.dumps(data, indent=2) + "\n")
    print(f"Exported {len(rows)} completed experiment/seed runs; original artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
