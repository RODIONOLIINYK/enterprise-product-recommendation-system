"""Stage 1 cohort evaluation and Stage 2 customer-clustered bootstrap intervals."""

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch


def stage1_cohort_metrics(
    logits: torch.Tensor,
    val_data: Dict[str, Any],
    val_source: Dict[str, Any],
    cohort_masks: Dict[str, torch.Tensor],
    *,
    save_predictions: Path | str | None = None,
) -> Dict[str, Any]:
    """Compute unseen Stage 1 metrics overall and across description/brand/family cohorts (1,857 denominator)."""
    unseen_scores = logits.masked_fill(~val_data["unseen"], -torch.inf)
    novel_orders = unseen_scores.argsort(dim=-1, descending=True, stable=True)
    top50 = novel_orders[:, :50]
    hit_25_matrix = val_data["new_target"].gather(1, top50[:, :25]) & val_data["unseen"].gather(1, top50[:, :25])
    hit_50_matrix = val_data["new_target"].gather(1, top50) & val_data["unseen"].gather(1, top50)

    target_rows, target_cols = torch.where(val_data["new_target"])
    total_new_targets = int(sum(len(row) for row in val_source["new_positive_pids"]))
    if len(target_rows) != total_new_targets:
        # Unrepresented targets (not in 841 pool) count as misses in the full denominator.
        pass

    hit25_per_target = np.zeros(len(target_rows), dtype=np.int64)
    hit50_per_target = np.zeros(len(target_rows), dtype=np.int64)
    for idx, (r, c) in enumerate(zip(target_rows.tolist(), target_cols.tolist())):
        if val_data["unseen"][r, c]:
            hit25_per_target[idx] = int((top50[r, :25] == c).any())
            hit50_per_target[idx] = int((top50[r] == c).any())

    def _cohort_stats(mask_1d: np.ndarray) -> Dict[str, Any]:
        n = int(mask_1d.sum())
        h25 = int(hit25_per_target[mask_1d].sum()) if n else 0
        h50 = int(hit50_per_target[mask_1d].sum()) if n else 0
        return {
            "targets": n,
            "hits@25": h25,
            "recall@25": float(h25 / n) if n else None,
            "hits@50": h50,
            "recall@50": float(h50 / n) if n else None,
        }

    has_desc = cohort_masks["has_description"][target_cols].numpy()
    in_desc = cohort_masks["in_description_file"][target_cols].numpy()
    brand_miss = cohort_masks["brand_missing"][target_cols].numpy()
    fam_miss = cohort_masks["family_missing"][target_cols].numpy()

    cohorts = {
        "with_description_nonempty": _cohort_stats(has_desc),
        "without_description_nonempty": _cohort_stats(~has_desc),
        "in_description_csv": _cohort_stats(in_desc),
        "not_in_description_csv": _cohort_stats(~in_desc),
        "brand_present": _cohort_stats(~brand_miss),
        "brand_missing": _cohort_stats(brand_miss),
        "family_present": _cohort_stats(~fam_miss),
        "family_missing": _cohort_stats(fam_miss),
    }

    if save_predictions is not None:
        np.savez_compressed(
            save_predictions,
            customers=np.array(val_source["customer_ids"]),
            hits25=hit_25_matrix.sum(dim=1).numpy(),
            hits50=hit_50_matrix.sum(dim=1).numpy(),
            new_targets=np.array([len(x) for x in val_source["new_positive_pids"]]),
        )
    return cohorts


def paired_stage1_intervals(
    control_npz_paths: Sequence[Path],
    treatment_npz_paths: Sequence[Path],
    *,
    repeats: int = 2000,
) -> Dict[str, list[float]]:
    """Nested seed + customer-cluster 95% bootstrap intervals for Stage 1 unseen recall@25 and @50 deltas."""
    a = [np.load(p) for p in control_npz_paths]
    b = [np.load(p) for p in treatment_npz_paths]
    for row in (*a, *b):
        if not np.array_equal(row["customers"], a[0]["customers"]) or not np.array_equal(
            row["new_targets"], a[0]["new_targets"]
        ):
            raise ValueError("Paired Stage 1 predictions require identical ordered validation rows.")
    _, groups = np.unique(a[0]["customers"], return_inverse=True)
    n_cust = int(groups.max()) + 1
    target_counts = np.bincount(groups, weights=a[0]["new_targets"])
    deltas = {
        key: np.stack([np.bincount(groups, weights=y[key] - x[key]) for x, y in zip(a, b)])
        for key in ("hits25", "hits50")
    }
    samples: Dict[str, list[float]] = {"new_recall@25_delta": [], "new_recall@50_delta": []}
    rng = np.random.default_rng(123)
    for _ in range(repeats):
        seeds = rng.integers(len(a), size=len(a))
        customers = rng.integers(n_cust, size=n_cust)
        denom = max(float(target_counts[customers].sum()), 1.0)
        samples["new_recall@25_delta"].append(float(deltas["hits25"][seeds][:, customers].mean(0).sum() / denom))
        samples["new_recall@50_delta"].append(float(deltas["hits50"][seeds][:, customers].mean(0).sum() / denom))
    return {name: np.quantile(vals, [0.025, 0.5, 0.975]).tolist() for name, vals in samples.items()}
