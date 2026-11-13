"""Full-pool evaluation (overall macro Recall@K + unseen discovery) and paired customer-clustered CIs."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from recommender.experiments.dynamic_selector.evaluate import ranking_metrics

from .config import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED


OVERALL_RECALL_KS: Tuple[int, ...] = (5, 10, 25, 50, 100)


def evaluate_full_pool_metrics(
    scores: torch.Tensor,
    val_data: Mapping[str, Any],
    *,
    overall_ks: Sequence[int] = (5, 10, 25, 50, 100),
    save_predictions: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Evaluate full-pool scores with overall macro Recall@K (all eligible) and unseen-only discovery@25/50/100."""
    scores_cpu = scores.detach().cpu()
    allowed = val_data["allowed"]
    unseen = val_data["unseen"]
    target = val_data["target"]
    new_target = val_data["new_target"]
    source = val_data["source"]

    # Reuse exact source semantics from dynamic_selector.evaluate.ranking_metrics.
    base_metrics = ranking_metrics(scores_cpu, val_data, indices=None, save=None)

    # Full eligible ranking (repeat + new, NO unseen filtering).
    masked = scores_cpu.masked_fill(~allowed, -torch.inf)
    order = masked.argsort(dim=-1, descending=True, stable=True)
    indices = torch.arange(scores_cpu.shape[1]).expand(len(scores_cpu), -1)
    ranked = indices.gather(1, order)
    eligible_ranked = allowed.gather(1, ranked)
    hits = (target.gather(1, ranked) & eligible_ranked).numpy()
    novel = (new_target.gather(1, ranked) & eligible_ranked).numpy()

    count = np.array([len(x) for x in source["positive_pids"]], dtype=np.float64)
    new_count = np.array([len(x) for x in source["new_positive_pids"]], dtype=np.int64)
    total_targets = int(count.sum())
    total_new = int(new_count.sum())

    per_query_recalls: Dict[int, np.ndarray] = {}
    overall_recalls: Dict[str, float] = {}
    for k in overall_ks:
        rec_k = hits[:, : int(k)].sum(1) / count
        per_query_recalls[int(k)] = rec_k
        overall_recalls[f"recall@{int(k)}"] = float(rec_k.mean())

    # Cross-check exact consistency with ranking_metrics on shared keys.
    if abs(overall_recalls["recall@5"] - base_metrics["recall@5"]) > 1e-12:
        raise AssertionError("Mismatch on overall recall@5 against ranking_metrics.")
    if abs(overall_recalls["recall@10"] - base_metrics["recall@10"]) > 1e-12:
        raise AssertionError("Mismatch on overall recall@10 against ranking_metrics.")

    discount = 1.0 / np.log2(np.arange(10) + 2.0)
    ideal = (discount[None] * (np.arange(10)[None] < count[:, None])).sum(1)
    ndcg10 = (hits[:, :10] * discount).sum(1) / ideal
    any_hit = hits.any(1)
    mrr = np.where(any_hit, 1.0 / (hits.argmax(1) + 1), 0.0)
    new_hits10 = novel[:, :10].sum(1).astype(np.int64)

    # Unseen-only ranking for discovery@25/50/100.
    unseen_scores = scores_cpu.masked_fill(~unseen, -torch.inf)
    unseen_order = unseen_scores.argsort(dim=-1, descending=True, stable=True)
    top100 = unseen_order[:, :100]
    valid100 = unseen.gather(1, top100)
    novel100 = (new_target.gather(1, top100) & valid100).numpy()
    new_hits25 = novel100[:, :25].sum(1).astype(np.int64)
    new_hits50 = novel100[:, :50].sum(1).astype(np.int64)
    new_hits100 = novel100[:, :100].sum(1).astype(np.int64)

    if save_predictions is not None:
        out_path = Path(save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            customers=np.array(source["customer_ids"]),
            positive_counts=count.astype(np.int64),
            new_targets=new_count,
            recall=per_query_recalls[10],
            recall5=per_query_recalls[5],
            recall10=per_query_recalls[10],
            recall25=per_query_recalls[25],
            recall50=per_query_recalls[50],
            recall100=per_query_recalls[100],
            ndcg=ndcg10,
            mrr=mrr,
            new_hits=new_hits10,
            new_hits25=new_hits25,
            new_hits50=new_hits50,
            new_hits100=new_hits100,
        )

    out = dict(base_metrics)
    out.update(
        {
            "queries": int(len(scores_cpu)),
            "total_targets": total_targets,
            "new_targets": total_new,
            **overall_recalls,
        }
    )
    return out


def paired_feature_ablation_intervals(
    control_npz: Union[Path, str],
    treatment_npz: Union[Path, str],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, list[float]]:
    """Customer-clustered 95% bootstrap intervals for treatment - control (single-seed conditional uncertainty)."""
    a = np.load(control_npz)
    b = np.load(treatment_npz)
    for key in ("customers", "new_targets", "positive_counts"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"Paired predictions require identical ordered validation rows ({key}).")

    _, groups = np.unique(a["customers"], return_inverse=True)
    n_cust = int(groups.max()) + 1
    rows = np.bincount(groups)
    new_targets = np.bincount(groups, weights=a["new_targets"])

    metric_specs = (
        ("recall@5_delta", "recall5", rows),
        ("recall@10_delta", "recall10", rows),
        ("recall@25_delta", "recall25", rows),
        ("recall@50_delta", "recall50", rows),
        ("recall@100_delta", "recall100", rows),
        ("ndcg@10_delta", "ndcg", rows),
        ("new_recall@10_delta", "new_hits", new_targets),
        ("new_recall@25_delta", "new_hits25", new_targets),
        ("new_recall@50_delta", "new_hits50", new_targets),
        ("new_recall@100_delta", "new_hits100", new_targets),
    )

    deltas = {
        out_name: np.bincount(groups, weights=b[key] - a[key])
        for out_name, key, _ in metric_specs
    }

    rng = np.random.default_rng(seed)
    draws = rng.integers(n_cust, size=(repeats, n_cust))
    out: Dict[str, list[float]] = {}
    for out_name, _, denom_vec in metric_specs:
        denom = np.maximum(denom_vec[draws].sum(1), 1.0)
        vals = deltas[out_name][draws].sum(1) / denom
        out[out_name] = np.quantile(vals, [0.025, 0.5, 0.975]).tolist()
    return out
