"""Full-pool MLP-only evaluation and paired customer-clustered bootstrap intervals."""

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


def save_mlp_per_query_predictions(
    scores: torch.Tensor,
    val_campaign_data: Mapping[str, Any],
    save_path: Path | str,
) -> Dict[str, Any]:
    """Compute and save per-query full-pool MLP metrics (matching dynamic_selector/evaluate.py semantics)."""
    scores = scores.detach().cpu()
    allowed = val_campaign_data["allowed"]
    unseen = val_campaign_data["unseen"]
    target = val_campaign_data["target"]
    new_target = val_campaign_data["new_target"]
    source = val_campaign_data["source"]

    masked = scores.masked_fill(~allowed, -torch.inf)
    order = masked.argsort(dim=-1, descending=True, stable=True)
    indices = torch.arange(scores.shape[1]).expand(len(scores), -1)
    ranked = indices.gather(1, order)
    eligible_ranked = allowed.gather(1, ranked)

    hits = (target.gather(1, ranked) & eligible_ranked).numpy()
    novel = (new_target.gather(1, ranked) & eligible_ranked).numpy()

    count = np.array([len(x) for x in source["positive_pids"]], dtype=np.float64)
    new_count = np.array([len(x) for x in source["new_positive_pids"]], dtype=np.int64)
    recall10 = hits[:, :10].sum(1) / count
    discount = 1.0 / np.log2(np.arange(10) + 2.0)
    ideal = (discount[None] * (np.arange(10)[None] < count[:, None])).sum(1)
    ndcg10 = (hits[:, :10] * discount).sum(1) / ideal
    new_hits10 = novel[:, :10].sum(1).astype(np.int64)

    unseen_scores = scores.masked_fill(~unseen, -torch.inf)
    unseen_order = unseen_scores.argsort(dim=-1, descending=True, stable=True)
    top100 = unseen_order[:, :100]
    valid100 = unseen.gather(1, top100)
    novel100 = (new_target.gather(1, top100) & valid100).numpy()
    new_hits25 = novel100[:, :25].sum(1).astype(np.int64)
    new_hits50 = novel100[:, :50].sum(1).astype(np.int64)
    new_hits100 = novel100[:, :100].sum(1).astype(np.int64)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        customers=np.array(source["customer_ids"]),
        recall=recall10,
        ndcg=ndcg10,
        new_hits=new_hits10,
        new_hits25=new_hits25,
        new_hits50=new_hits50,
        new_hits100=new_hits100,
        new_targets=new_count,
        positive_counts=count.astype(np.int64),
    )

    total_new = int(new_count.sum())
    return {
        "queries": int(len(scores)),
        "new_targets": total_new,
        "recall@10": float(recall10.mean()),
        "ndcg@10": float(ndcg10.mean()),
        "new_hits@10": int(new_hits10.sum()),
        "new_recall@10": float(new_hits10.sum() / max(total_new, 1)),
        "new_hits@25": int(new_hits25.sum()),
        "new_recall@25": float(new_hits25.sum() / max(total_new, 1)),
        "new_hits@50": int(new_hits50.sum()),
        "new_recall@50": float(new_hits50.sum() / max(total_new, 1)),
        "new_hits@100": int(new_hits100.sum()),
        "new_recall@100": float(new_hits100.sum() / max(total_new, 1)),
    }


def paired_ablation_intervals(
    control_npz: Path | str,
    treatment_npz: Path | str,
    *,
    repeats: int = 2000,
) -> Dict[str, list[float]]:
    """Customer-clustered 95% bootstrap intervals for full-pool MLP metrics (treatment - control)."""
    a = np.load(control_npz)
    b = np.load(treatment_npz)
    if not np.array_equal(a["customers"], b["customers"]) or not np.array_equal(
        a["new_targets"], b["new_targets"]
    ):
        raise ValueError("Paired predictions require identical ordered validation rows.")

    _, groups = np.unique(a["customers"], return_inverse=True)
    n_cust = int(groups.max()) + 1
    rows = np.bincount(groups)
    new_targets = np.bincount(groups, weights=a["new_targets"])

    metric_specs = (
        ("recall@10_delta", "recall", rows),
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

    rng = np.random.default_rng(123)
    draws = rng.integers(n_cust, size=(repeats, n_cust))
    out: Dict[str, list[float]] = {}
    for out_name, _, denom_vec in metric_specs:
        denom = np.maximum(denom_vec[draws].sum(1), 1.0)
        vals = deltas[out_name][draws].sum(1) / denom
        out[out_name] = np.quantile(vals, [0.025, 0.5, 0.975]).tolist()
    return out
