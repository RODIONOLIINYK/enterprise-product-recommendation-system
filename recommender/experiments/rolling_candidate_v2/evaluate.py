"""Full-denominator evaluation, paired bootstrap CIs, and candidate-attention diagnostics."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from recommender.experiments.stage2_mlp_score.evaluate import paired_stage2_intervals
from recommender.train_ranker import RankerLoss

from .config import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED
from .data import make_rolling_slate_loader, rolling_ranker_batch_inputs
from .model import RollingCandidateAttentionRanker


def evaluate_rolling_slate_metrics(
    scores: torch.Tensor,
    data: Mapping[str, Any],
    indices: torch.Tensor,
    save_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Compute complete-denominator ranking and slate-coverage metrics on candidate slates."""
    scores_cpu = scores.detach().cpu()
    if indices.shape != scores_cpu.shape:
        raise ValueError(
            f"scores shape {tuple(scores_cpu.shape)} != indices shape {tuple(indices.shape)}."
        )

    order = scores_cpu.argsort(dim=-1, descending=True, stable=True)
    ranked = indices.gather(1, order)
    eligible_ranked = data["allowed"].gather(1, ranked)
    hits = (data["target"].gather(1, ranked) & eligible_ranked).numpy()
    novel = (data["new_target"].gather(1, ranked) & eligible_ranked).numpy()

    source = data["source"]
    count = np.array([len(x) for x in source["positive_pids"]], dtype=np.int64)
    new_count = np.array([len(x) for x in source["new_positive_pids"]], dtype=np.int64)
    repeat_count = count - new_count
    if bool((repeat_count < 0).any()):
        raise ValueError("Found negative repeat target count (new_positive_pids > positive_pids).")

    recall5 = hits[:, :5].sum(1) / count
    recall10 = hits[:, :10].sum(1) / count
    recall25 = hits[:, :25].sum(1) / count
    recall50 = hits[:, :50].sum(1) / count
    recall100 = hits[:, :100].sum(1) / count

    discount = 1.0 / np.log2(np.arange(10) + 2.0)
    ideal = (discount[None] * (np.arange(10)[None] < count[:, None])).sum(1)
    ndcg10 = (hits[:, :10] * discount).sum(1) / np.maximum(ideal, 1e-12)

    any_hit = hits.any(1)
    mrr = np.where(any_hit, 1.0 / (hits.argmax(1) + 1.0), 0.0)

    new_hits10 = novel[:, :10].sum(1)
    new_hits25 = novel[:, :25].sum(1)
    new_hits50 = novel[:, :50].sum(1)
    new_hits100 = novel[:, :100].sum(1)

    eligible_indices = data["allowed"].gather(1, indices)
    target_in_slate = data["target"].gather(1, indices) & eligible_indices
    new_in_slate = data["new_target"].gather(1, indices) & eligible_indices
    repeat_in_slate = target_in_slate & ~new_in_slate

    present = target_in_slate.sum(1).numpy()
    new_present = new_in_slate.sum(1).numpy()
    repeat_present = repeat_in_slate.sum(1).numpy()

    repeat_hits10_per_row = hits[:, :10].sum(1) - new_hits10
    total_targets = int(count.sum())
    total_new_targets = int(new_count.sum())
    total_repeat_targets = int(repeat_count.sum())

    result: Dict[str, Any] = {
        "queries": int(len(count)),
        "total_targets": total_targets,
        "repeat_targets": total_repeat_targets,
        "new_targets": total_new_targets,
        "recall@5": float(recall5.mean()),
        "recall@10": float(recall10.mean()),
        "recall@25": float(recall25.mean()),
        "recall@50": float(recall50.mean()),
        "recall@100": float(recall100.mean()),
        "ndcg@10": float(ndcg10.mean()),
        "mrr": float(mrr.mean()),
        "new_hits@10": int(new_hits10.sum()),
        "new_recall@10": float(new_hits10.sum() / max(total_new_targets, 1)),
        "new_hits@25": int(new_hits25.sum()),
        "new_recall@25": float(new_hits25.sum() / max(total_new_targets, 1)),
        "new_hits@50": int(new_hits50.sum()),
        "new_recall@50": float(new_hits50.sum() / max(total_new_targets, 1)),
        "new_hits@100": int(new_hits100.sum()),
        "new_recall@100": float(new_hits100.sum() / max(total_new_targets, 1)),
        "conditional_new_recall@10": float(new_hits10.sum() / max(int(new_present.sum()), 1)),
        "repeat_recall@10_micro": float(repeat_hits10_per_row.sum() / max(total_repeat_targets, 1)),
        "repeat_recall@10_macro": (
            float(np.mean(repeat_hits10_per_row[repeat_count > 0] / repeat_count[repeat_count > 0]))
            if bool((repeat_count > 0).any())
            else None
        ),
        "mixed_coverage_macro": float((present / count).mean()),
        "mixed_coverage_micro": float(present.sum() / max(total_targets, 1)),
        "total_targets_in_slate": int(present.sum()),
        "new_targets_in_slate": int(new_present.sum()),
        "new_slate_coverage": float(new_present.sum() / max(total_new_targets, 1)),
        "repeat_targets_in_slate": int(repeat_present.sum()),
        "repeat_slate_coverage": float(repeat_present.sum() / max(total_repeat_targets, 1)),
        "new_reachable_lost_at10": int(new_present.sum() - new_hits10.sum()),
    }

    cohorts: Dict[str, Any] = {}
    observed = source["attention_mask"].sum(1).numpy()
    for name, mask in (
        ("cold_0", observed == 0),
        ("light_1_8", (observed > 0) & (observed <= 8)),
        ("warm_9_31", (observed > 8) & (observed < 32)),
        ("truncated_32", observed == 32),
    ):
        cohorts[name] = {
            "rows": int(mask.sum()),
            "recall@10": float(recall10[mask].mean()) if mask.any() else None,
            "ndcg@10": float(ndcg10[mask].mean()) if mask.any() else None,
            "new_hits@10": int(new_hits10[mask].sum()),
            "new_targets": int(new_count[mask].sum()),
        }
    result["history_cohorts"] = cohorts

    if "prediction_days" in source:
        dates = np.datetime64("2025-01-01") + source["prediction_days"].numpy().astype(np.int64).astype("timedelta64[D]")
        months = dates.astype("datetime64[M]").astype(str)
        result["validation_months"] = {
            month: {
                "rows": int((months == month).sum()),
                "recall@10": float(recall10[months == month].mean()),
                "ndcg@10": float(ndcg10[months == month].mean()),
                "new_hits@10": int(new_hits10[months == month].sum()),
                "new_targets": int(new_count[months == month].sum()),
            }
            for month in sorted(set(months))
        }

    if "features" in data:
        target_rows, target_ids = torch.where(data["new_target"])
        values = np.array(data["features"][target_rows.numpy(), target_ids.numpy()])
        selected = ranked[:, :10]
        reached = (selected[target_rows] == target_ids[:, None]).any(1).numpy()
        rare = values[:, 26] <= np.log1p(5) / np.log1p(1000)
        first_family = values[:, 22] > 0.5
        family_known = data.get("family_known", torch.ones(data["target"].shape[1], dtype=torch.bool))[target_ids].numpy()
        sparse = values[:, 38] <= 0.1
        new_cohorts = {
            name: {"targets": int(mask.sum()), "hits@10": int(reached[mask].sum())}
            for name, mask in (
                ("first_snapshot_family", first_family & family_known),
                ("known_snapshot_family", ~first_family & family_known),
                ("unknown_snapshot_family_metadata", ~family_known),
                ("sku_rare_rolling90d", rare),
                ("sku_frequent_rolling90d", ~rare),
                ("sparse_transition", sparse),
                ("supported_transition", ~sparse),
            )
        }
        new_cohorts["unrepresented_catalogue"] = {
            "targets": int(total_new_targets - len(target_ids)),
            "hits@10": 0,
        }
        result["new_target_cohorts"] = new_cohorts

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            customers=np.array(source["customer_ids"]),
            recall5=recall5,
            recall=recall10,
            recall25=recall25,
            recall50=recall50,
            recall100=recall100,
            ndcg=ndcg10,
            mrr=mrr,
            new_hits=new_hits10,
            new_targets=new_count,
            new_present=new_present,
        )
    return result


@torch.no_grad()
def predict_rolling_ranker(
    model: RollingCandidateAttentionRanker,
    slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
    *,
    disable_candidate_attention: bool = False,
) -> Dict[str, Any]:
    """Run full-slate inference and return logits, validation loss, and attention weights summary."""
    model.eval()
    loss_fn = RankerLoss().to(device)
    logits_list = []
    token0_list = []
    hist_list = []
    losses = []
    for batch in make_rolling_slate_loader(slate, shuffle=False):
        inputs = rolling_ranker_batch_inputs(batch, product_vectors, device)
        out = model(**inputs, disable_candidate_attention=disable_candidate_attention)
        loss = loss_fn(out.logits, batch["labels"].to(device))["loss"]
        losses.append(float(loss.detach().cpu()))
        logits_list.append(out.logits.detach().cpu())
        if out.token_0_weight is not None:
            token0_list.append(out.token_0_weight.mean(dim=-1).detach().cpu())
        if out.history_weights is not None:
            hist_list.append(out.history_weights.mean(dim=-1).detach().cpu())
    return {
        "scores": torch.cat(logits_list, dim=0),
        "val_loss": float(np.mean(losses)),
        "avg_token0_weight": float(torch.cat(token0_list, dim=0).mean().item()) if token0_list else None,
        "avg_history_weight": float(torch.cat(hist_list, dim=0).mean().item()) if hist_list else None,
    }


def paired_rolling_intervals(
    control_npz: Union[Path, str],
    treatment_npz: Union[Path, str],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, Any]]:
    """Customer-clustered paired bootstrap 95% confidence intervals (treatment - control)."""
    return paired_stage2_intervals(control_npz, treatment_npz, repeats=repeats, seed=seed)
