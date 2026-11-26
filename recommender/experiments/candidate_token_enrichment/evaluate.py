"""Evaluation, mechanism sensitivity diagnostics, and trained-weight stability checks."""

from typing import Any, Dict, Mapping

import numpy as np
import torch

from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import (
    evaluate_rolling_slate_metrics,
    paired_rolling_intervals,
)
from recommender.ranker import RankerLoss

from .config import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED
from .model import CandidateTokenEnrichmentRanker


@torch.no_grad()
def predict_enrichment_ranker(
    model: CandidateTokenEnrichmentRanker,
    slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
    *,
    disable_candidate_attention: bool = False,
    disable_token_enrichment: bool = False,
    disable_candidate_enrichment: bool = False,
    disable_profile_enrichment: bool = False,
    disable_history_enrichment: bool = False,
) -> Dict[str, Any]:
    """Run full-slate inference and return logits, validation loss, and attention summary."""
    model.eval()
    loss_fn = RankerLoss().to(device)
    logits_list = []
    token0_list = []
    hist_list = []
    losses = []
    for batch in make_rolling_slate_loader(slate, shuffle=False):
        inputs = rolling_ranker_batch_inputs(batch, product_vectors, device)
        out = model(
            **inputs,
            disable_candidate_attention=disable_candidate_attention,
            disable_token_enrichment=disable_token_enrichment,
            disable_candidate_enrichment=disable_candidate_enrichment,
            disable_profile_enrichment=disable_profile_enrichment,
            disable_history_enrichment=disable_history_enrichment,
        )
        loss = loss_fn(out.logits, batch["labels"].to(device), candidate_mask=inputs.get("candidate_mask"))["loss"]
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


@torch.no_grad()
def _stream_representation_and_logit_diff(
    model: CandidateTokenEnrichmentRanker,
    slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
    *,
    disable_candidate_attention: bool = False,
    disable_token_enrichment: bool = False,
    disable_candidate_enrichment: bool = False,
    disable_profile_enrichment: bool = False,
    disable_history_enrichment: bool = False,
) -> Dict[str, Any]:
    """Compute exact full-validation representation and logit diffs under a branch ablation."""
    model.eval()
    max_logit = 0.0
    sum_logit = 0.0
    count_logit = 0

    max_zcand = 0.0
    sum_zcand = 0.0
    count_zcand = 0

    max_mem = 0.0
    sum_mem = 0.0
    count_mem = 0

    max_hatt = 0.0
    sum_hatt = 0.0
    count_hatt = 0

    max_attn = 0.0
    sum_attn = 0.0
    count_attn = 0

    ablated_scores = []
    for batch in make_rolling_slate_loader(slate, shuffle=False):
        inputs = rolling_ranker_batch_inputs(batch, product_vectors, device)
        ref = model(**inputs)
        alt = model(
            **inputs,
            disable_candidate_attention=disable_candidate_attention,
            disable_token_enrichment=disable_token_enrichment,
            disable_candidate_enrichment=disable_candidate_enrichment,
            disable_profile_enrichment=disable_profile_enrichment,
            disable_history_enrichment=disable_history_enrichment,
        )
        ablated_scores.append(alt.logits.detach().cpu())

        ld = (ref.logits - alt.logits).abs()
        max_logit = max(max_logit, float(ld.max().item()))
        sum_logit += float(ld.sum().item())
        count_logit += int(ld.numel())

        zd = (ref.z_candidate - alt.z_candidate).abs()
        max_zcand = max(max_zcand, float(zd.max().item()))
        sum_zcand += float(zd.sum().item())
        count_zcand += int(zd.numel())

        md = (ref.memory - alt.memory).abs()
        max_mem = max(max_mem, float(md.max().item()))
        sum_mem += float(md.sum().item())
        count_mem += int(md.numel())

        hd = (ref.h_attended - alt.h_attended).abs()
        max_hatt = max(max_hatt, float(hd.max().item()))
        sum_hatt += float(hd.sum().item())
        count_hatt += int(hd.numel())

        if ref.attn_weights is not None and alt.attn_weights is not None:
            ad = (ref.attn_weights - alt.attn_weights).abs()
            max_attn = max(max_attn, float(ad.max().item()))
            sum_attn += float(ad.sum().item())
            count_attn += int(ad.numel())

    return {
        "ablated_scores": torch.cat(ablated_scores, dim=0),
        "max_abs_logit_diff": max_logit,
        "mean_abs_logit_diff": sum_logit / max(count_logit, 1),
        "max_abs_z_candidate_diff": max_zcand,
        "mean_abs_z_candidate_diff": sum_zcand / max(count_zcand, 1),
        "max_abs_memory_diff": max_mem,
        "mean_abs_memory_diff": sum_mem / max(count_mem, 1),
        "max_abs_h_attended_diff": max_hatt,
        "mean_abs_h_attended_diff": sum_hatt / max(count_hatt, 1),
        "max_abs_attn_weights_diff": max_attn,
        "mean_abs_attn_weights_diff": sum_attn / max(count_attn, 1),
    }


@torch.no_grad()
def verify_trained_permutation_and_padding_stability(
    model: CandidateTokenEnrichmentRanker,
    slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
    *,
    batch_rows: int = 32,
) -> Dict[str, Any]:
    """Verify permutation equivariance and NaN-padding isolation on fitted NONZERO weights."""
    model.eval()
    rows = min(int(batch_rows), len(slate["customer_ids"]))
    batch = next(iter(make_rolling_slate_loader(slate, batch_size=rows, shuffle=False)))
    inputs = rolling_ranker_batch_inputs(batch, product_vectors, device)
    ref_out = model(**inputs)

    # 1. Candidate order reversal (permutation equivariance)
    rev_inputs = {
        k: (
            torch.flip(v, dims=[1])
            if k in ("pairwise_features", "candidate_vectors", "candidate_mask", "candidate_indices")
            else v
        )
        for k, v in inputs.items()
    }
    rev_out = model(**rev_inputs)
    perm_diff = float((torch.flip(rev_out.logits, dims=[1]) - ref_out.logits).abs().max().item())

    # 2. Append 2 NaN-filled masked candidates + corrupt padded history positions with NaN/bad IDs
    pad_inputs = dict(inputs)
    pad_inputs["pairwise_features"] = torch.cat(
        [inputs["pairwise_features"], torch.full((rows, 2, 41), float("nan"), device=device)],
        dim=1,
    )
    pad_inputs["candidate_vectors"] = torch.cat(
        [inputs["candidate_vectors"], torch.full((rows, 2, 128), float("nan"), device=device)],
        dim=1,
    )
    pad_inputs["candidate_mask"] = torch.cat(
        [inputs["candidate_mask"], torch.zeros((rows, 2), dtype=torch.bool, device=device)],
        dim=1,
    )

    if inputs.get("candidate_indices") is not None:
        pad_inputs["candidate_indices"] = torch.cat(
            [inputs["candidate_indices"], torch.full((rows, 2), -999, dtype=torch.long, device=device)], dim=1)

    hist_mask = inputs["attention_mask"] != 0
    if bool((~hist_mask).any()):
        seq_nan = inputs["sequence"].clone()
        seq_nan[~hist_mask] = float("nan")
        dates_nan = inputs["seq_purchase_dates"].clone()
        dates_nan[~hist_mask] = float("nan")
        pids_bad = inputs["seq_product_ids"].clone()
        pids_bad[~hist_mask] = 999999
        pad_inputs["sequence"] = seq_nan
        pad_inputs["seq_purchase_dates"] = dates_nan
        pad_inputs["seq_product_ids"] = pids_bad

    pad_out = model(**pad_inputs)
    pad_valid_diff = float((pad_out.logits[:, :-2] - ref_out.logits).abs().max().item())
    pad_masked_neginf = bool(torch.all(pad_out.logits[:, -2:] == -torch.inf))
    return {
        "rows_checked": rows,
        "max_abs_logit_diff_reversed_slate": perm_diff,
        "max_abs_logit_diff_with_nan_cand_and_hist_padding": pad_valid_diff,
        "masked_pad_logit_neginf": pad_masked_neginf,
    }


@torch.no_grad()
def measure_mechanism_diagnostics(
    model: CandidateTokenEnrichmentRanker,
    slate: Mapping[str, Any],
    val_data: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
    *,
    baseline_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run mechanism sensitivity ablations (no retraining) and stability checks on fitted model."""
    out: Dict[str, Any] = {}
    cand_indices = slate["candidate_indices"]

    def _summarize(tag: str, **disable_flags) -> Dict[str, Any]:
        diff = _stream_representation_and_logit_diff(
            model, slate, product_vectors, device, **disable_flags
        )
        alt_metrics = evaluate_rolling_slate_metrics(diff.pop("ablated_scores"), val_data, cand_indices)
        return {
            **diff,
            "ablated_recall@10": float(alt_metrics["recall@10"]),
            "ablated_ndcg@10": float(alt_metrics["ndcg@10"]),
            "ablated_new_hits@10": int(alt_metrics["new_hits@10"]),
            "recall@10_delta_vs_intact": float(alt_metrics["recall@10"] - baseline_metrics["recall@10"]),
            "ndcg@10_delta_vs_intact": float(alt_metrics["ndcg@10"] - baseline_metrics["ndcg@10"]),
            "new_hits@10_delta_vs_intact": int(alt_metrics["new_hits@10"] - baseline_metrics["new_hits@10"]),
        }

    if model.candidate_attention_enabled:
        out["disable_candidate_attention_only"] = _summarize(
            "disable_candidate_attention",
            disable_candidate_attention=True,
        )

    if model.token_enrichment_enabled:
        out["disable_all_token_enrichment"] = _summarize(
            "disable_all_token_enrichment",
            disable_token_enrichment=True,
        )
        out["disable_candidate_enrichment_only"] = _summarize(
            "disable_candidate_enrichment",
            disable_candidate_enrichment=True,
        )
        out["disable_profile_enrichment_only"] = _summarize(
            "disable_profile_enrichment",
            disable_profile_enrichment=True,
        )
        out["disable_history_enrichment_only"] = _summarize(
            "disable_history_enrichment",
            disable_history_enrichment=True,
        )
        out["enricher_weight_norms"] = {
            name: {
                "final_weight_l2": float(mod.net[-1].weight.detach().norm().item()),
                "final_bias_l2": float(mod.net[-1].bias.detach().norm().item()),
                "first_weight_l2": float(mod.net[0].weight.detach().norm().item()),
                "stats_cols_weight_l2": float(mod.net[0].weight[:, 128:].detach().norm().item()),
            }
            for name, mod in (
                ("candidate_enricher", model.base.candidate_enricher),
                ("profile_enricher", model.base.profile_enricher),
                ("history_enricher", model.base.history_enricher),
            )
            if mod is not None
        }

    out["trained_stability"] = verify_trained_permutation_and_padding_stability(
        model, slate, product_vectors, device
    )
    return out


def compute_three_arm_paired_intervals(
    arm_a_npz: str,
    arm_b_npz: str,
    arm_c_npz: str,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Compute customer-clustered 95% bootstrap CIs for A->B, B->C, and A->C."""
    return {
        "A_to_B": paired_rolling_intervals(arm_a_npz, arm_b_npz, repeats=repeats, seed=seed),
        "B_to_C": paired_rolling_intervals(arm_b_npz, arm_c_npz, repeats=repeats, seed=seed),
        "A_to_C": paired_rolling_intervals(arm_a_npz, arm_c_npz, repeats=repeats, seed=seed),
    }
