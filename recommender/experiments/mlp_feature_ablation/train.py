"""Training, replay verification, comparison summary, and reporting for Stage 1 MLP feature ablation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.train import finish_run, run_directory
from recommender.shared_data import atomic_torch_save
from recommender.train_ranker import checkpoint_digest
from recommender.train_retriever import get_device

from .config import (
    ARTIFACT_ROOT,
    AUDIT_ARTIFACTS_PATH,
    BASELINE_EVAL_DIR,
    BATCH_SIZE,
    DYNAMIC_CACHE_DIR,
    EPOCHS,
    EVIDENCE_ROOT,
    LR,
    OMITTED_CONTEXT_FEATURES,
    POS_WEIGHT,
    PRIOR_300_EVIDENCE_DIR,
    PRIOR_300_RUN_DIR,
    REFERENCE_MANIFEST,
    REFERENCE_POINTER,
    REFERENCE_RETRIEVER,
    REFERENCE_SELECTOR,
    SEED,
    UPSTREAM_VARIANT_NAME,
    VARIANT_NAME,
    WEIGHT_DECAY,
)
from .data import load_verified_baseline_val_scores, load_verified_no_brand_campaign_data
from .evaluate import evaluate_full_pool_metrics, paired_feature_ablation_intervals
from .model import FeatureAblatedStage1Selector, load_ablated_selector_checkpoint
from .schema import (
    FeatureAblationSpec,
    build_feature_ablation_spec,
    build_input_contract,
    describe_retained_features,
    select_retained_context_features,
)


def snapshot_preserved_hashes(*, include_prior_300: bool = False) -> Dict[str, str]:
    """Capture SHA-256 digests of all 7 audit/artifacts.json files plus reference pointer/pair/manifest."""
    audit_entries = json.loads(AUDIT_ARTIFACTS_PATH.read_text(encoding="utf-8"))
    paths = [
        *audit_entries.keys(),
        str(REFERENCE_POINTER),
        str(REFERENCE_MANIFEST),
        str(REFERENCE_RETRIEVER),
        str(REFERENCE_SELECTOR),
        str(DYNAMIC_CACHE_DIR / "manifest.json"),
        str(BASELINE_EVAL_DIR / "evaluation.json"),
        str(BASELINE_EVAL_DIR / "predictions.npz"),
    ]
    if include_prior_300:
        paths.extend(
            [
                str(PRIOR_300_RUN_DIR / "best.pt"),
                str(PRIOR_300_RUN_DIR / "result.json"),
                str(PRIOR_300_RUN_DIR / "predictions.npz"),
                str(PRIOR_300_RUN_DIR / "val_scores.pt"),
                str(PRIOR_300_EVIDENCE_DIR / "summary.json"),
                str(PRIOR_300_EVIDENCE_DIR / "REPORT.md"),
                str(PRIOR_300_EVIDENCE_DIR / "baseline_predictions.npz"),
            ]
        )
    return {p: file_digest(p) for p in paths}


def _score_ablated_selector_split(
    model: FeatureAblatedStage1Selector,
    split_data: Mapping[str, Any],
    dev: torch.device,
    *,
    batch_size: int = 128,
    zero_context_38: bool = False,
    zero_retained_context: Optional[bool] = None,
    zero_customer_stats_6: bool = False,
) -> torch.Tensor:
    """Score all eligible products using canonical 41-col memmap (model slices retained cols before concat)."""
    zero_ctx = zero_context_38 if zero_retained_context is None else bool(zero_retained_context)
    model.eval()
    v_dev = split_data["v"].to(dev)
    u_all = split_data["u"]
    stats_all = split_data["customer_stats_6"]
    allowed_all = split_data["allowed"]
    feat_mmap = split_data["features"]
    if feat_mmap.shape[0] != len(u_all):
        raise ValueError(
            f"Dynamic feature memmap row count ({feat_mmap.shape[0]}) != split cohort size ({len(u_all)})."
        )

    chunks = []
    with torch.no_grad():
        for start in range(0, len(u_all), batch_size):
            end = min(start + batch_size, len(u_all))
            f41 = torch.from_numpy(np.array(feat_mmap[start:end], dtype=np.float32)).to(dev)
            stats = stats_all[start:end].to(dev)
            if zero_customer_stats_6:
                stats = torch.zeros_like(stats)
            if zero_ctx:
                f_retained_zero = torch.zeros_like(model.select_context(f41))
                logits = model(
                    u=u_all[start:end].to(dev),
                    v=v_dev,
                    retained_context=f_retained_zero,
                    customer_stats_6=stats,
                )
            else:
                logits = model(
                    u=u_all[start:end].to(dev),
                    v=v_dev,
                    features_41=f41,
                    customer_stats_6=stats,
                )
            masked = logits.masked_fill(~allowed_all[start:end].to(dev), -1e9)
            chunks.append(masked.cpu())
    return torch.cat(chunks, dim=0)


def _verify_runtime_omission_invariance(
    model: FeatureAblatedStage1Selector,
    split_data: Mapping[str, Any],
    dev: torch.device,
    *,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Empirically verify exact logit invariance to omitted columns and non-zero dependence on retained columns."""
    model.eval()
    n = min(batch_size, len(split_data["u"]))
    u_b = split_data["u"][:n].to(dev)
    v_b = split_data["v"].to(dev)
    stats_b = split_data["customer_stats_6"][:n].to(dev)
    f41_b = torch.from_numpy(np.array(split_data["features"][:n], dtype=np.float32)).to(dev)

    with torch.no_grad():
        base_logits = model(u=u_b, v=v_b, features_41=f41_b, customer_stats_6=stats_b)

        f41_omitted_perturbed = f41_b.clone()
        for col_idx in model.spec.omitted_indices:
            f41_omitted_perturbed[..., col_idx] = 999.0 if col_idx % 2 == 0 else -999.0
        omitted_logits = model(
            u=u_b, v=v_b, features_41=f41_omitted_perturbed, customer_stats_6=stats_b
        )
        omission_max_abs_diff = float((base_logits - omitted_logits).abs().max().item())

        f41_retained_perturbed = f41_b.clone()
        for col_idx in model.spec.retained_indices:
            f41_retained_perturbed[..., col_idx] = 0.0
        retained_logits = model(
            u=u_b, v=v_b, features_41=f41_retained_perturbed, customer_stats_6=stats_b
        )
        retained_max_abs_diff = float((base_logits - retained_logits).abs().max().item())

    if omission_max_abs_diff != 0.0:
        raise AssertionError(
            f"Omission invariance check failed: max_abs_diff={omission_max_abs_diff} != 0.0"
        )
    if retained_max_abs_diff <= 0.0:
        raise AssertionError(
            f"Retained feature dependence check failed: max_abs_diff={retained_max_abs_diff}"
        )
    return {
        "omission_invariance_max_abs_diff": omission_max_abs_diff,
        "retained_zero_max_abs_diff": retained_max_abs_diff,
    }


def measure_repeat_flag_influence(
    model: FeatureAblatedStage1Selector,
    split_data: Mapping[str, Any],
    dev: torch.device,
    *,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Measure selected-model logit sensitivity to retained `is_repeat_sku` across eligible validation pairs."""
    if "is_repeat_sku" not in model.spec.retained_names:
        raise ValueError("measure_repeat_flag_influence requires 'is_repeat_sku' in retained_names.")
    repeat_src_idx = model.spec.source_names.index("is_repeat_sku")

    model.eval()
    v_dev = split_data["v"].to(dev)
    u_all = split_data["u"]
    stats_all = split_data["customer_stats_6"]
    allowed_all = split_data["allowed"]
    feat_mmap = split_data["features"]

    max_abs_flip = 0.0
    max_abs_vs_base = 0.0
    sum_abs_flip = 0.0
    sum_signed_flip = 0.0
    total_eligible = 0

    with torch.no_grad():
        for start in range(0, len(u_all), batch_size):
            end = min(start + batch_size, len(u_all))
            f41 = torch.from_numpy(np.array(feat_mmap[start:end], dtype=np.float32)).to(dev)
            u_b = u_all[start:end].to(dev)
            stats_b = stats_all[start:end].to(dev)
            allowed_b = allowed_all[start:end].to(dev)

            base_logits = model(u=u_b, v=v_dev, features_41=f41, customer_stats_6=stats_b)
            f41_zero = f41.clone()
            f41_zero[..., repeat_src_idx] = 0.0
            logits_0 = model(u=u_b, v=v_dev, features_41=f41_zero, customer_stats_6=stats_b)

            f41_one = f41.clone()
            f41_one[..., repeat_src_idx] = 1.0
            logits_1 = model(u=u_b, v=v_dev, features_41=f41_one, customer_stats_6=stats_b)

            diff_1_minus_0 = (logits_1 - logits_0)[allowed_b]
            diff_vs_base = (base_logits - logits_0)[allowed_b]
            max_abs_flip = max(max_abs_flip, float(diff_1_minus_0.abs().max().item()))
            max_abs_vs_base = max(max_abs_vs_base, float(diff_vs_base.abs().max().item()))
            sum_abs_flip += float(diff_1_minus_0.abs().sum().item())
            sum_signed_flip += float(diff_1_minus_0.sum().item())
            total_eligible += int(allowed_b.sum().item())

    if max_abs_flip <= 0.0:
        raise AssertionError(
            f"Retained is_repeat_sku dependence check failed: max_abs_flip={max_abs_flip} <= 0.0"
        )
    return {
        "feature_name": "is_repeat_sku",
        "source_context_index": int(repeat_src_idx),
        "retained_context_index": int(model.spec.retained_names.index("is_repeat_sku")),
        "eligible_pairs_checked": int(total_eligible),
        "max_abs_logit_diff_1_vs_0": float(max_abs_flip),
        "mean_abs_logit_diff_1_vs_0": float(sum_abs_flip / max(total_eligible, 1)),
        "mean_signed_logit_diff_1_minus_0": float(sum_signed_flip / max(total_eligible, 1)),
        "max_abs_logit_diff_actual_vs_0": float(max_abs_vs_base),
    }


def fit_ablated_stage1_selector(
    spec: FeatureAblationSpec,
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    artifact_root: Path = ARTIFACT_ROOT,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    pos_weight: float = POS_WEIGHT,
    device: Optional[str] = None,
    stage_name: str = "stage1_compact_selector_feature_ablation_v1",
    run_zero_ablations: bool = True,
    check_repeat_flag_influence: bool = False,
) -> Dict[str, Any]:
    """Train a FeatureAblatedStage1Selector from scratch under the exact Stage 1 MLP protocol."""
    dev = get_device(device)
    input_contract = build_input_contract(spec)
    ref_hashes = meta["reference"]["hashes"]
    run_name = f"selector-{UPSTREAM_VARIANT_NAME}-{spec.name}-{spec.recipe}-seed{seed}"

    settings = {
        "stage": stage_name,
        "variant_name": spec.name,
        "upstream_variant": UPSTREAM_VARIANT_NAME,
        "upstream_reference_id": meta["reference"]["reference_pointer"]["active_reference_id"],
        "upstream_digest": ref_hashes["best_retriever_pt"],
        "baseline_selector_digest": ref_hashes["best_selector_pt"],
        "recipe": spec.recipe,
        "input_contract": input_contract,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "pos_weight": float(pos_weight),
        "loss": "BCEWithLogitsLoss_eligible_pool",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR(eta_min=1e-5)",
        "grad_clip_norm": 1.0,
        "selection": "new_hits@50, then ndcg@10, then recall@10",
        "dynamic_identity": meta["dynamic_identity"],
        "data_identity": meta["data_identity"],
        "enrichment_identity": meta["enrichment_identity"],
        "metadata_policy": "snapshot_unverified",
        "heldout_access": False,
    }

    directory, existing = run_directory(artifact_root, run_name, settings)
    if existing is not None:
        if not (directory / "val_scores.pt").is_file() or not (directory / "predictions.npz").is_file():
            raise FileNotFoundError(
                f"Completed run {directory} is missing val_scores.pt or predictions.npz."
            )
        return existing

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FeatureAblatedStage1Selector(spec).to(dev)
    if model.head[0].in_features != spec.input_dim:
        raise AssertionError(
            f"Expected {spec.input_dim} input weights, got {model.head[0].in_features}."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    tr = campaign_data["train"]
    va = campaign_data["val"]
    v_dev = tr["v"].to(dev)
    tr_feat_mmap = tr["features"]
    generator = torch.Generator().manual_seed(seed)
    pos_weight_tensor = torch.tensor(float(pos_weight), device=dev)

    best = (-1, -1.0, -1.0)
    history = []
    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        perm = torch.randperm(len(tr["u"]), generator=generator)
        for batch_rows in perm.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            f41 = torch.from_numpy(
                np.array(tr_feat_mmap[batch_rows.numpy()], dtype=np.float32)
            ).to(dev)
            stats = tr["customer_stats_6"][batch_rows].to(dev)
            logits = model(
                u=tr["u"][batch_rows].to(dev),
                v=v_dev,
                features_41=f41,
                customer_stats_6=stats,
            )
            allowed = tr["allowed"][batch_rows].to(dev)
            target = tr["target"][batch_rows].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(
                logits[allowed], target[allowed], pos_weight=pos_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        val_scores = _score_ablated_selector_split(model, va, dev)
        metrics = evaluate_full_pool_metrics(val_scores, va)
        key = (metrics["new_hits@50"], metrics["ndcg@10"], metrics["recall@10"])
        ep_sec = time.perf_counter() - ep_t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": ep_sec,
                "metrics": metrics,
            }
        )
        if key > best:
            best = key
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "input_contract": input_contract,
                    "model_state_dict": model.state_dict(),
                },
                directory / "best.pt",
            )
        print(
            f"selector [{spec.name}/{spec.recipe}] seed={seed} epoch={epoch:02d} "
            f"loss={np.mean(losses):.5f} rec5={metrics['recall@5']:.5f} rec10={metrics['recall@10']:.5f} "
            f"ndcg10={metrics['ndcg@10']:.5f} new25={metrics['new_hits@25']} new50={metrics['new_hits@50']} "
            f"new100={metrics['new_hits@100']} ({ep_sec:.2f}s)",
            flush=True,
        )
    sync(dev)
    elapsed = time.perf_counter() - started

    # Strict checkpoint reload & inference replay verification.
    reloaded_model, checkpoint = load_ablated_selector_checkpoint(
        directory / "best.pt",
        expected_spec=spec,
        map_location="cpu",
    )
    reloaded_model = reloaded_model.to(dev)
    reloaded_model.eval()

    sync(dev)
    inf_t0 = time.perf_counter()
    val_scores = _score_ablated_selector_split(reloaded_model, va, dev)
    sync(dev)
    val_inf_seconds = time.perf_counter() - inf_t0

    pred_path = directory / "predictions.npz"
    replayed = evaluate_full_pool_metrics(val_scores, va, save_predictions=pred_path)
    if replayed != checkpoint["metrics"]:
        raise AssertionError(f"Selected-weight replay mismatch for {run_name}")

    invariance_checks = _verify_runtime_omission_invariance(reloaded_model, va, dev)
    repeat_influence: Optional[Dict[str, Any]] = None
    if check_repeat_flag_influence and "is_repeat_sku" in spec.retained_names:
        repeat_influence = measure_repeat_flag_influence(reloaded_model, va, dev)

    train_scores = _score_ablated_selector_split(reloaded_model, tr, dev)
    best_digest = checkpoint_digest(directory / "best.pt")

    atomic_torch_save(
        {
            "selector_digest": best_digest,
            "upstream_digest": ref_hashes["best_retriever_pt"],
            "data_identity": info["identity"],
            "input_contract": input_contract,
            "scores": train_scores,
        },
        directory / "train_scores.pt",
    )
    atomic_torch_save(
        {
            "selector_digest": best_digest,
            "upstream_digest": ref_hashes["best_retriever_pt"],
            "data_identity": info["identity"],
            "input_contract": input_contract,
            "scores": val_scores,
        },
        directory / "val_scores.pt",
    )

    ablations: Optional[Dict[str, Any]] = None
    if run_zero_ablations:
        ablations = {"full": replayed}
        for abl_name, zero_ctx, zero_s6 in (
            (f"zero_context_{spec.num_retained_features}", True, False),
            (f"zero_customer_stats_{spec.num_customer_stats}", False, True),
            (f"zero_both_{spec.scalar_feature_count}", True, True),
        ):
            abl_scores = _score_ablated_selector_split(
                reloaded_model,
                va,
                dev,
                zero_retained_context=zero_ctx,
                zero_customer_stats_6=zero_s6,
            )
            ablations[abl_name] = evaluate_full_pool_metrics(abl_scores, va)

    result = {
        "settings": settings,
        "selected_epoch": int(checkpoint["epoch"]),
        "metrics": replayed,
        "invariance_checks": invariance_checks,
        "repeat_flag_influence": repeat_influence,
        "zero_ablations": ablations,
        "parameters": int(sum(p.numel() for p in reloaded_model.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in reloaded_model.parameters() if p.requires_grad)
        ),
        "first_layer_in_features": int(reloaded_model.head[0].in_features),
        "retained_context_dim": int(spec.num_retained_features),
        "scalar_feature_count": int(spec.scalar_feature_count),
        "training_seconds": float(elapsed),
        "val_inference_seconds": float(val_inf_seconds),
        "device": str(dev),
        "train_cohort_rows": int(len(tr["u"])),
        "val_cohort_rows": int(len(va["u"])),
        "selected_checkpoint": str(directory / "best.pt"),
        "selected_checkpoint_sha256": best_digest,
        "train_scores_path": str(directory / "train_scores.pt"),
        "val_scores_path": str(directory / "val_scores.pt"),
        "predictions_path": str(pred_path),
        "cache_lineage": meta["cache_lineage"],
    }
    atomic_json(history, directory / "history.json", overwrite=False)
    finish_run(directory, settings, result)
    return result


def _format_report_md(summary: Mapping[str, Any]) -> str:
    base = summary["baseline_303"]["metrics"]
    var = summary["variant_300"]["metrics"]
    cis = summary["paired_bootstrap_95_ci"]
    variant_meta = summary["variant_300"]
    spec_info = summary["feature_ablation_spec"]
    hashes_ok = summary["hash_preservation"]["all_matched"]

    def _pct(x: float) -> str:
        return f"{x * 100.0:.3f}%"

    def _ci_pp(key: str) -> str:
        lo, med, hi = cis[key]
        return f"{med * 100.0:+.3f} pp [{lo * 100.0:+.3f}, {hi * 100.0:+.3f}]"

    def _ci_val(key: str) -> str:
        lo, med, hi = cis[key]
        return f"{med:+.5f} [{lo:+.5f}, {hi:+.5f}]"

    lines = [
        "# Stage 1 MLP Feature-Removal Experiment (`no_repeat_no_month` / `uv_44`)",
        "",
        "## 1. Experiment Scope & Controlled Setup",
        "",
        f"- **Baseline Reference:** `{summary['reference']['reference_id']}` (`CompactStage1Selector`, `uv_47`, `303D = 128 + 128 + 41 + 6`, `47,617` params, selected epoch `1/10`, checkpoint SHA-256 `{summary['baseline_303']['selector_sha256']}`)",
        f"- **Treatment Variant:** `{spec_info['name']}` (`FeatureAblatedStage1Selector`, `{spec_info['recipe']}`, `300D = 128 + 128 + 38 + 6`, `{variant_meta['parameters']:,}` params, selected epoch `{variant_meta['selected_epoch']}/10`, checkpoint SHA-256 `{variant_meta['selected_checkpoint_sha256']}`)",
        "- **Frozen Upstream Towers:** `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt` (epoch 16, SHA-256 `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981`); fixed saved `u/v` purchase caches reused read-only with zero tower re-encoding.",
        f"- **Omitted Contextual Features (3 of 41, resolved by schema name):** `0: is_repeat_sku`, `31: month_sin`, `32: month_cos` (`38` retained context features + `6` customer stats = `44` scalar inputs; `head.0` physically `Linear(300, 128)`).",
        "- **Training Protocol:** `seed=42`, `10` epochs, `AdamW(lr=0.001, weight_decay=0.0001)`, `CosineAnnealingLR(eta_min=1e-5)`, `batch_size=64`, `BCEWithLogitsLoss(pos_weight=20.0)` over full eligible pool, `clip_grad_norm_(1.0)`, selection `(new_hits@50, ndcg@10, recall@10)`.",
        "- **Data Safety & Denominators:** Train `< 2025-10-01` (`26,253` queries with >=100 eligible SKUs); Val `2025-10-01..2026-02-18` (`6,191` queries, `1,857` new-to-customer targets, `10,276` total targets, `841` SKU universe, `735–771` eligible/query). Held-out split `>= 2026-02-19` untouched (`heldout_access=false`). No Stage 2 execution, no model promotion.",
        "",
        "## 2. Validation Comparison (All 6,191 Queries / 1,857 New Targets)",
        "",
        "> [!NOTE]",
        "> Confidence intervals below are paired **customer-clustered 95% bootstrap intervals** (`2,000` draws, seed `123`) over the `6,191` validation queries at selected weights for a single seed (`seed=42`). They measure query/customer sampling uncertainty conditional on these selected weights, not multi-seed training variance.",
        "",
        "### Overall Ranking Metrics (All Eligible Candidates, Repeat + New, No Unseen Filtering)",
        "",
        "| Metric | Baseline `303D` (`uv_47`, Ep 1) | Variant `300D` (`uv_44`, Ep "
        + f"{variant_meta['selected_epoch']}) | Delta | 95% Customer-Clustered Bootstrap CI |",
        "|---|---:|---:|---:|---:|",
        f"| Overall Macro `Recall@5` | {_pct(base['recall@5'])} (`{base['recall@5']:.6f}`) | {_pct(var['recall@5'])} (`{var['recall@5']:.6f}`) | `{(var['recall@5'] - base['recall@5']) * 100.0:+.3f} pp` | `{_ci_pp('recall@5_delta')}` |",
        f"| Overall Macro `Recall@10` | {_pct(base['recall@10'])} (`{base['recall@10']:.6f}`) | {_pct(var['recall@10'])} (`{var['recall@10']:.6f}`) | `{(var['recall@10'] - base['recall@10']) * 100.0:+.3f} pp` | `{_ci_pp('recall@10_delta')}` |",
        f"| Overall Macro `Recall@25` | {_pct(base['recall@25'])} (`{base['recall@25']:.6f}`) | {_pct(var['recall@25'])} (`{var['recall@25']:.6f}`) | `{(var['recall@25'] - base['recall@25']) * 100.0:+.3f} pp` | `{_ci_pp('recall@25_delta')}` |",
        f"| Overall Macro `Recall@50` | {_pct(base['recall@50'])} (`{base['recall@50']:.6f}`) | {_pct(var['recall@50'])} (`{var['recall@50']:.6f}`) | `{(var['recall@50'] - base['recall@50']) * 100.0:+.3f} pp` | `{_ci_pp('recall@50_delta')}` |",
        f"| Overall Macro `Recall@100` | {_pct(base['recall@100'])} (`{base['recall@100']:.6f}`) | {_pct(var['recall@100'])} (`{var['recall@100']:.6f}`) | `{(var['recall@100'] - base['recall@100']) * 100.0:+.3f} pp` | `{_ci_pp('recall@100_delta')}` |",
        f"| Overall `NDCG@10` | `{base['ndcg@10']:.6f}` | `{var['ndcg@10']:.6f}` | `{var['ndcg@10'] - base['ndcg@10']:+.6f}` | `{_ci_val('ndcg@10_delta')}` |",
        f"| Overall `MRR` | `{base['mrr']:.6f}` | `{var['mrr']:.6f}` | `{var['mrr'] - base['mrr']:+.6f}` | `—` |",
        "",
        "### New-Target Discovery Metrics (`1,857` New-to-Customer Target Instances)",
        "",
        "| Metric | Ranking Pool | Baseline `303D` (`uv_47`) | Variant `300D` (`uv_44`) | Delta | 95% Customer-Clustered Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|",
        f"| `new_hits@10` / `new_recall@10` | Ordinary Top-10 (all eligible) | `{base['new_hits@10']} / 1,857` ({_pct(base['new_recall@10'])}) | `{var['new_hits@10']} / 1,857` ({_pct(var['new_recall@10'])}) | `{var['new_hits@10'] - base['new_hits@10']:+d}` (`{(var['new_recall@10'] - base['new_recall@10']) * 100.0:+.3f} pp`) | `{_ci_pp('new_recall@10_delta')}` |",
        f"| `new_hits@25` / `new_recall@25` | Unseen-only | `{base['new_hits@25']} / 1,857` ({_pct(base['new_recall@25'])}) | `{var['new_hits@25']} / 1,857` ({_pct(var['new_recall@25'])}) | `{var['new_hits@25'] - base['new_hits@25']:+d}` (`{(var['new_recall@25'] - base['new_recall@25']) * 100.0:+.3f} pp`) | `{_ci_pp('new_recall@25_delta')}` |",
        f"| `new_hits@50` / `new_recall@50` (selection key) | Unseen-only | `{base['new_hits@50']} / 1,857` ({_pct(base['new_recall@50'])}) | `{var['new_hits@50']} / 1,857` ({_pct(var['new_recall@50'])}) | `{var['new_hits@50'] - base['new_hits@50']:+d}` (`{(var['new_recall@50'] - base['new_recall@50']) * 100.0:+.3f} pp`) | `{_ci_pp('new_recall@50_delta')}` |",
        f"| `new_hits@100` / `new_recall@100` | Unseen-only | `{base['new_hits@100']} / 1,857` ({_pct(base['new_recall@100'])}) | `{var['new_hits@100']} / 1,857` ({_pct(var['new_recall@100'])}) | `{var['new_hits@100'] - base['new_hits@100']:+d}` (`{(var['new_recall@100'] - base['new_recall@100']) * 100.0:+.3f} pp`) | `{_ci_pp('new_recall@100_delta')}` |",
        "",
        "## 3. Per-Epoch History (`300D` Variant, `seed=42`)",
        "",
        "| Epoch | Train Loss | Overall `Recall@5` | Overall `Recall@10` | Overall `Recall@25` | Overall `Recall@50` | Overall `Recall@100` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` | Selected |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["variant_300"]["history"]:
        m = row["metrics"]
        sel = "YES" if row["epoch"] == variant_meta["selected_epoch"] else ""
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.5f} | {m['recall@5']:.5f} | {m['recall@10']:.5f} | "
            f"{m['recall@25']:.5f} | {m['recall@50']:.5f} | {m['recall@100']:.5f} | {m['ndcg@10']:.5f} | "
            f"{m['new_hits@10']} | {m['new_hits@25']} | {m['new_hits@50']} | {m['new_hits@100']} | {sel} |"
        )

    lines.extend(
        [
            "",
            "## 4. Training Cost, Invariance Checks & Preserved Hashes",
            "",
            f"- **Device:** `{variant_meta['device']}`",
            f"- **Training time (10 epochs, 26,253 rows):** `{variant_meta['training_seconds']:.2f}s`",
            f"- **Validation replay inference time (6,191 rows x 841 SKUs):** `{variant_meta['val_inference_seconds']:.3f}s`",
            f"- **Parameters:** `{variant_meta['parameters']:,}` (`first_layer_in_features={variant_meta['first_layer_in_features']}`, reduced by `384 = 3 * 128` weights vs baseline `47,617`)",
            f"- **Omission invariance max abs diff (perturbing cols 0, 31, 32):** `{variant_meta['invariance_checks']['omission_invariance_max_abs_diff']}`",
            f"- **Retained feature sensitivity max abs diff (zeroing retained 38 cols):** `{variant_meta['invariance_checks']['retained_zero_max_abs_diff']:.6f}`",
            f"- **Preserved artifact/audit/reference hashes before & after:** `{'ALL MATCHED' if hashes_ok else 'MISMATCH'}`",
            "",
            "## 5. Retained Feature Contract (`38` Context + `6` Customer Stats = `44` Scalars, `300D` Total Input)",
            "",
            "Remaining scales are unchanged from `no-brand-v1`: `u` (`128D`) and `v` (`128D`) are L2-normalized signed vectors; `customer_stats_6` are in `[0, 1]`; of the `38` retained contextual features, `30` are in `[0, 1]` and `8` remain signed in `[-1, 1]` (`sku_growth_30d`, `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`, `line_last_lift`, `hierarchical_backoff`, `same_basket_lift`).",
            "",
            "| Retained Context Index | Source Cache Index | MLP Column | Feature Name |",
            "|---:|---:|---:|---|",
        ]
    )
    for item in summary["retained_feature_table"]:
        if item["block"] == "customer_stats_6":
            continue
        lines.append(
            f"| {item['ablated_context_index']} | {item['source_context_index']} | {item['mlp_column']} | `{item['name']}` |"
        )
    lines.extend(
        [
            "",
            "Plus `customer_stats_6` at MLP columns `294..299`: `distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`.",
            "",
            "## 6. Reproduction Commands",
            "",
            "```bash",
            "./venv/bin/python -m recommender.experiments.mlp_feature_ablation.train",
            "PYTHONPATH=. ./venv/bin/pytest tests/",
            "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_feature_ablation_experiment(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the controlled 300D MLP feature-removal experiment and paired evaluation."""
    hashes_before = snapshot_preserved_hashes()
    spec = build_feature_ablation_spec(OMITTED_CONTEXT_FEATURES, name=VARIANT_NAME)
    if spec.input_dim != 300 or spec.num_retained_features != 38 or spec.scalar_feature_count != 44:
        raise AssertionError(f"Unexpected spec dimensions: {asdict(spec)}")

    campaign_data, info, meta = load_verified_no_brand_campaign_data()
    va = campaign_data["val"]

    # 1. Baseline 303D evaluation directly from verified saved val_scores.pt (no extra training).
    baseline_scores, baseline_meta = load_verified_baseline_val_scores(info["identity"])
    evidence_dir = Path(evidence_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    baseline_npz = evidence_dir / "baseline_predictions.npz"
    baseline_metrics = evaluate_full_pool_metrics(
        baseline_scores, va, save_predictions=baseline_npz
    )

    ref_eval = baseline_meta["baseline_eval_json"]
    if ref_eval is not None:
        ref_m = ref_eval["metrics"]
        for k in ("new_hits@10", "new_hits@25", "new_hits@50", "new_hits@100", "queries", "new_targets"):
            if baseline_metrics[k] != ref_m[k]:
                raise AssertionError(f"Baseline mismatch on {k}: {baseline_metrics[k]} != {ref_m[k]}")
        for k in ("recall@10", "ndcg@10"):
            if abs(baseline_metrics[k] - ref_m[k]) > 1e-12:
                raise AssertionError(f"Baseline mismatch on {k}: {baseline_metrics[k]} != {ref_m[k]}")

    # 2. Train new 300D model from scratch (or reuse verified completed run in artifact_root).
    variant_res = fit_ablated_stage1_selector(
        spec,
        campaign_data,
        info,
        meta,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        device=device,
    )
    variant_run_dir = Path(variant_res["selected_checkpoint"]).parent
    variant_history = json.loads((variant_run_dir / "history.json").read_text(encoding="utf-8"))
    variant_npz = Path(variant_res["predictions_path"])

    # 3. Paired customer-clustered 95% bootstrap intervals.
    intervals = paired_feature_ablation_intervals(baseline_npz, variant_npz)

    hashes_after = snapshot_preserved_hashes()
    all_matched = hashes_before == hashes_after
    if not all_matched:
        diff_keys = [k for k in hashes_before if hashes_before[k] != hashes_after.get(k)]
        raise AssertionError(f"Preserved artifact hash modified during run: {diff_keys}")

    summary = {
        "experiment": "mlp_feature_ablation_v1",
        "variant": spec.name,
        "reference": {
            "reference_id": meta["reference"]["reference_pointer"]["active_reference_id"],
            "reference_pointer_path": str(REFERENCE_POINTER),
            "manifest_path": str(REFERENCE_MANIFEST),
            "retriever_path": str(REFERENCE_RETRIEVER),
            "selector_path": str(REFERENCE_SELECTOR),
            "promoted": False,
        },
        "data_safety": {
            "train_date_range": "< 2025-10-01",
            "val_date_range": "2025-10-01..2026-02-18",
            "heldout_split_cutoff": ">= 2026-02-19",
            "heldout_access": False,
            "metadata_policy": "snapshot_unverified",
            "train_queries": int(len(campaign_data["train"]["u"])),
            "val_queries": int(len(va["u"])),
            "val_new_targets": 1857,
            "val_total_targets": int(baseline_metrics["total_targets"]),
            "product_universe": 841,
            "val_eligible_min": int(va["allowed"].sum(1).min()),
            "val_eligible_max": int(va["allowed"].sum(1).max()),
        },
        "feature_ablation_spec": {
            "name": spec.name,
            "recipe": spec.recipe,
            "source_context_dim": spec.source_num_features,
            "omitted_names": list(spec.omitted_names),
            "omitted_indices": list(spec.omitted_indices),
            "retained_context_dim": spec.num_retained_features,
            "retained_names": list(spec.retained_names),
            "retained_indices": list(spec.retained_indices),
            "customer_stats_dim": spec.num_customer_stats,
            "customer_stats_names": list(spec.customer_stats_names),
            "scalar_feature_count": spec.scalar_feature_count,
            "input_dim": spec.input_dim,
            "input_contract": build_input_contract(spec),
        },
        "retained_feature_table": describe_retained_features(spec),
        "baseline_303": {
            "selector_sha256": meta["reference"]["hashes"]["best_selector_pt"],
            "val_scores_path": baseline_meta["val_scores_path"],
            "val_scores_sha256": baseline_meta["val_scores_sha256"],
            "predictions_npz": str(baseline_npz),
            "selected_epoch": 1,
            "parameters": 47617,
            "metrics": baseline_metrics,
        },
        "variant_300": {
            **variant_res,
            "history": variant_history,
        },
        "paired_bootstrap_95_ci": intervals,
        "bootstrap_note": (
            "Customer-clustered 95% bootstrap intervals (2,000 draws, seed=123) at selected weights "
            "for a single training seed (seed=42); conditional on selected weights, not multi-seed proof."
        ),
        "hash_preservation": {
            "all_matched": all_matched,
            "before": hashes_before,
            "after": hashes_after,
        },
    }

    summary_path = evidence_dir / "summary.json"
    report_path = evidence_dir / "REPORT.md"
    atomic_json(summary, summary_path, overwrite=True)
    report_path.write_text(_format_report_md(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled Stage 1 MLP feature-removal ablation (omit is_repeat_sku, month_sin, month_cos)."
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    summary = run_feature_ablation_experiment(
        artifact_root=args.artifact_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
    base = summary["baseline_303"]["metrics"]
    var = summary["variant_300"]["metrics"]
    print(
        f"COMPLETED epoch={summary['variant_300']['selected_epoch']} "
        f"rec5={var['recall@5']:.5f} (base {base['recall@5']:.5f}) "
        f"rec10={var['recall@10']:.5f} (base {base['recall@10']:.5f}) "
        f"ndcg10={var['ndcg@10']:.5f} (base {base['ndcg@10']:.5f}) "
        f"new50={var['new_hits@50']} (base {base['new_hits@50']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
