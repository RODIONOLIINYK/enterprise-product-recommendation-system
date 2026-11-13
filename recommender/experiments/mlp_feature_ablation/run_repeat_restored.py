"""Controlled Stage 1 MLP experiment restoring ONLY `is_repeat_sku` (`repeat_no_month` / `uv_45` / `301D`)."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional

import torch

from recommender.artifacts import atomic_json, file_digest

from .config import (
    EPOCHS,
    PRIOR_300_EVIDENCE_DIR,
    PRIOR_300_RUN_DIR,
    REFERENCE_MANIFEST,
    REFERENCE_POINTER,
    REFERENCE_RETRIEVER,
    REFERENCE_SELECTOR,
    REPEAT_RESTORED_ARTIFACT_ROOT,
    REPEAT_RESTORED_EVIDENCE_ROOT,
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
    SEED,
)
from .data import (
    load_verified_baseline_val_scores,
    load_verified_no_brand_campaign_data,
    load_verified_prior_300_val_scores,
)
from .evaluate import evaluate_full_pool_metrics, paired_feature_ablation_intervals
from .schema import (
    build_feature_ablation_spec,
    build_input_contract,
    describe_retained_features,
)
from .train import fit_ablated_stage1_selector, snapshot_preserved_hashes


def _persist_immutable_candidate(
    artifact_root: Path,
    *,
    variant_res: Mapping[str, Any],
    spec_contract: Mapping[str, Any],
    meta: Mapping[str, Any],
    baseline_303_metrics: Mapping[str, Any],
    prior_300_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Save byte-identical selected model copy and candidate manifest in a NEW candidate dir (not active reference)."""
    candidate_dir = Path(artifact_root) / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    src_ckpt = Path(variant_res["selected_checkpoint"])
    dst_ckpt = candidate_dir / "best_selector.pt"
    src_sha = variant_res["selected_checkpoint_sha256"]

    if dst_ckpt.exists():
        if file_digest(dst_ckpt) != src_sha:
            raise FileExistsError(
                f"Candidate checkpoint already exists with different digest at {dst_ckpt}."
            )
    else:
        tmp_ckpt = candidate_dir / ".best_selector.pt.tmp"
        shutil.copy2(src_ckpt, tmp_ckpt)
        tmp_ckpt.replace(dst_ckpt)

    dst_sha = file_digest(dst_ckpt)
    if dst_sha != src_sha:
        raise AssertionError(f"Candidate checkpoint copy hash mismatch: {dst_sha} != {src_sha}")

    manifest = {
        "manifest_version": 1,
        "candidate_id": "repeat-no-month-uv45-seed42",
        "is_active_reference": False,
        "promoted": False,
        "promotion_authority": "pending_parent_review",
        "scope": "stage1_compact_selector_mlp_candidate_only",
        "validation_only": True,
        "heldout_access": False,
        "stage2_executed": False,
        "metadata_policy": "snapshot_unverified",
        "upstream_reference": {
            "reference_id": meta["reference"]["reference_pointer"]["active_reference_id"],
            "reference_pointer_path": str(REFERENCE_POINTER),
            "retriever_path": str(REFERENCE_RETRIEVER),
            "retriever_sha256": meta["reference"]["hashes"]["best_retriever_pt"],
            "retriever_selected_epoch": 16,
        },
        "selector": {
            "path": str(dst_ckpt),
            "sha256": dst_sha,
            "bytes": int(dst_ckpt.stat().st_size),
            "byte_identical_to_run_checkpoint": True,
            "source_checkpoint_path": str(src_ckpt),
            "source_run_dir": str(src_ckpt.parent),
            "selected_epoch": int(variant_res["selected_epoch"]),
            "max_epochs": int(variant_res["settings"]["epochs"]),
            "seed": int(variant_res["settings"]["seed"]),
            "selection_criterion": variant_res["settings"]["selection"],
            "parameters": int(variant_res["parameters"]),
            "trainable_parameters": int(variant_res["trainable_parameters"]),
            "first_layer_in_features": int(variant_res["first_layer_in_features"]),
            "input_contract": spec_contract,
        },
        "normalization_status": {
            "legacy_normalization_unchanged": True,
            "all_inputs_unit_interval_0_1": False,
            "pending_work": (
                "Numeric normalization (8 signed lift/growth features in [-1, 1] plus unit-norm [-1, 1] u/v) "
                "and category-feature work remain pending."
            ),
        },
        "validation_metrics": variant_res["metrics"],
        "comparison_baselines": {
            "baseline_303_metrics": baseline_303_metrics,
            "prior_300_metrics": prior_300_metrics,
        },
    }
    manifest_path = candidate_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("selector", {}).get("sha256") != dst_sha:
            raise FileExistsError(
                f"Candidate manifest already exists for different selector hash at {manifest_path}."
            )
    else:
        atomic_json(manifest, manifest_path, overwrite=False)

    return {
        "candidate_dir": str(candidate_dir),
        "selector_path": str(dst_ckpt),
        "selector_sha256": dst_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_digest(manifest_path),
        "byte_identical": True,
        "promoted": False,
    }


def _extract_epoch_metrics(history: Optional[list[Dict[str, Any]]], target_epoch: int) -> Optional[Dict[str, Any]]:
    if not history:
        return None
    for row in history:
        if int(row.get("epoch", -1)) == int(target_epoch):
            return row.get("metrics")
    return None


def _format_report_md(summary: Mapping[str, Any]) -> str:
    b303 = summary["baseline_303"]["metrics"]
    p300 = summary["prior_300"]["metrics"]
    v301 = summary["variant_301"]["metrics"]
    ci_303 = summary["paired_bootstrap_vs_baseline_303"]
    ci_300 = summary["paired_bootstrap_vs_prior_300"]
    v301_meta = summary["variant_301"]
    spec_info = summary["feature_spec"]
    ep1 = summary["epoch1_comparison"]
    rep_inf = v301_meta["repeat_flag_influence"]
    cand = summary["candidate_artifact"]
    hashes_ok = summary["hash_preservation"]["all_matched"]

    def _pct(x: float) -> str:
        return f"{x * 100.0:.3f}%"

    def _observed_delta(cis: Mapping[str, Any], key: str) -> float:
        # Show the observed paired difference, rather than the bootstrap median.
        control = b303 if cis is ci_303 else p300
        metric = key.removesuffix("_delta")
        return float(v301[metric] - control[metric])

    def _ci_pp(cis: Mapping[str, Any], key: str) -> str:
        lo, _, hi = cis[key]
        delta = _observed_delta(cis, key)
        return f"{delta * 100.0:+.3f} pp [{lo * 100.0:+.3f}, {hi * 100.0:+.3f}]"

    def _ci_val(cis: Mapping[str, Any], key: str) -> str:
        lo, _, hi = cis[key]
        return f"{_observed_delta(cis, key):+.5f} [{lo:+.5f}, {hi:+.5f}]"

    lines = [
        "# Stage 1 MLP Repeat-Restored Experiment (`repeat_no_month` / `uv_45` / `301D`)",
        "",
        "## 1. Experiment Scope & Controlled Setup",
        "",
        f"- **Baseline Reference (`303D`):** `{summary['reference']['reference_id']}` (`CompactStage1Selector`, `uv_47`, `303D = 128 + 128 + 41 + 6`, `47,617` params, selected epoch `1/10`, SHA-256 `{summary['baseline_303']['selector_sha256']}`).",
        f"- **Prior Ablated Variant (`300D`):** `no_repeat_no_month` (`FeatureAblatedStage1Selector`, `uv_44`, `300D = 128 + 128 + 38 + 6`, `47,233` params, selected epoch `{summary['prior_300']['selected_epoch']}/10`, SHA-256 `{summary['prior_300']['selector_sha256']}`).",
        f"- **Treatment Variant (`301D`):** `{spec_info['name']}` (`FeatureAblatedStage1Selector`, `{spec_info['recipe']}`, `301D = 128 + 128 + 39 + 6`, `{v301_meta['parameters']:,}` params, selected epoch `{v301_meta['selected_epoch']}/10`, SHA-256 `{v301_meta['selected_checkpoint_sha256']}`).",
        "- **Frozen Upstream Towers:** `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt` (epoch 16, SHA-256 `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981`); saved train/val `u`/`v` purchase caches reused read-only with zero tower re-encoding.",
        "- **Feature Change Only:** Restores ONLY `0: is_repeat_sku` while keeping `31: month_sin` and `32: month_cos` physically omitted (`39` retained context features + `6` customer stats = `45` scalar inputs; `head.0` physically `Linear(301, 128)`).",
        "- **Training Protocol:** `seed=42`, `10` epochs from scratch, `AdamW(lr=0.001, weight_decay=0.0001)`, `CosineAnnealingLR(eta_min=1e-5)`, `batch_size=64`, `BCEWithLogitsLoss(pos_weight=20.0)` over full rolling eligible pool, `clip_grad_norm_(1.0)`, selection `(new_hits@50, ndcg@10, recall@10)`.",
        "- **Data Safety & Explicit Denominators:** Train `< 2025-10-01` (`26,253` queries with >=100 eligible SKUs); Val `2025-10-01..2026-02-18` (`6,191` queries, `10,276` total targets, `1,857` new-to-customer targets, `841` SKU universe, `735–771` eligible/query). Held-out split `>= 2026-02-19` untouched (`heldout_access=false`). Metadata policy: `snapshot_unverified`. No Stage 2 execution, no active-reference mutation (`promoted=false`).",
        "- **Pending Legacy Normalization:** Current numeric normalization remains legacy (`u`/`v` L2-normalized in `[-1, 1]`, `customer_stats_6` in `[0, 1]`, `31` retained context columns in `[0, 1]`, and `8` signed lift/growth columns in `[-1, 1]`). Numeric normalization and category-feature work remain pending.",
        "",
        "## 2. Validation Comparison vs Both Baselines (`6,191` Queries / `10,276` Total Targets / `1,857` New Targets)",
        "",
        "> [!NOTE]",
        "> Confidence intervals are paired **customer-clustered 95% bootstrap intervals** (`2,000` draws, seed `123`) over the `6,191` validation queries at selected weights for a single training seed (`seed=42`). They reflect query/customer sampling uncertainty conditional on these selected weights, **not** multi-seed training variance; do not claim repeatable superiority from query bootstrap alone.",
        "",
        "### Overall Ranking Metrics (All Eligible Candidates, Repeat + New, No Unseen Filtering)",
        "",
        f"| Metric | Baseline `303D` (`uv_47`, Ep 1) | Prior `300D` (`uv_44`, Ep {summary['prior_300']['selected_epoch']}) | Variant `301D` (`uv_45`, Ep {v301_meta['selected_epoch']}) | Delta vs `303D` [95% CI] | Delta vs `300D` [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Overall Macro `Recall@5` | {_pct(b303['recall@5'])} (`{b303['recall@5']:.6f}`) | {_pct(p300['recall@5'])} (`{p300['recall@5']:.6f}`) | {_pct(v301['recall@5'])} (`{v301['recall@5']:.6f}`) | `{_ci_pp(ci_303, 'recall@5_delta')}` | `{_ci_pp(ci_300, 'recall@5_delta')}` |",
        f"| Overall Macro `Recall@10` | {_pct(b303['recall@10'])} (`{b303['recall@10']:.6f}`) | {_pct(p300['recall@10'])} (`{p300['recall@10']:.6f}`) | {_pct(v301['recall@10'])} (`{v301['recall@10']:.6f}`) | `{_ci_pp(ci_303, 'recall@10_delta')}` | `{_ci_pp(ci_300, 'recall@10_delta')}` |",
        f"| Overall Macro `Recall@25` | {_pct(b303['recall@25'])} (`{b303['recall@25']:.6f}`) | {_pct(p300['recall@25'])} (`{p300['recall@25']:.6f}`) | {_pct(v301['recall@25'])} (`{v301['recall@25']:.6f}`) | `{_ci_pp(ci_303, 'recall@25_delta')}` | `{_ci_pp(ci_300, 'recall@25_delta')}` |",
        f"| Overall Macro `Recall@50` | {_pct(b303['recall@50'])} (`{b303['recall@50']:.6f}`) | {_pct(p300['recall@50'])} (`{p300['recall@50']:.6f}`) | {_pct(v301['recall@50'])} (`{v301['recall@50']:.6f}`) | `{_ci_pp(ci_303, 'recall@50_delta')}` | `{_ci_pp(ci_300, 'recall@50_delta')}` |",
        f"| Overall Macro `Recall@100` | {_pct(b303['recall@100'])} (`{b303['recall@100']:.6f}`) | {_pct(p300['recall@100'])} (`{p300['recall@100']:.6f}`) | {_pct(v301['recall@100'])} (`{v301['recall@100']:.6f}`) | `{_ci_pp(ci_303, 'recall@100_delta')}` | `{_ci_pp(ci_300, 'recall@100_delta')}` |",
        f"| Overall `NDCG@10` | `{b303['ndcg@10']:.6f}` | `{p300['ndcg@10']:.6f}` | `{v301['ndcg@10']:.6f}` | `{_ci_val(ci_303, 'ndcg@10_delta')}` | `{_ci_val(ci_300, 'ndcg@10_delta')}` |",
        f"| Overall `MRR` | `{b303['mrr']:.6f}` | `{p300['mrr']:.6f}` | `{v301['mrr']:.6f}` | `{(v301['mrr'] - b303['mrr']):+.6f}` | `{(v301['mrr'] - p300['mrr']):+.6f}` |",
        f"| Repeat `Recall@10` (Macro / Micro) | {_pct(b303['repeat_recall@10_macro'])} / {_pct(b303['repeat_recall@10_micro'])} | {_pct(p300['repeat_recall@10_macro'])} / {_pct(p300['repeat_recall@10_micro'])} | {_pct(v301['repeat_recall@10_macro'])} / {_pct(v301['repeat_recall@10_micro'])} | `{(v301['repeat_recall@10_macro'] - b303['repeat_recall@10_macro']) * 100.0:+.3f}` / `{(v301['repeat_recall@10_micro'] - b303['repeat_recall@10_micro']) * 100.0:+.3f} pp` | `{(v301['repeat_recall@10_macro'] - p300['repeat_recall@10_macro']) * 100.0:+.3f}` / `{(v301['repeat_recall@10_micro'] - p300['repeat_recall@10_micro']) * 100.0:+.3f} pp` |",
        "",
        "### New-Target Discovery Metrics (`1,857` New-to-Customer Target Instances)",
        "",
        "| Metric | Ranking Pool | Baseline `303D` (`uv_47`, Ep 1) | Prior `300D` (`uv_44`, Ep 5) | Variant `301D` (`uv_45`, Ep "
        + f"{v301_meta['selected_epoch']}) | Delta vs `303D` [95% CI] | Delta vs `300D` [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| `new_hits@10` / `new_recall@10` | Ordinary Top-10 (all eligible) | `{b303['new_hits@10']} / 1,857` ({_pct(b303['new_recall@10'])}) | `{p300['new_hits@10']} / 1,857` ({_pct(p300['new_recall@10'])}) | `{v301['new_hits@10']} / 1,857` ({_pct(v301['new_recall@10'])}) | `{v301['new_hits@10'] - b303['new_hits@10']:+d}` (`{_ci_pp(ci_303, 'new_recall@10_delta')}`) | `{v301['new_hits@10'] - p300['new_hits@10']:+d}` (`{_ci_pp(ci_300, 'new_recall@10_delta')}`) |",
        f"| `new_hits@25` / `new_recall@25` | Unseen-only | `{b303['new_hits@25']} / 1,857` ({_pct(b303['new_recall@25'])}) | `{p300['new_hits@25']} / 1,857` ({_pct(p300['new_recall@25'])}) | `{v301['new_hits@25']} / 1,857` ({_pct(v301['new_recall@25'])}) | `{v301['new_hits@25'] - b303['new_hits@25']:+d}` (`{_ci_pp(ci_303, 'new_recall@25_delta')}`) | `{v301['new_hits@25'] - p300['new_hits@25']:+d}` (`{_ci_pp(ci_300, 'new_recall@25_delta')}`) |",
        f"| `new_hits@50` / `new_recall@50` (selection key) | Unseen-only | `{b303['new_hits@50']} / 1,857` ({_pct(b303['new_recall@50'])}) | `{p300['new_hits@50']} / 1,857` ({_pct(p300['new_recall@50'])}) | `{v301['new_hits@50']} / 1,857` ({_pct(v301['new_recall@50'])}) | `{v301['new_hits@50'] - b303['new_hits@50']:+d}` (`{_ci_pp(ci_303, 'new_recall@50_delta')}`) | `{v301['new_hits@50'] - p300['new_hits@50']:+d}` (`{_ci_pp(ci_300, 'new_recall@50_delta')}`) |",
        f"| `new_hits@100` / `new_recall@100` | Unseen-only | `{b303['new_hits@100']} / 1,857` ({_pct(b303['new_recall@100'])}) | `{p300['new_hits@100']} / 1,857` ({_pct(p300['new_recall@100'])}) | `{v301['new_hits@100']} / 1,857` ({_pct(v301['new_recall@100'])}) | `{v301['new_hits@100'] - b303['new_hits@100']:+d}` (`{_ci_pp(ci_303, 'new_recall@100_delta')}`) | `{v301['new_hits@100'] - p300['new_hits@100']:+d}` (`{_ci_pp(ci_300, 'new_recall@100_delta')}`) |",
        "",
        "## 3. Descriptive Epoch-1 Comparison (Checkpoint-Selection Context)",
        "",
        "Epoch 1 metrics are shown descriptively below for all three variants to separate feature-set differences from checkpoint-selection epoch differences (`baseline_303` selected Ep 1; `prior_300` selected Ep 5; `variant_301` selected Ep "
        + f"{v301_meta['selected_epoch']}). No extra epoch was selected or retrained after viewing metrics.",
        "",
        "| Variant (Epoch 1) | `Recall@5` | `Recall@10` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, ep_m in (
        ("Baseline `303D` (`uv_47`, Ep 1)", ep1["baseline_303_epoch1"]),
        ("Prior `300D` (`uv_44`, Ep 1)", ep1["prior_300_epoch1"]),
        ("Variant `301D` (`uv_45`, Ep 1)", ep1["variant_301_epoch1"]),
    ):
        lines.append(
            f"| {label} | {ep_m['recall@5']:.5f} | {ep_m['recall@10']:.5f} | {ep_m['ndcg@10']:.5f} | "
            f"{ep_m['new_hits@10']} | {ep_m['new_hits@25']} | {ep_m['new_hits@50']} | {ep_m['new_hits@100']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Per-Epoch History (`301D` Variant `repeat_no_month` / `uv_45`, `seed=42`)",
            "",
            "| Epoch | Train Loss | Overall `Recall@5` | Overall `Recall@10` | Overall `Recall@25` | Overall `Recall@50` | Overall `Recall@100` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` | Selected |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in v301_meta["history"]:
        m = row["metrics"]
        sel = "YES" if row["epoch"] == v301_meta["selected_epoch"] else ""
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.5f} | {m['recall@5']:.5f} | {m['recall@10']:.5f} | "
            f"{m['recall@25']:.5f} | {m['recall@50']:.5f} | {m['recall@100']:.5f} | {m['ndcg@10']:.5f} | "
            f"{m['new_hits@10']} | {m['new_hits@25']} | {m['new_hits@50']} | {m['new_hits@100']} | {sel} |"
        )

    lines.extend(
        [
            "",
            "## 5. Runtime Cost, Invariance, `is_repeat_sku` Influence & Candidate Artifacts",
            "",
            f"- **Device:** `{v301_meta['device']}`",
            f"- **Training time (10 epochs, 26,253 rows):** `{v301_meta['training_seconds']:.2f}s`",
            f"- **Validation replay inference time (6,191 rows x 841 SKUs):** `{v301_meta['val_inference_seconds']:.3f}s`",
            f"- **Parameters:** `{v301_meta['parameters']:,}` (`first_layer_in_features={v301_meta['first_layer_in_features']}`, `-256 = -2 * 128` weights vs `303D`, `+128 = +1 * 128` weights vs `300D`)",
            f"- **Month omission invariance max abs diff (perturbing source cols 31, 32):** `{v301_meta['invariance_checks']['omission_invariance_max_abs_diff']}`",
            f"- **Retained feature sensitivity max abs diff (zeroing retained 39 cols):** `{v301_meta['invariance_checks']['retained_zero_max_abs_diff']:.6f}`",
            f"- **Retained `is_repeat_sku` influence (`1` vs `0` across {rep_inf['eligible_pairs_checked']:,} eligible val pairs):** max abs diff `{rep_inf['max_abs_logit_diff_1_vs_0']:.6f}`, mean abs diff `{rep_inf['mean_abs_logit_diff_1_vs_0']:.6f}`, mean signed `(1 - 0)` `{rep_inf['mean_signed_logit_diff_1_minus_0']:+.6f}`",
            f"- **Candidate checkpoint copy (not active reference):** `{cand['selector_path']}` (SHA-256 `{cand['selector_sha256']}`, byte-identical={cand['byte_identical']}, promoted={cand['promoted']})",
            f"- **Candidate manifest:** `{cand['manifest_path']}` (SHA-256 `{cand['manifest_sha256']}`)",
            f"- **Preserved audit/reference/prior-300 hashes before & after:** `{'ALL MATCHED' if hashes_ok else 'MISMATCH'}`",
            "",
            "## 6. Retained Feature Contract (`39` Context + `6` Customer Stats = `45` Scalars, `301D` Total Input)",
            "",
            "Canonical source column `0: is_repeat_sku` is retained at MLP column `256`; only `31: month_sin` and `32: month_cos` are omitted. Legacy numeric normalization remains unchanged (`u`/`v` L2-normalized in `[-1, 1]`, `customer_stats_6` in `[0, 1]`, `31` retained context columns in `[0, 1]`, `8` signed columns in `[-1, 1]`).",
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
            f"Plus `customer_stats_6` at MLP columns `{39 + 256}..{39 + 256 + 5}`: `distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`.",
            "",
            "## 7. Reproduction Commands",
            "",
            "```bash",
            "./venv/bin/python -m recommender.experiments.mlp_feature_ablation.run_repeat_restored",
            "PYTHONPATH=. ./venv/bin/pytest tests/",
            "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_repeat_restored_experiment(
    *,
    artifact_root: Path = REPEAT_RESTORED_ARTIFACT_ROOT,
    evidence_root: Path = REPEAT_RESTORED_EVIDENCE_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the controlled 301D `repeat_no_month` (`uv_45`) experiment and evaluate vs 303D and 300D."""
    hashes_before = snapshot_preserved_hashes(include_prior_300=True)
    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    if spec.input_dim != 301 or spec.num_retained_features != 39 or spec.scalar_feature_count != 45:
        raise AssertionError(f"Unexpected 301D spec dimensions: {asdict(spec)}")
    if spec.retained_names[0] != "is_repeat_sku" or spec.retained_indices[0] != 0:
        raise AssertionError("Expected is_repeat_sku retained at index 0.")
    if spec.omitted_names != ("month_sin", "month_cos") or spec.omitted_indices != (31, 32):
        raise AssertionError(f"Unexpected omitted month spec: {asdict(spec)}")

    campaign_data, info, meta = load_verified_no_brand_campaign_data()
    va = campaign_data["val"]

    evidence_dir = Path(evidence_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verified 303D baseline evaluation from saved val_scores.pt (no retraining).
    baseline_scores, baseline_meta = load_verified_baseline_val_scores(info["identity"])
    baseline_npz = evidence_dir / "baseline_303_predictions.npz"
    baseline_metrics = evaluate_full_pool_metrics(
        baseline_scores, va, save_predictions=baseline_npz
    )
    ref_eval = baseline_meta["baseline_eval_json"]
    if ref_eval is not None:
        ref_m = ref_eval["metrics"]
        for k in ("new_hits@10", "new_hits@25", "new_hits@50", "new_hits@100", "queries", "new_targets"):
            if baseline_metrics[k] != ref_m[k]:
                raise AssertionError(f"Baseline 303D mismatch on {k}: {baseline_metrics[k]} != {ref_m[k]}")
        for k in ("recall@10", "ndcg@10"):
            if abs(baseline_metrics[k] - ref_m[k]) > 1e-12:
                raise AssertionError(f"Baseline 303D mismatch on {k}: {baseline_metrics[k]} != {ref_m[k]}")

    # 2. Verified prior 300D evaluation from saved val_scores.pt (no retraining).
    prior_300_scores, prior_300_meta = load_verified_prior_300_val_scores(info["identity"])
    prior_300_npz = evidence_dir / "prior_300_predictions.npz"
    prior_300_metrics = evaluate_full_pool_metrics(
        prior_300_scores, va, save_predictions=prior_300_npz
    )
    prior_300_saved_m = prior_300_meta["result_json"]["metrics"]
    for k in ("new_hits@10", "new_hits@25", "new_hits@50", "new_hits@100", "queries", "new_targets"):
        if prior_300_metrics[k] != prior_300_saved_m[k]:
            raise AssertionError(f"Prior 300D mismatch on {k}: {prior_300_metrics[k]} != {prior_300_saved_m[k]}")
    for k in ("recall@10", "ndcg@10"):
        if abs(prior_300_metrics[k] - prior_300_saved_m[k]) > 1e-12:
            raise AssertionError(f"Prior 300D mismatch on {k}: {prior_300_metrics[k]} != {prior_300_saved_m[k]}")

    # 3. Fit 301D `repeat_no_month` (`uv_45`) from scratch with seed=42 (no extra zero-ablation grid).
    variant_res = fit_ablated_stage1_selector(
        spec,
        campaign_data,
        info,
        meta,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        device=device,
        stage_name="stage1_compact_selector_repeat_restored_v1",
        run_zero_ablations=False,
        check_repeat_flag_influence=True,
    )
    variant_run_dir = Path(variant_res["selected_checkpoint"]).parent
    variant_history = json.loads((variant_run_dir / "history.json").read_text(encoding="utf-8"))
    variant_run_npz = Path(variant_res["predictions_path"])
    variant_npz = evidence_dir / "repeat_restored_301_predictions.npz"
    shutil.copy2(variant_run_npz, variant_npz)

    # 4. Save immutable byte-identical candidate model + manifest (NOT active reference pointer).
    spec_contract = build_input_contract(spec)
    candidate_info = _persist_immutable_candidate(
        Path(artifact_root),
        variant_res=variant_res,
        spec_contract=spec_contract,
        meta=meta,
        baseline_303_metrics=baseline_metrics,
        prior_300_metrics=prior_300_metrics,
    )

    # 5. Paired customer-clustered 95% bootstrap intervals vs BOTH baselines.
    ci_vs_303 = paired_feature_ablation_intervals(baseline_npz, variant_npz)
    ci_vs_300 = paired_feature_ablation_intervals(prior_300_npz, variant_npz)

    # 6. Descriptive Epoch-1 comparison across all three models.
    b303_ep1 = _extract_epoch_metrics(baseline_meta.get("history"), 1)
    p300_ep1 = _extract_epoch_metrics(prior_300_meta.get("history"), 1)
    v301_ep1 = _extract_epoch_metrics(variant_history, 1)
    if b303_ep1 is None or p300_ep1 is None or v301_ep1 is None:
        raise AssertionError("Missing epoch 1 history metrics for descriptive comparison.")

    hashes_after = snapshot_preserved_hashes(include_prior_300=True)
    all_matched = hashes_before == hashes_after
    if not all_matched:
        diff_keys = [k for k in hashes_before if hashes_before[k] != hashes_after.get(k)]
        raise AssertionError(f"Preserved artifact hash modified during run: {diff_keys}")

    summary = {
        "experiment": "mlp_repeat_restored_v1",
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
        "feature_spec": {
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
            "input_contract": spec_contract,
        },
        "retained_feature_table": describe_retained_features(spec),
        "normalization_status": {
            "legacy_normalization_unchanged": True,
            "all_inputs_unit_interval_0_1": False,
            "pending_work": (
                "Numeric normalization (8 signed lift/growth features in [-1, 1] plus unit-norm [-1, 1] u/v) "
                "and category-feature work remain pending."
            ),
        },
        "baseline_303": {
            "variant_name": "no-brand-v1",
            "recipe": "uv_47",
            "input_dim": 303,
            "selector_sha256": meta["reference"]["hashes"]["best_selector_pt"],
            "val_scores_path": baseline_meta["val_scores_path"],
            "val_scores_sha256": baseline_meta["val_scores_sha256"],
            "predictions_npz": str(baseline_npz),
            "selected_epoch": 1,
            "parameters": 47617,
            "metrics": baseline_metrics,
        },
        "prior_300": {
            "variant_name": "no_repeat_no_month",
            "recipe": "uv_44",
            "input_dim": 300,
            "run_dir": str(PRIOR_300_RUN_DIR),
            "evidence_dir": str(PRIOR_300_EVIDENCE_DIR),
            "selector_path": prior_300_meta["selector_path"],
            "selector_sha256": prior_300_meta["selector_sha256"],
            "val_scores_path": prior_300_meta["val_scores_path"],
            "val_scores_sha256": prior_300_meta["val_scores_sha256"],
            "predictions_npz": str(prior_300_npz),
            "selected_epoch": int(prior_300_meta["result_json"]["selected_epoch"]),
            "parameters": int(prior_300_meta["result_json"]["parameters"]),
            "metrics": prior_300_metrics,
        },
        "variant_301": {
            **variant_res,
            "predictions_copy_npz": str(variant_npz),
            "history": variant_history,
        },
        "candidate_artifact": candidate_info,
        "epoch1_comparison": {
            "note": "Descriptive epoch-1 comparison only; no post-hoc epoch selection performed.",
            "baseline_303_epoch1": b303_ep1,
            "prior_300_epoch1": p300_ep1,
            "variant_301_epoch1": v301_ep1,
        },
        "paired_bootstrap_vs_baseline_303": ci_vs_303,
        "paired_bootstrap_vs_prior_300": ci_vs_300,
        "single_seed_and_bootstrap_limits": (
            "Single exploratory training seed (seed=42). Paired customer-clustered 95% bootstrap intervals "
            "(2,000 draws, seed=123) quantify validation query/customer sampling variability conditional on "
            "the selected checkpoint weights only, not training-seed or initialization variance; do not claim "
            "repeatable training superiority from query bootstrap alone."
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
        description=(
            "Run controlled Stage 1 MLP repeat-restored ablation "
            "(restore is_repeat_sku; omit month_sin, month_cos; recipe uv_45, 301D)."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=REPEAT_RESTORED_ARTIFACT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=REPEAT_RESTORED_EVIDENCE_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    summary = run_repeat_restored_experiment(
        artifact_root=args.artifact_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
    b303 = summary["baseline_303"]["metrics"]
    p300 = summary["prior_300"]["metrics"]
    v301 = summary["variant_301"]["metrics"]
    print(
        f"COMPLETED 301D epoch={summary['variant_301']['selected_epoch']} "
        f"rec10={v301['recall@10']:.5f} (303D {b303['recall@10']:.5f}, 300D {p300['recall@10']:.5f}) "
        f"ndcg10={v301['ndcg@10']:.5f} (303D {b303['ndcg@10']:.5f}, 300D {p300['ndcg@10']:.5f}) "
        f"new50={v301['new_hits@50']} (303D {b303['new_hits@50']}, 300D {p300['new_hits@50']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
