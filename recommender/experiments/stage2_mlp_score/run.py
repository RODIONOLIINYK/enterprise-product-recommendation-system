"""End-to-end runner and report generator for controlled Stage 2 + 301D MLP score experiment."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from recommender.artifacts import atomic_json

from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    CACHE_ROOT,
    EPOCHS,
    EVIDENCE_ROOT,
    SCORE_FEATURE_NAME,
    SCORE_TRANSFORM_VERSION,
    SEED,
    SLATE_POLICY,
)
from .data import (
    load_verified_stage2_campaign_data,
    prepare_or_load_stage2_slate_cache,
)
from .evaluate import (
    evaluate_stage2_slate_metrics,
    paired_stage2_intervals,
)
from .train import (
    fit_single_stage2_run,
    snapshot_preserved_hashes,
    verify_cache_vs_request_adapter_parity,
    verify_zero_extension_initialization_parity,
)


def _fmt_pct(val: Optional[float], digits: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{100.0 * val:.{digits}f}%"


def _fmt_num(val: Optional[float], digits: int = 5) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def _build_markdown_report(summary: Mapping[str, Any]) -> str:
    ctrl = summary["arms"]["control"]
    trt = summary["arms"]["treatment"]
    c_disc = ctrl["discovery"]["metrics"]
    t_disc = trt["discovery"]["metrics"]
    c_ndcg = ctrl["ndcg"]["metrics"]
    t_ndcg = trt["ndcg"]["metrics"]
    st1_ref = summary["baselines"]["stage1_301d_full_pool"]
    st1_slate = summary["baselines"]["stage1_301d_on_val_slate"]
    guard = summary["guardrails"]
    ci_disc = summary["paired_intervals"]["discovery_vs_discovery"]
    ci_ndcg = summary["paired_intervals"]["ndcg_vs_ndcg"]
    slate_cov = summary["slate_coverage_val"]

    lines = [
        "# Stage 2 UPS-CANR Integration with Frozen Stage 1 301D MLP Score",
        "",
        "## 1. Experiment Contract & Setup",
        f"- **Stage 1 Reference (`active_reference_id`)**: `{summary['reference']['active_reference_id']}` (`uv_45`, `301D`, selected epoch 5, `heldout_access={summary['heldout_access']}`)",
        f"- **Score Feature**: `{SCORE_FEATURE_NAME}` via `{SCORE_TRANSFORM_VERSION}` (`sigmoid(raw_logit)` in `[0, 1]` float32 after pre-sigmoid eligibility masking)",
        "- **Stage 2 Integration Path**: `Stage2MLPScoreRanker` wrapping `CandidateContextRanker('dynamic_features')` around `UPSCANRRanker` (`excluded_statistics=('history_repeat',)`, `num_pairwise_features=14` -> cross-attention `W[4, 15, 17]` + `398D` encoded features + `28D` dynamic cols `13:41` -> `426D` head)",
        f"- **Candidate Slate Policy**: `{SLATE_POLICY}` (100 SKUs: up to 50 eligible repeat + remaining top eligible unseen by frozen `301D` MLP score)",
        f"- **Controlled Arms (`seed={summary['seed']}`, `epochs={summary['epochs']}`, `batch_size={summary['batch_size']}`)**:",
        f"  - **Matched Control (`score_channel_enabled=False`)**: identical 426D architecture, identical initial weights (zero-init on new score columns), score channel forced to `0.0` before attention and head; `{ctrl['parameters']:,}` params (`{ctrl['trainable_parameters']:,}` trainable), `{ctrl['training_seconds']:.1f}s`",
        f"  - **Treatment (`score_channel_enabled=True`)**: identical initial weights and minibatch sequence, actual masked `sigmoid(mlp_logit)` fed to attention and head; `{trt['parameters']:,}` params (`{trt['trainable_parameters']:,}` trainable), `{trt['training_seconds']:.1f}s`",
        f"- **Quality Guardrails (`checkpoints/ranker/best_discovery_ranker.pt`)**: `Recall@10 >= {guard['recall_at_10']:.5f}`, `NDCG@10 >= {guard['ndcg_at_10']:.5f}`",
        "",
        "## 2. Quota100_50 Validation Slate Coverage (Frozen 301D MLP Selector)",
        f"- Validation queries: `{slate_cov['queries']}` (`2025-10-01..2026-02-18`), total targets: `{slate_cov['total_targets']}`, repeat targets: `{slate_cov['repeat_targets']}`, new targets: `{slate_cov['new_targets']}`",
        f"- **Overall targets in slate**: `{slate_cov['total_targets_in_slate']}/{slate_cov['total_targets']}` (`{_fmt_pct(slate_cov['mixed_coverage_micro'])}` micro, `{_fmt_pct(slate_cov['mixed_coverage_macro'])}` macro)",
        f"- **Repeat targets in slate**: `{slate_cov['repeat_targets_in_slate']}/{slate_cov['repeat_targets']}` (`{_fmt_pct(slate_cov['repeat_slate_coverage'])}`)",
        f"- **New targets in slate**: `{slate_cov['new_targets_in_slate']}/{slate_cov['new_targets']}` (`{_fmt_pct(slate_cov['new_slate_coverage'])}`)",
        "",
        "## 3. Main Validation Results",
        "",
        "### A. Discovery-Selected Checkpoints (Guardrail-Constrained)",
        "",
        "| Model / Arm | Ep | Qual | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | New R@10 | Cond New R@10 | Repeat R@10 (mic/mac) |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage 1 301D Teacher (full pool) | 5 | - | {_fmt_num(st1_ref['recall@5'])} | {_fmt_num(st1_ref['recall@10'])} | {_fmt_num(st1_ref['recall@25'])} | {_fmt_num(st1_ref['recall@50'])} | {_fmt_num(st1_ref['recall@100'])} | {_fmt_num(st1_ref['ndcg@10'])} | {_fmt_num(st1_ref['mrr'])} | {st1_ref['new_hits@10']}/{st1_ref['new_targets']} | {_fmt_pct(st1_ref['new_recall@10'])} | {_fmt_pct(st1_ref['conditional_new_recall@10'])} | {_fmt_pct(st1_ref['repeat_recall@10_micro'])} / {_fmt_pct(st1_ref['repeat_recall@10_macro'])} |",
        f"| Stage 1 301D Teacher (`quota100_50` order) | 5 | - | {_fmt_num(st1_slate['recall@5'])} | {_fmt_num(st1_slate['recall@10'])} | {_fmt_num(st1_slate['recall@25'])} | {_fmt_num(st1_slate['recall@50'])} | {_fmt_num(st1_slate['recall@100'])} | {_fmt_num(st1_slate['ndcg@10'])} | {_fmt_num(st1_slate['mrr'])} | {st1_slate['new_hits@10']}/{st1_slate['new_targets']} | {_fmt_pct(st1_slate['new_recall@10'])} | {_fmt_pct(st1_slate['conditional_new_recall@10'])} | {_fmt_pct(st1_slate['repeat_recall@10_micro'])} / {_fmt_pct(st1_slate['repeat_recall@10_macro'])} |",
        f"| Stage 2 Control (`score=0`) | {ctrl['discovery']['epoch']} | {ctrl['qualified']} | {_fmt_num(c_disc['recall@5'])} | {_fmt_num(c_disc['recall@10'])} | {_fmt_num(c_disc['recall@25'])} | {_fmt_num(c_disc['recall@50'])} | {_fmt_num(c_disc['recall@100'])} | {_fmt_num(c_disc['ndcg@10'])} | {_fmt_num(c_disc['mrr'])} | {c_disc['new_hits@10']}/{c_disc['new_targets']} | {_fmt_pct(c_disc['new_recall@10'])} | {_fmt_pct(c_disc['conditional_new_recall@10'])} | {_fmt_pct(c_disc['repeat_recall@10_micro'])} / {_fmt_pct(c_disc['repeat_recall@10_macro'])} |",
        f"| Stage 2 Treatment (`+301D MLP score`) | {trt['discovery']['epoch']} | {trt['qualified']} | {_fmt_num(t_disc['recall@5'])} | {_fmt_num(t_disc['recall@10'])} | {_fmt_num(t_disc['recall@25'])} | {_fmt_num(t_disc['recall@50'])} | {_fmt_num(t_disc['recall@100'])} | {_fmt_num(t_disc['ndcg@10'])} | {_fmt_num(t_disc['mrr'])} | {t_disc['new_hits@10']}/{t_disc['new_targets']} | {_fmt_pct(t_disc['new_recall@10'])} | {_fmt_pct(t_disc['conditional_new_recall@10'])} | {_fmt_pct(t_disc['repeat_recall@10_micro'])} / {_fmt_pct(t_disc['repeat_recall@10_macro'])} |",
        "",
        "### B. NDCG-Selected Checkpoints (`best_ndcg.pt`)",
        "",
        "| Model / Arm | Ep | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | New R@10 | Cond New R@10 | Repeat R@10 (mic/mac) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage 2 Control (`score=0`, `best_ndcg`) | {ctrl['ndcg']['epoch']} | {_fmt_num(c_ndcg['recall@5'])} | {_fmt_num(c_ndcg['recall@10'])} | {_fmt_num(c_ndcg['recall@25'])} | {_fmt_num(c_ndcg['recall@50'])} | {_fmt_num(c_ndcg['recall@100'])} | {_fmt_num(c_ndcg['ndcg@10'])} | {_fmt_num(c_ndcg['mrr'])} | {c_ndcg['new_hits@10']}/{c_ndcg['new_targets']} | {_fmt_pct(c_ndcg['new_recall@10'])} | {_fmt_pct(c_ndcg['conditional_new_recall@10'])} | {_fmt_pct(c_ndcg['repeat_recall@10_micro'])} / {_fmt_pct(c_ndcg['repeat_recall@10_macro'])} |",
        f"| Stage 2 Treatment (`+301D MLP score`, `best_ndcg`) | {trt['ndcg']['epoch']} | {_fmt_num(t_ndcg['recall@5'])} | {_fmt_num(t_ndcg['recall@10'])} | {_fmt_num(t_ndcg['recall@25'])} | {_fmt_num(t_ndcg['recall@50'])} | {_fmt_num(t_ndcg['recall@100'])} | {_fmt_num(t_ndcg['ndcg@10'])} | {_fmt_num(t_ndcg['mrr'])} | {t_ndcg['new_hits@10']}/{t_ndcg['new_targets']} | {_fmt_pct(t_ndcg['new_recall@10'])} | {_fmt_pct(t_ndcg['conditional_new_recall@10'])} | {_fmt_pct(t_ndcg['repeat_recall@10_micro'])} / {_fmt_pct(t_ndcg['repeat_recall@10_macro'])} |",
        "",
        "## 4. Paired Customer-Clustered 95% Bootstrap CIs (Treatment - Control)",
        "",
        "| Comparison | Metric | Point Delta | 95% CI Low | Median | 95% CI High |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for comp_name, comp in (("Discovery vs Discovery", ci_disc), ("NDCG vs NDCG", ci_ndcg)):
        for m_name in (
            "recall@5_delta",
            "recall@10_delta",
            "recall@25_delta",
            "recall@50_delta",
            "recall@100_delta",
            "ndcg@10_delta",
            "new_recall@10_delta",
        ):
            row = comp[m_name]
            lines.append(
                f"| {comp_name} | `{m_name}` | {row['point_delta']:+.5f} | {row['ci_95'][0]:+.5f} | {row['ci_95'][1]:+.5f} | {row['ci_95'][2]:+.5f} |"
            )

    lines.extend(["", "## 5. Epoch-by-Epoch Validation Trajectory", ""])
    lines.append("| Ep | Control Loss | Control R@10 | Control NDCG@10 | Control New@10 | Treatment Loss | Treatment R@10 | Treatment NDCG@10 | Treatment New@10 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ch, th in zip(ctrl["history"], trt["history"]):
        cm, tm = ch["metrics"], th["metrics"]
        lines.append(
            f"| {ch['epoch']} | {ch['train_loss']:.4f} | {cm['recall@10']:.5f} | {cm['ndcg@10']:.5f} | {cm['new_hits@10']} | "
            f"{th['train_loss']:.4f} | {tm['recall@10']:.5f} | {tm['ndcg@10']:.5f} | {tm['new_hits@10']} |"
        )

    init_parity = summary["initialization_parity"]
    t_diag_disc = trt["discovery"]["score_channel_diagnostics"]
    t_diag_ndcg = trt["ndcg"]["score_channel_diagnostics"]
    c_diag_disc = ctrl["discovery"]["score_channel_diagnostics"]
    adapter_par = summary["adapter_parity"]

    lines.extend(
        [
            "",
            "## 6. Mechanism, Observability & Parity Verification",
            f"- **Zero-extension initialization parity (425D `CandidateContextRanker` vs 426D `Stage2MLPScoreRanker`)**: max abs logit diff with actual score `{init_parity['max_abs_logit_diff_vs_425d_with_actual_score']:.2e}`, with zero score `{init_parity['max_abs_logit_diff_vs_425d_with_zero_score']:.2e}`, attn diff `{init_parity['max_abs_attn_diff_vs_425d_with_actual_score']:.2e}`.",
            f"- **Control exact score-invariance**: max abs val logit diff with vs without `mlp_score` = `{ctrl['discovery']['full_val_max_abs_logit_diff_vs_zero_score']:.1e}` (discovery) and `{ctrl['ndcg']['full_val_max_abs_logit_diff_vs_zero_score']:.1e}` (NDCG); `attention_score_weight_l2={c_diag_disc['attention_score_weight_l2']:.4f}`, `head_score_weight_l2={c_diag_disc['head_score_weight_l2']:.4f}`.",
            f"- **Treatment score observability (Discovery ckpt)**: `attention_score_weight_l2={t_diag_disc['attention_score_weight_l2']:.4f}` (`max_abs={t_diag_disc['attention_score_weight_max_abs']:.4f}`), `head_score_weight_l2={t_diag_disc['head_score_weight_l2']:.4f}` (`max_abs={t_diag_disc['head_score_weight_max_abs']:.4f}`); toggling `mlp_score` vs `0`: `max_abs_attn_diff={t_diag_disc['max_abs_attn_weights_diff_actual_vs_zero']:.5f}`, `max_abs_h_att_diff={t_diag_disc['max_abs_h_attended_diff_actual_vs_zero']:.5f}`, `max_abs_logit_diff={t_diag_disc['max_abs_logit_diff_actual_vs_zero']:.5f}` (full-val max abs logit diff `{trt['discovery']['full_val_max_abs_logit_diff_vs_zero_score']:.5f}`).",
            f"- **Treatment score observability (NDCG ckpt)**: `attention_score_weight_l2={t_diag_ndcg['attention_score_weight_l2']:.4f}`, `head_score_weight_l2={t_diag_ndcg['head_score_weight_l2']:.4f}`, `max_abs_attn_diff={t_diag_ndcg['max_abs_attn_weights_diff_actual_vs_zero']:.5f}`, `max_abs_logit_diff={t_diag_ndcg['max_abs_logit_diff_actual_vs_zero']:.5f}`.",
            f"- **Live request-time adapter vs sealed cache parity**: max abs `mlp_score` diff `{adapter_par['max_abs_mlp_score_diff_request_vs_cache']:.2e}`, max abs logit diff `{adapter_par['max_abs_logit_diff_request_vs_cache']:.2e}`; slate-order reversal diff `mlp_score={adapter_par['max_abs_mlp_score_diff_reversed_slate']:.2e}`, `logit={adapter_par['max_abs_logit_diff_reversed_slate']:.2e}`; NaN-padded invalid-candidate masking diff `mlp_score={adapter_par['max_abs_mlp_score_diff_with_nan_padding']:.2e}`, `logit={adapter_par['max_abs_logit_diff_with_nan_padding']:.2e}` (`masked_score_zero={adapter_par['masked_pad_mlp_score_zero']}`, `masked_logit_neginf={adapter_par['masked_pad_logit_neginf']}`).",
            f"- **Teacher checkpoint replay vs saved `val_scores.pt`**: max abs logit diff `{summary['teacher_replay_spot_check']['max_abs_logit_diff']:.2e}` across `{summary['teacher_replay_spot_check']['rows_checked']}` validation rows.",
            "",
            "## 7. History Cohort Breakdown (Discovery Checkpoints)",
            "",
            "| Cohort | Rows | New Targets | Control R@10 | Control New@10 | Treatment R@10 | Treatment New@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cohort_name in ("cold_0", "light_1_8", "warm_9_31", "truncated_32"):
        cc = c_disc["history_cohorts"][cohort_name]
        tc = t_disc["history_cohorts"][cohort_name]
        lines.append(
            f"| `{cohort_name}` | {cc['rows']} | {cc['new_targets']} | {_fmt_num(cc['recall@10'])} | {cc['new_hits@10']} | {_fmt_num(tc['recall@10'])} | {tc['new_hits@10']} |"
        )

    lines.extend(
        [
            "",
            "## 8. Methodological Disclosures & Scope Guardrails",
            "- **In-sample teacher score caveat**: `train_scores.pt` predictions were generated by the frozen `301D` MLP selector trained on the `< 2025-10-01` training split, so training-split MLP scores can appear sharper/better calibrated than out-of-sample validation MLP scores; Stage 2 can partially over-rely on `mlp_score` unless out-of-fold validation scores are evaluated in future work.",
            "- **Teacher feature transform**: `mlp_score = sigmoid(raw_logit)` is a bounded `[0, 1]` float32 feature (with `-1e9` / ineligible positions masked to `0.0` before sigmoid), **not** a calibrated probability claim.",
            "- **Metadata policy**: `snapshot_unverified` — catalog attributes (`product_family`, `brand`, etc.) reflect current snapshot metadata.",
            "- **Single-seed conditional CIs**: Results are for a single `seed=42` fit on fixed `quota100_50` slates; paired customer-clustered 95% CIs capture query/customer sampling uncertainty given this single seed and frozen Stage 1 slate, not multi-seed variance.",
            "- **Promotion & held-out status**: `heldout_access = false`; held-out split (`>= 2026-02-19`) was never touched. No Stage 2 model was promoted and all historical/production aliases (`checkpoints/ranker/`, `repeat-no-month-v1`) remain byte-identical.",
            "",
            "## 9. Preserved Hashes & Reproduction",
            f"- Preserved reference/upstream hashes verified identical before and after execution (`{len(summary['preserved_hashes_before'])}` files).",
            "- Re-run command: `./venv/bin/python -m recommender.experiments.stage2_mlp_score.run`",
            "- Verification tests: `PYTHONPATH=. ./venv/bin/pytest tests/test_stage2_mlp_score.py` and `PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only`",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage2_mlp_score_experiment(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete controlled Stage 2 MLP-score experiment on development splits only."""
    hashes_before = snapshot_preserved_hashes()
    campaign_data, info, meta = load_verified_stage2_campaign_data()
    slates, slate_manifest = prepare_or_load_stage2_slate_cache(
        campaign_data,
        info,
        meta,
        cache_root=cache_root,
    )

    product_vectors = campaign_data["train"]["v"]
    init_parity, init_model = verify_zero_extension_initialization_parity(
        slates["val"],
        product_vectors,
        seed=seed,
    )

    control_res = fit_single_stage2_run(
        arm_name="control-no_score",
        score_channel_enabled=False,
        init_model=init_model,
        train_slate=slates["train"],
        val_slate=slates["val"],
        val_data=campaign_data["val"],
        product_vectors=product_vectors,
        meta=meta,
        slate_manifest=slate_manifest,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    treatment_res = fit_single_stage2_run(
        arm_name="treatment-mlp_score",
        score_channel_enabled=True,
        init_model=init_model,
        train_slate=slates["train"],
        val_slate=slates["val"],
        val_data=campaign_data["val"],
        product_vectors=product_vectors,
        meta=meta,
        slate_manifest=slate_manifest,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    paired_disc = paired_stage2_intervals(
        control_res["discovery"]["predictions_npz"],
        treatment_res["discovery"]["predictions_npz"],
    )
    paired_ndcg = paired_stage2_intervals(
        control_res["ndcg"]["predictions_npz"],
        treatment_res["ndcg"]["predictions_npz"],
    )

    adapter_parity = verify_cache_vs_request_adapter_parity(
        teacher=meta["reference"]["teacher"],
        teacher_spec=meta["reference"]["teacher_spec"],
        treatment_ckpt_path=Path(treatment_res["discovery"]["checkpoint"]),
        val_data=campaign_data["val"],
        val_slate=slates["val"],
        provenance=treatment_res["provenance"],
        teacher_contract=meta["reference"]["teacher_input_contract"],
    )

    stage1_teacher_slate_metrics = evaluate_stage2_slate_metrics(
        slates["val"]["mlp_score"],
        campaign_data["val"],
        slates["val"]["candidate_indices"],
    )
    st1_manifest_metrics = meta["reference"]["manifest"]["validation_metrics"]["selector_uv45"]

    hashes_after = snapshot_preserved_hashes()
    if hashes_after != hashes_before:
        raise AssertionError("Frozen upstream/reference file hash changed during Stage 2 run!")

    summary: Dict[str, Any] = {
        "experiment": "stage2_mlp_score_v1",
        "heldout_access": False,
        "promoted": False,
        "metadata_policy": "snapshot_unverified",
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "slate_policy": SLATE_POLICY,
        "score_feature_name": SCORE_FEATURE_NAME,
        "score_transform_version": SCORE_TRANSFORM_VERSION,
        "reference": {
            "active_reference_id": meta["reference"]["reference_pointer"]["active_reference_id"],
            "hashes": meta["reference"]["hashes"],
            "teacher_contract_digest": meta["reference"]["teacher_input_contract"]["contract_digest"],
        },
        "teacher_replay_spot_check": meta["teacher_replay_spot_check"],
        "cache_lineage": meta["cache_lineage"],
        "slate_cache_manifest": slate_manifest,
        "slate_coverage_val": {
            "queries": stage1_teacher_slate_metrics["queries"],
            "total_targets": stage1_teacher_slate_metrics["total_targets"],
            "repeat_targets": stage1_teacher_slate_metrics["repeat_targets"],
            "new_targets": stage1_teacher_slate_metrics["new_targets"],
            "total_targets_in_slate": stage1_teacher_slate_metrics["total_targets_in_slate"],
            "mixed_coverage_micro": stage1_teacher_slate_metrics["mixed_coverage_micro"],
            "mixed_coverage_macro": stage1_teacher_slate_metrics["mixed_coverage_macro"],
            "repeat_targets_in_slate": stage1_teacher_slate_metrics["repeat_targets_in_slate"],
            "repeat_slate_coverage": stage1_teacher_slate_metrics["repeat_slate_coverage"],
            "new_targets_in_slate": stage1_teacher_slate_metrics["new_targets_in_slate"],
            "new_slate_coverage": stage1_teacher_slate_metrics["new_slate_coverage"],
        },
        "guardrails": {
            "recall_at_10": control_res["settings"]["recall_guardrail"],
            "ndcg_at_10": control_res["settings"]["ndcg_guardrail"],
            "artifact_sha256": control_res["settings"]["guardrail_artifact_sha256"],
        },
        "baselines": {
            "stage1_301d_full_pool": st1_manifest_metrics,
            "stage1_301d_on_val_slate": stage1_teacher_slate_metrics,
        },
        "initialization_parity": init_parity,
        "arms": {
            "control": control_res,
            "treatment": treatment_res,
        },
        "paired_intervals": {
            "discovery_vs_discovery": paired_disc,
            "ndcg_vs_ndcg": paired_ndcg,
        },
        "adapter_parity": adapter_parity,
        "preserved_hashes_before": hashes_before,
        "preserved_hashes_after": hashes_after,
    }

    ev_root = Path(evidence_root)
    ev_root.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, ev_root / "summary.json", overwrite=True)
    report_path = ev_root / "REPORT.md"
    tmp_report = ev_root / ".REPORT.md.tmp"
    tmp_report.write_text(_build_markdown_report(summary), encoding="utf-8")
    tmp_report.replace(report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    summary = run_stage2_mlp_score_experiment(
        artifact_root=args.artifact_root,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "control_disc": {
                    "epoch": summary["arms"]["control"]["discovery"]["epoch"],
                    "qualified": summary["arms"]["control"]["qualified"],
                    "recall@10": summary["arms"]["control"]["discovery"]["metrics"]["recall@10"],
                    "ndcg@10": summary["arms"]["control"]["discovery"]["metrics"]["ndcg@10"],
                    "new_hits@10": summary["arms"]["control"]["discovery"]["metrics"]["new_hits@10"],
                },
                "treatment_disc": {
                    "epoch": summary["arms"]["treatment"]["discovery"]["epoch"],
                    "qualified": summary["arms"]["treatment"]["qualified"],
                    "recall@10": summary["arms"]["treatment"]["discovery"]["metrics"]["recall@10"],
                    "ndcg@10": summary["arms"]["treatment"]["discovery"]["metrics"]["ndcg@10"],
                    "new_hits@10": summary["arms"]["treatment"]["discovery"]["metrics"]["new_hits@10"],
                },
                "control_ndcg": {
                    "epoch": summary["arms"]["control"]["ndcg"]["epoch"],
                    "recall@10": summary["arms"]["control"]["ndcg"]["metrics"]["recall@10"],
                    "ndcg@10": summary["arms"]["control"]["ndcg"]["metrics"]["ndcg@10"],
                    "new_hits@10": summary["arms"]["control"]["ndcg"]["metrics"]["new_hits@10"],
                },
                "treatment_ndcg": {
                    "epoch": summary["arms"]["treatment"]["ndcg"]["epoch"],
                    "recall@10": summary["arms"]["treatment"]["ndcg"]["metrics"]["recall@10"],
                    "ndcg@10": summary["arms"]["treatment"]["ndcg"]["metrics"]["ndcg@10"],
                    "new_hits@10": summary["arms"]["treatment"]["ndcg"]["metrics"]["new_hits@10"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
