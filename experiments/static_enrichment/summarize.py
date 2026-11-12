"""Aggregate Stage 1, cohort, and downstream Stage 2 metrics with paired customer-clustered intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.dynamic_selector.evaluate import paired_intervals
from recommender.experiments.static_enrichment.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.experiments.static_enrichment.evaluate import paired_stage1_intervals
from experiments.dynamic_campaign.summarize import nested_intervals


def _agg_mean_sd(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values,
    }


def main() -> None:
    raw = json.loads((EVIDENCE_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    enrichment_manifest = json.loads(
        (Path("data/interim/static_enrichment_v1/enrichment_manifest.json")).read_text(encoding="utf-8")
    )

    stage1_by_var = defaultdict(list)
    for key, res in raw["stage1"].items():
        stage1_by_var[res["settings"]["variant"]].append(res)
    for rows in stage1_by_var.values():
        rows.sort(key=lambda r: r["settings"]["seed"])

    stage1_summary = {}
    for var, rows in stage1_by_var.items():
        entry = {
            "n": len(rows),
            "seeds": [r["settings"]["seed"] for r in rows],
            "selected_epochs": [r["selected_epoch"] for r in rows],
            "parameters": rows[0]["parameters"],
            "training_seconds": _agg_mean_sd([r["training_seconds"] for r in rows]),
            "text_recipe": rows[0]["settings"]["spec"]["text_recipe"],
            "text_recipe_hash": rows[0]["settings"]["text_recipe_hash"],
            "metrics": {},
            "cohorts": {},
        }
        for m in (
            "new_hits@25",
            "stage1_new_recall@25",
            "new_hits@50",
            "stage1_new_recall@50",
            "val_recall@10",
            "val_ndcg@10",
            "new_hits@10",
            "new_recall@10",
        ):
            entry["metrics"][m] = _agg_mean_sd([r["metrics"][m] for r in rows])

        cohort_names = list(rows[0]["cohorts"].keys())
        for cname in cohort_names:
            entry["cohorts"][cname] = {
                "targets": rows[0]["cohorts"][cname]["targets"],
                "hits@25": _agg_mean_sd([r["cohorts"][cname]["hits@25"] for r in rows]),
                "recall@25": _agg_mean_sd([r["cohorts"][cname]["recall@25"] for r in rows if r["cohorts"][cname]["recall@25"] is not None]),
                "hits@50": _agg_mean_sd([r["cohorts"][cname]["hits@50"] for r in rows]),
                "recall@50": _agg_mean_sd([r["cohorts"][cname]["recall@50"] for r in rows if r["cohorts"][cname]["recall@50"] is not None]),
            }
        stage1_summary[var] = entry

    stage1_paired = {}
    c0_by_seed = {r["settings"]["seed"]: r for r in stage1_by_var.get("C0", [])}
    for var, rows in stage1_by_var.items():
        if var == "C0":
            continue
        seeds = [r["settings"]["seed"] for r in rows]
        if all(s in c0_by_seed for s in seeds):
            ctrl_paths = [Path(c0_by_seed[s]["selected_checkpoint"]).parent / "stage1_predictions.npz" for s in seeds]
            trt_paths = [Path(r["selected_checkpoint"]).parent / "stage1_predictions.npz" for r in rows]
            stage1_paired[var] = {
                "seeds": seeds,
                "nested_vs_C0": paired_stage1_intervals(ctrl_paths, trt_paths),
                "per_seed_vs_C0": {
                    str(s): paired_stage1_intervals([c_p], [t_p])
                    for s, c_p, t_p in zip(seeds, ctrl_paths, trt_paths)
                },
            }

    ds_by_cfg = defaultdict(list)
    for key, res in raw["downstream"].items():
        cfg = f"{res['settings']['upstream_variant']}-dot100-{res['settings']['mode']}"
        ds_by_cfg[cfg].append(res)
    for rows in ds_by_cfg.values():
        rows.sort(key=lambda r: r["settings"]["seed"])

    downstream_summary = {}
    for cfg, rows in ds_by_cfg.items():
        entry = {
            "n": len(rows),
            "seeds": [r["settings"]["seed"] for r in rows],
            "epochs": [r["epoch"] for r in rows],
            "qualified_count": sum(1 for r in rows if r["qualified"]),
            "qualified": [r["qualified"] for r in rows],
            "parameters": rows[0]["parameters"],
            "trainable_parameters": rows[0]["trainable_parameters"],
            "training_seconds": _agg_mean_sd([r["training_seconds"] for r in rows]),
            "metrics": {},
        }
        for m in (
            "new_hits@10",
            "new_recall@10",
            "recall@10",
            "ndcg@10",
            "conditional_new_recall@10",
            "repeat_recall@10_micro",
            "new_targets_in_slate",
        ):
            entry["metrics"][m] = _agg_mean_sd([r["metrics"][m] for r in rows])
        downstream_summary[cfg] = entry

    downstream_paired = {}
    for mode in ("control", "dynamic_features"):
        c0_cfg = f"C0-dot100-{mode}"
        if c0_cfg not in ds_by_cfg:
            continue
        c0_rows = {r["settings"]["seed"]: r for r in ds_by_cfg[c0_cfg]}
        for cfg, rows in ds_by_cfg.items():
            if cfg == c0_cfg or not cfg.endswith(f"-dot100-{mode}"):
                continue
            seeds = [r["settings"]["seed"] for r in rows]
            if all(s in c0_rows for s in seeds):
                ctrl_preds = [
                    Path(c0_rows[s].get("selected_checkpoint") or c0_rows[s]["diagnostic_checkpoint"]).parent / "predictions.npz"
                    for s in seeds
                ]
                trt_preds = [
                    Path(r.get("selected_checkpoint") or r["diagnostic_checkpoint"]).parent / "predictions.npz"
                    for r in rows
                ]
                downstream_paired[cfg] = {
                    "control": c0_cfg,
                    "seeds": seeds,
                    "nested_vs_C0": nested_intervals(ctrl_preds, trt_preds),
                    "per_seed_vs_C0": {
                        str(s): paired_intervals(cp, tp)
                        for s, cp, tp in zip(seeds, ctrl_preds, trt_preds)
                    },
                }

    summary = {
        "decisions": raw["decisions"],
        "enrichment_diagnostics": enrichment_manifest["diagnostics"],
        "stage1": stage1_summary,
        "stage1_paired_vs_C0": stage1_paired,
        "downstream": downstream_summary,
        "downstream_paired_vs_C0": downstream_paired,
        "heldout_access": False,
        "promoted": False,
        "source_sha256": {
            str(p): file_digest(p)
            for p in Path("recommender/experiments/static_enrichment").glob("*.py")
        },
    }
    atomic_json(summary, EVIDENCE_ROOT / "summary.json")

    lines = [
        "# Static Enrichment Campaign Comparison",
        "",
        "All metrics computed strictly on the 6,191 validation transitions / 1,857 new-to-customer targets. No test-split access.",
        "",
        "## Stage 1 Retrieval (30 epochs, S0 demand-disabled protocol)",
        "",
        "| Variant | n | Params | Fit (s) | Unseen Hits@25 | Recall@25 | Unseen Hits@50 | Recall@50 | With-Desc Hits@25 (514) | No-Desc Hits@25 (1,343) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(d, dec=2):
        if d["std"] is None:
            return f"{d['mean']:.{dec}f}"
        return f"{d['mean']:.{dec}f} ± {d['std']:.{dec}f}"

    order = ["C0", "T1", "T2", "A", "F_emb", "F_txt", "P", "P_no_unit", "U0", "U1"]
    for var in order:
        if var not in stage1_summary:
            continue
        s = stage1_summary[var]
        m = s["metrics"]
        c = s["cohorts"]
        lines.append(
            f"| {var} | {s['n']} | {s['parameters']:,} | {fmt(s['training_seconds'], 1)} | "
            f"{fmt(m['new_hits@25'], 2)} | {m['stage1_new_recall@25']['mean']*100:.2f}% | "
            f"{fmt(m['new_hits@50'], 2)} | {m['stage1_new_recall@50']['mean']*100:.2f}% | "
            f"{fmt(c['with_description_nonempty']['hits@25'], 2)} | "
            f"{fmt(c['without_description_nonempty']['hits@25'], 2)} |"
        )

    if downstream_summary:
        lines.extend([
            "",
            "## Downstream Stage 2 (dot/100 quota100_50, 10 epochs, 3 seeds)",
            "",
            "| Config | n | Qual | Params (Trainable) | Fit (s) | New Hits@10 | Recall@10 | NDCG@10 | Slate New Targets |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cfg, d in downstream_summary.items():
            m = d["metrics"]
            lines.append(
                f"| {cfg} | {d['n']} | {d['qualified_count']}/{d['n']} | "
                f"{d['parameters']:,} ({d['trainable_parameters']:,}) | {fmt(d['training_seconds'], 1)} | "
                f"{fmt(m['new_hits@10'], 2)} | {fmt(m['recall@10'], 6)} | {fmt(m['ndcg@10'], 6)} | "
                f"{fmt(m['new_targets_in_slate'], 1)} |"
            )

    (EVIDENCE_ROOT / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote summary.json and COMPARISON.md to", EVIDENCE_ROOT, flush=True)


def summarize_clean_v2() -> None:
    raw = json.loads((EVIDENCE_ROOT / "clean_v2_results.json").read_text(encoding="utf-8"))
    best_clean = raw["decisions"]["best_clean"]
    seed = raw["decisions"]["seed"]

    stage1_summary = {}
    for key, res in raw["stage1"].items():
        var = res["settings"]["variant"]
        stage1_summary[var] = {
            "seed": res["settings"]["seed"],
            "selected_epoch": res["selected_epoch"],
            "parameters": res["parameters"],
            "trainable_parameters": res["trainable_parameters"],
            "training_seconds": res["training_seconds"],
            "text_recipe": res["settings"]["spec"]["text_recipe"],
            "customer_tabular_features": res["settings"]["spec"]["customer_tabular_features"],
            "product_demand_features": res["settings"]["spec"]["product_demand_features"],
            "metrics": res["metrics"],
            "cohorts": res["cohorts"],
            "selected_checkpoint": res["selected_checkpoint"],
        }

    c0_pred = Path(stage1_summary["C0_clean"]["selected_checkpoint"]).parent / "stage1_predictions.npz"
    stage1_paired_vs_c0 = {}
    for var, entry in stage1_summary.items():
        if var == "C0_clean":
            continue
        trt_pred = Path(entry["selected_checkpoint"]).parent / "stage1_predictions.npz"
        stage1_paired_vs_c0[var] = paired_stage1_intervals([c0_pred], [trt_pred])

    stage1_paired_u0_vs_nosubcat = {}
    if "U0_t1_fam_clean" in stage1_summary and "U0_t1_fam_nosubcat" in stage1_summary:
        p_sub = Path(stage1_summary["U0_t1_fam_clean"]["selected_checkpoint"]).parent / "stage1_predictions.npz"
        p_nosub = Path(stage1_summary["U0_t1_fam_nosubcat"]["selected_checkpoint"]).parent / "stage1_predictions.npz"
        stage1_paired_u0_vs_nosubcat["U0_t1_fam_clean_vs_nosubcat"] = paired_stage1_intervals([p_nosub], [p_sub])

    ds_summary = {}
    for key, res in raw["downstream"].items():
        cfg = f"{res['settings']['upstream_variant']}-{res['settings']['mode']}"
        ds_summary[cfg] = {
            "seed": res["settings"]["seed"],
            "epoch": res["epoch"],
            "qualified": res["qualified"],
            "parameters": res["parameters"],
            "trainable_parameters": res["trainable_parameters"],
            "training_seconds": res["training_seconds"],
            "metrics": res["metrics"],
            "selected_checkpoint": res.get("selected_checkpoint"),
            "diagnostic_checkpoint": res.get("diagnostic_checkpoint"),
        }

    def _pred_path(cfg: str) -> Path:
        ckpt = ds_summary[cfg].get("selected_checkpoint") or ds_summary[cfg]["diagnostic_checkpoint"]
        return Path(ckpt).parent / "predictions.npz"

    ds_paired = {}
    c0_cfg = "C0_clean-dot100_41"
    best_41 = f"{best_clean}-dot100_41"
    best_47 = f"{best_clean}-dot100_47"
    best_mlp = f"{best_clean}-precomputed_mlp_47"
    for trt, ctrl in (
        (best_41, c0_cfg),
        (best_47, c0_cfg),
        (best_47, best_41),
        (best_mlp, c0_cfg),
        (best_mlp, best_47),
    ):
        if trt in ds_summary and ctrl in ds_summary:
            ds_paired[f"{trt}_vs_{ctrl}"] = paired_intervals(_pred_path(ctrl), _pred_path(trt))

    summary = {
        "decisions": raw["decisions"],
        "stage1": stage1_summary,
        "stage1_paired_vs_C0_clean": stage1_paired_vs_c0,
        "stage1_paired_ablations": stage1_paired_u0_vs_nosubcat,
        "downstream": ds_summary,
        "downstream_paired": ds_paired,
        "heldout_access": False,
        "promoted": False,
        "seed": seed,
        "source_sha256": {
            str(p): file_digest(p)
            for p in Path("recommender/experiments/static_enrichment").glob("*.py")
        },
    }
    atomic_json(summary, EVIDENCE_ROOT / "clean_v2_summary.json")

    lines = [
        "# Clean Static Towers + Transferred 6-Stat Scoring MLP (seed=42)",
        "",
        "All metrics computed strictly on the 6,191 validation transitions / 1,857 new-to-customer targets. No test-split access.",
        "",
        "## Stage 1 Clean Towers (0 customer tabular stats, 12D container_type only, 30 epochs, seed=42)",
        "",
        "| Variant | Ep | Params | Fit (s) | Unseen Hits@25 | Recall@25 | Unseen Hits@50 | Recall@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for var in ("C0_clean", "P_t1_fam_clean", "U0_t1_fam_clean", "U0_t1_fam_nosubcat"):
        if var not in stage1_summary:
            continue
        s = stage1_summary[var]
        m = s["metrics"]
        lines.append(
            f"| {var} | {s['selected_epoch']} | {s['parameters']:,} | {s['training_seconds']:.1f} | "
            f"{m['new_hits@25']} | {m['stage1_new_recall@25']*100:.2f}% | "
            f"{m['new_hits@50']} | {m['stage1_new_recall@50']*100:.2f}% |"
        )

    lines.extend([
        "",
        f"Selected `best_clean = {best_clean}`.",
        "",
        "## Downstream Stage 2 (quota100_50, 10 epochs, seed=42)",
        "",
        "| Config | Ep | Qual | Params (Trainable) | Fit (s) | New Hits@10 | New Recall@10 | Recall@10 | NDCG@10 | Slate New Targets |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cfg in (c0_cfg, best_41, best_47, best_mlp):
        if cfg not in ds_summary:
            continue
        d = ds_summary[cfg]
        m = d["metrics"]
        lines.append(
            f"| {cfg} | {d['epoch']} | {d['qualified']} | {d['parameters']:,} ({d['trainable_parameters']:,}) | "
            f"{d['training_seconds']:.1f} | {m['new_hits@10']} | {m['new_recall@10']*100:.2f}% | "
            f"{m['recall@10']:.6f} | {m['ndcg@10']:.6f} | {m['new_targets_in_slate']} |"
        )

    (EVIDENCE_ROOT / "clean_v2_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote clean_v2_summary.json and clean_v2_COMPARISON.md to", EVIDENCE_ROOT, flush=True)


def summarize_stage1_mlp_v3() -> None:
    raw = json.loads((EVIDENCE_ROOT / "stage1_mlp_v3_results.json").read_text(encoding="utf-8"))
    v2_raw_path = EVIDENCE_ROOT / "clean_v2_results.json"
    v2_raw = json.loads(v2_raw_path.read_text(encoding="utf-8")) if v2_raw_path.is_file() else {}
    seed = 42

    c0_v2 = v2_raw.get("downstream", {}).get(f"C0_clean-dot100_41-seed{seed}")
    c0_pred = (
        Path(c0_v2["selected_checkpoint"]).parent / "predictions.npz"
        if c0_v2 and c0_v2.get("selected_checkpoint")
        else None
    )

    def _pred_pair(cfg: str, kind: str = "discovery") -> Path:
        d = raw["downstream"][cfg]
        if kind == "ndcg":
            return Path(d["ndcg_selection"]["predictions"])
        return Path(d["discovery_selection"]["predictions"])

    paired_discovery = {}
    paired_ndcg = {}
    u0_dot_cfg = "U0-dot100_41"
    for cfg in raw["downstream"]:
        if c0_pred is not None and c0_pred.is_file():
            paired_discovery[f"{cfg}_vs_C0_clean-dot100_41"] = paired_intervals(c0_pred, _pred_pair(cfg, "discovery"))
            paired_ndcg[f"{cfg}_vs_C0_clean-dot100_41"] = paired_intervals(c0_pred, _pred_pair(cfg, "ndcg"))
        if cfg != u0_dot_cfg and u0_dot_cfg in raw["downstream"]:
            paired_discovery[f"{cfg}_vs_U0-dot100_41"] = paired_intervals(
                _pred_pair(u0_dot_cfg, "discovery"), _pred_pair(cfg, "discovery")
            )
            paired_ndcg[f"{cfg}_vs_U0-dot100_41"] = paired_intervals(
                _pred_pair(u0_dot_cfg, "ndcg"), _pred_pair(cfg, "ndcg")
            )

    summary = {
        "seed": seed,
        "heldout_access": False,
        "promoted": False,
        "stage1_clean_towers": {
            var: {
                "selected_epoch": res["selected_epoch"],
                "parameters": res["parameters"],
                "training_seconds": res["training_seconds"],
                "customer_embeddings": list(res["settings"]["spec"]["customer_embeddings"]),
                "customer_tabular_features": res["settings"]["spec"]["customer_tabular_features"],
                "product_demand_features": res["settings"]["spec"]["product_demand_features"],
                "full_metrics": res["metrics"],
                "dot_baseline": raw["dot_baseline"].get(var),
                "selected_checkpoint": res["selected_checkpoint"],
            }
            for var, res in raw["stage1"].items()
        },
        "compact_selectors": raw["selectors"],
        "downstream": raw["downstream"],
        "clean_v2_c0_reference": c0_v2,
        "downstream_paired_discovery": paired_discovery,
        "downstream_paired_ndcg": paired_ndcg,
        "latency": raw.get("latency", {}),
        "note_on_intervals": "95% CIs are customer-clustered bootstrap intervals within seed=42 (query-sampling uncertainty only, not multi-seed variance).",
        "source_sha256": {
            str(p): file_digest(p)
            for p in Path("recommender/experiments/static_enrichment").glob("*.py")
        },
    }
    atomic_json(summary, EVIDENCE_ROOT / "stage1_mlp_v3_summary.json")

    lines = [
        "# Compact Stage 1 Selector (`uv_47` / `uv_47_ref`) & Stage 2 Isolation Report (`seed=42`)",
        "",
        "All metrics computed strictly on validation transitions (`< 2026-02-19`, `6,191` transitions, `1,857` new targets). Zero test-split access.",
        "",
        "## 1. Stage 1 Full-Pool Ranking & `quota100_50` Slate Coverage",
        "",
        "| Stage 1 Scorer | Input Dim | Ep | Sales Branch | Unseen@25 | Unseen@50 | Recall@10 | NDCG@10 | Slate New Targets (Cov) | Slate Overall Cov |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for var, label in (
        ("C0_clean", "C0_clean + dot"),
        ("P_t1_fam_clean", "P_t1_fam_clean + dot (NO sales_branch)"),
        ("U0_t1_fam_clean", "U0_t1_fam_clean + dot (WITH sales_branch)"),
    ):
        if var in raw["stage1"]:
            s = raw["stage1"][var]
            db = raw["dot_baseline"][var]
            fp = db["full_pool"]
            cov = db["quota100_50_coverage"]
            sb = "Yes (16D)" if s["settings"]["spec"]["customer_embeddings"] else "No"
            lines.append(
                f"| {label} | 128D dot | {s['selected_epoch']} | {sb} | "
                f"{fp['new_hits@25']} | {fp['new_hits@50']} | {fp['recall@10']:.5f} | {fp['ndcg@10']:.5f} | "
                f"{cov['new_targets_in_slate']} ({cov['new_slate_coverage']*100:.2f}%) | {cov['overall_slate_coverage']*100:.2f}% |"
            )
    for key, label in (
        ("U0_t1_fam_clean-uv_47", "U0 + CompactStage1Selector (`uv_47`)"),
        ("U0_t1_fam_clean-uv_47_ref", "U0 + CompactStage1Selector (`uv_47_ref`)"),
    ):
        if key in raw["selectors"]:
            sr = raw["selectors"][key]
            fp = sr["metrics"]
            cov = sr["quota100_50_coverage"]
            dim = sr["settings"]["input_contract"]["input_dim"]
            lines.append(
                f"| {label} | {dim}D MLP | {sr['selected_epoch']} | Yes (16D in u) | "
                f"{fp['new_hits@25']} | {fp['new_hits@50']} | {fp['recall@10']:.5f} | {fp['ndcg@10']:.5f} | "
                f"{cov['new_targets_in_slate']} ({cov['new_slate_coverage']*100:.2f}%) | {cov['overall_slate_coverage']*100:.2f}% |"
            )

    lines.extend([
        "",
        "## 2. Stage 2 End-to-End `UPS-CANR` Results (Discovery vs NDCG Checkpoints)",
        "",
        "| Config | Sel | Ep (Disc/NDCG) | R@5 (Disc) | R@10 (Disc) | NDCG@10 (Disc) | Repeat R@10 | New@10 (Disc) | Cond New (Disc) | Slate New | R@10 (NDCG) | NDCG@10 (NDCG) | New@10 (NDCG) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if c0_v2:
        cm = c0_v2["metrics"]
        lines.append(
            f"| C0_clean-dot100_41 (v2 ref) | disc=ndcg | {c0_v2['epoch']} / {c0_v2['epoch']} | "
            f"{cm['recall@5']:.5f} | {cm['recall@10']:.5f} | {cm['ndcg@10']:.5f} | {cm['repeat_recall@10_micro']:.5f} | "
            f"{cm['new_hits@10']} ({cm['new_recall@10']*100:.2f}%) | {cm['conditional_new_recall@10']*100:.2f}% | "
            f"{cm['new_targets_in_slate']} | {cm['recall@10']:.5f} | {cm['ndcg@10']:.5f} | {cm['new_hits@10']} |"
        )
    for cfg, d in raw["downstream"].items():
        dm = d["discovery_selection"]["metrics"]
        nm = d["ndcg_selection"]["metrics"]
        de = d["discovery_selection"]["epoch"]
        ne = d["ndcg_selection"]["epoch"]
        lines.append(
            f"| {cfg} | {d['settings']['selector_type']} | {de} / {ne} | "
            f"{dm['recall@5']:.5f} | {dm['recall@10']:.5f} | {dm['ndcg@10']:.5f} | {dm['repeat_recall@10_micro']:.5f} | "
            f"{dm['new_hits@10']} ({dm['new_recall@10']*100:.2f}%) | {dm['conditional_new_recall@10']*100:.2f}% | "
            f"{dm['new_targets_in_slate']} | {nm['recall@10']:.5f} | {nm['ndcg@10']:.5f} | {nm['new_hits@10']} |"
        )

    lines.extend([
        "",
        "## 3. Warm Single-Request Latency (`U0_t1_fam_clean` Cached Vectors -> 100-Item Slate -> `UPS-CANR`)",
        "",
        "| Device | Pipeline | E2E Median (ms) | E2E p95 (ms) | Selector (ms) | Select100 (ms) | Ranker (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for dev_name, pipes in raw.get("latency", {}).items():
        for pname, vals in pipes.items():
            lines.append(
                f"| {dev_name} | {pname} | {vals['end_to_end_median_ms']:.3f} | {vals['end_to_end_p95_ms']:.3f} | "
                f"{vals['selector_median_ms']:.3f} | {vals['selection_median_ms']:.3f} | {vals['ranker_median_ms']:.3f} |"
            )

    (EVIDENCE_ROOT / "stage1_mlp_v3_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote stage1_mlp_v3_summary.json and stage1_mlp_v3_COMPARISON.md to", EVIDENCE_ROOT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("v1", "clean_v2", "stage1_mlp_v3"), default="v1")
    args = parser.parse_args()
    if args.campaign == "clean_v2":
        summarize_clean_v2()
    elif args.campaign == "stage1_mlp_v3":
        summarize_stage1_mlp_v3()
    else:
        main()
