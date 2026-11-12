"""Aggregate Stage 1, cohort, and downstream Stage 2 metrics with paired customer-clustered intervals."""

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


if __name__ == "__main__":
    main()
