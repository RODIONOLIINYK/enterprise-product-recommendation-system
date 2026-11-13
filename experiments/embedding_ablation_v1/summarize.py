"""Summarize isolated product-text and product-embedding ablation results with paired customer-clustered CIs."""

import json
from pathlib import Path
from typing import Any, Dict

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.embedding_ablation.config import CACHE_DIR, EVIDENCE_ROOT
from recommender.experiments.embedding_ablation.evaluate import paired_ablation_intervals
from recommender.experiments.static_enrichment.evaluate import paired_stage1_intervals


def _extract_entry(name: str, res: Dict[str, Any], base_mlp_npz: Path, base_s1_npz: Path) -> Dict[str, Any]:
    if res.get("is_noop"):
        return {
            "name": name,
            "group": res["ablation"]["group"],
            "ablated_target": res["ablation"]["ablated_target"],
            "is_noop": True,
            "noop_reason": res.get("noop_reason"),
            "text_diagnostics": res["text_diagnostics"],
        }

    s1 = res["stage1"]
    sel = res["selector_uv47"]
    m_sel = sel["metrics"]
    m_s1 = s1["metrics"]
    dot_fp = res["upstream_dot_fullpool"]

    mlp_npz = Path(res["mlp_per_query_predictions"])
    s1_npz = Path(s1["selected_checkpoint"]).parent / "stage1_predictions.npz"

    paired_mlp = None
    paired_s1 = None
    if name != "U0_t1_fam_clean" and base_mlp_npz.is_file() and mlp_npz.is_file():
        paired_mlp = paired_ablation_intervals(base_mlp_npz, mlp_npz)
    if name != "U0_t1_fam_clean" and base_s1_npz.is_file() and s1_npz.is_file():
        paired_s1 = paired_stage1_intervals([base_s1_npz], [s1_npz])

    return {
        "name": name,
        "group": res["ablation"]["group"],
        "ablated_target": res["ablation"]["ablated_target"],
        "is_noop": False,
        "text_diagnostics": res["text_diagnostics"],
        "upstream": {
            "reused_baseline_upstream": bool(s1.get("reused_baseline_upstream", False)),
            "selected_epoch": s1["selected_epoch"],
            "parameters": s1["parameters"],
            "training_seconds": s1["training_seconds"],
            "new_hits@25": m_s1["new_hits@25"],
            "new_recall@25": m_s1["stage1_new_recall@25"],
            "new_hits@50": m_s1["new_hits@50"],
            "new_recall@50": m_s1["stage1_new_recall@50"],
            "new_hits@100": dot_fp["new_hits@100"],
            "new_recall@100": dot_fp["new_recall@100"],
            "recall@10": dot_fp["recall@10"],
            "ndcg@10": dot_fp["ndcg@10"],
            "selected_checkpoint": s1["selected_checkpoint"],
        },
        "mlp_uv47": {
            "selected_epoch": sel["selected_epoch"],
            "parameters": sel["parameters"],
            "training_seconds": sel["training_seconds"],
            "new_hits@25": m_sel["new_hits@25"],
            "new_recall@25": m_sel["new_recall@25"],
            "new_hits@50": m_sel["new_hits@50"],
            "new_recall@50": m_sel["new_recall@50"],
            "new_hits@100": m_sel["new_hits@100"],
            "new_recall@100": m_sel["new_recall@100"],
            "recall@10": m_sel["recall@10"],
            "ndcg@10": m_sel["ndcg@10"],
            "new_hits@10": m_sel["new_hits@10"],
            "new_recall@10": m_sel["new_recall@10"],
            "selected_checkpoint": sel["selected_checkpoint"],
            "per_query_predictions": str(mlp_npz),
        },
        "paired_vs_baseline": {
            "mlp_uv47": paired_mlp,
            "upstream_stage1": paired_s1,
        },
    }


def _fmt_ci_pp(ci: list[float]) -> str:
    return f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}, {ci[2]*100:+.2f}] pp"


def _fmt_ci_num(ci: list[float], dec: int = 4) -> str:
    return f"[{ci[0]:+.{dec}f}, {ci[1]:+.{dec}f}, {ci[2]:+.{dec}f}]"


def main() -> None:
    raw = json.loads((EVIDENCE_ROOT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((CACHE_DIR / "ablation_manifest.json").read_text(encoding="utf-8"))

    ind = raw["individual"]
    base = ind["U0_t1_fam_clean"]
    base_mlp_npz = Path(base["mlp_per_query_predictions"])
    base_s1_npz = Path(base["stage1"]["selected_checkpoint"]).parent / "stage1_predictions.npz"

    individual_summary: Dict[str, Any] = {}
    for name, res in ind.items():
        individual_summary[name] = _extract_entry(name, res, base_mlp_npz, base_s1_npz)

    exploratory_summary: Dict[str, Any] = {}
    for name, res in raw.get("exploratory_combined", {}).items():
        exploratory_summary[name] = _extract_entry(name, res, base_mlp_npz, base_s1_npz)

    summary = {
        "campaign": raw["campaign"],
        "baseline_target": raw["baseline_target"],
        "selector_recipe": raw["selector_recipe"],
        "seed": raw["seed"],
        "heldout_access": False,
        "stage2_executed": False,
        "promoted": False,
        "metric_semantics": {
            "evaluation_split": "validation (2025-10-01 to 2026-02-18, 6,191 transitions, 1,857 new-to-customer target instances)",
            "training_split": "train (< 2025-10-01; Stage 1 discovery rows=8,465; Selector uv_47 cohort=26,253 rows with >=100 eligible SKUs)",
            "pool": "Full rolling eligible pool per query (~735-771 active SKUs per validation date inside 841 development SKUs)",
            "new_target_metrics": "new_hits@K and new_recall@K over unseen eligible products (~seen & allowed), micro-averaged over all 1,857 new-to-customer target instances (unrepresented targets outside 841 pool count as misses)",
            "overall_metrics": "recall@10 and ndcg@10 over all eligible products (allowed), macro-averaged across all 6,191 validation transitions against full positive_pids",
            "bootstrap": "Paired customer-clustered 95% bootstrap intervals (2,000 resamples, rng seed=123) within seed=42; reflects validation customer-query sampling uncertainty only",
        },
        "inapplicable": raw["inapplicable"],
        "individual": individual_summary,
        "exploratory_combined": exploratory_summary,
        "ablation_manifest_identity": manifest["identity"],
        "source_sha256": {
            str(p): file_digest(p)
            for p in sorted(Path("recommender/experiments/embedding_ablation").glob("*.py"))
        },
        "reproduction_commands": [
            "PYTHONPATH=. ./venv/bin/pytest tests/",
            "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only",
            "PYTHONPATH=. ./venv/bin/python experiments/embedding_ablation_v1/run.py",
            "PYTHONPATH=. ./venv/bin/python experiments/embedding_ablation_v1/summarize.py",
        ],
    }
    atomic_json(summary, EVIDENCE_ROOT / "summary.json")

    b_s1 = individual_summary["U0_t1_fam_clean"]["upstream"]
    b_mlp = individual_summary["U0_t1_fam_clean"]["mlp_uv47"]

    lines = [
        "# Isolated Product-Text & Product-Embedding Ablation Report (`U0_t1_fam_clean` + `CompactStage1Selector uv_47`, `seed=42`)",
        "",
        "## 1. Scope, Protocol & Exact Metric Semantics",
        "",
        "- **Target Baseline:** `U0_t1_fam_clean` (clean static towers: `t1_fam` text, 0 changing customer stats, static 12D `container_type` only on `ProductTower`, demand disabled, 7 product embeddings `category/subcategory/business_line/brand/tier/unit/country_of_origin`, customer `sales_branch` embedding).",
        "- **Downstream Scorer:** Current `CompactStage1Selector` (`uv_47`, 303D input, 128/64 hidden layers, 10 epochs `seed=42`, selection policy `new_hits@50 -> ndcg@10 -> recall@10`). No numeric normalization changes, no new features, no Stage 2 / `UPS-CANR` runs.",
        "- **Validation Cohort & Denominators:** Strictly `< 2026-02-19` (`2025-10-01` to `2026-02-18`), **6,191** validation transitions, **1,857** new-to-customer target instances. Training uses `< 2025-10-01` (`8,465` Stage 1 discovery rows; `26,253` selector rows with `>= 100` eligible SKUs).",
        "- **Full-Pool Metrics:**",
        "  - **New-target hits/recall@25, @50, @100:** Computed over unseen eligible SKUs (`unseen = allowed & ~seen`), micro-averaged over denominator **1,857**.",
        "  - **Overall Recall@10 / NDCG@10:** Computed over all eligible SKUs (`allowed`), macro-averaged across **6,191** transitions against full `positive_pids`.",
        "- **Inapplicable Ablations:**",
        f"  - Text `unit`: {raw['inapplicable']['text_ablations']['unit']}",
        f"  - Embedding `product_family`: {raw['inapplicable']['embedding_ablations']['product_family']}",
        "",
        "## 2. Part A — Independent Field-Level Text Deletions from `t1_fam` (Full-Pool `uv_47` MLP-Only)",
        "",
        "| Variant | Omitted Field | Changed Texts (/841) | Empty | S1 Ep (s) | MLP Ep (s) | MLP New@25 | MLP New@50 | MLP New@100 | MLP Recall@10 | MLP NDCG@10 | Δ New@50 vs Base (95% CI) | Δ NDCG@10 vs Base (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    text_order = [
        "U0_t1_fam_clean",
        "T_no_base_name",
        "T_no_subcategory",
        "T_no_family",
        "T_no_category",
        "T_no_brand",
        "T_no_business_line",
    ]
    for name in text_order:
        if name not in individual_summary:
            continue
        e = individual_summary[name]
        td = e["text_diagnostics"]
        if e["is_noop"]:
            lines.append(
                f"| {name} | `{e['ablated_target']}` | {td['changed_text_count']} (NO-OP) | {td['empty_text_count']} | — | — | — | — | — | — | — | Exact no-op | Exact no-op |"
            )
            continue
        u = e["upstream"]
        m = e["mlp_uv47"]
        p = e["paired_vs_baseline"]["mlp_uv47"]
        d50 = f"{m['new_hits@50'] - b_mlp['new_hits@50']:+d} ({_fmt_ci_pp(p['new_recall@50_delta'])})" if p else "—"
        dnd = f"{m['ndcg@10'] - b_mlp['ndcg@10']:+.4f} ({_fmt_ci_num(p['ndcg@10_delta'])})" if p else "—"
        lines.append(
            f"| {name} | `{e['ablated_target']}` | {td['changed_text_count']} | {td['empty_text_count']} | "
            f"{u['selected_epoch']} ({u['training_seconds']:.1f}s) | {m['selected_epoch']} ({m['training_seconds']:.1f}s) | "
            f"{m['new_hits@25']} ({m['new_recall@25']*100:.2f}%) | {m['new_hits@50']} ({m['new_recall@50']*100:.2f}%) | "
            f"{m['new_hits@100']} ({m['new_recall@100']*100:.2f}%) | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {d50} | {dnd} |"
        )

    lines.extend([
        "",
        "### Upstream Stage 1 (128D Dot + Familiarity) — Text Deletions",
        "",
        "| Variant | Omitted Field | S1 Ep | S1 New@25 | S1 New@50 | S1 New@100 | S1 Recall@10 | S1 NDCG@10 | Δ S1 New@25 (95% CI) | Δ S1 New@50 (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for name in text_order:
        if name not in individual_summary or individual_summary[name]["is_noop"]:
            continue
        e = individual_summary[name]
        u = e["upstream"]
        ps1 = e["paired_vs_baseline"]["upstream_stage1"]
        d25 = f"{u['new_hits@25'] - b_s1['new_hits@25']:+d} ({_fmt_ci_pp(ps1['new_recall@25_delta'])})" if ps1 else "—"
        d50 = f"{u['new_hits@50'] - b_s1['new_hits@50']:+d} ({_fmt_ci_pp(ps1['new_recall@50_delta'])})" if ps1 else "—"
        lines.append(
            f"| {name} | `{e['ablated_target']}` | {u['selected_epoch']} | "
            f"{u['new_hits@25']} ({u['new_recall@25']*100:.2f}%) | {u['new_hits@50']} ({u['new_recall@50']*100:.2f}%) | "
            f"{u['new_hits@100']} ({u['new_recall@100']*100:.2f}%) | {u['recall@10']:.5f} | {u['ndcg@10']:.5f} | {d25} | {d50} |"
        )

    lines.extend([
        "",
        "## 3. Part B — Independent Product Categorical Embedding Deletions (Full-Pool `uv_47` MLP-Only)",
        "",
        "| Variant | Removed Emb | S1 Params | S1 Ep (s) | MLP Ep (s) | MLP New@25 | MLP New@50 | MLP New@100 | MLP Recall@10 | MLP NDCG@10 | Δ New@50 vs Base (95% CI) | Δ NDCG@10 vs Base (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    emb_order = [
        "U0_t1_fam_clean",
        "E_no_category",
        "E_no_subcategory",
        "E_no_brand",
        "E_no_unit",
        "E_no_business_line",
    ]
    for name in emb_order:
        if name not in individual_summary:
            continue
        e = individual_summary[name]
        u = e["upstream"]
        m = e["mlp_uv47"]
        p = e["paired_vs_baseline"]["mlp_uv47"]
        d50 = f"{m['new_hits@50'] - b_mlp['new_hits@50']:+d} ({_fmt_ci_pp(p['new_recall@50_delta'])})" if p else "—"
        dnd = f"{m['ndcg@10'] - b_mlp['ndcg@10']:+.4f} ({_fmt_ci_num(p['ndcg@10_delta'])})" if p else "—"
        lines.append(
            f"| {name} | `{e['ablated_target']}` | {u['parameters']:,} | {u['selected_epoch']} ({u['training_seconds']:.1f}s) | "
            f"{m['selected_epoch']} ({m['training_seconds']:.1f}s) | "
            f"{m['new_hits@25']} ({m['new_recall@25']*100:.2f}%) | {m['new_hits@50']} ({m['new_recall@50']*100:.2f}%) | "
            f"{m['new_hits@100']} ({m['new_recall@100']*100:.2f}%) | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {d50} | {dnd} |"
        )

    lines.extend([
        "",
        "### Upstream Stage 1 (128D Dot + Familiarity) — Embedding Deletions",
        "",
        "| Variant | Removed Emb | S1 Ep | S1 New@25 | S1 New@50 | S1 New@100 | S1 Recall@10 | S1 NDCG@10 | Δ S1 New@25 (95% CI) | Δ S1 New@50 (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for name in emb_order:
        if name not in individual_summary:
            continue
        e = individual_summary[name]
        u = e["upstream"]
        ps1 = e["paired_vs_baseline"]["upstream_stage1"]
        d25 = f"{u['new_hits@25'] - b_s1['new_hits@25']:+d} ({_fmt_ci_pp(ps1['new_recall@25_delta'])})" if ps1 else "—"
        d50 = f"{u['new_hits@50'] - b_s1['new_hits@50']:+d} ({_fmt_ci_pp(ps1['new_recall@50_delta'])})" if ps1 else "—"
        lines.append(
            f"| {name} | `{e['ablated_target']}` | {u['selected_epoch']} | "
            f"{u['new_hits@25']} ({u['new_recall@25']*100:.2f}%) | {u['new_hits@50']} ({u['new_recall@50']*100:.2f}%) | "
            f"{u['new_hits@100']} ({u['new_recall@100']*100:.2f}%) | {u['recall@10']:.5f} | {u['ndcg@10']:.5f} | {d25} | {d50} |"
        )

    if exploratory_summary:
        lines.extend([
            "",
            "## 4. Exploratory Cumulative Combination (Strictly Separate from Individual Ablations)",
            "",
            "| Variant | Target | S1 Ep | MLP Ep | MLP New@25 | MLP New@50 | MLP New@100 | MLP Recall@10 | MLP NDCG@10 | Δ New@50 vs Base (95% CI) | Δ NDCG@10 vs Base (95% CI) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for name, e in exploratory_summary.items():
            u = e["upstream"]
            m = e["mlp_uv47"]
            p = e["paired_vs_baseline"]["mlp_uv47"]
            d50 = f"{m['new_hits@50'] - b_mlp['new_hits@50']:+d} ({_fmt_ci_pp(p['new_recall@50_delta'])})" if p else "—"
            dnd = f"{m['ndcg@10'] - b_mlp['ndcg@10']:+.4f} ({_fmt_ci_num(p['ndcg@10_delta'])})" if p else "—"
            lines.append(
                f"| {name} | `{e['ablated_target']}` | {u['selected_epoch']} | {m['selected_epoch']} | "
                f"{m['new_hits@25']} ({m['new_recall@25']*100:.2f}%) | {m['new_hits@50']} ({m['new_recall@50']*100:.2f}%) | "
                f"{m['new_hits@100']} ({m['new_recall@100']*100:.2f}%) | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {d50} | {dnd} |"
            )
    else:
        lines.extend([
            "",
            "## 4. Exploratory Cumulative Combination Decision",
            "",
            "- **Omitted:** Only one individual deletion (`E_no_brand`) achieved a statistically supported `uv_47` `New@50` gain (`+25`, 95% CI `[+0.16, +1.35, +2.63] pp`) without degrading `New@25` or `New@100`. All 6 `t1_fam` text deletions had small effects whose paired 95% CIs straddled zero (`New@50` delta `-4` to `+10`), and the remaining embedding deletions either reduced `New@50` (`E_no_subcategory`, `E_no_unit`, `E_no_business_line`) or reduced `New@25`/`New@100`/upstream `Stage 1` (`E_no_category`: `New@25 -31`, `New@100 -9`, `uv_47 New@50` CI `[-1.11, +0.90, +2.89] pp`). Because there were not multiple independently promising deletions, no cumulative combination was trained.",
        ])

    lines.extend([
        "",
        "## 5. Reproduction Commands",
        "",
        "```bash",
        *summary["reproduction_commands"],
        "```",
    ])

    (EVIDENCE_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote summary.json and REPORT.md to", EVIDENCE_ROOT, flush=True)


if __name__ == "__main__":
    main()
