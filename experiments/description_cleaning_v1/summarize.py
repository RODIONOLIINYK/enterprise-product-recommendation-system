"""Summarize description cleaning experiment results, subgroup CIs, examples, and provenance."""

import json
from pathlib import Path
from typing import Any, Dict

from recommender.artifacts import atomic_json, file_digest
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.description_cleaning.cleaner import (
    CLEANER_VERSION,
    CLEANING_RULES,
    CLEANING_RULES_HASH,
    load_descriptions_v1,
)
from recommender.experiments.description_cleaning.config import CACHE_DIR, EVIDENCE_ROOT
from recommender.experiments.description_cleaning.evaluate import (
    paired_subgroup_intervals,
    verify_hash_preservation,
)


def _fmt_ci_pp(ci: list[float]) -> str:
    return f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}, {ci[2]*100:+.2f}] pp"


def _fmt_ci_num(ci: list[float], dec: int = 4) -> str:
    return f"[{ci[0]:+.{dec}f}, {ci[1]:+.{dec}f}, {ci[2]:+.{dec}f}]"


def _collect_bounded_examples(dev_pids: set[str]) -> list[Dict[str, Any]]:
    df = load_descriptions_v1()
    in_dev = df.loc[df["product_id"].isin(dev_pids)].copy()
    chosen_ids = [
        "ТОВ-У001541",  # Helix HX7 10W-40: CTA headings + shellcheck QR URL + emojis + specs
        "ТОВ-У510621",  # Winter Screenwash Concentrate: <table> dilution ratios + -55°C/-30°C/-19°C/-10°C
        "ТОВ-У510357",  # Antifreeze Longlife: duplicate Ford WSS-M97B44-D + 'не містить' negations
        "ТОВ-У503715",  # Helix HX8 5W-30: MB 229.5 / VW 502.00 + promotional closing tagline
    ]
    examples: list[Dict[str, Any]] = []
    indexed = in_dev.set_index("product_id")
    for pid in chosen_ids:
        if pid not in indexed.index:
            continue
        row = indexed.loc[pid]
        stats = json.loads(row["cleaning_stats_json"])
        examples.append({
            "product_id": pid,
            "raw_name": row["raw_name"],
            "raw_chars": stats["raw_chars"],
            "html_basic_chars": len(row["legacy_html_description"]),
            "clean_v1_chars": stats["cleaned_chars"],
            "counts": {
                "tags_stripped": stats["tags_stripped"],
                "tables_parsed": stats["tables_parsed"],
                "emojis_removed": stats["emojis_removed"],
                "boilerplate_blocks_removed": stats["boilerplate_blocks_removed"],
                "boilerplate_sentences_removed": stats["boilerplate_sentences_removed"],
                "duplicate_blocks_removed": stats["duplicate_blocks_removed"],
                "duplicate_sentences_removed": stats["duplicate_sentences_removed"],
            },
            "removed_items": stats["removed_items"][:6],
            "before_html_basic_excerpt": row["legacy_html_description"][:500],
            "after_clean_v1_excerpt": row["cleaned_description_v1"][:500],
        })
    return examples


def _extract_variant_entry(
    name: str,
    res: Dict[str, Any],
    base_sub_npz: Path,
    html_sub_npz: Path | None,
) -> Dict[str, Any]:
    s1 = res["stage1"]
    sel = res["selector_uv47"]
    dot_fp = res["upstream_dot_fullpool"]
    sub_npz = Path(res["mlp_subgroup_predictions"])

    paired_vs_base = paired_subgroup_intervals(base_sub_npz, sub_npz)
    paired_vs_html = (
        paired_subgroup_intervals(html_sub_npz, sub_npz)
        if (name == "D_clean" and html_sub_npz is not None and html_sub_npz.is_file())
        else None
    )

    return {
        "name": name,
        "text_diagnostics": res["text_diagnostics"],
        "upstream": {
            "selected_epoch": s1["selected_epoch"],
            "parameters": s1["parameters"],
            "training_seconds": s1["training_seconds"],
            "metrics": s1["metrics"],
            "upstream_dot_fullpool": dot_fp,
            "selected_checkpoint": s1["selected_checkpoint"],
            "checkpoint_sha256": file_digest(s1["selected_checkpoint"]),
        },
        "mlp_uv47": {
            "selected_epoch": sel["selected_epoch"],
            "parameters": sel["parameters"],
            "training_seconds": sel["training_seconds"],
            "metrics": sel["metrics"],
            "selected_checkpoint": sel["selected_checkpoint"],
            "checkpoint_sha256": file_digest(sel["selected_checkpoint"]),
            "per_query_predictions": res["mlp_per_query_predictions"],
            "subgroup_predictions": res["mlp_subgroup_predictions"],
            "subgroups_html_mask": res["subgroups_html_mask"],
            "subgroups_clean_mask": res["subgroups_clean_mask"],
        },
        "paired_vs_baseline": paired_vs_base,
        "paired_vs_D_html": paired_vs_html,
    }


def main() -> None:
    raw = json.loads((EVIDENCE_ROOT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((CACHE_DIR / "description_manifest.json").read_text(encoding="utf-8"))
    hashes_ok = verify_hash_preservation()
    _, _, info, _, _ = prepare_retrieval_data()
    dev_pids = set(info["product_ids"])

    base = raw["baseline"]
    base_sub_npz = Path(base["mlp_subgroup_predictions"])
    html_res = raw["variants"].get("D_html")
    html_sub_npz = Path(html_res["mlp_subgroup_predictions"]) if html_res else None

    variants_summary: Dict[str, Any] = {}
    for name in ("D_html", "D_clean"):
        if name in raw["variants"]:
            variants_summary[name] = _extract_variant_entry(name, raw["variants"][name], base_sub_npz, html_sub_npz)

    examples = _collect_bounded_examples(dev_pids)

    summary = {
        "campaign": raw["campaign"],
        "baseline_target": raw["baseline_target"],
        "selector_recipe": raw["selector_recipe"],
        "selector_input_dim": raw["selector_input_dim"],
        "seed": raw["seed"],
        "validation_only": True,
        "heldout_access": False,
        "stage2_executed": False,
        "promoted": False,
        "metadata_policy": "snapshot_unverified",
        "cleaner_version": CLEANER_VERSION,
        "cleaner_rules_hash": CLEANING_RULES_HASH,
        "cleaning_rules": CLEANING_RULES,
        "cleaning_totals": manifest["diagnostics"]["cleaning_totals"],
        "validation_new_target_coverage": manifest["diagnostics"]["validation_new_target_coverage"],
        "text_recipes_provenance": manifest["diagnostics"]["text_recipes"],
        "bounded_cleaning_examples": examples,
        "hashes_preserved": hashes_ok,
        "baseline": base,
        "variants": variants_summary,
        "source_sha256": {
            str(p): file_digest(p)
            for p in sorted(Path("recommender/experiments/description_cleaning").glob("*.py"))
        },
        "reproduction_commands": [
            "PYTHONPATH=. ./venv/bin/pytest tests/",
            "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only",
            "PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --prepare-only",
            "PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --verify-baseline-only",
            "PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_html",
            "PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_clean",
            "PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/summarize.py",
        ],
    }
    atomic_json(summary, EVIDENCE_ROOT / "summary.json", overwrite=True)

    b_m = base["selector_uv47"]["saved_reference_metrics"]
    b_s1 = base["stage1"]["metrics"]
    b_sub = base["subgroups_html_mask"]["new_target_subgroups"]
    cov = manifest["diagnostics"]["validation_new_target_coverage"]
    ctot = manifest["diagnostics"]["cleaning_totals"]

    lines = [
        "# Controlled Product Description Cleaning Experiment (`no-brand-v1` + `CompactStage1Selector uv_47`, `seed=42`)",
        "",
        "## 1. Scope, Provenance & Experimental Contract",
        "",
        "- **Active Baseline Reference (Unchanged, Not Promoted):** `no-brand-v1` (`E_no_brand` epoch 16 + `CompactStage1Selector uv_47` epoch 1, `seed=42`, `t1_fam` text without descriptions, 6 product embeddings excluding `brand`, customer `sales_branch`, 0 tower customer stats, static 12D `container_type`, frozen 303-input `uv_47`).",
        "- **Variants Tested:**",
        "  - `D_html`: appends legacy `clean_html_description(raw_description)` to exact `t1_fam` prefix (`t1_fam_desc_html_v1`).",
        "  - `D_clean`: appends deterministic `desc_clean_v1` cleaned description to exact `t1_fam` prefix (`t1_fam_desc_clean_v1`).",
        "- **Stage 1 MLP Only:** Clean towers trained from scratch (30 epochs, `seed=42`) -> `CompactStage1Selector uv_47` (10 epochs, `seed=42`, `new_hits@50 -> ndcg@10 -> recall@10`). **Zero Stage 2 / `UPS-CANR` training or evaluation.**",
        "- **Strict Date Gate & Metadata Policy:** Train `< 2025-10-01`, Validation `2025-10-01..2026-02-18` (`6,191` transitions, `1,857` new targets), zero held-out (`>= 2026-02-19`) access. Description export (`data/raw/описание.csv`, SHA-256 `" + manifest["diagnostics"]["raw_description_sha256"] + "`) is **snapshot-unverified (`snapshot_unverified`)** — not historical effective-dated.",
        f"- **Data Coverage:** `{ctot['csv_rows']}` unique SKU rows in CSV (`{ctot['csv_nonempty_html']}` non-empty); `{ctot['dev_skus_in_csv']}` in 841 dev SKUs (`{ctot['dev_skus_nonempty_html']}` usable = `{ctot['dev_skus_pct_nonempty_html']:.4f}%`). Validation new targets: `{cov['nonempty_html']}` WITH description vs `{cov['without_description_html']}` WITHOUT description (sum `{cov['total']}`).",
        "",
        "## 2. Full-Pool `CompactStage1Selector uv_47` Results (6,191 Queries / 1,857 New Targets, Seed=42)",
        "",
        "| Variant | S1 Ep (s) | MLP Ep (s) | New@25 (/1,857) | New@50 (/1,857) | New@100 (/1,857) | Recall@10 (/6,191) | NDCG@10 (/6,191) | Δ New@50 vs Ref (95% CI) | Δ NDCG@10 vs Ref (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        f"| `no-brand-v1` (Ref) | {base['stage1']['selected_epoch']} (ref) | {base['selector_uv47']['selected_epoch']} (ref) | {b_m['new_hits@25']} ({b_m['new_recall@25']*100:.2f}%) | {b_m['new_hits@50']} ({b_m['new_recall@50']*100:.2f}%) | {b_m['new_hits@100']} ({b_m['new_recall@100']*100:.2f}%) | {b_m['recall@10']:.5f} | {b_m['ndcg@10']:.5f} | — | — |",
    ]
    for name in ("D_html", "D_clean"):
        if name not in variants_summary:
            continue
        v = variants_summary[name]
        u = v["upstream"]
        m = v["mlp_uv47"]["metrics"]
        p = v["paired_vs_baseline"]["overall"]
        d50 = f"{m['new_hits@50'] - b_m['new_hits@50']:+d} ({_fmt_ci_pp(p['new_recall@50_delta'])})"
        dnd = f"{m['ndcg@10'] - b_m['ndcg@10']:+.5f} ({_fmt_ci_num(p['ndcg@10_delta'])})"
        lines.append(
            f"| `{name}` | {u['selected_epoch']} ({u['training_seconds']:.1f}s) | {v['mlp_uv47']['selected_epoch']} ({v['mlp_uv47']['training_seconds']:.1f}s) | "
            f"{m['new_hits@25']} ({m['new_recall@25']*100:.2f}%) | {m['new_hits@50']} ({m['new_recall@50']*100:.2f}%) | "
            f"{m['new_hits@100']} ({m['new_recall@100']*100:.2f}%) | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {d50} | {dnd} |"
        )

    lines.extend([
        "",
        "### Upstream Stage 1 Tower Metrics (128D Dot + Familiarity)",
        "",
        "| Variant | S1 Ep | S1 New@25 | S1 New@50 | S1 Recall@10 | S1 NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `no-brand-v1` (Ref) | {base['stage1']['selected_epoch']} | {b_s1['new_hits@25']} ({b_s1['stage1_new_recall@25']*100:.2f}%) | {b_s1['new_hits@50']} ({b_s1['stage1_new_recall@50']*100:.2f}%) | {b_s1['val_recall@10']:.5f} | {b_s1['val_ndcg@10']:.5f} |",
    ])
    for name in ("D_html", "D_clean"):
        if name not in variants_summary:
            continue
        u = variants_summary[name]["upstream"]
        um = u["metrics"]
        lines.append(
            f"| `{name}` | {u['selected_epoch']} | {um['new_hits@25']} ({um['stage1_new_recall@25']*100:.2f}%) | {um['new_hits@50']} ({um['stage1_new_recall@50']*100:.2f}%) | {um['val_recall@10']:.5f} | {um['val_ndcg@10']:.5f} |"
        )

    lines.extend([
        "",
        "## 3. Subgroup Breakdown: Targets WITH vs WITHOUT Descriptions (Full Pool Fixed)",
        "",
        "Note: Even for targets WITHOUT descriptions (`1,343 / 1,857`), fresh tower weights and altered history vectors for customers who previously bought any of the 155 described SKUs can change predictions.",
        "",
        "| Variant | Subgroup | Targets | New@10 | New@25 | New@50 | New@100 | Δ New@50 vs Ref (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| `no-brand-v1` (Ref) | WITH desc | {b_sub['with_description']['new_targets']} | {b_sub['with_description']['new_hits@10']} ({b_sub['with_description']['new_recall@10']*100:.2f}%) | {b_sub['with_description']['new_hits@25']} ({b_sub['with_description']['new_recall@25']*100:.2f}%) | {b_sub['with_description']['new_hits@50']} ({b_sub['with_description']['new_recall@50']*100:.2f}%) | {b_sub['with_description']['new_hits@100']} ({b_sub['with_description']['new_recall@100']*100:.2f}%) | — |",
        f"| `no-brand-v1` (Ref) | WITHOUT desc | {b_sub['without_description']['new_targets']} | {b_sub['without_description']['new_hits@10']} ({b_sub['without_description']['new_recall@10']*100:.2f}%) | {b_sub['without_description']['new_hits@25']} ({b_sub['without_description']['new_recall@25']*100:.2f}%) | {b_sub['without_description']['new_hits@50']} ({b_sub['without_description']['new_recall@50']*100:.2f}%) | {b_sub['without_description']['new_hits@100']} ({b_sub['without_description']['new_recall@100']*100:.2f}%) | — |",
    ])
    for name in ("D_html", "D_clean"):
        if name not in variants_summary:
            continue
        v = variants_summary[name]
        sub = v["mlp_uv47"]["subgroups_html_mask"]["new_target_subgroups"]
        pv = v["paired_vs_baseline"]
        for sg_key, sg_label in (("with_description", "WITH desc"), ("without_description", "WITHOUT desc")):
            sm = sub[sg_key]
            bm = b_sub[sg_key]
            ci = pv[sg_key]["new_recall@50_delta"]
            lines.append(
                f"| `{name}` | {sg_label} | {sm['new_targets']} | "
                f"{sm['new_hits@10']} ({sm['new_recall@10']*100:.2f}%) | {sm['new_hits@25']} ({sm['new_recall@25']*100:.2f}%) | "
                f"{sm['new_hits@50']} ({sm['new_recall@50']*100:.2f}%) | {sm['new_hits@100']} ({sm['new_recall@100']*100:.2f}%) | "
                f"{sm['new_hits@50'] - bm['new_hits@50']:+d} ({_fmt_ci_pp(ci)}) |"
            )

    if "D_clean" in variants_summary and variants_summary["D_clean"].get("paired_vs_D_html"):
        p_dh = variants_summary["D_clean"]["paired_vs_D_html"]
        lines.extend([
            "",
            "### Paired `D_clean` vs `D_html` (Customer-Clustered 95% Bootstrap CIs)",
            "",
            f"- **Overall `New@50` Δ:** {_fmt_ci_pp(p_dh['overall']['new_recall@50_delta'])}",
            f"- **Overall `NDCG@10` Δ:** {_fmt_ci_num(p_dh['overall']['ndcg@10_delta'])}",
            f"- **WITH description `New@50` Δ:** {_fmt_ci_pp(p_dh['with_description']['new_recall@50_delta'])}",
            f"- **WITHOUT description `New@50` Δ:** {_fmt_ci_pp(p_dh['without_description']['new_recall@50_delta'])}",
        ])

    lines.extend([
        "",
        "## 4. Cleaning Rules (`desc_clean_v1`), Tokenizer Counts & Bounded Examples",
        "",
        f"- **Totals across 276 CSV rows:** `{ctot['tags_stripped']:,}` tags stripped, `{ctot['tables_parsed']}` tables formatted into key-value pairs, `{ctot['emojis_removed']:,}` decorative emojis/checkmarks removed, `{ctot['boilerplate_blocks_removed']}` CTA heading blocks + `{ctot['boilerplate_sentences_removed']}` marketing/QR/URL sentences removed, `{ctot['duplicate_blocks_removed']}` duplicate within-product blocks removed, **`{ctot['spec_token_losses_vs_legacy']}` spec-token losses**.",
    ])
    for rname in ("t1_fam", "t1_fam_desc_html_v1", "t1_fam_desc_clean_v1"):
        td = manifest["diagnostics"]["text_recipes"].get(rname)
        if td:
            ts = td["token_stats"]
            lines.append(
                f"- **Recipe `{rname}`:** changed `{td['changed_text_count']}/841` (reused `{td['reused_unchanged_count']}`, encoded `{td['newly_encoded_count']}`), max chars `{ts['max_chars']}`, max tokens `{ts['max_tokens']}` / `{ts['encoder_max_seq_length']}`, truncated `{ts['truncated_count']}`."
            )

    for ex in examples:
        lines.extend([
            "",
            f"### Example `{ex['product_id']}` ({ex['raw_name']})",
            f"- Chars: raw `{ex['raw_chars']}` -> `D_html` `{ex['html_basic_chars']}` -> `D_clean` `{ex['clean_v1_chars']}`; counts: {ex['counts']}",
            f"- Removed items (sample): {ex['removed_items']}",
            f"- **Before (`D_html` excerpt):** `{ex['before_html_basic_excerpt']}`",
            f"- **After (`D_clean` excerpt):** `{ex['after_clean_v1_excerpt']}`",
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
