"""Markdown report and self-contained HTML comparison artifact generators."""

import html
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import ARM_A_NAME, ARM_B_NAME, ARM_C_NAME
from .normalization import enrichment_normalization_spec


def _fmt(val: Any, digits: int = 5) -> str:
    if val is None:
        return "—"
    if isinstance(val, int) and not isinstance(val, bool):
        return f"{val:,}"
    return f"{float(val):.{digits}f}"


def _fmt_ci(entry: Mapping[str, Any], digits: int = 5) -> str:
    pt = float(entry["point_delta"])
    lo, _, hi = entry["ci_95"]
    return f"{pt:+.{digits}f} ([{lo:+.{digits}f}, {hi:+.{digits}f}])"


def build_reproduction_cli(
    *,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
) -> Dict[str, str]:
    return {
        "synthetic_tests_only": (
            "PYTHONPATH=. ./venv/bin/pytest tests/test_candidate_token_enrichment.py --synthetic-only"
        ),
        "full_synthetic_suite": "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only",
        "full_test_suite": "PYTHONPATH=. ./venv/bin/pytest tests/",
        "verify_init_only": (
            f"./venv/bin/python -m recommender.experiments.candidate_token_enrichment.train "
            f"--verify-init-only --seed {seed}"
        ),
        "execute_three_arm_campaign": (
            f"./venv/bin/python -m recommender.experiments.candidate_token_enrichment.train "
            f"--execute-training --seed {seed} --epochs {epochs} --batch-size {batch_size}"
        ),
    }


def generate_markdown_report(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    a = arms[ARM_A_NAME]
    b = arms[ARM_B_NAME]
    c = arms[ARM_C_NAME]
    norm = summary["scalar_normalization_spec"]
    cli = summary["reproduction_cli"]
    init_p = summary["three_arm_init_parity"]
    bench = summary["latency_benchmark"]

    lines = [
        "# Stage 2 Candidate-Set Self-Attention & Pre-Attention Token Enrichment (3-Arm Seed-42 Campaign)",
        "",
        f"> **Execution Scope:** `training_executed={summary['training_executed']}`, "
        f"`heldout_access={summary['heldout_access']}`, `promoted={summary['promoted']}`, "
        f"`statistics_policy={summary['statistics_policy']}`, `teacher_score_included={summary['teacher_score_included']}`, "
        f"`seed={summary['seed']}`, `epochs={summary['epochs']}`, `batch_size={summary['batch_size']}`.",
        f"> All 17 frozen reference hashes + {len(summary['preserved_hashes']) - 17} rolling v3 cache/evidence hashes verified identical before and after execution.",
        "",
        "## 1. Controlled 3-Arm Architecture & Parameter Summary",
        "",
        "All three arms share identical `rolling_observed_v2` `quota100_50` slates (`26,253` train rows / `6,191` val rows, `10,276` total targets, `1,857` new targets), identical Stage 1 `repeat-no-month-v1` `u`/`v` vectors, identical initial `base` + `base_head` + `candidate_attention` weights, and identical `AdamW(lr=1e-3, wd=1e-4)` + `CosineAnnealingLR(eta_min=1e-5)` + `RankerLoss` optimization.",
        "",
        "| Arm | Candidate Self-Attention | Pre-Attention Token Enrichment | Total Params | Trainable Params | Qualified Discovery | Discovery Epoch | NDCG Epoch | Train Time (s) |",
        "| --- | :---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for label, res in (("Arm A (Pointwise 425D Control)", a), ("Arm B (Cand-Attn 425D)", b), ("Arm C (Cand-Attn + Enriched Tokens 128D)", c)):
        lines.append(
            f"| {label} | `{res['candidate_attention_enabled']}` | `{res['token_enrichment_enabled']}` | "
            f"{res['parameters']:,} | {res['trainable_parameters']:,} | `{res['qualified']}` | "
            f"{res['discovery']['epoch']} | {res['ndcg']['epoch']} | {res['training_seconds']:.1f} |"
        )

    lines.extend([
        "",
        "### Architecture Dimensions & Placement",
        "- **Base UPS-CANR Cross-Attention & Bilinear Bias (Common A/B/C):** `33`-token customer memory width `128`, `4`-head cross-attention, `statistics_attention_weights` `(4, 14, 17)`, `excluded_statistics=('history_repeat',)`, `397D` encoded output + `28D` raw dynamic context -> **`425D`** `base_head` (`425 -> 256 -> 128 -> 64 -> 1`).",
        "- **Candidate-Set Self-Attention (Common B/C):** `425D -> Linear(425, 128)` -> masked unordered `TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, norm_first=True, activation='gelu')` (no candidate positional encodings) -> `residual_head(256 -> 128 -> 1)` zero-initialized.",
        "- **Pre-Attention Token Enrichment (Arm C only):** Base semantic tokens computed with zeros in numeric branches; three zero-output-initialized residual MLPs (`Linear -> LayerNorm(128) -> GELU -> Linear(128)`) inject `[0, 1]`-normalized statistics BEFORE `memory_encoder` and BEFORE `cross_attention`:",
        "  - Profile Token 0: `[z_prof (128), profile8] (136) -> 128 -> 128`, added BEFORE `type_embedding[0]`.",
        "  - History Tokens 1..32: `[t_hist (128), event9] (137) -> 128 -> 128`, added BEFORE `pos_embedding + type_embedding[1]` (padded tokens sanitized/masked to zero before embeddings and MLP).",
        "  - Candidate Queries: `[v_cand (128), all41] (169) -> 128 -> 128`, added BEFORE `cross_attention` (no positional encodings; padded candidates masked to zero before MLP).",
        f"- **Zero-Residual Init Parity:** `max_abs_logit_diff` vs `CandidateContextRanker(425D)` at init: Arm A `{init_p['max_abs_logit_diff_arm_a']:.1e}`, Arm B `{init_p['max_abs_logit_diff_arm_b']:.1e}`, Arm C `{init_p['max_abs_logit_diff_arm_c']:.1e}`; reversed-slate diff `{init_p['max_abs_logit_diff_reversed_slate_c']:.1e}`, NaN-padded diff `{init_p['max_abs_logit_diff_nan_padding_c']:.1e}`.",
        "",
        "## 2. Scalar Normalization, Sources & Limitations",
        f"- **Transform Version:** `{norm['transform_version']}` (`transform_digest={norm['transform_digest'][:12]}...`).",
        f"- **Exact Scalar Channel Counts:** `41` candidate context + `8` profile lifetime + `9` history event = **`{norm['total_enriched_scalar_channels']}`** enrichment channels (plus unchanged raw `41D`/`8D`/`9D` paths to bilinear bias and `425D` scoring/attention head, and preserved signed `128D` `u`/`v` latent vectors).",
        "- **10 Signed Candidate Context Columns Mapped via `(x + 1) / 2 -> [0, 1]`:** `" + "`, `".join(norm["signed_candidate_features"]) + "` (remaining 31 columns validated in `[0, 1]`).",
        f"- **Stage 2 vs Stage 1 `uv_45` Month Note:** {norm['stage2_vs_stage1_month_note']}",
        "- **Profile 8D (`LEGACY_PROFILE_FEATURES`):** `total_purchases`, `total_quantity`, `distinct_products`, `distinct_orders`, `distinct_categories`, `days_since_first_order`, `days_since_last_order`, `median_order_gap` (all `[0, 1]`).",
        "- **History 9D (`ENRICHED_FEATURE_NAMES`):** `days_since_purchase_norm` recomputed to `prediction_day` from `seq_purchase_dates` (`age/(age+90)`); `purchase_month_sin`/`purchase_month_cos` mapped via `(x + 1) / 2 -> [0, 1]`; `real_event_indicator` derived from `attention_mask`; `is_repeat_purchase` (`history_repeat`, col 7) strictly zeroed (`= 0.0`) across all enriched paths.",
        f"- **History Text & Metadata Limitations:** {norm['history_text_behavior']} Metadata policy remains `{norm['metadata_policy']}` (non-effective-dated catalogue/company attributes).",
        "",
        "## 3. Validation Performance (N = 6,191 Queries; 10,276 Total Targets; 1,857 New Targets)",
        "",
        "### 3.1 NDCG-Selected Checkpoints (`best_ndcg.pt`) & Discovery-Selected Checkpoints (`best.pt` / `best_unqualified.pt`)",
        "",
        "| Arm & Selection | Ep | Qual | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | NDCG@10 | MRR | Repeat Micro R@10 | Repeat Macro R@10 | New Hits@10 | New Recall@10 | Mixed Cov (Micro) | New Cov |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for sel_key, sel_label in (("ndcg", "NDCG-sel"), ("discovery", "Disc-sel")):
        for arm_short, res in (("Arm A", a), ("Arm B", b), ("Arm C", c)):
            s = res[sel_key]
            m = s["metrics"]
            lines.append(
                f"| {arm_short} ({sel_label}) | {s['epoch']} | `{res['qualified']}` | "
                f"{_fmt(m['recall@5'])} | {_fmt(m['recall@10'])} | {_fmt(m['recall@25'])} | {_fmt(m['recall@50'])} | {_fmt(m['recall@100'])} | "
                f"{_fmt(m['ndcg@10'])} | {_fmt(m['mrr'])} | {_fmt(m['repeat_recall@10_micro'])} | {_fmt(m['repeat_recall@10_macro'])} | "
                f"{m['new_hits@10']:,} | {_fmt(m['new_recall@10'])} | {_fmt(m['mixed_coverage_micro'])} | {_fmt(m['new_slate_coverage'])} |"
            )

    lines.extend([
        "",
        "### 3.2 Customer-Clustered Paired Bootstrap 95% CIs (2,000 draws, seed=123)",
        "",
        "| Selection | Comparison | Recall@10 Delta (95% CI) | NDCG@10 Delta (95% CI) | New Recall@10 Delta (95% CI) | Recall@5 Delta (95% CI) |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for sel_key in ("ndcg", "discovery"):
        ci_block = summary["paired_bootstrap_95ci"][sel_key]
        for cmp_key, cmp_label in (("A_to_B", "A -> B (B - A)"), ("B_to_C", "B -> C (C - B)"), ("A_to_C", "A -> C (C - A)")):
            row = ci_block[cmp_key]
            lines.append(
                f"| `{sel_key}` | {cmp_label} | {_fmt_ci(row['recall@10_delta'])} | "
                f"{_fmt_ci(row['ndcg@10_delta'])} | {_fmt_ci(row['new_recall@10_delta'])} | {_fmt_ci(row['recall@5_delta'])} |"
            )

    lines.extend([
        "",
        "### 3.3 History-Length & Validation-Month Cohorts (NDCG-Selected Checkpoints)",
        "> **Population Caveat:** Validation months (`2025-10`..`2026-02`) and history-length buckets contain different customer populations, basket sizes, and seasonal mixes; cohort deltas should not be conflated with stationary longitudinal effects.",
        "",
        "| Cohort | Rows | New Targets | A Recall@10 | B Recall@10 | C Recall@10 | A NDCG@10 | B NDCG@10 | C NDCG@10 | A New@10 | B New@10 | C New@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    am = a["ndcg"]["metrics"]
    bm = b["ndcg"]["metrics"]
    cm = c["ndcg"]["metrics"]
    for cname in ("cold_0", "light_1_8", "warm_9_31", "truncated_32"):
        ac, bc, cc = am["history_cohorts"][cname], bm["history_cohorts"][cname], cm["history_cohorts"][cname]
        lines.append(
            f"| `hist:{cname}` | {ac['rows']:,} | {ac['new_targets']:,} | "
            f"{_fmt(ac['recall@10'])} | {_fmt(bc['recall@10'])} | {_fmt(cc['recall@10'])} | "
            f"{_fmt(ac['ndcg@10'])} | {_fmt(bc['ndcg@10'])} | {_fmt(cc['ndcg@10'])} | "
            f"{ac['new_hits@10']:,} | {bc['new_hits@10']:,} | {cc['new_hits@10']:,} |"
        )
    for month in sorted(am.get("validation_months", {}).keys()):
        ac = am["validation_months"][month]
        bc = bm["validation_months"][month]
        cc = cm["validation_months"][month]
        lines.append(
            f"| `month:{month}` | {ac['rows']:,} | {ac['new_targets']:,} | "
            f"{_fmt(ac['recall@10'])} | {_fmt(bc['recall@10'])} | {_fmt(cc['recall@10'])} | "
            f"{_fmt(ac['ndcg@10'])} | {_fmt(bc['ndcg@10'])} | {_fmt(cc['ndcg@10'])} | "
            f"{ac['new_hits@10']:,} | {bc['new_hits@10']:,} | {cc['new_hits@10']:,} |"
        )

    lines.extend([
        "",
        "## 4. Mechanism Diagnostics (Fitted Checkpoints, No Retraining)",
        "",
        "| Checkpoint & Ablation | Max Abs Logit Diff | Mean Abs Logit Diff | Max Abs z_cand Diff | Max Abs Memory Diff | Recall@10 Delta | NDCG@10 Delta | New Hits@10 Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm_label, res in (("Arm B", b), ("Arm C", c)):
        for sel_key in ("ndcg", "discovery"):
            diag = res[sel_key]["mechanism_diagnostics"]
            for abl_key, abl_label in (
                ("disable_candidate_attention_only", "disable_cand_attn"),
                ("disable_all_token_enrichment", "disable_all_enrich"),
                ("disable_candidate_enrichment_only", "disable_cand_enrich"),
                ("disable_profile_enrichment_only", "disable_prof_enrich"),
                ("disable_history_enrichment_only", "disable_hist_enrich"),
            ):
                if abl_key not in diag:
                    continue
                d = diag[abl_key]
                lines.append(
                    f"| {arm_label} ({sel_key}) `{abl_label}` | {d['max_abs_logit_diff']:.4f} | {d['mean_abs_logit_diff']:.5f} | "
                    f"{d['max_abs_z_candidate_diff']:.4f} | {d['max_abs_memory_diff']:.4f} | "
                    f"{d['recall@10_delta_vs_intact']:+.5f} | {d['ndcg@10_delta_vs_intact']:+.5f} | {d['new_hits@10_delta_vs_intact']:+d} |"
                )

    lines.extend([
        "",
        "### Trained-Weight Stability Checks (Nonzero Weights, Reversed Slate & NaN Candidate/History Padding)",
        "",
        "| Checkpoint | Reversed Slate Max Diff | NaN Cand/Hist Pad Max Valid Diff | Masked Logits `-inf` |",
        "| --- | ---: | ---: | :---: |",
    ])
    for arm_label, res in (("Arm B", b), ("Arm C", c)):
        for sel_key in ("ndcg", "discovery"):
            st = res[sel_key]["mechanism_diagnostics"]["trained_stability"]
            lines.append(
                f"| {arm_label} ({sel_key}) | {st['max_abs_logit_diff_reversed_slate']:.3e} | "
                f"{st['max_abs_logit_diff_with_nan_cand_and_hist_padding']:.3e} | `{st['masked_pad_logit_neginf']}` |"
            )

    lines.extend([
        "",
        "## 5. Real-Input Stage 2 Latency Benchmark (CPU & MPS, Slate=100, Batch=1 & 64)",
        "",
        f"> Hardware: `{bench['hardware']['cpu_brand']}`, PyTorch `{bench['hardware']['torch_version']}`, `mps_available={bench['hardware']['mps_available']}`. "
        "Scope: `prepared_on_device_forward` measures pre-materialized tensors on target device; `host_to_device_transfer_plus_forward` includes CPU->device tensor transfer.",
        "",
        "| Device | Arm | Batch 1 Prepared Med/P90 (ms) | Batch 1 H2D+Fwd Med/P90 (ms) | Batch 64 Prepared Med/P90 (ms) | Batch 64 H2D+Fwd Med/P90 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for dev_name, dev_map in bench["devices"].items():
        for arm_name in (ARM_A_NAME, ARM_B_NAME, ARM_C_NAME):
            row = dev_map[arm_name]
            b1p = row["batch_1"]["prepared_on_device_forward"]
            b1h = row["batch_1"]["host_to_device_transfer_plus_forward"]
            b64p = row["batch_64"]["prepared_on_device_forward"]
            b64h = row["batch_64"]["host_to_device_transfer_plus_forward"]
            lines.append(
                f"| `{dev_name}` | `{arm_name}` | {b1p['median_ms']:.3f} / {b1p['p90_ms']:.3f} | "
                f"{b1h['median_ms']:.3f} / {b1h['p90_ms']:.3f} | "
                f"{b64p['median_ms']:.3f} / {b64p['p90_ms']:.3f} | "
                f"{b64h['median_ms']:.3f} / {b64h['p90_ms']:.3f} |"
            )

    lines.extend([
        "",
        "## 6. Reproduction CLI",
        "```bash",
        f"# 1. Synthetic unit & regression tests",
        cli["synthetic_tests_only"],
        f"# 2. Full synthetic suite",
        cli["full_synthetic_suite"],
        f"# 3. Full pytest suite",
        cli["full_test_suite"],
        f"# 4. Zero-init parity check (no training)",
        cli["verify_init_only"],
        f"# 5. Full 3-arm seed-42 10-epoch campaign + evaluation + benchmark",
        cli["execute_three_arm_campaign"],
        "```",
        "",
    ])
    return "\n".join(lines)


def generate_html_comparison(summary: Mapping[str, Any]) -> str:
    """Build a self-contained HTML comparison artifact suitable for html_preview/html_render."""
    arms = summary["arms"]
    a = arms[ARM_A_NAME]
    b = arms[ARM_B_NAME]
    c = arms[ARM_C_NAME]
    ci_ndcg = summary["paired_bootstrap_95ci"]["ndcg"]
    ci_disc = summary["paired_bootstrap_95ci"]["discovery"]
    bench = summary["latency_benchmark"]

    def _svg_ci_chart(ci_block: Mapping[str, Any], metric_key: str, title: str, scale: float = 100.0, unit: str = "pp") -> str:
        rows = [
            ("A -> B", ci_block["A_to_B"][metric_key]),
            ("B -> C", ci_block["B_to_C"][metric_key]),
            ("A -> C", ci_block["A_to_C"][metric_key]),
        ]
        vals = [v for _, e in rows for v in (e["ci_95"][0] * scale, e["ci_95"][2] * scale, e["point_delta"] * scale)]
        bound = max(max(abs(x) for x in vals) * 1.25, 0.2)
        w, h = 440, 145
        left, right = 75, 420
        mid = (left + right) / 2.0
        half = (right - left) / 2.0

        def _x(v: float) -> float:
            return mid + (v / bound) * half

        parts = [
            f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
            f'<text x="{left}" y="18" font-size="13" font-weight="600" fill="#1f2937">{html.escape(title)}</text>',
            f'<line x1="{mid:.1f}" y1="30" x2="{mid:.1f}" y2="120" stroke="#9ca3af" stroke-dasharray="3,3" stroke-width="1"/>',
            f'<text x="{mid:.1f}" y="136" font-size="10" text-anchor="middle" fill="#6b7280">0 {unit}</text>',
        ]
        for idx, (lbl, entry) in enumerate(rows):
            y = 48 + idx * 28
            lo = entry["ci_95"][0] * scale
            hi = entry["ci_95"][2] * scale
            pt = entry["point_delta"] * scale
            color = "#059669" if lo > 0 else ("#dc2626" if hi < 0 else "#2563eb")
            parts.append(f'<text x="8" y="{y + 4}" font-size="11" fill="#374151">{html.escape(lbl)}</text>')
            parts.append(f'<line x1="{_x(lo):.1f}" y1="{y}" x2="{_x(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<circle cx="{_x(pt):.1f}" cy="{y}" r="4.5" fill="{color}"/>')
            parts.append(
                f'<text x="{_x(hi) + 6:.1f}" y="{y + 4}" font-size="10" fill="#374151">{pt:+.3f}{unit}</text>'
            )
        parts.append("</svg>")
        return "\n".join(parts)

    def _arm_rows_html(sel_key: str) -> str:
        out = []
        for label, res in (("Arm A: Pointwise 425D", a), ("Arm B: Cand-Attn 425D", b), ("Arm C: Cand-Attn + Enriched 128D", c)):
            s = res[sel_key]
            m = s["metrics"]
            out.append(
                f"<tr><td><strong>{html.escape(label)}</strong></td>"
                f"<td>{s['epoch']}</td>"
                f"<td>{'Yes' if res['qualified'] else 'No'}</td>"
                f"<td>{m['recall@5']:.5f}</td>"
                f"<td>{m['recall@10']:.5f}</td>"
                f"<td>{m['recall@25']:.5f}</td>"
                f"<td>{m['recall@50']:.5f}</td>"
                f"<td>{m['recall@100']:.5f}</td>"
                f"<td>{m['ndcg@10']:.5f}</td>"
                f"<td>{m['mrr']:.5f}</td>"
                f"<td>{m['repeat_recall@10_micro']:.5f}</td>"
                f"<td>{m['repeat_recall@10_macro']:.5f}</td>"
                f"<td>{m['new_hits@10']:,} ({m['new_recall@10']:.5f})</td></tr>"
            )
        return "\n".join(out)

    def _bench_rows_html() -> str:
        out = []
        for dev_name, dev_map in bench["devices"].items():
            for arm_name in (ARM_A_NAME, ARM_B_NAME, ARM_C_NAME):
                r = dev_map[arm_name]
                b1p = r["batch_1"]["prepared_on_device_forward"]
                b1h = r["batch_1"]["host_to_device_transfer_plus_forward"]
                b64p = r["batch_64"]["prepared_on_device_forward"]
                b64h = r["batch_64"]["host_to_device_transfer_plus_forward"]
                out.append(
                    f"<tr><td><code>{html.escape(dev_name)}</code></td>"
                    f"<td><code>{html.escape(arm_name)}</code></td>"
                    f"<td>{r['parameters']:,}</td>"
                    f"<td>{b1p['median_ms']:.3f} / {b1p['p90_ms']:.3f}</td>"
                    f"<td>{b1h['median_ms']:.3f} / {b1h['p90_ms']:.3f}</td>"
                    f"<td>{b64p['median_ms']:.3f} / {b64p['p90_ms']:.3f}</td>"
                    f"<td>{b64h['median_ms']:.3f} / {b64h['p90_ms']:.3f}</td></tr>"
                )
        return "\n".join(out)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Stage 2 Candidate Self-Attention &amp; Pre-Attention Token Enrichment (Seed 42)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px auto; max-width: 1180px; color: #111827; background: #f9fafb; padding: 0 16px; }}
  h1, h2, h3 {{ color: #111827; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #e0f2fe; color: #0369a1; margin-right: 6px; }}
  .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 14px; }}
  .note {{ font-size: 12px; color: #4b5563; }}
</style>
</head>
<body>
  <h1>Stage 2 Candidate Self-Attention &amp; Token Statistics Enrichment</h1>
  <div>
    <span class="badge">Seed: {summary['seed']}</span>
    <span class="badge">Epochs: {summary['epochs']}</span>
    <span class="badge">Batch: {summary['batch_size']}</span>
    <span class="badge">Policy: {html.escape(summary['statistics_policy'])}</span>
    <span class="badge">Slate: {summary['slate_policy']} (26,253 train / 6,191 val)</span>
    <span class="badge">Teacher Score: False</span>
    <span class="badge">Heldout Access: False</span>
  </div>

  <div class="card">
    <h2>1. NDCG-Selected Validation Metrics (best_ndcg.pt)</h2>
    <table>
      <thead><tr><th>Arm</th><th>Ep</th><th>Qual</th><th>R@5</th><th>R@10</th><th>R@25</th><th>R@50</th><th>R@100</th><th>NDCG@10</th><th>MRR</th><th>Rep Micro</th><th>Rep Macro</th><th>New Hits@10</th></tr></thead>
      <tbody>{_arm_rows_html('ndcg')}</tbody>
    </table>
    <p class="note">Identical slates across all 3 arms: mixed coverage micro = {a['ndcg']['metrics']['mixed_coverage_micro']:.5f}, new slate coverage = {a['ndcg']['metrics']['new_slate_coverage']:.5f} ({a['ndcg']['metrics']['new_targets_in_slate']:,}/{a['ndcg']['metrics']['new_targets']:,}).</p>
  </div>

  <div class="card">
    <h2>2. Discovery-Selected Validation Metrics (best.pt / best_unqualified.pt)</h2>
    <table>
      <thead><tr><th>Arm</th><th>Ep</th><th>Qual</th><th>R@5</th><th>R@10</th><th>R@25</th><th>R@50</th><th>R@100</th><th>NDCG@10</th><th>MRR</th><th>Rep Micro</th><th>Rep Macro</th><th>New Hits@10</th></tr></thead>
      <tbody>{_arm_rows_html('discovery')}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>3. Customer-Clustered Paired 95% Bootstrap CIs (2,000 draws, seed=123)</h2>
    <div class="grid">
      <div>{_svg_ci_chart(ci_ndcg, 'recall@10_delta', 'NDCG-Selected: Recall@10 Delta (pp)', 100.0, 'pp')}</div>
      <div>{_svg_ci_chart(ci_ndcg, 'ndcg@10_delta', 'NDCG-Selected: NDCG@10 Delta (x100)', 100.0, '')}</div>
      <div>{_svg_ci_chart(ci_ndcg, 'new_recall@10_delta', 'NDCG-Selected: New Recall@10 Delta (pp)', 100.0, 'pp')}</div>
      <div>{_svg_ci_chart(ci_disc, 'new_recall@10_delta', 'Discovery-Selected: New Recall@10 Delta (pp)', 100.0, 'pp')}</div>
    </div>
  </div>

  <div class="card">
    <h2>4. Real-Input Stage 2 Request Latency (CPU &amp; MPS, Slate=100)</h2>
    <table>
      <thead><tr><th>Device</th><th>Arm</th><th>Params</th><th>Batch 1 Prepared (ms)</th><th>Batch 1 H2D+Fwd (ms)</th><th>Batch 64 Prepared (ms)</th><th>Batch 64 H2D+Fwd (ms)</th></tr></thead>
      <tbody>{_bench_rows_html()}</tbody>
    </table>
    <p class="note">Prepared measures pre-materialized tensors on target device; H2D+Fwd includes host-to-device batch transfer.</p>
  </div>
</body>
</html>
"""


def write_campaign_reports(summary: Mapping[str, Any], evidence_root: Path) -> Dict[str, str]:
    out_dir = Path(evidence_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"
    html_path = out_dir / "comparison.html"
    report_path.write_text(generate_markdown_report(summary), encoding="utf-8")
    html_path.write_text(generate_html_comparison(summary), encoding="utf-8")
    return {
        "report_md": str(report_path),
        "comparison_html": str(html_path),
    }
