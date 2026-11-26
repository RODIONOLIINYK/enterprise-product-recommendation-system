"""Report the enrichment-only follow-up without rewriting the A-C campaign."""

import json
from pathlib import Path

from .config import ARM_A_NAME, ARM_B_NAME, ARM_C_NAME
from .pointwise_enrichment import ARM_D_NAME, EVIDENCE_ROOT

LABELS = {
    ARM_A_NAME: "A · Pointwise control",
    ARM_B_NAME: "B · Candidate attention",
    ARM_C_NAME: "C · Attention + enrichment",
    ARM_D_NAME: "D · Pointwise + enrichment",
}


def write_reports(root=EVIDENCE_ROOT):
    summary = json.loads((root / "summary.json").read_text())
    arms = summary["arms"]
    lines = [
        "# Pointwise Stage 2 with 128D statistical token enrichment",
        "",
        "Only Arm D was trained in this follow-up: seed 42, 10 epochs, batch size 64, AdamW, MPS. "
        "Arms A-C are read-only results from the preceding matched experiment. "
        "No held-out access, reference promotion, commit or push occurred.",
        "",
        "## Result",
        "",
        "The enrichment-only model selects epoch 8 for both criteria. It does not improve over A: "
        "Recall@10 is 0.76885 versus 0.77104, NDCG@10 is 0.60034 versus 0.60223, and new hits@10 "
        "are 248 versus 255. D has slightly higher Recall@25 (0.85978 versus 0.85816). "
        "The paired intervals for Recall@10, NDCG@10 and new recall@10 versus A all include zero. "
        "This run supplies no evidence to replace A; it does not establish that enrichment is "
        "universally worse. D is smaller/faster than C but slower than A.",
        "",
        "## Architecture and controlled inputs",
        "",
        "D uses the same three pre-attention residual MLPs as C: candidate `[128 + 41] -> 128 -> 128`, "
        "profile Token 0 `[128 + 8] -> 128 -> 128`, and history Tokens 1-32 `[128 + 9] -> 128 -> 128`. "
        "Candidate-set attention is disabled and frozen. Customer-memory self-attention and "
        "candidate-to-memory cross-attention remain active. The original 425D scoring head and "
        "bilinear scalar bias remain unchanged. Stage 1 MLP scores are excluded.",
        "",
        "Common initial weights match A exactly; enrichment initialization matches C. "
        "All residual enrichers initially output zero, giving exact initial logit parity with A. "
        "Candidate statistics use the same [0,1] enrichment normalization as C, while legacy raw "
        "scalar/bias paths and signed latent vectors retain their previous representation.",
        "",
        "All arms use the same sealed `rolling_observed_v2` `quota100_50` slates: "
        "26,253 training / 6,191 validation queries, 10,276 total validation targets, 1,857 new targets. "
        "The follow-up refuses mismatched observed-source, feature, score, upstream or slate identities. "
        "A-C selected checkpoints, logits, prediction arrays and histories were verified before reuse; "
        "all 31 reference/rolling-cache evidence hashes and prior selected artifacts remained unchanged.",
        "",
        "## Validation metrics",
        "",
        "Discovery checkpoints maximize new hits@10 subject to the same Recall@10/NDCG@10 guardrails. "
        "NDCG checkpoints maximize NDCG@10 under the original ordering. Results below are complete-basket "
        "macro recalls; new recall uses all 1,857 new target instances, including missing-slate targets.",
    ]
    for selection in ("ndcg", "discovery"):
        lines += ["", f"### {selection.upper()}-selected checkpoints", "",
            "| Arm | Epoch | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | NDCG@10 | MRR | New hits@10 | Repeat micro R@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, a in arms.items():
            entry = a[selection]
            m = entry["metrics"]
            nums = [f"{m[key]:.5f}" for key in (
                "recall@5", "recall@10", "recall@25", "recall@50", "recall@100", "ndcg@10", "mrr"
            )]
            lines.append(f"| {LABELS[name]} | {entry['epoch']} | " + " | ".join(nums)
                         + f" | {m['new_hits@10']}/1,857 | {m['repeat_recall@10_micro']:.5f} |")
        lines += ["", "Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. "
                  "D minus the named comparator; these intervals do not measure training-seed variation.", "",
                  "| Comparator | Metric | Delta | 95% CI |", "| --- | --- | ---: | --- |"]
        for name in (ARM_A_NAME, ARM_B_NAME, ARM_C_NAME):
            for key in ("recall@10_delta", "ndcg@10_delta", "new_recall@10_delta"):
                ci = summary["paired_bootstrap_95ci"][selection][name + "_to_D"][key]
                low, _, high = ci["ci_95"]
                lines.append(f"| {LABELS[name]} | {key} | {ci['point_delta']:+.5f} | [{low:+.5f}, {high:+.5f}] |")
    lines += ["", "## Latency", "",
              "Median prepared-input forward time in ms. Measured again for all four selected NDCG "
              "models in the same benchmark; excludes feature extraction, Stage 1 retrieval and disk. "
              "MPS timings include the synchronization cost of runtime input guards.", "",
              "| Arm | CPU batch 1 | CPU batch 64 | MPS batch 1 | MPS batch 64 | Trainable parameters |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    bench = summary["latency_benchmark"]["devices"]
    for name, arm in arms.items():
        timings = [f"{bench[dev][name][batch]['prepared_on_device_forward']['median_ms']:.3f}"
                   if dev in bench else "n/a" for dev in ("cpu", "mps") for batch in ("batch_1", "batch_64")]
        lines.append(f"| {LABELS[name]} | " + " | ".join(timings) + f" | {arm['trainable_parameters']:,} |")
    lines += ["", "## Correctness review response", "",
        "- **Masked RankerLoss defect confirmed and fixed.** Invalid entries are excluded before BCE "
        "and softmax arithmetic. BCE is averaged over valid entries; listwise loss is averaged over "
        "rows with valid positives. All-masked rows contribute differentiable zero. Masked positive "
        "labels are rejected, and invalid-candidate gradients are zero. The six masked-ranker "
        "training/evaluation call sites now pass the candidate mask. Regression tests cover NaN/-inf "
        "padding, all-masked rows, no positives, variable lengths, unseen weighting, and exact all-valid "
        "value/gradient parity. Historical all-valid slate results are unaffected.",
        "- **Upstream snapshot coherence remains a production prerequisite.** This experiment verifies "
        "the same sealed source/upstream artifacts as A-C. It does not implement automatic refresh "
        "after ingestion or prove arbitrary external vectors share the feature snapshot. A complete "
        "inference snapshot contract remains separate work.",
        "- **Legacy frozen defaults and historical availability limits remain.** New experiment entry "
        "points explicitly select rolling statistics. Historical `CandidateRetriever.from_checkpoint` "
        "still uses the compatibility default; catalogue last-modified timestamps still cannot "
        "reconstruct effective-dated blocking history. Neither policy was silently changed.",
        "- **The review's candidate-attention training status is outdated relative to local work.** "
        "A-C completed ten-epoch fits before this follow-up. Their original evidence remains at "
        "[the three-arm report](../candidate_token_enrichment_v1/REPORT.md).",
        "- Broad data-module refactoring, feature-preparation optimization, documentation cleanup and "
        "basket-aware modeling were not mixed into this architecture comparison.",
        "", "## Limits and artifacts", "",
        "This is one exploratory training seed. Selected-checkpoint inference perturbations measure "
        "dependence on a fitted module, not the isolated causal value of statistics. Frozen towers "
        "and snapshot-unverified metadata remain unchanged. Do not infer final-test performance or "
        "promote a model solely from one metric.",
        "", "- [Machine-readable summary](summary.json)",
        "- [Interactive comparison](comparison.html)",
        "- [Latency measurements](benchmark.json)",
        "- New checkpoints: `checkpoints/pointwise-token-enrichment-v1/runs/`.",
        "", "Reproduction:", "", "```bash",
        "./venv/bin/python -m recommender.experiments.candidate_token_enrichment.pointwise_enrichment --execute-training --device mps",
        "./venv/bin/python -m recommender.experiments.candidate_token_enrichment.pointwise_report",
        "PYTHONPATH=. ./venv/bin/pytest tests/ -q",
        "PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q", "```", "",
    ]
    verification_path = root / "independent_verification.json"
    if verification_path.exists():
        check = json.loads(verification_path.read_text())
        replay_count = len(check["selected_checkpoint_checks"])
        max_diff = max(row["cpu_vs_saved_max_abs_diff"] for row in check["selected_checkpoint_checks"])
        lines += ["", "## Independent verification", "",
                  f"All {replay_count} selected checkpoint records were hash-verified and their saved logits "
                  f"reproduced the ranking metrics. CPU replay on 32 rows per record matched saved MPS "
                  f"logits within {max_diff:.3g}. All {check['preserved_hashes_matched']} frozen reference/rolling "
                  "cache evidence hashes matched. No additional training or held-out access occurred.", "",
                  f"- Full safe suite: {check['full_tests']['passed']} passed, {check['full_tests']['skipped']} skipped.",
                  f"- Synthetic-only suite: {check['synthetic_tests']['passed']} passed, {check['synthetic_tests']['skipped']} skipped.",
                  "- Masked-loss CPU regression and manual MPS check: finite losses/gradients, zero padding gradients.",
                  "- [Independent verification record](independent_verification.json)", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    data = {sel: [{"name": LABELS[name], "epoch": a[sel]["epoch"],
                   "recall": a[sel]["metrics"]["recall@10"] * 100,
                   "ndcg": a[sel]["metrics"]["ndcg@10"],
                   "hits": a[sel]["metrics"]["new_hits@10"]} for name, a in arms.items()]
            for sel in ("ndcg", "discovery")}
    page = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans)}.intro,.note{font-size:12px;color:var(--muted-foreground);line-height:1.5}.buttons{display:flex;gap:8px;margin:12px 0}button{padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius);font:inherit;font-size:13px;color:var(--foreground);background:var(--secondary);cursor:pointer}button[aria-pressed=true]{background:var(--primary);color:var(--primary-foreground)}.panel{background:var(--card);border-radius:var(--radius);padding:14px;margin:10px 0}.title{font-weight:600;font-size:14px}svg{display:block;width:100%;height:auto}text{fill:var(--foreground);font:12px var(--font-sans)}
</style></head><body><div class="intro">Stage 2 · 6,191 validation queries · same rolling slates · seed 42<br>D adds statistical token enrichment to the pointwise scorer. Only D was trained in this follow-up.</div><div class="buttons"><button id="ndcg" aria-pressed="true">NDCG-selected</button><button id="discovery" aria-pressed="false">Discovery-selected</button></div><div id="plots"></div><div class="note">All bars start at zero. New-hit denominator: all 1,857 new targets. No held-out access or reference promotion.</div><script>
const data=DATA;function render(sel){for(const name of ['ndcg','discovery'])document.getElementById(name).setAttribute('aria-pressed',name===sel);document.getElementById('plots').innerHTML=[['recall','Overall Recall@10',100,v=>v.toFixed(3)+'%'],['ndcg','NDCG@10',1,v=>v.toFixed(5)],['hits','New-product hits@10',300,v=>v+' / 1,857']].map(([key,title,max,fmt])=>'<div class="panel"><div class="title">'+title+'</div><svg viewBox="0 0 700 145" role="img" aria-label="'+title+'">'+data[sel].map((r,i)=>{let y=24+i*33;return '<text x="0" y="'+y+'">'+r.name+'</text><rect x="235" y="'+(y-12)+'" height="17" width="'+(r[key]/max*280)+'" rx="3" style="fill:var(--chart-'+(i+1)+')"/><text x="530" y="'+y+'">'+fmt(r[key])+'</text>'}).join('')+'</svg></div>').join('')}for(const sel of ['ndcg','discovery'])document.getElementById(sel).onclick=()=>render(sel);render('ndcg');</script></body></html>'''.replace('DATA', json.dumps(data))
    (root / "comparison.html").write_text(page)


if __name__ == "__main__":
    write_reports()
