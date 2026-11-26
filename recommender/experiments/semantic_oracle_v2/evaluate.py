"""Evaluate feature-parity v2 LLM predictions against exact matched rolling100 baselines.

Strictly validation-only. Does not touch semantic_oracle_v1 or any preserved checkpoints.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

from recommender.experiments.semantic_oracle.evaluate import (
    aggregate,
    case_metrics,
    paired_bootstrap,
)
from recommender.experiments.semantic_oracle.protocol import read_json, save_json

V1_ROOT = Path("data/interim/semantic_oracle_v1")
V2_ROOT = Path("data/interim/semantic_oracle_v2_parity")


def extract_json_object(text: str) -> dict:
    s = text.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return json.loads(s)


def validate_and_save_prediction(case_entry: dict, raw_text: str, meta: dict | None = None) -> dict:
    obj = extract_json_object(raw_text)
    cid = case_entry["case_id"]
    if obj.get("case_id") != cid:
        raise ValueError(f"case_id mismatch: expected {cid}, got {obj.get('case_id')}")
    rows = obj.get("ranked_products")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError(f"Expected 10 ranked_products, got {len(rows) if isinstance(rows, list) else type(rows)}")
    ranks = [int(r["rank"]) for r in rows]
    if ranks != list(range(1, 11)):
        raise ValueError(f"Ranks must be 1..10, got {ranks}")
    pids = [str(r["product_id"]).strip() for r in rows]
    valid = set(case_entry["shuffled_candidate_ids"])
    if len(set(pids)) != 10:
        raise ValueError(f"Duplicate product_ids in {cid}: {pids}")
    bad = [p for p in pids if p not in valid]
    if bad:
        raise ValueError(f"Invalid candidate product_ids in {cid}: {bad}")
    payload = {
        "case_id": cid,
        "idx": case_entry["idx"],
        "ranked_products": [
            {"rank": int(r["rank"]), "product_id": str(r["product_id"]).strip(), "reason": str(r.get("reason", "")).strip()}
            for r in rows
        ],
        "meta": meta or {},
    }
    out_path = V2_ROOT / "predictions" / f"{case_entry['idx']:02d}_{cid}.json"
    save_json(out_path, payload, exclusive=False)
    return payload


def evaluate_v2(v2_root: Path = V2_ROOT) -> dict:
    manifest = read_json(v2_root / "manifest.json")
    pred_dir = v2_root / "predictions"
    preds = {}
    for entry in manifest["cases_index"]:
        p = pred_dir / f"{entry['idx']:02d}_{entry['case_id']}.json"
        if p.exists():
            preds[entry["case_id"]] = read_json(p)

    done_ids = [e["case_id"] for e in manifest["cases_index"] if e["case_id"] in preds]
    if not done_ids:
        raise ValueError("No completed predictions found.")

    # Slice truth strictly to completed cases
    raw_truth = read_json(V1_ROOT / "vault" / "truth.json")
    truth = {cid: raw_truth[cid] for cid in done_ids}
    del raw_truth

    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    llm_only_hits = []
    neural_only_hits = []

    for cid in done_ids:
        case = truth[cid]
        cands = case["candidates"]["rolling100"]
        for bname, brank in case["baseline_rankings"]["rolling100"].items():
            by_model[bname][cid] = case_metrics(brank, case["targets"], case["new_targets"], cands)

        pred = preds[cid]
        ranking = [r["product_id"] for r in pred["ranked_products"]]
        by_model["llm_hydrogen_feature_parity"][cid] = case_metrics(
            ranking, case["targets"], case["new_targets"], cands
        )

        neural_rank = case["baseline_rankings"]["rolling100"]["pointwise425_ndcg"]
        neural_pos = {p: i + 1 for i, p in enumerate(neural_rank)}
        cb_rank = case["baseline_rankings"]["rolling100"]["catboost_original17_causal_retrain"]
        cb_pos = {p: i + 1 for i, p in enumerate(cb_rank)}
        neural_top10 = set(neural_rank[:10])
        llm_top10 = set(ranking[:10])
        tset = set(case["targets"])
        nset = set(case["new_targets"])

        for r in pred["ranked_products"]:
            pid = r["product_id"]
            if pid in (llm_top10 & tset) - neural_top10:
                llm_only_hits.append({
                    "case_id": cid, "product_id": pid, "is_new": pid in nset,
                    "llm_rank": r["rank"], "neural_rank": neural_pos.get(pid), "cb_rank": cb_pos.get(pid),
                    "reason": r.get("reason", ""),
                })
        for pid in (neural_top10 & tset) - llm_top10:
            neural_only_hits.append({
                "case_id": cid, "product_id": pid, "is_new": pid in nset,
                "neural_rank": neural_pos.get(pid), "cb_rank": cb_pos.get(pid),
            })

    agg = {m: aggregate([by_model[m][cid] for cid in done_ids]) for m in by_model}
    customers = [truth[cid]["company_id"] for cid in done_ids]
    pairs = [
        ("pointwise425_ndcg", "llm_hydrogen_feature_parity"),
        ("selector301_rolling", "llm_hydrogen_feature_parity"),
        ("catboost_original17_causal_retrain", "llm_hydrogen_feature_parity"),
        ("personal_repeat_prior", "llm_hydrogen_feature_parity"),
        ("popularity_prior", "llm_hydrogen_feature_parity"),
    ]
    boot = {}
    for a, b in pairs:
        boot[f"{a}__to__{b}"] = paired_bootstrap(
            [by_model[a][cid] for cid in done_ids],
            [by_model[b][cid] for cid in done_ids],
            customers,
            seed=42,
            draws=2000,
        )

    # Also compute exact 16-case overlap comparison with semantic_oracle_v1 if available
    v1_eval_path = V1_ROOT / "interim_evaluation_16cases.json"
    overlap16 = None
    if v1_eval_path.exists():
        v1_ev = read_json(v1_eval_path)
        v1_16 = [cid for cid in v1_ev["case_ids"] if cid in preds]
        if len(v1_16) == 16:
            overlap16 = {
                m: aggregate([by_model[m][cid] for cid in v1_16])
                for m in ("pointwise425_ndcg", "selector301_rolling", "catboost_original17_causal_retrain", "llm_hydrogen_feature_parity")
            }
            for arm in ("llm_customer_only", "llm_global_history", "llm_reduced_semantic"):
                overlap16["v1_" + arm] = v1_ev["aggregate"]["rolling100"][arm]

    out = {
        "evaluated_cases": len(done_ids),
        "benchmark": "rolling100",
        "aggregate": agg,
        "paired_customer_bootstrap95": boot,
        "overlap16_vs_v1": overlap16,
        "llm_only_hits": llm_only_hits,
        "neural_only_hits": neural_only_hits,
    }
    save_json(v2_root / "evaluation_v2.json", out, exclusive=False)

    print(f"\n=== FEATURE-PARITY V2 EVALUATION (rolling100, n={len(done_ids)} cases) ===")
    order = [
        "pointwise425_ndcg",
        "selector301_rolling",
        "catboost_original17_causal_retrain",
        "personal_repeat_prior",
        "popularity_prior",
        "llm_hydrogen_feature_parity",
    ]
    hdr = f"{'Model':36s} | {'Rec@10':>7s} | {'NDCG@10':>7s} | {'New@10':>13s} | {'Rep@10':>14s} | {'CondNew@10':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for m in order:
        r = agg[m]
        nr = f"{r['new_recall@10']*100:5.2f}% ({r['new_hits@10']}/{r['new_targets']})"
        rep_tot = r["targets"] - r["new_targets"]
        rep_hits = round(r["repeat_recall@10"] * rep_tot)
        rr = f"{r['repeat_recall@10']*100:5.2f}% ({rep_hits}/{rep_tot})"
        cnr = f"{r['conditional_new_recall@10']*100:6.2f}%" if r["conditional_new_recall@10"] is not None else "   n/a"
        print(f"{m:36s} | {r['recall@10']*100:6.2f}% | {r['ndcg@10']:7.4f} | {nr:>13s} | {rr:>14s} | {cnr:>10s}")
    ref = agg["pointwise425_ndcg"]
    print(
        f"Slate Coverage: macro={ref['candidate_coverage_macro']*100:.2f}%, micro={ref['candidate_coverage_micro']*100:.2f}%, "
        f"new_in_slate={ref['new_slate_coverage']*100:.2f}%"
    )
    print(
        f"Candidate-Constrained Ceilings: Recall@10={ref['candidate_constrained_recall_ceiling']*100:.2f}%, "
        f"NDCG@10={ref['candidate_constrained_ndcg_ceiling']:.4f}, "
        f"New@10={ref['candidate_constrained_new_recall_ceiling']*100:.2f}%, "
        f"Rep@10={ref['candidate_constrained_repeat_recall_ceiling']*100:.2f}%"
    )
    print("\n=== PAIRED BOOTSTRAP 95% CIs ===")
    for k, v in boot.items():
        rec, nd, nw, rp = v["recall@10"], v["ndcg@10"], v["new_recall@10"], v["repeat_recall@10"]
        print(
            f"  {k:60s} | dRec={rec['absolute_percentage_points']:+6.2f}pp [{rec['ci95'][0]*100:+6.2f},{rec['ci95'][1]*100:+6.2f}] "
            f"| dNDCG={nd['delta']:+6.4f} [{nd['ci95'][0]:+6.4f},{nd['ci95'][1]:+6.4f}] "
            f"| dNew={nw['absolute_percentage_points']:+6.2f}pp [{nw['ci95'][0]*100:+6.2f},{nw['ci95'][1]*100:+6.2f}] "
            f"| dRep={rp['absolute_percentage_points']:+6.2f}pp"
        )
    print(f"\nLLM-only hits vs pointwise425_ndcg: {len(llm_only_hits)} (new={sum(1 for h in llm_only_hits if h['is_new'])})")
    for h in llm_only_hits[:15]:
        print(f"  + {h['case_id']} {h['product_id']} new={h['is_new']} llm=#{h['llm_rank']} pw425=#{h['neural_rank']} cb=#{h['cb_rank']} | {h['reason'][:120]}")
    print(f"\nNeural-only hits vs llm_hydrogen: {len(neural_only_hits)} (new={sum(1 for h in neural_only_hits if h['is_new'])})")
    for h in neural_only_hits[:15]:
        print(f"  - {h['case_id']} {h['product_id']} new={h['is_new']} pw425=#{h['neural_rank']} cb=#{h['cb_rank']}")
    return out


if __name__ == "__main__":
    evaluate_v2()
