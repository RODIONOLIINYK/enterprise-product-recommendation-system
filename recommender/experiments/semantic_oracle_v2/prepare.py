"""Prepare 50 blind feature-parity prompts matching exact Stage 1 + Stage 2 rolling_observed_v2 inputs.

Strictly validation-only (< 2026-02-19). Zero edits to semantic_oracle_v1 or preserved artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiments.candidate_token_enrichment.train import snapshot_preserved_hashes
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES
from recommender.experiments.semantic_oracle.protocol import alias, file_sha, private_dir, read_json, save_json
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START

V1_ROOT = Path("data/interim/semantic_oracle_v1")
V2_ROOT = Path("data/interim/semantic_oracle_v2_parity")
VAL_SLATES_PATH = Path("data/interim/rolling_candidate_v2/v3/val_slates.pt")


def _clean(v, max_len: int = 80) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    s = " ".join(str(v).replace("|", "/").replace("\n", " ").split())
    return s[:max_len] if s else "-"


def _f(v: float, digits: int = 2) -> str:
    x = float(v)
    if abs(x) < 10 ** (-(digits + 1)):
        return "0"
    s = f"{x:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def build_prompts(v2_root: Path = V2_ROOT, seed: int = 42) -> dict:
    before_hashes = snapshot_preserved_hashes()
    private_dir(v2_root)
    private_dir(v2_root / "prompts")
    private_dir(v2_root / "predictions")

    # Load v1 truth ONLY to map exact 50 validation transition_keys & verify candidate parity
    # (never printed or exposed into prompts)
    v1_truth = read_json(V1_ROOT / "vault" / "truth.json")
    v1_jobs = read_json(V1_ROOT / "jobs.json")["jobs"]
    ordered_case_ids = []
    seen_cids = set()
    for j in v1_jobs:
        if j["case_id"] not in seen_cids:
            seen_cids.add(j["case_id"])
            ordered_case_ids.append(j["case_id"])
    assert len(ordered_case_ids) == 50, f"Expected 50 cases, got {len(ordered_case_ids)}"

    # Load sealed rolling100 val slates (exact 41 features fed to Stage 1 & Stage 2)
    slates = torch.load(VAL_SLATES_PATH, map_location="cpu", weights_only=False)
    assert len(DYNAMIC_FEATURE_NAMES) == 41
    feat_idx = {name: i for i, name in enumerate(DYNAMIC_FEATURE_NAMES)}

    # Build lookup from (alias(customer_id), prediction_date_str) -> val_slates row index
    # Note: prediction_days is days since 2025-01-01? Let's match via exact candidate set + customer alias!
    tkey_to_row = {}
    cand_sig_to_row = {}
    for row_i, (cid_raw, cand_pids) in enumerate(zip(slates["customer_ids"], slates["candidate_pids"])):
        cid_alias = alias(str(cid_raw), kind="company")
        sig = (cid_alias, tuple(sorted(cand_pids)))
        cand_sig_to_row[sig] = row_i

    # Load strictly development purchases (< TEST_START) + catalogue + companies
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    assert pd.to_datetime(purchases["purchase_date"]).max() < TEST_START, "Held-out leakage detected!"
    catalogue_df = load_product_catalogue(get_datasets_locs("items"))
    companies_df = load_companies(get_datasets_locs("companies"), purchases=purchases, minimum_purchases=1)
    paid = purchases.copy()
    paid["purchase_date"] = pd.to_datetime(paid["purchase_date"])
    paid["customer_id"] = paid["customer_id"].astype(str)
    paid["product_id"] = paid["product_id"].astype(str)
    paid = paid.sort_values(["purchase_date", "customer_id", "product_id"]).reset_index(drop=True)

    # Index catalogue & companies
    cat_map = {}
    for row in catalogue_df.to_dict(orient="records"):
        cat_map[str(row["product_id"])] = row
    comp_map = {}
    for row in companies_df.to_dict(orient="records"):
        comp_map[str(row["customer_id"])] = row

    # Pre-index purchases by customer and by product for fast strictly-prior lookups
    cust_groups = {cid: grp.sort_values("purchase_date") for cid, grp in paid.groupby("customer_id", sort=False)}
    sku_dates = {}
    sku_cids = {}
    for pid, grp in paid.groupby("product_id", sort=False):
        g = grp.sort_values("purchase_date")
        sku_dates[pid] = g["purchase_date"].to_numpy(dtype="datetime64[ns]")
        sku_cids[pid] = g["customer_id"].to_numpy()

    cases_index = []
    prompt_lengths = []

    for idx_case, case_id in enumerate(ordered_case_ids):
        t_entry = v1_truth[case_id]
        comp_alias = t_entry["company_id"]
        pred_date_str = t_entry["prediction_date"]
        pred_ts = pd.Timestamp(pred_date_str)
        assert pred_ts < TEST_START
        cands_v1 = t_entry["candidates"]["rolling100"]
        sig = (comp_alias, tuple(sorted(cands_v1)))
        assert sig in cand_sig_to_row, f"Could not align {case_id} to val_slates.pt!"
        row_i = cand_sig_to_row[sig]

        raw_cid = str(slates["customer_ids"][row_i])
        assert alias(raw_cid, kind="company") == comp_alias
        slate_pids = list(slates["candidate_pids"][row_i])
        pw = slates["pairwise_features"][row_i].numpy()  # (100, 41)
        assert len(slate_pids) == 100 and pw.shape == (100, 41)

        # Strictly prior purchases for this customer (< pred_ts)
        c_hist = cust_groups.get(raw_cid)
        if c_hist is not None:
            c_hist = c_hist[c_hist["purchase_date"] < pred_ts]
        else:
            c_hist = paid.iloc[0:0]

        # Exact raw customer lifetime & cadence stats (matching Stage 1 + Stage 2 Token 0)
        order_dates = sorted(c_hist["purchase_date"].unique())
        n_orders = len(order_dates)
        n_events = len(c_hist)
        total_qty = float(c_hist["quantity"].sum()) if n_events else 0.0
        n_prods = int(c_hist["product_id"].nunique()) if n_events else 0
        n_cats = int(c_hist["product_id"].map(lambda p: cat_map.get(p, {}).get("product_category", "")).nunique()) if n_events else 0
        if n_orders > 1:
            gaps = [(order_dates[i] - order_dates[i - 1]).days for i in range(1, n_orders)]
            med_gap = float(np.median(gaps))
        else:
            med_gap = 30.0
        med_qty = float(c_hist["quantity"].median()) if n_events else 0.0
        days_since_last = int((pred_ts - order_dates[-1]).days) if n_orders else 365
        overdue_ratio = days_since_last / max(med_gap, 1.0)
        orders_30d = sum(1 for d in order_dates if d > pred_ts - pd.Timedelta(days=30))
        orders_90d = sum(1 for d in order_dates if d > pred_ts - pd.Timedelta(days=90))

        # Per-SKU customer purchase stats
        sku_cust_stats = {}
        for pid, grp in c_hist.groupby("product_id", sort=False):
            dts = sorted(grp["purchase_date"].unique())
            g_days = [(dts[i] - dts[i - 1]).days for i in range(1, len(dts))]
            sku_cust_stats[pid] = {
                "n": len(dts),
                "sum_q": float(grp["quantity"].sum()),
                "last_q": float(grp.iloc[-1]["quantity"]),
                "days_ago": int((pred_ts - dts[-1]).days),
                "med_gap": float(np.median(g_days)) if g_days else None,
            }

        # Company profile metadata
        cinfo = comp_map.get(raw_cid, {})
        comp_industry = _clean(cinfo.get("industry_branch"), 40)
        comp_desc = _clean(cinfo.get("description"), 180)
        comp_city = _clean(cinfo.get("city"), 40)
        comp_region = _clean(cinfo.get("region_code"), 30)
        comp_branch = _clean(cinfo.get("sales_branch"), 40)

        # Month seasonality (exact Stage 2 values)
        m_sin = float(pw[0, feat_idx["month_sin"]])
        m_cos = float(pw[0, feat_idx["month_cos"]])

        # Deterministically shuffle candidate order (blinding: never leak Stage 1/2 order!)
        rng = random.Random(int(hashlib.sha256(f"parity-v2:{seed}:{case_id}".encode()).hexdigest()[:12], 16))
        perm = list(range(len(slate_pids)))
        rng.shuffle(perm)
        shuffled_pids = [slate_pids[i] for i in perm]

        # Raw global prior counts (< pred_ts) for each candidate (interpretable companion to normalized features)
        pred_64 = pred_ts.to_datetime64()
        s7_64 = (pred_ts - pd.Timedelta(days=7)).to_datetime64()
        s30_64 = (pred_ts - pd.Timedelta(days=30)).to_datetime64()
        s90_64 = (pred_ts - pd.Timedelta(days=90)).to_datetime64()

        # Build prompt text
        lines = []
        lines.append(
            f"TASK: Rank the TOP 10 most likely products to appear in this B2B customer's NEXT order on {pred_date_str} "
            f"from the 100 shuffled candidate SKUs below. Output ONLY strict JSON (no markdown fences, no commentary, no tool calls)."
        )
        lines.append("")
        lines.append("## 1. CUSTOMER PROFILE & LIFETIME STATS (strictly < prediction_date)")
        lines.append(
            f"CASE: {case_id} | DATE: {pred_date_str} (month_sin={_f(m_sin)}, month_cos={_f(m_cos)}) | COMPANY: {comp_alias}"
        )
        lines.append(
            f"INDUSTRY: {comp_industry} | CITY: {comp_city} | REGION: {comp_region} | BRANCH: {comp_branch}"
        )
        lines.append(f"DESC: {comp_desc}")
        lines.append(
            f"LIFETIME: orders={n_orders}, events={n_events}, total_qty={_f(total_qty,1)}, distinct_skus={n_prods}, "
            f"distinct_cats={n_cats}, med_order_gap={_f(med_gap,1)}d, med_qty={_f(med_qty,1)}, "
            f"days_since_last={days_since_last}d, overdue_ratio={_f(overdue_ratio,2)}, orders_30d={orders_30d}, orders_90d={orders_90d}"
        )

        # Last 32 purchase events (exact Stage 1 & Stage 2 sequence window)
        lines.append("")
        lines.append(f"## 2. RECENT PURCHASE SEQUENCE (last {min(32, n_events)} of {n_events} events, chronological)")
        recent_32 = c_hist.tail(32)
        prev_dt = None
        for r in recent_32.itertuples(index=False):
            dt = r.purchase_date
            gap_s = "-" if prev_dt is None else f"{(dt - prev_dt).days}d"
            prev_dt = dt
            cm = cat_map.get(r.product_id, {})
            nm = _clean(cm.get("short_name") or cm.get("product_name"), 52)
            fam = _clean(cm.get("product_family"), 18)
            cont = _clean(cm.get("container_type"), 14)
            lines.append(
                f"{dt.strftime('%Y-%m-%d')} (+{gap_s}) | {r.product_id} | qty={_f(r.quantity,1)} | "
                f"{_clean(cm.get('brand'),12)}/{_clean(cm.get('product_category'),16)}/{fam}/{cont} | {nm}"
            )

        # If customer has older SKUs beyond last 32 events, list their summary too
        in_last32 = set(recent_32["product_id"])
        older_skus = [p for p in sku_cust_stats if p not in in_last32]
        if older_skus:
            older_skus.sort(key=lambda p: (-sku_cust_stats[p]["n"], sku_cust_stats[p]["days_ago"]))
            lines.append(f"OLDER_PRIOR_SKUS (outside last-32 window, {len(older_skus)} SKUs):")
            for p in older_skus[:25]:
                st = sku_cust_stats[p]
                cm = cat_map.get(p, {})
                lines.append(
                    f"  {p} | n={st['n']} sum_q={_f(st['sum_q'],1)} ago={st['days_ago']}d | "
                    f"{_clean(cm.get('brand'),12)}/{_clean(cm.get('product_family'),18)} | {_clean(cm.get('short_name'),45)}"
                )

        # Candidate table with ALL 41 exact rolling_observed_v2 features + raw counts + catalogue text
        lines.append("")
        lines.append("## 3. 100 SHUFFLED CANDIDATES WITH EXACT STAGE-1/STAGE-2 CAUSAL FEATURES (< prediction_date)")
        lines.append(
            "FIELD KEY: "
            "PREV=repeat SKU(1/0); uN=user orders(raw/norm); uQ=user qty share; uAgo=days since user buy(raw/norm); urg=sku urgency; "
            "uSh=user(cat/sub/br/fam) shares; repF=repeat flags(cat/br/line); "
            "cCad=cat(age/ovd/never); fCad=fam(age/ovd/never); "
            "gPop=global_pop_norm(raw_life_ord/buyers); indAff=industry_sku_affinity; "
            "dem=global_orders(7d/30d/90d/buyers30d) [norm:7/30/90/b30]; gr30=30d_growth[-1..1]; catRel=30d_cat_share; gAge=global_sale_age; "
            "lift=cross_co_lifts(skuLast/skuRec3/catL/famL/lineL/supConf/backoff/sameBask)"
        )

        for rank_pos, orig_i in enumerate(perm, start=1):
            pid = slate_pids[orig_i]
            fvec = pw[orig_i]
            cm = cat_map.get(pid, {})
            cst = sku_cust_stats.get(pid)

            # Raw global counts strictly < pred_ts
            dts = sku_dates.get(pid)
            if dts is not None:
                stop = int(dts.searchsorted(pred_64, side="left"))
                s7 = int(dts.searchsorted(s7_64, side="left"))
                s30 = int(dts.searchsorted(s30_64, side="left"))
                s90 = int(dts.searchsorted(s90_64, side="left"))
                n_life = stop
                n_life_b = len(set(sku_cids[pid][:stop])) if stop > 0 else 0
                n7, n30, n90 = stop - s7, stop - s30, stop - s90
                b30 = len(set(sku_cids[pid][s30:stop])) if stop > s30 else 0
            else:
                n_life = n_life_b = n7 = n30 = n90 = b30 = 0

            is_rep = int(round(fvec[feat_idx["is_repeat_sku"]]))
            u_n_raw = cst["n"] if cst else 0
            u_ago_raw = f"{cst['days_ago']}d" if cst else "-"
            u_n_norm = _f(fvec[feat_idx["user_sku_order_count_norm"]])
            u_q = _f(fvec[feat_idx["user_sku_qty_share"]])
            u_ago_norm = _f(fvec[feat_idx["days_since_last_sku_buy_norm"]])
            urg = _f(fvec[feat_idx["sku_urgency"]])

            u_sh = "/".join([
                _f(fvec[feat_idx["user_cat_order_share"]]),
                _f(fvec[feat_idx["user_subcat_order_share"]]),
                _f(fvec[feat_idx["user_brand_order_share"]]),
                _f(fvec[feat_idx["family_history_share"]]),
            ])
            rep_f = f"{int(round(fvec[feat_idx['is_repeat_category']]))}{int(round(fvec[feat_idx['is_repeat_brand']]))}{int(round(fvec[feat_idx['is_repeat_business_line']]))}"
            c_cad = f"{_f(fvec[feat_idx['category_age']])}/{_f(fvec[feat_idx['category_overdue_ratio']])}/{int(round(fvec[feat_idx['category_never_seen']]))}"
            f_cad = f"{_f(fvec[feat_idx['family_age']])}/{_f(fvec[feat_idx['family_overdue_ratio']])}/{int(round(fvec[feat_idx['family_never_seen']]))}"

            g_pop = _f(fvec[feat_idx["global_sku_popularity_norm"]])
            ind_aff = _f(fvec[feat_idx["industry_sku_affinity"]])
            gr30 = _f(fvec[feat_idx["sku_growth_30d"]])
            cat_rel = _f(fvec[feat_idx["category_relative_share"]])
            g_age = _f(fvec[feat_idx["global_sale_age"]])
            lifts = "/".join([
                _f(fvec[feat_idx["sku_last_lift"]]),
                _f(fvec[feat_idx["sku_recent3_lift"]]),
                _f(fvec[feat_idx["category_last_lift"]]),
                _f(fvec[feat_idx["family_last_lift"]]),
                _f(fvec[feat_idx["line_last_lift"]]),
                _f(fvec[feat_idx["sku_support_confidence"]]),
                _f(fvec[feat_idx["hierarchical_backoff"]]),
                _f(fvec[feat_idx["same_basket_lift"]]),
            ])

            brand = _clean(cm.get("brand"), 14)
            cat_sub = f"{_clean(cm.get('product_category'),16)}/{_clean(cm.get('product_subcategory'),16)}"
            line_fam = f"{_clean(cm.get('business_line'),12)}/{_clean(cm.get('product_family'),18)}"
            pack = f"{_clean(cm.get('container_type'),12)}/{_clean(cm.get('unit'),6)}"
            tier_orig = f"{_clean(cm.get('tier'),10)}/{_clean(cm.get('country_of_origin'),10)}"
            name = _clean(cm.get("short_name") or cm.get("product_name"), 56)

            lines.append(
                f"{rank_pos:03d}|{pid}|PREV={is_rep}|"
                f"{brand}|{cat_sub}|{line_fam}|{pack}|{tier_orig}|{name}|"
                f"uN={u_n_raw}({u_n_norm}) uQ={u_q} uAgo={u_ago_raw}({u_ago_norm}) urg={urg}|"
                f"uSh={u_sh} repF={rep_f} cCad={c_cad} fCad={f_cad}|"
                f"gPop={g_pop}({n_life}o/{n_life_b}c) indAff={ind_aff} dem={n7}/{n30}/{n90}/{b30} gr30={gr30} catRel={cat_rel} gAge={g_age}|"
                f"lift={lifts}"
            )

        lines.append("")
        lines.append("## OUTPUT FORMAT (strict JSON only, no markdown, no tools)")
        lines.append(
            f'{{"case_id":"{case_id}","ranked_products":['
            '{"rank":1,"product_id":"<exact_pid_from_100_candidates>","reason":"<brief 1-sentence reason>"},'
            '... exactly 10 unique product_ids ranked 1 to 10]}}'
        )

        prompt_text = "\n".join(lines) + "\n"
        prompt_path = v2_root / "prompts" / f"{idx_case:02d}_{case_id}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        prompt_path.chmod(0o600)
        prompt_lengths.append(len(prompt_text))

        cases_index.append({
            "idx": idx_case,
            "case_id": case_id,
            "company_id": comp_alias,
            "prediction_date": pred_date_str,
            "val_slates_row": row_i,
            "candidate_count": len(shuffled_pids),
            "shuffled_candidate_ids": shuffled_pids,
            "prompt_path": str(prompt_path.resolve()),
            "prompt_chars": len(prompt_text),
            "prompt_sha256": file_sha(prompt_path),
        })

    assert snapshot_preserved_hashes() == before_hashes, "Preserved hashes changed!"

    manifest = {
        "version": "semantic_oracle_v2_feature_parity",
        "heldout_access": False,
        "training_executed": False,
        "promoted": False,
        "benchmark": "rolling100",
        "slate_size": 100,
        "cases": len(cases_index),
        "features_included": list(DYNAMIC_FEATURE_NAMES),
        "val_slates_sha256": file_sha(VAL_SLATES_PATH),
        "prompt_chars_min": min(prompt_lengths),
        "prompt_chars_avg": round(sum(prompt_lengths) / len(prompt_lengths), 1),
        "prompt_chars_max": max(prompt_lengths),
        "cases_index": cases_index,
    }
    save_json(v2_root / "manifest.json", manifest, exclusive=False)
    print(
        f"Prepared {len(cases_index)} blind feature-parity prompts in {v2_root} "
        f"(chars min={manifest['prompt_chars_min']}, avg={manifest['prompt_chars_avg']}, max={manifest['prompt_chars_max']})."
    )
    return manifest


if __name__ == "__main__":
    build_prompts()
