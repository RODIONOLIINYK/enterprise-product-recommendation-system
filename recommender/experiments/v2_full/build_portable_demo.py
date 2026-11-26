"""Build a polished, zero-install portable ZIP demo with the promoted `w2_2mem_recency_hl365` model built in.

Runs real PyTorch inference via `V2FullLiveDemoEngine` over a stratified set of out-of-time
`v2_full` Validation baskets (strictly < 2026-07-01, zero held-out Test access), pre-extracts
each case's exact Top-15 Repeat + Top-15 Unseen neural logits so the standalone HTML/JS app
performs mathematically exact client-side Top-10 ranking across all 3 modes (Blended, Pure
Cross-Sell, Repeat Only) plus a live "What-If Exploration Boost" slider.

Outputs:
    exports/B2B_Recommender_Demo_v2_final.zip
      ├── OPEN_DEMO.html          (self-contained bilingual UA/EN interactive web app)
      ├── README_START.txt        (quick-start guide in UA & EN)
      └── model/
          ├── best_discovery.pt   (promoted w2_2mem_recency_hl365 Ep 10 checkpoint)
          └── v2_full_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple
import zipfile

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import file_digest

from .config import V2_FULL_REFERENCE_PATH
from .demo_engine import C42_IDX, U11_IDX, V2FullLiveDemoEngine
from .extra_features import assemble_cand_42, prune_cust_7


DEFAULT_OUTPUT_ZIP = Path("exports/B2B_Recommender_Demo_v2_final.zip")


def _extract_drivers(
    f_row: np.ndarray,
    *,
    is_new_candidate: bool,
) -> List[Dict[str, Any]]:
    """Return structured, localizable signal drivers from exact causal cand_42 features."""
    out: List[Dict[str, Any]] = []
    if f_row[C42_IDX["has_bought_base_sku"]] > 0.5:
        approx_orders = max(
            1,
            int(round(math.expm1(float(f_row[C42_IDX["user_base_sku_order_count_norm"]]) * math.log1p(50.0)))),
        )
        approx_days = int(
            round(math.expm1(float(f_row[C42_IDX["days_since_last_base_sku_buy_norm"]]) * math.log1p(365.0)))
        )
        out.append({"t": "repeat", "n": approx_orders, "d": approx_days})
        urg = float(f_row[C42_IDX["base_sku_urgency"]]) * 5.0
        if urg >= 0.6:
            out.append({"t": "due", "v": round(urg, 1)})
    else:
        cf_rec = float(f_row[C42_IDX["full_cf_recency_weighted"]])
        cf_cos = float(f_row[C42_IDX["full_cf_cosine"]])
        if cf_rec > 0.08:
            out.append({"t": "cf_rec", "v": round(cf_rec, 2)})
        elif cf_cos > 0.08:
            out.append({"t": "cf_cos", "v": round(cf_cos, 2)})

    if f_row[C42_IDX["base_name_recent3_lift"]] > 0.15:
        out.append({"t": "seq_lift", "v": round(float(f_row[C42_IDX["base_name_recent3_lift"]]), 2)})
    if f_row[C42_IDX["mgr_base_name_share_30d"]] > 0.02:
        out.append({"t": "mgr30", "v": round(100.0 * float(f_row[C42_IDX["mgr_base_name_share_30d"]]), 1)})
    elif f_row[C42_IDX["mgr_industry_base_name_share_90d"]] > 0.02:
        out.append({"t": "mgr_ind", "v": round(100.0 * float(f_row[C42_IDX["mgr_industry_base_name_share_90d"]]), 1)})
    elif f_row[C42_IDX["mgr_base_name_share_90d"]] > 0.02:
        out.append({"t": "mgr90", "v": round(100.0 * float(f_row[C42_IDX["mgr_base_name_share_90d"]]), 1)})
    if f_row[C42_IDX["region_base_name_share_90d"]] > 0.025 and is_new_candidate:
        out.append({"t": "reg90", "v": round(100.0 * float(f_row[C42_IDX["region_base_name_share_90d"]]), 1)})
    if f_row[C42_IDX["base_name_industry_affinity"]] > 0.03:
        out.append({"t": "ind", "v": round(100.0 * float(f_row[C42_IDX["base_name_industry_affinity"]]), 1)})
    if f_row[C42_IDX["app_matches_company_equipment"]] > 0.5:
        out.append({"t": "equip"})
    if f_row[C42_IDX["user_viscosity_order_share"]] > 0.0 and is_new_candidate:
        out.append({"t": "visc", "v": round(100.0 * float(f_row[C42_IDX["user_viscosity_order_share"]]), 0)})
    return out[:4]


def _score_case_for_export(
    engine: V2FullLiveDemoEngine,
    idx: int,
    *,
    top_k_each: int = 15,
) -> Dict[str, Any]:
    """Run real neural inference for one validation case and export exact Top-K repeat + unseen logits."""
    cid = engine.customer_ids[idx]
    date = engine.pred_dates[idx]
    is_first = bool(engine.first_obs_np[idx])
    target_row = engine.target_np[idx]
    new_row = engine.new_target_np[idx]
    allowed_row = engine.allowed_np[idx]
    unseen_row = engine.unseen_np[idx]
    sku_allowed_row = engine.sku_allowed_np[idx]
    sku_target_row = engine.sku_target_np[idx]

    t_feat0 = time.perf_counter()
    cand_42 = assemble_cand_42(engine.val_cand38_mm[idx], engine.val_extra5_mm[idx])
    cust_11_t = engine.val_meta["tabular_features"][idx : idx + 1].float()
    cust_7_t = prune_cust_7(cust_11_t)
    cust_11_np = cust_11_t.squeeze(0).numpy()
    t_feat_ms = 1000.0 * (time.perf_counter() - t_feat0)

    t_inf0 = time.perf_counter()
    allowed_t = engine.val_meta["allowed_bn"][idx : idx + 1].bool().to(engine.dev)
    with torch.no_grad():
        res = engine.ranker(
            company_text_embedding=engine.val_meta["text_embedding"][idx : idx + 1].float().to(engine.dev),
            city_id=engine.val_meta["city_id"][idx : idx + 1].long().to(engine.dev),
            region_id=engine.val_meta["region_id"][idx : idx + 1].long().to(engine.dev),
            industry_id=engine.val_meta["industry_id"][idx : idx + 1].long().to(engine.dev),
            customer_features=cust_7_t.to(engine.dev),
            sequence=engine.val_meta["sequence"][idx : idx + 1].float().to(engine.dev),
            attention_mask=engine.val_meta["attention_mask"][idx : idx + 1].long().to(engine.dev),
            seq_product_ids=engine.val_meta["seq_product_ids"][idx : idx + 1].long().to(engine.dev),
            seq_purchase_dates=engine.val_meta["seq_purchase_dates"][idx : idx + 1].float().to(engine.dev),
            prediction_day=engine.val_meta["prediction_days"][idx : idx + 1].float().to(engine.dev),
            candidate_features=torch.from_numpy(cand_42).unsqueeze(0).to(engine.dev),
            candidate_mask=allowed_t,
            precomputed_v_bn=engine.v_bn,
            return_aux=False,
        )
        s_main = res["output"].logits.squeeze(0).float().cpu().numpy()
        s_unseen = res["unseen_logits"].squeeze(0).float().cpu().numpy()
    t_inf_ms = 1000.0 * (time.perf_counter() - t_inf0)

    fold_id = int(engine.oof_folds[idx])
    a_cal, b_cal = engine.fold_ab.get(fold_id, engine.full_fit_ab)

    elig_rep = allowed_row & (~unseen_row)
    elig_uns = allowed_row & unseen_row
    s_blend = np.where(
        allowed_row,
        np.where(unseen_row, a_cal * np.where(elig_uns, s_unseen, 0.0) + b_cal, np.where(elig_rep, s_main, 0.0)),
        -np.inf,
    )
    s_main_elig = np.where(allowed_row, s_main, -np.inf)
    s_uns_elig = np.where(elig_uns, s_unseen, -np.inf)

    ord_blend = torch.from_numpy(s_blend.astype(np.float64)).argsort(descending=True, stable=True).numpy()
    ord_main = torch.from_numpy(s_main_elig.astype(np.float64)).argsort(descending=True, stable=True).numpy()
    ord_uns = torch.from_numpy(s_uns_elig.astype(np.float64)).argsort(descending=True, stable=True).numpy()

    valid_all = int(allowed_row.sum())
    valid_uns = int(elig_uns.sum())
    rk_blend = {int(b): p + 1 for p, b in enumerate(ord_blend[:valid_all].tolist())}
    rk_main = {int(b): p + 1 for p, b in enumerate(ord_main[:valid_all].tolist())}
    rk_uns = {int(b): p + 1 for p, b in enumerate(ord_uns[:valid_uns].tolist())}

    # Exact Top-K repeat (by s_main) + Top-K unseen (by s_unseen) + Top-K main-head (for main_only mode)
    rep_scores = np.where(elig_rep, s_main, -np.inf)
    n_rep = min(int(top_k_each), int(elig_rep.sum()))
    n_uns = min(int(top_k_each), valid_uns)
    top_rep = torch.from_numpy(rep_scores.astype(np.float64)).argsort(descending=True, stable=True).numpy()[:n_rep]
    top_uns = ord_uns[:n_uns]
    top_main_only = ord_main[: min(10, valid_all)]
    pool_bns = list(dict.fromkeys([int(x) for x in top_rep] + [int(x) for x in top_uns] + [int(x) for x in top_main_only]))

    t_res0 = time.perf_counter()
    resolved_skus = engine._fast_resolve_skus(np.asarray(pool_bns, dtype=np.int64), cid, date, sku_allowed_row)
    t_res_ms = 1000.0 * (time.perf_counter() - t_res0)

    candidates = []
    for pos, bn_i in enumerate(pool_bns):
        meta = engine.bn_meta[bn_i]
        r_pidx = int(resolved_skus[pos])
        sku_label = "—"
        sku_hit = False
        if r_pidx >= 0:
            r_pid = engine.product_ids[r_pidx]
            sku_hit = bool(sku_target_row[r_pidx])
            if r_pid in engine.cat_indexed.index:
                r_row = engine.cat_indexed.loc[r_pid]
                cont = str(r_row.get("container_type", "")) if pd.notna(r_row.get("container_type", "")) else ""
                pname = str(r_row.get("short_name", r_row.get("product_name", r_pid)))
                if not pname or pname == "nan":
                    pname = r_pid
                sku_label = f"{r_pid} ({cont})" if cont else f"{r_pid}: {pname[:32]}"
            else:
                sku_label = r_pid
        is_new_cand = bool(unseen_row[bn_i])
        candidates.append(
            {
                "b": bn_i,
                "n": meta["base_name"],
                "br": meta["brand"],
                "cat": meta["category"],
                "sku": sku_label,
                "sh": sku_hit,
                "r": not is_new_cand,
                "t": bool(target_row[bn_i]),
                "nt": bool(new_row[bn_i]),
                "sm": round(float(s_main[bn_i]), 4) if np.isfinite(s_main[bn_i]) else -30.0,
                "su": round(float(s_unseen[bn_i]), 4) if np.isfinite(s_unseen[bn_i]) else -30.0,
                "dr": _extract_drivers(cand_42[bn_i], is_new_candidate=is_new_cand),
            }
        )

    gt_items = []
    for bn_i in np.flatnonzero(target_row).tolist():
        meta = engine.bn_meta[bn_i]
        gt_items.append(
            {
                "b": bn_i,
                "n": meta["base_name"],
                "br": meta["brand"],
                "cat": meta["category"],
                "nw": bool(new_row[bn_i]),
                "el": bool(allowed_row[bn_i]),
                "rb": rk_blend.get(bn_i),
                "ru": rk_uns.get(bn_i),
                "rm": rk_main.get(bn_i),
            }
        )
    gt_items.sort(key=lambda x: (x["rb"] is None, x["rb"] or 9999))

    # Prior history (< date, top 12 lines)
    hist_df = engine.histories.get(cid)
    prior_df = (
        hist_df.loc[hist_df["purchase_date"] < date].sort_values("purchase_date", ascending=False)
        if hist_df is not None
        else engine.dev_purchases.iloc[:0]
    )
    prior_dates = sorted(prior_df["purchase_date"].unique())
    median_gap = (
        round(float(np.median(np.diff(np.array(prior_dates, dtype="datetime64[D]")).astype(float))), 1)
        if len(prior_dates) >= 2
        else None
    )
    days_since_last = int((date - pd.Timestamp(prior_dates[-1])).days) if prior_dates else None
    all_pid_to_bn = engine.universe["all_pid_to_bn"]
    prior_bns = {all_pid_to_bn.get(p, "") for p in prior_df["product_id"]} - {""}

    prior_rows = []
    if not prior_df.empty:
        p_view = prior_df.assign(bn=prior_df["product_id"].map(all_pid_to_bn).fillna(prior_df["product_id"]))
        agg_h = (
            p_view.groupby(["purchase_date", "bn"], as_index=False)["quantity"]
            .sum()
            .sort_values(["purchase_date", "quantity"], ascending=[False, False])
        )
        by_bn = engine.universe["bn_to_idx"]
        for _, r in agg_h.head(12).iterrows():
            bn_s = str(r["bn"])
            meta = engine.bn_meta[by_bn[bn_s]] if bn_s in by_bn else {"brand": "", "category": ""}
            prior_rows.append(
                {
                    "dt": str(pd.Timestamp(r["purchase_date"]).date()),
                    "ago": int((date - pd.Timestamp(r["purchase_date"])).days),
                    "n": bn_s,
                    "br": meta["brand"],
                    "cat": meta["category"],
                    "q": round(float(r["quantity"]), 1),
                }
            )

    comp = engine.company_map.get(cid, {})
    desc_str = str(comp.get("description", "")) if pd.notna(comp.get("description", "")) else ""
    ind_str = str(comp.get("industry_branch", "")) if pd.notna(comp.get("industry_branch", "")) else ""

    n_tgt = int(target_row.sum())
    n_new = int(new_row.sum())
    top10_blend = set(ord_blend[: min(10, valid_all)].tolist())
    hits10 = sum(1 for bn_i in np.flatnonzero(target_row).tolist() if bn_i in top10_blend)
    new_hits10 = sum(1 for bn_i in np.flatnonzero(new_row).tolist() if bn_i in top10_blend)

    return {
        "id": int(idx),
        "dt": str(date.date()),
        "a": round(float(a_cal), 4),
        "b": round(float(b_cal), 4),
        "el": valid_all,
        "elu": valid_uns,
        "ms": [round(t_feat_ms, 2), round(t_inf_ms, 2), round(t_res_ms, 2)],
        "fo": is_first,
        "hn": (not is_first) and (n_new > 0),
        "mi": n_tgt >= 2,
        "h10": hits10,
        "ntgt": n_tgt,
        "nh10": new_hits10,
        "nnew": n_new,
        "c": {
            "cid": cid,
            "nm": str(comp.get("company_name", cid)) if pd.notna(comp.get("company_name", "")) else cid,
            "ct": str(comp.get("city", "—")) if pd.notna(comp.get("city", "")) else "—",
            "rg": str(comp.get("region_code", "—")) if pd.notna(comp.get("region_code", "")) else "—",
            "ind": ind_str or "—",
            "desc": (desc_str[:220] + "…") if len(desc_str) > 220 else desc_str,
            "ord": int(len(prior_dates)),
            "bns": int(len(prior_bns)),
            "gap": median_gap,
            "last": days_since_last,
            "prep": round(100.0 * float(cust_11_np[U11_IDX["cust_prepaid_or_cash_ratio"]]), 1),
            "def": round(90.0 * float(cust_11_np[U11_IDX["cust_avg_deferral_days_norm"]]), 1),
            "ovd": round(100.0 * float(cust_11_np[U11_IDX["cust_overdue_unpaid_ratio"]]), 1),
        },
        "hist": prior_rows,
        "cands": candidates,
        "gt": gt_items,
    }


def _select_stratified_cases(
    engine: V2FullLiveDemoEngine,
    *,
    n_returning: int = 90,
    n_has_new: int = 85,
    n_first_obs: int = 50,
    n_multi: int = 55,
    n_showcase_scan: int = 180,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministically sample a diverse, honest stratified pool plus highlight indices."""
    rng = np.random.default_rng(seed)

    def _pick(pool: Sequence[int], k: int) -> List[int]:
        arr = np.asarray(pool, dtype=np.int64)
        if len(arr) <= k:
            return arr.tolist()
        return rng.choice(arr, size=k, replace=False).tolist()

    selected: List[int] = []
    seen = set()
    for pool_key, k in (
        ("has_new", n_has_new),
        ("first_observed", n_first_obs),
        ("multi_item", n_multi),
        ("returning", n_returning),
    ):
        for idx in _pick(engine.filter_indices[pool_key], k):
            if idx not in seen:
                seen.add(idx)
                selected.append(int(idx))

    # Also scan a small deterministic subset of has_new + first_observed to tag clean showcase examples
    scan_pool = [i for i in selected if i in set(engine.filter_indices["has_new"]) or i in set(engine.filter_indices["first_observed"])]
    return selected, scan_pool[:n_showcase_scan]


PORTABLE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>B2B Product Recommender — Interactive Demo (w2_2mem_recency_hl365)</title>
<style>
  :root {
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.10);
    --green: #059669;
    --green-bg: rgba(16, 185, 129, 0.12);
    --green-border: #10b981;
    --amber: #d97706;
    --amber-bg: rgba(245, 158, 11, 0.14);
    --amber-border: #f59e0b;
    --purple: #7c3aed;
    --purple-bg: rgba(124, 58, 237, 0.12);
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 4px 14px rgba(15, 23, 42, 0.04);
    --radius: 14px;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  body.dark {
    --bg: #0b1120;
    --surface: #111827;
    --surface-2: #172033;
    --border: #24324a;
    --text: #e6edf7;
    --muted: #94a3b8;
    --primary: #3b82f6;
    --primary-soft: rgba(59, 130, 246, 0.16);
    --green: #34d399;
    --green-bg: rgba(16, 185, 129, 0.15);
    --amber: #fbbf24;
    --amber-bg: rgba(245, 158, 11, 0.16);
    --purple: #c4b5fd;
    --purple-bg: rgba(139, 92, 246, 0.18);
    --shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    padding: 20px 22px 48px;
    transition: background 0.2s, color 0.2s;
  }
  .wrap { max-width: 1360px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
  .topbar {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px 22px; box-shadow: var(--shadow);
  }
  .brand-title { font-size: 20px; font-weight: 800; letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px; }
  .badge-live {
    font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px;
    background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border);
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .brand-sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .top-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .tabs { display: flex; gap: 6px; background: var(--surface-2); padding: 4px; border-radius: 10px; border: 1px solid var(--border); }
  .tab-btn {
    border: none; background: transparent; color: var(--muted); font-weight: 600; font-size: 13px;
    padding: 7px 14px; border-radius: 8px; cursor: pointer; transition: all 0.15s;
  }
  .tab-btn.active { background: var(--primary); color: #fff; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.28); }
  .icon-btn {
    border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
    font-weight: 700; font-size: 12.5px; padding: 7px 12px; border-radius: 9px; cursor: pointer;
  }
  .icon-btn:hover { border-color: var(--primary); }

  .section { display: none; flex-direction: column; gap: 16px; }
  .section.active { display: flex; }

  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .kpi {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 18px; box-shadow: var(--shadow);
  }
  .kpi .lbl { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 700; }
  .kpi .num { font-size: 28px; font-weight: 800; font-family: var(--mono); margin-top: 6px; letter-spacing: -0.02em; }
  .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .num.green { color: var(--green); }
  .num.amber { color: var(--amber); }
  .num.blue { color: var(--primary); }

  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px 20px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 14px;
  }
  .card-hdr { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; }
  .card-title { font-size: 14px; font-weight: 800; letter-spacing: -0.01em; }
  .two-col { display: grid; grid-template-columns: 1.25fr 1fr; gap: 16px; align-items: start; }
  .explorer-grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; align-items: start; }

  .pipe-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .pipe-box {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .pipe-num {
    width: 24px; height: 24px; border-radius: 50%; background: var(--primary-soft); color: var(--primary);
    font-weight: 800; font-size: 12px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px;
  }
  .pipe-box h4 { font-size: 13.5px; font-weight: 700; margin-bottom: 4px; }
  .pipe-box p { font-size: 12.5px; color: var(--muted); }

  .evo-bars { display: flex; flex-direction: column; gap: 10px; }
  .evo-row { display: grid; grid-template-columns: 190px 1fr 78px 82px; gap: 10px; align-items: center; font-size: 12.5px; }
  .evo-track { height: 12px; background: var(--surface-2); border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
  .evo-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #60a5fa, #2563eb); }
  .evo-row.best .evo-fill { background: linear-gradient(90deg, #10b981, #059669); }
  .evo-row.best { font-weight: 700; }

  .controls-bar {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; box-shadow: var(--shadow);
  }
  select, input[type="text"], button.btn {
    font-family: var(--sans); font-size: 13px; border-radius: 9px;
    border: 1px solid var(--border); background: var(--surface-2); color: var(--text); padding: 8px 12px;
  }
  input[type="text"] { min-width: 220px; flex: 1; }
  button.btn { cursor: pointer; font-weight: 600; }
  button.btn:hover { border-color: var(--primary); }
  button.btn.primary { background: var(--primary); color: #fff; border-color: var(--primary); }
  button.btn.primary:hover { opacity: 0.92; }

  .whatif-bar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
    background: var(--primary-soft); border: 1px solid rgba(37, 99, 235, 0.25);
    border-radius: 11px; padding: 10px 14px;
  }
  .whatif-bar input[type="range"] { width: 220px; accent-color: var(--primary); cursor: pointer; }

  .pill {
    display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 700;
    padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface-2); color: var(--muted);
    white-space: nowrap;
  }
  .pill.hit { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
  .pill.newhit { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber); }
  .pill.cold { background: var(--purple-bg); border-color: var(--purple); color: var(--purple); }
  .pill.ret { background: var(--primary-soft); border-color: var(--primary); color: var(--primary); }
  .pill.newtag { background: var(--amber-bg); border-color: rgba(245, 158, 11, 0.45); color: var(--amber); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); padding: 8px 8px; border-bottom: 1px solid var(--border);
  }
  td { padding: 10px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr.row-hit { background: var(--green-bg); }
  tr.row-hit td:first-child { border-left: 4px solid var(--green-border); }
  tr.row-newhit { background: var(--amber-bg); }
  tr.row-newhit td:first-child { border-left: 4px solid var(--amber-border); }
  .rank-col { font-family: var(--mono); font-weight: 800; width: 36px; }
  .prod-nm { font-weight: 700; }
  .prod-sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .drivers { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
  .drv {
    font-size: 11px; padding: 2px 7px; border-radius: 5px;
    background: var(--surface-2); border: 1px solid var(--border); color: var(--muted); font-family: var(--mono);
  }
  .score-box { width: 86px; }
  .score-bar { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; margin-top: 4px; }
  .score-fill { height: 100%; background: var(--primary); border-radius: 999px; }
  tr.row-hit .score-fill { background: var(--green-border); }
  tr.row-newhit .score-fill { background: var(--amber-border); }

  .gt-list { display: flex; flex-direction: column; gap: 7px; }
  .gt-row {
    display: flex; justify-content: space-between; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px; border: 1px solid var(--border); background: var(--surface-2);
  }
  .gt-row.hit { background: var(--green-bg); border-color: var(--green-border); }
  .gt-row.newhit { background: var(--amber-bg); border-color: var(--amber-border); }

  .mini-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .mstat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 9px 11px; }
  .mstat .k { font-size: 11px; color: var(--muted); }
  .mstat .v { font-size: 14px; font-weight: 800; font-family: var(--mono); margin-top: 2px; }
  .case-kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .hist-scroll { max-height: 290px; overflow-y: auto; }
  .mono { font-family: var(--mono); }
  .footer {
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; font-size: 12px;
    color: var(--muted); padding: 12px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  }
  @media (max-width: 1020px) {
    .kpi-grid, .pipe-grid, .two-col, .explorer-grid, .case-kpi-strip { grid-template-columns: 1fr; }
    .evo-row { grid-template-columns: 140px 1fr 64px 68px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <div class="brand-title">
        <span data-i18n="appTitle">Система AI-рекомендацій B2B</span>
        <span class="badge-live" data-i18n="finalModelBadge">Фінальна модель · v2_full</span>
      </div>
      <div class="brand-sub" id="top-subtitle"></div>
    </div>
    <div class="top-actions">
      <div class="tabs">
        <button class="tab-btn active" data-tab="explorer" data-i18n="tabExplorer">Інтерактивні прогнози</button>
        <button class="tab-btn" data-tab="overview" data-i18n="tabOverview">Огляд і результати</button>
        <button class="tab-btn" data-tab="package" data-i18n="tabPackage">Модель у пакеті</button>
      </div>
      <button class="icon-btn" id="btn-lang">UA / EN</button>
      <button class="icon-btn" id="btn-theme">🌙</button>
    </div>
  </div>

  <!-- TAB 1: INTERACTIVE EXPLORER -->
  <div class="section active" id="sec-explorer">
    <div class="controls-bar">
      <select id="sel-cohort"></select>
      <select id="sel-mode"></select>
      <input type="text" id="inp-search" placeholder="Пошук компанії, міста, галузі або товару..." />
      <button class="btn" id="btn-prev">←</button>
      <button class="btn" id="btn-next">→</button>
      <button class="btn primary" id="btn-random" data-i18n="btnRandom">Випадковий кейс</button>
      <span class="pill mono" id="pool-counter">—</span>
    </div>

    <div class="whatif-bar" id="whatif-wrap">
      <div>
        <strong data-i18n="whatIfTitle">Баланс «Повтори ↔ Нові товари (Cross-Sell)»</strong>
        <span style="font-size:12px;color:var(--muted);margin-left:6px;" id="whatif-desc"></span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:12px;color:var(--muted);" data-i18n="sliderLeft">Більше повторів</span>
        <input type="range" id="rng-boost" min="-2.5" max="2.5" step="0.1" value="0" />
        <span style="font-size:12px;color:var(--muted);" data-i18n="sliderRight">Більше нових</span>
        <span class="pill mono" id="boost-val">0.0</span>
        <button class="btn" id="btn-reset-boost" style="padding:4px 9px;font-size:12px;" data-i18n="btnReset">Оптимум</button>
      </div>
    </div>

    <div class="case-kpi-strip">
      <div class="kpi">
        <div class="lbl" id="lbl-c-r10">Точність кейсу (Recall@10)</div>
        <div class="num green" id="c-r10">—</div>
        <div class="sub" id="c-hits">—</div>
      </div>
      <div class="kpi">
        <div class="lbl" id="lbl-c-ndcg">Якість ранжування (NDCG@10)</div>
        <div class="num blue" id="c-ndcg">—</div>
        <div class="sub" data-i18n="ndcgHint">Враховує позицію в Топ-10</div>
      </div>
      <div class="kpi">
        <div class="lbl" data-i18n="lblNewHits">Знайдено нових товарів у Топ-10</div>
        <div class="num amber" id="c-new">—</div>
        <div class="sub" data-i18n="newHint">Клієнт ніколи не купував раніше</div>
      </div>
      <div class="kpi">
        <div class="lbl" data-i18n="lblSpeed">Швидкість прогнозу (CPU)</div>
        <div class="num" id="c-ms" style="font-size:22px;padding-top:4px;">—</div>
        <div class="sub" id="c-date-sub">—</div>
      </div>
    </div>

    <div class="explorer-grid">
      <div class="card">
        <div class="card-hdr">
          <div class="card-title" data-i18n="top10Title">Топ-10 рекомендацій нейромережі</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <span class="pill hit" data-i18n="legendHit">✓ Вгадано (Повтор)</span>
            <span class="pill newhit" data-i18n="legendNewHit">★ Вгадано (Новий товар!)</span>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th data-i18n="thProduct">Рекомендований продукт і причини</th>
              <th data-i18n="thSku">Фасування (SKU)</th>
              <th data-i18n="thConf">Оцінка</th>
              <th data-i18n="thStatus">Факт</th>
            </tr>
          </thead>
          <tbody id="pred-tbody"></tbody>
        </table>

        <div style="margin-top:6px;">
          <div class="card-hdr" style="margin-bottom:8px;">
            <div class="card-title"><span data-i18n="gtTitle">Що клієнт реально купив</span> <span class="mono" id="gt-date"></span></div>
            <span class="pill" id="gt-badge">—</span>
          </div>
          <div class="gt-list" id="gt-list"></div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="card">
          <div class="card-hdr">
            <div class="card-title" data-i18n="custTitle">Профіль клієнта (до дати замовлення)</div>
            <span class="pill" id="cust-cohort">—</span>
          </div>
          <div>
            <div style="font-size:16px;font-weight:800;" id="cust-name">—</div>
            <div style="font-size:12.5px;color:var(--muted);margin-top:2px;" id="cust-loc">—</div>
          </div>
          <div id="cust-desc" style="font-size:12.5px;color:var(--muted);background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;">—</div>
          <div class="mini-stats">
            <div class="mstat"><div class="k" data-i18n="stOrders">Минулих замовлень</div><div class="v" id="st-ord">—</div></div>
            <div class="mstat"><div class="k" data-i18n="stBns">Унікальних товарів</div><div class="v" id="st-bns">—</div></div>
            <div class="mstat"><div class="k" data-i18n="stGap">Періодичність</div><div class="v" id="st-gap">—</div></div>
            <div class="mstat"><div class="k" data-i18n="stLast">Днів з останнього</div><div class="v" id="st-last">—</div></div>
            <div class="mstat"><div class="k" data-i18n="stPrep">Передплата / готівка</div><div class="v" id="st-prep">—</div></div>
            <div class="mstat"><div class="k" data-i18n="stDef">Відстрочка / простр.</div><div class="v" id="st-def">—</div></div>
          </div>
        </div>

        <div class="card">
          <div class="card-hdr">
            <div class="card-title" data-i18n="histTitle">Останні покупки клієнта (&lt; дати прогнозу)</div>
            <span class="pill" id="hist-badge">—</span>
          </div>
          <div class="hist-scroll">
            <table>
              <thead><tr><th data-i18n="thDate">Дата</th><th data-i18n="thHistProd">Продукт</th><th data-i18n="thQty">К-сть</th></tr></thead>
              <tbody id="hist-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: EXECUTIVE OVERVIEW -->
  <div class="section" id="sec-overview">
    <div class="kpi-grid">
      <div class="kpi">
        <div class="lbl">Recall@10</div>
        <div class="num green">82.26%</div>
        <div class="sub" data-i18n="kpiR10Sub">Вгадано товарів у Топ-10 (16 623 замовлення)</div>
      </div>
      <div class="kpi">
        <div class="lbl">NDCG@10</div>
        <div class="num blue">65.48%</div>
        <div class="sub" data-i18n="kpiNdcgSub">Точність порядку у Топ-10</div>
      </div>
      <div class="kpi">
        <div class="lbl" data-i18n="kpiNewLbl">Нові товари у Топ-10</div>
        <div class="num amber">27.91%</div>
        <div class="sub" data-i18n="kpiNewSub">1 255 із 4 496 нових покупок клієнтів</div>
      </div>
      <div class="kpi">
        <div class="lbl" data-i18n="kpiPureLbl">Чистий Cross-Sell@10</div>
        <div class="num amber">52.27%</div>
        <div class="sub" data-i18n="kpiPureSub">2 350 із 4 496 при пошуку лише новинок</div>
      </div>
      <div class="kpi">
        <div class="lbl" data-i18n="kpiSpeedLbl">Швидкість (1 запит)</div>
        <div class="num">2.9 ms</div>
        <div class="sub" data-i18n="kpiSpeedSub">По всіх 1 141 формуляціях на CPU</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title" data-i18n="howTitle">Як працює система (без витоку майбутніх даних)</div>
      <div class="pipe-grid">
        <div class="pipe-box">
          <div class="pipe-num">1</div>
          <h4 data-i18n="p1t">Уніфікація каталогу</h4>
          <p data-i18n="p1d">1 660 SKU об'єднано у 1 141 канонічний продукт (без дублювання каністр 4л / 20л / 209л у Топ-10).</p>
        </div>
        <div class="pipe-box">
          <div class="pipe-num">2</div>
          <h4 data-i18n="p2t">42 сигнали на дату замовлення</h4>
          <p data-i18n="p2d">Історія повторів, колаборативна фільтрація, статистика менеджера й регіону, галузь, в'язкість, фінанси — строго до дати.</p>
        </div>
        <div class="pipe-box">
          <div class="pipe-num">3</div>
          <h4 data-i18n="p3t">2-шаровий Transformer + 2 голови</h4>
          <p data-i18n="p3d">Окрема голова для регулярних повторів (Main Head) та окрема голова для відкриття новинок (Option A Head), навчені з акцентом на свіжі 2025–2026 продажі.</p>
        </div>
        <div class="pipe-box">
          <div class="pipe-num">4</div>
          <h4 data-i18n="p4t">Підбір фасування (SKU)</h4>
          <p data-i18n="p4d">Для кожного з Топ-10 продуктів система автоматично підбирає найімовірнішу тару клієнта (бочка, каністра тощо).</p>
        </div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-hdr">
          <div class="card-title" data-i18n="evoTitle">Еволюція якості моделі (валідація)</div>
          <span class="pill hit">Recall@10 &amp; New@10</span>
        </div>
        <div class="evo-bars" id="evo-bars"></div>
      </div>

      <div class="card">
        <div class="card-title" data-i18n="cohortTitle">Результати по сегментах (16 623 замовлення)</div>
        <table>
          <thead>
            <tr>
              <th data-i18n="thSeg">Сегмент</th>
              <th>Recall@10</th>
              <th>NDCG@10</th>
              <th>New@10</th>
            </tr>
          </thead>
          <tbody>
            <tr><td data-i18n="segAll">Усі замовлення (16 623)</td><td class="mono"><strong>82.26%</strong></td><td class="mono"><strong>65.48%</strong></td><td class="mono"><strong>27.91%</strong> (1 255)</td></tr>
            <tr><td data-i18n="segRet">Постійні клієнти (16 215)</td><td class="mono">83.02%</td><td class="mono">66.19%</td><td class="mono">20.96% (761)</td></tr>
            <tr><td data-i18n="segCold">Нові клієнти / 1-ше замовлення (408)</td><td class="mono">52.00%</td><td class="mono">36.93%</td><td class="mono">57.04% (494)</td></tr>
            <tr><td data-i18n="segPure">Режим лише новинок (Option A, Ep 9)</td><td class="mono">—</td><td class="mono">32.63%</td><td class="mono"><strong>52.27%</strong> (2 350)</td></tr>
            <tr><td data-i18n="segSku">Точне фасування (Resolved SKU)</td><td class="mono">55.24%</td><td class="mono">45.51%</td><td class="mono">663 SKU</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB 3: MODEL PACKAGE INFO -->
  <div class="section" id="sec-package">
    <div class="card">
      <div class="card-title" data-i18n="pkgTitle">Вбудована фінальна модель у цьому ZIP-пакеті</div>
      <p style="color:var(--muted);" data-i18n="pkgDesc">Разом із цим демо у папці <code>model/</code> знаходиться сама навчена нейромережа PyTorch та повна специфікація з контрольними сумами SHA-256.</p>
      <div class="mini-stats" id="pkg-grid" style="grid-template-columns:repeat(3,1fr);"></div>
      <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px 14px;font-size:12.5px;">
        <div><strong>Reference ID:</strong> <span class="mono" id="pkg-refid"></span></div>
        <div style="margin-top:4px;"><strong>Model Checkpoint:</strong> <span class="mono">model/best_discovery.pt</span> (<span id="pkg-size"></span>)</div>
        <div style="margin-top:4px;"><strong>SHA-256:</strong> <span class="mono" id="pkg-sha" style="word-break:break-all;"></span></div>
        <div style="margin-top:4px;"><strong>Train / Val Split:</strong> <span class="mono">Train &lt; 2026-02-01 (137,765 baskets) · Val 2026-02-01..2026-06-30 (16,623 baskets) · Held-out Test &gt;= 2026-07-01 untouched</span></div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div id="footer-left"></div>
    <div data-i18n="footerRight">Усі прогнози згенеровано реальною нейромережею w2_2mem_recency_hl365 без доступу до майбутніх даних.</div>
  </div>
</div>

<script>
const DEMO_PAYLOAD = __DEMO_PAYLOAD_JSON__;

const I18N = {
  uk: {
    appTitle: "Система AI-рекомендацій B2B",
    finalModelBadge: "Фінальна модель · v2_full",
    tabExplorer: "Інтерактивні прогнози",
    tabOverview: "Огляд і результати",
    tabPackage: "Модель у пакеті",
    btnRandom: "Випадковий кейс",
    whatIfTitle: "Баланс «Повтори ↔ Нові товари (Cross-Sell)»",
    sliderLeft: "Більше повторів",
    sliderRight: "Більше нових",
    btnReset: "Оптимум",
    ndcgHint: "Враховує позицію в Топ-10",
    lblNewHits: "Знайдено нових товарів у Топ-10",
    newHint: "Клієнт ніколи не купував раніше",
    lblSpeed: "Швидкість прогнозу (CPU)",
    top10Title: "Топ-10 рекомендацій нейромережі",
    legendHit: "✓ Вгадано (Повтор)",
    legendNewHit: "★ Вгадано (Новий товар!)",
    thProduct: "Рекомендований продукт і причини",
    thSku: "Фасування (SKU)",
    thConf: "Оцінка",
    thStatus: "Факт",
    gtTitle: "Що клієнт реально купив",
    custTitle: "Профіль клієнта (до дати замовлення)",
    stOrders: "Минулих замовлень",
    stBns: "Унікальних товарів",
    stGap: "Періодичність",
    stLast: "Днів з останнього",
    stPrep: "Передплата / готівка",
    stDef: "Відстрочка / простр.",
    histTitle: "Останні покупки клієнта (< дати прогнозу)",
    thDate: "Дата",
    thHistProd: "Продукт",
    thQty: "К-сть",
    kpiR10Sub: "Вгадано товарів у Топ-10 (16 623 замовлення)",
    kpiNdcgSub: "Точність порядку у Топ-10",
    kpiNewLbl: "Нові товари у Топ-10",
    kpiNewSub: "1 255 із 4 496 нових покупок клієнтів",
    kpiPureLbl: "Чистий Cross-Sell@10",
    kpiPureSub: "2 350 із 4 496 при пошуку лише новинок",
    kpiSpeedLbl: "Швидкість (1 запит)",
    kpiSpeedSub: "По всіх 1 141 формуляціях на CPU",
    howTitle: "Як працює система (без витоку майбутніх даних)",
    p1t: "Уніфікація каталогу",
    p1d: "1 660 SKU об'єднано у 1 141 канонічний продукт (без дублювання каністр 4л / 20л / 209л у Топ-10).",
    p2t: "42 сигнали на дату замовлення",
    p2d: "Історія повторів, колаборативна фільтрація, статистика менеджера й регіону, галузь, в'язкість, фінанси — строго до дати.",
    p3t: "2-шаровий Transformer + 2 голови",
    p3d: "Окрема голова для регулярних повторів (Main Head) та окрема голова для відкриття новинок (Option A Head), навчені з акцентом на свіжі 2025–2026 продажі.",
    p4t: "Підбір фасування (SKU)",
    p4d: "Для кожного з Топ-10 продуктів система автоматично підбирає найімовірнішу тару клієнта (бочка, каністра тощо).",
    evoTitle: "Еволюція якості моделі (валідація)",
    cohortTitle: "Результати по сегментах (16 623 замовлення)",
    thSeg: "Сегмент",
    segAll: "Усі замовлення (16 623)",
    segRet: "Постійні клієнти (16 215)",
    segCold: "Нові клієнти / 1-ше замовлення (408)",
    segPure: "Режим лише новинок (Option A, Ep 9)",
    segSku: "Точне фасування (Resolved SKU)",
    pkgTitle: "Вбудована фінальна модель у цьому ZIP-пакеті",
    pkgDesc: "Разом із цим демо у папці model/ знаходиться сама навчена нейромережа PyTorch та повна специфікація з контрольними сумами SHA-256.",
    footerRight: "Усі прогнози згенеровано реальною нейромережею w2_2mem_recency_hl365 без доступу до майбутніх даних.",
    searchPh: "Пошук компанії, міста, галузі або товару...",
    cohorts: {
      showcase: "Найкращі приклади (Showcase)",
      all: "Усі приклади демо",
      has_new: "Постійні клієнти з новинками",
      first_observed: "Нові клієнти (1-ше замовлення)",
      multi_item: "Великі кошики (≥2 товари)",
      returning: "Постійні клієнти",
    },
    modes: {
      blended: "Змішаний (Повтори + Новинки)",
      pure_unseen: "Лише нові товари (Cross-Sell)",
      main_only: "Лише регулярні повтори",
    },
    hitRep: "✓ Вгадано · Повтор",
    hitNew: "★ Вгадано · Новий!",
    tagRep: "Повтор",
    tagNew: "Новий товар",
    tagCold: "Новий клієнт (0 покупок)",
    tagReturning: "Постійний клієнт",
    r10Lbl: "Точність кейсу (Recall@10)",
    r10PureLbl: "Точність новинок (Pure Unseen@10)",
    ndcgLbl: "Якість порядку (NDCG@10)",
    ndcgPureLbl: "Якість порядку новинок",
    ofBought: (h, t) => `${h} із ${t} куплених продуктів у Топ-10`,
    ofNewBought: (h, t) => `${h} із ${t} нових продуктів у Топ-10`,
    rankHit: r => `Місце #${r} ✓`,
    rankMiss: r => `Місце #${r}`,
    maskedRep: "Приховано (Повтор)",
    noHist: "Перше замовлення клієнта — історії покупок ще немає. Прогноз базується на галузі, регіоні, профілі компанії та попиті.",
    whatIfOpt: (a, b) => `Оптимальна OOF-калібровка: a=${a}, b=${b}`,
    whatIfAdj: (a, b, d) => `Зміщено на ${d > 0 ? '+' : ''}${d.toFixed(1)} (a=${a}, b=${b})`,
    drv: {
      repeat: x => `Купував ${x.n}× · ${x.d}д тому`,
      due: x => `Час повтору ${x.v}×`,
      cf_rec: x => `Схожі клієнти ${x.v}`,
      cf_cos: x => `CF схожість ${x.v}`,
      seq_lift: x => `Зв'язка +${x.v}`,
      mgr30: x => `Менеджер 30д ${x.v}%`,
      mgr_ind: x => `Мен×Галузь ${x.v}%`,
      mgr90: x => `Менеджер 90д ${x.v}%`,
      reg90: x => `Регіон 90д ${x.v}%`,
      ind: x => `Галузь ${x.v}%`,
      equip: () => `Збіг техніки`,
      visc: x => `В'язкість ${x.v}%`,
    },
  },
  en: {
    appTitle: "B2B AI Product Recommender",
    finalModelBadge: "Final Model · v2_full",
    tabExplorer: "Interactive Predictions",
    tabOverview: "Overview & Metrics",
    tabPackage: "Bundled Model",
    btnRandom: "Random Case",
    whatIfTitle: "Repeat ↔ New Product (Cross-Sell) Balance",
    sliderLeft: "More Repeats",
    sliderRight: "More New",
    btnReset: "Optimal",
    ndcgHint: "Position-weighted Top-10 quality",
    lblNewHits: "New Discoveries in Top-10",
    newHint: "Never purchased by customer before",
    lblSpeed: "Inference Latency (CPU)",
    top10Title: "Top-10 Neural Network Predictions",
    legendHit: "✓ Repeat Hit",
    legendNewHit: "★ New Discovery Hit",
    thProduct: "Recommended Formulation & Drivers",
    thSku: "Resolved Pack SKU",
    thConf: "Score",
    thStatus: "Outcome",
    gtTitle: "Actual Ground-Truth Basket on",
    custTitle: "Customer Profile (Strictly Prior)",
    stOrders: "Prior Orders",
    stBns: "Distinct Products",
    stGap: "Median Cadence",
    stLast: "Days Since Last",
    stPrep: "Prepaid / Cash",
    stDef: "Deferral / Overdue",
    histTitle: "Recent Prior Purchases (< Date)",
    thDate: "Date",
    thHistProd: "Product",
    thQty: "Qty",
    kpiR10Sub: "Formulations caught in Top-10 (16,623 baskets)",
    kpiNdcgSub: "Top-10 ranking order quality",
    kpiNewLbl: "New Products@10",
    kpiNewSub: "1,255 / 4,496 truly new adoptions",
    kpiPureLbl: "Pure Cross-Sell@10",
    kpiPureSub: "2,350 / 4,496 when ranking unseen only",
    kpiSpeedLbl: "Latency (1 Request)",
    kpiSpeedSub: "All 1,141 formulations scored on CPU",
    howTitle: "How the Pipeline Works (Zero Future Leakage)",
    p1t: "Catalogue Unification",
    p1d: "1,660 SKUs unified into 1,141 canonical formulations so pack variants (4L / 20L / 209L) never crowd Top-10.",
    p2t: "42 Strictly Causal Signals",
    p2d: "Repeat cadence, collaborative filtering, manager & region shares, industry, viscosity, financials — all < order date.",
    p3t: "2-Layer Transformer + Dual Heads",
    p3d: "Dedicated Repeat Head (Main) + Cross-Sell Head (Option A), trained with 365-day recency weighting for the 2025–2026 regime.",
    p4t: "Pack-Size SKU Resolver",
    p4d: "Automatically resolves each Top-10 formulation to the customer's most likely pack size (drum, pail, jug).",
    evoTitle: "Model Quality Progression (Validation)",
    cohortTitle: "Performance by Segment (16,623 Baskets)",
    thSeg: "Segment",
    segAll: "All Validation Baskets (16,623)",
    segRet: "Returning Customers (16,215)",
    segCold: "First-Observed / Cold-Start (408)",
    segPure: "Pure Unseen Cross-Sell (Option A, Ep 9)",
    segSku: "Resolved Exact Pack SKU",
    pkgTitle: "Final Model Bundled in This ZIP Packet",
    pkgDesc: "The model/ folder inside this ZIP contains the trained PyTorch checkpoint and full SHA-256 verified reference metadata.",
    footerRight: "All predictions were generated by the real w2_2mem_recency_hl365 neural network with zero future leakage.",
    searchPh: "Search company, city, industry, or product...",
    cohorts: {
      showcase: "Curated Highlights (Showcase)",
      all: "All Demo Cases",
      has_new: "Returning w/ True New Product",
      first_observed: "Cold-Start (First Order)",
      multi_item: "Multi-Product Baskets (≥2)",
      returning: "Returning Customers",
    },
    modes: {
      blended: "Blended (Repeats + Cross-Sell)",
      pure_unseen: "Pure New Products (Cross-Sell)",
      main_only: "Repeat Replenishment Only",
    },
    hitRep: "✓ HIT · Repeat",
    hitNew: "★ HIT · NEW!",
    tagRep: "Repeat",
    tagNew: "New Candidate",
    tagCold: "Cold-Start (0 prior)",
    tagReturning: "Returning Customer",
    r10Lbl: "Case Recall@10",
    r10PureLbl: "Pure Unseen Recall@10",
    ndcgLbl: "Case NDCG@10",
    ndcgPureLbl: "Pure Unseen NDCG@10",
    ofBought: (h, t) => `${h} of ${t} bought formulations in Top-10`,
    ofNewBought: (h, t) => `${h} of ${t} new formulations in Top-10`,
    rankHit: r => `Rank #${r} ✓`,
    rankMiss: r => `Rank #${r}`,
    maskedRep: "Masked (Repeat)",
    noHist: "First observed order — zero prior history. Predictions use company profile, industry, region & rolling demand.",
    whatIfOpt: (a, b) => `Optimal OOF calibration: a=${a}, b=${b}`,
    whatIfAdj: (a, b, d) => `Adjusted by ${d > 0 ? '+' : ''}${d.toFixed(1)} (a=${a}, b=${b})`,
    drv: {
      repeat: x => `Repeat ${x.n}× · ${x.d}d ago`,
      due: x => `Due ${x.v}×`,
      cf_rec: x => `CF rec ${x.v}`,
      cf_cos: x => `CF cos ${x.v}`,
      seq_lift: x => `Seq lift +${x.v}`,
      mgr30: x => `Mgr 30d ${x.v}%`,
      mgr_ind: x => `Mgr×Ind ${x.v}%`,
      mgr90: x => `Mgr 90d ${x.v}%`,
      reg90: x => `Reg 90d ${x.v}%`,
      ind: x => `Industry ${x.v}%`,
      equip: () => `Equipment match`,
      visc: x => `Viscosity ${x.v}%`,
    },
  }
};

let lang = 'uk';
let filteredCases = [];
let curPos = 0;

const esc = s => String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));

function applyLang() {
  const t = I18N[lang];
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (t[k]) el.textContent = t[k];
  });
  document.getElementById('inp-search').placeholder = t.searchPh;
  const meta = DEMO_PAYLOAD.meta;
  document.getElementById('top-subtitle').textContent =
    `${meta.arm} (Ep ${meta.epoch} · 2× Transformer · HL=365d · ${(meta.per_request_params/1e6).toFixed(2)}M per-req) · ` +
    `${meta.num_formulations.toLocaleString()} formulations · ${DEMO_PAYLOAD.cases.length} real validation cases`;
  document.getElementById('footer-left').textContent =
    `Ref: ${meta.reference_id} · Validation ${meta.date_range[0]} → ${meta.date_range[1]} (${meta.validation_baskets.toLocaleString()} baskets)`;

  const prevCohort = document.getElementById('sel-cohort').value || 'showcase';
  const prevMode = document.getElementById('sel-mode').value || 'blended';
  const counts = {
    showcase: DEMO_PAYLOAD.cases.filter(x => x.sc).length,
    all: DEMO_PAYLOAD.cases.length,
    has_new: DEMO_PAYLOAD.cases.filter(x => x.hn).length,
    first_observed: DEMO_PAYLOAD.cases.filter(x => x.fo).length,
    multi_item: DEMO_PAYLOAD.cases.filter(x => x.mi).length,
    returning: DEMO_PAYLOAD.cases.filter(x => !x.fo).length,
  };
  document.getElementById('sel-cohort').innerHTML = Object.entries(t.cohorts)
    .map(([k, lbl]) => `<option value="${k}">${esc(lbl)} (${counts[k]})</option>`).join('');
  document.getElementById('sel-cohort').value = prevCohort;

  document.getElementById('sel-mode').innerHTML = Object.entries(t.modes)
    .map(([k, lbl]) => `<option value="${k}">${esc(lbl)}</option>`).join('');
  document.getElementById('sel-mode').value = prevMode;

  renderEvolution();
  renderPackageInfo();
  renderCurrentCase();
}

function renderEvolution() {
  const rows = [
    { name: "1. Initial SKU Baseline (v1)", r10: 76.15, new10: "11.25%" },
    { name: "2. v1 Shared-UV Formulations", r10: 80.89, new10: "21.62%" },
    { name: "3. CatBoost 42-feat (v2_full)", r10: 81.61, new10: "25.33%" },
    { name: "4. v2_full Wave 1 E2E (arm4)", r10: 81.84, new10: "25.98%" },
    { name: "5. v2_full Wave 2C (Option A)", r10: 82.12, new10: "27.11%" },
    { name: "6. v2_full 2× Memory Blocks", r10: 82.18, new10: "27.31%" },
    { name: "7. Final: 2× Mem + HL=365d ⭐", r10: 82.26, new10: "27.91%", best: true },
  ];
  document.getElementById('evo-bars').innerHTML = rows.map(r => {
    const pct = Math.max(8, Math.min(100, ((r.r10 - 74) / (83 - 74)) * 100));
    return `<div class="evo-row ${r.best ? 'best' : ''}">
      <div>${esc(r.name)}</div>
      <div class="evo-track"><div class="evo-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="mono" style="text-align:right;">${r.r10.toFixed(2)}%</div>
      <div class="mono" style="text-align:right;color:var(--amber);">${r.new10}</div>
    </div>`;
  }).join('');
}

function renderPackageInfo() {
  const m = DEMO_PAYLOAD.meta;
  document.getElementById('pkg-refid').textContent = m.reference_id;
  document.getElementById('pkg-size').textContent = `${(m.checkpoint_bytes / (1024*1024)).toFixed(2)} MB`;
  document.getElementById('pkg-sha').textContent = m.checkpoint_sha256;
  const items = [
    ["Architecture", "Wave2BE2ERanker (2× mem)"],
    ["Train / Per-Req Params", `${(m.train_params/1e6).toFixed(2)}M / ${(m.per_request_params/1e6).toFixed(2)}M`],
    ["Features", "cand_42 + CUST_7 + SEQ_5"],
    ["Recency Weighting", "HL=365d (min=0.1)"],
    ["Validation R@10 / NDCG", "82.26% / 65.48%"],
    ["New@10 / PureU@10", "27.91% / 52.27%"],
  ];
  document.getElementById('pkg-grid').innerHTML = items.map(([k, v]) =>
    `<div class="mstat"><div class="k">${esc(k)}</div><div class="v" style="font-size:13px;">${esc(v)}</div></div>`
  ).join('');
}

function rebuildFilter() {
  const coh = document.getElementById('sel-cohort').value || 'showcase';
  const q = (document.getElementById('inp-search').value || '').trim().toLowerCase();
  filteredCases = DEMO_PAYLOAD.cases.filter(x => {
    if (coh === 'showcase' && !x.sc) return false;
    if (coh === 'has_new' && !x.hn) return false;
    if (coh === 'first_observed' && !x.fo) return false;
    if (coh === 'multi_item' && !x.mi) return false;
    if (coh === 'returning' && x.fo) return false;
    if (q) {
      const hay = `${x.c.nm} ${x.c.ct} ${x.c.ind} ${x.c.cid} ${x.gt.map(g=>g.n).join(' ')} ${x.cands.map(p=>p.n).join(' ')}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  curPos = 0;
  renderCurrentCase();
}

function rankCaseClient(d, mode, boostDelta) {
  const bEff = d.b + boostDelta;
  let pool = d.cands.filter(p => mode !== 'pure_unseen' || !p.r);
  const scored = pool.map(p => {
    let sc;
    if (mode === 'pure_unseen') sc = p.su;
    else if (mode === 'main_only') sc = p.sm;
    else sc = p.r ? p.sm : (d.a * p.su + bEff);
    return { ...p, sc };
  });
  scored.sort((x, y) => (y.sc - x.sc) || (x.b - y.b));
  return { top10: scored.slice(0, 10), bEff };
}

function renderCurrentCase() {
  const t = I18N[lang];
  const cntEl = document.getElementById('pool-counter');
  if (!filteredCases.length) {
    cntEl.textContent = '0 / 0';
    document.getElementById('pred-tbody').innerHTML = `<tr><td colspan="5" style="color:var(--muted);padding:18px;">—</td></tr>`;
    return;
  }
  curPos = ((curPos % filteredCases.length) + filteredCases.length) % filteredCases.length;
  const d = filteredCases[curPos];
  cntEl.textContent = `${curPos + 1} / ${filteredCases.length} (Case #${d.id})`;

  const mode = document.getElementById('sel-mode').value || 'blended';
  const boostDelta = mode === 'blended' ? parseFloat(document.getElementById('rng-boost').value || '0') : 0;
  document.getElementById('whatif-wrap').style.display = mode === 'blended' ? '' : 'none';
  document.getElementById('boost-val').textContent = (boostDelta >= 0 ? '+' : '') + boostDelta.toFixed(1);

  const { top10, bEff } = rankCaseClient(d, mode, boostDelta);
  document.getElementById('whatif-desc').textContent =
    Math.abs(boostDelta) < 1e-6 ? t.whatIfOpt(d.a.toFixed(2), d.b.toFixed(2)) : t.whatIfAdj(d.a.toFixed(2), bEff.toFixed(2), boostDelta);

  const isPure = mode === 'pure_unseen';
  const denom = isPure ? d.nnew : d.ntgt;
  const rel = top10.map(p => (isPure ? p.nt : p.t) ? 1 : 0);
  const hits = rel.reduce((a, b) => a + b, 0);
  const newHits = top10.filter(p => p.nt).length;
  let dcg = 0, idcg = 0;
  for (let i = 0; i < rel.length; i++) dcg += rel[i] / Math.log2(i + 2);
  for (let i = 0; i < Math.min(denom, 10); i++) idcg += 1 / Math.log2(i + 2);
  const r10 = denom > 0 ? hits / denom : null;
  const ndcg = (denom > 0 && idcg > 0) ? dcg / idcg : null;

  document.getElementById('lbl-c-r10').textContent = isPure ? t.r10PureLbl : t.r10Lbl;
  document.getElementById('lbl-c-ndcg').textContent = isPure ? t.ndcgPureLbl : t.ndcgLbl;
  document.getElementById('c-r10').textContent = r10 !== null ? `${(100*r10).toFixed(1)}%` : 'N/A';
  document.getElementById('c-hits').textContent = isPure ? t.ofNewBought(hits, denom) : t.ofBought(hits, denom);
  document.getElementById('c-ndcg').textContent = ndcg !== null ? `${(100*ndcg).toFixed(1)}%` : 'N/A';
  document.getElementById('c-new').textContent = `${newHits} / ${d.nnew}`;
  document.getElementById('c-ms').textContent = `${d.ms[1].toFixed(2)} ms`;
  document.getElementById('c-date-sub').textContent = `${d.dt} · ${d.el.toLocaleString()} eligible`;

  const top10Rank = {};
  top10.forEach((p, idx) => { top10Rank[p.b] = idx + 1; });

  document.getElementById('pred-tbody').innerHTML = top10.map((p, idx) => {
    const isHit = isPure ? p.nt : p.t;
    const cls = p.nt ? 'row-newhit' : (isHit ? 'row-hit' : '');
    const badge = p.nt ? `<span class="pill newhit">${t.hitNew}</span>`
      : (isHit ? `<span class="pill hit">${t.hitRep}</span>`
      : (p.r ? `<span class="pill">${t.tagRep}</span>` : `<span class="pill newtag">${t.tagNew}</span>`));
    const prob = sigmoid(p.sc);
    const drvHtml = p.dr.map(x => `<span class="drv">${esc((t.drv[x.t] || (()=>x.t))(x))}</span>`).join('');
    const skuMark = p.sh ? ' <span style="color:var(--green);font-weight:800;">✓</span>' : '';
    return `<tr class="${cls}">
      <td class="rank-col">#${idx + 1}</td>
      <td>
        <div class="prod-nm">${esc(p.n)}</div>
        <div class="prod-sub">${esc(p.br)} · ${esc(p.cat)}</div>
        <div class="drivers">${drvHtml}</div>
      </td>
      <td class="mono" style="font-size:12px;">${esc(p.sku)}${skuMark}</td>
      <td class="score-box">
        <div class="mono" style="font-size:12px;font-weight:700;">${(100*prob).toFixed(1)}%</div>
        <div class="score-bar"><div class="score-fill" style="width:${Math.min(100, Math.round(100*prob))}%"></div></div>
      </td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  document.getElementById('gt-date').textContent = d.dt;
  document.getElementById('gt-badge').textContent = `${d.ntgt} (${d.nnew} new)`;
  document.getElementById('gt-list').innerHTML = d.gt.map(g => {
    let rk = top10Rank[g.b] || null;
    if (rk === null && Math.abs(boostDelta) < 1e-6) {
      rk = isPure ? g.ru : (mode === 'main_only' ? g.rm : g.rb);
    }
    const inTop10 = top10Rank[g.b] !== undefined;
    const cls = inTop10 ? (g.nw ? 'newhit' : 'hit') : '';
    const rkTxt = rk !== null ? (inTop10 ? t.rankHit(rk) : t.rankMiss(rk)) : (isPure && !g.nw ? t.maskedRep : '> #10');
    const tag = g.nw ? `<span class="pill newtag">${t.tagNew}</span>` : `<span class="pill">${t.tagRep}</span>`;
    return `<div class="gt-row ${cls}">
      <div><div class="prod-nm">${esc(g.n)}</div><div class="prod-sub">${esc(g.br)} · ${esc(g.cat)}</div></div>
      <div style="display:flex;gap:6px;align-items:center;">${tag}<span class="pill mono ${cls}">${esc(rkTxt)}</span></div>
    </div>`;
  }).join('');

  const c = d.c;
  document.getElementById('cust-cohort').className = `pill ${d.fo ? 'cold' : 'ret'}`;
  document.getElementById('cust-cohort').textContent = d.fo ? t.tagCold : t.tagReturning;
  document.getElementById('cust-name').textContent = c.nm;
  document.getElementById('cust-loc').textContent = `${c.ct} (${c.rg}) · ${c.ind} · ID: ${c.cid}`;
  document.getElementById('cust-desc').textContent = c.desc || '—';
  document.getElementById('st-ord').textContent = c.ord;
  document.getElementById('st-bns').textContent = c.bns;
  document.getElementById('st-gap').textContent = c.gap !== null ? `${c.gap}d` : '—';
  document.getElementById('st-last').textContent = c.last !== null ? `${c.last}d` : '—';
  document.getElementById('st-prep').textContent = `${c.prep}%`;
  document.getElementById('st-def').textContent = `${c.def}d / ${c.ovd}%`;

  document.getElementById('hist-badge').textContent = d.hist.length ? `${d.hist.length}` : '0';
  document.getElementById('hist-tbody').innerHTML = d.hist.length
    ? d.hist.map(h => `<tr>
        <td class="mono" style="font-size:11.5px;white-space:nowrap;">${esc(h.dt)}<div style="color:var(--muted);font-size:10.5px;">-${h.ago}d</div></td>
        <td><div style="font-weight:600;">${esc(h.n)}</div><div style="color:var(--muted);font-size:11px;">${esc(h.br)} · ${esc(h.cat)}</div></td>
        <td class="mono">${h.q}</td>
      </tr>`).join('')
    : `<tr><td colspan="3" style="color:var(--muted);padding:14px;">${esc(t.noHist)}</td></tr>`;
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('sec-' + btn.dataset.tab).classList.add('active');
  });
});
document.getElementById('btn-lang').addEventListener('click', () => { lang = lang === 'uk' ? 'en' : 'uk'; applyLang(); });
document.getElementById('btn-theme').addEventListener('click', () => {
  document.body.classList.toggle('dark');
  document.getElementById('btn-theme').textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
});
document.getElementById('sel-cohort').addEventListener('change', rebuildFilter);
document.getElementById('sel-mode').addEventListener('change', renderCurrentCase);
document.getElementById('inp-search').addEventListener('input', rebuildFilter);
document.getElementById('btn-prev').addEventListener('click', () => { curPos -= 1; renderCurrentCase(); });
document.getElementById('btn-next').addEventListener('click', () => { curPos += 1; renderCurrentCase(); });
document.getElementById('btn-random').addEventListener('click', () => {
  if (filteredCases.length > 1) {
    let nxt = curPos;
    while (nxt === curPos) nxt = Math.floor(Math.random() * filteredCases.length);
    curPos = nxt;
    renderCurrentCase();
  }
});
document.getElementById('rng-boost').addEventListener('input', renderCurrentCase);
document.getElementById('btn-reset-boost').addEventListener('click', () => {
  document.getElementById('rng-boost').value = '0';
  renderCurrentCase();
});

applyLang();
rebuildFilter();
</script>
</body>
</html>
"""


README_START_TXT = """====================================================================
  B2B AI Product Recommender — Interactive Demo + Final Model
====================================================================

[UA] ЯК ВІДКРИТИ ДЕМО:
  1. Розпакуйте цей ZIP-архів у будь-яку папку.
  2. Двічі клікніть на файл OPEN_DEMO.html — він відкриється у браузері
     (Chrome / Safari / Edge) без інтернету та без встановлення програм.
  3. У вкладці «Інтерактивні прогнози» можна перемикати режими
     (Повтори + Новинки / Лише новинки), шукати компанії та рухати
     повзунок балансу повторів і нових товарів у реальному часі.

  У папці model/ лежить сама навчена модель PyTorch (best_discovery.pt)
  та її повна специфікація (v2_full_reference.json).

[EN] HOW TO OPEN:
  1. Unzip this archive anywhere.
  2. Double-click OPEN_DEMO.html to launch the interactive demo directly
     in any web browser (offline, zero installation required).
  3. The model/ folder contains the trained PyTorch checkpoint
     (best_discovery.pt, w2_2mem_recency_hl365 Ep 10) and verified metadata.
====================================================================
"""


def build_portable_demo_zip(
    *,
    output_zip: Path = DEFAULT_OUTPUT_ZIP,
    seed: int = 42,
) -> Path:
    t0 = time.perf_counter()
    engine = V2FullLiveDemoEngine(checkpoint_kind="discovery")
    ref = json.loads(V2_FULL_REFERENCE_PATH.read_text(encoding="utf-8"))
    ckpt_path = Path(ref["checkpoints"]["ranker"]["path"])

    selected_ids, _ = _select_stratified_cases(engine, seed=seed)
    print(f"[portable-demo] Running real inference on {len(selected_ids)} stratified validation baskets...", flush=True)
    cases = [_score_case_for_export(engine, idx) for idx in selected_ids]

    # Mark clean showcase highlights (real cases with strong recall + true new discoveries or cold-start hits)
    showcase_scored = sorted(
        cases,
        key=lambda d: (
            (d["nh10"] > 0 and d["h10"] >= min(d["ntgt"], 2)),
            d["nh10"],
            d["h10"] / max(d["ntgt"], 1),
            d["ntgt"],
        ),
        reverse=True,
    )
    showcase_ids = {d["id"] for d in showcase_scored[:45]}
    for d in cases:
        d["sc"] = d["id"] in showcase_ids
    # Order so showcase highlights appear first
    cases.sort(key=lambda d: (not d["sc"], -d["nh10"], -(d["h10"] / max(d["ntgt"], 1)), d["id"]))

    payload = {
        "meta": {
            "reference_id": ref["reference_id"],
            "arm": ref["active_architecture"]["arm"],
            "epoch": int(engine.active_epoch),
            "train_params": int(ref["active_architecture"]["trainable_params"]),
            "per_request_params": int(ref["active_architecture"]["per_request_params"]),
            "num_formulations": int(engine.n_bn),
            "validation_baskets": int(engine.n_cases),
            "date_range": [str(engine.pred_dates[0].date()), str(engine.pred_dates[-1].date())],
            "checkpoint_bytes": int(ckpt_path.stat().st_size),
            "checkpoint_sha256": file_digest(ckpt_path),
        },
        "cases": cases,
    }

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_out = PORTABLE_HTML_TEMPLATE.replace("__DEMO_PAYLOAD_JSON__", payload_json)

    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    root_folder = "B2B_Recommender_Demo"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(f"{root_folder}/OPEN_DEMO.html", html_out.encode("utf-8"))
        zf.writestr(f"{root_folder}/README_START.txt", README_START_TXT.encode("utf-8"))
        zf.write(ckpt_path, arcname=f"{root_folder}/model/best_discovery.pt")
        zf.write(V2_FULL_REFERENCE_PATH, arcname=f"{root_folder}/model/v2_full_reference.json")

    elapsed = time.perf_counter() - t0
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(
        f"[portable-demo] Built {output_zip} ({size_mb:.2f} MB, {len(cases)} real cases, "
        f"{len(showcase_ids)} showcase highlights) in {elapsed:.1f}s",
        flush=True,
    )
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_portable_demo_zip(output_zip=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
