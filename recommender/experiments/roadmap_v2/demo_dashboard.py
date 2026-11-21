"""Interactive live demo dashboard for Stage 5 No-MLP Full-Pool UPS-CANR on real held-out Test cases.

Usage (requires explicit --allow-heldout-test flag since cases come from >= 2026-02-19):
    PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --allow-heldout-test --port 8765
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.catalogue import RollingCandidatePool, load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    load_enriched_companies,
)
from recommender.features import (
    LEGACY_PROFILE_FEATURES,
    PointInTimeNormalizer,
    compute_point_in_time_features,
)
from recommender.scope import get_datasets_locs
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.split import TEST_START
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata, get_device

from .base_name_features import BaseNameRollingFeatureBuilder
from .commercial_features import CommercialRollingFeatureBuilder
from .config import EVIDENCE_ROOT
from .dated_baskets import (
    ENRICHMENT,
    FEATURE_NAMES,
    VOCABS,
    dated_candidate_features,
    first_observed_pack_indices,
)
from .evaluate_stage5_test import (
    CLEANED_PURCHASES_PATH,
    RAW_WORKBOOK_PATH,
    build_test_basket_cases,
    load_aligned_full_commercial_events,
    load_all_cleaned_purchases,
)
from .parsed_spec_features import ParsedSpecRollingFeatureBuilder, parse_company_equipment_domains
from .run_stage5 import _stage5_tower
from .run_stage5_bigger_ranker import (
    MODEST_VARIANTS,
    OUT_ROOT as BIGGER_ROOT,
    SWEEP2_VARIANTS,
    BiggerStage5BaseNameRanker,
)
from .stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    resolve_base_names_to_skus,
)


ACTIVE_REF_PATH = EVIDENCE_ROOT / "stage5_reference.json"
F_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class Stage5LiveDemoEngine:
    """Loads the frozen Stage 5 No-MLP Full-Pool UPS-CANR pipeline and evaluates real Test baskets live."""

    def __init__(
        self,
        *,
        allow_heldout_test: bool = False,
        device: Optional[str] = None,
        variant_override: Optional[str] = None,
        checkpoint_override: Optional[Path] = None,
    ):
        if not allow_heldout_test:
            raise PermissionError("Demo uses held-out Test baskets (>= 2026-02-19); pass --allow-heldout-test.")

        t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.dev = get_device(device)
        self.reference = json.loads(ACTIVE_REF_PATH.read_text(encoding="utf-8"))

        print("[demo] Loading catalogue, purchases, companies & frozen static bundles...", flush=True)
        dev_purchases = load_development_purchases(str(CLEANED_PURCHASES_PATH))
        self.all_purchases = load_all_cleaned_purchases(CLEANED_PURCHASES_PATH)
        self.catalogue = load_product_catalogue(get_datasets_locs("items"))
        self.companies = load_enriched_companies(purchases=self.all_purchases, minimum_purchases=0)
        self.company_map = self.companies.drop_duplicates("customer_id").set_index("customer_id").to_dict("index")

        enrichment = torch.load(ENRICHMENT, map_location="cpu", weights_only=False)
        self.product_ids = list(enrichment["product_ids"])
        self.vocabs = torch.load(VOCABS, map_location="cpu", weights_only=False)["vocabs"]
        config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["config"]
        text_file = company_embeddings_path(DEFAULT_RETRIEVER, config["company_cache_name"])
        self.texts = load_cached_embeddings(text_file, "customer_id")
        self.normalizer = PointInTimeNormalizer.from_state(config["profile_normalizer"])
        self.profiles = build_customer_metadata(self.companies, self.vocabs)
        self.sales_branch = self.companies.set_index("customer_id")["sales_branch"].to_dict()
        self.sales_branch_vocab = enrichment["customer_vocabs"]["sales_branch"]

        dev_pool = RollingCandidatePool(self.catalogue, dev_purchases)
        self.universe = build_base_name_universe(self.product_ids, self.catalogue, dev_purchases)
        self.n_bn = self.universe["num_base_names"]
        self.reps = first_observed_pack_indices(
            self.product_ids, self.universe["bn_to_member_pid_indices"], dev_pool.first_sales
        )
        self.representatives = (
            self.catalogue.set_index("product_id").loc[np.asarray(self.product_ids)[self.reps]].reset_index()
        )
        self.cat_indexed = self.catalogue.drop_duplicates("product_id").set_index("product_id")

        # Per-formulation display metadata
        self.bn_meta: List[Dict[str, str]] = []
        for b_i, bn_name in enumerate(self.universe["base_names"]):
            rep_pid = self.product_ids[int(self.reps[b_i])]
            row = self.cat_indexed.loc[rep_pid] if rep_pid in self.cat_indexed.index else {}
            self.bn_meta.append(
                {
                    "bn_idx": b_i,
                    "base_name": bn_name,
                    "brand": str(row.get("brand", "")) if pd.notna(row.get("brand", "")) else "",
                    "category": str(row.get("product_category", "")) if pd.notna(row.get("product_category", "")) else "",
                    "subcategory": str(row.get("product_subcategory", "")) if pd.notna(row.get("product_subcategory", "")) else "",
                    "business_line": str(row.get("business_line", "")) if pd.notna(row.get("business_line", "")) else "",
                    "rep_pid": rep_pid,
                }
            )

        # Load commercial events & strictly-causal rolling feature builders (allow_non_monotonic=True for random sampling!)
        print("[demo] Reconciling commercial ledger & initializing strictly-causal feature builders...", flush=True)
        full_commercial = load_aligned_full_commercial_events(RAW_WORKBOOK_PATH, self.all_purchases, self.catalogue)
        self.test_pool = RollingCandidatePool(self.catalogue, self.all_purchases)
        self.dynamic = DynamicFeatureBuilder(
            self.all_purchases[["purchase_date", "customer_id", "product_id", "quantity"]],
            self.catalogue,
            self.companies,
            self.product_ids,
            statistics_policy=STATISTICS_POLICY_ROLLING,
            allow_non_monotonic=True,
            allow_heldout_test=True,
        )
        self.base = BaseNameRollingFeatureBuilder(
            self.all_purchases,
            self.catalogue,
            self.companies,
            self.product_ids,
            allow_non_monotonic=True,
            allow_heldout_test=True,
        )
        self.comm = CommercialRollingFeatureBuilder(
            full_commercial, self.catalogue, self.product_ids, allow_heldout_test=True
        )
        self.specs = ParsedSpecRollingFeatureBuilder(
            self.all_purchases, self.catalogue, self.companies, self.product_ids, allow_heldout_test=True
        )
        self.histories = dict(tuple(self.all_purchases.groupby("customer_id", sort=False)))

        # Precompute held-out test basket index (>= 2026-02-19)
        self.cases, self.cohort_counts = build_test_basket_cases(
            self.all_purchases, self.product_ids, self.universe, self.test_pool
        )
        # Keep raw unmodeled formulation names per test basket for transparent display
        self.case_unmodeled_bns: List[List[str]] = []
        by_bn = self.universe["bn_to_idx"]
        all_pid_to_bn = self.universe["all_pid_to_bn"]
        test_grouped = {
            (dt, cid): grp["product_id"].tolist()
            for (dt, cid), grp in self.all_purchases.loc[self.all_purchases["purchase_date"] >= TEST_START].groupby(
                ["purchase_date", "customer_id"], sort=False
            )
        }
        for c in self.cases:
            pids_day = test_grouped.get((c.date, c.customer), [])
            unmod = sorted({all_pid_to_bn.get(p, p) for p in pids_day if all_pid_to_bn.get(p, "") not in by_bn})
            self.case_unmodeled_bns.append(unmod)

        # Pre-index filter pools for instant UI filtering
        self.filter_indices = {
            "any": list(range(len(self.cases))),
            "returning": [i for i, c in enumerate(self.cases) if not c.first],
            "first_observed": [i for i, c in enumerate(self.cases) if c.first],
            "has_new": [i for i, c in enumerate(self.cases) if bool(c.new.any()) and not c.first],
            "multi_item": [i for i, c in enumerate(self.cases) if int(c.target.sum()) >= 2],
        }

        # Load frozen Two-Tower v_bn & active No-MLP Full-Pool UPS-CANR ranker
        print("[demo] Loading frozen Two-Tower vectors & active No-MLP Full-Pool UPS-CANR...", flush=True)
        tt_ckpt_path = Path(self.reference["checkpoints"]["two_tower"]["path"])
        tower = _stage5_tower(self.universe, enrichment).to(self.dev)
        tt_ckpt = torch.load(tt_ckpt_path, map_location="cpu", weights_only=False)
        tower.load_state_dict(tt_ckpt["model_state_dict"], strict=True)
        tower.eval()
        with torch.no_grad():
            p_dev = {
                "text_embedding": enrichment["text_embeddings"]["t1_fam"][self.reps].float().to(self.dev),
                "demand_features": torch.zeros(len(self.reps), 12, device=self.dev),
                "attr_ids": {k: v[self.reps].to(self.dev) for k, v in enrichment["product_attr_ids"].items()},
            }
            self.v_bn = tower.product_vectors(p_dev).detach()  # (585, 128) on device
        del tower

        orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]
        all_cfgs = {**MODEST_VARIANTS, **SWEEP2_VARIANTS}
        arch = dict(self.reference["ranker_architecture"])
        if variant_override:
            if variant_override not in all_cfgs:
                raise ValueError(f"Unknown variant {variant_override!r}; available: {list(all_cfgs)}")
            arch = {"variant": variant_override, **all_cfgs[variant_override]}
            ckpt_path = checkpoint_override or (BIGGER_ROOT / f"{variant_override}-seed42" / "best_ndcg.pt")
        else:
            ckpt_path = checkpoint_override or Path(self.reference["checkpoints"]["ranker"]["path"])

        self.active_variant = str(arch.get("variant", "modest_wider_mlp_384"))
        self.active_ckpt_path = str(ckpt_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.active_epoch = int(ck.get("epoch", 0))
        self.active_val_metrics = ck.get("metrics", {})

        self.ranker = BiggerStage5BaseNameRanker(
            num_base_names=self.n_bn,
            num_features=len(FEATURE_NAMES),
            num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
            num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
            num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
            num_categories=len(self.vocabs["category_vocab"]) + 1,
            num_subcategories=len(self.vocabs["subcategory_vocab"]) + 1,
            num_business_lines=len(self.vocabs["business_line_vocab"]) + 1,
            d_model=int(arch["d_model"]),
            num_heads=int(arch["num_heads"]),
            memory_layers=int(arch["memory_layers"]),
            memory_ffn_mult=int(arch.get("memory_ffn_mult", 2)),
            memory_ffn_dim=arch.get("memory_ffn_dim"),
            cross_attn_layers=int(arch["cross_attn_layers"]),
            mlp_widths=tuple(arch["mlp_widths"]),
            dropout=float(arch.get("dropout", 0.1)),
            aux_60d_enabled=False,
        ).to(self.dev)
        self.ranker.load_state_dict(ck["model_state_dict"], strict=True)
        self.ranker.eval()
        self.trainable_params = sum(p.numel() for p in self.ranker.parameters() if p.requires_grad)

        # Warmup once so live inference timings reflect steady-state latency
        self.evaluate_case(0)
        print(
            f"[demo] Ready in {time.perf_counter() - t0:.1f}s | {len(self.cases):,} held-out test baskets | "
            f"model={self.active_variant} (ep {self.active_epoch}, {self.trainable_params:,} params, dev={self.dev})",
            flush=True,
        )

    def metadata_payload(self) -> Dict[str, Any]:
        return {
            "reference_id": self.reference["reference_id"],
            "active_variant": self.active_variant,
            "active_epoch": self.active_epoch,
            "trainable_params": self.trainable_params,
            "device": str(self.dev),
            "num_base_names": self.n_bn,
            "num_features": len(FEATURE_NAMES),
            "test_baskets_total": len(self.cases),
            "test_date_range": [str(self.cases[0].date.date()), str(self.cases[-1].date.date())],
            "cohort_counts": self.cohort_counts,
            "filter_counts": {k: len(v) for k, v in self.filter_indices.items()},
            "val_metrics": self.active_val_metrics,
        }

    def evaluate_case(self, case_idx: int) -> Dict[str, Any]:
        with self.lock:
            c = self.cases[case_idx]
            cid = c.customer
            date = c.date
            day_int = int((date - PURCHASE_DATE_ORIGIN).days)

            # 1. Strictly-prior feature & sequence computation (< date)
            t_feat0 = time.perf_counter()
            feats_np = dated_candidate_features(
                cid, date, self.reps, self.dynamic, self.base, self.comm, self.specs
            ).astype(np.float32)

            hist_df = self.histories[cid]
            prior_df = hist_df.loc[hist_df["purchase_date"] < date].sort_values("purchase_date", ascending=False)
            stats_norm = compute_point_in_time_features(
                hist_df, cid, date, normalizer=self.normalizer, include_median_purchase_quantity=True
            )
            profile8 = [float(stats_norm[k]) for k in LEGACY_PROFILE_FEATURES]

            seqs = build_base_name_sequences(
                prior_df,
                [cid],
                [day_int],
                self.universe,
                self.representatives,
                self.vocabs["category_vocab"],
                self.vocabs["subcategory_vocab"],
                self.vocabs["business_line_vocab"],
                allow_heldout_test=True,
            )
            t_feat_ms = 1000.0 * (time.perf_counter() - t_feat0)

            # 2. Live No-MLP Full-Pool UPS-CANR forward pass (all 585 formulations!)
            t_inf0 = time.perf_counter()
            prof_ids = self.profiles.get(cid, {})
            text_vec = torch.from_numpy(self.texts.get(cid, np.zeros(1024, dtype=np.float32))).unsqueeze(0).to(self.dev)
            allowed_t = torch.from_numpy(c.allowed).unsqueeze(0).to(self.dev)
            with torch.no_grad():
                inp = {
                    "text_embedding": text_vec,
                    "city_id": torch.tensor([prof_ids.get("city_id", 0)], dtype=torch.long, device=self.dev),
                    "region_id": torch.tensor([prof_ids.get("region_id", 0)], dtype=torch.long, device=self.dev),
                    "industry_id": torch.tensor([prof_ids.get("industry_id", 0)], dtype=torch.long, device=self.dev),
                    "tabular_features": torch.tensor([profile8], dtype=torch.float32, device=self.dev),
                    "sequence": seqs["sequence"].to(self.dev),
                    "attention_mask": seqs["attention_mask"].to(self.dev),
                    "seq_product_ids": seqs["seq_product_ids"].to(self.dev),
                    "seq_category_ids": seqs["seq_category_ids"].to(self.dev),
                    "seq_subcategory_ids": seqs["seq_subcategory_ids"].to(self.dev),
                    "seq_business_line_ids": seqs["seq_business_line_ids"].to(self.dev),
                    "seq_purchase_dates": seqs["seq_purchase_dates"].to(self.dev),
                    "prediction_day": torch.tensor([float(day_int)], dtype=torch.float32, device=self.dev),
                    "pairwise_features": torch.from_numpy(feats_np).unsqueeze(0).to(self.dev),
                    "candidate_vectors": self.v_bn.unsqueeze(0),
                    "candidate_mask": allowed_t,
                    "candidate_indices": torch.arange(self.n_bn, device=self.dev).unsqueeze(0),
                }
                logits = self.ranker(**inp).logits.squeeze(0).cpu()
                if self.dev.type == "mps":
                    torch.mps.synchronize()
            t_inf_ms = 1000.0 * (time.perf_counter() - t_inf0)

            order = logits.argsort(descending=True, stable=True).numpy()
            rank_of_bn = {int(bn_i): pos + 1 for pos, bn_i in enumerate(order.tolist())}
            top10 = order[:10]

            # 3. Resolve Top-10 base_names -> concrete eligible pack SKUs strictly < date
            t_res0 = time.perf_counter()
            resolved_sku_idx = resolve_base_names_to_skus(
                top10[None, :],
                [cid],
                [day_int],
                torch.from_numpy(c.sku_allowed).unsqueeze(0),
                prior_df,
                self.catalogue,
                self.product_ids,
                self.universe,
                top_k=10,
            )[0]
            t_res_ms = 1000.0 * (time.perf_counter() - t_res0)

            # 4. Human-readable strictly-prior customer stats & purchase history
            prior_dates = sorted(prior_df["purchase_date"].unique())
            if len(prior_dates) >= 2:
                gaps = np.diff(np.array(prior_dates, dtype="datetime64[D]")).astype(float)
                median_gap_days = float(np.median(gaps))
            else:
                median_gap_days = None
            days_since_last = int((date - pd.Timestamp(prior_dates[-1])).days) if prior_dates else None

            all_pid_to_bn = self.universe["all_pid_to_bn"]
            prior_bns = {all_pid_to_bn.get(p, "") for p in prior_df["product_id"]} - {""}
            prior_history_rows = []
            if not prior_df.empty:
                # Aggregate same-day same-base_name lines (matching model sequence view!)
                p_view = prior_df.assign(bn=prior_df["product_id"].map(all_pid_to_bn).fillna(prior_df["product_id"]))
                agg_h = (
                    p_view.groupby(["purchase_date", "bn"], as_index=False)["quantity"]
                    .sum()
                    .sort_values(["purchase_date", "quantity"], ascending=[False, False])
                )
                by_bn = self.universe["bn_to_idx"]
                for _, r in agg_h.head(18).iterrows():
                    bn_s = str(r["bn"])
                    meta = self.bn_meta[by_bn[bn_s]] if bn_s in by_bn else {"brand": "", "category": ""}
                    prior_history_rows.append(
                        {
                            "date": str(pd.Timestamp(r["purchase_date"]).date()),
                            "days_ago": int((date - pd.Timestamp(r["purchase_date"])).days),
                            "base_name": bn_s,
                            "brand": meta["brand"],
                            "category": meta["category"],
                            "quantity": round(float(r["quantity"]), 2),
                        }
                    )

            # Customer profile + financial/equipment signals from feats_np[0] (customer-level columns are broadcast)
            comp = self.company_map.get(cid, {})
            desc_str = str(comp.get("description", "")) if pd.notna(comp.get("description", "")) else ""
            ind_str = str(comp.get("industry_branch", "")) if pd.notna(comp.get("industry_branch", "")) else ""
            equip_domains = sorted(parse_company_equipment_domains(desc_str, ind_str))

            customer_panel = {
                "customer_id": cid,
                "company_name": str(comp.get("company_name", cid)) if pd.notna(comp.get("company_name", "")) else cid,
                "city": str(comp.get("city", "—")) if pd.notna(comp.get("city", "")) else "—",
                "region": str(comp.get("region_code", "—")) if pd.notna(comp.get("region_code", "")) else "—",
                "industry": ind_str or "—",
                "sales_branch": str(comp.get("sales_branch", "—")) if pd.notna(comp.get("sales_branch", "")) else "—",
                "description_snippet": (desc_str[:220] + "…") if len(desc_str) > 220 else (desc_str or "No company description recorded."),
                "equipment_domains": equip_domains,
                "is_first_observed": bool(c.first),
                "prior_order_days": int(len(prior_dates)),
                "prior_line_items": int(len(prior_df)),
                "prior_distinct_formulations": int(len(prior_bns)),
                "prior_total_quantity": round(float(prior_df["quantity"].sum()), 1) if not prior_df.empty else 0.0,
                "median_order_gap_days": round(median_gap_days, 1) if median_gap_days is not None else None,
                "days_since_last_order": days_since_last,
                "financial": {
                    "prepaid_or_cash_pct": round(100.0 * float(feats_np[0, F_IDX["cust_prepaid_or_cash_ratio"]]), 1),
                    "avg_deferral_days": round(90.0 * float(feats_np[0, F_IDX["cust_avg_deferral_days_norm"]]), 1),
                    "actual_vs_contract_ratio": round(2.0 * float(feats_np[0, F_IDX["cust_actual_vs_contract_pay_ratio"]]), 2),
                    "overdue_unpaid_pct": round(100.0 * float(feats_np[0, F_IDX["cust_overdue_unpaid_ratio"]]), 1),
                    "mgr_changed_recently": bool(feats_np[0, F_IDX["mgr_changed_recently"]] > 0.5),
                },
            }

            # 5. Build Top-10 predictions list with signal drivers & hit highlighting
            predictions = []
            rel_flags = []
            new_hits_10 = 0
            for rank_0, bn_i in enumerate(top10.tolist()):
                meta = self.bn_meta[bn_i]
                f_row = feats_np[bn_i]
                logit_v = float(logits[bn_i])
                prob_v = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit_v)))))
                is_hit = bool(c.target[bn_i])
                is_new_candidate = bool(c.unseen[bn_i])
                is_new_hit = bool(c.new[bn_i])
                rel_flags.append(1.0 if is_hit else 0.0)
                if is_new_hit:
                    new_hits_10 += 1

                # Concrete resolved pack SKU name
                r_pidx = int(resolved_sku_idx[rank_0])
                resolved_sku_label = "—"
                if r_pidx >= 0:
                    r_pid = self.product_ids[r_pidx]
                    if r_pid in self.cat_indexed.index:
                        r_row = self.cat_indexed.loc[r_pid]
                        pname = str(r_row.get("product_name", r_pid)) if pd.notna(r_row.get("product_name", "")) else r_pid
                        cont = str(r_row.get("container_type", "")) if pd.notna(r_row.get("container_type", "")) else ""
                        resolved_sku_label = f"{r_pid} ({cont})" if cont else f"{r_pid}: {pname[:36]}"
                    else:
                        resolved_sku_label = r_pid

                # Build interpretable drivers from the exact 55 causal features
                drivers: List[str] = []
                if f_row[F_IDX["has_bought_base_sku"]] > 0.5:
                    approx_orders = max(1, int(round(math.expm1(float(f_row[F_IDX["user_base_sku_order_count_norm"]]) * math.log1p(50.0)))))
                    approx_days = int(round(math.expm1(float(f_row[F_IDX["days_since_last_base_sku_buy_norm"]]) * math.log1p(365.0))))
                    drivers.append(f"Repeat ({approx_orders}× · {approx_days}d ago)")
                    urg = float(f_row[F_IDX["base_sku_urgency"]]) * 5.0
                    if urg >= 0.6:
                        drivers.append(f"Due ratio {urg:.1f}×")
                else:
                    drivers.append("New to customer")
                if f_row[F_IDX["base_name_recent3_lift"]] > 0.15:
                    drivers.append(f"Seq lift +{f_row[F_IDX['base_name_recent3_lift']]:.2f}")
                if f_row[F_IDX["mgr_base_name_share_90d"]] > 0.02:
                    drivers.append(f"Mgr 90d {100*f_row[F_IDX['mgr_base_name_share_90d']]:.1f}%")
                if f_row[F_IDX["base_name_industry_affinity"]] > 0.03:
                    drivers.append(f"Industry {100*f_row[F_IDX['base_name_industry_affinity']]:.1f}%")
                if f_row[F_IDX["app_matches_company_equipment"]] > 0.5:
                    drivers.append("Equipment match")
                if f_row[F_IDX["same_viscosity_grade_bought"]] > 0.5 and is_new_candidate:
                    drivers.append("Same viscosity grade")

                predictions.append(
                    {
                        "rank": rank_0 + 1,
                        "bn_idx": bn_i,
                        "base_name": meta["base_name"],
                        "brand": meta["brand"],
                        "category": meta["category"],
                        "resolved_sku": resolved_sku_label,
                        "logit": round(logit_v, 3),
                        "probability": round(prob_v, 4),
                        "is_repeat_candidate": not is_new_candidate,
                        "is_hit": is_hit,
                        "is_new_hit": is_new_hit,
                        "drivers": drivers[:4],
                    }
                )

            # 6. Ground-truth basket items & exact case metrics (Full Catalogue denominator)
            gt_items = []
            for bn_i in np.flatnonzero(c.target).tolist():
                meta = self.bn_meta[bn_i]
                rk = rank_of_bn.get(bn_i) if c.allowed[bn_i] else None
                gt_items.append(
                    {
                        "base_name": meta["base_name"],
                        "brand": meta["brand"],
                        "category": meta["category"],
                        "is_new": bool(c.new[bn_i]),
                        "is_eligible": bool(c.allowed[bn_i]),
                        "is_modeled": True,
                        "predicted_rank": rk,
                        "hit_top10": (rk is not None and rk <= 10),
                    }
                )
            for u_bn in self.case_unmodeled_bns[case_idx]:
                gt_items.append(
                    {
                        "base_name": u_bn,
                        "brand": "New Test Launch",
                        "category": "Unmodeled (>2026-02-18)",
                        "is_new": True,
                        "is_eligible": False,
                        "is_modeled": False,
                        "predicted_rank": None,
                        "hit_top10": False,
                    }
                )
            gt_items.sort(key=lambda x: (x["predicted_rank"] is None, x["predicted_rank"] or 9999))

            n_pos_all = int(c.target.sum()) + int(c.extra_bn_targets)
            n_new_all = int(c.new.sum()) + int(c.extra_new_bn_targets)
            hits_10 = int(sum(rel_flags))
            disc = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
            dcg = float(np.sum(np.array(rel_flags, dtype=np.float64) * disc[: len(rel_flags)]))
            idcg = float(np.sum(disc[: min(n_pos_all, 10)])) if n_pos_all > 0 else 1.0
            case_ndcg = dcg / idcg if idcg > 0 else 0.0
            case_r10 = (hits_10 / n_pos_all) if n_pos_all > 0 else 0.0

            return {
                "case_idx": int(case_idx),
                "prediction_date": str(date.date()),
                "eligible_formulations": int(c.allowed.sum()),
                "total_formulations": int(self.n_bn),
                "timings_ms": {
                    "features_and_sequence": round(t_feat_ms, 2),
                    "fullpool_ups_canr": round(t_inf_ms, 2),
                    "pack_resolution": round(t_res_ms, 2),
                },
                "case_metrics": {
                    "hits_at_10": hits_10,
                    "total_targets": n_pos_all,
                    "new_hits_at_10": int(new_hits_10),
                    "total_new_targets": n_new_all,
                    "recall_at_10": round(case_r10, 4),
                    "ndcg_at_10": round(case_ndcg, 4),
                    "unmodeled_targets": int(c.extra_bn_targets),
                },
                "customer": customer_panel,
                "prior_history": prior_history_rows,
                "predictions_top10": predictions,
                "ground_truth_basket": gt_items,
            }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Stage 5 No-MLP Full-Pool UPS-CANR — Live Test-Case Demo</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #121826;
    --panel-2: #171f30;
    --border: #243047;
    --text: #e6edf7;
    --muted: #8fa1bf;
    --accent: #3b82f6;
    --accent-soft: rgba(59, 130, 246, 0.14);
    --hit-bg: rgba(16, 185, 129, 0.14);
    --hit-border: #10b981;
    --hit-text: #34d399;
    --new-bg: rgba(245, 158, 11, 0.16);
    --new-border: #f59e0b;
    --new-text: #fbbf24;
    --miss-bg: rgba(148, 163, 184, 0.08);
    --radius: 10px;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13.5px;
    line-height: 1.45;
    padding: 20px 24px 40px;
  }
  .container { max-width: 1380px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
  .header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
    background: linear-gradient(135deg, #121a2d 0%, #162238 100%);
    border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px;
  }
  .header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }
  .header .sub { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  select, button {
    font-family: var(--sans); font-size: 13px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    padding: 8px 12px; cursor: pointer;
  }
  button.primary {
    background: var(--accent); border-color: #2563eb; color: #fff; font-weight: 600;
    padding: 8px 16px; transition: opacity 0.15s;
  }
  button.primary:hover { opacity: 0.92; }
  button.primary:disabled { opacity: 0.5; cursor: wait; }

  .pipeline-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .pipe-step {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 11px 14px;
  }
  .pipe-step .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
  .pipe-step .v { font-size: 14px; font-weight: 700; margin-top: 3px; font-family: var(--mono); }
  .pipe-step .d { font-size: 11.5px; color: var(--muted); margin-top: 2px; }

  .metrics-bar {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
  }
  .metric-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px;
  }
  .metric-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .metric-card .val { font-size: 22px; font-weight: 700; font-family: var(--mono); margin-top: 4px; }
  .metric-card .hint { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .val.good { color: var(--hit-text); }
  .val.gold { color: var(--new-text); }

  .main-grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; align-items: start; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 18px; display: flex; flex-direction: column; gap: 12px;
  }
  .card-hdr { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .card-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #cbd5e1; }
  .pill {
    display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
    white-space: nowrap;
  }
  .pill.hit { background: var(--hit-bg); border-color: var(--hit-border); color: var(--hit-text); }
  .pill.newhit { background: var(--new-bg); border-color: var(--new-border); color: var(--new-text); }
  .pill.cold { background: rgba(168, 85, 247, 0.16); border-color: #a855f7; color: #d8b4fe; }
  .pill.ret { background: var(--accent-soft); border-color: #3b82f6; color: #93c5fd; }
  .pill.newtag { background: rgba(245, 158, 11, 0.10); border-color: rgba(245, 158, 11, 0.45); color: #fcd34d; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th {
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); padding: 7px 8px; border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 8px; border-bottom: 1px solid rgba(36, 48, 71, 0.55); vertical-align: middle; }
  tr.row-hit { background: var(--hit-bg); }
  tr.row-hit td:first-child { border-left: 3px solid var(--hit-border); }
  tr.row-newhit { background: var(--new-bg); }
  tr.row-newhit td:first-child { border-left: 3px solid var(--new-border); }
  .rank-num { font-family: var(--mono); font-weight: 700; font-size: 13px; width: 32px; }
  .prod-name { font-weight: 600; color: #f8fafc; }
  .prod-meta { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .drivers { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .drv {
    font-size: 10.5px; font-family: var(--mono); padding: 1px 6px; border-radius: 4px;
    background: rgba(148, 163, 184, 0.12); color: #cbd5e1;
  }
  .prob-wrap { width: 88px; }
  .prob-txt { font-family: var(--mono); font-size: 11.5px; font-weight: 600; }
  .prob-bar { height: 5px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 3px; }
  .prob-fill { height: 100%; background: var(--accent); border-radius: 3px; }
  tr.row-hit .prob-fill { background: var(--hit-border); }
  tr.row-newhit .prob-fill { background: var(--new-border); }

  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .stat-box {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px;
  }
  .stat-box .sk { font-size: 10.5px; color: var(--muted); }
  .stat-box .sv { font-size: 14px; font-weight: 700; font-family: var(--mono); margin-top: 2px; }
  .desc-box {
    font-size: 12px; color: var(--muted); background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px;
  }
  .gt-list { display: flex; flex-direction: column; gap: 6px; }
  .gt-item {
    display: flex; justify-content: space-between; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); background: var(--panel-2);
  }
  .gt-item.hit { background: var(--hit-bg); border-color: var(--hit-border); }
  .gt-item.newhit { background: var(--new-bg); border-color: var(--new-border); }
  .hist-scroll { max-height: 290px; overflow-y: auto; }
  .session-bar {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    font-size: 12px; color: var(--muted); padding: 8px 14px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  }
  .session-bar strong { color: var(--text); font-family: var(--mono); }
  @media (max-width: 1020px) {
    .main-grid, .pipeline-strip, .metrics-bar { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Stage 5 No-MLP Full-Pool UPS-CANR — Live Held-Out Test Demo</h1>
      <div class="sub" id="model-subtitle">Loading model metadata...</div>
    </div>
    <div class="controls">
      <select id="cohort-filter" title="Filter held-out test baskets">
        <option value="any">Any Test Basket (6,582)</option>
        <option value="returning">Returning Customers</option>
        <option value="first_observed">First-Observed (Cold-Start)</option>
        <option value="has_new">Returning w/ True New Product</option>
        <option value="multi_item">Multi-Product Baskets (≥2)</option>
      </select>
      <button class="primary" id="btn-random">🎲 Pick Random Test Case</button>
    </div>
  </div>

  <div class="pipeline-strip">
    <div class="pipe-step">
      <div class="k">1. Strictly Causal Cutoff</div>
      <div class="v" id="p-date">—</div>
      <div class="d">Only orders &amp; settlements &lt; date</div>
    </div>
    <div class="pipe-step">
      <div class="k">2. 55D Feature + 32-Seq Build</div>
      <div class="v" id="p-feat-ms">—</div>
      <div class="d">base_name + 3A fin + 3C mgr + 3E specs</div>
    </div>
    <div class="pipe-step">
      <div class="k">3. Full-Pool UPS-CANR (No MLP)</div>
      <div class="v" id="p-inf-ms">—</div>
      <div class="d" id="p-pool">Scores all 585 formulations directly</div>
    </div>
    <div class="pipe-step">
      <div class="k">4. Causal Pack-Size Resolver</div>
      <div class="v" id="p-res-ms">—</div>
      <div class="d">Maps Top-10 base_name → eligible SKU</div>
    </div>
  </div>

  <div class="metrics-bar">
    <div class="metric-card">
      <div class="label">Case Recall@10</div>
      <div class="val good" id="m-r10">—</div>
      <div class="hint" id="m-hits">—</div>
    </div>
    <div class="metric-card">
      <div class="label">Case NDCG@10</div>
      <div class="val" id="m-ndcg">—</div>
      <div class="hint">Position-weighted ranking quality</div>
    </div>
    <div class="metric-card">
      <div class="label">True New Discoveries@10</div>
      <div class="val gold" id="m-new">—</div>
      <div class="hint">Never purchased before this date</div>
    </div>
    <div class="metric-card">
      <div class="label">Customer Cohort</div>
      <div class="val" id="m-cohort" style="font-size:16px;padding-top:4px;">—</div>
      <div class="hint" id="m-prior-summary">—</div>
    </div>
    <div class="metric-card">
      <div class="label">Session Avg (Sampled Cases)</div>
      <div class="val" id="m-sess-r10">—</div>
      <div class="hint" id="m-sess-cnt">0 test cases sampled</div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card">
      <div class="card-hdr">
        <div class="card-title">Top-10 Model Predictions (Correct Hits Highlighted)</div>
        <div style="display:flex;gap:6px;">
          <span class="pill hit">✓ Repeat Hit</span>
          <span class="pill newhit">★ New Discovery Hit</span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Recommended Formulation (base_name) &amp; Signal Drivers</th>
            <th>Resolved Pack SKU</th>
            <th>Score</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody id="pred-tbody">
          <tr><td colspan="5" style="color:var(--muted);padding:18px;">Loading first random held-out test case...</td></tr>
        </tbody>
      </table>

      <div style="margin-top:6px;">
        <div class="card-hdr" style="margin-bottom:8px;">
          <div class="card-title">Actual Ground-Truth Basket Bought on <span id="gt-date">—</span></div>
          <span class="pill" id="gt-count">—</span>
        </div>
        <div class="gt-list" id="gt-list"></div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Customer Profile &amp; Strictly-Prior Causal Stats</div>
          <span class="pill" id="cust-id-pill">—</span>
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;" id="cust-name">—</div>
          <div style="font-size:12px;color:var(--muted);margin-top:2px;" id="cust-loc">—</div>
        </div>
        <div class="desc-box" id="cust-desc">—</div>
        <div class="drivers" id="cust-equip"></div>

        <div class="stats-grid">
          <div class="stat-box"><div class="sk">Prior Order Days</div><div class="sv" id="s-orders">—</div></div>
          <div class="stat-box"><div class="sk">Distinct Formulations</div><div class="sv" id="s-bns">—</div></div>
          <div class="stat-box"><div class="sk">Median Cadence</div><div class="sv" id="s-gap">—</div></div>
          <div class="stat-box"><div class="sk">Days Since Last</div><div class="sv" id="s-last">—</div></div>
          <div class="stat-box"><div class="sk">Prepaid / Cash</div><div class="sv" id="s-prep">—</div></div>
          <div class="stat-box"><div class="sk">Avg Deferral / Overdue</div><div class="sv" id="s-fin">—</div></div>
        </div>
      </div>

      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Strictly-Prior Purchase History (&lt; Prediction Date)</div>
          <span class="pill" id="hist-count">—</span>
        </div>
        <div class="hist-scroll">
          <table>
            <thead>
              <tr><th>Date</th><th>Formulation (base_name)</th><th>Qty</th></tr>
            </thead>
            <tbody id="hist-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="session-bar">
    <div id="benchmark-footer">Validation benchmarks loading...</div>
    <div>Zero future leakage: every feature, history token, rolling market count &amp; payment settlement is strictly <strong>&lt; prediction_date</strong>.</div>
  </div>
</div>

<script>
const session = { count: 0, sumR10: 0, sumNdcg: 0, hits: 0, targets: 0, newHits: 0, newTargets: 0 };

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

async function loadMeta() {
  const res = await fetch('/api/meta');
  const m = await res.json();
  document.getElementById('model-subtitle').textContent =
    `Model: ${m.active_variant} (ep ${m.active_epoch}, ${(m.trainable_params/1e3).toFixed(0)}K params, ${m.device}) · ` +
    `${m.num_base_names} canonical base_names · Held-out Test window ${m.test_date_range[0]} → ${m.test_date_range[1]} (${m.test_baskets_total.toLocaleString()} baskets)`;
  const sel = document.getElementById('cohort-filter');
  sel.options[0].text = `Any Test Basket (${m.filter_counts.any.toLocaleString()})`;
  sel.options[1].text = `Returning Customers (${m.filter_counts.returning.toLocaleString()})`;
  sel.options[2].text = `First-Observed / Cold-Start (${m.filter_counts.first_observed.toLocaleString()})`;
  sel.options[3].text = `Returning w/ True New Product (${m.filter_counts.has_new.toLocaleString()})`;
  sel.options[4].text = `Multi-Product Baskets ≥2 (${m.filter_counts.multi_item.toLocaleString()})`;
  const vm = m.val_metrics || {};
  if (vm['bn_recall@10'] !== undefined) {
    document.getElementById('benchmark-footer').innerHTML =
      `Validation (6,480 baskets): <strong>Recall@10 ${(100*vm['bn_recall@10']).toFixed(2)}%</strong> · ` +
      `<strong>NDCG@10 ${vm['bn_ndcg@10'].toFixed(5)}</strong> · ` +
      `<strong>New@10 ${vm['true_new_bn_hits@10']}/${vm['true_new_bn_targets']}</strong>`;
  }
}

async function loadRandomCase() {
  const btn = document.getElementById('btn-random');
  const flt = document.getElementById('cohort-filter').value;
  btn.disabled = true;
  btn.textContent = 'Computing live...';
  try {
    const res = await fetch(`/api/case?filter=${encodeURIComponent(flt)}`);
    const d = await res.json();
    renderCase(d);
  } finally {
    btn.disabled = false;
    btn.textContent = '🎲 Pick Random Test Case';
  }
}

function renderCase(d) {
  session.count += 1;
  session.sumR10 += d.case_metrics.recall_at_10;
  session.sumNdcg += d.case_metrics.ndcg_at_10;
  session.hits += d.case_metrics.hits_at_10;
  session.targets += d.case_metrics.total_targets;
  session.newHits += d.case_metrics.new_hits_at_10;
  session.newTargets += d.case_metrics.total_new_targets;

  document.getElementById('p-date').textContent = `${d.prediction_date} (Case #${d.case_idx})`;
  document.getElementById('p-feat-ms').textContent = `${d.timings_ms.features_and_sequence.toFixed(1)} ms`;
  document.getElementById('p-inf-ms').textContent = `${d.timings_ms.fullpool_ups_canr.toFixed(2)} ms`;
  document.getElementById('p-pool').textContent = `${d.eligible_formulations} eligible / ${d.total_formulations} formulations`;
  document.getElementById('p-res-ms').textContent = `${d.timings_ms.pack_resolution.toFixed(2)} ms`;

  const cm = d.case_metrics;
  document.getElementById('m-r10').textContent = `${(100 * cm.recall_at_10).toFixed(1)}%`;
  document.getElementById('m-hits').textContent = `${cm.hits_at_10} of ${cm.total_targets} bought formulations in Top-10`;
  document.getElementById('m-ndcg').textContent = cm.ndcg_at_10.toFixed(3);
  document.getElementById('m-new').textContent = `${cm.new_hits_at_10} / ${cm.total_new_targets}`;

  const c = d.customer;
  document.getElementById('m-cohort').innerHTML = c.is_first_observed
    ? '<span class="pill cold">First-Observed (Cold-Start)</span>'
    : '<span class="pill ret">Returning Customer</span>';
  document.getElementById('m-prior-summary').textContent = c.is_first_observed
    ? '0 prior purchases — profile & market only'
    : `${c.prior_order_days} prior orders · ${c.prior_distinct_formulations} formulations`;

  document.getElementById('m-sess-r10').textContent = `${(100 * session.sumR10 / session.count).toFixed(1)}%`;
  document.getElementById('m-sess-cnt').textContent =
    `${session.count} sampled · ${session.hits}/${session.targets} hits · ${session.newHits}/${session.newTargets} new`;

  // Top-10 table
  const tbody = document.getElementById('pred-tbody');
  tbody.innerHTML = d.predictions_top10.map(p => {
    const cls = p.is_new_hit ? 'row-newhit' : (p.is_hit ? 'row-hit' : '');
    const badge = p.is_new_hit
      ? '<span class="pill newhit">★ HIT · NEW</span>'
      : (p.is_hit
          ? '<span class="pill hit">✓ HIT · Repeat</span>'
          : (p.is_repeat_candidate ? '<span class="pill">Repeat</span>' : '<span class="pill newtag">New Candidate</span>'));
    const drvHtml = p.drivers.map(x => `<span class="drv">${esc(x)}</span>`).join('');
    return `<tr class="${cls}">
      <td class="rank-num">#${p.rank}</td>
      <td>
        <div class="prod-name">${esc(p.base_name)}</div>
        <div class="prod-meta">${esc(p.brand)} · ${esc(p.category)}</div>
        <div class="drivers">${drvHtml}</div>
      </td>
      <td style="font-family:var(--mono);font-size:11.5px;color:#cbd5e1;">${esc(p.resolved_sku)}</td>
      <td class="prob-wrap">
        <div class="prob-txt">${(100*p.probability).toFixed(1)}%</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${Math.min(100, Math.round(100*p.probability))}%"></div></div>
      </td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  // Ground truth basket
  document.getElementById('gt-date').textContent = d.prediction_date;
  document.getElementById('gt-count').textContent = `${cm.total_targets} formulations (${cm.total_new_targets} new)`;
  document.getElementById('gt-list').innerHTML = d.ground_truth_basket.map(g => {
    const cls = g.hit_top10 ? (g.is_new ? 'newhit' : 'hit') : '';
    const rkTxt = g.predicted_rank !== null
      ? (g.hit_top10 ? `Rank #${g.predicted_rank} ✓` : `Rank #${g.predicted_rank}`)
      : (g.is_modeled ? 'Ineligible on date' : 'Unmodeled test launch');
    const tag = g.is_new ? '<span class="pill newtag">Truly New</span>' : '<span class="pill">Repeat</span>';
    return `<div class="gt-item ${cls}">
      <div>
        <div class="prod-name">${esc(g.base_name)}</div>
        <div class="prod-meta">${esc(g.brand)} · ${esc(g.category)}</div>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        ${tag}
        <span class="pill ${g.hit_top10 ? (g.is_new ? 'newhit' : 'hit') : ''}" style="font-family:var(--mono);">${esc(rkTxt)}</span>
      </div>
    </div>`;
  }).join('');

  // Customer panel
  document.getElementById('cust-id-pill').textContent = c.customer_id;
  document.getElementById('cust-name').textContent = c.company_name;
  document.getElementById('cust-loc').textContent = `${c.city} (${c.region}) · Industry: ${c.industry} · Branch: ${c.sales_branch}`;
  document.getElementById('cust-desc').textContent = c.description_snippet;
  document.getElementById('cust-equip').innerHTML = c.equipment_domains.length
    ? c.equipment_domains.map(e => `<span class="drv">equip:${esc(e)}</span>`).join('')
    : '<span class="drv">No parsed equipment tags</span>';

  document.getElementById('s-orders').textContent = c.prior_order_days;
  document.getElementById('s-bns').textContent = c.prior_distinct_formulations;
  document.getElementById('s-gap').textContent = c.median_order_gap_days !== null ? `${c.median_order_gap_days}d` : '—';
  document.getElementById('s-last').textContent = c.days_since_last_order !== null ? `${c.days_since_last_order}d` : 'Cold';
  document.getElementById('s-prep').textContent = `${c.financial.prepaid_or_cash_pct}%`;
  document.getElementById('s-fin').textContent = `${c.financial.avg_deferral_days}d / ${c.financial.overdue_unpaid_pct}%`;

  // Prior history table
  document.getElementById('hist-count').textContent =
    c.is_first_observed ? '0 prior orders' : `Showing ${d.prior_history.length} of ${c.prior_line_items} prior lines`;
  const hbody = document.getElementById('hist-tbody');
  if (!d.prior_history.length) {
    hbody.innerHTML = '<tr><td colspan="3" style="color:var(--muted);padding:14px;">First observed order — zero prior purchase history. Predictions come from company profile, industry affinity, equipment match &amp; rolling market signals.</td></tr>';
  } else {
    hbody.innerHTML = d.prior_history.map(h => `<tr>
      <td style="font-family:var(--mono);font-size:11.5px;white-space:nowrap;">${esc(h.date)}<div style="color:var(--muted);font-size:10.5px;">-${h.days_ago}d</div></td>
      <td><div style="font-weight:600;">${esc(h.base_name)}</div><div style="color:var(--muted);font-size:11px;">${esc(h.brand)} · ${esc(h.category)}</div></td>
      <td style="font-family:var(--mono);">${h.quantity}</td>
    </tr>`).join('');
  }
}

document.getElementById('btn-random').addEventListener('click', loadRandomCase);
document.getElementById('cohort-filter').addEventListener('change', loadRandomCase);

loadMeta().then(loadRandomCase);
</script>
</body>
</html>
"""


def create_server(engine: Stage5LiveDemoEngine, host: str, port: int) -> ThreadingHTTPServer:
    class DemoHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/meta":
                self._send_json(engine.metadata_payload())
            elif parsed.path == "/api/case":
                qs = parse_qs(parsed.query)
                if "idx" in qs:
                    idx = max(0, min(len(engine.cases) - 1, int(qs["idx"][0])))
                else:
                    flt = qs.get("filter", ["any"])[0]
                    pool_ids = engine.filter_indices.get(flt, engine.filter_indices["any"])
                    idx = random.choice(pool_ids)
                self._send_json(engine.evaluate_case(idx))
            else:
                self._send_json({"error": "Not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), DemoHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-heldout-test",
        action="store_true",
        help="Explicitly authorize live demo inference over held-out Test baskets (>= 2026-02-19).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    engine = Stage5LiveDemoEngine(
        allow_heldout_test=args.allow_heldout_test,
        device=args.device,
        variant_override=args.variant,
        checkpoint_override=args.checkpoint,
    )
    server = create_server(engine, args.host, args.port)
    print(f"[demo] Live dashboard running at http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
