"""Interactive live demo dashboard for the active recommendation model (`v2_full` default, `v1` optional).

Usage:
    # Default: canonical v2_full single-stage E2E Wave2BE2ERanker + Option A (strictly Validation, zero Test access):
    PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --port 8765

    # Legacy v1 Stage 5 Shared-UV model on held-out Test baskets (>= 2026-02-19):
    PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --model v1 --allow-heldout-test --port 8765
"""

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

from recommender.catalogue import RollingCandidatePool, load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER
from recommender.experiments.static_enrichment.metadata import load_enriched_companies
from recommender.scope import get_datasets_locs
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata

from .clean_ranker import CleanStage5BaseNameRanker, SharedUVStage5Ranker
from .config import EVIDENCE_ROOT
from .dated_baskets import (
    ENRICHMENT,
    VOCABS,
    first_observed_pack_indices,
)
from .evaluate_stage5_test import (
    CLEANED_PURCHASES_PATH,
    RAW_WORKBOOK_PATH,
    build_test_basket_cases,
    load_aligned_full_commercial_events,
    load_all_cleaned_purchases,
)
from .native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    build_base_name_static_representations,
    candidate_storage_dtype,
    observed_pack_ids,
    representative_attr_ids,
)
from .parsed_spec_features import parse_company_equipment_domains
from .run_stage5 import _stage5_tower, _verify_reference_files
from .stage5_base_entity import build_base_name_universe


ACTIVE_REF_PATH = EVIDENCE_ROOT / "stage5_reference.json"
C_IDX = {name: i for i, name in enumerate(CANDIDATE_38_FEATURE_NAMES)}
U_IDX = {name: i for i, name in enumerate(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)}


def _case_filter_indices(cases):
    return {
        "any": list(range(len(cases))),
        "returning": [i for i, c in enumerate(cases) if not c.first],
        "first_observed": [i for i, c in enumerate(cases) if c.first],
        "has_new": [i for i, c in enumerate(cases) if (c.new.any() or c.extra_new_bn_targets > 0) and not c.first],
        "multi_item": [i for i, c in enumerate(cases) if int(c.target.sum()) + c.extra_bn_targets >= 2],
    }


def _live_candidate_features(values, storage_dtype):
    return values.astype(storage_dtype, copy=False).astype(np.float32, copy=False)


class Stage5LiveDemoEngine:
    """Loads the frozen Stage 5 No-MLP Full-Pool UPS-CANR pipeline and evaluates real Test baskets live."""

    def __init__(
        self,
        *,
        allow_heldout_test: bool = False,
        device: Optional[str] = None,
        variant_override: Optional[str] = None,
        checkpoint_override: Optional[Path] = None,
        frozen_v5_replay: bool = False,
    ):
        if not allow_heldout_test:
            raise PermissionError("Demo uses held-out Test baskets (>= 2026-02-19); pass --allow-heldout-test.")

        t0 = time.perf_counter()
        self.lock = threading.Lock()
        # Default to CPU for single-request real-time serving (~1.13 ms median, zero MPS sync overhead!)
        self.dev = torch.device(device) if device else torch.device("cpu")
        self.reference = json.loads(ACTIVE_REF_PATH.read_text(encoding="utf-8"))
        _verify_reference_files(self.reference)
        self.frozen_v5_replay = frozen_v5_replay
        if frozen_v5_replay and self.reference.get("reference_id") != "stage5-v5-shared-uv-ep10":
            raise ValueError("Historical v5 replay requires the pinned stage5-v5-shared-uv-ep10 reference.")
        self.candidate_dtype = candidate_storage_dtype(self.reference.get("candidate_feature_storage_dtype", "float16"))

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
        self.profiles = build_customer_metadata(self.companies, self.vocabs)

        dev_pool = RollingCandidatePool(self.catalogue, dev_purchases)
        observed_before = (None if self.reference.get("metadata_policy") == "snapshot_unverified"
                           else self.reference.get("observed_before", TRAIN_CUTOFF))
        allowed_members = observed_pack_ids(dev_pool.first_sales, observed_before)
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

        # Load commercial events & single unified NativeBaseNameFeatureEngine (~0.55 ms/req!)
        print("[demo] Reconciling commercial ledger & initializing NativeBaseNameFeatureEngine...", flush=True)
        full_commercial = load_aligned_full_commercial_events(RAW_WORKBOOK_PATH, self.all_purchases, self.catalogue)
        self.test_pool = RollingCandidatePool(self.catalogue, self.all_purchases)
        self.native_engine = NativeBaseNameFeatureEngine(
            self.all_purchases[["purchase_date", "customer_id", "product_id", "quantity"]],
            full_commercial,
            self.catalogue,
            self.companies,
            self.product_ids,
            self.universe,
            self.reps,
            allowed_member_pids=allowed_members,
            frozen_v5_replay=frozen_v5_replay,
            allow_non_monotonic=True,
            allow_heldout_test=True,
        )
        self.histories = dict(tuple(self.all_purchases.groupby("customer_id", sort=False)))

        # Fast startup pre-indexing (eliminates per-click DataFrame scans & 2.5-year table replays!)
        self._init_fast_sequence_and_resolver_indices()
        self._init_relationship_checkpoints()

        # Precompute held-out test basket index (>= 2026-02-19)
        self.cases, self.cohort_counts = build_test_basket_cases(
            self.all_purchases, self.product_ids, self.universe, self.test_pool
        )
        # Pre-index filter pools for instant UI filtering
        self.filter_indices = _case_filter_indices(self.cases)

        # Load frozen Pure Pack-Free + 65D Structured Multi-Hot Two-Tower (keep CustomerTower live for per-request u_bn!)
        print("[demo] Loading frozen 65D Two-Tower (cached v_bn + live CustomerTower u_bn) & active ranker...", flush=True)
        bn_static = build_base_name_static_representations(
            self.universe, self.product_ids, self.catalogue, self.reps, allowed_member_pids=allowed_members,
            frozen_v5_specs=self.reference["reference_id"] == "stage5-v5-shared-uv-ep10",
        )
        attr_ids = representative_attr_ids(enrichment, self.product_ids, self.reps, allowed_members)
        tt_ckpt_path = Path(self.reference["checkpoints"]["two_tower"]["path"])
        spec_dim = int(self.reference.get("num_product_structured_specs", 65))
        self.tower = _stage5_tower(
            self.universe,
            enrichment,
            num_demand_features=spec_dim,
            num_customer_tabular_features=0,
            drop_sales_branch=not bool(self.reference.get("sales_branch_enabled", False)),
        ).to(self.dev)
        tt_ckpt = torch.load(tt_ckpt_path, map_location="cpu", weights_only=False)
        self.tower.load_state_dict(tt_ckpt["model_state_dict"], strict=True)
        self.tower.eval()
        with torch.no_grad():
            p_dev = {
                "text_embedding": bn_static["text_pack_free"].float().to(self.dev),
                "demand_features": bn_static["full_specs_multihot"].float().to(self.dev),
                "attr_ids": {k: v.to(self.dev) for k, v in attr_ids.items()},
            }
            self.v_bn = self.tower.product_vectors(p_dev).detach()  # (585, 128) on device

        orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]
        arch = dict(self.reference["ranker_architecture"])
        ckpt_path = checkpoint_override or Path(self.reference["checkpoints"]["ranker"]["path"])
        self.active_variant = str(variant_override or arch.get("variant", "v5_shared_uv"))
        self.active_ranker_class = str(arch.get("class", "SharedUVStage5Ranker"))
        self.active_ckpt_path = str(ckpt_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.active_epoch = int(ck.get("epoch", 0))
        self.active_val_metrics = ck.get("metrics", {})

        if self.active_ranker_class == "SharedUVStage5Ranker":
            self.ranker: torch.nn.Module = SharedUVStage5Ranker(
                num_candidate_features=int(arch.get("num_candidate_features", len(CANDIDATE_38_FEATURE_NAMES))),
                num_customer_features=int(arch.get("num_customer_features", len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES))),
                num_pairwise_bias_features=13,
                d_model=int(arch["d_model"]),
                num_heads=int(arch["num_heads"]),
                memory_ffn_dim=int(arch["memory_ffn_dim"]),
                mlp_widths=tuple(arch["mlp_widths"]),
                dropout=float(arch.get("dropout", 0.12)),
                aux_60d_enabled=False,
            ).to(self.dev)
        else:
            self.ranker = CleanStage5BaseNameRanker(
                num_base_names=self.n_bn,
                num_candidate_features=int(arch.get("num_candidate_features", len(CANDIDATE_38_FEATURE_NAMES))),
                num_customer_features=int(arch.get("num_customer_features", len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES))),
                num_pairwise_bias_features=13,
                num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
                num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
                num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
                num_sales_branches=int(arch.get("num_sales_branches", 0)),
                num_categories=len(self.vocabs["category_vocab"]) + 1,
                num_subcategories=len(self.vocabs["subcategory_vocab"]) + 1,
                num_business_lines=len(self.vocabs["business_line_vocab"]) + 1,
                d_model=int(arch["d_model"]),
                num_heads=int(arch["num_heads"]),
                memory_ffn_dim=int(arch["memory_ffn_dim"]),
                mlp_widths=tuple(arch["mlp_widths"]),
                dropout=float(arch.get("dropout", 0.12)),
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

    def _init_fast_sequence_and_resolver_indices(self) -> None:
        """Pre-index customer base_name sequences & pack-resolver timelines ONCE at startup (<0.2 ms/req)."""
        from .base_name_features import _clean_base_name

        # 1. Pre-index base_name -> (cat_id, subcat_id, line_id) from representatives
        cat_vocab = self.vocabs["category_vocab"]
        subcat_vocab = self.vocabs["subcategory_vocab"]
        line_vocab = self.vocabs["business_line_vocab"]
        rep_df = self.representatives.copy()
        rep_df["bn"] = rep_df["base_name"].map(_clean_base_name)
        bn_meta_ids: Dict[str, tuple[int, int, int]] = {}
        for bn, grp in rep_df.loc[rep_df["bn"] != ""].groupby("bn", sort=False):
            c_v = str(grp["product_category"].dropna().mode().iloc[0]).strip() if grp["product_category"].notna().any() else ""
            sc_v = str(grp["product_subcategory"].dropna().mode().iloc[0]).strip() if grp["product_subcategory"].notna().any() else ""
            bl_v = str(grp["business_line"].dropna().mode().iloc[0]).strip() if grp["business_line"].notna().any() else ""
            bn_meta_ids[bn] = (
                int(cat_vocab.get(c_v, 0)),
                int(subcat_vocab.get(sc_v, 0)),
                int(line_vocab.get(bl_v, 0)),
            )

        all_pid_to_bn = self.universe["all_pid_to_bn"]
        bn_to_idx = self.universe["bn_to_idx"]
        df = self.all_purchases[["customer_id", "purchase_date", "product_id", "quantity"]].copy()
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).clip(lower=0.0)
        df["bn"] = df["product_id"].map(all_pid_to_bn).fillna("")
        agg = (
            df.loc[df["bn"] != ""]
            .groupby(["customer_id", "purchase_date", "bn"], as_index=False, sort=False)["quantity"]
            .sum()
            .sort_values(["customer_id", "purchase_date", "bn"], kind="stable")
        )
        self._cust_seq_events: Dict[str, tuple[np.ndarray, list[tuple[int, str, int, float, int, int, int, int]]]] = {}
        for cid, grp in agg.groupby("customer_id", sort=False):
            dts = ((grp["purchase_date"] - PURCHASE_DATE_ORIGIN).dt.days).to_numpy(dtype=np.int64)
            months = grp["purchase_date"].dt.month.to_numpy(dtype=np.int64)
            evs = []
            for d_i, bn_s, q_v, m_i in zip(dts.tolist(), grp["bn"].tolist(), grp["quantity"].tolist(), months.tolist()):
                b_id = bn_to_idx.get(bn_s, -1) + 1
                c_i, sc_i, bl_i = bn_meta_ids.get(bn_s, (0, 0, 0))
                evs.append((d_i, bn_s, b_id, float(q_v), c_i, sc_i, bl_i, int(m_i)))
            self._cust_seq_events[cid] = (dts, evs)

        # 2. Pre-index pack-size resolver timelines
        pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        self._pid_cont = [
            str(self.cat_indexed.at[p, "container_type"]).strip().upper()
            if p in self.cat_indexed.index and pd.notna(self.cat_indexed.at[p, "container_type"])
            else ""
            for p in self.product_ids
        ]
        df_sorted = df.sort_values("purchase_date", kind="stable")
        self._sku_sale_dates: Dict[int, np.ndarray] = {}
        for pid, grp in df_sorted.groupby("product_id", sort=False):
            if pid in pid_to_idx:
                self._sku_sale_dates[pid_to_idx[pid]] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
        self._cust_sku_hist: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cid, grp in df_sorted.groupby("customer_id", sort=False):
            dts = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
            pidxs = np.array([pid_to_idx.get(p, -1) for p in grp["product_id"]], dtype=np.int64)
            self._cust_sku_hist[cid] = (dts, pidxs)

    def _init_relationship_checkpoints(self) -> None:
        """Snapshot native 585-BN RelationshipTable states biweekly across Test window so random jumps advance <=14d."""
        self._rel_tables = (
            *self.native_engine._tables.values(),
            self.native_engine._complements,
        )
        max_dt = self.all_purchases["purchase_date"].max()
        anchors = list(pd.date_range(TEST_START, max_dt, freq="14D"))
        self._rel_checkpoints: list[tuple[pd.Timestamp, list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]]] = []
        for tbl in self._rel_tables:
            tbl.reset()
        for dt in anchors:
            snap = []
            for tbl in self._rel_tables:
                tbl.advance(dt)
                snap.append((
                    tbl.pairs.astype(np.float32, copy=True),
                    tbl.source.astype(np.float32, copy=True),
                    tbl.dest.astype(np.float32, copy=True),
                    int(tbl.total),
                    int(tbl.cursor),
                ))
            self._rel_checkpoints.append((dt, snap))
        self.native_engine._last_date = anchors[-1]

    def _seek_relationship_tables(self, target_date: pd.Timestamp) -> None:
        """Restore nearest <= biweekly snapshot whenever target_date jumps backward or >14 days forward."""
        if self.native_engine._last_date is not None and 0 <= (target_date - self.native_engine._last_date).days <= 14:
            return
        chosen_dt, chosen_snap = self._rel_checkpoints[0]
        for dt, snap in self._rel_checkpoints:
            if dt <= target_date:
                chosen_dt, chosen_snap = dt, snap
            else:
                break
        for tbl, (pairs, src, dst, tot, cur) in zip(self._rel_tables, chosen_snap):
            tbl.pairs = pairs.astype(np.float64, copy=True)
            tbl.source = src.astype(np.float64, copy=True)
            tbl.dest = dst.astype(np.float64, copy=True)
            tbl.total = tot
            tbl.cursor = cur
        self.native_engine._last_date = chosen_dt

    def _fast_build_sequence(self, cid: str, pday: int, seq_len: int = 32, *, frozen_v5_replay=False) -> Dict[str, torch.Tensor]:
        """Bit-exact single-customer base_name 32-event sequence builder (<0.1 ms)."""
        sequence = np.zeros((1, seq_len, 9), dtype=np.float32)
        seq_rel_qty = np.zeros((1, seq_len), dtype=np.float32)
        cust_median_gap_days = np.full((1,), 30.0, dtype=np.float32)
        attn_mask = np.zeros((1, seq_len), dtype=np.int64)
        seq_bn_ids = np.zeros((1, seq_len), dtype=np.int64)
        seq_cat_ids = np.zeros((1, seq_len), dtype=np.int64)
        seq_subcat_ids = np.zeros((1, seq_len), dtype=np.int64)
        seq_line_ids = np.zeros((1, seq_len), dtype=np.int64)
        seq_dates = np.zeros((1, seq_len), dtype=np.float32)

        entry = self._cust_seq_events.get(cid)
        if entry is not None:
            dts, evs = entry
            stop = int(dts.searchsorted(pday, side="left"))
            if stop > 0:
                log60 = math.log1p(60.0)
                log50 = math.log1p(50.0)
                prior_evs = evs[:stop]
                uniq_days = []
                for ev in prior_evs:
                    if not uniq_days or ev[0] != uniq_days[-1]:
                        uniq_days.append(ev[0])
                if len(uniq_days) >= 2:
                    day_gaps = [b - a for a, b in zip(uniq_days, uniq_days[1:])]
                    cust_median_gap_days[0] = float(np.median(day_gaps))
                bn_counts: Dict[str, int] = {}
                bn_vols: Dict[str, float] = {}
                cum_vol = 0.0
                start_keep = max(0, stop - seq_len)
                max_window_log_q = max(
                    (math.log1p(max(0.0, ev[3])) for ev in prior_evs[start_keep:]),
                    default=0.0,
                )
                prev_day = prior_evs[0][0]
                last_day = prior_evs[-1][0]
                for idx_e, (d_i, bn_s, b_id, q_v, c_i, sc_i, bl_i, m_i) in enumerate(prior_evs):
                    if frozen_v5_replay and b_id == 0:
                        bn_s = ""
                    prev_cnt = bn_counts.get(bn_s, 0)
                    bn_counts[bn_s] = prev_cnt + 1
                    bn_vols[bn_s] = bn_vols.get(bn_s, 0.0) + q_v
                    cum_vol += q_v
                    gap_days = max(0, d_i - prev_day) if idx_e > 0 else 0
                    prev_day = d_i
                    if idx_e >= start_keep:
                        slot = idx_e - start_keep
                        attn_mask[0, slot] = 1
                        seq_bn_ids[0, slot] = b_id
                        seq_cat_ids[0, slot] = c_i
                        seq_subcat_ids[0, slot] = sc_i
                        seq_line_ids[0, slot] = bl_i
                        seq_dates[0, slot] = float(d_i) / 365.25
                        seq_rel_qty[0, slot] = (
                            math.log1p(max(0.0, q_v)) / max_window_log_q
                            if max_window_log_q > 0.0
                            else 0.0
                        )
                        age_rel = max(0, last_day - d_i)
                        angle = 2.0 * math.pi * (m_i - 1) / 12.0
                        sequence[0, slot, 0] = min(1.0, math.log1p(q_v) / log60)
                        sequence[0, slot, 1] = float(age_rel) / (float(age_rel) + 90.0)
                        sequence[0, slot, 2] = min(1.0, float(gap_days) / 365.0)
                        sequence[0, slot, 3] = math.sin(angle)
                        sequence[0, slot, 4] = math.cos(angle)
                        sequence[0, slot, 5] = min(1.0, math.log1p(prev_cnt) / log50)
                        sequence[0, slot, 6] = min(1.0, bn_vols[bn_s] / max(cum_vol, 1e-6))
                        sequence[0, slot, 7] = 1.0 if prev_cnt > 0 else 0.0
                        sequence[0, slot, 8] = 1.0
        return {
            "sequence": torch.from_numpy(sequence),
            "seq_rel_qty": torch.from_numpy(seq_rel_qty),
            "cust_median_gap_days": torch.from_numpy(cust_median_gap_days),
            "attention_mask": torch.from_numpy(attn_mask),
            "seq_product_ids": torch.from_numpy(seq_bn_ids),
            "seq_category_ids": torch.from_numpy(seq_cat_ids),
            "seq_subcategory_ids": torch.from_numpy(seq_subcat_ids),
            "seq_business_line_ids": torch.from_numpy(seq_line_ids),
            "seq_purchase_dates": torch.from_numpy(seq_dates),
        }

    def _fast_resolve_skus(self, top_bns: np.ndarray, cid: str, date: pd.Timestamp, row_allowed: np.ndarray) -> np.ndarray:
        """Bit-exact single-customer causal pack-size resolver (<0.2 ms)."""
        dt_64 = date.to_datetime64()
        sku_cnt: Counter[int] = Counter()
        sku_last: Dict[int, int] = {}
        cont_cnt: Counter[str] = Counter()
        entry = self._cust_sku_hist.get(cid)
        if entry is not None:
            dts, pidxs = entry
            stop = int(dts.searchsorted(dt_64, side="left"))
            for pos_e in range(stop):
                p_i = int(pidxs[pos_e])
                if p_i >= 0:
                    sku_cnt[p_i] += 1
                    sku_last[p_i] = pos_e
                    if self._pid_cont[p_i]:
                        cont_cnt[self._pid_cont[p_i]] += 1
        dom_cont = cont_cnt.most_common(1)[0][0] if cont_cnt else ""
        members_map = self.universe["bn_to_member_pid_indices"]
        out = np.full(len(top_bns), -1, dtype=np.int64)
        for k, b_idx in enumerate(top_bns.tolist()):
            if b_idx < 0:
                continue
            elig = [p_i for p_i in members_map[b_idx] if row_allowed[p_i]]
            if not elig:
                continue
            if len(elig) == 1:
                out[k] = elig[0]
                continue
            prior_m = [p_i for p_i in elig if sku_cnt[p_i] > 0]
            if prior_m:
                out[k] = max(prior_m, key=lambda p_i: (sku_cnt[p_i], sku_last[p_i]))
                continue
            if dom_cont:
                cont_m = [p_i for p_i in elig if self._pid_cont[p_i] == dom_cont]
                if cont_m:
                    out[k] = max(
                        cont_m,
                        key=lambda p_i: int(self._sku_sale_dates[p_i].searchsorted(dt_64, side="left"))
                        if p_i in self._sku_sale_dates else 0,
                    )
                    continue
            out[k] = max(
                elig,
                key=lambda p_i: int(self._sku_sale_dates[p_i].searchsorted(dt_64, side="left"))
                if p_i in self._sku_sale_dates else 0,
            )
        return out

    def metadata_payload(self) -> Dict[str, Any]:
        return {
            "model_family": "v1",
            "title": "Stage 5 No-MLP Full-Pool UPS-CANR — Live Held-Out Test Demo",
            "split_label": "Held-Out Test",
            "split_window_note": f"Held-out Test window {self.cases[0].date.date()} → {self.cases[-1].date.date()}",
            "pipeline_labels": {
                "step2_title": "2. 38D Cand + 11D Cust + 32-Seq",
                "step2_desc": "base_name + 3A fin + 3C mgr + 3E specs",
                "step3_title": "3. Full-Pool UPS-CANR (No MLP)",
                "step3_desc": f"Scores all {self.n_bn} formulations directly",
            },
            "supports_ranking_modes": False,
            "ranking_modes": [],
            "reference_id": self.reference["reference_id"],
            "active_variant": self.active_variant,
            "active_epoch": self.active_epoch,
            "trainable_params": self.trainable_params,
            "device": str(self.dev),
            "num_base_names": self.n_bn,
            "num_features": len(CANDIDATE_38_FEATURE_NAMES),
            "num_customer_features": len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
            "test_baskets_total": len(self.cases),
            "test_date_range": [str(self.cases[0].date.date()), str(self.cases[-1].date.date())],
            "cohort_counts": self.cohort_counts,
            "filter_counts": {k: len(v) for k, v in self.filter_indices.items()},
            "val_metrics": self.active_val_metrics,
        }

    def evaluate_case(self, case_idx: int, *, mode: str = "blended") -> Dict[str, Any]:
        with self.lock:
            c = self.cases[case_idx]
            cid = c.customer
            date = c.date
            day_int = int((date - PURCHASE_DATE_ORIGIN).days)

            # Time-travel seek (demo-only artifact; live production stays at 'today'!)
            self._seek_relationship_tables(date)

            # 1. Strictly-prior native 38D candidate + 11D current customer state + 32-event sequence (< date)
            t_feat0 = time.perf_counter()
            cand_38, cust_11 = self.native_engine.build_split(cid, date)
            cand_38 = _live_candidate_features(cand_38, self.candidate_dtype)
            seqs = self._fast_build_sequence(cid, day_int, frozen_v5_replay=self.frozen_v5_replay)
            seqs["sequence"][..., 3:5] = 0.0  # calendar month permanently zeroed
            t_feat_ms = 1000.0 * (time.perf_counter() - t_feat0)

            hist_df = self.histories[cid]
            prior_df = hist_df.loc[hist_df["purchase_date"] < date].sort_values("purchase_date", ascending=False)

            # 2. Live CustomerTower u_bn + Full-Pool UPS-CANR forward pass (ALL 585 eligible formulations!)
            t_inf0 = time.perf_counter()
            prof_ids = self.profiles.get(cid, {})
            text_vec = torch.from_numpy(self.texts.get(cid, np.zeros(1024, dtype=np.float32))).unsqueeze(0).to(self.dev)
            city_t = torch.tensor([prof_ids.get("city_id", 0)], dtype=torch.long, device=self.dev)
            reg_t = torch.tensor([prof_ids.get("region_id", 0)], dtype=torch.long, device=self.dev)
            ind_t = torch.tensor([prof_ids.get("industry_id", 0)], dtype=torch.long, device=self.dev)
            seq_t = seqs["sequence"].to(self.dev)
            mask_t = seqs["attention_mask"].to(self.dev)
            seq_pid_t = seqs["seq_product_ids"].to(self.dev)
            seq_dt_t = seqs["seq_purchase_dates"].to(self.dev)
            pday_t = torch.tensor([float(day_int)], dtype=torch.float32, device=self.dev)
            allowed_t = torch.from_numpy(c.allowed).unsqueeze(0).to(self.dev)
            with torch.no_grad():
                if isinstance(self.ranker, SharedUVStage5Ranker):
                    c_in = {
                        "text_embedding": text_vec,
                        "city_id": city_t,
                        "region_id": reg_t,
                        "industry_id": ind_t,
                        "tabular_features": torch.zeros((1, 0), dtype=torch.float32, device=self.dev),
                        "sequence": seq_t,
                        "attention_mask": mask_t,
                        "seq_product_ids": seq_pid_t,
                        "seq_purchase_dates": seq_dt_t,
                    }
                    u_bn = self.tower.encode_purchase(c_in, self.v_bn).squeeze(1)  # (1, 128) live!
                    inp = {
                        "customer_vector": u_bn,
                        "customer_features": torch.from_numpy(cust_11).unsqueeze(0).to(self.dev),
                        "catalog_product_vectors": self.v_bn,
                        "sequence": seq_t,
                        "attention_mask": mask_t,
                        "seq_product_ids": seq_pid_t,
                        "seq_purchase_dates": seq_dt_t,
                        "prediction_day": pday_t,
                        "candidate_features": torch.from_numpy(cand_38).unsqueeze(0).to(self.dev),
                        "candidate_mask": allowed_t,
                    }
                else:
                    inp = {
                        "text_embedding": text_vec,
                        "city_id": city_t,
                        "region_id": reg_t,
                        "industry_id": ind_t,
                        "customer_features": torch.from_numpy(cust_11).unsqueeze(0).to(self.dev),
                        "sequence": seq_t,
                        "attention_mask": mask_t,
                        "seq_product_ids": seq_pid_t,
                        "seq_category_ids": seqs["seq_category_ids"].to(self.dev),
                        "seq_subcategory_ids": seqs["seq_subcategory_ids"].to(self.dev),
                        "seq_business_line_ids": seqs["seq_business_line_ids"].to(self.dev),
                        "seq_purchase_dates": seq_dt_t,
                        "prediction_day": pday_t,
                        "candidate_features": torch.from_numpy(cand_38).unsqueeze(0).to(self.dev),
                        "candidate_vectors": self.v_bn.unsqueeze(0),
                        "candidate_mask": allowed_t,
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
            resolved_sku_idx = self._fast_resolve_skus(top10, cid, date, c.sku_allowed)
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
            equip_domains = sorted(parse_company_equipment_domains(desc_str, ind_str, frozen_v5=self.frozen_v5_replay))

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
                    "prepaid_or_cash_pct": round(100.0 * float(cust_11[U_IDX["cust_prepaid_or_cash_ratio"]]), 1),
                    "avg_deferral_days": round(90.0 * float(cust_11[U_IDX["cust_avg_deferral_days_norm"]]), 1),
                    "actual_vs_contract_ratio": round(2.0 * float(cust_11[U_IDX["cust_actual_vs_contract_pay_ratio"]]), 2),
                    "overdue_unpaid_pct": round(100.0 * float(cust_11[U_IDX["cust_overdue_unpaid_ratio"]]), 1),
                    "mgr_changed_recently": bool(cust_11[U_IDX["mgr_changed_recently"]] > 0.5),
                },
            }

            # 5. Build Top-10 predictions list with signal drivers & hit highlighting
            predictions = []
            rel_flags = []
            new_hits_10 = 0
            for rank_0, bn_i in enumerate(top10.tolist()):
                meta = self.bn_meta[bn_i]
                f_row = cand_38[bn_i]
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

                # Build interpretable drivers from the exact 38 causal candidate features
                drivers: List[str] = []
                if f_row[C_IDX["has_bought_base_sku"]] > 0.5:
                    approx_orders = max(1, int(round(math.expm1(float(f_row[C_IDX["user_base_sku_order_count_norm"]]) * math.log1p(50.0)))))
                    approx_days = int(round(math.expm1(float(f_row[C_IDX["days_since_last_base_sku_buy_norm"]]) * math.log1p(365.0))))
                    drivers.append(f"Repeat ({approx_orders}× · {approx_days}d ago)")
                    urg = float(f_row[C_IDX["base_sku_urgency"]]) * 5.0
                    if urg >= 0.6:
                        drivers.append(f"Due ratio {urg:.1f}×")
                else:
                    drivers.append("New to customer")
                if f_row[C_IDX["base_name_recent3_lift"]] > 0.15:
                    drivers.append(f"Seq lift +{f_row[C_IDX['base_name_recent3_lift']]:.2f}")
                if f_row[C_IDX["mgr_base_name_share_90d"]] > 0.02:
                    drivers.append(f"Mgr 90d {100*f_row[C_IDX['mgr_base_name_share_90d']]:.1f}%")
                if f_row[C_IDX["base_name_industry_affinity"]] > 0.03:
                    drivers.append(f"Industry {100*f_row[C_IDX['base_name_industry_affinity']]:.1f}%")
                if f_row[C_IDX["app_matches_company_equipment"]] > 0.5:
                    drivers.append("Equipment match")
                if f_row[C_IDX["user_viscosity_order_share"]] > 0.0 and is_new_candidate:
                    drivers.append(f"Viscosity {100*f_row[C_IDX['user_viscosity_order_share']]:.0f}%")

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
            for u_bn, is_new in c.unmodeled_novelty:
                gt_items.append(
                    {
                        "base_name": u_bn,
                        "brand": "",
                        "category": "Unmodeled (>2026-02-18)",
                        "is_new": is_new,
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
    color-scheme: light;
    --bg: #f4f7fb;
    --panel: #ffffff;
    --panel-2: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.10);
    --hit-bg: rgba(16, 185, 129, 0.12);
    --hit-border: #10b981;
    --hit-text: #059669;
    --new-bg: rgba(245, 158, 11, 0.14);
    --new-border: #f59e0b;
    --new-text: #b45309;
    --miss-bg: rgba(148, 163, 184, 0.10);
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
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }
  .header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }
  .header .sub { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  select, button, input.case-input {
    font-family: var(--sans); font-size: 13px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    padding: 8px 12px; cursor: pointer;
  }
  input.case-input {
    width: 92px; font-family: var(--mono); cursor: text; padding: 8px 10px;
  }
  button.secondary { padding: 8px 11px; font-weight: 600; }
  button.secondary:hover { border-color: var(--accent); }
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
  .card-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #334155; }
  .pill {
    display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
    white-space: nowrap;
  }
  .pill.hit { background: var(--hit-bg); border-color: var(--hit-border); color: var(--hit-text); }
  .pill.newhit { background: var(--new-bg); border-color: var(--new-border); color: var(--new-text); }
  .pill.cold { background: rgba(124, 58, 237, 0.12); border-color: #7c3aed; color: #6d28d9; }
  .pill.ret { background: var(--accent-soft); border-color: #2563eb; color: #1d4ed8; }
  .pill.newtag { background: rgba(245, 158, 11, 0.14); border-color: rgba(245, 158, 11, 0.50); color: #b45309; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th {
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); padding: 7px 8px; border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr.row-hit { background: var(--hit-bg); }
  tr.row-hit td:first-child { border-left: 3px solid var(--hit-border); }
  tr.row-newhit { background: var(--new-bg); }
  tr.row-newhit td:first-child { border-left: 3px solid var(--new-border); }
  .rank-num { font-family: var(--mono); font-weight: 700; font-size: 13px; width: 32px; }
  .prod-name { font-weight: 600; color: var(--text); }
  .prod-meta { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .drivers { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .drv {
    font-size: 10.5px; font-family: var(--mono); padding: 1px 6px; border-radius: 4px;
    background: var(--panel-2); border: 1px solid var(--border); color: #475569;
  }
  .prob-wrap { width: 88px; }
  .prob-txt { font-family: var(--mono); font-size: 11.5px; font-weight: 600; }
  .prob-bar { height: 5px; background: #e2e8f0; border-radius: 3px; overflow: hidden; margin-top: 3px; }
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
      <h1 id="page-title">Live Product Recommendation Demo</h1>
      <div class="sub" id="model-subtitle">Loading model metadata...</div>
    </div>
    <div class="controls">
      <select id="ranking-mode" title="Ranking head / calibration mode" style="display:none;"></select>
      <select id="cohort-filter" title="Filter evaluation baskets">
        <option value="any">Any Basket</option>
        <option value="returning">Returning Customers</option>
        <option value="first_observed">First-Observed (Cold-Start)</option>
        <option value="has_new">Returning w/ True New Product</option>
        <option value="multi_item">Multi-Product Baskets (≥2)</option>
      </select>
      <input type="number" min="0" id="case-idx-input" class="case-input" placeholder="Case #" title="Jump to specific case index" />
      <button class="secondary" id="btn-go" title="Load specific case index">Go</button>
      <button class="primary" id="btn-random">🎲 Pick Random Case</button>
    </div>
  </div>

  <div class="pipeline-strip">
    <div class="pipe-step">
      <div class="k">1. Strictly Causal Cutoff</div>
      <div class="v" id="p-date">—</div>
      <div class="d">Only orders &amp; settlements &lt; date</div>
    </div>
    <div class="pipe-step">
      <div class="k" id="p-step2-title">2. Causal Features &amp; Sequence</div>
      <div class="v" id="p-feat-ms">—</div>
      <div class="d" id="p-step2-desc">—</div>
    </div>
    <div class="pipe-step">
      <div class="k" id="p-step3-title">3. Full-Pool Neural Ranker</div>
      <div class="v" id="p-inf-ms">—</div>
      <div class="d" id="p-pool">—</div>
    </div>
    <div class="pipe-step">
      <div class="k">4. Causal Pack-Size Resolver</div>
      <div class="v" id="p-res-ms">—</div>
      <div class="d">Maps Top-10 base_name → eligible SKU</div>
    </div>
  </div>

  <div class="metrics-bar">
    <div class="metric-card">
      <div class="label" id="lbl-r10">Case Recall@10</div>
      <div class="val good" id="m-r10">—</div>
      <div class="hint" id="m-hits">—</div>
    </div>
    <div class="metric-card">
      <div class="label" id="lbl-ndcg">Case NDCG@10</div>
      <div class="val" id="m-ndcg">—</div>
      <div class="hint" id="m-ndcg-hint">Position-weighted ranking quality</div>
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
let activeMeta = null;
let currentCaseIdx = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function activeMode() {
  const sel = document.getElementById('ranking-mode');
  return (sel && sel.style.display !== 'none' && sel.value) ? sel.value : 'blended';
}

async function loadMeta() {
  const res = await fetch('/api/meta');
  const m = await res.json();
  activeMeta = m;

  if (m.title) {
    document.title = m.title;
    document.getElementById('page-title').textContent = m.title;
  }
  const paramStr = m.per_request_params
    ? `${(m.per_request_params/1e3).toFixed(0)}K per-req / ${(m.trainable_params/1e6).toFixed(2)}M total`
    : `${(m.trainable_params/1e3).toFixed(0)}K params`;
  const splitNote = m.split_window_note || `${m.test_date_range[0]} → ${m.test_date_range[1]}`;
  document.getElementById('model-subtitle').textContent =
    `Model: ${m.active_variant} (ep ${m.active_epoch}, ${paramStr}, ${m.device}) · ` +
    `${m.num_base_names.toLocaleString()} canonical base_names · ${splitNote} (${m.test_baskets_total.toLocaleString()} baskets)`;

  if (m.pipeline_labels) {
    document.getElementById('p-step2-title').textContent = m.pipeline_labels.step2_title;
    document.getElementById('p-step2-desc').textContent = m.pipeline_labels.step2_desc;
    document.getElementById('p-step3-title').textContent = m.pipeline_labels.step3_title;
    document.getElementById('p-pool').textContent = m.pipeline_labels.step3_desc;
  }

  const modeSel = document.getElementById('ranking-mode');
  if (m.supports_ranking_modes && m.ranking_modes && m.ranking_modes.length) {
    modeSel.innerHTML = m.ranking_modes.map(o => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join('');
    modeSel.style.display = '';
  } else {
    modeSel.style.display = 'none';
  }

  const sel = document.getElementById('cohort-filter');
  const splitPrefix = m.model_family === 'v2_full' ? 'Validation' : 'Test';
  sel.options[0].text = `Any ${splitPrefix} Basket (${m.filter_counts.any.toLocaleString()})`;
  sel.options[1].text = `Returning Customers (${m.filter_counts.returning.toLocaleString()})`;
  sel.options[2].text = `First-Observed / Cold-Start (${m.filter_counts.first_observed.toLocaleString()})`;
  sel.options[3].text = `Returning w/ True New Product (${m.filter_counts.has_new.toLocaleString()})`;
  sel.options[4].text = `Multi-Product Baskets ≥2 (${m.filter_counts.multi_item.toLocaleString()})`;

  document.getElementById('case-idx-input').max = String(Math.max(0, m.test_baskets_total - 1));

  const vm = m.val_metrics || {};
  if (vm['bn_recall@10'] !== undefined) {
    const pureTxt = vm['pure_unseen_hits@10'] !== undefined
      ? ` · <strong>Pure Unseen@10 ${vm['pure_unseen_hits@10'].toLocaleString()}/${vm['true_new_bn_targets'].toLocaleString()} (${(100*vm['pure_unseen_recall@10']).toFixed(1)}%)</strong>`
      : '';
    const nValStr = m.model_family === 'v2_full' ? m.test_baskets_total.toLocaleString() : '6,480';
    document.getElementById('benchmark-footer').innerHTML =
      `Validation (${nValStr} baskets): <strong>Recall@10 ${(100*vm['bn_recall@10']).toFixed(2)}%</strong> · ` +
      `<strong>NDCG@10 ${vm['bn_ndcg@10'].toFixed(5)}</strong> · ` +
      `<strong>New@10 ${vm['true_new_bn_hits@10'].toLocaleString()}/${vm['true_new_bn_targets'].toLocaleString()}</strong>${pureTxt}`;
  }
}

async function fetchCase(url, { countSession = true } = {}) {
  const btn = document.getElementById('btn-random');
  btn.disabled = true;
  btn.textContent = 'Computing live...';
  try {
    const res = await fetch(url);
    const d = await res.json();
    renderCase(d, { countSession });
  } finally {
    btn.disabled = false;
    btn.textContent = '🎲 Pick Random Case';
  }
}

async function loadRandomCase() {
  const flt = document.getElementById('cohort-filter').value;
  const mode = activeMode();
  await fetchCase(`/api/case?filter=${encodeURIComponent(flt)}&mode=${encodeURIComponent(mode)}`, { countSession: true });
}

async function loadCaseByIdx(idx, { countSession = true } = {}) {
  const mode = activeMode();
  await fetchCase(`/api/case?idx=${encodeURIComponent(idx)}&mode=${encodeURIComponent(mode)}`, { countSession });
}

function renderCase(d, { countSession = true } = {}) {
  currentCaseIdx = d.case_idx;
  document.getElementById('case-idx-input').value = String(d.case_idx);

  const isPureUnseen = d.ranking_mode === 'pure_unseen';
  if (countSession) {
    session.count += 1;
    session.sumR10 += d.case_metrics.recall_at_10;
    session.sumNdcg += d.case_metrics.ndcg_at_10;
    session.hits += d.case_metrics.hits_at_10;
    session.targets += d.case_metrics.total_targets;
    session.newHits += d.case_metrics.new_hits_at_10;
    session.newTargets += d.case_metrics.total_new_targets;
  }

  document.getElementById('p-date').textContent = `${d.prediction_date} (Case #${d.case_idx})`;
  document.getElementById('p-feat-ms').textContent = `${d.timings_ms.features_and_sequence.toFixed(2)} ms`;
  document.getElementById('p-inf-ms').textContent = `${d.timings_ms.fullpool_ups_canr.toFixed(2)} ms`;
  const abNote = (d.affine_blend && d.ranking_mode === 'blended')
    ? ` · OOF a=${d.affine_blend.a}, b=${d.affine_blend.b}`
    : (isPureUnseen ? ' · Pure Unseen (repeats masked)' : (d.ranking_mode === 'main_only' ? ' · Main Head only' : ''));
  document.getElementById('p-pool').textContent =
    `${d.eligible_formulations.toLocaleString()} eligible / ${d.total_formulations.toLocaleString()} formulations${abNote}`;
  document.getElementById('p-res-ms').textContent = `${d.timings_ms.pack_resolution.toFixed(2)} ms`;

  const cm = d.case_metrics;
  document.getElementById('lbl-r10').textContent = isPureUnseen ? 'Pure Unseen Recall@10' : 'Case Recall@10';
  document.getElementById('lbl-ndcg').textContent = isPureUnseen ? 'Pure Unseen NDCG@10' : 'Case NDCG@10';
  document.getElementById('m-r10').textContent = cm.total_targets > 0 ? `${(100 * cm.recall_at_10).toFixed(1)}%` : 'N/A';
  document.getElementById('m-hits').textContent = isPureUnseen
    ? `${cm.hits_at_10} of ${cm.total_targets} truly new formulations in Top-10`
    : `${cm.hits_at_10} of ${cm.total_targets} bought formulations in Top-10`;
  document.getElementById('m-ndcg').textContent = cm.total_targets > 0 ? cm.ndcg_at_10.toFixed(3) : 'N/A';
  document.getElementById('m-new').textContent = `${cm.new_hits_at_10} / ${cm.total_new_targets}`;

  const c = d.customer;
  document.getElementById('m-cohort').innerHTML = c.is_first_observed
    ? '<span class="pill cold">First-Observed (Cold-Start)</span>'
    : '<span class="pill ret">Returning Customer</span>';
  document.getElementById('m-prior-summary').textContent = c.is_first_observed
    ? '0 prior purchases — profile & market only'
    : `${c.prior_order_days} prior orders · ${c.prior_distinct_formulations} formulations`;

  document.getElementById('m-sess-r10').textContent = session.count > 0
    ? `${(100 * session.sumR10 / session.count).toFixed(1)}%`
    : '—';
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
    const skuMark = p.resolved_sku_hit ? ' <span style="color:var(--hit-text);font-weight:700;">✓</span>' : '';
    return `<tr class="${cls}">
      <td class="rank-num">#${p.rank}</td>
      <td>
        <div class="prod-name">${esc(p.base_name)}</div>
        <div class="prod-meta">${esc(p.brand)} · ${esc(p.category)}</div>
        <div class="drivers">${drvHtml}</div>
      </td>
      <td style="font-family:var(--mono);font-size:11.5px;color:#cbd5e1;">${esc(p.resolved_sku)}${skuMark}</td>
      <td class="prob-wrap">
        <div class="prob-txt">${(100*p.probability).toFixed(1)}%</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${Math.min(100, Math.round(100*p.probability))}%"></div></div>
      </td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  // Ground truth basket
  const basketTotal = cm.basket_total_targets !== undefined ? cm.basket_total_targets : cm.total_targets;
  document.getElementById('gt-date').textContent = d.prediction_date;
  document.getElementById('gt-count').textContent = `${basketTotal} formulations (${cm.total_new_targets} new)`;
  document.getElementById('gt-list').innerHTML = d.ground_truth_basket.map(g => {
    const cls = g.hit_top10 ? (g.is_new ? 'newhit' : 'hit') : '';
    const rkTxt = g.predicted_rank !== null
      ? (g.hit_top10 ? `Rank #${g.predicted_rank} ✓` : `Rank #${g.predicted_rank}`)
      : (isPureUnseen && !g.is_new
          ? 'Masked (Repeat)'
          : (g.is_modeled ? 'Ineligible on date' : 'Outside modeled catalogue'));
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
    hbody.innerHTML = '<tr><td colspan="3" style="color:var(--muted);padding:14px;">First observed order — zero prior purchase history. Predictions come from company profile, industry/regional affinity, equipment match &amp; rolling market signals.</td></tr>';
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
document.getElementById('ranking-mode').addEventListener('change', () => {
  if (currentCaseIdx !== null) {
    loadCaseByIdx(currentCaseIdx, { countSession: false });
  } else {
    loadRandomCase();
  }
});
document.getElementById('btn-go').addEventListener('click', () => {
  const val = parseInt(document.getElementById('case-idx-input').value, 10);
  if (!Number.isNaN(val)) loadCaseByIdx(val, { countSession: true });
});
document.getElementById('case-idx-input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') {
    const val = parseInt(ev.target.value, 10);
    if (!Number.isNaN(val)) loadCaseByIdx(val, { countSession: true });
  }
});

loadMeta().then(loadRandomCase);
</script>
</body>
</html>
"""


def create_server(engine: Any, host: str, port: int) -> ThreadingHTTPServer:
    n_cases = int(getattr(engine, "n_cases", 0) or len(engine.cases))

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
                mode = qs.get("mode", ["blended"])[0]
                if "idx" in qs:
                    idx = max(0, min(n_cases - 1, int(qs["idx"][0])))
                else:
                    flt = qs.get("filter", ["any"])[0]
                    pool_ids = engine.filter_indices.get(flt, engine.filter_indices["any"])
                    idx = random.choice(pool_ids)
                self._send_json(engine.evaluate_case(idx, mode=mode))
            else:
                self._send_json({"error": "Not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), DemoHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        choices=("v2_full", "v1"),
        default="v2_full",
        help="Model engine to serve: 'v2_full' (default, canonical Wave2BE2ERanker + Option A on Validation) or 'v1'.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=("discovery", "ndcg"),
        default="discovery",
        help="v2_full checkpoint selection: 'discovery' (Ep 6 primary cal-Discovery==cal-Unseen) or 'ndcg' (Ep 10 companion).",
    )
    parser.add_argument(
        "--allow-heldout-test",
        action="store_true",
        help="Required only for legacy --model v1 live demo over held-out Test baskets (>= 2026-02-19).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replay-frozen-features", action="store_true", help="Legacy v1: reproduce original v5 validation feature recipe.")
    args = parser.parse_args()

    if args.model == "v2_full":
        from recommender.experiments.v2_full.demo_engine import V2FullLiveDemoEngine

        engine: Any = V2FullLiveDemoEngine(
            device=args.device,
            checkpoint_kind=args.checkpoint_kind,
            checkpoint_override=args.checkpoint,
        )
    else:
        engine = Stage5LiveDemoEngine(
            allow_heldout_test=args.allow_heldout_test,
            device=args.device,
            variant_override=args.variant,
            checkpoint_override=args.checkpoint,
            frozen_v5_replay=args.replay_frozen_features,
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
