"""Live interactive demo engine for the canonical `v2_full` E2E Option A model.

Serves the promoted single-stage `Wave2BE2ERanker` (`v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6`,
`cand_42 + CUST_7 + SEQ_5`, Pure-Q `Q = v_bn`, Full-42 cross-attention bias, clean separated
`OptionAUnseenHead`, and 2-parameter customer-grouped OOF affine blend) over the `16,623` strictly
out-of-time `v2_full` Validation baskets (`2026-02-01 .. 2026-06-30`).

Strictly zero held-out Test (`>= 2026-07-01`) access.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import file_digest
from recommender.experiments.roadmap_v2.dated_baskets import validate_dated_split
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
)
from recommender.experiments.roadmap_v2.parsed_spec_features import parse_company_equipment_domains
from recommender.experiments.static_enrichment.metadata import (
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.sequences import PURCHASE_DATE_ORIGIN

from .config import (
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    RAW_ITEMS_PATH,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
    V2_FULL_REFERENCE_PATH,
    load_v2_development_purchases,
)
from .extra_features import (
    CAND_42_FEATURE_NAMES,
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    NUM_EXTRA_CAND_FEATURES,
    SEQ_5_ACTIVE_INDICES,
    assemble_cand_42,
    prune_cust_7,
)
from .wave2b_models import Wave2BE2ERanker, _customer_fold_assignment


C42_IDX: Dict[str, int] = {name: i for i, name in enumerate(CAND_42_FEATURE_NAMES)}
U11_IDX: Dict[str, int] = {name: i for i, name in enumerate(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)}

SUPPORTED_RANKING_MODES: Tuple[str, ...] = ("blended", "pure_unseen", "main_only")


def _verify_v2_checkpoint(entry: Mapping[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing v2_full reference {label}: {path}")
    expected_sha = entry.get("sha256")
    if expected_sha:
        actual_sha = file_digest(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Checksum mismatch for v2_full reference {label} ({path}): "
                f"expected {expected_sha}, got {actual_sha}"
            )
    return path


def _locate_verified_val_extra_cand_5(
    prepared_root: Path,
    n_val: int,
    n_bn: int,
) -> Tuple[Path, Dict[str, Any]]:
    """Find and SHA-256-verify the sealed `val_extra_cand_5.npy` memmap without loading train_meta."""
    extra_root = prepared_root / "extra_cand_5"
    if not extra_root.is_dir():
        raise FileNotFoundError(
            f"Missing extra_cand_5 directory at {extra_root}. "
            "Run `PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.extra_features` first."
        )
    candidates = sorted(
        extra_root.glob("*/extra_cand_5_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for mpath in candidates:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        val_shape = manifest.get("shapes", {}).get("val")
        if val_shape != [n_val, n_bn, NUM_EXTRA_CAND_FEATURES]:
            continue
        if pd.Timestamp(manifest.get("train_cutoff", "")) != TRAIN_CUTOFF_V2:
            continue
        if pd.Timestamp(manifest.get("test_start", "")) != TEST_START_V2:
            continue
        if manifest.get("heldout_access", True) is not False:
            continue
        va_path = mpath.parent / "val_extra_cand_5.npy"
        if not va_path.is_file():
            continue
        expected_sha = manifest.get("output_sha256", {}).get("val")
        if expected_sha and file_digest(va_path) != expected_sha:
            continue
        return va_path, manifest
    raise FileNotFoundError(
        f"No verified extra_cand_5 validation memmap with shape [{n_val}, {n_bn}, {NUM_EXTRA_CAND_FEATURES}] "
        f"found under {extra_root}."
    )


def _v2_filter_indices(
    first_observed: np.ndarray,
    target_bn: np.ndarray,
    new_target_bn: np.ndarray,
) -> Dict[str, List[int]]:
    n_rows = len(first_observed)
    returning = ~first_observed
    has_new = returning & new_target_bn.any(axis=1)
    multi_item = target_bn.sum(axis=1) >= 2
    return {
        "any": list(range(n_rows)),
        "returning": np.flatnonzero(returning).tolist(),
        "first_observed": np.flatnonzero(first_observed).tolist(),
        "has_new": np.flatnonzero(has_new).tolist(),
        "multi_item": np.flatnonzero(multi_item).tolist(),
    }


def _resolve_checkpoint_kind(
    reference: Mapping[str, Any],
    checkpoint_kind: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    kind = str(checkpoint_kind).strip().lower()
    if kind in ("discovery", "primary", "unseen"):
        return (
            reference["checkpoints"]["ranker"],
            reference["validation_metrics_primary_cal_discovery"],
        )
    if kind == "ndcg":
        return (
            reference["checkpoints"]["ndcg_ranker"],
            reference["validation_metrics_companion_cal_ndcg"],
        )
    raise ValueError(f"Unsupported checkpoint_kind={checkpoint_kind!r}; expected 'discovery' or 'ndcg'.")


class V2FullLiveDemoEngine:
    """Loads the canonical `v2_full` Wave2BE2ERanker + Option A head and serves live Validation cases."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        reference_path: Path = V2_FULL_REFERENCE_PATH,
        prepared_root: Path = PREPARED_ROOT,
        checkpoint_kind: str = "discovery",
        checkpoint_override: Optional[Path] = None,
    ):
        t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.dev = torch.device(device) if device else torch.device("cpu")
        self.reference_path = Path(reference_path)
        self.reference = json.loads(self.reference_path.read_text(encoding="utf-8"))
        if self.reference.get("heldout_access", True) is not False:
            raise ValueError("v2_full reference must preserve heldout_access=False.")

        self.checkpoint_kind = str(checkpoint_kind).strip().lower()
        ckpt_entry, ref_val_metrics = _resolve_checkpoint_kind(self.reference, self.checkpoint_kind)
        if checkpoint_override is not None:
            ckpt_path = Path(checkpoint_override)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint override not found: {ckpt_path}")
        else:
            ckpt_path = _verify_v2_checkpoint(ckpt_entry, f"ranker ({self.checkpoint_kind})")

        print(f"[demo-v2] Loading v2_full prepared validation metadata & catalogue (< {TEST_START_V2.date()})...", flush=True)
        prepared_root = Path(prepared_root)
        self.shared_meta = torch.load(prepared_root / "shared_meta.pt", map_location="cpu", weights_only=False)
        if (
            pd.Timestamp(self.shared_meta["train_cutoff"]) != TRAIN_CUTOFF_V2
            or pd.Timestamp(self.shared_meta["test_start"]) != TEST_START_V2
        ):
            raise ValueError("Prepared v2_full shared_meta split boundaries mismatch.")

        self.val_meta = torch.load(prepared_root / "val_meta.pt", map_location="cpu", weights_only=False)
        validate_dated_split(self.val_meta, "val", train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2)
        if (
            tuple(self.val_meta["feature_names"]) != CANDIDATE_38_FEATURE_NAMES
            or tuple(self.val_meta["customer_feature_names"]) != CUSTOMER_DYNAMIC_11_FEATURE_NAMES
        ):
            raise ValueError("v2_full val_meta feature schema mismatch.")

        self.universe = self.shared_meta["universe"]
        self.product_ids: List[str] = list(self.shared_meta["product_ids"])
        self.reps = self.shared_meta["reps"]
        self.vocabs = self.shared_meta["vocabs"]
        self.enrichment = self.shared_meta["enrichment"]
        self.n_bn = int(self.universe["num_base_names"])
        self.n_cases = len(self.val_meta["customer_ids"])

        # Memory-mapped float16 validation feature arrays (zero RAM duplication, <0.2ms per case)
        cand38_path = Path(self.val_meta["cand_38_path"])
        if not cand38_path.is_file():
            cand38_path = prepared_root / "val_cand_38.npy"
        self.val_cand38_mm = np.load(cand38_path, mmap_mode="r")
        if self.val_cand38_mm.shape != (self.n_cases, self.n_bn, len(CANDIDATE_38_FEATURE_NAMES)):
            raise ValueError(f"Unexpected val_cand_38 shape: {self.val_cand38_mm.shape}")

        va_extra_path, _ = _locate_verified_val_extra_cand_5(prepared_root, self.n_cases, self.n_bn)
        self.val_extra5_mm = np.load(va_extra_path, mmap_mode="r")
        if self.val_extra5_mm.shape != (self.n_cases, self.n_bn, NUM_EXTRA_CAND_FEATURES):
            raise ValueError(f"Unexpected val_extra_cand_5 shape: {self.val_extra5_mm.shape}")

        # Strictly development purchases (< 2026-07-01), catalogue & companies for human-readable display + pack resolver
        self.dev_purchases = load_v2_development_purchases()
        if bool(self.dev_purchases["purchase_date"].ge(TEST_START_V2).any()):
            raise ValueError("Held-out test event detected in development purchases!")
        self.catalogue = load_enriched_catalogue(RAW_ITEMS_PATH, descriptions_path=None)
        self.companies = load_enriched_companies(RAW_CUSTOMERS_PATH, purchases=self.dev_purchases, minimum_purchases=0)
        self.company_map = self.companies.drop_duplicates("customer_id").set_index("customer_id").to_dict("index")
        self.cat_indexed = self.catalogue.drop_duplicates("product_id").set_index("product_id")
        self.histories = dict(tuple(self.dev_purchases.groupby("customer_id", sort=False)))

        # Per-formulation display metadata
        self.bn_meta: List[Dict[str, Any]] = []
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

        # Fast pack-size resolver timelines (strictly < date, <0.2 ms/req)
        self._init_fast_resolver_indices()

        # Pre-index validation case metadata & filters
        self.customer_ids: List[str] = [str(c) for c in self.val_meta["customer_ids"]]
        self.pred_days_np = self.val_meta["prediction_days"].cpu().numpy().astype(np.int64)
        self.pred_dates: List[pd.Timestamp] = [
            PURCHASE_DATE_ORIGIN + pd.Timedelta(days=int(d)) for d in self.pred_days_np.tolist()
        ]
        self.first_obs_np = self.val_meta["first_observed"].bool().cpu().numpy()
        self.target_np = self.val_meta["target_bn"].bool().cpu().numpy()
        self.new_target_np = self.val_meta["new_target_bn"].bool().cpu().numpy()
        self.allowed_np = self.val_meta["allowed_bn"].bool().cpu().numpy()
        self.unseen_np = self.val_meta["unseen_bn"].bool().cpu().numpy()
        self.sku_allowed_np = self.val_meta["sku_val"]["allowed"].bool().cpu().numpy()
        self.sku_target_np = self.val_meta["sku_val"]["target"].bool().cpu().numpy()

        self.filter_indices = _v2_filter_indices(self.first_obs_np, self.target_np, self.new_target_np)
        self.cohort_counts = {
            "returning_baskets": int((~self.first_obs_np).sum()),
            "first_observed_baskets": int(self.first_obs_np.sum()),
            "total_bn_targets": int(self.target_np.sum()),
            "true_new_bn_targets": int(self.new_target_np.sum()),
        }

        # Deterministic customer-grouped OOF affine calibration (exact zero-leakage validation blend)
        cal = ref_val_metrics.get("affine_oof_calibration", {})
        self.oof_n_folds = int(cal.get("n_folds", 2))
        self.oof_folds = _customer_fold_assignment(self.customer_ids, n_folds=self.oof_n_folds)
        self.fold_ab: Dict[int, Tuple[float, float]] = {
            int(f["fold"]): (float(f["a"]), float(f["b"])) for f in cal.get("folds", [])
        }
        dep_fit = cal.get("deployment_full_fit", {})
        self.full_fit_ab: Tuple[float, float] = (
            float(dep_fit.get("a", 1.0)),
            float(dep_fit.get("b", 0.0)),
        )

        # Load Wave2BE2ERanker & precompute cached v_bn (1,141 x 128)
        print("[demo-v2] Loading Wave2BE2ERanker + OptionAUnseenHead & caching 1,141 v_bn vectors...", flush=True)
        arch = dict(self.reference["active_architecture"])
        self.active_variant = str(arch.get("arm", "w2c_exp1_clean_sep_w10"))
        self.active_ckpt_path = str(ckpt_path)
        n_demand = int(self.shared_meta["prod_inputs"]["demand_features"].shape[1])

        self.ranker = Wave2BE2ERanker(
            product_attr_vocab_sizes={k: len(v) for k, v in self.enrichment["product_vocabs"].items()},
            num_cities=len(self.vocabs["city_vocab"]) + 1,
            num_regions=len(self.vocabs["region_vocab"]) + 1,
            num_industries=len(self.vocabs["industry_vocab"]) + 1,
            num_demand_features=n_demand,
            num_candidate_features=int(arch.get("num_candidate_features", NUM_CAND_42_FEATURES)),
            num_customer_features=int(arch.get("num_customer_features", NUM_CUST_7_FEATURES)),
            num_pairwise_bias_features=int(arch.get("num_pairwise_bias_features", NUM_CAND_42_FEATURES)),
            active_seq_stat_indices=tuple(arch.get("active_seq_stat_indices", SEQ_5_ACTIVE_INDICES)),
            pure_v_query=bool(arch.get("pure_v_query", True)),
            option_a_unseen_head=bool(arch.get("option_a_unseen_head", True)),
            unseen_dropout_cold_prob=0.0,
            unseen_attn_aux_enabled=False,
            d_model=int(arch["d_model"]),
            num_heads=int(arch["num_heads"]),
            memory_ffn_dim=int(arch["memory_ffn_dim"]),
            memory_layers=int(arch.get("memory_layers", 1)),
            mlp_widths=tuple(int(w) for w in arch["mlp_widths"]),
            dropout=float(arch.get("dropout", 0.12)),
            aux_60d_enabled=False,
        ).to(self.dev)

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ranker.load_state_dict(ck["model_state_dict"], strict=True)
        self.ranker.eval()

        self.active_epoch = int(ck.get("epoch", ref_val_metrics.get("epoch", 0)))
        self.active_val_metrics = {
            "bn_recall@10": float(ref_val_metrics.get("bn_recall@10", 0.0)),
            "bn_ndcg@10": float(ref_val_metrics.get("bn_ndcg@10", 0.0)),
            "true_new_bn_hits@10": int(ref_val_metrics.get("true_new_bn_hits@10", 0)),
            "true_new_bn_targets": int(self.reference.get("validation_true_new_bn_targets", self.cohort_counts["true_new_bn_targets"])),
            "pure_unseen_hits@10": int(ref_val_metrics.get("pure_unseen_hits@10", 0)),
            "pure_unseen_recall@10": float(ref_val_metrics.get("pure_unseen_recall@10", 0.0)),
            "sku_recall@10": float(ref_val_metrics.get("sku_recall@10", 0.0)),
            "sku_ndcg@10": float(ref_val_metrics.get("sku_ndcg@10", 0.0)),
        }
        self.trainable_params = int(arch.get("inference_params", sum(p.numel() for p in self.ranker.parameters())))
        self.per_request_params = int(
            arch.get(
                "per_request_params",
                sum(p.numel() for k, p in self.ranker.named_parameters() if not k.startswith("product_tower.")),
            )
        )

        prod_inputs_dev = {
            "text_embedding": self.shared_meta["prod_inputs"]["text_embedding"].float().to(self.dev),
            "demand_features": self.shared_meta["prod_inputs"]["demand_features"].float().to(self.dev),
            "attr_ids": {k: v.to(self.dev) for k, v in self.shared_meta["prod_inputs"]["attr_ids"].items()},
        }
        with torch.no_grad():
            self.v_bn = self.ranker.precompute_product_vectors(prod_inputs_dev).detach()

        # Warmup once so live timings reflect steady-state serving latency
        self.evaluate_case(0)
        print(
            f"[demo-v2] Ready in {time.perf_counter() - t0:.1f}s | {self.n_cases:,} v2_full validation baskets "
            f"({self.pred_dates[0].date()}..{self.pred_dates[-1].date()}, held-out >= {TEST_START_V2.date()} untouched) | "
            f"model={self.active_variant} (ep {self.active_epoch}, {self.per_request_params:,} per-req params, dev={self.dev})",
            flush=True,
        )

    def _init_fast_resolver_indices(self) -> None:
        pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        self._pid_cont = [
            str(self.cat_indexed.at[p, "container_type"]).strip().upper()
            if p in self.cat_indexed.index and pd.notna(self.cat_indexed.at[p, "container_type"])
            else ""
            for p in self.product_ids
        ]
        df_sorted = self.dev_purchases[["customer_id", "purchase_date", "product_id"]].sort_values(
            "purchase_date", kind="stable"
        )
        self._sku_sale_dates: Dict[int, np.ndarray] = {}
        for pid, grp in df_sorted.groupby("product_id", sort=False):
            if pid in pid_to_idx:
                self._sku_sale_dates[pid_to_idx[pid]] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
        self._cust_sku_hist: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for cid, grp in df_sorted.groupby("customer_id", sort=False):
            dts = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
            pidxs = np.array([pid_to_idx.get(p, -1) for p in grp["product_id"]], dtype=np.int64)
            self._cust_sku_hist[cid] = (dts, pidxs)

    def _fast_resolve_skus(self, top_bns: np.ndarray, cid: str, date: pd.Timestamp, row_allowed: np.ndarray) -> np.ndarray:
        """Bit-exact single-customer causal pack-size resolver strictly < date (<0.2 ms)."""
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
            "model_family": "v2_full",
            "title": "v2_full Single-Stage E2E Wave2BE2ERanker + Option A — Live Demo",
            "split_label": "Out-of-Time Validation",
            "split_window_note": f"Validation {self.pred_dates[0].date()} → {self.pred_dates[-1].date()} (Held-out Test >= {TEST_START_V2.date()} strictly untouched)",
            "pipeline_labels": {
                "step2_title": "2. cand_42 + CUST_7 + SEQ_5",
                "step2_desc": "42 cand (incl. CF, mgr 30d, region 90d) + 7 cust + 5 seq",
                "step3_title": "3. E2E Wave2BE2ERanker + Option A",
                "step3_desc": f"Pure-Q + Full-42 bias · Main & Option A · {self.n_bn:,} formulations",
            },
            "supports_ranking_modes": True,
            "ranking_modes": [
                {"value": "blended", "label": "Blended (Main + Option A OOF)"},
                {"value": "pure_unseen", "label": "Pure Unseen Only (Option A Head)"},
                {"value": "main_only", "label": "Main Repeat Head Only"},
            ],
            "reference_id": self.reference["reference_id"],
            "active_variant": self.active_variant,
            "active_epoch": self.active_epoch,
            "checkpoint_kind": self.checkpoint_kind,
            "trainable_params": self.trainable_params,
            "per_request_params": self.per_request_params,
            "device": str(self.dev),
            "num_base_names": self.n_bn,
            "num_features": NUM_CAND_42_FEATURES,
            "num_customer_features": NUM_CUST_7_FEATURES,
            "test_baskets_total": self.n_cases,
            "test_date_range": [str(self.pred_dates[0].date()), str(self.pred_dates[-1].date())],
            "cohort_counts": self.cohort_counts,
            "filter_counts": {k: len(v) for k, v in self.filter_indices.items()},
            "val_metrics": self.active_val_metrics,
            "full_fit_affine": {"a": round(self.full_fit_ab[0], 4), "b": round(self.full_fit_ab[1], 4)},
        }

    def evaluate_case(self, case_idx: int, *, mode: str = "blended") -> Dict[str, Any]:
        mode_clean = str(mode or "blended").strip().lower()
        if mode_clean not in SUPPORTED_RANKING_MODES:
            mode_clean = "blended"

        with self.lock:
            idx = max(0, min(self.n_cases - 1, int(case_idx)))
            cid = self.customer_ids[idx]
            date = self.pred_dates[idx]
            is_first = bool(self.first_obs_np[idx])
            target_row = self.target_np[idx]
            new_row = self.new_target_np[idx]
            allowed_row = self.allowed_np[idx]
            unseen_row = self.unseen_np[idx]
            sku_allowed_row = self.sku_allowed_np[idx]
            sku_target_row = self.sku_target_np[idx]

            # 1. Assemble strictly causal cand_42 + CUST_7 + SEQ_5 (< date)
            t_feat0 = time.perf_counter()
            cand_42 = assemble_cand_42(self.val_cand38_mm[idx], self.val_extra5_mm[idx])
            cust_11_t = self.val_meta["tabular_features"][idx : idx + 1].float()
            cust_7_t = prune_cust_7(cust_11_t)
            cust_11_np = cust_11_t.squeeze(0).numpy()
            t_feat_ms = 1000.0 * (time.perf_counter() - t_feat0)

            # 2. Live single-stage E2E Wave2BE2ERanker forward pass over ALL 1,141 formulations
            t_inf0 = time.perf_counter()
            allowed_t = self.val_meta["allowed_bn"][idx : idx + 1].bool().to(self.dev)
            with torch.no_grad():
                res = self.ranker(
                    company_text_embedding=self.val_meta["text_embedding"][idx : idx + 1].float().to(self.dev),
                    city_id=self.val_meta["city_id"][idx : idx + 1].long().to(self.dev),
                    region_id=self.val_meta["region_id"][idx : idx + 1].long().to(self.dev),
                    industry_id=self.val_meta["industry_id"][idx : idx + 1].long().to(self.dev),
                    customer_features=cust_7_t.to(self.dev),
                    sequence=self.val_meta["sequence"][idx : idx + 1].float().to(self.dev),
                    attention_mask=self.val_meta["attention_mask"][idx : idx + 1].long().to(self.dev),
                    seq_product_ids=self.val_meta["seq_product_ids"][idx : idx + 1].long().to(self.dev),
                    seq_purchase_dates=self.val_meta["seq_purchase_dates"][idx : idx + 1].float().to(self.dev),
                    prediction_day=self.val_meta["prediction_days"][idx : idx + 1].float().to(self.dev),
                    candidate_features=torch.from_numpy(cand_42).unsqueeze(0).to(self.dev),
                    candidate_mask=allowed_t,
                    precomputed_v_bn=self.v_bn,
                    return_aux=False,
                )
                s_main = res["output"].logits.squeeze(0).float().cpu().numpy()
                s_unseen = res["unseen_logits"].squeeze(0).float().cpu().numpy()
                if self.dev.type == "mps":
                    torch.mps.synchronize()
            t_inf_ms = 1000.0 * (time.perf_counter() - t_inf0)

            fold_id = int(self.oof_folds[idx])
            a_cal, b_cal = self.fold_ab.get(fold_id, self.full_fit_ab)

            elig_unseen = allowed_row & unseen_row
            if mode_clean == "pure_unseen":
                active_mask = elig_unseen
                scores = np.where(elig_unseen, s_unseen, -np.inf)
            elif mode_clean == "main_only":
                active_mask = allowed_row
                scores = np.where(allowed_row, s_main, -np.inf)
            else:
                active_mask = allowed_row
                s_u_safe = np.where(elig_unseen, s_unseen, 0.0)
                s_m_safe = np.where(allowed_row, s_main, 0.0)
                scores = np.where(unseen_row, a_cal * s_u_safe + b_cal, s_m_safe)
                scores = np.where(allowed_row, scores, -np.inf)

            # Stable descending rank over active candidates (matching rank_eligible_formulations)
            scores_t = torch.from_numpy(scores.astype(np.float64))
            order = scores_t.argsort(descending=True, stable=True).numpy()
            valid_count = int(active_mask.sum())
            rank_of_bn = {
                int(bn_i): pos + 1
                for pos, bn_i in enumerate(order[:valid_count].tolist())
            }
            top10 = order[: min(10, valid_count)]

            # 3. Resolve Top-10 base_names -> concrete eligible pack SKUs strictly < date
            t_res0 = time.perf_counter()
            resolved_sku_idx = self._fast_resolve_skus(top10, cid, date, sku_allowed_row)
            t_res_ms = 1000.0 * (time.perf_counter() - t_res0)

            # 4. Human-readable strictly-prior customer stats & purchase history (< date)
            hist_df = self.histories.get(cid)
            if hist_df is not None:
                prior_df = hist_df.loc[hist_df["purchase_date"] < date].sort_values("purchase_date", ascending=False)
            else:
                prior_df = self.dev_purchases.iloc[:0]

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

            comp = self.company_map.get(cid, {})
            desc_str = str(comp.get("description", "")) if pd.notna(comp.get("description", "")) else ""
            ind_str = str(comp.get("industry_branch", "")) if pd.notna(comp.get("industry_branch", "")) else ""
            equip_domains = sorted(parse_company_equipment_domains(desc_str, ind_str, frozen_v5=False))

            customer_panel = {
                "customer_id": cid,
                "company_name": str(comp.get("company_name", cid)) if pd.notna(comp.get("company_name", "")) else cid,
                "city": str(comp.get("city", "—")) if pd.notna(comp.get("city", "")) else "—",
                "region": str(comp.get("region_code", "—")) if pd.notna(comp.get("region_code", "")) else "—",
                "industry": ind_str or "—",
                "sales_branch": str(comp.get("sales_branch", "—")) if pd.notna(comp.get("sales_branch", "")) else "—",
                "description_snippet": (desc_str[:220] + "…") if len(desc_str) > 220 else (desc_str or "No company description recorded."),
                "equipment_domains": equip_domains,
                "is_first_observed": is_first,
                "prior_order_days": int(len(prior_dates)),
                "prior_line_items": int(len(prior_df)),
                "prior_distinct_formulations": int(len(prior_bns)),
                "prior_total_quantity": round(float(prior_df["quantity"].sum()), 1) if not prior_df.empty else 0.0,
                "median_order_gap_days": round(median_gap_days, 1) if median_gap_days is not None else None,
                "days_since_last_order": days_since_last,
                "financial": {
                    "prepaid_or_cash_pct": round(100.0 * float(cust_11_np[U11_IDX["cust_prepaid_or_cash_ratio"]]), 1),
                    "avg_deferral_days": round(90.0 * float(cust_11_np[U11_IDX["cust_avg_deferral_days_norm"]]), 1),
                    "actual_vs_contract_ratio": round(2.0 * float(cust_11_np[U11_IDX["cust_actual_vs_contract_pay_ratio"]]), 2),
                    "overdue_unpaid_pct": round(100.0 * float(cust_11_np[U11_IDX["cust_overdue_unpaid_ratio"]]), 1),
                    "mgr_changed_recently": bool(cust_11_np[U11_IDX["mgr_changed_recently"]] > 0.5),
                },
            }

            # 5. Build Top-10 predictions list with cand_42 signal drivers & Option A / Main head provenance
            predictions = []
            rel_flags = []
            new_hits_10 = 0
            for rank_0, bn_i in enumerate(top10.tolist()):
                meta = self.bn_meta[bn_i]
                f_row = cand_42[bn_i]
                logit_v = float(scores[bn_i])
                prob_v = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit_v)))))
                is_new_candidate = bool(unseen_row[bn_i])
                is_hit = bool(new_row[bn_i]) if mode_clean == "pure_unseen" else bool(target_row[bn_i])
                is_new_hit = bool(new_row[bn_i])
                rel_flags.append(1.0 if is_hit else 0.0)
                if is_new_hit:
                    new_hits_10 += 1

                r_pidx = int(resolved_sku_idx[rank_0])
                resolved_sku_label = "—"
                resolved_sku_hit = False
                if r_pidx >= 0:
                    r_pid = self.product_ids[r_pidx]
                    resolved_sku_hit = bool(sku_target_row[r_pidx])
                    if r_pid in self.cat_indexed.index:
                        r_row = self.cat_indexed.loc[r_pid]
                        pname = str(r_row.get("short_name", r_row.get("product_name", r_pid)))
                        if not pname or pname == "nan":
                            pname = r_pid
                        cont = str(r_row.get("container_type", "")) if pd.notna(r_row.get("container_type", "")) else ""
                        resolved_sku_label = f"{r_pid} ({cont})" if cont else f"{r_pid}: {pname[:36]}"
                    else:
                        resolved_sku_label = r_pid

                # Build interpretable drivers from exact cand_42 causal features
                drivers: List[str] = []
                if mode_clean == "blended":
                    drivers.append("Option A Head" if is_new_candidate else "Main Head")
                elif mode_clean == "pure_unseen":
                    drivers.append("Option A Head")
                else:
                    drivers.append("Main Head")

                if f_row[C42_IDX["has_bought_base_sku"]] > 0.5:
                    approx_orders = max(
                        1, int(round(math.expm1(float(f_row[C42_IDX["user_base_sku_order_count_norm"]]) * math.log1p(50.0))))
                    )
                    approx_days = int(
                        round(math.expm1(float(f_row[C42_IDX["days_since_last_base_sku_buy_norm"]]) * math.log1p(365.0)))
                    )
                    drivers.append(f"Repeat ({approx_orders}× · {approx_days}d ago)")
                    urg = float(f_row[C42_IDX["base_sku_urgency"]]) * 5.0
                    if urg >= 0.6:
                        drivers.append(f"Due {urg:.1f}×")
                else:
                    cf_rec = float(f_row[C42_IDX["full_cf_recency_weighted"]])
                    cf_cos = float(f_row[C42_IDX["full_cf_cosine"]])
                    if cf_rec > 0.08:
                        drivers.append(f"CF rec {cf_rec:.2f}")
                    elif cf_cos > 0.08:
                        drivers.append(f"CF cos {cf_cos:.2f}")

                if f_row[C42_IDX["base_name_recent3_lift"]] > 0.15:
                    drivers.append(f"Seq lift +{f_row[C42_IDX['base_name_recent3_lift']]:.2f}")
                if f_row[C42_IDX["mgr_base_name_share_30d"]] > 0.02:
                    drivers.append(f"Mgr 30d {100*f_row[C42_IDX['mgr_base_name_share_30d']]:.1f}%")
                elif f_row[C42_IDX["mgr_industry_base_name_share_90d"]] > 0.02:
                    drivers.append(f"Mgr×Ind {100*f_row[C42_IDX['mgr_industry_base_name_share_90d']]:.1f}%")
                elif f_row[C42_IDX["mgr_base_name_share_90d"]] > 0.02:
                    drivers.append(f"Mgr 90d {100*f_row[C42_IDX['mgr_base_name_share_90d']]:.1f}%")
                if f_row[C42_IDX["region_base_name_share_90d"]] > 0.025 and is_new_candidate:
                    drivers.append(f"Reg 90d {100*f_row[C42_IDX['region_base_name_share_90d']]:.1f}%")
                if f_row[C42_IDX["base_name_industry_affinity"]] > 0.03:
                    drivers.append(f"Industry {100*f_row[C42_IDX['base_name_industry_affinity']]:.1f}%")
                if f_row[C42_IDX["app_matches_company_equipment"]] > 0.5:
                    drivers.append("Equipment match")
                if f_row[C42_IDX["user_viscosity_order_share"]] > 0.0 and is_new_candidate:
                    drivers.append(f"Viscosity {100*f_row[C42_IDX['user_viscosity_order_share']]:.0f}%")

                predictions.append(
                    {
                        "rank": rank_0 + 1,
                        "bn_idx": bn_i,
                        "base_name": meta["base_name"],
                        "brand": meta["brand"],
                        "category": meta["category"],
                        "resolved_sku": resolved_sku_label,
                        "resolved_sku_hit": resolved_sku_hit,
                        "logit": round(logit_v, 3),
                        "main_logit": round(float(s_main[bn_i]), 3) if np.isfinite(s_main[bn_i]) else None,
                        "unseen_logit": round(float(s_unseen[bn_i]), 3) if np.isfinite(s_unseen[bn_i]) else None,
                        "probability": round(prob_v, 4),
                        "is_repeat_candidate": not is_new_candidate,
                        "is_hit": is_hit,
                        "is_new_hit": is_new_hit,
                        "drivers": drivers[:4],
                    }
                )

            # 6. Ground-truth basket items & exact case metrics
            gt_items = []
            for bn_i in np.flatnonzero(target_row).tolist():
                meta = self.bn_meta[bn_i]
                is_elig = bool(active_mask[bn_i])
                rk = rank_of_bn.get(bn_i) if is_elig else None
                gt_items.append(
                    {
                        "base_name": meta["base_name"],
                        "brand": meta["brand"],
                        "category": meta["category"],
                        "is_new": bool(new_row[bn_i]),
                        "is_eligible": is_elig,
                        "is_modeled": True,
                        "predicted_rank": rk,
                        "hit_top10": (rk is not None and rk <= 10),
                    }
                )
            gt_items.sort(key=lambda x: (x["predicted_rank"] is None, x["predicted_rank"] or 9999))

            n_pos_all = int(new_row.sum()) if mode_clean == "pure_unseen" else int(target_row.sum())
            n_new_all = int(new_row.sum())
            hits_10 = int(sum(rel_flags))
            disc = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
            dcg = float(np.sum(np.array(rel_flags, dtype=np.float64) * disc[: len(rel_flags)]))
            idcg = float(np.sum(disc[: min(n_pos_all, 10)])) if n_pos_all > 0 else 1.0
            case_ndcg = dcg / idcg if (n_pos_all > 0 and idcg > 0) else 0.0
            case_r10 = (hits_10 / n_pos_all) if n_pos_all > 0 else 0.0

            return {
                "case_idx": idx,
                "ranking_mode": mode_clean,
                "affine_blend": {"fold": fold_id, "a": round(a_cal, 3), "b": round(b_cal, 3)},
                "prediction_date": str(date.date()),
                "eligible_formulations": int(active_mask.sum()),
                "total_formulations": int(self.n_bn),
                "timings_ms": {
                    "features_and_sequence": round(t_feat_ms, 2),
                    "fullpool_ups_canr": round(t_inf_ms, 2),
                    "pack_resolution": round(t_res_ms, 2),
                },
                "case_metrics": {
                    "hits_at_10": hits_10,
                    "total_targets": n_pos_all,
                    "basket_total_targets": int(target_row.sum()),
                    "new_hits_at_10": int(new_hits_10),
                    "total_new_targets": n_new_all,
                    "recall_at_10": round(case_r10, 4),
                    "ndcg_at_10": round(case_ndcg, 4),
                    "unmodeled_targets": 0,
                },
                "customer": customer_panel,
                "prior_history": prior_history_rows,
                "predictions_top10": predictions,
                "ground_truth_basket": gt_items,
            }
