"""Inference-only CPU/MPS vector recomputation and Stage 1/Stage 2 latency benchmark (no training)."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiment_signals import sync
from recommender.experiments.embedding_ablation.config import build_ablation_specs
from recommender.experiments.static_enrichment.train import (
    _make_downstream_ranker,
    _stage1_customer_inputs,
    _stage1_product_inputs,
    encode_and_score_variant,
    make_variant_model,
)
from recommender.train_ranker import ranker_batch_inputs

from .config import (
    ABLATION_BUNDLE_PATH,
    EVIDENCE_ROOT,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    SEED,
    SLATE_SIZE,
    STAGE1_MLP_REF_RETRIEVER,
    UPSTREAM_STAGE1_RUN_DIR,
)
from .data import RollingDynamicFeatureAdapter, assert_dev_dates_strict, verify_active_reference
from .model import RollingCandidateAttentionRanker
from .train import snapshot_preserved_hashes


def _hardware_metadata() -> Dict[str, Any]:
    cpu_brand = platform.processor() or platform.machine()
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            cpu_brand = out
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu_brand,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def _time_callable(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int = 3,
    repeats: int = 15,
    items: int = 1,
) -> Dict[str, Any]:
    for _ in range(max(int(warmup), 0)):
        fn()
    sync(device)

    samples_ms: List[float] = []
    for _ in range(int(repeats)):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(samples_ms, dtype=np.float64)
    med_ms = float(np.median(arr))
    return {
        "items": int(items),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "median_ms": med_ms,
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "per_item_us": float((med_ms * 1000.0) / max(int(items), 1)),
    }


def _deduplicate_unique_customers(
    train_ret: Mapping[str, Any],
    val_ret: Mapping[str, Any],
    train_src: Mapping[str, Any],
    val_src: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    spec,
) -> Dict[str, Any]:
    """Select the latest observed snapshot per unique customer across train, val, and combined dev."""
    assert_dev_dates_strict("train", train_src["prediction_days"])
    assert_dev_dates_strict("val", val_src["prediction_days"])

    def _collect(pairs: Sequence[Tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
        latest_key: Dict[str, Tuple[int, int, int]] = {}
        latest_loc: Dict[str, Tuple[int, int]] = {}
        for split_order, (split_name, ret_data, src_data) in enumerate(pairs):
            days = src_data["prediction_days"].tolist()
            cids = src_data["customer_ids"]
            for row_idx, (cid, day) in enumerate(zip(cids, days)):
                key = (int(day), int(split_order), int(row_idx))
                if cid not in latest_key or key > latest_key[cid]:
                    latest_key[cid] = key
                    latest_loc[cid] = (split_order, row_idx)

        ordered_cids = sorted(latest_loc.keys())
        # Group indices per split then restore canonical order
        split_buckets: List[List[Tuple[int, int]]] = [[] for _ in pairs]
        for out_pos, cid in enumerate(ordered_cids):
            sp_ord, r_idx = latest_loc[cid]
            split_buckets[sp_ord].append((out_pos, r_idx))

        first_ret = pairs[0][1]
        n = len(ordered_cids)
        combined: Dict[str, Any] = {}
        fields = (
            "text_embedding",
            "city_id",
            "region_id",
            "industry_id",
            "tabular_features",
            "sequence",
            "attention_mask",
            "seq_product_ids",
            "seq_purchase_dates",
            "familiarity",
        )
        for field in fields:
            ref_t = first_ret[field]
            combined[field] = torch.empty((n, *ref_t.shape[1:]), dtype=ref_t.dtype)
        extra_attrs: Dict[str, torch.Tensor] = {}
        for attr_name in spec.customer_embeddings:
            ref_a = enrichment[f"{pairs[0][0]}_customer_attrs"][attr_name]
            extra_attrs[attr_name] = torch.empty((n, *ref_a.shape[1:]), dtype=ref_a.dtype)

        for sp_ord, bucket in enumerate(split_buckets):
            if not bucket:
                continue
            sp_name, ret_data, _ = pairs[sp_ord]
            out_idx = torch.tensor([x[0] for x in bucket], dtype=torch.long)
            src_idx = torch.tensor([x[1] for x in bucket], dtype=torch.long)
            for field in fields:
                combined[field][out_idx] = ret_data[field][src_idx]
            for attr_name in spec.customer_embeddings:
                extra_attrs[attr_name][out_idx] = enrichment[f"{sp_name}_customer_attrs"][attr_name][src_idx]

        return {
            "count": n,
            "customer_ids": ordered_cids,
            "inputs": combined,
            "extra_attrs": extra_attrs,
        }

    return {
        "train_unique": _collect([("train", train_ret, train_src)]),
        "val_unique": _collect([("val", val_ret, val_src)]),
        "dev_unique": _collect([("train", train_ret, train_src), ("val", val_ret, val_src)]),
        "val_all_transitions_count": int(len(val_src["customer_ids"])),
        "train_all_transitions_count": int(len(train_src["customer_ids"])),
    }


def _to_device_customer_inputs(
    cohort: Mapping[str, Any],
    rows: torch.Tensor,
    device: torch.device,
    spec,
) -> Dict[str, Any]:
    src = cohort["inputs"]
    out: Dict[str, Any] = {
        key: src[key][rows].to(device)
        for key in (
            "text_embedding",
            "city_id",
            "region_id",
            "industry_id",
            "tabular_features",
            "sequence",
            "attention_mask",
            "seq_product_ids",
            "seq_purchase_dates",
        )
    }
    if spec.customer_embeddings:
        out["extra_attr_ids"] = {
            name: cohort["extra_attrs"][name][rows].to(device)
            for name in spec.customer_embeddings
        }
    return out


@torch.no_grad()
def _benchmark_device(
    device_name: str,
    *,
    e_no_brand_spec,
    retriever_ckpt: Mapping[str, Any],
    selector,
    val_ret: Mapping[str, Any],
    val_src: Mapping[str, Any],
    info: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    cohorts: Mapping[str, Any],
    warmup: int = 3,
    repeats: int = 15,
    batch_size: int = 256,
) -> Dict[str, Any]:
    dev = torch.device(device_name)
    out: Dict[str, Any] = {"device": device_name}

    # 1. Active repeat-no-month-v1 E_no_brand towers (12D product container_type, 0D customer stats)
    t_load0 = time.perf_counter()
    model = make_variant_model(e_no_brand_spec.variant_spec, info, enrichment)
    model.load_state_dict(retriever_ckpt["model_state_dict"], strict=True)
    model = model.to(dev)
    model.eval()
    prod_inputs_all = _stage1_product_inputs(e_no_brand_spec.variant_spec, info, enrichment, dev)
    prod_inputs_one = _stage1_product_inputs(e_no_brand_spec.variant_spec, info, enrichment, dev, limit=1)
    sync(dev)
    out["active_model_transfer_ms"] = (time.perf_counter() - t_load0) * 1000.0

    # Pre-materialize customer batches on device to isolate steady-state neural forward latency
    # (separately we also report end-to-end host->device slice+forward where relevant).
    prod_vecs_dev = model.product_vectors(prod_inputs_all)
    n_prods = int(prod_vecs_dev.shape[0])
    prod_indices = torch.arange(n_prods, device=dev)

    def _prep_batches(cohort_key: str) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], int]:
        cohort = cohorts[cohort_key]
        n = int(cohort["count"])
        batches = []
        fam_batches = []
        for rows in torch.arange(n).split(int(batch_size)):
            batches.append(_to_device_customer_inputs(cohort, rows, dev, e_no_brand_spec.variant_spec))
            fam_batches.append(cohort["inputs"]["familiarity"][rows].to(dev))
        return batches, fam_batches, n

    dev_batches, dev_fams, n_dev_uniq = _prep_batches("dev_unique")
    train_batches, _, n_train_uniq = _prep_batches("train_unique")
    val_uniq_batches, _, n_val_uniq = _prep_batches("val_unique")
    single_cust_batch = _to_device_customer_inputs(
        cohorts["dev_unique"], torch.tensor([0]), dev, e_no_brand_spec.variant_spec
    )
    single_fam = cohorts["dev_unique"]["inputs"]["familiarity"][:1].to(dev)

    val_trans_batches = []
    n_val_trans = len(val_ret["target"])
    for rows in torch.arange(n_val_trans).split(int(batch_size)):
        val_trans_batches.append(
            _stage1_customer_inputs(e_no_brand_spec.variant_spec, val_ret, rows, enrichment, "val", dev)
        )

    def _encode_cohort(batches: List[Dict[str, Any]]) -> torch.Tensor:
        vecs = [model.encode_purchase(b, prod_vecs_dev) for b in batches]
        return torch.cat(vecs, dim=0)

    def _score_cohort(batches: List[Dict[str, Any]], fams: List[torch.Tensor]) -> torch.Tensor:
        out_scores = []
        for b, f in zip(batches, fams):
            u = model.encode_purchase(b, prod_vecs_dev)
            out_scores.append(model.score_cached(u, prod_vecs_dev, prod_indices, f))
        return torch.cat(out_scores, dim=0)

    active_timings = {
        "product_tower_all_841": _time_callable(
            lambda: model.product_vectors(prod_inputs_all), dev, warmup=warmup, repeats=repeats, items=n_prods
        ),
        "product_tower_single_1": _time_callable(
            lambda: model.product_vectors(prod_inputs_one), dev, warmup=warmup, repeats=repeats, items=1
        ),
        "customer_tower_dev_unique_2571_bs256": _time_callable(
            lambda: _encode_cohort(dev_batches), dev, warmup=warmup, repeats=repeats, items=n_dev_uniq
        ),
        "customer_tower_train_unique_2319_bs256": _time_callable(
            lambda: _encode_cohort(train_batches), dev, warmup=warmup, repeats=repeats, items=n_train_uniq
        ),
        "customer_tower_val_unique_1676_bs256": _time_callable(
            lambda: _encode_cohort(val_uniq_batches), dev, warmup=warmup, repeats=repeats, items=n_val_uniq
        ),
        "customer_tower_val_all_transitions_6191_bs256": _time_callable(
            lambda: _encode_cohort(val_trans_batches), dev, warmup=warmup, repeats=repeats, items=n_val_trans
        ),
        "customer_tower_single_1": _time_callable(
            lambda: model.encode_purchase(single_cust_batch, prod_vecs_dev),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=1,
        ),
        "end_to_end_full_dev_recompute_841p_2571c_encode_only": _time_callable(
            lambda: (
                lambda pv: torch.cat([model.encode_purchase(b, pv) for b in dev_batches], dim=0)
            )(model.product_vectors(prod_inputs_all)),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=n_prods + n_dev_uniq,
        ),
        "end_to_end_dev_unique_2571_plus_dot_scoring_841": _time_callable(
            lambda: _score_cohort(dev_batches, dev_fams),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=n_dev_uniq,
        ),
        "single_customer_encode_plus_dot_841": _time_callable(
            lambda: model.score_cached(
                model.encode_purchase(single_cust_batch, prod_vecs_dev),
                prod_vecs_dev,
                prod_indices,
                single_fam,
            ),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=1,
        ),
    }
    out["active_e_no_brand_12d_0d"] = active_timings

    # 2. Shape-only untrained latency probe with 18D product numeric (12D container + 6D demand)
    #    and 6D customer tabular stats restored into towers. NEVER trained or claimed as quality evidence!
    probe_spec = replace(
        e_no_brand_spec.variant_spec,
        name="shape_only_probe_18d_6d",
        product_demand_features=18,
        customer_tabular_features=6,
    )
    probe_model = make_variant_model(probe_spec, info, enrichment).to(dev)
    probe_model.eval()
    probe_vecs_dev = probe_model.product_vectors(prod_inputs_all)
    probe_timings = {
        "shape_only_untrained_latency_probe": True,
        "trained": False,
        "product_demand_features": 18,
        "customer_tabular_features": 6,
        "product_tower_all_841": _time_callable(
            lambda: probe_model.product_vectors(prod_inputs_all),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=n_prods,
        ),
        "product_tower_single_1": _time_callable(
            lambda: probe_model.product_vectors(prod_inputs_one),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=1,
        ),
        "customer_tower_dev_unique_2571_bs256": _time_callable(
            lambda: torch.cat([probe_model.encode_purchase(b, probe_vecs_dev) for b in dev_batches], dim=0),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=n_dev_uniq,
        ),
        "customer_tower_single_1": _time_callable(
            lambda: probe_model.encode_purchase(single_cust_batch, probe_vecs_dev),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=1,
        ),
        "end_to_end_full_dev_recompute_841p_2571c_encode_only": _time_callable(
            lambda: (
                lambda pv: torch.cat([probe_model.encode_purchase(b, pv) for b in dev_batches], dim=0)
            )(probe_model.product_vectors(prod_inputs_all)),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=n_prods + n_dev_uniq,
        ),
    }
    out["shape_only_probe_with_tower_stats_18d_6d"] = probe_timings

    # 3. Active Stage 1 Selector (301D) forward timing over full 841 SKUs
    sel_dev = selector
    import copy
    sel_copy = copy.deepcopy(sel_dev).to(dev)
    sel_copy.eval()
    u1 = torch.randn(1, 128, device=dev)
    u256 = torch.randn(256, 128, device=dev)
    f41_1 = torch.randn(1, n_prods, 41, device=dev)
    f41_256 = torch.randn(256, n_prods, 41, device=dev)
    cs1 = torch.randn(1, 6, device=dev)
    cs256 = torch.randn(256, 6, device=dev)
    out["stage1_active_selector_301d"] = {
        "batch1_x_841": _time_callable(
            lambda: sel_copy(u=u1, v=prod_vecs_dev, features_41=f41_1, customer_stats_6=cs1),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=1,
        ),
        "batch256_x_841": _time_callable(
            lambda: sel_copy(u=u256, v=prod_vecs_dev, features_41=f41_256, customer_stats_6=cs256),
            dev,
            warmup=warmup,
            repeats=repeats,
            items=256,
        ),
    }

    # 4. Stage 2 RollingCandidateAttentionRanker (control 425D vs treatment candidate self-attention)
    torch.manual_seed(SEED)
    dyn_ranker = _make_downstream_ranker(
        "dynamic_features", SEED, STAGE1_MLP_REF_RETRIEVER, is_control_stage1=False
    )
    ctrl_ranker = RollingCandidateAttentionRanker.from_dynamic_context_ranker(
        dyn_ranker, candidate_attention_enabled=False
    ).to(dev)
    treat_ranker = ctrl_ranker.clone_with_candidate_attention(True).to(dev)
    ctrl_ranker.eval()
    treat_ranker.eval()

    def _dummy_ranker_inputs(bs: int) -> Dict[str, Any]:
        return {
            "text_embedding": torch.randn(bs, 1024, device=dev),
            "city_id": torch.zeros(bs, dtype=torch.long, device=dev),
            "region_id": torch.zeros(bs, dtype=torch.long, device=dev),
            "industry_id": torch.zeros(bs, dtype=torch.long, device=dev),
            "tabular_features": torch.zeros(bs, 8, device=dev),
            "sequence": torch.zeros(bs, 32, 9, device=dev),
            "attention_mask": torch.ones(bs, 32, dtype=torch.bool, device=dev),
            "seq_product_ids": torch.ones(bs, 32, dtype=torch.long, device=dev),
            "seq_category_ids": torch.ones(bs, 32, dtype=torch.long, device=dev),
            "seq_subcategory_ids": torch.ones(bs, 32, dtype=torch.long, device=dev),
            "seq_business_line_ids": torch.ones(bs, 32, dtype=torch.long, device=dev),
            "pairwise_features": torch.randn(bs, SLATE_SIZE, 41, device=dev),
            "candidate_vectors": torch.randn(bs, SLATE_SIZE, 128, device=dev),
            "seq_purchase_dates": torch.zeros(bs, 32, dtype=torch.long, device=dev),
            "prediction_day": torch.full((bs,), 300, dtype=torch.long, device=dev),
            "candidate_mask": torch.ones(bs, SLATE_SIZE, dtype=torch.bool, device=dev),
            "cached_customer_vectors": torch.randn(bs, 128, device=dev),
        }

    r_in_1 = _dummy_ranker_inputs(1)
    r_in_64 = _dummy_ranker_inputs(64)
    out["stage2_rolling_ranker_slate100"] = {
        "control_pointwise_425d_batch1": _time_callable(
            lambda: ctrl_ranker(**r_in_1), dev, warmup=warmup, repeats=repeats, items=1
        ),
        "treatment_cand_attn_425d_batch1": _time_callable(
            lambda: treat_ranker(**r_in_1), dev, warmup=warmup, repeats=repeats, items=1
        ),
        "control_pointwise_425d_batch64": _time_callable(
            lambda: ctrl_ranker(**r_in_64), dev, warmup=warmup, repeats=repeats, items=64
        ),
        "treatment_cand_attn_425d_batch64": _time_callable(
            lambda: treat_ranker(**r_in_64), dev, warmup=warmup, repeats=repeats, items=64
        ),
    }

    return out


def run_benchmark(
    *,
    evidence_root: Path = EVIDENCE_ROOT,
    warmup: int = 3,
    repeats: int = 15,
) -> Dict[str, Any]:
    """Run strict inference-only CPU+MPS latency benchmark and save artifacts."""
    before_hashes = snapshot_preserved_hashes()
    ref_info = verify_active_reference(load_selector=True)

    t_cold0 = time.perf_counter()
    retriever_ckpt = torch.load(STAGE1_MLP_REF_RETRIEVER, map_location="cpu", weights_only=False)
    enrichment = torch.load(ABLATION_BUNDLE_PATH, map_location="cpu", weights_only=False)
    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()
    e_no_brand = build_ablation_specs()["E_no_brand"]
    cold_artifact_load_ms = (time.perf_counter() - t_cold0) * 1000.0

    # Verify non-timed exact cache parity against val_purchase_cache.pt on CPU.
    cpu_model = make_variant_model(e_no_brand.variant_spec, info, enrichment)
    cpu_model.load_state_dict(retriever_ckpt["model_state_dict"], strict=True)
    cpu_model.eval()
    with torch.no_grad():
        prod_vecs, val_cust_vecs, _ = encode_and_score_variant(
            cpu_model,
            e_no_brand.variant_spec,
            val_ret,
            info,
            enrichment,
            "val",
            torch.device("cpu"),
        )
    val_cache = torch.load(UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt", map_location="cpu", weights_only=False)
    max_prod_diff = float((prod_vecs - val_cache["product_vectors"]).abs().max().item())
    max_cust_diff = float((val_cust_vecs - val_cache["customer_vectors"]).abs().max().item())
    if max_prod_diff > 1e-4 or max_cust_diff > 1e-4:
        raise AssertionError(
            f"Parity check against frozen val_purchase_cache.pt failed: "
            f"prod={max_prod_diff:.3e}, cust={max_cust_diff:.3e}"
        )

    cohorts = _deduplicate_unique_customers(train_ret, val_ret, train_src, val_src, enrichment, e_no_brand.variant_spec)

    device_results: Dict[str, Any] = {}
    devices_to_run = ["cpu"]
    if torch.backends.mps.is_available():
        devices_to_run.append("mps")
    for dname in devices_to_run:
        device_results[dname] = _benchmark_device(
            dname,
            e_no_brand_spec=e_no_brand,
            retriever_ckpt=retriever_ckpt,
            selector=ref_info["selector"],
            val_ret=val_ret,
            val_src=val_src,
            info=info,
            enrichment=enrichment,
            cohorts=cohorts,
            warmup=warmup,
            repeats=repeats,
        )

    # Measure CPU request-time RollingDynamicFeatureAdapter construction + cold/warm single-customer feature extraction
    t_adapt0 = time.perf_counter()
    adapter = RollingDynamicFeatureAdapter.from_development_data(
        info["product_ids"],
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    adapter_init_ms = (time.perf_counter() - t_adapt0) * 1000.0

    origin = pd.Timestamp("2025-01-01")
    mid_idx = len(val_src["customer_ids"]) // 2
    cid_sample = val_src["customer_ids"][mid_idx]
    as_of_sample = origin + pd.Timedelta(days=int(val_src["prediction_days"][mid_idx]))
    # Find another customer with the same or nearby date for warm global_cache test
    cid_other = next(
        (c for c in val_src["customer_ids"] if c != cid_sample),
        cid_sample,
    )

    def _adapter_cold_day() -> np.ndarray:
        adapter.builder.global_cache.clear()
        return adapter.build_quantized(cid_sample, as_of_sample)

    adapter_cold_timing = _time_callable(
        _adapter_cold_day, torch.device("cpu"), warmup=2, repeats=10, items=EXPECTED_PRODUCT_UNIVERSE
    )
    # Prime global_cache for warm hit
    adapter.build_quantized(cid_sample, as_of_sample)
    adapter_warm_timing = _time_callable(
        lambda: adapter.build_quantized(cid_other, as_of_sample),
        torch.device("cpu"),
        warmup=3,
        repeats=15,
        items=EXPECTED_PRODUCT_UNIVERSE,
    )
    def _global_features_cold() -> np.ndarray:
        adapter.builder.global_cache.clear()
        return adapter.builder.global_features(as_of_sample)

    global_feat_cold_timing = _time_callable(
        _global_features_cold, torch.device("cpu"), warmup=2, repeats=10, items=EXPECTED_PRODUCT_UNIVERSE
    )

    after_hashes = snapshot_preserved_hashes()
    if after_hashes != before_hashes:
        raise AssertionError("Preserved artifact SHA-256 mismatch detected after benchmark!")

    report = {
        "training_executed": False,
        "heldout_access": False,
        "promoted": False,
        "active_reference_id": EXPECTED_REFERENCE_ID,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "hardware": _hardware_metadata(),
        "cold_artifact_load_ms": float(cold_artifact_load_ms),
        "cache_parity_check_non_timed": {
            "max_abs_product_vector_diff": max_prod_diff,
            "max_abs_val_customer_vector_diff": max_cust_diff,
        },
        "universe_counts": {
            "products": int(len(info["product_ids"])),
            "dev_unique_customers": int(cohorts["dev_unique"]["count"]),
            "train_unique_customers": int(cohorts["train_unique"]["count"]),
            "val_unique_customers": int(cohorts["val_unique"]["count"]),
            "train_all_transitions": int(cohorts["train_all_transitions_count"]),
            "val_all_transitions": int(cohorts["val_all_transitions_count"]),
        },
        "devices": device_results,
        "cpu_rolling_feature_adapter": {
            "cold_adapter_init_ms": float(adapter_init_ms),
            "single_request_841_skus_cold_date": adapter_cold_timing,
            "single_request_841_skus_warm_date": adapter_warm_timing,
            "global_features_841_skus_cold_date": global_feat_cold_timing,
        },
        "preserved_hashes": after_hashes,
        "limitations": [
            "Inference-only latency benchmark; training_executed=false, heldout_access=false, promoted=false.",
            "Steady-state tower timings measure pre-materialized device inputs after warmup and sync(); cold disk load is reported separately.",
            "shape_only_probe_with_tower_stats_18d_6d uses untrained weights strictly to measure forward-pass latency if 6D demand/6D customer stats were added; zero quality claims.",
            "Snapshot metadata (snapshot_unverified) caveat remains: product/company attributes are not effective-dated.",
        ],
    }

    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(report, root / "benchmark.json", overwrite=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference-only CPU/MPS vector recomputation & rolling Stage 1/2 latency benchmark."
    )
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    report = run_benchmark(
        evidence_root=args.evidence_root,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
