"""Read-only verified cache adapter for no-brand-v1 Stage 1 u/v and 41-col dynamic features."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch

from recommender.artifacts import file_digest
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import sliced
from recommender.experiments.dynamic_selector.feature_schema import (
    NAMES as SOURCE_CONTEXT_NAMES,
    contract as source_feature_contract,
)
from recommender.experiments.static_enrichment.data import (
    load_dynamic_campaign_features_readonly,
    quantize_transferred_stats,
)
from recommender.split import TEST_START, TRAIN_CUTOFF

from .config import (
    BASELINE_EVAL_DIR,
    BASELINE_SELECTOR_RUN_DIR,
    E1_E5_RETRIEVAL_CACHE_DIR,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PRIOR_300_SELECTOR_SHA256,
    EXPECTED_REFERENCE_ID,
    EXPECTED_RETRIEVER_SHA256,
    EXPECTED_SELECTOR_SHA256,
    MIN_ELIGIBLE_SKUS,
    OMITTED_CONTEXT_FEATURES,
    PRIOR_300_EVIDENCE_DIR,
    PRIOR_300_RUN_DIR,
    RECIPE_NAME,
    REFERENCE_MANIFEST,
    REFERENCE_POINTER,
    REFERENCE_RETRIEVER,
    REFERENCE_SELECTOR,
    UPSTREAM_STAGE1_RUN_DIR,
    VARIANT_NAME,
)
from .schema import build_feature_ablation_spec, build_input_contract, validate_input_contract


def _assert_dev_dates_strict(split: str, prediction_days: torch.Tensor) -> None:
    origin = pd.Timestamp("2025-01-01")
    train_cutoff_day = (TRAIN_CUTOFF - origin).days
    test_start_day = (TEST_START - origin).days
    if bool((prediction_days >= test_start_day).any()):
        raise ValueError(f"Held-out dates >= {TEST_START.date()} detected in {split} split.")
    if split == "train" and bool((prediction_days >= train_cutoff_day).any()):
        raise ValueError(f"Validation/test dates >= {TRAIN_CUTOFF.date()} detected in train split.")
    if split == "val" and bool((prediction_days < train_cutoff_day).any()):
        raise ValueError(f"Training dates < {TRAIN_CUTOFF.date()} detected in val split.")


def verify_no_brand_reference() -> Dict[str, Any]:
    """Verify the immutable no-brand-v1 reference pointer, manifest, and checkpoint hashes."""
    for path in (
        REFERENCE_POINTER,
        REFERENCE_MANIFEST,
        REFERENCE_RETRIEVER,
        REFERENCE_SELECTOR,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required reference artifact missing: {path}")

    ref_ptr = json.loads(REFERENCE_POINTER.read_text(encoding="utf-8"))
    if ref_ptr.get("active_reference_id") != EXPECTED_REFERENCE_ID:
        raise ValueError(
            f"Expected active_reference_id={EXPECTED_REFERENCE_ID!r}, got {ref_ptr.get('active_reference_id')!r}."
        )
    if ref_ptr.get("heldout_access") is not False:
        raise ValueError("Reference pointer must record heldout_access=false.")

    manifest_sha = file_digest(REFERENCE_MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Reference manifest hash mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}")
    retriever_sha = file_digest(REFERENCE_RETRIEVER)
    if retriever_sha != EXPECTED_RETRIEVER_SHA256:
        raise ValueError(f"Reference retriever hash mismatch: {retriever_sha} != {EXPECTED_RETRIEVER_SHA256}")
    selector_sha = file_digest(REFERENCE_SELECTOR)
    if selector_sha != EXPECTED_SELECTOR_SHA256:
        raise ValueError(f"Reference selector hash mismatch: {selector_sha} != {EXPECTED_SELECTOR_SHA256}")

    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    return {
        "reference_pointer": ref_ptr,
        "manifest": manifest,
        "hashes": {
            "reference_json": file_digest(REFERENCE_POINTER),
            "manifest_json": manifest_sha,
            "best_retriever_pt": retriever_sha,
            "best_selector_pt": selector_sha,
        },
    }


def load_verified_no_brand_campaign_data() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Load verified Stage 1 cached u/v and 41-col dynamic features without ever encoding towers or touching held-out data."""
    ref_info = verify_no_brand_reference()
    manifest = ref_info["manifest"]
    expected_enrichment_identity = manifest["source_identities"]["enrichment_identity"]

    for name in ("train.pt", "val.pt", "manifest.pt"):
        if not (E1_E5_RETRIEVAL_CACHE_DIR / name).is_file():
            raise FileNotFoundError(
                f"Required retrieval cache {E1_E5_RETRIEVAL_CACHE_DIR / name} missing; refusing raw regeneration."
            )

    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()
    if len(info["product_ids"]) != 841:
        raise ValueError(f"Expected 841 product universe, got {len(info['product_ids'])}.")

    dyn_manifest, train_feat_mmap, val_feat_mmap = load_dynamic_campaign_features_readonly()
    if dyn_manifest.get("source_identity") != info["identity"]:
        raise ValueError("Dynamic campaign source_identity mismatch against e1_e5_retrieval_v1.")
    if dyn_manifest.get("feature_contract") != source_feature_contract():
        raise ValueError("Dynamic campaign feature_contract mismatch.")
    if tuple(dyn_manifest["feature_contract"]["names"]) != tuple(SOURCE_CONTEXT_NAMES):
        raise ValueError("Dynamic campaign feature names do not match ordered canonical schema.")
    if len(SOURCE_CONTEXT_NAMES) != 41:
        raise ValueError(f"Expected 41 canonical context features, got {len(SOURCE_CONTEXT_NAMES)}.")

    campaign_data: Dict[str, Dict[str, Any]] = {}
    cache_lineage: Dict[str, Any] = {}
    for split, ret_data, src_data, feat_mmap in (
        ("train", train_ret, train_src, train_feat_mmap),
        ("val", val_ret, val_src, val_feat_mmap),
    ):
        cache_path = UPSTREAM_STAGE1_RUN_DIR / f"{split}_purchase_cache.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Verified Stage 1 purchase cache missing: {cache_path}. Never fall back to tower encoding."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("model_digest") != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(
                f"{cache_path} model_digest mismatch: {payload.get('model_digest')} != {EXPECTED_RETRIEVER_SHA256}"
            )
        if payload.get("data_identity") != info["identity"]:
            raise ValueError(f"{cache_path} data_identity mismatch.")
        if split == "val":
            if payload.get("enrichment_identity") != expected_enrichment_identity:
                raise ValueError(f"{cache_path} enrichment_identity mismatch.")
            expected_val_cache_sha = manifest["provenance_and_caches"]["stage1_val_purchase_cache"]["sha256"]
            if file_digest(cache_path) != expected_val_cache_sha:
                raise ValueError(f"{cache_path} sha256 mismatch against reference manifest.")

        rows = torch.where(ret_data["allowed"].sum(1) >= MIN_ELIGIBLE_SKUS)[0]
        sliced_src = sliced(src_data, rows, len(src_data["customer_ids"]))
        sliced_ret = sliced(ret_data, rows, len(src_data["customer_ids"]))
        _assert_dev_dates_strict(split, sliced_src["prediction_days"])

        u = payload["customer_vectors"][rows].squeeze(1).contiguous()
        v = payload["product_vectors"].contiguous()
        dot_scores = payload["scores"][rows].contiguous()
        if u.shape != (len(rows), 128) or v.shape != (len(info["product_ids"]), 128):
            raise ValueError(f"Unexpected u/v shape in {cache_path}: u={u.shape}, v={v.shape}.")
        if feat_mmap.shape != (len(rows), len(info["product_ids"]), 41):
            raise ValueError(
                f"Dynamic feature memmap shape mismatch for {split}: {feat_mmap.shape} vs "
                f"({len(rows)}, {len(info['product_ids'])}, 41)."
            )

        sliced_ret.update(
            source=sliced_src,
            cohort_rows=rows,
            u=u,
            v=v,
            dot_scores=dot_scores,
            features=feat_mmap,
            customer_stats_6=quantize_transferred_stats(sliced_src["retriever_tabular_features"]),
        )
        campaign_data[split] = sliced_ret
        cache_lineage[split] = {
            "purchase_cache_path": str(cache_path),
            "purchase_cache_sha256": file_digest(cache_path),
            "recorded_enrichment_identity": payload.get("enrichment_identity"),
            "rows": int(len(rows)),
            "dynamic_features_path": str(Path("data/interim/dynamic_campaign_v1") / f"{split}_features.npy"),
            "dynamic_features_sha256": dyn_manifest["feature_sha256"][split],
        }

    if not torch.equal(campaign_data["train"]["v"], campaign_data["val"]["v"]):
        raise ValueError("Train and val cached product_vectors differ.")

    # Explicit cohort / denominator checks matching reference.
    if len(campaign_data["train"]["u"]) != 26253:
        raise ValueError(f"Expected 26253 train rows, got {len(campaign_data['train']['u'])}.")
    if len(campaign_data["val"]["u"]) != 6191:
        raise ValueError(f"Expected 6191 val rows, got {len(campaign_data['val']['u'])}.")
    val_eligible = campaign_data["val"]["allowed"].sum(1)
    if int(val_eligible.min()) != 735 or int(val_eligible.max()) != 771:
        raise ValueError(
            f"Unexpected val eligible range: [{int(val_eligible.min())}, {int(val_eligible.max())}]"
        )
    total_new_val = int(sum(len(x) for x in campaign_data["val"]["source"]["new_positive_pids"]))
    if total_new_val != 1857:
        raise ValueError(f"Expected 1857 val new targets, got {total_new_val}.")

    meta = {
        "reference": ref_info,
        "data_identity": info["identity"],
        "enrichment_identity": expected_enrichment_identity,
        "dynamic_identity": dyn_manifest["identity"],
        "product_ids": list(info["product_ids"]),
        "cache_lineage": cache_lineage,
    }
    return campaign_data, info, meta


def load_verified_baseline_val_scores(expected_data_identity: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load and validate the saved 303-input baseline MLP validation score tensor."""
    val_scores_path = BASELINE_SELECTOR_RUN_DIR / "val_scores.pt"
    if not val_scores_path.is_file():
        raise FileNotFoundError(f"Baseline val_scores.pt missing: {val_scores_path}")
    payload = torch.load(val_scores_path, map_location="cpu", weights_only=False)
    if payload.get("selector_digest") != EXPECTED_SELECTOR_SHA256:
        raise ValueError(
            f"Baseline val_scores selector_digest mismatch: {payload.get('selector_digest')} != {EXPECTED_SELECTOR_SHA256}"
        )
    if payload.get("upstream_digest") != EXPECTED_RETRIEVER_SHA256:
        raise ValueError(
            f"Baseline val_scores upstream_digest mismatch: {payload.get('upstream_digest')} != {EXPECTED_RETRIEVER_SHA256}"
        )
    if payload.get("data_identity") != expected_data_identity:
        raise ValueError("Baseline val_scores data_identity mismatch.")
    scores = payload["scores"].detach().cpu()
    if scores.shape != (6191, 841):
        raise ValueError(f"Unexpected baseline val_scores shape: {tuple(scores.shape)}.")

    eval_json_path = BASELINE_EVAL_DIR / "evaluation.json"
    eval_json = json.loads(eval_json_path.read_text(encoding="utf-8")) if eval_json_path.is_file() else None
    history_path = BASELINE_SELECTOR_RUN_DIR / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else None
    return scores, {
        "val_scores_path": str(val_scores_path),
        "val_scores_sha256": file_digest(val_scores_path),
        "baseline_eval_json": eval_json,
        "history": history,
    }


def load_verified_prior_300_val_scores(expected_data_identity: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load and verify the saved 300-input no_repeat_no_month (uv_44) validation score tensor and history."""
    val_scores_path = PRIOR_300_RUN_DIR / "val_scores.pt"
    ckpt_path = PRIOR_300_RUN_DIR / "best.pt"
    result_path = PRIOR_300_RUN_DIR / "result.json"
    history_path = PRIOR_300_RUN_DIR / "history.json"
    summary_path = PRIOR_300_EVIDENCE_DIR / "summary.json"
    for path in (val_scores_path, ckpt_path, result_path, history_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required prior 300D artifact missing: {path}")

    ckpt_sha = file_digest(ckpt_path)
    if ckpt_sha != EXPECTED_PRIOR_300_SELECTOR_SHA256:
        raise ValueError(
            f"Prior 300D checkpoint hash mismatch: {ckpt_sha} != {EXPECTED_PRIOR_300_SELECTOR_SHA256}"
        )

    payload = torch.load(val_scores_path, map_location="cpu", weights_only=False)
    if payload.get("selector_digest") != EXPECTED_PRIOR_300_SELECTOR_SHA256:
        raise ValueError(
            f"Prior 300D val_scores selector_digest mismatch: {payload.get('selector_digest')} != {EXPECTED_PRIOR_300_SELECTOR_SHA256}"
        )
    if payload.get("upstream_digest") != EXPECTED_RETRIEVER_SHA256:
        raise ValueError(
            f"Prior 300D val_scores upstream_digest mismatch: {payload.get('upstream_digest')} != {EXPECTED_RETRIEVER_SHA256}"
        )
    if payload.get("data_identity") != expected_data_identity:
        raise ValueError("Prior 300D val_scores data_identity mismatch.")

    expected_300_spec = build_feature_ablation_spec(
        OMITTED_CONTEXT_FEATURES,
        name=VARIANT_NAME,
        recipe=RECIPE_NAME,
    )
    validate_input_contract(payload.get("input_contract"), build_input_contract(expected_300_spec))

    scores = payload["scores"].detach().cpu()
    if scores.shape != (6191, 841):
        raise ValueError(f"Unexpected prior 300D val_scores shape: {tuple(scores.shape)}.")

    result_json = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    return scores, {
        "selector_path": str(ckpt_path),
        "selector_sha256": ckpt_sha,
        "val_scores_path": str(val_scores_path),
        "val_scores_sha256": file_digest(val_scores_path),
        "result_json": result_json,
        "history": history,
        "summary_json": summary_json,
    }
