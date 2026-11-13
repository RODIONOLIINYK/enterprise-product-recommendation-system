"""Verified data, reference validation, and sealed quota100_50 slate/score caches for Stage 2."""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import sliced
from recommender.experiments.dynamic_selector.feature_schema import (
    NAMES as DYNAMIC_FEATURE_NAMES,
    contract as source_feature_contract,
)
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.mlp_feature_ablation.config import (
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
)
from recommender.experiments.mlp_feature_ablation.model import load_ablated_selector_checkpoint
from recommender.experiments.mlp_feature_ablation.schema import (
    FeatureAblationSpec,
    build_feature_ablation_spec,
    build_input_contract,
    validate_input_contract,
)
from recommender.experiments.static_enrichment.data import (
    load_dynamic_campaign_features_readonly,
    quantize_transferred_stats,
)
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_ranker import RankerDataset, collate_ranker_batch

from .config import (
    BATCH_SIZE,
    CACHE_ROOT,
    DEFAULT_DISCOVERY_RANKER,
    DEFAULT_RETRIEVER,
    E1_E5_RETRIEVAL_CACHE_DIR,
    EXPECTED_CONTRACT_DIGEST,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    EXPECTED_RETRIEVER_SHA256,
    EXPECTED_SELECTOR_DIM,
    EXPECTED_SELECTOR_RECIPE,
    EXPECTED_SELECTOR_SHA256,
    EXPECTED_SELECTOR_VARIANT,
    EXPECTED_TRAIN_PURCHASE_CACHE_SHA256,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SCORES_SHA256,
    EXPECTED_UPSTREAM_VARIANT,
    EXPECTED_VAL_NEW_TARGETS,
    EXPECTED_VAL_PURCHASE_CACHE_SHA256,
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SCORES_SHA256,
    EXPECTED_VAL_TOTAL_TARGETS,
    MIN_ELIGIBLE_SKUS,
    SCORE_FEATURE_NAME,
    SCORE_TRANSFORM_VERSION,
    SEED,
    SELECTOR_SOURCE_RUN_DIR,
    SLATE_POLICY,
    SLATE_SIZE,
    STAGE1_MLP_REF_MANIFEST,
    STAGE1_MLP_REF_POINTER,
    STAGE1_MLP_REF_RETRIEVER,
    STAGE1_MLP_REF_SELECTOR,
    UPSTREAM_STAGE1_RUN_DIR,
)
from .model import (
    gather_teacher_scores_by_canonical_indices,
    transform_teacher_logits_to_mlp_score,
)


SLATE_CACHE_VERSION = 1


def expected_teacher_spec() -> FeatureAblationSpec:
    """Canonical 301D FeatureAblatedStage1Selector spec for repeat-no-month-v1."""
    return build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )


def assert_dev_dates_strict(split: str, prediction_days: torch.Tensor) -> None:
    """Enforce zero held-out access (>= 2026-02-19) and split boundary invariance."""
    origin = pd.Timestamp("2025-01-01")
    train_cutoff_day = (TRAIN_CUTOFF - origin).days
    test_start_day = (TEST_START - origin).days
    if bool((prediction_days >= test_start_day).any()):
        raise ValueError(f"Held-out dates >= {TEST_START.date()} detected in {split} split.")
    if split == "train" and bool((prediction_days >= train_cutoff_day).any()):
        raise ValueError(f"Validation/test dates >= {TRAIN_CUTOFF.date()} detected in train split.")
    if split == "val" and bool((prediction_days < train_cutoff_day).any()):
        raise ValueError(f"Training dates < {TRAIN_CUTOFF.date()} detected in val split.")


def verify_repeat_no_month_reference() -> Dict[str, Any]:
    """Verify frozen repeat-no-month-v1 pointer, manifest, selector, retriever, and source caches."""
    required_paths = (
        STAGE1_MLP_REF_POINTER,
        STAGE1_MLP_REF_MANIFEST,
        STAGE1_MLP_REF_RETRIEVER,
        STAGE1_MLP_REF_SELECTOR,
        SELECTOR_SOURCE_RUN_DIR / "train_scores.pt",
        SELECTOR_SOURCE_RUN_DIR / "val_scores.pt",
        UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
        UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
        DEFAULT_RETRIEVER,
        DEFAULT_DISCOVERY_RANKER,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")

    ref_ptr = json.loads(STAGE1_MLP_REF_POINTER.read_text(encoding="utf-8"))
    if ref_ptr.get("active_reference_id") != EXPECTED_REFERENCE_ID:
        raise ValueError(
            f"Expected active_reference_id={EXPECTED_REFERENCE_ID!r}, got {ref_ptr.get('active_reference_id')!r}."
        )
    if ref_ptr.get("heldout_access") is not False:
        raise ValueError("Reference pointer must record heldout_access=false.")
    if ref_ptr.get("upstream_variant") != EXPECTED_UPSTREAM_VARIANT:
        raise ValueError(f"Unexpected upstream_variant: {ref_ptr.get('upstream_variant')!r}")
    if ref_ptr.get("variant") != EXPECTED_SELECTOR_VARIANT:
        raise ValueError(f"Unexpected selector variant: {ref_ptr.get('variant')!r}")
    if ref_ptr.get("selector_recipe") != EXPECTED_SELECTOR_RECIPE:
        raise ValueError(f"Unexpected selector_recipe: {ref_ptr.get('selector_recipe')!r}")
    if int(ref_ptr.get("selector_input_dim", -1)) != EXPECTED_SELECTOR_DIM:
        raise ValueError(f"Unexpected selector_input_dim: {ref_ptr.get('selector_input_dim')}")

    expected_hashes = {
        "reference_manifest": (STAGE1_MLP_REF_MANIFEST, EXPECTED_MANIFEST_SHA256),
        "best_retriever": (STAGE1_MLP_REF_RETRIEVER, EXPECTED_RETRIEVER_SHA256),
        "best_selector": (STAGE1_MLP_REF_SELECTOR, EXPECTED_SELECTOR_SHA256),
        "train_scores": (SELECTOR_SOURCE_RUN_DIR / "train_scores.pt", EXPECTED_TRAIN_SCORES_SHA256),
        "val_scores": (SELECTOR_SOURCE_RUN_DIR / "val_scores.pt", EXPECTED_VAL_SCORES_SHA256),
        "train_purchase_cache": (
            UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
            EXPECTED_TRAIN_PURCHASE_CACHE_SHA256,
        ),
        "val_purchase_cache": (
            UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
            EXPECTED_VAL_PURCHASE_CACHE_SHA256,
        ),
    }
    actual_hashes: Dict[str, str] = {"reference_json": file_digest(STAGE1_MLP_REF_POINTER)}
    for label, (path, exp_sha) in expected_hashes.items():
        actual = file_digest(path)
        if actual != exp_sha:
            raise ValueError(f"Hash mismatch for {label} ({path}): {actual} != {exp_sha}")
        actual_hashes[label] = actual

    manifest = json.loads(STAGE1_MLP_REF_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("reference_id") != EXPECTED_REFERENCE_ID:
        raise ValueError("Reference manifest reference_id mismatch.")

    spec = expected_teacher_spec()
    expected_contract = build_input_contract(spec)
    if expected_contract["contract_digest"] != EXPECTED_CONTRACT_DIGEST:
        raise AssertionError(
            f"Contract digest mismatch: {expected_contract['contract_digest']} != {EXPECTED_CONTRACT_DIGEST}."
        )
    teacher, selector_ckpt = load_ablated_selector_checkpoint(
        STAGE1_MLP_REF_SELECTOR,
        expected_spec=spec,
        map_location="cpu",
    )
    teacher.eval()
    teacher.requires_grad_(False)
    selector_upstream = selector_ckpt.get("upstream_digest") or selector_ckpt.get("settings", {}).get("upstream_digest")
    if selector_upstream != EXPECTED_RETRIEVER_SHA256:
        raise ValueError(
            f"Selector upstream_digest mismatch: {selector_upstream} != {EXPECTED_RETRIEVER_SHA256}."
        )
    if int(selector_ckpt.get("epoch", -1)) != int(ref_ptr["selector"]["selected_epoch"]):
        raise ValueError(
            f"Selector epoch mismatch: {selector_ckpt.get('epoch')} != {ref_ptr['selector']['selected_epoch']}."
        )

    actual_hashes["default_retriever"] = file_digest(DEFAULT_RETRIEVER)
    actual_hashes["default_discovery_ranker"] = file_digest(DEFAULT_DISCOVERY_RANKER)
    return {
        "reference_pointer": ref_ptr,
        "manifest": manifest,
        "teacher": teacher,
        "teacher_ckpt": selector_ckpt,
        "teacher_spec": spec,
        "teacher_input_contract": expected_contract,
        "hashes": actual_hashes,
    }


def load_verified_stage2_campaign_data(
    *,
    replay_spot_rows: int = 64,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Load verified precomputed u/v, 41D dynamic features, and frozen 301D MLP scores."""
    ref_info = verify_repeat_no_month_reference()
    manifest = ref_info["manifest"]
    teacher = ref_info["teacher"]
    expected_contract = ref_info["teacher_input_contract"]
    expected_enrichment_identity = manifest["source_identities"]["enrichment_identity"]

    for name in ("train.pt", "val.pt", "manifest.pt"):
        if not (E1_E5_RETRIEVAL_CACHE_DIR / name).is_file():
            raise FileNotFoundError(
                f"Required retrieval cache {E1_E5_RETRIEVAL_CACHE_DIR / name} missing; refusing raw regeneration."
            )

    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()
    if len(info["product_ids"]) != EXPECTED_PRODUCT_UNIVERSE:
        raise ValueError(
            f"Expected {EXPECTED_PRODUCT_UNIVERSE} product universe, got {len(info['product_ids'])}."
        )

    dyn_manifest, train_feat_mmap, val_feat_mmap = load_dynamic_campaign_features_readonly()
    if dyn_manifest.get("source_identity") != info["identity"]:
        raise ValueError("Dynamic campaign source_identity mismatch against e1_e5_retrieval_v1.")
    if dyn_manifest.get("feature_contract") != source_feature_contract():
        raise ValueError("Dynamic campaign feature_contract mismatch.")
    if tuple(dyn_manifest["feature_contract"]["names"]) != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError("Dynamic campaign feature names do not match ordered canonical 41-col schema.")
    if list(dyn_manifest.get("product_ids", [])) != list(info["product_ids"]):
        raise ValueError("Dynamic campaign product_ids do not match canonical retrieval product_ids.")

    diagnostic_catalogue = load_product_catalogue(get_datasets_locs("items")).set_index("product_id")
    family_values = diagnostic_catalogue["product_family"].reindex(info["product_ids"]).fillna("").astype(str)
    family_known = torch.tensor(family_values.str.strip().ne("").to_numpy(dtype=bool, copy=True))

    campaign_data: Dict[str, Dict[str, Any]] = {}
    cache_lineage: Dict[str, Any] = {}
    for split, ret_data, src_data, feat_mmap, expected_rows in (
        ("train", train_ret, train_src, train_feat_mmap, EXPECTED_TRAIN_ROWS),
        ("val", val_ret, val_src, val_feat_mmap, EXPECTED_VAL_ROWS),
    ):
        purchase_cache_path = UPSTREAM_STAGE1_RUN_DIR / f"{split}_purchase_cache.pt"
        payload = torch.load(purchase_cache_path, map_location="cpu", weights_only=False)
        if payload.get("model_digest") != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(f"{purchase_cache_path} model_digest mismatch.")
        if payload.get("data_identity") != info["identity"]:
            raise ValueError(f"{purchase_cache_path} data_identity mismatch.")
        if split == "val" and payload.get("enrichment_identity") != expected_enrichment_identity:
            raise ValueError(f"{purchase_cache_path} enrichment_identity mismatch.")

        scores_path = SELECTOR_SOURCE_RUN_DIR / f"{split}_scores.pt"
        scores_payload = torch.load(scores_path, map_location="cpu", weights_only=False)
        if scores_payload.get("selector_digest") != EXPECTED_SELECTOR_SHA256:
            raise ValueError(f"{scores_path} selector_digest mismatch.")
        if scores_payload.get("upstream_digest") != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(f"{scores_path} upstream_digest mismatch.")
        if scores_payload.get("data_identity") != info["identity"]:
            raise ValueError(f"{scores_path} data_identity mismatch.")
        validate_input_contract(scores_payload.get("input_contract"), expected_contract)

        rows = torch.where(ret_data["allowed"].sum(1) >= MIN_ELIGIBLE_SKUS)[0]
        if len(rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} {split} rows, got {len(rows)}.")
        sliced_src = sliced(src_data, rows, len(src_data["customer_ids"]))
        sliced_ret = sliced(ret_data, rows, len(src_data["customer_ids"]))
        assert_dev_dates_strict(split, sliced_src["prediction_days"])

        u = payload["customer_vectors"][rows].squeeze(1).contiguous()
        v = payload["product_vectors"].contiguous()
        raw_teacher_scores = scores_payload["scores"].detach().cpu().to(dtype=torch.float32).contiguous()

        if u.shape != (expected_rows, 128) or v.shape != (EXPECTED_PRODUCT_UNIVERSE, 128):
            raise ValueError(f"Unexpected u/v shape for {split}: u={tuple(u.shape)}, v={tuple(v.shape)}.")
        if raw_teacher_scores.shape != (expected_rows, EXPECTED_PRODUCT_UNIVERSE):
            raise ValueError(
                f"Unexpected {split}_scores.pt shape: {tuple(raw_teacher_scores.shape)} != ({expected_rows}, {EXPECTED_PRODUCT_UNIVERSE})."
            )
        if feat_mmap.shape != (expected_rows, EXPECTED_PRODUCT_UNIVERSE, 41):
            raise ValueError(f"Unexpected {split}_features.npy shape: {feat_mmap.shape}.")

        customer_stats_6 = quantize_transferred_stats(sliced_src["retriever_tabular_features"])
        if customer_stats_6.shape != (expected_rows, 6):
            raise ValueError(f"Unexpected customer_stats_6 shape: {tuple(customer_stats_6.shape)}.")

        # Validate saved raw scores against allowed eligibility mask: eligible entries must be finite.
        eligible_logits = raw_teacher_scores[sliced_ret["allowed"]]
        if not bool(torch.isfinite(eligible_logits).all()):
            raise ValueError(f"Non-finite teacher logits found on eligible products in {split}_scores.pt.")

        sliced_ret.update(
            source=sliced_src,
            cohort_rows=rows,
            u=u,
            v=v,
            raw_teacher_scores=raw_teacher_scores,
            features=feat_mmap,
            customer_stats_6=customer_stats_6,
            family_known=family_known,
        )
        campaign_data[split] = sliced_ret
        cache_lineage[split] = {
            "purchase_cache_path": str(purchase_cache_path),
            "purchase_cache_sha256": ref_info["hashes"][f"{split}_purchase_cache"],
            "teacher_scores_path": str(scores_path),
            "teacher_scores_sha256": ref_info["hashes"][f"{split}_scores"],
            "dynamic_features_path": str(Path("data/interim/dynamic_campaign_v1") / f"{split}_features.npy"),
            "dynamic_features_sha256": dyn_manifest["feature_sha256"][split],
            "rows": int(expected_rows),
        }

    if not torch.equal(campaign_data["train"]["v"], campaign_data["val"]["v"]):
        raise ValueError("Train and val cached product_vectors differ.")

    total_val_targets = int(sum(len(x) for x in campaign_data["val"]["source"]["positive_pids"]))
    total_val_new = int(sum(len(x) for x in campaign_data["val"]["source"]["new_positive_pids"]))
    if total_val_targets != EXPECTED_VAL_TOTAL_TARGETS:
        raise ValueError(f"Expected {EXPECTED_VAL_TOTAL_TARGETS} val total targets, got {total_val_targets}.")
    if total_val_new != EXPECTED_VAL_NEW_TARGETS:
        raise ValueError(f"Expected {EXPECTED_VAL_NEW_TARGETS} val new targets, got {total_val_new}.")

    # Spot-check frozen 301D teacher replay on validation rows (eval() / no_grad / frozen params).
    if any(p.requires_grad for p in teacher.parameters()):
        raise AssertionError("Frozen teacher must have requires_grad=False on all parameters.")
    val_spot = campaign_data["val"]
    n_spot = min(int(replay_spot_rows), len(val_spot["u"]))
    with torch.no_grad():
        teacher.eval()
        f41_spot = torch.from_numpy(np.array(val_spot["features"][:n_spot], dtype=np.float32))
        replayed_logits = teacher(
            u=val_spot["u"][:n_spot],
            v=val_spot["v"],
            features_41=f41_spot,
            customer_stats_6=val_spot["customer_stats_6"][:n_spot],
        )
        spot_allowed = val_spot["allowed"][:n_spot]
        saved_logits = val_spot["raw_teacher_scores"][:n_spot]
        max_abs_replay_diff = float((replayed_logits[spot_allowed] - saved_logits[spot_allowed]).abs().max().item())
        if max_abs_replay_diff > 1e-4:
            raise AssertionError(
                f"Teacher checkpoint replay diff {max_abs_replay_diff:.6e} exceeded 1e-4 vs val_scores.pt."
            )
        replayed_prob = transform_teacher_logits_to_mlp_score(replayed_logits, valid_mask=spot_allowed)
        saved_prob = transform_teacher_logits_to_mlp_score(saved_logits, valid_mask=spot_allowed)
        max_prob_replay_diff = float((replayed_prob - saved_prob).abs().max().item())

    meta = {
        "reference": ref_info,
        "data_identity": info["identity"],
        "enrichment_identity": expected_enrichment_identity,
        "dynamic_identity": dyn_manifest["identity"],
        "product_ids": list(info["product_ids"]),
        "cache_lineage": cache_lineage,
        "teacher_replay_spot_check": {
            "rows_checked": n_spot,
            "max_abs_logit_diff": max_abs_replay_diff,
            "max_abs_sigmoid_diff": max_prob_replay_diff,
        },
    }
    return campaign_data, info, meta


def _build_slate_payload(
    split: str,
    data: Mapping[str, Any],
    product_ids: Sequence[str],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_scores = data["raw_teacher_scores"]
    indices = select_candidates(raw_scores, data["allowed"], data["unseen"], SLATE_POLICY)
    if indices.shape != (len(data["u"]), SLATE_SIZE):
        raise ValueError(f"Expected {split} slate shape ({len(data['u'])}, {SLATE_SIZE}), got {tuple(indices.shape)}.")
    # Every row must contain SLATE_SIZE unique eligible candidates.
    eligible_selected = data["allowed"].gather(1, indices)
    if not bool(eligible_selected.all()):
        raise ValueError(f"Ineligible candidate selected in {split} slates.")
    if any(len(set(row)) != SLATE_SIZE for row in indices.tolist()):
        raise ValueError(f"Duplicate candidate index in {split} slates.")

    mlp_score, masked_raw_logits = gather_teacher_scores_by_canonical_indices(
        raw_scores,
        indices,
        canonical_product_ids=product_ids,
        score_product_ids=product_ids,
        valid_mask=eligible_selected,
    )
    if not bool(torch.isfinite(masked_raw_logits).all()):
        raise ValueError(f"Non-finite gathered teacher logits in {split} slates.")

    src = data["source"]
    row_arange = np.arange(len(indices))[:, None]
    pairwise_features = torch.from_numpy(
        np.array(data["features"][row_arange, indices.numpy()], dtype=np.float32)
    ).contiguous()
    if not bool(torch.isfinite(pairwise_features).all()):
        raise ValueError(f"Non-finite pairwise_features in {split} slates.")

    payload: Dict[str, Any] = {
        "cache_version": SLATE_CACHE_VERSION,
        "split": split,
        "policy": SLATE_POLICY,
        "slate_size": SLATE_SIZE,
        "score_feature_name": SCORE_FEATURE_NAME,
        "score_transform_version": SCORE_TRANSFORM_VERSION,
        "teacher_sha256": meta["reference"]["hashes"]["best_selector"],
        "upstream_sha256": meta["reference"]["hashes"]["best_retriever"],
        "teacher_scores_sha256": meta["cache_lineage"][split]["teacher_scores_sha256"],
        "dynamic_features_sha256": meta["cache_lineage"][split]["dynamic_features_sha256"],
        "data_identity": meta["data_identity"],
        "dynamic_identity": meta["dynamic_identity"],
        "contract_digest": meta["reference"]["teacher_input_contract"]["contract_digest"],
        "product_ids": list(product_ids),
        "transition_keys": list(src["transition_keys"]),
        "customer_ids": list(src["customer_ids"]),
        "positive_pids": list(src["positive_pids"]),
        "new_positive_pids": list(src["new_positive_pids"]),
        "prediction_days": src["prediction_days"].clone(),
        "text_embedding": src["text_embedding"].clone(),
        "city_id": src["city_id"].clone(),
        "region_id": src["region_id"].clone(),
        "industry_id": src["industry_id"].clone(),
        "tabular_features": src["tabular_features"].clone(),
        "retriever_tabular_features": src["retriever_tabular_features"].clone(),
        "sequence": src["sequence"].clone(),
        "attention_mask": src["attention_mask"].clone(),
        "seq_product_ids": src["seq_product_ids"].clone(),
        "seq_purchase_dates": src["seq_purchase_dates"].clone(),
        "seq_category_ids": src["seq_category_ids"].clone(),
        "seq_subcategory_ids": src["seq_subcategory_ids"].clone(),
        "seq_business_line_ids": src["seq_business_line_ids"].clone(),
        "stage1_customer_vectors": data["u"].clone(),
        "candidate_indices": indices.to(dtype=torch.long).contiguous(),
        "candidate_mask": eligible_selected.to(dtype=torch.bool).contiguous(),
        "candidate_pids": [[product_ids[i] for i in row] for row in indices.tolist()],
        "pairwise_features": pairwise_features,
        "mlp_score": mlp_score.to(dtype=torch.float32).contiguous(),
        "teacher_raw_logit": masked_raw_logits.to(dtype=torch.float32).contiguous(),
        "labels": data["target"].gather(1, indices).to(dtype=torch.float32).contiguous(),
    }
    cache_identity = json_digest(
        {
            "cache_version": SLATE_CACHE_VERSION,
            "split": split,
            "policy": SLATE_POLICY,
            "slate_size": SLATE_SIZE,
            "score_feature_name": SCORE_FEATURE_NAME,
            "score_transform_version": SCORE_TRANSFORM_VERSION,
            "teacher_sha256": payload["teacher_sha256"],
            "upstream_sha256": payload["upstream_sha256"],
            "teacher_scores_sha256": payload["teacher_scores_sha256"],
            "dynamic_features_sha256": payload["dynamic_features_sha256"],
            "data_identity": payload["data_identity"],
            "dynamic_identity": payload["dynamic_identity"],
            "contract_digest": payload["contract_digest"],
            "product_ids": payload["product_ids"],
            "transition_keys": payload["transition_keys"],
            "indices": indices.tolist(),
        }
    )
    payload["cache_identity"] = cache_identity
    return payload


def _validate_loaded_slate_payload(
    split: str,
    payload: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    expected_rows: int,
) -> None:
    if payload.get("cache_version") != SLATE_CACHE_VERSION:
        raise ValueError(f"Slate cache version mismatch for {split}.")
    if payload.get("split") != split or payload.get("policy") != SLATE_POLICY:
        raise ValueError(f"Slate cache split/policy mismatch for {split}.")
    if payload.get("score_transform_version") != SCORE_TRANSFORM_VERSION:
        raise ValueError(f"Slate cache score_transform_version mismatch for {split}.")
    for key, val in expected_identity.items():
        if payload.get(key) != val:
            raise ValueError(f"Slate cache {split} provenance mismatch on {key!r}: {payload.get(key)!r} != {val!r}.")
    assert_dev_dates_strict(split, payload["prediction_days"])
    if payload["candidate_indices"].shape != (expected_rows, SLATE_SIZE):
        raise ValueError(f"Slate cache {split} candidate_indices shape mismatch.")
    if payload["pairwise_features"].shape != (expected_rows, SLATE_SIZE, 41):
        raise ValueError(f"Slate cache {split} pairwise_features shape mismatch.")
    if payload["mlp_score"].shape != (expected_rows, SLATE_SIZE):
        raise ValueError(f"Slate cache {split} mlp_score shape mismatch.")
    scores = payload["mlp_score"]
    if not bool(torch.isfinite(scores).all()) or not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
        raise ValueError(f"Slate cache {split} mlp_score out of [0, 1] or non-finite.")


def prepare_or_load_stage2_slate_cache(
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    cache_root: Path = CACHE_ROOT,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Build or strictly validate sealed train/val quota100_50 + sigmoid(mlp_score) caches."""
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    product_ids = list(info["product_ids"])

    expected_shared = {
        "teacher_sha256": meta["reference"]["hashes"]["best_selector"],
        "upstream_sha256": meta["reference"]["hashes"]["best_retriever"],
        "data_identity": meta["data_identity"],
        "dynamic_identity": meta["dynamic_identity"],
        "contract_digest": meta["reference"]["teacher_input_contract"]["contract_digest"],
        "product_ids": product_ids,
    }

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("heldout_access") is not False:
            raise ValueError("Slate cache manifest must record heldout_access=false.")
        if manifest.get("policy") != SLATE_POLICY or manifest.get("score_transform_version") != SCORE_TRANSFORM_VERSION:
            raise ValueError("Slate cache manifest policy/transform mismatch.")
        for key, val in expected_shared.items():
            if key == "product_ids":
                continue
            if manifest.get(key) != val:
                raise ValueError(f"Slate cache manifest mismatch on {key!r}.")
        slates: Dict[str, Dict[str, Any]] = {}
        for split in ("train", "val"):
            split_path = root / f"{split}_slate_scores.pt"
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing slate cache split file: {split_path}")
            actual_sha = file_digest(split_path)
            if actual_sha != manifest["split_sha256"][split]:
                raise ValueError(f"Slate cache SHA-256 mismatch for {split}: {actual_sha} != {manifest['split_sha256'][split]}")
            payload = torch.load(split_path, map_location="cpu", weights_only=False)
            exp_split = {
                **expected_shared,
                "teacher_scores_sha256": meta["cache_lineage"][split]["teacher_scores_sha256"],
                "dynamic_features_sha256": meta["cache_lineage"][split]["dynamic_features_sha256"],
                "transition_keys": list(campaign_data[split]["source"]["transition_keys"]),
            }
            _validate_loaded_slate_payload(split, payload, exp_split, len(campaign_data[split]["u"]))
            if payload.get("cache_identity") != manifest["split_cache_identity"][split]:
                raise ValueError(f"Slate cache_identity mismatch for {split}.")
            # Also verify saved mlp_score matches raw logits transformed via sigmoid.
            recomputed_score, _ = gather_teacher_scores_by_canonical_indices(
                campaign_data[split]["raw_teacher_scores"],
                payload["candidate_indices"],
                canonical_product_ids=product_ids,
                valid_mask=payload["candidate_mask"],
            )
            torch.testing.assert_close(payload["mlp_score"], recomputed_score, atol=0.0, rtol=0.0)
            slates[split] = payload
        return slates, manifest

    if any(root.glob("*_slate_scores.pt")):
        raise ValueError(f"Incomplete slate cache in {root}; refusing silent overwrite.")

    slates = {}
    split_sha256: Dict[str, str] = {}
    split_cache_identity: Dict[str, str] = {}
    split_stats: Dict[str, Any] = {}
    for split in ("train", "val"):
        payload = _build_slate_payload(split, campaign_data[split], product_ids, meta)
        out_path = root / f"{split}_slate_scores.pt"
        atomic_torch_save(payload, out_path)
        slates[split] = payload
        split_sha256[split] = file_digest(out_path)
        split_cache_identity[split] = payload["cache_identity"]
        split_stats[split] = {
            "rows": int(len(payload["customer_ids"])),
            "slate_size": SLATE_SIZE,
            "mlp_score_min": float(payload["mlp_score"].min().item()),
            "mlp_score_max": float(payload["mlp_score"].max().item()),
            "mlp_score_mean": float(payload["mlp_score"].mean().item()),
            "raw_logit_min": float(payload["teacher_raw_logit"].min().item()),
            "raw_logit_max": float(payload["teacher_raw_logit"].max().item()),
        }

    manifest = {
        "cache_version": SLATE_CACHE_VERSION,
        "heldout_access": False,
        "policy": SLATE_POLICY,
        "slate_size": SLATE_SIZE,
        "score_feature_name": SCORE_FEATURE_NAME,
        "score_transform_version": SCORE_TRANSFORM_VERSION,
        "teacher_sha256": expected_shared["teacher_sha256"],
        "upstream_sha256": expected_shared["upstream_sha256"],
        "data_identity": expected_shared["data_identity"],
        "dynamic_identity": expected_shared["dynamic_identity"],
        "contract_digest": expected_shared["contract_digest"],
        "teacher_scores_sha256": {
            s: meta["cache_lineage"][s]["teacher_scores_sha256"] for s in ("train", "val")
        },
        "dynamic_features_sha256": {
            s: meta["cache_lineage"][s]["dynamic_features_sha256"] for s in ("train", "val")
        },
        "split_sha256": split_sha256,
        "split_cache_identity": split_cache_identity,
        "split_stats": split_stats,
        "product_universe": len(product_ids),
    }
    atomic_json(manifest, manifest_path, overwrite=False)
    return slates, manifest


class Stage2MLPScoreDataset(RankerDataset):
    """RankerDataset extended with per-candidate bounded mlp_score and candidate_mask."""

    def __init__(self, data: Mapping[str, Any]):
        super().__init__(dict(data))
        self.mlp_score = data["mlp_score"]
        self.candidate_mask = data.get("candidate_mask")
        self.teacher_raw_logit = data.get("teacher_raw_logit")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item["mlp_score"] = self.mlp_score[idx]
        item["candidate_mask"] = (
            self.candidate_mask[idx]
            if self.candidate_mask is not None
            else torch.ones_like(self.candidate_indices[idx], dtype=torch.bool)
        )
        if self.teacher_raw_logit is not None:
            item["teacher_raw_logit"] = self.teacher_raw_logit[idx]
        return item


def collate_stage2_mlp_score_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate Stage 2 slate batch preserving mlp_score and candidate_mask."""
    collated = collate_ranker_batch(batch)
    collated["mlp_score"] = torch.stack([x["mlp_score"] for x in batch], dim=0)
    collated["candidate_mask"] = torch.stack([x["candidate_mask"] for x in batch], dim=0)
    if batch[0].get("teacher_raw_logit") is not None:
        collated["teacher_raw_logit"] = torch.stack([x["teacher_raw_logit"] for x in batch], dim=0)
    return collated


def stage2_mlp_score_batch_inputs(
    batch: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Prepare device tensors for Stage2MLPScoreRanker.forward from a collated batch."""
    names = (
        "text_embedding",
        "city_id",
        "region_id",
        "industry_id",
        "tabular_features",
        "sequence",
        "attention_mask",
        "seq_product_ids",
        "seq_category_ids",
        "seq_subcategory_ids",
        "seq_business_line_ids",
        "pairwise_features",
        "seq_purchase_dates",
        "prediction_day",
        "mlp_score",
        "candidate_mask",
    )
    inputs: Dict[str, Any] = {
        name: batch[name].to(device) if batch.get(name) is not None else None
        for name in names
    }
    vectors = batch.get("candidate_vectors")
    if vectors is None:
        cand_idx = batch["candidate_indices"].to(device)
        mask = inputs["candidate_mask"]
        safe_idx = torch.where(mask, cand_idx, cand_idx.new_zeros(())) if mask is not None else cand_idx
        inputs["candidate_vectors"] = product_vectors[safe_idx]
    else:
        inputs["candidate_vectors"] = vectors.to(device)
    if batch.get("stage1_customer_vectors") is not None:
        inputs["cached_customer_vectors"] = batch["stage1_customer_vectors"].to(device)
    return inputs


def make_stage2_loader(
    slate: Mapping[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = SEED,
) -> DataLoader:
    """Create deterministic DataLoader matching existing Stage 2 shuffle order."""
    return DataLoader(
        Stage2MLPScoreDataset(slate),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collate_stage2_mlp_score_batch,
        generator=torch.Generator().manual_seed(int(seed)),
    )
