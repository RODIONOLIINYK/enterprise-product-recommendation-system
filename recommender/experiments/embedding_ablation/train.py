"""Isolated training and verification orchestration for product-text and embedding ablations (Stage 1 + uv_47 MLP only)."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional

import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.experiment_retrieval import metrics_from_scores
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.dynamic_selector.train import finish_run, run_directory
from recommender.experiments.static_enrichment.evaluate import stage1_cohort_metrics
from recommender.experiments.static_enrichment.train import (
    _load_stage1_campaign_data,
    encode_and_score_variant,
    fit_compact_stage1_selector,
    make_variant_model,
    train_stage1_variant,
)
from recommender.shared_data import atomic_torch_save
from recommender.train_ranker import checkpoint_digest
from recommender.train_retriever import get_device

from .config import (
    AblationSpec,
    BASELINE_CUSTOMER_EMBEDDINGS,
    BASELINE_PRODUCT_EMBEDDINGS,
    CLEAN_V2_U0_DIR,
    STATIC_ENRICHMENT_V1_MANIFEST,
)
from .evaluate import save_mlp_per_query_predictions


def verify_and_stage_clean_v2_baseline(
    baseline_spec: AblationSpec,
    train: Mapping[str, Any],
    val: Mapping[str, Any],
    info: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    clean_v2_dir: Path = CLEAN_V2_U0_DIR,
    static_v1_manifest_path: Optional[Path] = STATIC_ENRICHMENT_V1_MANIFEST,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only verify clean_v2 U0_t1_fam_clean checkpoint/caches and stage into isolated namespace."""
    src_dir = Path(clean_v2_dir)
    src_ckpt_path = src_dir / "best.pt"
    src_val_cache_path = src_dir / "val_purchase_cache.pt"
    src_train_cache_path = src_dir / "train_purchase_cache.pt"
    src_result_path = src_dir / "result.json"
    for p in (src_ckpt_path, src_val_cache_path, src_train_cache_path, src_result_path):
        if not p.is_file():
            raise FileNotFoundError(f"Verified clean_v2 baseline artifact missing: {p}")

    ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    src_result = json.loads(src_result_path.read_text(encoding="utf-8"))
    settings = ckpt.get("settings", {})
    spec_dict = settings.get("spec", {})
    if settings.get("variant") != "U0_t1_fam_clean" or settings.get("seed") != 42:
        raise ValueError("clean_v2 checkpoint is not U0_t1_fam_clean seed=42.")
    if settings.get("identity") != info["identity"]:
        raise ValueError("clean_v2 checkpoint data identity mismatch.")
    if settings.get("text_recipe_hash") != enrichment["text_recipe_hashes"]["t1_fam"]:
        raise ValueError("clean_v2 checkpoint t1_fam text_recipe_hash mismatch.")
    if tuple(spec_dict.get("product_embeddings", ())) != BASELINE_PRODUCT_EMBEDDINGS:
        raise ValueError("clean_v2 checkpoint product_embeddings mismatch.")
    if tuple(spec_dict.get("customer_embeddings", ())) != BASELINE_CUSTOMER_EMBEDDINGS:
        raise ValueError("clean_v2 checkpoint customer_embeddings mismatch.")
    if int(spec_dict.get("customer_tabular_features", -1)) != 0 or int(spec_dict.get("product_demand_features", -1)) != 12:
        raise ValueError("clean_v2 checkpoint clean-tower dimensions mismatch.")

    if static_v1_manifest_path is not None and Path(static_v1_manifest_path).is_file():
        v1_manifest = json.loads(Path(static_v1_manifest_path).read_text(encoding="utf-8"))
        v1_diag = v1_manifest.get("diagnostics", {})
        cur_diag = enrichment.get("diagnostics", {})
        if v1_diag.get("product_vocab_sizes") != cur_diag.get("product_vocab_sizes"):
            raise ValueError("Product vocab sizes differ from static_enrichment_v1 manifest.")
        if v1_diag.get("customer_vocab_sizes") != cur_diag.get("customer_vocab_sizes"):
            raise ValueError("Customer vocab sizes differ from static_enrichment_v1 manifest.")

    src_ckpt_sha256 = file_digest(src_ckpt_path)
    src_val_cache = torch.load(src_val_cache_path, map_location="cpu", weights_only=False)
    src_train_cache = torch.load(src_train_cache_path, map_location="cpu", weights_only=False)
    if src_val_cache.get("model_digest") != src_ckpt_sha256 or src_train_cache.get("model_digest") != src_ckpt_sha256:
        raise ValueError("clean_v2 purchase cache digest does not match best.pt.")
    if src_val_cache.get("data_identity") != info["identity"] or src_train_cache.get("data_identity") != info["identity"]:
        raise ValueError("clean_v2 purchase cache data identity mismatch.")

    dev = get_device(device)
    model = make_variant_model(baseline_spec.variant_spec, info, enrichment).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    vectors, val_customers, val_logits = encode_and_score_variant(
        model, baseline_spec.variant_spec, val, info, enrichment, "val", dev
    )
    replayed = metrics_from_scores(val_logits, val, val_source)
    if replayed != ckpt["metrics"]:
        raise AssertionError("clean_v2 baseline Stage 1 replay metrics mismatch.")

    # Verify compatibility against saved clean_v2 val_purchase_cache.pt
    torch.testing.assert_close(vectors, src_val_cache["product_vectors"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(val_customers, src_val_cache["customer_vectors"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(val_logits, src_val_cache["scores"], atol=1e-3, rtol=1e-3)

    _, train_customers, train_logits = encode_and_score_variant(
        model, baseline_spec.variant_spec, train, info, enrichment, "train", dev
    )
    torch.testing.assert_close(train_customers, src_train_cache["customer_vectors"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(train_logits, src_train_cache["scores"], atol=1e-3, rtol=1e-3)

    staged_settings = {
        **settings,
        "enrichment_identity": enrichment["identity"],
        "reused_from": str(src_ckpt_path),
        "reused_checkpoint_sha256": src_ckpt_sha256,
    }
    directory, existing = run_directory(output_root, f"stage1-{baseline_spec.name}-seed42", staged_settings)
    if existing is not None:
        return existing

    # Copy best.pt byte-for-byte so checkpoint_digest matches clean_v2 identically.
    shutil.copy2(src_ckpt_path, directory / "best.pt")
    if file_digest(directory / "best.pt") != src_ckpt_sha256:
        raise AssertionError("Copied baseline checkpoint digest mismatch.")

    cohorts = stage1_cohort_metrics(
        val_logits,
        val,
        val_source,
        enrichment["cohort_masks"],
        save_predictions=directory / "stage1_predictions.npz",
    )
    atomic_torch_save(
        {
            "cache_version": 1,
            "model_digest": src_ckpt_sha256,
            "data_identity": info["identity"],
            "enrichment_identity": enrichment["identity"],
            "customer_vectors": val_customers,
            "product_vectors": vectors,
            "scores": val_logits,
        },
        directory / "val_purchase_cache.pt",
    )
    atomic_torch_save(
        {
            "cache_version": 1,
            "model_digest": src_ckpt_sha256,
            "data_identity": info["identity"],
            "enrichment_identity": enrichment["identity"],
            "customer_vectors": train_customers,
            "product_vectors": vectors,
            "scores": train_logits,
        },
        directory / "train_purchase_cache.pt",
    )
    atomic_json(
        {
            "source_dir": str(src_dir),
            "source_checkpoint": str(src_ckpt_path),
            "source_checkpoint_sha256": src_ckpt_sha256,
            "verified_replay_metrics": replayed,
            "clean_v2_mutated": False,
        },
        directory / "baseline_provenance.json",
    )

    result = {
        "settings": staged_settings,
        "selected_epoch": ckpt["epoch"],
        "metrics": replayed,
        "cohorts": cohorts,
        "training_seconds": src_result.get("training_seconds", 0.0),
        "reused_baseline_upstream": True,
        "source_checkpoint": str(src_ckpt_path),
        "source_checkpoint_sha256": src_ckpt_sha256,
        "parameters": src_result.get("parameters", sum(p.numel() for p in model.parameters())),
        "trainable_parameters": src_result.get("trainable_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "device": str(dev),
        "training_discovery_rows": src_result.get("training_discovery_rows"),
        "selected_checkpoint": str(directory / "best.pt"),
    }
    finish_run(directory, staged_settings, result)
    return result


def run_ablation_variant(
    ablation_spec: AblationSpec,
    seed: int,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    stage1_epochs: int = 30,
    selector_epochs: int = 10,
    reuse_clean_v2_baseline: bool = True,
    clean_v2_dir: Path = CLEAN_V2_U0_DIR,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Stage 1 upstream training (or verified reuse/no-op) + CompactStage1Selector uv_47 fit."""
    dev = get_device(device)
    spec = ablation_spec.variant_spec
    text_diag = enrichment["diagnostics"]["text_recipes"][spec.text_recipe]

    # Exact no-op guard for text deletions that change zero rendered strings.
    if ablation_spec.group == "text_deletion" and text_diag.get("is_noop", False):
        return {
            "ablation": asdict(ablation_spec),
            "is_noop": True,
            "noop_reason": "Text field omission changed 0 rendered strings after token deduplication; reusing baseline results without retraining.",
            "text_diagnostics": text_diag,
            "stage1": None,
            "upstream_dot_fullpool": None,
            "selector_uv47": None,
            "mlp_per_query_predictions": None,
        }

    if ablation_spec.name == "U0_t1_fam_clean" and reuse_clean_v2_baseline and Path(clean_v2_dir).is_dir():
        stage1_res = verify_and_stage_clean_v2_baseline(
            ablation_spec,
            train_retrieval,
            val_retrieval,
            info,
            val_source,
            enrichment,
            output_root,
            clean_v2_dir=clean_v2_dir,
            device=str(dev),
        )
    else:
        stage1_res = train_stage1_variant(
            spec,
            seed,
            train_retrieval,
            val_retrieval,
            info,
            train_source,
            val_source,
            enrichment,
            output_root,
            epochs=stage1_epochs,
            device=str(dev),
            save_train_cache=True,
        )

    stage1_dir = Path(stage1_res["selected_checkpoint"]).parent
    cdata, _, _, stage1_digest = _load_stage1_campaign_data(
        spec,
        stage1_dir,
        train_retrieval,
        val_retrieval,
        info,
        train_source,
        val_source,
        enrichment,
        dev,
    )
    va = cdata["val"]
    dot_fullpool = ranking_metrics(va["dot_scores"], va, None)

    selector_res = fit_compact_stage1_selector(
        ablation_spec.name,
        spec,
        "uv_47",
        seed,
        train_retrieval,
        val_retrieval,
        info,
        train_source,
        val_source,
        enrichment,
        stage1_dir,
        output_root,
        epochs=selector_epochs,
        device=str(dev),
    )

    selector_dir = Path(selector_res["selected_checkpoint"]).parent
    val_scores_payload = torch.load(selector_dir / "val_scores.pt", map_location="cpu", weights_only=False)
    if val_scores_payload.get("upstream_digest") != stage1_digest:
        raise AssertionError(f"Selector val_scores upstream_digest mismatch for {ablation_spec.name}")
    if val_scores_payload.get("selector_digest") != checkpoint_digest(selector_dir / "best.pt"):
        raise AssertionError(f"Selector val_scores selector_digest mismatch for {ablation_spec.name}")

    per_query_npz = selector_dir / "mlp_fullpool_predictions.npz"
    per_query_summary = save_mlp_per_query_predictions(val_scores_payload["scores"], va, per_query_npz)

    # Cross-check exact consistency with fit_compact_stage1_selector metrics.
    for k in ("new_hits@25", "new_hits@50", "new_hits@100", "new_hits@10"):
        if per_query_summary[k] != selector_res["metrics"][k]:
            raise AssertionError(f"Per-query metric mismatch on {k} for {ablation_spec.name}")
    for k in ("recall@10", "ndcg@10", "new_recall@25", "new_recall@50", "new_recall@100"):
        if abs(per_query_summary[k] - selector_res["metrics"][k]) > 1e-6:
            raise AssertionError(f"Per-query float metric mismatch on {k} for {ablation_spec.name}")

    return {
        "ablation": asdict(ablation_spec),
        "is_noop": False,
        "text_diagnostics": text_diag,
        "stage1": stage1_res,
        "upstream_dot_fullpool": {
            "new_hits@25": dot_fullpool["new_hits@25"],
            "new_recall@25": dot_fullpool["new_recall@25"],
            "new_hits@50": dot_fullpool["new_hits@50"],
            "new_recall@50": dot_fullpool["new_recall@50"],
            "new_hits@100": dot_fullpool["new_hits@100"],
            "new_recall@100": dot_fullpool["new_recall@100"],
            "recall@10": dot_fullpool["recall@10"],
            "ndcg@10": dot_fullpool["ndcg@10"],
        },
        "selector_uv47": selector_res,
        "mlp_per_query_predictions": str(per_query_npz),
    }
