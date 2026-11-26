"""Isolated Stage 1 + uv_47 MLP training orchestration for description cleaning variants."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.embedding_ablation.evaluate import save_mlp_per_query_predictions
from recommender.experiments.static_enrichment.train import (
    _load_stage1_campaign_data,
    fit_compact_stage1_selector,
    train_stage1_variant,
)
from recommender.train_ranker import checkpoint_digest
from recommender.train_retriever import get_device

from .config import DescriptionVariantSpec
from .evaluate import evaluate_description_subgroups


def run_description_variant(
    variant: DescriptionVariantSpec,
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
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train clean Stage 1 towers from scratch, then fit CompactStage1Selector uv_47 (NO Stage 2)."""
    if variant.name == "no_brand_v1":
        raise ValueError("Baseline no_brand_v1 must be verified/replayed via verify_and_replay_baseline_reference, not retrained.")

    dev = get_device(device)
    spec = variant.variant_spec
    text_diag = enrichment["diagnostics"]["text_recipes"][spec.text_recipe]

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
        variant.name,
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
        raise AssertionError(f"Selector val_scores upstream_digest mismatch for {variant.name}")
    if val_scores_payload.get("selector_digest") != checkpoint_digest(selector_dir / "best.pt"):
        raise AssertionError(f"Selector val_scores selector_digest mismatch for {variant.name}")

    per_query_npz = selector_dir / "mlp_fullpool_predictions.npz"
    per_query_summary = save_mlp_per_query_predictions(val_scores_payload["scores"], va, per_query_npz)

    for k in ("new_hits@25", "new_hits@50", "new_hits@100", "new_hits@10"):
        if per_query_summary[k] != selector_res["metrics"][k]:
            raise AssertionError(f"Per-query metric mismatch on {k} for {variant.name}")
    for k in ("recall@10", "ndcg@10", "new_recall@25", "new_recall@50", "new_recall@100"):
        if abs(per_query_summary[k] - selector_res["metrics"][k]) > 1e-6:
            raise AssertionError(f"Per-query float metric mismatch on {k} for {variant.name}")

    subgroup_npz = selector_dir / "mlp_subgroup_predictions.npz"
    subgroups_html = evaluate_description_subgroups(
        val_scores_payload["scores"],
        va,
        enrichment["cohort_masks"]["has_description_html"],
        save_path=subgroup_npz,
    )
    subgroups_clean = evaluate_description_subgroups(
        val_scores_payload["scores"],
        va,
        enrichment["cohort_masks"]["has_description_clean"],
    )

    return {
        "variant": asdict(variant),
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
        "mlp_subgroup_predictions": str(subgroup_npz),
        "subgroups_html_mask": subgroups_html,
        "subgroups_clean_mask": subgroups_clean,
    }
