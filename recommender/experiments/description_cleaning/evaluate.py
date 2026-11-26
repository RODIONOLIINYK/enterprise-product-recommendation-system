"""Baseline reference provenance/replay verification and subgroup full-pool evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from recommender.artifacts import file_digest
from recommender.experiment_retrieval import metrics_from_scores
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.embedding_ablation.evaluate import (
    paired_ablation_intervals,
    save_mlp_per_query_predictions,
)
from recommender.experiments.static_enrichment.model import CompactStage1Selector
from recommender.experiments.static_enrichment.train import (
    _load_stage1_campaign_data,
    _score_compact_selector_split,
    encode_and_score_variant,
    make_variant_model,
)
from recommender.train_ranker import checkpoint_digest
from recommender.train_retriever import get_device

from .config import BASELINE_REFERENCE_DIR, BASELINE_REFERENCE_JSON, build_description_variant_specs


def evaluate_description_subgroups(
    scores: torch.Tensor,
    val_campaign_data: Mapping[str, Any],
    has_description_mask: torch.Tensor,
    *,
    save_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Evaluate full-pool MLP predictions overall and across targets WITH vs WITHOUT descriptions.

    Always keeps the full eligible candidate pool fixed (no description-only candidate filtering).
    """
    scores = scores.detach().cpu()
    allowed = val_campaign_data["allowed"].cpu()
    unseen = val_campaign_data["unseen"].cpu()
    target = val_campaign_data["target"].cpu()
    new_target = val_campaign_data["new_target"].cpu()
    source = val_campaign_data["source"]
    desc_mask = has_description_mask.detach().cpu().bool()

    if scores.shape != allowed.shape or desc_mask.shape != (scores.shape[1],):
        raise ValueError("Shape mismatch in evaluate_description_subgroups.")

    # Full-pool overall rankings (allowed)
    masked = scores.masked_fill(~allowed, -torch.inf)
    order = masked.argsort(dim=-1, descending=True, stable=True)
    indices = torch.arange(scores.shape[1]).expand(len(scores), -1)
    ranked = indices.gather(1, order)
    eligible_ranked = allowed.gather(1, ranked)

    hits_all = (target.gather(1, ranked) & eligible_ranked).numpy()
    k10 = min(10, ranked.shape[1])
    top10_ranked = ranked[:, :k10]
    top10_novel = (new_target.gather(1, top10_ranked) & eligible_ranked[:, :k10]).numpy()
    top10_desc = desc_mask[top10_ranked].numpy()

    count = np.array([len(x) for x in source["positive_pids"]], dtype=np.float64)
    new_count = np.array([len(x) for x in source["new_positive_pids"]], dtype=np.int64)
    recall10 = hits_all[:, :k10].sum(1) / count
    discount = 1.0 / np.log2(np.arange(k10) + 2.0)
    ideal = (discount[None] * (np.arange(k10)[None] < count[:, None])).sum(1)
    ndcg10 = (hits_all[:, :k10] * discount).sum(1) / ideal

    # Unseen rankings over full unseen eligible pool (unseen = allowed & ~seen)
    unseen_scores = scores.masked_fill(~unseen, -torch.inf)
    unseen_order = unseen_scores.argsort(dim=-1, descending=True, stable=True)
    k100 = min(100, unseen_order.shape[1])
    top100 = unseen_order[:, :k100]
    valid100 = unseen.gather(1, top100)
    novel100 = (new_target.gather(1, top100) & valid100).numpy()
    desc100 = desc_mask[top100].numpy()

    # Per-query target denominators for WITH vs WITHOUT description new targets
    new_target_with_desc_mat = new_target & desc_mask.unsqueeze(0)
    new_target_without_desc_mat = new_target & (~desc_mask).unsqueeze(0)
    with_desc_targets = new_target_with_desc_mat.sum(dim=1).numpy().astype(np.int64)
    without_desc_targets = new_target_without_desc_mat.sum(dim=1).numpy().astype(np.int64)

    # Check exact partition of represented new targets against source denominator
    total_new = int(new_count.sum())
    if int((with_desc_targets + without_desc_targets).sum()) != total_new:
        raise ValueError("Subgroup new-target counts do not sum to total new_targets denominator.")

    with_hits10 = (top10_novel & top10_desc).sum(1).astype(np.int64)
    without_hits10 = (top10_novel & (~top10_desc)).sum(1).astype(np.int64)
    all_hits10 = (top10_novel).sum(1).astype(np.int64)

    with_hits25 = (novel100[:, : min(25, k100)] & desc100[:, : min(25, k100)]).sum(1).astype(np.int64)
    with_hits50 = (novel100[:, : min(50, k100)] & desc100[:, : min(50, k100)]).sum(1).astype(np.int64)
    with_hits100 = (novel100[:, :k100] & desc100[:, :k100]).sum(1).astype(np.int64)

    without_hits25 = (novel100[:, : min(25, k100)] & (~desc100[:, : min(25, k100)])).sum(1).astype(np.int64)
    without_hits50 = (novel100[:, : min(50, k100)] & (~desc100[:, : min(50, k100)])).sum(1).astype(np.int64)
    without_hits100 = (novel100[:, :k100] & (~desc100[:, :k100])).sum(1).astype(np.int64)

    all_hits25 = novel100[:, : min(25, k100)].sum(1).astype(np.int64)
    all_hits50 = novel100[:, : min(50, k100)].sum(1).astype(np.int64)
    all_hits100 = novel100[:, :k100].sum(1).astype(np.int64)

    # Query-level overall subgroup (whether any positive target in basket has description)
    query_has_desc_target = (target & desc_mask.unsqueeze(0)).any(dim=1).numpy()

    def _unseen_summary(h10: np.ndarray, h25: np.ndarray, h50: np.ndarray, h100: np.ndarray, den: np.ndarray) -> Dict[str, Any]:
        tot = int(den.sum())
        return {
            "new_targets": tot,
            "new_hits@10": int(h10.sum()),
            "new_recall@10": float(h10.sum() / tot) if tot else None,
            "new_hits@25": int(h25.sum()),
            "new_recall@25": float(h25.sum() / tot) if tot else None,
            "new_hits@50": int(h50.sum()),
            "new_recall@50": float(h50.sum() / tot) if tot else None,
            "new_hits@100": int(h100.sum()),
            "new_recall@100": float(h100.sum() / tot) if tot else None,
        }

    def _query_overall_summary(mask: np.ndarray) -> Dict[str, Any]:
        n_q = int(mask.sum())
        return {
            "queries": n_q,
            "recall@10": float(recall10[mask].mean()) if n_q else None,
            "ndcg@10": float(ndcg10[mask].mean()) if n_q else None,
        }

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            customers=np.array(source["customer_ids"]),
            recall=recall10,
            ndcg=ndcg10,
            new_hits=all_hits10,
            new_hits25=all_hits25,
            new_hits50=all_hits50,
            new_hits100=all_hits100,
            new_targets=new_count,
            with_desc_hits10=with_hits10,
            with_desc_hits25=with_hits25,
            with_desc_hits50=with_hits50,
            with_desc_hits100=with_hits100,
            with_desc_targets=with_desc_targets,
            without_desc_hits10=without_hits10,
            without_desc_hits25=without_hits25,
            without_desc_hits50=without_hits50,
            without_desc_hits100=without_hits100,
            without_desc_targets=without_desc_targets,
            query_has_desc_target=query_has_desc_target,
            positive_counts=count.astype(np.int64),
        )

    return {
        "pool_policy": "full_rolling_eligible_pool_fixed",
        "overall": {
            "queries": int(len(scores)),
            "recall@10": float(recall10.mean()),
            "ndcg@10": float(ndcg10.mean()),
            **_unseen_summary(all_hits10, all_hits25, all_hits50, all_hits100, new_count),
        },
        "new_target_subgroups": {
            "with_description": _unseen_summary(
                with_hits10, with_hits25, with_hits50, with_hits100, with_desc_targets
            ),
            "without_description": _unseen_summary(
                without_hits10, without_hits25, without_hits50, without_hits100, without_desc_targets
            ),
        },
        "query_overall_subgroups": {
            "queries_with_described_positive": _query_overall_summary(query_has_desc_target),
            "queries_without_described_positive": _query_overall_summary(~query_has_desc_target),
        },
    }


def paired_subgroup_intervals(
    control_subgroup_npz: Path | str,
    treatment_subgroup_npz: Path | str,
    *,
    repeats: int = 2000,
) -> Dict[str, Dict[str, list[float]]]:
    """Paired customer-clustered 95% bootstrap CIs for overall and description subgroups."""
    a = np.load(control_subgroup_npz)
    b = np.load(treatment_subgroup_npz)
    for k in ("customers", "new_targets", "with_desc_targets", "without_desc_targets"):
        if not np.array_equal(a[k], b[k]):
            raise ValueError(f"Paired subgroup predictions mismatch on {k}.")

    _, groups = np.unique(a["customers"], return_inverse=True)
    n_cust = int(groups.max()) + 1
    rows_denom = np.bincount(groups)
    all_new_denom = np.bincount(groups, weights=a["new_targets"])
    with_denom = np.bincount(groups, weights=a["with_desc_targets"])
    without_denom = np.bincount(groups, weights=a["without_desc_targets"])

    rng = np.random.default_rng(123)
    draws = rng.integers(n_cust, size=(repeats, n_cust))

    def _boot(specs: tuple[tuple[str, str, np.ndarray], ...]) -> Dict[str, list[float]]:
        out: Dict[str, list[float]] = {}
        for out_name, key, denom_vec in specs:
            grp_delta = np.bincount(groups, weights=b[key] - a[key])
            denom = np.maximum(denom_vec[draws].sum(1), 1.0)
            vals = grp_delta[draws].sum(1) / denom
            out[out_name] = np.quantile(vals, [0.025, 0.5, 0.975]).tolist()
        return out

    return {
        "overall": _boot((
            ("recall@10_delta", "recall", rows_denom),
            ("ndcg@10_delta", "ndcg", rows_denom),
            ("new_recall@10_delta", "new_hits", all_new_denom),
            ("new_recall@25_delta", "new_hits25", all_new_denom),
            ("new_recall@50_delta", "new_hits50", all_new_denom),
            ("new_recall@100_delta", "new_hits100", all_new_denom),
        )),
        "with_description": _boot((
            ("new_recall@10_delta", "with_desc_hits10", with_denom),
            ("new_recall@25_delta", "with_desc_hits25", with_denom),
            ("new_recall@50_delta", "with_desc_hits50", with_denom),
            ("new_recall@100_delta", "with_desc_hits100", with_denom),
        )),
        "without_description": _boot((
            ("new_recall@10_delta", "without_desc_hits10", without_denom),
            ("new_recall@25_delta", "without_desc_hits25", without_denom),
            ("new_recall@50_delta", "without_desc_hits50", without_denom),
            ("new_recall@100_delta", "without_desc_hits100", without_denom),
        )),
    }


def verify_hash_preservation() -> Dict[str, Any]:
    """Verify all 7 historical audit/artifacts.json hashes and all no-brand-v1 reference hashes."""
    audit_path = Path("audit/artifacts.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_verified: Dict[str, str] = {}
    for p_str, info in audit.items():
        actual = file_digest(p_str)
        expected = info["sha256"]
        if actual != expected:
            raise AssertionError(f"Audit hash mismatch for {p_str}: {actual} != {expected}")
        audit_verified[p_str] = actual

    ref = json.loads(BASELINE_REFERENCE_JSON.read_text(encoding="utf-8"))
    ref_verified: Dict[str, str] = {}
    for key in ("manifest", "retriever", "selector"):
        p_str = ref[key]["path"]
        actual = file_digest(p_str)
        expected = ref[key]["sha256"]
        if actual != expected:
            raise AssertionError(f"Reference hash mismatch for {p_str}: {actual} != {expected}")
        ref_verified[p_str] = actual
        if "source_path" in ref[key]:
            src_str = ref[key]["source_path"]
            src_actual = file_digest(src_str)
            if src_actual != expected:
                raise AssertionError(f"Reference source hash mismatch for {src_str}: {src_actual} != {expected}")
            ref_verified[src_str] = src_actual

    return {
        "audit_artifacts_count": len(audit_verified),
        "audit_artifacts_sha256": audit_verified,
        "reference_artifacts_sha256": ref_verified,
    }


def verify_and_replay_baseline_reference(
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify no-brand-v1 reference provenance, replay inference (no retraining), and record baseline outputs."""
    hashes = verify_hash_preservation()
    ref = json.loads(BASELINE_REFERENCE_JSON.read_text(encoding="utf-8"))
    manifest = json.loads((BASELINE_REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8"))

    retriever_path = BASELINE_REFERENCE_DIR / "best_retriever.pt"
    selector_path = BASELINE_REFERENCE_DIR / "best_selector.pt"
    retriever_digest = checkpoint_digest(retriever_path)
    selector_digest = checkpoint_digest(selector_path)

    dev = get_device(device)
    base_spec = build_description_variant_specs()["no_brand_v1"].variant_spec

    # Replay Stage 1 retriever on validation split
    ret_ckpt = torch.load(retriever_path, map_location="cpu", weights_only=False)
    ret_model = make_variant_model(base_spec, info, enrichment).to(dev)
    ret_model.load_state_dict(ret_ckpt["model_state_dict"], strict=True)
    prod_vecs, val_cust_vecs, val_logits = encode_and_score_variant(
        ret_model, base_spec, val_retrieval, info, enrichment, "val", dev
    )
    replayed_s1 = metrics_from_scores(val_logits, val_retrieval, val_source)
    if replayed_s1 != ret_ckpt["metrics"]:
        raise AssertionError("Baseline Stage 1 checkpoint replay metrics mismatch.")

    # Load upstream run dir for selector campaign data & saved score verification
    upstream_run_dir = Path(ref["retriever"]["source_path"]).parent
    cdata, _, _, stage1_digest = _load_stage1_campaign_data(
        base_spec,
        upstream_run_dir,
        train_retrieval,
        val_retrieval,
        info,
        train_source,
        val_source,
        enrichment,
        dev,
    )
    if stage1_digest != retriever_digest:
        raise AssertionError("Baseline upstream digest mismatch.")
    va = cdata["val"]
    torch.testing.assert_close(prod_vecs, va["v"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(val_cust_vecs.squeeze(1), va["u"], atol=1e-4, rtol=1e-4)

    # Replay CompactStage1Selector uv_47 inference on validation split
    sel_ckpt = torch.load(selector_path, map_location="cpu", weights_only=False)
    sel_model = CompactStage1Selector(recipe="uv_47").to(dev)
    sel_model.load_state_dict(sel_ckpt["model_state_dict"], strict=True)
    val_scores = _score_compact_selector_split(sel_model, va, dev)
    replayed_mlp = ranking_metrics(val_scores, va, None)

    saved_metrics = ref["validation_metrics"]
    for k in ("new_hits@10", "new_hits@25", "new_hits@50", "new_hits@100"):
        if replayed_mlp[k] != saved_metrics[k]:
            raise AssertionError(f"Baseline MLP replay mismatch on {k}: {replayed_mlp[k]} != {saved_metrics[k]}")
    for k in ("recall@10", "ndcg@10", "new_recall@10", "new_recall@25", "new_recall@50", "new_recall@100"):
        if abs(replayed_mlp[k] - saved_metrics[k]) > 1e-5:
            raise AssertionError(f"Baseline MLP replay float mismatch on {k}: {replayed_mlp[k]} != {saved_metrics[k]}")

    # Verify saved val_scores.pt and mlp_fullpool_predictions.npz if present
    sel_run_dir = Path(ref["selector"]["source_path"]).parent
    saved_scores_path = sel_run_dir / "val_scores.pt"
    if saved_scores_path.is_file():
        saved_scores_payload = torch.load(saved_scores_path, map_location="cpu", weights_only=False)
        if saved_scores_payload.get("selector_digest") != selector_digest:
            raise AssertionError("Saved baseline val_scores.pt selector_digest mismatch.")
        torch.testing.assert_close(val_scores, saved_scores_payload["scores"], atol=1e-4, rtol=1e-4)
        # Use exact saved val_scores tensor so saved baseline metrics match identically
        canonical_scores = saved_scores_payload["scores"]
    else:
        canonical_scores = val_scores

    baseline_dir = Path(output_root) / "baseline_verified"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    per_query_npz = baseline_dir / "no_brand_v1_mlp_fullpool_predictions.npz"
    subgroup_npz = baseline_dir / "no_brand_v1_mlp_subgroup_predictions.npz"
    save_mlp_per_query_predictions(canonical_scores, va, per_query_npz)

    saved_npz = sel_run_dir / "mlp_fullpool_predictions.npz"
    if saved_npz.is_file():
        a = np.load(saved_npz)
        b = np.load(per_query_npz)
        for key in ("customers", "new_hits", "new_hits25", "new_hits50", "new_hits100", "new_targets"):
            if not np.array_equal(a[key], b[key]):
                raise AssertionError(f"Baseline per-query npz mismatch on {key}.")

    subgroups_html = evaluate_description_subgroups(
        canonical_scores,
        va,
        enrichment["cohort_masks"]["has_description_html"],
        save_path=subgroup_npz,
    )
    subgroups_clean = evaluate_description_subgroups(
        canonical_scores,
        va,
        enrichment["cohort_masks"]["has_description_clean"],
    )

    dot_fullpool = ranking_metrics(va["dot_scores"], va, None)
    return {
        "name": "no_brand_v1",
        "reused_reference": True,
        "reference_json": str(BASELINE_REFERENCE_JSON),
        "reference_dir": str(BASELINE_REFERENCE_DIR),
        "hashes_verified": hashes,
        "stage1": {
            "selected_epoch": ret_ckpt["epoch"],
            "parameters": sum(p.numel() for p in ret_model.parameters()),
            "selected_checkpoint": str(retriever_path),
            "sha256": retriever_digest,
            "metrics": replayed_s1,
            "upstream_dot_fullpool": {
                k: dot_fullpool[k]
                for k in ("new_hits@25", "new_recall@25", "new_hits@50", "new_recall@50", "new_hits@100", "new_recall@100", "recall@10", "ndcg@10")
            },
        },
        "selector_uv47": {
            "selected_epoch": sel_ckpt["epoch"],
            "parameters": sum(p.numel() for p in sel_model.parameters()),
            "selected_checkpoint": str(selector_path),
            "sha256": selector_digest,
            "saved_reference_metrics": saved_metrics,
            "replayed_metrics": replayed_mlp,
            "training_seconds": manifest["artifacts"]["selector"].get("training_seconds"),
        },
        "mlp_per_query_predictions": str(per_query_npz),
        "mlp_subgroup_predictions": str(subgroup_npz),
        "subgroups_html_mask": subgroups_html,
        "subgroups_clean_mask": subgroups_clean,
    }
