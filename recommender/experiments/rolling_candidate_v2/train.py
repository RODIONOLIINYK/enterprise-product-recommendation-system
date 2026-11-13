"""Matched 10-epoch Stage 2 control vs candidate-attention fits on rolling_observed_v2 slates (no MLP score)."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from recommender.artifacts import atomic_json, create_run_directory, file_digest, json_digest
from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.feature_schema import rolling_contract
from recommender.experiments.static_enrichment.train import _make_downstream_ranker
from recommender.ranker import RankerLoss
from recommender.train_retriever import get_device

from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    CACHE_ROOT,
    DEFAULT_DISCOVERY_RANKER,
    DEFAULT_RETRIEVER,
    EPOCHS,
    ETA_MIN,
    EVIDENCE_ROOT,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    GRAD_CLIP_NORM,
    LR,
    SEED,
    SELECTOR_SOURCE_RUN_DIR,
    SLATE_POLICY,
    STAGE1_MLP_REF_MANIFEST,
    STAGE1_MLP_REF_POINTER,
    STAGE1_MLP_REF_RETRIEVER,
    STAGE1_MLP_REF_SELECTOR,
    UPSTREAM_STAGE1_RUN_DIR,
    WEIGHT_DECAY,
)
from .data import (
    make_rolling_slate_loader,
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
    verify_active_reference,
    verify_rolling_cache_vs_adapter_parity,
)
from .evaluate import (
    evaluate_rolling_slate_metrics,
    paired_rolling_intervals,
    predict_rolling_ranker,
)
from .model import (
    RollingCandidateAttentionRanker,
    load_rolling_ranker_checkpoint,
    save_rolling_ranker_checkpoint,
)


def snapshot_preserved_hashes() -> Dict[str, str]:
    """Capture SHA-256 digests of frozen reference/historical artifacts."""
    targets = {
        "stage1_mlp_ref_pointer": STAGE1_MLP_REF_POINTER,
        "stage1_mlp_ref_manifest": STAGE1_MLP_REF_MANIFEST,
        "stage1_mlp_ref_retriever": STAGE1_MLP_REF_RETRIEVER,
        "stage1_mlp_ref_selector": STAGE1_MLP_REF_SELECTOR,
        "embedding_ablation_ref_pointer": Path("experiments/embedding_ablation_v1/reference.json"),
        "source_selector_best": SELECTOR_SOURCE_RUN_DIR / "best.pt",
        "source_train_scores": SELECTOR_SOURCE_RUN_DIR / "train_scores.pt",
        "source_val_scores": SELECTOR_SOURCE_RUN_DIR / "val_scores.pt",
        "upstream_retriever_best": UPSTREAM_STAGE1_RUN_DIR / "best.pt",
        "upstream_train_purchase_cache": UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
        "upstream_val_purchase_cache": UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
        "default_retriever": DEFAULT_RETRIEVER,
        "default_discovery_ranker": DEFAULT_DISCOVERY_RANKER,
        "default_best_ranker": Path("checkpoints/ranker/best_ranker.pt"),
        "default_ndcg_ranker": Path("checkpoints/ranker/best_ndcg_ranker.pt"),
        "legacy_ranker_train_slates": Path("data/interim/ranker_train_slates_v3_repeat_0.5_rolling_v1.pt"),
        "legacy_ranker_val_slates": Path("data/interim/ranker_val_slates_v3_repeat_0.5_rolling_v1.pt"),
    }
    return {name: file_digest(path) for name, path in targets.items() if path.is_file()}


@torch.no_grad()
def verify_zero_residual_initialization_parity(
    val_slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    *,
    seed: int = SEED,
    batch_rows: int = 32,
) -> Tuple[Dict[str, Any], RollingCandidateAttentionRanker]:
    """Verify exact init parity (atol=0) vs 425D pointwise scorer, permutation equivariance, and pad masking."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dyn_ranker = _make_downstream_ranker(
        "dynamic_features",
        int(seed),
        STAGE1_MLP_REF_RETRIEVER,
        is_control_stage1=False,
    )
    dyn_ranker.eval()

    ctrl_model = RollingCandidateAttentionRanker.from_dynamic_context_ranker(
        dyn_ranker,
        candidate_attention_enabled=False,
    )
    treat_model = ctrl_model.clone_with_candidate_attention(True)
    ctrl_model.eval()
    treat_model.eval()

    rows = min(int(batch_rows), len(val_slate["customer_ids"]))
    batch = next(iter(make_rolling_slate_loader(val_slate, batch_size=rows, shuffle=False)))
    cpu = torch.device("cpu")
    inputs_new = rolling_ranker_batch_inputs(batch, product_vectors.cpu(), cpu)
    inputs_old = {k: v for k, v in inputs_new.items() if k != "candidate_mask"}

    old_out = dyn_ranker(**inputs_old)
    ctrl_out = ctrl_model(**inputs_new)
    treat_out = treat_model(**inputs_new)
    treat_no_attn = treat_model(**inputs_new, disable_candidate_attention=True)

    diff_ctrl = float((ctrl_out.logits - old_out.logits).abs().max().item())
    diff_treat = float((treat_out.logits - old_out.logits).abs().max().item())
    diff_treat_no_attn = float((treat_no_attn.logits - old_out.logits).abs().max().item())
    diff_attn = float((treat_out.attn_weights - old_out.attn_weights).abs().max().item())

    if diff_ctrl != 0.0 or diff_treat != 0.0 or diff_treat_no_attn != 0.0 or diff_attn != 0.0:
        raise AssertionError(
            "Zero-residual init parity failed: "
            f"ctrl={diff_ctrl:.3e}, treat={diff_treat:.3e}, "
            f"treat_no_attn={diff_treat_no_attn:.3e}, attn={diff_attn:.3e}."
        )

    # Reverse candidate order to verify permutation equivariance.
    rev_inputs = {
        k: (
            torch.flip(v, dims=[1])
            if k in ("pairwise_features", "candidate_vectors", "candidate_mask")
            else v
        )
        for k, v in inputs_new.items()
    }
    rev_out = treat_model(**rev_inputs)
    perm_diff = float((torch.flip(rev_out.logits, dims=[1]) - treat_out.logits).abs().max().item())
    if perm_diff > 1e-6:
        raise AssertionError(f"Permutation equivariance failed at init: {perm_diff:.3e}")

    # Append 2 NaN-filled masked candidates to verify isolation.
    pad_inputs = dict(inputs_new)
    pad_inputs["pairwise_features"] = torch.cat(
        [inputs_new["pairwise_features"], torch.full((rows, 2, 41), float("nan"))],
        dim=1,
    )
    pad_inputs["candidate_vectors"] = torch.cat(
        [inputs_new["candidate_vectors"], torch.full((rows, 2, 128), float("nan"))],
        dim=1,
    )
    pad_inputs["candidate_mask"] = torch.cat(
        [inputs_new["candidate_mask"], torch.zeros((rows, 2), dtype=torch.bool)],
        dim=1,
    )
    pad_out = treat_model(**pad_inputs)
    pad_valid_diff = float((pad_out.logits[:, :-2] - treat_out.logits).abs().max().item())
    pad_masked_neginf = bool(torch.all(pad_out.logits[:, -2:] == -torch.inf))
    if pad_valid_diff > 1e-6 or not pad_masked_neginf:
        raise AssertionError("Padding/mask isolation failed at init.")

    diag = {
        "rows_checked": rows,
        "dyn_425d_parameters": int(sum(p.numel() for p in dyn_ranker.parameters())),
        "control_parameters": int(sum(p.numel() for p in ctrl_model.parameters())),
        "control_trainable_parameters": int(sum(p.numel() for p in ctrl_model.parameters() if p.requires_grad)),
        "treatment_parameters": int(sum(p.numel() for p in treat_model.parameters())),
        "treatment_trainable_parameters": int(sum(p.numel() for p in treat_model.parameters() if p.requires_grad)),
        "max_abs_logit_diff_control_vs_425d": diff_ctrl,
        "max_abs_logit_diff_treatment_vs_425d": diff_treat,
        "max_abs_logit_diff_treatment_disable_attn_vs_425d": diff_treat_no_attn,
        "max_abs_attn_diff_vs_425d": diff_attn,
        "max_abs_logit_diff_reversed_slate": perm_diff,
        "max_abs_logit_diff_with_nan_padding": pad_valid_diff,
        "masked_pad_logit_neginf": pad_masked_neginf,
    }
    return diag, ctrl_model


def _run_dir_for_fit(
    root: Path, run_name: str, settings: Mapping[str, Any]
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    directory = Path(root) / "runs" / run_name
    settings_digest = json_digest(settings)
    if directory.exists():
        run_manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run_manifest.get("settings_digest") != settings_digest:
            raise ValueError(
                f"Existing run {directory} has different settings_digest; refusing overwrite."
            )
        if run_manifest.get("status") != "complete":
            raise ValueError(f"Incomplete existing run {directory}; use a clean artifact root.")
        for ckpt_key, sha_key in (
            ("discovery_checkpoint", "discovery_sha256"),
            ("ndcg_checkpoint", "ndcg_sha256"),
        ):
            rel = run_manifest.get(ckpt_key)
            if not rel or file_digest(directory / rel) != run_manifest.get(sha_key):
                raise ValueError(f"Checkpoint checksum mismatch in {directory} for {ckpt_key}.")
        return directory, json.loads((directory / "result.json").read_text(encoding="utf-8"))

    directory = create_run_directory(root, run_name)
    atomic_json(
        {
            "version": 1,
            "status": "running",
            "settings_digest": settings_digest,
            "discovery_checkpoint": None,
            "ndcg_checkpoint": None,
        },
        directory / "run.json",
    )
    return directory, None


def _build_provenance(
    *,
    candidate_attention_enabled: bool,
    meta: Mapping[str, Any],
    slate_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "reference_id": EXPECTED_REFERENCE_ID,
        "upstream_retriever_path": str(STAGE1_MLP_REF_RETRIEVER),
        "upstream_retriever_sha256": meta["reference"]["hashes"]["best_retriever"],
        "candidate_source_selector_path": str(STAGE1_MLP_REF_SELECTOR),
        "candidate_source_selector_sha256": meta["reference"]["hashes"]["best_selector"],
        "teacher_score_included": False,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_schema_digest": meta["feature_schema_digest"],
        "data_identity": meta["data_identity"],
        "rolling_features_identity": meta["rolling_features_identity"],
        "slate_policy": SLATE_POLICY,
        "train_slate_identity": slate_manifest["split_cache_identity"]["train"],
        "val_slate_identity": slate_manifest["split_cache_identity"]["val"],
        "candidate_attention_enabled": bool(candidate_attention_enabled),
        "heldout_access": False,
        "metadata_policy": "snapshot_unverified",
        "product_universe": EXPECTED_PRODUCT_UNIVERSE,
    }


def fit_single_rolling_run(
    *,
    arm_name: str,
    candidate_attention_enabled: bool,
    init_model: RollingCandidateAttentionRanker,
    train_slate: Mapping[str, Any],
    val_slate: Mapping[str, Any],
    val_data: Mapping[str, Any],
    product_vectors: torch.Tensor,
    meta: Mapping[str, Any],
    slate_manifest: Mapping[str, Any],
    artifact_root: Path = ARTIFACT_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    eta_min: float = ETA_MIN,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit one Stage 2 rolling arm (control pointwise vs candidate self-attention) and replay checkpoints."""
    dev = get_device(device)
    ref_disc = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu", weights_only=False)
    recall_guard = float(ref_disc["discovery_min_overall_recall"])
    ndcg_guard = float(ref_disc["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)

    provenance = _build_provenance(
        candidate_attention_enabled=candidate_attention_enabled,
        meta=meta,
        slate_manifest=slate_manifest,
    )
    feature_contract_digest = meta["feature_schema_digest"]

    run_name = f"stage2-{arm_name}-seed{seed}"
    settings = {
        "stage": "rolling_candidate_v2",
        "arm_name": arm_name,
        "candidate_attention_enabled": bool(candidate_attention_enabled),
        "teacher_score_included": False,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_contract": rolling_contract(),
        "feature_contract_digest": feature_contract_digest,
        "policy": SLATE_POLICY,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "eta_min": float(eta_min),
        "grad_clip_norm": float(grad_clip_norm),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "RankerLoss",
        "data_identity": meta["data_identity"],
        "rolling_features_identity": meta["rolling_features_identity"],
        "upstream_sha256": provenance["upstream_retriever_sha256"],
        "candidate_source_selector_sha256": provenance["candidate_source_selector_sha256"],
        "train_slate_identity": provenance["train_slate_identity"],
        "val_slate_identity": provenance["val_slate_identity"],
        "recall_guardrail": recall_guard,
        "ndcg_guardrail": ndcg_guard,
        "guardrail_artifact_sha256": guard_sha,
        "heldout_access": False,
        "selection_discovery": "qualified=(recall@10>=guard and ndcg@10>=guard), new_hits@10, ndcg@10",
        "selection_ndcg": "ndcg@10, recall@10, new_hits@10",
    }

    directory, existing = _run_dir_for_fit(artifact_root, run_name, settings)
    if existing is not None:
        return existing

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = init_model.clone_with_candidate_attention(candidate_attention_enabled).to(dev)
    vectors_dev = product_vectors.to(dev)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=float(eta_min))
    loss_fn = RankerLoss().to(dev)
    train_loader = make_rolling_slate_loader(
        train_slate,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed),
    )

    best_disc_key = (-1, -1, -1.0)
    best_ndcg_key = (-1.0, -1.0, -1)
    history = []

    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = rolling_ranker_batch_inputs(batch, vectors_dev, dev)
            logits = model(**inputs).logits
            loss = loss_fn(logits, batch["labels"].to(dev))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_pred = predict_rolling_ranker(model, val_slate, vectors_dev, dev)
        metrics = evaluate_rolling_slate_metrics(
            val_pred["scores"],
            val_data,
            val_slate["candidate_indices"],
        )
        metrics["val_loss"] = val_pred["val_loss"]
        metrics["avg_token0_weight"] = val_pred["avg_token0_weight"]
        metrics["avg_history_weight"] = val_pred["avg_history_weight"]

        qualified_epoch = bool(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard)
        disc_key = (int(qualified_epoch), int(metrics["new_hits@10"]), float(metrics["ndcg@10"]))
        ndcg_key = (float(metrics["ndcg@10"]), float(metrics["recall@10"]), int(metrics["new_hits@10"]))
        train_loss_mean = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "qualified": qualified_epoch,
                "metrics": metrics,
            }
        )

        if disc_key > best_disc_key:
            best_disc_key = disc_key
            save_rolling_ranker_checkpoint(
                directory / "best.pt",
                model,
                epoch=epoch,
                metrics=metrics,
                selection_criterion="discovery_guardrail_new_hits_at_10",
                provenance=provenance,
                feature_contract_digest=feature_contract_digest,
                settings=settings,
            )

        if ndcg_key > best_ndcg_key:
            best_ndcg_key = ndcg_key
            save_rolling_ranker_checkpoint(
                directory / "best_ndcg.pt",
                model,
                epoch=epoch,
                metrics=metrics,
                selection_criterion="val_ndcg_at_10",
                provenance=provenance,
                feature_contract_digest=feature_contract_digest,
                settings=settings,
            )

        print(
            f"[rolling_candidate_v2/{arm_name}] seed={seed} epoch={epoch:02d} loss={train_loss_mean:.4f} "
            f"rec10={metrics['recall@10']:.5f} ndcg10={metrics['ndcg@10']:.5f} "
            f"new10={metrics['new_hits@10']}/{metrics['new_targets']} qual={qualified_epoch}",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - started

    qualified = bool(best_disc_key[0])
    disc_rel = "best.pt" if qualified else "best_unqualified.pt"
    if not qualified:
        (directory / "best.pt").replace(directory / disc_rel)
    disc_path = directory / disc_rel
    ndcg_path = directory / "best_ndcg.pt"

    disc_model, disc_ckpt = load_rolling_ranker_checkpoint(
        disc_path,
        expected_provenance=provenance,
        expected_feature_contract_digest=feature_contract_digest,
        expected_candidate_attention_enabled=candidate_attention_enabled,
        map_location="cpu",
    )
    disc_model = disc_model.to(dev)
    disc_pred = predict_rolling_ranker(disc_model, val_slate, vectors_dev, dev)
    disc_replayed = evaluate_rolling_slate_metrics(
        disc_pred["scores"],
        val_data,
        val_slate["candidate_indices"],
        save_path=directory / "predictions_discovery.npz",
    )
    disc_replayed["val_loss"] = disc_pred["val_loss"]
    disc_replayed["avg_token0_weight"] = disc_pred["avg_token0_weight"]
    disc_replayed["avg_history_weight"] = disc_pred["avg_history_weight"]
    if disc_replayed != disc_ckpt["metrics"]:
        raise AssertionError(f"Discovery checkpoint replay mismatch for {run_name}.")

    disc_no_attn = predict_rolling_ranker(
        disc_model, val_slate, vectors_dev, dev, disable_candidate_attention=True
    )
    disc_attn_diff = float((disc_pred["scores"] - disc_no_attn["scores"]).abs().max().item())

    ndcg_model, ndcg_ckpt = load_rolling_ranker_checkpoint(
        ndcg_path,
        expected_provenance=provenance,
        expected_feature_contract_digest=feature_contract_digest,
        expected_candidate_attention_enabled=candidate_attention_enabled,
        map_location="cpu",
    )
    ndcg_model = ndcg_model.to(dev)
    ndcg_pred = predict_rolling_ranker(ndcg_model, val_slate, vectors_dev, dev)
    ndcg_replayed = evaluate_rolling_slate_metrics(
        ndcg_pred["scores"],
        val_data,
        val_slate["candidate_indices"],
        save_path=directory / "predictions_ndcg.npz",
    )
    ndcg_replayed["val_loss"] = ndcg_pred["val_loss"]
    ndcg_replayed["avg_token0_weight"] = ndcg_pred["avg_token0_weight"]
    ndcg_replayed["avg_history_weight"] = ndcg_pred["avg_history_weight"]
    if ndcg_replayed != ndcg_ckpt["metrics"]:
        raise AssertionError(f"NDCG checkpoint replay mismatch for {run_name}.")

    ndcg_no_attn = predict_rolling_ranker(
        ndcg_model, val_slate, vectors_dev, dev, disable_candidate_attention=True
    )
    ndcg_attn_diff = float((ndcg_pred["scores"] - ndcg_no_attn["scores"]).abs().max().item())

    result: Dict[str, Any] = {
        "arm_name": arm_name,
        "run_dir": str(directory),
        "candidate_attention_enabled": bool(candidate_attention_enabled),
        "teacher_score_included": False,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "settings": settings,
        "provenance": provenance,
        "qualified": qualified,
        "parameters": int(sum(p.numel() for p in disc_model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in disc_model.parameters() if p.requires_grad)),
        "training_seconds": float(elapsed),
        "device": str(dev),
        "discovery": {
            "epoch": int(disc_ckpt["epoch"]),
            "checkpoint": str(disc_path),
            "checkpoint_sha256": file_digest(disc_path),
            "predictions_npz": str(directory / "predictions_discovery.npz"),
            "metrics": disc_replayed,
            "full_val_max_abs_logit_diff_vs_disable_candidate_attention": disc_attn_diff,
        },
        "ndcg": {
            "epoch": int(ndcg_ckpt["epoch"]),
            "checkpoint": str(ndcg_path),
            "checkpoint_sha256": file_digest(ndcg_path),
            "predictions_npz": str(directory / "predictions_ndcg.npz"),
            "metrics": ndcg_replayed,
            "full_val_max_abs_logit_diff_vs_disable_candidate_attention": ndcg_attn_diff,
        },
        "history": history,
    }
    atomic_json(history, directory / "history.json", overwrite=False)
    atomic_json(result, directory / "result.json", overwrite=False)
    atomic_json(
        {
            "version": 1,
            "status": "complete",
            "settings_digest": json_digest(settings),
            "discovery_checkpoint": disc_rel,
            "discovery_sha256": result["discovery"]["checkpoint_sha256"],
            "ndcg_checkpoint": "best_ndcg.pt",
            "ndcg_sha256": result["ndcg"]["checkpoint_sha256"],
            "qualified": qualified,
        },
        directory / "run.json",
        overwrite=True,
    )
    return result


def run_matched_experiment(
    *,
    cache_root: Path = CACHE_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run paired control vs candidate-attention training on rolling_observed_v2 (explicit invocation only)."""
    before_hashes = snapshot_preserved_hashes()
    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=cache_root,
        build=True,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    _, scores_manifest = prepare_or_load_rolling_selector_scores(
        campaign_data, info, meta, cache_root=cache_root
    )
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(
        campaign_data, info, meta, cache_root=cache_root
    )
    parity = verify_rolling_cache_vs_adapter_parity(
        campaign_data, info, slates=slates, n_rows=16
    )
    product_vectors = campaign_data["train"]["v"]
    init_diag, init_model = verify_zero_residual_initialization_parity(
        slates["val"], product_vectors, seed=seed
    )

    ctrl = fit_single_rolling_run(
        arm_name="control_pointwise_425d",
        candidate_attention_enabled=False,
        init_model=init_model,
        train_slate=slates["train"],
        val_slate=slates["val"],
        val_data=campaign_data["val"],
        product_vectors=product_vectors,
        meta=meta,
        slate_manifest=slate_manifest,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    treat = fit_single_rolling_run(
        arm_name="treatment_cand_attn_425d",
        candidate_attention_enabled=True,
        init_model=init_model,
        train_slate=slates["train"],
        val_slate=slates["val"],
        val_data=campaign_data["val"],
        product_vectors=product_vectors,
        meta=meta,
        slate_manifest=slate_manifest,
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    intervals = {
        "discovery": paired_rolling_intervals(
            ctrl["discovery"]["predictions_npz"],
            treat["discovery"]["predictions_npz"],
        ),
        "ndcg": paired_rolling_intervals(
            ctrl["ndcg"]["predictions_npz"],
            treat["ndcg"]["predictions_npz"],
        ),
    }

    after_hashes = snapshot_preserved_hashes()
    if after_hashes != before_hashes:
        raise AssertionError("Preserved reference/historical hashes changed during execution!")

    summary = {
        "training_executed": True,
        "heldout_access": False,
        "promoted": False,
        "active_reference_id": EXPECTED_REFERENCE_ID,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "teacher_score_included": False,
        "feature_bundle_identity": feat_manifest["identity"],
        "rolling_selector_scores_identity": scores_manifest["identity"],
        "slate_cache_identities": slate_manifest["split_cache_identity"],
        "adapter_parity": parity,
        "zero_residual_init_parity": init_diag,
        "control": ctrl,
        "treatment": treat,
        "paired_bootstrap_95ci": intervals,
        "preserved_hashes": after_hashes,
    }
    Path(evidence_root).mkdir(parents=True, exist_ok=True)
    atomic_json(summary, Path(evidence_root) / "training_summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling candidate-set self-attention experiment for repeat-no-month-v1 "
            "(rolling_observed_v2, NO Stage 1 teacher score)."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify reference and prepare sealed rolling feature + slate caches without training.",
    )
    parser.add_argument(
        "--verify-init-only",
        action="store_true",
        help="Verify zero-residual initialization parity without any optimizer steps.",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Explicitly execute the matched 10-epoch control vs treatment Stage 2 training runs.",
    )
    args = parser.parse_args()

    if not args.execute_training and not args.prepare_only and not args.verify_init_only:
        parser.error(
            "Pass one of --prepare-only, --verify-init-only, or --execute-training."
        )

    if args.execute_training:
        summary = run_matched_experiment(
            cache_root=args.cache_root,
            artifact_root=args.artifact_root,
            evidence_root=args.evidence_root,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, default=str))
        return

    verify_active_reference(load_selector=True)
    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=args.cache_root,
        build=True,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    _, scores_manifest = prepare_or_load_rolling_selector_scores(
        campaign_data, info, meta, cache_root=args.cache_root
    )
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(
        campaign_data, info, meta, cache_root=args.cache_root
    )
    out: Dict[str, Any] = {
        "training_executed": False,
        "heldout_access": False,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_bundle_identity": feat_manifest["identity"],
        "rolling_selector_scores_identity": scores_manifest["identity"],
        "slate_cache_identities": slate_manifest["split_cache_identity"],
    }
    if args.verify_init_only:
        init_diag, _ = verify_zero_residual_initialization_parity(
            slates["val"], campaign_data["train"]["v"], seed=args.seed
        )
        out["zero_residual_init_parity"] = init_diag
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
