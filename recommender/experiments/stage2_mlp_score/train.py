"""Controlled matched 10-epoch Stage 2 fits with frozen 301D MLP score channel."""

import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from recommender.artifacts import atomic_json, create_run_directory, file_digest, json_digest
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.feature_schema import contract as dynamic_feature_contract
from recommender.experiments.static_enrichment.train import _make_downstream_ranker
from recommender.ranker import RankerLoss
from recommender.train_retriever import get_device

from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    DEFAULT_DISCOVERY_RANKER,
    DEFAULT_RETRIEVER,
    EPOCHS,
    ETA_MIN,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    EXPECTED_RETRIEVER_SHA256,
    EXPECTED_SELECTOR_SHA256,
    GRAD_CLIP_NORM,
    LR,
    SCORE_FEATURE_NAME,
    SCORE_TRANSFORM_VERSION,
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
    make_stage2_loader,
    stage2_mlp_score_batch_inputs,
)
from .evaluate import (
    evaluate_stage2_slate_metrics,
    measure_score_channel_diagnostics,
    predict_stage2_ranker,
)
from .model import (
    Stage2MLPScoreInferenceAdapter,
    Stage2MLPScoreRanker,
    load_stage2_mlp_score_checkpoint,
    save_stage2_mlp_score_checkpoint,
)


def snapshot_preserved_hashes() -> Dict[str, str]:
    """Capture SHA-256 digests of all frozen upstream/reference files that must remain untouched."""
    targets = {
        "agents_md": Path("AGENTS.md"),
        "stage1_mlp_ref_pointer": STAGE1_MLP_REF_POINTER,
        "stage1_mlp_ref_manifest": STAGE1_MLP_REF_MANIFEST,
        "stage1_mlp_ref_retriever": STAGE1_MLP_REF_RETRIEVER,
        "stage1_mlp_ref_selector": STAGE1_MLP_REF_SELECTOR,
        "source_selector_best": SELECTOR_SOURCE_RUN_DIR / "best.pt",
        "source_train_scores": SELECTOR_SOURCE_RUN_DIR / "train_scores.pt",
        "source_val_scores": SELECTOR_SOURCE_RUN_DIR / "val_scores.pt",
        "upstream_retriever_best": UPSTREAM_STAGE1_RUN_DIR / "best.pt",
        "upstream_train_purchase_cache": UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
        "upstream_val_purchase_cache": UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
        "default_retriever": DEFAULT_RETRIEVER,
        "default_discovery_ranker": DEFAULT_DISCOVERY_RANKER,
    }
    optional_ranker = Path("checkpoints/ranker/best_ranker.pt")
    if optional_ranker.is_file():
        targets["default_best_ranker"] = optional_ranker
    return {name: file_digest(path) for name, path in targets.items() if path.is_file()}


def verify_zero_extension_initialization_parity(
    val_slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    *,
    seed: int = SEED,
    batch_rows: int = 32,
) -> Dict[str, Any]:
    """Construct shared 426D base and prove exact parity against unexpanded 425D dynamic ranker at init."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dyn_ranker = _make_downstream_ranker(
        "dynamic_features",
        int(seed),
        STAGE1_MLP_REF_RETRIEVER,
        is_control_stage1=False,
    )
    dyn_ranker.eval()
    expanded = Stage2MLPScoreRanker.from_dynamic_context_ranker(
        dyn_ranker,
        score_channel_enabled=True,
    )
    expanded.eval()

    batch = next(iter(make_stage2_loader(val_slate, batch_size=min(batch_rows, len(val_slate["customer_ids"])), shuffle=False)))
    cpu = torch.device("cpu")
    inputs_new = stage2_mlp_score_batch_inputs(batch, product_vectors.cpu(), cpu)
    inputs_old = {
        k: v
        for k, v in inputs_new.items()
        if k not in ("mlp_score", "candidate_mask")
    }
    with torch.no_grad():
        old_out = dyn_ranker(**inputs_old)
        new_actual = expanded(**inputs_new)
        inputs_zero = dict(inputs_new)
        inputs_zero["mlp_score"] = torch.zeros_like(inputs_new["mlp_score"])
        new_zero = expanded(**inputs_zero)

    diff_actual = float((new_actual.logits - old_out.logits).abs().max().item())
    diff_zero = float((new_zero.logits - old_out.logits).abs().max().item())
    diff_attn = float((new_actual.attn_weights - old_out.attn_weights).abs().max().item())
    if diff_actual > 1e-6 or diff_zero > 1e-6 or diff_attn > 1e-6:
        raise AssertionError(
            f"Initial zero-extension parity check failed: "
            f"diff_actual={diff_actual:.3e}, diff_zero={diff_zero:.3e}, diff_attn={diff_attn:.3e}."
        )
    return {
        "max_abs_logit_diff_vs_425d_with_actual_score": diff_actual,
        "max_abs_logit_diff_vs_425d_with_zero_score": diff_zero,
        "max_abs_attn_diff_vs_425d_with_actual_score": diff_attn,
        "dyn_425d_parameters": int(sum(p.numel() for p in dyn_ranker.parameters())),
        "expanded_426d_parameters": int(sum(p.numel() for p in expanded.parameters())),
        "expanded_426d_trainable_parameters": int(
            sum(p.numel() for p in expanded.parameters() if p.requires_grad)
        ),
    }, expanded


def _run_dir_for_fit(root: Path, run_name: str, settings: Mapping[str, Any]) -> Tuple[Path, Optional[Dict[str, Any]]]:
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
    score_channel_enabled: bool,
    meta: Mapping[str, Any],
    slate_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "reference_id": EXPECTED_REFERENCE_ID,
        "teacher_checkpoint_path": str(STAGE1_MLP_REF_SELECTOR),
        "teacher_checkpoint_sha256": meta["reference"]["hashes"]["best_selector"],
        "upstream_retriever_path": str(STAGE1_MLP_REF_RETRIEVER),
        "upstream_retriever_sha256": meta["reference"]["hashes"]["best_retriever"],
        "teacher_train_scores_sha256": meta["cache_lineage"]["train"]["teacher_scores_sha256"],
        "teacher_val_scores_sha256": meta["cache_lineage"]["val"]["teacher_scores_sha256"],
        "data_identity": meta["data_identity"],
        "dynamic_identity": meta["dynamic_identity"],
        "slate_policy": SLATE_POLICY,
        "train_slate_identity": slate_manifest["split_cache_identity"]["train"],
        "val_slate_identity": slate_manifest["split_cache_identity"]["val"],
        "score_feature_name": SCORE_FEATURE_NAME,
        "score_feature_transform": SCORE_TRANSFORM_VERSION,
        "score_channel_enabled": bool(score_channel_enabled),
        "heldout_access": False,
        "metadata_policy": "snapshot_unverified",
        "product_universe": EXPECTED_PRODUCT_UNIVERSE,
    }


def fit_single_stage2_run(
    *,
    arm_name: str,
    score_channel_enabled: bool,
    init_model: Stage2MLPScoreRanker,
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
    """Fit one 10-epoch Stage 2 ranker arm (matched control or MLP-score treatment) and verify strict reload."""
    dev = get_device(device)
    ref_disc = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu", weights_only=False)
    recall_guard = float(ref_disc["discovery_min_overall_recall"])
    ndcg_guard = float(ref_disc["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)

    teacher_contract = meta["reference"]["teacher_input_contract"]
    provenance = _build_provenance(
        score_channel_enabled=score_channel_enabled,
        meta=meta,
        slate_manifest=slate_manifest,
    )

    run_name = f"stage2-{arm_name}-seed{seed}"
    settings = {
        "stage": "stage2_mlp_score_v1",
        "arm_name": arm_name,
        "score_channel_enabled": bool(score_channel_enabled),
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
        "dynamic_identity": meta["dynamic_identity"],
        "feature_contract": dynamic_feature_contract(),
        "teacher_contract_digest": teacher_contract["contract_digest"],
        "teacher_sha256": provenance["teacher_checkpoint_sha256"],
        "upstream_sha256": provenance["upstream_retriever_sha256"],
        "train_slate_identity": provenance["train_slate_identity"],
        "val_slate_identity": provenance["val_slate_identity"],
        "score_feature_transform": SCORE_TRANSFORM_VERSION,
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
    model = init_model.clone_with_score_channel(score_channel_enabled).to(dev)
    vectors_dev = product_vectors.to(dev)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=float(eta_min))
    loss_fn = RankerLoss().to(dev)
    train_loader = make_stage2_loader(
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
            inputs = stage2_mlp_score_batch_inputs(batch, vectors_dev, dev)
            logits = model(**inputs).logits
            loss = loss_fn(logits, batch["labels"].to(dev))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_pred = predict_stage2_ranker(model, val_slate, vectors_dev, dev)
        metrics = evaluate_stage2_slate_metrics(
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
            save_stage2_mlp_score_checkpoint(
                directory / "best.pt",
                model,
                epoch=epoch,
                metrics=metrics,
                selection_criterion="discovery_guardrail_new_hits_at_10",
                provenance=provenance,
                teacher_input_contract=teacher_contract,
                settings=settings,
            )

        if ndcg_key > best_ndcg_key:
            best_ndcg_key = ndcg_key
            save_stage2_mlp_score_checkpoint(
                directory / "best_ndcg.pt",
                model,
                epoch=epoch,
                metrics=metrics,
                selection_criterion="val_ndcg_at_10",
                provenance=provenance,
                teacher_input_contract=teacher_contract,
                settings=settings,
            )

        print(
            f"[stage2/{arm_name}] seed={seed} epoch={epoch:02d} loss={train_loss_mean:.4f} "
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

    # Strict replay verification for BOTH discovery and NDCG checkpoints.
    disc_model, disc_ckpt = load_stage2_mlp_score_checkpoint(
        disc_path,
        expected_provenance=provenance,
        expected_teacher_input_contract=teacher_contract,
        expected_score_channel_enabled=score_channel_enabled,
        map_location="cpu",
    )
    disc_model = disc_model.to(dev)
    disc_pred = predict_stage2_ranker(disc_model, val_slate, vectors_dev, dev)
    disc_replayed = evaluate_stage2_slate_metrics(
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
    disc_diag = measure_score_channel_diagnostics(disc_model, val_slate, vectors_dev, dev)

    ndcg_model, ndcg_ckpt = load_stage2_mlp_score_checkpoint(
        ndcg_path,
        expected_provenance=provenance,
        expected_teacher_input_contract=teacher_contract,
        expected_score_channel_enabled=score_channel_enabled,
        map_location="cpu",
    )
    ndcg_model = ndcg_model.to(dev)
    ndcg_pred = predict_stage2_ranker(ndcg_model, val_slate, vectors_dev, dev)
    ndcg_replayed = evaluate_stage2_slate_metrics(
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
    ndcg_diag = measure_score_channel_diagnostics(ndcg_model, val_slate, vectors_dev, dev)

    # Zero-score diagnostics on replayed checkpoints:
    # - Control must be EXACTLY invariant (diff == 0.0) when mlp_score is supplied vs zero.
    # - Treatment reports post-training zero-score inference ablation (strictly labeled diagnostic).
    disc_zero_pred = predict_stage2_ranker(disc_model, val_slate, vectors_dev, dev, force_zero_score=True)
    disc_zero_diff = float((disc_pred["scores"] - disc_zero_pred["scores"]).abs().max().item())
    if not score_channel_enabled and disc_zero_diff != 0.0:
        raise AssertionError(f"Control discovery model altered output on mlp_score: {disc_zero_diff}")
    post_zero_metrics_disc = evaluate_stage2_slate_metrics(
        disc_zero_pred["scores"], val_data, val_slate["candidate_indices"]
    )
    ndcg_zero_pred = predict_stage2_ranker(ndcg_model, val_slate, vectors_dev, dev, force_zero_score=True)
    ndcg_zero_diff = float((ndcg_pred["scores"] - ndcg_zero_pred["scores"]).abs().max().item())
    if not score_channel_enabled and ndcg_zero_diff != 0.0:
        raise AssertionError(f"Control NDCG model altered output on mlp_score: {ndcg_zero_diff}")
    post_zero_metrics_ndcg = evaluate_stage2_slate_metrics(
        ndcg_zero_pred["scores"], val_data, val_slate["candidate_indices"]
    )

    result: Dict[str, Any] = {
        "arm_name": arm_name,
        "run_dir": str(directory),
        "score_channel_enabled": bool(score_channel_enabled),
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
            "score_channel_diagnostics": disc_diag,
            "full_val_max_abs_logit_diff_vs_zero_score": disc_zero_diff,
            "post_training_zero_score_diagnostic": post_zero_metrics_disc,
        },
        "ndcg": {
            "epoch": int(ndcg_ckpt["epoch"]),
            "checkpoint": str(ndcg_path),
            "checkpoint_sha256": file_digest(ndcg_path),
            "predictions_npz": str(directory / "predictions_ndcg.npz"),
            "metrics": ndcg_replayed,
            "score_channel_diagnostics": ndcg_diag,
            "full_val_max_abs_logit_diff_vs_zero_score": ndcg_zero_diff,
            "post_training_zero_score_diagnostic": post_zero_metrics_ndcg,
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


@torch.no_grad()
def verify_cache_vs_request_adapter_parity(
    *,
    teacher,
    teacher_spec,
    treatment_ckpt_path: Path,
    val_data: Mapping[str, Any],
    val_slate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    teacher_contract: Mapping[str, Any],
    n_rows: int = 32,
) -> Dict[str, Any]:
    """Verify live request-time adapter matches sealed cache scores, is permutation-equivariant, and masks pads."""
    ranker, _ = load_stage2_mlp_score_checkpoint(
        treatment_ckpt_path,
        expected_provenance=provenance,
        expected_teacher_input_contract=teacher_contract,
        expected_score_channel_enabled=True,
        map_location="cpu",
    )
    adapter = Stage2MLPScoreInferenceAdapter(
        teacher,
        ranker,
        expected_teacher_spec=teacher_spec,
    )

    rows = min(int(n_rows), len(val_slate["customer_ids"]))
    batch = next(iter(make_stage2_loader(val_slate, batch_size=rows, shuffle=False)))
    cpu = torch.device("cpu")
    v = val_data["v"].cpu()
    u = val_data["u"][:rows].cpu()
    stats6 = val_data["customer_stats_6"][:rows].cpu()
    cand_idx = batch["candidate_indices"]
    cand_mask = batch["candidate_mask"]
    inputs = stage2_mlp_score_batch_inputs(batch, v, cpu)

    # 1. Request-time adapter vs sealed cache mlp_score & logits
    req_mlp_score = adapter.compute_mlp_score(
        u=u,
        v=v,
        features_41=inputs["pairwise_features"],
        customer_stats_6=stats6,
        candidate_indices=cand_idx,
        candidate_mask=cand_mask,
    )
    cache_mlp_score = inputs["mlp_score"]
    mlp_score_diff = float((req_mlp_score - cache_mlp_score).abs().max().item())

    req_out = adapter.score_candidates(
        {k: val for k, val in inputs.items() if k not in ("mlp_score", "candidate_mask")},
        u=u,
        v=v,
        customer_stats_6=stats6,
        candidate_indices=cand_idx,
        candidate_mask=cand_mask,
    )
    cached_out = ranker(**inputs)
    logit_diff = float((req_out.logits - cached_out.logits).abs().max().item())

    # 2. Slate-order independence: reverse candidate order
    rev_idx = torch.flip(cand_idx, dims=[1])
    rev_mask = torch.flip(cand_mask, dims=[1])
    rev_inputs = {
        k: (
            torch.flip(val, dims=[1])
            if k in ("pairwise_features", "candidate_vectors")
            else val
        )
        for k, val in inputs.items()
        if k not in ("mlp_score", "candidate_mask")
    }
    rev_mlp_score = adapter.compute_mlp_score(
        u=u,
        v=v,
        features_41=rev_inputs["pairwise_features"],
        customer_stats_6=stats6,
        candidate_indices=rev_idx,
        candidate_mask=rev_mask,
    )
    perm_score_diff = float((torch.flip(rev_mlp_score, dims=[1]) - req_mlp_score).abs().max().item())
    rev_out = adapter.score_candidates(
        rev_inputs,
        u=u,
        v=v,
        customer_stats_6=stats6,
        candidate_indices=rev_idx,
        candidate_mask=rev_mask,
    )
    perm_logit_diff = float((torch.flip(rev_out.logits, dims=[1]) - req_out.logits).abs().max().item())

    # 3. Padding/mask isolation: append 2 invalid candidates with NaNs & out-of-bounds indices
    pad_idx = torch.cat([cand_idx, torch.full((rows, 2), -1, dtype=cand_idx.dtype)], dim=1)
    pad_mask = torch.cat([cand_mask, torch.zeros((rows, 2), dtype=torch.bool)], dim=1)
    pad_f41 = torch.cat([inputs["pairwise_features"], torch.full((rows, 2, 41), float("nan"))], dim=1)
    pad_vecs = torch.cat([inputs["candidate_vectors"], torch.full((rows, 2, 128), float("nan"))], dim=1)
    pad_inputs = {
        k: val for k, val in inputs.items() if k not in ("mlp_score", "candidate_mask", "pairwise_features", "candidate_vectors")
    }
    pad_inputs["pairwise_features"] = pad_f41
    pad_inputs["candidate_vectors"] = pad_vecs
    pad_mlp_score = adapter.compute_mlp_score(
        u=u,
        v=v,
        features_41=pad_f41,
        customer_stats_6=stats6,
        candidate_indices=pad_idx,
        candidate_mask=pad_mask,
    )
    pad_out = adapter.score_candidates(
        pad_inputs,
        u=u,
        v=v,
        customer_stats_6=stats6,
        candidate_indices=pad_idx,
        candidate_mask=pad_mask,
    )
    pad_score_valid_diff = float((pad_mlp_score[:, :-2] - req_mlp_score).abs().max().item())
    pad_logit_valid_diff = float((pad_out.logits[:, :-2] - req_out.logits).abs().max().item())
    pad_masked_score_zero = bool(torch.all(pad_mlp_score[:, -2:] == 0.0))
    pad_masked_logit_neginf = bool(torch.all(pad_out.logits[:, -2:] == -torch.inf))

    if mlp_score_diff > 1e-4 or logit_diff > 1e-4:
        raise AssertionError(f"Cache vs request adapter parity failed: score={mlp_score_diff}, logit={logit_diff}")
    if perm_score_diff > 1e-5 or perm_logit_diff > 1e-4:
        raise AssertionError(f"Slate permutation invariance failed: score={perm_score_diff}, logit={perm_logit_diff}")
    if pad_score_valid_diff > 1e-6 or pad_logit_valid_diff > 1e-5 or not pad_masked_score_zero or not pad_masked_logit_neginf:
        raise AssertionError("Padding/mask isolation check failed.")

    return {
        "rows_checked": rows,
        "max_abs_mlp_score_diff_request_vs_cache": mlp_score_diff,
        "max_abs_logit_diff_request_vs_cache": logit_diff,
        "max_abs_mlp_score_diff_reversed_slate": perm_score_diff,
        "max_abs_logit_diff_reversed_slate": perm_logit_diff,
        "max_abs_mlp_score_diff_with_nan_padding": pad_score_valid_diff,
        "max_abs_logit_diff_with_nan_padding": pad_logit_valid_diff,
        "masked_pad_mlp_score_zero": pad_masked_score_zero,
        "masked_pad_logit_neginf": pad_masked_logit_neginf,
    }
