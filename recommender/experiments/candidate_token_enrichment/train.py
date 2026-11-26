"""Controlled 3-arm Stage 2 campaign: pointwise (A) vs cand-attn (B) vs cand-attn + enriched tokens (C)."""

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
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader,
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
    verify_active_reference,
)
from recommender.experiments.rolling_candidate_v2.train import (
    snapshot_preserved_hashes as snapshot_core_17_hashes,
)
from recommender.experiments.static_enrichment.train import _make_downstream_ranker
from recommender.ranker import RankerLoss
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .benchmark import run_stage2_latency_benchmark
from .config import (
    ARM_A_NAME,
    ARM_B_NAME,
    ARM_C_NAME,
    ARTIFACT_ROOT,
    BATCH_SIZE,
    CACHE_ROOT,
    DEFAULT_DISCOVERY_RANKER,
    ENRICHMENT_INIT_SEED_OFFSET,
    EPOCHS,
    ETA_MIN,
    EVIDENCE_ROOT,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    GRAD_CLIP_NORM,
    LR,
    SEED,
    SLATE_POLICY,
    STAGE1_MLP_REF_RETRIEVER,
    STAGE1_MLP_REF_SELECTOR,
    WEIGHT_DECAY,
)
from .evaluate import (
    compute_three_arm_paired_intervals,
    evaluate_rolling_slate_metrics,
    measure_mechanism_diagnostics,
    predict_enrichment_ranker,
)
from .model import (
    CandidateTokenEnrichmentRanker,
    load_enrichment_ranker_checkpoint,
    save_enrichment_ranker_checkpoint,
)
from .normalization import ENRICHMENT_SCALAR_TRANSFORM_VERSION, enrichment_normalization_spec
from .report import build_reproduction_cli, write_campaign_reports


def snapshot_preserved_hashes(cache_root: Path = CACHE_ROOT) -> Dict[str, str]:
    """Capture all 17 core reference/historical digests plus sealed rolling v3 cache & report digests."""
    hashes = dict(snapshot_core_17_hashes())
    v3_dir = Path(cache_root)
    if v3_dir.is_dir():
        for path in sorted(v3_dir.iterdir()):
            if path.is_file():
                hashes[f"rolling_v3/{path.name}"] = file_digest(path)
    roll_ev = Path("experiments/rolling_candidate_v2")
    if roll_ev.is_dir():
        for path in sorted(roll_ev.iterdir()):
            if path.is_file():
                hashes[f"rolling_v2_evidence/{path.name}"] = file_digest(path)
    return hashes


def build_initial_arms(
    seed: int = SEED,
) -> Tuple[Any, CandidateTokenEnrichmentRanker, CandidateTokenEnrichmentRanker, CandidateTokenEnrichmentRanker]:
    """Create shared initial weights and clone identically across Arms A, B, and C."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dyn_ranker = _make_downstream_ranker(
        "dynamic_features",
        int(seed),
        STAGE1_MLP_REF_RETRIEVER,
        is_control_stage1=False,
    )
    dyn_ranker.eval()

    shared_base = CandidateTokenEnrichmentRanker.from_dynamic_context_ranker(
        dyn_ranker,
        candidate_attention_enabled=False,
        token_enrichment_enabled=False,
    )
    arm_a = shared_base.clone_arm(
        candidate_attention_enabled=False,
        token_enrichment_enabled=False,
    )
    arm_b = shared_base.clone_arm(
        candidate_attention_enabled=True,
        token_enrichment_enabled=False,
    )
    arm_c = shared_base.clone_arm(
        candidate_attention_enabled=True,
        token_enrichment_enabled=True,
        enrichment_seed=int(seed) + ENRICHMENT_INIT_SEED_OFFSET,
    )
    return dyn_ranker, arm_a, arm_b, arm_c


@torch.no_grad()
def verify_three_arm_init_parity(
    val_slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    *,
    seed: int = SEED,
    batch_rows: int = 32,
) -> Tuple[Dict[str, Any], Dict[str, CandidateTokenEnrichmentRanker]]:
    """Verify exact init parity (atol=0) across Arms A/B/C, shared weights, permutation, and padding."""
    dyn_ranker, arm_a, arm_b, arm_c = build_initial_arms(seed=seed)
    arm_a.eval()
    arm_b.eval()
    arm_c.eval()

    # Check shared initial weights match verbatim across A, B, C
    a_state = arm_a.state_dict()
    b_state = arm_b.state_dict()
    c_state = arm_c.state_dict()
    for key, val_a in a_state.items():
        if not torch.equal(val_a, b_state[key]) or not torch.equal(val_a, c_state[key]):
            raise AssertionError(f"Shared initial parameter mismatch across arms on {key!r}.")

    rows = min(int(batch_rows), len(val_slate["customer_ids"]))
    batch = next(iter(make_rolling_slate_loader(val_slate, batch_size=rows, shuffle=False)))
    cpu = torch.device("cpu")
    inputs_new = rolling_ranker_batch_inputs(batch, product_vectors.cpu(), cpu)
    inputs_old = {k: v for k, v in inputs_new.items() if k != "candidate_mask"}

    old_out = dyn_ranker(**inputs_old)
    out_a = arm_a(**inputs_new)
    out_b = arm_b(**inputs_new)
    out_c = arm_c(**inputs_new)
    out_c_no_attn = arm_c(**inputs_new, disable_candidate_attention=True)
    out_c_no_enrich = arm_c(**inputs_new, disable_token_enrichment=True)

    diff_a = float((out_a.logits - old_out.logits).abs().max().item())
    diff_b = float((out_b.logits - old_out.logits).abs().max().item())
    diff_c = float((out_c.logits - old_out.logits).abs().max().item())
    diff_c_no_attn = float((out_c_no_attn.logits - old_out.logits).abs().max().item())
    diff_c_no_enrich = float((out_c_no_enrich.logits - old_out.logits).abs().max().item())
    diff_mem_c = float((out_c.memory - out_b.memory).abs().max().item())
    diff_zcand_c = float((out_c.z_candidate - out_b.z_candidate).abs().max().item())
    diff_attn_c = float((out_c.attn_weights - out_b.attn_weights).abs().max().item())

    if any(
        d != 0.0
        for d in (
            diff_a,
            diff_b,
            diff_c,
            diff_c_no_attn,
            diff_c_no_enrich,
            diff_mem_c,
            diff_zcand_c,
            diff_attn_c,
        )
    ):
        raise AssertionError(
            "Three-arm zero-residual init parity failed: "
            f"a={diff_a:.3e}, b={diff_b:.3e}, c={diff_c:.3e}, "
            f"mem={diff_mem_c:.3e}, zcand={diff_zcand_c:.3e}, attn={diff_attn_c:.3e}."
        )

    # Candidate permutation equivariance on C
    rev_inputs = {
        k: (
            torch.flip(v, dims=[1])
            if k in ("pairwise_features", "candidate_vectors", "candidate_mask")
            else v
        )
        for k, v in inputs_new.items()
    }
    rev_c = arm_c(**rev_inputs)
    perm_diff_c = float((torch.flip(rev_c.logits, dims=[1]) - out_c.logits).abs().max().item())
    if perm_diff_c > 1e-6:
        raise AssertionError(f"Permutation equivariance failed at init for C: {perm_diff_c:.3e}")

    # NaN-padded candidate & history isolation on C
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
    hist_mask = inputs_new["attention_mask"] != 0
    if bool((~hist_mask).any()):
        seq_nan = inputs_new["sequence"].clone()
        seq_nan[~hist_mask] = float("nan")
        dates_nan = inputs_new["seq_purchase_dates"].clone()
        dates_nan[~hist_mask] = float("nan")
        pad_inputs["sequence"] = seq_nan
        pad_inputs["seq_purchase_dates"] = dates_nan

    pad_c = arm_c(**pad_inputs)
    pad_valid_diff_c = float((pad_c.logits[:, :-2] - out_c.logits).abs().max().item())
    pad_neginf_c = bool(torch.all(pad_c.logits[:, -2:] == -torch.inf))
    if pad_valid_diff_c > 1e-6 or not pad_neginf_c:
        raise AssertionError("Padding/mask isolation failed at init for C.")

    diag = {
        "rows_checked": rows,
        "shared_weights_identical_across_arms": True,
        "dyn_425d_parameters": int(sum(p.numel() for p in dyn_ranker.parameters())),
        "arm_a_parameters": int(sum(p.numel() for p in arm_a.parameters())),
        "arm_a_trainable_parameters": int(sum(p.numel() for p in arm_a.parameters() if p.requires_grad)),
        "arm_b_parameters": int(sum(p.numel() for p in arm_b.parameters())),
        "arm_b_trainable_parameters": int(sum(p.numel() for p in arm_b.parameters() if p.requires_grad)),
        "arm_c_parameters": int(sum(p.numel() for p in arm_c.parameters())),
        "arm_c_trainable_parameters": int(sum(p.numel() for p in arm_c.parameters() if p.requires_grad)),
        "max_abs_logit_diff_arm_a": diff_a,
        "max_abs_logit_diff_arm_b": diff_b,
        "max_abs_logit_diff_arm_c": diff_c,
        "max_abs_logit_diff_c_disable_attn": diff_c_no_attn,
        "max_abs_logit_diff_c_disable_enrich": diff_c_no_enrich,
        "max_abs_memory_diff_c_vs_b": diff_mem_c,
        "max_abs_zcand_diff_c_vs_b": diff_zcand_c,
        "max_abs_attn_diff_c_vs_b": diff_attn_c,
        "max_abs_logit_diff_reversed_slate_c": perm_diff_c,
        "max_abs_logit_diff_nan_padding_c": pad_valid_diff_c,
        "masked_pad_logit_neginf_c": pad_neginf_c,
    }
    return diag, {ARM_A_NAME: arm_a, ARM_B_NAME: arm_b, ARM_C_NAME: arm_c}


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
    arm_name: str,
    candidate_attention_enabled: bool,
    token_enrichment_enabled: bool,
    meta: Mapping[str, Any],
    slate_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "arm_name": arm_name,
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
        "observed_source_digest": meta["observed_source_digest"],
        "slate_policy": SLATE_POLICY,
        "train_slate_identity": slate_manifest["split_cache_identity"]["train"],
        "val_slate_identity": slate_manifest["split_cache_identity"]["val"],
        "candidate_attention_enabled": bool(candidate_attention_enabled),
        "token_enrichment_enabled": bool(token_enrichment_enabled),
        "enrichment_transform_version": ENRICHMENT_SCALAR_TRANSFORM_VERSION,
        "heldout_access": False,
        "metadata_policy": "snapshot_unverified",
        "product_universe": EXPECTED_PRODUCT_UNIVERSE,
    }


def fit_single_enrichment_arm(
    *,
    arm_name: str,
    init_model: CandidateTokenEnrichmentRanker,
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
    checkpoint_writer=save_enrichment_ranker_checkpoint,
    checkpoint_loader=load_enrichment_ranker_checkpoint,
    experiment_name: str = "candidate_token_enrichment_v1",
) -> Dict[str, Any]:
    """Fit one Stage 2 arm, strictly reload both selected checkpoints, replay, and run diagnostics."""
    dev = get_device(device)
    ref_disc = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu", weights_only=False)
    recall_guard = float(ref_disc["discovery_min_overall_recall"])
    ndcg_guard = float(ref_disc["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)

    cand_attn = bool(init_model.candidate_attention_enabled)
    tok_enrich = bool(init_model.token_enrichment_enabled)
    provenance = _build_provenance(
        arm_name=arm_name,
        candidate_attention_enabled=cand_attn,
        token_enrichment_enabled=tok_enrich,
        meta=meta,
        slate_manifest=slate_manifest,
    )
    feature_contract_digest = meta["feature_schema_digest"]

    run_name = f"stage2-{arm_name}-seed{seed}"
    settings = {
        "stage": experiment_name,
        "arm_name": arm_name,
        "candidate_attention_enabled": cand_attn,
        "token_enrichment_enabled": tok_enrich,
        "enrichment_transform_version": ENRICHMENT_SCALAR_TRANSFORM_VERSION,
        "enrichment_normalization": enrichment_normalization_spec(
            excluded_statistics=init_model.base.excluded_statistics
        ),
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
        "observed_source_digest": meta["observed_source_digest"],
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

    # Paired training RNG & loader alignment
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = init_model.clone_arm(
        candidate_attention_enabled=cand_attn,
        token_enrichment_enabled=tok_enrich,
    )
    if tok_enrich:
        # Preserve exact initial enrichment weights cloned from init_model
        model.load_state_dict(init_model.state_dict(), strict=True)
    model = model.to(dev)
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
            loss = loss_fn(logits, batch["labels"].to(dev), candidate_mask=inputs.get("candidate_mask"))["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss in {arm_name}, epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_pred = predict_enrichment_ranker(model, val_slate, vectors_dev, dev)
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
            checkpoint_writer(
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
            checkpoint_writer(
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
            f"[candidate_token_enrichment/{arm_name}] seed={seed} ep={epoch:02d} loss={train_loss_mean:.4f} "
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

    def _replay_and_diagnose(ckpt_path: Path, tag: str) -> Dict[str, Any]:
        loaded, ckpt = checkpoint_loader(
            ckpt_path,
            expected_provenance=provenance,
            expected_feature_contract_digest=feature_contract_digest,
            expected_candidate_attention_enabled=cand_attn,
            expected_token_enrichment_enabled=tok_enrich,
            map_location="cpu",
        )
        loaded = loaded.to(dev)
        pred = predict_enrichment_ranker(loaded, val_slate, vectors_dev, dev)
        pred_npz_path = directory / f"predictions_{tag}.npz"
        logits_pt_path = directory / f"logits_{tag}.pt"
        replayed = evaluate_rolling_slate_metrics(
            pred["scores"],
            val_data,
            val_slate["candidate_indices"],
            save_path=pred_npz_path,
        )
        replayed["val_loss"] = pred["val_loss"]
        replayed["avg_token0_weight"] = pred["avg_token0_weight"]
        replayed["avg_history_weight"] = pred["avg_history_weight"]
        if replayed != ckpt["metrics"]:
            raise AssertionError(f"{tag} checkpoint replay mismatch for {run_name}.")

        atomic_torch_save({"logits": pred["scores"].to(dtype=torch.float32)}, logits_pt_path)
        diag = measure_mechanism_diagnostics(
            loaded,
            val_slate,
            val_data,
            vectors_dev,
            dev,
            baseline_metrics=replayed,
        )
        return {
            "epoch": int(ckpt["epoch"]),
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_digest(ckpt_path),
            "predictions_npz": str(pred_npz_path),
            "predictions_sha256": file_digest(pred_npz_path),
            "logits_pt": str(logits_pt_path),
            "logits_sha256": file_digest(logits_pt_path),
            "metrics": replayed,
            "mechanism_diagnostics": diag,
        }

    disc_info = _replay_and_diagnose(disc_path, "discovery")
    ndcg_info = _replay_and_diagnose(ndcg_path, "ndcg")

    history_path = directory / "history.json"
    atomic_json(history, history_path, overwrite=False)
    result: Dict[str, Any] = {
        "arm_name": arm_name,
        "run_dir": str(directory),
        "candidate_attention_enabled": cand_attn,
        "token_enrichment_enabled": tok_enrich,
        "enrichment_transform_version": ENRICHMENT_SCALAR_TRANSFORM_VERSION,
        "teacher_score_included": False,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "settings": settings,
        "provenance": provenance,
        "qualified": qualified,
        "parameters": int(sum(p.numel() for p in init_model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in init_model.parameters() if p.requires_grad)),
        "training_seconds": float(elapsed),
        "device": str(dev),
        "discovery": disc_info,
        "ndcg": ndcg_info,
        "history_json": str(history_path),
        "history_sha256": file_digest(history_path),
        "history": history,
    }
    atomic_json(result, directory / "result.json", overwrite=False)
    atomic_json(
        {
            "version": 1,
            "status": "complete",
            "settings_digest": json_digest(settings),
            "discovery_checkpoint": disc_rel,
            "discovery_sha256": disc_info["checkpoint_sha256"],
            "ndcg_checkpoint": "best_ndcg.pt",
            "ndcg_sha256": ndcg_info["checkpoint_sha256"],
            "qualified": qualified,
        },
        directory / "run.json",
        overwrite=True,
    )
    return result


def run_three_arm_campaign(
    *,
    cache_root: Path = CACHE_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full 3-arm seed-42 10-epoch campaign, evaluation, diagnostics, and latency benchmark."""
    before_hashes = snapshot_preserved_hashes(cache_root=cache_root)
    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=cache_root,
        build=False,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    _, scores_manifest = prepare_or_load_rolling_selector_scores(
        campaign_data, info, meta, cache_root=cache_root
    )
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(
        campaign_data, info, meta, cache_root=cache_root
    )
    product_vectors = campaign_data["train"]["v"]

    init_diag, init_arms = verify_three_arm_init_parity(
        slates["val"], product_vectors, seed=seed
    )

    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name in (ARM_A_NAME, ARM_B_NAME, ARM_C_NAME):
        arm_results[arm_name] = fit_single_enrichment_arm(
            arm_name=arm_name,
            init_model=init_arms[arm_name],
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
        sel_key: compute_three_arm_paired_intervals(
            arm_results[ARM_A_NAME][sel_key]["predictions_npz"],
            arm_results[ARM_B_NAME][sel_key]["predictions_npz"],
            arm_results[ARM_C_NAME][sel_key]["predictions_npz"],
        )
        for sel_key in ("discovery", "ndcg")
    }

    # Load fitted NDCG checkpoints on CPU for real-input CPU & MPS latency benchmarking
    bench_models = {}
    for arm_name in (ARM_A_NAME, ARM_B_NAME, ARM_C_NAME):
        loaded_m, _ = load_enrichment_ranker_checkpoint(
            arm_results[arm_name]["ndcg"]["checkpoint"],
            map_location="cpu",
        )
        bench_models[arm_name] = loaded_m

    latency = run_stage2_latency_benchmark(
        bench_models,
        slates["val"],
        product_vectors,
        evidence_root=evidence_root,
        save=True,
    )

    after_hashes = snapshot_preserved_hashes(cache_root=cache_root)
    if after_hashes != before_hashes:
        raise AssertionError("Preserved reference or rolling v3 hashes changed during execution!")

    summary: Dict[str, Any] = {
        "training_executed": True,
        "heldout_access": False,
        "promoted": False,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "active_reference_id": EXPECTED_REFERENCE_ID,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "slate_policy": SLATE_POLICY,
        "teacher_score_included": False,
        "observed_source_digest": meta["observed_source_digest"],
        "feature_bundle_identity": feat_manifest["identity"],
        "rolling_selector_scores_identity": scores_manifest["identity"],
        "slate_cache_identities": slate_manifest["split_cache_identity"],
        "scalar_normalization_spec": enrichment_normalization_spec(
            excluded_statistics=("history_repeat",)
        ),
        "three_arm_init_parity": init_diag,
        "arms": arm_results,
        "paired_bootstrap_95ci": intervals,
        "latency_benchmark": latency,
        "reproduction_cli": build_reproduction_cli(seed=seed, epochs=epochs, batch_size=batch_size),
        "preserved_hashes": after_hashes,
    }
    ev_dir = Path(evidence_root)
    ev_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, ev_dir / "summary.json", overwrite=True)
    report_files = write_campaign_reports(summary, ev_dir)
    summary["report_files"] = report_files
    atomic_json(summary, ev_dir / "summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled 3-arm Stage 2 campaign: pointwise (A) vs candidate self-attention (B) "
            "vs candidate self-attention + pre-attention token enrichment (C)."
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
        "--verify-init-only",
        action="store_true",
        help="Verify 3-arm zero-residual init parity without training.",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Execute the full 3-arm seed-42 10-epoch campaign, evaluation, and benchmark.",
    )
    args = parser.parse_args()

    if not args.verify_init_only and not args.execute_training:
        parser.error("Pass either --verify-init-only or --execute-training.")

    if args.execute_training:
        summary = run_three_arm_campaign(
            cache_root=args.cache_root,
            artifact_root=args.artifact_root,
            evidence_root=args.evidence_root,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "training_executed": summary["training_executed"],
                    "heldout_access": summary["heldout_access"],
                    "promoted": summary["promoted"],
                    "report_files": summary["report_files"],
                    "arms": {
                        name: {
                            "qualified": res["qualified"],
                            "discovery_epoch": res["discovery"]["epoch"],
                            "discovery_recall@10": res["discovery"]["metrics"]["recall@10"],
                            "discovery_ndcg@10": res["discovery"]["metrics"]["ndcg@10"],
                            "discovery_new_hits@10": res["discovery"]["metrics"]["new_hits@10"],
                            "ndcg_epoch": res["ndcg"]["epoch"],
                            "ndcg_recall@10": res["ndcg"]["metrics"]["recall@10"],
                            "ndcg_ndcg@10": res["ndcg"]["metrics"]["ndcg@10"],
                            "ndcg_new_hits@10": res["ndcg"]["metrics"]["new_hits@10"],
                        }
                        for name, res in summary["arms"].items()
                    },
                },
                indent=2,
            )
        )
        return

    verify_active_reference(load_selector=False)
    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=args.cache_root,
        build=False,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    _, scores_manifest = prepare_or_load_rolling_selector_scores(
        campaign_data, info, meta, cache_root=args.cache_root
    )
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(
        campaign_data, info, meta, cache_root=args.cache_root
    )
    init_diag, _ = verify_three_arm_init_parity(
        slates["val"], campaign_data["train"]["v"], seed=args.seed
    )
    print(
        json.dumps(
            {
                "training_executed": False,
                "heldout_access": False,
                "observed_source_digest": meta["observed_source_digest"],
                "feature_bundle_identity": feat_manifest["identity"],
                "rolling_selector_scores_identity": scores_manifest["identity"],
                "slate_cache_identities": slate_manifest["split_cache_identity"],
                "three_arm_init_parity": init_diag,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
