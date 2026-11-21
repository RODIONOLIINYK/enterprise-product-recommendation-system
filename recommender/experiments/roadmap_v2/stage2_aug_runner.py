"""Stage 2 runner: Multi-Task 60-Day Auxiliary Head + History-SKU Dropout on BOTH Stage 1 MLP and Stage 2 Transformer."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.development import load_development_purchases
from recommender.experiment_signals import sync
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_DISCOVERY_RANKER
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader,
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .augmentation import (
    apply_stage1_history_sku_dropout,
    apply_stage2_history_sku_dropout,
    build_seq_universe_indices,
    build_strict_60d_aux_labels,
)
from .base_name_features import (
    assemble_arm_features_numpy,
    prepare_or_load_base_name_feature_bundle,
)
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE1_BATCH_SIZE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE1_OMITTED_CALENDAR_FEATURES,
    STAGE1_POS_WEIGHT,
    STAGE1_WEIGHT_DECAY,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_LR,
    STAGE2_WEIGHT_DECAY,
    feature_names_for_arm,
)
from .stage1_mlp import (
    RoadmapStage1Selector,
    compute_pack_switch_masks,
    evaluate_stage1_with_breakdown,
    score_roadmap_stage1_split,
)
from .stage2_ranker import (
    RoadmapPointwiseRanker,
    attach_arm_features_to_fixed_slate,
    build_slate_from_stage1_scores,
    evaluate_stage2_with_breakdown,
    predict_roadmap_stage2,
)


def _prod_bn_id_tensor(meta: Mapping[str, Any]) -> torch.Tensor:
    bns = meta["product_base_names"]
    uniq = {bn: i + 1 for i, bn in enumerate(sorted(set(bns) - {""}))}
    return torch.tensor([uniq.get(bn, 0) for bn in bns], dtype=torch.long)


def fit_stage1_augmented(
    *,
    arm: str,
    variant_name: str,
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    aux_60d_labels: Optional[torch.Tensor] = None,
    aux_lambda: float = 0.0,
    aux_pos_weight: float = 10.0,
    p_drop: float = 0.0,
    artifact_root: Path = ARTIFACT_ROOT / "stage2_augmentation",
    seed: int = SEED,
    epochs: int = STAGE1_EPOCHS,
    batch_size: int = STAGE1_BATCH_SIZE,
    lr: float = STAGE1_LR,
    weight_decay: float = STAGE1_WEIGHT_DECAY,
    pos_weight: float = STAGE1_POS_WEIGHT,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train Stage 1 MLP with optional 60-day auxiliary head (discarded at inference) and/or History-SKU Dropout."""
    dev = get_device(device)
    feature_names = feature_names_for_arm(arm)
    use_aux = aux_lambda > 0.0 and aux_60d_labels is not None
    run_name = f"stage1-{arm}-{variant_name}-seed{seed}"
    run_dir = Path(artifact_root) / "runs" / run_name

    settings = {
        "stage": "roadmap_v2_stage2_aug_stage1",
        "arm": arm,
        "variant_name": variant_name,
        "aux_60d_enabled": bool(use_aux),
        "aux_lambda": float(aux_lambda),
        "aux_pos_weight": float(aux_pos_weight),
        "p_drop": float(p_drop),
        "feature_names": list(feature_names),
        "statistics_policy": "rolling_observed_v2",
        "rolling_features_identity": meta["rolling_features_identity"],
        "base_name_identity": meta["base_name_identity"],
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "pos_weight": float(pos_weight),
        "heldout_access": False,
    }
    settings_digest = json_digest(settings)

    if (run_dir / "result.json").is_file() and (run_dir / "run.json").is_file():
        rm = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if rm.get("status") == "complete" and rm.get("settings_digest") == settings_digest:
            return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "running", "settings_digest": settings_digest}, run_dir / "run.json", overwrite=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = RoadmapStage1Selector(feature_names, aux_60d_enabled=use_aux).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    tr = campaign_data["train"]
    va = campaign_data["val"]
    v_dev = tr["v"].to(dev)
    tr_f41 = tr["features"]
    tr_f19 = tr.get("base_name_features")
    prod_bn_ids = _prod_bn_id_tensor(meta)
    pack_switch_mask, true_new_bn_mask = compute_pack_switch_masks(
        va, info["product_ids"], meta["all_pid_to_base_name"]
    )

    perm_gen = torch.Generator().manual_seed(int(seed))
    drop_gen = torch.Generator().manual_seed(int(seed) + 10007)
    pos_weight_t = torch.tensor(float(pos_weight), device=dev)
    aux_pw_t = torch.tensor(float(aux_pos_weight), device=dev)

    best_key = (-1, -1.0, -1.0)
    best_epoch = 0
    history = []

    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses, main_losses, aux_losses = [], [], []
        total_dropped = 0
        perm = torch.randperm(len(tr["u"]), generator=perm_gen)
        for batch_rows in perm.split(int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            rows_np = batch_rows.numpy()
            f41_np = np.array(tr_f41[rows_np], dtype=np.float32)
            f19_np = np.array(tr_f19[rows_np], dtype=np.float32) if tr_f19 is not None else None
            feats = torch.from_numpy(assemble_arm_features_numpy(f41_np, f19_np, arm)).to(dev)
            allowed = tr["allowed"][batch_rows].to(dev)
            target = tr["target"][batch_rows].to(dev).float()

            if p_drop > 0.0:
                feats, n_drop = apply_stage1_history_sku_dropout(
                    feats, target, prod_bn_ids, feature_names, p_drop=p_drop, generator=drop_gen
                )
                total_dropped += n_drop

            if use_aux:
                main_logits, aux_logits = model(
                    u=tr["u"][batch_rows].to(dev),
                    v=v_dev,
                    features=feats,
                    customer_stats_6=tr["customer_stats_6"][batch_rows].to(dev),
                    return_aux=True,
                )
                loss_main = F.binary_cross_entropy_with_logits(
                    main_logits[allowed], target[allowed], pos_weight=pos_weight_t
                )
                aux_t = aux_60d_labels[batch_rows].to(dev).float()
                loss_aux = F.binary_cross_entropy_with_logits(
                    aux_logits[allowed], aux_t[allowed], pos_weight=aux_pw_t
                )
                loss = loss_main + float(aux_lambda) * loss_aux
                aux_losses.append(float(loss_aux.detach().cpu()))
            else:
                main_logits = model(
                    u=tr["u"][batch_rows].to(dev),
                    v=v_dev,
                    features=feats,
                    customer_stats_6=tr["customer_stats_6"][batch_rows].to(dev),
                    return_aux=False,
                )
                loss_main = F.binary_cross_entropy_with_logits(
                    main_logits[allowed], target[allowed], pos_weight=pos_weight_t
                )
                loss = loss_main

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            main_losses.append(float(loss_main.detach().cpu()))
        scheduler.step()

        # Inference uses ONLY main_head (aux_60d_head untouched!)
        val_scores = score_roadmap_stage1_split(model, va, arm, dev)
        metrics = evaluate_stage1_with_breakdown(val_scores, va, pack_switch_mask, true_new_bn_mask)
        key = (int(metrics["new_hits@50"]), float(metrics["ndcg@10"]), float(metrics["recall@10"]))
        ep_sec = time.perf_counter() - ep_t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "main_loss": float(np.mean(main_losses)),
                "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
                "dropped_queries": int(total_dropped),
                "epoch_seconds": ep_sec,
                "metrics": metrics,
            }
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            # Strip aux_60d_head from saved inference state so inference artifact is identical in size/speed!
            inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "feature_names": list(feature_names),
                    "omitted_names": list(STAGE1_OMITTED_CALENDAR_FEATURES),
                    "input_dim": model.input_dim,
                    "aux_60d_stripped_at_inference": bool(use_aux),
                    "model_state_dict": inf_state,
                },
                run_dir / "best.pt",
            )
        print(
            f"[stage2-aug/S1/{variant_name}] ep={epoch:02d} loss={np.mean(losses):.5f} drop={total_dropped} "
            f"R@5={metrics['recall@5']:.5f} R@10={metrics['recall@10']:.5f} NDCG@10={metrics['ndcg@10']:.5f} "
            f"New@10={metrics['new_hits@10']} (ps={metrics['pack_switch_hits@10']}, tn={metrics['true_new_bn_hits@10']}) "
            f"New@25={metrics['new_hits@25']} New@50={metrics['new_hits@50']} New@100={metrics['new_hits@100']} ({ep_sec:.1f}s)",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - started

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    reloaded = RoadmapStage1Selector(feature_names, aux_60d_enabled=False).to(dev)
    reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    val_scores = score_roadmap_stage1_split(reloaded, va, arm, dev)
    replayed = evaluate_stage1_with_breakdown(
        val_scores, va, pack_switch_mask, true_new_bn_mask, save_predictions=run_dir / "predictions.npz"
    )
    if replayed != ckpt["metrics"]:
        raise AssertionError(f"Stage 1 augmented replay mismatch for {run_name}.")

    train_scores = score_roadmap_stage1_split(reloaded, tr, arm, dev)
    best_sha = file_digest(run_dir / "best.pt")
    atomic_torch_save({"arm": arm, "variant": variant_name, "selector_sha256": best_sha, "scores": train_scores}, run_dir / "train_scores.pt")
    atomic_torch_save({"arm": arm, "variant": variant_name, "selector_sha256": best_sha, "scores": val_scores}, run_dir / "val_scores.pt")

    result = {
        "arm": arm,
        "variant_name": variant_name,
        "run_dir": str(run_dir),
        "selected_epoch": int(best_epoch),
        "training_seconds": float(elapsed),
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_sha256": best_sha,
        "predictions_npz": str(run_dir / "predictions.npz"),
        "train_scores_pt": str(run_dir / "train_scores.pt"),
        "val_scores_pt": str(run_dir / "val_scores.pt"),
        "metrics": replayed,
        "settings": settings,
        "history": history,
    }
    atomic_json(history, run_dir / "history.json", overwrite=True)
    atomic_json(result, run_dir / "result.json", overwrite=True)
    atomic_json({"status": "complete", "settings_digest": settings_digest, "checkpoint_sha256": best_sha}, run_dir / "run.json", overwrite=True)
    return result


def fit_stage2_augmented(
    *,
    arm: str,
    variant_name: str,
    train_slate: Mapping[str, Any],
    val_slate: Mapping[str, Any],
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    product_vectors: torch.Tensor,
    product_ids: Sequence[str],
    meta: Mapping[str, Any],
    aux_60d_labels: Optional[torch.Tensor] = None,
    aux_lambda: float = 0.0,
    aux_pos_weight: float = 5.0,
    p_drop: float = 0.0,
    slate_tag: str = "fixed_rolling100",
    artifact_root: Path = ARTIFACT_ROOT / "stage2_augmentation",
    seed: int = SEED,
    epochs: int = STAGE2_EPOCHS,
    batch_size: int = STAGE2_BATCH_SIZE,
    lr: float = STAGE2_LR,
    weight_decay: float = STAGE2_WEIGHT_DECAY,
    eta_min: float = STAGE2_ETA_MIN,
    grad_clip_norm: float = STAGE2_GRAD_CLIP_NORM,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train Stage 2 UPS-CANR with optional 60-day aux head (stripped at inference) and/or History-SKU Dropout."""
    dev = get_device(device)
    ref_disc = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu", weights_only=False)
    recall_guard = float(ref_disc["discovery_min_overall_recall"])
    ndcg_guard = float(ref_disc["val_metrics"]["val_ndcg@10"])

    feature_names = feature_names_for_arm(arm)
    num_features = int(train_slate["pairwise_features"].shape[-1])
    use_aux = aux_lambda > 0.0 and aux_60d_labels is not None
    run_name = f"stage2-{arm}-{variant_name}-{slate_tag}-seed{seed}"
    run_dir = Path(artifact_root) / "runs" / run_name

    settings = {
        "stage": "roadmap_v2_stage2_aug_stage2",
        "arm": arm,
        "variant_name": variant_name,
        "slate_tag": slate_tag,
        "num_features": num_features,
        "aux_60d_enabled": bool(use_aux),
        "aux_lambda": float(aux_lambda),
        "aux_pos_weight": float(aux_pos_weight),
        "p_drop": float(p_drop),
        "statistics_policy": "rolling_observed_v2",
        "rolling_features_identity": meta["rolling_features_identity"],
        "base_name_identity": meta["base_name_identity"],
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "heldout_access": False,
    }
    settings_digest = json_digest(settings)

    if (run_dir / "result.json").is_file() and (run_dir / "run.json").is_file():
        rm = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if rm.get("status") == "complete" and rm.get("settings_digest") == settings_digest:
            return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "running", "settings_digest": settings_digest}, run_dir / "run.json", overwrite=True)

    pack_switch_mask, true_new_bn_mask = compute_pack_switch_masks(
        val_data, product_ids, meta["all_pid_to_base_name"]
    )
    prod_bn_ids = _prod_bn_id_tensor(meta)
    seq_u_all = build_seq_universe_indices(train_data["source"]["seq_canonical_skus"], product_ids)
    # Gather per-slate 60-day auxiliary labels cleanly (N_train, 100)
    aux_slate_labels = (
        aux_60d_labels.gather(1, train_slate["candidate_indices"]).float()
        if use_aux
        else None
    )

    model = RoadmapPointwiseRanker(num_features=num_features, seed=seed, aux_60d_enabled=use_aux).to(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    vectors_dev = product_vectors.to(dev)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=float(eta_min))
    loss_fn = RankerLoss().to(dev)
    aux_loss_fn = RankerLoss(pos_weight=float(aux_pos_weight), listwise_weight=1.0).to(dev)

    # Index-based deterministic batching so we can align batch_rows with seq_u_all & aux_slate_labels
    n_train = len(train_slate["customer_ids"])
    perm_gen = torch.Generator().manual_seed(int(seed))
    drop_gen = torch.Generator().manual_seed(int(seed) + 20011)

    best_disc_key = (-1, -1, -1.0)
    best_ndcg_key = (-1.0, -1.0, -1)
    history = []

    def _slice_batch(row_idx: torch.Tensor) -> Dict[str, Any]:
        return {
            "text_embedding": train_slate["text_embedding"][row_idx],
            "city_id": train_slate["city_id"][row_idx],
            "region_id": train_slate["region_id"][row_idx],
            "industry_id": train_slate["industry_id"][row_idx],
            "tabular_features": train_slate["tabular_features"][row_idx],
            "sequence": train_slate["sequence"][row_idx],
            "attention_mask": train_slate["attention_mask"][row_idx],
            "seq_product_ids": train_slate["seq_product_ids"][row_idx],
            "seq_purchase_dates": train_slate["seq_purchase_dates"][row_idx],
            "seq_category_ids": train_slate["seq_category_ids"][row_idx],
            "seq_subcategory_ids": train_slate["seq_subcategory_ids"][row_idx],
            "seq_business_line_ids": train_slate["seq_business_line_ids"][row_idx],
            "prediction_day": train_slate["prediction_days"][row_idx],
            "stage1_customer_vectors": train_slate["stage1_customer_vectors"][row_idx],
            "candidate_indices": train_slate["candidate_indices"][row_idx],
            "candidate_mask": train_slate["candidate_mask"][row_idx],
            "pairwise_features": train_slate["pairwise_features"][row_idx],
            "labels": train_slate["labels"][row_idx],
        }

    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses, main_losses, aux_losses = [], [], []
        total_dropped = 0
        perm = torch.randperm(n_train, generator=perm_gen)
        for batch_rows in perm.split(int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            batch = _slice_batch(batch_rows)
            inputs = rolling_ranker_batch_inputs(batch, vectors_dev, dev)
            labels = batch["labels"].to(dev)

            if p_drop > 0.0:
                inputs, n_drop = apply_stage2_history_sku_dropout(
                    inputs,
                    labels,
                    batch["candidate_indices"],
                    seq_u_all[batch_rows],
                    prod_bn_ids,
                    feature_names,
                    p_drop=p_drop,
                    drop_base_name_in_sequence=True,
                    generator=drop_gen,
                )
                total_dropped += n_drop

            if use_aux:
                out, aux_logits = model(**inputs, return_aux=True)
                loss_main = loss_fn(out.logits, labels, candidate_mask=inputs.get("candidate_mask"))["loss"]
                aux_y = aux_slate_labels[batch_rows].to(dev)
                loss_aux = aux_loss_fn(aux_logits, aux_y, candidate_mask=inputs.get("candidate_mask"))["loss"]
                loss = loss_main + float(aux_lambda) * loss_aux
                aux_losses.append(float(loss_aux.detach().cpu()))
            else:
                out = model(**inputs, return_aux=False)
                loss_main = loss_fn(out.logits, labels, candidate_mask=inputs.get("candidate_mask"))["loss"]
                loss = loss_main

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            main_losses.append(float(loss_main.detach().cpu()))
        scheduler.step()

        # Validation uses ONLY main_head (aux_60d_head untouched!)
        val_pred = predict_roadmap_stage2(model, val_slate, vectors_dev, dev)
        metrics = evaluate_stage2_with_breakdown(
            val_pred["scores"], val_data, val_slate["candidate_indices"], pack_switch_mask, true_new_bn_mask
        )
        metrics["val_loss"] = val_pred["val_loss"]
        qualified_epoch = bool(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard)
        disc_key = (int(qualified_epoch), int(metrics["new_hits@10"]), float(metrics["ndcg@10"]))
        ndcg_key = (float(metrics["ndcg@10"]), float(metrics["recall@10"]), int(metrics["new_hits@10"]))
        ep_sec = time.perf_counter() - ep_t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "main_loss": float(np.mean(main_losses)),
                "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
                "dropped_queries": int(total_dropped),
                "qualified": qualified_epoch,
                "epoch_seconds": ep_sec,
                "metrics": metrics,
            }
        )

        # Strip aux_60d_head from saved inference weights!
        inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}
        if disc_key > best_disc_key:
            best_disc_key = disc_key
            atomic_torch_save(
                {"settings": settings, "epoch": epoch, "metrics": metrics, "num_features": num_features, "model_state_dict": inf_state},
                run_dir / "best_discovery.pt",
            )
        if ndcg_key > best_ndcg_key:
            best_ndcg_key = ndcg_key
            atomic_torch_save(
                {"settings": settings, "epoch": epoch, "metrics": metrics, "num_features": num_features, "model_state_dict": inf_state},
                run_dir / "best_ndcg.pt",
            )
        print(
            f"[stage2-aug/S2/{variant_name}/{slate_tag}] ep={epoch:02d} loss={np.mean(losses):.4f} drop={total_dropped} "
            f"R@5={metrics['recall@5']:.5f} R@10={metrics['recall@10']:.5f} NDCG@10={metrics['ndcg@10']:.5f} "
            f"New@10={metrics['new_hits@10']} (ps={metrics['pack_switch_hits@10']}, tn={metrics['true_new_bn_hits@10']}) ({ep_sec:.1f}s)",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - started

    selections: Dict[str, Any] = {}
    for sel_name, fname in (("discovery", "best_discovery.pt"), ("ndcg", "best_ndcg.pt")):
        ckpt_path = run_dir / fname
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reloaded = RoadmapPointwiseRanker(num_features=num_features, seed=seed, aux_60d_enabled=False).to(dev)
        reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
        pred = predict_roadmap_stage2(reloaded, val_slate, vectors_dev, dev)
        replayed = evaluate_stage2_with_breakdown(
            pred["scores"], val_data, val_slate["candidate_indices"], pack_switch_mask, true_new_bn_mask,
            save_path=run_dir / f"predictions_{sel_name}.npz",
        )
        replayed["val_loss"] = pred["val_loss"]
        if replayed != ckpt["metrics"]:
            raise AssertionError(f"Stage 2 augmented {sel_name} replay mismatch for {run_name}.")
        selections[sel_name] = {
            "epoch": int(ckpt["epoch"]),
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_digest(ckpt_path),
            "predictions_npz": str(run_dir / f"predictions_{sel_name}.npz"),
            "metrics": replayed,
        }

    result = {
        "arm": arm,
        "variant_name": variant_name,
        "slate_tag": slate_tag,
        "run_dir": str(run_dir),
        "training_seconds": float(elapsed),
        "discovery": selections["discovery"],
        "ndcg": selections["ndcg"],
        "settings": settings,
        "history": history,
    }
    atomic_json(history, run_dir / "history.json", overwrite=True)
    atomic_json(result, run_dir / "result.json", overwrite=True)
    atomic_json(
        {
            "status": "complete",
            "settings_digest": settings_digest,
            "discovery_sha256": selections["discovery"]["checkpoint_sha256"],
            "ndcg_sha256": selections["ndcg"]["checkpoint_sha256"],
        },
        run_dir / "run.json",
        overwrite=True,
    )
    return result
