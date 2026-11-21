"""Stage 2 Transformer (UPS-CANR pointwise) for roadmap_v2: extensible features, aux 60d head & dropout."""

import copy
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.rolling_candidate_v2.config import (
    DEFAULT_DISCOVERY_RANKER,
    SLATE_POLICY,
    SLATE_SIZE,
    STAGE1_MLP_REF_RETRIEVER,
)
from recommender.experiments.rolling_candidate_v2.data import (
    RollingSlateDataset,
    make_rolling_slate_loader,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import (
    evaluate_rolling_slate_metrics,
    paired_rolling_intervals,
)
from recommender.experiments.static_enrichment.train import _make_downstream_ranker
from recommender.ranker import RankerLoss, RankerOutput
from recommender.shared_data import atomic_torch_save
from recommender.train_ranker import collate_ranker_batch
from recommender.train_retriever import get_device

from .base_name_features import assemble_arm_features_numpy
from .config import (
    ARTIFACT_ROOT,
    SEED,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_LR,
    STAGE2_WEIGHT_DECAY,
    feature_names_for_arm,
)
from .stage1_mlp import compute_pack_switch_masks


class RoadmapPointwiseRanker(nn.Module):
    """UPS-CANR pointwise ranker supporting arbitrary feature widths and optional training-only 60d aux head.

    Exact initialization parity with `arm_a_control_pointwise_425d` (`pointwise425_ndcg`) when `num_features == 41`.
    """

    def __init__(
        self,
        *,
        num_features: int = 41,
        seed: int = SEED,
        aux_60d_enabled: bool = False,
        aux_60d_dim: int = 1,
    ):
        super().__init__()
        if int(num_features) < 13:
            raise ValueError(f"Expected num_features >= 13, got {num_features}.")
        self.num_features = int(num_features)
        self.extra_dim = self.num_features - 13
        self.aux_60d_enabled = bool(aux_60d_enabled)

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        dyn = _make_downstream_ranker(
            "dynamic_features",
            int(seed),
            STAGE1_MLP_REF_RETRIEVER,
            is_control_stage1=False,
        )
        self.base = dyn.base
        self.base.configure_cached_training(True)
        self.base.mlp.requires_grad_(False)

        # Extract 425D scoring head from dyn.head and expand/preserve cleanly if num_features != 41
        src_head = copy.deepcopy(dyn.head)
        base_encoded_dim = int(self.base.hidden_dim) * 3 + 13  # 397
        target_in_dim = base_encoded_dim + self.extra_dim
        old_lin: nn.Linear = src_head[0]
        if target_in_dim != old_lin.in_features:
            new_lin = nn.Linear(target_in_dim, old_lin.out_features)
            with torch.no_grad():
                nn.init.zeros_(new_lin.weight)
                copy_cols = min(target_in_dim, old_lin.in_features)
                new_lin.weight[:, :copy_cols].copy_(old_lin.weight[:, :copy_cols])
                new_lin.bias.copy_(old_lin.bias)
            src_head[0] = new_lin

        # Split into trunk (.. -> 64) and final linear head (64 -> 1)
        self.head_trunk = nn.Sequential(*list(src_head.children())[:-1])
        self.main_head = list(src_head.children())[-1]
        penult_dim = int(self.main_head.in_features)
        self.aux_60d_head: Optional[nn.Linear] = (
            nn.Linear(penult_dim, int(aux_60d_dim)) if self.aux_60d_enabled else None
        )
        self.head_input_dim = target_in_dim

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        cached_customer_vectors: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        **kwargs,
    ) -> Any:
        core_13 = pairwise_features[..., :13]
        extra = pairwise_features[..., 13:]
        encoded = self.base.encode_candidate_features(
            pairwise_features=core_13,
            memory_value_statistics=False,
            **kwargs,
        )
        fused = torch.cat([encoded.features, extra], dim=-1)
        hidden = self.head_trunk(fused)
        logits = self.main_head(hidden).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)
        out = encoded.to_output(logits)
        if return_aux and self.aux_60d_head is not None:
            aux_logits = self.aux_60d_head(hidden).squeeze(-1)
            if candidate_mask is not None:
                aux_logits = aux_logits.masked_fill(~candidate_mask, -torch.inf)
            return out, aux_logits
        return out


def build_slate_from_stage1_scores(
    base_slate: Mapping[str, Any],
    split_data: Mapping[str, Any],
    stage1_scores: torch.Tensor,
    product_ids: Sequence[str],
    arm: str,
) -> Dict[str, Any]:
    """Rebuild a 100-item quota100_50 slate from new Stage 1 scores with arm-specific rolling features."""
    indices = select_candidates(stage1_scores, split_data["allowed"], split_data["unseen"], SLATE_POLICY)
    eligible = split_data["allowed"].gather(1, indices)
    if not bool(eligible.all()):
        raise ValueError("Ineligible candidate selected during slate rebuild.")
    row_arange = np.arange(len(indices))[:, None]
    idx_np = indices.numpy()
    f41 = np.array(split_data["features"][row_arange, idx_np], dtype=np.float32)
    f19 = (
        np.array(split_data["base_name_features"][row_arange, idx_np], dtype=np.float32)
        if "base_name_features" in split_data
        else None
    )
    feats = torch.from_numpy(assemble_arm_features_numpy(f41, f19, arm)).contiguous()

    slate = dict(base_slate)
    slate["candidate_indices"] = indices.to(dtype=torch.long).contiguous()
    slate["candidate_mask"] = eligible.to(dtype=torch.bool).contiguous()
    slate["candidate_pids"] = [[product_ids[i] for i in row] for row in indices.tolist()]
    slate["pairwise_features"] = feats
    slate["labels"] = split_data["target"].gather(1, indices).to(dtype=torch.float32).contiguous()
    return slate


def attach_arm_features_to_fixed_slate(
    base_slate: Mapping[str, Any],
    split_data: Mapping[str, Any],
    arm: str,
) -> Dict[str, Any]:
    """Keep candidate_indices/labels identical to base_slate while swapping pairwise_features to arm."""
    indices = base_slate["candidate_indices"]
    row_arange = np.arange(len(indices))[:, None]
    idx_np = indices.numpy()
    f41 = base_slate["pairwise_features"].numpy()
    f19 = (
        np.array(split_data["base_name_features"][row_arange, idx_np], dtype=np.float32)
        if "base_name_features" in split_data
        else None
    )
    feats = torch.from_numpy(assemble_arm_features_numpy(f41, f19, arm)).contiguous()
    slate = dict(base_slate)
    slate["pairwise_features"] = feats
    return slate


@torch.no_grad()
def predict_roadmap_stage2(
    model: RoadmapPointwiseRanker,
    slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    dev: torch.device,
    *,
    batch_size: int = 128,
) -> Dict[str, Any]:
    model.eval()
    loader = make_rolling_slate_loader(slate, batch_size=batch_size, shuffle=False)
    loss_fn = RankerLoss().to(dev)
    all_scores = []
    losses = []
    t0_weights = []
    hist_weights = []

    for batch in loader:
        inputs = rolling_ranker_batch_inputs(batch, product_vectors, dev)
        out = model(**inputs)
        loss = loss_fn(out.logits, batch["labels"].to(dev), candidate_mask=inputs.get("candidate_mask"))["loss"]
        losses.append(float(loss.detach().cpu()))
        all_scores.append(out.logits.detach().cpu())
        if out.token_0_weight is not None:
            t0_weights.append(float(out.token_0_weight.detach().mean().cpu()))
        if out.history_weights is not None:
            hist_weights.append(float(out.history_weights.detach().mean().cpu()))

    return {
        "scores": torch.cat(all_scores, dim=0).contiguous(),
        "val_loss": float(np.mean(losses)),
        "avg_token0_weight": float(np.mean(t0_weights)) if t0_weights else 0.0,
        "avg_history_weight": float(np.mean(hist_weights)) if hist_weights else 0.0,
    }


def evaluate_stage2_with_breakdown(
    scores: torch.Tensor,
    val_data: Mapping[str, Any],
    candidate_indices: torch.Tensor,
    pack_switch_mask: torch.Tensor,
    true_new_bn_mask: torch.Tensor,
    *,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = evaluate_rolling_slate_metrics(scores, val_data, candidate_indices, save_path=save_path)
    ps_in_slate = pack_switch_mask.gather(1, candidate_indices)
    tn_in_slate = true_new_bn_mask.gather(1, candidate_indices)
    metrics["pack_switch_targets"] = int(pack_switch_mask.sum().item())
    metrics["true_new_bn_targets"] = int(true_new_bn_mask.sum().item())
    metrics["pack_switch_in_slate"] = int(ps_in_slate.sum().item())
    metrics["true_new_bn_in_slate"] = int(tn_in_slate.sum().item())

    top10_pos = scores.argsort(dim=-1, descending=True, stable=True)[:, :10]
    metrics["pack_switch_hits@10"] = int(ps_in_slate.gather(1, top10_pos).sum().item())
    metrics["true_new_bn_hits@10"] = int(tn_in_slate.gather(1, top10_pos).sum().item())
    return metrics


def fit_stage2_roadmap_run(
    *,
    arm_name: str,
    train_slate: Mapping[str, Any],
    val_slate: Mapping[str, Any],
    val_data: Mapping[str, Any],
    product_vectors: torch.Tensor,
    product_ids: Sequence[str],
    meta: Mapping[str, Any],
    artifact_root: Path = ARTIFACT_ROOT / "stage2_base_name",
    seed: int = SEED,
    epochs: int = STAGE2_EPOCHS,
    batch_size: int = STAGE2_BATCH_SIZE,
    lr: float = STAGE2_LR,
    weight_decay: float = STAGE2_WEIGHT_DECAY,
    eta_min: float = STAGE2_ETA_MIN,
    grad_clip_norm: float = STAGE2_GRAD_CLIP_NORM,
    device: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    slate_tag: str = "fixed_rolling100",
) -> Dict[str, Any]:
    """Train one Stage 2 UPS-CANR pointwise arm and verify exact checkpoint replay."""
    dev = get_device(device)
    ref_disc = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu", weights_only=False)
    recall_guard = float(ref_disc["discovery_min_overall_recall"])
    ndcg_guard = float(ref_disc["val_metrics"]["val_ndcg@10"])

    num_features = int(train_slate["pairwise_features"].shape[-1])
    run_name = f"stage2-{arm_name}-{slate_tag}-seed{seed}"
    run_dir = Path(artifact_root) / "runs" / run_name

    settings = {
        "stage": "roadmap_v2_stage2",
        "arm_name": arm_name,
        "slate_tag": slate_tag,
        "num_features": num_features,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "statistics_policy": "rolling_observed_v2",
        "rolling_features_identity": meta["rolling_features_identity"],
        "base_name_identity": meta.get("base_name_identity"),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "eta_min": float(eta_min),
        "grad_clip_norm": float(grad_clip_norm),
        "recall_guardrail": recall_guard,
        "ndcg_guardrail": ndcg_guard,
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

    model = RoadmapPointwiseRanker(num_features=num_features, seed=seed).to(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    vectors_dev = product_vectors.to(dev)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=float(eta_min))
    loss_fn = RankerLoss().to(dev)
    train_loader = make_rolling_slate_loader(
        train_slate, batch_size=int(batch_size), shuffle=True, seed=int(seed)
    )

    best_disc_key = (-1, -1, -1.0)
    best_ndcg_key = (-1.0, -1.0, -1)
    history = []

    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = rolling_ranker_batch_inputs(batch, vectors_dev, dev)
            logits = model(**inputs).logits
            loss = loss_fn(logits, batch["labels"].to(dev), candidate_mask=inputs.get("candidate_mask"))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_pred = predict_roadmap_stage2(model, val_slate, vectors_dev, dev)
        metrics = evaluate_stage2_with_breakdown(
            val_pred["scores"], val_data, val_slate["candidate_indices"], pack_switch_mask, true_new_bn_mask
        )
        metrics["val_loss"] = val_pred["val_loss"]
        metrics["avg_token0_weight"] = val_pred["avg_token0_weight"]
        metrics["avg_history_weight"] = val_pred["avg_history_weight"]

        qualified_epoch = bool(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard)
        disc_key = (int(qualified_epoch), int(metrics["new_hits@10"]), float(metrics["ndcg@10"]))
        ndcg_key = (float(metrics["ndcg@10"]), float(metrics["recall@10"]), int(metrics["new_hits@10"]))
        ep_sec = time.perf_counter() - ep_t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "qualified": qualified_epoch,
                "epoch_seconds": ep_sec,
                "metrics": metrics,
            }
        )

        if disc_key > best_disc_key:
            best_disc_key = disc_key
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "num_features": num_features,
                    "model_state_dict": model.state_dict(),
                },
                run_dir / "best_discovery.pt",
            )
        if ndcg_key > best_ndcg_key:
            best_ndcg_key = ndcg_key
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "num_features": num_features,
                    "model_state_dict": model.state_dict(),
                },
                run_dir / "best_ndcg.pt",
            )
        print(
            f"[stage2/{arm_name}/{slate_tag}] seed={seed} ep={epoch:02d} loss={np.mean(losses):.4f} "
            f"R@5={metrics['recall@5']:.5f} R@10={metrics['recall@10']:.5f} NDCG@10={metrics['ndcg@10']:.5f} "
            f"New@10={metrics['new_hits@10']} (ps={metrics['pack_switch_hits@10']}/{metrics['pack_switch_in_slate']}, "
            f"tn={metrics['true_new_bn_hits@10']}/{metrics['true_new_bn_in_slate']}) ({ep_sec:.1f}s)",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - started

    selections: Dict[str, Any] = {}
    for sel_name, fname in (("discovery", "best_discovery.pt"), ("ndcg", "best_ndcg.pt")):
        ckpt_path = run_dir / fname
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reloaded = RoadmapPointwiseRanker(num_features=num_features, seed=seed).to(dev)
        reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
        pred = predict_roadmap_stage2(reloaded, val_slate, vectors_dev, dev)
        replayed = evaluate_stage2_with_breakdown(
            pred["scores"],
            val_data,
            val_slate["candidate_indices"],
            pack_switch_mask,
            true_new_bn_mask,
            save_path=run_dir / f"predictions_{sel_name}.npz",
        )
        replayed["val_loss"] = pred["val_loss"]
        replayed["avg_token0_weight"] = pred["avg_token0_weight"]
        replayed["avg_history_weight"] = pred["avg_history_weight"]
        if replayed != ckpt["metrics"]:
            raise AssertionError(f"Stage 2 {sel_name} replay mismatch for {run_name}.")
        selections[sel_name] = {
            "epoch": int(ckpt["epoch"]),
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_digest(ckpt_path),
            "predictions_npz": str(run_dir / f"predictions_{sel_name}.npz"),
            "metrics": replayed,
        }

    result = {
        "arm_name": arm_name,
        "slate_tag": slate_tag,
        "run_dir": str(run_dir),
        "num_features": num_features,
        "head_input_dim": int(model.head_input_dim),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "training_seconds": float(elapsed),
        "device": str(dev),
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
