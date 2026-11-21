"""Stage 1 MLP selector for roadmap_v2 (strictly rolling_observed_v2, extensible for aux head & dropout)."""

import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.experiment_signals import sync
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.mlp_feature_ablation.evaluate import evaluate_full_pool_metrics
from recommender.models import OUTPUT_DIM
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .base_name_features import assemble_arm_features_numpy
from .config import (
    ARTIFACT_ROOT,
    SEED,
    STAGE1_BATCH_SIZE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE1_OMITTED_CALENDAR_FEATURES,
    STAGE1_POS_WEIGHT,
    STAGE1_WEIGHT_DECAY,
    feature_names_for_arm,
)


def compute_pack_switch_masks(
    val_data: Mapping[str, Any],
    product_ids: Sequence[str],
    all_pid_to_base_name: Mapping[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split val `new_target` (1,857) into `pack_switch_target` (455) and `true_new_bn_target` (1,402).

    Uses strictly-causal `base_name_features[..., 0]` (`has_bought_base_sku` < as_of_date).
    """
    new_target = val_data["new_target"].bool()
    has_bought_bn = torch.from_numpy(
        np.array(val_data["base_name_features"][:, :, 0], dtype=np.float32) > 0.5
    )
    pack_switch = new_target & has_bought_bn
    true_new_bn = new_target & (~has_bought_bn)
    return pack_switch, true_new_bn


def evaluate_stage1_with_breakdown(
    scores: torch.Tensor,
    val_data: Mapping[str, Any],
    pack_switch_mask: torch.Tensor,
    true_new_bn_mask: torch.Tensor,
    *,
    save_predictions: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute canonical full-pool metrics plus pack-switch vs truly-new base_name discovery hits."""
    metrics = evaluate_full_pool_metrics(scores, val_data, save_predictions=save_predictions)
    allowed = val_data["allowed"].bool()
    unseen = val_data["unseen"].bool()
    # Ordinary top-10 (all eligible) and unseen top-K (strictly unseen SKUs)
    all_order = scores.masked_fill(~allowed, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    unseen_order = scores.masked_fill(~unseen, -torch.inf).argsort(dim=-1, descending=True, stable=True)

    ps_total = int(pack_switch_mask.sum().item())
    tn_total = int(true_new_bn_mask.sum().item())
    metrics["pack_switch_targets"] = ps_total
    metrics["true_new_bn_targets"] = tn_total

    top10_all = all_order[:, :10]
    metrics["pack_switch_hits@10"] = int(pack_switch_mask.gather(1, top10_all).sum().item())
    metrics["true_new_bn_hits@10"] = int(true_new_bn_mask.gather(1, top10_all).sum().item())

    for k in (25, 50, 100):
        topk_unseen = unseen_order[:, :k]
        metrics[f"pack_switch_hits@{k}"] = int(pack_switch_mask.gather(1, topk_unseen).sum().item())
        metrics[f"true_new_bn_hits@{k}"] = int(true_new_bn_mask.gather(1, topk_unseen).sum().item())
    return metrics


class RoadmapStage1Selector(nn.Module):
    """Pointwise Stage 1 MLP selector over [u, v, retained_features, customer_stats_6].

    Architecture matches `FeatureAblatedStage1Selector` (128 -> 64 -> 1) and supports an optional
    training-only 60-day auxiliary head (`aux_60d_head: Linear(64, 1)`) discarded at inference.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        omitted_names: Sequence[str] = STAGE1_OMITTED_CALENDAR_FEATURES,
        embedding_dim: int = OUTPUT_DIM,
        num_customer_stats: int = 6,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        aux_60d_enabled: bool = False,
    ):
        super().__init__()
        self.source_names = tuple(str(n) for n in feature_names)
        omit_set = set(str(n) for n in omitted_names)
        self.omitted_names = tuple(n for n in self.source_names if n in omit_set)
        self.retained_indices = tuple(i for i, n in enumerate(self.source_names) if n not in omit_set)
        self.retained_names = tuple(self.source_names[i] for i in self.retained_indices)
        self.embedding_dim = int(embedding_dim)
        self.num_customer_stats = int(num_customer_stats)
        self.num_retained_features = len(self.retained_indices)
        self.input_dim = self.embedding_dim * 2 + self.num_retained_features + self.num_customer_stats
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.aux_60d_enabled = bool(aux_60d_enabled)

        layers: list[nn.Module] = []
        prev = self.input_dim
        for w in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, w),
                    nn.LayerNorm(w),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev = w
        self.trunk = nn.Sequential(*layers)
        self.main_head = nn.Linear(prev, 1)
        self.aux_60d_head: Optional[nn.Linear] = nn.Linear(prev, 1) if self.aux_60d_enabled else None

        self.register_buffer(
            "retained_indices_tensor",
            torch.tensor(self.retained_indices, dtype=torch.long),
            persistent=False,
        )

    def encode_hidden(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        features: torch.Tensor,
        customer_stats_6: torch.Tensor,
    ) -> torch.Tensor:
        idx = self.retained_indices_tensor
        if idx.device != features.device:
            idx = idx.to(features.device)
        retained = torch.index_select(features, -1, idx)
        bsz, n_prod, _ = retained.shape
        u_exp = u.unsqueeze(1).expand(bsz, n_prod, self.embedding_dim)
        stats_exp = customer_stats_6.unsqueeze(1).expand(bsz, n_prod, self.num_customer_stats)
        v_exp = v.unsqueeze(0).expand(bsz, n_prod, self.embedding_dim) if v.ndim == 2 else v
        x = torch.cat([u_exp, v_exp, retained, stats_exp], dim=-1)
        return self.trunk(x)

    def forward(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        features: torch.Tensor,
        customer_stats_6: torch.Tensor,
        return_aux: bool = False,
    ) -> Any:
        h = self.encode_hidden(u=u, v=v, features=features, customer_stats_6=customer_stats_6)
        main_logits = self.main_head(h).squeeze(-1)
        if return_aux and self.aux_60d_head is not None:
            aux_logits = self.aux_60d_head(h).squeeze(-1)
            return main_logits, aux_logits
        return main_logits


@torch.no_grad()
def score_roadmap_stage1_split(
    model: RoadmapStage1Selector,
    split_data: Mapping[str, Any],
    arm: str,
    dev: torch.device,
    *,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    v_dev = split_data["v"].to(dev)
    u_all = split_data["u"]
    stats_all = split_data["customer_stats_6"]
    allowed_all = split_data["allowed"]
    f41_mmap = split_data["features"]
    f19_mmap = split_data.get("base_name_features")

    chunks = []
    for start in range(0, len(u_all), batch_size):
        end = min(start + batch_size, len(u_all))
        f41_np = np.array(f41_mmap[start:end], dtype=np.float32)
        f19_np = np.array(f19_mmap[start:end], dtype=np.float32) if f19_mmap is not None else None
        feats = torch.from_numpy(assemble_arm_features_numpy(f41_np, f19_np, arm)).to(dev)
        logits = model(
            u=u_all[start:end].to(dev),
            v=v_dev,
            features=feats,
            customer_stats_6=stats_all[start:end].to(dev),
        )
        masked = logits.masked_fill(~allowed_all[start:end].to(dev), -1e9)
        chunks.append(masked.cpu())
    return torch.cat(chunks, dim=0).contiguous()


def fit_stage1_base_name_arm(
    arm: str,
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    artifact_root: Path = ARTIFACT_ROOT / "stage1_base_name",
    seed: int = SEED,
    epochs: int = STAGE1_EPOCHS,
    batch_size: int = STAGE1_BATCH_SIZE,
    lr: float = STAGE1_LR,
    weight_decay: float = STAGE1_WEIGHT_DECAY,
    pos_weight: float = STAGE1_POS_WEIGHT,
    device: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Train one Stage 1 MLP arm natively under rolling_observed_v2 and verify strict replay."""
    dev = get_device(device)
    feature_names = feature_names_for_arm(arm)
    run_name = run_tag or f"stage1-{arm}-seed{seed}"
    run_dir = Path(artifact_root) / "runs" / run_name

    settings = {
        "stage": "roadmap_v2_stage1_base_name",
        "arm": arm,
        "feature_names": list(feature_names),
        "omitted_names": list(STAGE1_OMITTED_CALENDAR_FEATURES),
        "statistics_policy": "rolling_observed_v2",
        "rolling_features_identity": meta["rolling_features_identity"],
        "base_name_identity": meta["base_name_identity"],
        "data_identity": meta["data_identity"],
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "pos_weight": float(pos_weight),
        "selection": "new_hits@50, ndcg@10, recall@10 (on rolling_observed_v2 val)",
        "heldout_access": False,
    }
    settings_digest = json_digest(settings)

    if (run_dir / "result.json").is_file() and (run_dir / "run.json").is_file():
        run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if run_meta.get("status") == "complete" and run_meta.get("settings_digest") == settings_digest:
            return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "running", "settings_digest": settings_digest}, run_dir / "run.json", overwrite=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = RoadmapStage1Selector(feature_names).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    tr = campaign_data["train"]
    va = campaign_data["val"]
    v_dev = tr["v"].to(dev)
    tr_f41 = tr["features"]
    tr_f19 = tr.get("base_name_features")
    pack_switch_mask, true_new_bn_mask = compute_pack_switch_masks(
        va, info["product_ids"], meta["all_pid_to_base_name"]
    )

    generator = torch.Generator().manual_seed(int(seed))
    pos_weight_tensor = torch.tensor(float(pos_weight), device=dev)

    best_key = (-1, -1.0, -1.0)
    best_epoch = 0
    history = []

    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        perm = torch.randperm(len(tr["u"]), generator=generator)
        for batch_rows in perm.split(int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            rows_np = batch_rows.numpy()
            f41_np = np.array(tr_f41[rows_np], dtype=np.float32)
            f19_np = np.array(tr_f19[rows_np], dtype=np.float32) if tr_f19 is not None else None
            feats = torch.from_numpy(assemble_arm_features_numpy(f41_np, f19_np, arm)).to(dev)
            logits = model(
                u=tr["u"][batch_rows].to(dev),
                v=v_dev,
                features=feats,
                customer_stats_6=tr["customer_stats_6"][batch_rows].to(dev),
            )
            allowed = tr["allowed"][batch_rows].to(dev)
            target = tr["target"][batch_rows].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(
                logits[allowed], target[allowed], pos_weight=pos_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_scores = score_roadmap_stage1_split(model, va, arm, dev)
        metrics = evaluate_stage1_with_breakdown(val_scores, va, pack_switch_mask, true_new_bn_mask)
        key = (int(metrics["new_hits@50"]), float(metrics["ndcg@10"]), float(metrics["recall@10"]))
        ep_sec = time.perf_counter() - ep_t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": ep_sec,
                "metrics": metrics,
            }
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "feature_names": list(feature_names),
                    "omitted_names": list(STAGE1_OMITTED_CALENDAR_FEATURES),
                    "input_dim": model.input_dim,
                    "model_state_dict": model.state_dict(),
                },
                run_dir / "best.pt",
            )
        print(
            f"[stage1/{arm}] seed={seed} ep={epoch:02d} loss={np.mean(losses):.5f} "
            f"R@5={metrics['recall@5']:.5f} R@10={metrics['recall@10']:.5f} NDCG@10={metrics['ndcg@10']:.5f} "
            f"New@10={metrics['new_hits@10']} (ps={metrics['pack_switch_hits@10']}/455, tn={metrics['true_new_bn_hits@10']}/1402) "
            f"New@25={metrics['new_hits@25']} New@50={metrics['new_hits@50']} New@100={metrics['new_hits@100']} ({ep_sec:.1f}s)",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - started

    # Strict checkpoint reload & replay verification
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    reloaded = RoadmapStage1Selector(feature_names).to(dev)
    reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    val_scores = score_roadmap_stage1_split(reloaded, va, arm, dev)
    replayed = evaluate_stage1_with_breakdown(
        val_scores,
        va,
        pack_switch_mask,
        true_new_bn_mask,
        save_predictions=run_dir / "predictions.npz",
    )
    if replayed != ckpt["metrics"]:
        raise AssertionError(f"Stage 1 replay mismatch for {run_name}.")

    train_scores = score_roadmap_stage1_split(reloaded, tr, arm, dev)
    best_sha = file_digest(run_dir / "best.pt")
    atomic_torch_save({"arm": arm, "selector_sha256": best_sha, "scores": train_scores}, run_dir / "train_scores.pt")
    atomic_torch_save({"arm": arm, "selector_sha256": best_sha, "scores": val_scores}, run_dir / "val_scores.pt")

    result = {
        "arm": arm,
        "run_dir": str(run_dir),
        "input_dim": int(reloaded.input_dim),
        "num_retained_features": int(reloaded.num_retained_features),
        "parameters": int(sum(p.numel() for p in reloaded.parameters())),
        "selected_epoch": int(best_epoch),
        "training_seconds": float(elapsed),
        "device": str(dev),
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
    atomic_json(
        {
            "status": "complete",
            "settings_digest": settings_digest,
            "checkpoint_sha256": best_sha,
            "selected_epoch": int(best_epoch),
        },
        run_dir / "run.json",
        overwrite=True,
    )
    return result
