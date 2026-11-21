"""Stage 3 CLI runner: One-group-at-a-time causal ablation for 3A financial, 3B pricing, 3C manager, 3D warehouse, 3E parsed specs."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.development import load_development_purchases
from recommender.experiment_signals import sync
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_DISCOVERY_RANKER
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
from .base_name_features import assemble_arm_features_numpy, prepare_or_load_base_name_feature_bundle
from .commercial_features import (
    COMMERCIAL_FEATURE_NAMES,
    COMMERCIAL_GROUPS,
    prepare_or_load_commercial_feature_bundle,
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
    VALID_BASE_NAME_ARMS,
    feature_names_for_arm,
)
from .parsed_spec_features import PARSED_SPEC_FEATURE_NAMES, prepare_or_load_parsed_spec_feature_bundle
from .stage1_mlp import (
    RoadmapStage1Selector,
    compute_pack_switch_masks,
    evaluate_stage1_with_breakdown,
)
from .stage2_aug_runner import _prod_bn_id_tensor
from .stage2_ranker import (
    RoadmapPointwiseRanker,
    attach_arm_features_to_fixed_slate,
    build_slate_from_stage1_scores,
    evaluate_stage2_with_breakdown,
    predict_roadmap_stage2,
)

ALL_STAGE3_GROUPS: Tuple[str, ...] = (
    "3a_financial",
    "3b_pricing",
    "3c_manager",
    "3d_warehouse",
    "3e_parsed_specs",
)


def extra_feature_names_for_groups(groups: Sequence[str]) -> Tuple[str, ...]:
    names: List[str] = []
    for g in groups:
        if g in COMMERCIAL_GROUPS:
            names.extend(COMMERCIAL_FEATURE_NAMES[i] for i in COMMERCIAL_GROUPS[g])
        elif g == "3e_parsed_specs":
            names.extend(PARSED_SPEC_FEATURE_NAMES)
        else:
            raise ValueError(f"Unknown Stage 3 group: {g!r}")
    return tuple(names)


def slice_extra_group_features(
    comm_mmap: np.ndarray,
    spec_mmap: np.ndarray,
    rows: Any,
    groups: Sequence[str],
    cand_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if not groups:
        return None
    parts = []
    for g in groups:
        if g in COMMERCIAL_GROUPS:
            cols = list(COMMERCIAL_GROUPS[g])
            arr = (
                np.array(comm_mmap[rows, cand_indices], dtype=np.float32)[..., cols]
                if cand_indices is not None
                else np.array(comm_mmap[rows], dtype=np.float32)[..., cols]
            )
            parts.append(arr)
        elif g == "3e_parsed_specs":
            arr = (
                np.array(spec_mmap[rows, cand_indices], dtype=np.float32)
                if cand_indices is not None
                else np.array(spec_mmap[rows], dtype=np.float32)
            )
            parts.append(arr)
        else:
            raise ValueError(f"Unknown Stage 3 group: {g!r}")
    return np.concatenate(parts, axis=-1)


@torch.no_grad()
def _score_stage1_with_groups(
    model: RoadmapStage1Selector,
    split_data: Mapping[str, Any],
    comm_mmap: np.ndarray,
    spec_mmap: np.ndarray,
    arm: str,
    groups: Sequence[str],
    dev: torch.device,
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
        sl = slice(start, end)
        base = assemble_arm_features_numpy(
            np.array(f41_mmap[sl], dtype=np.float32),
            np.array(f19_mmap[sl], dtype=np.float32) if f19_mmap is not None else None,
            arm,
        )
        extra = slice_extra_group_features(comm_mmap, spec_mmap, sl, groups)
        full = np.concatenate([base, extra], axis=-1) if extra is not None else base
        logits = model(
            u=u_all[sl].to(dev),
            v=v_dev,
            features=torch.from_numpy(full).to(dev),
            customer_stats_6=stats_all[sl].to(dev),
        )
        chunks.append(logits.masked_fill(~allowed_all[sl].to(dev), -1e9).cpu())
    return torch.cat(chunks, dim=0).contiguous()


def fit_stage1_with_groups(
    *,
    arm: str,
    groups: Sequence[str],
    tag: str,
    campaign_data: Mapping[str, Mapping[str, Any]],
    comm_mmaps: Mapping[str, np.ndarray],
    spec_mmaps: Mapping[str, np.ndarray],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    aux_60d_labels: Optional[torch.Tensor] = None,
    aux_lambda: float = 0.0,
    p_drop: float = 0.0,
    artifact_root: Path = ARTIFACT_ROOT / "stage3_commercial",
    seed: int = SEED,
    epochs: int = STAGE1_EPOCHS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    dev = get_device(device)
    feature_names = feature_names_for_arm(arm) + extra_feature_names_for_groups(groups)
    use_aux = aux_lambda > 0.0 and aux_60d_labels is not None
    run_name = f"stage1-{arm}-{tag}-seed{seed}"
    run_dir = Path(artifact_root) / "runs" / run_name
    settings = {
        "stage": "roadmap_v2_stage3_s1",
        "arm": arm,
        "groups": list(groups),
        "tag": tag,
        "aux_lambda": float(aux_lambda),
        "p_drop": float(p_drop),
        "feature_names": list(feature_names),
        "seed": int(seed),
        "epochs": int(epochs),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    tr, va = campaign_data["train"], campaign_data["val"]
    v_dev = tr["v"].to(dev)
    prod_bn_ids = _prod_bn_id_tensor(meta)
    ps_mask, tn_mask = compute_pack_switch_masks(va, info["product_ids"], meta["all_pid_to_base_name"])
    perm_gen = torch.Generator().manual_seed(int(seed))
    drop_gen = torch.Generator().manual_seed(int(seed) + 10007)
    pw_t = torch.tensor(float(STAGE1_POS_WEIGHT), device=dev)
    aux_pw_t = torch.tensor(10.0, device=dev)

    best_key = (-1, -1.0, -1.0)
    best_epoch = 0
    history = []
    sync(dev)
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        perm = torch.randperm(len(tr["u"]), generator=perm_gen)
        for batch_rows in perm.split(STAGE1_BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            rows_np = batch_rows.numpy()
            base = assemble_arm_features_numpy(
                np.array(tr["features"][rows_np], dtype=np.float32),
                np.array(tr["base_name_features"][rows_np], dtype=np.float32),
                arm,
            )
            extra = slice_extra_group_features(comm_mmaps["train"], spec_mmaps["train"], rows_np, groups)
            full = np.concatenate([base, extra], axis=-1) if extra is not None else base
            feats = torch.from_numpy(full).to(dev)
            allowed = tr["allowed"][batch_rows].to(dev)
            target = tr["target"][batch_rows].to(dev).float()
            if p_drop > 0.0:
                feats, _ = apply_stage1_history_sku_dropout(
                    feats, target, prod_bn_ids, feature_names, p_drop=p_drop, generator=drop_gen
                )
            if use_aux:
                m_log, a_log = model(
                    u=tr["u"][batch_rows].to(dev),
                    v=v_dev,
                    features=feats,
                    customer_stats_6=tr["customer_stats_6"][batch_rows].to(dev),
                    return_aux=True,
                )
                l_main = F.binary_cross_entropy_with_logits(m_log[allowed], target[allowed], pos_weight=pw_t)
                l_aux = F.binary_cross_entropy_with_logits(
                    a_log[allowed], aux_60d_labels[batch_rows].to(dev).float()[allowed], pos_weight=aux_pw_t
                )
                loss = l_main + float(aux_lambda) * l_aux
            else:
                m_log = model(
                    u=tr["u"][batch_rows].to(dev),
                    v=v_dev,
                    features=feats,
                    customer_stats_6=tr["customer_stats_6"][batch_rows].to(dev),
                )
                loss = F.binary_cross_entropy_with_logits(m_log[allowed], target[allowed], pos_weight=pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_scores = _score_stage1_with_groups(
            model, va, comm_mmaps["val"], spec_mmaps["val"], arm, groups, dev
        )
        metrics = evaluate_stage1_with_breakdown(val_scores, va, ps_mask, tn_mask)
        key = (int(metrics["new_hits@50"]), float(metrics["ndcg@10"]), float(metrics["recall@10"]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})
        if key > best_key:
            best_key = key
            best_epoch = epoch
            inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}
            atomic_torch_save(
                {"settings": settings, "epoch": epoch, "metrics": metrics, "model_state_dict": inf_state},
                run_dir / "best.pt",
            )
        print(
            f"[stage3/S1/{tag}] ep={epoch:02d} R@5={metrics['recall@5']:.5f} R@10={metrics['recall@10']:.5f} "
            f"NDCG@10={metrics['ndcg@10']:.5f} New@10={metrics['new_hits@10']} "
            f"(ps={metrics['pack_switch_hits@10']}, tn={metrics['true_new_bn_hits@10']}) "
            f"New@25={metrics['new_hits@25']} New@50={metrics['new_hits@50']} New@100={metrics['new_hits@100']} "
            f"({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    sync(dev)
    elapsed = time.perf_counter() - t0
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    reloaded = RoadmapStage1Selector(feature_names, aux_60d_enabled=False).to(dev)
    reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    val_scores = _score_stage1_with_groups(reloaded, va, comm_mmaps["val"], spec_mmaps["val"], arm, groups, dev)
    replayed = evaluate_stage1_with_breakdown(
        val_scores, va, ps_mask, tn_mask, save_predictions=run_dir / "predictions.npz"
    )
    if replayed != ckpt["metrics"]:
        raise AssertionError(f"Stage 3 S1 replay mismatch for {run_name}")
    tr_scores = _score_stage1_with_groups(reloaded, tr, comm_mmaps["train"], spec_mmaps["train"], arm, groups, dev)
    best_sha = file_digest(run_dir / "best.pt")
    atomic_torch_save({"scores": tr_scores, "sha": best_sha}, run_dir / "train_scores.pt")
    atomic_torch_save({"scores": val_scores, "sha": best_sha}, run_dir / "val_scores.pt")
    result = {
        "tag": tag,
        "groups": list(groups),
        "input_dim": int(reloaded.input_dim),
        "selected_epoch": int(best_epoch),
        "training_seconds": float(elapsed),
        "checkpoint": str(run_dir / "best.pt"),
        "predictions_npz": str(run_dir / "predictions.npz"),
        "train_scores_pt": str(run_dir / "train_scores.pt"),
        "val_scores_pt": str(run_dir / "val_scores.pt"),
        "metrics": replayed,
        "settings": settings,
    }
    atomic_json(result, run_dir / "result.json", overwrite=True)
    atomic_json({"status": "complete", "settings_digest": settings_digest, "checkpoint_sha256": best_sha}, run_dir / "run.json", overwrite=True)
    return result


def attach_stage3_groups_to_slate(
    base_slate: Mapping[str, Any],
    split_data: Mapping[str, Any],
    comm_mmap: np.ndarray,
    spec_mmap: np.ndarray,
    arm: str,
    groups: Sequence[str],
) -> Dict[str, Any]:
    slate = attach_arm_features_to_fixed_slate(base_slate, split_data, arm)
    if not groups:
        return slate
    idx_np = slate["candidate_indices"].numpy()
    row_arange = np.arange(len(idx_np))[:, None]
    extra = slice_extra_group_features(comm_mmap, spec_mmap, row_arange, groups, cand_indices=idx_np)
    slate["pairwise_features"] = torch.cat(
        [slate["pairwise_features"], torch.from_numpy(extra)], dim=-1
    ).contiguous()
    return slate


def fit_stage2_with_groups(
    *,
    arm: str,
    groups: Sequence[str],
    tag: str,
    train_slate: Mapping[str, Any],
    val_slate: Mapping[str, Any],
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    product_vectors: torch.Tensor,
    product_ids: Sequence[str],
    meta: Mapping[str, Any],
    aux_60d_labels: Optional[torch.Tensor] = None,
    aux_lambda: float = 0.0,
    p_drop: float = 0.0,
    artifact_root: Path = ARTIFACT_ROOT / "stage3_commercial",
    seed: int = SEED,
    epochs: int = STAGE2_EPOCHS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    from .stage2_aug_runner import fit_stage2_augmented

    # Temporarily patch feature names by passing a custom feature_names list?
    # Note: fit_stage2_augmented only needs exact-SKU/base_name personal feature indices (which sit at the front of arm features!).
    # So calling fit_stage2_augmented directly works identically because extra Stage 3 columns are appended at the end!
    return fit_stage2_augmented(
        arm=arm,
        variant_name=tag,
        train_slate=train_slate,
        val_slate=val_slate,
        train_data=train_data,
        val_data=val_data,
        product_vectors=product_vectors,
        product_ids=product_ids,
        meta=meta,
        aux_60d_labels=aux_60d_labels,
        aux_lambda=aux_lambda,
        p_drop=p_drop,
        slate_tag="stage3",
        artifact_root=artifact_root,
        seed=seed,
        epochs=epochs,
        device=device,
    )


def run_stage3(
    *,
    arm: str,
    s1_aux_lambda: float = 0.0,
    s1_p_drop: float = 0.0,
    s2_aux_lambda: float = 0.0,
    s2_p_drop: float = 0.0,
    slate_source_dir: Optional[Path] = None,
    groups_to_test: Sequence[str] = ALL_STAGE3_GROUPS,
    combined_groups: Optional[Sequence[str]] = None,
    seed: int = SEED,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    before_hashes = snapshot_preserved_hashes()
    campaign_data, info, _, meta = prepare_or_load_base_name_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(campaign_data, info, meta)
    base_slates, _ = prepare_or_load_rolling_slate_cache(campaign_data, info, meta)
    comm_mmaps, comm_manifest = prepare_or_load_commercial_feature_bundle(build=False)
    spec_mmaps, spec_manifest = prepare_or_load_parsed_spec_feature_bundle(build=False)

    product_ids = list(info["product_ids"])
    product_vectors = campaign_data["train"]["v"]
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    tr_src = campaign_data["train"]["source"]
    aux_60d = (
        build_strict_60d_aux_labels(purchases, tr_src["customer_ids"], tr_src["prediction_days"].tolist(), product_ids)
        if (s1_aux_lambda > 0 or s2_aux_lambda > 0)
        else None
    )

    # Determine base slates for Stage 2 one-group-at-a-time ablations
    if slate_source_dir is not None and (Path(slate_source_dir) / "train_scores.pt").is_file():
        s1_tr = torch.load(Path(slate_source_dir) / "train_scores.pt", map_location="cpu", weights_only=False)["scores"]
        s1_va = torch.load(Path(slate_source_dir) / "val_scores.pt", map_location="cpu", weights_only=False)["scores"]
        tr_base_slate = build_slate_from_stage1_scores(base_slates["train"], campaign_data["train"], s1_tr, product_ids, arm)
        va_base_slate = build_slate_from_stage1_scores(base_slates["val"], campaign_data["val"], s1_va, product_ids, arm)
        # Keep original 41D pairwise_features on base_slate so attach_stage3_groups_to_slate cleanly rebuilds arm + groups
        tr_idx = tr_base_slate["candidate_indices"].numpy()
        va_idx = va_base_slate["candidate_indices"].numpy()
        tr_base_slate["pairwise_features"] = torch.from_numpy(
            np.array(campaign_data["train"]["features"][np.arange(len(tr_idx))[:, None], tr_idx], dtype=np.float32)
        )
        va_base_slate["pairwise_features"] = torch.from_numpy(
            np.array(campaign_data["val"]["features"][np.arange(len(va_idx))[:, None], va_idx], dtype=np.float32)
        )
    else:
        tr_base_slate = base_slates["train"]
        va_base_slate = base_slates["val"]

    # Control run (0 extra groups)
    s1_results: Dict[str, Any] = {}
    s2_results: Dict[str, Any] = {}

    runs_plan: List[Tuple[str, Tuple[str, ...]]] = [("control", ())]
    for g in groups_to_test:
        runs_plan.append((g, (g,)))
    if combined_groups:
        runs_plan.append(("combined_selected", tuple(combined_groups)))

    for tag, grp_tuple in runs_plan:
        print(f"\n================ STAGE 3 (Stage 1 MLP): {tag} {grp_tuple} ================", flush=True)
        s1_results[tag] = fit_stage1_with_groups(
            arm=arm,
            groups=grp_tuple,
            tag=tag,
            campaign_data=campaign_data,
            comm_mmaps=comm_mmaps,
            spec_mmaps=spec_mmaps,
            info=info,
            meta=meta,
            aux_60d_labels=aux_60d,
            aux_lambda=s1_aux_lambda,
            p_drop=s1_p_drop,
            seed=seed,
            device=device,
        )

    for tag, grp_tuple in runs_plan:
        print(f"\n================ STAGE 3 (Stage 2 Transformer): {tag} {grp_tuple} ================", flush=True)
        tr_s = attach_stage3_groups_to_slate(tr_base_slate, campaign_data["train"], comm_mmaps["train"], spec_mmaps["train"], arm, grp_tuple)
        va_s = attach_stage3_groups_to_slate(va_base_slate, campaign_data["val"], comm_mmaps["val"], spec_mmaps["val"], arm, grp_tuple)
        s2_results[tag] = fit_stage2_with_groups(
            arm=arm,
            groups=grp_tuple,
            tag=tag,
            train_slate=tr_s,
            val_slate=va_s,
            train_data=campaign_data["train"],
            val_data=campaign_data["val"],
            product_vectors=product_vectors,
            product_ids=product_ids,
            meta=meta,
            aux_60d_labels=aux_60d,
            aux_lambda=s2_aux_lambda,
            p_drop=s2_p_drop,
            seed=seed,
            device=device,
        )

    s1_intervals = {
        f"ctrl_to_{tag}": paired_rolling_intervals(s1_results["control"]["predictions_npz"], s1_results[tag]["predictions_npz"])
        for tag, _ in runs_plan
        if tag != "control"
    }
    s2_intervals = {
        sel: {
            f"ctrl_to_{tag}": paired_rolling_intervals(
                s2_results["control"][sel]["predictions_npz"], s2_results[tag][sel]["predictions_npz"]
            )
            for tag, _ in runs_plan
            if tag != "control"
        }
        for sel in ("ndcg", "discovery")
    }

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")

    summary = {
        "stage": "stage3_commercial_ablations",
        "arm": arm,
        "s1_aux_lambda": s1_aux_lambda,
        "s1_p_drop": s1_p_drop,
        "s2_aux_lambda": s2_aux_lambda,
        "s2_p_drop": s2_p_drop,
        "seed": int(seed),
        "heldout_access": False,
        "stage1_mlp": s1_results,
        "stage1_paired_bootstrap_95ci": s1_intervals,
        "stage2_transformer": s2_results,
        "stage2_paired_bootstrap_95ci": s2_intervals,
        "preserved_hashes_verified": True,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, EVIDENCE_ROOT / "stage3_commercial_summary.json", overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=str, required=True, choices=VALID_BASE_NAME_ARMS)
    parser.add_argument("--s1-aux-lambda", type=float, default=0.0)
    parser.add_argument("--s1-p-drop", type=float, default=0.0)
    parser.add_argument("--s2-aux-lambda", type=float, default=0.0)
    parser.add_argument("--s2-p-drop", type=float, default=0.0)
    parser.add_argument("--slate-source-dir", type=Path, default=None)
    parser.add_argument("--groups", nargs="*", default=list(ALL_STAGE3_GROUPS))
    parser.add_argument("--combined-groups", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    summary = run_stage3(
        arm=args.arm,
        s1_aux_lambda=args.s1_aux_lambda,
        s1_p_drop=args.s1_p_drop,
        s2_aux_lambda=args.s2_aux_lambda,
        s2_p_drop=args.s2_p_drop,
        slate_source_dir=args.slate_source_dir,
        groups_to_test=args.groups,
        combined_groups=args.combined_groups,
        seed=args.seed,
        device=args.device,
    )
    print("\n================ STAGE 3 COMMERCIAL ABLATION SUMMARY ================", flush=True)
    for k, res in summary["stage1_mlp"].items():
        m = res["metrics"]
        print(
            f"S1 [{k:18s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']}) "
            f"New@25={m['new_hits@25']} New@50={m['new_hits@50']} New@100={m['new_hits@100']}"
        )
    for k, res in summary["stage2_transformer"].items():
        m = res["ndcg"]["metrics"]
        print(
            f"S2 [{k:18s}] R@5={m['recall@5']:.5f} R@10={m['recall@10']:.5f} NDCG@10={m['ndcg@10']:.5f} "
            f"New@10={m['new_hits@10']} (ps={m['pack_switch_hits@10']}, tn={m['true_new_bn_hits@10']})"
        )


if __name__ == "__main__":
    main()
