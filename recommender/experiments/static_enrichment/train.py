"""Stage 1 S0-protocol training and downstream dot/100 ranker execution for static enrichment."""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.experiment_retrieval import (
    FIELDS,
    metrics_from_scores,
    serving_latency,
)
from recommender.experiment_signals import sync
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.data import make_slates, sliced
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.dynamic_selector.feature_schema import contract
from recommender.experiments.dynamic_selector.train import run_directory, finish_run
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.shared_data import DEFAULT_RETRIEVER, atomic_torch_save
from recommender.train_ranker import checkpoint_digest, ranker_batch_inputs
from recommender.train_retriever import get_device
from recommender.train_shared_ranker import make_loader

from .config import DEFAULT_DISCOVERY_RANKER, TRANSFERRED_CUSTOMER_STAT_NAMES, VariantSpec
from .data import load_dynamic_campaign_features_readonly, quantize_transferred_stats
from .evaluate import stage1_cohort_metrics
from .model import (
    PrecomputedEmbeddingMLPRanker,
    StaticEnrichedRetriever,
    TransferredStatsDynamicRanker,
)


def make_variant_model(
    spec: VariantSpec,
    info: Mapping[str, Any],
    enrichment: Mapping[str, Any],
) -> StaticEnrichedRetriever:
    """Construct C0 (exact S0 architecture) or an enriched static variant from scratch."""
    original = torch.load(DEFAULT_RETRIEVER, map_location="cpu")
    s = original["model_state_dict"]
    demand_dim = (
        12
        if int(spec.product_demand_features) == 12
        else info["product_inputs"]["demand_features"].shape[1]
    )
    return StaticEnrichedRetriever(
        spec,
        num_cities=s["customer_tower.city_embedding.weight"].shape[0],
        num_regions=s["customer_tower.region_embedding.weight"].shape[0],
        num_industries=s["customer_tower.industry_embedding.weight"].shape[0],
        num_catalog_products=len(info["product_ids"]),
        num_demand_features=demand_dim,
        product_attr_vocab_sizes={
            name: len(vocab) for name, vocab in enrichment["product_vocabs"].items()
        },
        customer_attr_vocab_sizes={
            name: len(vocab) for name, vocab in enrichment["customer_vocabs"].items()
        },
    )


def _stage1_product_inputs(
    spec: VariantSpec,
    info: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    device: torch.device,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    raw_inputs = info["product_inputs"]
    end = limit if limit is not None else len(info["product_ids"])
    out: Dict[str, Any] = {
        key: value[:end].to(device)
        for key, value in raw_inputs.items()
        if key != "text_embedding"
    }
    out["text_embedding"] = enrichment["text_embeddings"][spec.text_recipe][:end].to(device)
    if spec.product_embeddings:
        out["attr_ids"] = {
            name: enrichment["product_attr_ids"][name][:end].to(device)
            for name in spec.product_embeddings
        }
    return out


def _stage1_customer_inputs(
    spec: VariantSpec,
    data: Mapping[str, Any],
    rows: torch.Tensor,
    enrichment: Mapping[str, Any],
    split: str,
    device: torch.device,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        key: data[key][rows].to(device)
        for key in (*FIELDS, "tabular_features", "seq_product_ids")
    }
    if spec.customer_embeddings:
        src = enrichment[f"{split}_customer_attrs"]
        out["extra_attr_ids"] = {
            name: src[name][rows].to(device)
            for name in spec.customer_embeddings
        }
    return out


@torch.no_grad()
def encode_and_score_variant(
    model: StaticEnrichedRetriever,
    spec: VariantSpec,
    data: Mapping[str, Any],
    info: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    split: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    prod_inputs = _stage1_product_inputs(spec, info, enrichment, device)
    vectors = model.product_vectors(prod_inputs)
    encoded, scores = [], []
    indices = torch.arange(len(vectors), device=device)
    for rows in torch.arange(len(data["target"])).split(256):
        customers = model.encode_purchase(
            _stage1_customer_inputs(spec, data, rows, enrichment, split, device),
            vectors,
        )
        encoded.append(customers.cpu())
        scores.append(
            model.score_cached(customers, vectors, indices, data["familiarity"][rows].to(device)).cpu()
        )
    return vectors.cpu(), torch.cat(encoded), torch.cat(scores)


def train_stage1_variant(
    spec: VariantSpec,
    seed: int,
    train: Mapping[str, Any],
    val: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    epochs: int = 30,
    device: Optional[str] = None,
    save_train_cache: bool = False,
) -> Dict[str, Any]:
    """Train one Stage 1 variant using the exact S0 protocol and verify strict replay."""
    dev = get_device(device)
    run_name = f"stage1-{spec.name}-seed{seed}"
    settings = {
        "stage": "stage1",
        "variant": spec.name,
        "spec": asdict(spec),
        "seed": seed,
        "epochs": epochs,
        "identity": info["identity"],
        "enrichment_identity": enrichment["identity"],
        "text_recipe_hash": enrichment["text_recipe_hashes"][spec.text_recipe],
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "pos_weight": 20.0,
        "selection": "new_hits@25, then new_hits@50; first epoch wins ties",
        "demand_policy": "disabled_v1",
        "metadata_policy": "snapshot_unverified",
        "initialization": "scratch",
    }
    directory, existing = run_directory(output_root, run_name, settings)
    if existing is not None:
        if save_train_cache and not (directory / "train_purchase_cache.pt").is_file():
            _ensure_train_purchase_cache(spec, directory, train, info, enrichment, dev)
        return existing

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_variant_model(spec, info, enrichment).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    useful = torch.where((train["target"] & train["unseen"]).any(-1))[0]
    generator = torch.Generator().manual_seed(seed)
    n_train = int(info["trained_products"])
    train_prod_inputs = _stage1_product_inputs(spec, info, enrichment, dev, limit=n_train)
    train_indices = torch.arange(n_train, device=dev)

    history = []
    best = (-1, -1)
    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for rows in useful[torch.randperm(len(useful), generator=generator)].split(256):
            optimizer.zero_grad(set_to_none=True)
            vectors = model.product_vectors(train_prod_inputs)
            customers = model.encode_purchase(
                _stage1_customer_inputs(spec, train, rows, enrichment, "train", dev),
                vectors,
            )
            logits = model.score_cached(
                customers,
                vectors,
                train_indices,
                train["familiarity"][rows, :n_train].to(dev),
            )
            allowed = train["unseen"][rows, :n_train].to(dev)
            target = train["target"][rows, :n_train].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(
                logits[allowed], target[allowed], pos_weight=torch.tensor(20.0, device=dev)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        _, _, val_logits = encode_and_score_variant(model, spec, val, info, enrichment, "val", dev)
        metrics = metrics_from_scores(val_logits, val, val_source)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})
        key = (metrics["new_hits@25"], metrics["new_hits@50"])
        if key > best:
            best = key
            atomic_torch_save(
                {"settings": settings, "model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                directory / "best.pt",
            )
        print(
            f"[{spec.name}] seed={seed} epoch={epoch:02d} new25={metrics['new_hits@25']} "
            f"new50={metrics['new_hits@50']} rec10={metrics['val_recall@10']:.4f}",
            flush=True,
        )
    sync(dev)
    elapsed = time.perf_counter() - started

    checkpoint = torch.load(directory / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    vectors, customers, val_logits = encode_and_score_variant(model, spec, val, info, enrichment, "val", dev)
    replayed = metrics_from_scores(val_logits, val, val_source)
    if replayed != checkpoint["metrics"]:
        raise AssertionError(f"Selected checkpoint replay mismatch for {run_name}")

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
            "model_digest": checkpoint_digest(directory / "best.pt"),
            "data_identity": info["identity"],
            "enrichment_identity": enrichment["identity"],
            "customer_vectors": customers,
            "product_vectors": vectors,
            "scores": val_logits,
        },
        directory / "val_purchase_cache.pt",
    )
    if save_train_cache:
        _ensure_train_purchase_cache(spec, directory, train, info, enrichment, dev, model=model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "settings": settings,
        "selected_epoch": checkpoint["epoch"],
        "metrics": replayed,
        "cohorts": cohorts,
        "training_seconds": elapsed,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(dev),
        "training_discovery_rows": len(useful),
        "selected_checkpoint": str(directory / "best.pt"),
    }
    atomic_json(history, directory / "history.json", overwrite=False)
    finish_run(directory, settings, result)
    return result


def _ensure_train_purchase_cache(
    spec: VariantSpec,
    directory: Path,
    train: Mapping[str, Any],
    info: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    device: torch.device,
    *,
    model: Optional[StaticEnrichedRetriever] = None,
) -> Path:
    cache_path = directory / "train_purchase_cache.pt"
    best_path = directory / "best.pt"
    best_digest = checkpoint_digest(best_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("model_digest") == best_digest and payload.get("enrichment_identity") == enrichment["identity"]:
            return cache_path
    if model is None:
        checkpoint = torch.load(best_path, map_location="cpu")
        model = make_variant_model(spec, info, enrichment).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    vectors, customers, logits = encode_and_score_variant(model, spec, train, info, enrichment, "train", device)
    atomic_torch_save(
        {
            "cache_version": 1,
            "model_digest": best_digest,
            "data_identity": info["identity"],
            "enrichment_identity": enrichment["identity"],
            "customer_vectors": customers,
            "product_vectors": vectors,
            "scores": logits,
        },
        cache_path,
    )
    return cache_path


def _make_downstream_ranker(
    mode: str,
    seed: int,
    stage1_checkpoint_path: Path,
    *,
    is_control_stage1: bool,
) -> CandidateContextRanker:
    torch.manual_seed(seed)
    base = UPSCANRRanker.from_retriever_checkpoint(
        DEFAULT_RETRIEVER,
        statistics_attention=True,
        excluded_statistics=("history_repeat",),
        cached_only=True,
    )

    def reset(module):
        if isinstance(module, torch.nn.MultiheadAttention):
            module._reset_parameters()
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

    base.apply(reset)
    state = torch.load(stage1_checkpoint_path, map_location="cpu")["model_state_dict"]
    prefix = "towers.customer_tower." if is_control_stage1 else "customer_tower."
    with torch.no_grad():
        for name in (
            "text_proj.0.weight",
            "text_proj.0.bias",
            "city_embedding.weight",
            "region_embedding.weight",
            "industry_embedding.weight",
        ):
            dict(base.named_parameters())[name].copy_(state[prefix + name])
        ref_config = torch.load(DEFAULT_RETRIEVER, map_location="cpu")["config"]
        target = base.tabular_proj[0].weight
        source = state[prefix + "tabular_proj.0.weight"]
        target[:, :48].copy_(source[:, :48])
        # Extra customer categorical embeddings (if any) sit between :48 and the last 6 profile features.
        # Clean v2 Stage 1 towers delete the 6 profile stats, so only copy them when present.
        if source.shape[1] >= 48 + len(ref_config["profile_features"]):
            profile_offset = source.shape[1] - len(ref_config["profile_features"])
            for i, name in enumerate(ref_config["profile_features"]):
                if name in LEGACY_PROFILE_FEATURES:
                    target[:, 48 + LEGACY_PROFILE_FEATURES.index(name)].copy_(source[:, profile_offset + i])
        base.tabular_proj[0].bias.copy_(state[prefix + "tabular_proj.0.bias"])
    with torch.random.fork_rng(devices=[]):
        return CandidateContextRanker(base, mode)


def run_downstream_for_stage1(
    variant_name: str,
    spec: VariantSpec,
    seed: int,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    modes: tuple[str, ...] = ("control", "dynamic_features"),
    epochs: int = 10,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run dynamic-campaign dot/100 original ranker and scalar-features ranker for a Stage 1 run."""
    dev = get_device(device)
    stage1_dir = Path(output_root) / "runs" / f"stage1-{variant_name}-seed{seed}"
    stage1_ckpt = stage1_dir / "best.pt"
    stage1_digest = file_digest(stage1_ckpt)
    _ensure_train_purchase_cache(spec, stage1_dir, train_retrieval, info, enrichment, dev)

    dyn_manifest, train_feat_mmap, val_feat_mmap = load_dynamic_campaign_features_readonly()
    if dyn_manifest["source_identity"] != info["identity"]:
        raise ValueError("Dynamic campaign source identity mismatch.")

    campaign_data: Dict[str, Dict[str, Any]] = {}
    slates: Dict[str, Dict[str, Any]] = {}
    for split, ret_data, src_data, feat_mmap in (
        ("train", train_retrieval, train_source, train_feat_mmap),
        ("val", val_retrieval, val_source, val_feat_mmap),
    ):
        rows = torch.where(ret_data["allowed"].sum(1) >= 100)[0]
        sliced_src = sliced(src_data, rows, len(src_data["customer_ids"]))
        sliced_ret = sliced(ret_data, rows, len(src_data["customer_ids"]))
        payload = torch.load(stage1_dir / f"{split}_purchase_cache.pt", map_location="cpu")
        if payload["model_digest"] != stage1_digest or payload["data_identity"] != info["identity"]:
            raise ValueError(f"Purchase cache provenance mismatch in {stage1_dir}")
        sliced_ret.update(
            source=sliced_src,
            u=payload["customer_vectors"][rows].squeeze(1),
            v=payload["product_vectors"],
            dot_scores=payload["scores"][rows],
            features=feat_mmap,
        )
        campaign_data[split] = sliced_ret
        slates[split] = make_slates(sliced_ret, sliced_ret["dot_scores"], info["product_ids"], "quota100_50")

    reference = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu")
    recall_guard = float(reference["discovery_min_overall_recall"])
    ndcg_guard = float(reference["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)
    vectors_dev = campaign_data["train"]["v"].to(dev)

    results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        run_name = f"ranker-{variant_name}-dot100-{mode}-seed{seed}"
        settings = {
            "stage": "ranker",
            "upstream_variant": variant_name,
            "mode": mode,
            "policy": "quota100_50",
            "seed": seed,
            "epochs": epochs,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "identity": dyn_manifest["identity"],
            "feature_contract": contract(),
            "upstream_digest": stage1_digest,
            "train_slate_identity": slates["train"]["cache_identity"],
            "val_slate_identity": slates["val"]["cache_identity"],
            "recall_guardrail": recall_guard,
            "ndcg_guardrail": ndcg_guard,
            "guardrail_artifact_sha256": guard_sha,
            "selection": "new_hits@10 then NDCG subject to both reference quality floors",
        }
        directory, existing = run_directory(output_root, run_name, settings)
        if existing is not None:
            results[mode] = existing
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _make_downstream_ranker(
            mode, seed, stage1_ckpt, is_control_stage1=spec.is_control
        ).to(dev)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
        loss_fn = RankerLoss().to(dev)
        loader = make_loader(slates["train"], shuffle=True, seed=seed)

        best = (-1, -1.0, -1)
        history = []
        sync(dev)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                inputs = ranker_batch_inputs(batch, vectors_dev, model, dev)
                inputs["cached_customer_vectors"] = batch["stage1_customer_vectors"].to(dev)
                logits = model(**inputs).logits
                loss = loss_fn(logits, batch["labels"].to(dev))["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_scores = torch.cat([
                    model(
                        **ranker_batch_inputs(b, vectors_dev, model, dev),
                        cached_customer_vectors=b["stage1_customer_vectors"].to(dev),
                    ).logits.cpu()
                    for b in make_loader(slates["val"])
                ])
            metrics = ranking_metrics(val_scores, campaign_data["val"], slates["val"]["candidate_indices"])
            key = (
                int(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard),
                metrics["new_hits@10"],
                metrics["ndcg@10"],
            )
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})
            if key > best:
                best = key
                atomic_torch_save(
                    {
                        "settings": settings,
                        "epoch": epoch,
                        "metrics": metrics,
                        "model_state_dict": model.state_dict(),
                    },
                    directory / "best.pt",
                )
            print(
                f"ranker [{variant_name}/dot100/{mode}] seed={seed} epoch={epoch:02d} "
                f"new={metrics['new_hits@10']} rec={metrics['recall@10']:.5f} ndcg={metrics['ndcg@10']:.5f}",
                flush=True,
            )
        sync(dev)
        elapsed = time.perf_counter() - started

        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            scores = torch.cat([
                model(
                    **ranker_batch_inputs(b, vectors_dev, model, dev),
                    cached_customer_vectors=b["stage1_customer_vectors"].to(dev),
                ).logits.cpu()
                for b in make_loader(slates["val"])
            ])
        metrics = ranking_metrics(
            scores, campaign_data["val"], slates["val"]["candidate_indices"], directory / "predictions.npz"
        )
        if metrics != checkpoint["metrics"]:
            raise AssertionError(f"Downstream checkpoint replay mismatch for {run_name}")
        result = {
            "settings": settings,
            "epoch": checkpoint["epoch"],
            "metrics": metrics,
            "qualified": bool(best[0]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training_seconds": elapsed,
            "selected_checkpoint": str(directory / "best.pt"),
        }
        atomic_json(history, directory / "history.json", overwrite=False)
        finish_run(directory, settings, result)
        results[mode] = result
    return results


def _make_clean_v2_ranker(
    mode: str,
    seed: int,
    stage1_checkpoint_path: Path,
    *,
    is_control_stage1: bool,
) -> torch.nn.Module:
    if mode == "dot100_41":
        return _make_downstream_ranker(
            "dynamic_features",
            seed,
            stage1_checkpoint_path,
            is_control_stage1=is_control_stage1,
        )
    if mode == "dot100_47":
        dyn = _make_downstream_ranker(
            "dynamic_features",
            seed,
            stage1_checkpoint_path,
            is_control_stage1=is_control_stage1,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return TransferredStatsDynamicRanker(
                dyn, num_customer_stats=len(TRANSFERRED_CUSTOMER_STAT_NAMES)
            )
    if mode == "precomputed_mlp_47":
        torch.manual_seed(seed)
        return PrecomputedEmbeddingMLPRanker(
            embedding_dim=128,
            num_pairwise_features=41,
            num_customer_stats=len(TRANSFERRED_CUSTOMER_STAT_NAMES),
        )
    raise ValueError(f"Unknown clean_v2 downstream mode: {mode}")


def run_clean_v2_downstream(
    stage1_runs: Sequence[tuple[str, VariantSpec, str]],
    seed: int,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    output_root: Path,
    *,
    epochs: int = 10,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run clean_v2 downstream configs: (variant_name, spec, mode) tuples on seed=42."""
    dev = get_device(device)
    dyn_manifest, train_feat_mmap, val_feat_mmap = load_dynamic_campaign_features_readonly()
    if dyn_manifest["source_identity"] != info["identity"]:
        raise ValueError("Dynamic campaign source identity mismatch.")

    reference = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu")
    recall_guard = float(reference["discovery_min_overall_recall"])
    ndcg_guard = float(reference["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)

    slates_cache: Dict[str, tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Path, str]] = {}
    results: Dict[str, Dict[str, Any]] = {}

    for variant_name, spec, mode in stage1_runs:
        if variant_name not in slates_cache:
            stage1_dir = Path(output_root) / "runs" / f"stage1-{variant_name}-seed{seed}"
            stage1_ckpt = stage1_dir / "best.pt"
            stage1_digest = file_digest(stage1_ckpt)
            _ensure_train_purchase_cache(spec, stage1_dir, train_retrieval, info, enrichment, dev)

            campaign_data: Dict[str, Dict[str, Any]] = {}
            slates: Dict[str, Dict[str, Any]] = {}
            for split, ret_data, src_data, feat_mmap in (
                ("train", train_retrieval, train_source, train_feat_mmap),
                ("val", val_retrieval, val_source, val_feat_mmap),
            ):
                rows = torch.where(ret_data["allowed"].sum(1) >= 100)[0]
                sliced_src = sliced(src_data, rows, len(src_data["customer_ids"]))
                sliced_ret = sliced(ret_data, rows, len(src_data["customer_ids"]))
                payload = torch.load(stage1_dir / f"{split}_purchase_cache.pt", map_location="cpu")
                if payload["model_digest"] != stage1_digest or payload["data_identity"] != info["identity"]:
                    raise ValueError(f"Purchase cache provenance mismatch in {stage1_dir}")
                sliced_ret.update(
                    source=sliced_src,
                    u=payload["customer_vectors"][rows].squeeze(1),
                    v=payload["product_vectors"],
                    dot_scores=payload["scores"][rows],
                    features=feat_mmap,
                )
                campaign_data[split] = sliced_ret
                slate = make_slates(sliced_ret, sliced_ret["dot_scores"], info["product_ids"], "quota100_50")
                # Quantize the 6 point-in-time log1p-scaled customer stats identically to float16->float32.
                slate["retriever_tabular_features"] = quantize_transferred_stats(
                    slate["retriever_tabular_features"]
                )
                slates[split] = slate
            slates_cache[variant_name] = (campaign_data, slates, stage1_ckpt, stage1_digest)

        campaign_data, slates, stage1_ckpt, stage1_digest = slates_cache[variant_name]
        vectors_dev = campaign_data["train"]["v"].to(dev)
        config_key = f"{variant_name}-{mode}"
        run_name = f"ranker-{config_key}-seed{seed}"
        settings = {
            "stage": "ranker_clean_v2",
            "upstream_variant": variant_name,
            "mode": mode,
            "policy": "quota100_50",
            "seed": seed,
            "epochs": epochs,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "identity": dyn_manifest["identity"],
            "feature_contract": contract(),
            "transferred_customer_stats": list(TRANSFERRED_CUSTOMER_STAT_NAMES)
            if mode.endswith("_47") else [],
            "upstream_digest": stage1_digest,
            "train_slate_identity": slates["train"]["cache_identity"],
            "val_slate_identity": slates["val"]["cache_identity"],
            "recall_guardrail": recall_guard,
            "ndcg_guardrail": ndcg_guard,
            "guardrail_artifact_sha256": guard_sha,
            "selection": "new_hits@10 then NDCG subject to both reference quality floors",
        }
        directory, existing = run_directory(output_root, run_name, settings)
        if existing is not None:
            results[config_key] = existing
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _make_clean_v2_ranker(
            mode, seed, stage1_ckpt, is_control_stage1=spec.is_control
        ).to(dev)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
        loss_fn = RankerLoss().to(dev)
        loader = make_loader(slates["train"], shuffle=True, seed=seed)
        needs_stats = mode.endswith("_47")

        def _forward_batch(batch: Mapping[str, Any]) -> torch.Tensor:
            inputs = ranker_batch_inputs(batch, vectors_dev, model, dev)
            inputs["cached_customer_vectors"] = batch["stage1_customer_vectors"].to(dev)
            if needs_stats:
                inputs["transferred_customer_stats"] = batch["retriever_tabular_features"].to(dev)
            return model(**inputs).logits

        best = (-1, -1.0, -1)
        history = []
        sync(dev)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_batch(batch)
                loss = loss_fn(logits, batch["labels"].to(dev))["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_scores = torch.cat([_forward_batch(b).cpu() for b in make_loader(slates["val"])])
            metrics = ranking_metrics(val_scores, campaign_data["val"], slates["val"]["candidate_indices"])
            key = (
                int(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard),
                metrics["new_hits@10"],
                metrics["ndcg@10"],
            )
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})
            if key > best:
                best = key
                atomic_torch_save(
                    {
                        "settings": settings,
                        "epoch": epoch,
                        "metrics": metrics,
                        "model_state_dict": model.state_dict(),
                    },
                    directory / "best.pt",
                )
            print(
                f"ranker [{config_key}] seed={seed} epoch={epoch:02d} "
                f"new={metrics['new_hits@10']} rec={metrics['recall@10']:.5f} ndcg={metrics['ndcg@10']:.5f}",
                flush=True,
            )
        sync(dev)
        elapsed = time.perf_counter() - started

        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            scores = torch.cat([_forward_batch(b).cpu() for b in make_loader(slates["val"])])
        metrics = ranking_metrics(
            scores, campaign_data["val"], slates["val"]["candidate_indices"], directory / "predictions.npz"
        )
        if metrics != checkpoint["metrics"]:
            raise AssertionError(f"Clean v2 downstream replay mismatch for {run_name}")
        result = {
            "settings": settings,
            "epoch": checkpoint["epoch"],
            "metrics": metrics,
            "qualified": bool(best[0]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training_seconds": elapsed,
            "selected_checkpoint": str(directory / "best.pt"),
        }
        atomic_json(history, directory / "history.json", overwrite=False)
        finish_run(directory, settings, result)
        results[config_key] = result
    return results
