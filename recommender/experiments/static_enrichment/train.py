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
from recommender.experiments.dynamic_selector.evaluate import paired_intervals, ranking_metrics
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES, contract
from recommender.experiments.dynamic_selector.model import select_candidates
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
    CompactStage1Selector,
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
                {
                    "settings": settings,
                    "customer_tabular_features": int(spec.customer_tabular_features),
                    "customer_profile_features": (
                        list(TRANSFERRED_CUSTOMER_STAT_NAMES[: int(spec.customer_tabular_features)])
                        if int(spec.customer_tabular_features) > 0
                        else []
                    ),
                    "customer_embeddings": list(spec.customer_embeddings),
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
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


def _resolve_stage1_customer_tabular_contract(
    checkpoint: Mapping[str, Any],
    source_width: int,
    ref_profile_features: Sequence[str],
    checkpoint_path: Path,
) -> tuple[int, list[str], list[str]]:
    """Extract explicit Stage 1 customer tabular metadata and reject ambiguous widths."""
    meta_spec = checkpoint.get("settings", {}).get("spec")
    if "customer_tabular_features" in checkpoint:
        num_tab = int(checkpoint["customer_tabular_features"])
        prof_names = list(
            checkpoint.get("customer_profile_features", ref_profile_features if num_tab > 0 else [])
        )
        cust_embs = list(checkpoint.get("customer_embeddings", []))
    elif isinstance(meta_spec, Mapping) and "customer_tabular_features" in meta_spec:
        num_tab = int(meta_spec["customer_tabular_features"])
        prof_names = list(ref_profile_features) if num_tab > 0 else []
        cust_embs = list(meta_spec.get("customer_embeddings", []))
    else:
        if source_width == 48:
            return 0, [], []
        raise ValueError(
            f"Checkpoint {checkpoint_path} has customer tabular input width {source_width} "
            "without explicit customer_tabular_features / customer_embeddings metadata."
        )

    expected_width = 48 + 16 * len(cust_embs) + len(prof_names)
    if source_width != expected_width:
        raise ValueError(
            f"Customer tabular width mismatch in {checkpoint_path}: got {source_width}, "
            f"expected {expected_width} from customer_embeddings={cust_embs} and "
            f"customer_profile_features={prof_names}."
        )
    if len(prof_names) != num_tab:
        raise ValueError(
            f"Customer profile feature count mismatch in {checkpoint_path}: "
            f"customer_tabular_features={num_tab} vs len(customer_profile_features)={len(prof_names)}."
        )
    return num_tab, prof_names, cust_embs


def _make_downstream_ranker(
    mode: str,
    seed: int,
    stage1_checkpoint_path: Optional[Path] = None,
    *,
    is_control_stage1: bool = False,
    init_mode: str = "stage1_copied",
    init_checkpoint_path: Optional[Path] = None,
) -> CandidateContextRanker:
    if init_mode not in ("stage1_copied", "common_checkpoint", "scratch"):
        raise ValueError(f"Unsupported ranker init_mode: {init_mode!r}")
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
    if init_mode != "scratch":
        ckpt_path = Path(init_checkpoint_path) if init_checkpoint_path is not None else stage1_checkpoint_path
        if ckpt_path is None:
            raise ValueError(f"init_mode={init_mode!r} requires a Stage 1 checkpoint path.")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint["model_state_dict"]
        if "towers.customer_tower.text_proj.0.weight" in state:
            prefix = "towers.customer_tower."
        elif "customer_tower.text_proj.0.weight" in state:
            prefix = "customer_tower."
        else:
            prefix = "towers.customer_tower." if is_control_stage1 else "customer_tower."
        ref_config = torch.load(DEFAULT_RETRIEVER, map_location="cpu")["config"]
        source = state[prefix + "tabular_proj.0.weight"]
        num_tab, prof_names, cust_embs = _resolve_stage1_customer_tabular_contract(
            checkpoint, int(source.shape[1]), ref_config["profile_features"], Path(ckpt_path)
        )
        with torch.no_grad():
            for name in (
                "text_proj.0.weight",
                "text_proj.0.bias",
                "city_embedding.weight",
                "region_embedding.weight",
                "industry_embedding.weight",
            ):
                dict(base.named_parameters())[name].copy_(state[prefix + name])
            target = base.tabular_proj[0].weight
            target[:, :48].copy_(source[:, :48])
            # Static customer embeddings (e.g. sales_branch 16D) occupy [48 : 48 + 16*k].
            # Only copy profile stats when explicit metadata confirms num_tab > 0.
            if num_tab > 0:
                profile_offset = 48 + 16 * len(cust_embs)
                for i, name in enumerate(prof_names):
                    if name in LEGACY_PROFILE_FEATURES:
                        target[:, 48 + LEGACY_PROFILE_FEATURES.index(name)].copy_(
                            source[:, profile_offset + i]
                        )
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


def _load_stage1_campaign_data(
    spec: VariantSpec,
    stage1_dir: Path,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    dev: torch.device,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Path, str]:
    """Load split caches aligned to the >=100 eligible-SKU cohort used by dynamic_campaign."""
    stage1_ckpt = Path(stage1_dir) / "best.pt"
    stage1_digest = file_digest(stage1_ckpt)
    _ensure_train_purchase_cache(spec, Path(stage1_dir), train_retrieval, info, enrichment, dev)

    dyn_manifest, train_feat_mmap, val_feat_mmap = load_dynamic_campaign_features_readonly()
    if dyn_manifest["source_identity"] != info["identity"]:
        raise ValueError("Dynamic campaign source identity mismatch.")

    campaign_data: Dict[str, Dict[str, Any]] = {}
    for split, ret_data, src_data, feat_mmap in (
        ("train", train_retrieval, train_source, train_feat_mmap),
        ("val", val_retrieval, val_source, val_feat_mmap),
    ):
        rows = torch.where(ret_data["allowed"].sum(1) >= 100)[0]
        sliced_src = sliced(src_data, rows, len(src_data["customer_ids"]))
        sliced_ret = sliced(ret_data, rows, len(src_data["customer_ids"]))
        payload = torch.load(Path(stage1_dir) / f"{split}_purchase_cache.pt", map_location="cpu")
        if payload["model_digest"] != stage1_digest or payload["data_identity"] != info["identity"]:
            raise ValueError(f"Purchase cache provenance mismatch in {stage1_dir} ({split})")
        sliced_ret.update(
            source=sliced_src,
            cohort_rows=rows,
            u=payload["customer_vectors"][rows].squeeze(1),
            v=payload["product_vectors"],
            dot_scores=payload["scores"][rows],
            features=feat_mmap,
            customer_stats_6=quantize_transferred_stats(sliced_src["retriever_tabular_features"]),
        )
        campaign_data[split] = sliced_ret
    return campaign_data, dyn_manifest, stage1_ckpt, stage1_digest


def slate_coverage_summary(
    candidate_indices: torch.Tensor,
    target: torch.Tensor,
    unseen: torch.Tensor,
) -> Dict[str, Any]:
    """Compute overall, repeat, and new-to-customer target coverage inside a candidate slate."""
    slate_targets = target.gather(1, candidate_indices)
    slate_unseen = unseen.gather(1, candidate_indices)
    total_targets = int(target.sum().item())
    new_targets = int((target & unseen).sum().item())
    repeat_targets = total_targets - new_targets
    targets_in_slate = int(slate_targets.sum().item())
    new_in_slate = int((slate_targets & slate_unseen).sum().item())
    repeat_in_slate = targets_in_slate - new_in_slate
    return {
        "slate_size": int(candidate_indices.shape[1]),
        "total_targets": total_targets,
        "targets_in_slate": targets_in_slate,
        "overall_slate_coverage": targets_in_slate / max(total_targets, 1),
        "repeat_targets": repeat_targets,
        "repeat_targets_in_slate": repeat_in_slate,
        "repeat_slate_coverage": repeat_in_slate / max(repeat_targets, 1),
        "new_targets": new_targets,
        "new_targets_in_slate": new_in_slate,
        "new_slate_coverage": new_in_slate / max(new_targets, 1),
    }


def _score_compact_selector_split(
    model: CompactStage1Selector,
    split_data: Mapping[str, Any],
    dev: torch.device,
    *,
    batch_size: int = 128,
    zero_features_41: bool = False,
    zero_customer_stats_6: bool = False,
) -> torch.Tensor:
    """Score all eligible products in split_data with optional zero-ablation of dynamic blocks."""
    model.eval()
    v_dev = split_data["v"].to(dev)
    u_all = split_data["u"]
    stats_all = split_data["customer_stats_6"]
    allowed_all = split_data["allowed"]
    feat_mmap = split_data["features"]
    dot_all = split_data["dot_scores"]
    if feat_mmap.shape[0] != len(u_all):
        raise ValueError(
            f"Dynamic feature memmap row count ({feat_mmap.shape[0]}) != split cohort size ({len(u_all)})."
        )

    chunks = []
    with torch.no_grad():
        for start in range(0, len(u_all), batch_size):
            end = min(start + batch_size, len(u_all))
            f41 = torch.from_numpy(np.array(feat_mmap[start:end], dtype=np.float32)).to(dev)
            if zero_features_41:
                f41.zero_()
            stats = stats_all[start:end].to(dev)
            if zero_customer_stats_6:
                stats = torch.zeros_like(stats)
            ref = dot_all[start:end].to(dev) if model.recipe == "uv_47_ref" else None
            logits = model(
                u=u_all[start:end].to(dev),
                v=v_dev,
                features_41=f41,
                customer_stats_6=stats,
                reference_scores=ref,
            )
            masked = logits.masked_fill(~allowed_all[start:end].to(dev), -1e9)
            chunks.append(masked.cpu())
    return torch.cat(chunks, dim=0)


def fit_compact_stage1_selector(
    variant_name: str,
    spec: VariantSpec,
    recipe: str,
    seed: int,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    stage1_dir: Path,
    selector_root: Path,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    pos_weight: float = 20.0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train CompactStage1Selector on the full rolling pool to score Stage 1 quota100_50 slates."""
    dev = get_device(device)
    campaign_data, dyn_manifest, _, stage1_digest = _load_stage1_campaign_data(
        spec, stage1_dir, train_retrieval, val_retrieval, info, train_source, val_source, enrichment, dev
    )
    torch.manual_seed(seed)
    probe_model = CompactStage1Selector(recipe=recipe)
    input_contract = {
        **probe_model.input_contract(),
        "features_41_names": list(DYNAMIC_FEATURE_NAMES),
        "customer_stats_6_names": list(TRANSFERRED_CUSTOMER_STAT_NAMES),
    }

    run_name = f"selector-{variant_name}-{recipe}-seed{seed}"
    settings = {
        "stage": "stage1_compact_selector_v3",
        "upstream_variant": variant_name,
        "upstream_digest": stage1_digest,
        "recipe": recipe,
        "input_contract": input_contract,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pos_weight": pos_weight,
        "loss": "BCEWithLogitsLoss_eligible_pool",
        "selection": "new_hits@50, then ndcg@10, then recall@10",
        "dynamic_identity": dyn_manifest["identity"],
        "data_identity": info["identity"],
        "enrichment_identity": enrichment["identity"],
    }
    directory, existing = run_directory(selector_root, run_name, settings)
    if existing is not None:
        if (directory / "train_scores.pt").is_file() and (directory / "val_scores.pt").is_file():
            return existing
        raise FileNotFoundError(f"Completed selector run {directory} is missing cached score tensors.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CompactStage1Selector(recipe=recipe).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    tr = campaign_data["train"]
    va = campaign_data["val"]
    v_dev = tr["v"].to(dev)
    tr_feat_mmap = tr["features"]
    if tr_feat_mmap.shape[0] != len(tr["u"]) or va["features"].shape[0] != len(va["u"]):
        raise ValueError(
            f"Dynamic feature memmap cohort mismatch: train=({tr_feat_mmap.shape[0]} vs {len(tr['u'])}), "
            f"val=({va['features'].shape[0]} vs {len(va['u'])})."
        )
    generator = torch.Generator().manual_seed(seed)
    pos_weight_tensor = torch.tensor(float(pos_weight), device=dev)

    best = (-1, -1.0, -1.0)
    history = []
    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        perm = torch.randperm(len(tr["u"]), generator=generator)
        for batch_rows in perm.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            f41 = torch.from_numpy(np.array(tr_feat_mmap[batch_rows.numpy()], dtype=np.float32)).to(dev)
            stats = tr["customer_stats_6"][batch_rows].to(dev)
            ref = tr["dot_scores"][batch_rows].to(dev) if recipe == "uv_47_ref" else None
            logits = model(
                u=tr["u"][batch_rows].to(dev),
                v=v_dev,
                features_41=f41,
                customer_stats_6=stats,
                reference_scores=ref,
            )
            allowed = tr["allowed"][batch_rows].to(dev)
            target = tr["target"][batch_rows].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(logits[allowed], target[allowed], pos_weight=pos_weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        val_scores = _score_compact_selector_split(model, va, dev)
        metrics = ranking_metrics(val_scores, va, None)
        key = (metrics["new_hits@50"], metrics["ndcg@10"], metrics["recall@10"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})
        if key > best:
            best = key
            atomic_torch_save(
                {
                    "settings": settings,
                    "epoch": epoch,
                    "metrics": metrics,
                    "input_contract": input_contract,
                    "model_state_dict": model.state_dict(),
                },
                directory / "best.pt",
            )
        print(
            f"selector [{variant_name}/{recipe}] seed={seed} epoch={epoch:02d} "
            f"new25={metrics['new_hits@25']} new50={metrics['new_hits@50']} "
            f"rec10={metrics['recall@10']:.5f} ndcg={metrics['ndcg@10']:.5f}",
            flush=True,
        )
    sync(dev)
    elapsed = time.perf_counter() - started

    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    val_scores = _score_compact_selector_split(model, va, dev)
    replayed = ranking_metrics(val_scores, va, None, directory / "predictions.npz")
    if replayed != checkpoint["metrics"]:
        raise AssertionError(f"Selector replay mismatch for {run_name}")
    train_scores = _score_compact_selector_split(model, tr, dev)

    atomic_torch_save(
        {
            "selector_digest": checkpoint_digest(directory / "best.pt"),
            "upstream_digest": stage1_digest,
            "data_identity": info["identity"],
            "scores": train_scores,
        },
        directory / "train_scores.pt",
    )
    atomic_torch_save(
        {
            "selector_digest": checkpoint_digest(directory / "best.pt"),
            "upstream_digest": stage1_digest,
            "data_identity": info["identity"],
            "scores": val_scores,
        },
        directory / "val_scores.pt",
    )

    val_cand = select_candidates(val_scores, va["allowed"], va["unseen"], policy="quota100_50")
    val_slate_cov = slate_coverage_summary(val_cand, va["target"], va["unseen"])

    ablations: Dict[str, Any] = {
        "full": {"full_pool": replayed, "quota100_50_coverage": val_slate_cov},
    }
    for abl_name, zero_f41, zero_s6 in (
        ("zero_features_41", True, False),
        ("zero_customer_stats_6", False, True),
        ("zero_both_47", True, True),
    ):
        abl_scores = _score_compact_selector_split(
            model, va, dev, zero_features_41=zero_f41, zero_customer_stats_6=zero_s6
        )
        abl_cand = select_candidates(abl_scores, va["allowed"], va["unseen"], policy="quota100_50")
        ablations[abl_name] = {
            "full_pool": ranking_metrics(abl_scores, va, None),
            "quota100_50_coverage": slate_coverage_summary(abl_cand, va["target"], va["unseen"]),
        }

    result = {
        "settings": settings,
        "selected_epoch": checkpoint["epoch"],
        "metrics": replayed,
        "quota100_50_coverage": val_slate_cov,
        "zero_ablations": ablations,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": elapsed,
        "selected_checkpoint": str(directory / "best.pt"),
        "train_scores_path": str(directory / "train_scores.pt"),
        "val_scores_path": str(directory / "val_scores.pt"),
        "predictions_path": str(directory / "predictions.npz"),
    }
    atomic_json(history, directory / "history.json", overwrite=False)
    finish_run(directory, settings, result)
    return result


def run_v3_stage2_ranker(
    run_name: str,
    variant_name: str,
    spec: VariantSpec,
    stage1_dir: Path,
    output_root: Path,
    seed: int,
    train_retrieval: Mapping[str, Any],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    train_source: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    *,
    selector_type: str = "dot",
    selector_dir: Optional[Path] = None,
    init_mode: str = "stage1_copied",
    init_checkpoint_path: Optional[Path] = None,
    slate_override_stage1_dir: Optional[Path] = None,
    slate_override_spec: Optional[VariantSpec] = None,
    epochs: int = 10,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train UPS-CANR (dynamic_features, quota100_50) and preserve both discovery & NDCG checkpoints."""
    dev = get_device(device)
    campaign_data, dyn_manifest, stage1_ckpt, stage1_digest = _load_stage1_campaign_data(
        spec, stage1_dir, train_retrieval, val_retrieval, info, train_source, val_source, enrichment, dev
    )

    if slate_override_stage1_dir is not None:
        if selector_type != "dot":
            raise ValueError("slate_override_stage1_dir only supports selector_type='dot'.")
        override_spec = slate_override_spec or spec
        ref_data, _, _, ref_stage1_digest = _load_stage1_campaign_data(
            override_spec,
            Path(slate_override_stage1_dir),
            train_retrieval,
            val_retrieval,
            info,
            train_source,
            val_source,
            enrichment,
            dev,
        )
        train_scores_for_slate = ref_data["train"]["dot_scores"]
        val_scores_for_slate = ref_data["val"]["dot_scores"]
        selector_digest = ref_stage1_digest
    elif selector_type == "dot":
        train_scores_for_slate = campaign_data["train"]["dot_scores"]
        val_scores_for_slate = campaign_data["val"]["dot_scores"]
        selector_digest = stage1_digest
    else:
        if selector_dir is None:
            raise ValueError(f"selector_dir is required when selector_type={selector_type!r}.")
        tr_payload = torch.load(Path(selector_dir) / "train_scores.pt", map_location="cpu")
        va_payload = torch.load(Path(selector_dir) / "val_scores.pt", map_location="cpu")
        if tr_payload["upstream_digest"] != stage1_digest or va_payload["upstream_digest"] != stage1_digest:
            raise ValueError("Selector score cache does not match Stage 1 checkpoint digest.")
        train_scores_for_slate = tr_payload["scores"]
        val_scores_for_slate = va_payload["scores"]
        selector_digest = tr_payload["selector_digest"]

    slates = {
        "train": make_slates(campaign_data["train"], train_scores_for_slate, info["product_ids"], "quota100_50"),
        "val": make_slates(campaign_data["val"], val_scores_for_slate, info["product_ids"], "quota100_50"),
    }
    if slate_override_stage1_dir is not None:
        ref_val_cand = select_candidates(
            val_scores_for_slate,
            campaign_data["val"]["allowed"],
            campaign_data["val"]["unseen"],
            policy="quota100_50",
        )
        if not torch.equal(slates["val"]["candidate_indices"], ref_val_cand):
            raise AssertionError("Slate override candidate_indices mismatch against reference slates.")

    val_slate_cov = slate_coverage_summary(
        slates["val"]["candidate_indices"],
        campaign_data["val"]["target"],
        campaign_data["val"]["unseen"],
    )

    reference = torch.load(DEFAULT_DISCOVERY_RANKER, map_location="cpu")
    recall_guard = float(reference["discovery_min_overall_recall"])
    ndcg_guard = float(reference["val_metrics"]["val_ndcg@10"])
    guard_sha = file_digest(DEFAULT_DISCOVERY_RANKER)

    resolved_init_ckpt = Path(init_checkpoint_path) if init_checkpoint_path is not None else stage1_ckpt
    settings = {
        "stage": "ranker_stage1_mlp_v3",
        "run_name": run_name,
        "upstream_variant": variant_name,
        "upstream_digest": stage1_digest,
        "selector_type": selector_type,
        "selector_digest": selector_digest,
        "init_mode": init_mode,
        "init_checkpoint": str(resolved_init_ckpt) if init_mode != "scratch" else None,
        "init_checkpoint_digest": file_digest(resolved_init_ckpt) if init_mode != "scratch" else None,
        "slate_override_stage1_dir": str(slate_override_stage1_dir) if slate_override_stage1_dir else None,
        "policy": "quota100_50",
        "ranker_mode": "dynamic_features",
        "seed": seed,
        "epochs": epochs,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "dynamic_identity": dyn_manifest["identity"],
        "feature_contract": contract(),
        "train_slate_identity": slates["train"]["cache_identity"],
        "val_slate_identity": slates["val"]["cache_identity"],
        "recall_guardrail": recall_guard,
        "ndcg_guardrail": ndcg_guard,
        "guardrail_artifact_sha256": guard_sha,
    }
    directory, existing = run_directory(output_root, run_name, settings)
    if existing is not None:
        return existing

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _make_downstream_ranker(
        "dynamic_features",
        seed,
        stage1_ckpt,
        is_control_stage1=spec.is_control,
        init_mode=init_mode,
        init_checkpoint_path=init_checkpoint_path,
    ).to(dev)
    vectors_dev = campaign_data["train"]["v"].to(dev)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    loss_fn = RankerLoss().to(dev)
    loader = make_loader(slates["train"], shuffle=True, seed=seed)

    def _forward(batch: Mapping[str, Any]) -> torch.Tensor:
        return model(
            **ranker_batch_inputs(batch, vectors_dev, model, dev),
            cached_customer_vectors=batch["stage1_customer_vectors"].to(dev),
        ).logits

    best_disc = (-1, -1.0, -1.0)
    best_ndcg = (-1.0, -1.0, -1)
    history = []
    sync(dev)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(batch)
            loss = loss_fn(logits, batch["labels"].to(dev))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.cat([_forward(b).cpu() for b in make_loader(slates["val"])])
        metrics = ranking_metrics(val_scores, campaign_data["val"], slates["val"]["candidate_indices"])
        disc_key = (
            int(metrics["recall@10"] >= recall_guard and metrics["ndcg@10"] >= ndcg_guard),
            metrics["new_hits@10"],
            metrics["ndcg@10"],
        )
        ndcg_key = (metrics["ndcg@10"], metrics["recall@10"], metrics["new_hits@10"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics})

        if disc_key > best_disc:
            best_disc = disc_key
            atomic_torch_save(
                {"settings": settings, "epoch": epoch, "metrics": metrics, "model_state_dict": model.state_dict()},
                directory / "best.pt",
            )
        if ndcg_key > best_ndcg:
            best_ndcg = ndcg_key
            atomic_torch_save(
                {"settings": settings, "epoch": epoch, "metrics": metrics, "model_state_dict": model.state_dict()},
                directory / "best_ndcg.pt",
            )
        print(
            f"v3 [{run_name}] epoch={epoch:02d} new={metrics['new_hits@10']} "
            f"rec={metrics['recall@10']:.5f} ndcg={metrics['ndcg@10']:.5f}",
            flush=True,
        )
    sync(dev)
    elapsed = time.perf_counter() - started

    def _replay(ckpt_file: Path, pred_files: Sequence[Path]) -> Dict[str, Any]:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            scores = torch.cat([_forward(b).cpu() for b in make_loader(slates["val"])])
        first_pred = pred_files[0]
        replayed = ranking_metrics(scores, campaign_data["val"], slates["val"]["candidate_indices"], first_pred)
        if replayed != ckpt["metrics"]:
            raise AssertionError(f"Replay mismatch for {ckpt_file}")
        for extra_pred in pred_files[1:]:
            ranking_metrics(scores, campaign_data["val"], slates["val"]["candidate_indices"], extra_pred)
        return {"epoch": ckpt["epoch"], "metrics": replayed, "checkpoint": str(ckpt_file), "predictions": str(first_pred)}

    disc_summary = _replay(directory / "best.pt", [directory / "predictions.npz", directory / "predictions_discovery.npz"])
    ndcg_summary = _replay(directory / "best_ndcg.pt", [directory / "predictions_ndcg.npz"])

    result = {
        "settings": settings,
        "epoch": disc_summary["epoch"],
        "metrics": disc_summary["metrics"],
        "qualified": bool(best_disc[0]),
        "discovery_selection": disc_summary,
        "ndcg_selection": ndcg_summary,
        "slate_coverage": val_slate_cov,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": elapsed,
        "selected_checkpoint": str(directory / "best.pt"),
        "ndcg_checkpoint": str(directory / "best_ndcg.pt"),
    }
    atomic_json(history, directory / "history.json", overwrite=False)
    finish_run(directory, settings, result)
    return result


def _timed_ms(operation, dev: torch.device, *, warmup: int = 10, repeats: int = 50) -> Dict[str, float]:
    for _ in range(warmup):
        operation()
    sync(dev)
    samples = []
    for _ in range(repeats):
        sync(dev)
        t0 = time.perf_counter()
        operation()
        sync(dev)
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def measure_v3_warm_latency(
    spec: VariantSpec,
    stage1_dir: Path,
    ranker_results: Mapping[str, Mapping[str, Any]],
    selector_results: Mapping[str, Mapping[str, Any]],
    val_retrieval: Mapping[str, Any],
    info: Mapping[str, Any],
    val_source: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    *,
    repeats: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Measure CPU and MPS warm single-request latency using cached Stage 1 vectors."""
    _, _, feat_mmap_val = load_dynamic_campaign_features_readonly()
    stage1_ckpt = torch.load(Path(stage1_dir) / "best.pt", map_location="cpu", weights_only=False)
    val_cache = torch.load(Path(stage1_dir) / "val_purchase_cache.pt", map_location="cpu", weights_only=False)

    stage1_cpu = make_variant_model(spec, info, enrichment)
    stage1_cpu.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
    stage1_cpu.eval()

    row_idx = 0
    u_cpu = val_cache["customer_vectors"][row_idx : row_idx + 1].squeeze(1).contiguous()  # (1, 128)
    v_cpu = val_cache["product_vectors"].contiguous()  # (P, 128)
    stats_cpu = quantize_transferred_stats(val_source["retriever_tabular_features"][row_idx : row_idx + 1])
    f41_cpu = torch.from_numpy(np.array(feat_mmap_val[row_idx : row_idx + 1], dtype=np.float32))
    allowed_cpu = val_retrieval["allowed"][row_idx : row_idx + 1]
    unseen_cpu = val_retrieval["unseen"][row_idx : row_idx + 1]
    fam_cpu = val_retrieval["familiarity"][row_idx : row_idx + 1]
    all_prod_idx_cpu = torch.arange(len(info["product_ids"]))

    dummy_slate = make_slates(
        {
            "u": u_cpu,
            "v": v_cpu,
            "allowed": allowed_cpu,
            "unseen": unseen_cpu,
            "target": val_retrieval["target"][row_idx : row_idx + 1],
            "features": feat_mmap_val,
            "source": sliced(val_source, torch.tensor([row_idx]), len(val_source["customer_ids"])),
        },
        val_cache["scores"][row_idx : row_idx + 1],
        info["product_ids"],
        "quota100_50",
    )
    ranker_batch_cpu = next(iter(make_loader(dummy_slate, batch_size=1, shuffle=False)))

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    report: Dict[str, Dict[str, Any]] = {}
    for dev in devices:
        dev_key = str(dev)
        report[dev_key] = {}

        u_d = u_cpu.to(dev)
        v_d = v_cpu.to(dev)
        stats_d = stats_cpu.to(dev)
        f41_d = f41_cpu.to(dev)
        allowed_d = allowed_cpu.to(dev)
        unseen_d = unseen_cpu.to(dev)
        fam_d = fam_cpu.to(dev)
        idx_d = all_prod_idx_cpu.to(dev)
        stage1_d = make_variant_model(spec, info, enrichment).to(dev)
        stage1_d.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
        stage1_d.eval()

        for label, sel_recipe, ranker_key in (
            ("U0-dot100_41", "dot", "U0-dot100_41"),
            ("U0-mlp303_100_41", "uv_47", "U0-mlp303_100_41"),
            ("U0-mlp304_100_41", "uv_47_ref", "U0-mlp304_100_41"),
        ):
            ranker = _make_downstream_ranker(
                "dynamic_features",
                seed=42,
                stage1_checkpoint_path=Path(stage1_dir) / "best.pt",
                is_control_stage1=spec.is_control,
            ).to(dev)
            r_ckpt = torch.load(ranker_results[ranker_key]["selected_checkpoint"], map_location="cpu", weights_only=False)
            ranker.load_state_dict(r_ckpt["model_state_dict"], strict=True)
            ranker.eval()

            selector = None
            if sel_recipe != "dot":
                sel_ckpt = torch.load(selector_results[sel_recipe]["selected_checkpoint"], map_location="cpu", weights_only=False)
                selector = CompactStage1Selector(recipe=sel_recipe).to(dev)
                selector.load_state_dict(sel_ckpt["model_state_dict"], strict=True)
                selector.eval()

            base_inputs = ranker_batch_inputs(ranker_batch_cpu, v_d, ranker, dev)

            with torch.no_grad():
                def _score_stage1():
                    if sel_recipe == "dot":
                        return stage1_d.score_cached(u_d, v_d, idx_d, fam_d)
                    ref = stage1_d.score_cached(u_d, v_d, idx_d, fam_d) if sel_recipe == "uv_47_ref" else None
                    return selector(
                        u=u_d,
                        v=v_d,
                        features_41=f41_d,
                        customer_stats_6=stats_d,
                        reference_scores=ref,
                    )

                sample_scores = _score_stage1()
                def _select():
                    return select_candidates(sample_scores, allowed_d, unseen_d, policy="quota100_50")

                sample_cand = _select()
                def _run_ranker():
                    cand_pf = f41_d[0, sample_cand[0]].unsqueeze(0)
                    cand_vec = v_d[sample_cand[0]].unsqueeze(0)
                    return ranker(
                        **{
                            **base_inputs,
                            "candidate_vectors": cand_vec,
                            "pairwise_features": cand_pf,
                            "cached_customer_vectors": u_d,
                        }
                    ).logits

                def _e2e():
                    sc = _score_stage1()
                    cand = select_candidates(sc, allowed_d, unseen_d, policy="quota100_50")
                    cand_pf = f41_d[0, cand[0]].unsqueeze(0)
                    cand_vec = v_d[cand[0]].unsqueeze(0)
                    return ranker(
                        **{
                            **base_inputs,
                            "candidate_vectors": cand_vec,
                            "pairwise_features": cand_pf,
                            "cached_customer_vectors": u_d,
                        }
                    ).logits

                sel_t = _timed_ms(_score_stage1, dev, repeats=repeats)
                pick_t = _timed_ms(_select, dev, repeats=repeats)
                rnk_t = _timed_ms(_run_ranker, dev, repeats=repeats)
                e2e_t = _timed_ms(_e2e, dev, repeats=repeats)

            report[dev_key][label] = {
                "end_to_end_median_ms": e2e_t["median_ms"],
                "end_to_end_p95_ms": e2e_t["p95_ms"],
                "selector_median_ms": sel_t["median_ms"],
                "selection_median_ms": pick_t["median_ms"],
                "ranker_median_ms": rnk_t["median_ms"],
            }
    return report

