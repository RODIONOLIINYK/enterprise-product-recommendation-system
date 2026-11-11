"""E1/E2 screening, paired repeated-seed fits and isolated evidence.

PYTHONPATH=. ./venv/bin/python -m recommender.experiment_signals --seeds 42
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from recommender.experimental_signals import SignalRanker, transition_features
from recommender.ranker import UPSCANRRanker, RankerLoss
from recommender.shared_data import DEFAULT_CACHE, DEFAULT_RETRIEVER, atomic_torch_save, load_development_purchases, validate_shared_cache
from recommender.train_ranker import checkpoint_digest, evaluate_ranker
from recommender.train_shared_ranker import make_loader, batch_inputs, diagnostics, paired_customer_bootstrap
from recommender.train_retriever import get_device

OUTPUT = Path("checkpoints/e1-e5/signals-v1")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def latency(model, data, vectors, device):
    result = {}
    model.eval()
    with torch.no_grad():
        for size in (1, 64):
            batch = next(iter(make_loader(data, batch_size=size)))
            inputs = batch_inputs(batch, vectors, model, device)
            for _ in range(5):
                model(**inputs)
            sync(device)
            measurements = []
            for _ in range(30):
                start = time.perf_counter()
                model(**inputs)
                sync(device)
                measurements.append((time.perf_counter() - start) * 1000)
            result[str(size)] = {"median_ms": float(np.median(measurements)), "p95_ms": float(np.quantile(measurements, .95))}
    return result


def load_inputs():
    manifest = torch.load(DEFAULT_CACHE / "manifest.pt", map_location="cpu")
    if manifest["stage1_digest"] != checkpoint_digest(DEFAULT_RETRIEVER):
        raise ValueError("Frozen retriever provenance mismatch")
    data = [torch.load(DEFAULT_CACHE / f"{split}.pt", map_location="cpu") for split in ("train", "val")]
    for split, value in zip(("train", "val"), data):
        validate_shared_cache(value, f"{manifest['identity']}:{split}", manifest["product_ids"], manifest["stage1_digest"])
    return *data, manifest


def make_model(mode):
    base = UPSCANRRanker.from_retriever_checkpoint(DEFAULT_RETRIEVER, statistics_attention=True, excluded_statistics=("history_repeat",))
    return SignalRanker(base, mode)


def fit(mode, seed, train, val, manifest, output=OUTPUT, epochs=10, device=None):
    device = get_device(device)
    run_dir = Path(output) / f"{mode}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {"mode": mode, "seed": seed, "epochs": epochs, "cache_identity": manifest["identity"],
                "stage1_digest": manifest["stage1_digest"], "alpha": 20., "version": 1,
                "batch_size": 64, "lr": .001, "weight_decay": .0001, "pos_weight": 10., "listwise_weight": 1.}
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["settings"] != settings:
            raise ValueError("Existing experiment settings differ")
        return result
    if list(run_dir.glob("*.pt")):
        raise ValueError(f"Incomplete run in {run_dir}; use a new output directory")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(mode)
    if manifest.get("clean_stage1_state") is not None:
        from recommender.experimental_retrieval import reset_reference
        state = manifest["clean_stage1_state"]
        # Reset all Stage 2 learned branches before transferring the clean retriever.
        def reset(module):
            if isinstance(module, torch.nn.MultiheadAttention):
                module._reset_parameters()
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()
        model.apply(reset)
        with torch.no_grad():
            for name in ("text_proj.0.weight", "text_proj.0.bias", "city_embedding.weight", "region_embedding.weight", "industry_embedding.weight"):
                dict(model.base.named_parameters())[name].copy_(state["towers.customer_tower." + name])
            from recommender.features import LEGACY_PROFILE_FEATURES
            original_config = torch.load(DEFAULT_RETRIEVER, map_location="cpu")["config"]
            target = model.base.tabular_proj[0].weight
            source = state["towers.customer_tower.tabular_proj.0.weight"]
            target[:, :48].copy_(source[:, :48])
            for i, name in enumerate(original_config["profile_features"]):
                if name in LEGACY_PROFILE_FEATURES:
                    target[:, 48 + LEGACY_PROFILE_FEATURES.index(name)].copy_(source[:, 48 + i])
            model.base.tabular_proj[0].bias.copy_(state["towers.customer_tower.tabular_proj.0.bias"])
    model.to(device)
    vectors = manifest["product_vectors"].to(device)
    train_loader = make_loader(train, shuffle=True, seed=seed)
    val_loader = make_loader(val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = RankerLoss().to(device)
    guard = torch.load("checkpoints/ranker/best_discovery_ranker.pt", map_location="cpu").get("discovery_min_overall_recall")
    best, best_ndcg, unconstrained = -1, -1., -1
    history = []
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(**batch_inputs(batch, vectors, model, device))
            loss = loss_fn(output_batch.logits, batch["labels"].to(device))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate_ranker(model, val_loader, vectors, device, loss_fn)
        residual = float(model.residual_weight.detach()) if hasattr(model, "residual_weight") else None
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "metrics": metrics, "residual_weight": residual})
        payload = {"settings": settings, "model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics,
                   "recall_guardrail": guard}
        if metrics["new_hits@10"] > unconstrained:
            unconstrained = metrics["new_hits@10"]
            atomic_torch_save(payload, run_dir / "best_unconstrained.pt")
        if metrics["val_ndcg@10"] > best_ndcg:
            best_ndcg = metrics["val_ndcg@10"]
            atomic_torch_save(payload, run_dir / "best_ndcg.pt")
        if (guard is None or metrics["val_recall@10"] >= guard) and metrics["new_hits@10"] > best:
            best = metrics["new_hits@10"]
            atomic_torch_save(payload, run_dir / "best_discovery.pt")
        print(f"{mode} seed={seed} epoch={epoch} new={metrics['new_hits@10']} rec={metrics['val_recall@10']:.6f} ndcg={metrics['val_ndcg@10']:.6f} residual={residual}", flush=True)
    sync(device)
    elapsed = time.perf_counter() - start
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    selected_path = run_dir / ("best_discovery.pt" if best >= 0 else "best_unconstrained.pt")
    checkpoint = torch.load(selected_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    replay = evaluate_ranker(model, val_loader, vectors, device, loss_fn)
    assert replay == checkpoint["metrics"], "Selected checkpoint replay mismatch"
    diag = diagnostics(model, val, vectors, device, run_dir / "predictions.npz")
    result = {"settings": settings, "epoch": checkpoint["epoch"], "qualified_recall_guardrail": best >= 0, "selected_checkpoint": str(selected_path), "metrics": replay, "diagnostics": diag,
              "parameters": sum(p.numel() for p in model.parameters()), "training_seconds": elapsed,
              "latency": latency(model, val, vectors, device), "device": str(device),
              "residual_weight": float(model.residual_weight.detach()) if hasattr(model, "residual_weight") else None,
              "ablation_zero_signal": None}
    if mode != "control":
        ablated = dict(val)
        ablated["pairwise_features"] = val["pairwise_features"].clone()
        if mode == "unseen":
            with torch.no_grad():
                model.residual_weight.zero_()
        else:
            ablated["pairwise_features"][..., 13] = 0
        result["ablation_zero_signal"] = evaluate_ranker(model, make_loader(ablated), vectors, device, loss_fn)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--modes", nargs="+", default=["control", "unseen", "transition_residual", "transition_mlp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train, val, manifest = load_inputs()
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    features = transition_features(purchases, [train, val])
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            datasets = [dict(train), dict(val)]
            if mode.startswith("transition"):
                for data, feature in zip(datasets, features):
                    data["pairwise_features"] = torch.cat([data["pairwise_features"], feature.unsqueeze(-1)], -1)
            results.append(fit(mode, seed, *datasets, manifest, args.output, args.epochs, args.device))
    comparison = {}
    for seed in args.seeds:
        control = args.output / f"control-seed{seed}" / "predictions.npz"
        if control.exists():
            for mode in args.modes:
                if mode != "control":
                    comparison[f"{mode}-seed{seed}"] = paired_customer_bootstrap(control, args.output / f"{mode}-seed{seed}" / "predictions.npz")
    (args.output / "comparison.json").write_text(json.dumps({"results": results, "paired": comparison}, indent=2) + "\n")


if __name__ == "__main__":
    main()
