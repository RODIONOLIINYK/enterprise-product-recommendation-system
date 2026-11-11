"""Controlled, development-only experiments; never overwrite reference artifacts.

PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R0 R1 R2 R3 R4
"""

import argparse
import json
import math
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.shared_data import (
    DEFAULT_CACHE,
    DEFAULT_RETRIEVER,
    atomic_torch_save,
    prepare_shared_caches,
    validate_shared_cache,
)
from recommender.shared_ranker import SharedRankerConfig, SharedStage1Ranker
from recommender.train_ranker import (
    RankerDataset,
    collate_ranker_batch,
    evaluate_ranker,
)
from recommender.train_retriever import get_device

DEFAULT_OUTPUT = Path("checkpoints/shared-stage1-experiments")
RUN_CONFIGS = {
    "R0W": SharedRankerConfig(representation="legacy_width"),
    "R1": SharedRankerConfig(),
    "R2": SharedRankerConfig(memory_statistics=True),
    "R3": SharedRankerConfig(memory_statistics=True, query_statistics=True),
    "R4": SharedRankerConfig(memory_statistics=True, query_statistics=True, bias_schema="reduced16"),
    "R3P0": SharedRankerConfig(memory_statistics=True, query_statistics=True, profile_token=False),
    "R3H0": SharedRankerConfig(memory_statistics=True, query_statistics=True, history_tokens=False),
    "R5": SharedRankerConfig(width=192, memory_statistics=True, query_statistics=True),
    "R6": SharedRankerConfig(width=192, memory_statistics=True, query_statistics=True, learned_relationships=True),
}


def make_loader(data, batch_size=64, shuffle=False, seed=42):
    return DataLoader(RankerDataset(data), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_ranker_batch, generator=torch.Generator().manual_seed(seed))


def match_width_control_initialization(model, product_vectors, seed):
    """Match the 176D attention/readout/scorer initial state to R1.

    Legacy input construction consumes different CPU random draws. Isolate that
    difference from the common modules without altering the MPS dropout stream.
    """
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        shared = SharedStage1Ranker(product_vectors, RUN_CONFIGS["R1"])
    for name in ("memory_encoder", "cross_attention", "context_readout", "mlp", "statistical_bias"):
        getattr(model, name).load_state_dict(getattr(shared, name).state_dict())


def batch_inputs(batch, product_vectors, model, device):
    names = ("text_embedding", "city_id", "region_id", "industry_id", "tabular_features", "sequence",
             "attention_mask", "seq_product_ids", "seq_category_ids", "seq_subcategory_ids",
             "seq_business_line_ids", "pairwise_features", "seq_purchase_dates", "prediction_day")
    inputs = {name: batch[name].to(device) for name in names}
    indices = batch["candidate_indices"].to(device)
    inputs["candidate_vectors"] = product_vectors[indices]
    if getattr(model, "requires_shared_stage1_inputs", False):
        inputs.update(stage1_customer_vectors=batch["stage1_customer_vectors"].to(device),
                      seq_catalog_indices=batch["seq_catalog_indices"].to(device), candidate_indices=indices)
    return inputs


def diagnostics(model, data, product_vectors, device, output_path=None):
    """Same-checkpoint per-basket evidence, conditional recall and attention summaries."""
    model.eval()
    new_count = new_present = new_hits = target_count = target_present = 0
    per_row, logits_rows, profile_rows, entropy_rows = [], [], [], []
    with torch.no_grad():
        for batch in make_loader(data):
            output = model(**batch_inputs(batch, product_vectors, model, device))
            logits = output.logits.cpu().numpy()
            logits_rows.append(logits)
            profile_rows.extend(output.token_0_weight.mean(1).cpu().tolist())
            entropy = -(output.attn_weights * output.attn_weights.clamp_min(1e-12).log()).sum(-1).mean(1)
            entropy_rows.extend(entropy.cpu().tolist())
            for row, scores in enumerate(logits):
                targets = set(batch["positive_pids"][row])
                new = set(batch["new_positive_pids"][row])
                candidates = batch["candidate_pids"][row]
                ranked = [candidates[i] for i in np.argsort(-scores)]
                hits = len(targets.intersection(ranked[:10]))
                novel_hits = len(new.intersection(ranked[:10]))
                novel_present = len(new.intersection(candidates))
                new_count += len(new)
                new_present += novel_present
                new_hits += novel_hits
                target_count += len(targets)
                target_present += len(targets.intersection(candidates))
                dcg = sum(1 / math.log2(rank + 2) for rank, sku in enumerate(ranked[:10]) if sku in targets)
                ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(targets))))
                per_row.append({
                    "customer": batch["customer_ids"][row], "history_events": int(batch["attention_mask"][row].sum()),
                    "recall@10": hits / len(targets), "ndcg@10": dcg / ideal,
                    "new_hits@10": novel_hits, "new_targets": len(new), "new_present": novel_present,
                    "candidate_coverage": len(targets.intersection(candidates)) / len(targets),
                })
    groups = {}
    for name, low, high in (("0", 0, 0), ("1-8", 1, 8), ("9-16", 9, 16), ("17-31", 17, 31), ("32+_truncated", 32, 32)):
        rows = [row for row in per_row if low <= row["history_events"] <= high]
        count = sum(row["new_targets"] for row in rows)
        groups[name] = {"baskets": len(rows), "recall@10": float(np.mean([row["recall@10"] for row in rows])) if rows else None,
                        "new_recall@10": sum(row["new_hits@10"] for row in rows) / count if count else None}
    if output_path is not None:
        np.savez_compressed(output_path, logits=np.concatenate(logits_rows),
                            customers=np.array([row["customer"] for row in per_row]),
                            recall=np.array([row["recall@10"] for row in per_row]),
                            ndcg=np.array([row["ndcg@10"] for row in per_row]),
                            new_hits=np.array([row["new_hits@10"] for row in per_row]),
                            new_targets=np.array([row["new_targets"] for row in per_row]))
    return {
        "new_targets": new_count, "new_targets_in_slate": new_present, "new_hits@10": new_hits,
        "new_candidate_coverage": new_present / max(new_count, 1),
        "conditional_new_recall@10": new_hits / max(new_present, 1),
        "target_candidate_coverage_micro": target_present / max(target_count, 1),
        "candidate_coverage_macro": float(np.mean([row["candidate_coverage"] for row in per_row])),
        "profile_attention": float(np.mean(profile_rows)), "history_attention": 1 - float(np.mean(profile_rows)),
        "mean_attention_entropy_nats_averaged_heads": float(np.mean(entropy_rows)),
        "attention_quantile_levels": [0., .1, .5, .9, 1.],
        "profile_attention_quantiles_per_basket": np.quantile(profile_rows, [0., .1, .5, .9, 1.]).tolist(),
        "entropy_quantiles_per_basket_nats": np.quantile(entropy_rows, [0., .1, .5, .9, 1.]).tolist(),
        "profile_attention_histogram_10_bins": np.histogram(profile_rows, bins=np.linspace(0, 1, 11))[0].tolist(),
        "by_observed_history_events": groups,
    }


def parameter_utilization(model, data, vectors, device):
    model.train()
    model.zero_grad(set_to_none=True)
    batch = next(iter(make_loader(data)))
    output = model(**batch_inputs(batch, vectors, model, device))
    loss = RankerLoss()(output.logits, batch["labels"].to(device))["loss"]
    loss.backward()
    result = [{"name": name, "count": parameter.numel(), "grad_none": parameter.grad is None,
               "grad_norm": float(parameter.grad.norm()) if parameter.grad is not None else None,
               "zero_gradient_entries": int((parameter.grad == 0).sum()) if parameter.grad is not None else None}
              for name, parameter in model.named_parameters()]
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def evaluate_checkpoint(path, data, manifest, device):
    """One reconstruction path for validation and later permitted inference."""
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("settings", {}).get("cache_identity", manifest["identity"]) != manifest["identity"]:
        raise ValueError("Experimental baseline checkpoint cache identity mismatch.")
    if "model_metadata" in checkpoint:
        model = SharedStage1Ranker.from_checkpoint(
            checkpoint, manifest["product_vectors"], manifest["product_ids"],
            manifest["stage1_digest"], manifest["identity"],
        )
    else:
        model = UPSCANRRanker.from_retriever_checkpoint(
            DEFAULT_RETRIEVER, statistics_attention=True, excluded_statistics=("history_repeat",),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if checkpoint.get("retriever_digest", manifest["stage1_digest"]) != manifest["stage1_digest"]:
            raise ValueError("Baseline checkpoint retriever digest mismatch.")
    model.to(device).eval()
    vectors = manifest["product_vectors"].to(device)
    metrics = evaluate_ranker(model, make_loader(data), vectors, device, RankerLoss().to(device))
    return model, metrics


def train_experiment(run, seed, train_data, val_data, manifest, output_dir, device, epochs=10):
    run_dir = Path(output_dir) / f"{run}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    settings = {"run": run, "seed": seed, "epochs": epochs, "batch_size": 64,
                "lr": .001, "weight_decay": .0001, "pos_weight": 10., "listwise_weight": 1.,
                "new_positive_weight": 1., "scheduler": "cosine eta_min=1e-5",
                "selection": "new_recall@10; strict improvements; same fixed recall guardrail",
                "cache_identity": manifest["identity"], "stage1_digest": manifest["stage1_digest"], "device": str(device)}
    saved_baseline = torch.load("checkpoints/ranker/best_discovery_ranker.pt", map_location="cpu")
    guard = saved_baseline.get("discovery_min_overall_recall")
    settings["recall_guardrail"] = guard
    if run == "R0W":
        settings["initialization_policy"] = "common 176D attention/readout/scorer matched to R1 at same seed"
    if run != "R0":
        config = RUN_CONFIGS[run] if run != "R7" else replace(
            RUN_CONFIGS["R6"], width=224,
            categorical_sizes=tuple(max(manifest["vocabs"][name].values(), default=0) + 1
                                    for name in ("city_vocab", "region_vocab", "industry_vocab", "category_vocab", "subcategory_vocab", "business_line_vocab")),
        )
        settings["model_config"] = asdict(config)
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if "model_config" in existing["settings"]:
            # Newly optional scale overrides default to the original common scale.
            existing["settings"]["model_config"] = asdict(SharedRankerConfig(**existing["settings"]["model_config"]))
        if existing["settings"] != json.loads(json.dumps(settings)):
            raise ValueError(f"Refusing to reuse incompatible completed experiment {run_dir}.")
        print(f"Reusing completed {run}-seed{seed}", flush=True)
        return existing
    torch.manual_seed(seed)
    np.random.seed(seed)
    if run == "R0":
        model = UPSCANRRanker.from_retriever_checkpoint(
            DEFAULT_RETRIEVER, statistics_attention=True, excluded_statistics=("history_repeat",),
        )
    else:
        legacy_model = (UPSCANRRanker.from_retriever_checkpoint(
            DEFAULT_RETRIEVER, statistics_attention=True, excluded_statistics=("history_repeat",),
        ) if run == "R0W" else None)
        model = SharedStage1Ranker(manifest["product_vectors"], config, manifest["product_metadata_ids"], legacy_model)
        if run == "R0W":
            match_width_control_initialization(model, manifest["product_vectors"], seed)
    model.to(device)
    vectors = manifest["product_vectors"].to(device)
    loader = make_loader(train_data, shuffle=True, seed=seed)
    val_loader = make_loader(val_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = RankerLoss().to(device)
    history, best_new, best_qualified, best_ndcg = [], -1, -1, -1.
    peak_mps = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch_inputs(batch, vectors, model, device))
            loss = loss_fn(output.logits, batch["labels"].to(device))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if device.type == "mps":
                peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
        scheduler.step()
        metrics = evaluate_ranker(model, val_loader, vectors, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics,
                        "seconds": time.perf_counter() - epoch_start})
        print(f"{run} seed={seed} epoch={epoch}/{epochs} train={np.mean(losses):.4f} "
              f"rec10={metrics['val_recall@10']:.6f} ndcg={metrics['val_ndcg@10']:.6f} "
              f"new={metrics['new_hits@10']}/{metrics['new_target_count']} "
              f"seconds={history[-1]['seconds']:.1f}", flush=True)
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": metrics,
                   "settings": settings, "retriever_digest": manifest["stage1_digest"]}
        if run != "R0":
            payload["model_metadata"] = model.metadata(manifest["stage1_digest"], manifest["product_ids"], manifest["identity"])
        if metrics["new_hits@10"] > best_new:
            best_new = metrics["new_hits@10"]
            atomic_torch_save(payload, run_dir / "best_unconstrained.pt")
        if (guard is None or metrics["val_recall@10"] >= guard) and metrics["new_hits@10"] > best_qualified:
            best_qualified = metrics["new_hits@10"]
            atomic_torch_save(payload, run_dir / "best_discovery.pt")
        if metrics["val_ndcg@10"] > best_ndcg:
            best_ndcg = metrics["val_ndcg@10"]
            atomic_torch_save(payload, run_dir / "best_ndcg.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    elapsed = time.perf_counter() - started
    selected_path = run_dir / ("best_discovery.pt" if best_qualified >= 0 else "best_unconstrained.pt")
    del optimizer, model
    selected, metrics = evaluate_checkpoint(selected_path, val_data, manifest, device)
    diag = diagnostics(selected, val_data, vectors, device, run_dir / "predictions.npz")
    ablations = {}
    if run != "R0":
        original_config = selected.config
        # These are inference perturbations, not newly trained ablation models.
        for name, changes in {
            "no_customer_token_information": {"profile_token": False},
            "no_history_tokens": {"history_tokens": False},
            "no_numerical_memory": {"memory_statistics": False},
            "no_numerical_queries": {"query_statistics": False},
        }.items():
            changed = replace(original_config, **changes)
            if changed == original_config:
                continue
            selected.config = changed
            selected.token_builder.config = changed
            ablations[name] = evaluate_ranker(selected, val_loader, vectors, device, loss_fn)
        selected.config = selected.token_builder.config = original_config
    result = {"settings": settings, "selected_checkpoint": str(selected_path),
              "selected_epoch": torch.load(selected_path, map_location="cpu")["epoch"],
              "qualified_recall_guardrail": best_qualified >= 0, "metrics": metrics, "diagnostics": diag,
              "parameter_count": sum(p.numel() for p in selected.parameters()),
              "training_seconds": elapsed, "peak_mps_driver_bytes_sampled": peak_mps,
              "process_peak_rss_bytes_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_ndcg_other_checkpoint": best_ndcg, "inference_perturbations": ablations,
              "parameter_utilization": parameter_utilization(selected, train_data, vectors, device)}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def paired_customer_bootstrap(left_path, right_path, seed=2026, repeats=2000):
    left, right = np.load(left_path), np.load(right_path)
    if not np.array_equal(left["customers"], right["customers"]):
        raise ValueError("Paired comparisons require identical validation basket order.")
    customers, inverse = np.unique(left["customers"], return_inverse=True)
    count = len(customers)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ("recall", "ndcg", "new_hits"):
        delta = right[metric] - left[metric]
        per_customer = np.bincount(inverse, weights=delta)
        denominators = np.bincount(inverse, weights=left["new_targets"]) if metric == "new_hits" else sizes
        values = []
        for _ in range(repeats):
            sample = rng.integers(count, size=count)
            values.append(per_customer[sample].sum() / max(denominators[sample].sum(), 1))
        result[metric] = {"delta": float(delta.sum() / max(denominators.sum(), 1)),
                          "customer_bootstrap_95_interval": np.quantile(values, [.025, .975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["R0", "R0W", "R1", "R2", "R3", "R4"], choices=["R0", *RUN_CONFIGS, "R7"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = get_device(args.device)
    train, val, manifest = prepare_shared_caches(device, cache_dir=args.cache_dir)
    for split, data in (("train", train), ("val", val)):
        validate_shared_cache(data, f"{manifest['identity']}:{split}", manifest["product_ids"], manifest["stage1_digest"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for name in ("best_ranker", "best_discovery_ranker"):
        path = Path("checkpoints/ranker") / f"{name}.pt"
        model, metrics = evaluate_checkpoint(path, val, manifest, device)
        baselines[name] = {"path": str(path), "epoch": torch.load(path, map_location="cpu")["epoch"],
                           "reproduced_metrics": metrics,
                           "diagnostics": diagnostics(model, val, manifest["product_vectors"].to(device), device,
                                                      args.output_dir / f"reference-{name}.npz")}
    (args.output_dir / "baseline_replay.json").write_text(json.dumps(baselines, indent=2) + "\n")
    for run in args.runs:
        train_experiment(run, args.seed, train, val, manifest, args.output_dir, device, args.epochs)


if __name__ == "__main__":
    main()
