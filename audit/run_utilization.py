"""Reproduce the audit without raw purchases, evaluation, or optimizer updates.

Run from the repository root:
    PYTHONPATH=. ./venv/bin/python audit/run_utilization.py

Reads fixed training-cache and checkpoint paths only. JSON outputs are qualitative
utilization evidence and artifact metadata, not freshly evaluated quality metrics.
"""

import hashlib
import json
from pathlib import Path

import torch

from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.split import TRAIN_CUTOFF
from recommender.train_ranker import RankerDataset, collate_ranker_batch

RETRIEVER = Path("checkpoints/retriever-discovery-reference/best_retriever.pt")
TRAIN_CACHE = Path("data/interim/ranker_train_slates_v3_repeat_0.5_rolling_v1.pt")
VAL_CACHE = Path("data/interim/ranker_val_slates_v3_repeat_0.5_rolling_v1.pt")
RANKER = Path("checkpoints/ranker/best_ranker.pt")
OUTPUT = Path(__file__).resolve().parent


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def artifact_inventory():
    paths = [RETRIEVER, Path("checkpoints/retriever/best_retriever.pt"), RANKER,
             RANKER.with_name("best_discovery_ranker.pt"), RANKER.with_name("best_ndcg_ranker.pt")]
    inventory = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        inventory[str(path)] = {
            "sha256": file_hash(path), "bytes": path.stat().st_size,
            "epoch": ckpt.get("epoch"), "config": ckpt.get("config"),
            "best_metric_name": ckpt.get("best_metric_name"),
            "best_metric_value": ckpt.get("best_metric_value"),
            "val_metrics": ckpt.get("val_metrics"),
            "retriever_digest": ckpt.get("retriever_digest"),
            "statistics_attention": ckpt.get("statistics_attention"),
            "excluded_statistics": ckpt.get("excluded_statistics"),
            "slate_schema_version": ckpt.get("slate_schema_version"),
            "catalogue_size": len(ckpt.get("product_ids", [])),
            "vocab_sizes": {key: len(value) for key, value in ckpt.get("vocabs", {}).items()},
        }
    for path in [TRAIN_CACHE, VAL_CACHE]:
        data = torch.load(path, map_location="cpu")
        inventory[str(path)] = {
            "sha256": file_hash(path), "bytes": path.stat().st_size,
            "cache_identity": data.get("cache_identity"),
            "tensors": {key: list(value.shape) for key, value in data.items() if isinstance(value, torch.Tensor)},
        }
    return inventory


def utilization():
    torch.manual_seed(42)
    model = UPSCANRRanker.from_retriever_checkpoint(
        RETRIEVER, statistics_attention=True, excluded_statistics=("history_repeat",),
    )
    model.load_state_dict(torch.load(RANKER, map_location="cpu")["model_state_dict"])
    dataset = RankerDataset(torch.load(TRAIN_CACHE, map_location="cpu"))
    batch = collate_ranker_batch([dataset[i] for i in range(64)])
    inputs = {key: batch[key] for key in (
        "text_embedding", "city_id", "region_id", "industry_id", "tabular_features",
        "sequence", "attention_mask", "seq_product_ids", "seq_category_ids",
        "seq_subcategory_ids", "seq_business_line_ids", "seq_purchase_dates", "pairwise_features",
    )}
    inputs["candidate_vectors"] = torch.nn.functional.normalize(torch.randn(64, 50, 128), dim=-1)
    inputs["prediction_day"] = torch.full((64,), float((TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 1))
    report = {
        "probe": "64 cached training rows; seeded synthetic normalized candidate vectors; synthetic prediction day 2025-09-30; no optimizer step",
        "seed": 42, "device": "cpu", "torch_version": str(torch.__version__), "runs": {},
    }
    for training in [True, False]:
        model.train(training)
        model.zero_grad(set_to_none=True)
        text_inputs = []
        hook = model.product_text_proj.register_forward_pre_hook(
            lambda _module, args, captured=text_inputs: captured.append(args[0].detach()),
        )
        output = model(**inputs)
        hook.remove()
        loss = RankerLoss()(output.logits, batch["labels"])["loss"]
        loss.backward()
        rows = []
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            rows.append({
                "name": name, "count": parameter.numel(),
                "participates_in_graph": grad is not None, "grad_none": grad is None,
                "grad_zero": bool(torch.count_nonzero(grad) == 0) if grad is not None else None,
                "grad_norm": float(grad.norm()) if grad is not None else None,
                "zero_gradient_elements": int((grad == 0).sum()) if grad is not None else None,
            })
        tabular = model.tabular_proj[0].weight.grad
        padding = torch.cat([torch.zeros(64, 1, dtype=torch.bool), batch["attention_mask"] == 0], 1)
        report["runs"]["train" if training else "eval_with_grad"] = {
            "loss": float(loss.detach()), "parameters": rows,
            "total_parameters": sum(row["count"] for row in rows),
            "parameters_without_gradient": sum(row["count"] for row in rows if row["grad_none"]),
            "parameters_in_all_zero_gradient_tensors": sum(row["count"] for row in rows if row["grad_zero"]),
            "historical_text_input_nonzero": int(torch.count_nonzero(text_inputs[0])),
            "tabular_numerical_zero": bool((tabular[:, 48:] == 0).all()),
            "tabular_categorical_norm": float(tabular[:, :48].norm()),
            "padded_attention_max": float(output.attn_weights.masked_select(padding[:, None]).abs().max().detach()),
        }
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        model(**inputs)
    report["eval_no_grad_all_none"] = all(parameter.grad is None for parameter in model.parameters())
    return report


if __name__ == "__main__":
    inventory = artifact_inventory()
    report = utilization()
    for filename, payload in [("artifacts.json", inventory), ("gradients.json", report)]:
        (OUTPUT / filename).write_text(json.dumps(payload, indent=2) + "\n")
    # Verify that even this read-only probe has not altered any audited binary.
    assert all(file_hash(Path(path)) == info["sha256"] for path, info in inventory.items())
    print("Saved artifact inventory and train/eval gradient evidence; binary hashes unchanged.")
