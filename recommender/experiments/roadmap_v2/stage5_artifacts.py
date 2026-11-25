"""Run isolation and ordered tensor identities for the formulation pipeline."""

import json
from pathlib import Path
from uuid import uuid4

import torch

from recommender.artifacts import json_digest
from recommender.shared_ranker import tensor_digest
from .config import EVIDENCE_ROOT


def stage5_run_paths(root, output_dir=None, summary_path=None):
    """Reserve a new output tree before feature preparation can write any files."""
    directory = Path(output_dir or (Path(root) / "runs" / uuid4().hex)).resolve()
    summary = Path(summary_path or directory / "summary.json").resolve()
    for name in ("stage5_reference.json", "stage5_3stage_reference.json"):
        pointer = EVIDENCE_ROOT / name
        artifacts = json.loads(pointer.read_text())["checkpoints"].values()
        for path in (pointer, *(Path(a["path"]) for a in artifacts)):
            pinned = path.resolve()
            if pinned == summary or pinned == directory or directory in pinned.parents:
                raise FileExistsError(f"Stage 5 output conflicts with pinned file {pinned}")
    if summary.exists() or summary == directory or summary in directory.parents:
        raise FileExistsError(f"Summary path unavailable: {summary}")
    directory.mkdir(parents=True, exist_ok=False)
    summary.parent.mkdir(parents=True, exist_ok=True)
    return directory, summary


def product_inputs_digest(inputs):
    return json_digest(
        {
            "text_embedding": tensor_digest(inputs["text_embedding"]),
            "demand_features": tensor_digest(inputs["demand_features"]),
            "attr_ids": {k: tensor_digest(v) for k, v in inputs["attr_ids"].items()},
        }
    )


def vector_identity(split, names, tower_hash, inputs_hash):
    return {
        "tower_checkpoint_sha256": tower_hash,
        "ordered_base_names": list(names),
        "ordered_keys": [
            (str(cid), int(day))
            for cid, day in zip(split["customer_ids"], split["prediction_days"].tolist(), strict=True)
        ],
        "product_inputs_digest": inputs_hash,
    }


def read_stage5_vectors(directory, *, allow_unsealed_legacy=False):
    """Read both exports together. A partial seal can never fall back to legacy mode."""
    identity_fields = {
        "tower_checkpoint_sha256",
        "ordered_base_names",
        "ordered_keys",
        "product_inputs_digest",
    }
    loaded = {}
    for split in ("train", "val"):
        loaded[split] = torch.load(
            Path(directory) / f"{split}_vectors.pt", map_location="cpu", weights_only=True, mmap=True
        )
    seals = [identity_fields.intersection(data) for data in loaded.values()]
    legacy = not any(seals)
    if legacy:
        if not allow_unsealed_legacy:
            raise ValueError(
                "Missing vector identity. Pre-seal pairs require --allow-unsealed-legacy-vectors."
            )
    elif any(fields != identity_fields for fields in seals):
        raise ValueError("Partial vector identity: train and validation exports must both be fully sealed.")
    return loaded, legacy


def validate_stage5_vectors(loaded, splits, names, tower_hash, inputs, *, legacy=False, d_model=128):
    inputs_hash = product_inputs_digest(inputs)
    for name, data in loaded.items():
        expected = vector_identity(splits[name], names, tower_hash, inputs_hash)
        if not legacy or expected.keys() & data.keys():
            for key, value in expected.items():
                if data.get(key) != value:
                    raise ValueError(f"{name} vector identity mismatched: {key}")
        for key, count in (("u_bn", len(expected["ordered_keys"])), ("v_bn", len(names))):
            tensor = data.get(key)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (count, d_model)
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid {name} {key} vectors")
    if not torch.equal(loaded["train"]["v_bn"], loaded["val"]["v_bn"]):
        raise ValueError("Train and validation catalogue vectors disagree")
    return loaded["train"]["v_bn"], loaded["train"]["u_bn"], loaded["val"]["u_bn"]
