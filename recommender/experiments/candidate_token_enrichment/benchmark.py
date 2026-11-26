"""Real-input Stage 2 CPU & MPS latency benchmark (batch=1 and batch=64) for Arms A, B, and C."""

import copy
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from recommender.artifacts import atomic_json
from recommender.experiment_signals import sync
from recommender.experiments.rolling_candidate_v2.benchmark import _hardware_metadata
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader,
    rolling_ranker_batch_inputs,
)

from .config import EVIDENCE_ROOT, SLATE_SIZE
from .model import CandidateTokenEnrichmentRanker


def _time_callable(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int = 5,
    repeats: int = 30,
    items: int = 1,
) -> Dict[str, Any]:
    for _ in range(max(int(warmup), 0)):
        fn()
    sync(device)

    samples_ms: List[float] = []
    for _ in range(int(repeats)):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(samples_ms, dtype=np.float64)
    med_ms = float(np.median(arr))
    return {
        "items": int(items),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "median_ms": med_ms,
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "per_item_us": float((med_ms * 1000.0) / max(int(items), 1)),
    }


@torch.no_grad()
def run_stage2_latency_benchmark(
    models: Mapping[str, CandidateTokenEnrichmentRanker],
    val_slate: Mapping[str, Any],
    product_vectors: torch.Tensor,
    *,
    evidence_root: Optional[Path] = EVIDENCE_ROOT,
    warmup: int = 5,
    repeats: int = 30,
    save: bool = True,
) -> Dict[str, Any]:
    """Benchmark real validation-slate Stage 2 request latency on CPU and MPS (batch=1 and 64).

    Explicit scope:
    - `prepared_on_device_forward`: inputs and product_vectors already materialized on target device.
    - `host_to_device_transfer_plus_forward`: CPU batch dict transferred to target device + forward.
    - Excludes cold disk loading, feature extraction, and Stage 1 candidate retrieval.
    """
    batch_1 = next(iter(make_rolling_slate_loader(val_slate, batch_size=1, shuffle=False)))
    batch_64 = next(iter(make_rolling_slate_loader(val_slate, batch_size=64, shuffle=False)))
    pv_cpu = product_vectors.detach().cpu()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results: Dict[str, Any] = {
        "hardware": _hardware_metadata(),
        "slate_size": int(SLATE_SIZE),
        "input_source": "real_val_slates_v3_rows_0_and_0_64",
        "scope_notes": {
            "prepared_on_device_forward": "Pre-materialized tensors on target device; excludes H2D copy and disk I/O.",
            "host_to_device_transfer_plus_forward": "Includes rolling_ranker_batch_inputs CPU->device copy + model.forward.",
        },
        "devices": {},
    }

    for dev_name in devices:
        dev = torch.device(dev_name)
        pv_dev = pv_cpu.to(dev)
        in_1_dev = rolling_ranker_batch_inputs(batch_1, pv_dev, dev)
        in_64_dev = rolling_ranker_batch_inputs(batch_64, pv_dev, dev)

        dev_res: Dict[str, Any] = {}
        for arm_name, src_model in models.items():
            m = copy.deepcopy(src_model).to(dev)
            m.eval()
            dev_res[arm_name] = {
                "parameters": int(sum(p.numel() for p in m.parameters())),
                "batch_1": {
                    "prepared_on_device_forward": _time_callable(
                        lambda m=m, inp=in_1_dev: m(**inp),
                        dev,
                        warmup=warmup,
                        repeats=repeats,
                        items=1,
                    ),
                    "host_to_device_transfer_plus_forward": _time_callable(
                        lambda m=m, b=batch_1, pv=pv_dev, d=dev: m(**rolling_ranker_batch_inputs(b, pv, d)),
                        dev,
                        warmup=warmup,
                        repeats=repeats,
                        items=1,
                    ),
                },
                "batch_64": {
                    "prepared_on_device_forward": _time_callable(
                        lambda m=m, inp=in_64_dev: m(**inp),
                        dev,
                        warmup=warmup,
                        repeats=repeats,
                        items=64,
                    ),
                    "host_to_device_transfer_plus_forward": _time_callable(
                        lambda m=m, b=batch_64, pv=pv_dev, d=dev: m(**rolling_ranker_batch_inputs(b, pv, d)),
                        dev,
                        warmup=warmup,
                        repeats=repeats,
                        items=64,
                    ),
                },
            }
        results["devices"][dev_name] = dev_res

    if save and evidence_root is not None:
        out_dir = Path(evidence_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(results, out_dir / "benchmark.json", overwrite=True)
    return results
