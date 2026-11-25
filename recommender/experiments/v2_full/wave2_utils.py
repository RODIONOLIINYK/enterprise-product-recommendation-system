"""Wave 2 building blocks for `v2_full` (strictly leak-free, zero held-out Test access):

1. `WeightedRankerLoss`: Exact per-basket weighted generalization of `RankerLoss` (BCE + listwise
   softmax CE) that reduces identically to `RankerLoss` when `sample_weight` is `None` or all-ones.
2. `compute_recency_sample_weights`: Strictly train-only exponential time-decay basket weights
   (`w_i = max(min_weight, 2^(-(T_cutoff - d_i) / half_life_days))`, mean-normalized to `1.0`).
3. `ensure_v2_sequences_for_length`: Builds and caches longer strictly point-in-time base-name
   purchase sequences (`seq_len = 64`, `96`, ...) directly from `cleaned_purchases.csv` (< TEST_START_V2)
   and `items.csv` without rebuilding the 12 GB `cand_38.npy` memmap.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from recommender.artifacts import json_digest
from recommender.catalogue import RollingCandidatePool
from recommender.experiments.roadmap_v2.dated_baskets import validate_dated_split
from recommender.experiments.roadmap_v2.native_base_name import observed_pack_ids
from recommender.experiments.roadmap_v2.stage5_base_entity import build_base_name_sequences
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .config import PREPARED_ROOT, TEST_START_V2, TRAIN_CUTOFF_V2

SEQUENCE_BUILDER_VERSION = "v2_seq_builder_v1"


def is_tpu_device(dev: torch.device) -> bool:
    """Return True when `dev` targets a TPU backend (`xla` via torch_xla or native `tpu`)."""
    return dev.type in ("xla", "tpu")


def supports_native_bf16(dev: torch.device) -> bool:
    """Return True only when hardware supports native bfloat16 (no software emulation)."""
    if dev.type == "cuda":
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    if is_tpu_device(dev):
        return True
    return False


def _try_get_tpu_device(preferred: Optional[str] = None) -> Optional[torch.device]:
    """Resolve an active Colab/Cloud TPU device via `torch_xla` or native `torch_tpu`."""
    if preferred != "xla":
        try:
            import torch_tpu  # noqa: F401

            return torch.device("tpu")
        except Exception:
            pass
    if preferred != "tpu":
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla

            if hasattr(torch_xla, "device"):
                return torch_xla.device()
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        except Exception:
            pass
    return None


def resolve_wave2_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve CUDA/MPS/TPU/CPU device cleanly, supporting explicit `--device xla|tpu` and auto-detect."""
    if requested_device is not None:
        req = str(requested_device).strip().lower()
        if req in ("xla", "tpu"):
            dev = _try_get_tpu_device(preferred=req)
            if dev is None:
                raise RuntimeError(
                    f"Requested device={requested_device!r}, but no active TPU runtime "
                    "(torch_xla or torch_tpu) was found."
                )
            return dev
        return get_device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    tpu_dev = _try_get_tpu_device()
    if tpu_dev is not None:
        return tpu_dev
    return torch.device("cpu")


def wave2_autocast_context(dev: torch.device, enabled: bool):
    """Enable native BF16 autocast on CUDA (Ampere/Ada/Hopper/Blackwell) and TPU (`xla`/`tpu`)."""
    if not enabled:
        return nullcontext()
    if dev.type == "cuda" and supports_native_bf16(dev):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if dev.type == "tpu":
        return torch.amp.autocast("tpu", dtype=torch.bfloat16)
    if dev.type == "xla":
        try:
            return torch.amp.autocast("xla", dtype=torch.bfloat16)
        except Exception:
            from torch_xla.amp import autocast as xla_autocast

            return xla_autocast(dev, dtype=torch.bfloat16)
    return nullcontext()


def tpu_mark_step(dev: torch.device) -> None:
    """Trigger XLA graph execution boundary on `torch_xla` devices; no-op elsewhere."""
    if dev.type != "xla":
        return
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def tpu_optimizer_step(opt: torch.optim.Optimizer, dev: torch.device) -> None:
    """Execute optimizer step cleanly on CUDA/CPU/MPS/native-TPU and `torch_xla` TPU."""
    if dev.type == "xla":
        try:
            import torch_xla.core.xla_model as xm

            xm.optimizer_step(opt, barrier=True)
            return
        except Exception:
            opt.step()
            tpu_mark_step(dev)
            return
    opt.step()


SEQUENCE_PAYLOAD_KEYS: Tuple[str, ...] = (
    "sequence",
    "seq_rel_qty",
    "cust_median_gap_days",
    "attention_mask",
    "seq_product_ids",
    "seq_category_ids",
    "seq_subcategory_ids",
    "seq_business_line_ids",
    "seq_purchase_dates",
)


class WeightedRankerLoss(nn.Module):
    """Per-basket weighted Multi-Label BCE + Listwise Softmax Cross-Entropy Loss.

    Exact generalization of `recommender.ranker.RankerLoss`:
      - Weighted BCE averages over valid `(basket, candidate)` entries weighted by `w_b`:
            sum_b w_b * sum_{c in valid(b)} bce_{b,c} / sum_b (w_b * |valid(b)|)
      - Weighted Listwise CE averages over rows with >= 1 valid positive weighted by `w_b`:
            sum_{b : has_pos(b)} w_b * lw_b / sum_{b : has_pos(b)} w_b
      - When `sample_weight` is `None` or all-ones, both terms match `RankerLoss` identically.
    """

    def __init__(self, pos_weight: float = 10.0, listwise_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.listwise_weight = float(listwise_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        strict_checks: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 2 or targets.shape != logits.shape:
            raise ValueError("logits and targets must have matching (B, C) shapes.")
        if targets.device != logits.device:
            raise ValueError("logits and targets must be on the same device.")

        if candidate_mask is None:
            cmask = torch.ones_like(logits, dtype=torch.bool)
        else:
            if candidate_mask.dtype != torch.bool or candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask must be boolean with the same (B, C) shape as logits.")
            if candidate_mask.device != logits.device:
                raise ValueError("candidate_mask must be on the same device as logits.")
            cmask = candidate_mask

        on_tpu = is_tpu_device(logits.device)
        do_checks = (not on_tpu) if strict_checks is None else bool(strict_checks)
        if do_checks:
            if not bool(torch.isfinite(logits[cmask]).all()):
                raise ValueError("Valid candidate logits must be finite.")
            if not bool(torch.isfinite(targets[cmask]).all()):
                raise ValueError("Valid candidate targets must be finite.")
            if bool(((targets > 0.5) & ~cmask).any()):
                raise ValueError("Positive targets cannot refer to masked candidates.")

        if sample_weight is None:
            w = torch.ones((logits.size(0),), dtype=logits.dtype, device=logits.device)
        else:
            if sample_weight.ndim != 1 or sample_weight.size(0) != logits.size(0):
                raise ValueError(
                    f"sample_weight must have shape ({logits.size(0)},), got {tuple(sample_weight.shape)}."
                )
            w = sample_weight.to(device=logits.device, dtype=logits.dtype)
            if do_checks:
                if not bool(torch.isfinite(w).all()):
                    raise ValueError("sample_weight must be finite.")
                if bool((w < 0.0).any()):
                    raise ValueError("sample_weight must be non-negative.")

        safe_logits = torch.where(cmask, logits, torch.zeros_like(logits))
        safe_targets = torch.where(cmask, targets, torch.zeros_like(targets))

        # Use float64 accumulation on CPU/CUDA (overflow-proof); float32 on TPU (no native fp64).
        acc_dtype = torch.float32 if on_tpu else torch.float64
        pw = logits.new_tensor([self.pos_weight])
        bce_terms = F.binary_cross_entropy_with_logits(
            safe_logits, safe_targets, pos_weight=pw, reduction="none"
        )
        bce_valid = torch.where(cmask, bce_terms, torch.zeros_like(bce_terms))
        w_acc = w.to(acc_dtype)
        row_valid_count = cmask.sum(dim=-1).to(acc_dtype)
        bce_denom = (w_acc * row_valid_count).sum()
        has_bce = bce_denom > 0.0
        safe_bce_denom = torch.where(has_bce, bce_denom, torch.ones_like(bce_denom))
        bce_num = (w_acc * bce_valid.to(acc_dtype).sum(dim=-1)).sum()
        loss_bce = torch.where(has_bce, bce_num / safe_bce_denom, torch.zeros_like(bce_num)).to(logits.dtype)

        # Listwise Softmax Cross-Entropy (never feeds all -inf into softmax or 0 * -inf; branchless for XLA)
        row_has_valid = cmask.any(dim=-1, keepdim=True)
        softmax_logits = safe_logits.masked_fill(~cmask, -torch.inf)
        softmax_logits = torch.where(row_has_valid, softmax_logits, torch.zeros_like(softmax_logits))
        log_probs = F.log_softmax(softmax_logits, dim=-1)
        log_probs = torch.where(cmask, log_probs, torch.zeros_like(log_probs))

        pos_mask = (safe_targets > 0.5) & cmask
        pos_counts = pos_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        per_row = -((safe_targets / pos_counts) * log_probs).to(acc_dtype).sum(dim=-1)
        row_has_pos = pos_mask.any(dim=-1)
        lw_weights = torch.where(row_has_pos, w_acc, torch.zeros_like(w_acc))
        lw_denom = lw_weights.sum()
        has_lw = lw_denom > 0.0
        safe_lw_denom = torch.where(has_lw, lw_denom, torch.ones_like(lw_denom))
        lw_num = (lw_weights * per_row).sum()
        loss_lw = torch.where(has_lw, lw_num / safe_lw_denom, torch.zeros_like(lw_num)).to(logits.dtype)

        return {
            "loss": loss_bce + self.listwise_weight * loss_lw,
            "bce_loss": loss_bce,
            "list_loss": loss_lw,
        }


def compute_recency_sample_weights(
    prediction_days: torch.Tensor,
    *,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
    half_life_days: float = 365.0,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Compute strictly train-only exponential time-decay basket weights normalized to mean 1.0.

    Formula:
        age_i = max(0, cutoff_day - prediction_day_i)
        raw_i = max(min_weight, 2^(-age_i / half_life_days))
        w_i   = raw_i / mean(raw)

    Raises `ValueError` if any basket has `prediction_day >= cutoff_day` (prevents val/test leakage).
    """
    if not np.isfinite(half_life_days) or half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
    if not (0.0 < float(min_weight) <= 1.0):
        raise ValueError(f"min_weight must be in (0, 1], got {min_weight}")
    days = torch.as_tensor(prediction_days, dtype=torch.float32)
    if days.ndim != 1 or days.numel() == 0:
        raise ValueError(f"prediction_days must be a non-empty 1D tensor, got shape {tuple(days.shape)}")
    if not bool(torch.isfinite(days).all()):
        raise ValueError("prediction_days must be finite.")

    cutoff_day = float((pd.Timestamp(train_cutoff) - PURCHASE_DATE_ORIGIN).days)
    if bool((days >= cutoff_day).any()):
        raise ValueError(
            f"Found prediction_days >= train_cutoff day ({cutoff_day}); "
            "recency sample weights are strictly for training baskets (< train_cutoff)."
        )

    age_days = (cutoff_day - days).clamp_min(0.0)
    raw = torch.pow(2.0, -age_days / float(half_life_days)).clamp_min(float(min_weight))
    return raw / raw.mean().clamp_min(1e-12)


def _split_identity_digest(customer_ids: Sequence[str], prediction_days: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(f"n={len(customer_ids)}".encode("utf-8"))
    for cid in customer_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\n")
    h.update(torch.as_tensor(prediction_days, dtype=torch.int64).cpu().numpy().tobytes())
    return h.hexdigest()


def ensure_v2_sequences_for_length(
    seq_len: int,
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    dev_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    cache_dir: Path = PREPARED_ROOT,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF_V2,
    test_start: pd.Timestamp = TEST_START_V2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return `(tr_meta_L, va_meta_L)` with strictly point-in-time `seq_len`-event histories.

    - When `seq_len` matches the already-prepared sequence length (32), returns shallow copies
      immediately with zero compute.
    - When `seq_len > 32` (e.g. 64 or 96), builds & caches `cache_dir / f"sequences_len{seq_len}.pt"`
      using the exact `prepare_v2_full.py` representative-catalogue and zeroed-month-slot contract,
      verifying zero held-out Test (`>= test_start`) rows and validating both splits.
    """
    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    p_dates = pd.to_datetime(dev_purchases["purchase_date"], format="mixed").dt.normalize()
    if not bool(p_dates.lt(test_start).all()):
        raise ValueError(f"Held-out test event (>= {test_start.date()}) leaked into dev_purchases!")

    if int(tr_meta["attention_mask"].size(1)) == seq_len and int(va_meta["attention_mask"].size(1)) == seq_len:
        validate_dated_split(tr_meta, "train", train_cutoff=train_cutoff, test_start=test_start)
        validate_dated_split(va_meta, "val", train_cutoff=train_cutoff, test_start=test_start)
        return dict(tr_meta), dict(va_meta)

    tr_cids = list(tr_meta["customer_ids"])
    va_cids = list(va_meta["customer_ids"])
    tr_days = [int(round(float(d))) for d in tr_meta["prediction_days"].tolist()]
    va_days = [int(round(float(d))) for d in va_meta["prediction_days"].tolist()]
    expected_ident = {
        "builder_version": SEQUENCE_BUILDER_VERSION,
        "seq_len": seq_len,
        "train_digest": _split_identity_digest(tr_cids, tr_meta["prediction_days"]),
        "val_digest": _split_identity_digest(va_cids, va_meta["prediction_days"]),
        "train_cutoff": str(train_cutoff),
        "test_start": str(test_start),
    }

    def _sequence_payload_digest(train_payload: Mapping[str, Any], val_payload: Mapping[str, Any]) -> str:
        h = hashlib.sha256()
        for split_name, payload in (("train", train_payload), ("val", val_payload)):
            h.update(split_name.encode("utf-8"))
            for key in SEQUENCE_PAYLOAD_KEYS:
                t = torch.as_tensor(payload[key]).detach().cpu().contiguous()
                h.update(f"{key}:{tuple(t.shape)}:{t.dtype}".encode("utf-8"))
                h.update(t.numpy().tobytes())
        return h.hexdigest()

    def _valid_cached_payload(cached_obj: Mapping[str, Any]) -> bool:
        if cached_obj.get("identity") != expected_ident:
            return False
        stored_digest = cached_obj.get("payload_sha256")
        if not isinstance(stored_digest, str) or not stored_digest:
            return False
        for split_name, n_expected in (("train", len(tr_cids)), ("val", len(va_cids))):
            part = cached_obj.get(split_name)
            if not isinstance(part, Mapping):
                return False
            if any(k not in part for k in SEQUENCE_PAYLOAD_KEYS):
                return False
            for key in SEQUENCE_PAYLOAD_KEYS:
                t = part[key]
                if not isinstance(t, torch.Tensor) or t.size(0) != n_expected:
                    return False
                if key == "cust_median_gap_days":
                    if t.ndim != 1:
                        return False
                elif t.ndim < 2 or t.size(1) != seq_len:
                    return False
            if bool((part["sequence"][..., 3:5] != 0.0).any()):
                return False
        return _sequence_payload_digest(cached_obj["train"], cached_obj["val"]) == stored_digest

    print(f"[v2-w2] Building strictly point-in-time seq_len={seq_len} sequences (train + val)...", flush=True)
    t0 = time.perf_counter()
    product_ids = list(shared_meta["product_ids"])
    reps = np.asarray(shared_meta["reps"], dtype=np.int64)
    universe = shared_meta["universe"]
    vocabs = shared_meta["vocabs"]

    pool = RollingCandidatePool(catalogue, dev_purchases)
    allowed_train_pids = observed_pack_ids(pool.first_sales, train_cutoff)
    if allowed_train_pids is None:
        raise RuntimeError("No train-observed pack IDs found prior to train_cutoff.")
    representatives = catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()
    representatives = representatives.loc[representatives["product_id"].isin(allowed_train_pids)]

    def frame_digest(frame: pd.DataFrame) -> str:
        hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        return hashlib.sha256(hashes.tobytes()).hexdigest()

    expected_ident.update(
        purchases=frame_digest(dev_purchases[["customer_id", "product_id", "purchase_date", "quantity"]]),
        representatives=frame_digest(representatives[
            ["product_id", "base_name", "product_category", "product_subcategory", "business_line"]
        ]),
        encoding=json_digest([universe["all_pid_to_bn"], universe["bn_to_idx"], vocabs]),
        quantity_reference_length=32,
    )
    cache_path = Path(cache_dir) / f"sequences_len{seq_len}.pt"
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            cached = {}
        if isinstance(cached, Mapping) and _valid_cached_payload(cached):
            print(f"[v2-w2] Loaded cached seq_len={seq_len} sequences from {cache_path}", flush=True)
            tr_out = dict(tr_meta)
            va_out = dict(va_meta)
            tr_out.update(cached["train"])
            va_out.update(cached["val"])
            validate_dated_split(tr_out, "train", train_cutoff=train_cutoff, test_start=test_start)
            validate_dated_split(va_out, "val", train_cutoff=train_cutoff, test_start=test_start)
            return tr_out, va_out

    built: Dict[str, Dict[str, torch.Tensor]] = {}
    for split, cids, days, base_meta in (
        ("train", tr_cids, tr_days, tr_meta),
        ("val", va_cids, va_days, va_meta),
    ):
        seqs = build_base_name_sequences(
            dev_purchases,
            cids,
            days,
            universe,
            representatives,
            vocabs["category_vocab"],
            vocabs["subcategory_vocab"],
            vocabs["business_line_vocab"],
            seq_len=seq_len,
            quantity_reference_length=32,
            frozen_v5_replay=False,
            test_start=test_start,
        )
        # Match prepare_v2_full.py: zero calendar month sin/cos slots (cols 3:5)
        seqs["sequence"][..., 3:5] = 0.0
        merged = dict(base_meta)
        merged.update(seqs)
        validate_dated_split(merged, split, train_cutoff=train_cutoff, test_start=test_start)
        built[split] = {k: seqs[k] for k in SEQUENCE_PAYLOAD_KEYS}

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    payload_sha = _sequence_payload_digest(built["train"], built["val"])
    atomic_torch_save(
        {
            "identity": expected_ident,
            "payload_sha256": payload_sha,
            "train": built["train"],
            "val": built["val"],
        },
        cache_path,
    )
    print(f"[v2-w2] Cached seq_len={seq_len} sequences -> {cache_path} ({time.perf_counter() - t0:.1f}s)", flush=True)

    tr_out = dict(tr_meta)
    va_out = dict(va_meta)
    tr_out.update(built["train"])
    va_out.update(built["val"])
    return tr_out, va_out
