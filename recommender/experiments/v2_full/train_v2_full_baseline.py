"""Train the clean `v5_shared_uv` 2-stage baseline on the 4-year `v2_full` dataset (validation-only).

Runs the exact canonical v5_shared_uv pipeline on `data/v2_full/` (strictly Train < 2026-02-01 ->
Validation 2026-02-01..2026-06-30, ZERO held-out Test >= 2026-07-01 access, ZERO v1 reference mutation):
  1. Stage 1 Two-Tower (`StaticEnrichedRetriever`, E_no_brand + 65D structured specs, pack-free BGE-M3,
     0D Stage-1 tabular, no `sales_branch`) -> sealed `train_vectors.pt` & `val_vectors.pt`.
  2. Stage 2 `SharedUVStage5Ranker` (1-layer 8-head `memory_encoder` + 8-head cross-attention +
     (384, 192, 96) MLP + training-only 60d aux head lambda=0.3), streaming float16 memmap batches
     cleanly to float32 on MPS (< 2.5 GB RAM).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.embedding_ablation.config import build_ablation_specs
from recommender.experiments.roadmap_v2.clean_ranker import SharedUVStage5Ranker
from recommender.experiments.roadmap_v2.config import EVIDENCE_ROOT as V1_EVIDENCE_ROOT
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
)
from recommender.experiments.roadmap_v2.run_stage5 import _verify_reference_files
from recommender.experiments.roadmap_v2.run_stage5_ablations import _evaluate_with_cohorts_and_sku
from recommender.experiments.roadmap_v2.stage5_artifacts import (
    product_inputs_digest, read_stage5_vectors, validate_stage5_vectors, vector_identity,
)
from recommender.experiments.roadmap_v2.stage5_base_entity import evaluate_base_name_predictions
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.experiments.static_enrichment.model import StaticEnrichedRetriever
from recommender.ranker import RankerLoss
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .config import (
    ARTIFACT_ROOT,
    load_v2_development_purchases,
    EVIDENCE_ROOT,
    PREPARED_ROOT,
    RAW_ITEMS_PATH,
    SEED,
    STAGE1_EPOCHS,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_WEIGHT_DECAY,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
)
from .provenance import input_digest, prepared_dataset_digest
from .prepare_v2_full import load_prepared_metadata, prepare_v2_full_dataset


def build_v2_stage1_tower(
    shared_meta: Mapping[str, Any],
    *,
    num_demand_features: int = 65,
    d_model: int = 128,
    max_seq_len: int = 32,
) -> StaticEnrichedRetriever:
    """Build Stage 1 pack-free 65D-spec Two-Tower sized natively to v2_full train-only vocabularies."""
    spec = build_ablation_specs()["E_no_brand"].variant_spec
    spec = replace(
        spec,
        product_demand_features=int(num_demand_features),
        customer_tabular_features=0,
        customer_embeddings=tuple(c for c in spec.customer_embeddings if c != "sales_branch"),
    )
    vocabs = shared_meta["vocabs"]
    enrichment = shared_meta["enrichment"]
    return StaticEnrichedRetriever(
        spec,
        num_cities=len(vocabs["city_vocab"]) + 1,
        num_regions=len(vocabs["region_vocab"]) + 1,
        num_industries=len(vocabs["industry_vocab"]) + 1,
        num_catalog_products=int(shared_meta["universe"]["num_base_names"]),
        num_demand_features=int(num_demand_features),
        product_attr_vocab_sizes={k: len(v) for k, v in enrichment["product_vocabs"].items()},
        customer_attr_vocab_sizes={k: len(v) for k, v in enrichment["customer_vocabs"].items()},
        output_dim=int(d_model),
        hidden_dim=int(d_model),
        max_seq_len=int(max_seq_len),
    )


def _customer_vector_input_digest(split):
    fields = ("text_embedding", "city_id", "region_id", "industry_id", "tabular_features",
              "sequence", "attention_mask", "seq_purchase_dates", "seq_product_ids", "sales_branch")
    return input_digest({name: split[name] for name in fields})


def load_v2_stage1_vectors(directory, shared_meta, tr_meta, va_meta, d_model=128):
    loaded, _ = read_stage5_vectors(directory)
    if any("customer_inputs_sha256" in export for export in loaded.values()):
        for name, split in (("train", tr_meta), ("val", va_meta)):
            if loaded[name].get("customer_inputs_sha256") != _customer_vector_input_digest(split):
                raise ValueError(f"{name} vectors use different customer inputs.")
    return validate_stage5_vectors(
        loaded, {"train": tr_meta, "val": va_meta}, shared_meta["universe"]["base_names"],
        file_digest(directory / "best.pt"), shared_meta["prod_inputs"], d_model=d_model,
    )


def train_v2_stage1_two_tower(
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    out_dir: Path,
    seed: int = SEED,
    epochs: int = STAGE1_EPOCHS,
    batch_size: int = 512,
    dev: torch.device,
    d_model: int = 128,
    max_seq_len: int = 32,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train Stage 1 Two-Tower on v2_full and export sealed train/val u_bn & v_bn vectors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "result.json"
    tr_vec_p = out_dir / "train_vectors.pt"
    va_vec_p = out_dir / "val_vectors.pt"
    best_pt = out_dir / "best.pt"
    stage1_settings = dict(seed=seed, epochs=epochs, batch_size=batch_size, d_model=d_model, max_seq_len=max_seq_len)
    stage1_settings["input_sha256"] = input_digest((shared_meta, tr_meta, va_meta))
    if res_path.is_file() and best_pt.is_file() and tr_vec_p.is_file() and va_vec_p.is_file():
        print(f"[v2-s1] Loading existing sealed Two-Tower vectors from {out_dir}...", flush=True)
        res = json.loads(res_path.read_text(encoding="utf-8"))
        legacy_reference = out_dir == ARTIFACT_ROOT / f"baseline_v5_shared_uv-seed{seed}" / "two_tower"
        legacy_settings_match = (
            "run_settings" not in res and legacy_reference and epochs == STAGE1_EPOCHS
            and (batch_size, d_model, max_seq_len) == (512, 128, 32)
        )
        if (res.get("run_settings") == stage1_settings or legacy_settings_match) and (
            res.get("tower_checkpoint_sha256") == file_digest(best_pt)
        ):
            return (res, *load_v2_stage1_vectors(out_dir, shared_meta, tr_meta, va_meta, d_model))

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    prod_inputs = shared_meta["prod_inputs"]
    universe = shared_meta["universe"]
    n_bn = int(universe["num_base_names"])
    n_train_bn = int(universe["trained_base_names"])

    model = build_v2_stage1_tower(
        shared_meta,
        num_demand_features=int(prod_inputs["demand_features"].shape[1]),
        d_model=int(d_model),
        max_seq_len=int(max_seq_len),
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v2-s1] Training StaticEnrichedRetriever ({n_params:,} params, {n_bn} formulations)...", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=1e-5)
    useful = torch.where((tr_meta["target_bn"][:, :n_train_bn] & tr_meta["unseen_bn"][:, :n_train_bn]).any(-1))[0]
    gen = torch.Generator().manual_seed(int(seed))

    def _p_in(limit: Optional[int] = None) -> Dict[str, Any]:
        end = limit if limit is not None else n_bn
        return {
            "text_embedding": prod_inputs["text_embedding"][:end].to(dev),
            "demand_features": prod_inputs["demand_features"][:end].to(dev),
            "attr_ids": {k: v[:end].to(dev) for k, v in prod_inputs["attr_ids"].items()},
        }

    def _c_in(split_d: Mapping[str, Any], rows: torch.Tensor) -> Dict[str, Any]:
        return {
            "text_embedding": split_d["text_embedding"][rows].to(dev),
            "city_id": split_d["city_id"][rows].to(dev),
            "region_id": split_d["region_id"][rows].to(dev),
            "industry_id": split_d["industry_id"][rows].to(dev),
            "tabular_features": split_d["tabular_features"][rows].to(dev),
            "sequence": split_d["sequence"][rows].to(dev),
            "attention_mask": split_d["attention_mask"][rows].to(dev),
            "seq_purchase_dates": split_d["seq_purchase_dates"][rows].to(dev),
            "seq_product_ids": split_d["seq_product_ids"][rows].to(dev),
            "extra_attr_ids": {"sales_branch": split_d["sales_branch"][rows].to(dev)},
        }

    @torch.no_grad()
    def _eval_unseen_hits(model) -> Tuple[int, int, int]:
        model.eval()
        vecs = model.product_vectors(_p_in())
        idxs = torch.arange(n_bn, device=dev)
        h25 = h50 = h100 = 0
        for rows in torch.arange(len(va_meta["customer_ids"])).split(512):
            u_b = model.encode_purchase(_c_in(va_meta, rows), vecs)
            sc = model.score_cached(u_b, vecs, idxs, va_meta["familiarity_bn"][rows].to(dev))
            unseen_m = va_meta["unseen_bn"][rows].to(dev)
            sc = sc.masked_fill(~unseen_m, -torch.inf)
            top100 = sc.topk(k=min(100, n_bn), dim=-1).indices.cpu()
            new_t = va_meta["new_target_bn"][rows]
            hits = (new_t & va_meta["unseen_bn"][rows]).gather(1, top100)
            h25 += int(hits[:, :25].sum().item())
            h50 += int(hits[:, :50].sum().item())
            h100 += int(hits[:, :100].sum().item())
        return h25, h50, h100

    train_p = _p_in(limit=n_train_bn)
    train_idxs = torch.arange(n_train_bn, device=dev)
    pw_t = torch.tensor(20.0, device=dev)
    total_val_new = int(va_meta["new_target_bn"].sum().item())

    best_key = (-1, -1, -1)
    best_ep = 0
    t0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        ep_t = time.perf_counter()
        model.train()
        losses = []
        for rows in useful[torch.randperm(len(useful), generator=gen)].split(batch_size):
            opt.zero_grad(set_to_none=True)
            vecs = model.product_vectors(train_p)
            u_b = model.encode_purchase(_c_in(tr_meta, rows), vecs)
            logits = model.score_cached(u_b, vecs, train_idxs, tr_meta["familiarity_bn"][rows, :n_train_bn].to(dev))
            allowed = tr_meta["unseen_bn"][rows, :n_train_bn].to(dev)
            target = tr_meta["target_bn"][rows, :n_train_bn].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(logits[allowed], target[allowed], pos_weight=pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sch.step()

        h25, h50, h100 = _eval_unseen_hits(model)
        key = (h50, h25, h100)
        if key > best_key:
            best_key = key
            best_ep = ep
            atomic_torch_save(
                {"epoch": ep, "hits25": h25, "hits50": h50, "hits100": h100, "model_state_dict": model.state_dict()},
                best_pt,
            )
        if ep in (1, 5, 10, 15, 20, 25, 30) or key == best_key:
            print(
                f"[v2-s1] ep={ep:02d}/{epochs} loss={np.mean(losses):.4f} "
                f"unseen@25={h25}/{total_val_new} @50={h50}/{total_val_new} @100={h100}/{total_val_new} "
                f"({time.perf_counter() - ep_t:.1f}s)",
                flush=True,
            )

    # Reload best and export sealed train & val vectors
    best_ck = torch.load(best_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(best_ck["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        v_bn = model.product_vectors(_p_in())
        u_splits: Dict[str, torch.Tensor] = {}
        for sname, smeta in (("train", tr_meta), ("val", va_meta)):
            chunks = []
            for rows in torch.arange(len(smeta["customer_ids"])).split(512):
                chunks.append(model.encode_purchase(_c_in(smeta, rows), v_bn).squeeze(1).cpu())
            u_splits[sname] = torch.cat(chunks, dim=0)
            if not torch.isfinite(u_splits[sname]).all():
                raise ValueError(f"Non-finite Stage 1 u_bn in {sname}.")
        v_bn_cpu = v_bn.cpu()

    tower_sha = file_digest(best_pt)
    inputs_hash = product_inputs_digest(prod_inputs)
    names = list(universe["base_names"])
    for sname, smeta, out_p in (("train", tr_meta, tr_vec_p), ("val", va_meta, va_vec_p)):
        ident = vector_identity(smeta, names, tower_sha, inputs_hash)
        ident["customer_inputs_sha256"] = _customer_vector_input_digest(smeta)
        atomic_torch_save({"v_bn": v_bn_cpu, "u_bn": u_splits[sname], **ident}, out_p)

    res = {
        "run_settings": stage1_settings,
        "best_epoch": int(best_ep),
        "val_unseen_hits@25": int(best_ck["hits25"]),
        "val_unseen_hits@50": int(best_ck["hits50"]),
        "val_unseen_hits@100": int(best_ck["hits100"]),
        "val_total_new_targets": total_val_new,
        "trainable_params": int(n_params),
        "tower_checkpoint_sha256": tower_sha,
        "training_seconds": float(time.perf_counter() - t0),
    }
    atomic_json(res, res_path, overwrite=True)
    del model, opt, sch
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    return res, v_bn_cpu, u_splits["train"], u_splits["val"]


def iter_block_shuffled_batches(
    cand_mm: np.ndarray,
    n_rows: int,
    batch_size: int,
    gen: torch.Generator,
    *,
    block_rows: int = 512,
    blocks_per_buffer: int = 8,
):
    """Yield `(rows_tensor, cand_float32_np)` via sequential multi-block NVMe reads + cross-timeline shuffling.

    Reads `blocks_per_buffer` random contiguous blocks of `block_rows` (~44 MB each on v2_full,
    mixing 8 distinct time periods per buffer) sequentially into RAM, then randomly permutes all
    rows inside the buffer. Avoids random 12 GB memmap page-fault thrashing while keeping RAM < 2.5 GB.
    """
    starts = list(range(0, n_rows, int(block_rows)))
    block_order = torch.randperm(len(starts), generator=gen).tolist()
    for b_off in range(0, len(block_order), int(blocks_per_buffer)):
        chosen = block_order[b_off : b_off + int(blocks_per_buffer)]
        row_chunks = []
        cand_chunks = []
        for blk_i in chosen:
            s0 = starts[blk_i]
            s1 = min(n_rows, s0 + int(block_rows))
            row_chunks.append(np.arange(s0, s1, dtype=np.int64))
            # Sequential slice read from memmap (fast NVMe streaming!)
            cand_chunks.append(np.asarray(cand_mm[s0:s1], dtype=np.float32))
        buf_rows = np.concatenate(row_chunks, axis=0)
        buf_cand = np.concatenate(cand_chunks, axis=0)
        perm = torch.randperm(len(buf_rows), generator=gen).numpy()
        for off in range(0, len(buf_rows), int(batch_size)):
            idx = perm[off : off + int(batch_size)]
            yield torch.from_numpy(buf_rows[idx].copy()), buf_cand[idx]
        del buf_rows, buf_cand, row_chunks, cand_chunks


def _build_memmap_batch(
    split_meta: Mapping[str, Any],
    cand_np: np.ndarray,
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    return {
        "customer_vector": u_bn_split[rows].to(dev, dtype=torch.float32),
        "customer_features": split_meta["tabular_features"][rows].to(dev, dtype=torch.float32),
        "catalog_product_vectors": v_bn_dev,
        "sequence": split_meta["sequence"][rows].to(dev),
        "attention_mask": split_meta["attention_mask"][rows].to(dev),
        "seq_product_ids": split_meta["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_meta["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_meta["prediction_days"][rows].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand_np, dtype=np.float32)).to(dev),
        "candidate_mask": split_meta["allowed_bn"][rows].to(dev),
    }


def rank_eligible_formulations(scores: torch.Tensor, allowed: torch.Tensor) -> np.ndarray:
    """Keep unavailable entries out of metrics even when fewer than 50 candidates qualify."""
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError("Eligible formulations require finite scores.")
    order = scores.masked_fill(~allowed, -torch.inf).argsort(descending=True, stable=True)[:, :50]
    return order.masked_fill(~allowed.gather(1, order), -1).numpy()


@torch.no_grad()
def _score_memmap(
    model: SharedUVStage5Ranker,
    split_meta: Mapping[str, Any],
    cand_mm: np.ndarray,
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    dev: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    n_rows = len(split_meta["customer_ids"])
    chunks = []
    # Read val memmap in contiguous 2048-row slices for sequential SSD speed
    for s0 in range(0, n_rows, 2048):
        s1 = min(n_rows, s0 + 2048)
        buf_cand = np.asarray(cand_mm[s0:s1], dtype=np.float32)
        for off in range(0, s1 - s0, batch_size):
            rows = torch.arange(s0 + off, min(s1, s0 + off + batch_size), dtype=torch.long)
            out = model(
                **_build_memmap_batch(split_meta, buf_cand[off : off + len(rows)], u_bn_split, v_bn_dev, rows, dev),
                return_aux=False,
            )
            chunks.append(out.logits.cpu())
        del buf_cand
    return torch.cat(chunks, dim=0)


def iter_training_batches(
    cand_arr: np.ndarray,
    n_rows: int,
    batch_size: int,
    gen: torch.Generator,
    *,
    in_memory: bool = False,
    block_rows: int = 512,
    blocks_per_buffer: int = 8,
):
    """Yield `(rows_tensor, cand_float32_np)` cleanly from RAM (`in_memory=True`) or NVMe memmap."""
    if in_memory:
        perm = torch.randperm(n_rows, generator=gen)
        for rows in perm.split(int(batch_size)):
            yield rows, np.asarray(cand_arr[rows.numpy()], dtype=np.float32)
        return
    yield from iter_block_shuffled_batches(
        cand_arr,
        n_rows,
        batch_size,
        gen,
        block_rows=block_rows,
        blocks_per_buffer=blocks_per_buffer,
    )


def amp_autocast_context(dev: torch.device, enabled: bool):
    """Enable native BF16 autocast on CUDA (Ampere/Ada/Hopper); no-op elsewhere."""
    if enabled and dev.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


SELECTED_OUTPUTS = ("best_ndcg.pt", "best_discovery.pt", "predictions_ndcg.npz", "predictions_discovery.npz")


def completed_arm(directory, run_settings, summary_path=None):
    summary_path = summary_path or directory / "arm_result.json"
    if not summary_path.is_file():
        return None
    try:
        result = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if result.get("run_settings") != run_settings:
        return None
    for name in SELECTED_OUTPUTS:
        path = directory / name
        if not path.is_file() or file_digest(path) != result.get("output_sha256", {}).get(name):
            return None
    return result


def sync_outputs_to_dir(sync_dir: Optional[Path], paths: Tuple[Path, ...]) -> None:
    """Copy completed checkpoints/JSON summaries into a persistent sync directory (e.g. Google Drive)."""
    if sync_dir is None:
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if not src.exists():
            continue
        if src.is_absolute():
            relative = src.relative_to(Path.cwd()) if src.is_relative_to(Path.cwd()) else Path(src.name)
        else:
            relative = src
        dst = sync_dir / relative
        if src.resolve() == dst.resolve():
            continue
        if src.is_dir() and dst.resolve().is_relative_to(src.resolve()):
            raise ValueError(f"Cannot sync a directory into itself: {src} -> {dst}")
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[v2-sync] Synced outputs -> {sync_dir}", flush=True)


def run_v2_full_baseline(
    *,
    seed: int = SEED,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    batch_size: int = STAGE2_BATCH_SIZE,
    device: Optional[str] = None,
    amp: bool = False,
    in_memory: bool = False,
    skip_v1_verify: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    before_hashes = None
    ref_v1 = ref_v1_3s = None
    if not skip_v1_verify:
        before_hashes = snapshot_preserved_hashes()
        ref_v1 = json.loads((V1_EVIDENCE_ROOT / "stage5_reference.json").read_text(encoding="utf-8"))
        ref_v1_3s = json.loads((V1_EVIDENCE_ROOT / "stage5_3stage_reference.json").read_text(encoding="utf-8"))
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    summary_path = EVIDENCE_ROOT / f"stage5_v5_baseline_seed{seed}.json"
    run_dir = ARTIFACT_ROOT / f"baseline_v5_shared_uv-seed{seed}"
    s2_dir = run_dir / "ranker_v5_control"
    run_settings = {
        "selection_version": 2, "seed": seed, "epochs": stage2_epochs, "stage1_epochs": stage1_epochs,
        "batch_size": batch_size, "in_memory": in_memory,
        "bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()),
    }
    manifest = prepare_v2_full_dataset()
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)

    run_dir.mkdir(parents=True, exist_ok=True)
    s1_res, v_bn, u_tr, u_va = train_v2_stage1_two_tower(
        shared_meta=shared_meta,
        tr_meta=tr_meta,
        va_meta=va_meta,
        out_dir=run_dir / "two_tower",
        seed=seed,
        epochs=stage1_epochs,
        dev=dev,
    )

    mmap_mode = None if in_memory else "r"
    if in_memory:
        print("[v2-s2] Loading float16 candidate features into RAM (--in-memory)...", flush=True)
    tr_cand_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)
    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)
    run_settings["dataset_identity"] = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand_mm, va_cand_mm, dev_purchases, catalogue
    )
    run_settings["stage1_identity"] = input_digest((s1_res["tower_checkpoint_sha256"], v_bn, u_tr, u_va))
    cached = completed_arm(s2_dir, run_settings, summary_path)
    if cached is not None:
        sync_outputs_to_dir(sync_dir, (s2_dir, summary_path))
        return cached

    v_dev = v_bn.to(dev)
    aux_60d_bn = tr_meta["aux_60d_bn"]
    full_window = tr_meta["full_window"]

    s2_dir.mkdir(parents=True, exist_ok=True)

    def _make_ranker(aux: bool) -> SharedUVStage5Ranker:
        return SharedUVStage5Ranker(
            num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
            num_customer_features=len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
            num_pairwise_bias_features=13,
            d_model=128,
            num_heads=8,
            memory_ffn_dim=384,
            mlp_widths=(384, 192, 96),
            dropout=0.12,
            aux_60d_enabled=aux,
            use_memory_encoder=True,
            self_attention_bias_mode="none",
            seed=seed,
        ).to(dev)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _make_ranker(aux=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    n_tr = len(tr_meta["customer_ids"])
    n_va = len(va_meta["customer_ids"])
    tot_va_new = int(va_meta["new_target_bn"].sum().item())
    use_bf16 = bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported())
    print(
        f"\n[v2-s2] Training SharedUVStage5Ranker ({n_params:,} params) on {n_tr:,} train / {n_va:,} val baskets "
        f"(device={dev}, batch_size={batch_size}, epochs={stage2_epochs}, bf16={use_bf16}, in_memory={in_memory})...",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=STAGE2_WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(stage2_epochs), eta_min=STAGE2_ETA_MIN)
    loss_fn = RankerLoss().to(dev)
    aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    best_ndcg_key = (-1.0, -1, -1.0)
    best_disc_key = (-1, -1.0, -1.0)
    epoch_history = []
    t_s2 = time.perf_counter()

    for ep in range(1, int(stage2_epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        for rows, cand_np in iter_training_batches(tr_cand_mm, n_tr, batch_size, gen, in_memory=in_memory):
            opt.zero_grad(set_to_none=True)
            inp = _build_memmap_batch(tr_meta, cand_np, u_tr, v_dev, rows, dev)
            cmask = inp["candidate_mask"]
            y = tr_meta["target_bn"][rows].to(dev, dtype=torch.float32) * cmask.float()
            with amp_autocast_context(dev, use_bf16):
                out, a_lg = model(**inp, return_aux=True)
            l_main = loss_fn(out.logits.float(), y, candidate_mask=cmask)["loss"]
            aux_rows = full_window[rows].to(dev)
            loss = l_main
            if aux_rows.any():
                y_aux = aux_60d_bn[rows].to(dev, dtype=torch.float32)[aux_rows] * cmask[aux_rows].float()
                l_aux = aux_loss_fn(a_lg[aux_rows].float(), y_aux, candidate_mask=cmask[aux_rows])["loss"]
                loss = loss + 0.3 * l_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, STAGE2_GRAD_CLIP_NORM)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sch.step()

        va_lg = _score_memmap(model, va_meta, va_cand_mm, u_va, v_dev, dev, batch_size=batch_size)
        ranked = rank_eligible_formulations(va_lg, va_meta["allowed_bn"])
        m_ep = evaluate_base_name_predictions(
            ranked, va_meta["target_bn"], va_meta["new_target_bn"], va_meta["customer_ids"]
        )
        inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}

        nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
        dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, s2_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, s2_dir / "best_discovery.pt")

        epoch_history.append(
            {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "bn_recall@5": float(m_ep["bn_recall@5"]),
                "bn_recall@10": float(m_ep["bn_recall@10"]),
                "bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
                "true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
            }
        )
        print(
            f"[v2-s2] ep={ep:02d}/{stage2_epochs} loss={np.mean(losses):.4f} "
            f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
            f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/{tot_va_new} "
            f"({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    # Full cohort + resolved exact-SKU evaluation & CPU latency benchmark

    product_ids = shared_meta["product_ids"]
    universe = shared_meta["universe"]
    sku_val = va_meta["sku_val"]

    s2_res: Dict[str, Any] = {
        "trainable_params": int(n_params),
        "training_seconds": float(time.perf_counter() - t_s2),
        "epoch_history": epoch_history,
    }
    cpu_bench_ms = {}
    for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
        ck = torch.load(s2_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_ranker(aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_lg = _score_memmap(eval_m, va_meta, va_cand_mm, u_va, v_dev, dev, batch_size=batch_size)
        ranked = rank_eligible_formulations(va_lg, va_meta["allowed_bn"])
        npz_p = s2_dir / f"predictions_{sel_name}.npz"
        ev = _evaluate_with_cohorts_and_sku(
            ranked, va_meta, sku_val, dev_purchases, catalogue, product_ids, universe, npz_p
        )
        if ev["bn_metrics"] != ck["metrics"]:
            raise AssertionError(f"v2_full {sel_name} replay mismatch!")
        ev["epoch"] = int(ck["epoch"])
        s2_res[sel_name] = ev

        if sel_name == "ndcg":
            eval_cpu = eval_m.to("cpu").eval()
            one_cand = np.asarray(va_cand_mm[0:1], dtype=np.float32)
            one_inp = _build_memmap_batch(va_meta, one_cand, u_va, v_bn, torch.tensor([0]), torch.device("cpu"))
            with torch.no_grad():
                for _ in range(10):
                    eval_cpu(**one_inp)
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    eval_cpu(**one_inp)
                    ts.append((time.perf_counter() - t0) * 1000.0)
            cpu_bench_ms = {
                "median_ms": float(np.median(ts)),
                "p95_ms": float(np.percentile(ts, 95)),
                "mean_ms": float(np.mean(ts)),
            }

    if not skip_v1_verify:
        if snapshot_preserved_hashes() != before_hashes:
            raise AssertionError("v1 preserved reference hashes changed!")
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "v2_full_stage5_v5_shared_uv_baseline",
        "dataset": "v2_full",
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "amp_bf16": use_bf16,
        "data_manifest": manifest,
        "stage1_two_tower": s1_res,
        "stage2_ranker": s2_res,
        "cpu_single_request_ranker_ms": cpu_bench_ms,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    results["run_settings"] = run_settings
    results["output_sha256"] = {name: file_digest(s2_dir / name) for name in SELECTED_OUTPUTS}
    atomic_json(results, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (s2_dir, summary_path))
    print(f"\nSaved v2_full baseline summary -> {summary_path} ({results['elapsed_seconds']:.1f}s)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=STAGE2_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--skip-v1-verify", action="store_true")
    parser.add_argument("--sync-dir", type=Path, default=None)
    args = parser.parse_args()
    run_v2_full_baseline(
        seed=args.seed,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        device=args.device,
        amp=args.amp,
        in_memory=args.in_memory,
        skip_v1_verify=args.skip_v1_verify,
        sync_dir=args.sync_dir,
    )


if __name__ == "__main__":
    main()
